# -*- coding: utf-8 -*-
"""
★ EditKG 主训练脚本 ★
对应论文 Algorithm 1: EditKG 整体训练流程

整体流程概览:
    1. 加载数据（用户-物品交互、知识图谱三元组）
    2. 计算物品间 NPMI 相关性（Section 3.1, Eq.1）
    3. 创建 Recommender 模型 + PCGrad 优化器
    4. 100 轮训练循环:
       - 每3轮: 生成候选KG → 训练KGC模型 → 更新Potential KG
       - 每轮: CF训练（PCGrad处理MMD和推荐损失的梯度冲突）
       - 每轮: 评估 Recall/NDCG/Precision/Hit@K
    5. 早停策略保存最优模型

@author: comp
Created on Tue May 16 23:45:11 2023
"""
import os

import sys
import math
import random
import torch
import itertools
import numpy as np
import pickle as pkl
from math import log
from tqdm import tqdm
from time import time 
import multiprocessing
import scipy.sparse as sp
from collections import Counter, defaultdict

from utils.parser import parse_args
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from utils.data_loader import load_data
from sklearn.metrics.pairwise import cosine_similarity

from modules.KGR_model import KGR
from modules.EDKG import Recommender       # 核心推荐模型
from modules.pcgrad import PCGrad          # 梯度冲突解决优化器

from utils.evaluate import test            # 多进程评估函数
from utils.helper import early_stopping, _generate_candi_kg, _cal_npmi


cores = multiprocessing.cpu_count()       # CPU核心数（用于多进程评估）
# 全局变量（在 main 中赋值）
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_data(train_entity_pairs, train_user_set):
    """
    将训练的用户-物品对转为字典格式
    
    参数:
        train_entity_pairs: [N, 2] 张量，每行 [user_id, item_id]
        train_user_set: 训练集用户交互字典（此函数中未使用）
    
    返回:
        feed_dict: {'users': 用户ID张量, 'pos_items': 正样本物品ID张量}
    """
    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = torch.LongTensor(entity_pairs[:, 0])
    feed_dict['pos_items'] = torch.LongTensor(entity_pairs[:, 1])

    return feed_dict

def _mulp_neg(triplet):
    """
    为单个KG三元组生成一个负样本（多进程辅助函数）
    
    负采样策略（随机选择一种）：
        - 33%概率: 替换关系r
        - 33%概率: 替换头实体h  
        - 33%概率: 替换尾实体t（尾实体从非物品实体中采样）
    
    参数:
        triplet: [h, r, t] 原始三元组
    
    返回:
        new_triplet: [h', r', t'] 负样本三元组
    """
    h, r, t = triplet
    rand_value = random.random()
    while True:
        if rand_value < 0.33:
            # 替换关系
            new_rel = random.randint(1, n_relations - 1)
            if new_rel != r:
                new_triplet = [h, new_rel, t]
                break
        elif rand_value >= 0.33 and rand_value > 0.66:
            # 替换头实体
            new_head = random.randint(0, n_entities - 1)
            if new_head != h:
                new_triplet = [new_head, r, t]
                break
        else:
            # 替换尾实体（从非物品实体中采样，避免将物品实体作为尾）
            new_t = random.randint(n_items, n_entities - 1)
            if new_t != t:
                new_triplet = [h, r, new_t]
                break
    return new_triplet

def _get_KGC_neg_data(triplets, neg_times=4):
    """
    为KGC训练生成负样本数据
    
    使用多进程并行生成，每个正三元组生成 neg_times 轮负样本。
    最终去重返回所有唯一的负三元组。
    
    参数:
        triplets: [N, 3] 正三元组数组
        neg_times: 负采样轮次（默认4轮）
    
    返回:
        去重后的负三元组 numpy 数组 [M, 3]
    """
    def negative_sampling(triplets):
        pool = multiprocessing.Pool(cores)
        neg_triplets = list(tqdm(iterable=pool.imap(_mulp_neg, triplets.tolist()), total=len(triplets)))
        pool.close()
        return np.array(neg_triplets)
    all_neg = []
    for i in range(min(neg_times, 2)):
        neg_triplets = negative_sampling(triplets)
        all_neg.append(neg_triplets)
    return np.unique(np.concatenate(all_neg, axis=0), axis=0)


def train_kgr_model(model, kgr_optimizer, triplets, kg_mask=[], epochs=2, threshold=0.5):
    """
    训练 KGC（知识图谱补全）模型
    对应论文 Section 3.2.2 的 KGC 训练过程
    
    流程:
        1. 如果有 KG 掩码（APL筛选结果），只保留被选中的三元组
        2. 正样本标签=1，生成负样本标签=0
        3. 按 epoch 循环训练:
           - 每4轮重新采样负样本
           - 前向传播计算 BCE 损失
           - 反向传播更新模型参数
           - 在验证集上计算准确率
    
    参数:
        model: Recommender 模型（通过 model.gcn.kgc 访问KGC子模型）
        kgr_optimizer: AdamW 优化器
        triplets: [N, 3] 所有KG三元组
        kg_mask: APL 的筛选掩码（非零=选中）
        epochs: 训练轮次
        threshold: 分类阈值（默认0.5）
    """
    # 如果有KG掩码，只保留被APL选中的三元组
    if kg_mask != []:
        triplets = triplets[kg_mask.reshape(-1) != 0.]

    kgr_batch = 1024

    # 构造正样本数据：三元组 + 标签1
    pos_label = np.ones((triplets.shape[0], 1))
    pos_data = np.concatenate([triplets, pos_label], axis=-1)
    index = np.arange(len(pos_data))
    np.random.shuffle(index)
    pos_data = pos_data[index]

    # 划分训练集和验证集（5%作为验证）
    pos_valid_num = int(len(pos_data) * 0.05)
    pos_train = pos_data[:-pos_valid_num]
    pos_valid = pos_data[-pos_valid_num:]

    model.train()
    
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        # 每4轮重新生成负样本（避免重复使用相同负样本）
        if epoch % 4 == 0:
            kgc_neg_data = _get_KGC_neg_data(triplets, neg_times=4)
            neg_label = np.zeros((kgc_neg_data.shape[0], 1))
            neg_data = np.concatenate([kgc_neg_data, neg_label], axis=-1)
            
            neg_train = neg_data[:-pos_valid_num]
            neg_valid = neg_data[-pos_valid_num:]
            # 正负样本合并后打乱
            kgr_train_data = np.concatenate([pos_train, neg_train], axis=0)
            index = np.arange(len(kgr_train_data))
            np.random.shuffle(index)
            kgr_train_data = kgr_train_data[index]
            kgr_valid_data = pos_valid  # 验证集只用正样本

            kgr_train_data = torch.LongTensor(kgr_train_data)
            kgr_valid_data = torch.LongTensor(kgr_valid_data)

            kgr_iter = math.ceil(len(kgr_train_data) / kgr_batch)
            kgr_val_iter = math.ceil(len(kgr_valid_data) / kgr_batch)

        # ===== 训练循环 =====
        batch_kg = dict()
        total_kg_loss = 0
        for i in range(kgr_iter):
            batch_kg["hr_pair"] = kgr_train_data[i * kgr_batch:(i + 1) * kgr_batch].to(device)

            # 通过 Recommender.forward(mode="kgc") 调用 KGC 模型
            kgr_batch_loss = model(batch_kg, mode="kgc")
            total_kg_loss += kgr_batch_loss.item()

            kgr_optimizer.zero_grad()
            kgr_batch_loss.backward()
            kgr_optimizer.step()

        # ===== 验证：计算分类准确率 =====
        model.eval()
        pred_label = []
        true_label = []
        for i in range(kgr_val_iter):
            batch_kg["hr_pair"] = kgr_valid_data[i * kgr_batch:(i + 1) * kgr_batch].to(device)
            pre = model.gcn.kgc(batch_kg, eval=True)   # 推理模式返回分数
            pre = (pre >= threshold).squeeze(-1).float()  # 阈值分类
            pre = pre.detach().cpu().numpy()
            label = batch_kg["hr_pair"][:, -1]
            label = label.cpu().numpy()
            pred_label.append(pre)
            true_label.append(label)
            
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        
        acc = accuracy_score(pred_label, true_label)  # 分类准确率（仅用于监控）

        

def _process_kg_attr(canditate_kg, tripltes, kg_mask=None):
    """
    扩展候选KG：将候选尾实体作为头实体的已有三元组也加入候选集
    
    示例：候选KG中有 (item_A, r1, entity_X)，
    如果原始KG中有 (entity_X, r2, entity_Y)，也把它加入候选集。
    这样可以保留候选实体的属性关系。
    
    参数:
        canditate_kg: [N, 3] NPMI生成的候选三元组
        tripltes: 所有原始KG三元组
        kg_mask: APL掩码（可选）
    
    返回:
        去重后的扩展候选KG [M, 3]
    """
    canditate_kg = np.unique(canditate_kg, axis=0)
    if kg_mask != None:
        tripltes = tripltes[kg_mask.reshape(-1) != 0]
        
    # 收集候选KG中尾实体作为头实体的已有三元组
    attr_set = set(np.unique(canditate_kg[:, -1]))   # 候选KG中所有尾实体
    out_attr_kg = []
    for tirp in tqdm(tripltes):
        if tirp[0] in attr_set:  # 如果原始三元组的头实体是候选的尾实体
            out_attr_kg.append(tirp)
    out_attr_kg = np.asarray(out_attr_kg)
    # 将扩展的属性三元组与候选KG合并
    if out_attr_kg.shape[0] > 0:
        all_candi_kg = np.concatenate([canditate_kg, out_attr_kg], axis=0)
    else:
        all_candi_kg = canditate_kg
    return np.unique(all_candi_kg, axis=0)
    


if __name__ == '__main__':
    """
    ===== EditKG 主训练入口 =====
    对应论文 Algorithm 1
    """
    # ===== 1. 读取命令行参数 =====
    global args, device, train_user_set, kg_dict, item_lists_dict, ent_lists_dict
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # 固定随机种子（用于显著性检验的多次实验）
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SEED] 随机种子已设置为 {seed}")

    # ===== 2. 加载数据 =====
    # train_cf: 训练交互对 [N, 2]
    # test_cf: 测试交互对
    # user_dict: {'train_user_set': {uid: [iid, ...]}, 'test_user_set': ...}
    # n_params: 数据规模参数
    # graph: NetworkX KG图
    # ui_sparse_graph: 归一化的用户-物品稀疏交互矩阵
    # item_rel_mask: 物品-关系缺失掩码 [n_items, n_relations]
    # triplets: 所有KG三元组 [N, 3]
    # kg_dict: {head: [(tail, relation), ...]} 邻接字典
    train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, all_sparse_graph, item_rel_mask, triplets, kg_dict, v_feat, t_feat = load_data(args)

    # 如果指定 --no_mm，禁用多模态特征（作为基线对照）
    if args.no_mm:
        v_feat = None
        t_feat = None
        print("[NO_MM] 已禁用多模态特征（基线模式）")

    # ===== 3. 计算物品间 NPMI（论文 Section 3.1, Eq.1）=====
    # NPMI(i,j) 衡量物品 i 和 j 的共现关联强度
    item_pmi_dict = _cal_npmi(user_dict['train_user_set'])
    pkl.dump(item_pmi_dict, open(args.dataset + "item_pair_pmi.pkl", "wb"))
    
    # 设置全局变量（被 _mulp_neg 等多进程函数使用）
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']

    # ===== 4. 准备CF训练数据 =====
    train_cf_pairs = (np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))#其实 train_cf 本身已经是 [N, 2] 格式了，这行代码主要作用就是确保数据类型是 int32 的 numpy 数组，方便后续做索引和打乱操作。

    # ===== 5. 创建模型 =====
    model = Recommender(n_params, args, graph, ui_sparse_graph, item_rel_mask, v_feat=v_feat, t_feat=t_feat).to(device)

    # ===== 6. 创建优化器 =====
    # 主优化器：Adam + PCGrad（用于处理MMD和推荐损失的梯度冲突，论文 Section 3.4.1）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = PCGrad(optimizer)
    
    # KGC专用优化器：AdamW（权重衰减版Adam）
    kgr_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  
    
    # ===== 7. 训练状态初始化 =====
    cur_best = 0              # 当前最优 Recall@20
    stopping_step = 0         # 早停计数器
    should_stop = False       # 早停标志
    best_metric = {"recall": 0, "ndcg": 0, "precision": 0, "hit_ratio": 0}   # 各指标最优值
    best_epoch = {"recall": 0, "ndcg": 0, "precision": 0, "hit_ratio": 0}    # 各指标最优轮次
    
    # 批次参数
    iter = math.ceil(len(train_cf_pairs) / args.batch_size)  # 每轮的批次数
    cl_batch = 512
    cl_iter = math.ceil(n_items / cl_batch)
    item_embs = []      # 物品嵌入（用于候选KG生成的余弦相似度筛选）
    KG_mask = []         # APL掩码（指示哪些三元组被保留）
    
    # ===== 8. 主训练循环（100轮）=====
    total_epochs = 2 if args.quick_test else 100
    if args.quick_test:
        print("\n" + "="*60)
        print("[QUICK TEST] 快速测试模式：仅跑2轮，验证代码路径")
        print("="*60 + "\n")
    for epoch in range(total_epochs):
        torch.cuda.empty_cache()
        
        # ----- 每10轮或第0轮：重新打乱训练数据 -----防止模型学到数据的排列顺序，而非数据本身的规律
        if epoch % 10 == 0 or epoch == 0:
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            # 构造训练 feed 数据
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'])
            all_feed_data['pos_index'] = torch.LongTensor(index)

        # ----- 每3轮：Knowledge Generator + Knowledge Deleter 更新 -----
        # 对应论文 Section 3.1（候选KG生成）和 Section 3.2（KGC训练）
        if epoch % 3 == 0:
            # Step 1: 基于NPMI生成候选KG三元组（论文 Eq.2-4）
            # pmi_threshold=0.6: NPMI阈值（不同数据集需调整: yelp/MIND=0.7, others=0.5）
            candi_kg = _generate_candi_kg(item_pmi_dict, n_items, kg_dict,
                                          item_embeds=item_embs,
                                          pmi_threshold=0.7, cos_threshold=0.95)
            
            # Step 2: 扩展候选KG（加入候选尾实体的属性三元组）
            all_candi_kg = _process_kg_attr(candi_kg, triplets)
            
            # Step 3: 训练KGC模型（论文 Section 3.2.2, Eq.10-11）
            # 第0轮训练10个epoch来预热，之后只训练1个epoch来微调
            if epoch == 0:
                kgr_epoch = 1 if args.quick_test else 10    # 预热轮次（MIND:6, others:8, book:10）
            else:
                kgr_epoch = 1     # 微调轮次（MIND:1, others:2）
            train_kgr_model(model, kgr_optimizer, triplets, kg_mask=KG_mask, epochs=kgr_epoch)
            
            # Step 4: 将候选KG更新到模型中（作为 Potential KG）
            all_candi_kg = torch.LongTensor(all_candi_kg).to(device)
            model.gcn._update_knowledge(all_candi_kg)
            
        # ----- CF推荐训练 -----
        # 对应论文 Section 3.3 + Section 3.4
        model.train()
        loss = 0
        train_s_t = time()

        for i in tqdm(range(iter)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            # 构造当前批次
            batch = dict()
            batch['pos_index'] = all_feed_data['pos_index'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['users'] = all_feed_data['users'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i * args.batch_size:(i + 1) * args.batch_size].to(device)

            # 前向传播：返回 (推荐损失, MMD损失)
            batch_loss, batch_mmd_loss = model(batch)
            
            # ★ PCGrad 反向传播（论文 Section 3.4.1, Eq.25-28）★
            # 先传辅助任务损失(MMD)，再传主任务损失(推荐)
            # 如果梯度冲突，将辅助梯度投影到主梯度的法平面上
            loss_list = [batch_mmd_loss, batch_loss]
            optimizer.pc_backward(loss_list)
            optimizer.step()#根据梯度更新模型参数
            loss += batch_loss.item()

        train_e_t = time()
        
        # ----- 生成当前嵌入和KG掩码（用于下一轮的候选KG生成）-----
        item_embs, KG_mask = model.generate(for_kgc=True)
        item_embs = item_embs.detach().cpu().numpy()    # 物品嵌入 [n_items, dim*3]
        KG_mask = KG_mask.detach().cpu().numpy()         # APL掩码

        # ----- 评估（每轮都评估）-----
        if epoch % 1 == 0:
            model.eval()
            test_s_t = time()
            with torch.no_grad():
                # 计算 Recall@K, NDCG@K, Precision@K, Hit@K
                ret = test(model, user_dict, n_params)
            test_e_t = time()
            
            # 更新最优指标
            for k in best_metric.keys():
                if ret[k][0] > best_metric[k]:
                    best_metric[k] = ret[k][0]
                    best_epoch[k] = epoch
            
            # 打印训练结果表格
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            
            # 保存训练日志到文件
            f = open('./result/{}_exp_cxks_kg_xr_v2.txt'.format(args.dataset), 'a+')
            f.write(str(best_metric) + '\n')
            f.write(str(best_epoch) + '\n')
            f.write(str(train_res) + '\n')
            f.write('\n')
            f.close()

            # ===== 早停检查 =====
            # 如果 Recall@20 连续10轮评估没有提升，则停止训练
            cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=2)
            
            if should_stop:
                break
            
            # ===== 保存最优模型 =====
            if ret['recall'][0] == cur_best and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
            
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))

    # ===== 输出 JSON 格式结果（便于显著性检验脚本解析）=====
    import json
    result_summary = {
        'seed': args.seed,
        'no_mm': args.no_mm,
        'best_recall_20': float(best_metric['recall']),
        'best_ndcg_20': float(best_metric['ndcg']),
        'best_precision_20': float(best_metric['precision']),
        'best_hit_ratio_20': float(best_metric['hit_ratio']),
        'best_epoch': best_epoch,
        'early_stop_epoch': epoch
    }
    print('\n[RESULT_JSON]' + json.dumps(result_summary))

    # 追加写入结果文件
    mode_tag = 'baseline' if args.no_mm else 'mm_text'
    result_file = f'./result/{args.dataset}_significance_{mode_tag}.jsonl'
    os.makedirs('./result', exist_ok=True)
    with open(result_file, 'a') as f:
        f.write(json.dumps(result_summary) + '\n')
    print(f'[RESULT] 结果已追加到 {result_file}')
