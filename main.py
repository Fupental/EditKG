# -*- coding: utf-8 -*-
"""
★ EditKG 主训练脚本 ★
对应论文 Algorithm 1: EditKG 整体训练流程

整体流程概览:
    1. 加载数据（用户-物品交互、知识图谱三元组）
    2. 计算物品间 NPMI 相关性（Section 3.1, Eq.1）
    3. 创建 Recommender 模型 + Adam 优化器
    4. 100 轮训练循环:
       - 每3轮: 生成候选KG → 更新Potential KG
       - 每轮: CF推荐训练（交叉熵损失）
       - 每轮: 评估 Recall/NDCG/Precision/Hit@K
    5. 早停策略保存最优模型

@author: comp
Created on Tue May 16 23:45:11 2023
"""
import os
import subprocess

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
from utils.path_utils import ensure_dir, resolve_dataset_dir, models_root
from utils.memory_monitor import log_cuda_mem, MemTimer
from prettytable import PrettyTable
# [已删除] accuracy_score — KGC验证不再需要
from utils.data_loader import load_data
from sklearn.metrics.pairwise import cosine_similarity

# [已删除] KGR模型不再使用
from modules.EDKG import Recommender       # 核心推荐模型

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


def precompute_llm_scores_subprocess(triplets_tensor, tag, llm_score_dir, data_dir, args, llm_score_cache_path):
    triplets_path = os.path.join(llm_score_dir, f"{tag}_triplets.pt")
    scores_path = os.path.join(llm_score_dir, f"{tag}_scores.pt")
    torch.save(triplets_tensor.detach().cpu(), triplets_path)
    print(f"[LLM] {tag} 三元组已保存: {triplets_path}")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "utils", "precompute_llm_scores.py"),
        "--triplets_path", triplets_path,
        "--output_path", scores_path,
        "--data_dir", data_dir,
        "--model_path", args.llm_model_path,
        "--adapter_path", args.llm_adapter_path,
        "--cache_path", llm_score_cache_path,
        "--batch_size", str(args.llm_batch_size),
        "--gpu_id", str(args.gpu_id),
    ]
    if args.mem_debug:
        cmd.append("--mem_debug")
    print(f"[LLM] 启动独立进程打分({tag}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return torch.load(scores_path, map_location=device)


def extract_active_primary_triplets(triplets, n_items, context_hops):
    """
    仅保留在当前 message passing 深度内可能影响 item embedding 的 Primary KG 边。
    对本模型的消息方向，保留：
    - 第1层：head 是 item 的边
    - 第2层及以后：head 是上一层 tail 实体的边
    """
    active_heads = set(range(int(n_items)))
    kept_masks = []
    heads = triplets[:, 0]
    tails = triplets[:, 2]
    for _ in range(int(context_hops)):
        head_list = np.fromiter(active_heads, dtype=np.int64)
        if head_list.size == 0:
            break
        mask = np.isin(heads, head_list)
        kept_masks.append(mask)
        active_heads = set(tails[mask].tolist())
    if not kept_masks:
        return triplets
    final_mask = kept_masks[0].copy()
    for mask in kept_masks[1:]:
        final_mask |= mask
    return triplets[final_mask]
    


if __name__ == '__main__':
    """
    ===== EditKG 主训练入口 =====
    对应论文 Algorithm 1
    """
    # ===== 1. 读取命令行参数 =====
    global args, device, train_user_set, kg_dict, item_lists_dict, ent_lists_dict
    args = parse_args()
    if args.llm_adapter_path and not os.path.exists(args.llm_adapter_path):
        alt_adapter = os.path.join(str(models_root()), os.path.basename(args.llm_adapter_path))
        if os.path.exists(alt_adapter):
            print(f"[PATH] llm_adapter_path 不存在，自动回退到: {alt_adapter}")
            args.llm_adapter_path = alt_adapter
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    data_dir = resolve_dataset_dir(args.data_path, args.dataset)
    llm_score_dir = args.llm_score_dir or os.path.join(data_dir, "llm_scores")
    llm_score_dir = ensure_dir(llm_score_dir)
    llm_score_cache_path = args.llm_score_cache_path or os.path.join(llm_score_dir, "triplet_score_cache.pt")

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
    log_cuda_mem("startup.after_seed", device, args.mem_debug)

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
    log_cuda_mem("after.load_data", device, args.mem_debug)
    primary_triplets = extract_active_primary_triplets(triplets, n_params['n_items'], args.context_hops)
    if args.quick_test and len(primary_triplets) > 20000:
        primary_triplets = primary_triplets[:20000]
        print(f"[QUICK TEST] Primary KG 活跃子图裁切到 {len(primary_triplets)} 条边")
    print(f"[Primary KG] 原始边数={len(triplets)}, 活跃子图边数={len(primary_triplets)}, context_hops={args.context_hops}")

    # 如果指定 --no_mm，禁用多模态特征（作为基线对照）
    if args.no_mm:
        v_feat = None
        t_feat = None
        print("[NO_MM] 已禁用多模态特征（基线模式）")

    # ===== 3. 计算物品间 NPMI（论文 Section 3.1, Eq.1）=====
    # NPMI(i,j) 衡量物品 i 和 j 的共现关联强度
    item_pmi_dict = _cal_npmi(user_dict['train_user_set'])
    log_cuda_mem("after.npmi", device, args.mem_debug)
    pmi_cache_path = os.path.join(resolve_dataset_dir(args.data_path, args.dataset), "item_pair_pmi.pkl")
    pkl.dump(item_pmi_dict, open(pmi_cache_path, "wb"))
    
    # 设置全局变量
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']

    # ===== 4. 准备CF训练数据 =====
    train_cf_pairs = (np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))#其实 train_cf 本身已经是 [N, 2] 格式了，这行代码主要作用就是确保数据类型是 int32 的 numpy 数组，方便后续做索引和打乱操作。

    # ===== 5. 创建模型 =====
    model = Recommender(
        n_params, args, graph, ui_sparse_graph, item_rel_mask, triplets=primary_triplets, v_feat=v_feat, t_feat=t_feat
    ).to(device)
    log_cuda_mem("after.model_init", device, args.mem_debug)

    # ===== 5.5 初始化LLM评分器（替代KGC教师）=====
    llm_scorer = None
    if not args.llm_model_path:
        raise ValueError("当前代码已重构为 LLM-only KG 筛边，必须提供 --llm_model_path")

    if args.llm_model_path and args.llm_score_mode == "online":
        from modules.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(
            model_path=args.llm_model_path,
            adapter_path=args.llm_adapter_path,
            data_dir=data_dir,
            device=f"cuda:{args.gpu_id}",
            batch_size=args.llm_batch_size,
            mem_debug=args.mem_debug,
        )
        print(f"[LLM] LLM评分器已就绪（LLM-only 模式）")
    elif args.llm_model_path and args.llm_score_mode == "subprocess":
        print("[LLM] 使用 subprocess 模式：Primary KG 预打分一次，Potential KG 每3轮独立进程打分")

    # ===== 5.6 预计算 Primary KG 的 LLM 真实性分数 =====
    primary_triplets = torch.LongTensor(primary_triplets).to(device)
    if args.llm_score_mode == "subprocess":
        with MemTimer("primary_kg.llm_subprocess", device, args.mem_debug):
            primary_scores = precompute_llm_scores_subprocess(
                primary_triplets, "primary_kg", llm_score_dir, data_dir, args, llm_score_cache_path
            )
    else:
        with MemTimer("primary_kg.llm_online", device, args.mem_debug):
            primary_scores = llm_scorer.score_triplets(primary_triplets, target_device=device)
    model.gcn.set_llm_scores(primary_scores_tensor=primary_scores)
    print(f"[LLM] Primary KG 打分完成, shape={primary_scores.shape}, 均值={primary_scores.float().mean():.3f}")

    # ===== 6. 创建优化器 =====
    # 主优化器：Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_cuda_mem("after.optimizer_init", device, args.mem_debug)
    # [已删除] PCGrad包装 — 仅单一损失，使用标准Adam
    
    # [已删除] kgr_optimizer — KGC训练已移至LLM独立训练
    
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
        # Knowledge Generator: 生成候选KG并更新Potential KG
        if epoch % 3 == 0:
            log_cuda_mem(f"epoch{epoch}.kg_refresh.before", device, args.mem_debug)
            # Step 1: 基于NPMI生成候选KG三元组（论文 Eq.2-4）
            # pmi_threshold=0.6: NPMI阈值（不同数据集需调整: yelp/MIND=0.7, others=0.5）
            candi_kg = _generate_candi_kg(item_pmi_dict, n_items, kg_dict,
                                          item_embeds=item_embs,
                                          pmi_threshold=0.7, cos_threshold=0.95)
            
            # Step 2: 扩展候选KG（加入候选尾实体的属性三元组）
            all_candi_kg = _process_kg_attr(candi_kg, triplets)
            if args.quick_test and len(all_candi_kg) > 20000:
                all_candi_kg = all_candi_kg[:20000]
                print(f"[QUICK TEST] Potential KG 裁切到 {len(all_candi_kg)} 条边")
            
            # [已删除] KGC训练步骤 → LLM训练独立进行（MS-Swift）
            
            # Step 4: 将候选KG更新到模型中（作为 Potential KG）
            all_candi_kg = torch.LongTensor(all_candi_kg).to(device)
            model.gcn._update_knowledge(all_candi_kg)
            log_cuda_mem(f"epoch{epoch}.kg_refresh.after_update", device, args.mem_debug,
                         extra=f"candidates={len(all_candi_kg)}")
            
            # Step 5: 预计算 Potential KG 所有三元组的 LLM 真实性分数
            if args.llm_model_path and args.llm_score_mode == "subprocess":
                with MemTimer(f"epoch{epoch}.llm_subprocess", device, args.mem_debug):
                    llm_scores = precompute_llm_scores_subprocess(
                        all_candi_kg, f"candidate_epoch_{epoch:03d}", llm_score_dir, data_dir, args, llm_score_cache_path
                    )
                model.gcn.set_llm_scores(potential_scores_tensor=llm_scores)
                print(f"[LLM] subprocess 打分完成, shape={llm_scores.shape}, 均值={llm_scores.mean():.3f}")
                log_cuda_mem(f"epoch{epoch}.llm_scores_loaded", device, args.mem_debug,
                             extra=f"shape={tuple(llm_scores.shape)}")
            elif llm_scorer is not None:
                print(f"[LLM] 对 {len(all_candi_kg)} 条候选三元组进行真实性评分...")
                with MemTimer(f"epoch{epoch}.llm_online", device, args.mem_debug):
                    llm_scores = llm_scorer.score_triplets(all_candi_kg, target_device=device)
                model.gcn.set_llm_scores(potential_scores_tensor=llm_scores)
                print(f"[LLM] 分数已缓存, shape={llm_scores.shape}, 均值={llm_scores.mean():.3f}")
                log_cuda_mem(f"epoch{epoch}.llm_scores_cached", device, args.mem_debug,
                             extra=f"shape={tuple(llm_scores.shape)}")
            
        # ----- CF推荐训练 -----
        # 对应论文 Section 3.3 + Section 3.4
        model.train()
        loss = 0
        train_s_t = time()

        for i in tqdm(range(iter)):
            torch.cuda.empty_cache()
            
            # 构造当前批次
            batch = dict()
            batch['pos_index'] = all_feed_data['pos_index'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['users'] = all_feed_data['users'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i * args.batch_size:(i + 1) * args.batch_size].to(device)

            # 前向传播：返回推荐损失
            if args.mem_debug and (i == 0 or (i + 1) % args.mem_debug_interval == 0):
                log_cuda_mem(f"epoch{epoch}.batch{i}.before_forward", device, True)
            cet_loss = model(batch)
            batch_loss = cet_loss
            
            optimizer.zero_grad()
            if args.mem_debug and (i == 0 or (i + 1) % args.mem_debug_interval == 0):
                log_cuda_mem(f"epoch{epoch}.batch{i}.after_forward", device, True,
                             extra=f"cet={cet_loss.item():.4f}")
            batch_loss.backward()
            if args.mem_debug and (i == 0 or (i + 1) % args.mem_debug_interval == 0):
                log_cuda_mem(f"epoch{epoch}.batch{i}.after_backward", device, True)
            optimizer.step()
            if args.mem_debug and (i == 0 or (i + 1) % args.mem_debug_interval == 0):
                log_cuda_mem(f"epoch{epoch}.batch{i}.after_step", device, True)
            loss += batch_loss.item()

        train_e_t = time()
        
        # ----- 生成当前嵌入和KG掩码（用于下一轮的候选KG生成）-----
        with MemTimer(f"epoch{epoch}.generate", device, args.mem_debug):
            item_embs, KG_mask = model.generate(for_kgc=True)
        item_embs = item_embs.detach().cpu().numpy()    # 物品嵌入 [n_items, dim*3]
        KG_mask = KG_mask.detach().cpu().numpy()         # APL掩码
        log_cuda_mem(f"epoch{epoch}.after_generate_to_cpu", device, args.mem_debug)

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

    # ===== 显存峰值统计 =====
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        print(f'[GPU Memory] 训练期间显存峰值: {peak_mem_mb:.1f} MB ({peak_mem_gb:.2f} GB)')
        print(f'[GPU Memory] 最低显卡显存需求 (建议预留约20%余量): {peak_mem_gb * 1.2:.2f} GB')

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
