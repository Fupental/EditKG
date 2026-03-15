# -*- coding: utf-8 -*-
"""
==========================================================================
data_loader.py — 数据加载与图构建模块
==========================================================================
功能说明：
    负责所有数据的读取、预处理和图结构的构建，是整个系统的数据基础。
    
    核心功能：
    1. read_cf(): 读取用户-物品交互数据（train.txt / test.txt）
    2. remap_item(): 统计用户/物品数量，构建用户交互集合
    3. read_triplets(): 读取知识图谱三元组（kg_final.txt），构建 KG 字典
    4. build_graph(): 构建协同知识图（CKG = CF图 + KG）
    5. build_sparse_graph(): 构建稀疏邻接矩阵，用于 GNN 的消息传递
    6. build_kg_set(): 构建物品-关系掩码，标记每个物品缺失的属性类型
    7. load_data(): 统一调度以上所有函数

文件角色：
    被 main.py 调用 load_data() 加载所有数据，返回值作为模型构建的输入。

数据文件格式：
    - train.txt / test.txt: 每行 "用户ID 物品ID1 物品ID2 ..."
    - kg_final.txt: 每行 "头实体 关系 尾实体"
==========================================================================
"""

import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import os

# ========== 全局变量：数据规模统计 ==========
n_users = 0       # 用户总数
n_items = 0       # 物品总数
n_entities = 0    # 实体总数（物品 + KG中的其他实体）
n_relations = 0   # 关系总数（KG中的关系类型数 + 'interact'关系）
n_nodes = 0       # 节点总数（n_entities + n_users）

# 用户交互字典
train_user_set = defaultdict(list)  # {用户ID: [训练集中交互过的物品ID列表]}
test_user_set = defaultdict(list)   # {用户ID: [测试集中交互过的物品ID列表]}


def read_cf(file_name):
    """
    读取用户-物品交互数据文件（协同过滤数据）
    
    文件格式：每行 "用户ID 物品ID1 物品ID2 ..."
    例如：0 12 34 56 表示用户0与物品12、34、56有交互
    
    参数:
        file_name: 数据文件路径（如 train.txt 或 test.txt）
    
    返回:
        numpy数组，形状 [N, 2]，每行是一个 [用户ID, 物品ID] 交互对
    """
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]  # 第一个元素是用户ID，后面都是物品ID
        pos_ids = list(set(pos_ids))  # 去重
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])  # 展开为逐条交互记录

    return np.array(inter_mat)

 
def remap_item(train_data, test_data):
    """
    统计用户和物品的总数，并构建用户交互集合字典。
    
    注意：这里的物品ID和用户ID都是从原始数据中直接获取的，
    物品ID空间是 [0, n_items)，在构建图时会偏移到 [n_users, n_users+n_items)。
    
    参数:
        train_data: 训练集交互对数组 [N_train, 2]
        test_data: 测试集交互对数组 [N_test, 2]
    """
    global n_users, n_items
    # 通过最大值+1确定用户和物品的总数
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    # 构建用户交互字典：{用户ID: [物品ID列表]}
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    """
    读取知识图谱三元组数据（kg_final.txt）
    
    关键处理：
    1. 可选地添加逆向三元组（inverse_r参数控制）
       → 例如 (item, has-genre, Action) 生成逆向 (Action, is-genre-of, item)
    2. 所有关系ID +1，为关系ID=0 留出"interact"交互关系
       → 这样 CF图和KG可以统一到一个图中（关系0=用户交互）
    
    参数:
        file_name: KG文件路径（kg_final.txt）
    
    返回:
        triplets: 处理后的三元组数组 [N, 3]，每行 [头实体, 关系, 尾实体]
        kg_dict: KG字典 {头实体: [(关系, 尾实体), ...]}
    """
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)  # 去除重复三元组

    if args.inverse_r:
        # 添加逆向三元组：交换头尾实体，关系ID偏移
        # 例如 (A, r, B) → (B, r+max_r+1, A)
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]  # 头尾交换
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1  # 逆向关系ID偏移
        # 所有关系ID +1，为'interact'关系（ID=0）腾出位置
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # 合并正向和逆向三元组
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # 仅添加'interact'关系偏移
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    # 统计实体、节点和关系的总数
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # 实体总数（含物品和KG实体）
    n_nodes = n_entities + n_users   # 总节点数 = 实体数 + 用户数
    n_relations = max(triplets[:, 1]) + 1  # 关系总数

    def _get_kg_dict(triplets):
        """将三元组数组转换为字典格式：{头实体: [(关系, 尾实体), ...]}"""
        kg_dict = defaultdict(list)
        for h, r, t in triplets:
            kg_dict[h].append((r, t))
        return kg_dict
    
    kg_dict = _get_kg_dict(triplets)
    return triplets, kg_dict


def generate_polluted_cf_data(train_cf, rate):
    """
    生成带噪声的CF数据（用于鲁棒性实验）
    
    将 rate 比例的交互替换为随机负采样的物品，模拟噪声数据。
    这个函数在论文的鲁棒性实验（Section 4.4）中使用。
    
    参数:
        train_cf: 原始训练集交互对
        rate: 噪声比例（如 0.2 表示替换20%的交互）
    """
    index = np.arange(len(train_cf))
    np.random.shuffle(index)
    train_cf = train_cf[index]

    n_noise = int(len(train_cf) * rate)      # 需要替换的交互数量
    train_cf_noise = train_cf[:n_noise]       # 待替换的交互
    train_cf_ori = train_cf[n_noise:]         # 保留的原始交互

    # 将噪声部分的物品替换为用户没有交互过的随机物品
    train_total = []
    for u, i in train_cf_noise:
        while 1:
            n = np.random.randint(low=0, high=n_items, size=1)[0]
            if n not in train_user_set[u]:
                train_total.append([u, n])
                break
    train_total = np.vstack((np.array(train_total), train_cf_ori))

    # 输出为文件
    train_dict = defaultdict(list)
    for u, i in train_total:
        train_dict[int(u)].append(int(i))

    f = open('./data/{}/train_noise_{}.txt'.format(args.dataset, rate), 'w')
    for key, val in train_dict.items():
        val = [key] + val
        val = ' '.join(str(x) for x in val)
        val = val + '\n'
        f.write(val)
    f.close()


def generate_polluted_kg_data(file_name, rate):
    """
    生成带噪声的KG数据（用于鲁棒性实验）
    
    将 rate 比例的三元组尾实体替换为随机实体，模拟KG噪声。
    
    参数:
        file_name: 原始KG文件路径
        rate: 噪声比例
    """
    triplets_np = np.loadtxt(file_name, dtype=np.int32)
    triplets_np = np.unique(triplets_np, axis=0)

    tri_dict = defaultdict(list)
    for h, r, t in triplets_np:
        tri_dict[int(h)].append(int(t))

    index = np.arange(len(triplets_np))
    np.random.shuffle(index)
    triplets_np = triplets_np[index]

    n_noise = int(len(triplets_np) * rate)
    triplets_np_noise = triplets_np[:n_noise]
    triplets_np_ori = triplets_np[n_noise:]

    # 将噪声部分的尾实体替换为随机实体
    triplets_np_total = []
    for h, r, t in triplets_np_noise:
        while 1:
            n = np.random.randint(low=0, high=n_entities, size=1)[0]
            if n not in tri_dict[h]:
                triplets_np_total.append([h, r, n])
                break
    triplets_np_total = np.vstack((np.array(triplets_np_total), triplets_np_ori))

    f = open('./data/{}/kg_noise_{}.txt'.format(args.dataset, rate), 'w')
    for h, r, t in triplets_np_total:
        f.write(str(h) + ' ' + str(r) + ' ' + str(t) + '\n')
    f.close()


def build_graph(train_data, triplets):
    """
    构建协同知识图（CKG = CF图 + KG）
    
    CKG 统一了用户-物品交互和知识图谱：
    - 关系0 = "interact"（用户-物品交互边）
    - 关系1,2,... = KG中的各种属性关系
    
    参数:
        train_data: 训练集交互对 [N, 2]
        triplets: KG三元组 [M, 3]
    
    返回:
        ckg_graph: NetworkX有向多重图（当前代码中实际未使用此图对象）
        rd: 关系字典 {关系ID: [[头, 尾], ...]}，用于后续构建稀疏矩阵
    """
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    # 用户-物品交互边，关系ID = 0
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    # KG三元组边
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_graph(relation_dict):
    """
    构建稀疏邻接矩阵，用于 GNN 的稀疏矩阵乘法消息传递。
    
    包含两个归一化方法（代码中注释了对称归一化，使用左归一化）：
    - D^{-1/2}AD^{-1/2}: 对称归一化（注释掉的 _bi_norm_lap）
    - D^{-1}A: 左归一化（当前使用的 _si_norm_lap）
    
    参数:
        relation_dict: {关系ID: [[头, 尾], ...]}
    
    返回:
        ui_mat_list: 用户-物品交互的稀疏矩阵 [n_users × n_items]
        adj_mat_list: 所有关系的邻接矩阵列表
    
    关键细节：
        - 关系0（interact）的物品ID需要偏移n_users，因为在统一图中
          用户占 [0, n_users)，物品占 [n_users, n_users+n_items)
    """
    # 注释掉的对称归一化方法：D^{-1/2}AD^{-1/2}
    # def _bi_norm_lap(adj):
    #     rowsum = np.array(adj.sum(1))
    #     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #     bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    #     return bi_lap.tocoo()

    def _si_norm_lap(adj):
        """左归一化：D^{-1}A，即每行除以其度数"""
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.  # 处理度为0的节点（避免除零）
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []

    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            # 关系0是用户-物品交互（interact关系）
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # 物品ID偏移：[0, n_items) → [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            # KG关系（实体-实体边）
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)
    
    # 提取用户-物品交互的子矩阵 [n_users × n_items]
    # 从完整邻接矩阵中切片出用户行、物品列
    ui_mat_list = adj_mat_list[0].tocsr()[:n_users, n_users:n_users + n_items].tocoo()
    return ui_mat_list, adj_mat_list


def build_kg_set(triplets):
    """
    构建物品-关系掩码矩阵，标记每个物品缺失了哪些关系类型。
    
    这是 Knowledge Deleter 中 APL（属性净化层）的辅助数据：
    item_rel_mask_rev[item][relation] = 1 表示该物品不具有该关系类型
    → 用于判断哪些属性是"可能需要补充的"
    
    参数:
        triplets: KG三元组数组
    
    返回:
        item_rel_mask_rev: 反转后的物品-关系掩码 [n_items × n_relations]
            值为1表示该物品缺少该关系，值为0表示已有
    
    缓存机制：首次计算后保存为 item_rel_mask_rev.pkl，后续直接加载
    """
    if os.path.exists("item_rel_mask_rev.pkl"):
        item_rel_mask_rev = pkl.load(open("item_rel_mask_rev.pkl", "rb"))
    else:
        # 构建物品-关系掩码：item_rel_mask[item][relation] = 1 表示物品拥有该关系
        item_rel_mask = np.zeros((n_items, n_relations))
        for item in tqdm(range(n_items)):
            # 找出该物品出现过的所有关系类型
            item_rel = triplets[triplets[:, 0] == item][:, 1]
            item_rel = np.unique(item_rel)
            item_rel_map = item_rel_mask[item]
            item_rel_map[item_rel] = 1
            if sum(item_rel_map) == 0:
                pass  # 该物品没有任何KG关系（知识匮乏的物品）
            item_rel_mask[item] = item_rel_map
        
        # 取反：1变0，0变1 → 标记物品缺失的关系
        item_rel_mask_rev = (~(item_rel_mask > 0)).astype("float")
        item_rel_mask_rev[:, 0] = 0  # 关系0是"interact"，不视为缺失属性
        pkl.dump(item_rel_mask_rev, open("item_rel_mask_rev.pkl", "wb"))
    return item_rel_mask_rev


def static_kg(triplets):
    """
    KG统计分析函数（调试用途）
    
    统计三元组中：
    - 头尾都不是物品的三元组数
    - 头尾都是物品的三元组数
    - 只有一端是物品的三元组数
    """
    head = triplets[:, 0]
    tail = triplets[:, 2]
    cnt_both_non = 0   # 头尾都不是物品
    cnt_both_in = 0    # 头尾都是物品
    cnt_one_in = 0     # 仅一端是物品
    for i in range(len(head)):
        if head[i] > n_items and tail[i] > n_items:
            cnt_both_non += 1
        elif head[i] < n_items and tail[i] < n_items:
            cnt_both_in += 1
        else:
            cnt_one_in += 1


def load_multimodal_features(data_dir):
    """
    加载预提取的多模态特征（视觉 + 文本）。
    
    参考 R2MR 的 abstract_recommender.py 中的特征加载方式：
    - 从 .npy 文件加载预提取特征
    - 对视觉特征做 PCA 降维
    - 对两种特征都做 tanh 激活
    
    参数:
        data_dir: 数据集目录路径（如 data/amazon-book/）
    
    返回:
        v_feat: 视觉特征 numpy 数组 [n_items, feat_dim]，不存在则返回 None
        t_feat: 文本特征 numpy 数组 [n_items, feat_dim]，不存在则返回 None
    """
    from sklearn.decomposition import PCA
    
    v_feat = None
    t_feat = None
    
    v_feat_path = os.path.join(data_dir, 'image_feat.npy')
    t_feat_path = os.path.join(data_dir, 'text_feat.npy')
    
    if os.path.isfile(v_feat_path):
        v_feat = np.load(v_feat_path, allow_pickle=True)
        print(f"[多模态] 加载视觉特征: {v_feat_path}, 原始形状: {v_feat.shape}")
        # 参考 R2MR：对视觉特征做 PCA 降维到 384 维（= dim*3 = 128*3，可直接与 ID 嵌入相加）
        v_zero_mask = np.all(v_feat == 0, axis=1)  # 标记原始零向量行
        if v_feat.shape[1] > 384:
            v_pca = PCA(n_components=384)
            v_feat = v_pca.fit_transform(v_feat)
            print(f"[多模态]   PCA 降维后形状: {v_feat.shape},  explained_var_ratio 前5: {v_pca.explained_variance_ratio_[:5]}")
        v_feat = np.tanh(v_feat)  # tanh 激活（参考 R2MR）
        v_feat[v_zero_mask] = 0.0  # 恢复零向量行，避免PCA引入噪声
        # ---------- 调试信息 ----------
        nonzero = np.sum(np.any(v_feat != 0, axis=1))
        print(f"[多模态]   tanh后 → 形状: {v_feat.shape}, 有效行: {nonzero}/{v_feat.shape[0]}, "
              f"均值: {v_feat.mean():.6f}, 标准差: {v_feat.std():.6f}, "
              f"范数(前3): {[f'{np.linalg.norm(v_feat[i]):.4f}' for i in range(min(3, len(v_feat)))]}")
    else:
        print(f"[多模态] 未找到视觉特征文件: {v_feat_path}，跳过")
    
    if os.path.isfile(t_feat_path):
        t_feat = np.load(t_feat_path, allow_pickle=True)
        print(f"[多模态] 加载文本特征: {t_feat_path}, 原始形状: {t_feat.shape}")
        # 参考 R2MR：对文本特征做 PCA 降维到 384 维（= dim*3 = 128*3）
        t_zero_mask = np.all(t_feat == 0, axis=1)  # 标记原始零向量行
        if t_feat.shape[1] > 384:
            t_pca = PCA(n_components=384)
            t_feat = t_pca.fit_transform(t_feat)
            print(f"[多模态]   PCA 降维后形状: {t_feat.shape},  explained_var_ratio 前5: {t_pca.explained_variance_ratio_[:5]}")
        t_feat = np.tanh(t_feat)
        t_feat[t_zero_mask] = 0.0  # 恢复零向量行，避免PCA引入噪声
        # ---------- 调试信息 ----------
        nonzero = np.sum(np.any(t_feat != 0, axis=1))
        print(f"[多模态]   tanh后 → 形状: {t_feat.shape}, 有效行: {nonzero}/{t_feat.shape[0]}, "
              f"均值: {t_feat.mean():.6f}, 标准差: {t_feat.std():.6f}, "
              f"范数(前3): {[f'{np.linalg.norm(t_feat[i]):.4f}' for i in range(min(3, len(t_feat)))]}")
    else:
        print(f"[多模态] 未找到文本特征文件: {t_feat_path}，跳过")
    
    return v_feat, t_feat


def load_data(model_args):
    """
    主数据加载函数 —— 统一调度所有数据读取和预处理。
    
    这是 main.py 调用的入口函数，依次执行：
    1. 读取用户-物品交互数据（train.txt, test.txt）
    2. 统计用户/物品数量
    3. 读取知识图谱三元组
    4. 构建物品-关系掩码
    5. 构建CKG图和关系字典
    6. 构建稀疏邻接矩阵
    7. 加载多模态特征（视觉+文本）
    
    参数:
        model_args: 命令行参数对象（来自 parser.py）
    
    返回:
        train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph,
        all_sparse_graph, item_rel_mask, triplets, kg_dict, v_feat, t_feat
    """
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    # 第一步：读取交互数据
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    # 第二步：读取KG和构建辅助数据
    triplets, kg_dict = read_triplets(directory + 'kg_final.txt')
    static_kg(triplets)
    item_rel_mask = build_kg_set(triplets)

    # 第三步：构建图结构
    graph, relation_dict = build_graph(train_cf, triplets)

    # 第四步：构建稀疏矩阵
    ui_sparse_graph, all_sparse_graph = build_sparse_graph(relation_dict)
    
    # 封装数据规模参数
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    # 封装用户交互字典
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    # ===== 加载多模态特征（如果存在）=====
    v_feat, t_feat = load_multimodal_features(directory)

    return train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, all_sparse_graph, item_rel_mask, triplets, kg_dict, v_feat, t_feat
