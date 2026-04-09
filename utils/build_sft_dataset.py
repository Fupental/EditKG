# -*- coding: utf-8 -*-
"""
build_sft_dataset.py — 构建 LLM SFT 微调数据集 (llm_sft_xujia)

数据组成：
  - 30万条正样本 (从 kg_final 随机抽取真实三元组)
  - 40万条负样本:
      - 16万条简单负样本 (随机替换头/关系/尾)
      - 24万条困难负样本 (用 Qwen3-4B embedding 找语义相似实体替换)

使用方式:
    python utils/build_sft_dataset.py --dataset amazon-book
"""
import os
import sys
import json
import random
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    from utils.path_utils import default_base_model_path, default_dataset_root, resolve_dataset_dir
except ImportError:
    from path_utils import default_base_model_path, default_dataset_root, resolve_dataset_dir

SYSTEM_PROMPT = (
    "You are a knowledge graph validation expert. "
    "Given the following statement about a book or literary work, "
    "determine if it describes a factually correct relationship.\n"
    "Answer only True or False."
)

NUM_POSITIVE = 300_000
NUM_NEG_SIMPLE = 160_000
NUM_NEG_HARD = 240_000
SEED = 2023


def load_entity_names(data_dir):
    names = {}
    with open(os.path.join(data_dir, 'entity_list.txt')) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and parts[1].strip().isdigit():
                eid = int(parts[1])
                name = parts[2].strip()
                if name:
                    names[eid] = name
    print(f"[Entity] 加载 {len(names)} 个有名称的实体")
    return names


def load_relation_names(data_dir):
    rel_names = {}
    with open(os.path.join(data_dir, 'relation_list.txt')) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                uri, rid = parts[0], int(parts[1])
                short = uri.split('/')[-1]
                if '#' in short:
                    short = short.split('#')[-1]
                rel_names[rid] = short
    print(f"[Relation] 加载 {len(rel_names)} 个关系")
    return rel_names


def load_relation_templates():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from relation_templates import REL_TEMPLATES
    except ImportError:
        print("ERROR: 无法导入 relation_templates.py")
        sys.exit(1)
    print(f"[Templates] 加载 {len(REL_TEMPLATES)} 个关系模板")
    return REL_TEMPLATES


def load_kg(data_dir):
    triplets = []
    triplet_set = set()
    with open(os.path.join(data_dir, 'kg_final.txt')) as f:
        for line in f:
            h, r, t = map(int, line.strip().split())
            triplets.append((h, r, t))
            triplet_set.add((h, r, t))
    print(f"[KG] 加载 {len(triplets)} 条三元组")
    return triplets, triplet_set


def make_nl_statement(h, r, t, entity_names, rel_names, templates):
    h_name = entity_names.get(h, f"entity_{h}")
    t_name = entity_names.get(t, f"entity_{t}")
    r_short = rel_names.get(r, f"relation_{r}")
    template = templates.get(r_short)
    if template:
        try:
            return template.format(head=h_name, tail=t_name)
        except (KeyError, IndexError):
            pass
    r_readable = r_short.replace('.', ' > ').replace('_', ' ')
    return f'"{h_name}" has the relationship "{r_readable}" with "{t_name}"'


def make_sft_sample(statement, label):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": statement},
            {"role": "assistant", "content": "True" if label else "False"}
        ]
    }


def generate_positives(triplets, entity_names, rel_names, templates, n):
    valid = [(h, r, t) for h, r, t in triplets
             if h in entity_names and t in entity_names]
    print(f"[正样本] 可用三元组: {len(valid)}, 需要: {n}")
    sampled = random.sample(valid, min(n, len(valid)))
    samples = []
    for h, r, t in tqdm(sampled, desc="生成正样本"):
        stmt = make_nl_statement(h, r, t, entity_names, rel_names, templates)
        samples.append(make_sft_sample(stmt, True))
    return samples, sampled


def generate_simple_negatives(pos_triplets, triplet_set, entity_names,
                               rel_names, templates, n):
    entity_ids = list(entity_names.keys())
    rel_ids = list(rel_names.keys())
    samples = []

    for _ in tqdm(range(n), desc="生成简单负样本"):
        h, r, t = random.choice(pos_triplets)
        mode = random.choice(['head', 'tail', 'relation'])
        for _ in range(50):
            if mode == 'head':
                new_h = random.choice(entity_ids)
                candidate = (new_h, r, t)
            elif mode == 'tail':
                new_t = random.choice(entity_ids)
                candidate = (h, r, new_t)
            else:
                new_r = random.choice(rel_ids)
                candidate = (h, new_r, t)
            if candidate not in triplet_set:
                nh, nr, nt = candidate
                if nh in entity_names and nt in entity_names:
                    stmt = make_nl_statement(nh, nr, nt, entity_names,
                                             rel_names, templates)
                    samples.append(make_sft_sample(stmt, False))
                    break

    print(f"[简单负样本] 生成 {len(samples)} 条")
    return samples


def compute_entity_embeddings(entity_names, model_path, batch_size=256):
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"[Embedding] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    eids = sorted(entity_names.keys())
    names = [entity_names[eid] for eid in eids]
    embeddings = []

    print(f"[Embedding] 计算 {len(names)} 个实体的 embedding...")
    for i in tqdm(range(0, len(names), batch_size), desc="Embedding"):
        batch = names[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(pooled.cpu().float().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    print(f"[Embedding] shape: {embeddings.shape}")

    del model
    torch.cuda.empty_cache()

    return eids, embeddings


def build_similarity_index(eids, embeddings, top_k=20):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"[FAISS] 搜索 top-{top_k} 相似实体...")
    batch_size = 10000
    all_neighbors = {}

    for i in tqdm(range(0, len(eids), batch_size), desc="FAISS搜索"):
        batch_emb = embeddings[i:i + batch_size].astype(np.float32)
        _, indices = index.search(batch_emb, top_k + 1)
        for j, idx_row in enumerate(indices):
            eid = eids[i + j]
            neighbors = [eids[k] for k in idx_row if eids[k] != eid][:top_k]
            all_neighbors[eid] = neighbors

    return all_neighbors


def generate_hard_negatives(pos_triplets, triplet_set, entity_names,
                             rel_names, templates, neighbors, n):
    samples = []

    for _ in tqdm(range(n), desc="生成困难负样本"):
        h, r, t = random.choice(pos_triplets)
        mode = random.choice(['head', 'tail'])

        for _ in range(50):
            if mode == 'head' and h in neighbors and neighbors[h]:
                new_h = random.choice(neighbors[h])
                candidate = (new_h, r, t)
            elif mode == 'tail' and t in neighbors and neighbors[t]:
                new_t = random.choice(neighbors[t])
                candidate = (h, r, new_t)
            else:
                if mode == 'head':
                    new_h = random.choice(list(entity_names.keys()))
                    candidate = (new_h, r, t)
                else:
                    new_t = random.choice(list(entity_names.keys()))
                    candidate = (h, r, new_t)

            if candidate not in triplet_set:
                nh, nr, nt = candidate
                if nh in entity_names and nt in entity_names:
                    stmt = make_nl_statement(nh, nr, nt, entity_names,
                                             rel_names, templates)
                    samples.append(make_sft_sample(stmt, False))
                    break

    print(f"[困难负样本] 生成 {len(samples)} 条")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--data_path", default=default_dataset_root())
    parser.add_argument("--model_path",
                        default=default_base_model_path())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_pos", type=int, default=NUM_POSITIVE)
    parser.add_argument("--num_neg_simple", type=int, default=NUM_NEG_SIMPLE)
    parser.add_argument("--num_neg_hard", type=int, default=NUM_NEG_HARD)
    parser.add_argument("--embedding_batch", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = resolve_dataset_dir(args.data_path, args.dataset)
    out_dir = os.path.join(data_dir, "llm_sft_xujia")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("阶段1: 加载数据")
    print("=" * 60)
    entity_names = load_entity_names(data_dir)
    rel_names = load_relation_names(data_dir)
    templates = load_relation_templates()
    triplets, triplet_set = load_kg(data_dir)

    print("\n" + "=" * 60)
    print("阶段2: 生成正样本")
    print("=" * 60)
    pos_samples, pos_triplets = generate_positives(
        triplets, entity_names, rel_names, templates, args.num_pos)

    print("\n" + "=" * 60)
    print("阶段3: 生成简单负样本")
    print("=" * 60)
    simple_neg_samples = generate_simple_negatives(
        pos_triplets, triplet_set, entity_names, rel_names, templates,
        args.num_neg_simple)

    print("\n" + "=" * 60)
    print("阶段4: 计算实体 embedding (Qwen3-4B)")
    print("=" * 60)
    eids, embeddings = compute_entity_embeddings(
        entity_names, args.model_path, args.embedding_batch)

    emb_path = os.path.join(out_dir, "entity_embeddings.npy")
    np.save(emb_path, embeddings)
    eid_path = os.path.join(out_dir, "entity_eids.npy")
    np.save(eid_path, np.array(eids))
    print(f"[保存] embedding -> {emb_path}")

    print("\n" + "=" * 60)
    print("阶段5: 构建相似度索引 & 生成困难负样本")
    print("=" * 60)
    import faiss
    neighbors = build_similarity_index(eids, embeddings, args.top_k)

    hard_neg_samples = generate_hard_negatives(
        pos_triplets, triplet_set, entity_names, rel_names, templates,
        neighbors, args.num_neg_hard)

    print("\n" + "=" * 60)
    print("阶段6: 合并并保存数据集")
    print("=" * 60)

    all_samples = pos_samples + simple_neg_samples + hard_neg_samples
    random.shuffle(all_samples)

    n_true = sum(1 for s in all_samples
                 if s["messages"][-1]["content"] == "True")
    n_false = len(all_samples) - n_true
    print(f"[总计] {len(all_samples)} 条: True={n_true}, False={n_false}")

    n_val = int(len(all_samples) * args.val_ratio)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n[保存] train: {len(train_samples)} -> {train_path}")
    print(f"[保存] val:   {len(val_samples)} -> {val_path}")

    stats = {
        "total": len(all_samples),
        "positive": len(pos_samples),
        "negative_simple": len(simple_neg_samples),
        "negative_hard": len(hard_neg_samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "seed": args.seed,
    }
    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[保存] stats -> {stats_path}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
