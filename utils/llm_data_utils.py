# -*- coding: utf-8 -*-
"""
llm_data_utils.py — LLM训练/推理数据生成
  模板来自 relation_templates.py（全部39个关系均有模板）
  简单负样本(90%): 随机替换头/尾/关系
  困难负样本(10%): 由LLM生成对抗性替换（见 generate_hard_negatives.py）
"""
import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

try:
    from utils.relation_templates import REL_TEMPLATES
except ImportError:
    from relation_templates import REL_TEMPLATES

# 22-rdf-syntax-ns#type 视为 type.object.type
RDF_TYPE_ALIAS = '22-rdf-syntax-ns#type'
RDF_TYPE_TARGET = 'type.object.type'

SYSTEM_PROMPT = (
    "You are a knowledge graph validation expert. "
    "Given the following statement about a book or literary work, "
    "determine if it describes a factually correct relationship.\n"
    "Answer only True or False."
)


# ======== 数据加载 ========

def load_item_meta(data_dir):
    item_ids = set()
    isbn_map = {}
    with open(os.path.join(data_dir, 'item_list.txt')) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                isbn, rid = parts[0], int(parts[1])
                item_ids.add(rid)
                isbn_map[isbn] = rid
    isbn_to_meta = {}
    meta_path = os.path.join(data_dir, 'meta_Books.jsonl')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = m.get('parent_asin', '')
                if asin and asin in isbn_map:
                    isbn_to_meta[asin] = m
    print(f"[Meta] item数: {len(item_ids)}, meta匹配: {len(isbn_to_meta)}")
    return item_ids, isbn_map, isbn_to_meta


def load_relations(data_dir):
    """加载关系，不再跳过任何关系。将22-rdf-syntax-ns#type映射为type.object.type。"""
    rel_names = {}
    with open(os.path.join(data_dir, 'relation_list.txt')) as f:
        next(f)
        for line in f:
            parts = line.strip().rsplit(maxsplit=1)
            if len(parts) == 2:
                org, rid = parts[0], int(parts[1])
                short = org.split('/')[-1] if '/' in org else org
                # 22-rdf-syntax-ns#type → type.object.type
                if short == RDF_TYPE_ALIAS:
                    short = RDF_TYPE_TARGET
                rel_names[rid] = short
    print(f"[Rel] 总关系: {len(rel_names)}, 全部启用（无跳过）")
    return rel_names


def load_kg(data_dir):
    triplets = []
    with open(os.path.join(data_dir, 'kg_final.txt')) as f:
        for line in f:
            h, r, t = [int(x) for x in line.strip().split()]
            triplets.append((h, r, t))
    print(f"[KG] 三元组: {len(triplets)}")
    return triplets


def build_entity_names(item_ids, isbn_map, isbn_to_meta, triplets, rel_names, data_dir):
    """
    为实体赋予可读名称。来源优先级：
    1. meta_Books.jsonl 中的 title（item实体）
    2. meta 中推断的 author/genre/subject 名
    3. entity_list.txt 中的原始名（非 m.xxx 的实体，如类型名、XMLSchema等）
    """
    entity_names = {}

    # === 来源1：meta title ===
    for isbn, rid in isbn_map.items():
        if isbn in isbn_to_meta:
            title = isbn_to_meta[isbn].get('title', '')
            if title:
                entity_names[rid] = title

    # === 来源2：meta推断的author/genre/subject ===
    item_to_authors = defaultdict(set)
    item_to_genres = defaultdict(set)
    item_to_subjects = defaultdict(set)
    rid_to_isbn = {v: k for k, v in isbn_map.items()}

    for h, r, t in triplets:
        rn = rel_names.get(r, '')
        if rn == 'book.written_work.author' and h in item_ids:
            isbn = rid_to_isbn.get(h)
            if isbn: item_to_authors[isbn].add(t)
        elif rn == 'book.author.works_written' and t in item_ids:
            isbn = rid_to_isbn.get(t)
            if isbn: item_to_authors[isbn].add(h)
        if rn in ('book.book.genre', 'book.short_story.genre', 'theater.play.genre') and h in item_ids:
            isbn = rid_to_isbn.get(h)
            if isbn: item_to_genres[isbn].add(t)
        elif rn in ('media_common.literary_genre.books_in_this_genre',
                     'media_common.literary_genre.stories_in_this_genre',
                     'theater.theater_genre.plays_in_this_genre') and t in item_ids:
            isbn = rid_to_isbn.get(t)
            if isbn: item_to_genres[isbn].add(h)
        if rn == 'book.written_work.subjects' and h in item_ids:
            isbn = rid_to_isbn.get(h)
            if isbn: item_to_subjects[isbn].add(t)
        elif rn == 'book.book_subject.works' and t in item_ids:
            isbn = rid_to_isbn.get(t)
            if isbn: item_to_subjects[isbn].add(h)

    author_named, genre_named, subject_named = 0, 0, 0
    for isbn, meta in isbn_to_meta.items():
        author_name = None
        if meta.get('author') and isinstance(meta['author'], dict):
            author_name = meta['author'].get('name', '')
        elif meta.get('store'):
            s = meta['store']
            for suffix in [' (Author)', ' (Editor)', ' (Translator)', ' (Narrator)']:
                if suffix in s:
                    author_name = s.split(suffix)[0].strip()
                    break
        if author_name and isbn in item_to_authors:
            authors = item_to_authors[isbn]
            if len(authors) == 1:
                eid = next(iter(authors))
                if eid not in entity_names:
                    entity_names[eid] = author_name
                    author_named += 1

        cats = meta.get('categories', [])
        if cats and len(cats) > 1:
            real_cats = [c for c in cats[1:] if c and c != 'Books']
            if isbn in item_to_genres:
                genres = item_to_genres[isbn]
                if len(genres) == 1 and real_cats:
                    eid = next(iter(genres))
                    if eid not in entity_names:
                        entity_names[eid] = real_cats[-1]
                        genre_named += 1
                elif len(genres) <= len(real_cats):
                    for i, eid in enumerate(sorted(genres)):
                        if eid not in entity_names and i < len(real_cats):
                            entity_names[eid] = real_cats[-(i+1)]
                            genre_named += 1
            if isbn in item_to_subjects:
                subjects = item_to_subjects[isbn]
                if len(subjects) == 1 and real_cats:
                    eid = next(iter(subjects))
                    if eid not in entity_names:
                        entity_names[eid] = real_cats[0]
                        subject_named += 1

    # === 来源3：entity_list.txt（3列格式：org_id\tremap_id\tname）===
    # 第3列有 Wikidata 解析出的可读名称，直接使用
    entity_list_named = 0
    entity_list_path = os.path.join(data_dir, 'entity_list.txt')
    if os.path.exists(entity_list_path):
        with open(entity_list_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[1].strip().isdigit():
                    eid = int(parts[1])
                    name = parts[2].strip()
                    if name and eid not in entity_names:
                        entity_names[eid] = name
                        entity_list_named += 1
                elif len(parts) >= 2 and parts[1].strip().isdigit():
                    # 兼容旧的2列格式
                    eid = int(parts[1])
                    orig_name = parts[0]
                    if eid not in entity_names and not orig_name.startswith('m.'):
                        clean = orig_name.strip('"').strip()
                        if clean.endswith('"@en'):
                            clean = clean[:-4].strip('"')
                        if clean:
                            entity_names[eid] = clean
                            entity_list_named += 1

    n_item = sum(1 for e in entity_names if e in item_ids)
    n_non_item = len(entity_names) - n_item
    print(f"[Entity] item命名: {n_item}/{len(item_ids)}, "
          f"非item命名: {n_non_item} "
          f"(author:{author_named}, genre:{genre_named}, subject:{subject_named}, "
          f"entity_list:{entity_list_named})")
    return entity_names


def make_nl_statement(h, r, t, entity_names, rel_names):
    rn = rel_names.get(r)
    if rn not in REL_TEMPLATES:
        return None
    tpl = REL_TEMPLATES[rn]
    if not tpl:  # 模板为空字符串的关系仍然跳过
        return None
    h_name = entity_names.get(h)
    t_name = entity_names.get(t)
    if not h_name or not t_name:
        return None
    return tpl.format(head=h_name, tail=t_name)


# ======== 简单负样本 ========

def generate_easy_negative(h, r, t, rel_heads, rel_tails, entity_names,
                           rel_names, triplet_set):
    """简单负样本：随机替换头/尾/关系"""
    replace = random.choice(['head', 'tail', 'relation'])
    for _ in range(10):
        if replace == 'tail':
            pool = list(rel_tails.get(r, set()))
            if len(pool) < 2: return None
            new_t = random.choice(pool)
            if new_t != t and (h, r, new_t) not in triplet_set:
                return make_nl_statement(h, r, new_t, entity_names, rel_names)
        elif replace == 'head':
            pool = list(rel_heads.get(r, set()))
            if len(pool) < 2: return None
            new_h = random.choice(pool)
            if new_h != h and (new_h, r, t) not in triplet_set:
                return make_nl_statement(new_h, r, t, entity_names, rel_names)
        elif replace == 'relation':
            valid_rels = [rid for rid in rel_tails if rid != r
                          and rel_names.get(rid) in REL_TEMPLATES
                          and REL_TEMPLATES.get(rel_names.get(rid), '') != '']
            if not valid_rels: return None
            new_r = random.choice(valid_rels)
            if (h, new_r, t) not in triplet_set:
                return make_nl_statement(h, new_r, t, entity_names, rel_names)
    return None


# ======== 训练数据构建 ========

def build_train_data(data_dir, output_dir, max_samples=25000, seed=2023,
                     hard_neg_path=None, hard_neg_ratio=0.1):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    item_ids, isbn_map, isbn_to_meta = load_item_meta(data_dir)
    rel_names = load_relations(data_dir)
    triplets = load_kg(data_dir)
    entity_names = build_entity_names(item_ids, isbn_map, isbn_to_meta, triplets, rel_names, data_dir)

    # 筛选：关系有模板且非空，头尾都有名字
    named_triplets = [(h, r, t) for h, r, t in triplets
                      if rel_names.get(r) in REL_TEMPLATES
                      and REL_TEMPLATES.get(rel_names.get(r), '') != ''
                      and h in entity_names and t in entity_names]
    print(f"可用三元组: {len(named_triplets)}")
    triplet_set = set(triplets)

    # 按关系构建实体池
    rel_heads = defaultdict(set)
    rel_tails = defaultdict(set)
    for h, r, t in named_triplets:
        rel_heads[r].add(h)
        rel_tails[r].add(t)

    selected = random.sample(named_triplets, min(max_samples, len(named_triplets)))

    positives = []
    easy_negatives = []

    for h, r, t in tqdm(selected, desc="构建正样本+简单负样本"):
        stmt = make_nl_statement(h, r, t, entity_names, rel_names)
        if not stmt:
            continue
        positives.append({"role": "pos", "input": stmt, "output": "True"})
        easy = generate_easy_negative(h, r, t, rel_heads, rel_tails,
                                      entity_names, rel_names, triplet_set)
        if easy:
            easy_negatives.append({"role": "neg_easy", "input": easy, "output": "False"})

    # 加载LLM困难负样本
    hard_negatives = []
    if hard_neg_path and os.path.exists(hard_neg_path):
        with open(hard_neg_path, 'r', encoding='utf-8') as f:
            for line in f:
                hard_negatives.append(json.loads(line.strip()))
        print(f"[困难负样本] 加载: {len(hard_negatives)} from {hard_neg_path}")
    else:
        print(f"[困难负样本] 未找到文件，仅使用简单负样本")

    # 按比例混合
    n_pos = len(positives)
    n_hard_target = int(n_pos * hard_neg_ratio) if hard_negatives else 0
    n_easy_target = n_pos - n_hard_target

    if len(hard_negatives) > n_hard_target:
        hard_negatives = random.sample(hard_negatives, n_hard_target)
    else:
        n_hard_target = len(hard_negatives)
        n_easy_target = n_pos - n_hard_target

    if len(easy_negatives) > n_easy_target:
        easy_negatives = random.sample(easy_negatives, n_easy_target)
    else:
        n_easy_target = len(easy_negatives)

    total_neg = len(easy_negatives) + len(hard_negatives)
    if n_pos > total_neg:
        positives = random.sample(positives, total_neg)
    elif total_neg > n_pos:
        all_neg = easy_negatives + hard_negatives
        random.shuffle(all_neg)
        all_neg = all_neg[:n_pos]
        easy_negatives = [s for s in all_neg if s['role'] == 'neg_easy']
        hard_negatives = [s for s in all_neg if s['role'] == 'neg_hard']

    balanced = positives + easy_negatives + hard_negatives
    random.shuffle(balanced)

    n_final_pos = len(positives)
    n_final_easy = len(easy_negatives)
    n_final_hard = len(hard_negatives)
    total_neg_final = n_final_easy + n_final_hard
    print(f"\n[最终数据]")
    print(f"  正样本: {n_final_pos}")
    if total_neg_final > 0:
        print(f"  简单负样本: {n_final_easy} ({n_final_easy/total_neg_final*100:.1f}%)")
        print(f"  困难负样本: {n_final_hard} ({n_final_hard/total_neg_final*100:.1f}%)")
    print(f"  总计: {len(balanced)}")

    def to_swift(s):
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["input"]},
            {"role": "assistant", "content": s["output"]}
        ]}

    split_idx = int(len(balanced) * 0.95)
    train = balanced[:split_idx]
    val = balanced[split_idx:]
    for name, data in [('train_swift.jsonl', train), ('val_swift.jsonl', val)]:
        path = os.path.join(output_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            for s in data:
                f.write(json.dumps(to_swift(s), ensure_ascii=False) + '\n')
        print(f"  {name}: {len(data)} -> {path}")


# ======== 推理数据构建 ========

def build_inference_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    item_ids, isbn_map, isbn_to_meta = load_item_meta(data_dir)
    rel_names = load_relations(data_dir)
    triplets = load_kg(data_dir)
    entity_names = build_entity_names(item_ids, isbn_map, isbn_to_meta, triplets, rel_names, data_dir)

    out_path = os.path.join(output_dir, 'inference_swift.jsonl')
    n_written = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, (h, r, t) in enumerate(tqdm(triplets, desc="构建推理数据")):
            stmt = make_nl_statement(h, r, t, entity_names, rel_names)
            if stmt is None:
                continue
            rec = {
                "custom_id": str(idx),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stmt},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_written += 1
    print(f"[推理数据] 可翻译: {n_written}/{len(triplets)} -> {out_path}")


# ======== 预测解析 ========

def parse_mask(prediction_path, n_triplets, output_mask_path):
    mask = np.ones(n_triplets, dtype=np.float32)
    parsed = 0
    with open(prediction_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            idx = int(item.get('custom_id', item.get('triple_idx', -1)))
            if idx < 0 or idx >= n_triplets:
                continue
            pred = item.get('prediction', item.get('output', '')).strip().lower()
            if pred in ('false', '0', 'no'):
                mask[idx] = 0.0
            parsed += 1
    np.save(output_mask_path, mask)
    n_kept = int(mask.sum())
    print(f"[掩码] 解析: {parsed}/{n_triplets}, "
          f"保留: {n_kept} ({n_kept/n_triplets*100:.1f}%), "
          f"移除: {n_triplets-n_kept}")
    return mask


# ======== CLI ========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazon-book')
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--mode', choices=['train', 'inference', 'parse_mask'], default='train')
    parser.add_argument('--max_samples', type=int, default=25000)
    parser.add_argument('--hard_neg_path', default=None)
    parser.add_argument('--hard_neg_ratio', type=float, default=0.1)
    parser.add_argument('--prediction_path', default=None)
    parser.add_argument('--n_triplets', type=int, default=None)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_path, args.dataset)
    if args.output_dir is None:
        args.output_dir = os.path.join(data_dir, 'llm_data')

    if args.mode == 'train':
        if args.hard_neg_path is None:
            args.hard_neg_path = os.path.join(args.output_dir, 'hard_negatives.jsonl')
        build_train_data(data_dir, args.output_dir, max_samples=args.max_samples,
                         hard_neg_path=args.hard_neg_path,
                         hard_neg_ratio=args.hard_neg_ratio)
    elif args.mode == 'inference':
        build_inference_data(data_dir, args.output_dir)
    elif args.mode == 'parse_mask':
        if not args.prediction_path:
            raise ValueError("需要 --prediction_path")
        if not args.n_triplets:
            with open(os.path.join(data_dir, 'kg_final.txt')) as f:
                args.n_triplets = sum(1 for _ in f)
        mask_path = os.path.join(args.output_dir, 'llm_mask.npy')
        parse_mask(args.prediction_path, args.n_triplets, mask_path)
