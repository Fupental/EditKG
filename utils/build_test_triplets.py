"""
构建 LLM 零样本测试数据：
1. 从 amazon-book KG 中采样真实三元组
2. 构造虚假三元组（打乱 tail）
3. 解析实体/关系名称为可读文本
4. 输出 test_triplets.jsonl
"""

import json
import random
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/amazon-book")
random.seed(42)
np.random.seed(42)


def load_relation_names():
    rel_map = {}
    with open(DATA_DIR / "relation_list.txt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                uri, rid = parts[0], int(parts[1])
                name = uri.split("/")[-1]
                name = name.replace("_", " ").replace(".", " > ")
                rel_map[rid] = name
    return rel_map


def load_item_id_to_isbn():
    id2isbn = {}
    with open(DATA_DIR / "item_list.txt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                isbn, rid = parts[0], int(parts[1])
                id2isbn[rid] = isbn
    return id2isbn


def load_entity_id_to_freebase():
    id2fb = {}
    with open(DATA_DIR / "entity_list.txt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split('	')
            if len(parts) >= 2 and parts[1].strip().isdigit():
                fb = parts[0]
                rid = int(parts[1])
                id2fb[rid] = fb
    return id2fb


def load_isbn_to_title():
    cache_path = DATA_DIR / "isbn_to_title.json"
    if cache_path.exists():
        print(f"Loading cached ISBN->title from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print("Building ISBN->title mapping from meta_Books.jsonl...")
    isbn2title = {}
    with open(DATA_DIR / "meta_Books.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            pasin = d.get("parent_asin", "")
            title = d.get("title", "")
            if pasin and title:
                isbn2title[pasin] = title
            if (i + 1) % 1000000 == 0:
                print(f"  processed {i+1} entries...")

    with open(cache_path, "w") as f:
        json.dump(isbn2title, f, ensure_ascii=False)
    print(f"Saved {len(isbn2title)} ISBN->title mappings to {cache_path}")
    return isbn2title


def resolve_entity_name(eid, id2isbn, isbn2title, id2fb, n_items):
    if eid in id2isbn:
        isbn = id2isbn[eid]
        title = isbn2title.get(isbn)
        if title:
            return title
        return f"Book(ISBN:{isbn})"
    fb = id2fb.get(eid, f"entity_{eid}")
    return fb


def build_test_data(n_real=100, n_fake_corrupt=50, n_fake_random=50):
    rel_map = load_relation_names()
    id2isbn = load_item_id_to_isbn()
    id2fb = load_entity_id_to_freebase()
    isbn2title = load_isbn_to_title()
    triplets = np.loadtxt(DATA_DIR / "kg_final.txt", dtype=np.int32)

    n_items = len(id2isbn)
    n_entities = len(id2fb)

    print(f"\nItems: {n_items}, Entities: {n_entities}, Relations: {len(rel_map)}")

    item_ids = set(id2isbn.keys())
    good_triplets = []
    for h, r, t in triplets:
        h, r, t = int(h), int(r), int(t)
        h_name = resolve_entity_name(h, id2isbn, isbn2title, id2fb, n_items)
        t_name = resolve_entity_name(t, id2isbn, isbn2title, id2fb, n_items)
        h_is_book = h in id2isbn and not h_name.startswith("m.") and not h_name.startswith("Book(")
        t_is_book = t in id2isbn and not t_name.startswith("m.") and not t_name.startswith("Book(")
        if h_is_book or t_is_book:
            good_triplets.append((h, r, t))

    print(f"Triplets with readable book name: {len(good_triplets)}")

    real_samples = random.sample(good_triplets, min(n_real, len(good_triplets)))

    all_entities = list(range(n_entities))
    fake_corrupt = []
    for h, r, t in random.sample(good_triplets, min(n_fake_corrupt * 3, len(good_triplets))):
        new_t = random.choice(all_entities)
        while new_t == t:
            new_t = random.choice(all_entities)
        fake_corrupt.append((h, r, new_t))
        if len(fake_corrupt) >= n_fake_corrupt:
            break

    fake_semantic = []
    book_ids = [eid for eid in id2isbn if isbn2title.get(id2isbn[eid])]
    for _ in range(n_fake_random):
        h = random.choice(book_ids)
        r = random.choice(list(rel_map.keys()))
        t = random.choice(book_ids)
        while t == h:
            t = random.choice(book_ids)
        fake_semantic.append((h, r, t))

    test_data = []
    for h, r, t in real_samples:
        test_data.append({
            "head": resolve_entity_name(h, id2isbn, isbn2title, id2fb, n_items),
            "relation": rel_map.get(r, f"relation_{r}"),
            "tail": resolve_entity_name(t, id2isbn, isbn2title, id2fb, n_items),
            "head_id": h, "relation_id": r, "tail_id": t,
            "label": "real", "type": "ground_truth"
        })

    for h, r, t in fake_corrupt:
        test_data.append({
            "head": resolve_entity_name(h, id2isbn, isbn2title, id2fb, n_items),
            "relation": rel_map.get(r, f"relation_{r}"),
            "tail": resolve_entity_name(t, id2isbn, isbn2title, id2fb, n_items),
            "head_id": h, "relation_id": r, "tail_id": t,
            "label": "fake", "type": "corrupted_tail"
        })

    for h, r, t in fake_semantic:
        test_data.append({
            "head": resolve_entity_name(h, id2isbn, isbn2title, id2fb, n_items),
            "relation": rel_map.get(r, f"relation_{r}"),
            "tail": resolve_entity_name(t, id2isbn, isbn2title, id2fb, n_items),
            "head_id": h, "relation_id": r, "tail_id": t,
            "label": "fake", "type": "semantic_mismatch"
        })

    random.shuffle(test_data)

    out_path = DATA_DIR / "test_triplets.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(test_data)} test triplets to {out_path}")
    print(f"  Real: {sum(1 for d in test_data if d['label'] == 'real')}")
    print(f"  Fake (corrupted): {sum(1 for d in test_data if d['type'] == 'corrupted_tail')}")
    print(f"  Fake (semantic): {sum(1 for d in test_data if d['type'] == 'semantic_mismatch')}")

    print("\n=== Sample entries ===")
    for item in test_data[:8]:
        print(f"  [{item['label']:4s}|{item['type']:18s}] ({item['head']}, {item['relation']}, {item['tail']})")


if __name__ == "__main__":
    build_test_data(n_real=100, n_fake_corrupt=50, n_fake_random=50)
