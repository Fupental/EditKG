# -*- coding: utf-8 -*-
"""
resolve_freebase_mids.py — 通过 Wikidata SPARQL 将 Freebase MID 映射为可读名称
方法：Freebase MID -> Wikidata P646 属性 -> QID -> label

使用方式：
    # 测试前10个
    python utils/resolve_freebase_mids.py --dataset amazon-book --limit 10

    # 全量解析（分批SPARQL，无需API key）
    python utils/resolve_freebase_mids.py --dataset amazon-book

    # 从缓存继续
    python utils/resolve_freebase_mids.py --dataset amazon-book --resume

输出：直接在 entity_list.txt 第三列写入解析出的名称
"""
import os
import sys
import json
import time
import argparse
import requests
from tqdm import tqdm

try:
    from utils.path_utils import default_dataset_root, resolve_dataset_dir
except ImportError:
    from path_utils import default_dataset_root, resolve_dataset_dir

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "EditKG-MID-Resolver/1.0 (research project)"

# Wikidata SPARQL 一次最多查多少个 MID（VALUES 子句限制）
BATCH_SIZE = 50


def sparql_query(query: str, max_retries: int = 3) -> list:
    """执行 SPARQL 查询，返回 results bindings 列表"""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query},
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = min(60, 10 * (attempt + 1))
                print(f"  [429 限流] 等待 {wait}s 后重试...")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                wait = min(120, 30 * (attempt + 1))
                print(f"  [403 forbidden] IP可能被临时限制，等待 {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [HTTP {resp.status_code}] {resp.text[:200]}")
                time.sleep(5)
                continue
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  [请求异常] {e}, 重试 {attempt+1}/{max_retries}")
            time.sleep(5 * (attempt + 1))
    return []


def batch_resolve_mids(mids: list, batch_size: int = BATCH_SIZE,
                       delay: float = 1.5, cache: dict = None,
                       cache_path: str = None) -> dict:
    """
    批量通过 Wikidata SPARQL 将 Freebase MID 转为可读名称。
    """
    results = {}
    if cache is None:
        cache = {}
    total_batches = (len(mids) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(tqdm(range(0, len(mids), batch_size),
                  total=total_batches, desc="SPARQL批量查询")):
        batch = mids[i:i + batch_size]

        # 构建 VALUES 子句: "/m/06dz2p0" "/m/0691ppl" ...
        values_str = " ".join(
            f'"/m/{mid[2:]}"' for mid in batch  # m.xxx -> /m/xxx
        )

        query = f"""
SELECT ?mid_val ?label WHERE {{
  VALUES ?mid_val {{ {values_str} }}
  ?item wdt:P646 ?mid_val .
  ?item rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}}
"""
        bindings = sparql_query(query)

        for b in bindings:
            mid_val = b.get("mid_val", {}).get("value", "")  # "/m/06dz2p0"
            label = b.get("label", {}).get("value", "")
            if mid_val and label:
                # /m/06dz2p0 -> m.06dz2p0
                mid_key = "m." + mid_val[3:]
                results[mid_key] = label
                cache[mid_key] = label

        # 定期保存缓存
        if cache_path and (batch_idx + 1) % 20 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            tqdm.write(f"  [缓存] 已保存 {len(cache)} 条 (batch {batch_idx+1}/{total_batches})")

        # 控制请求速率
        if i + batch_size < len(mids):
            time.sleep(delay)

    return results


def resolve_g_mids(g_mids: list, delay: float = 1.5, cache: dict = None,
                   cache_path: str = None) -> dict:
    """解析 g. 开头的 MID"""
    results = {}
    if cache is None:
        cache = {}
    batch_size = BATCH_SIZE
    total_batches = (len(g_mids) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(g_mids), batch_size),
                  total=total_batches, desc="SPARQL查询g.MID"):
        batch = g_mids[i:i + batch_size]

        values_str = " ".join(
            f'"/g/{mid[2:]}"' for mid in batch
        )

        query = f"""
SELECT ?mid_val ?label WHERE {{
  VALUES ?mid_val {{ {values_str} }}
  ?item wdt:P646 ?mid_val .
  ?item rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}}
"""
        bindings = sparql_query(query)
        for b in bindings:
            mid_val = b.get("mid_val", {}).get("value", "")
            label = b.get("label", {}).get("value", "")
            if mid_val and label:
                mid_key = "g." + mid_val[3:]
                results[mid_key] = label
                cache[mid_key] = label

        if i + batch_size < len(g_mids):
            time.sleep(delay)

    return results


def read_entity_list(filepath):
    """
    读取 entity_list.txt，返回 [(org_id, remap_id, existing_name_or_None), ...]
    支持2列和3列格式
    """
    entities = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                entities.append((parts[0], parts[1], parts[2]))
            else:
                # 2列格式: "m.045wq1q 0"（空格分隔，取最后一个作为ID）
                parts2 = line.rsplit(maxsplit=1)
                if len(parts2) == 2:
                    entities.append((parts2[0], parts2[1], None))
    return header, entities


def write_entity_list(filepath, header, entities):
    """写回 entity_list.txt（3列：org_id remap_id label）"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for org_id, remap_id, name in entities:
            if name:
                f.write(f"{org_id}\t{remap_id}\t{name}\n")
            else:
                f.write(f"{org_id}\t{remap_id}\n")


def main():
    parser = argparse.ArgumentParser(description="通过Wikidata SPARQL解析Freebase MID")
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--data_path", default=default_dataset_root())
    parser.add_argument("--limit", type=int, default=0,
                        help="最多解析多少个MID（0=全部）")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="批次间延迟秒数（默认1.5s）")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"每批查询个数（默认{BATCH_SIZE}）")
    parser.add_argument("--resume", action="store_true",
                        help="从缓存继续解析")
    args = parser.parse_args()

    data_dir = resolve_dataset_dir(args.data_path, args.dataset)
    entity_file = os.path.join(data_dir, "entity_list.txt")
    cache_path = os.path.join(data_dir, "mid_names_cache.json")

    # 读取实体列表
    header, entities = read_entity_list(entity_file)
    print(f"[实体列表] 共 {len(entities)} 个实体")

    # 筛选需要解析的 MID
    m_mids = [org_id for org_id, _, name in entities
              if org_id.startswith("m.") and not name]
    g_mids = [org_id for org_id, _, name in entities
              if org_id.startswith("g.") and not name]
    already_named = sum(1 for _, _, name in entities if name)
    print(f"[统计] m.MID: {len(m_mids)}, g.MID: {len(g_mids)}, "
          f"已有名称: {already_named}, 类型名: "
          f"{len(entities) - len(m_mids) - len(g_mids) - already_named}")

    # 加载缓存
    cache = {}
    if args.resume and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[缓存] 已加载 {len(cache)} 条")
        # 从缓存中去掉已解析的
        m_mids = [m for m in m_mids if m not in cache]
        g_mids = [g for g in g_mids if g not in cache]
        print(f"[缓存后] 待解析 m.MID: {len(m_mids)}, g.MID: {len(g_mids)}")

    if args.limit > 0:
        m_mids = m_mids[:args.limit]
        g_mids = g_mids[:max(0, args.limit - len(m_mids))]
        print(f"[限制] 只解析前 {args.limit} 个")

    # 批量解析
    total_to_resolve = len(m_mids) + len(g_mids)
    total_batches = (total_to_resolve + args.batch_size - 1) // args.batch_size
    est_time = total_batches * args.delay
    print(f"\n{'='*60}")
    print(f"开始通过 Wikidata SPARQL 解析 Freebase MID (无需API Key)")
    print(f"批次大小: {args.batch_size}, 延迟: {args.delay}s")
    print(f"待解析: {total_to_resolve}, 预计批次: {total_batches}")
    print(f"预计最短耗时: {est_time/60:.0f} 分钟")
    print(f"{'='*60}\n")

    resolved = {}
    if m_mids:
        resolved.update(batch_resolve_mids(
            m_mids, args.batch_size, args.delay, cache, cache_path))
    if g_mids:
        resolved.update(resolve_g_mids(
            g_mids, args.delay, cache, cache_path))

    # 保存缓存
    cache.update(resolved)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\n[缓存保存] {len(cache)} 条 -> {cache_path}")

    # 将解析结果合并（包含之前缓存的）
    all_resolved = {k: v for k, v in cache.items() if v}

    # 更新 entity_list.txt 第三列
    updated = 0
    new_entities = []
    for org_id, remap_id, name in entities:
        if not name and org_id in all_resolved:
            new_entities.append((org_id, remap_id, all_resolved[org_id]))
            updated += 1
        else:
            new_entities.append((org_id, remap_id, name))

    # 更新header
    if "label" not in header.lower():
        header = "org_id\tremap_id\tlabel"

    write_entity_list(entity_file, header, new_entities)

    # 统计
    total = len(entities)
    named = sum(1 for _, _, n in new_entities if n)
    print(f"\n{'='*60}")
    print(f"[完成] 本次新解析: {len(resolved)}, 成功写入entity_list.txt: {updated}")
    print(f"[总计] 有名称: {named}/{total} = {named/total*100:.1f}%")
    print(f"[输出] {entity_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
