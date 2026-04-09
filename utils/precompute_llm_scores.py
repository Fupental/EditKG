# -*- coding: utf-8 -*-
"""
precompute_llm_scores.py — 对候选KG三元组执行LLM打分，并维护历史缓存
"""
import argparse
import gc
import os
import sys

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm_scorer import LLMScorer
from utils.memory_monitor import log_cuda_mem, MemTimer


def load_cache(cache_path):
    if cache_path and os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        if isinstance(cache, dict):
            return cache
    return {}


def save_cache(cache, cache_path):
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)


def triplet_key(triplet_row):
    return tuple(int(x) for x in triplet_row.tolist())


def score_with_retry(missing_tensor, args):
    batch_size = args.batch_size
    last_error = None

    while batch_size >= 1:
        scorer = None
        try:
            print(f"[LLM-Precompute] 尝试 batch_size={batch_size}")
            scorer = LLMScorer(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                data_dir=args.data_dir,
                device=f"cuda:{args.gpu_id}",
                batch_size=batch_size,
                mem_debug=args.mem_debug,
            )
            new_scores = scorer.score_triplets(missing_tensor, target_device="cpu").cpu()
            scorer.unload()
            return new_scores, batch_size
        except torch.OutOfMemoryError as exc:
            last_error = exc
            print(f"[LLM-Precompute] OOM: batch_size={batch_size} 失败，准备降批重试")
            if scorer is not None:
                scorer.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size == 1:
                break
            batch_size = max(1, batch_size // 2)

    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets_path", required=True, help="候选KG三元组文件(.pt)")
    parser.add_argument("--output_path", required=True, help="输出分数文件(.pt)")
    parser.add_argument("--data_dir", required=True, help="数据集目录")
    parser.add_argument("--model_path", required=True, help="LLM基座模型路径")
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter路径")
    parser.add_argument("--cache_path", default=None, help="历史三元组分数缓存(.pt)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mem_debug", action="store_true")
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}"

    triplets = torch.load(args.triplets_path, map_location="cpu")
    if not isinstance(triplets, torch.Tensor):
        triplets = torch.as_tensor(triplets, dtype=torch.long)
    triplets = triplets.long()

    cache = load_cache(args.cache_path)
    scores = torch.empty((triplets.shape[0], 1), dtype=torch.float32)
    missing_indices = []
    missing_triplets = []

    for idx, row in enumerate(triplets):
        key = triplet_key(row)
        if key in cache:
            scores[idx, 0] = float(cache[key])
        else:
            missing_indices.append(idx)
            missing_triplets.append(row)

    print(f"[LLM-Precompute] 候选三元组: {triplets.shape[0]}")
    print(f"[LLM-Precompute] 命中历史缓存: {triplets.shape[0] - len(missing_indices)}")
    print(f"[LLM-Precompute] 需要新打分: {len(missing_indices)}")
    log_cuda_mem("llm_precompute.after_cache_scan", device, args.mem_debug)

    if missing_triplets:
        missing_tensor = torch.stack(missing_triplets, dim=0)
        with MemTimer("llm_precompute.score_with_retry", device, args.mem_debug):
            new_scores, final_batch_size = score_with_retry(missing_tensor, args)
        print(f"[LLM-Precompute] 实际使用 batch_size={final_batch_size}")

        for pos, idx in enumerate(missing_indices):
            val = float(new_scores[pos, 0].item())
            scores[idx, 0] = val
            cache[triplet_key(triplets[idx])] = val
    else:
        new_scores = None

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(scores, args.output_path)
    save_cache(cache, args.cache_path)
    print(f"[LLM-Precompute] 分数已保存: {args.output_path}")
    if args.cache_path:
        print(f"[LLM-Precompute] 历史缓存已保存: {args.cache_path}")


if __name__ == "__main__":
    main()
