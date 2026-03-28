# -*- coding: utf-8 -*-
"""
==========================================================================
significance_test.py — 显著性检验分析脚本
==========================================================================
功能：
    读取基线和文本模态两组实验的 jsonl 结果文件，
    对 Recall@20, NDCG@20, Precision@20, Hit@20 四个指标
    执行 配对t检验 和 Wilcoxon符号秩检验。

用法：
    python utils/significance_test.py --dataset amazon-book

输出:
    每个指标的 均值±标准差、提升幅度、p值、是否显著
==========================================================================
"""

import os
import json
import argparse
import numpy as np
from scipy import stats


def load_results(filepath):
    """从 jsonl 文件加载所有实验结果"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser(description="显著性检验分析")
    parser.add_argument("--dataset", type=str, default="amazon-book")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平（默认0.05）")
    args = parser.parse_args()

    baseline_path = f'./result/{args.dataset}_significance_baseline.jsonl'
    mm_path = f'./result/{args.dataset}_significance_mm_text.jsonl'

    # 检查文件是否存在
    for path in [baseline_path, mm_path]:
        if not os.path.isfile(path):
            print(f"错误: 找不到结果文件 {path}")
            print("请先运行 bash run_significance_test.sh 完成所有实验")
            return

    baseline_results = load_results(baseline_path)
    mm_results = load_results(mm_path)

    print(f"\n{'='*70}")
    print(f"  显著性检验分析 — 数据集: {args.dataset}")
    print(f"  基线实验次数: {len(baseline_results)}, 文本模态实验次数: {len(mm_results)}")
    print(f"  显著性水平 α = {args.alpha}")
    print(f"{'='*70}")

    if len(baseline_results) != len(mm_results):
        print(f"警告: 两组实验次数不一致 ({len(baseline_results)} vs {len(mm_results)})")

    n = min(len(baseline_results), len(mm_results))
    if n < 2:
        print("错误: 至少需要2次实验才能进行显著性检验")
        return

    # 按 seed 排序，确保配对正确
    baseline_results.sort(key=lambda x: x.get('seed', 0))
    mm_results.sort(key=lambda x: x.get('seed', 0))

    # 打印每次实验的种子
    print(f"\n  基线种子:     {[r.get('seed') for r in baseline_results[:n]]}")
    print(f"  文本模态种子: {[r.get('seed') for r in mm_results[:n]]}")

    # 待检验的指标
    metrics = [
        ('Recall@20', 'best_recall_20'),
        ('NDCG@20', 'best_ndcg_20'),
        ('Precision@20', 'best_precision_20'),
        ('Hit@20', 'best_hit_ratio_20'),
    ]

    print(f"\n{'─'*70}")
    print(f"{'指标':<16} {'基线 (mean±std)':<22} {'文本模态 (mean±std)':<22} {'提升%':<8} {'t-test p':<10} {'Wilcoxon p':<10} {'显著?'}")
    print(f"{'─'*70}")

    for metric_name, key in metrics:
        bl_vals = np.array([r[key] for r in baseline_results[:n]])
        mm_vals = np.array([r[key] for r in mm_results[:n]])

        bl_mean = bl_vals.mean()
        bl_std = bl_vals.std()
        mm_mean = mm_vals.mean()
        mm_std = mm_vals.std()

        # 相对提升
        improvement = (mm_mean - bl_mean) / (bl_mean + 1e-10) * 100

        # 配对 t 检验（双尾）
        t_stat, t_pval = stats.ttest_rel(mm_vals, bl_vals)

        # Wilcoxon 符号秩检验（非参数方法，样本量小时更稳健）
        try:
            w_stat, w_pval = stats.wilcoxon(mm_vals, bl_vals)
        except ValueError:
            # 当差异全为0时 wilcoxon 会报错
            w_pval = 1.0

        # 是否显著
        significant = "✓" if t_pval < args.alpha else "✗"

        print(f"{metric_name:<16} {bl_mean:.4f}±{bl_std:.4f}      {mm_mean:.4f}±{mm_std:.4f}      {improvement:>+6.2f}%  {t_pval:<10.6f} {w_pval:<10.6f} {significant}")

    print(f"{'─'*70}")

    # 详细结果
    print(f"\n{'='*70}")
    print("  各次实验详细结果")
    print(f"{'='*70}")
    for metric_name, key in metrics:
        print(f"\n  {metric_name}:")
        print(f"    {'Seed':<8} {'基线':<12} {'文本模态':<12} {'差值':<12}")
        for i in range(n):
            bl_v = baseline_results[i][key]
            mm_v = mm_results[i][key]
            seed = baseline_results[i].get('seed', '?')
            print(f"    {seed:<8} {bl_v:<12.4f} {mm_v:<12.4f} {mm_v - bl_v:<+12.4f}")

    print(f"\n{'='*70}")
    print("  说明:")
    print(f"  - 配对t检验: 适用于正态分布假设, p < {args.alpha} 表示差异显著")
    print(f"  - Wilcoxon符号秩检验: 非参数方法, 不依赖正态假设, 样本量少时更可靠")
    print(f"  - 建议: 当两种检验的 p 值都 < {args.alpha} 时, 可以认为差异显著")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
