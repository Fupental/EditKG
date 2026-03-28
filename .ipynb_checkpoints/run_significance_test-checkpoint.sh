#!/bin/bash
# ==========================================================================
# run_significance_test.sh — 显著性检验批量运行脚本
# ==========================================================================
# 用法：
#   chmod +x run_significance_test.sh
#   bash run_significance_test.sh
#
# 说明：
#   使用3个不同随机种子分别运行 基线(无多模态) 和 文本模态 两组实验，
#   每组3次，共6次训练。结果自动保存到 result/ 目录下的 jsonl 文件。
#   运行完毕后，执行 python utils/significance_test.py 进行显著性检验。
# ==========================================================================

SEEDS=(2023 2024 2025)

# 公共参数（与你之前的训练命令一致，根据需要调整）
COMMON_ARGS="--dataset amazon-book --lr 0.0005 --dim 128 --channel 128 --context_hops 2 --margin 0.2 --max_iter 2 --batch_size 4096 --test_batch_size 2048 --gpu_id 0"

echo "=========================================="
echo "  显著性检验：共 ${#SEEDS[@]} 个种子 × 2 组 = $((${#SEEDS[@]} * 2)) 次实验"
echo "=========================================="

# 清空旧结果文件
rm -f ./result/amazon-book_significance_baseline.jsonl
rm -f ./result/amazon-book_significance_mm_text.jsonl

# ===== 第一组：基线（无多模态）=====
echo ""
echo "========== 第一组：基线（--no_mm）=========="
for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> [基线] 种子=${seed} 开始训练..."
    python main.py ${COMMON_ARGS} --seed ${seed} --no_mm
    echo ">>> [基线] 种子=${seed} 训练完成"
    echo ""
done

# ===== 第二组：带文本模态 =====
echo ""
echo "========== 第二组：文本模态（多模态）=========="
for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> [文本模态] 种子=${seed} 开始训练..."
    python main.py ${COMMON_ARGS} --seed ${seed}
    echo ">>> [文本模态] 种子=${seed} 训练完成"
    echo ""
done

echo ""
echo "=========================================="
echo "  所有实验已完成！"
echo "  基线结果:     ./result/amazon-book_significance_baseline.jsonl"
echo "  文本模态结果: ./result/amazon-book_significance_mm_text.jsonl"
echo ""
echo "  运行显著性检验分析："
echo "    python utils/significance_test.py --dataset amazon-book"
echo "=========================================="
