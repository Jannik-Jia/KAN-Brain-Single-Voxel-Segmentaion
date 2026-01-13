#!/bin/bash
# M-Layer 模型交叉验证训练脚本 (Operator-based Implementation)
#
# 理论背景：
#   基于 arXiv:2008.03936 "Intelligent Matrix Exponentiation"
#   核心公式：y = exp(M(x)) @ x
#   - 将输入 x 视为状态（State）
#   - 将 exp(M) 视为演化算子（Evolution Operator）
#   - 让算子直接作用于状态，保留李群几何结构
#
# 参数量：
#   Baseline (4×4096 MLP): ~52.2M 参数
#   M-Layer (Operator-based): ~35.4M 参数 (减少约 32%)

# ===== 配置参数 =====
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"
OUTPUT_DIR="./results_mlayer_operator_cv"

# 训练参数
EPOCHS=25
BATCH_SIZE=128          # 如果显存不足，改为 64 或 32
PATIENCE=5

# M-Layer 参数
M_MATRIX_SIZE=64        # 矩阵大小 (64x64=4096)
M_SCALE=0.01            # 数值稳定性缩放
M_CLIP=10.0             # 裁剪范围 (-1 表示不裁剪)

# ===== 开始训练 =====
echo "============================================================"
echo "M-Layer 模型交叉验证训练 (Operator-based)"
echo "============================================================"
echo ""
echo "数据目录:"
echo "  1D: ${DATA_DIR_1D}"
echo "  3D: ${DATA_DIR_3D}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "训练参数:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Patience: ${PATIENCE}"
echo ""
echo "M-Layer 参数:"
echo "  Matrix Size: ${M_MATRIX_SIZE}"
echo "  Scale: ${M_SCALE}"
echo "  Clip: ${M_CLIP}"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

python train_1d_with_3d_dataset.py \
    --data_dir_1d ${DATA_DIR_1D} \
    --data_dir_3d ${DATA_DIR_3D} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --m_matrix_size ${M_MATRIX_SIZE} \
    --m_scale ${M_SCALE} \
    --m_clip ${M_CLIP} \
    --cross_validation

echo ""
echo "============================================================"
echo "训练完成！(Operator-based M-Layer)"
echo "结果保存在: ${OUTPUT_DIR}"
echo "查看汇总: cat ${OUTPUT_DIR}/cv_summary.json"
echo "============================================================"
