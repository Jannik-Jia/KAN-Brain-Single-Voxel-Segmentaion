#!/bin/bash
# 交叉验证实验批量训练脚本
# 用于运行不同排除策略下的完整交叉验证

# ===== 配置参数 =====
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"
OUTPUT_BASE_DIR="./results_cv_experiments"
EXCLUDE_FILE="./exclude_subjects.txt"

# 训练参数
EPOCHS=25
BATCH_SIZE=4096
PATIENCE=5

# ===== 读取排除列表 =====
echo "读取配准问题数据集列表..."
mapfile -t EXCLUDE_SUBJECTS < <(grep -v '^#' "${EXCLUDE_FILE}" | grep -v '^$')

echo "找到 ${#EXCLUDE_SUBJECTS[@]} 个配准问题数据集:"
for subject in "${EXCLUDE_SUBJECTS[@]}"; do
    echo "  - ${subject}"
done
echo ""

# ===== 实验1: 排除所有错误数据集后的交叉验证 =====
echo "===== 实验1: 排除所有错误数据集后的交叉验证 ====="
echo ""

output_dir="${OUTPUT_BASE_DIR}/cv_exclude_all"
echo "输出目录: ${output_dir}"
mkdir -p "${output_dir}"

cmd="python train_1d_with_3d_dataset.py \
    --data_dir_1d ${DATA_DIR_1D} \
    --data_dir_3d ${DATA_DIR_3D} \
    --output_dir ${output_dir} \
    --exclude_subjects ${EXCLUDE_FILE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --cross_validation"

echo "执行命令: ${cmd}"
eval ${cmd}

echo "实验1 完成！"
echo ""

# ===== 实验2-N: 每次只排除1个错误数据集的交叉验证 =====
echo "===== 开始每次只排除1个错误数据集的交叉验证实验 ====="
echo ""

for i in "${!EXCLUDE_SUBJECTS[@]}"; do
    experiment_num=$((i + 2))
    exclude_subject="${EXCLUDE_SUBJECTS[$i]}"
    output_dir="${OUTPUT_BASE_DIR}/cv_exclude_${exclude_subject}"

    echo ">>> 实验 ${experiment_num}: 只排除 ${exclude_subject} 的交叉验证"
    echo "输出目录: ${output_dir}"

    mkdir -p "${output_dir}"

    cmd="python train_1d_with_3d_dataset.py \
        --data_dir_1d ${DATA_DIR_1D} \
        --data_dir_3d ${DATA_DIR_3D} \
        --output_dir ${output_dir} \
        --exclude_single ${exclude_subject} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --cross_validation"

    echo "执行命令: ${cmd}"
    eval ${cmd}

    echo "实验 ${experiment_num} 完成！"
    echo ""
done

# ===== 实验N+1: 不排除任何数据集的完整交叉验证（基线）=====
echo "===== 基线实验: 不排除任何数据集的完整38折交叉验证 ====="
echo ""

output_dir="${OUTPUT_BASE_DIR}/cv_baseline_no_exclude"
echo "输出目录: ${output_dir}"
mkdir -p "${output_dir}"

cmd="python train_1d_with_3d_dataset.py \
    --data_dir_1d ${DATA_DIR_1D} \
    --data_dir_3d ${DATA_DIR_3D} \
    --output_dir ${output_dir} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --cross_validation"

echo "执行命令: ${cmd}"
eval ${cmd}

echo "基线实验 完成！"
echo ""

echo "===== 所有交叉验证实验完成！ ====="
echo "结果保存在: ${OUTPUT_BASE_DIR}"
echo ""
echo "每个实验目录下包含:"
echo "  - fold_XX_被试名/: 每个fold的模型和历史"
echo "  - cv_summary.json: 交叉验证汇总报告"
