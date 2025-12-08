#!/bin/bash
# 排除实验批量训练脚本
# 用于对比排除不同配准问题数据集对训练结果的影响

# ===== 配置参数 =====
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"
OUTPUT_BASE_DIR="./results_exclude_experiments"
EXCLUDE_FILE="./exclude_subjects.txt"

# 训练参数（与原始一致）
EPOCHS=25
BATCH_SIZE=4096
SAMPLES=50000

# 固定的测试被试（选一个正确的被试，用于所有实验）
# 修改为你想要的正确被试名称（如 "ODP_01" 或者留空使用默认的被试1）
FIXED_TEST_SUBJECT="ODP_01"  # 例如: "ODP_01" 或 "qhlazec"

# ===== 读取排除列表 =====
echo "读取配准问题数据集列表..."
mapfile -t EXCLUDE_SUBJECTS < <(grep -v '^#' "${EXCLUDE_FILE}" | grep -v '^$')

echo "找到 ${#EXCLUDE_SUBJECTS[@]} 个配准问题数据集:"
for subject in "${EXCLUDE_SUBJECTS[@]}"; do
    echo "  - ${subject}"
done
echo ""

# ===== 实验1-6: 每次排除1个错误数据集 =====
echo "===== 开始实验1-6: 每次排除1个错误数据集 ====="
echo ""

for i in "${!EXCLUDE_SUBJECTS[@]}"; do
    experiment_num=$((i + 1))
    exclude_subject="${EXCLUDE_SUBJECTS[$i]}"
    output_dir="${OUTPUT_BASE_DIR}/exp${experiment_num}_exclude_${exclude_subject}"

    echo ">>> 实验 ${experiment_num}/6: 排除 ${exclude_subject}"
    echo "输出目录: ${output_dir}"

    mkdir -p "${output_dir}"

    # 构建命令
    cmd="python train_1d_with_3d_dataset.py \
        --data_dir_1d ${DATA_DIR_1D} \
        --data_dir_3d ${DATA_DIR_3D} \
        --output_dir ${output_dir} \
        --exclude_single ${exclude_subject} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --samples_per_subject ${SAMPLES} \
        --save_predictions"

    # 如果设置了固定测试被试，添加参数
    if [ -n "${FIXED_TEST_SUBJECT}" ]; then
        cmd="${cmd} --fixed_test_subject ${FIXED_TEST_SUBJECT}"
    else
        cmd="${cmd} --test_subject 1"
    fi

    # 执行训练
    eval ${cmd}

    echo "实验 ${experiment_num} 完成！"
    echo ""
done

# ===== 实验7: 排除所有错误数据集 =====
echo "===== 开始实验7: 排除所有错误数据集 ====="
echo ""

output_dir="${OUTPUT_BASE_DIR}/exp7_exclude_all"
echo ">>> 实验 7/7: 排除所有配准问题数据集"
echo "输出目录: ${output_dir}"

mkdir -p "${output_dir}"

# 构建命令
cmd="python train_1d_with_3d_dataset.py \
    --data_dir_1d ${DATA_DIR_1D} \
    --data_dir_3d ${DATA_DIR_3D} \
    --output_dir ${output_dir} \
    --exclude_subjects ${EXCLUDE_FILE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --samples_per_subject ${SAMPLES} \
    --save_predictions"

# 如果设置了固定测试被试，添加参数
if [ -n "${FIXED_TEST_SUBJECT}" ]; then
    cmd="${cmd} --fixed_test_subject ${FIXED_TEST_SUBJECT}"
else
    cmd="${cmd} --test_subject 1"
fi

# 执行训练
eval ${cmd}

echo "实验 7 完成！"
echo ""

# ===== 生成汇总报告 =====
echo "===== 生成详细汇总报告 ====="
python compare_exclude_results_extended.py \
    --results_dir ${OUTPUT_BASE_DIR} \
    --output_file ${OUTPUT_BASE_DIR}/comparison_report_extended.json

echo ""
echo "===== 所有实验完成！ ====="
echo "结果保存在: ${OUTPUT_BASE_DIR}"
echo "详细报告: ${OUTPUT_BASE_DIR}/comparison_report_extended.json"
echo ""
echo "生成的可视化图表:"
echo "  - comprehensive_metrics_comparison.png  (综合指标对比)"
echo "  - topk_accuracy_comparison.png          (Top-K准确率对比)"
echo "  - metrics_correlation_heatmap.png       (指标相关性热力图)"
echo "  - single_vs_all_boxplot.png             (单个vs所有箱线图)"
