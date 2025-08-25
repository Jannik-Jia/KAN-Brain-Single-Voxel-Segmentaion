#!/bin/bash
# 运行体素级概率图平滑后处理和评估脚本

# 设置参数
RESULTS_DIR="../B0_1D_training/results_1d_with_3d"    # 1D训练结果目录
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据目录（真实标签）
OUTPUT_DIR="./smooth_eval_results"              # 平滑评估结果目录

# 测试被试编号
TEST_SUBJECT=38

echo "开始体素级概率图平滑后处理评估..."
echo "预测结果目录: ${RESULTS_DIR}"
echo "3D数据目录: ${DATA_DIR_3D}"
echo "输出目录: ${OUTPUT_DIR}"
echo "测试被试: ${TEST_SUBJECT}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 检查预测文件是否存在
PRED_FILE="${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat"
if [ ! -f "${PRED_FILE}" ]; then
    echo "错误: 预测文件不存在: ${PRED_FILE}"
    echo "请先运行1D训练生成预测结果"
    exit 1
fi

# 获取对应的真实标签文件（按字母顺序第38个）
GT_FILE=$(ls ${DATA_DIR_3D}/*.mat | sort | sed -n "${TEST_SUBJECT}p")
if [ ! -f "${GT_FILE}" ]; then
    echo "错误: 真实标签文件不存在: ${GT_FILE}"
    echo "请检查3D数据目录路径"
    exit 1
fi

echo "预测文件: ${PRED_FILE}"
echo "真实标签文件: ${GT_FILE}"
echo ""

# 运行平滑评估（使用快速算法）
echo ">>> 执行平滑后处理和评估"
python smooth_postprocess_eval.py \
    --pred_file "${PRED_FILE}" \
    --gt_file "${GT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --fast_smooth

echo ""
echo "平滑评估完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""

# 显示结果文件
if [ -f "${OUTPUT_DIR}/evaluation_results.json" ]; then
    echo "===== 评估结果摘要 ====="
    python -c "
import json
with open('${OUTPUT_DIR}/evaluation_results.json', 'r') as f:
    results = json.load(f)

print('Method\\t\\tAccuracy\\tMacro F1\\tKappa\\t\\tAUPRC')
print('-' * 60)
for method, metrics in results.items():
    acc = metrics['accuracy']
    f1 = metrics['macro_f1'] 
    kappa = metrics['kappa']
    auprc = metrics.get('macro_auprc', 0)
    print(f'{method:<12}\\t{acc:.4f}\\t\\t{f1:.4f}\\t\\t{kappa:.4f}\\t\\t{auprc:.4f}')

# 计算改进
if 'original' in results and 'smooth_k3' in results:
    print('\\n===== 改进效果 =====')
    orig = results['original']
    k3 = results['smooth_k3']
    k7 = results['smooth_k7']
    
    print('3x3 平滑 vs 原始:')
    print(f'  ΔAccuracy: {k3[\"accuracy\"] - orig[\"accuracy\"]:+.4f}')
    print(f'  ΔMacro F1: {k3[\"macro_f1\"] - orig[\"macro_f1\"]:+.4f}')
    print(f'  ΔAUPRC: {k3.get(\"macro_auprc\", 0) - orig.get(\"macro_auprc\", 0):+.4f}')
    
    print('\\n7x7 平滑 vs 原始:')
    print(f'  ΔAccuracy: {k7[\"accuracy\"] - orig[\"accuracy\"]:+.4f}')
    print(f'  ΔMacro F1: {k7[\"macro_f1\"] - orig[\"macro_f1\"]:+.4f}')
    print(f'  ΔAUPRC: {k7.get(\"macro_auprc\", 0) - orig.get(\"macro_auprc\", 0):+.4f}')
"
else
    echo "未找到评估结果文件"
fi