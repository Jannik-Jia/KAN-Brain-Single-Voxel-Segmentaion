#!/bin/bash
# 门控平滑快速测试脚本

echo "=== 门控平滑快速测试 ==="

# 基础设置
RESULTS_DIR="../B0_1D_training/results_1d_with_3d"
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"
OUTPUT_DIR="./gated_smooth_test"
TEST_SUBJECT=38

# 清理旧结果
rm -rf ${OUTPUT_DIR}

echo "测试1: 标准平滑（baseline）"
python smooth_postprocess_eval.py \
    --pred_file "${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat" \
    --gt_file auto \
    --output_dir "${OUTPUT_DIR}/baseline" \
    --fast_smooth

echo ""
echo "测试2: 同类门控平滑"
python smooth_postprocess_eval.py \
    --pred_file "${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat" \
    --gt_file auto \
    --output_dir "${OUTPUT_DIR}/class_gated" \
    --fast_smooth \
    --use_class_gating

echo ""
echo "测试3: 不确定性门控平滑（熵-based）"
python smooth_postprocess_eval.py \
    --pred_file "${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat" \
    --gt_file auto \
    --output_dir "${OUTPUT_DIR}/uncertainty_entropy" \
    --fast_smooth \
    --use_uncertainty_gating \
    --uncertainty_type entropy

echo ""
echo "测试4: 全门控平滑（所有方法对比）"
python smooth_postprocess_eval.py \
    --pred_file "${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat" \
    --gt_file auto \
    --output_dir "${OUTPUT_DIR}/all_methods" \
    --fast_smooth \
    --use_class_gating \
    --use_uncertainty_gating \
    --uncertainty_type margin \
    --uncertainty_tau 0.3 \
    --uncertainty_kappa 0.15

echo ""
echo "=== 测试完成 ==="
echo "结果保存在: ${OUTPUT_DIR}/"
echo ""
echo "查看详细结果:"
echo "  cat ${OUTPUT_DIR}/all_methods/evaluation_results.json"