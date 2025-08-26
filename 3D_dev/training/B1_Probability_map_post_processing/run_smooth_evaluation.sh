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

# 检查预测文件是否存在（检查多个可能的位置）
PRED_FILE="${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat"
if [ ! -f "${PRED_FILE}" ]; then
    # 尝试其他可能的位置
    ALT_PRED_FILE="../predictions/predictions_3d_test${TEST_SUBJECT}.mat"
    if [ -f "${ALT_PRED_FILE}" ]; then
        PRED_FILE="${ALT_PRED_FILE}"
        echo "找到预测文件: ${PRED_FILE}"
    else
        echo "错误: 预测文件不存在于以下位置:"
        echo "  ${RESULTS_DIR}/predictions_3d_test${TEST_SUBJECT}.mat"
        echo "  ../predictions/predictions_3d_test${TEST_SUBJECT}.mat"
        echo "请先运行1D训练生成预测结果，或检查--output_dir设置"
        exit 1
    fi
fi

# 获取对应的真实标签文件（从预测文件HDF5属性中读取，确保一致性）
echo ">>> 从预测文件读取GT文件路径..."
GT_FILE=$(python3 - "${PRED_FILE}" "${DATA_DIR_3D}" <<'PY'
import h5py, sys, os, glob

if len(sys.argv) < 3:
    print('', file=sys.stderr)
    exit()

pred_file, data_dir_3d = sys.argv[1], sys.argv[2]

try:
    with h5py.File(pred_file, 'r') as f:
        print(f"读取预测文件属性...", file=sys.stderr)
        
        # 显示所有属性用于调试
        attrs = dict(f.attrs)
        print(f"预测文件包含属性: {list(attrs.keys())}", file=sys.stderr)
        
        # 优先使用训练时保存的完整路径
        test_file_3d = f.attrs.get('test_file_3d')
        if test_file_3d is not None:
            tf3d_str = test_file_3d.decode() if isinstance(test_file_3d, bytes) else str(test_file_3d)
            print(f"找到test_file_3d属性: {tf3d_str}", file=sys.stderr)
            if os.path.exists(tf3d_str):
                print(f"3D文件存在，使用: {tf3d_str}", file=sys.stderr)
                print(tf3d_str)
                exit()
            else:
                print(f"3D文件不存在: {tf3d_str}", file=sys.stderr)
        
        # 回退：从test_subject属性匹配文件名
        test_subject = f.attrs.get('test_subject', 0)
        if isinstance(test_subject, bytes):
            test_subject = test_subject.decode()
        test_subject = int(test_subject)
        
        print(f"使用test_subject属性: {test_subject}", file=sys.stderr)
        
        if test_subject > 0:
            # 查找所有3D验证文件
            pattern = os.path.join(data_dir_3d, '*_3d_validated.mat')
            files = sorted(glob.glob(pattern))
            print(f"在 {data_dir_3d} 找到 {len(files)} 个3D文件", file=sys.stderr)
            
            if 1 <= test_subject <= len(files):
                selected_file = files[test_subject-1]
                print(f"选择第{test_subject}个文件: {selected_file}", file=sys.stderr)
                print(selected_file)
                exit()
        
        print('匹配失败', file=sys.stderr)
        print('')
except Exception as e:
    print(f'Python错误: {e}', file=sys.stderr)
    print('')
PY
)

if [ -z "$GT_FILE" ] || [ ! -f "$GT_FILE" ]; then
    echo "错误: 未能从预测文件属性定位真实标签文件"
    echo "预测文件: ${PRED_FILE}"
    echo "3D数据目录: ${DATA_DIR_3D}"
    echo "请检查："
    echo "  1. 预测文件是否为正确的HDF5格式"
    echo "  2. 预测文件是否包含test_file_3d或test_subject属性"
    echo "  3. 3D数据目录路径是否正确"
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