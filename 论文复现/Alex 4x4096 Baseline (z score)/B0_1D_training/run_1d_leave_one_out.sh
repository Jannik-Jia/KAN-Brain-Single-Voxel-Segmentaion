#!/bin/bash
# 1D训练Leave-one-out脚本
# 对所有38个被试分别作为测试集进行训练
# 【Per-Patient Z-Score 版本】每个患者独立进行逐通道z-score标准化

# 设置参数
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"  # 1D数据集
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据集
OUTPUT_DIR="./results_1d_leave_one_out_perpatient_zscore"  # 更新目录名以区分标准化方式

# Alex的超参数
EPOCHS=25
BATCH_SIZE=8192

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "【Per-Patient Z-Score 版本】Leave-One-Out 训练"
echo "=============================================="
echo ""
echo "标准化方式: Per-Patient Z-Score（每个患者独立逐通道标准化）"
echo "1D数据目录: ${DATA_DIR_1D}"
echo "3D数据目录: ${DATA_DIR_3D}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "训练参数:"
echo "  - Epochs: ${EPOCHS} (Alex的设置)"
echo "  - Batch Size: ${BATCH_SIZE} (Alex的设置)"
echo "  - Learning Rate: 0.00001 (Alex的设置)"
echo "  - Normalisation: Per-Patient Z-Score"
echo ""

# Leave-One-Out训练循环（38次）
for test_subject in {1..38}
do
    echo ""
    echo ">>> 训练 ${test_subject}/38 (测试被试${test_subject})"

    python train_1d_with_3d_dataset.py \
        --data_dir_1d ${DATA_DIR_1D} \
        --data_dir_3d ${DATA_DIR_3D} \
        --output_dir ${OUTPUT_DIR} \
        --test_subject ${test_subject} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --save_predictions

    # 可选：只运行前几个被试进行测试
    # if [ ${test_subject} -eq 3 ]; then
    #     break
    # fi
done

echo ""
echo "所有训练完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""

# 生成汇总报告
echo "生成汇总报告..."
python -c "
import json
import numpy as np
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
results = []

for i in range(1, 39):
    history_file = output_dir / f'history_test{i}.json'
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
            best_f1 = max(history['test_f1'])
            results.append({
                'test_subject': i,
                'best_test_f1': best_f1,
                'final_train_f1': history['train_f1'][-1],
                'final_test_f1': history['test_f1'][-1]
            })

if results:
    best_f1s = [r['best_test_f1'] for r in results]
    print(f'\\n==============================================')
    print(f'【Per-Patient Z-Score 版本】Leave-One-Out 结果汇总')
    print(f'==============================================')
    print(f'标准化方式: Per-Patient Z-Score')
    print(f'完成训练: {len(results)}/38')
    print(f'平均最佳测试F1: {np.mean(best_f1s):.4f} ± {np.std(best_f1s):.4f}')
    print(f'最好结果: {np.max(best_f1s):.4f}')
    print(f'最差结果: {np.min(best_f1s):.4f}')

    # 保存汇总结果
    summary_file = output_dir / 'summary_results.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'normalisation_mode': 'per_patient_zscore',
            'method_key': 'alex_4x4096_perpatient_zscore',
            'subfamily': 'baseline_4x4096_per_patient',
            'results': results,
            'statistics': {
                'mean_f1': float(np.mean(best_f1s)),
                'std_f1': float(np.std(best_f1s)),
                'max_f1': float(np.max(best_f1s)),
                'min_f1': float(np.min(best_f1s)),
                'n_subjects': len(results)
            }
        }, f, indent=2)
    print(f'汇总结果保存至: {summary_file}')
else:
    print('未找到任何结果文件')
"

echo ""
echo "汇总报告完成！"
