#!/bin/bash
# Leave-One-Out交叉验证训练脚本
# 对每个被试分别作为测试集进行训练
# 每个fold的结果保存在独立的子目录中

# ============================================================================
# 配置参数
# ============================================================================

# 数据路径（请根据实际情况修改）
DATA_ROOT="/path/to/your/data"  # 包含1d/和3d/子目录的数据根目录

# 输出目录
OUTPUT_DIR="./runs/leave_one_out"

# 训练超参数
EPOCHS=3
BATCH_SIZE=256
LR=1e-5
WEIGHT_DECAY=1e-5
GRAD_CLIP_NORM=1.0

# 类别权重设置（默认不使用）
USE_CLASS_WEIGHTS=false  # 设置为true启用类别权重
CLASS_WEIGHT_ALPHA=0.5

# 其他设置
SEED=42
MAX_VOX_PER_SUBJECT=""  # 留空表示使用全部体素

# ============================================================================
# 自动检测被试列表
# ============================================================================

echo "检测数据目录中的被试..."

# 检查数据目录是否存在
if [ ! -d "${DATA_ROOT}/1d" ]; then
    echo "错误: 数据目录不存在: ${DATA_ROOT}/1d"
    echo "请修改脚本中的 DATA_ROOT 变量"
    exit 1
fi

# 从1d目录中获取被试列表
SUBJECT_FILES=($(ls ${DATA_ROOT}/1d/*_1d.npz 2>/dev/null | sort))

if [ ${#SUBJECT_FILES[@]} -eq 0 ]; then
    echo "错误: 未找到任何被试数据文件"
    echo "请检查数据目录: ${DATA_ROOT}/1d"
    exit 1
fi

# 提取被试ID（去除路径和后缀）
SUBJECTS=()
for file in "${SUBJECT_FILES[@]}"; do
    basename=$(basename "$file" _1d.npz)
    SUBJECTS+=("$basename")
done

N_SUBJECTS=${#SUBJECTS[@]}

echo "找到 ${N_SUBJECTS} 个被试"
echo "被试列表: ${SUBJECTS[@]}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# ============================================================================
# 训练配置摘要
# ============================================================================

echo "训练配置:"
echo "  - 数据根目录: ${DATA_ROOT}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - 训练轮数: ${EPOCHS}"
echo "  - 批大小: ${BATCH_SIZE}"
echo "  - 学习率: ${LR}"
echo "  - 权重衰减: ${WEIGHT_DECAY}"
echo "  - 梯度裁剪: ${GRAD_CLIP_NORM}"
echo "  - 使用类权重: ${USE_CLASS_WEIGHTS}"
if [ "$USE_CLASS_WEIGHTS" = true ]; then
    echo "  - 类权重alpha: ${CLASS_WEIGHT_ALPHA}"
fi
echo "  - 随机种子: ${SEED}"
if [ -n "$MAX_VOX_PER_SUBJECT" ]; then
    echo "  - 每被试最大体素数: ${MAX_VOX_PER_SUBJECT}"
fi
echo ""

# 保存配置到文件
cat > ${OUTPUT_DIR}/config.txt <<EOF
Leave-One-Out Cross-Validation Configuration
=============================================

Data Root: ${DATA_ROOT}
Output Directory: ${OUTPUT_DIR}
Number of Subjects: ${N_SUBJECTS}

Training Parameters:
  - Epochs: ${EPOCHS}
  - Batch Size: ${BATCH_SIZE}
  - Learning Rate: ${LR}
  - Weight Decay: ${WEIGHT_DECAY}
  - Gradient Clipping: ${GRAD_CLIP_NORM}
  - Use Class Weights: ${USE_CLASS_WEIGHTS}
  - Class Weight Alpha: ${CLASS_WEIGHT_ALPHA}
  - Random Seed: ${SEED}
  - Max Voxels per Subject: ${MAX_VOX_PER_SUBJECT:-"All"}

Subjects (${N_SUBJECTS}):
$(printf '  - %s\n' "${SUBJECTS[@]}")

Started: $(date)
EOF

echo "配置已保存到: ${OUTPUT_DIR}/config.txt"
echo ""

# ============================================================================
# Leave-One-Out训练循环
# ============================================================================

echo "开始Leave-One-Out交叉验证训练..."
echo "========================================"
echo ""

START_TIME=$(date +%s)

for i in "${!SUBJECTS[@]}"; do
    TEST_SUBJECT="${SUBJECTS[$i]}"
    FOLD_NUM=$((i + 1))
    FOLD_NAME="fold_${FOLD_NUM}_test_${TEST_SUBJECT}"

    echo ""
    echo ">>> Fold ${FOLD_NUM}/${N_SUBJECTS}"
    echo ">>> 测试被试: ${TEST_SUBJECT}"
    echo ">>> 保存目录: ${OUTPUT_DIR}/${FOLD_NAME}"
    echo "----------------------------------------"

    # 构建训练命令
    CMD="python train_runner.py \
        --data-root ${DATA_ROOT} \
        --test-id ${TEST_SUBJECT} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --grad-clip-norm ${GRAD_CLIP_NORM} \
        --save-dir ${OUTPUT_DIR} \
        --fold-name ${FOLD_NAME}"

    # 添加可选参数
    if [ "$USE_CLASS_WEIGHTS" = true ]; then
        CMD="${CMD} --use-class-weights --class-weight-alpha ${CLASS_WEIGHT_ALPHA}"
    fi

    if [ -n "$MAX_VOX_PER_SUBJECT" ]; then
        CMD="${CMD} --max-vox-per-subject ${MAX_VOX_PER_SUBJECT}"
    fi

    # 执行训练
    echo "执行命令: ${CMD}"
    echo ""

    eval ${CMD}

    if [ $? -eq 0 ]; then
        echo "✅ Fold ${FOLD_NUM} 训练完成"
    else
        echo "❌ Fold ${FOLD_NUM} 训练失败"
        echo "错误发生在测试被试: ${TEST_SUBJECT}"
        # 可选：继续训练其他fold，或者退出
        # exit 1
    fi

    echo ""

    # 可选：只运行前几个fold进行测试
    # if [ ${FOLD_NUM} -eq 3 ]; then
    #     echo "测试模式：只运行前3个fold"
    #     break
    # fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================"
echo "所有训练完成！"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# 生成汇总报告
# ============================================================================

echo "生成汇总报告..."

python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
results = []

# 遍历所有fold目录
fold_dirs = sorted(output_dir.glob("fold_*"))

print(f"\n找到 {len(fold_dirs)} 个fold目录")

for fold_dir in fold_dirs:
    # 读取run_summary.json
    summary_file = fold_dir / 'run_summary.json'

    if not summary_file.exists():
        print(f"警告: {fold_dir.name} 缺少 run_summary.json")
        continue

    try:
        with open(summary_file) as f:
            summary = json.load(f)

        # 提取关键指标
        fold_result = {
            'fold_name': fold_dir.name,
            'test_subject': summary.get('test_id', 'unknown'),
            'val_subject': summary.get('val_id', 'unknown'),
            'n_train_subjects': summary.get('n_train_subjects', 0),
            'best_epoch': summary.get('best_epoch', 0),
            'best_val_nll': summary.get('best_val_nll', float('inf')),
            # 验证集指标
            'val_gross_acc': summary.get('val_metrics', {}).get('gross_accuracy', 0),
            'val_nll': summary.get('val_metrics', {}).get('nll', 0),
            'val_macro_f1': summary.get('val_metrics', {}).get('macro_f1', 0),
            'val_soft_ece': summary.get('val_metrics', {}).get('soft_ece', 0),
            # 测试集指标
            'test_gross_acc': summary.get('test_metrics', {}).get('gross_accuracy', 0),
            'test_nll': summary.get('test_metrics', {}).get('nll', 0),
            'test_macro_f1': summary.get('test_metrics', {}).get('macro_f1', 0),
            'test_soft_ece': summary.get('test_metrics', {}).get('soft_ece', 0),
            # 3D指标
            'val_3d_gross_acc': summary.get('val_3d_gross_acc', 0),
            'test_3d_gross_acc': summary.get('test_3d_gross_acc', 0),
        }

        results.append(fold_result)

    except Exception as e:
        print(f"错误: 读取 {fold_dir.name} 失败 - {e}")

if results:
    # 计算统计量
    val_accs = [r['val_gross_acc'] for r in results]
    test_accs = [r['test_gross_acc'] for r in results]
    val_nlls = [r['val_nll'] for r in results]
    test_nlls = [r['test_nll'] for r in results]
    val_f1s = [r['val_macro_f1'] for r in results]
    test_f1s = [r['test_macro_f1'] for r in results]

    # 打印汇总结果
    print('\n' + '=' * 80)
    print('Leave-One-Out Cross-Validation Results')
    print('=' * 80)
    print(f'完成训练: {len(results)}/${len(fold_dirs)} folds')
    print()
    print('验证集指标 (平均 ± 标准差):')
    print(f'  Gross Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}')
    print(f'  NLL:            {np.mean(val_nlls):.4f} ± {np.std(val_nlls):.4f}')
    print(f'  Macro F1:       {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}')
    print()
    print('测试集指标 (平均 ± 标准差):')
    print(f'  Gross Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}')
    print(f'  NLL:            {np.mean(test_nlls):.4f} ± {np.std(test_nlls):.4f}')
    print(f'  Macro F1:       {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}')
    print()
    print('测试集准确率范围:')
    print(f'  最好: {np.max(test_accs):.4f}')
    print(f'  最差: {np.min(test_accs):.4f}')
    print('=' * 80)

    # 保存汇总结果
    summary_results = {
        'n_folds': len(results),
        'statistics': {
            'val': {
                'gross_acc_mean': float(np.mean(val_accs)),
                'gross_acc_std': float(np.std(val_accs)),
                'nll_mean': float(np.mean(val_nlls)),
                'nll_std': float(np.std(val_nlls)),
                'macro_f1_mean': float(np.mean(val_f1s)),
                'macro_f1_std': float(np.std(val_f1s)),
            },
            'test': {
                'gross_acc_mean': float(np.mean(test_accs)),
                'gross_acc_std': float(np.std(test_accs)),
                'gross_acc_min': float(np.min(test_accs)),
                'gross_acc_max': float(np.max(test_accs)),
                'nll_mean': float(np.mean(test_nlls)),
                'nll_std': float(np.std(test_nlls)),
                'macro_f1_mean': float(np.mean(test_f1s)),
                'macro_f1_std': float(np.std(test_f1s)),
            }
        },
        'fold_results': results
    }

    summary_path = output_dir / 'cross_validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f'\n汇总结果已保存到: {summary_path}')

    # 创建CSV格式的结果表
    import csv
    csv_path = output_dir / 'cross_validation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['fold_name', 'test_subject', 'val_subject',
                     'val_gross_acc', 'val_nll', 'val_macro_f1',
                     'test_gross_acc', 'test_nll', 'test_macro_f1',
                     'val_3d_gross_acc', 'test_3d_gross_acc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f'CSV结果表已保存到: {csv_path}')

else:
    print('\n未找到任何有效的fold结果')

print()
EOF

# 更新配置文件，添加完成时间
cat >> ${OUTPUT_DIR}/config.txt <<EOF

Completed: $(date)
Duration: ${HOURS} hours ${MINUTES} minutes
EOF

echo ""
echo "汇总报告生成完成！"
echo ""
echo "========================================"
echo "查看结果:"
echo "  - 配置信息: ${OUTPUT_DIR}/config.txt"
echo "  - JSON汇总: ${OUTPUT_DIR}/cross_validation_summary.json"
echo "  - CSV表格:  ${OUTPUT_DIR}/cross_validation_results.csv"
echo "  - 各fold详细结果: ${OUTPUT_DIR}/fold_*/"
echo "========================================"
