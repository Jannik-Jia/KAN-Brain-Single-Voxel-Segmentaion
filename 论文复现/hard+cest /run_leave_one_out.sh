#!/bin/bash
# Leave-One-Out交叉验证训练脚本
# 对每个被试分别作为测试集进行训练
# 每个fold的结果保存在独立的子目录中
#
# 支持 Soft+CEST 和 Hard+CEST 对照实验
# 两种模式使用完全一致的数据、split规则、超参数，仅训练标签不同

# ============================================================================
# 配置参数
# ============================================================================

# 数据路径
DATA_ROOT="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling/3d"  # 数据根目录

# 输出目录
OUTPUT_DIR="./runs/leave_one_out"

# 需要排除的被试列表（可为空）。默认使用 Fullyconnected_exclude 中的列表。
# 列表格式：一行一个关键字，文件名中包含该关键字的被试将被过滤掉。
EXCLUDE_FILE="/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/Fullyconnected_exclude/B0_1D_training/exclude_subjects.txt"

# ==================== 关键参数：训练监督模式 ====================
# soft: 使用软标签 q_i 训练 (Soft+CEST)
# hard: 使用硬标签 one_hot(argmax(q_i)) 训练 (Hard+CEST)
SUPERVISION="hard"  # 可选: soft, hard

# 训练超参数
EPOCHS=25
BATCH_SIZE=256
LR=1e-5
WEIGHT_DECAY=1e-5
GRAD_CLIP_NORM=1.0

# 类别权重设置（默认不使用）
USE_CLASS_WEIGHTS=false  # 设置为true启用类别权重
CLASS_WEIGHT_ALPHA=0.5

# 标签来源（默认自动检测）
LABELS_SOURCE="auto"  # 可选: auto, seg_one_hot_1d, proba_labels_3d

# ECE bins数量
ECE_N_BINS=15

# 其他设置
SEED=42
MAX_VOX_PER_SUBJECT=""  # 留空表示使用全部体素
SAVE_PREPOST_PREDS=true  # 是否保存校准前/后的预测

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

# 读取排除列表（如果提供）
EXCLUDE_SUBJECTS=()
if [ -n "$EXCLUDE_FILE" ]; then
    if [ ! -f "$EXCLUDE_FILE" ]; then
        echo "错误: 排除列表不存在: ${EXCLUDE_FILE}"
        exit 1
    fi
    mapfile -t EXCLUDE_SUBJECTS < <(grep -v '^#' "${EXCLUDE_FILE}" | grep -v '^[[:space:]]*$')
    echo "从排除列表读取到 ${#EXCLUDE_SUBJECTS[@]} 个被试: ${EXCLUDE_SUBJECTS[@]}"
    # 过滤被试列表
    FILTERED_SUBJECTS=()
    for sid in "${SUBJECTS[@]}"; do
        skip=false
        for excl in "${EXCLUDE_SUBJECTS[@]}"; do
            if [[ "$sid" == *"$excl"* ]]; then
                skip=true
                break
            fi
        done
        if [ "$skip" = false ]; then
            FILTERED_SUBJECTS+=("$sid")
        fi
    done
    SUBJECTS=("${FILTERED_SUBJECTS[@]}")
fi

N_SUBJECTS=${#SUBJECTS[@]}

if [ ${N_SUBJECTS} -lt 2 ]; then
    echo "错误: 排除后被试数量不足以进行Leave-One-Out (当前 ${N_SUBJECTS})"
    exit 1
fi

echo "找到 ${N_SUBJECTS} 个被试"
echo "被试列表: ${SUBJECTS[@]}"
echo ""

# 创建输出目录（包含supervision标记）
OUTPUT_DIR_FULL="${OUTPUT_DIR}_${SUPERVISION}"
mkdir -p ${OUTPUT_DIR_FULL}

# ============================================================================
# 训练配置摘要
# ============================================================================

echo "训练配置:"
echo "  - 监督模式: ${SUPERVISION}"
echo "  - 数据根目录: ${DATA_ROOT}"
echo "  - 输出目录: ${OUTPUT_DIR_FULL}"
echo "  - 训练轮数: ${EPOCHS}"
echo "  - 批大小: ${BATCH_SIZE}"
echo "  - 学习率: ${LR}"
echo "  - 权重衰减: ${WEIGHT_DECAY}"
echo "  - 梯度裁剪: ${GRAD_CLIP_NORM}"
echo "  - 使用类权重: ${USE_CLASS_WEIGHTS}"
if [ "$USE_CLASS_WEIGHTS" = true ]; then
    echo "  - 类权重alpha: ${CLASS_WEIGHT_ALPHA}"
fi
echo "  - 标签来源: ${LABELS_SOURCE}"
echo "  - ECE bins: ${ECE_N_BINS}"
echo "  - 随机种子: ${SEED}"
if [ -n "$MAX_VOX_PER_SUBJECT" ]; then
    echo "  - 每被试最大体素数: ${MAX_VOX_PER_SUBJECT}"
fi
echo ""

# 保存配置到文件
cat > ${OUTPUT_DIR_FULL}/config.txt <<EOF
Leave-One-Out Cross-Validation Configuration
=============================================

Supervision Mode: ${SUPERVISION}
Data Root: ${DATA_ROOT}
Output Directory: ${OUTPUT_DIR_FULL}
Number of Subjects: ${N_SUBJECTS}

Training Parameters:
  - Epochs: ${EPOCHS}
  - Batch Size: ${BATCH_SIZE}
  - Learning Rate: ${LR}
  - Weight Decay: ${WEIGHT_DECAY}
  - Gradient Clipping: ${GRAD_CLIP_NORM}
  - Use Class Weights: ${USE_CLASS_WEIGHTS}
  - Class Weight Alpha: ${CLASS_WEIGHT_ALPHA}
  - Labels Source: ${LABELS_SOURCE}
  - ECE Bins: ${ECE_N_BINS}
  - Random Seed: ${SEED}
  - Max Voxels per Subject: ${MAX_VOX_PER_SUBJECT:-"All"}
  - Save Pre/Post Predictions: ${SAVE_PREPOST_PREDS}

Subjects (${N_SUBJECTS}):
$(printf '  - %s\n' "${SUBJECTS[@]}")

Started: $(date)
EOF

echo "配置已保存到: ${OUTPUT_DIR_FULL}/config.txt"
echo ""

# ============================================================================
# Leave-One-Out训练循环
# ============================================================================

echo "开始Leave-One-Out交叉验证训练 (${SUPERVISION^^})..."
echo "========================================"
echo ""

START_TIME=$(date +%s)

for i in "${!SUBJECTS[@]}"; do
    TEST_SUBJECT="${SUBJECTS[$i]}"

    # 计算验证集被试：循环后继 (i+1) % N
    VAL_IDX=$(( (i + 1) % N_SUBJECTS ))
    VAL_SUBJECT="${SUBJECTS[$VAL_IDX]}"

    FOLD_NUM=$((i + 1))
    FOLD_NAME="fold_${FOLD_NUM}_test_${TEST_SUBJECT}_${SUPERVISION}"

    echo ""
    echo ">>> Fold ${FOLD_NUM}/${N_SUBJECTS} (${SUPERVISION^^})"
    echo ">>> 测试被试: ${TEST_SUBJECT}"
    echo ">>> 验证被试: ${VAL_SUBJECT}"
    echo ">>> 保存目录: ${OUTPUT_DIR_FULL}/${FOLD_NAME}"
    echo "----------------------------------------"

    # 构建训练命令
    CMD="python train_runner.py \
        --data-root ${DATA_ROOT} \
        --test-id ${TEST_SUBJECT} \
        --val-id ${VAL_SUBJECT} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --grad-clip-norm ${GRAD_CLIP_NORM} \
        --save-dir ${OUTPUT_DIR_FULL} \
        --fold-name ${FOLD_NAME} \
        --supervision ${SUPERVISION} \
        --labels-source ${LABELS_SOURCE} \
        --ece-n-bins ${ECE_N_BINS}"

    # 添加可选参数
    if [ "$USE_CLASS_WEIGHTS" = true ]; then
        CMD="${CMD} --use-class-weights --class-weight-alpha ${CLASS_WEIGHT_ALPHA}"
    fi

    if [ -n "$MAX_VOX_PER_SUBJECT" ]; then
        CMD="${CMD} --max-vox-per-subject ${MAX_VOX_PER_SUBJECT}"
    fi

    if [ "$SAVE_PREPOST_PREDS" = true ]; then
        CMD="${CMD} --save-prepost-preds"
    else
        CMD="${CMD} --no-save-prepost-preds"
    fi

    # 执行训练
    echo "执行命令: ${CMD}"
    echo ""

    eval ${CMD}

    if [ $? -eq 0 ]; then
        echo "Fold ${FOLD_NUM} 训练完成"
    else
        echo "Fold ${FOLD_NUM} 训练失败"
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
echo "结果保存在: ${OUTPUT_DIR_FULL}"
echo ""

# ============================================================================
# 生成汇总报告
# ============================================================================

echo "生成汇总报告..."

python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("${OUTPUT_DIR_FULL}")
supervision = "${SUPERVISION}"
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
            'supervision': summary.get('supervision', 'unknown'),
            'n_train_subjects': summary.get('n_train_subjects', 0),
            'best_epoch': summary.get('best_epoch', 0),
            'best_val_nll': summary.get('best_val_nll', float('inf')),
            'optimal_temperature': summary.get('optimal_temperature', 1.0),
        }

        # 验证集指标 (calibrated - 论文主表用)
        val_cal = summary.get('val_metrics_calibrated', {})
        fold_result['val_gross_acc'] = val_cal.get('gross_accuracy', 0)
        fold_result['val_top3_acc'] = val_cal.get('top3_accuracy', 0)
        fold_result['val_top5_acc'] = val_cal.get('top5_accuracy', 0)
        fold_result['val_nll'] = val_cal.get('nll', 0)
        fold_result['val_soft_ece'] = val_cal.get('soft_ece', 0)
        fold_result['val_macro_f1'] = val_cal.get('macro_f1', 0)
        fold_result['val_brier'] = val_cal.get('brier_score', 0)

        # 测试集指标 (calibrated - 论文主表用)
        test_cal = summary.get('test_metrics_calibrated', {})
        fold_result['test_gross_acc'] = test_cal.get('gross_accuracy', 0)
        fold_result['test_balanced_acc'] = test_cal.get('balanced_accuracy', 0)
        fold_result['test_top3_acc'] = test_cal.get('top3_accuracy', 0)
        fold_result['test_top5_acc'] = test_cal.get('top5_accuracy', 0)
        fold_result['test_nll'] = test_cal.get('nll', 0)
        fold_result['test_soft_ece'] = test_cal.get('soft_ece', 0)
        fold_result['test_macro_f1'] = test_cal.get('macro_f1', 0)
        fold_result['test_weighted_f1'] = test_cal.get('weighted_f1', 0)
        fold_result['test_kappa'] = test_cal.get('kappa', 0)
        fold_result['test_brier'] = test_cal.get('brier_score', 0)

        # Chapter 6 选择性预测指标 (calibrated)
        fold_result['test_risk_soft_at_95cov'] = test_cal.get('risk_soft_at_95cov', 0)
        fold_result['test_risk_hard_at_95cov'] = test_cal.get('risk_hard_at_95cov', 0)
        fold_result['test_coverage_at_5pct_risk_soft'] = test_cal.get('coverage_at_5pct_risk_soft', 0)

        # 3D指标 (calibrated)
        fold_result['val_3d_gross_acc'] = summary.get('val_3d_gross_acc_postT', 0)
        fold_result['val_3d_soft_dice_macro'] = summary.get('val_3d_soft_dice_macro_postT', 0)
        fold_result['test_3d_gross_acc'] = summary.get('test_3d_gross_acc_postT', 0)
        fold_result['test_3d_soft_dice_macro'] = summary.get('test_3d_soft_dice_macro_postT', 0)

        # 未校准指标（用于对比）
        val_uncal = summary.get('val_metrics_uncalibrated', {})
        test_uncal = summary.get('test_metrics_uncalibrated', {})
        fold_result['val_nll_uncal'] = val_uncal.get('nll', 0)
        fold_result['val_soft_ece_uncal'] = val_uncal.get('soft_ece', 0)
        fold_result['test_nll_uncal'] = test_uncal.get('nll', 0)
        fold_result['test_soft_ece_uncal'] = test_uncal.get('soft_ece', 0)

        results.append(fold_result)

    except Exception as e:
        print(f"错误: 读取 {fold_dir.name} 失败 - {e}")

if results:
    # 计算统计量
    def compute_stats(values):
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

    # 提取各指标
    test_accs = [r['test_gross_acc'] for r in results]
    test_balanced_accs = [r['test_balanced_acc'] for r in results]
    test_top3 = [r['test_top3_acc'] for r in results]
    test_top5 = [r['test_top5_acc'] for r in results]
    test_nlls = [r['test_nll'] for r in results]
    test_eces = [r['test_soft_ece'] for r in results]
    test_f1s = [r['test_macro_f1'] for r in results]
    test_weighted_f1s = [r['test_weighted_f1'] for r in results]
    test_kappas = [r['test_kappa'] for r in results]
    test_briers = [r['test_brier'] for r in results]
    test_3d_dices = [r['test_3d_soft_dice_macro'] for r in results]

    # Chapter 6 选择性预测指标
    test_risk_soft_95 = [r['test_risk_soft_at_95cov'] for r in results]
    test_risk_hard_95 = [r['test_risk_hard_at_95cov'] for r in results]
    test_cov_5pct_risk = [r['test_coverage_at_5pct_risk_soft'] for r in results]

    val_accs = [r['val_gross_acc'] for r in results]
    val_nlls = [r['val_nll'] for r in results]
    val_eces = [r['val_soft_ece'] for r in results]
    val_f1s = [r['val_macro_f1'] for r in results]

    # 打印汇总结果
    print('\n' + '=' * 80)
    print(f'Leave-One-Out Cross-Validation Results ({supervision.upper()})')
    print('=' * 80)
    print(f'完成训练: {len(results)}/{len(fold_dirs)} folds')
    print()

    print('=' * 80)
    print('论文主表指标 (Test Set - Calibrated)')
    print('=' * 80)
    print(f'  Top-1 (Gross Acc): {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}')
    print(f'  Balanced Acc:      {np.mean(test_balanced_accs):.4f} +/- {np.std(test_balanced_accs):.4f}')
    print(f'  Top-3 Accuracy:    {np.mean(test_top3):.4f} +/- {np.std(test_top3):.4f}')
    print(f'  Top-5 Accuracy:    {np.mean(test_top5):.4f} +/- {np.std(test_top5):.4f}')
    print(f'  NLL (q):           {np.mean(test_nlls):.4f} +/- {np.std(test_nlls):.4f}')
    print(f'  ECE^soft:          {np.mean(test_eces):.4f} +/- {np.std(test_eces):.4f}')
    print(f'  Macro-F1:          {np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f}')
    print(f'  Weighted-F1:       {np.mean(test_weighted_f1s):.4f} +/- {np.std(test_weighted_f1s):.4f}')
    print(f'  Kappa:             {np.mean(test_kappas):.4f} +/- {np.std(test_kappas):.4f}')
    print(f'  Brier Score:       {np.mean(test_briers):.4f} +/- {np.std(test_briers):.4f}')
    print(f'  3D Soft Dice:      {np.mean(test_3d_dices):.4f} +/- {np.std(test_3d_dices):.4f}')
    print()

    print('Chapter 6 选择性预测指标 (Test Set - Calibrated):')
    print(f'  Risk^soft @95%cov:      {np.mean(test_risk_soft_95):.4f} +/- {np.std(test_risk_soft_95):.4f}')
    print(f'  Risk^hard @95%cov:      {np.mean(test_risk_hard_95):.4f} +/- {np.std(test_risk_hard_95):.4f}')
    print(f'  Coverage @5%risk^soft:  {np.mean(test_cov_5pct_risk):.4f} +/- {np.std(test_cov_5pct_risk):.4f}')
    print()

    print('验证集指标 (Calibrated):')
    print(f'  Gross Accuracy: {np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}')
    print(f'  NLL:            {np.mean(val_nlls):.4f} +/- {np.std(val_nlls):.4f}')
    print(f'  ECE^soft:       {np.mean(val_eces):.4f} +/- {np.std(val_eces):.4f}')
    print(f'  Macro F1:       {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}')
    print()

    print('测试集准确率范围:')
    print(f'  最好: {np.max(test_accs):.4f}')
    print(f'  最差: {np.min(test_accs):.4f}')
    print('=' * 80)

    # 保存汇总结果
    summary_results = {
        'supervision': supervision,
        'n_folds': len(results),
        'paper_metrics': {
            # Hard-comparable
            'test_gross_acc': compute_stats(test_accs),
            'test_balanced_acc': compute_stats(test_balanced_accs),
            'test_top3_acc': compute_stats(test_top3),
            'test_top5_acc': compute_stats(test_top5),
            'test_macro_f1': compute_stats(test_f1s),
            'test_weighted_f1': compute_stats(test_weighted_f1s),
            'test_kappa': compute_stats(test_kappas),
            # Probabilistic
            'test_nll': compute_stats(test_nlls),
            'test_soft_ece': compute_stats(test_eces),
            'test_brier': compute_stats(test_briers),
            'test_3d_soft_dice_macro': compute_stats(test_3d_dices),
            # Chapter 6 选择性预测指标
            'test_risk_soft_at_95cov': compute_stats(test_risk_soft_95),
            'test_risk_hard_at_95cov': compute_stats(test_risk_hard_95),
            'test_coverage_at_5pct_risk_soft': compute_stats(test_cov_5pct_risk),
        },
        'val_metrics': {
            'val_gross_acc': compute_stats(val_accs),
            'val_nll': compute_stats(val_nlls),
            'val_soft_ece': compute_stats(val_eces),
            'val_macro_f1': compute_stats(val_f1s),
        },
        'fold_results': results
    }

    summary_path = output_dir / 'cross_validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f'\n汇总结果已保存到: {summary_path}')

    # 创建CSV格式的结果表（论文主表格式）
    import csv
    csv_path = output_dir / 'cross_validation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'fold_name', 'test_subject', 'val_subject', 'supervision',
            'test_gross_acc', 'test_balanced_acc', 'test_top3_acc', 'test_top5_acc',
            'test_macro_f1', 'test_weighted_f1', 'test_kappa',
            'test_nll', 'test_soft_ece', 'test_brier',
            'test_3d_soft_dice_macro',
            'test_risk_soft_at_95cov', 'test_risk_hard_at_95cov', 'test_coverage_at_5pct_risk_soft',
            'val_gross_acc', 'val_nll', 'val_soft_ece', 'val_macro_f1',
            'optimal_temperature', 'best_epoch'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f'CSV结果表已保存到: {csv_path}')

    # 生成论文主表格式的汇总（方便复制）
    paper_table_path = output_dir / 'paper_table_summary.txt'
    with open(paper_table_path, 'w') as f:
        f.write(f"# Leave-One-Out Cross-Validation Results\n")
        f.write(f"# Supervision: {supervision.upper()}\n")
        f.write(f"# N Folds: {len(results)}\n\n")
        f.write("## Hard-Comparable Metrics (based on y=argmax(q))\n\n")
        f.write("| Metric | Mean +/- Std |\n")
        f.write("|--------|-------------|\n")
        f.write(f"| Top-1 (Gross Acc) | {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f} |\n")
        f.write(f"| Balanced Acc | {np.mean(test_balanced_accs):.4f} +/- {np.std(test_balanced_accs):.4f} |\n")
        f.write(f"| Top-3 | {np.mean(test_top3):.4f} +/- {np.std(test_top3):.4f} |\n")
        f.write(f"| Top-5 | {np.mean(test_top5):.4f} +/- {np.std(test_top5):.4f} |\n")
        f.write(f"| Macro-F1 | {np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f} |\n")
        f.write(f"| Weighted-F1 | {np.mean(test_weighted_f1s):.4f} +/- {np.std(test_weighted_f1s):.4f} |\n")
        f.write(f"| Kappa | {np.mean(test_kappas):.4f} +/- {np.std(test_kappas):.4f} |\n")
        f.write("\n")
        f.write("## Probabilistic Metrics (based on soft target q)\n\n")
        f.write("| Metric | Mean +/- Std |\n")
        f.write("|--------|-------------|\n")
        f.write(f"| NLL (q) | {np.mean(test_nlls):.4f} +/- {np.std(test_nlls):.4f} |\n")
        f.write(f"| ECE^soft | {np.mean(test_eces):.4f} +/- {np.std(test_eces):.4f} |\n")
        f.write(f"| Brier | {np.mean(test_briers):.4f} +/- {np.std(test_briers):.4f} |\n")
        f.write(f"| 3D Soft Dice | {np.mean(test_3d_dices):.4f} +/- {np.std(test_3d_dices):.4f} |\n")
        f.write("\n")
        f.write("## Chapter 6 Selective Prediction Metrics\n\n")
        f.write("| Metric | Mean +/- Std |\n")
        f.write("|--------|-------------|\n")
        f.write(f"| Risk^soft @95%cov | {np.mean(test_risk_soft_95):.4f} +/- {np.std(test_risk_soft_95):.4f} |\n")
        f.write(f"| Risk^hard @95%cov | {np.mean(test_risk_hard_95):.4f} +/- {np.std(test_risk_hard_95):.4f} |\n")
        f.write(f"| Coverage @5%risk^soft | {np.mean(test_cov_5pct_risk):.4f} +/- {np.std(test_cov_5pct_risk):.4f} |\n")
        f.write("\n")
        f.write("## Metric Definitions\n")
        f.write("- gross_accuracy = Top-1^hard: argmax(p) == argmax(q)\n")
        f.write("- balanced_accuracy = mean recall per class (only over classes present in test fold)\n")
        f.write("- kappa = Cohen's kappa coefficient\n")
        f.write("- soft_ece = ECE^soft: calibration against soft labels q\n")
        f.write("- nll = NLL(q, p): negative log-likelihood against soft labels\n")
        f.write("- risk_soft_at_95cov: 1 - mean(q[i, argmax(p_i)]) over top 95% confident samples\n")
        f.write("- risk_hard_at_95cov: 1 - mean(1[argmax(p)==argmax(q)]) over top 95% confident samples\n")
        f.write("- coverage_at_5pct_risk_soft: max coverage where risk^soft <= 5%\n")
        f.write("- All metrics computed AFTER temperature scaling calibration (log_T parameterization)\n")

    print(f'论文主表汇总已保存到: {paper_table_path}')

else:
    print('\n未找到任何有效的fold结果')

print()
EOF

# 更新配置文件，添加完成时间
cat >> ${OUTPUT_DIR_FULL}/config.txt <<EOF

Completed: $(date)
Duration: ${HOURS} hours ${MINUTES} minutes
EOF

echo ""
echo "汇总报告生成完成！"
echo ""
echo "========================================"
echo "查看结果:"
echo "  - 配置信息: ${OUTPUT_DIR_FULL}/config.txt"
echo "  - JSON汇总: ${OUTPUT_DIR_FULL}/cross_validation_summary.json"
echo "  - CSV表格:  ${OUTPUT_DIR_FULL}/cross_validation_results.csv"
echo "  - 论文主表: ${OUTPUT_DIR_FULL}/paper_table_summary.txt"
echo "  - 各fold详细结果: ${OUTPUT_DIR_FULL}/fold_*/"
echo "========================================"
echo ""
echo "指标定义 (论文口径):"
echo "  - gross_accuracy = Top-1^hard (argmax(p) == argmax(q))"
echo "  - soft_ece = ECE^soft (calibration against q)"
echo "  - nll = NLL(q, p) (soft cross-entropy)"
echo "  - 所有指标基于温度缩放校准后的预测"
echo "========================================"
