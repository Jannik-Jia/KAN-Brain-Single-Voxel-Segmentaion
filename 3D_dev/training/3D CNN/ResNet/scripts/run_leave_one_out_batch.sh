#!/bin/bash
# Leave-One-Out Cross-Validation for MRI ResNet (批量加载版本)
#
# 自动为所有38个被试运行Leave-One-Out训练，使用批量数据加载避免内存不足问题。
# 每个被试轮流作为测试集，其余37个作为训练集。
#
# 用法:
#     bash run_leave_one_out_batch.sh
#
# 配置:
#     修改下面的变量来调整训练参数。

set -e  # 遇到错误时退出

# ==================== 配置 ====================

# 数据目录设置 - 与3D CNN基线相同
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据 (2D patch提取)
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"              # 1D数据

# ResNet使用3D数据（用于2D patch提取）
DATA_DIR=$DATA_DIR_3D

# 输出目录
OUTPUT_BASE="./results_leave_one_out_resnet_batch"

# ResNet配置
BASE_WIDTH=104          # 约50M参数
INPUT_CHANNELS=351      # MRI特征通道数
NUM_CLASSES=102         # 脑区数量
PATCH_SIZE=7            # 7×7 patches

# 批量加载配置（重要！）
BATCH_FILES=3           # 每次加载的文件数量（根据服务器内存调整）
                        # 内存 >= 16GB: 3-4个文件
                        # 内存 8-16GB: 2-3个文件
                        # 内存 < 8GB:  1-2个文件

# 训练参数
EPOCHS=100                 # ResNet可能需要比3D CNN更多的epoch
BATCH_SIZE=256             # 批量版本可以使用较小的batch size
LEARNING_RATE=0.0001       # 稍微降低学习率
WEIGHT_DECAY=0.0001
PATIENCE=15

# 数据参数
SAMPLES_PER_SUBJECT=10000  # 匹配3D CNN基线默认值
NUM_WORKERS=4              # 批量版本建议减少worker数量

# 损失和优化
LOSS_TYPE="cb_focal"       # cb_focal, logit_adj, focal, weighted_ce, ce
# USE_MIXUP="--use_mixup"  # 可选启用Mixup
MIXUP_ALPHA=0.2
USE_EMA="--use_ema"

# 硬件
DEVICE="cuda"
SEED=42

# 其他选项
VERBOSE="--verbose"

# ==================== 验证 ====================

echo "=== MRI ResNet Leave-One-Out Cross-Validation (批量加载版本) ==="
echo "数据目录: $DATA_DIR (与3D CNN基线相同)"
echo "输出目录: $OUTPUT_BASE"
echo "批量加载配置: 每次加载 $BATCH_FILES 个文件"
echo "配置:"
echo "  - 基础宽度: $BASE_WIDTH (≈50M参数)"
echo "  - Patch大小: ${PATCH_SIZE}×${PATCH_SIZE}"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 批量文件数: $BATCH_FILES"
echo "  - 损失类型: $LOSS_TYPE"
echo "  - Epochs: $EPOCHS"
echo "  - 设备: $DEVICE"

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    echo "请修改此脚本中的DATA_DIR指向你的MAT文件。"
    exit 1
fi

# 检查MAT文件是否存在
MAT_COUNT=$(find "$DATA_DIR" -name "*.mat" | wc -l)
if [ "$MAT_COUNT" -eq 0 ]; then
    echo "错误: 在 $DATA_DIR 中未找到MAT文件"
    echo "请确保目录包含.mat文件。"
    exit 1
fi

echo "在数据目录中找到 $MAT_COUNT 个MAT文件"

# 检查Python脚本是否存在
TRAIN_SCRIPT="./train_mri_resnet_batch.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本未找到: $TRAIN_SCRIPT"
    echo "请确保从scripts目录运行此脚本。"
    exit 1
fi

# 如果使用CUDA，检查GPU可用性
if [ "$DEVICE" = "cuda" ]; then
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'"
    if [ $? -ne 0 ]; then
        echo "错误: CUDA不可用。设置DEVICE='cpu'或安装CUDA。"
        exit 1
    fi

    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEMORY=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')")
    echo "使用GPU: $GPU_NAME (数量: $GPU_COUNT, 内存: $GPU_MEMORY)"

    # 根据GPU内存给出批量文件数建议
    GPU_MEMORY_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory/1024**3))")
    if [ "$GPU_MEMORY_GB" -lt 12 ]; then
        echo "警告: GPU内存较少 (${GPU_MEMORY_GB}GB)，建议设置 BATCH_FILES=2 和 BATCH_SIZE=128"
    fi
fi

# 检查系统内存
SYSTEM_MEMORY=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().total/1024**3:.1f}GB')")
AVAILABLE_MEMORY=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().available/1024**3:.1f}GB')")
echo "系统内存: $SYSTEM_MEMORY (可用: $AVAILABLE_MEMORY)"

# 根据可用内存给出建议
AVAILABLE_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available/1024**3))")
if [ "$AVAILABLE_GB" -lt 8 ]; then
    echo "⚠️  警告: 可用内存不足8GB，强烈建议设置:"
    echo "   BATCH_FILES=2"
    echo "   BATCH_SIZE=128"
    echo "   SAMPLES_PER_SUBJECT=5000"
elif [ "$AVAILABLE_GB" -lt 16 ]; then
    echo "💡 建议: 可用内存${AVAILABLE_GB}GB，建议设置:"
    echo "   BATCH_FILES=2-3"
    echo "   BATCH_SIZE=256"
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 保存配置
CONFIG_FILE="$OUTPUT_BASE/training_config.txt"
cat > "$CONFIG_FILE" << EOF
MRI ResNet Leave-One-Out 批量加载配置
=====================================
日期: $(date)
主机: $(hostname)
用户: $(whoami)
工作目录: $(pwd)

数据配置:
- 数据目录: $DATA_DIR (3D验证数据用于2D patch提取)
- 替代1D目录: $DATA_DIR_1D (ResNet不使用)
- 找到MAT文件: $MAT_COUNT
- Patch大小: ${PATCH_SIZE}×${PATCH_SIZE}
- 输入通道: $INPUT_CHANNELS
- 输出类别: $NUM_CLASSES
- 每个被试样本数: ${SAMPLES_PER_SUBJECT:-"所有有效样本"}

批量加载配置:
- 每批次文件数: $BATCH_FILES
- 内存节省估计: ~$((100 - 100 * BATCH_FILES / MAT_COUNT))%
- 系统内存: $SYSTEM_MEMORY (可用: $AVAILABLE_MEMORY)

模型配置:
- 架构: ResNet-50 (3-4-6-3 Bottleneck)
- 基础宽度: $BASE_WIDTH (~50M参数)
- Stem: 3×3/s1/p1, expand-first (351→512通道)
- 空间流: 7→7→4→2→1
- 通道流: 512→256→512→1024→2048→102

训练配置:
- Epochs: $EPOCHS
- 批次大小: $BATCH_SIZE
- 学习率: $LEARNING_RATE
- 权重衰减: $WEIGHT_DECAY
- 早停耐心: $PATIENCE
- 损失函数: $LOSS_TYPE
- Mixup: $([ -n "$USE_MIXUP" ] && echo "启用 (α=$MIXUP_ALPHA)" || echo "禁用")
- EMA: $([ -n "$USE_EMA" ] && echo "启用" || echo "禁用")

硬件:
- 设备: $DEVICE
- Workers: $NUM_WORKERS
- 随机种子: $SEED
EOF

echo ""
echo "配置已保存到: $CONFIG_FILE"
echo ""

# ==================== 训练循环 ====================

# 构建命令参数
ARGS=""
ARGS="$ARGS --data_dir $DATA_DIR"
ARGS="$ARGS --output_dir $OUTPUT_BASE"
ARGS="$ARGS --batch_files $BATCH_FILES"  # 批量加载关键参数
ARGS="$ARGS --base_width $BASE_WIDTH"
ARGS="$ARGS --input_channels $INPUT_CHANNELS"
ARGS="$ARGS --num_classes $NUM_CLASSES"
ARGS="$ARGS --epochs $EPOCHS"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --learning_rate $LEARNING_RATE"
ARGS="$ARGS --weight_decay $WEIGHT_DECAY"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --patch_size $PATCH_SIZE"
ARGS="$ARGS --num_workers $NUM_WORKERS"
ARGS="$ARGS --loss_type $LOSS_TYPE"
ARGS="$ARGS --mixup_alpha $MIXUP_ALPHA"
ARGS="$ARGS --device $DEVICE"
ARGS="$ARGS --seed $SEED"

# 添加可选参数
[ -n "$SAMPLES_PER_SUBJECT" ] && ARGS="$ARGS --samples_per_subject $SAMPLES_PER_SUBJECT"
[ -n "$USE_MIXUP" ] && ARGS="$ARGS $USE_MIXUP"
[ -n "$USE_EMA" ] && ARGS="$ARGS $USE_EMA"
[ -n "$VERBOSE" ] && ARGS="$ARGS $VERBOSE"

# 开始训练循环
TOTAL_SUBJECTS=38
START_TIME=$(date +%s)

echo "开始为 $TOTAL_SUBJECTS 个被试进行Leave-One-Out训练..."
echo "使用批量加载，每次处理 $BATCH_FILES 个文件"
echo "根据硬件配置，这可能需要几个小时。"
echo ""

# 整体进度日志文件
PROGRESS_LOG="$OUTPUT_BASE/training_progress.log"
echo "Leave-One-Out 批量训练进度 - 开始于 $(date)" > "$PROGRESS_LOG"

# 内存监控函数
monitor_memory() {
    local subject=$1
    local log_file="$OUTPUT_BASE/memory_usage_subject_$subject.log"
    echo "时间,内存使用(GB),可用内存(GB)" > "$log_file"

    while [ -f "/tmp/training_subject_$subject.pid" ]; do
        python3 -c "
import psutil
import time
vm = psutil.virtual_memory()
used_gb = (vm.total - vm.available) / 1024**3
avail_gb = vm.available / 1024**3
print(f'{time.strftime(\"%H:%M:%S\")},{used_gb:.1f},{avail_gb:.1f}')
" >> "$log_file"
        sleep 30  # 每30秒记录一次
    done &
}

for test_subject in $(seq 1 $TOTAL_SUBJECTS); do
    echo "=========================================="
    echo "训练被试 $test_subject/$TOTAL_SUBJECTS"
    echo "测试被试: $test_subject"
    echo "=========================================="

    SUBJECT_START=$(date +%s)

    # 创建PID文件用于内存监控
    touch "/tmp/training_subject_$test_subject.pid"

    # 启动内存监控
    monitor_memory $test_subject
    MONITOR_PID=$!

    # 运行训练
    CMD="python3 $TRAIN_SCRIPT $ARGS --test_subject $test_subject"
    echo "命令: $CMD"
    echo ""

    if eval $CMD; then
        SUBJECT_END=$(date +%s)
        SUBJECT_TIME=$((SUBJECT_END - SUBJECT_START))

        echo ""
        echo "被试 $test_subject 成功完成，用时 ${SUBJECT_TIME}秒"
        echo "被试 $test_subject: 成功 (${SUBJECT_TIME}秒)" >> "$PROGRESS_LOG"

        # 检查结果是否存在
        SUBJECT_DIR="$OUTPUT_BASE/resnet_batch_test_subject_$test_subject"
        if [ -f "$SUBJECT_DIR/training_results.json" ]; then
            BEST_F1=$(python3 -c "
import json
try:
    with open('$SUBJECT_DIR/training_results.json') as f:
        data = json.load(f)
        print(f'{data[\"best_f1\"]:.4f}')
except:
    print('N/A')
" 2>/dev/null)
            echo "最佳F1分数: $BEST_F1"
            echo "  最佳F1: $BEST_F1" >> "$PROGRESS_LOG"

            # 记录批次信息
            BATCH_INFO=$(python3 -c "
import json
try:
    with open('$SUBJECT_DIR/training_results.json') as f:
        data = json.load(f)
        batch_info = data.get('batch_training_info', {})
        print(f'批次数: {batch_info.get(\"total_batches\", \"N/A\")}, 每批文件数: {batch_info.get(\"batch_files\", \"N/A\")}')
except:
    print('批次信息: N/A')
" 2>/dev/null)
            echo "  $BATCH_INFO" >> "$PROGRESS_LOG"
        fi
    else
        echo ""
        echo "错误: 被试 $test_subject 训练失败"
        echo "被试 $test_subject: 失败" >> "$PROGRESS_LOG"

        # 可选择继续或退出
        read -p "继续其余被试吗? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "用户停止训练。"
            exit 1
        fi
    fi

    # 清理监控
    rm -f "/tmp/training_subject_$test_subject.pid"
    kill $MONITOR_PID 2>/dev/null || true

    echo ""
done

# ==================== 完成 ====================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo "=========================================="
echo "Leave-One-Out 批量训练完成!"
echo "=========================================="
echo "总用时: ${HOURS}小时 ${MINUTES}分钟"
echo "结果保存在: $OUTPUT_BASE"
echo ""

# 摘要
echo "训练摘要:" >> "$PROGRESS_LOG"
echo "总用时: ${HOURS}小时 ${MINUTES}分钟" >> "$PROGRESS_LOG"
echo "完成时间: $(date)" >> "$PROGRESS_LOG"

# 统计成功运行次数
SUCCESS_COUNT=$(grep "成功" "$PROGRESS_LOG" | wc -l)
FAIL_COUNT=$(grep "失败" "$PROGRESS_LOG" | wc -l)

echo "成功运行: $SUCCESS_COUNT/$TOTAL_SUBJECTS"
echo "失败运行: $FAIL_COUNT/$TOTAL_SUBJECTS"

if [ $SUCCESS_COUNT -eq $TOTAL_SUBJECTS ]; then
    echo "🎉 所有被试都成功完成!"
else
    echo "⚠️  一些被试失败了。查看单独的日志了解详情。"
fi

echo ""
echo "下一步:"
echo "1. 运行分析脚本聚合结果:"
echo "   python3 analyze_resnet_results.py --results_dir $OUTPUT_BASE"
echo ""
echo "2. 与基线方法比较"
echo ""
echo "3. 生成详细报告和可视化"
echo ""

# 如果脚本存在，尝试自动运行分析
ANALYSIS_SCRIPT="./analyze_resnet_results.py"
if [ -f "$ANALYSIS_SCRIPT" ] && [ $SUCCESS_COUNT -gt 0 ]; then
    echo "运行自动分析..."
    python3 "$ANALYSIS_SCRIPT" --results_dir "$OUTPUT_BASE" || echo "分析脚本失败"
fi

# 生成内存使用摘要
echo ""
echo "内存使用摘要:"
for subject in $(seq 1 $SUCCESS_COUNT); do
    MEMORY_LOG="$OUTPUT_BASE/memory_usage_subject_$subject.log"
    if [ -f "$MEMORY_LOG" ]; then
        MAX_MEMORY=$(tail -n +2 "$MEMORY_LOG" | cut -d',' -f2 | sort -n | tail -1)
        echo "  被试 $subject: 最大内存使用 ${MAX_MEMORY}GB"
    fi
done

echo ""
echo "批量训练完成! 查看 $OUTPUT_BASE 获取所有结果。"
echo "内存节省: 约$((100 - 100 * BATCH_FILES / MAT_COUNT))% 相比原始方法"