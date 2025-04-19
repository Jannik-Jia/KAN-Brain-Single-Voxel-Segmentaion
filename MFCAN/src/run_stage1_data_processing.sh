#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置日志文件路径
LOG_FILE="logs/shell/stage1_data_analysis_$(date +%Y%m%d_%H%M%S).log"

# 设置参数
MAX_CLASSES=20           # 只处理前20个类别
MAX_SAMPLES=1000         # 每个类别最多处理1000个样本
USE_GPU=true             # 启用GPU加速
BATCH_SIZE=10            # 类别批处理大小

echo "Starting Stage 1 Data Analysis with optimized parameters..."
echo "Max Classes: $MAX_CLASSES"
echo "Max Samples per Class: $MAX_SAMPLES"
echo "Use GPU: $USE_GPU"
echo "Batch Size: $BATCH_SIZE"
echo "Log file: $LOG_FILE"

# 运行阶段一数据分析脚本
nohup python scripts/stage1_data_analysis.py \
    --config configs/data_config.json \
    --max_classes $MAX_CLASSES \
    --max_samples $MAX_SAMPLES \
    $([ "$USE_GPU" = true ] && echo "--use_gpu") \
    --batch_size $BATCH_SIZE \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "Process started with PID: $PID"
echo "To check progress, use: tail -f $LOG_FILE"
echo "To check if process is running, use: ps -p $PID"