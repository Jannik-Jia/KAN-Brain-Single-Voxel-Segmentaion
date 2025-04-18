#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置日志文件路径
LOG_FILE="logs/shell/stage1_data_analysis_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Stage 1 Data Analysis..."
echo "Log file: $LOG_FILE"

# 运行阶段一数据分析脚本
nohup python scripts/stage1_data_analysis.py \
    --config configs/data_config.json \
    --data_path data/processed/feature_groups.h5 \
    --output_dir results/analysis \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "Process started with PID: $PID"
echo "To check progress, use: tail -f $LOG_FILE"
echo "To check if process is running, use: ps -p $PID"