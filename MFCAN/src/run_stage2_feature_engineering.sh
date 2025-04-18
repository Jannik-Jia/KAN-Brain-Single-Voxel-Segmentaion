#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置日志文件路径
LOG_FILE="logs/shell/stage2_feature_engineering_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Stage 2 Feature Engineering..."
echo "Log file: $LOG_FILE"

# 设置最新分析结果目录（如果有）
ANALYSIS_DIR=$(ls -td results/analysis/20* | head -1)
if [ -n "$ANALYSIS_DIR" ]; then
  ANALYSIS_PARAM="--analysis_dir $ANALYSIS_DIR"
  echo "Using latest analysis results: $ANALYSIS_DIR"
else
  ANALYSIS_PARAM=""
  echo "No analysis directory specified"
fi

# 运行阶段二特征工程脚本
nohup python scripts/stage2_feature_engineering.py \
    --config configs/base_config.json \
    --data_path data/processed/feature_groups.h5 \
    --output_dir results/feature_engineering \
    $ANALYSIS_PARAM \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "Process started with PID: $PID"
echo "To check progress, use: tail -f $LOG_FILE"
echo "To check if process is running, use: ps -p $PID"