#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置默认参数
MODALITY="all"  # 可选: diffusion, qti, cest, all
CONFIG_PATH="configs/encoders_config.json"
DATA_PATH="data/processed/feature_groups.h5"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --modality)
      MODALITY="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 设置日志文件路径
LOG_FILE="logs/shell/stage3_encoder_pretraining_${MODALITY}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Stage 3 Encoder Pretraining..."
echo "Modality: $MODALITY"
echo "Config file: $CONFIG_PATH"
echo "Data path: $DATA_PATH"
echo "Log file: $LOG_FILE"

# 设置最新特征工程结果目录
FEATURE_ENG_DIR=$(ls -td results/feature_engineering/20* | head -1)
echo "Using feature engineering results from: $FEATURE_ENG_DIR"

# 设置输出目录
OUTPUT_DIR="models/encoders/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 运行阶段三编码器预训练脚本
nohup python scripts/stage3_encoder_pretraining.py \
    --config "$CONFIG_PATH" \
    --modality "$MODALITY" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "Process started with PID: $PID"
echo "To check progress, use: tail -f $LOG_FILE"
echo "To check if process is running, use: ps -p $PID"