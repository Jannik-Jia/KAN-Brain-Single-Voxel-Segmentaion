#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置默认参数
MODEL_TYPE="deep_mlp"  # 可选: mlp, group_mlp, deep_mlp
FEATURE_TYPE="selected"  # 可选: original, selected, pca, combined

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --feature_type)
      FEATURE_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 设置日志文件路径
LOG_FILE="logs/shell/stage2_baseline_training_${MODEL_TYPE}_${FEATURE_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Stage 2 Baseline Training..."
echo "Model Type: $MODEL_TYPE"
echo "Feature Type: $FEATURE_TYPE"
echo "Log file: $LOG_FILE"

# 设置最新特征工程结果目录
FEATURE_ENG_DIR=$(ls -td results/feature_engineering/20* | head -1)
echo "Using feature engineering results from: $FEATURE_ENG_DIR"

# 设置特征文件路径参数
if [ "$FEATURE_TYPE" = "selected" ]; then
  FEATURE_PARAMS="--selected_features $FEATURE_ENG_DIR/selected_features.h5"
elif [ "$FEATURE_TYPE" = "pca" ]; then
  FEATURE_PARAMS="--transformed_features $FEATURE_ENG_DIR/transformed_features.h5"
elif [ "$FEATURE_TYPE" = "combined" ]; then
  FEATURE_PARAMS="--selected_features $FEATURE_ENG_DIR/selected_features.h5 --transformed_features $FEATURE_ENG_DIR/transformed_features.h5"
else
  FEATURE_PARAMS=""
fi

# 运行阶段二基线训练脚本
nohup python scripts/stage2_baseline_training.py \
    --config configs/base_config.json \
    --model_type $MODEL_TYPE \
    --feature_type $FEATURE_TYPE \
    $FEATURE_PARAMS \
    --output_dir results/baseline \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "Process started with PID: $PID"
echo "To check progress, use: tail -f $LOG_FILE"
echo "To check if process is running, use: ps -p $PID"