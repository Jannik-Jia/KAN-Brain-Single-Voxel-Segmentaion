#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 默认值
MODEL_PATH="./results/latest_model/best_model.pth"
DATA_PATH="./demo/DEMO38.mat"
OUTPUT_DIR="./prediction_results"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行预测脚本
python predict.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device 0 \
  --normalize \
  --save_3d \
  --colormap "jet" \
  --threshold 0.5

echo "预测完成! 结果已保存到: $OUTPUT_DIR"