#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 默认值
MODEL_PATH="./best_model_results/BrainVoxel_BestParams_20250513_090622_epoch_15_acc_0.7078_f1_0.6980.pth"
DATA_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat"
OUTPUT_DIR="./prediction_results"
INTERACTIVE=""
FEATURES_KEY=""
REGION_KEY=""
TRAIN_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train"
TEST_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test"
VAL_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"

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
    --interactive)
      INTERACTIVE="--interactive"
      shift
      ;;
    --features_key)
      FEATURES_KEY="--features_key $2"
      shift 2
      ;;
    --region_key)
      REGION_KEY="--region_key $2"
      shift 2
      ;;
    --train_dir)
      TRAIN_DIR="$2"
      shift 2
      ;;
    --test_dir)
      TEST_DIR="$2"
      shift 2
      ;;
    --val_dir)
      VAL_DIR="$2"
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
  --threshold 0.5 \
  --train_dir "$TRAIN_DIR" \
  --test_dir "$TEST_DIR" \
  --val_dir "$VAL_DIR" \
  $INTERACTIVE \
  $FEATURES_KEY \
  $REGION_KEY

echo "预测完成! 结果已保存到: $OUTPUT_DIR"