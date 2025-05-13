#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 指定模型路径和数据路径
MODEL_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/Fullyconnected/best_model_results/BrainVoxel_BestParams_20250513_141710_epoch_9_acc_0.7071_f1_0.6970.pth"
DATA_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat"

# 运行预测
python predict_standardized.py \
  --model "$MODEL_PATH" \
  --data "$DATA_PATH" \
  --output "./prediction_results_correctly_standardized" \
  --batch_size 128

echo "预测完成! 结果已保存到: ./prediction_results_correctly_standardized"