#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 实验名称（使用已训练模型的目录名）
EXPERIMENT_NAME="BrainVoxel_MLP_MAT_[DATE]_[TIME]" # 替换为您的实验目录名
MODEL_PATH="./results/$EXPERIMENT_NAME/best_model.pth" # 替换为最佳模型路径

# 运行预测
python predict.py \
    --model_path $MODEL_PATH \
    --input_mat_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat" \
    --output_path "./results/$EXPERIMENT_NAME/predictions/demo38_predictions.mat" \
    --device 0