#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 实验名称（使用已训练模型的目录名）
EXPERIMENT_NAME="BrainVoxel_MLP_MAT_[DATE]_[TIME]" # 替换为您的实验目录名
MODEL_PATH="./results/$EXPERIMENT_NAME/best_model.pth" # 替换为最佳模型路径

# 运行评估
python evaluate.py \
    --model_path $MODEL_PATH \
    --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat" \
    --batch_size 128 \
    --device 0 \
    --save_dir "./results/$EXPERIMENT_NAME/evaluation"