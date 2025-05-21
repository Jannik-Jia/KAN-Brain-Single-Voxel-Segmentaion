#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# 实验名称
EXPERIMENT_NAME="BrainVoxel_BestParams_$(date +%Y%m%d_%H%M%S)"

# 创建日志目录
mkdir -p logs

# 运行最优参数训练脚本
nohup python -u train_best_params.py \
    --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat" \
    --experiment_name $EXPERIMENT_NAME \
    --device 0 \
    --batch_size 128 \
    --epochs 30 \
    --save_dir "./results" \
    --log_dir "./logs" \
    --seed 666 \
    --save_every 1 \     # 每个epoch保存一次
    --eval_every 2 \     # 每个epoch评估一次
    > logs/${EXPERIMENT_NAME}.log 2>&1 &

PID=$!
echo "Started experiment $EXPERIMENT_NAME with best parameters. PID: $PID"
echo "Check logs with: tail -f logs/${EXPERIMENT_NAME}.log"
echo "Or monitor: watch -n 1 'tail -n 20 logs/${EXPERIMENT_NAME}.log'"