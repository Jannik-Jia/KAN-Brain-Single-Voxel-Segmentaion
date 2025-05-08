#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
mkdir -p logs
mkdir -p results

# 实验名称
EXPERIMENT_NAME="BrainVoxel_MLP_$(date +%Y%m%d_%H%M%S)"

# 运行代码
nohup python main.py \
    --experiment_name $EXPERIMENT_NAME \
    --train_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train" \
    --test_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test" \
    --val_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val" \
    --batch_size 128 \
    --epochs 30 \
    --val_epochs 3 \
    --device 0 \
    --model_type "base_mlp" \
    --hidden_units "4096,4096,4096,4096" \
    --activation "swish" \
    --dropout_rate 0.5 \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --optimizer "adamw" \
    --use_lr_scheduler \
    --lr_scheduler_type "cosine" \
    --old_serialization \
    --save_dir "./results" \
    --log_dir "./logs" \
    --run_bayesian_opt \
    --n_trials 50 \
    > logs/${EXPERIMENT_NAME}.log 2>&1 &

echo "Started experiment $EXPERIMENT_NAME. PID: $!"
echo "Check logs with: tail -f logs/${EXPERIMENT_NAME}.log"