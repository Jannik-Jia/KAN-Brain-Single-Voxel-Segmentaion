#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 实验目录和模型路径
EXPERIMENT_DIR="./results/BrainVoxel_102Class_MLP_20250508_111619"
MODEL_PATH="${EXPERIMENT_DIR}/BrainVoxel_102Class_MLP_20250508_111619_epoch_30_acc_0.8643_f1_0.8516.pth"

# 创建日志目录
mkdir -p logs

# 运行评估代码
python evaluate.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --train_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train" \
    --test_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test" \
    --val_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val" \
    --device 0 \
    --batch_size 128 \
    > logs/evaluation_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "评估完成，查看日志: logs/evaluation_$(date +%Y%m%d_%H%M%S).log"