#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# 创建预处理脚本
cat > prepare_env.py << 'EOL'
import torch
import torch.serialization
import numpy as np
import sys

# 禁用tqdm进度条
import tqdm
tqdm.tqdm = lambda *args, **kwargs: args[0]

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print("禁用tqdm进度条")
sys.stdout.flush()

try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    print("Successfully added numpy.core.multiarray.scalar to safe globals")
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")
EOL

# 运行预处理脚本
python prepare_env.py

# 实验名称 - 使用最优参数
EXPERIMENT_NAME="BrainVoxel_BestParams_MAT_$(date +%Y%m%d_%H%M%S)"

# 创建日志目录
mkdir -p logs

# 运行代码 - 使用最优参数和MAT格式
nohup python -u main.py \
    --experiment_name $EXPERIMENT_NAME \
    --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat" \
    --demo_mat_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat" \
    --batch_size 128 \
    --epochs 50 \
    --device 0 \
    --model_type "base_mlp" \
    --hidden_units "4096,4096,4096,4096" \
    --activation "gelu" \
    --dropout_rate 0.1840571430245583 \
    --lr 9.247073789581185e-05 \
    --weight_decay 1.041622803083192e-06 \
    --optimizer "adamw" \
    --use_lr_scheduler \
    --lr_scheduler_type "step" \
    --old_serialization \
    --save_dir "./results/best_params" \
    --log_dir "./logs" \
    > logs/${EXPERIMENT_NAME}.log 2>&1 &

PID=$!
echo "Started experiment $EXPERIMENT_NAME with best parameters. PID: $PID"
echo "Check logs with: tail -f logs/${EXPERIMENT_NAME}.log"
echo "Or monitor: watch -n 1 'tail -n 20 logs/${EXPERIMENT_NAME}.log'"