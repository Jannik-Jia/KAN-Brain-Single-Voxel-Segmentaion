#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
# 禁用所有进度条
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

# 实验名称
EXPERIMENT_NAME="BrainVoxel_MLP_MAT_$(date +%Y%m%d_%H%M%S)"

# 创建日志目录
mkdir -p logs

# 运行代码 - 使用config.py中的默认配置，仅传递必要参数
nohup python -u main.py \
    --experiment_name $EXPERIMENT_NAME \
    --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat" \
    --demo_mat_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat" \
    --device 0 \
    --old_serialization \
    --save_dir "./results" \
    --log_dir "./logs" \
    > logs/${EXPERIMENT_NAME}.log 2>&1 &

PID=$!
echo "Started experiment $EXPERIMENT_NAME. PID: $PID"
echo "Check logs with: tail -f logs/${EXPERIMENT_NAME}.log"
echo "Or monitor: watch -n 1 'tail -n 20 logs/${EXPERIMENT_NAME}.log'"