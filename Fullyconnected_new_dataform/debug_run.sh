#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 创建预处理脚本来添加安全全局变量
cat > prepare_env.py << 'EOL'
import torch
import torch.serialization
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    print("Successfully added numpy.core.multiarray.scalar to safe globals")
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")
EOL

# 运行预处理脚本
python prepare_env.py

# 实验名称 - 添加DEBUG前缀
EXPERIMENT_NAME="DEBUG_BrainVoxel_MLP_$(date +%Y%m%d_%H%M%S)"

# 运行代码 - 使用最小参数
python main.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --train_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train" \
    --test_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test" \
    --val_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val" \
    --batch_size 128 \
    --epochs 1 \
    --device 0 \
    --model_type "base_mlp" \
    --hidden_units "1024,1024" \
    --activation "relu" \
    --dropout_rate 0.5 \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --optimizer "adam" \
    --save_dir "./debug_results" \
    --log_dir "./debug_logs"

echo "Debug run completed. Check for any errors."