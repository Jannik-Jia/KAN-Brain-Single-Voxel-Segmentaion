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

# 实验名称 - 添加患者数据标识
EXPERIMENT_NAME="BrainVoxel_MLP_PatientBased_$(date +%Y%m%d_%H%M%S)"

# 创建日志目录
mkdir -p logs

# 运行代码 - 更新命令行参数支持患者数据加载和优化贝叶斯优化
nohup python -u main.py \
    --experiment_name $EXPERIMENT_NAME \
    --use_patient_based_loading \
    --patient_data_base_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/reorganized_fold_data" \
    --batch_size 128 \
    --epochs 30 \
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
    --n_trials 30 \
    --bo_max_epochs 100 \
    --early_stop_patience 10 \
    --pruner_type "hyperband" \
    --norm True \
    --save_scaler True \
    > logs/${EXPERIMENT_NAME}.log 2>&1 &

PID=$!
echo "Started experiment $EXPERIMENT_NAME. PID: $PID"
echo "Check logs with: tail -f logs/${EXPERIMENT_NAME}.log"
echo "Or monitor: watch -n 1 'tail -n 20 logs/${EXPERIMENT_NAME}.log'"