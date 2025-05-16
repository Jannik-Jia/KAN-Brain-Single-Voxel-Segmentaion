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
    > logs/${EXPERIMENT_NAME}.log 2>&1 &

PID=$!
echo "Started experiment $EXPERIMENT_NAME. PID: $PID"
echo "Check logs with: tail -f logs/${EXPERIMENT_NAME}.log"
echo "Or monitor: watch -n 1 'tail -n 20 logs/${EXPERIMENT_NAME}.log'"

# 创建验证最佳模型的辅助脚本
cat > verify_best_model.sh << 'EOL'
#!/bin/bash

# 使用第一个命令行参数作为实验名称，如果未提供则使用最近的实验
EXPERIMENT_NAME=$1
if [ -z "$EXPERIMENT_NAME" ]; then
    # 查找最近创建的实验目录
    EXPERIMENT_NAME=$(ls -td ./results/BrainVoxel_MLP_* | head -n 1 | xargs basename)
    echo "No experiment name provided, using most recent: $EXPERIMENT_NAME"
fi

BEST_MODEL_DIR="./results/$EXPERIMENT_NAME/best_model"

# 检查最佳模型是否已生成
if [ ! -d "$BEST_MODEL_DIR" ]; then
    echo "Error: Best model directory not found: $BEST_MODEL_DIR"
    echo "The optimization might still be running or failed."
    exit 1
fi

echo "====== 最佳模型信息 ======"
cd "$BEST_MODEL_DIR"

# 如果architecture.json存在，显示模型架构信息
if [ -f "architecture.json" ]; then
    echo "Model architecture:"
    cat architecture.json | python -m json.tool
else
    echo "Warning: architecture.json not found"
fi

# 如果best_params.json存在，显示最佳参数
if [ -f "best_params.json" ]; then
    echo -e "\nBest parameters:"
    cat best_params.json | python -m json.tool
else
    echo "Warning: best_params.json not found"
fi

# 检查部署文件
if [ -d "deployment" ]; then
    echo -e "\nDeployment files:"
    ls -la deployment/
    
    # 检查推理脚本是否存在
    if [ -f "inference.py" ]; then
        echo -e "\nInference script is available."
    fi
else
    echo -e "\nWarning: Deployment directory not found"
fi

echo -e "\n====== 验证完成 ======"
EOL

chmod +x verify_best_model.sh
echo "Created verification script: ./verify_best_model.sh"

# 创建一个简易的推理演示脚本
cat > run_inference_demo.sh << 'EOL'
#!/bin/bash

# 使用第一个命令行参数作为实验名称，如果未提供则使用最近的实验
EXPERIMENT_NAME=$1
if [ -z "$EXPERIMENT_NAME" ]; then
    # 查找最近创建的实验目录
    EXPERIMENT_NAME=$(ls -td ./results/BrainVoxel_MLP_* | head -n 1 | xargs basename)
    echo "No experiment name provided, using most recent: $EXPERIMENT_NAME"
fi

BEST_MODEL_DIR="./results/$EXPERIMENT_NAME/best_model"

# 检查最佳模型是否已生成
if [ ! -d "$BEST_MODEL_DIR" ]; then
    echo "Error: Best model directory not found: $BEST_MODEL_DIR"
    exit 1
fi

# 进入最佳模型目录
cd "$BEST_MODEL_DIR"

# 检查推理脚本是否存在
if [ ! -f "inference.py" ]; then
    echo "Error: inference.py script not found"
    exit 1
fi

# 运行推理演示
echo "Running inference demo..."
python inference.py

echo "Inference demo completed!"
EOL

chmod +x run_inference_demo.sh
echo "Created inference demo script: ./run_inference_demo.sh"

# 等待1秒后开始显示日志
sleep 1
echo "Initial log output:"
tail -n 20 logs/${EXPERIMENT_NAME}.log
echo -e "\nExperiment is running in the background. Use the monitoring commands above to check progress."