#!/bin/bash

# 集成版本运行脚本 - 支持原版和MAT两种数据格式
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
import os
os.environ['TQDM_DISABLE'] = '1'

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print("环境准备完成")
sys.stdout.flush()

try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    print("Successfully added numpy.core.multiarray.scalar to safe globals")
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")
EOL

# 运行预处理脚本
python prepare_env.py

# 创建日志目录
mkdir -p logs
echo "=========================================="
echo "脑体素分类训练脚本 - 集成版本"
echo "支持原版格式和MAT格式数据"
echo "支持可配置的背景处理"
echo "=========================================="

# 🔧 新增：背景处理选项
echo "请选择背景处理模式:"
echo "1) 过滤背景 (默认，保持向后兼容)"
echo "2) 保留背景但忽略 (背景→ignore_index=-1)"
echo "3) 背景作为分类类别 (103个类别)"
read -p "请输入选择 (1/2/3, 默认1): " bg_choice

case $bg_choice in
    2)
        echo "使用保留背景但忽略模式"
        BG_PARAMS="--filter_background False --include_background_in_classes False --background_label_target -1"
        EXPERIMENT_SUFFIX="_BGIgnore"
        ;;
    3)
        echo "使用背景作为分类类别模式"
        BG_PARAMS="--filter_background False --include_background_in_classes True --background_label_target 0 --num_class 103"
        EXPERIMENT_SUFFIX="_BGClass"
        ;;
    *)
        echo "使用过滤背景模式（默认）"
        BG_PARAMS="--filter_background True"
        EXPERIMENT_SUFFIX="_BGFilter"
        ;;
esac

# 检查用户选择的数据格式
echo "请选择数据格式:"
echo "1) 原版格式 (分散的.npy文件)"
echo "2) MAT格式 (单个TRAIN38.mat文件)"
echo "3) 自动检测"
read -p "请输入选择 (1/2/3, 默认3): " format_choice

case $format_choice in
    1)
        echo "使用原版数据格式"
        DATA_FORMAT="original"
        ;;
    2)
        echo "使用MAT数据格式"
        DATA_FORMAT="mat"
        ;;
    *)
        echo "使用自动检测"
        DATA_FORMAT="auto"
        ;;
esac

# 根据数据格式设置参数
if [ "$DATA_FORMAT" = "original" ]; then
    # 原版格式参数
    EXPERIMENT_NAME="BrainVoxel_MLP_Original${EXPERIMENT_SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    
    # 检查是否启用贝叶斯优化
    read -p "是否启用贝叶斯优化? (y/n, 默认y): " enable_bayes
    if [[ $enable_bayes =~ ^[Nn]$ ]]; then
        BAYES_PARAMS=""
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_NoBayes"
    else
        BAYES_PARAMS="--run_bayesian_opt --n_trials 30"
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_Bayes"
    fi
    
    nohup python -u main.py \
        --experiment_name $EXPERIMENT_NAME \
        --train_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train" \
        --test_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test" \
        --val_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val" \
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
        $BG_PARAMS \
        $BAYES_PARAMS \
        > logs/${EXPERIMENT_NAME}.log 2>&1 &

elif [ "$DATA_FORMAT" = "mat" ]; then
    # MAT格式参数
    EXPERIMENT_NAME="BrainVoxel_MLP_MAT${EXPERIMENT_SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    
    # 检查是否启用贝叶斯优化
    read -p "是否启用贝叶斯优化? (y/n, 默认y): " enable_bayes
    if [[ $enable_bayes =~ ^[Nn]$ ]]; then
        BAYES_PARAMS=""
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_NoBayes"
    else
        BAYES_PARAMS="--run_bayesian_opt --n_trials 30"
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_Bayes"
    fi
    
    nohup python -u main.py \
        --experiment_name $EXPERIMENT_NAME \
        --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat" \
        --demo_mat_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat" \
        --test_size 0.01 \
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
        $BG_PARAMS \
        $BAYES_PARAMS \
        > logs/${EXPERIMENT_NAME}.log 2>&1 &

else
    # 自动检测格式
    EXPERIMENT_NAME="BrainVoxel_MLP_Auto${EXPERIMENT_SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    
    # 检查是否启用贝叶斯优化
    read -p "是否启用贝叶斯优化? (y/n, 默认y): " enable_bayes
    if [[ $enable_bayes =~ ^[Nn]$ ]]; then
        BAYES_PARAMS=""
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_NoBayes"
    else
        BAYES_PARAMS="--run_bayesian_opt --n_trials 30"
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_Bayes"
    fi
    
    # 提供两种数据路径，让配置自动检测
    nohup python -u main.py \
        --experiment_name $EXPERIMENT_NAME \
        --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat" \
        --train_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train" \
        --test_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test" \
        --val_dir "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val" \
        --batch_size 128 \
        --epochs 30 \
        --device 0 \
        --old_serialization \
        --save_dir "./results" \
        --log_dir "./logs" \
        $BG_PARAMS \
        $BAYES_PARAMS \
        > logs/${EXPERIMENT_NAME}.log 2>&1 &
fi

PID=$!
echo "=========================================="
echo "实验启动成功!"
echo "实验名称: $EXPERIMENT_NAME"
echo "进程ID: $PID"
echo "数据格式: $DATA_FORMAT"
echo "背景处理参数: $BG_PARAMS"
echo "=========================================="
echo ""
echo "监控命令:"
echo "实时日志: tail -f logs/${EXPERIMENT_NAME}.log"
echo "定期检查: watch -n 5 'tail -n 20 logs/${EXPERIMENT_NAME}.log'"
echo ""
echo "停止实验: kill $PID"
echo "查看进程: ps aux | grep $PID"
echo ""

# 显示初始几行日志
echo "初始日志预览:"
echo "----------------------------------------"
sleep 2
head -n 20 logs/${EXPERIMENT_NAME}.log 2>/dev/null || echo "日志文件正在生成中..."