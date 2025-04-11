#!/bin/bash

# 实验类型设置
EXPERIMENT_TYPE="fs"  # 可选："pca" 或 "fs"（特征选择）

# 设置日志文件
LOG_FILE="full_experiments_${EXPERIMENT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

# 设置数据目录 - 请根据实际路径修改
TRAIN_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train"
TEST_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test"
VAL_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"

# 设置结果保存目录
EXPERIMENT_DIR="pseudoinverse_${EXPERIMENT_TYPE}_experiments_fixed_mode"

# 设置最大实验数量 - 增加实验数量以确保覆盖更多参数组合
MAX_EXPERIMENTS=300

# GPU相关设置
USE_GPU=true                # 是否使用GPU加速（true/false）
GPU_MEMORY_FRACTION=0.8     # GPU内存使用比例上限（0.0-1.0）

# 添加数据处理参数
MAX_SAMPLES_PER_LABEL=1   # 每个标签最多使用的样本数

# 特征选择时是否跳过PCA
SKIP_PCA=true               # 特征选择时是否完全跳过PCA（true/false）

# 构建GPU参数
GPU_ARGS=""
if [ "$USE_GPU" = true ]; then
    GPU_ARGS="--use_gpu --gpu_memory_fraction $GPU_MEMORY_FRACTION"
    echo "启用GPU加速，内存使用上限：${GPU_MEMORY_FRACTION}（${GPU_MEMORY_FRACTION}*100%）"
else
    echo "仅使用CPU计算"
fi

# 构建实验类型参数
TYPE_ARGS=""
if [ "$EXPERIMENT_TYPE" = "fs" ]; then
    TYPE_ARGS="--focus_on_fs"
    echo "专注于直接特征选择实验"
    
    # 添加skip_pca参数
    if [ "$SKIP_PCA" = true ]; then
        TYPE_ARGS="$TYPE_ARGS --skip_pca"
        echo "特征选择时将完全跳过PCA，直接在原始特征上进行选择"
    fi
else
    echo "专注于PCA实验"
fi

# 确保Python模块路径正确
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# 打印实验设置
echo "实验设置：" 
echo " - 实验类型: ${EXPERIMENT_TYPE}"
echo " - 最大实验数量: $MAX_EXPERIMENTS"
echo " - 每个标签最大样本数: $MAX_SAMPLES_PER_LABEL"
echo " - 特征选择模式: global"
echo " - 特征选择时跳过PCA: $SKIP_PCA"
echo " - 结果保存目录: $EXPERIMENT_DIR"
echo " - PYTHONPATH: $PYTHONPATH"

# 创建结果目录
mkdir -p "$EXPERIMENT_DIR"

# 运行完整实验
echo "启动完整实验，日志将保存到 $LOG_FILE"
nohup python -m src.main \
    --train_dir "$TRAIN_DIR" \
    --test_dir "$TEST_DIR" \
    --val_dir "$VAL_DIR" \
    --exp_dir "$EXPERIMENT_DIR" \
    --fs_mode "global" \
    --max_experiments $MAX_EXPERIMENTS \
    --max_samples_per_label $MAX_SAMPLES_PER_LABEL \
    $GPU_ARGS \
    $TYPE_ARGS \
    > "$LOG_FILE" 2>&1 &

# 记录进程ID
PID=$!
echo "实验进程启动，PID: $PID"
echo "查看实验日志: tail -f $LOG_FILE"