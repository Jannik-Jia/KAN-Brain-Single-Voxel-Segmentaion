#!/bin/bash

# 设置实验类型：pca 或 fs (特征选择)
EXPERIMENT_TYPE="pca"  # 可选："pca" 或 "fs"

export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# 设置日志文件
LOG_FILE="baseline_${EXPERIMENT_TYPE}_experiment_$(date +%Y%m%d_%H%M%S).log"

# 设置数据目录 - 请根据实际路径修改
TRAIN_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train"
TEST_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test"
VAL_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"

# 设置结果保存目录
EXPERIMENT_DIR="pseudoinverse_${EXPERIMENT_TYPE}_baseline_experiments"

# GPU相关设置
USE_GPU=true                # 是否使用GPU加速（true/false）
GPU_MEMORY_FRACTION=0.8     # GPU内存使用比例上限（0.0-1.0）

# 添加数据处理参数
MAX_SAMPLES_PER_LABEL=500   # 每个标签最多使用的样本数

# PCA设置 (仅在PCA模式下使用)
APPLY_PCA=true              # 应用PCA降维
AUTO_VARIANCE=0.95          # 自动选择解释95%方差的PCA组件数量

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
    APPLY_PCA=false
    echo "运行特征选择基线实验"
else
    echo "运行PCA基线实验"
fi

# 打印实验设置
echo "实验设置：" 
echo " - 实验类型: ${EXPERIMENT_TYPE}"
echo " - 运行基准测试模式"
echo " - 每个标签最大样本数: $MAX_SAMPLES_PER_LABEL"
if [ "$EXPERIMENT_TYPE" = "pca" ]; then
    echo " - 应用PCA: $APPLY_PCA" 
    echo " - 自动PCA方差阈值: $AUTO_VARIANCE"
fi
echo " - 特征选择模式: global" 
echo " - 结果保存目录: $EXPERIMENT_DIR"

# 运行基线实验
echo "启动基线实验，日志将保存到 $LOG_FILE"
nohup python -m src.main \
    --train_dir "$TRAIN_DIR" \
    --test_dir "$TEST_DIR" \
    --val_dir "$VAL_DIR" \
    --exp_dir "$EXPERIMENT_DIR" \
    --fs_mode "global" \
    --run_baseline \
    --max_samples_per_label $MAX_SAMPLES_PER_LABEL \
    --apply_pca $APPLY_PCA \
    --auto_pca_variance $AUTO_VARIANCE \
    $GPU_ARGS \
    $TYPE_ARGS \
    > "$LOG_FILE" 2>&1 &

# 记录进程ID
PID=$!
echo "实验进程启动，PID: $PID"
echo "查看实验日志: tail -f $LOG_FILE"