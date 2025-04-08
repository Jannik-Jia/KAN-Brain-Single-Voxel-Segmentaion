#!/bin/bash

export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# 设置日志文件
LOG_FILE="baseline_pca_experiment_$(date +%Y%m%d_%H%M%S).log"

# 设置数据目录 - 请根据实际路径修改
TRAIN_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train"
TEST_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test"
VAL_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"

# 设置结果保存目录
EXPERIMENT_DIR="pseudoinverse_pca_baseline_experiments"

# GPU相关设置
USE_GPU=true                # 是否使用GPU加速（true/false）
GPU_MEMORY_FRACTION=0.8     # GPU内存使用比例上限（0.0-1.0）

# 添加数据处理参数
MAX_SAMPLES_PER_LABEL=500   # 每个标签最多使用的样本数
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

# 打印实验设置
echo "实验设置：" 
echo " - 运行基准测试模式 (自动PCA组件数量)"
echo " - 每个标签最大样本数: $MAX_SAMPLES_PER_LABEL"
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
    > "$LOG_FILE" 2>&1 &

# 记录进程ID
PID=$!
echo "实验进程启动，PID: $PID"
echo "查看实验日志: tail -f $LOG_FILE"