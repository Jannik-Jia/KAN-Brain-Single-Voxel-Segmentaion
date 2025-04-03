# scripts/run_full_experiments.sh
#!/bin/bash

# 设置日志文件
LOG_FILE="full_experiments_$(date +%Y%m%d_%H%M%S).log"

# 设置数据目录 - 请根据实际路径修改
TRAIN_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train"
TEST_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test"
VAL_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"

# 设置结果保存目录
EXPERIMENT_DIR="pseudoinverse_with_fs_experiments"

# 设置最大实验数量
MAX_EXPERIMENTS=50

# 运行完整实验
echo "启动完整实验，日志将保存到 $LOG_FILE"
nohup python -m src.main \
    --train_dir "$TRAIN_DIR" \
    --test_dir "$TEST_DIR" \
    --val_dir "$VAL_DIR" \
    --exp_dir "$EXPERIMENT_DIR" \
    --fs_mode "global" \
    --max_experiments $MAX_EXPERIMENTS \
    > "$LOG_FILE" 2>&1 &

# 记录进程ID
PID=$!
echo "实验进程启动，PID: $PID"
echo "查看实验日志: tail -f $LOG_FILE"