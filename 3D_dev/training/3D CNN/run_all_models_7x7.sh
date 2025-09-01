#!/bin/bash

# 批量运行不同参数量的模型进行对比实验 - 7×7 patch版本
# 按照从大到小的顺序训练，避免显存碎片问题

# 数据目录设置
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据（实际是2D patch）
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"  # 1D数据

# 使用3D_validated数据（2D patch）
DATA_DIR=$DATA_DIR_3D

OUTPUT_DIR="./results_comparison_7x7"
PATCH_SIZE=7  # 使用7×7 patch
TEST_SUBJECT=1
EPOCHS=50

# 创建输出目录和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/logs

# 设置日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
MAIN_LOG="$LOG_DIR/training_7x7_${TIMESTAMP}.log"
SUMMARY_LOG="$LOG_DIR/summary_7x7_${TIMESTAMP}.log"

# 创建日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_summary() {
    echo "$1" | tee -a "$SUMMARY_LOG"
}

log_message "===== 开始训练实验 (7×7 Patch) ====="
log_message "数据目录: $DATA_DIR"
log_message "输出目录: $OUTPUT_DIR"
log_message "Patch大小: ${PATCH_SIZE}×${PATCH_SIZE}"
log_message "测试被试: Subject $TEST_SUBJECT"
log_message "训练轮数: $EPOCHS"
log_message "日志文件: $MAIN_LOG"
log_message ""

# 记录系统信息
log_summary "===== 系统信息 ====="
log_summary "主机名: $(hostname)"
log_summary "开始时间: $(date)"
log_summary "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | tee -a "$SUMMARY_LOG"
log_summary ""
log_summary "===== 训练配置 ====="
log_summary "Patch大小: ${PATCH_SIZE}×${PATCH_SIZE}"
log_summary "测试被试: Subject $TEST_SUBJECT"
log_summary "训练轮数: $EPOCHS"
log_summary ""

# 1. 先运行69M模型（最大，需要最多显存）
log_message "===== [1/4] 运行69M参数模型 (7×7) ====="
log_message "预计参数量: ~69,000,000"
log_summary "[1/4] 69M模型 - 开始时间: $(date)"

MODEL_LOG="$LOG_DIR/model_69M_${TIMESTAMP}.log"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/69M \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 64 \
    --epochs $EPOCHS \
    --model_size 69M \
    --samples_per_subject 10000 2>&1 | tee "$MODEL_LOG"

# 提取关键结果
FINAL_F1=$(grep "最佳测试 F1:" "$MODEL_LOG" | tail -1 | sed 's/.*F1: \([0-9.]*\).*/\1/')
log_summary "[1/4] 69M模型 - 完成时间: $(date) - 最佳F1: $FINAL_F1"

log_message ""
log_message "69M模型训练完成，等待10秒释放显存..."
sleep 10

# 2. 运行52M模型
log_message "===== [2/4] 运行52M参数模型 (7×7) ====="
log_message "预计参数量: ~52,000,000"
log_summary "[2/4] 52M模型 - 开始时间: $(date)"

MODEL_LOG="$LOG_DIR/model_52M_${TIMESTAMP}.log"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/52M \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 128 \
    --epochs $EPOCHS \
    --model_size 52M \
    --samples_per_subject 10000 2>&1 | tee "$MODEL_LOG"

# 提取关键结果
FINAL_F1=$(grep "最佳测试 F1:" "$MODEL_LOG" | tail -1 | sed 's/.*F1: \([0-9.]*\).*/\1/')
log_summary "[2/4] 52M模型 - 完成时间: $(date) - 最佳F1: $FINAL_F1"

log_message ""
log_message "52M模型训练完成，等待10秒释放显存..."
sleep 10

# 3. 运行35M模型
log_message "===== [3/4] 运行35M参数模型 (7×7) ====="
log_message "预计参数量: ~35,000,000"
log_summary "[3/4] 35M模型 - 开始时间: $(date)"

MODEL_LOG="$LOG_DIR/model_35M_${TIMESTAMP}.log"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/35M \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 128 \
    --epochs $EPOCHS \
    --model_size 35M \
    --samples_per_subject 10000 2>&1 | tee "$MODEL_LOG"

# 提取关键结果
FINAL_F1=$(grep "最佳测试 F1:" "$MODEL_LOG" | tail -1 | sed 's/.*F1: \([0-9.]*\).*/\1/')
log_summary "[3/4] 35M模型 - 完成时间: $(date) - 最佳F1: $FINAL_F1"

log_message ""
log_message "35M模型训练完成，等待10秒释放显存..."
sleep 10

# 4. 最后运行基线模型（最小，~58K）
log_message "===== [4/4] 运行基线模型 (7×7) ====="
log_message "预计参数量: ~58,000"
log_summary "[4/4] Baseline模型 - 开始时间: $(date)"

MODEL_LOG="$LOG_DIR/model_baseline_${TIMESTAMP}.log"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/baseline \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 256 \
    --epochs $EPOCHS \
    --model_size baseline \
    --samples_per_subject 10000 2>&1 | tee "$MODEL_LOG"

# 提取关键结果
FINAL_F1=$(grep "最佳测试 F1:" "$MODEL_LOG" | tail -1 | sed 's/.*F1: \([0-9.]*\).*/\1/')
log_summary "[4/4] Baseline模型 - 完成时间: $(date) - 最佳F1: $FINAL_F1"

log_message ""
log_message "===== 所有7×7模型训练完成 ====="
log_message "结果保存在: $OUTPUT_DIR"
log_message "日志保存在: $LOG_DIR"
log_message ""
log_message "模型参数量总结 (7×7 patch):"
log_message "- Baseline: ~58K"
log_message "- 35M: ~35,000,000"
log_message "- 52M: ~52,000,000"
log_message "- 69M: ~69,000,000"

# 写入最终总结
log_summary ""
log_summary "===== 训练完成总结 ====="
log_summary "完成时间: $(date)"
log_summary "所有日志文件保存在: $LOG_DIR"
log_summary "主日志: $MAIN_LOG"
log_summary "总结日志: $SUMMARY_LOG"

echo ""
echo "提示：查看详细日志请运行:"
echo "  主日志: cat $MAIN_LOG"
echo "  总结: cat $SUMMARY_LOG"
echo "  实时监控: tail -f $MAIN_LOG"