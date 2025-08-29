#!/bin/bash

# 批量运行不同参数量的模型进行对比实验
# 按照从大到小的顺序训练，避免显存碎片问题

# 数据目录设置
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据（实际是2D patch）
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"  # 1D数据

# 使用3D_validated数据（2D patch）
DATA_DIR=$DATA_DIR_3D

OUTPUT_DIR="./results_comparison"
PATCH_SIZE=3
TEST_SUBJECT=1
EPOCHS=50

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "===== 开始训练实验 ====="
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Patch大小: ${PATCH_SIZE}×${PATCH_SIZE}"
echo "测试被试: Subject $TEST_SUBJECT"
echo "训练轮数: $EPOCHS"
echo ""

# 1. 先运行69M模型（最大，需要最多显存）
echo "===== [1/4] 运行69M参数模型 ====="
echo "预计参数量: ~69,000,000"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/69M \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 128 \
    --epochs $EPOCHS \
    --model_size 69M \
    --samples_per_subject 10000

echo ""
echo "69M模型训练完成，等待10秒释放显存..."
sleep 10

# 2. 运行52M模型
echo "===== [2/4] 运行52M参数模型 ====="
echo "预计参数量: ~52,000,000"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/52M \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 256 \
    --epochs $EPOCHS \
    --model_size 52M \
    --samples_per_subject 10000

echo ""
echo "52M模型训练完成，等待10秒释放显存..."
sleep 10

# 3. 运行35M模型
echo "===== [3/4] 运行35M参数模型 ====="
echo "预计参数量: ~35,000,000"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/35M \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 256 \
    --epochs $EPOCHS \
    --model_size 35M \
    --samples_per_subject 10000

echo ""
echo "35M模型训练完成，等待10秒释放显存..."
sleep 10

# 4. 最后运行基线模型（最小，~58K）
echo "===== [4/4] 运行基线模型 ====="
echo "预计参数量: ~58,000"
python train_baseline_3x3_7x7.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/baseline \
    --patch_size $PATCH_SIZE \
    --test_subject $TEST_SUBJECT \
    --batch_size 512 \
    --epochs $EPOCHS \
    --model_size baseline \
    --samples_per_subject 10000

echo ""
echo "===== 所有模型训练完成 ====="
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "模型参数量总结:"
echo "- Baseline: ~58K"
echo "- 35M: ~35,000,000"
echo "- 52M: ~52,000,000"
echo "- 69M: ~69,000,000"