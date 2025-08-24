#!/bin/bash
# Leave-one-out 训练脚本
# 分别训练3×3和7×7模型，每个被试作为测试集运行一次

# 设置参数
DATA_DIR="/path/to/your/3d/mat/files"  # 请修改为实际的3D MAT文件目录
OUTPUT_DIR="./results_leave_one_out"
EPOCHS=50
BATCH_SIZE=256
SAMPLES_PER_SUBJECT=10000

# 创建输出目录
mkdir -p ${OUTPUT_DIR}/3x3
mkdir -p ${OUTPUT_DIR}/7x7

echo "开始 Leave-one-out 训练..."
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "每个被试采样数: ${SAMPLES_PER_SUBJECT}"

# 3×3 模型训练
echo ""
echo "========== 训练 3×3 模型 =========="
for test_subject in {1..38}
do
    echo ""
    echo ">>> 测试被试 ${test_subject}/38 (3×3 patch)"
    
    python train_baseline_3x3_7x7.py \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/3x3 \
        --patch_size 3 \
        --test_subject ${test_subject} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --samples_per_subject ${SAMPLES_PER_SUBJECT} \
        --use_se \
        --lr 1e-4
    
    # 可选：只运行前几个被试进行测试
    # if [ ${test_subject} -eq 3 ]; then
    #     break
    # fi
done

# 7×7 模型训练
echo ""
echo "========== 训练 7×7 模型 =========="
for test_subject in {1..38}
do
    echo ""
    echo ">>> 测试被试 ${test_subject}/38 (7×7 patch)"
    
    python train_baseline_3x3_7x7.py \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/7x7 \
        --patch_size 7 \
        --test_subject ${test_subject} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --samples_per_subject ${SAMPLES_PER_SUBJECT} \
        --use_se \
        --lr 1e-4
    
    # 可选：只运行前几个被试进行测试
    # if [ ${test_subject} -eq 3 ]; then
    #     break
    # fi
done

echo ""
echo "所有训练完成！"
echo "结果保存在: ${OUTPUT_DIR}"