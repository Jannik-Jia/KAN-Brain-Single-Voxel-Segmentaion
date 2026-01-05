#!/bin/bash
# 1D训练脚本（使用3D数据集）
# 复现Alex的架构，使用341维输入
# 【Per-Patient Z-Score 版本】每个患者独立进行逐通道z-score标准化

# 设置参数
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"  # 1D数据集（去除_3d后缀）
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据集（用于mask）
OUTPUT_DIR="./results_1d_perpatient_zscore"  # 更新目录名以区分标准化方式

# Alex的超参数
EPOCHS=25
BATCH_SIZE=8192

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "【Per-Patient Z-Score 版本】1D训练"
echo "=============================================="
echo ""
echo "标准化方式: Per-Patient Z-Score（每个患者独立逐通道标准化）"
echo "1D数据目录: ${DATA_DIR_1D}"
echo "3D数据目录: ${DATA_DIR_3D}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "训练参数:"
echo "  - Epochs: ${EPOCHS} (Alex的设置)"
echo "  - Batch Size: ${BATCH_SIZE} (Alex的设置)"
echo "  - Learning Rate: 0.00001 (Alex的设置)"
echo "  - Normalisation: Per-Patient Z-Score"
echo ""

# 测试单个被试（被试38作为测试集，复现Alex的设置）
echo ">>> 训练模型（测试被试38）"
python train_1d_with_3d_dataset.py \
    --data_dir_1d ${DATA_DIR_1D} \
    --data_dir_3d ${DATA_DIR_3D} \
    --output_dir ${OUTPUT_DIR} \
    --test_subject 38 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --save_predictions

echo ""
echo "训练完成！"
echo ""

# 如果训练成功，进行可视化
if [ -f "${OUTPUT_DIR}/predictions_3d_test38.mat" ]; then
    echo "生成可视化..."

    # 获取测试文件路径（假设按字母顺序第38个）
    TEST_FILE=$(ls ${DATA_DIR_3D}/*.mat | sort | sed -n '38p')

    python visualize_1d_3d_predictions.py \
        --pred_file ${OUTPUT_DIR}/predictions_3d_test38.mat \
        --gt_file ${TEST_FILE} \
        --slice_idx 128 \
        --output_dir ${OUTPUT_DIR}/visualizations

    echo "可视化完成！结果保存在: ${OUTPUT_DIR}/visualizations"
else
    echo "未找到预测文件，跳过可视化"
fi

echo ""
echo "=============================================="
echo "完整Leave-One-Out训练（可选）"
echo "=============================================="
echo "如果要运行完整的38次Leave-One-Out训练，请运行："
echo "bash run_1d_leave_one_out.sh"
