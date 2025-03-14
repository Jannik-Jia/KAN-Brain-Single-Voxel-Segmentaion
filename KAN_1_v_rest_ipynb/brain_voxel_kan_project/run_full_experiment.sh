#!/bin/bash
# run_full_experiment.sh

# 设置参数
LABEL_ID=1
EPOCHS=200
BATCH_SIZE=640
DEVICE=0  # GPU ID

# 创建必要的目录
mkdir -p Results/BrainVoxel_1DKAN/BrainVoxel

# 步骤1：训练模型
echo "开始训练模型，总共${EPOCHS}个epoch..."
python main.py --label_id $LABEL_ID --epochs $EPOCHS --batch_size $BATCH_SIZE --device $DEVICE

# 步骤2：评估所有epoch
echo "训练完成，开始评估所有epoch..."
python evaluate_epochs.py --label_id $LABEL_ID --start_epoch 1 --end_epoch $EPOCHS --batch_size $BATCH_SIZE --device $DEVICE --save_results --plot_results

echo "实验完成!"

# chmod +x run_full_experiment.sh
# ./run_full_experiment.sh