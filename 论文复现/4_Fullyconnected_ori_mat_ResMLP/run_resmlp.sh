#!/bin/bash
# Residual MLP 4x4096 训练脚本
# 测试被试: YHC_10_lncguay

# 默认参数
EPOCHS=30
BATCH_SIZE=8192
LR=1e-5
DEVICE=0

# 运行训练
python main_resmlp.py \
    --test_subject "YHC_10_lncguay" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --save_dir "./results"

echo "Residual MLP 训练完成！"
