#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置固定参数 - 可以直接在这里修改
MODALITY="all"  # 可选: diffusion, qti, cest, all
CONFIG_PATH="configs/encoders_config.json"

# ===== 自定义H5文件路径 - 在这里修改 =====
# 直接指定您想使用的H5文件的完整路径
DATA_PATH="results/feature_engineering/combined_20240422_154030/combined_selected_features.h5"

# 设置日志文件路径
LOG_FILE="logs/shell/stage3_encoder_pretraining_${MODALITY}_$(date +%Y%m%d_%H%M%S).log"

echo "开始阶段3: 编码器预训练..."
echo "模态: $MODALITY"
echo "配置文件: $CONFIG_PATH"
echo "数据路径: $DATA_PATH"
echo "日志文件: $LOG_FILE"

# 检查文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件 $CONFIG_PATH 不存在"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件 $DATA_PATH 不存在"
    exit 1
fi

# 设置输出目录
OUTPUT_DIR="models/encoders/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 运行阶段三编码器预训练脚本
nohup python scripts/stage3_encoder_pretraining.py \
    --config "$CONFIG_PATH" \
    --modality "$MODALITY" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "进程已启动，PID: $PID"
echo "查看进度使用: tail -f $LOG_FILE"
echo "检查进程是否运行使用: ps -p $PID"