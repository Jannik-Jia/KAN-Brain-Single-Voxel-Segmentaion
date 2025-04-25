#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置默认参数
CONFIG_PATH="configs/mfcan_config.json"
DATA_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/MFCAN/src/data/processed/reorganized_encoder_data.h5"
MODEL_PATH="results/mfcan/20250424_104759/encoder_pretrain_best_epoch_30.pth"  # 预训练模型路径
OUTPUT_DIR="results/hyperopt/"  # 默认使用配置文件中的设置
HYPEROPT_DIR=""  # 超参数优化结果目录

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --hyperopt_dir)
      HYPEROPT_DIR="$2"
      shift 2
      ;;
    *)
      echo "未知选项: $1"
      echo "可用选项: --config, --data_path, --model_path, --output_dir, --hyperopt_dir"
      exit 1
      ;;
  esac
done

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/shell/continue_training_${TIMESTAMP}.log"

# 如果没有指定超参数优化目录，则通过OUTPUT_DIR推断
if [ -z "$HYPEROPT_DIR" ]; then
    if [ -z "$OUTPUT_DIR" ]; then
        # 尝试查找最近的超参数优化结果
        LATEST_HYPEROPT_DIR=$(find results/hyperopt -maxdepth 1 -type d -not -name "hyperopt" | sort -r | head -n 1)
        if [ -n "$LATEST_HYPEROPT_DIR" ]; then
            HYPEROPT_DIR="${LATEST_HYPEROPT_DIR}"
            echo "自动检测到最近的超参数优化目录: ${HYPEROPT_DIR}"
        else
            echo "未找到超参数优化目录，将使用原始配置"
        fi
    else
        # 如果指定了OUTPUT_DIR，检查其中是否有hyperopt子目录
        if [ -d "${OUTPUT_DIR}/hyperopt" ]; then
            HYPEROPT_DIR="${OUTPUT_DIR}/hyperopt"
            echo "使用指定输出目录下的超参数优化结果: ${HYPEROPT_DIR}"
        fi
    fi
fi

# 查找最佳配置
BEST_CONFIG=""
if [ -n "$HYPEROPT_DIR" ]; then
    if [ -f "${HYPEROPT_DIR}/best_config.json" ]; then
        BEST_CONFIG="${HYPEROPT_DIR}/best_config.json"
        echo "使用超参数优化的最佳配置: ${BEST_CONFIG}"
        CONFIG_PATH="${BEST_CONFIG}"
    else
        echo "警告: 在 ${HYPEROPT_DIR} 中未找到best_config.json，将使用原始配置"
    fi
fi

echo "开始继续MFCAN训练..."
echo "配置文件: $CONFIG_PATH"
echo "数据路径: $DATA_PATH"
echo "预训练模型: $MODEL_PATH"
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

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件 $MODEL_PATH 不存在"
    exit 1
fi

# 设置输出目录（如果未指定）
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="results/mfcan/${TIMESTAMP}_continue"
    echo "自动设置输出目录: $OUTPUT_DIR"
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "运行继续训练脚本..."

# 运行命令并重定向输出到日志文件
nohup python scripts/continue_training.py \
    --config ${CONFIG_PATH} \
    --data_path ${DATA_PATH} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} > "${LOG_FILE}" 2>&1 &

# 获取进程ID
PID=$!
echo "进程已启动，PID: $PID"
echo "查看进度使用: tail -f $LOG_FILE"
echo "检查进程是否运行使用: ps -p $PID"