#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0


# 指定新的模型路径
MODEL_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/Fullyconnected/best_model_results/BrainVoxel_BestParams_20250513_141710_epoch_9_acc_0.7071_f1_0.6970.pth"

# 从模型路径中提取实验名称前缀
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_BASENAME=$(basename "$MODEL_PATH")

# 假设格式是: 实验名_epoch_X_acc_Y_f1_Z.pth
# 先检查文件名是否包含"_epoch_"
if [[ "$MODEL_BASENAME" == *"_epoch_"* ]]; then
    # 包含，则提取_epoch_之前的部分
    EXPERIMENT_PREFIX=$(echo "$MODEL_BASENAME" | sed 's/\(.*\)_epoch_.*/\1/')
else
    # 不包含，则去掉扩展名作为前缀
    EXPERIMENT_PREFIX=$(echo "$MODEL_BASENAME" | sed 's/\..*//')
fi

echo "提取的实验名称前缀: $EXPERIMENT_PREFIX"

# 查找对应的scaler文件
SCALER_PATH="${MODEL_DIR}/${EXPERIMENT_PREFIX}_scaler.pkl"

# 检查scaler文件是否存在
if [ -f "$SCALER_PATH" ]; then
    echo "找到对应的scaler文件: $SCALER_PATH"
    SCALER_ARG="--scaler_path $SCALER_PATH"
else
    echo "警告: 未找到对应scaler文件 ${SCALER_PATH}"
    echo "尝试在目录中查找任何scaler文件..."
    
    # 尝试在目录中查找任何*_scaler.pkl文件
    FOUND_SCALER=$(find "$MODEL_DIR" -name "*_scaler.pkl" | head -n 1)
    
    if [ -n "$FOUND_SCALER" ]; then
        echo "找到可替代的scaler文件: $FOUND_SCALER"
        SCALER_ARG="--scaler_path $FOUND_SCALER"
    else
        echo "未找到任何scaler文件，将尝试从模型中提取标准化参数"
        SCALER_ARG=""
    fi
fi

# 检查原始数据文件是否存在
DATA_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 找不到数据文件 $DATA_PATH"
    echo "请确认数据路径是否正确"
    exit 1
fi

# 创建结果目录（确保不存在）
RESULT_DIR="./prediction_results_$(basename "$MODEL_PATH" .pth)"
if [ -d "$RESULT_DIR" ]; then
    # 添加时间戳避免覆盖
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_DIR="${RESULT_DIR}_${TIMESTAMP}"
    echo "目录已存在，使用新目录: $RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"

# 记录系统信息
echo "记录系统信息..."
{
    echo "系统信息:"
    echo "--------"
    echo "日期时间: $(date)"
    echo "主机名: $(hostname)"
    echo "Python版本: $(python --version 2>&1)"
    echo "CUDA可见设备: $CUDA_VISIBLE_DEVICES"
    echo ""
    echo "使用的模型: $MODEL_PATH"
    echo "使用的scaler: ${SCALER_PATH:-'从模型中提取'}"
    echo "数据文件: $DATA_PATH"
} > "$RESULT_DIR/system_info.txt"

# 添加PyTorch安全全局变量 - 修复语法错误，使用多行语法
cat > "$RESULT_DIR/prepare.py" << 'EOF'
import torch
import numpy as np

try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    print('添加安全全局变量成功')
except Exception as e:
    print(f'添加安全全局变量失败 (可忽略): {e}')
EOF

python "$RESULT_DIR/prepare.py"

# 预测使用训练时的标准化参数
echo "开始预测..."
python predict.py \
  --model "$MODEL_PATH" \
  $SCALER_ARG \
  --data_path "$DATA_PATH" \
  --features_key multidim_data \
  --region_key region \
  --output_dir "$RESULT_DIR" \
  --save_3d 2>&1 | tee "$RESULT_DIR/prediction_log.txt"

# 检查预测是否成功
if [ $? -eq 0 ]; then
    echo "预测成功完成! 结果已保存到: $RESULT_DIR"
    
    # 保存使用的命令和参数到结果目录，便于追踪
    {
        echo "#!/bin/bash"
        echo "# 预测命令记录 - $(date)"
        echo "python predict.py \\"
        echo "  --model \"$MODEL_PATH\" \\"
        echo "  $SCALER_ARG \\"
        echo "  --data_path \"$DATA_PATH\" \\"
        echo "  --features_key multidim_data \\"
        echo "  --region_key region \\"
        echo "  --output_dir \"$RESULT_DIR\" \\"
        echo "  --save_3d"
    } > "$RESULT_DIR/prediction_command.sh"
    chmod +x "$RESULT_DIR/prediction_command.sh"
    
    echo "使用的命令已保存到: $RESULT_DIR/prediction_command.sh"
    
    # 复制脚本自身到结果目录
    cp "$0" "$RESULT_DIR/$(basename "$0")"
    
    # 创建简单的结果摘要
    if [ -f "$RESULT_DIR/prediction_stats.txt" ]; then
        echo "结果摘要:" > "$RESULT_DIR/summary.txt"
        echo "--------" >> "$RESULT_DIR/summary.txt"
        echo "" >> "$RESULT_DIR/summary.txt"
        grep -A 10 "预测类别分布" "$RESULT_DIR/prediction_stats.txt" >> "$RESULT_DIR/summary.txt" 2>/dev/null
    fi
else
    echo "预测过程中出现错误，请检查日志: $RESULT_DIR/prediction_log.txt"
    exit 1
fi