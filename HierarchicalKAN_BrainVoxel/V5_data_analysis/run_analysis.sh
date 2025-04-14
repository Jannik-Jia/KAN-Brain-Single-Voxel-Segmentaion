#!/bin/bash

# 脑MRI数据特性分析启动脚本
# 使用固定路径的数据

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，根据需要修改

# 创建输出和日志目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="brain_analysis_results_$TIMESTAMP"
LOG_DIR="logs"
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 定义日志文件
LOG_FILE="$LOG_DIR/brain_analysis_$TIMESTAMP.log"

echo "开始脑MRI数据分析任务..."
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"

# 添加一个简单的配置，用于选择分析类型
# 默认为全部分析
ANALYSIS_TYPE="all"  # 可选值: all, basic, feature, dim_reduction

# 如果提供了参数，则使用参数作为分析类型
if [ "$#" -ge 1 ]; then
    ANALYSIS_TYPE=$1
    echo "分析类型: $ANALYSIS_TYPE"
fi

# 根据分析类型设置参数
EXTRA_ARGS=""
case $ANALYSIS_TYPE in
    "basic")
        EXTRA_ARGS="--skip_feature --skip_dim_reduction"
        ;;
    "feature")
        EXTRA_ARGS="--skip_basic --skip_dim_reduction"
        ;;
    "dim_reduction")
        EXTRA_ARGS="--skip_basic --skip_feature"
        ;;
    *)
        EXTRA_ARGS=""  # 默认全部分析
        ;;
esac

# 启动分析脚本
python main.py \
    --data_dir "dummy" \
    --output_dir "$OUTPUT_DIR" \
    --normalize none \
    --gpu \
    $EXTRA_ARGS > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
echo "分析任务已在后台启动，进程ID: $PID"
echo "可以使用以下命令查看日志: tail -f $LOG_FILE"
echo "$PID" > "$LOG_DIR/pid_$TIMESTAMP.txt"

# 显示如何杀死进程的说明
echo "要停止分析任务，请使用: kill $PID"