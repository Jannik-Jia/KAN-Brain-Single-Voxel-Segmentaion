#!/bin/bash

# MRI数据特性分析启动脚本
# 使用nohup在后台运行分析任务

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU，根据需要修改

# 创建输出和日志目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="analysis_results_$TIMESTAMP"
LOG_DIR="logs"
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 获取数据目录参数
if [ "$#" -lt 1 ]; then
    echo "用法: $0 <数据目录> [其他参数]"
    echo "例如: $0 /path/to/mri_data --gpu --subset val"
    exit 1
fi

DATA_DIR=$1
shift  # 移除第一个参数，保留其余参数

# 定义日志文件
LOG_FILE="$LOG_DIR/analysis_$TIMESTAMP.log"

echo "开始MRI数据分析任务..."
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "附加参数: $@"

# 启动分析程序
nohup python main.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR $@ > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
echo "分析任务已在后台启动，进程ID: $PID"
echo "可以使用以下命令查看日志: tail -f $LOG_FILE"
echo "$PID" > "$LOG_DIR/pid_$TIMESTAMP.txt"

# 显示如何杀死进程的说明
echo "要停止分析任务，请使用: kill $PID"