#!/bin/bash
# 运行BrainVoxel分析脚本
# 使用nohup在后台运行，即使终端断开也会继续执行

# 设置环境变量（根据需要调整）
# export CUDA_VISIBLE_DEVICES=0

# 创建时间戳目录名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./nohup_output"
mkdir -p $OUTPUT_DIR

# 添加执行权限
chmod +x brainvoxel_analysis.py

# 使用nohup在后台运行脚本，将标准输出和错误输出重定向到文件
nohup python brainvoxel_analysis.py > "${OUTPUT_DIR}/brainvoxel_analysis_${TIMESTAMP}.out" 2>&1 &

# 获取运行进程的PID
PID=$!
echo "分析任务已在后台启动，PID: ${PID}"
echo "输出将保存到: ${OUTPUT_DIR}/brainvoxel_analysis_${TIMESTAMP}.out"
echo "可以使用 'tail -f ${OUTPUT_DIR}/brainvoxel_analysis_${TIMESTAMP}.out' 查看实时输出"
echo "使用 'ps -p ${PID}' 检查进程是否仍在运行"