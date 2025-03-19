#!/bin/bash
# 脑体素分层分类项目运行脚本 - GPU加速版本

# 设置日期时间作为运行标识
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PREFIX="BrainVoxel_GPU_${TIMESTAMP}"
LOG_FILE="output/logs/${OUTPUT_PREFIX}.log"

# 确保输出目录存在
mkdir -p output/logs
mkdir -p output/figures
mkdir -p output/results
mkdir -p output/models

# 打印启动信息
echo "============================================================"
echo "脑体素分层分类项目 (GPU加速版本) - 运行开始: $(date)"
echo "输出前缀: ${OUTPUT_PREFIX}"
echo "日志文件: ${LOG_FILE}"
echo "============================================================"

# 检查GPU可用性
if nvidia-smi &> /dev/null; then
    echo "GPU已检测到，将使用GPU加速计算" | tee -a ${LOG_FILE}
    nvidia-smi | tee -a ${LOG_FILE}
else
    echo "警告: 未检测到GPU，将使用CPU计算" | tee -a ${LOG_FILE}
fi


# 后台运行主程序并将输出重定向到日志文件
nohup python main.py \
  --data_subset=val \
  --normalize=robust \
  --pca \
  --feature_selection \
  --min_clusters=2 \
  --max_clusters=10 \
  --output_prefix=${OUTPUT_PREFIX} \
  > ${LOG_FILE} 2>&1 &

# 保存进程ID
PID=$!
echo "进程已启动，PID: $PID"
echo "进程ID已保存到: ${OUTPUT_PREFIX}.pid"
echo $PID > ${OUTPUT_PREFIX}.pid

# 打印监控指令
echo ""
echo "监控日志: tail -f ${LOG_FILE}"
echo "监控GPU使用: nvidia-smi -l 5"  # 每5秒更新一次GPU使用情况
echo "检查进程: ps -p $PID"
echo "终止进程: kill $PID"
echo ""
echo "脚本已在后台运行。将显示最新的日志输出:"
echo "============================================================"
sleep 2
tail -f ${LOG_FILE}