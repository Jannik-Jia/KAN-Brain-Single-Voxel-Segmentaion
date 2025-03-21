#!/bin/bash
# 脑体素分层分类项目运行脚本 - GPU加速版本 - 带错误监控

# 设置日期时间作为运行标识
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PREFIX="BrainVoxel_GPU_${TIMESTAMP}"
LOG_FILE="output/logs/${OUTPUT_PREFIX}.log"
STATUS_FILE="output/logs/${OUTPUT_PREFIX}_status.txt"
ERROR_LOG="output/logs/${OUTPUT_PREFIX}_error.log"

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
echo "状态文件: ${STATUS_FILE}"
echo "============================================================"

# 记录初始状态
echo "STATUS:RUNNING" > ${STATUS_FILE}
echo "START_TIME:$(date)" >> ${STATUS_FILE}

# 检查GPU可用性
if nvidia-smi &> /dev/null; then
    echo "GPU已检测到，将使用GPU加速计算" | tee -a ${LOG_FILE}
    nvidia-smi | tee -a ${LOG_FILE}
else
    echo "警告: 未检测到GPU，将使用CPU计算" | tee -a ${LOG_FILE}
fi

# 设置监控函数
monitor_process() {
    local pid=$1
    local log_file=$2
    local status_file=$3
    local error_log=$4
    
    # 检查进程是否存在
    if ! ps -p $pid > /dev/null; then
        echo "错误: 进程 $pid 未启动或已退出" | tee -a ${error_log}
        echo "STATUS:FAILED" > ${status_file}
        echo "END_TIME:$(date)" >> ${status_file}
        echo "ERROR:进程启动失败" >> ${status_file}
        return 1
    fi
    
    # 启动监控循环
    while ps -p $pid > /dev/null; do
        # 检查GPU内存使用
        if nvidia-smi &> /dev/null; then
            gpu_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep $pid | awk '{print $2}')
            if [ ! -z "$gpu_mem" ]; then
                echo "$(date): GPU内存使用: $gpu_mem" >> ${status_file}
                # 记录到状态文件但不打印到控制台，避免过多输出
            fi
            
            # 检查GPU错误
            nvidia_errors=$(nvidia-smi | grep -i "error")
            if [ ! -z "$nvidia_errors" ]; then
                echo "检测到GPU错误:" | tee -a ${error_log}
                echo "$nvidia_errors" | tee -a ${error_log}
            fi
        fi
        
        # 检查系统内存使用
        mem_percent=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
        if (( $(echo "$mem_percent > 95" | bc -l) )); then
            echo "警告: 内存使用率超过95%: ${mem_percent}%" | tee -a ${error_log}
            echo "MEMORY_WARNING:${mem_percent}%" >> ${status_file}
        fi
        
        # 检查日志文件中的错误
        recent_errors=$(tail -100 ${log_file} | grep -i "error\|exception\|traceback")
        if [ ! -z "$recent_errors" ]; then
            echo "检测到日志中的错误:" | tee -a ${error_log}
            echo "$recent_errors" | tee -a ${error_log}
            # 避免重复报告同一错误
            sleep 60
        fi
        
        # 每30秒检查一次
        sleep 30
    done
    
    # 进程已结束
    if grep -q "分析完成" ${log_file}; then
        echo "进程正常完成" | tee -a ${status_file}
        echo "STATUS:COMPLETED" > ${status_file}
    else
        echo "进程异常退出" | tee -a ${error_log}
        echo "STATUS:FAILED" > ${status_file}
        # 提取最后100行日志作为错误上下文
        echo "最后日志内容:" | tee -a ${error_log}
        tail -100 ${log_file} | tee -a ${error_log}
        
        # 发送桌面通知（如果支持）
        if command -v notify-send &> /dev/null; then
            notify-send "分析任务异常退出" "脑体素分层分类任务已异常退出，请检查日志文件"
        fi
    fi
    
    echo "END_TIME:$(date)" >> ${status_file}
}

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

# 启动监控进程
monitor_process $PID ${LOG_FILE} ${STATUS_FILE} ${ERROR_LOG} &
MONITOR_PID=$!
echo "监控进程已启动，PID: $MONITOR_PID"

# 设置捕获CTRL+C信号
trap 'echo "接收到中断信号，正在终止进程..."; kill $PID 2>/dev/null; echo "STATUS:INTERRUPTED" > ${STATUS_FILE}; echo "END_TIME:$(date)" >> ${STATUS_FILE}; exit 1' INT

# 打印监控指令
echo ""
echo "监控日志: tail -f ${LOG_FILE}"
echo "监控状态: cat ${STATUS_FILE}"
echo "监控错误: tail -f ${ERROR_LOG}"
echo "监控GPU使用: nvidia-smi -l 5"  # 每5秒更新一次GPU使用情况
echo "检查进程: ps -p $PID"
echo "终止进程: kill $PID"
echo ""
echo "脚本已在后台运行。将显示最新的日志输出:"
echo "============================================================"
sleep 2
tail -f ${LOG_FILE}