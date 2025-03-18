#!/bin/bash
# 运行BrainVoxel分层分类分析脚本 (新版本)
# 使用nohup在后台运行，即使终端断开也会继续执行

# 设置环境变量（根据需要调整）
# export CUDA_VISIBLE_DEVICES=0

# 激活 conda 环境（如果需要）
# eval "$(conda shell.bash hook)"
# conda activate your_environment_name

# 创建时间戳目录名作为唯一标识
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="brainvoxel_analysis_${TIMESTAMP}"
OUTPUT_DIR="./nohup_output"
mkdir -p $OUTPUT_DIR

# 添加执行权限
chmod +x main.py

# 记录使用的 Python 和 CUDA 版本
echo "======== 环境信息 ========" > "${OUTPUT_DIR}/${JOB_NAME}.log"
echo "主机名: $(hostname)" >> "${OUTPUT_DIR}/${JOB_NAME}.log"
echo "Python版本: $(python --version 2>&1)" >> "${OUTPUT_DIR}/${JOB_NAME}.log" 
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA信息:" >> "${OUTPUT_DIR}/${JOB_NAME}.log"
    nvidia-smi | head -n 10 >> "${OUTPUT_DIR}/${JOB_NAME}.log"
fi
echo "=========================" >> "${OUTPUT_DIR}/${JOB_NAME}.log"
echo "" >> "${OUTPUT_DIR}/${JOB_NAME}.log"

# 记录当前时间
echo "开始时间: $(date)" >> "${OUTPUT_DIR}/${JOB_NAME}.log"
echo "" >> "${OUTPUT_DIR}/${JOB_NAME}.log"

# 使用nohup在后台运行脚本，将标准输出和错误输出重定向到文件
echo "启动分析任务..."
nohup python main.py > "${OUTPUT_DIR}/${JOB_NAME}.out" 2>&1 &

# 获取运行进程的PID
PID=$!
echo "分析任务已在后台启动:"
echo "  作业名称: ${JOB_NAME}"
echo "  进程ID (PID): ${PID}"
echo "  输出文件: ${OUTPUT_DIR}/${JOB_NAME}.out"
echo "  日志文件: ${OUTPUT_DIR}/${JOB_NAME}.log"
echo ""
echo "使用以下命令查看实时输出:"
echo "  tail -f ${OUTPUT_DIR}/${JOB_NAME}.out"
echo ""
echo "使用以下命令检查进程是否仍在运行:"
echo "  ps -p ${PID}"

# 将 PID 记录到日志文件
echo "进程 ID: ${PID}" >> "${OUTPUT_DIR}/${JOB_NAME}.log"

# 创建一个简单的查看和管理脚本
cat > "${OUTPUT_DIR}/check_${JOB_NAME}.sh" << EOL
#!/bin/bash
# 查看和管理 ${JOB_NAME} 任务的脚本

function check_status() {
  if ps -p ${PID} > /dev/null; then
    echo "✅ 任务正在运行 (PID: ${PID})"
  else
    echo "❌ 任务已停止运行"
    
    # 检查是否有结束时间记录
    if grep -q "分析完成" "${OUTPUT_DIR}/${JOB_NAME}.out"; then
      echo "✓ 任务已正常完成"
      grep "分析完成" "${OUTPUT_DIR}/${JOB_NAME}.out" -A 3
    else
      echo "! 任务可能异常终止，检查输出文件末尾:"
      tail -n 20 "${OUTPUT_DIR}/${JOB_NAME}.out"
    fi
  fi
}

function show_output() {
  tail -n \${1:-50} "${OUTPUT_DIR}/${JOB_NAME}.out"
}

function follow_output() {
  tail -f "${OUTPUT_DIR}/${JOB_NAME}.out"
}

case "\$1" in
  status)
    check_status
    ;;
  output)
    show_output \$2
    ;;
  follow)
    follow_output
    ;;
  kill)
    echo "正在停止任务 (PID: ${PID})..."
    kill \$2 ${PID}
    ;;
  *)
    echo "使用方法: \$0 {status|output [行数]|follow|kill [-9]}"
    echo "  status - 检查任务状态"
    echo "  output [行数] - 显示输出文件的最后N行 (默认50行)"
    echo "  follow - 实时跟踪输出"
    echo "  kill [-9] - 停止任务 (可选-9参数强制终止)"
    ;;
esac
EOL

chmod +x "${OUTPUT_DIR}/check_${JOB_NAME}.sh"
echo ""
echo "已创建管理脚本:"
echo "  ${OUTPUT_DIR}/check_${JOB_NAME}.sh"
echo "使用以下命令管理任务:"
echo "  ${OUTPUT_DIR}/check_${JOB_NAME}.sh status  # 检查任务状态"
echo "  ${OUTPUT_DIR}/check_${JOB_NAME}.sh output  # 查看输出"
echo "  ${OUTPUT_DIR}/check_${JOB_NAME}.sh follow  # 实时跟踪输出"
echo "  ${OUTPUT_DIR}/check_${JOB_NAME}.sh kill    # 停止任务"