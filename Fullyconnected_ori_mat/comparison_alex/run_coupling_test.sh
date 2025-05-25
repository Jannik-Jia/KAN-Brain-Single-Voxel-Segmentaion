#!/bin/bash

# 脑体素分类 - 架构超参数耦合验证测试脚本
# 基于原始brain_voxel.ipynb训练流程

set -e  # 遇到错误立即停止

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建必要的目录
print_info "创建目录结构..."
mkdir -p coupling_test_results
mkdir -p logs

# 配置参数
EXPERIMENT_NAME="coupling_test_$(date +%Y%m%d_%H%M%S)"
MAT_FILE_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
OUTPUT_DIR="./coupling_test_results"
LOG_FILE="logs/${EXPERIMENT_NAME}.log"

print_info "实验配置:"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  数据文件: $MAT_FILE_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  日志文件: $LOG_FILE"

# 检查数据文件是否存在
if [ ! -f "$MAT_FILE_PATH" ]; then
    print_error "数据文件不存在: $MAT_FILE_PATH"
    print_info "请确认以下路径之一是否正确:"
    echo "  /home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
    echo "  ./TRAIN38.mat"
    echo "  ../TRAIN38.mat"
    echo ""
    echo "或者修改脚本中的 MAT_FILE_PATH 变量"
    exit 1
fi

print_success "数据文件检查通过: $MAT_FILE_PATH"

# 检查Python环境
print_info "检查Python环境..."
python3 -c "
import torch
import numpy as np
import h5py
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ NumPy: {np.__version__}')
print(f'✓ Scikit-learn: {sklearn.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠ CUDA not available, will use CPU')
"

if [ $? -ne 0 ]; then
    print_error "Python环境检查失败，请确保安装了所需的包"
    exit 1
fi

print_success "Python环境检查通过"

# 创建耦合测试脚本 (如果不存在)
COUPLING_SCRIPT="coupling_test.py"
if [ ! -f "$COUPLING_SCRIPT" ]; then
    print_info "创建耦合测试脚本..."
    
    # 这里我们假设耦合测试的Python代码已经存在
    # 如果不存在，用户需要先创建 coupling_test.py 文件
    if [ ! -f "coupling_test.py" ]; then
        print_error "未找到 coupling_test.py 脚本文件"
        print_info "请确保 coupling_test.py 文件存在于当前目录"
        print_info "或者将完整的耦合测试代码保存为 coupling_test.py"
        exit 1
    fi
fi

# 显示实验信息
print_info "实验详情:"
echo "  🏗️  测试架构:"
echo "     - original: [4096, 4096, 4096, 4096] (基准)"
echo "     - wide_shallow: [8192, 8192] (宽浅)"
echo "     - narrow_deep: [2048×6] (窄深)"
echo ""
echo "  📊 测试学习率:"
echo "     - 5e-6 (原始的0.5倍)"
echo "     - 1e-5 (原始值，baseline)"
echo "     - 2e-5 (原始的2倍)"
echo "     - 5e-5 (原始的5倍)"
echo ""
echo "  ⚙️  固定参数:"
echo "     - Batch size: 128"
echo "     - Epochs: 12 (快速测试)"
echo "     - Dropout: 0.5"
echo "     - L2 regularization: 1e-5"
echo ""
echo "  📈 总实验数: 12 (3架构 × 4学习率)"
echo "  ⏱️  预计时间: 3-4小时"

# 询问用户是否继续
echo ""
read -p "是否开始耦合测试? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "用户取消了实验"
    exit 0
fi

# 开始实验
print_info "启动耦合测试实验..."

# 运行耦合测试
nohup python3 -u coupling_test.py \
    --mat_file_path "$MAT_FILE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "auto" \
    > "$LOG_FILE" 2>&1 &

PID=$!

# 等待几秒检查进程是否正常启动
sleep 3
if ps -p $PID > /dev/null; then
    print_success "耦合测试已成功启动!"
else
    print_error "耦合测试启动失败"
    print_info "检查日志文件: $LOG_FILE"
    if [ -f "$LOG_FILE" ]; then
        echo "最近的错误信息:"
        tail -n 10 "$LOG_FILE"
    fi
    exit 1
fi

# 显示监控信息
echo ""
echo "🚀 实验运行信息:"
echo "   进程ID: $PID"
echo "   日志文件: $LOG_FILE"
echo "   输出目录: $OUTPUT_DIR"
echo ""
echo "📊 监控命令:"
echo "   实时日志: tail -f $LOG_FILE"
echo "   定期检查: watch -n 30 'tail -n 15 $LOG_FILE'"
echo "   进程状态: ps aux | grep $PID"
echo ""
echo "⏹️  停止实验: kill $PID"
echo ""

# 显示预期的输出文件
print_info "实验完成后，将生成以下文件:"
echo "   📄 coupling_results_[timestamp].json - 详细结果数据"
echo "   📋 coupling_report_[timestamp].txt - 分析报告"
echo "   📊 实验将回答以下关键问题:"
echo "      1. 不同架构是否需要不同的学习率?"
echo "      2. 架构和学习率之间的耦合强度如何?"
echo "      3. 应该采用什么优化策略?"

# 创建一个简单的监控脚本
cat > monitor_coupling.sh << 'EOF'
#!/bin/bash
PID=$1
LOG_FILE=$2

if [ -z "$PID" ] || [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <PID> <LOG_FILE>"
    exit 1
fi

echo "监控耦合测试进程 $PID"
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 退出监控"
echo ""

while ps -p $PID > /dev/null; do
    clear
    echo "🔄 耦合测试进行中... (PID: $PID)"
    echo "⏰ $(date)"
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        echo "📋 最新日志 (最后15行):"
        echo "----------------------------------------"
        tail -n 15 "$LOG_FILE" | sed 's/^/   /'
        echo "----------------------------------------"
    else
        echo "等待日志文件生成..."
    fi
    
    sleep 30
done

echo ""
echo "✅ 耦合测试已完成或进程已停止"
if [ -f "$LOG_FILE" ]; then
    echo "📋 最终日志:"
    tail -n 20 "$LOG_FILE"
fi
EOF

chmod +x monitor_coupling.sh

echo "🔍 启动监控: ./monitor_coupling.sh $PID $LOG_FILE"
echo ""

# 可选：立即开始监控
read -p "是否立即开始监控? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "开始监控模式..."
    ./monitor_coupling.sh $PID $LOG_FILE
else
    print_info "后台运行中，可稍后使用监控命令查看进度"
    print_success "耦合测试脚本执行完成"
fi