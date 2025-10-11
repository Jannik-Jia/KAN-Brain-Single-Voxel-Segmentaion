#!/bin/bash
##############################################################################
# MRI Multi-modal Downsampling Pipeline - 批量处理启动脚本
##############################################################################
#
# 使用方法:
#   ./run_batch_downsampling.sh [mode]
#
# 模式:
#   test    - 测试模式（处理前3个被试）
#   full    - 完整处理（所有被试）
#   resume  - 断点续传（继续未完成的任务）
#   force   - 强制重新处理（不跳过已存在文件）
#
# 示例:
#   ./run_batch_downsampling.sh test
#   ./run_batch_downsampling.sh full
#
##############################################################################

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认路径（根据实际情况修改）
INPUT_DIR="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated"
OUTPUT_DIR="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling/3d"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 打印带颜色的消息
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "未找到Python。请安装Python 3.7+。"
        exit 1
    fi

    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python版本: $PYTHON_VERSION"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖包..."

    python -c "import numpy, scipy, h5py, SimpleITK, skimage" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "依赖包可能缺失。尝试安装..."
        pip install -r requirements_downsampling.txt
    else
        print_info "所有依赖包已安装 ✓"
    fi
}

# 检查输入目录
check_input_dir() {
    if [ ! -d "$INPUT_DIR" ]; then
        print_error "输入目录不存在: $INPUT_DIR"
        print_info "请修改脚本中的INPUT_DIR变量"
        exit 1
    fi

    NUM_FILES=$(ls -1 "$INPUT_DIR"/*_3d_validated.mat 2>/dev/null | wc -l)
    if [ "$NUM_FILES" -eq 0 ]; then
        print_error "未找到3D验证文件在: $INPUT_DIR"
        exit 1
    fi

    print_info "找到 $NUM_FILES 个3D验证文件 ✓"
}

# 显示系统信息
show_system_info() {
    print_header "系统信息"

    # CPU信息
    if command -v lscpu &> /dev/null; then
        CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
        CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
        print_info "CPU: $CPU_MODEL ($CPU_CORES cores)"
    fi

    # 内存信息
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
        AVAIL_MEM=$(free -h | awk '/^Mem:/ {print $7}')
        print_info "内存: $AVAIL_MEM 可用 / $TOTAL_MEM 总计"
    fi

    # 磁盘空间
    DISK_SPACE=$(df -h "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo "未知")
    print_info "输出目录可用空间: $DISK_SPACE"

    echo ""
}

# 测试模式
run_test_mode() {
    print_header "测试模式 - 处理前3个被试"

    python "$SCRIPT_DIR/batch_downsampling_pipeline.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --test-only \
        --log-level INFO

    if [ $? -eq 0 ]; then
        print_info "测试完成 ✓"
    else
        print_error "测试失败"
        exit 1
    fi
}

# 完整处理模式
run_full_mode() {
    print_header "完整处理模式 - 处理所有被试"

    # 确认
    echo -e "${YELLOW}即将处理所有被试，这可能需要数小时。${NC}"
    read -p "确认继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi

    python "$SCRIPT_DIR/batch_downsampling_pipeline.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --log-level INFO

    if [ $? -eq 0 ]; then
        print_info "完整处理完成 ✓"
    else
        print_error "处理过程中出现错误"
        exit 1
    fi
}

# 断点续传模式
run_resume_mode() {
    print_header "断点续传模式 - 继续未完成的任务"

    CHECKPOINT_FILE="$OUTPUT_DIR/downsampling_checkpoint.pkl"
    if [ -f "$CHECKPOINT_FILE" ]; then
        print_info "找到检查点文件 ✓"
    else
        print_warning "未找到检查点文件，将从头开始"
    fi

    python "$SCRIPT_DIR/batch_downsampling_pipeline.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --log-level INFO

    if [ $? -eq 0 ]; then
        print_info "续传完成 ✓"
    else
        print_error "处理过程中出现错误"
        exit 1
    fi
}

# 强制重新处理模式
run_force_mode() {
    print_header "强制重新处理模式 - 覆盖已存在文件"

    # 警告
    echo -e "${RED}警告: 这将覆盖所有已存在的输出文件！${NC}"
    read -p "确认继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi

    python "$SCRIPT_DIR/batch_downsampling_pipeline.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --no-skip-existing \
        --log-level INFO

    if [ $? -eq 0 ]; then
        print_info "强制处理完成 ✓"
    else
        print_error "处理过程中出现错误"
        exit 1
    fi
}

# 显示使用说明
show_usage() {
    cat << EOF
${BLUE}MRI Multi-modal Downsampling Pipeline - 批量处理${NC}

使用方法:
    $0 [mode]

模式:
    test    - 测试模式（处理前3个被试）
    full    - 完整处理（所有被试）
    resume  - 断点续传（继续未完成的任务）
    force   - 强制重新处理（不跳过已存在文件）

配置:
    输入目录: $INPUT_DIR
    输出目录: $OUTPUT_DIR

示例:
    $0 test         # 测试模式
    $0 full         # 完整处理
    $0 resume       # 续传

EOF
    exit 0
}

##############################################################################
# 主程序
##############################################################################

main() {
    # 检查参数
    if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        show_usage
    fi

    MODE=$1

    # 打印标题
    print_header "MRI Downsampling Pipeline - Batch Processing"
    echo -e "${BLUE}Version: 1.2.0 (with BUGFIX v1.2)${NC}"
    echo ""

    # 环境检查
    check_python
    check_dependencies
    check_input_dir

    # 显示系统信息
    show_system_info

    # 根据模式执行
    case $MODE in
        test)
            run_test_mode
            ;;
        full)
            run_full_mode
            ;;
        resume)
            run_resume_mode
            ;;
        force)
            run_force_mode
            ;;
        *)
            print_error "未知模式: $MODE"
            show_usage
            ;;
    esac

    # 显示输出位置
    echo ""
    print_header "处理完成"
    print_info "输出目录: $OUTPUT_DIR"
    print_info "日志目录: $OUTPUT_DIR/logs"
    print_info "查看报告: ls $OUTPUT_DIR/*.json"
    echo ""
}

# 执行主程序
main "$@"
