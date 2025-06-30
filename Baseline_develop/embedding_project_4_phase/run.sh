#!/bin/bash

# 脑区感知Subject Embedding分析系统运行脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "脑区感知Subject Embedding分析系统"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  all                    运行所有Phase (0-4)"
    echo "  phase <N>              运行单个Phase (N=0,1,2,3,4)"
    echo "  status                 显示所有Phase的执行状态"
    echo "  clean                  清理所有输出"
    echo "  help                   显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 all                 # 运行完整分析"
    echo "  $0 phase 0             # 只运行数据准备"
    echo "  $0 phase 1             # 只运行受试者分析"
    echo "  $0 status              # 查看执行状态"
    echo ""
    echo "高级用法:"
    echo "  运行Phase 0并指定数据路径:"
    echo "  python orchestrator.py --run phase --phase 0 --data-path /path/to/data.mat"
    echo ""
    echo "  从Phase 2开始运行:"
    echo "  python orchestrator.py --run all --start-from 2"
    echo ""
    echo "  跳过可视化:"
    echo "  python orchestrator.py --run all --skip-visualization"
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未安装"
        exit 1
    fi
    
    print_info "Python版本: $(python3 --version)"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖包..."
    
    required_packages=("numpy" "scipy" "sklearn" "torch" "matplotlib" "h5py")
    
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_success "$package ✓"
        else
            print_error "$package ✗ (需要安装: pip install $package)"
            exit 1
        fi
    done
}

# 清理输出
clean_outputs() {
    print_info "清理输出目录..."
    
    if [ -d "data_exchange" ]; then
        rm -rf data_exchange/*
        print_success "输出目录已清理"
    fi
}

# 主逻辑
case "$1" in
    all)
        check_python
        check_dependencies
        print_info "运行完整分析流程..."
        python3 orchestrator.py --run all
        ;;
    
    phase)
        if [ -z "$2" ]; then
            print_error "请指定Phase编号"
            show_help
            exit 1
        fi
        check_python
        check_dependencies
        print_info "运行Phase $2..."
        python3 orchestrator.py --run phase --phase "$2"
        ;;
    
    status)
        check_python
        python3 orchestrator.py --list-status
        ;;
    
    clean)
        clean_outputs
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        print_error "无效的命令: $1"
        show_help
        exit 1
        ;;
esac