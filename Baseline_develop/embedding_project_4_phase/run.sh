#!/bin/bash

# 脑区感知Subject Embedding分析系统运行脚本 (增强版)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_section() {
    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# 显示帮助信息
show_help() {
    print_section "脑区感知Subject Embedding分析系统 v2.0"
    
    echo "用法: $0 [选项]"
    echo ""
    echo "${CYAN}基本选项:${NC}"
    echo "  all                    运行所有Phase (0-4)"
    echo "  phase <N>              运行单个Phase (N=0,1,2,3,4)"
    echo "  status                 显示所有Phase的执行状态"
    echo "  clean                  清理所有输出"
    echo "  help                   显示此帮助信息"
    echo ""
    echo "${CYAN}Phase 1专用选项:${NC}"
    echo "  phase1-quick           运行Phase 1但跳过UMAP（快速模式）"
    echo "  phase1-cpu             运行Phase 1使用CPU（不用GPU）"
    echo "  phase1-full            运行Phase 1完整版（包含所有可视化）"
    echo ""
    echo "${CYAN}Phase 2专用选项:${NC}"
    echo "  phase2-step <N>        从Phase 2的指定步骤开始 (N=1-6)"
    echo ""
    echo "${CYAN}快速启动:${NC}"
    echo "  quick                  运行Phase 0和1（跳过耗时的UMAP）"
    echo "  full-gpu               运行所有Phase并启用所有GPU加速"
    echo ""
    echo "${CYAN}系统检查:${NC}"
    echo "  gpu-check              检查GPU状态和CUDA环境"
    echo "  mem-check              检查系统内存"
    echo "  env-check              完整环境检查（推荐首次运行）"
    echo ""
    echo "${CYAN}示例:${NC}"
    echo "  $0 env-check           # 首次运行前检查环境"
    echo "  $0 quick               # 快速分析（适合初步探索）"
    echo "  $0 phase1-full         # 运行完整的Phase 1分析"
    echo "  $0 full-gpu            # GPU全速运行所有分析"
    echo ""
    echo "${CYAN}Phase 2步骤说明:${NC}"
    echo "  1: Baseline性能测试"
    echo "  2: LOSO评估（新的epoch-wise版本）"
    echo "  3: 深度网络权威分析"
    echo "  4: 分脑区Subject分类性能"
    echo "  5: Embedding需求评估"
    echo "  6: 生成可视化"
    echo ""
    echo "${CYAN}高级用法:${NC}"
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
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python版本: $python_version"
    
    # 检查Python版本是否>=3.7
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" 2>/dev/null; then
        print_error "需要Python 3.7或更高版本"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    print_section "检查依赖包"
    
    required_packages=("numpy" "scipy" "sklearn" "torch" "matplotlib" "h5py" "pandas" "seaborn")
    optional_packages=("cupy" "cuml" "umap" "plotly")
    
    all_ok=true
    
    # 检查必需包
    print_info "检查核心依赖包..."
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            # 获取版本信息
            version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            print_success "$package ✓ (version: $version)"
        else
            print_error "$package ✗ (需要安装: pip install $package)"
            all_ok=false
        fi
    done
    
    # 检查可选包
    echo ""
    print_info "检查可选包（用于GPU加速和高级功能）..."
    for package in "${optional_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            print_success "$package ✓ (version: $version)"
        else
            print_warning "$package ✗ (可选，用于加速)"
        fi
    done
    
    if [ "$all_ok" = false ]; then
        print_error "缺少必需的依赖包"
        exit 1
    fi
}

# 检查GPU
check_gpu() {
    print_section "检查GPU状态"
    
    # 检查CUDA
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        print_success "CUDA可用 ✓"
        
        # 显示GPU信息
        gpu_info=$(python3 -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'GPU设备: {torch.cuda.get_device_name(device)}')
    props = torch.cuda.get_device_properties(device)
    print(f'显存容量: {props.total_memory / 1024**3:.1f} GB')
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'PyTorch版本: {torch.__version__}')
" 2>/dev/null)
        echo "$gpu_info" | while IFS= read -r line; do
            print_info "$line"
        done
        
        # 检查当前显存使用
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            print_info "当前GPU使用情况:"
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{printf "  GPU %d: %s | 显存: %s/%s MB | 使用率: %s%%\n", $1, $2, $3, $4, $5}'
        fi
    else
        print_warning "CUDA不可用，将使用CPU进行计算"
        print_info "提示: Phase 1的多标签分析和UMAP可能会较慢"
    fi
    
    echo ""
    # 检查cupy
    if python3 -c "import cupy" 2>/dev/null; then
        print_success "CuPy已安装 ✓ (GPU加速的NumPy)"
        cupy_version=$(python3 -c "import cupy; print(cupy.__version__)" 2>/dev/null)
        print_info "CuPy版本: $cupy_version"
    else
        print_warning "CuPy未安装，多标签分析将使用CPU"
        print_info "安装命令: pip install cupy-cuda11x  # 根据CUDA版本选择"
    fi
    
    echo ""
    # 检查cuML
    if python3 -c "from cuml import UMAP" 2>/dev/null; then
        print_success "cuML已安装 ✓ (GPU加速的机器学习)"
        cuml_version=$(python3 -c "import cuml; print(cuml.__version__)" 2>/dev/null || echo "unknown")
        print_info "cuML版本: $cuml_version"
    else
        print_warning "cuML未安装，UMAP将使用CPU版本"
        print_info "cuML安装较复杂，请参考: https://rapids.ai/start.html"
    fi
}

# 检查内存
check_memory() {
    print_section "检查系统内存"
    
    # 获取内存信息
    if command -v free &> /dev/null; then
        mem_info=$(free -h | grep "^Mem:")
        total_mem=$(echo $mem_info | awk '{print $2}')
        used_mem=$(echo $mem_info | awk '{print $3}')
        available_mem=$(echo $mem_info | awk '{print $7}')
        
        print_info "总内存: $total_mem"
        print_info "已使用: $used_mem"
        print_info "可用内存: $available_mem"
        
        # 转换为GB进行比较
        available_gb=$(free -g | awk '/^Mem:/{print $7}')
        
        if [ "$available_gb" -lt 10 ]; then
            print_error "可用内存不足10GB！"
            print_warning "Phase 1的UMAP可视化可能会失败"
            print_warning "建议: 关闭其他程序或使用 --skip_umap 参数"
        elif [ "$available_gb" -lt 20 ]; then
            print_warning "可用内存较少（<20GB）"
            print_warning "处理大数据集时可能需要使用采样策略"
        else
            print_success "内存充足 ✓"
        fi
    else
        print_warning "无法检查内存（free命令不可用）"
    fi
}

# 完整环境检查
env_check() {
    print_section "完整环境检查"
    
    check_python
    echo ""
    check_dependencies
    echo ""
    check_gpu
    echo ""
    check_memory
    echo ""
    
    # 检查输出目录
    print_info "检查工作目录..."
    if [ -w "." ]; then
        print_success "当前目录可写 ✓"
    else
        print_error "当前目录不可写"
        exit 1
    fi
    
    # 检查数据交换目录
    if [ -d "data_exchange" ]; then
        print_info "data_exchange目录已存在"
        # 检查是否有之前的运行结果
        if [ -d "data_exchange/phase0_output" ] && [ "$(ls -A data_exchange/phase0_output 2>/dev/null)" ]; then
            print_warning "检测到之前的运行结果"
        fi
    else
        print_info "将创建data_exchange目录"
    fi
    
    print_section "环境检查完成"
    print_success "系统准备就绪！"
}

# 清理输出
clean_outputs() {
    print_section "清理输出"
    
    if [ -d "data_exchange" ]; then
        print_info "清理data_exchange目录..."
        
        # 显示将要删除的内容大小
        if command -v du &> /dev/null; then
            size=$(du -sh data_exchange 2>/dev/null | cut -f1)
            print_info "将删除 $size 的数据"
        fi
        
        # 确认删除
        read -p "确定要删除所有输出吗？[y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf data_exchange/*
            print_success "输出目录已清理"
        else
            print_info "取消清理"
        fi
    else
        print_info "没有需要清理的内容"
    fi
}

# 运行Phase 1快速模式
run_phase1_quick() {
    print_section "运行Phase 1 (快速模式)"
    print_info "将跳过耗时的UMAP可视化..."
    
    python3 phase1_subject_analysis/main.py --skip_umap "$@"
}

# 运行Phase 1 CPU模式
run_phase1_cpu() {
    print_section "运行Phase 1 (CPU模式)"
    print_info "将使用CPU进行所有计算..."
    
    python3 phase1_subject_analysis/main.py --use_gpu false "$@"
}

# 运行Phase 1完整模式
run_phase1_full() {
    print_section "运行Phase 1 (完整模式)"
    print_info "将生成所有可视化（包括102个脑区的UMAP）..."
    print_warning "预计耗时较长，请耐心等待"
    
    python3 phase1_subject_analysis/main.py "$@"
}

# 运行Phase 2的特定步骤
run_phase2_step() {
    local step=$1
    shift  # 移除第一个参数，剩下的都是额外参数
    
    print_section "Phase 2步骤模式"
    print_info "从步骤 $step 开始执行..."
    
    # 检查是否有baseline结果
    if [ "$step" -gt 1 ] && [ ! -f "data_exchange/phase2_output/baseline_results.json" ]; then
        print_warning "未找到baseline_results.json"
        print_info "将使用--use_existing_baseline参数"
        python3 phase2_separability/main.py --start_from_step "$step" --use_existing_baseline "$@"
    else
        python3 phase2_separability/main.py --start_from_step "$step" "$@"
    fi
}

# 快速分析模式
run_quick_analysis() {
    print_section "快速分析模式"
    print_info "将运行Phase 0和Phase 1（跳过UMAP）"
    
    # Phase 0
    print_info "\n运行Phase 0: 数据准备..."
    if ! python3 orchestrator.py --run phase --phase 0; then
        print_error "Phase 0执行失败"
        exit 1
    fi
    
    # Phase 1 快速版
    print_info "\n运行Phase 1: 受试者分析（快速版）..."
    if ! python3 phase1_subject_analysis/main.py --skip_umap; then
        print_error "Phase 1执行失败"
        exit 1
    fi
    
    print_success "\n快速分析完成！"
    print_info "您可以查看初步结果，决定是否需要运行完整分析"
}

# GPU全速模式
run_full_gpu() {
    print_section "GPU全速模式"
    
    # 先进行环境检查
    check_gpu
    check_memory
    
    # 确认GPU可用
    if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        print_error "GPU不可用，无法运行GPU全速模式"
        exit 1
    fi
    
    print_info "使用GPU加速运行所有Phase..."
    python3 orchestrator.py --run all
}

# 启动时检查
startup_check() {
    # 基础Python检查
    check_python
    
    # 如果是特定命令，进行额外检查
    case "$1" in
        *gpu*|all|full-gpu)
            print_info "此命令需要GPU支持，正在检查..."
            check_gpu
            ;;
        phase)
            if [ "$2" == "1" ]; then
                print_info "Phase 1可能需要大量内存，正在检查..."
                check_memory
            fi
            ;;
        phase1-*)
            print_info "Phase 1需要额外资源，正在检查..."
            check_memory
            ;;
    esac
}

# 主逻辑
main() {
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 根据命令执行相应操作
    case "$1" in
        all)
            startup_check "$@"
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
            startup_check "$@"
            check_dependencies
            print_info "运行Phase $2..."
            python3 orchestrator.py --run phase --phase "$2"
            ;;
        
        phase1-quick)
            startup_check "$@"
            check_dependencies
            shift
            run_phase1_quick "$@"
            ;;
        
        phase1-cpu)
            startup_check "$@"
            check_dependencies
            shift
            run_phase1_cpu "$@"
            ;;
        
        phase1-full)
            startup_check "$@"
            check_dependencies
            shift
            run_phase1_full "$@"
            ;;
        
        phase2-step)
            if [ -z "$2" ]; then
                print_error "请指定Phase 2的步骤编号 (1-6)"
                show_help
                exit 1
            fi
            
            step_num="$2"
            
            # 验证步骤编号
            if ! [[ "$step_num" =~ ^[1-6]$ ]]; then
                print_error "无效的步骤编号: $step_num (必须是1-6)"
                exit 1
            fi
            
            startup_check "$@"
            check_dependencies
            
            # 获取额外参数
            shift 2
            run_phase2_step "$step_num" "$@"
            ;;
        
        quick)
            startup_check "$@"
            check_dependencies
            run_quick_analysis
            ;;
        
        full-gpu)
            startup_check "$@"
            check_dependencies
            run_full_gpu
            ;;
        
        status)
            check_python
            python3 orchestrator.py --list-status
            ;;
        
        clean)
            clean_outputs
            ;;
        
        gpu-check)
            check_gpu
            ;;
        
        mem-check)
            check_memory
            ;;
        
        env-check)
            env_check
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
}

# 执行主函数
main "$@"