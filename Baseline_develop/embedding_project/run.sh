#!/bin/bash

# 脑区感知Subject Embedding分析启动脚本 - 修复版
# 🔥 新增功能：存档点管理、交互式恢复、智能重启

# 🔥 改进：更严格的错误处理
set -euo pipefail  # 更严格的错误处理
IFS=$'\n\t'       # 更安全的字段分隔符

# 颜色定义
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# 打印彩色信息函数
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
    echo -e "${RED}[ERROR]${NC} $1" >&2  # 错误信息输出到stderr
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_checkpoint() {
    echo -e "${CYAN}[CHECKPOINT]${NC} $1"
}

# 🔥 新增：错误处理函数
cleanup_on_error() {
    local exit_code=$?
    print_error "脚本执行失败，退出码: $exit_code"
    
    # 如果有PID文件，尝试清理进程
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            print_warning "清理后台进程: $pid"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    
    exit $exit_code
}

trap cleanup_on_error ERR

# 默认参数
DATA_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
TRAIN_START=1
TRAIN_END=31
VAL_START=31
VAL_END=38
TEST_SUBJECT=38
OUTPUT_DIR="./brain_aware_analysis_output"
RANDOM_STATE=42
LOG_LEVEL="INFO"
SKIP_VIZ=false
CONDA_ENV=""
PYTHON_CMD="python"

# 🔥 新增：存档点相关参数
DISABLE_CHECKPOINTS=false
RESUME_LATEST=false
RESUME_FROM=""
LIST_CHECKPOINTS=false
CHECKPOINT_AND_EXIT=""
FORCE_RESTART=false
INTERACTIVE_RECOVERY=false

# 全局变量
PID_FILE=""
NOHUP_LOG=""

# 🔥 增强版帮助信息
show_help() {
    cat << EOF
脑区感知Subject Embedding分析启动脚本 - 增强版 (带存档点支持)

用法: $0 [选项]

基础选项:
    -d, --data-path PATH        数据文件路径 (默认: $DATA_PATH)
    --train-start NUM           训练集受试者起始ID (默认: $TRAIN_START)
    --train-end NUM             训练集受试者结束ID (默认: $TRAIN_END)
    --val-start NUM             验证集受试者起始ID (默认: $VAL_START)
    --val-end NUM               验证集受试者结束ID (默认: $VAL_END)
    --test-subject NUM          测试集受试者ID (默认: $TEST_SUBJECT)
    -o, --output-dir PATH       输出目录路径 (默认: $OUTPUT_DIR)
    -r, --random-state NUM      随机种子 (默认: $RANDOM_STATE)
    -l, --log-level LEVEL       日志级别 [DEBUG|INFO|WARNING|ERROR] (默认: $LOG_LEVEL)
    --skip-viz                  跳过可视化生成
    -e, --conda-env ENV         指定conda环境名
    -p, --python-cmd CMD        指定Python命令 (默认: $PYTHON_CMD)

🔥 存档点选项:
    --disable-checkpoints       禁用存档点系统
    --resume-latest             自动恢复最新存档点
    --resume-from CHECKPOINT    从指定存档点恢复 (文件名或阶段名)
    --list-checkpoints          列出所有可用存档点并退出
    --checkpoint-and-exit NAME  创建指定名称的存档点后退出
    --force-restart             强制重新开始，忽略所有存档点
    --interactive-recovery      启用交互式恢复选择

    -h, --help                  显示此帮助信息

存档点使用示例:
    # 启用交互式恢复（推荐）
    $0 --interactive-recovery
    
    # 自动恢复最新存档点
    $0 --resume-latest
    
    # 从指定存档点恢复
    $0 --resume-from phase2_separability
    
    # 列出所有存档点
    $0 --list-checkpoints
    
    # 强制重新开始
    $0 --force-restart

EOF
}

# 🔥 新增：存档点管理函数
list_available_checkpoints() {
    local checkpoint_dir="$OUTPUT_DIR/checkpoints"
    
    if [[ ! -d "$checkpoint_dir" ]]; then
        print_warning "存档点目录不存在: $checkpoint_dir"
        return 1
    fi
    
    local checkpoint_files=($(find "$checkpoint_dir" -name "*.ckpt" -type f 2>/dev/null | sort -t_ -k3 -r))
    
    if [[ ${#checkpoint_files[@]} -eq 0 ]]; then
        print_warning "未找到任何存档点文件"
        return 1
    fi
    
    print_header "📂 可用存档点列表"
    echo ""
    
    local i=0
    for checkpoint_file in "${checkpoint_files[@]}"; do
        local filename=$(basename "$checkpoint_file")
        local size=$(ls -lh "$checkpoint_file" | awk '{print $5}')
        local date=$(stat -c %y "$checkpoint_file" 2>/dev/null | cut -d'.' -f1 || echo "Unknown")
        
        echo "[$i] $filename"
        echo "    大小: $size, 创建时间: $date"
        echo ""
        
        ((i++))
        
        # 只显示前10个
        if [[ $i -ge 10 ]]; then
            echo "... (还有 $((${#checkpoint_files[@]} - 10)) 个存档点)"
            break
        fi
    done
}

check_checkpoint_compatibility() {
    local checkpoint_path="$1"
    
    if [[ ! -f "$checkpoint_path" ]]; then
        print_error "存档点文件不存在: $checkpoint_path"
        return 1
    fi
    
    # 基础检查：文件是否可读
    if [[ ! -r "$checkpoint_path" ]]; then
        print_error "存档点文件不可读: $checkpoint_path"
        return 1
    fi
    
    # 检查文件大小
    local file_size=$(stat -c%s "$checkpoint_path" 2>/dev/null || echo "0")
    if [[ "$file_size" -lt 1000 ]]; then
        print_error "存档点文件可能损坏 (文件过小): $checkpoint_path"
        return 1
    fi
    
    print_success "存档点兼容性检查通过: $(basename "$checkpoint_path")"
    return 0
}

# 🔥 改进：参数验证函数
validate_parameters() {
    # 验证数值参数
    if ! [[ "$TRAIN_START" =~ ^[0-9]+$ ]] || (( TRAIN_START < 1 )); then
        print_error "训练集起始ID必须是正整数: $TRAIN_START"
        return 1
    fi
    
    if ! [[ "$TRAIN_END" =~ ^[0-9]+$ ]] || (( TRAIN_END <= TRAIN_START )); then
        print_error "训练集结束ID必须大于起始ID: $TRAIN_END > $TRAIN_START"
        return 1
    fi
    
    if ! [[ "$VAL_START" =~ ^[0-9]+$ ]] || (( VAL_START < 1 )); then
        print_error "验证集起始ID必须是正整数: $VAL_START"
        return 1
    fi
    
    if ! [[ "$VAL_END" =~ ^[0-9]+$ ]] || (( VAL_END <= VAL_START )); then
        print_error "验证集结束ID必须大于起始ID: $VAL_END > $VAL_START"
        return 1
    fi
    
    if ! [[ "$TEST_SUBJECT" =~ ^[0-9]+$ ]] || (( TEST_SUBJECT < 1 )); then
        print_error "测试受试者ID必须是正整数: $TEST_SUBJECT"
        return 1
    fi
    
    if ! [[ "$RANDOM_STATE" =~ ^[0-9]+$ ]]; then
        print_error "随机种子必须是非负整数: $RANDOM_STATE"
        return 1
    fi
    
    # 验证日志级别
    case "$LOG_LEVEL" in
        DEBUG|INFO|WARNING|ERROR) ;;
        *) print_error "无效的日志级别: $LOG_LEVEL"; return 1 ;;
    esac
    
    # 🔥 验证存档点相关参数
    if [[ "$RESUME_LATEST" == true && -n "$RESUME_FROM" ]]; then
        print_error "不能同时使用 --resume-latest 和 --resume-from"
        return 1
    fi
    
    if [[ "$FORCE_RESTART" == true && ("$RESUME_LATEST" == true || -n "$RESUME_FROM" || "$INTERACTIVE_RECOVERY" == true) ]]; then
        print_error "--force-restart 不能与其他恢复选项同时使用"
        return 1
    fi
    
    return 0
}

# 解析命令行参数（增强版）
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --train-start)
            TRAIN_START="$2"
            shift 2
            ;;
        --train-end)
            TRAIN_END="$2"
            shift 2
            ;;
        --val-start)
            VAL_START="$2"
            shift 2
            ;;
        --val-end)
            VAL_END="$2"
            shift 2
            ;;
        --test-subject)
            TEST_SUBJECT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--random-state)
            RANDOM_STATE="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --skip-viz)
            SKIP_VIZ=true
            shift
            ;;
        -e|--conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -p|--python-cmd)
            PYTHON_CMD="$2"
            shift 2
            ;;
        # 🔥 新增：存档点选项
        --disable-checkpoints)
            DISABLE_CHECKPOINTS=true
            shift
            ;;
        --resume-latest)
            RESUME_LATEST=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --list-checkpoints)
            LIST_CHECKPOINTS=true
            shift
            ;;
        --checkpoint-and-exit)
            CHECKPOINT_AND_EXIT="$2"
            shift 2
            ;;
        --force-restart)
            FORCE_RESTART=true
            shift
            ;;
        --interactive-recovery)
            INTERACTIVE_RECOVERY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 🔥 新增：验证参数
if ! validate_parameters; then
    print_error "参数验证失败"
    exit 1
fi

# 🔥 处理存档点列表请求
if [[ "$LIST_CHECKPOINTS" == true ]]; then
    list_available_checkpoints
    exit 0
fi

# 打印启动信息
print_header "🧠 脑区感知Subject Embedding分析 - 增强版"

print_info "分析参数:"
echo "  📁 数据路径: $DATA_PATH"
echo "  🎯 训练集: 受试者${TRAIN_START}-$((TRAIN_END-1))"
echo "  🎯 验证集: 受试者${VAL_START}-$((VAL_END-1))"
echo "  🎯 测试集: 受试者${TEST_SUBJECT}"
echo "  📂 输出目录: $OUTPUT_DIR"
echo "  🎲 随机种子: $RANDOM_STATE"
echo "  📊 日志级别: $LOG_LEVEL"
echo "  🎨 跳过可视化: $SKIP_VIZ"

# 🔥 新增：存档点配置显示
echo ""
print_checkpoint "存档点配置:"
echo "  💾 存档点系统: $([ "$DISABLE_CHECKPOINTS" == true ] && echo "禁用" || echo "启用")"
echo "  🔄 自动恢复最新: $RESUME_LATEST"
echo "  📂 从存档点恢复: ${RESUME_FROM:-"无"}"
echo "  🔧 交互式恢复: $INTERACTIVE_RECOVERY"
echo "  🚀 强制重新开始: $FORCE_RESTART"

# 检查数据文件是否存在
if [[ ! -f "$DATA_PATH" ]]; then
    print_error "数据文件不存在: $DATA_PATH"
    exit 1
fi

print_success "数据文件验证通过"

# 创建输出目录
if ! mkdir -p "$OUTPUT_DIR"; then
    print_error "无法创建输出目录: $OUTPUT_DIR"
    exit 1
fi
print_success "输出目录创建完成: $OUTPUT_DIR"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR" || {
    print_error "无法切换到脚本目录: $SCRIPT_DIR"
    exit 1
}

print_info "当前工作目录: $(pwd)"

# 🔥 改进：检查项目结构
required_files=("main.py" "src/analyzer.py" "src/data_loader.py" "src/utils.py" "config/settings.py" "checkpoint_manager.py")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "缺少必要文件: $file"
        exit 1
    fi
done
print_success "项目结构检查通过（包含存档点模块）"

# 检查Python环境
if [[ -n "$CONDA_ENV" ]]; then
    print_info "激活conda环境: $CONDA_ENV"
    
    # 检查conda是否可用
    if ! command -v conda &> /dev/null; then
        print_error "conda命令未找到，请确保已安装Anaconda/Miniconda"
        exit 1
    fi
    
    # 激活环境
    source "$(conda info --base)/etc/profile.d/conda.sh" || {
        print_error "无法加载conda配置"
        exit 1
    }
    
    conda activate "$CONDA_ENV" || {
        print_error "无法激活conda环境: $CONDA_ENV"
        exit 1
    }
    
    print_success "conda环境激活成功: $CONDA_ENV"
    PYTHON_CMD="python"
fi

# 检查Python和依赖
print_info "检查Python环境..."
if ! "$PYTHON_CMD" --version; then
    print_error "Python命令执行失败: $PYTHON_CMD"
    exit 1
fi

# 🔥 改进：更详细的依赖检查（包含存档点相关依赖）
print_info "检查Python依赖包..."
required_packages=("numpy" "pandas" "matplotlib" "sklearn" "scipy" "h5py" "torch" "psutil")

for package in "${required_packages[@]}"; do
    if ! "$PYTHON_CMD" -c "import $package" &> /dev/null; then
        print_error "缺少必要的Python包: $package"
        print_info "请安装: pip install $package"
        exit 1
    else
        # 获取包版本
        version=$("$PYTHON_CMD" -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        print_info "  ✓ $package: $version"
    fi
done

print_success "Python环境检查通过（包含存档点依赖）"

# 🔥 改进：设置环境变量
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg  # 确保matplotlib使用非交互式后端

# 检查GPU（如果可用）
if command -v nvidia-smi &> /dev/null; then
    print_info "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | \
    while IFS=, read -r index name total used; do
        echo "  GPU $index: $name (${used}MB/${total}MB)"
    done
else
    print_warning "未检测到NVIDIA GPU，将使用CPU进行计算"
fi

# 🔥 构建Python命令参数（增强版）
PYTHON_ARGS=(
    "--data_path" "$DATA_PATH"
    "--train_start" "$TRAIN_START"
    "--train_end" "$TRAIN_END"
    "--val_start" "$VAL_START"
    "--val_end" "$VAL_END"
    "--test_subject" "$TEST_SUBJECT"
    "--output_dir" "$OUTPUT_DIR"
    "--random_state" "$RANDOM_STATE"
    "--log_level" "$LOG_LEVEL"
)

if [[ "$SKIP_VIZ" == true ]]; then
    PYTHON_ARGS+=("--skip_visualization")
fi

# 🔥 添加存档点相关参数
if [[ "$DISABLE_CHECKPOINTS" == true ]]; then
    PYTHON_ARGS+=("--disable_checkpoints")
fi

if [[ "$RESUME_LATEST" == true ]]; then
    PYTHON_ARGS+=("--resume_latest")
fi

if [[ -n "$RESUME_FROM" ]]; then
    PYTHON_ARGS+=("--resume_from" "$RESUME_FROM")
    
    # 检查指定的存档点是否存在
    checkpoint_path="$OUTPUT_DIR/checkpoints/$RESUME_FROM"
    if [[ ! "$RESUME_FROM" == *".ckpt" ]]; then
        checkpoint_path="$OUTPUT_DIR/checkpoints/${RESUME_FROM}.ckpt"
    fi
    
    if [[ -f "$checkpoint_path" ]]; then
        check_checkpoint_compatibility "$checkpoint_path"
    else
        print_warning "指定的存档点文件不存在: $checkpoint_path"
        print_info "将尝试按阶段名查找存档点..."
    fi
fi

if [[ "$INTERACTIVE_RECOVERY" == true ]]; then
    PYTHON_ARGS+=("--interactive_recovery")
fi

if [[ "$FORCE_RESTART" == true ]]; then
    PYTHON_ARGS+=("--force_restart")
fi

if [[ -n "$CHECKPOINT_AND_EXIT" ]]; then
    PYTHON_ARGS+=("--checkpoint_and_exit" "$CHECKPOINT_AND_EXIT")
fi

# 准备日志文件
readonly TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NOHUP_LOG="$OUTPUT_DIR/run_${TIMESTAMP}.log"
PID_FILE="$OUTPUT_DIR/analysis.pid"

print_info "准备启动分析..."
print_info "nohup日志文件: $NOHUP_LOG"
print_info "PID文件: $PID_FILE"

# 🔥 新增：存档点状态显示
if [[ "$DISABLE_CHECKPOINTS" != true ]]; then
    print_checkpoint "存档点状态检查..."
    checkpoint_dir="$OUTPUT_DIR/checkpoints"
    if [[ -d "$checkpoint_dir" ]]; then
        checkpoint_count=$(find "$checkpoint_dir" -name "*.ckpt" -type f 2>/dev/null | wc -l)
        if [[ $checkpoint_count -gt 0 ]]; then
            print_checkpoint "发现 $checkpoint_count 个存档点"
            
            # 显示最新的几个存档点
            recent_checkpoints=($(find "$checkpoint_dir" -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2-))
            if [[ ${#recent_checkpoints[@]} -gt 0 ]]; then
                print_checkpoint "最近的存档点:"
                for ckpt_path in "${recent_checkpoints[@]}"; do
                    ckpt_name=$(basename "$ckpt_path")
                    ckpt_date=$(stat -c %y "$ckpt_path" 2>/dev/null | cut -d'.' -f1 || echo "Unknown")
                    echo "  • $ckpt_name ($ckpt_date)"
                done
            fi
        else
            print_checkpoint "未发现存档点，将执行完整分析"
        fi
    else
        print_checkpoint "存档点目录不存在，将执行完整分析"
    fi
fi

# 启动分析
print_header "🚀 启动分析进程"

# 🔥 改进：更好的进程启动
print_info "执行命令: $PYTHON_CMD main.py ${PYTHON_ARGS[*]}"

# 使用nohup在后台运行
nohup "$PYTHON_CMD" main.py "${PYTHON_ARGS[@]}" > "$NOHUP_LOG" 2>&1 &
ANALYSIS_PID=$!

# 保存PID
echo $ANALYSIS_PID > "$PID_FILE" || {
    print_error "无法保存PID文件"
    kill $ANALYSIS_PID 2>/dev/null || true
    exit 1
}

print_success "分析进程已启动"
print_info "进程PID: $ANALYSIS_PID"
print_info "nohup日志: $NOHUP_LOG"

# 🔥 改进：更长时间等待和更好的进程检查
print_info "等待进程稳定启动..."
sleep 5

if kill -0 $ANALYSIS_PID 2>/dev/null; then
    print_success "分析进程运行正常"
    
    print_info "监控命令:"
    echo "  查看实时日志: tail -f $NOHUP_LOG"
    echo "  查看进程状态: ps aux | grep $ANALYSIS_PID"
    echo "  停止分析:     kill $ANALYSIS_PID"
    echo "  强制停止:     kill -9 $ANALYSIS_PID"
    echo "  后台进程管理: jobs"
    
    # 🔥 新增：存档点管理命令
    if [[ "$DISABLE_CHECKPOINTS" != true ]]; then
        echo ""
        print_checkpoint "存档点管理命令:"
        echo "  列出存档点:   $0 --list-checkpoints"
        echo "  从存档点恢复: $0 --resume-from <checkpoint>"
        echo "  交互式恢复:   $0 --interactive-recovery"
        echo "  强制重新开始: $0 --force-restart"
    fi
    
    print_info "分析预计耗时: 10-30分钟（取决于数据规模和硬件配置）"
    
    # 🔥 改进：检查初始日志
    if [[ -f "$NOHUP_LOG" ]]; then
        echo ""
        print_info "初始日志内容:"
        head -n 10 "$NOHUP_LOG" 2>/dev/null | sed 's/^/  /'
    fi
    
    # 提供日志监控选项
    echo ""
    read -p "是否现在查看实时日志？[y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "开始监控日志（按Ctrl+C退出监控，不会停止分析）..."
        sleep 1
        tail -f "$NOHUP_LOG"
    else
        print_info "分析在后台运行中..."
        print_info "要查看日志，请运行: tail -f $NOHUP_LOG"
        
        # 🔥 新增：存档点进度提示
        if [[ "$DISABLE_CHECKPOINTS" != true ]]; then
            echo ""
            print_checkpoint "存档点进度提示:"
            echo "  • Phase 0: 数据准备完成后会创建第一个存档点"
            echo "  • Phase 1-4: 每个阶段完成后自动创建存档点"
            echo "  • 如果分析中断，可以使用存档点快速恢复"
            echo "  • 异常时会自动创建紧急存档点"
        fi
    fi
    
else
    print_error "分析进程启动失败"
    print_info "请检查日志文件: $NOHUP_LOG"
    
    # 显示错误日志
    if [[ -f "$NOHUP_LOG" ]]; then
        echo ""
        print_error "错误日志："
        tail -n 20 "$NOHUP_LOG" 2>/dev/null | sed 's/^/  /'
    fi
    
    # 清理PID文件
    rm -f "$PID_FILE"
    exit 1
fi

print_success "脚本执行完成"