#!/bin/bash

# =============================================================================
# 多模型训练脚本 - 交互式选择和独立保存
# Multi-Model Training Script - Interactive Selection with Separate Outputs
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# =============================================================================
# 配置参数
# =============================================================================

# 训练参数（可通过环境变量覆盖）
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
ROOT_DIR="${ROOT_DIR:-/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS}"
INCLUDE_BACKGROUND="${INCLUDE_BACKGROUND:-false}"

# 脚本配置
SCRIPT_NAME="train.py"
BASE_OUTPUT_DIR="./training_runs"
LOG_DIR="./logs"

# 可用模型列表
declare -A MODELS
MODELS=(
    ["1"]="reg_model|深度全连接神经网络（Alex identical）"
    ["2"]="resnet_mlp|带残差连接的全连接网络"
    ["3"]="simple_mlp|简单多层感知机（轻量级）"
    ["4"]="kan|KAN模型（自动类权重）"
    ["5"]="deep_mlp|超深度MLP（特征交互+残差+自注意力）"
)

# =============================================================================
# 辅助函数
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# =============================================================================
# 主界面
# =============================================================================

print_header "多模型MRI体素分割训练系统"

echo -e "${GREEN}开始时间: $(date)${NC}"
echo -e "${CYAN}当前配置:${NC}"
echo -e "  训练轮数: ${YELLOW}$EPOCHS${NC}"
echo -e "  批大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  学习率: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "  包含背景: ${YELLOW}$INCLUDE_BACKGROUND${NC}"

# =============================================================================
# 环境检查
# =============================================================================

print_section "环境检查"

# 检查Python
echo -n "检查Python环境... "
if python --version &> /dev/null; then
    python_version=$(python --version 2>&1)
    echo -e "${GREEN}✓ $python_version${NC}"
else
    echo -e "${RED}✗ Python未安装${NC}"
    exit 1
fi

# 检查必要的包
echo -n "检查PyTorch... "
if python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ 未安装${NC}"
    exit 1
fi

echo -n "检查nibabel... "
if python -c "import nibabel" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ 未安装${NC}"
    exit 1
fi

echo -n "检查sklearn... "
if python -c "import sklearn" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ 未安装${NC}"
    exit 1
fi

echo -n "检查fastkan（KAN模型需要）... "
if python -c "import fastkan" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
    fastkan_available=true
else
    echo -e "${YELLOW}✗ 未安装（如果不使用KAN模型可忽略）${NC}"
    fastkan_available=false
fi

# 检查训练脚本
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo -e "${RED}✗ 找不到训练脚本: $SCRIPT_NAME${NC}"
    exit 1
fi

# =============================================================================
# 模型选择
# =============================================================================

print_section "模型选择"

echo -e "${CYAN}可用模型:${NC}"
echo ""

for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort -n); do
    IFS='|' read -r model_name description <<< "${MODELS[$key]}"

    # 特殊标记KAN模型
    if [[ "$model_name" == "kan" ]]; then
        if [[ "$fastkan_available" == "true" ]]; then
            echo -e "  ${MAGENTA}[$key]${NC} ${YELLOW}$model_name${NC} - $description ${GREEN}✓${NC}"
        else
            echo -e "  ${MAGENTA}[$key]${NC} ${YELLOW}$model_name${NC} - $description ${RED}(需要fastkan)${NC}"
        fi
    else
        echo -e "  ${MAGENTA}[$key]${NC} ${YELLOW}$model_name${NC} - $description"
    fi
done

echo ""
echo -e "${CYAN}请选择要训练的模型（可多选）:${NC}"
echo -e "${YELLOW}输入格式: 用空格分隔，例如 '1 2 4' 表示训练 reg_model, resnet_mlp, 和 kan${NC}"
echo -e "${YELLOW}输入 'all' 训练所有模型${NC}"
echo -n -e "${GREEN}您的选择: ${NC}"

read -r user_input

# 解析用户输入
selected_models=()

if [[ "$user_input" == "all" ]]; then
    # 选择所有模型
    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r model_name description <<< "${MODELS[$key]}"

        # 如果是KAN模型且fastkan未安装，跳过
        if [[ "$model_name" == "kan" && "$fastkan_available" == "false" ]]; then
            echo -e "${YELLOW}⚠️  跳过KAN模型（fastkan未安装）${NC}"
            continue
        fi

        selected_models+=("$model_name")
    done
else
    # 解析用户选择的数字
    for num in $user_input; do
        if [[ -n "${MODELS[$num]}" ]]; then
            IFS='|' read -r model_name description <<< "${MODELS[$num]}"

            # 检查KAN模型的依赖
            if [[ "$model_name" == "kan" && "$fastkan_available" == "false" ]]; then
                echo -e "${RED}✗ KAN模型需要fastkan库，请先安装: pip install fastkan${NC}"
                exit 1
            fi

            selected_models+=("$model_name")
        else
            echo -e "${RED}✗ 无效的选择: $num${NC}"
            exit 1
        fi
    done
fi

# 验证是否选择了模型
if [[ ${#selected_models[@]} -eq 0 ]]; then
    echo -e "${RED}✗ 未选择任何模型${NC}"
    exit 1
fi

# 显示选择的模型
print_section "训练计划"

echo -e "${GREEN}将训练以下 ${#selected_models[@]} 个模型:${NC}"
for i in "${!selected_models[@]}"; do
    model_name="${selected_models[$i]}"
    echo -e "  ${CYAN}$((i+1)).${NC} ${YELLOW}$model_name${NC}"
done

echo ""
echo -e "${YELLOW}每个模型将花费约 15-60 分钟（取决于模型和数据量）${NC}"
echo -e "${YELLOW}总预计时间: $(( ${#selected_models[@]} * 30 )) 分钟左右${NC}"
echo ""
echo -n -e "${GREEN}确认开始训练？(y/n): ${NC}"
read -r confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    exit 0
fi

# =============================================================================
# 创建输出目录结构
# =============================================================================

# 创建基础目录
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 生成总时间戳
MASTER_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# 训练函数
# =============================================================================

train_single_model() {
    local model_name=$1
    local model_index=$2
    local total_models=$3

    print_header "训练模型 [$model_index/$total_models]: $model_name"

    # 创建模型专用输出目录
    local bg_str
    if [[ "$INCLUDE_BACKGROUND" == "true" ]]; then
        bg_str="incl"
    else
        bg_str="excl"
    fi

    local output_dir="${BASE_OUTPUT_DIR}/${model_name}_bg_${bg_str}_${MASTER_TIMESTAMP}"
    mkdir -p "$output_dir/results"
    mkdir -p "$output_dir/logs"

    echo -e "${CYAN}输出目录: $output_dir${NC}"

    # 记录开始时间
    local start_time=$(date +%s)

    # 构建训练命令
    local cmd="python $SCRIPT_NAME"
    cmd="$cmd --model $model_name"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --lr $LEARNING_RATE"
    cmd="$cmd --root_dir $ROOT_DIR"
    cmd="$cmd --exclude_features 14"

    # 添加背景参数
    if [[ "$INCLUDE_BACKGROUND" == "true" ]]; then
        cmd="$cmd --include_background"
    fi

    echo -e "${GREEN}执行命令:${NC}"
    echo -e "${YELLOW}$cmd${NC}"
    echo ""

    # 日志文件
    local log_file="$output_dir/logs/training_${model_name}_${MASTER_TIMESTAMP}.log"

    # 运行训练
    echo -e "${BLUE}开始训练... (日志: $log_file)${NC}"

    # 临时修改输出目录（通过修改train.py中的results路径）
    # 我们需要在当前目录创建一个临时的results链接
    local old_results_dir="./results"
    local temp_results_dir="${output_dir}/results"

    # 创建目标results目录
    mkdir -p "$temp_results_dir"

    # 保存原始results目录（如果存在）
    if [[ -d "$old_results_dir" ]] && [[ ! -L "$old_results_dir" ]]; then
        mv "$old_results_dir" "${old_results_dir}_backup_$$"
    elif [[ -L "$old_results_dir" ]]; then
        rm "$old_results_dir"
    fi

    # 创建符号链接（使用绝对路径）
    local abs_temp_results_dir=$(cd "$(dirname "$temp_results_dir")" && pwd)/$(basename "$temp_results_dir")
    ln -sf "$abs_temp_results_dir" "$old_results_dir"

    # 验证符号链接
    if [[ ! -L "$old_results_dir" ]]; then
        echo -e "${RED}❌ 创建符号链接失败${NC}"
        return 1
    fi

    # 运行训练
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))

        echo -e "${GREEN}✅ 模型 $model_name 训练完成!${NC}"
        echo -e "${GREEN}   耗时: ${minutes}分${seconds}秒${NC}"

        # 恢复results目录
        if [[ -L "$old_results_dir" ]]; then
            rm "$old_results_dir"
        elif [[ -d "$old_results_dir" ]]; then
            rm -rf "$old_results_dir"
        fi
        if [[ -d "${old_results_dir}_backup_$$" ]]; then
            mv "${old_results_dir}_backup_$$" "$old_results_dir"
        fi

        # 整理输出文件
        organize_outputs "$output_dir" "$model_name"

        return 0
    else
        echo -e "${RED}❌ 模型 $model_name 训练失败!${NC}"

        # 恢复results目录
        if [[ -L "$old_results_dir" ]]; then
            rm "$old_results_dir"
        elif [[ -d "$old_results_dir" ]]; then
            rm -rf "$old_results_dir"
        fi
        if [[ -d "${old_results_dir}_backup_$$" ]]; then
            mv "${old_results_dir}_backup_$$" "$old_results_dir"
        fi

        return 1
    fi
}

# =============================================================================
# 整理输出文件
# =============================================================================

organize_outputs() {
    local output_dir=$1
    local model_name=$2

    echo -e "${CYAN}整理输出文件...${NC}"

    # 检查results目录中的文件
    local results_dir="${output_dir}/results"

    if [[ -d "$results_dir" ]]; then
        # 统计文件（去除空格）
        local pth_count=$(find "$results_dir" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')
        local nii_count=$(find "$results_dir" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        local png_count=$(find "$results_dir" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        local analysis_count=$(find "$results_dir" -name "per_class_analysis_*" -type d 2>/dev/null | wc -l | tr -d ' ')

        # 调试信息
        echo -e "${YELLOW}  调试: 检查目录 $results_dir${NC}"
        echo -e "${YELLOW}  调试: 目录内容:${NC}"
        ls -la "$results_dir" 2>/dev/null | head -10

        echo -e "${GREEN}✓ 模型文件: $pth_count${NC}"
        echo -e "${GREEN}✓ 3D Softmax: $nii_count${NC}"
        echo -e "${GREEN}✓ 训练曲线: $png_count${NC}"
        echo -e "${GREEN}✓ Per-class分析: $analysis_count${NC}"

        # 创建汇总文件
        cat > "${output_dir}/SUMMARY.txt" << EOF
训练汇总报告
================================================================================
模型名称: $model_name
训练时间: $MASTER_TIMESTAMP
训练参数:
  - Epochs: $EPOCHS
  - Batch Size: $BATCH_SIZE
  - Learning Rate: $LEARNING_RATE
  - Include Background: $INCLUDE_BACKGROUND

输出文件统计:
  - 模型权重 (.pth): $pth_count
  - 3D Softmax (.nii.gz): $nii_count
  - 训练曲线 (.png): $png_count
  - Per-class分析目录: $analysis_count

文件位置:
  - 主目录: $output_dir
  - 结果: $output_dir/results/
  - 日志: $output_dir/logs/

生成时间: $(date)
================================================================================
EOF

        echo -e "${GREEN}✓ 汇总报告: ${output_dir}/SUMMARY.txt${NC}"
    else
        echo -e "${YELLOW}⚠️  未找到results目录${NC}"
    fi
}

# =============================================================================
# 主训练循环
# =============================================================================

print_section "开始训练"

# 记录总开始时间
TOTAL_START_TIME=$(date +%s)

# 训练统计
successful_models=()
failed_models=()

# 逐个训练模型
for i in "${!selected_models[@]}"; do
    model_name="${selected_models[$i]}"
    model_index=$((i+1))
    total_models=${#selected_models[@]}

    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}进度: [$model_index/$total_models]${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if train_single_model "$model_name" "$model_index" "$total_models"; then
        successful_models+=("$model_name")
    else
        failed_models+=("$model_name")

        # 询问是否继续
        if [[ $model_index -lt $total_models ]]; then
            echo ""
            echo -n -e "${YELLOW}是否继续训练下一个模型？(y/n): ${NC}"
            read -r continue_training

            if [[ ! "$continue_training" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}训练已中止${NC}"
                break
            fi
        fi
    fi

    # 如果不是最后一个模型，稍作休息
    if [[ $model_index -lt $total_models ]]; then
        echo ""
        echo -e "${CYAN}等待5秒后继续下一个模型...${NC}"
        sleep 5
    fi
done

# =============================================================================
# 训练完成总结
# =============================================================================

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

print_header "训练完成汇总"

echo -e "${GREEN}完成时间: $(date)${NC}"
echo -e "${GREEN}总耗时: ${TOTAL_HOURS}小时${TOTAL_MINUTES}分${TOTAL_SECONDS}秒${NC}"
echo ""

# 成功的模型
if [[ ${#successful_models[@]} -gt 0 ]]; then
    echo -e "${GREEN}✅ 成功训练的模型 (${#successful_models[@]}):${NC}"
    for model in "${successful_models[@]}"; do
        echo -e "   ${GREEN}✓${NC} $model"
    done
    echo ""
fi

# 失败的模型
if [[ ${#failed_models[@]} -gt 0 ]]; then
    echo -e "${RED}❌ 失败的模型 (${#failed_models[@]}):${NC}"
    for model in "${failed_models[@]}"; do
        echo -e "   ${RED}✗${NC} $model"
    done
    echo ""
fi

# 输出目录位置
echo -e "${BLUE}📁 所有训练结果保存在:${NC}"
echo -e "   ${CYAN}$BASE_OUTPUT_DIR/${NC}"
echo ""

# 列出输出目录
echo -e "${BLUE}📊 训练结果目录:${NC}"
for model in "${successful_models[@]}"; do
    local bg_str
    if [[ "$INCLUDE_BACKGROUND" == "true" ]]; then
        bg_str="incl"
    else
        bg_str="excl"
    fi
    local dir="${BASE_OUTPUT_DIR}/${model}_bg_${bg_str}_${MASTER_TIMESTAMP}"

    if [[ -d "$dir" ]]; then
        echo ""
        echo -e "${YELLOW}$model:${NC}"
        echo -e "   目录: ${CYAN}$dir${NC}"
        echo -e "   汇总: ${CYAN}$dir/SUMMARY.txt${NC}"

        # 列出主要文件
        if [[ -d "$dir/results" ]]; then
            local model_file=$(find "$dir/results" -name "*.pth" 2>/dev/null | head -1)
            local softmax_file=$(find "$dir/results" -name "test_softmax_3d_*.nii.gz" 2>/dev/null | head -1)
            local history_file=$(find "$dir/results" -name "training_history_*.png" 2>/dev/null | head -1)
            local analysis_dir=$(find "$dir/results" -name "per_class_analysis_*" -type d 2>/dev/null | head -1)

            [[ -n "$model_file" ]] && echo -e "   模型: ${CYAN}$(basename $model_file)${NC}"
            [[ -n "$softmax_file" ]] && echo -e "   Softmax: ${CYAN}$(basename $softmax_file)${NC}"
            [[ -n "$history_file" ]] && echo -e "   曲线图: ${CYAN}$(basename $history_file)${NC}"
            [[ -n "$analysis_dir" ]] && echo -e "   分析: ${CYAN}$(basename $analysis_dir)${NC}"
        fi
    fi
done

# 创建总汇总报告
MASTER_SUMMARY="${BASE_OUTPUT_DIR}/MASTER_SUMMARY_${MASTER_TIMESTAMP}.txt"
cat > "$MASTER_SUMMARY" << EOF
多模型训练总汇总报告
================================================================================
训练批次: $MASTER_TIMESTAMP
训练时间: $(date)
总耗时: ${TOTAL_HOURS}小时${TOTAL_MINUTES}分${TOTAL_SECONDS}秒

训练配置:
  - Epochs: $EPOCHS
  - Batch Size: $BATCH_SIZE
  - Learning Rate: $LEARNING_RATE
  - Include Background: $INCLUDE_BACKGROUND

训练结果:
  - 总模型数: ${#selected_models[@]}
  - 成功: ${#successful_models[@]}
  - 失败: ${#failed_models[@]}

成功的模型:
EOF

for model in "${successful_models[@]}"; do
    echo "  ✓ $model" >> "$MASTER_SUMMARY"
done

if [[ ${#failed_models[@]} -gt 0 ]]; then
    echo "" >> "$MASTER_SUMMARY"
    echo "失败的模型:" >> "$MASTER_SUMMARY"
    for model in "${failed_models[@]}"; do
        echo "  ✗ $model" >> "$MASTER_SUMMARY"
    done
fi

echo "" >> "$MASTER_SUMMARY"
echo "输出目录结构:" >> "$MASTER_SUMMARY"
tree -L 2 "$BASE_OUTPUT_DIR" >> "$MASTER_SUMMARY" 2>/dev/null || ls -la "$BASE_OUTPUT_DIR" >> "$MASTER_SUMMARY"

echo "" >> "$MASTER_SUMMARY"
echo "生成时间: $(date)" >> "$MASTER_SUMMARY"
echo "================================================================================" >> "$MASTER_SUMMARY"

echo ""
echo -e "${GREEN}✓ 总汇总报告: ${CYAN}$MASTER_SUMMARY${NC}"

# =============================================================================
# 后续建议
# =============================================================================

echo ""
print_section "后续建议"

echo -e "${YELLOW}1. 查看训练曲线对比不同模型的性能${NC}"
echo -e "${YELLOW}2. 查看per-class分析了解每个类别的表现${NC}"
echo -e "${YELLOW}3. 使用生成的3D softmax进行可视化${NC}"
echo -e "${YELLOW}4. 查看各模型的SUMMARY.txt了解详细信息${NC}"
echo ""

print_header "全部完成！祝分析愉快！"

echo ""
