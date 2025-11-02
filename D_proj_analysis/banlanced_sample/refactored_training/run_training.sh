#!/bin/bash

# =============================================================================
# 重构版训练脚本 - 支持多种模型
# Refactored Training Script - Support Multiple Models
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================

# 模型选择 (可选: reg_model, resnet_mlp, simple_mlp)
MODEL_NAME="${MODEL_NAME:-reg_model}"

# 训练参数
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"

# 数据参数
ROOT_DIR="${ROOT_DIR:-/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS}"
INCLUDE_BACKGROUND="${INCLUDE_BACKGROUND:-false}"  # true 或 false

# 脚本和目录
SCRIPT_NAME="train.py"
LOG_DIR="./logs"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# 初始化
# =============================================================================

# 创建日志目录
mkdir -p $LOG_DIR

# 打印开始信息
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}                    重构版MRI体素分割训练脚本${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}开始时间: $(date)${NC}"
echo -e "${CYAN}模型: $MODEL_NAME${NC}"
echo -e "${GREEN}训练轮数: $EPOCHS${NC}"
echo -e "${GREEN}批大小: $BATCH_SIZE${NC}"
echo -e "${GREEN}学习率: $LEARNING_RATE${NC}"
echo -e "${GREEN}包含背景: $INCLUDE_BACKGROUND${NC}"
echo ""

# 检查脚本是否存在
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo -e "${RED}❌ 错误: 找不到训练脚本 $SCRIPT_NAME${NC}"
    echo -e "${YELLOW}请确保脚本在当前目录下${NC}"
    exit 1
fi

# 检查Python环境
echo -e "${BLUE}🔍 检查运行环境...${NC}"
python --version
if ! python -c "import torch, nibabel, sklearn"; then
    echo -e "${RED}❌ 错误: 缺少必要的Python包${NC}"
    echo -e "${YELLOW}请安装: torch, nibabel, scikit-learn${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python环境检查通过${NC}"
echo ""

# =============================================================================
# 训练函数
# =============================================================================

run_training() {
    local model=$1
    local bg_mode=$2
    local mode_name=$3

    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}🚀 开始训练: ${model} - ${mode_name}${NC}"
    echo -e "${BLUE}===============================================================================${NC}"

    # 记录开始时间
    local start_time=$(date +%s)

    # 构建命令
    local cmd="python $SCRIPT_NAME --model $model --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LEARNING_RATE"

    # 添加背景参数
    if [[ "$bg_mode" == "true" ]]; then
        cmd="$cmd --include_background"
    fi

    # 添加排除特征
    cmd="$cmd --exclude_features 14"

    # 添加数据根目录
    cmd="$cmd --root_dir $ROOT_DIR"

    echo -e "${GREEN}执行命令:${NC} $cmd"
    echo ""

    # 运行训练
    local log_suffix
    if [[ "$bg_mode" == "true" ]]; then
        log_suffix="incl"
    else
        log_suffix="excl"
    fi

    if eval "$cmd" 2>&1 | tee "$LOG_DIR/training_${model}_bg_${log_suffix}_$(date +%Y%m%d_%H%M%S).log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}✅ 训练完成!${NC}"
        echo -e "${GREEN}耗时: $(($duration / 60))分$(($duration % 60))秒${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}❌ 训练失败!${NC}"
        return 1
    fi
}

# =============================================================================
# 主训练流程
# =============================================================================

echo -e "${CYAN}📊 可用模型列表:${NC}"
echo -e "${YELLOW}  • reg_model    - 深度全连接神经网络（Alex identical structure）${NC}"
echo -e "${YELLOW}  • resnet_mlp   - 带残差连接的全连接神经网络${NC}"
echo -e "${YELLOW}  • simple_mlp   - 简单多层感知机（轻量级）${NC}"
echo -e "${YELLOW}  • kan          - KAN模型（自动使用类权重）⭐${NC}"
echo ""

# 验证模型名称
if [[ ! "$MODEL_NAME" =~ ^(reg_model|resnet_mlp|simple_mlp|kan)$ ]]; then
    echo -e "${RED}❌ 错误: 无效的模型名称 '$MODEL_NAME'${NC}"
    echo -e "${YELLOW}请使用: reg_model, resnet_mlp, simple_mlp, 或 kan${NC}"
    exit 1
fi

# 根据INCLUDE_BACKGROUND决定运行模式
if [[ "$INCLUDE_BACKGROUND" == "true" ]]; then
    # 只运行包含背景模式
    run_training "$MODEL_NAME" "true" "包含背景"
    exit_code=$?
elif [[ "$INCLUDE_BACKGROUND" == "false" ]]; then
    # 只运行排除背景模式
    run_training "$MODEL_NAME" "false" "排除背景 (推荐)"
    exit_code=$?
else
    echo -e "${RED}❌ 错误: INCLUDE_BACKGROUND 必须是 'true' 或 'false'${NC}"
    exit 1
fi

# =============================================================================
# 训练完成汇总
# =============================================================================

if [[ $exit_code -eq 0 ]]; then
    echo ""
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}🎉 训练完成!${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${GREEN}完成时间: $(date)${NC}"
    echo ""

    echo -e "${BLUE}📁 生成的文件:${NC}"
    echo -e "${YELLOW}训练日志:${NC}"
    ls -lh $LOG_DIR/*.log 2>/dev/null | tail -5 || echo "  (没有找到日志文件)"

    echo ""
    echo -e "${YELLOW}模型和结果文件 (在 ./results/ 目录):${NC}"
    ls -lh ./results/${MODEL_NAME}* 2>/dev/null | head -10 || echo "  (没有找到结果文件)"

    echo ""
    echo -e "${BLUE}💡 后续建议:${NC}"
    echo -e "${YELLOW}1. 检查 results/ 目录中的训练历史图表${NC}"
    echo -e "${YELLOW}2. 查看 per_class_analysis_*/ 目录中的详细分析${NC}"
    echo -e "${YELLOW}3. 使用生成的3D softmax进行可视化${NC}"
    echo -e "${YELLOW}4. 根据结果尝试不同的模型或参数${NC}"

    echo ""
    echo -e "${GREEN}🎯 主要输出文件:${NC}"
    echo -e "${GREEN}  • {model}_bg_{excl/incl}_*.pth        - 训练好的模型权重${NC}"
    echo -e "${GREEN}  • training_history_*.png              - 训练曲线图${NC}"
    echo -e "${GREEN}  • test_softmax_3d_*.nii.gz            - 3D softmax概率图${NC}"
    echo -e "${GREEN}  • per_class_analysis_*/               - 详细的per-class分析${NC}"
else
    echo -e "${RED}❌ 训练过程中出现错误${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}✨ 全部完成! ✨${NC}"
echo -e "${BLUE}===============================================================================${NC}"
