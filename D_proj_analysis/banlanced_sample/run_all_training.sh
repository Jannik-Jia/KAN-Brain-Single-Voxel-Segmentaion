#!/bin/bash

# =============================================================================
# Balanced MRI数据全模式训练脚本
# 自动运行排除背景和包含背景两种训练模式
# =============================================================================

set -e  # 遇到错误立即退出

# 配置参数
SCRIPT_NAME="train_with_3d_prediction_save.py"
LOG_DIR="./logs"
EPOCHS=25
BATCH_SIZE=8192
LEARNING_RATE=0.00001

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p $LOG_DIR

# 打印开始信息
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}                    Balanced MRI 数据全模式训练脚本${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}开始时间: $(date)${NC}"
echo -e "${GREEN}训练轮数: $EPOCHS${NC}"
echo -e "${GREEN}批大小: $BATCH_SIZE${NC}"
echo -e "${GREEN}学习率: $LEARNING_RATE${NC}"
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
# 训练模式1: 排除背景 (推荐模式)
# =============================================================================

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}🚀 开始训练模式1: 排除背景 (推荐)${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${YELLOW}特点:${NC}"
echo -e "${YELLOW}  • 仅训练非背景体素 (~45万个)${NC}"
echo -e "${YELLOW}  • 训练速度快，内存需求小${NC}"
echo -e "${YELLOW}  • 适合快速验证和主要分析${NC}"
echo ""

# 记录开始时间
mode1_start=$(date +%s)

echo -e "${GREEN}执行命令:${NC} python $SCRIPT_NAME --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LEARNING_RATE"
echo ""

# 运行排除背景模式
if python "$SCRIPT_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --exclude_features 14 \
    2>&1 | tee "$LOG_DIR/training_bg_excl_$(date +%Y%m%d_%H%M%S).log"; then
    
    mode1_end=$(date +%s)
    mode1_duration=$((mode1_end - mode1_start))
    echo -e "${GREEN}✅ 模式1 (排除背景) 训练完成!${NC}"
    echo -e "${GREEN}耗时: $(($mode1_duration / 60))分$(($mode1_duration % 60))秒${NC}"
    
    echo ""
    echo -e "${BLUE}✅ 模式1训练和Per-Class分析完成 (已集成在训练脚本中)${NC}"
    
else
    echo -e "${RED}❌ 模式1 (排除背景) 训练失败!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}⏳ 等待5秒后开始下一个模式...${NC}"
sleep 5
echo ""

# =============================================================================
# 训练模式2: 包含背景 (对比模式)
# =============================================================================

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}🚀 开始训练模式2: 包含背景 (对比)${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${YELLOW}特点:${NC}"
echo -e "${YELLOW}  • 训练所有体素 (~3700万个)${NC}"
echo -e "${YELLOW}  • 训练时间长，完整数据${NC}"
echo -e "${YELLOW}  • 适合全面对比分析${NC}"
echo ""

# 记录开始时间
mode2_start=$(date +%s)

echo -e "${GREEN}执行命令:${NC} python $SCRIPT_NAME --include_background --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LEARNING_RATE"
echo ""

# 运行包含背景模式
if python "$SCRIPT_NAME" \
    --include_background \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --exclude_features 14 \
    2>&1 | tee "$LOG_DIR/training_bg_incl_$(date +%Y%m%d_%H%M%S).log"; then
    
    mode2_end=$(date +%s)
    mode2_duration=$((mode2_end - mode2_start))
    echo -e "${GREEN}✅ 模式2 (包含背景) 训练完成!${NC}"
    echo -e "${GREEN}耗时: $(($mode2_duration / 60))分$(($mode2_duration % 60))秒${NC}"
    
    echo ""
    echo -e "${BLUE}✅ 模式2训练和Per-Class分析完成 (已集成在训练脚本中)${NC}"
    
else
    echo -e "${RED}❌ 模式2 (包含背景) 训练失败!${NC}"
    exit 1
fi

# =============================================================================
# 训练完成汇总
# =============================================================================

total_end=$(date +%s)
total_start=$(date +%s)
if [[ -n "$mode1_start" ]]; then
    total_start=$mode1_start
fi
total_duration=$((total_end - total_start))

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}🎉 所有训练模式完成!${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}完成时间: $(date)${NC}"
echo -e "${GREEN}总耗时: $(($total_duration / 3600))小时$(($total_duration % 3600 / 60))分$(($total_duration % 60))秒${NC}"
echo ""

echo -e "${BLUE}📊 训练结果汇总:${NC}"
echo -e "${YELLOW}模式1 (排除背景):${NC}"
echo -e "${YELLOW}  耗时: $(($mode1_duration / 60))分$(($mode1_duration % 60))秒${NC}"

if [[ -n "$mode2_duration" ]]; then
    echo -e "${YELLOW}模式2 (包含背景):${NC}"
    echo -e "${YELLOW}  耗时: $(($mode2_duration / 60))分$(($mode2_duration % 60))秒${NC}"
    
    if [[ $mode2_duration -gt 0 && $mode1_duration -gt 0 ]]; then
        speed_ratio=$((mode2_duration / mode1_duration))
        echo -e "${YELLOW}速度对比: 模式2比模式1慢 ${speed_ratio}x${NC}"
    fi
fi

echo ""
echo -e "${BLUE}📁 生成的文件:${NC}"
echo -e "${YELLOW}训练日志:${NC}"
ls -la $LOG_DIR/*.log 2>/dev/null | tail -10 || echo "  (没有找到日志文件)"

echo ""
echo -e "${YELLOW}模型和结果文件:${NC}"
echo -e "${YELLOW}  排除背景模式:${NC}"
ls -la *bg_excl_*.* 2>/dev/null | head -5 || echo "    (没有找到bg_excl文件)"

echo -e "${YELLOW}  包含背景模式:${NC}" 
ls -la *bg_incl_*.* 2>/dev/null | head -5 || echo "    (没有找到bg_incl文件)"

echo ""
echo -e "${BLUE}💡 后续建议:${NC}"
echo -e "${YELLOW}1. 检查训练历史图表，观察训练效果${NC}"
echo -e "${YELLOW}2. 对比两种模式的性能指标${NC}"
echo -e "${YELLOW}3. 使用生成的3D softmax进行可视化分析${NC}"
echo -e "${YELLOW}4. 根据结果调整超参数进行优化${NC}"

echo ""
echo -e "${GREEN}🎯 主要输出文件说明:${NC}"
echo -e "${GREEN}  • *_softmax_3d_*.nii.gz  : 3D softmax概率图 (用于可视化)${NC}"
echo -e "${GREEN}  • balanced_3d_*.pth      : 训练好的模型权重${NC}"
echo -e "${GREEN}  • training_history_*.png : 训练曲线图${NC}"
echo -e "${GREEN}  • *.log                  : 详细训练日志${NC}"

# =============================================================================
# 最终对比总结
# =============================================================================

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}📊 两种模式Per-Class分析总结${NC}"
echo -e "${BLUE}===============================================================================${NC}"

echo -e "${GREEN}✅ 训练和分析完成！两种模式的详细per-class量化对比已生成${NC}"
echo ""
echo -e "${YELLOW}📁 Per-Class分析结果位置:${NC}"
echo -e "${YELLOW}  • ./results/per_class_analysis_bg_excl_*/ - 排除背景模式详细分析${NC}"
echo -e "${YELLOW}  • ./results/per_class_analysis_bg_incl_*/ - 包含背景模式详细分析${NC}"
echo ""
echo -e "${YELLOW}🎯 每个分析目录包含:${NC}"
echo -e "${YELLOW}  • comprehensive_per_class_analysis.png - 完整的逐class F1/Dice/Precision/Recall对比${NC}"
echo -e "${YELLOW}  • detailed_performance_ranking_analysis.png - 性能排名和分布分析${NC}"
echo -e "${YELLOW}  • detailed_correlation_analysis.png - 各指标相关性分析${NC}"
echo -e "${YELLOW}  • per_class_detailed_metrics.csv - 每个class的详细量化数据${NC}"
echo -e "${YELLOW}  • per_class_summary_report.txt - 人类友好的分析报告${NC}"
echo ""
echo -e "${BLUE}💡 关键改进:${NC}"
echo -e "${YELLOW}  ✅ Per-class分析直接集成在训练脚本中，无需重新加载模型${NC}"
echo -e "${YELLOW}  ✅ 训练完成后立即进行分析，避免模型加载错误${NC}"
echo -e "${YELLOW}  ✅ 每个训练模式完成后立即看到详细的F1/Precision/Recall等指标对比${NC}"

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}✨ 全部训练和分析完成! 祝你分析愉快! ✨${NC}"
echo -e "${BLUE}===============================================================================${NC}"

echo ""
echo -e "${BLUE}📊 完整输出文件清单:${NC}"
echo -e "${GREEN}训练结果:${NC}"
echo -e "${GREEN}  • *_softmax_3d_*.nii.gz    : 3D softmax概率图${NC}"
echo -e "${GREEN}  • balanced_3d_*.pth        : 训练好的模型权重${NC}"
echo -e "${GREEN}  • training_history_*.png   : 训练曲线图${NC}"
echo ""
echo -e "${GREEN}Per-Class分析结果:${NC}"
echo -e "${GREEN}  • per_class_analysis_*/    : 详细的per-class可视化分析${NC}"
echo -e "${GREEN}  • per_class_comparison_*.txt : 对比分析报告${NC}"
echo ""
echo -e "${BLUE}💡 建议的分析流程:${NC}"
echo -e "${YELLOW}1. 查看训练曲线图确认训练效果${NC}"
echo -e "${YELLOW}2. 打开per_class分析目录查看详细可视化${NC}"
echo -e "${YELLOW}3. 重点关注comprehensive_per_class_analysis.png${NC}"
echo -e "${YELLOW}4. 对比两种训练模式的per-class性能差异${NC}"
echo -e "${YELLOW}5. 根据分析结果优化模型或数据处理策略${NC}"