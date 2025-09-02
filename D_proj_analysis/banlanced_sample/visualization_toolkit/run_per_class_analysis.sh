#!/bin/bash

echo "🧮 Per-Class Performance Analysis Tool"
echo "=================================================="

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
if [[ $# -eq 0 ]]; then
    echo -e "${YELLOW}📖 使用方法:${NC}"
    echo "  $0 [options]"
    echo ""
    echo -e "${BLUE}参数说明:${NC}"
    echo "  -s, --softmax FILE     Softmax预测文件 (.nii.gz) [必需]"
    echo "  -l, --labels FILE      标签文件 (.nii.gz) [必需]"
    echo "  -i, --info FILE        Info文件 (.json) [可选]"
    echo "  -o, --output DIR       输出目录 [默认: per_class_analysis_TIMESTAMP]"
    echo "  -h, --help             显示帮助信息"
    echo ""
    echo -e "${BLUE}示例用法:${NC}"
    echo "  # 基本用法"
    echo -e "  ${GREEN}$0 -s softmax.nii.gz -l labels.nii.gz${NC}"
    echo ""
    echo "  # 包含info文件"
    echo -e "  ${GREEN}$0 -s softmax.nii.gz -l labels.nii.gz -i info.json${NC}"
    echo ""
    echo "  # 自定义输出目录"  
    echo -e "  ${GREEN}$0 -s softmax.nii.gz -l labels.nii.gz -o my_analysis${NC}"
    echo ""
    echo -e "${BLUE}🔍 自动查找最新文件的示例:${NC}"
    
    # 显示可用的文件
    RESULTS_DIR="../results"
    if [[ -d "$RESULTS_DIR" ]]; then
        echo ""
        echo -e "${YELLOW}📁 在 $RESULTS_DIR 中找到的文件:${NC}"
        
        echo -e "${BLUE}  Softmax files:${NC}"
        find "$RESULTS_DIR" -name "*softmax*.nii.gz" -type f 2>/dev/null | head -5 | while read -r file; do
            echo "    • $(basename "$file")"
        done
        
        echo -e "${BLUE}  Label files:${NC}"
        find /home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS -name "*labels*.nii.gz" -type f 2>/dev/null | head -3 | while read -r file; do
            echo "    • $file"
        done
    fi
    
    exit 0
fi

# 检查脚本是否存在
if [[ ! -f "per_class_analyzer.py" ]]; then
    echo -e "${RED}❌ 错误: 找不到 per_class_analyzer.py${NC}"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

# 解析参数
SOFTMAX_FILE=""
LABELS_FILE=""
INFO_FILE=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--softmax)
            SOFTMAX_FILE="$2"
            shift 2
            ;;
        -l|--labels)
            LABELS_FILE="$2"
            shift 2
            ;;
        -i|--info)
            INFO_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            exec $0  # 显示帮助
            ;;
        *)
            echo -e "${RED}❌ 未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [[ -z "$SOFTMAX_FILE" ]]; then
    echo -e "${RED}❌ 错误: 需要指定softmax文件 (-s)${NC}"
    exit 1
fi

if [[ -z "$LABELS_FILE" ]]; then
    echo -e "${RED}❌ 错误: 需要指定labels文件 (-l)${NC}"
    exit 1
fi

# 检查文件是否存在
if [[ ! -f "$SOFTMAX_FILE" ]]; then
    echo -e "${RED}❌ 错误: Softmax文件不存在: $SOFTMAX_FILE${NC}"
    exit 1
fi

if [[ ! -f "$LABELS_FILE" ]]; then
    echo -e "${RED}❌ 错误: Labels文件不存在: $LABELS_FILE${NC}"
    exit 1
fi

if [[ -n "$INFO_FILE" && ! -f "$INFO_FILE" ]]; then
    echo -e "${RED}❌ 错误: Info文件不存在: $INFO_FILE${NC}"
    exit 1
fi

# 设置默认输出目录
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="per_class_analysis_$(date +%Y%m%d_%H%M%S)"
fi

# 显示运行信息
echo -e "${GREEN}🚀 开始Per-Class性能分析...${NC}"
echo "📊 分析设置:"
echo "  Softmax文件: $(basename "$SOFTMAX_FILE")"
echo "  Labels文件:  $(basename "$LABELS_FILE")"
if [[ -n "$INFO_FILE" ]]; then
    echo "  Info文件:    $(basename "$INFO_FILE")"
fi
echo "  输出目录:    $OUTPUT_DIR"
echo "  开始时间:    $(date)"
echo ""

# 构建命令
CMD="python per_class_analyzer.py -s \"$SOFTMAX_FILE\" -l \"$LABELS_FILE\" -o \"$OUTPUT_DIR\""
if [[ -n "$INFO_FILE" ]]; then
    CMD="$CMD -i \"$INFO_FILE\""
fi

# 运行分析
echo -e "${BLUE}🔧 执行命令: $CMD${NC}"
eval $CMD

# 检查运行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ Per-Class分析完成！${NC}"
    echo "⏱️  完成时间: $(date)"
    echo ""
    echo -e "${BLUE}📁 生成的文件:${NC}"
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "  📊 可视化图表:"
        find "$OUTPUT_DIR" -name "*.png" -type f | while read -r file; do
            echo "    • $(basename "$file")"
        done
        
        echo "  📋 数据报告:"
        find "$OUTPUT_DIR" -name "*.csv" -o -name "*.json" -o -name "*.txt" | while read -r file; do
            echo "    • $(basename "$file")"
        done
        
        echo ""
        echo -e "${YELLOW}💡 建议查看的文件:${NC}"
        echo "  1. ${OUTPUT_DIR}/comprehensive_per_class_analysis.png - 综合性能分析图"
        echo "  2. ${OUTPUT_DIR}/per_class_summary_report.txt - 详细文字报告"
        echo "  3. ${OUTPUT_DIR}/per_class_detailed_metrics.csv - 详细数据表格"
    fi
    
    echo ""
    echo -e "${BLUE}🎯 关键指标说明:${NC}"
    echo "  • F1 Score: 精确率和召回率的调和平均，综合性能指标"
    echo "  • Dice Coefficient: 医学图像分割的标准指标"
    echo "  • Precision: 预测为该类的准确率"
    echo "  • Recall: 真实该类的检出率"
    echo "  • Support: 该类别的真实样本数量"
    echo "  • IoU: 交并比，另一个重要的分割指标"
    
else
    echo ""
    echo -e "${RED}❌ 分析失败${NC}"
    echo "请检查上面的错误信息"
    exit 1
fi