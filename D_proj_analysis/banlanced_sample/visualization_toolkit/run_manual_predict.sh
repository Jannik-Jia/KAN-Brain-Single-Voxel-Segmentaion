#!/bin/bash

echo "🔮 手动softmax预测工具"
echo "=================================================="

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否提供了模型路径参数
if [[ $# -eq 0 ]]; then
    echo -e "${YELLOW}📖 使用方法:${NC}"
    echo "  $0 <model_path> [options]"
    echo ""
    echo -e "${BLUE}参数说明:${NC}"
    echo "  model_path              模型文件路径 (.pth) [必需]"
    echo "  -o, --output DIR       输出目录 (默认: ../results)"
    echo "  -s, --subject PATH     受试者数据路径"
    echo "  -n, --name PREFIX      输出文件名前缀 (默认: current_prediction)"
    echo ""
    echo -e "${BLUE}示例用法:${NC}"
    echo "  # 基本用法"
    echo -e "  ${GREEN}$0 ../results/my_model.pth${NC}"
    echo ""
    echo "  # 指定输出目录和文件名"
    echo -e "  ${GREEN}$0 ../results/my_model.pth -o /path/to/output -n my_prediction${NC}"
    echo ""
    echo "  # 使用不同受试者数据"
    echo -e "  ${GREEN}$0 ../results/my_model.pth -s /path/to/subject/data${NC}"
    echo ""
    echo -e "${BLUE}可用模型文件:${NC}"
    
    # 显示可用的模型文件
    if [[ -d "../results" ]]; then
        echo "  在 ../results/ 中找到:"
        find ../results -name "*.pth" -type f | head -10 | while read -r file; do
            size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "    • $(basename "$file") (${size})"
        done
        
        if [[ $(find ../results -name "*.pth" -type f | wc -l) -gt 10 ]]; then
            echo "    ... 等等"
        fi
    fi
    
    exit 0
fi

# 检查脚本是否存在
if [[ ! -f "manual_predict_softmax.py" ]]; then
    echo -e "${RED}❌ 错误: 找不到 manual_predict_softmax.py${NC}"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

# 检查模型文件是否存在
MODEL_PATH="$1"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${RED}❌ 错误: 模型文件不存在: $MODEL_PATH${NC}"
    exit 1
fi

# 显示运行信息
echo -e "${GREEN}🚀 开始预测...${NC}"
echo "📦 模型文件: $MODEL_PATH"
echo "⏱️  开始时间: $(date)"
echo ""

# 运行预测
python manual_predict_softmax.py "$@"

# 检查运行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ 预测完成！${NC}"
    echo "⏱️  完成时间: $(date)"
    echo ""
    echo -e "${BLUE}📁 检查输出文件:${NC}"
    
    # 显示最新生成的文件
    OUTPUT_DIR="${2:-../results}"
    if [[ "$2" == "-o" ]]; then
        OUTPUT_DIR="$3"
    fi
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "  最新生成的softmax文件:"
        find "$OUTPUT_DIR" -name "*softmax*.nii.gz" -newermt "1 minute ago" 2>/dev/null | head -3 | while read -r file; do
            size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "    • $(basename "$file") (${size})"
        done
    fi
else
    echo ""
    echo -e "${RED}❌ 预测失败${NC}"
    echo "请检查上面的错误信息"
    exit 1
fi