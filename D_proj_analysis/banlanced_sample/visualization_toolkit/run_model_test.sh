#!/bin/bash

echo "🚀 运行模型预测验证测试"
echo "=================================================="

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [[ ! -f "test_model_prediction.py" ]]; then
    echo -e "${RED}❌ 错误: 请在包含test_model_prediction.py的目录下运行此脚本${NC}"
    exit 1
fi

# 显示脚本信息
echo -e "${BLUE}📍 脚本功能:${NC}"
echo "  • 加载训练好的模型"  
echo "  • 进行预测并生成softmax概率"
echo "  • 检测保存前后数据的正确性"
echo "  • 验证数据映射是否正确"
echo ""

# 运行测试
echo -e "${GREEN}🔍 开始运行测试...${NC}"
python test_model_prediction.py

# 检查运行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ 测试完成！${NC}"
    echo ""
    echo -e "${BLUE}📊 检查生成的文件:${NC}"
    ls -la ../../results/verification_test_softmax*
    echo ""
    echo -e "${BLUE}💡 下一步建议:${NC}"
    echo "  1. 检查生成的verification文件中是否有非背景预测"
    echo "  2. 如果验证成功，可以修复原始的predictions_to_3d_volume函数"
    echo "  3. 重新训练或重新生成原始的softmax文件"
else
    echo ""
    echo -e "${RED}❌ 测试失败，请检查错误信息${NC}"
fi