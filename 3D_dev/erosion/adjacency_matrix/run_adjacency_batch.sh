#!/bin/bash

# 批量计算3D脑区邻接矩阵脚本
# 使用方法: bash run_adjacency_batch.sh

# 设置错误时退出
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  3D脑区邻接矩阵批量计算脚本${NC}"
echo -e "${BLUE}======================================${NC}"

# 配置参数
# 🔧 修改这些路径以匹配你的数据位置
DATA_DIR="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated"
OUTPUT_DIR="./results"
CONNECTIVITY=6  # 6, 18, 或 26连通性
STANDARD_MATRIX_SIZE=102  # 标准化矩阵大小 (默认102，对应标签0-101)
START_SUBJECT=1
END_SUBJECT=38

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_DIR${NC}"
    echo -e "${YELLOW}请修改脚本中的DATA_DIR变量${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"

# 检查Python脚本是否存在
SCRIPT_PATH="./compute_adjacency_matrices.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}错误: 计算脚本不存在: $SCRIPT_PATH${NC}"
    exit 1
fi

# 显示配置
echo -e "${BLUE}配置信息:${NC}"
echo -e "  数据目录: ${DATA_DIR}"
echo -e "  输出目录: ${OUTPUT_DIR}"
echo -e "  连通性: ${CONNECTIVITY}"
echo -e "  标准矩阵大小: ${STANDARD_MATRIX_SIZE}×${STANDARD_MATRIX_SIZE}"
echo -e "  被试范围: ${START_SUBJECT}-${END_SUBJECT}"
echo

# 询问是否继续
read -p "是否开始处理? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}用户取消操作${NC}"
    exit 0
fi

# 1. 首先测试单个被试
echo -e "${YELLOW}步骤1: 测试单个被试...${NC}"
python "$SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --connectivity "$CONNECTIVITY" \
    --standard_matrix_size "$STANDARD_MATRIX_SIZE" \
    --test_only \
    --verbose

if [ $? -ne 0 ]; then
    echo -e "${RED}测试失败，请检查数据和脚本${NC}"
    exit 1
fi

echo -e "${GREEN}测试成功！${NC}"
echo

# 询问是否继续处理所有被试
read -p "测试成功，是否处理所有被试? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}只完成测试，退出处理${NC}"
    exit 0
fi

# 2. 处理所有被试
echo -e "${YELLOW}步骤2: 处理所有被试 (${START_SUBJECT}-${END_SUBJECT})...${NC}"
echo -e "${BLUE}这可能需要几分钟到几小时，取决于数据大小${NC}"

# 记录开始时间
START_TIME=$(date +%s)

python "$SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --connectivity "$CONNECTIVITY" \
    --standard_matrix_size "$STANDARD_MATRIX_SIZE" \
    --start_subject "$START_SUBJECT" \
    --end_subject "$END_SUBJECT" \
    --verbose

PYTHON_EXIT_CODE=$?

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo
echo -e "${BLUE}======================================${NC}"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 批量处理完成！${NC}"
    echo -e "总耗时: ${DURATION}秒 ($(($DURATION / 60))分钟)"

    # 统计输出文件
    ADJACENCY_FILES=$(find "$OUTPUT_DIR" -name "*_adjacency_*.h5" | wc -l)
    LOG_FILES=$(find "$OUTPUT_DIR" -name "*.log" | wc -l)
    REPORT_FILES=$(find "$OUTPUT_DIR" -name "*_report_*.json" | wc -l)

    echo -e "${GREEN}生成文件统计:${NC}"
    echo -e "  邻接矩阵文件: ${ADJACENCY_FILES}个"
    echo -e "  日志文件: ${LOG_FILES}个"
    echo -e "  报告文件: ${REPORT_FILES}个"
    echo

    echo -e "${BLUE}下一步建议:${NC}"
    echo -e "1. 检查日志文件了解处理详情"
    echo -e "2. 使用adjacency_analysis_utils.py分析结果"
    echo -e "3. 与混淆矩阵进行Hadamard乘积分析"

else
    echo -e "${RED}❌ 批量处理失败 (退出码: $PYTHON_EXIT_CODE)${NC}"
    echo -e "${YELLOW}请检查日志文件了解错误详情${NC}"
fi

echo -e "${BLUE}======================================${NC}"

# 显示输出目录内容
echo -e "${BLUE}输出目录内容:${NC}"
ls -la "$OUTPUT_DIR" | head -10

if [ $(ls -1 "$OUTPUT_DIR" | wc -l) -gt 10 ]; then
    echo "... (显示前10个文件)"
fi

exit $PYTHON_EXIT_CODE