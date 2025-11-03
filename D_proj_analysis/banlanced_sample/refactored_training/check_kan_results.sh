#!/bin/bash

# =============================================================================
# KAN训练结果检查和恢复脚本
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "================================================================================"
echo "KAN训练结果检查"
echo "================================================================================"
echo ""

# 查找KAN训练目录
echo -e "${CYAN}1. 查找KAN训练目录...${NC}"
kan_dirs=$(find ./training_runs -name "kan_bg_*" -type d 2>/dev/null)

if [[ -z "$kan_dirs" ]]; then
    echo -e "${YELLOW}未找到KAN训练目录${NC}"
else
    echo -e "${GREEN}找到以下KAN训练目录:${NC}"
    echo "$kan_dirs"
fi

echo ""

# 查找results目录
echo -e "${CYAN}2. 查找所有results目录...${NC}"
results_dirs=$(find . -name "results" -type d 2>/dev/null | grep -v ".git")

echo -e "${GREEN}找到以下results目录:${NC}"
for dir in $results_dirs; do
    echo "  - $dir"
done

echo ""

# 查找KAN相关文件
echo -e "${CYAN}3. 查找KAN相关文件...${NC}"

echo -e "${YELLOW}模型文件 (.pth):${NC}"
find . -name "kan_bg_*.pth" -type f 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  ✓ $file ($size)"
done

echo ""

echo -e "${YELLOW}3D Softmax文件 (.nii.gz):${NC}"
find . -name "*kan*.nii.gz" -type f 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  ✓ $file ($size)"
done

echo ""

echo -e "${YELLOW}训练曲线 (.png):${NC}"
find . -name "training_history_kan*.png" -type f 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  ✓ $file ($size)"
done

echo ""

# 检查特定的KAN训练目录
echo -e "${CYAN}4. 检查KAN训练目录内容...${NC}"

for kan_dir in $kan_dirs; do
    echo ""
    echo -e "${BLUE}目录: $kan_dir${NC}"
    echo "----------------------------------------"

    # 检查results子目录
    results_path="${kan_dir}/results"
    if [[ -d "$results_path" ]]; then
        echo -e "${GREEN}results目录存在${NC}"

        # 统计文件
        pth_count=$(find "$results_path" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')
        nii_count=$(find "$results_path" -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
        png_count=$(find "$results_path" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        analysis_count=$(find "$results_path" -name "per_class_analysis_*" -type d 2>/dev/null | wc -l | tr -d ' ')

        echo "  模型文件: $pth_count"
        echo "  3D Softmax: $nii_count"
        echo "  训练曲线: $png_count"
        echo "  Per-class分析: $analysis_count"

        # 列出文件
        if [[ $pth_count -gt 0 ]] || [[ $nii_count -gt 0 ]] || [[ $png_count -gt 0 ]]; then
            echo ""
            echo "  文件列表:"
            ls -lh "$results_path" | grep -v "^total" | grep -v "^d"
        fi
    else
        echo -e "${RED}results目录不存在${NC}"
    fi

    # 检查日志
    logs_path="${kan_dir}/logs"
    if [[ -d "$logs_path" ]]; then
        echo ""
        echo -e "${GREEN}logs目录存在${NC}"
        log_count=$(find "$logs_path" -name "*.log" 2>/dev/null | wc -l | tr -d ' ')
        echo "  日志文件: $log_count"

        # 显示最新的日志文件
        latest_log=$(find "$logs_path" -name "*.log" 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            echo "  最新日志: $latest_log"
            echo ""
            echo "  最后50行:"
            tail -50 "$latest_log" | head -20
            echo "  ..."
        fi
    fi
done

echo ""
echo ""

# 查找可能在错误位置的文件
echo -e "${CYAN}5. 查找可能需要移动的文件...${NC}"

orphan_files=false

# 在当前results目录查找
if [[ -d "./results" ]]; then
    kan_files=$(find ./results -name "*kan*" 2>/dev/null)
    if [[ -n "$kan_files" ]]; then
        echo -e "${YELLOW}在 ./results 目录中找到KAN文件:${NC}"
        echo "$kan_files"
        orphan_files=true
    fi
fi

if ! $orphan_files; then
    echo -e "${GREEN}所有文件似乎都在正确位置${NC}"
fi

echo ""
echo ""

# 提供建议
echo "================================================================================"
echo "总结和建议"
echo "================================================================================"
echo ""

if [[ -z "$kan_dirs" ]]; then
    echo -e "${YELLOW}建议: 运行KAN训练${NC}"
    echo "  EPOCHS=25 ./run_multiple_models.sh"
    echo "  # 选择: 4 (kan)"
elif [[ $orphan_files == true ]]; then
    echo -e "${YELLOW}建议: 手动移动孤立文件到正确位置${NC}"
    echo "  详见 MULTI_MODEL_FIX.md 中的恢复步骤"
else
    echo -e "${GREEN}✓ KAN训练结果看起来正常${NC}"
    echo ""
    echo "如果要重新训练:"
    echo "  EPOCHS=5 ./run_multiple_models.sh  # 快速测试"
    echo "  EPOCHS=25 ./run_multiple_models.sh  # 完整训练"
fi

echo ""
echo "完成!"
echo ""
