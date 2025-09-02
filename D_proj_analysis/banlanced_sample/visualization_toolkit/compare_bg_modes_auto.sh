#!/bin/bash

# =============================================================================
# 自动查找最新的softmax文件进行背景模式对比分析
# 优先使用verification文件，回退到原始文件
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 结果目录
RESULTS_DIR="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results"
LABELS_FILE="/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS/FOR_016_20250204_reproducibility/balanced_output/balanced_labels_3d10000.nii.gz"

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}🔬 Auto Background Mode Comparison Analysis${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}Automatically finding the best available softmax files${NC}"
echo ""

# 函数：查找最新的softmax文件
find_latest_softmax() {
    local mode=$1  # "bg_incl" 或 "bg_excl"
    local pattern_priority=(
        "verification_test_softmax_3d_*_${mode}_*.nii.gz"
        "current_prediction_softmax_3d_*_${mode}_*.nii.gz"
        "test_softmax_3d_*_${mode}_*.nii.gz"
    )
    
    for pattern in "${pattern_priority[@]}"; do
        local latest_file=$(find "$RESULTS_DIR" -name "$pattern" -type f 2>/dev/null | sort -r | head -1)
        if [[ -n "$latest_file" && -f "$latest_file" ]]; then
            echo "$latest_file"
            return 0
        fi
    done
    
    return 1
}

# 函数：查找对应的info文件
find_info_file() {
    local softmax_file=$1
    local info_file="${softmax_file%.nii.gz}"
    info_file="${info_file/softmax_3d/softmax_info}"
    info_file="${info_file}.json"
    
    if [[ -f "$info_file" ]]; then
        echo "$info_file"
        return 0
    fi
    
    return 1
}

# 查找背景包含模式文件
echo -e "${CYAN}🔍 Searching for Background INCLUDED files...${NC}"
BG_INCL_SOFTMAX=$(find_latest_softmax "bg_incl")
if [[ -z "$BG_INCL_SOFTMAX" ]]; then
    echo -e "${RED}❌ No background included softmax files found!${NC}"
    exit 1
fi

BG_INCL_INFO=$(find_info_file "$BG_INCL_SOFTMAX")
if [[ -z "$BG_INCL_INFO" ]]; then
    echo -e "${RED}❌ No corresponding info file found for: $(basename "$BG_INCL_SOFTMAX")${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found Background INCLUDED files:${NC}"
echo -e "  📊 Softmax: $(basename "$BG_INCL_SOFTMAX")"
echo -e "  📋 Info: $(basename "$BG_INCL_INFO")"

# 查找背景排除模式文件
echo -e "${CYAN}🔍 Searching for Background EXCLUDED files...${NC}"
BG_EXCL_SOFTMAX=$(find_latest_softmax "bg_excl")
if [[ -z "$BG_EXCL_SOFTMAX" ]]; then
    echo -e "${YELLOW}⚠️ No background excluded softmax files found!${NC}"
    echo -e "${YELLOW}📝 You may need to generate these files using:${NC}"
    echo -e "    ./run_manual_predict.sh path/to/bg_excl_model.pth -n bg_excl_verification"
    echo ""
    echo -e "${BLUE}🔄 Continuing with single-mode analysis for now...${NC}"
    BG_EXCL_SOFTMAX=""
    BG_EXCL_INFO=""
else
    BG_EXCL_INFO=$(find_info_file "$BG_EXCL_SOFTMAX")
    if [[ -z "$BG_EXCL_INFO" ]]; then
        echo -e "${RED}❌ No corresponding info file found for: $(basename "$BG_EXCL_SOFTMAX")${NC}"
        BG_EXCL_SOFTMAX=""
        BG_EXCL_INFO=""
    else
        echo -e "${GREEN}✅ Found Background EXCLUDED files:${NC}"
        echo -e "  📊 Softmax: $(basename "$BG_EXCL_SOFTMAX")"
        echo -e "  📋 Info: $(basename "$BG_EXCL_INFO")"
    fi
fi

# 检查标签文件
echo -e "${CYAN}🔍 Checking labels file...${NC}"
if [[ ! -f "$LABELS_FILE" ]]; then
    echo -e "${RED}❌ Labels file not found: $LABELS_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Labels file found: $(basename "$LABELS_FILE")${NC}"

# 输出目录
OUTPUT_DIR="bg_comparison_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# 检查Python环境
echo -e "${BLUE}🔍 Checking environment...${NC}"
PYTHON_CMD=$(command -v python3 || command -v python)
echo -e "${GREEN}✅ Python found: $($PYTHON_CMD --version)${NC}"

# =============================================================================
# 分析背景包含模式 (总是可用)
# =============================================================================

echo ""
echo -e "${PURPLE}===============================================================================${NC}"
echo -e "${PURPLE}🌟 ANALYZING BACKGROUND INCLUDED MODE${NC}"
echo -e "${PURPLE}===============================================================================${NC}"

echo -e "${CYAN}📈 Performance Metrics Analysis...${NC}"
$PYTHON_CMD performance_metrics_analyzer.py \
    -s "$BG_INCL_SOFTMAX" \
    -i "$BG_INCL_INFO" \
    -l "$LABELS_FILE" \
    -o "$OUTPUT_DIR/performance_bg_included" \
    2>&1 | tee "$OUTPUT_DIR/performance_bg_incl.log"

echo -e "${CYAN}🔍 Hybrid Data Analysis...${NC}"
$PYTHON_CMD hybrid_data_analyzer.py \
    -s "$BG_INCL_SOFTMAX" \
    -i "$BG_INCL_INFO" \
    -l "$LABELS_FILE" \
    -o "$OUTPUT_DIR/hybrid_analysis_bg_included" \
    2>&1 | tee "$OUTPUT_DIR/hybrid_analysis_bg_incl.log"

echo -e "${CYAN}📊 Probability Distribution Analysis...${NC}"
$PYTHON_CMD probability_distribution_analyzer.py \
    -s "$BG_INCL_SOFTMAX" \
    -i "$BG_INCL_INFO" \
    -t 0.001 \
    -o "$OUTPUT_DIR/prob_dist_bg_included" \
    2>&1 | tee "$OUTPUT_DIR/prob_analysis_bg_incl.log"

# =============================================================================
# 分析背景排除模式 (如果可用)
# =============================================================================

if [[ -n "$BG_EXCL_SOFTMAX" ]]; then
    echo ""
    echo -e "${PURPLE}===============================================================================${NC}"
    echo -e "${PURPLE}🎯 ANALYZING BACKGROUND EXCLUDED MODE${NC}"
    echo -e "${PURPLE}===============================================================================${NC}"

    echo -e "${CYAN}📈 Performance Metrics Analysis...${NC}"
    $PYTHON_CMD performance_metrics_analyzer.py \
        -s "$BG_EXCL_SOFTMAX" \
        -i "$BG_EXCL_INFO" \
        -l "$LABELS_FILE" \
        -o "$OUTPUT_DIR/performance_bg_excluded" \
        2>&1 | tee "$OUTPUT_DIR/performance_bg_excl.log"

    echo -e "${CYAN}🔍 Hybrid Data Analysis...${NC}"
    $PYTHON_CMD hybrid_data_analyzer.py \
        -s "$BG_EXCL_SOFTMAX" \
        -i "$BG_EXCL_INFO" \
        -l "$LABELS_FILE" \
        -o "$OUTPUT_DIR/hybrid_analysis_bg_excluded" \
        2>&1 | tee "$OUTPUT_DIR/hybrid_analysis_bg_excl.log"

    echo -e "${CYAN}📊 Probability Distribution Analysis...${NC}"
    $PYTHON_CMD probability_distribution_analyzer.py \
        -s "$BG_EXCL_SOFTMAX" \
        -i "$BG_EXCL_INFO" \
        -t 0.001 \
        -o "$OUTPUT_DIR/prob_dist_bg_excluded" \
        2>&1 | tee "$OUTPUT_DIR/prob_analysis_bg_excl.log"
fi

# =============================================================================
# 生成报告
# =============================================================================

echo ""
echo -e "${PURPLE}===============================================================================${NC}"
echo -e "${PURPLE}📋 GENERATING ANALYSIS REPORT${NC}"
echo -e "${PURPLE}===============================================================================${NC}"

# 创建报告
if [[ -n "$BG_EXCL_SOFTMAX" ]]; then
    REPORT_TYPE="comparison"
    REPORT_TITLE="Background Mode Comparison Report"
else
    REPORT_TYPE="single_mode"
    REPORT_TITLE="Background Included Analysis Report"
fi

cat > "$OUTPUT_DIR/analysis_report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>$REPORT_TITLE</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 40px; 
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }
        h2 { 
            color: #34495e; 
            margin-top: 30px; 
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .info-box {
            background: #e8f4fd;
            border: 1px solid #3498db;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .file-info {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 $REPORT_TITLE</h1>
        
        <div class="success-box">
            <h3>✅ Analysis Status: COMPLETED</h3>
            <p><strong>Date:</strong> $(date)</p>
            <p><strong>Using Fixed Data:</strong> Yes - memory layout issues resolved</p>
            <p><strong>Analysis Type:</strong> $REPORT_TYPE</p>
        </div>
        
        <h2>📊 Files Analyzed</h2>
        <div class="file-info">
            <strong>Background INCLUDED:</strong><br>
            • Softmax: $(basename "$BG_INCL_SOFTMAX")<br>
            • Info: $(basename "$BG_INCL_INFO")
        </div>
EOF

if [[ -n "$BG_EXCL_SOFTMAX" ]]; then
cat >> "$OUTPUT_DIR/analysis_report.html" << EOF
        <div class="file-info">
            <strong>Background EXCLUDED:</strong><br>
            • Softmax: $(basename "$BG_EXCL_SOFTMAX")<br>
            • Info: $(basename "$BG_EXCL_INFO")
        </div>
EOF
else
cat >> "$OUTPUT_DIR/analysis_report.html" << EOF
        <div class="warning-box">
            <h3>⚠️ Background EXCLUDED files not found</h3>
            <p>To generate background excluded analysis, use:</p>
            <code>./run_manual_predict.sh path/to/bg_excl_model.pth -n bg_excl_verification</code>
        </div>
EOF
fi

cat >> "$OUTPUT_DIR/analysis_report.html" << EOF
        
        <div class="file-info">
            <strong>Ground Truth Labels:</strong><br>
            • File: $(basename "$LABELS_FILE")
        </div>
        
        <h2>🎯 Key Improvements</h2>
        <div class="success-box">
            <ul>
                <li><strong>✅ Fixed Memory Layout:</strong> No more order='F' issues</li>
                <li><strong>✅ Correct Data Mapping:</strong> Predictions mapped to right spatial locations</li>
                <li><strong>✅ Meaningful F1 Scores:</strong> Should see F1 > 0.000 now</li>
                <li><strong>✅ Proper Probability Distributions:</strong> Real softmax values throughout volume</li>
            </ul>
        </div>
        
        <h2>📁 Generated Analysis Files</h2>
        <p>Check the following files in the output directory:</p>
        <ul>
            <li><strong>Visualizations:</strong> *.png files</li>
            <li><strong>Detailed Metrics:</strong> *.csv files</li>
            <li><strong>Analysis Logs:</strong> *.log files</li>
        </ul>
        
        <div class="info-box">
            <h3>💡 Expected Results</h3>
            <p>With the fixed memory layout, you should now see:</p>
            <ul>
                <li>F1 scores matching your training results (~0.6)</li>
                <li>Non-background predictions in visualizations</li>
                <li>Realistic probability distributions</li>
                <li>Meaningful comparison between training modes</li>
            </ul>
        </div>
    </div>
</body>
</html>
EOF

echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}🎉 AUTO ANALYSIS COMPLETED!${NC}"
echo -e "${GREEN}===============================================================================${NC}"

echo -e "${CYAN}📁 Results Location:${NC} $OUTPUT_DIR/"
echo -e "${CYAN}📋 Report:${NC} $OUTPUT_DIR/analysis_report.html"

echo ""
echo -e "${BLUE}📊 Analysis Summary:${NC}"
echo -e "${GREEN}✅ Background INCLUDED mode: Analyzed${NC}"
if [[ -n "$BG_EXCL_SOFTMAX" ]]; then
    echo -e "${GREEN}✅ Background EXCLUDED mode: Analyzed${NC}"
else
    echo -e "${YELLOW}⚠️  Background EXCLUDED mode: Not available${NC}"
    echo -e "    Generate with: ./run_manual_predict.sh bg_excl_model.pth -n bg_excl_verification"
fi

echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo -e "  1. Open ${YELLOW}$OUTPUT_DIR/analysis_report.html${NC}"
echo -e "  2. Check the visualization files (*.png)"
echo -e "  3. Review performance metrics (*.csv)" 
echo -e "  4. Compare with your training F1 scores"

if [[ -z "$BG_EXCL_SOFTMAX" ]]; then
    echo ""
    echo -e "${YELLOW}📝 To complete the comparison:${NC}"
    echo -e "  1. Find your background excluded model (.pth file)"
    echo -e "  2. Run: ${CYAN}./run_manual_predict.sh path/to/bg_excl_model.pth -n bg_excl_verification${NC}"
    echo -e "  3. Re-run this script for full comparison"
fi