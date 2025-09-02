#!/bin/bash

# =============================================================================
# 背景包含 vs 排除模式对比分析脚本
# 自动分析两种训练模式的概率分布和性能指标差异
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

# 文件路径配置
BG_INCL_SOFTMAX="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_incl_20250827_154857.nii.gz"
BG_INCL_INFO="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.json"

BG_EXCL_SOFTMAX="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.nii.gz"
BG_EXCL_INFO="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.json"

LABELS_FILE="/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS/FOR_016_20250204_reproducibility/balanced_output/balanced_labels_3d10000.nii.gz"

# 输出目录
OUTPUT_DIR="bg_comparison_results_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}🔬 Background Include vs Exclude Comparison Analysis${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}Analyzing probability distributions and performance metrics${NC}"
echo -e "${GREEN}for both background inclusion modes${NC}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"

# 检查Python环境
echo -e "${BLUE}🔍 Checking environment...${NC}"
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found!${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo -e "${GREEN}✅ Python found: $($PYTHON_CMD --version)${NC}"

# 检查文件是否存在
echo -e "${BLUE}📂 Checking input files...${NC}"
files=(
    "$BG_INCL_SOFTMAX"
    "$BG_INCL_INFO" 
    "$BG_EXCL_SOFTMAX"
    "$BG_EXCL_INFO"
    "$LABELS_FILE"
)

for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}❌ File not found: $file${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ All input files found${NC}"

cd "$SCRIPT_DIR"

# =============================================================================
# 概率分布分析
# =============================================================================

echo ""
echo -e "${PURPLE}===============================================================================${NC}"
echo -e "${PURPLE}📊 PROBABILITY DISTRIBUTION ANALYSIS${NC}"
echo -e "${PURPLE}===============================================================================${NC}"

# 分析背景包含模式
echo -e "${CYAN}🌟 Analyzing Background INCLUDED mode...${NC}"
$PYTHON_CMD probability_distribution_analyzer.py \
    -s "$BG_INCL_SOFTMAX" \
    -i "$BG_INCL_INFO" \
    -t 0.01 \
    -o "$OUTPUT_DIR/prob_dist_bg_included" \
    2>&1 | tee "$OUTPUT_DIR/prob_analysis_bg_incl.log"

echo ""
echo -e "${CYAN}🎯 Analyzing Background EXCLUDED mode...${NC}"
$PYTHON_CMD probability_distribution_analyzer.py \
    -s "$BG_EXCL_SOFTMAX" \
    -i "$BG_EXCL_INFO" \
    -t 0.01 \
    -o "$OUTPUT_DIR/prob_dist_bg_excluded" \
    2>&1 | tee "$OUTPUT_DIR/prob_analysis_bg_excl.log"

# =============================================================================
# 性能指标分析  
# =============================================================================

echo ""
echo -e "${PURPLE}===============================================================================${NC}"
echo -e "${PURPLE}📈 PERFORMANCE METRICS ANALYSIS${NC}"
echo -e "${PURPLE}===============================================================================${NC}"

# 分析背景包含模式
echo -e "${CYAN}🌟 Analyzing Performance with Background INCLUDED...${NC}"
$PYTHON_CMD performance_metrics_analyzer.py \
    -s "$BG_INCL_SOFTMAX" \
    -i "$BG_INCL_INFO" \
    -l "$LABELS_FILE" \
    -o "$OUTPUT_DIR/performance_bg_included" \
    2>&1 | tee "$OUTPUT_DIR/performance_bg_incl.log"

echo ""
echo -e "${CYAN}🎯 Analyzing Performance with Background EXCLUDED...${NC}"
$PYTHON_CMD performance_metrics_analyzer.py \
    -s "$BG_EXCL_SOFTMAX" \
    -i "$BG_EXCL_INFO" \
    -l "$LABELS_FILE" \
    -o "$OUTPUT_DIR/performance_bg_excluded" \
    2>&1 | tee "$OUTPUT_DIR/performance_bg_excl.log"

# =============================================================================
# 生成对比报告
# =============================================================================

echo ""
echo -e "${PURPLE}===============================================================================${NC}"
echo -e "${PURPLE}📋 GENERATING COMPARISON REPORT${NC}"
echo -e "${PURPLE}===============================================================================${NC}"

# 创建HTML对比报告
cat > "$OUTPUT_DIR/comparison_report.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Background Mode Comparison Report</title>
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
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 20px 0;
        }
        .mode-section {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
        }
        .mode-included {
            border-color: #27ae60;
            background: #f8fff8;
        }
        .mode-excluded {
            border-color: #e74c3c;
            background: #fff8f8;
        }
        .mode-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        .included-title { color: #27ae60; }
        .excluded-title { color: #e74c3c; }
        .image-container {
            text-align: center;
            margin: 15px 0;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .log-section {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 300px;
            overflow-y: auto;
        }
        .summary-box {
            background: #e8f4fd;
            border: 1px solid #3498db;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .file-list {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        .file-list ul {
            margin: 0;
            padding-left: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Background Mode Comparison Analysis</h1>
        
        <div class="summary-box">
            <h3>📊 Analysis Overview</h3>
            <p><strong>Purpose:</strong> Compare probability distributions and performance metrics between background inclusion and exclusion training modes.</p>
            <p><strong>Subject:</strong> FOR_016_20250204_reproducibility</p>
            <p><strong>Analysis Date:</strong> TIMESTAMP_PLACEHOLDER</p>
        </div>
        
        <h2>📈 Probability Distribution Analysis</h2>
        <div class="comparison-grid">
            <div class="mode-section mode-included">
                <div class="mode-title included-title">🌟 Background INCLUDED</div>
                <div class="image-container">
                    <img src="prob_dist_bg_included.png" alt="Background Included Probability Distribution">
                </div>
                <p><strong>Characteristics:</strong></p>
                <ul>
                    <li>All voxels (background + foreground) analyzed</li>
                    <li>True softmax predictions for entire volume</li>
                    <li>More realistic uncertainty patterns</li>
                </ul>
            </div>
            
            <div class="mode-section mode-excluded">
                <div class="mode-title excluded-title">🎯 Background EXCLUDED</div>
                <div class="image-container">
                    <img src="prob_dist_bg_excluded.png" alt="Background Excluded Probability Distribution">
                </div>
                <p><strong>Characteristics:</strong></p>
                <ul>
                    <li>Only foreground voxels analyzed</li>
                    <li>Background artificially set to perfect predictions</li>
                    <li>May show artificially low uncertainty</li>
                </ul>
            </div>
        </div>
        
        <h2>🎯 Performance Metrics Analysis</h2>
        <div class="comparison-grid">
            <div class="mode-section mode-included">
                <div class="mode-title included-title">🌟 Background INCLUDED</div>
                <div class="image-container">
                    <img src="performance_bg_included_visualization.png" alt="Background Included Performance">
                </div>
                <p><strong>Analysis Scope:</strong></p>
                <ul>
                    <li>Full volume analysis</li>
                    <li>Background class performance included</li>
                    <li>True model generalization assessment</li>
                </ul>
            </div>
            
            <div class="mode-section mode-excluded">
                <div class="mode-title excluded-title">🎯 Background EXCLUDED</div>
                <div class="image-container">
                    <img src="performance_bg_excluded_visualization.png" alt="Background Excluded Performance">
                </div>
                <p><strong>Analysis Scope:</strong></p>
                <ul>
                    <li>Foreground-only analysis</li>
                    <li>Background class artificially perfect</li>
                    <li>Focus on anatomical structure performance</li>
                </ul>
            </div>
        </div>
        
        <h2>📁 Generated Files</h2>
        <div class="file-list">
            <h4>🎨 Visualizations:</h4>
            <ul>
                <li><code>prob_dist_bg_included.png</code> - Probability distribution analysis (background included)</li>
                <li><code>prob_dist_bg_excluded.png</code> - Probability distribution analysis (background excluded)</li>
                <li><code>performance_bg_included_visualization.png</code> - Performance metrics (background included)</li>
                <li><code>performance_bg_excluded_visualization.png</code> - Performance metrics (background excluded)</li>
            </ul>
            
            <h4>📊 Data Reports:</h4>
            <ul>
                <li><code>prob_dist_bg_included.csv</code> - Detailed probability statistics (background included)</li>
                <li><code>prob_dist_bg_excluded.csv</code> - Detailed probability statistics (background excluded)</li>
                <li><code>performance_bg_included.csv</code> - Performance metrics (background included)</li>
                <li><code>performance_bg_excluded.csv</code> - Performance metrics (background excluded)</li>
            </ul>
            
            <h4>📝 Log Files:</h4>
            <ul>
                <li><code>prob_analysis_bg_incl.log</code> - Probability analysis log (background included)</li>
                <li><code>prob_analysis_bg_excl.log</code> - Probability analysis log (background excluded)</li>
                <li><code>performance_bg_incl.log</code> - Performance analysis log (background included)</li>
                <li><code>performance_bg_excl.log</code> - Performance analysis log (background excluded)</li>
            </ul>
        </div>
        
        <h2>💡 Key Questions to Investigate</h2>
        <div class="summary-box">
            <h4>🔍 Probability Distribution Comparison:</h4>
            <ul>
                <li>Which mode shows more realistic probability distributions?</li>
                <li>How does background handling affect class confidence patterns?</li>
                <li>Are there significant differences in uncertainty patterns?</li>
            </ul>
            
            <h4>📊 Performance Metrics Comparison:</h4>
            <ul>
                <li>Which anatomical structures benefit from background inclusion?</li>
                <li>How does overall F1 score compare between modes?</li>
                <li>Are there classes that perform better with background exclusion?</li>
            </ul>
            
            <h4>🎯 Practical Implications:</h4>
            <ul>
                <li>Which training strategy is more suitable for your use case?</li>
                <li>How do computational costs compare?</li>
                <li>What are the trade-offs in model interpretability?</li>
            </ul>
        </div>
        
        <h2>🚀 Next Steps</h2>
        <div class="summary-box">
            <ol>
                <li><strong>Review Visualizations:</strong> Compare the generated plots side by side</li>
                <li><strong>Analyze CSV Reports:</strong> Dive into detailed numerical comparisons</li>
                <li><strong>Check Log Files:</strong> Review detailed analysis outputs and statistics</li>
                <li><strong>Make Decision:</strong> Choose the optimal training strategy based on your requirements</li>
            </ol>
        </div>
    </div>
</body>
</html>
EOF

# 更新时间戳
sed -i "s/TIMESTAMP_PLACEHOLDER/$(date)/" "$OUTPUT_DIR/comparison_report.html"

# =============================================================================
# 完成总结
# =============================================================================

echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}🎉 COMPARISON ANALYSIS COMPLETED!${NC}"
echo -e "${GREEN}===============================================================================${NC}"

echo -e "${CYAN}📁 Results Location:${NC} $OUTPUT_DIR/"
echo -e "${CYAN}📋 Main Report:${NC} $OUTPUT_DIR/comparison_report.html"

echo ""
echo -e "${BLUE}📊 Generated Analysis Files:${NC}"
echo -e "${YELLOW}Visualizations:${NC}"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  (Visualizations will be generated during analysis)"

echo -e "${YELLOW}Data Reports:${NC}"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (CSV reports will be generated during analysis)"

echo -e "${YELLOW}Log Files:${NC}"
ls -la "$OUTPUT_DIR"/*.log 2>/dev/null || echo "  (Log files will be generated during analysis)"

echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo -e "  1. Open ${YELLOW}$OUTPUT_DIR/comparison_report.html${NC} in your browser"
echo -e "  2. Compare the visualization results"
echo -e "  3. Review the CSV reports for detailed metrics"
echo -e "  4. Check log files for analysis summaries"

# 尝试打开报告
if command -v open &> /dev/null; then  # macOS
    read -p "🌐 Open the comparison report now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT_DIR/comparison_report.html"
    fi
elif command -v xdg-open &> /dev/null; then  # Linux
    read -p "🌐 Open the comparison report now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open "$OUTPUT_DIR/comparison_report.html"
    fi
fi

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}✨ Background Mode Comparison Analysis Complete! ✨${NC}"
echo -e "${BLUE}===============================================================================${NC}"