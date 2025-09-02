#!/bin/bash

# =============================================================================
# Brain Segmentation Visualization Toolkit - One-Click Runner
# =============================================================================
# 
# This script automatically detects your segmentation results and runs
# comprehensive visualization and analysis.
#
# Usage:
#   ./run_visualization.sh                    # Auto-detect files
#   ./run_visualization.sh path/to/softmax.nii.gz  # Specify softmax file
#   ./run_visualization.sh -h                 # Show help
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Banner
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}🧠 Brain Segmentation Visualization Toolkit${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}Comprehensive analysis and visualization of brain segmentation results${NC}"
echo -e "${GREEN}Generated softmax predictions, uncertainty analysis, and performance metrics${NC}"
echo ""

# Function to show help
show_help() {
    echo -e "${CYAN}Usage:${NC}"
    echo "  $0                                    # Auto-detect files and run analysis"
    echo "  $0 path/to/softmax.nii.gz            # Specify softmax file"
    echo "  $0 -s softmax.nii.gz -l labels.nii.gz  # Specify softmax and labels"
    echo "  $0 -h, --help                        # Show this help"
    echo ""
    echo -e "${CYAN}Options:${NC}"
    echo "  -s, --softmax FILE    Softmax prediction file (.nii.gz)"
    echo "  -i, --info FILE       Info JSON file (auto-detected if not specified)"
    echo "  -l, --labels FILE     Ground truth labels file (optional, enables performance analysis)"
    echo "  -c, --config FILE     Configuration file (default: config.yaml)"
    echo "  -o, --output DIR      Output directory (default: auto-generated)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  # Auto-detect and analyze latest results"
    echo "  $0"
    echo ""
    echo "  # Analyze specific softmax file"
    echo "  $0 ../results/test_softmax_3d_FOR_016_bg_excl_20250827.nii.gz"
    echo ""
    echo "  # Full analysis with labels for performance metrics"
    echo "  $0 -s softmax.nii.gz -l balanced_labels_3d10000.nii.gz"
    echo ""
    echo -e "${CYAN}File Detection:${NC}"
    echo "  The script automatically searches for:"
    echo "  - *softmax_3d*.nii.gz files"
    echo "  - *softmax_info*.json files"
    echo "  - *labels*.nii.gz files"
    echo "  in current directory, results/, and parent directories"
}

# Parse command line arguments
SOFTMAX_FILE=""
INFO_FILE=""
LABELS_FILE=""
CONFIG_FILE="config.yaml"
OUTPUT_DIR=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--softmax)
            SOFTMAX_FILE="$2"
            shift 2
            ;;
        -i|--info)
            INFO_FILE="$2"
            shift 2
            ;;
        -l|--labels)
            LABELS_FILE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -*|--*)
            echo -e "${RED}Unknown option $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Handle positional argument (softmax file)
if [[ ${#POSITIONAL_ARGS[@]} -eq 1 ]]; then
    SOFTMAX_FILE="${POSITIONAL_ARGS[0]}"
elif [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    echo -e "${RED}❌ Too many positional arguments${NC}"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Check Python environment
echo -e "${BLUE}🔍 Checking environment...${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found! Please install Python 3.7+${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}✅ Python found: $($PYTHON_CMD --version)${NC}"

# Check required Python packages
echo -e "${BLUE}📦 Checking required packages...${NC}"
REQUIRED_PACKAGES=(
    "numpy"
    "pandas" 
    "matplotlib"
    "seaborn"
    "nibabel"
    "scikit-learn"
    "scipy"
    "plotly"
    "pyyaml"
)

MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${BLUE}📥 Installing missing packages...${NC}"
    
    # Try pip install
    if command -v pip3 &> /dev/null; then
        pip3 install "${MISSING_PACKAGES[@]}"
    elif command -v pip &> /dev/null; then
        pip install "${MISSING_PACKAGES[@]}"
    else
        echo -e "${RED}❌ pip not found! Please install missing packages manually:${NC}"
        echo "pip install ${MISSING_PACKAGES[*]}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ All required packages found${NC}"
fi

# Prepare arguments for Python script
PYTHON_ARGS=()

if [[ -n "$SOFTMAX_FILE" ]]; then
    PYTHON_ARGS+=("-s" "$SOFTMAX_FILE")
fi

if [[ -n "$INFO_FILE" ]]; then
    PYTHON_ARGS+=("-i" "$INFO_FILE")
fi

if [[ -n "$LABELS_FILE" ]]; then
    PYTHON_ARGS+=("-l" "$LABELS_FILE")
fi

if [[ -n "$CONFIG_FILE" ]]; then
    PYTHON_ARGS+=("-c" "$CONFIG_FILE")
fi

# Add search directories
SEARCH_DIRS=(
    "."
    "../results"
    "results"
    "../output"
    "output"
    "../banlanced"
    "banlanced"
    "../logs"
    "logs"
)

PYTHON_ARGS+=("--search-dirs" "${SEARCH_DIRS[@]}")

echo ""
echo -e "${PURPLE}🚀 Starting analysis...${NC}"
echo -e "${BLUE}Command: $PYTHON_CMD main_analyzer.py ${PYTHON_ARGS[*]}${NC}"
echo ""

# Run the main analyzer
if $PYTHON_CMD main_analyzer.py "${PYTHON_ARGS[@]}"; then
    echo ""
    echo -e "${GREEN}===============================================================================${NC}"
    echo -e "${GREEN}🎉 ANALYSIS COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}===============================================================================${NC}"
    
    # Find the output directory (most recent)
    OUTPUT_PATTERN="visualization_results_*"
    if compgen -G "$OUTPUT_PATTERN" > /dev/null; then
        LATEST_OUTPUT=$(ls -dt $OUTPUT_PATTERN 2>/dev/null | head -n1)
        if [[ -n "$LATEST_OUTPUT" ]]; then
            echo -e "${CYAN}📁 Results Location:${NC} $LATEST_OUTPUT/"
            echo -e "${CYAN}📋 Main Report:${NC} $LATEST_OUTPUT/index.html"
            echo ""
            echo -e "${BLUE}💡 Next Steps:${NC}"
            echo -e "  1. Open ${YELLOW}$LATEST_OUTPUT/index.html${NC} in your web browser"
            echo -e "  2. Explore the interactive visualizations"
            echo -e "  3. Review the performance metrics (if labels provided)"
            echo -e "  4. Check individual analysis folders for detailed results"
            
            # Offer to open the report
            if command -v open &> /dev/null; then  # macOS
                echo ""
                read -p "🌐 Open the report in your browser now? (y/N): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    open "$LATEST_OUTPUT/index.html"
                fi
            elif command -v xdg-open &> /dev/null; then  # Linux
                echo ""
                read -p "🌐 Open the report in your browser now? (y/N): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    xdg-open "$LATEST_OUTPUT/index.html"
                fi
            fi
        fi
    fi
else
    echo ""
    echo -e "${RED}===============================================================================${NC}"
    echo -e "${RED}❌ ANALYSIS FAILED${NC}"
    echo -e "${RED}===============================================================================${NC}"
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo -e "  1. Check that your softmax files exist and are valid"
    echo -e "  2. Ensure you have sufficient disk space"
    echo -e "  3. Verify all required Python packages are installed"
    echo -e "  4. Check the log files for detailed error messages"
    echo -e "  5. Run with specific file paths if auto-detection fails:"
    echo -e "     ${CYAN}$0 -s your_softmax_file.nii.gz${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}✨ Thank you for using Brain Segmentation Visualization Toolkit! ✨${NC}"
echo -e "${BLUE}===============================================================================${NC}"