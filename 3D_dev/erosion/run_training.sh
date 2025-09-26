#!/bin/bash

# 38-Fold Cross Validation Training Control Script
# Usage:
#   ./run_training.sh single <fold_number>  # Train single fold
#   ./run_training.sh all                    # Train all 38 folds
#   ./run_training.sh test                   # Test with fold 38

set -e  # Exit on error

# Configuration
PYTHON_SCRIPT="train_38fold.py"
DATASET_INDEX="/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/dataset_index_validated.json"
OUTPUT_BASE="./output"
EPOCHS=25
BATCH_SIZE=128
LR=0.00001
LOG_INTERVAL=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check Python dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."

    python3 -c "import torch" 2>/dev/null || {
        print_error "PyTorch not installed. Please install with: pip install torch"
        exit 1
    }

    python3 -c "import sklearn" 2>/dev/null || {
        print_error "scikit-learn not installed. Please install with: pip install scikit-learn"
        exit 1
    }

    python3 -c "import h5py" 2>/dev/null || {
        print_error "h5py not installed. Please install with: pip install h5py"
        exit 1
    }

    print_info "All dependencies satisfied"
}

# Function to train single fold
train_single_fold() {
    local fold=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_dir="${OUTPUT_BASE}/single_fold_${fold}_${timestamp}"

    print_info "Training single fold: ${fold}"
    print_info "Output directory: ${output_dir}"

    python3 ${PYTHON_SCRIPT} \
        --dataset-index ${DATASET_INDEX} \
        --fold ${fold} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --output-dir ${output_dir} \
        --log-interval ${LOG_INTERVAL}

    if [ $? -eq 0 ]; then
        print_info "Training completed successfully for fold ${fold}"
        print_info "Results saved in: ${output_dir}"
    else
        print_error "Training failed for fold ${fold}"
        exit 1
    fi
}

# Function to train all folds
train_all_folds() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_dir="${OUTPUT_BASE}/all_folds_${timestamp}"

    print_info "Training all 38 folds"
    print_info "Output directory: ${output_dir}"
    print_warning "This will take several hours to complete"

    python3 ${PYTHON_SCRIPT} \
        --dataset-index ${DATASET_INDEX} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --output-dir ${output_dir} \
        --log-interval ${LOG_INTERVAL}

    if [ $? -eq 0 ]; then
        print_info "All folds training completed successfully"
        print_info "Results saved in: ${output_dir}"

        # Display summary if results file exists
        if [ -f "${output_dir}/cross_validation_results.json" ]; then
            print_info "Cross-validation summary:"
            python3 -c "
import json
with open('${output_dir}/cross_validation_results.json', 'r') as f:
    results = json.load(f)
    valid = [r for r in results if 'test_accuracy' in r]
    if valid:
        import numpy as np
        accs = [r['test_accuracy'] for r in valid]
        f1s = [r['test_f1'] for r in valid]
        print(f'  Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
        print(f'  Average Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
        print(f'  Successful folds: {len(valid)}/{len(results)}')
"
        fi
    else
        print_error "Training failed"
        exit 1
    fi
}

# Function to run test fold (fold 38)
test_fold() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_dir="${OUTPUT_BASE}/test_fold38_${timestamp}"

    print_info "Running test with fold 38"
    print_info "Output directory: ${output_dir}"

    # Run with fewer epochs for testing
    python3 ${PYTHON_SCRIPT} \
        --dataset-index ${DATASET_INDEX} \
        --fold 38 \
        --epochs 3 \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --output-dir ${output_dir} \
        --log-interval 1

    if [ $? -eq 0 ]; then
        print_info "Test completed successfully"
        print_info "Test results saved in: ${output_dir}"
    else
        print_error "Test failed"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 {single|all|test|help} [fold_number]"
    echo ""
    echo "Commands:"
    echo "  single <fold>  Train a single fold (1-38)"
    echo "  all            Train all 38 folds"
    echo "  test           Quick test with fold 38 (3 epochs)"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 single 5    # Train fold 5 only"
    echo "  $0 all         # Train all 38 folds"
    echo "  $0 test        # Quick test run"
    echo ""
    echo "Current Configuration:"
    echo "  Dataset: ${DATASET_INDEX}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Learning rate: ${LR}"
    echo "  Output base: ${OUTPUT_BASE}"
}

# Main execution
main() {
    # Check if Python script exists
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        print_error "Python script not found: ${PYTHON_SCRIPT}"
        exit 1
    fi

    # Parse command line arguments
    case "$1" in
        single)
            if [ -z "$2" ]; then
                print_error "Please specify fold number (1-38)"
                show_usage
                exit 1
            fi

            if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -lt 1 ] || [ "$2" -gt 38 ]; then
                print_error "Invalid fold number: $2. Must be between 1 and 38."
                exit 1
            fi

            check_dependencies
            train_single_fold $2
            ;;

        all)
            check_dependencies

            # Confirm with user for long training
            read -p "This will train all 38 folds and may take several hours. Continue? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                train_all_folds
            else
                print_info "Training cancelled"
                exit 0
            fi
            ;;

        test)
            check_dependencies
            test_fold
            ;;

        help)
            show_usage
            ;;

        *)
            print_error "Invalid command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"