#!/bin/bash
# Leave-One-Out Cross-Validation for MRI ResNet
#
# Automatically runs training for all 38 subjects in Leave-One-Out fashion.
# Each subject is used once as the test set while the remaining 37 are used for training.
#
# Usage:
#     bash run_leave_one_out.sh
#
# Configuration:
#     Modify the variables below to adjust training parameters.

set -e  # Exit on any error

# ==================== CONFIGURATION ====================

# Data directory settings - same as 3D CNN baseline
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D data (2D patches)
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"              # 1D data

# ResNet uses 3D data (for 2D patch extraction)
DATA_DIR=$DATA_DIR_3D

# Output directory
OUTPUT_BASE="./results_leave_one_out_resnet"

# ResNet configuration
BASE_WIDTH=104          # For ~50M parameters
INPUT_CHANNELS=351      # MRI feature channels
NUM_CLASSES=102         # Brain regions
PATCH_SIZE=7            # 7×7 patches

# Training parameters
EPOCHS=20                  # 减少epoch数：因为现在每个epoch包含71M patches vs 原来370K
BATCH_SIZE=2048           # 保持不变，已经优化过
LEARNING_RATE=0.0008      # 保持不变，已经根据batch size调整过
WEIGHT_DECAY=0.0002
PATIENCE=8                # 减少patience：完整数据训练收敛更快

# Data parameters - 内存高效模式
SAMPLES_PER_SUBJECT=10000  # 在内存高效模式下此参数被忽略，实际使用全部71M patches
NUM_WORKERS=0              # 重要：内存高效模式必须设为0 (h5py不支持多进程)
MEMORY_EFFICIENT="--memory_efficient"  # 启用内存高效模式

# Loss and optimization
LOSS_TYPE="cb_focal"       # cb_focal, logit_adj, focal, weighted_ce, ce
# USE_MIXUP="--use_mixup"  # 禁用Mixup - 对脑区分类任务不适用
MIXUP_ALPHA=0.2
USE_EMA="--use_ema"

# Hardware
DEVICE="cuda"
SEED=42

# Other options
VERBOSE="--verbose"

# ==================== VALIDATION ====================

echo "=== MRI ResNet Leave-One-Out Cross-Validation (Memory-Efficient) ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_BASE"
echo "Configuration:"
echo "  - Base width: $BASE_WIDTH (≈50M parameters)"
echo "  - Patch size: ${PATCH_SIZE}×${PATCH_SIZE}"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Loss type: $LOSS_TYPE"
echo "  - Epochs: $EPOCHS (reduced due to 71M patches per epoch)"
echo "  - Memory mode: EFFICIENT (1.2GB vs 450GB)"
echo "  - Workers: $NUM_WORKERS (0 for h5py compatibility)"
echo "  - Device: $DEVICE"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    echo "Please modify DATA_DIR in this script to point to your MAT files."
    exit 1
fi

# Check if MAT files exist
MAT_COUNT=$(find "$DATA_DIR" -name "*.mat" | wc -l)
if [ "$MAT_COUNT" -eq 0 ]; then
    echo "ERROR: No MAT files found in $DATA_DIR"
    echo "Please ensure the directory contains .mat files."
    exit 1
fi

echo "Found $MAT_COUNT MAT files in data directory"

# Check if Python script exists
TRAIN_SCRIPT="./train_mri_resnet.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    echo "Please ensure you're running this from the scripts directory."
    exit 1
fi

# Check GPU availability if using CUDA
if [ "$DEVICE" = "cuda" ]; then
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    if [ $? -ne 0 ]; then
        echo "ERROR: CUDA not available. Set DEVICE='cpu' or install CUDA."
        exit 1
    fi
    
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "Using GPU: $GPU_NAME (Count: $GPU_COUNT)"
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Save configuration
CONFIG_FILE="$OUTPUT_BASE/training_config.txt"
cat > "$CONFIG_FILE" << EOF
MRI ResNet Leave-One-Out Configuration
=====================================
Date: $(date)
Host: $(hostname)
User: $(whoami)
Working Directory: $(pwd)

Data Configuration:
- Data Directory: $DATA_DIR (3D validated data for 2D patch extraction)
- Alternative 1D Directory: $DATA_DIR_1D (not used by ResNet)
- MAT Files Found: $MAT_COUNT
- Patch Size: ${PATCH_SIZE}×${PATCH_SIZE}
- Input Channels: $INPUT_CHANNELS
- Output Classes: $NUM_CLASSES
- Samples Per Subject: ${SAMPLES_PER_SUBJECT:-"All valid samples"}

Model Configuration:
- Architecture: ResNet-50 (3-4-6-3 Bottleneck)
- Base Width: $BASE_WIDTH (~50M parameters)
- Stem: 3×3/s1/p1, expand-first (351→512 channels)
- Spatial Flow: 7→7→4→2→1
- Channel Flow: 512→256→512→1024→2048→102

Training Configuration:
- Epochs: $EPOCHS
- Batch Size: $BATCH_SIZE
- Learning Rate: $LEARNING_RATE
- Weight Decay: $WEIGHT_DECAY
- Early Stopping Patience: $PATIENCE
- Loss Function: $LOSS_TYPE
- Mixup: $([ -n "$USE_MIXUP" ] && echo "Enabled (α=$MIXUP_ALPHA)" || echo "Disabled")
- EMA: $([ -n "$USE_EMA" ] && echo "Enabled" || echo "Disabled")

Hardware:
- Device: $DEVICE
- Workers: $NUM_WORKERS
- Random Seed: $SEED
EOF

echo ""
echo "Configuration saved to: $CONFIG_FILE"
echo ""

# ==================== TRAINING LOOP ====================

# Build command arguments
ARGS=""
ARGS="$ARGS --data_dir $DATA_DIR"
ARGS="$ARGS --output_dir $OUTPUT_BASE"
ARGS="$ARGS --base_width $BASE_WIDTH"
ARGS="$ARGS --input_channels $INPUT_CHANNELS"
ARGS="$ARGS --num_classes $NUM_CLASSES"
ARGS="$ARGS --epochs $EPOCHS"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --learning_rate $LEARNING_RATE"
ARGS="$ARGS --weight_decay $WEIGHT_DECAY"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --patch_size $PATCH_SIZE"
ARGS="$ARGS --num_workers $NUM_WORKERS"
ARGS="$ARGS --loss_type $LOSS_TYPE"
ARGS="$ARGS --mixup_alpha $MIXUP_ALPHA"
ARGS="$ARGS --device $DEVICE"
ARGS="$ARGS --seed $SEED"

# Add config file path to ensure default_config.json is loaded
ARGS="$ARGS --config ../configs/default_config.json"

# Add optional arguments
[ -n "$SAMPLES_PER_SUBJECT" ] && ARGS="$ARGS --samples_per_subject $SAMPLES_PER_SUBJECT"
# [ -n "$USE_MIXUP" ] && ARGS="$ARGS $USE_MIXUP"  # Mixup已禁用
[ -n "$USE_EMA" ] && ARGS="$ARGS $USE_EMA"
[ -n "$MEMORY_EFFICIENT" ] && ARGS="$ARGS $MEMORY_EFFICIENT"  # 内存高效模式
[ -n "$VERBOSE" ] && ARGS="$ARGS $VERBOSE"

# Start training loop
TOTAL_SUBJECTS=38
START_TIME=$(date +%s)

echo "Starting Leave-One-Out training for $TOTAL_SUBJECTS subjects..."
echo "This may take several hours depending on your hardware."
echo ""

# Log file for overall progress
PROGRESS_LOG="$OUTPUT_BASE/training_progress.log"
echo "Leave-One-Out Training Progress - Started $(date)" > "$PROGRESS_LOG"

for test_subject in $(seq 1 $TOTAL_SUBJECTS); do
    echo "=========================================="
    echo "Training Subject $test_subject/$TOTAL_SUBJECTS"
    echo "Test Subject: $test_subject"
    echo "=========================================="
    
    SUBJECT_START=$(date +%s)
    
    # Run training
    CMD="python3 $TRAIN_SCRIPT $ARGS --test_subject $test_subject"
    echo "Command: $CMD"
    echo ""
    
    if eval $CMD; then
        SUBJECT_END=$(date +%s)
        SUBJECT_TIME=$((SUBJECT_END - SUBJECT_START))
        
        echo ""
        echo "Subject $test_subject completed successfully in ${SUBJECT_TIME}s"
        echo "Subject $test_subject: SUCCESS (${SUBJECT_TIME}s)" >> "$PROGRESS_LOG"
        
        # Check if results exist
        SUBJECT_DIR="$OUTPUT_BASE/resnet_test_subject_$test_subject"
        if [ -f "$SUBJECT_DIR/training_results.json" ]; then
            BEST_F1=$(python3 -c "
import json
with open('$SUBJECT_DIR/training_results.json') as f:
    data = json.load(f)
    print(f'{data[\"best_f1\"]:.4f}')
" 2>/dev/null || echo "N/A")
            echo "Best F1 Score: $BEST_F1"
            echo "  Best F1: $BEST_F1" >> "$PROGRESS_LOG"
        fi
    else
        echo ""
        echo "ERROR: Training failed for subject $test_subject"
        echo "Subject $test_subject: FAILED" >> "$PROGRESS_LOG"
        
        # Optionally continue or exit
        read -p "Continue with remaining subjects? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Training stopped by user."
            exit 1
        fi
    fi
    
    echo ""
done

# ==================== COMPLETION ====================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo "=========================================="
echo "Leave-One-Out Training Completed!"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved in: $OUTPUT_BASE"
echo ""

# Summary
echo "Training Summary:" >> "$PROGRESS_LOG"
echo "Total time: ${HOURS}h ${MINUTES}m" >> "$PROGRESS_LOG"
echo "Completed: $(date)" >> "$PROGRESS_LOG"

# Count successful runs
SUCCESS_COUNT=$(grep "SUCCESS" "$PROGRESS_LOG" | wc -l)
FAIL_COUNT=$(grep "FAILED" "$PROGRESS_LOG" | wc -l)

echo "Successful runs: $SUCCESS_COUNT/$TOTAL_SUBJECTS"
echo "Failed runs: $FAIL_COUNT/$TOTAL_SUBJECTS"

if [ $SUCCESS_COUNT -eq $TOTAL_SUBJECTS ]; then
    echo "🎉 All subjects completed successfully!"
else
    echo "⚠️  Some subjects failed. Check individual logs for details."
fi

echo ""
echo "Next steps:"
echo "1. Run analysis script to aggregate results:"
echo "   python3 analyze_resnet_results.py --results_dir $OUTPUT_BASE"
echo ""
echo "2. Compare with baseline methods"
echo ""
echo "3. Generate detailed reports and visualizations"
echo ""

# Try to run analysis automatically if script exists
ANALYSIS_SCRIPT="./analyze_resnet_results.py"
if [ -f "$ANALYSIS_SCRIPT" ] && [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Running automatic analysis..."
    python3 "$ANALYSIS_SCRIPT" --results_dir "$OUTPUT_BASE" || echo "Analysis script failed"
fi

echo "Training completed! Check $OUTPUT_BASE for all results."