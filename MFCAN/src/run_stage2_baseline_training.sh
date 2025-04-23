#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置默认参数
MODEL_TYPE="deep_mlp"
FEATURE_TYPE="selected"
FEATURE_SELECTION="rf"  # 默认使用随机森林特征选择
FEATURE_SUBSET=""  # 可选的特征子集
CONFIG_PATH="configs/base_config.json"
SELECTED_FEATURES=""
TRANSFORMED_FEATURES=""
OUTPUT_DIR=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --feature_type)
      FEATURE_TYPE="$2"
      shift 2
      ;;
    --feature_selection)
      FEATURE_SELECTION="$2"
      shift 2
      ;;
    --feature_subset)
      FEATURE_SUBSET="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --selected_features)
      SELECTED_FEATURES="$2"
      shift 2
      ;;
    --transformed_features)
      TRANSFORMED_FEATURES="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 设置日志文件路径
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/shell/stage2_baseline_${MODEL_TYPE}_${FEATURE_TYPE}_${FEATURE_SELECTION}_${TIMESTAMP}.log"

echo "Starting Stage 2 Baseline Training (with Macro F1 as primary metric)..."
echo "Model type: $MODEL_TYPE"
echo "Feature type: $FEATURE_TYPE"
echo "Feature selection method: $FEATURE_SELECTION"
if [ ! -z "$FEATURE_SUBSET" ]; then
  echo "Feature subset: $FEATURE_SUBSET"
fi
echo "Config file: $CONFIG_PATH"
echo "Log file: $LOG_FILE"

# 如果未指定特征文件路径，尝试自动查找最新的特征工程结果
if [ -z "$SELECTED_FEATURES" ]; then
  # 根据特征选择方法设置子目录
  FEATURE_PATH_PATTERN="results/feature_engineering/*/*selected_features*.h5"
  SELECTED_FEATURES=$(ls -td ${FEATURE_PATH_PATTERN} 2>/dev/null | head -1)
  
  if [ ! -z "$SELECTED_FEATURES" ]; then
    echo "Automatically selected feature file: $SELECTED_FEATURES"
  else
    echo "Warning: Could not find feature selection result for method ${FEATURE_SELECTION}"
    # 尝试找到任意特征选择结果
    SELECTED_FEATURES=$(ls -td results/feature_engineering/*/feature_selection*.h5 2>/dev/null | head -1)
    
    if [ ! -z "$SELECTED_FEATURES" ]; then
      echo "Using alternative feature selection result: $SELECTED_FEATURES"
    else
      echo "Error: No feature selection files found"
      exit 1
    fi
  fi
fi

# 对于combined特征类型，确保也有transformed_features文件
if [ "$FEATURE_TYPE" == "combined" ] && [ -z "$TRANSFORMED_FEATURES" ]; then
  TRANSFORMED_FEATURES=$(ls -td results/feature_engineering/*/transformed_features*.h5 2>/dev/null | head -1)
  
  if [ ! -z "$TRANSFORMED_FEATURES" ]; then
    echo "Automatically selected transformed feature file: $TRANSFORMED_FEATURES"
  else
    echo "Error: No transformed feature files found for combined mode"
    exit 1
  fi
fi

# 构建命令
CMD="python scripts/stage2_baseline_training.py --model_type ${MODEL_TYPE} --feature_type ${FEATURE_TYPE} --config ${CONFIG_PATH}"

# 添加特征选择方法（如果不是"original"特征类型）
if [ "$FEATURE_TYPE" != "original" ]; then
  CMD="${CMD} --feature_selection ${FEATURE_SELECTION}"
fi

# 添加特征子集（如果指定了）
if [ ! -z "$FEATURE_SUBSET" ]; then
  CMD="${CMD} --feature_subset ${FEATURE_SUBSET}"
fi

# 添加特征文件路径
if [ ! -z "$SELECTED_FEATURES" ]; then
  CMD="${CMD} --selected_features ${SELECTED_FEATURES}"
fi

if [ ! -z "$TRANSFORMED_FEATURES" ]; then
  CMD="${CMD} --transformed_features ${TRANSFORMED_FEATURES}"
fi

# 添加输出目录（如果指定了）
if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="${CMD} --output_dir ${OUTPUT_DIR}"
fi

echo "Running command: ${CMD}"

# 运行命令并重定向输出到日志文件
nohup ${CMD} > "${LOG_FILE}" 2>&1 &

# 获取进程ID
PID=$!
echo "Process started with PID: $PID"
echo "To check progress, use: tail -f $LOG_FILE"
echo "To check if process is running, use: ps -p $PID"

# 添加指导信息，提醒用户测试集评估将在训练结束后完成
echo ""
echo "Note: Test set evaluation will be performed ONLY ONCE after training completes."
echo "      This ensures an unbiased estimation of model performance."
echo "      Look for 'Final Test Set Performance Summary' in the log file when training is done."
echo "      Detailed evaluation reports and visualizations will be saved in the output directory."
