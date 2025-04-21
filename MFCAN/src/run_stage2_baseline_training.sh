#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置默认参数
MODEL_TYPE="deep_mlp"  # 可选: mlp, group_mlp, deep_mlp
FEATURE_TYPE="selected"  # 可选: original, selected, pca, combined
FEATURE_SELECTION="rf"  # 可选: importance, nonredundant, rf
NUM_RUNS=3  # 默认运行次数，用于获取更稳定的结果

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
    --num_runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 设置最新特征工程结果目录
FEATURE_ENG_DIR=$(ls -td results/feature_engineering/20* | head -1)
echo "Using feature engineering results from: $FEATURE_ENG_DIR"

# 检查选定的特征选择方法是否有效
IMPORTANCE_FILE="$FEATURE_ENG_DIR/importance/all_feature_ranking.csv"
if [ ! -f "$IMPORTANCE_FILE" ]; then
  echo "Warning: Feature importance file not found at $IMPORTANCE_FILE"
  echo "Will use default feature selection from the H5 file"
fi

# 设置模型参数组合 (针对DeepMLP的超参数调优)
if [ "$MODEL_TYPE" = "deep_mlp" ]; then
  # 创建参数组合数组
  declare -a PARAM_COMBINATIONS=(
    # 参数格式: "hidden_dims dropout_rate num_attn_heads"
    "1024,512,256,128 0.3 12"     # 默认配置
    "2048,1024,512,256 0.4 16"    # 更深更宽的网络
    "1024,1024,512,512,256 0.5 8" # 更多层，更强正则化
  )
else
  # 对于其他模型，只使用一组默认参数
  declare -a PARAM_COMBINATIONS=(
    "default"
  )
fi

# 创建实验结果摘要文件
RESULTS_SUMMARY="$FEATURE_ENG_DIR/baseline_experiments_summary.csv"
echo "model_type,feature_type,feature_selection,params,run,accuracy,f1_score,timestamp" > "$RESULTS_SUMMARY"

# 运行每种参数组合
for PARAMS in "${PARAM_COMBINATIONS[@]}"; do
  # 对每种参数组合运行多次以获得稳定结果
  for RUN in $(seq 1 $NUM_RUNS); do
    # 设置日志文件路径
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    if [ "$PARAMS" = "default" ]; then
      PARAM_STR="default"
      CONFIG_PATH="configs/base_config.json"
    else
      # 解析参数
      IFS=' ' read -r HIDDEN_DIMS DROPOUT NUM_HEADS <<< "$PARAMS"
      PARAM_STR="${HIDDEN_DIMS//,/_}_d${DROPOUT}_h${NUM_HEADS}"
      
      # 创建临时配置文件
      CONFIG_PATH="configs/temp_${MODEL_TYPE}_${PARAM_STR}.json"
      
      # 基于基础配置创建临时配置
      cp configs/base_config.json "$CONFIG_PATH"
      
      # 修改配置文件中的参数 (基于jq或sed，这里用sed简化处理)
      if [[ "$HIDDEN_DIMS" =~ ^[0-9,]+$ ]]; then
        HIDDEN_DIMS_JSON="[$(echo $HIDDEN_DIMS | sed 's/,/,/g')]"
        sed -i "s/\"hidden_dims\": \[[0-9, ]*\]/\"hidden_dims\": $HIDDEN_DIMS_JSON/g" "$CONFIG_PATH"
      fi
      sed -i "s/\"dropout_rate\": [0-9.]\+/\"dropout_rate\": $DROPOUT/g" "$CONFIG_PATH"
      sed -i "s/\"num_attn_heads\": [0-9]\+/\"num_attn_heads\": $NUM_HEADS/g" "$CONFIG_PATH"
    fi
    
    LOG_FILE="logs/shell/stage2_baseline_${MODEL_TYPE}_${FEATURE_TYPE}_${FEATURE_SELECTION}_${PARAM_STR}_run${RUN}_${TIMESTAMP}.log"
    
    echo "Starting run $RUN/$NUM_RUNS with parameters: $PARAM_STR"
    echo "Model Type: $MODEL_TYPE"
    echo "Feature Type: $FEATURE_TYPE"
    echo "Feature Selection: $FEATURE_SELECTION"
    echo "Log file: $LOG_FILE"
    
    # 设置特征文件路径参数
    if [ "$FEATURE_TYPE" = "selected" ]; then
      FEATURE_PARAMS="--selected_features $FEATURE_ENG_DIR/selected_features.h5 --feature_selection $FEATURE_SELECTION"
    elif [ "$FEATURE_TYPE" = "pca" ]; then
      FEATURE_PARAMS="--transformed_features $FEATURE_ENG_DIR/transformed_features.h5"
    elif [ "$FEATURE_TYPE" = "combined" ]; then
      FEATURE_PARAMS="--selected_features $FEATURE_ENG_DIR/selected_features.h5 --transformed_features $FEATURE_ENG_DIR/transformed_features.h5 --feature_selection $FEATURE_SELECTION"
    else
      FEATURE_PARAMS=""
    fi
    
    # 运行阶段二基线训练脚本
    python scripts/stage2_baseline_training.py \
        --config "$CONFIG_PATH" \
        --model_type $MODEL_TYPE \
        --feature_type $FEATURE_TYPE \
        $FEATURE_PARAMS \
        --output_dir "results/baseline/${MODEL_TYPE}_${FEATURE_TYPE}_${FEATURE_SELECTION}_${PARAM_STR}_${TIMESTAMP}" \
        > "$LOG_FILE" 2>&1 &
    
    # 获取进程ID
    PID=$!
    echo "Process started with PID: $PID"
    
    # 等待进程完成
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
      echo "Run completed successfully"
      
      # 提取结果并添加到摘要文件
      ACCURACY=$(grep -o "准确率: [0-9.]\+" "$LOG_FILE" | tail -1 | awk '{print $2}')
      F1_SCORE=$(grep -o "F1分数: [0-9.]\+" "$LOG_FILE" | tail -1 | awk '{print $2}')
      
      if [ -n "$ACCURACY" ] && [ -n "$F1_SCORE" ]; then
        echo "$MODEL_TYPE,$FEATURE_TYPE,$FEATURE_SELECTION,$PARAM_STR,$RUN,$ACCURACY,$F1_SCORE,$TIMESTAMP" >> "$RESULTS_SUMMARY"
        echo "Results recorded: Accuracy=$ACCURACY, F1=$F1_SCORE"
      else
        echo "Could not extract results from log file"
      fi
    else
      echo "Run failed with exit code $EXIT_CODE"
    fi
    
    # 如果不是默认配置，删除临时配置文件
    if [ "$PARAMS" != "default" ]; then
      rm -f "$CONFIG_PATH"
    fi
    
    echo "-----------------------------------"
  done
done

echo "All experiments completed."
echo "Results summary saved to: $RESULTS_SUMMARY"

# 分析最佳模型
echo "Analyzing best models..."
echo "Top 3 models by accuracy:"
sort -t, -k6,6nr "$RESULTS_SUMMARY" | head -4 | column -t -s,

echo "Complete!"