#!/bin/bash

# 创建日志目录
mkdir -p logs/shell

# 设置默认参数
MODE="full"  # 默认运行完整训练流程
CONFIG_PATH="configs/mfcan_config.json"
DATA_PATH="/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/MFCAN/src/data/processed/reorganized_encoder_data.h5"
MODEL_PATH=""  # 可选的预训练模型路径
OUTPUT_DIR=""  # 默认使用配置文件中的设置
DO_HYPEROPT="true"  # 是否进行超参数优化
SEARCH_METHOD="grid"  # 超参数搜索方法
N_ITER="10"  # 迭代次数
USE_BALANCED_SAMPLER="false"  # 是否使用平衡采样
SAMPLES_PER_CLASS="1000"      # 每个类别采样数量
ANALYZE_GROUPS="false"        # 是否分析特征组贡献
MAX_GROUP_SIZE="3"            # 特征组分析的最大组合大小

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --balanced_sampling)
      USE_BALANCED_SAMPLER="true"
      shift
      ;;
    --samples_per_class)
      SAMPLES_PER_CLASS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --hyperopt)
      DO_HYPEROPT="true"
      shift
      ;;
    --search_method)
      SEARCH_METHOD="$2"
      shift 2
      ;;
    --n_iter)
      N_ITER="$2"
      shift 2
      ;;
    --analyze_groups)
      ANALYZE_GROUPS="true"
      shift
      ;;
    --max_group_size)
      MAX_GROUP_SIZE="$2"
      shift 2
      ;;
    *)
      echo "未知选项: $1"
      echo "可用选项: --mode, --config, --data_path, --model_path, --output_dir, --hyperopt, --search_method, --n_iter, --balanced_sampling, --samples_per_class, --analyze_groups, --max_group_size"
      exit 1
      ;;
  esac
done

# 设置日志文件路径
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/shell/stage4_mfcan_${MODE}_${TIMESTAMP}.log"

# 如果指定了超参数优化，运行优化脚本
if [ "$DO_HYPEROPT" = "true" ]; then
    echo "开始阶段4: MFCAN超参数优化..."
    echo "搜索方法: $SEARCH_METHOD"
    echo "配置文件: $CONFIG_PATH"
    echo "数据路径: $DATA_PATH"
    
    HYPEROPT_LOG_FILE="logs/shell/stage4_hyperopt_${SEARCH_METHOD}_${TIMESTAMP}.log"
    echo "日志文件: $HYPEROPT_LOG_FILE"
    
    # 设置超参数优化输出目录
    if [ -z "$OUTPUT_DIR" ]; then
        HYPEROPT_OUTPUT_DIR="results/hyperopt/${TIMESTAMP}"
    else
        HYPEROPT_OUTPUT_DIR="${OUTPUT_DIR}/hyperopt"
    fi
    
    # 构建超参数优化命令
    HYPEROPT_CMD="python scripts/hyperparameter_search.py --config ${CONFIG_PATH} --data_path ${DATA_PATH} --output_dir ${HYPEROPT_OUTPUT_DIR} --search_method ${SEARCH_METHOD} --n_iter ${N_ITER}"
    
    # 添加平衡采样参数（如果启用）
    if [ "$USE_BALANCED_SAMPLER" = "true" ]; then
        HYPEROPT_CMD="${HYPEROPT_CMD} --balanced_sampling --samples_per_class ${SAMPLES_PER_CLASS}"
    fi
    
    # 运行超参数优化脚本
    echo "运行超参数优化命令: ${HYPEROPT_CMD}"
    eval "${HYPEROPT_CMD} > ${HYPEROPT_LOG_FILE} 2>&1"
    
    # 检查是否成功
    if [ $? -ne 0 ]; then
        echo "超参数优化失败，请查看日志: ${HYPEROPT_LOG_FILE}"
        exit 1
    fi
    
    # 使用最佳配置
    BEST_CONFIG="${HYPEROPT_OUTPUT_DIR}/best_config.json"
    if [ -f "$BEST_CONFIG" ]; then
        echo "超参数优化完成，使用最佳配置: ${BEST_CONFIG}"
        CONFIG_PATH="${BEST_CONFIG}"
        
        # 如果没有指定输出目录，使用时间戳创建新目录
        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="results/mfcan/${TIMESTAMP}_best"
        fi
    else
        echo "警告: 未找到最佳配置文件，将使用原始配置"
    fi
fi

echo "开始阶段4: MFCAN训练..."
echo "训练模式: $MODE"
echo "配置文件: $CONFIG_PATH"
echo "数据路径: $DATA_PATH"
if [ ! -z "$MODEL_PATH" ]; then
  echo "预训练模型: $MODEL_PATH"
fi
echo "日志文件: $LOG_FILE"

# 检查文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件 $CONFIG_PATH 不存在"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件 $DATA_PATH 不存在"
    exit 1
fi

# 验证训练模式
if [[ "$MODE" != "full" && "$MODE" != "encoders" && "$MODE" != "fusion" && "$MODE" != "finetune" ]]; then
    echo "错误: 无效的训练模式 $MODE"
    echo "有效的模式: full, encoders, fusion, finetune"
    exit 1
fi

# 如果指定了MODEL_PATH，检查文件是否存在
if [ ! -z "$MODEL_PATH" ]; then
    if [ ! -f "$MODEL_PATH" ]; then
        echo "错误: 模型文件 $MODEL_PATH 不存在"
        exit 1
    fi
fi

# 设置输出目录（如果未指定）
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="results/mfcan/${TIMESTAMP}"
    echo "自动设置输出目录: $OUTPUT_DIR"
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python scripts/train_mfcan.py --mode ${MODE} --config ${CONFIG_PATH} --data_path ${DATA_PATH}"

# 添加模型路径（如果指定）
if [ ! -z "$MODEL_PATH" ]; then
    CMD="${CMD} --model_path ${MODEL_PATH}"
fi

# 添加输出目录
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

# 添加平衡采样参数（如果启用）
if [ "$USE_BALANCED_SAMPLER" = "true" ]; then
    CMD="${CMD} --balanced_sampling --samples_per_class ${SAMPLES_PER_CLASS}"
fi

# 添加特征组分析参数（如果启用）
if [ "$ANALYZE_GROUPS" = "true" ]; then
    CMD="${CMD} --analyze_groups --max_group_size ${MAX_GROUP_SIZE}"
    echo "将进行特征组贡献分析，最大组合大小: ${MAX_GROUP_SIZE}"
fi

echo "运行命令: ${CMD}"

# 运行命令并重定向输出到日志文件
nohup ${CMD} > "${LOG_FILE}" 2>&1 &

# 获取进程ID
PID=$!
echo "进程已启动，PID: $PID"
echo "查看进度使用: tail -f $LOG_FILE"
echo "检查进程是否运行使用: ps -p $PID"