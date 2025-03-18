#!/bin/bash
# 脑体素分层分类项目 - 实验批量运行脚本

# 设置日期时间作为运行标识
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_PREFIX="BrainVoxel_Exp_${TIMESTAMP}"
LOG_DIR="output/logs"
LOG_MASTER="${LOG_DIR}/${BASE_PREFIX}_master.log"

# 确保输出目录存在
mkdir -p output/logs
mkdir -p output/figures
mkdir -p output/results
mkdir -p output/models

# 打印启动信息
echo "============================================================" | tee -a ${LOG_MASTER}
echo "脑体素分层分类项目 - 批量实验开始: $(date)" | tee -a ${LOG_MASTER}
echo "基础前缀: ${BASE_PREFIX}" | tee -a ${LOG_MASTER}
echo "主日志文件: ${LOG_MASTER}" | tee -a ${LOG_MASTER}
echo "============================================================" | tee -a ${LOG_MASTER}

# 定义实验设置
declare -a NORMALIZE_METHODS=("robust" "standard" "none")
declare -a PCA_OPTIONS=("--pca" "--no_pca")
declare -a FEATURE_SELECTION_OPTIONS=("--feature_selection" "--no_feature_selection")
declare -a DATA_SUBSETS=("val")

# 运行特定实验脚本
echo "运行特征选择实验..." | tee -a ${LOG_MASTER}
EXP1_LOG="${LOG_DIR}/${BASE_PREFIX}_feature_selection.log"
nohup python experiments/feature_selection_exp.py > ${EXP1_LOG} 2>&1 &
PID1=$!
echo "特征选择实验已启动，PID: $PID1，日志: ${EXP1_LOG}" | tee -a ${LOG_MASTER}

echo "运行聚类实验..." | tee -a ${LOG_MASTER}
EXP2_LOG="${LOG_DIR}/${BASE_PREFIX}_clustering.log"
nohup python experiments/clustering_exp.py > ${EXP2_LOG} 2>&1 &
PID2=$!
echo "聚类实验已启动，PID: $PID2，日志: ${EXP2_LOG}" | tee -a ${LOG_MASTER}

echo "运行全特征分析实验..." | tee -a ${LOG_MASTER}
EXP3_LOG="${LOG_DIR}/${BASE_PREFIX}_all_features.log"
nohup python experiments/all_features_exp.py > ${EXP3_LOG} 2>&1 &
PID3=$!
echo "全特征分析实验已启动，PID: $PID3，日志: ${EXP3_LOG}" | tee -a ${LOG_MASTER}

# 运行参数网格实验
echo "开始参数网格实验..." | tee -a ${LOG_MASTER}
EXP_COUNT=1

for SUBSET in "${DATA_SUBSETS[@]}"; do
  for NORM in "${NORMALIZE_METHODS[@]}"; do
    for PCA_OPT in "${PCA_OPTIONS[@]}"; do
      for FS_OPT in "${FEATURE_SELECTION_OPTIONS[@]}"; do
        # 设置实验标识符
        EXP_ID="exp${EXP_COUNT}_${SUBSET}_${NORM}_${PCA_OPT//--/}_${FS_OPT//--/}"
        EXP_PREFIX="${BASE_PREFIX}_${EXP_ID}"
        EXP_LOG="${LOG_DIR}/${EXP_PREFIX}.log"
        
        echo "启动实验 $EXP_COUNT: $EXP_ID" | tee -a ${LOG_MASTER}
        echo "  参数: --data_subset=${SUBSET} --normalize=${NORM} ${PCA_OPT} ${FS_OPT}" | tee -a ${LOG_MASTER}
        
        # 运行实验
        nohup python main.py \
          --data_subset=${SUBSET} \
          --normalize=${NORM} \
          ${PCA_OPT} \
          ${FS_OPT} \
          --min_clusters=2 \
          --max_clusters=10 \
          --output_prefix=${EXP_PREFIX} \
          --skip_plots \
          > ${EXP_LOG} 2>&1 &
        
        PID=$!
        echo "  进程已启动，PID: $PID，日志: ${EXP_LOG}" | tee -a ${LOG_MASTER}
        echo $PID > "${EXP_PREFIX}.pid"
        
        # 增加计数器
        EXP_COUNT=$((EXP_COUNT+1))
        
        # 避免同时启动太多进程，等待一下
        sleep 2
      done
    done
  done
done

# 运行比较实验结果的脚本
echo "启动实验结果比较分析..." | tee -a ${LOG_MASTER}
COMPARE_LOG="${LOG_DIR}/${BASE_PREFIX}_comparison.log"
nohup python run_experiments.py > ${COMPARE_LOG} 2>&1 &
PID_COMPARE=$!
echo "比较分析已启动，PID: $PID_COMPARE，日志: ${COMPARE_LOG}" | tee -a ${LOG_MASTER}

echo "============================================================" | tee -a ${LOG_MASTER}
echo "共启动 $((EXP_COUNT-1)) 个参数网格实验和 4 个特定实验" | tee -a ${LOG_MASTER}
echo "所有实验都在后台运行。" | tee -a ${LOG_MASTER}
echo "查看实验进度: tail -f ${LOG_MASTER}" | tee -a ${LOG_MASTER}
echo "查看具体实验: tail -f ${LOG_DIR}/${BASE_PREFIX}_*.log" | tee -a ${LOG_MASTER}
echo "============================================================" | tee -a ${LOG_MASTER}