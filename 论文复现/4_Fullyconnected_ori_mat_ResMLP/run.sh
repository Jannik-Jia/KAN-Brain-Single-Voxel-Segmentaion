#!/bin/bash

# 101标签数据集专用运行脚本
# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "脑体素分类训练脚本 - 101标签数据集版本"
echo "数据集: TRAIN38_no_label43.mat"
echo "标签范围: 0-100 (背景0 + 有效标签1-100)"
echo "自动模式: 将测试所有架构(base_mlp, deep_mlp, residual_mlp)"
echo "=========================================="

# 标准化方法选择
echo "请选择标准化方法:"
echo "1) Patientwise标准化 (推荐)"
echo "2) Global标准化"
read -p "请输入选择 (1/2, 默认1): " std_choice

case $std_choice in
    2)
        STD_METHOD="global"
        STD_SUFFIX="_Global"
        ;;
    *)
        STD_METHOD="patientwise"
        STD_SUFFIX="_Patientwise"
        ;;
esac

# 背景处理选项
echo ""
echo "请选择背景处理模式:"
echo "1) 背景作为分类类别 (推荐，101个类别)"
echo "2) 过滤背景 (100个类别)"
read -p "请输入选择 (1/2, 默认1): " bg_choice

case $bg_choice in
    2)
        BG_PARAMS="--filter_background True --num_class 100"
        EXPERIMENT_SUFFIX="${STD_SUFFIX}_BGFilter100"
        ;;
    *)
        BG_PARAMS="--filter_background False --include_background_in_classes True --num_class 101"
        EXPERIMENT_SUFFIX="${STD_SUFFIX}_BGClass101"
        ;;
esac

# Patient-aware采样器选项
if [ "$STD_METHOD" = "patientwise" ]; then
    echo ""
    read -p "是否使用Patient-aware批次采样器? (y/n, 默认y): " sampler_choice
    
    if [[ $sampler_choice =~ ^[Nn]$ ]]; then
        SAMPLER_PARAMS=""
    else
        SAMPLER_PARAMS="--use_patient_aware_sampler --min_patients_per_batch 3"
    fi
fi

# 训练参数
echo ""
echo "设置优化参数:"
read -p "贝叶斯优化试验次数 (默认50，将自动分配给3种架构): " n_trials
read -p "最终训练轮数 (默认30): " epochs
read -p "批次大小 (默认128): " batch_size

n_trials=${n_trials:-50}
epochs=${epochs:-30}
batch_size=${batch_size:-128}

# 实验名称
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX}_AutoSelect"
EXPERIMENT_NAME="BrainVoxel_101Labels${EXPERIMENT_SUFFIX}_$(date +%Y%m%d_%H%M%S)"

# 运行命令
echo ""
echo "=========================================="
echo "实验配置摘要:"
echo "- 自动测试3种架构: base_mlp, deep_mlp, residual_mlp"
echo "- 贝叶斯优化试验: ${n_trials}次"
echo "- 每种架构至少测试: $((n_trials/3))次"
echo "- 最终选择: 验证集F1分数最高的架构和参数"
echo "=========================================="
read -p "确认执行? (y/n): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    # 运行贝叶斯优化，自动选择最佳架构
    nohup python -u main.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --mat_file_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38_no_label43.mat" \
        --standardization_method "$STD_METHOD" \
        --val_prob_idx 20 \
        --test_prob_idx 38 \
        --batch_size "$batch_size" \
        --epochs "$epochs" \
        --device 0 \
        --save_dir ./results \
        --log_dir ./logs \
        --run_bayesian_opt \
        --n_trials "$n_trials" \
        $BG_PARAMS \
        $SAMPLER_PARAMS \
        > "logs/${EXPERIMENT_NAME}.log" 2>&1 &
    PID=$!
    
    echo ""
    echo "=========================================="
    echo "实验启动成功!"
    echo "实验名称: $EXPERIMENT_NAME"
    echo "进程ID: $PID"
    echo "标准化方法: $STD_METHOD"
    echo "类别数: $([ "$bg_choice" = "2" ] && echo "100" || echo "101")"
    echo "贝叶斯优化: ${n_trials}次试验"
    echo "=========================================="
    echo ""
    echo "监控命令:"
    echo "tail -f logs/${EXPERIMENT_NAME}.log"
    echo ""
    echo "查看架构分布:"
    echo "grep '当前架构分布情况' -A 5 logs/${EXPERIMENT_NAME}.log"
    echo ""
    
    # 等待并显示初始日志
    sleep 3
    echo "初始日志:"
    echo "----------------------------------------"
    head -n 30 logs/${EXPERIMENT_NAME}.log 2>/dev/null || echo "日志生成中..."
else
    echo "已取消执行"
fi