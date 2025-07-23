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

# 基础训练参数
echo ""
echo "设置训练参数:"
read -p "训练轮数 (默认25): " epochs
read -p "批次大小 (默认128): " batch_size
read -p "学习率 (默认1e-5): " lr

epochs=${epochs:-25}
batch_size=${batch_size:-128}
lr=${lr:-1e-5}

# 实验名称
EXPERIMENT_NAME="BrainVoxel_101Labels${EXPERIMENT_SUFFIX}_$(date +%Y%m%d_%H%M%S)"

# 运行命令
CMD="python -u main.py \
    --experiment_name $EXPERIMENT_NAME \
    --mat_file_path '/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38_no_label43.mat' \
    --standardization_method $STD_METHOD \
    --val_prob_idx 20 \
    --test_prob_idx 38 \
    --batch_size $batch_size \
    --epochs $epochs \
    --device 0 \
    --model_type base_mlp \
    --hidden_units '4096,4096,4096,4096' \
    --activation relu \
    --dropout_rate 0.5 \
    --lr $lr \
    --weight_decay 1e-5 \
    --optimizer adam \
    --use_lr_scheduler \
    --lr_scheduler_type cosine \
    --save_dir ./results \
    --log_dir ./logs \
    $BG_PARAMS \
    $SAMPLER_PARAMS"

echo ""
echo "即将执行命令:"
echo "$CMD"
echo ""
read -p "确认执行? (y/n): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    nohup $CMD > logs/${EXPERIMENT_NAME}.log 2>&1 &
    PID=$!
    
    echo ""
    echo "=========================================="
    echo "实验启动成功!"
    echo "实验名称: $EXPERIMENT_NAME"
    echo "进程ID: $PID"
    echo "标准化方法: $STD_METHOD"
    echo "类别数: $([ "$bg_choice" = "2" ] && echo "100" || echo "101")"
    echo "=========================================="
    echo ""
    echo "监控命令:"
    echo "tail -f logs/${EXPERIMENT_NAME}.log"
    echo ""
    
    # 等待并显示初始日志
    sleep 3
    echo "初始日志:"
    echo "----------------------------------------"
    head -n 30 logs/${EXPERIMENT_NAME}.log 2>/dev/null || echo "日志生成中..."
else
    echo "已取消执行"
fi