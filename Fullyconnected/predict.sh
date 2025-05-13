#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 预测使用特征级标准化
python predict.py \
  --data_path "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat" \
  --features_key multidim_data \
  --region_key region \
  --output_dir "./prediction_results_feature_wise" \
  --normalize feature_wise \
  --save_3d

echo "预测完成! 结果已保存到: ./prediction_results_feature_wise"