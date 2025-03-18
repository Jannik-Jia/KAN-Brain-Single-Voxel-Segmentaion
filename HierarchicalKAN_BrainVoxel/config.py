#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件 - 包含所有全局参数和常量
"""

import os
from datetime import datetime

# 设置随机种子，确保实验可重复性
RANDOM_SEED = 666

# 数据采样参数
SAMPLE_RATIO = 0.1  # 使用10%的数据进行分析
USE_SAMPLING = True  # 是否使用数据采样

# 数据预处理参数
APPLY_PCA = True   # 是否应用PCA降维
NORM = True        # 是否对数据进行标准化/归一化处理

# 特征分组
DIFF_FEATURES = list(range(0, 15))     # 扩散特征 (1-15)
QTI_FEATURES = list(range(15, 225))    # QTI特征 (16-225)
CEST_FEATURES = list(range(225, 341))  # CEST特征 (226-341)

# 定义模型名称，用于结果保存和模型标识
MODEL_NAME = 'HierarchicalKAN_BrainVoxel'

# 大类定义 (初始设置，可能需要根据实际数据调整)
DEFAULT_NUM_BIG_CLASSES = 5  # 大类数量

# 指定数据集名称
DATASET = 'BrainVoxel'

# 训练参数
EPOCH = 50         # 总训练轮数
VAL_EPOCH = 1      # 每隔多少轮进行一次验证
LR = 0.001         # 学习率
WEIGHT_DECAY = 1e-6  # 权重衰减系数，用于L2正则化
BATCH_SIZE = 640    # 批处理大小，固定不变

# 计算设备选择
DEVICE = 0         # -1表示使用CPU，0表示使用第一块GPU(cuda:0)

# 数据参数
FEATURE_DIM = 341  # 输入特征总维度
NUM_CLASS = 102    # 细分类别数量
FIXED_GRID = 10    # 固定网格大小，不进行网格扩展

# 专家模型参数
DIFF_HIDDEN_DIM = 64    # 扩散专家隐藏层维度
QTI_HIDDEN_DIM = 128    # QTI专家隐藏层维度
CEST_HIDDEN_DIM = 64    # CEST专家隐藏层维度

# PCA参数
DIFF_PCA_COMPONENTS = 10   # 扩散特征PCA组件数
QTI_PCA_COMPONENTS = 30    # QTI特征PCA组件数
CEST_PCA_COMPONENTS = 20   # CEST特征PCA组件数

# 模型检查点路径
CHECK_POINT = None  # 加载预训练模型的路径，None表示从头开始训练

# 结果保存路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"./Results/{MODEL_NAME}/{DATASET}/{timestamp}"

# 数据目录
DATA_DIRS = {
    'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
    'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
    'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
}

# 确保保存目录存在
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)