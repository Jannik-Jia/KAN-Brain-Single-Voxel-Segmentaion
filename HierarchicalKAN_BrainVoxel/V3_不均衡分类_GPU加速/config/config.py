#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脑体素分层分类项目的配置文件
"""

import os
import torch
import numpy as np
from datetime import datetime

# 基本路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# 确保输出目录存在
for dir_path in [OUTPUT_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 随机种子，确保实验可重复性
RANDOM_SEED = 666

# 数据相关配置
DATA_DIRS = {
    'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
    'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
    'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
}

# 数据采样参数
SAMPLE_RATIO = 0.1  # 使用10%的数据进行分析（仅用于快速实验）
USE_SAMPLING = False  # 默认不使用数据采样，使用全部数据

# 数据预处理参数
NORMALIZATION_METHOD = 'robust'  # 'standard', 'robust', 或 'none'

# 特征分组
FEATURE_GROUPS = {
    'diffusion': list(range(0, 15)),     # 扩散特征 (0-14)
    'qti': list(range(15, 225)),         # QTI特征 (15-224)
    'cest': list(range(225, 341)),       # CEST特征 (225-340)
    'all_features': list(range(0, 341))  # 全部特征 (0-340)
}

# PCA参数
PCA_CONFIG = {
    'diffusion': {'apply': True, 'n_components': 10},
    'qti': {'apply': True, 'n_components': 30},
    'cest': {'apply': True, 'n_components': 20},
    'all_features': {'apply': True, 'n_components': 60}
}

# 特征选择参数
FEATURE_SELECTION = {
    'apply': True,
    'n_features': {
        'diffusion': 10,
        'qti': 30,
        'cest': 20,
        'all_features': 60
    },
    'method': 'f_classif'  # 'f_classif', 'mutual_info', 'chi2'
}

# 聚类分析参数
CLUSTERING = {
    'methods': ['kmeans', 'spectral', 'agglomerative'],
    'min_clusters': 2,
    'max_clusters': 10
}

# 模型参数
NUM_CLASS = 102    # 细分类别数量
DEFAULT_NUM_BIG_CLASSES = 7  # 默认大类数量

# 模型名称
PROJECT_NAME = 'BrainVoxel_Hierarchical'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{PROJECT_NAME}_{TIMESTAMP}"

# 计算设备选择
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型评估参数
EVALUATION = {
    'classification': {
        'cv_folds': 5,
        'metrics': ['accuracy', 'balanced_accuracy', 'f1_weighted', 'kappa']
    },
    'clustering': {
        'metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    }
}

# 可视化参数
VISUALIZATION = {
    'dpi': 300,
    'fig_width': 12,
    'fig_height': 8,
    'font_size': 12,
    'color_palette': 'viridis',
    'save_format': 'png'
}

# 日志配置
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = os.path.join(LOGS_DIR, f"{RUN_ID}.log")

# 预定义大类
DEFAULT_BIG_CLASS_NAMES = [
    "脑室系统 (Ventricular System)",     # 0
    "白质 (White Matter)",              # 1
    "灰质-皮层 (Cortical Gray Matter)", # 2
    "深部灰质核团 (Deep Gray Nuclei)",   # 3
    "边缘系统 (Limbic System)",         # 4
    "脑干 (Brain Stem)",               # 5
    "其他结构 (Other Structures)"       # 6
]
