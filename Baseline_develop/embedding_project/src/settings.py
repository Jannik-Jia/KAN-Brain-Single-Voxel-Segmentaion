#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
包含所有分析相关的配置参数
"""

import os
import warnings

# 🔥 修复：在导入matplotlib之前设置环境变量
os.environ['MPLBACKEND'] = 'Agg'  # 确保非交互式后端

import matplotlib
matplotlib.use('Agg', force=True)  # 强制设置非交互式后端

import matplotlib.pyplot as plt
import numpy as np

# 忽略matplotlib相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class Config:
    """配置类"""
    
    # 数据配置
    DEFAULT_DATA_PATH = '/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat'
    DEFAULT_RANDOM_STATE = 42
    
    # 受试者分割配置
    DEFAULT_TRAIN_RANGE = (1, 31)
    DEFAULT_VAL_RANGE = (31, 38)
    DEFAULT_TEST_SUBJECT = 38
    
    # 分析配置
    MIN_VOXELS_PER_REGION = 50  # 每个脑区的最少体素数
    MIN_SAMPLES_FOR_ANALYSIS = 100  # 分析所需的最少样本数
    MIN_SUBJECTS_FOR_ANALYSIS = 5  # 分析所需的最少受试者数
    
    # 可视化配置
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    FIGURE_SIZE_LARGE = (18, 12)
    FIGURE_SIZE_MEDIUM = (12, 8)
    FIGURE_SIZE_SMALL = (8, 6)
    
    # 聚类配置
    MAX_CLUSTERS = 8
    MIN_CLUSTER_SIZE = 3
    
    # Embedding配置
    MIN_EMBEDDING_DIM = 16
    MAX_EMBEDDING_DIM = 128
    DEFAULT_EMBEDDING_DIM = 64
    
    # 性能配置
    MAX_SAMPLES_FOR_EFFICIENCY = 5000  # 为提高效率限制的最大样本数
    
    @classmethod
    def setup_matplotlib(cls):
        """设置matplotlib参数"""
        # 确保后端正确设置
        matplotlib.use('Agg', force=True)
        
        # 设置基本参数
        plt.rcParams.update({
            'font.size': 12,
            'figure.figsize': cls.FIGURE_SIZE_MEDIUM,
            'figure.dpi': cls.FIGURE_DPI,
            'savefig.dpi': cls.FIGURE_DPI,
            'savefig.format': cls.FIGURE_FORMAT,
            'savefig.bbox': 'tight',
            'figure.max_open_warning': 0,  # 禁用过多图形警告
            'agg.path.chunksize': 10000    # 提高大图性能
        })
        
        # 中文字体支持（如果需要）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass  # 如果没有中文字体，使用默认字体
    
    @classmethod
    def get_feature_groups(cls):
        """获取特征分组配置"""
        return {
            'qti_params': list(range(0, 15)),           # QTI参数 (15个)
            'raw_b_tensors': list(range(15, 225)),      # 原始b-tensor值 (210个)
            'cest_params': list(range(225, 229)),       # CEST参数 (4个)
            'z_spectrum': list(range(229, 341))         # Z-spectrum值 (112个)
        }
    
    @classmethod
    def get_decision_weights(cls):
        """获取决策权重配置"""
        return {
            'difference_significance': 0.15,
            'pattern_linearity': 0.15,
            'subject_separability': 0.15,
            'class_consistency': 0.10,
            'sample_adequacy': 0.10,
            'embedding_feasibility': 0.15,
            'intrinsic_dimensionality': 0.10,
            'high_dim_performance': 0.10
        }
    
    @classmethod
    def get_embedding_thresholds(cls):
        """获取embedding阈值配置"""
        return {
            'high_necessity': 0.7,
            'medium_necessity': 0.5,
            'low_necessity': 0.3,
            'min_improvement': 0.02,
            'significant_improvement': 0.05
        }

    @classmethod
    def check_environment(cls):
        """检查运行环境"""
        env_info = {
            'matplotlib_backend': matplotlib.get_backend(),
            'display_available': 'DISPLAY' in os.environ,
            'ssh_connection': 'SSH_CONNECTION' in os.environ,
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'None')
        }
        return env_info


# 初始化matplotlib配置
Config.setup_matplotlib()