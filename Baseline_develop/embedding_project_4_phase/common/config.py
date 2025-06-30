#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置模块
包含所有分析相关的配置参数
"""

import os
import warnings
from pathlib import Path

# 设置matplotlib后端
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class Config:
    """全局配置类"""
    
    # 数据配置
    DEFAULT_DATA_PATH = '/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38_no_label43.mat'
    DEFAULT_RANDOM_STATE = 42
    
    # 受试者分割配置
    DEFAULT_TRAIN_RANGE = (1, 31)
    DEFAULT_VAL_RANGE = (31, 38)
    DEFAULT_TEST_SUBJECT = 38
    
    # 分析配置
    MIN_VOXELS_PER_REGION = 50
    MIN_SAMPLES_FOR_ANALYSIS = 100
    MIN_SUBJECTS_FOR_ANALYSIS = 5
    MIN_SAMPLES_FOR_SUBJECT_REGION = 50  # 每个受试者-脑区组合的最小样本数
    
    # 可视化配置
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    FIGURE_SIZE_LARGE = (18, 12)
    FIGURE_SIZE_MEDIUM = (12, 8)
    FIGURE_SIZE_SMALL = (8, 6)
    
    # 深度网络配置（alex版本超参数）
    ALEX_HYPERPARAMS = {
        'batch_size': 128,
        'no_epochs': 25,
        'learning_rate': 0.00001,
        'weight_decay': 0.00001,
        'dropout_rate': 0.5
    }
    
    # 数据交换目录
    DATA_EXCHANGE_DIR = Path('./data_exchange')
    PHASE0_OUTPUT_DIR = DATA_EXCHANGE_DIR / 'phase0_output'
    PHASE1_OUTPUT_DIR = DATA_EXCHANGE_DIR / 'phase1_output'
    PHASE2_OUTPUT_DIR = DATA_EXCHANGE_DIR / 'phase2_output'
    PHASE3_OUTPUT_DIR = DATA_EXCHANGE_DIR / 'phase3_output'
    
    # 文件名配置
    TRAIN_DATA_FILE = 'train_data.npz'
    VAL_DATA_FILE = 'val_data.npz'
    TEST_DATA_FILE = 'test_data.npz'
    DATA_STATS_FILE = 'data_statistics.json'
    LABEL_MAPPING_FILE = 'label_mapping.json'
    SCALER_FILE = 'scaler.pkl'
    
    @classmethod
    def setup_matplotlib(cls):
        """设置matplotlib参数"""
        matplotlib.use('Agg', force=True)
        
        plt.rcParams.update({
            'font.size': 12,
            'figure.figsize': cls.FIGURE_SIZE_MEDIUM,
            'figure.dpi': cls.FIGURE_DPI,
            'savefig.dpi': cls.FIGURE_DPI,
            'savefig.format': cls.FIGURE_FORMAT,
            'savefig.bbox': 'tight',
            'figure.max_open_warning': 0,
            'agg.path.chunksize': 10000
        })
        
        # 中文字体支持
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
    
    @classmethod
    def get_feature_groups(cls):
        """获取特征分组配置"""
        return {
            'qti_params': list(range(0, 15)),
            'raw_b_tensors': list(range(15, 225)),
            'cest_params': list(range(225, 229)),
            'z_spectrum': list(range(229, 341))
        }
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        cls.DATA_EXCHANGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.PHASE0_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PHASE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PHASE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PHASE3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 初始化配置
Config.setup_matplotlib()
Config.ensure_directories()