#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件，定义了模型训练和评估的默认参数
"""

import os
import json
import torch

# 基础配置
CONFIG = {
    # 随机种子和设备设置
    'random_seed': 666,
    'device': 0,  # -1表示CPU，>=0表示使用对应索引的GPU
    
    # 添加mat文件相关配置
    'mat_file_path': None,  # TRAIN38.mat文件路径
    'test_size': 0.01,      # 从训练集中分割出测试集的比例
    'demo_mat_path': None,  # 用于评估的DEMO38.mat文件路径

    # 添加患者ID配置 - 自定义数据集分割
    'dataset_split': {
        'train_patients': [28, 5, 25, 30, 34, 32, 33, 11, 12, 20, 29, 17, 37, 7, 26, 1, 36, 14, 19, 3, 35, 31, 22, 8],
        'val_patients': [4, 24, 9, 15, 16, 18, 2],
        'test_patients': [38, 6, 21, 13, 10, 23, 27]
    },

    'apply_pca': False,  # 是否应用PCA降维
    'n_pca': 0,          # PCA保留的主成分数量，0表示不进行PCA
    'norm': True,        # 是否进行数据标准化
    
    # 模型基本参数
    'model_name': 'BrainVoxel_102Class_MLP',
    'dataset_name': 'BrainVoxel',
    'feature_dim': 341,  # 原始特征维度
    'num_class': 102,    # 类别数量
    
    # 模型架构参数
    'model_type': 'base_mlp',  # 'base_mlp', 'deep_mlp', 'residual_mlp'
    'hidden_units': [4096, 4096, 4096, 4096],  # MLP隐藏层大小
    'dropout_rate': 0.5,        # Dropout率
    'activation': 'swish',       # 激活函数: 'relu', 'gelu', 'swish'
    
    # 训练参数
    'epochs': 30,        # 训练轮数
    'val_epochs': 3,     # 验证频率
    'batch_size': 128,   # 批处理大小
    'lr': 1e-5,          # 学习率
    'weight_decay': 1e-5,# 权重衰减
    
    # 学习率调度参数
    'use_lr_scheduler': True,           # 是否使用学习率调度
    'lr_scheduler_type': 'cosine',      # 'multistep', 'cosine', 'plateau'
    'lr_milestones': [10, 20],          # 多步调度的里程碑轮次
    'lr_gamma': 0.5,                    # 学习率降低的倍数因子
    
    # 优化器设置
    'optimizer': 'adamw',  # 'adam', 'adamw'
    
    # 贝叶斯优化参数
    'run_bayesian_opt': True,  # 是否运行贝叶斯优化
    'n_trials': 30,            # 贝叶斯优化的试验次数
    'pruning_patience': 5,     # 提前终止的耐心值
    
    # 结果保存
    'save_dir': './results',    # 结果保存目录
    'log_dir': './logs',        # 日志保存目录
    'save_checkpoints': True,   # 是否保存检查点
    'use_old_zipfile_serialization': True,  # 使用旧的zipfile序列化方式（兼容性更好）
}

def load_config(config_file=None):
    """
    从JSON文件加载配置，并与默认配置合并
    
    参数:
        config_file: 配置文件路径，None表示使用默认配置
    
    返回:
        config: 合并后的配置字典
    """
    config = CONFIG.copy()
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # 确保结果和日志目录存在
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    return config

def save_config(config, filepath):
    """
    保存配置到JSON文件
    
    参数:
        config: 配置字典
        filepath: 保存路径
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
    return filepath