#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件，定义了模型训练和评估的默认参数（支持两种数据格式）
"""

import os
import json
import torch

# 基础配置
CONFIG = {
    # 随机种子和设备设置
    'random_seed': 666,
    'device': 0,  # -1表示CPU，>=0表示使用对应索引的GPU
    
    # 数据路径和处理设置 - 支持两种格式
    # 方式1：原版分散文件格式
    'data_dirs': {
        'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
        'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
        'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
    },
    
    # 方式2：MAT文件格式（如果设置了mat_file_path，将优先使用这种方式）
    'mat_file_path': None,  # 例如: "/path/to/TRAIN38.mat"
    'test_size': 0.01,      # 从训练集中分割出测试集的比例
    'demo_mat_path': None,  # 用于评估的DEMO38.mat文件路径
    
    # MAT文件的患者ID配置 - 自定义数据集分割
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
    
    # 标签处理配置
    'label_format': 'auto',  # 'original' (1-102, 背景-1), 'mat' (0-101, 背景0), 'auto' (自动检测)
    
    # 模型架构参数
    'model_type': 'base_mlp',  # 'base_mlp', 'deep_mlp', 'residual_mlp'
    'hidden_units': [4096, 4096, 4096, 4096],  # MLP隐藏层大小
    'dropout_rate': 0.5,        # Dropout率
    'activation': 'relu',       # 激活函数: 'relu', 'gelu', 'swish'
    
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
    
    # 数据格式自适应逻辑
    _auto_detect_data_format(config)
    
    return config


def _auto_detect_data_format(config):
    """
    自动检测数据格式并设置相应参数
    """
    # 如果设置了mat_file_path且文件存在，优先使用MAT格式
    if config.get('mat_file_path') and os.path.exists(config['mat_file_path']):
        print(f"检测到MAT文件: {config['mat_file_path']}")
        print("将使用MAT数据加载格式")
        
        # 确保MAT格式的相关配置
        if config.get('label_format') == 'auto':
            config['label_format'] = 'mat'
        
        # 简化方案：不使用ignore_index
        config['ignore_index'] = None  # 修改：不忽略任何标签
        
    # 否则检查是否有传统的数据目录
    elif config.get('data_dirs') and all(
        config['data_dirs'].get(key) and os.path.exists(config['data_dirs'][key]) 
        for key in ['train_dir', 'test_dir', 'val_dir']
    ):
        print("检测到传统数据目录格式")
        print("将使用原版数据加载格式")
        
        # 确保原版格式的相关配置
        if config.get('label_format') == 'auto':
            config['label_format'] = 'original'
        
        # 简化方案：不使用ignore_index
        config['ignore_index'] = None  # 修改：不忽略任何标签
        
    else:
        print("警告: 未检测到有效的数据源配置")
        print("请设置 'mat_file_path' 或完整的 'data_dirs'")
    
    # 修改：更新标签格式说明
    print("标签格式说明:")
    print('- 原始数据: 0=背景, 1-102=有效类别')
    print('- 处理后数据: 1-102=有效类别 (背景已过滤)')  
    print('- 训练时: 0-101=有效类别 (映射后)')
    print("- 训练时: 0=背景, 1-101=有效类别 (所有类别都参与训练)")
    print("- 模型输出: 102个类别 (0-101)")

# def _auto_detect_data_format(config):
#     """
#     自动检测数据格式并设置相应参数
#     """
#     # 如果设置了mat_file_path且文件存在，优先使用MAT格式
#     if config.get('mat_file_path') and os.path.exists(config['mat_file_path']):
#         print(f"检测到MAT文件: {config['mat_file_path']}")
#         print("将使用MAT数据加载格式")
        
#         # 确保MAT格式的相关配置
#         if config.get('label_format') == 'auto':
#             config['label_format'] = 'mat'
            
#         # 两种格式都使用相同的背景标签索引
#         config['ignore_index'] = -1  # 训练时背景标签都是-1
        
#     # 否则检查是否有传统的数据目录
#     elif config.get('data_dirs') and all(
#         config['data_dirs'].get(key) and os.path.exists(config['data_dirs'][key]) 
#         for key in ['train_dir', 'test_dir', 'val_dir']
#     ):
#         print("检测到传统数据目录格式")
#         print("将使用原版数据加载格式")
        
#         # 确保原版格式的相关配置
#         if config.get('label_format') == 'auto':
#             config['label_format'] = 'original'
            
#         # 设置背景标签索引
#         config['ignore_index'] = -1  # 原版格式背景标签也是-1
        
#     else:
#         print("警告: 未检测到有效的数据源配置")
#         print("请设置 'mat_file_path' 或完整的 'data_dirs'")
        
#     # 补充说明标签格式
#     print("标签格式说明:")
#     print("- 数据中: 0=背景, 1-102=有效类别")
#     print("- 训练时: -1=背景(忽略), 0-101=有效类别")

def save_config(config, filepath):
    """
    保存配置到JSON文件
    
    参数:
        config: 配置字典
        filepath: 保存路径
    """
    # 创建一个副本，移除不可序列化的对象
    config_to_save = config.copy()
    
    # 移除可能存在的不可序列化对象
    if 'scaler' in config_to_save:
        del config_to_save['scaler']
    if 'pca_model' in config_to_save:
        del config_to_save['pca_model']
    
    with open(filepath, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    
    return filepath

def get_data_format(config):
    """
    获取当前配置使用的数据格式
    
    返回:
        'mat' 或 'original'
    """
    if config.get('mat_file_path') and os.path.exists(config['mat_file_path']):
        return 'mat'
    elif config.get('data_dirs') and all(config['data_dirs'].values()):
        return 'original'
    else:
        return 'unknown'

def is_mat_format(config):
    """
    检查是否使用MAT格式
    """
    return get_data_format(config) == 'mat'

def is_original_format(config):
    """
    检查是否使用原版格式
    """
    return get_data_format(config) == 'original'