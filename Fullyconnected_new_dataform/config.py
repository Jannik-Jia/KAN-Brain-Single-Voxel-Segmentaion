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
    
    # 数据路径和处理设置
    'data_dirs': {
        'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
        'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
        'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
    },

    # 患者数据设置
    'use_patient_based_loading': True,  # 是否使用基于患者ID的数据加载
    'patient_data_base_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/reorganized_fold_data",  # 重组数据的基础目录
    'fixed_patient_split': {  # 固定的患者分组
        'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'valid': [21, 22, 23, 24, 25, 26, 27, 28], 
        'test': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    },
    'test_patient_id': 38,  # 默认测试患者ID
    
    # 数据处理参数
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

    # Early Stopping相关配置项
    'bo_max_epochs': 100,         # 贝叶斯优化中的最大训练轮数
    'early_stop_patience': 10,    # 早停的耐心值
    'early_stop_min_delta': 0.001, # 性能改进最小阈值
    'save_trial_checkpoints': True, # 是否保存每个trial的最佳检查点
    'pruner_type': 'hyperband',    # pruner类型: 'median', 'hyperband'

    # Early Stopping和优化配置
    'bo_early_stopping': {
        'patience': 10,             # 早停轮数
        'min_delta': 0.001,         # 最小改进阈值
        'restart_from_best': True,  # 如果性能下降是否回到最佳权重
    },

    # 模型部署配置
    'deployment': {
        'prepare_deployment': True,   # 是否准备部署版本
        'export_onnx': True,          # 是否导出ONNX
        'export_torchscript': True,   # 是否导出TorchScript
        'quantize': False,            # 是否量化模型（可选）
        'create_serving_scripts': True, # 是否创建服务部署脚本
        'optimize_for_inference': True, # 是否优化模型以加速推理
        'model_metadata': {
            'creator': 'BrainVoxel BO Framework',
            'version': '1.0.0',
            'description': '脑体素分类器',
            'license': 'Private',
        }
    },

    # 生产环境配置
    'production': {
        'batch_inference': True,       # 是否支持批处理推理
        'preprocessing_pipeline': True, # 是否包含预处理管道
        'max_batch_size': 64,          # 最大批处理大小
        'timeout_ms': 100,             # 推理超时（毫秒）
    },
    
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