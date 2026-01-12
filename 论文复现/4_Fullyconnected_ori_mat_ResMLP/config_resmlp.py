#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Residual MLP 4x4096 配置文件
固定配置，用于 Alex7T 数据集的体素分类任务
"""

import os

# ==================== 基础配置 ====================
CONFIG = {
    # 随机种子
    'random_seed': 666,
    'device': 0,  # -1 表示 CPU，>=0 表示对应 GPU

    # ==================== 数据配置 ====================
    # 1D 和 3D 数据目录
    'data_dir_1d': "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D",
    'data_dir_3d': "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated",

    # 测试被试名称
    'test_subject': "YHC_10_lncguay",

    # 特征维度
    'feature_dim': 341,
    'num_class': 102,  # Alex7T 102 类

    # 标准化
    'use_zscore': True,  # Per-Patient Z-Score

    # ==================== 模型配置 (固定 Residual MLP 4x4096) ====================
    'model_type': 'residual_mlp',
    'hidden_units': [4096, 4096, 4096, 4096],  # 4x4096
    'activation': 'relu',  # 原始 ResMLP 使用 ReLU
    'dropout_rate': 0.5,   # 原始配置

    # ResMLP 特有参数
    'use_bottleneck': False,
    'bottleneck_factor': 0.5,

    # ==================== 训练配置 ====================
    'epochs': 30,
    'batch_size': 128,
    'lr': 1e-5,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',

    # 学习率调度
    'use_lr_scheduler': True,
    'lr_scheduler_type': 'cosine',

    # ==================== 保存配置 ====================
    'save_dir': './results',
    'log_dir': './logs',

    # ==================== Experiment JSON 配置 ====================
    'experiment_id': 'alex_resmlp_4x4096_patientwise_v1',
    'method_name': 'Residual MLP 4x4096 (ReLU, Per-Patient Z-Score)',
    'method_key': 'resmlp_4x4096_patientwise',
    'family': 'mlp',
    'subfamily': 'resmlp_4x4096',
}


def load_config(config_file=None):
    """加载配置"""
    config = CONFIG.copy()

    # 从 JSON 文件更新配置（如果提供）
    if config_file and os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)

    # 确保目录存在
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    return config


def save_config(config, filepath):
    """保存配置到 JSON 文件"""
    import json

    config_to_save = {k: v for k, v in config.items()
                      if not callable(v) and k not in ['scaler']}

    with open(filepath, 'w') as f:
        json.dump(config_to_save, f, indent=4)

    return filepath


def print_config(config):
    """打印配置信息"""
    print("\n" + "=" * 60)
    print("Residual MLP 4x4096 配置")
    print("=" * 60)
    print(f"模型: {config['model_type']}")
    print(f"隐藏层: {config['hidden_units']}")
    print(f"激活函数: {config['activation']}")
    print(f"Dropout: {config['dropout_rate']}")
    print(f"Bottleneck: {config.get('use_bottleneck', False)}")
    print(f"学习率: {config['lr']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"测试被试: {config['test_subject']}")
    print(f"标准化: {'Per-Patient Z-Score' if config['use_zscore'] else '无'}")
    print("=" * 60 + "\n")
