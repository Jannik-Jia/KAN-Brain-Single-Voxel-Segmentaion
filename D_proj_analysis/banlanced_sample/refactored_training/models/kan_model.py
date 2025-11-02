#!/usr/bin/env python
# coding: utf-8

"""
BrainVoxelKAN - KAN-based Model for Brain Voxel Classification
基于KAN的大脑体素分类模型
"""

import torch.nn as nn

try:
    from fastkan import FastKAN
    FASTKAN_AVAILABLE = True
except ImportError:
    FASTKAN_AVAILABLE = False
    print("⚠️  Warning: fastkan not installed. Please install: pip install fastkan")


class BrainVoxelKAN(nn.Module):
    """
    轻量封装的 KAN 分类头

    基于FastKAN实现，用于MRI体素级别分类
    - layers_hidden = [input_dim] + hidden_dims + [num_classes]
    - 不需要PCA，直接使用原始特征
    - 支持类权重处理类别不平衡

    Parameters:
    -----------
    input_dim : int
        输入特征维度（例如42，排除第14个特征后）
    hidden_dims : int or list of int
        隐藏层维度，可以是单个int或list
        例如：64 或 [256, 128, 64]
    num_classes : int
        输出类别数（52个FreeSurfer分区，包含所有类别）
    grid_size : int
        KAN的分段网格数（默认8）
        - 8-16通常是稳妥起点
        - 越大表达力越强但更易过拟合、计算更重
    """

    def __init__(self, input_dim=42, num_classes=52, hidden_dims=None,
                 grid_size=8):
        super(BrainVoxelKAN, self).__init__()

        if not FASTKAN_AVAILABLE:
            raise ImportError(
                "fastkan is required for BrainVoxelKAN. "
                "Please install it: pip install fastkan"
            )

        # 处理hidden_dims参数
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]  # 默认三层
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # 构建层结构：[input_dim] + hidden_dims + [num_classes]
        layers = [input_dim] + list(hidden_dims) + [num_classes]

        # 创建FastKAN模型
        self.kan = FastKAN(layers_hidden=layers, num_grids=grid_size)

        # 保存配置信息
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.grid_size = grid_size

    def forward(self, x):
        """
        前向传播

        Parameters:
        -----------
        x : torch.Tensor
            输入特征，形状 (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor
            输出logits，形状 (batch_size, num_classes)
            注意：输出是logits，不是概率
            训练时配合CrossEntropyLoss使用（Loss内部会做softmax）
        """
        # x: (B, input_dim) → logits: (B, num_classes)
        return self.kan(x)

    def get_config(self):
        """获取模型配置信息"""
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_dims': self.hidden_dims,
            'grid_size': self.grid_size
        }


def get_model_config():
    """
    获取模型默认配置

    Returns:
    --------
    dict : 模型配置字典
    """
    return {
        'model_name': 'BrainVoxelKAN',
        'hidden_dims': [256, 128, 64],
        'grid_size': 8,
        'use_class_weights': True,  # 标记需要使用类权重
        'description': 'KAN-based模型，使用Kolmogorov-Arnold网络进行体素分类'
    }
