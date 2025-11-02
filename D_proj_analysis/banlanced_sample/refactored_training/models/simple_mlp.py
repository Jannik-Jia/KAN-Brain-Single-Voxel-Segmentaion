#!/usr/bin/env python
# coding: utf-8

"""
SimpleMLP - Simple Multi-Layer Perceptron
简单的多层感知机模型
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MRI voxel classification

    简单的多层感知机
    - 3层隐藏层
    - BatchNorm + Dropout
    - 参数量较少，训练速度快
    """

    def __init__(self, input_dim=42, num_classes=52, hidden_dims=[512, 256, 128],
                 dropout_rate=0.3):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        num_classes : int
            输出类别数
        hidden_dims : list of int
            各隐藏层的维度（默认[512, 256, 128]）
        dropout_rate : float
            Dropout比率（默认0.3）
        """
        super(SimpleMLP, self).__init__()

        # 构建层序列
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

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
        """
        return self.network(x)


def get_model_config():
    """
    获取模型默认配置

    Returns:
    --------
    dict : 模型配置字典
    """
    return {
        'model_name': 'SimpleMLP',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'description': '简单多层感知机（轻量级，训练快速）'
    }
