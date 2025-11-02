#!/usr/bin/env python
# coding: utf-8

"""
ResNetMLP - ResNet-style MLP with Residual Connections
带残差连接的全连接神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, hidden_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))

        out += identity  # 残差连接
        out = F.relu(out)

        return out


class ResNetMLP(nn.Module):
    """
    ResNet-style MLP for MRI voxel classification

    带残差连接的全连接神经网络
    - 使用ResidualBlock
    - BatchNorm + Dropout
    - 残差连接帮助训练更深的网络
    """

    def __init__(self, input_dim=42, num_classes=52, hidden_dim=2048,
                 num_residual_blocks=3, dropout_rate=0.3):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        num_classes : int
            输出类别数
        hidden_dim : int
            隐藏层维度（默认2048）
        num_residual_blocks : int
            残差块数量（默认3）
        dropout_rate : float
            Dropout比率（默认0.3）
        """
        super(ResNetMLP, self).__init__()

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # 残差块序列
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_residual_blocks)
        ])

        # 输出层
        self.output = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

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
        # 输入投影
        x = F.relu(self.input_bn(self.input_proj(x)))
        x = self.dropout(x)

        # 通过残差块
        for block in self.residual_blocks:
            x = block(x)

        # 输出层
        x = self.output(x)

        return x


def get_model_config():
    """
    获取模型默认配置

    Returns:
    --------
    dict : 模型配置字典
    """
    return {
        'model_name': 'ResNetMLP',
        'hidden_dim': 2048,
        'num_residual_blocks': 3,
        'dropout_rate': 0.3,
        'description': '带残差连接的全连接神经网络（ResNet-style）'
    }
