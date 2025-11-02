#!/usr/bin/env python
# coding: utf-8

"""
RegModel - Deep Fully Connected Network
原始全连接神经网络模型（Alex identical structure）
"""

import torch.nn as nn
import torch.nn.functional as F


class RegModel(nn.Module):
    """
    Deep fully connected network for MRI voxel classification (Alex identical structure)

    深度全连接神经网络，用于MRI体素分类
    - 4层隐藏层，每层4096神经元
    - ReLU激活函数 + Dropout
    - 无BatchNorm
    """

    def __init__(self, input_dim=42, num_classes=52, hidden_dim=4096,
                 num_hidden_layers=4, dropout_rate=0.5):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度（默认42，排除第14个特征后）
        num_classes : int
            输出类别数（默认52个FreeSurfer分区）
        hidden_dim : int
            隐藏层神经元数（默认4096）
        num_hidden_layers : int
            隐藏层数量（默认4）
        dropout_rate : float
            Dropout比率（默认0.5）
        """
        super(RegModel, self).__init__()

        # 完全对应Alex的结构：无BatchNorm，使用单独的层定义
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_classes)
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
        # 严格按照Alex的前向传播结构
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # 输出层无激活函数
        return x


def get_model_config():
    """
    获取模型默认配置

    Returns:
    --------
    dict : 模型配置字典
    """
    return {
        'model_name': 'RegModel',
        'hidden_dim': 4096,
        'num_hidden_layers': 4,
        'dropout_rate': 0.5,
        'description': '深度全连接神经网络（Alex identical structure）'
    }
