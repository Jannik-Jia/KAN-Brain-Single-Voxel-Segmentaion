#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础多层感知器模型实现
"""

import torch
import torch.nn as nn

class BrainVoxelMLP(nn.Module):
    """
    用于脑体素分类的多层感知器模型，多分类版本
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.5, activation='relu'):
        """
        初始化模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如[4096, 4096, 4096, 4096]
            num_classes: 类别数量 (102)
            dropout_rate: Dropout比率
            activation: 激活函数，支持'relu', 'gelu', 'swish'
        """
        super(BrainVoxelMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 添加输入层到第一个隐藏层
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'swish':
            act_fn = nn.SiLU()  # PyTorch中的SiLU就是Swish激活函数
        else:
            act_fn = nn.ReLU()  # 默认使用ReLU
            
        self.layers.append(act_fn)
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 添加中间隐藏层
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(act_fn)
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 最后的分类层
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, input_dim)
            
        返回:
            output: 模型输出，形状为(batch_size, num_classes)
        """
        for layer in self.layers:
            x = layer(x)
        return x