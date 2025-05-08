#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深层多层感知器模型实现
"""

import torch
import torch.nn as nn
from .base_mlp import BrainVoxelMLP

class DeepMLP(nn.Module):
    """
    深层MLP变体，支持不同数量的层和逐渐变窄或变宽的架构
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.5, 
                 activation='relu', use_skip_connections=False):
        """
        初始化模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量
            dropout_rate: Dropout比率
            activation: 激活函数，支持'relu', 'gelu', 'swish'
            use_skip_connections: 是否在深层网络中添加跳跃连接
        """
        super(DeepMLP, self).__init__()
        
        self.use_skip_connections = use_skip_connections
        
        # 选择激活函数
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'swish':
            self.act_fn = nn.SiLU()  # PyTorch中的SiLU就是Swish激活函数
        else:
            self.act_fn = nn.ReLU()  # 默认使用ReLU
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.skip_adapters = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(self.act_fn)
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 中间隐藏层
        for i in range(len(hidden_dims) - 1):
            # 主路径层
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(self.act_fn)
            self.layers.append(nn.Dropout(dropout_rate))
            
            # 跳跃连接适配器 - 如果维度不同，需要一个线性变换
            if use_skip_connections:
                if i > 0 and hidden_dims[i-1] != hidden_dims[i+1]:
                    self.skip_adapters.append(nn.Linear(hidden_dims[i-1], hidden_dims[i+1]))
                else:
                    self.skip_adapters.append(nn.Identity())
        
        # 最后的分类层
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
    
    def forward(self, x):
        """
        前向传播，包含可选的跳跃连接
        """
        # 输入层处理
        x = self.layers[0](x)  # 线性层
        x = self.layers[1](x)  # 激活
        x = self.layers[2](x)  # dropout
        
        layer_outputs = [x]  # 保存中间层输出用于跳跃连接
        
        # 处理中间隐藏层
        layer_idx = 3
        skip_idx = 0
        
        for i in range(len(self.layers) // 3 - 1):  # 每三个为一组(linear, act, dropout)
            # 获取当前层的输入
            current_input = x
            
            # 处理当前块
            x = self.layers[layer_idx](x)    # Linear
            x = self.layers[layer_idx+1](x)  # Activation
            x = self.layers[layer_idx+2](x)  # Dropout
            
            # 添加跳跃连接（如果启用）
            if self.use_skip_connections and i > 0:
                skip_connection = self.skip_adapters[skip_idx](layer_outputs[-2])
                x = x + skip_connection
                skip_idx += 1
            
            layer_outputs.append(x)
            layer_idx += 3
        
        # 输出层
        x = self.layers[-1](x)
        
        return x