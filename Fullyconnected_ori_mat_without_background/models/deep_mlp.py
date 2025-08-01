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
        super(DeepMLP, self).__init__()
        
        self.use_skip_connections = use_skip_connections
        self.input_dim = input_dim  # 保存以供get_model_info使用
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        # 选择激活函数
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'swish':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.ReLU()
            
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
            
            # 跳跃连接适配器 - 修复：正确计算源层和目标层的维度
            if use_skip_connections and i > 0:
                # 从i-1层跳到i+1层
                source_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
                target_dim = hidden_dims[i+1]
                
                if source_dim != target_dim:
                    self.skip_adapters.append(nn.Linear(source_dim, target_dim))
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
        
        num_hidden_layers = (len(self.layers) - 4) // 3  # 减去输入层(3个)和输出层(1个)
        
        for i in range(num_hidden_layers):
            # 保存当前层输入
            current_input = x
            
            # 处理当前块
            x = self.layers[layer_idx](x)    # Linear
            x = self.layers[layer_idx+1](x)  # Activation
            x = self.layers[layer_idx+2](x)  # Dropout
            
            # 添加跳跃连接（如果启用）
            if self.use_skip_connections and i > 0 and skip_idx < len(self.skip_adapters):
                # 使用适配器调整维度
                skip_connection = self.skip_adapters[skip_idx](layer_outputs[-1])
                x = x + skip_connection
                skip_idx += 1
            
            layer_outputs.append(x)
            layer_idx += 3
        
        # 输出层
        x = self.layers[-1](x)
        
        return x
    


    # deep_mlp.py
    def get_model_info(self):
        """获取模型的架构信息"""
        return {
            'model_type': 'deep_mlp',
            'input_dim': self.input_dim if hasattr(self, 'input_dim') else self.layers[0].in_features,
            'hidden_dims': self.hidden_dims if hasattr(self, 'hidden_dims') else [l.out_features for l in self.layers[::3] if isinstance(l, nn.Linear)][:-1],
            'num_classes': self.num_classes if hasattr(self, 'num_classes') else self.layers[-1].out_features,
            'dropout_rate': self.dropout_rate if hasattr(self, 'dropout_rate') else 0.5,
            'activation': self.activation_name if hasattr(self, 'activation_name') else 'relu',
            'use_skip_connections': self.use_skip_connections,
            'num_layers': len([l for l in self.layers if isinstance(l, nn.Linear)]),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

