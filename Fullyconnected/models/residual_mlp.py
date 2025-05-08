#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
带残差连接的多层感知器模型实现
"""

import torch
import torch.nn as nn

class ResidualBrainVoxelMLP(nn.Module):
    """
    带残差连接的脑体素MLP网络，支持更多配置选项
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.5, 
                 activation='relu', use_bottleneck=False, bottleneck_factor=0.5):
        """
        初始化模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如[4096, 4096, 4096, 4096]
            num_classes: 类别数量 (102)
            dropout_rate: Dropout比率
            activation: 激活函数，支持'relu', 'gelu', 'swish'
            use_bottleneck: 是否在残差块中使用bottleneck结构
            bottleneck_factor: bottleneck的压缩因子，仅在use_bottleneck=True时使用
        """
        super(ResidualBrainVoxelMLP, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'swish':
            self.act_fn = nn.SiLU()  # PyTorch中的SiLU就是Swish激活函数
        else:
            self.act_fn = nn.ReLU()  # 默认使用ReLU
        
        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_act = self.act_fn
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            # 判断是否需要维度转换
            if hidden_dims[i] != hidden_dims[i+1]:
                shortcut = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            else:
                shortcut = nn.Identity()
            
            # 主路径 - 根据是否使用bottleneck选择不同结构
            if use_bottleneck and hidden_dims[i] > 1000:  # 只在大型层使用bottleneck
                bottleneck_dim = int(hidden_dims[i] * bottleneck_factor)
                main_path = nn.Sequential(
                    nn.Linear(hidden_dims[i], bottleneck_dim),  # 下降维度
                    self.act_fn,
                    nn.Dropout(dropout_rate),
                    nn.Linear(bottleneck_dim, hidden_dims[i+1]),  # 恢复或变换维度
                    nn.Dropout(dropout_rate)
                )
            else:
                main_path = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    self.act_fn,
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                    nn.Dropout(dropout_rate)
                )
            
            self.residual_blocks.append(nn.ModuleDict({
                'main_path': main_path,
                'shortcut': shortcut
            }))
            
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
    
    # forward方法保持不变
    def forward(self, x):
        # 输入层
        x = self.input_layer(x)
        x = self.input_act(x)
        x = self.input_dropout(x)
        
        # 残差块
        for block in self.residual_blocks:
            identity = x
            x = block['main_path'](x)
            x = block['shortcut'](identity) + x  # 残差连接
            x = self.act_fn(x)  # 残差后的激活函数
        
        # 输出层
        x = self.output_layer(x)
        return x