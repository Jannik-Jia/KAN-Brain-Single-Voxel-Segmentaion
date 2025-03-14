"""
模型定义模块
"""
import torch
import torch.nn as nn
from fastkan import FastKAN

class BrainVoxelKAN(nn.Module):
    """
    用于脑体素分类的KAN模型，简化版
    """
    def __init__(self, input_dim, hidden_dim, num_classes, grid_size=3):
        """
        初始化模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数量
            grid_size: 网格大小，保持固定
        """
        super(BrainVoxelKAN, self).__init__()
        
        self.kan = FastKAN(
            layers_hidden=[input_dim, hidden_dim, num_classes],
            num_grids=grid_size
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, input_dim)
            
        返回:
            output: 模型输出，形状为(batch_size, num_classes)
        """
        return self.kan(x)
