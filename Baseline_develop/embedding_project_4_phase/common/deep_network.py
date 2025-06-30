#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度网络模块
定义4×4096深度网络架构（alex版本）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from common.config import Config

logger = logging.getLogger(__name__)


class RegModel(nn.Module):
    """4×4096深度网络模型（严格对应alex版本）"""
    
    def __init__(self, input_dim: int = 341, num_classes: int = 101):
        super(RegModel, self).__init__()
        
        # 严格对应alex的TensorFlow版本的Dense层
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, num_classes)
        
        # 严格使用alex版本的dropout率
        self.dropout = nn.Dropout(Config.ALEX_HYPERPARAMS['dropout_rate'])
    
    def forward(self, x):
        # 严格按照alex的TensorFlow模型的结构
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # 不应用softmax，让CrossEntropyLoss处理
        return x


class DeepNetworkUtils:
    """深度网络工具类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"深度网络将使用设备: {self.device}")
        
        # 设置随机种子确保可重现性
        self._set_random_seeds()
    
    def _set_random_seeds(self, seed: int = 42):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_network(self, input_dim: int = 341, num_classes: Optional[int] = None) -> RegModel:
        """
        创建深度网络
        
        Args:
            input_dim: 输入维度
            num_classes: 类别数（如果为None，将自动检测）
            
        Returns:
            网络模型
        """
        if num_classes is None:
            num_classes = 101  # 默认值（标签范围0-101）
        
        model = RegModel(input_dim=input_dim, num_classes=num_classes).to(self.device)
        logger.info(f"创建4×4096深度网络: {input_dim} → 4096×4 → {num_classes}")
        
        return model
    
    def kernel_l2_regularization(self, model: nn.Module, weight_decay: float) -> torch.Tensor:
        """
        L2正则化（只对权重矩阵）
        
        Args:
            model: 模型
            weight_decay: 权重衰减系数
            
        Returns:
            L2正则化损失
        """
        l2_reg = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                l2_reg += torch.norm(param, p=2) ** 2
        return weight_decay * l2_reg
    
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                    is_training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        准备PyTorch数据
        
        Args:
            X: 特征数据
            y: 标签数据
            is_training: 是否为训练模式
            
        Returns:
            准备好的张量数据
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if is_training and y is not None:
            # 处理one-hot编码
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_indices = np.argmax(y, axis=1)
            else:
                y_indices = y.astype(int)
            
            y_tensor = torch.LongTensor(y_indices).to(self.device)
            return X_tensor, y_tensor
        else:
            return X_tensor, None