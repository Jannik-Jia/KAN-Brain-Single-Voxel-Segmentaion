import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffusionEncoder(nn.Module):
    """扩散特征编码器，处理扩散特征并提取有用表示"""
    
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=32, dropout=0.1):
        """
        初始化扩散特征编码器
        
        参数:
            input_dim: 输入特征维度，扩散特征通常为15维
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super(DiffusionEncoder, self).__init__()
        
        # 特征映射层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # 辅助分类器，用于预训练
        self.classifier = nn.Linear(output_dim, 102)  # 假设有102个类别
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        参数:
            x: 输入特征
            return_features: 是否返回特征而不是分类结果
            
        返回:
            如果return_features为True，则返回特征表示
            否则返回分类logits
        """
        embedded = self.embedding(x)
        features = self.feature_extraction(embedded)
        
        if return_features:
            return features
            
        logits = self.classifier(features)
        return logits