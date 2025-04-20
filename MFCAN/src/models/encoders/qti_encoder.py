import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QTIEncoder(nn.Module):
    """QTI特征编码器，处理QTI特征并提取有用表示"""
    
    def __init__(self, input_dim=210, hidden_dims=[512, 256], output_dim=128, dropout=0.3):
        """
        初始化QTI特征编码器
        
        参数:
            input_dim: 输入特征维度，QTI特征通常为210维
            hidden_dims: 隐藏层维度列表
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super(QTIEncoder, self).__init__()
        
        # 构建多层特征提取网络
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(*layers)
        
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
        features = self.feature_extractor(x)
        
        if return_features:
            return features
            
        logits = self.classifier(features)
        return logits