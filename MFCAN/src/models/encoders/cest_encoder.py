import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CESTEncoder(nn.Module):
    """CEST特征编码器，处理CEST特征并提取有用表示"""
    
    def __init__(self, input_dim=116, hidden_dim=256, output_dim=128, dropout=0.5):
        """
        初始化CEST特征编码器
        
        参数:
            input_dim: 输入特征维度，CEST特征通常为116维
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super(CESTEncoder, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
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
        features = self.feature_extractor(x)
        
        if return_features:
            return features
            
        logits = self.classifier(features)
        return logits