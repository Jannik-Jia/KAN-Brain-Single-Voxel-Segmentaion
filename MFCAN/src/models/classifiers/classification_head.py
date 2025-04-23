import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    分类头，将融合特征映射到类别空间，提供分类决策和类别概率
    """
    
    def __init__(self, input_dim=384, hidden_dim=512, num_classes=102, dropout=0.4):
        """
        初始化分类头
        
        参数:
            input_dim: 输入特征维度，默认384
            hidden_dim: 隐藏层维度，默认512
            num_classes: 类别数量，默认102
            dropout: Dropout比率，默认0.4
        """
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # 降低第二层的dropout率
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            
        返回:
            logits: 分类logits [batch_size, num_classes]
        """
        return self.classifier(x)