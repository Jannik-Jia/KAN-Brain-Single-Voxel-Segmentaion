import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassConditionedAttention(nn.Module):
    """
    类别条件注意力机制，根据类别先验信息动态调整不同模态的重要性
    """
    
    def __init__(self, feature_dim=320, num_classes=102):
        """
        初始化类别条件注意力模块
        
        参数:
            feature_dim: 最终特征维度，默认320
            num_classes: 类别数量，默认102
        """
        super(ClassConditionedAttention, self).__init__()
        
        # 类别嵌入层
        self.class_embedding = nn.Embedding(num_classes, 128)
        
        # 注意力生成网络
        self.attention_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3个模态的注意力权重
        )
        
        # 模态特定转换层 - 将各模态特征转换为统一维度
        self.modal_transforms = nn.ModuleDict({
            'diffusion': nn.Linear(32, feature_dim // 3),
            'qti': nn.Linear(128, feature_dim // 3),
            'cest': nn.Linear(128, feature_dim // 3)
        })
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, class_priors=None):
        """
        前向传播
        
        参数:
            features: 包含三个模态特征的字典 {'diffusion': tensor, 'qti': tensor, 'cest': tensor}
            class_priors: 可选的类别先验信息 [batch_size]
            
        返回:
            attended_features: 注意力加权后的特征 [batch_size, feature_dim]
            attention_weights: 每个模态的注意力权重 [batch_size, 3]
        """
        batch_size = features['diffusion'].size(0)
        device = features['diffusion'].device
        
        # 如果没有提供类别先验，使用均匀分布
        if class_priors is None:
            attention_weights = torch.ones(batch_size, 3, device=device)
            attention_weights = F.softmax(attention_weights, dim=1)
        else:
            # 获取类别嵌入
            class_embeds = self.class_embedding(class_priors)
            
            # 生成注意力权重
            attention_weights = self.attention_net(class_embeds)
            attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, 3]
        
        # 转换每个模态的特征
        transformed_features = {
            'diffusion': self.modal_transforms['diffusion'](features['diffusion']),
            'qti': self.modal_transforms['qti'](features['qti']),
            'cest': self.modal_transforms['cest'](features['cest'])
        }
        
        # 应用注意力权重
        modal_features = []
        for i, modal_name in enumerate(['diffusion', 'qti', 'cest']):
            weighted_feature = transformed_features[modal_name] * attention_weights[:, i].unsqueeze(1)
            modal_features.append(weighted_feature)
        
        # 合并特征
        attended_features = torch.cat(modal_features, dim=1)
        
        return attended_features, attention_weights