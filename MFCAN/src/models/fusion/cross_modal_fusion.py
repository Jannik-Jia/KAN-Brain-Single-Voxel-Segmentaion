import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    """交叉模态融合层，捕获不同模态之间的相互关系和互补信息"""
    
    def __init__(self, modal_dims=[32, 128, 128], fusion_dim=384):
        """
        初始化交叉模态融合层
        
        参数:
            modal_dims: 三个模态特征的维度列表 [diffusion_dim, qti_dim, cest_dim]
            fusion_dim: 融合后的特征维度
        """
        super(CrossModalFusion, self).__init__()
        
        self.modal_names = ['diffusion', 'qti', 'cest']
        self.fusion_dim = fusion_dim
        
        # 交叉注意力机制 - 每对模态之间的互相影响
        self.cross_attention = nn.ModuleDict()
        for src in self.modal_names:
            for tgt in self.modal_names:
                if src != tgt:
                    src_idx = self.modal_names.index(src)
                    tgt_idx = self.modal_names.index(tgt)
                    self.cross_attention[f"{src}_{tgt}"] = nn.Sequential(
                        nn.Linear(modal_dims[src_idx], modal_dims[tgt_idx]),
                        nn.Sigmoid()
                    )
        
        # 融合后的特征转换
        self.fusion_transform = nn.Sequential(
            nn.Linear(sum(modal_dims), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 残差连接 (可选)
        if sum(modal_dims) == fusion_dim:
            self.use_residual = True
        else:
            self.use_residual = False
            self.residual_projection = nn.Linear(sum(modal_dims), fusion_dim)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        前向传播
        
        参数:
            features: 包含三个模态特征的字典 {'diffusion': tensor, 'qti': tensor, 'cest': tensor}
            
        返回:
            fused_features: 融合后的特征 [batch_size, fusion_dim]
        """
        # 交叉注意力增强
        enhanced_features = {}
        for tgt in self.modal_names:
            enhanced_features[tgt] = features[tgt].clone()  # 避免原地修改
            
            # 其他模态对当前模态的影响
            for src in self.modal_names:
                if src != tgt:
                    # 源模态对目标模态的注意力权重
                    attention = self.cross_attention[f"{src}_{tgt}"](features[src])
                    # 注意力加权
                    enhanced_features[tgt] = enhanced_features[tgt] * attention
        
        # 连接增强后的特征
        concatenated = torch.cat([enhanced_features[modal] for modal in self.modal_names], dim=1)
        
        # 转换融合特征
        fused_feature = self.fusion_transform(concatenated)
        
        # 应用残差连接 (如果维度匹配)
        if self.use_residual:
            fused_feature = fused_feature + concatenated
        elif hasattr(self, 'residual_projection'):
            # 如果维度不匹配，使用线性投影
            fused_feature = fused_feature + self.residual_projection(concatenated)
        
        return fused_feature