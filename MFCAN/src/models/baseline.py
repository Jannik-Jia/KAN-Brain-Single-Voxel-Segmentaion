import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.logging_utils import Logger

class BaselineMLP(nn.Module):
    """简单的多层感知机基线模型"""
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        """
        初始化基线MLP模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，如[512, 256, 128]
            num_classes: 输出类别数
            dropout_rate: Dropout比率
        """
        super(BaselineMLP, self).__init__()
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
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
    
    def forward(self, x):
        """前向传播"""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

class FeatureGroupMLP(nn.Module):
    """分特征组的MLP模型"""
    
    def __init__(self, group_dims, hidden_dims, num_classes, fusion_method='concat', dropout_rate=0.3):
        """
        初始化分组MLP模型
        
        参数:
            group_dims: 字典，键为特征组名称，值为该组特征维度
            hidden_dims: 各特征组的隐藏层维度字典
            num_classes: 输出类别数
            fusion_method: 特征融合方法，'concat'或'attention'
            dropout_rate: Dropout比率
        """
        super(FeatureGroupMLP, self).__init__()
        
        self.group_names = list(group_dims.keys())
        self.fusion_method = fusion_method
        
        # 为每个特征组创建特征提取器
        self.feature_extractors = nn.ModuleDict()
        self.feature_dims = {}
        
        for group_name, input_dim in group_dims.items():
            layers = []
            prev_dim = input_dim
            
            for i, dim in enumerate(hidden_dims.get(group_name, [128, 64])):
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = dim
            
            self.feature_extractors[group_name] = nn.Sequential(*layers)
            self.feature_dims[group_name] = prev_dim
        
        # 特征融合层
        if fusion_method == 'concat':
            fusion_dim = sum(self.feature_dims.values())
            self.fusion_layer = nn.Identity()
        elif fusion_method == 'attention':
            # 实现注意力融合
            fusion_dim = max(self.feature_dims.values())
            self.group_projections = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim) 
                for name, dim in self.feature_dims.items()
            })
            self.attention_weights = nn.Parameter(torch.ones(len(group_dims), fusion_dim) / len(group_dims))
            self.fusion_layer = self._attention_fusion
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, num_classes)
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _attention_fusion(self, features_dict):
        """注意力融合机制"""
        # 投影所有特征到相同维度
        projected_features = {
            name: self.group_projections[name](features)
            for name, features in features_dict.items()
        }
        
        # 应用注意力权重
        attention_weights = F.softmax(self.attention_weights, dim=0)  # [num_groups, fusion_dim]
        
        # 融合特征
        fused_features = None
        for i, name in enumerate(self.group_names):
            weighted_feature = projected_features[name] * attention_weights[i]
            if fused_features is None:
                fused_features = weighted_feature
            else:
                fused_features += weighted_feature
                
        return fused_features
    
    def forward(self, x_dict):
        """
        前向传播
        
        参数:
            x_dict: 字典，键为特征组名称，值为该组特征
            
        返回:
            logits: 分类logits
        """
        # 提取各组特征
        features_dict = {}
        for name in self.group_names:
            if name in x_dict:
                features_dict[name] = self.feature_extractors[name](x_dict[name])
        
        # 特征融合
        if self.fusion_method == 'concat':
            # 拼接所有特征
            concatenated_features = torch.cat([features_dict[name] for name in self.group_names], dim=1)
            fused_features = self.fusion_layer(concatenated_features)
        else:  # 'attention'
            fused_features = self.fusion_layer(features_dict)
        
        # 分类
        logits = self.classifier(fused_features)
        return logits

# 创建增强版的深度MLP模型
class DeepMLP(nn.Module):
    """增强版深度MLP模型，支持残差连接、自注意力和特征交互"""
    
    def __init__(self, input_dim, hidden_dims, num_classes, 
                 use_residual=True, use_self_attention=True, use_feature_interaction=True,
                 dropout_rates=None, num_attn_heads=8, attn_layers=None):
        """
        初始化增强版DeepMLP模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 输出类别数
            use_residual: 是否使用残差连接
            use_self_attention: 是否使用自注意力
            use_feature_interaction: 是否使用特征交互
            dropout_rates: 各层的dropout率，默认为None（所有层使用0.3）
            num_attn_heads: 注意力头数
            attn_layers: 使用注意力的层索引列表
        """
        super(DeepMLP, self).__init__()
        log_manager = Logger("DeepMLP", log_dir="logs/models")
        self.logger = log_manager.get_logger()

        self.use_residual = use_residual
        self.use_self_attention = use_self_attention
        self.use_feature_interaction = use_feature_interaction
        
        # 设置dropout率
        if dropout_rates is None:
            dropout_rates = [0.3] * len(hidden_dims)
        elif len(dropout_rates) < len(hidden_dims):
            # 如果dropout_rates长度不够，扩展它
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(hidden_dims) - len(dropout_rates))
        
        # 设置使用注意力的层
        if attn_layers is None and use_self_attention:
            # 默认在1/3和2/3处添加注意力层
            attn_layers = [len(hidden_dims) // 3, 2 * len(hidden_dims) // 3]
        elif not use_self_attention:
            attn_layers = []
            
        # 构建网络层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            # 添加线性层
            self.layers.append(nn.Linear(prev_dim, dim))
            
            # 添加BN和激活函数
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rates[i])
            ))
            
            # 如果使用残差连接，并且维度匹配或可以投影
            if use_residual and i > 0:
                # 修改这部分逻辑
                prev_dim = hidden_dims[i-1]  # 前一层的维度
                current_dim = dim  # 当前层的维度
                
                if prev_dim == current_dim:
                    # 维度相同，直接使用恒等残差连接
                    self.layers.append(ResidualConnection(current_dim))
                else:
                    # 维度不同，需要投影
                    self.layers.append(ResidualConnection(current_dim, prev_dim))
                    

            
            # 如果当前层需要添加注意力
            if use_self_attention and i in attn_layers:
                self.layers.append(SelfAttention(dim, num_attn_heads))
            
            # 如果需要特征交互且不是最后一层
            if use_feature_interaction and i < len(hidden_dims) - 1:
                self.layers.append(FeatureInteraction(dim))
            
            prev_dim = dim
        
        # 分类器
        self.classifier = nn.Linear(prev_dim, num_classes)
        
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
 
    def forward(self, x):
        """前向传播"""
        previous_output = None
        layer_outputs = {}
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ResidualConnection):
                # 为残差连接传递前一个线性层的输出
                if previous_output is not None:
                    # 确保添加日志来调试维度问题
                    if i > 1:  # 避免在第一层打印
                        self.logger.info(f"Layer {i}: x shape={x.shape}, previous_output shape={previous_output.shape}")
                    x = layer(x, previous_output)
                else:
                    x = layer(x)
            else:
                x = layer(x)
                # 如果是线性层，记录其输出用于残差连接
                if isinstance(layer, nn.Linear):
                    previous_output = x
        
        # 注意这里的缩进修正 - 在循环外调用分类器
        logits = self.classifier(x)
        return logits

class ResidualConnection(nn.Module):
    """残差连接模块"""
    
    def __init__(self, dim, input_dim=None):
        """
        初始化残差连接
        
        参数:
            dim: 输出维度
            input_dim: 输入维度，如果与输出维度不同需要投影
        """
        super(ResidualConnection, self).__init__()
        
        self.needs_projection = input_dim is not None and input_dim != dim
        
        if self.needs_projection:
            self.projection = nn.Linear(input_dim, dim)

    def forward(self, x, residual=None):
        """
        前向传播
        
        参数:
            x: 当前特征
            residual: 残差特征，默认为None（使用x作为残差）
        """
        if residual is None:
            residual = x
            
        if self.needs_projection:
            residual = self.projection(residual)
            
        return x + residual


class SelfAttention(nn.Module):
    """多头自注意力模块"""
    
    def __init__(self, dim, num_heads=8):
        """
        初始化自注意力模块
        
        参数:
            dim: 特征维度
            num_heads: 注意力头数
        """
        super(SelfAttention, self).__init__()
        
        assert dim % num_heads == 0, f"维度 {dim} 必须能被注意力头数 {num_heads} 整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV投影
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, feature_dim]
        """
        # 调整形状以适应注意力机制
        batch_size = x.shape[0]
        
        # 添加一个虚拟的序列维度
        x_seq = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 第一个残差块 - 自注意力
        residual = x_seq
        
        # 层归一化
        x_norm = self.norm1(x_seq)
        
        # 计算QKV
        q = self.query(x_norm).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        k = self.key(x_norm).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)    # [batch, num_heads, 1, head_dim]
        v = self.value(x_norm).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [batch, num_heads, 1, 1]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(batch_size, 1, self.dim)  # [batch, 1, dim]
        out = self.proj(out)
        
        # 第一个残差连接
        out = out + residual
        
        # 第二个残差块 - MLP
        residual = out
        out = self.norm2(out)
        out = self.mlp(out)
        out = out + residual
        
        # 去掉序列维度
        return out.squeeze(1)  # [batch, dim]


class FeatureInteraction(nn.Module):
    """特征交互模块，用于捕获特征之间的相互关系"""
    
    def __init__(self, dim, reduction_ratio=8):
        """
        初始化特征交互模块
        
        参数:
            dim: 特征维度
            reduction_ratio: 降维比例，用于减少参数数量
        """
        super(FeatureInteraction, self).__init__()
        
        # 全局平均池化已经隐含在操作中
        # 降维层
        self.fc1 = nn.Linear(dim, dim // reduction_ratio)
        # 升维层
        self.fc2 = nn.Linear(dim // reduction_ratio, dim)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, feature_dim]
        """
        # 第一步：全局特征提取
        pooled = x  # 这里我们直接使用特征，因为已经是全局表示
        
        # 第二步：通过两层FC实现特征交互和重要性评估
        pooled = self.fc1(pooled)
        pooled = self.relu(pooled)
        pooled = self.fc2(pooled)
        pooled = self.sigmoid(pooled)
        
        # 第三步：重新标定特征重要性
        return x * pooled