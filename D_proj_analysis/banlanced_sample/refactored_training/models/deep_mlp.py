#!/usr/bin/env python
# coding: utf-8

"""
DeepMLP - 超深度多层感知机模型
Ultra-Deep Multi-Layer Perceptron with Advanced Features

特性:
- 残差连接 (Residual connections)
- 特征交互层 (Feature interaction layer)
- 自注意力机制 (Self-attention)
- 混合激活函数 (Mixed activation: GELU/SiLU)
- Shake-Shake正则化 (Shake-Shake regularization)
- 随机深度 (Stochastic depth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureInteractionLayer(nn.Module):
    """
    特征交互层 - 支持双线性和三线性交互
    """
    def __init__(self, input_dim, use_trilinear=True):
        super(FeatureInteractionLayer, self).__init__()
        self.input_dim = input_dim
        self.use_trilinear = use_trilinear

        # 双线性交互的输出维度
        self.bilinear_dim = input_dim * (input_dim - 1) // 2

        # 三线性交互的输出维度（如果启用）
        if use_trilinear:
            self.trilinear_dim = input_dim * (input_dim - 1) * (input_dim - 2) // 6
        else:
            self.trilinear_dim = 0

        # 总输出维度
        self.output_dim = input_dim + self.bilinear_dim + self.trilinear_dim

    def forward(self, x):
        """
        x: (B, input_dim)
        返回: (B, output_dim) - 包含原始特征、双线性交互、三线性交互
        """
        batch_size = x.size(0)

        # 原始特征
        features = [x]

        # 双线性交互: x_i * x_j for all i < j
        bilinear_features = []
        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                bilinear_features.append((x[:, i] * x[:, j]).unsqueeze(1))

        if bilinear_features:
            features.append(torch.cat(bilinear_features, dim=1))

        # 三线性交互: x_i * x_j * x_k for all i < j < k
        if self.use_trilinear:
            trilinear_features = []
            for i in range(self.input_dim):
                for j in range(i + 1, self.input_dim):
                    for k in range(j + 1, self.input_dim):
                        trilinear_features.append((x[:, i] * x[:, j] * x[:, k]).unsqueeze(1))

            if trilinear_features:
                features.append(torch.cat(trilinear_features, dim=1))

        return torch.cat(features, dim=1)


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力层
    """
    def __init__(self, dim, num_heads=16, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, dim)
        """
        B = x.size(0)

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, head_dim)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 应用注意力到V
        out = (attn @ v).transpose(1, 2).reshape(B, -1)  # (B, dim)
        out = self.proj(out)

        return out


class ResidualBlock(nn.Module):
    """
    残差块 - 带瓶颈结构
    """
    def __init__(self, dim, bottleneck_dim=None, dropout=0.3, activation='gelu', use_shake_shake=False):
        super(ResidualBlock, self).__init__()

        if bottleneck_dim is None:
            bottleneck_dim = dim // 4

        self.use_shake_shake = use_shake_shake

        # 瓶颈结构: dim -> bottleneck_dim -> dim
        self.fc1 = nn.Linear(dim, bottleneck_dim)
        self.bn1 = nn.BatchNorm1d(bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # Shake-Shake正则化（仅在训练时）
        if self.training and self.use_shake_shake:
            alpha = torch.rand(1, device=x.device)
            out = alpha * out + (1 - alpha) * identity
        else:
            out = out + identity

        out = self.activation(out)

        return out


class DeepMLP(nn.Module):
    """
    超深度MLP模型 - 集成多种先进技术

    架构特点:
    - 可选的特征交互层
    - 7层深度网络
    - 残差连接
    - 多头自注意力
    - 混合激活函数
    - Shake-Shake正则化
    - 随机深度
    """

    def __init__(
        self,
        input_dim=42,
        num_classes=52,
        hidden_dims=None,
        use_feature_interaction=True,
        use_trilinear=True,
        use_residual=True,
        use_self_attention=True,
        num_attn_heads=16,
        attn_layers=None,
        use_mixed_activation=True,
        use_shake_shake=True,
        stochastic_depth_rate=0.2,
        dropout_rate=0.3
    ):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        num_classes : int
            输出类别数
        hidden_dims : list
            隐藏层维度列表，默认 [2048, 1536, 1536, 1536, 1536, 2048, 2048]
        use_feature_interaction : bool
            是否使用特征交互层
        use_trilinear : bool
            是否使用三线性交互（仅当use_feature_interaction=True时有效）
        use_residual : bool
            是否使用残差连接
        use_self_attention : bool
            是否使用自注意力机制
        num_attn_heads : int
            自注意力头数
        attn_layers : list
            在哪些层添加自注意力，默认 [1, 3, 5]
        use_mixed_activation : bool
            是否使用混合激活函数（GELU和SiLU交替）
        use_shake_shake : bool
            是否使用Shake-Shake正则化
        stochastic_depth_rate : float
            随机深度的drop rate
        dropout_rate : float
            Dropout率
        """
        super(DeepMLP, self).__init__()

        # 默认隐藏层维度
        if hidden_dims is None:
            hidden_dims = [2048, 1536, 1536, 1536, 1536, 2048, 2048]

        # 默认自注意力层位置
        if attn_layers is None:
            attn_layers = [1, 3, 5]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.use_feature_interaction = use_feature_interaction
        self.use_residual = use_residual
        self.use_self_attention = use_self_attention
        self.stochastic_depth_rate = stochastic_depth_rate

        # 特征交互层
        if use_feature_interaction:
            self.feature_interaction = FeatureInteractionLayer(input_dim, use_trilinear)
            current_dim = self.feature_interaction.output_dim
        else:
            self.feature_interaction = None
            current_dim = input_dim

        # 输入投影层
        self.input_proj = nn.Linear(current_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_activation = nn.GELU() if use_mixed_activation else nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)

        # 构建深层网络
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleDict()

        for i, hidden_dim in enumerate(hidden_dims):
            layer_modules = nn.ModuleDict()

            # 主层（残差块或普通线性层）
            if use_residual:
                # 激活函数选择（交替使用GELU和SiLU）
                activation = 'gelu' if (not use_mixed_activation or i % 2 == 0) else 'silu'
                layer_modules['block'] = ResidualBlock(
                    hidden_dim,
                    bottleneck_dim=hidden_dim // 4,
                    dropout=dropout_rate,
                    activation=activation,
                    use_shake_shake=use_shake_shake
                )
            else:
                layer_modules['linear'] = nn.Linear(hidden_dim, hidden_dim)
                layer_modules['bn'] = nn.BatchNorm1d(hidden_dim)
                layer_modules['dropout'] = nn.Dropout(dropout_rate)

            self.layers.append(layer_modules)

            # 自注意力层
            if use_self_attention and i in attn_layers:
                self.attention_layers[f'attn_{i}'] = MultiHeadSelfAttention(
                    hidden_dim,
                    num_heads=num_attn_heads,
                    dropout=dropout_rate
                )

            # LayerNorm
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 输出层
        self.output_fc = nn.Linear(hidden_dims[-1], num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, input_dim)
        返回: (B, num_classes)
        """
        # 特征交互
        if self.feature_interaction is not None:
            x = self.feature_interaction(x)

        # 输入投影
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)

        # 通过深层网络
        for i, layer_modules in enumerate(self.layers):
            # 随机深度（训练时随机跳过某些层）
            if self.training and self.stochastic_depth_rate > 0:
                if torch.rand(1).item() < self.stochastic_depth_rate * (i / len(self.layers)):
                    continue  # 跳过这一层

            # 主层
            if 'block' in layer_modules:
                x = layer_modules['block'](x)
            else:
                identity = x
                x = layer_modules['linear'](x)
                x = layer_modules['bn'](x)
                x = F.relu(x)
                x = layer_modules['dropout'](x)
                x = x + identity  # 手动残差连接

            # 自注意力
            attn_key = f'attn_{i}'
            if attn_key in self.attention_layers:
                attn_out = self.attention_layers[attn_key](x)
                x = x + attn_out  # 残差连接

            # LayerNorm
            x = self.layer_norms[i](x)

        # 输出
        logits = self.output_fc(x)

        return logits


def get_model_config():
    """
    获取DeepMLP模型的默认配置
    """
    return {
        'model_name': 'DeepMLP',
        'hidden_dims': [2048, 1536, 1536, 1536, 1536, 2048, 2048],
        'use_feature_interaction': True,
        'use_trilinear': True,
        'use_residual': True,
        'use_self_attention': True,
        'num_attn_heads': 16,
        'attn_layers': [1, 3, 5],
        'use_mixed_activation': True,
        'use_shake_shake': True,
        'stochastic_depth_rate': 0.2,
        'dropout_rate': 0.3,
        'use_class_weights': True,  # 使用类权重
        'description': '超深度MLP - 集成特征交互、残差连接、自注意力等先进技术'
    }


# 测试代码
if __name__ == '__main__':
    # 创建模型
    model = DeepMLP(input_dim=42, num_classes=52)

    # 打印模型信息
    print("=" * 80)
    print("DeepMLP 模型测试")
    print("=" * 80)
    print(f"\n模型结构:\n{model}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # 测试前向传播
    batch_size = 16
    x = torch.randn(batch_size, 42)

    print(f"\n输入形状: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"输出形状: {output.shape}")
    print(f"输出logits范围: [{output.min():.4f}, {output.max():.4f}]")

    # 测试softmax
    probs = torch.softmax(output, dim=1)
    print(f"Softmax概率和: {probs.sum(dim=1).mean():.4f} (应该接近1.0)")

    print("\n✅ DeepMLP模型测试通过！")
