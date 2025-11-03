#!/usr/bin/env python
# coding: utf-8

"""
测试DeepMLP修复是否正确
"""

import torch
import numpy as np
from models.deep_mlp import DeepMLP

print("=" * 80)
print("测试DeepMLP维度修复")
print("=" * 80)

# 测试配置
input_dim = 41  # 实际数据的输入维度
num_classes = 52
batch_size = 16

# 创建模型（使用默认的不同维度配置）
model = DeepMLP(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_dims=[2048, 1536, 1536, 1536, 1536, 2048, 2048],
    use_feature_interaction=True,
    use_trilinear=True,
    use_residual=True,
    use_self_attention=True
)

print(f"\n模型配置:")
print(f"  输入维度: {input_dim}")
print(f"  输出类别: {num_classes}")
print(f"  隐藏层维度: [2048, 1536, 1536, 1536, 1536, 2048, 2048]")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"  总参数量: {total_params:,}")

# 测试前向传播
print(f"\n测试前向传播:")
x = torch.randn(batch_size, input_dim)
print(f"  输入形状: {x.shape}")

model.eval()
try:
    with torch.no_grad():
        output = model(x)
    print(f"  输出形状: {output.shape}")
    print(f"  ✅ 前向传播成功！")

    # 测试softmax
    probs = torch.softmax(output, dim=1)
    print(f"  Softmax概率和: {probs.sum(dim=1).mean():.4f} (应该接近1.0)")

except Exception as e:
    print(f"  ❌ 前向传播失败: {e}")
    raise

# 测试训练模式（包含随机深度）
print(f"\n测试训练模式（含随机深度）:")
model.train()
try:
    for i in range(5):
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        loss = output.sum()  # 模拟损失
        loss.backward()
    print(f"  ✅ 训练模式测试通过（5次迭代）")
except Exception as e:
    print(f"  ❌ 训练模式失败: {e}")
    raise

print("\n" + "=" * 80)
print("✅ 所有测试通过！DeepMLP已修复")
print("=" * 80)
