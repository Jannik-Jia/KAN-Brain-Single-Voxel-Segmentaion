#!/usr/bin/env python
# coding: utf-8

"""
KAN模型测试脚本
用于验证KAN模型是否正确安装和工作
"""

import torch
import numpy as np

print("=" * 80)
print("KAN模型测试脚本")
print("=" * 80)

# 步骤1: 检查fastkan是否安装
print("\n步骤1: 检查fastkan库...")
try:
    from fastkan import FastKAN
    print("✅ fastkan已安装")
except ImportError:
    print("❌ fastkan未安装")
    print("请运行: pip install fastkan")
    exit(1)

# 步骤2: 导入KAN模型
print("\n步骤2: 导入BrainVoxelKAN模型...")
try:
    from models.kan_model import BrainVoxelKAN
    print("✅ BrainVoxelKAN模型导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# 步骤3: 创建模型
print("\n步骤3: 创建KAN模型实例...")
try:
    model = BrainVoxelKAN(
        input_dim=42,
        num_classes=52,
        hidden_dims=[256, 128, 64],
        grid_size=8
    )
    print("✅ 模型创建成功")
    print(f"   输入维度: 42")
    print(f"   输出类别: 52")
    print(f"   隐藏层: [256, 128, 64]")
    print(f"   Grid Size: 8")
except Exception as e:
    print(f"❌ 创建失败: {e}")
    exit(1)

# 步骤4: 计算参数量
print("\n步骤4: 计算模型参数...")
try:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
except Exception as e:
    print(f"❌ 失败: {e}")
    exit(1)

# 步骤5: 前向传播测试
print("\n步骤5: 测试前向传播...")
try:
    # 创建测试输入
    batch_size = 4
    x = torch.randn(batch_size, 42)

    # 前向传播
    with torch.no_grad():
        logits = model(x)

    print(f"✅ 前向传播成功")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {logits.shape}")

    # 验证输出形状
    assert logits.shape == (batch_size, 52), f"输出形状错误: {logits.shape}"
    print(f"   ✓ 输出形状正确: {logits.shape}")

except Exception as e:
    print(f"❌ 失败: {e}")
    exit(1)

# 步骤6: Softmax概率测试
print("\n步骤6: 测试Softmax概率...")
try:
    # 计算概率
    probs = torch.softmax(logits, dim=1)

    # 检查概率和为1
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6), \
        f"概率和不为1: {prob_sums}"

    print(f"✅ Softmax概率正确")
    print(f"   概率形状: {probs.shape}")
    print(f"   概率和: {prob_sums.tolist()}")
    print(f"   ✓ 所有样本的概率和都为1.0")

except Exception as e:
    print(f"❌ 失败: {e}")
    exit(1)

# 步骤7: 梯度测试
print("\n步骤7: 测试梯度反向传播...")
try:
    # 创建训练数据
    x_train = torch.randn(4, 42, requires_grad=True)
    y_train = torch.randint(0, 52, (4,))

    # 前向传播
    logits = model(x_train)

    # 计算损失
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, y_train)

    # 反向传播
    loss.backward()

    print(f"✅ 梯度反向传播成功")
    print(f"   损失值: {loss.item():.4f}")

    # 检查梯度
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "模型参数没有梯度"
    print(f"   ✓ 模型参数有梯度")

except Exception as e:
    print(f"❌ 失败: {e}")
    exit(1)

# 步骤8: GPU测试（如果可用）
print("\n步骤8: 测试GPU支持...")
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_gpu = model.to(device)
        x_gpu = torch.randn(4, 42).to(device)

        with torch.no_grad():
            logits_gpu = model_gpu(x_gpu)

        print(f"✅ GPU测试成功")
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"   输出设备: {logits_gpu.device}")
    else:
        print("⚠️  CUDA不可用，跳过GPU测试")
        print("   模型可以在CPU上运行")

except Exception as e:
    print(f"❌ GPU测试失败: {e}")
    print("   但模型仍可在CPU上使用")

# 最终总结
print("\n" + "=" * 80)
print("🎉 所有测试通过！KAN模型已准备就绪")
print("=" * 80)
print("\n可以使用以下命令开始训练:")
print("  Shell脚本: MODEL_NAME=kan ./run_training.sh")
print("  Python脚本: python train.py --model kan --epochs 25")
print("\n更多信息请查看 README.md")
print("=" * 80)
