#!/usr/bin/env python3
"""
测试训练配置，确保禁用Mixup后能正常运行
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

import torch
import torch.nn as nn
from losses import create_loss_function
from resnet import mri_resnet50

def test_cb_focal_loss():
    """测试cb_focal损失函数"""
    print("=== 测试 CB Focal Loss ===")
    
    # 模拟您的实际类别分布（102个脑区的不平衡分布）
    # 假设一些脑区样本很多，一些很少
    class_counts = torch.zeros(102)
    class_counts[:20] = 50000   # 20个大脑区
    class_counts[20:60] = 5000  # 40个中等脑区
    class_counts[60:] = 500     # 42个小脑区
    
    print(f"类别分布：")
    print(f"  - 大脑区(1-20): 各50000样本")
    print(f"  - 中脑区(21-60): 各5000样本")
    print(f"  - 小脑区(61-102): 各500样本")
    
    # 创建损失函数
    loss_fn = create_loss_function(
        'cb_focal',
        class_counts=class_counts,
        beta=0.9999,
        gamma=1.5,
        reduction='mean'
    )
    
    # 测试前向传播
    batch_size = 256
    num_classes = 102
    
    # 模拟模型输出
    logits = torch.randn(batch_size, num_classes)
    
    # 模拟不同类别的标签
    labels = torch.cat([
        torch.zeros(50, dtype=torch.long),     # 多数类
        torch.ones(50, dtype=torch.long) * 30,  # 中等类
        torch.ones(156, dtype=torch.long) * 80  # 少数类
    ])
    
    # 计算损失
    loss = loss_fn(logits, labels)
    print(f"\n损失值: {loss.item():.4f}")
    print("✓ CB Focal Loss 工作正常，无需Mixup包装")
    
    return True

def test_model_forward():
    """测试模型前向传播"""
    print("\n=== 测试 ResNet-50 模型 ===")
    
    # 创建模型
    model = mri_resnet50(
        input_channels=351,
        num_classes=102,
        base_width=104
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 351, 7, 7)
    
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    assert y.shape == (batch_size, 102), f"输出形状错误: {y.shape}"
    print("✓ 模型前向传播正常")
    
    return True

def test_training_step():
    """测试一个训练步骤（不使用Mixup）"""
    print("\n=== 测试训练步骤（无Mixup） ===")
    
    # 设置
    device = 'cpu'  # 本地测试用CPU
    model = mri_resnet50(input_channels=351, num_classes=102, base_width=104).to(device)
    
    # 类别分布
    class_counts = torch.ones(102) * 1000  # 简化的均匀分布
    
    # 创建损失函数（不包装Mixup）
    criterion = create_loss_function(
        'cb_focal',
        class_counts=class_counts,
        beta=0.9999,
        gamma=1.5
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 模拟一个batch
    images = torch.randn(8, 351, 7, 7).to(device)
    labels = torch.randint(0, 102, (8,)).to(device)
    
    # 前向传播
    outputs = model(images)
    
    # 计算损失（直接使用，无需Mixup的4个参数）
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"损失值: {loss.item():.4f}")
    print("✓ 训练步骤正常执行（无Mixup）")
    
    return True

if __name__ == "__main__":
    print("测试训练配置（禁用Mixup后）\n")
    
    try:
        # 运行所有测试
        test_cb_focal_loss()
        test_model_forward()
        test_training_step()
        
        print("\n" + "="*50)
        print("✅ 所有测试通过！")
        print("建议：")
        print("1. cb_focal损失函数正常工作，适合处理不平衡数据")
        print("2. 模型前向传播正常，参数量约50M")
        print("3. 训练循环无需Mixup即可正常运行")
        print("\n您现在可以运行 bash run_leave_one_out.sh 开始训练")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()