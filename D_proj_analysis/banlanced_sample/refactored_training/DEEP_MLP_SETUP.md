# DeepMLP 模型设置与说明

## 📋 概述

DeepMLP是一个超深度多层感知机模型，集成了多种先进的深度学习技术，包括特征交互、残差连接、自注意力机制等。本文档详细说明了模型的架构、配置和使用方法。

---

## 🏗️ 模型架构

### 核心特性

1. **特征交互层** (Feature Interaction Layer)
   - 双线性交互: x_i * x_j for all i < j
   - 三线性交互: x_i * x_j * x_k for all i < j < k
   - 自动扩展输入特征维度，捕获特征间的高阶交互

2. **深层网络** (Deep Network)
   - 默认7层隐藏层: [2048, 1536, 1536, 1536, 1536, 2048, 2048]
   - 每层都有BatchNorm和LayerNorm
   - 总参数量: ~150-200M (适配42特征输入)

3. **残差连接** (Residual Connections)
   - 瓶颈结构: dim → bottleneck_dim → dim
   - 瓶颈维度 = hidden_dim // 4
   - 缓解梯度消失问题

4. **多头自注意力** (Multi-Head Self-Attention)
   - 16个注意力头
   - 在第1、3、5层添加自注意力
   - 增强模型的表达能力

5. **混合激活函数** (Mixed Activation)
   - 偶数层使用GELU
   - 奇数层使用SiLU
   - 提供不同的非线性变换

6. **正则化技术**
   - Shake-Shake正则化 (训练时随机混合残差路径)
   - 随机深度 (Stochastic Depth, drop_rate=0.2)
   - Dropout (0.3)

7. **类权重** (Class Weights)
   - 自动使用类权重处理类别不平衡
   - 通过`use_class_weights=True`启用

---

## 📊 参数适配说明

### 与原始规格的差异

原始DeepMLP规格（用户提供）:
```python
input_dim = 341  # FEATURE_DIM
num_classes = 102  # NUM_CLASS
hidden_dims = [8192, 6144, 6144, 6144, 6144, 8192, 8192]
总参数量: ~1.47B
```

**当前实现（已适配）**:
```python
input_dim = 42  # 当前数据集特征数（排除feature 14后）
num_classes = 52  # 当前数据集类别数（FreeSurfer标签）
hidden_dims = [2048, 1536, 1536, 1536, 1536, 2048, 2048]  # 按比例缩放
总参数量: ~150-200M
```

### 适配原因

1. **输入维度**: 当前数据集有43个原始特征，排除feature 14后为42个特征
2. **输出类别**: FreeSurfer标签映射后为52个类别（0-51）
3. **隐藏层缩放**: 按输入维度比例缩放 (42/341 ≈ 0.123)，将8192缩放至2048
4. **参数量控制**: 1.47B参数对于42特征输入过大，可能导致严重过拟合

### 特征交互维度

启用特征交互后，输入维度扩展：
- 原始特征: 42
- 双线性交互: 42 * 41 / 2 = 861
- 三线性交互: 42 * 41 * 40 / 6 = 11,480
- **总输入维度**: 42 + 861 + 11,480 = **12,383**

这使得第一层实际输入非常高维，能够捕获复杂的特征关系。

---

## 🚀 使用方法

### 基础训练

```bash
# 使用默认配置
python train.py --model deep_mlp --epochs 25 --batch_size 8192

# 使用Shell脚本
MODEL_NAME=deep_mlp ./run_training.sh

# 多模型训练（推荐）
./run_multiple_models.sh
# 选择: 5 (deep_mlp)
```

### 自定义配置

```bash
# 禁用特征交互（减少参数量）
python train.py --model deep_mlp --use_feature_interaction False

# 禁用三线性交互（保留双线性）
python train.py --model deep_mlp --use_trilinear False

# 禁用自注意力（加快训练）
python train.py --model deep_mlp --use_self_attention False

# 调整隐藏层维度（更大的模型）
python train.py --model deep_mlp --hidden_dims 3072 2304 2304 2304 2304 3072 3072

# 组合配置
python train.py --model deep_mlp \
    --epochs 30 \
    --batch_size 4096 \
    --use_trilinear False \
    --dropout_rate 0.4
```

---

## 🔧 配置选项

### 可配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 42 | 输入特征维度 |
| `num_classes` | 52 | 输出类别数 |
| `hidden_dims` | [2048, 1536, ...] | 隐藏层维度列表 |
| `use_feature_interaction` | True | 是否使用特征交互层 |
| `use_trilinear` | True | 是否使用三线性交互 |
| `use_residual` | True | 是否使用残差连接 |
| `use_self_attention` | True | 是否使用自注意力 |
| `num_attn_heads` | 16 | 注意力头数 |
| `attn_layers` | [1, 3, 5] | 在哪些层添加自注意力 |
| `use_mixed_activation` | True | 是否使用混合激活 |
| `use_shake_shake` | True | 是否使用Shake-Shake |
| `stochastic_depth_rate` | 0.2 | 随机深度drop rate |
| `dropout_rate` | 0.3 | Dropout率 |

### 功能开关建议

**追求最高性能**（默认配置）:
```python
use_feature_interaction=True
use_trilinear=True
use_residual=True
use_self_attention=True
use_mixed_activation=True
use_shake_shake=True
stochastic_depth_rate=0.2
```

**快速训练**（减少计算量）:
```python
use_feature_interaction=False  # 减少输入维度
use_trilinear=False
use_self_attention=False  # 跳过自注意力
stochastic_depth_rate=0.0  # 不使用随机深度
```

**防止过拟合**（增强正则化）:
```python
dropout_rate=0.5
stochastic_depth_rate=0.3
use_shake_shake=True
```

---

## 💾 内存和性能

### GPU内存需求

| 配置 | 模型大小 | 训练内存 (batch=8192) | 训练内存 (batch=4096) |
|------|----------|----------------------|---------------------|
| 完整配置 | ~800MB | ~12-16GB | ~8-10GB |
| 无特征交互 | ~300MB | ~6-8GB | ~4-6GB |
| 无三线性 | ~400MB | ~8-10GB | ~5-7GB |
| 最小配置 | ~200MB | ~4-6GB | ~3-4GB |

### 训练速度

相比其他模型（单epoch，batch_size=8192）:
- RegModel: ~2-3分钟
- ResNetMLP: ~1-2分钟
- SimpleMLP: ~30秒-1分钟
- KAN: ~3-5分钟
- **DeepMLP**: ~8-12分钟 (完整配置)

### 性能优化建议

1. **减小batch_size**: 如果GPU内存不足
   ```bash
   python train.py --model deep_mlp --batch_size 4096
   ```

2. **禁用三线性交互**: 显著减少参数量
   ```bash
   python train.py --model deep_mlp --use_trilinear False
   ```

3. **减少隐藏层维度**: 按比例缩小
   ```bash
   python train.py --model deep_mlp --hidden_dims 1024 768 768 768 768 1024 1024
   ```

4. **使用混合精度训练** (如果PyTorch支持):
   ```python
   # 在train.py中添加
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

---

## 🧪 测试模型

### 快速测试

```bash
# 测试模型创建和前向传播
python models/deep_mlp.py
```

输出示例:
```
================================================================================
DeepMLP 模型测试
================================================================================

总参数量: 156,234,789
可训练参数: 156,234,789
模型大小: 595.89 MB (FP32)

输入形状: torch.Size([16, 42])
输出形状: torch.Size([16, 52])
输出logits范围: [-0.1234, 0.2345]
Softmax概率和: 1.0000 (应该接近1.0)

✅ DeepMLP模型测试通过！
```

### 完整训练测试

```bash
# 快速训练测试（5 epochs）
EPOCHS=5 ./run_multiple_models.sh
# 选择: 5
```

---

## 📈 模型优势

### 相比RegModel

| 特性 | RegModel | DeepMLP |
|------|----------|---------|
| 参数量 | ~84M | ~150-200M |
| 特征交互 | ❌ | ✅ (双线性+三线性) |
| 残差连接 | ❌ | ✅ (瓶颈结构) |
| 自注意力 | ❌ | ✅ (多头) |
| BatchNorm | ❌ | ✅ |
| LayerNorm | ❌ | ✅ |
| 正则化 | Dropout | Dropout + Shake-Shake + Stochastic Depth |
| 类权重 | ❌ | ✅ (自动) |

### 相比KAN

| 特性 | KAN | DeepMLP |
|------|-----|---------|
| 参数量 | ~5-10M | ~150-200M |
| 网络深度 | 3层 | 7层 |
| 特征交互 | 内置在KAN层 | 显式双线性/三线性 |
| 自注意力 | ❌ | ✅ |
| 训练速度 | 中等 | 较慢 |
| 适用场景 | 探索性实验 | 追求最高性能 |

---

## 🐛 常见问题

### Q1: GPU内存不足

**A**: 尝试以下方法：
1. 减小batch_size: `--batch_size 4096`
2. 禁用特征交互: `--use_feature_interaction False`
3. 禁用三线性: `--use_trilinear False`
4. 减小hidden_dims: `--hidden_dims 1024 768 768 768 768 1024 1024`

### Q2: 训练太慢

**A**:
1. 禁用自注意力: `--use_self_attention False`
2. 减少epochs: `--epochs 15`
3. 使用更小的模型配置

### Q3: 模型过拟合

**A**:
1. 增加dropout: `--dropout_rate 0.5`
2. 增加随机深度: `--stochastic_depth_rate 0.3`
3. 减少模型容量（减小hidden_dims）
4. 使用数据增强

### Q4: 如何调整自注意力

**A**:
```bash
# 更多注意力头
python train.py --model deep_mlp --num_attn_heads 32

# 在不同层添加注意力
python train.py --model deep_mlp --attn_layers 0 2 4 6

# 只在最后一层添加
python train.py --model deep_mlp --attn_layers 6
```

### Q5: 与原始1.47B参数版本对比

**A**:
- 当前版本是针对42特征、52类别的**适配版本**
- 如果需要训练1.47B参数的完整版本，需要：
  1. 有341特征的数据集
  2. 102个输出类别
  3. 至少40GB+ GPU内存
  4. 修改`get_model_config()`中的hidden_dims

---

## 📚 技术细节

### 特征交互计算

双线性交互数量: `n * (n-1) / 2`
- 42特征: 861个交互项

三线性交互数量: `n * (n-1) * (n-2) / 6`
- 42特征: 11,480个交互项

### 参数量估算

```python
# 输入层（含特征交互）
12383 * 2048 ≈ 25M

# 隐藏层
for each layer:
    bottleneck: 2048 * 512 * 2 ≈ 2M
    attention: 2048 * 2048 * 3 + 2048 * 2048 ≈ 17M
Total hidden ≈ 100M

# 输出层
2048 * 52 ≈ 100K

# 总计: ~150M
```

### 自注意力机制

```python
# 对于每个注意力层
Q = Linear(dim, dim)  # Query
K = Linear(dim, dim)  # Key
V = Linear(dim, dim)  # Value
Attention = softmax(Q @ K^T / sqrt(d)) @ V
Output = Linear(dim, dim)
```

---

## 🎓 引用和参考

DeepMLP集成了多种经典技术：

1. **残差连接**: He et al., "Deep Residual Learning for Image Recognition" (2016)
2. **瓶颈结构**: He et al., "Identity Mappings in Deep Residual Networks" (2016)
3. **自注意力**: Vaswani et al., "Attention Is All You Need" (2017)
4. **Shake-Shake**: Gastaldi, "Shake-Shake regularization" (2017)
5. **随机深度**: Huang et al., "Deep Networks with Stochastic Depth" (2016)
6. **特征交互**: Blondel et al., "Higher-order Factorization Machines" (2016)

---

## 🎯 最佳实践

### 训练策略

1. **分阶段训练**:
   ```bash
   # 阶段1: 快速验证（5 epochs，简化配置）
   EPOCHS=5 python train.py --model deep_mlp \
       --use_trilinear False \
       --use_self_attention False

   # 阶段2: 完整训练（25 epochs，完整配置）
   EPOCHS=25 python train.py --model deep_mlp
   ```

2. **学习率调优**:
   ```bash
   # DeepMLP可能需要更小的学习率
   python train.py --model deep_mlp --lr 0.000005
   ```

3. **批量大小选择**:
   - GPU < 12GB: batch_size=2048
   - GPU 12-16GB: batch_size=4096
   - GPU 16-24GB: batch_size=8192
   - GPU > 24GB: batch_size=16384

### 实验建议

1. **基线对比**: 先训练SimpleMLP和RegModel建立基线
2. **逐步添加特性**: 从基础配置开始，逐步启用高级特性
3. **Per-class分析**: 重点关注少数类别的性能提升
4. **资源监控**: 使用`nvidia-smi`监控GPU使用

---

## 📞 支持

如有问题，请查看：
- 主文档: [README.md](README.md)
- 多模型训练指南: [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md)
- KAN设置: [KAN_SETUP.md](KAN_SETUP.md)

---

**创建时间**: 2025-01-02
**版本**: 1.0
**维护者**: Training System Team
