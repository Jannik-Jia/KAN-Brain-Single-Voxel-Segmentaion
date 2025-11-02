# KAN模型设置指南

## 概述

BrainVoxelKAN是基于Kolmogorov-Arnold网络（KAN）的MRI体素分类模型。本指南将帮助您快速设置和使用KAN模型。

---

## 📦 安装依赖

### 1. 安装FastKAN库

KAN模型依赖于fastkan库：

```bash
pip install fastkan
```

### 2. 验证安装

运行测试脚本验证KAN模型是否正确安装：

```bash
cd refactored_training
python test_kan.py
```

如果所有测试通过，您将看到：
```
🎉 所有测试通过！KAN模型已准备就绪
```

---

## 🚀 快速开始

### 使用Shell脚本

```bash
# 基础训练
MODEL_NAME=kan ./run_training.sh

# 自定义epochs
MODEL_NAME=kan EPOCHS=30 ./run_training.sh
```

### 使用Python脚本

```bash
# 基础训练
python train.py --model kan --epochs 25 --batch_size 8192

# 自定义grid_size
python train.py --model kan --epochs 25 --grid_size 16

# 完整自定义
python train.py \
    --model kan \
    --epochs 30 \
    --batch_size 4096 \
    --lr 0.0001 \
    --grid_size 12
```

---

## ⚙️ KAN模型配置

### 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_dims | [256, 128, 64] | 隐藏层维度 |
| grid_size | 8 | KAN网格分段数 |
| use_class_weights | True | 自动使用类权重 |
| num_classes | 52 | 输出类别数（包含所有类别）|

### 关键特性

✅ **自动类权重**: KAN模型自动计算和使用类权重，处理类别不平衡
✅ **无需PCA**: 直接使用原始特征，无需降维
✅ **类别包含性**: 标签0被作为正常类别（非背景）
✅ **灵活配置**: grid_size和hidden_dims可自定义

---

## 📊 模型对比

与其他模型的对比：

| 模型 | 参数量 | 类权重 | 特点 |
|------|--------|--------|------|
| RegModel | ~84M | 否 | 原始基准，深层MLP |
| ResNetMLP | ~25M | 否 | 残差连接，训练稳定 |
| SimpleMLP | ~1M | 否 | 轻量级，快速 |
| **KAN** | ~待测 | **是** | KAN架构，自动平衡 |

---

## 🔧 参数调优建议

### grid_size选择

- **8 (默认)**: 平衡性能和计算效率，适合初次尝试
- **12-16**: 更强表达力，适合复杂数据，但计算量更大
- **4-6**: 更快训练，适合快速实验

### hidden_dims选择

- **[256, 128, 64] (默认)**: 三层递减，适合大多数情况
- **[512, 256, 128]**: 更大容量，数据充足时使用
- **[128, 64]**: 两层，训练更快
- **[64]**: 单层，最轻量级

### 学习率调整

KAN模型可能需要不同的学习率：

```bash
# 尝试更大的学习率
python train.py --model kan --lr 0.0001

# 或更小的学习率
python train.py --model kan --lr 0.000001
```

---

## 📈 训练监控

### 查看训练日志

```bash
# 实时查看
tail -f logs/kan_bg_excl_*.log

# 查看类权重信息
grep "类权重统计" logs/kan_bg_excl_*.log -A 20
```

### 检查类权重

训练开始时会显示：
```
⚖️ 计算类权重以处理类别不平衡...

类别分布统计:
类别ID     样本数            占比(%)
----------------------------------------
0          12345            2.50
1          45678            9.25
...

类权重统计 (method=inverse_freq):
类别ID     权重              样本数
---------------------------------------------
0          8.1234           12,345
1          2.1890           45,678
...
```

---

## 🐛 故障排除

### 问题1: ImportError: No module named 'fastkan'

**解决方案**:
```bash
pip install fastkan
```

### 问题2: CUDA out of memory

**解决方案**:
```bash
# 减小batch_size
python train.py --model kan --batch_size 4096

# 或减小grid_size
python train.py --model kan --grid_size 4
```

### 问题3: 训练速度慢

**解决方案**:
- 减小grid_size: `--grid_size 4`
- 减少隐藏层: `--hidden_dims` (需修改代码)
- 减小batch_size以获得更快的epoch

### 问题4: 验证KAN是否正常工作

**解决方案**:
```bash
python test_kan.py
```

查看所有测试是否通过。

---

## 📝 使用示例

### 示例1: 基础训练

```bash
# 使用默认配置训练10个epoch
MODEL_NAME=kan EPOCHS=10 ./run_training.sh
```

### 示例2: 调优grid_size

```bash
# 尝试不同的grid_size
for grid in 4 8 12 16; do
    python train.py --model kan --grid_size $grid --epochs 10
done
```

### 示例3: 完整训练

```bash
# 完整25 epochs训练
python train.py \
    --model kan \
    --epochs 25 \
    --batch_size 8192 \
    --lr 0.00001 \
    --grid_size 8
```

---

## 💡 最佳实践

1. **首次使用**: 从默认配置开始（grid_size=8）
2. **验证模型**: 运行 `python test_kan.py` 确保正确安装
3. **监控类权重**: 检查日志中的类权重分布是否合理
4. **对比实验**: 与RegModel对比，查看性能提升
5. **参数搜索**: 尝试不同的grid_size (4, 8, 12, 16)

---

## 📚 相关资源

- **FastKAN库**: https://github.com/ZiyaoLi/fast-kan
- **KAN论文**: Kolmogorov-Arnold Networks (arxiv.org)
- **项目README**: 查看 README.md 了解完整功能

---

## ❓ 常见问题

**Q: KAN模型比其他模型更好吗？**
A: 不一定。KAN在某些任务上表现更好，但需要实验对比。优势在于自动使用类权重。

**Q: 为什么KAN自动使用类权重？**
A: 因为KAN特别适合处理不平衡数据，我们将其作为默认配置。

**Q: 可以禁用类权重吗？**
A: 可以修改 `models/kan_model.py` 中的 `use_class_weights` 为 `False`。

**Q: grid_size越大越好吗？**
A: 不是。太大会过拟合且计算慢。建议从8开始，根据验证集性能调整。

---

祝您训练顺利！如有问题，请查看主README或运行测试脚本。
