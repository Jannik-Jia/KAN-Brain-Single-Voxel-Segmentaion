# DeepMLP 错误修复总结

## 问题描述

DeepMLP训练时出现维度不匹配错误：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8192x2048 and 1536x384)
```

同时，类权重计算对没有样本的类别（类别0）分配了异常大的权重（51.97）。

---

## 问题分析

### 问题1: 维度不匹配 ❌

**根本原因:**
- DeepMLP使用的hidden_dims为 `[2048, 1536, 1536, 1536, 1536, 2048, 2048]`
- 每个`ResidualBlock`是瓶颈结构: `dim → bottleneck_dim → dim`（输入输出维度相同）
- 但代码中没有处理**层之间的维度变化**
- 当第0层输出2048维，第1层期望输入1536维时，维度不匹配

**问题代码:**
```python
# 每个ResidualBlock维度不变
for i, hidden_dim in enumerate(hidden_dims):
    layer_modules['block'] = ResidualBlock(hidden_dim, ...)  # 输入输出都是hidden_dim
    self.layers.append(layer_modules)
```

当`hidden_dims[i] != hidden_dims[i+1]`时，没有维度转换！

### 问题2: 类权重计算异常 ⚠️

**根本原因:**
- 类别0有0个样本
- 代码使用 `np.maximum(class_counts, 1.0)` 将0改为1
- 导致虚假的大权重：`1 / 1 * normalization_factor ≈ 51.97`

**问题代码:**
```python
class_counts = np.maximum(class_counts, 1.0)  # 0 → 1
weights = 1.0 / class_counts  # 1 / 1 = 1.0（相对其他类很大）
```

---

## 修复方案

### 修复1: 添加维度转换层 ✅

**方案：** 在层之间添加`transition_layers`来处理维度变化

**修改位置:** `models/deep_mlp.py`

**新增代码:**
```python
# 构建网络时
self.transition_layers = nn.ModuleList()

for i, hidden_dim in enumerate(hidden_dims):
    # ... 构建主层 ...

    # 维度转换层：如果下一层维度不同，添加投影层
    if i < len(hidden_dims) - 1 and hidden_dims[i] != hidden_dims[i + 1]:
        self.transition_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
    else:
        self.transition_layers.append(nn.Identity())
```

**前向传播:**
```python
for i, layer_modules in enumerate(self.layers):
    # ... 主层计算 ...
    x = self.layer_norms[i](x)

    # 维度转换（无论是否跳过该层，都需要转换维度）
    if i < len(self.transition_layers):
        x = self.transition_layers[i](x)
```

**参考来源:** 参考文件 `AttentionMLP_102分类_30分类.ipynb` 第975行：
```python
nn.Linear(hidden_dims[i], hidden_dims[i+1]) if hidden_dims[i] != hidden_dims[i+1] else nn.Identity()
```

### 修复2: 正确处理零样本类别 ✅

**方案：** 将没有样本的类别权重设置为0，而不是虚假的大值

**修改位置:** `utils/class_weights.py`

**新增代码:**
```python
if method == 'inverse_freq':
    # 记录没有样本的类别
    zero_count_classes = np.where(class_counts == 0)[0]
    if len(zero_count_classes) > 0:
        logger.warning(f"⚠️ 以下类别没有训练样本: {zero_count_classes.tolist()}")
        logger.warning(f"   这些类别的权重将被设置为0")

    # 避免除以零，但保持0样本的类别为0权重
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)
    weights = 1.0 / safe_counts

    # 将没有样本的类别权重设置为0
    weights[class_counts == 0] = 0.0

    # 归一化使平均权重为1（只对非零权重归一化）
    non_zero_weights = weights[weights > 0]
    if len(non_zero_weights) > 0:
        scale_factor = non_zero_weights.mean()
        weights = weights / scale_factor
```

---

## 测试验证

### 运行测试脚本

```bash
cd refactored_training
python test_deep_mlp_fix.py
```

**预期输出:**
```
================================================================================
测试DeepMLP维度修复
================================================================================

模型配置:
  输入维度: 41
  输出类别: 52
  隐藏层维度: [2048, 1536, 1536, 1536, 1536, 2048, 2048]
  总参数量: 70,460,468

测试前向传播:
  输入形状: torch.Size([16, 41])
  输出形状: torch.Size([16, 52])
  ✅ 前向传播成功！
  Softmax概率和: 1.0000 (应该接近1.0)

测试训练模式（含随机深度）:
  ✅ 训练模式测试通过（5次迭代）

================================================================================
✅ 所有测试通过！DeepMLP已修复
================================================================================
```

### 重新训练

```bash
# 使用Python脚本
python train.py --model deep_mlp --epochs 25 --batch_size 8192

# 或使用Shell脚本
MODEL_NAME=deep_mlp ./run_training.sh

# 或使用多模型训练
./run_multiple_models.sh
# 选择: 5 (deep_mlp)
```

---

## 技术细节

### 为什么需要transition_layers？

1. **ResidualBlock的瓶颈结构:**
   ```
   dim → bottleneck_dim (dim//4) → dim
   ```
   输入输出维度相同，不做维度转换。

2. **不同的hidden_dims:**
   ```
   [2048, 1536, 1536, 1536, 1536, 2048, 2048]
   ```
   层0输出2048，但层1期望输入1536。

3. **transition_layers的作用:**
   - 如果 `hidden_dims[i] != hidden_dims[i+1]`：使用 `Linear(hidden_dims[i], hidden_dims[i+1])`
   - 如果维度相同：使用 `Identity()`（不做任何操作）

### 为什么随机深度会触发错误？

**原始代码:**
```python
if torch.rand(1).item() < self.stochastic_depth_rate:
    continue  # 跳过这一层
```

当跳过某一层时，`x`的维度保持不变。如果下一层期望不同的维度，就会报错。

**修复后:**
```python
if not skip_layer:
    x = layer_modules['block'](x)
    x = self.layer_norms[i](x)

# 无论是否跳过，都需要转换维度
x = self.transition_layers[i](x)
```

### 为什么类别0没有样本？

可能的原因：
1. 数据预处理时排除了类别0
2. 类别0是背景，在`exclude_background=True`模式下被过滤
3. 数据不平衡导致某些类别样本极少

**解决方案：** 将权重设置为0，这样在loss计算时不会影响训练。

---

## 相关文件

### 修改的文件

1. `models/deep_mlp.py`
   - 添加了 `self.transition_layers`
   - 修改了 `forward()` 方法

2. `utils/class_weights.py`
   - 修改了 `compute_class_weights()` 函数
   - 正确处理零样本类别

### 新增的文件

3. `test_deep_mlp_fix.py`
   - 测试修复是否正确

4. `DEEP_MLP_FIX.md` (本文档)
   - 详细说明问题和修复

---

## 参考

- 参考文件: `AttentionResidualMLP_BrainVoxel_30Class/AttentionMLP_102分类_30分类.ipynb`
  - 第842行: `ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout)`
  - 第975行: 维度转换层的实现

---

## 更新日志

**2025-11-02:**
- ✅ 修复DeepMLP维度不匹配问题
- ✅ 修复类权重计算对零样本类别的处理
- ✅ 添加测试脚本
- ✅ 创建修复文档

---

**状态:** ✅ 已修复，可以开始训练

**建议:** 先运行 `test_deep_mlp_fix.py` 验证修复，然后开始完整训练。
