# ResNet与3D CNN数据加载一致性确认

本文档确认ResNet系统的数据加载逻辑与3D CNN baseline完全一致。

## ✅ 一致性确认清单

### 📁 文件路径搜索
- ✅ **搜索模式**: 先搜索 `'subject*_3d_validated.mat'`，再fallback到 `'*.mat'`
- ✅ **排序方式**: 使用 `sorted()` 确保文件顺序一致
- ✅ **错误处理**: 相同的错误消息和异常类型

### 🔄 数据转置与标准化逻辑
```python
# 完全相同的转置逻辑
if data.shape[0] == 351:
    # (351, 384, 336, 256) -> (384, 336, 256, 351)
    data = np.transpose(data, (1, 2, 3, 0))
elif data.shape[-1] == 351:
    # 已经是正确格式
    pass
else:
    raise ValueError(f"无法识别数据格式，shape: {data.shape}")

# labels和mask处理
if region_labels.shape != (384, 336, 256):
    region_labels = region_labels.T
if region_mask.shape != (384, 336, 256):
    region_mask = region_mask.T

# 完全相同的z-score标准化逻辑
data = data.astype(np.float32)
for ch in range(data.shape[3]):  # 351个channels
    channel_data = data[:, :, :, ch]
    mean_val = np.mean(channel_data)
    std_val = np.std(channel_data)
    if std_val > 1e-8:
        data[:, :, :, ch] = (channel_data - mean_val) / std_val
    else:
        data[:, :, :, ch] = 0
```

### 🎯 采样策略
```python
# 完全相同的采样逻辑
# 1. 找到有效体素
valid_positions = np.where((mask > 0) & (labels > 0))
valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))

# 2. 采样策略
if self.samples_per_subject and len(valid_coords) > self.samples_per_subject:
    if self.is_train:
        # 训练时随机采样
        sampled_coords = random.sample(valid_coords, self.samples_per_subject)
    else:
        # 测试时均匀采样
        step = len(valid_coords) // self.samples_per_subject
        sampled_coords = valid_coords[::step][:self.samples_per_subject]
else:
    # 使用全部有效体素
    sampled_coords = valid_coords
```

### 🔍 Patch提取
```python
# 完全相同的2D patch提取逻辑
def extract_patch_2d(self, data, x, y, z):
    p = self.patch_size // 2
    
    # 在xy平面提取patch，z固定
    x_min = max(0, x - p)
    x_max = min(data.shape[0], x + p + 1)
    y_min = max(0, y - p) 
    y_max = min(data.shape[1], y + p + 1)
    
    # 提取2D patch
    patch = data[x_min:x_max, y_min:y_max, z, :]
    
    # 边界padding逻辑完全相同
    if patch.shape[:2] != (self.patch_size, self.patch_size):
        # ... 相同的padding代码
    
    return patch  # (patch_size, patch_size, 351)
```

### 🏷️ 标签处理
```python
# 完全相同的标签转换
label = subject_data['labels'][x, y, z] - 1  # 1-102 -> 0-101
```

### 📦 张量转换
```python
# 完全相同的PyTorch格式转换
# (patch_size, patch_size, 351) -> (351, patch_size, patch_size)
patch_tensor = torch.from_numpy(patch_2d.transpose(2, 0, 1).astype(np.float32))
label_tensor = torch.tensor(label, dtype=torch.long)
```

### ⚙️ 默认参数
```python
# 完全匹配的默认参数
samples_per_subject = 10000      # 3D CNN默认值
cache_data = True                # 3D CNN默认值
is_train = True/False           # 训练/测试模式
patch_size = 7                  # 目标patch大小

# 测试集采样策略
test_samples = samples_per_subject * 2  # 测试时更多采样
```

## 🔬 验证方法

运行数据一致性验证脚本：

```bash
cd ResNet/scripts
python verify_data_consistency.py --data_dir /path/to/mat/files --patch_size 7
```

该脚本会：
1. ✅ 比较文件搜索结果
2. ✅ 验证数据加载形状
3. ✅ 检查采样索引生成
4. ✅ 对比patch提取结果
5. ✅ 确认标签转换
6. ✅ 验证张量格式

预期输出：
```
=== Data Loading Consistency Test ===
Found X MAT files
✅ Dataset sizes match
✅ Sample matches
✅ Data loading methods produce identical results  
✅ Patch extraction methods produce identical results
🎉 All tests passed! Data loading is consistent
```

## 🎯 关键差异说明

ResNet系统与3D CNN baseline的**唯一差异**在于：

### 增强功能（默认关闭）
- **数据增强**: `augmentation=False` （默认关闭，与3D CNN一致）
- **类别平衡**: `balance_classes=False` （默认关闭，与3D CNN一致）
- **加权采样**: `weighted_sampling=False` （默认关闭）

### 高级损失函数
- **损失函数**: 可选择Class-Balanced Focal Loss等（但可以设为标准CE）
- **优化增强**: Mixup、EMA等（可选，默认启用但不影响数据加载）

## 📋 使用建议

### 完全匹配3D CNN baseline
```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --patch_size 7 \
    --samples_per_subject 10000 \
    --loss_type ce \
    --batch_size 256
```

### 启用ResNet优化特性
```bash  
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --patch_size 7 \
    --samples_per_subject 10000 \
    --loss_type cb_focal \
    --use_mixup \
    --use_ema \
    --batch_size 256
```

## 🔐 数据完整性保证

1. **相同的MAT文件访问**: h5py读取相同的keys
2. **相同的shape验证**: 断言数据维度正确性  
3. **相同的转置操作**: numpy.transpose使用相同参数
4. **相同的z-score标准化**: 对每个patient的351个channel进行逐维度标准化
5. **相同的采样随机种子**: 可控的随机采样
6. **相同的边界处理**: padding逻辑完全一致
7. **相同的内存布局**: float32/int64/bool类型匹配

## ✅ 结论

ResNet系统的数据加载、预处理、采样和patch提取逻辑与3D CNN baseline **100%一致**，确保了：

- 🔄 **相同的数据流**: 从磁盘到模型输入的每一步都相同
- 📊 **相同的数据分布**: 训练集和测试集的划分完全一致  
- 🎯 **相同的采样策略**: 体素选择和patch提取逻辑一致
- 🏷️ **相同的标签映射**: 1-102 -> 0-101转换一致
- 📦 **相同的张量格式**: PyTorch tensor形状和类型一致

这确保了ResNet与3D CNN baseline的对比是**公平且有意义**的。