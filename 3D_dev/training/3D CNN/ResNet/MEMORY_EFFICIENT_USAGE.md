# 内存高效数据加载使用指南

## 概述

新的内存高效数据加载系统已实现，主要特性：

1. **只加载Labels/Masks到内存** (~1.2GB total)
2. **71M个patch完整训练** (不再限制samples_per_subject)
3. **真正随机化** (每个epoch重新shuffle所有71M坐标)
4. **按需patch提取** (每个patch只读68KB)
5. **向后兼容** (保持原有接口)

## 使用方法

### 在训练脚本中启用内存高效模式

修改你的训练脚本中的数据加载部分：

```python
# 原来的方式
train_loader, test_loader = create_data_loaders(
    train_files=train_files,
    test_files=test_files,
    batch_size=batch_size,
    num_workers=num_workers,
    samples_per_subject=samples_per_subject,
    memory_efficient=False  # 原来的方式
)

# 新的内存高效方式
train_loader, test_loader = create_data_loaders(
    train_files=train_files,
    test_files=test_files,
    batch_size=batch_size,
    num_workers=0,  # 重要：内存高效模式必须设为0
    samples_per_subject=None,  # 被忽略，将使用所有71M patches
    memory_efficient=True   # 启用内存高效模式
)
```

### 训练循环中的epoch管理

```python
# 训练循环
for epoch in range(num_epochs):
    # 重要：每个epoch开始时重新shuffle数据
    if hasattr(train_loader.dataset, 'set_epoch'):
        train_loader.dataset.set_epoch(epoch)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 正常训练代码...
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 主要变化

### 1. 数据集大小变化
```python
# 原来：每个subject采样10000个patch
# 37个训练subjects × 10000 = 370,000 patches per epoch

# 现在：所有有效patch
# 约71,000,000 patches per epoch (完整数据集)
```

### 2. 内存使用
```python
# 原来：预加载全部3D数据
# ~37 subjects × 12GB = ~450GB (不可行)

# 现在：只加载labels/masks
# ~37 subjects × 32MB = ~1.2GB + 每个batch的临时patch数据
```

### 3. 训练时间估算
```python
# 原来：370k patches / 2048 batch_size ≈ 180 batches/epoch
# 现在：71M patches / 2048 batch_size ≈ 34,700 batches/epoch

# 预期每个epoch时间增加，但收敛更快（因为使用完整数据）
```

## 配置建议

### 推荐配置（内存高效模式）
```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --batch_size 2048 \
    --epochs 10 \
    --num_workers 0 \
    --memory_efficient \
    --patience 5  # 更激进的early stopping
```

### 参数说明
- `--num_workers 0`: h5py不支持多进程，必须设为0
- `--epochs 10`: 因为每个epoch包含完整71M数据，可以减少epoch数
- `--patience 5`: 更快收敛，可以更激进地early stop
- `samples_per_subject`: 在内存高效模式下被忽略

## 向后兼容

如果需要使用原来的方式（比如测试或调试）：

```python
# 禁用内存高效模式
train_loader, test_loader = create_data_loaders(
    train_files=train_files,
    test_files=test_files,
    memory_efficient=False,  # 使用原来的方式
    samples_per_subject=10000,
    num_workers=4
)
```

## 预期效果

### 优势
1. **完整数据训练**: 不再丢弃数据，使用全部71M有效patch
2. **内存高效**: 只需1.2GB内存 vs 450GB
3. **真正随机**: 每个epoch完全重新随机化
4. **更好收敛**: 完整数据集训练，期望更好的泛化性能

### 权衡
1. **更长的epoch**: 每个epoch从180 batches增加到34,700 batches
2. **磁盘I/O**: 按需读取增加磁盘访问，但每次只读68KB
3. **单线程**: num_workers=0，但h5py的I/O通常是瓶颈

## 故障排除

### 如果遇到内存不足
```python
# 减少batch size
--batch_size 1024

# 或者回退到原来的模式
memory_efficient=False
```

### 如果训练太慢
```python
# 检查是否在SSD上
# 确认batch_size足够大
# 考虑减少epochs（因为每个epoch数据更多）
```

### 测试内存高效模式
```python
# 运行测试脚本
cd models/
python dataset.py
```

这个指南帮助你理解和使用新的内存高效数据加载系统了吗？