# Mixup 使用说明

## 概述

Mixup 是一种数据增强技术，通过混合两个训练样本来创建新的虚拟样本。虽然已经修复了实现bug，但**对于脑区分类任务，我们建议暂时不使用Mixup**。

## 修复说明

### 原始Bug
```python
# 错误：所有损失都被包装成MixupLoss
if use_mixup:
    self.criterion = MixupLoss(self.criterion)
    
# 导致标准训练分支调用失败
loss = self.criterion(outputs, labels)  # MixupLoss需要4个参数！
```

### 修复方案
```python
# 正确：分离base和mixup损失
self.base_criterion = create_loss_function(...)  # 基础损失

if use_mixup:
    self.mixup_criterion = MixupLoss(self.base_criterion)  # Mixup版本
    
# 使用时根据情况选择
if use_mixup and random() < 0.5:
    loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)  # 4参数
else:
    loss = self.base_criterion(outputs, labels)  # 2参数
```

## 使用方法

### 启用Mixup
```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --use_mixup \
    --mixup_alpha 0.2
```

### 禁用Mixup（推荐）
```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1
# 不添加 --use_mixup 标志
```

## 参数说明

- `--use_mixup`: 启用Mixup数据增强
- `--mixup_alpha`: Beta分布参数，控制混合强度（默认0.2）
  - 0.1: 轻微混合
  - 0.2: 标准混合（默认）
  - 0.4: 强混合

## 为什么不推荐用于脑区分类

### 1. 缺乏生物学意义
- 混合海马体和皮层的MRI信号没有实际意义
- 脑区是离散的解剖结构，不是连续的

### 2. 已有足够的数据多样性
- 7000万+训练样本已经很充分
- 7×7邻域本身提供了空间多样性

### 3. 其他正则化已足够
- `cb_focal`损失已处理类别不平衡
- EMA提供了模型平均
- 大batch size稳定训练

## 测试修复

运行测试脚本验证修复：
```bash
cd scripts/
python test_mixup_fix.py
```

预期输出：
```
✅ 所有测试通过！Mixup已成功修复
```

## 建议

1. **默认配置**：不使用Mixup，依靠cb_focal处理不平衡
2. **如果过拟合严重**：先尝试增加weight_decay或label_smoothing
3. **实验性使用**：如果想尝试Mixup，建议alpha=0.1-0.2

## 相关文件

- 训练脚本：`scripts/train_mri_resnet.py`
- 损失函数：`models/losses.py`
- 测试脚本：`scripts/test_mixup_fix.py`
- 批量训练：`scripts/run_leave_one_out.sh`