# 训练逻辑说明

## 数据分割策略

### 38-Fold交叉验证
- **训练集**: 37个被试
- **测试集**: 1个被试（轮流）
- **验证集**: 从37个训练被试的数据中随机分割1%（与原notebook保持一致）

### Patient-wise Z-score标准化
每个被试的数据独立进行Z-score标准化：

```python
# 对每个被试独立进行标准化
for subject_id, subject_data in all_subjects_data.items():
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(subject_data['data'])  # (n_voxels, 351)
```

这确保了：
1. **被试间独立性**: 每个被试的特征分布独立标准化
2. **避免数据泄露**: 测试集被试不会影响训练集的标准化
3. **更好的泛化**: 模型学习跨被试的共同模式而非特定分布

## 与原notebook的对比

### 相同点
- **模型架构**: 4x4096全连接层 + Dropout(0.5)
- **训练参数**:
  - Batch size: 128
  - Epochs: 25
  - Learning rate: 1e-5
  - L2 regularization: 1e-5
- **优化器**: Adam
- **损失函数**: CrossEntropyLoss
- **训练/验证分割**: 99%/1%

### 改进点
1. **数据源**: 使用38个独立的1D数据集（351维），而非合并的TRAIN38.mat（341维）
2. **标准化策略**: Patient-wise z-score，而非全局标准化
3. **交叉验证**: 完整的38-fold，每个被试轮流作为测试集
4. **评估指标**: 增加了macro-F1、per-class metrics、gross accuracy
5. **输出格式**: 增加ONNX导出，便于部署

## 数据维度

- **输入特征**: 351维（多模态MRI特征）
- **输出类别**: 102个脑区域
- **被试数量**: 38个

## 验证策略

每个fold的数据分配：
```
Fold 1: Train=[2-38], Test=[1], Val=[从2-38中随机1%]
Fold 2: Train=[1,3-38], Test=[2], Val=[从1,3-38中随机1%]
...
Fold 38: Train=[1-37], Test=[38], Val=[从1-37中随机1%]
```

## 使用方法

### 训练单个fold（使用patient-wise z-score）
```bash
python train_38fold.py --fold 5
```

### 训练所有38个fold
```bash
python train_38fold.py  # 自动使用patient-wise z-score
```

### 使用原始全局标准化（仅用于对比）
需要修改代码中的`patient_wise_zscore=False`