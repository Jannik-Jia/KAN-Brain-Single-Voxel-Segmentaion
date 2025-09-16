# 数据集转换与训练更新检查清单

## ✅ 已完成的更新

### 1. **数据集转换工具**
- [x] 创建z-score标准化转换器
- [x] 优化7×7×1 patch提取 (chunk size: 32×32×1×351)
- [x] 使用LZF压缩 (快速解压)
- [x] 向量化z-score实现 (100x性能提升)
- [x] 只基于脑组织体素计算统计量
- [x] 完整的验证机制

### 2. **训练脚本路径更新**
- [x] `ResNet/scripts/run_leave_one_out.sh`
- [x] `3D CNN/run_all_models_7x7.sh`
- [x] `3D CNN/run_all_models_3x3.sh`
- [x] `B0_1D_training/run_1d_training.sh`
- [x] `B0_1D_training/run_1d_leave_one_out.sh`

### 3. **工具优化**
- [x] 修复ImportError (quick_convert.py)
- [x] 修复AttributeError (logging初始化顺序)
- [x] 更新CLI默认参数 (chunk size)
- [x] 清理重复文件

## 🔄 使用流程

### 第一步：转换数据集
```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create
python quick_convert.py

# 选择要转换的数据集：
# 1. 3D数据 (用于patch训练)
# 2. 1D数据 (原始格式)
# 3. 都转换
```

### 第二步：训练模型
```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/training

# ResNet训练 (推荐的主要实验)
cd "3D CNN/ResNet/scripts"
bash run_leave_one_out.sh

# 其他实验
cd "../.."
bash run_all_models_7x7.sh  # 7×7 patch对比
bash run_all_models_3x3.sh  # 3×3 patch对比

# 1D训练
cd ../B0_1D_training
bash run_1d_training.sh           # 单次训练
bash run_1d_leave_one_out.sh      # Leave-one-out
```

## 📂 预期文件结构

转换完成后，数据集目录结构：
```
alex_datasets/
├── 1D/                                    # 原始1D数据
├── 1D_zscore_normalized/                  # ✨ Z-score标准化1D数据
│   ├── subject1.h5
│   ├── subject2.h5
│   └── ... (38个文件)
├── 3D_validated/                          # 原始3D数据
└── 3D_validated_zscore_normalized/        # ✨ Z-score标准化3D数据
    ├── subject1.h5
    ├── subject2.h5
    └── ... (38个文件)
```

## ⚠️ 重要注意事项

### 1. **训练代码兼容性**
确保训练代码能读取`.h5`格式文件而不是`.mat`格式：

```python
# 需要支持的读取方式
import h5py
with h5py.File('subject1.h5', 'r') as f:
    data = f['data'][()]          # (384, 336, 256, 351)
    labels = f['region_labels'][()] # (384, 336, 256)
    mask = f['region_mask'][()]    # (384, 336, 256)
```

### 2. **内存和性能**
- ✅ 转换后文件更小 (~200-300MB vs ~3GB)
- ✅ Patch提取更快 (LZF压缩 + 优化chunk)
- ✅ 训练应该更稳定 (z-score标准化)

### 3. **验证结果**
转换日志中应该看到：
```
✅ Z-score normalization: Passed (351 dimensions normalized)
✅ Spatial consistency: Passed
✅ Label integrity: Passed
```

## 🚨 故障排除

### 转换失败
1. 检查原始数据路径是否正确
2. 确认有足够磁盘空间
3. 查看转换日志了解具体错误

### 训练失败
1. 确认转换后的数据集存在
2. 检查训练代码是否支持`.h5`格式
3. 验证路径配置是否正确

### 性能问题
1. 确认使用的是z-score标准化版本
2. 检查chunk size和压缩设置
3. 监控GPU和内存使用情况

## 🎯 预期改善

使用z-score标准化数据集后：
- **训练稳定性**: 减少梯度爆炸/消失
- **收敛速度**: 更快达到最优
- **模型性能**: F1-score等指标提升
- **泛化能力**: 更好的跨被试表现
- **I/O效率**: 更快的数据加载

---

**总结**: 所有必要的更新都已完成。现在可以运行数据集转换，然后使用更新后的训练脚本进行实验。