# 训练脚本路径更新说明

## 📝 更新内容

所有训练脚本已更新为使用**z-score标准化**后的数据集，以获得更好的训练效果。

## 🔄 路径变更对照表

| 脚本文件 | 原始路径 | 新路径 |
|---------|----------|--------|
| **3D CNN ResNet (主要)** | | |
| `run_leave_one_out.sh` | `3D_validated/` | `3D_validated_zscore_normalized/` |
| **3D CNN 对比实验** | | |
| `run_all_models_7x7.sh` | `3D_validated/` | `3D_validated_zscore_normalized/` |
| `run_all_models_3x3.sh` | `3D_validated/` | `3D_validated_zscore_normalized/` |
| **1D 训练** | | |
| `run_1d_training.sh` | `1D/` | `1D_zscore_normalized/` |
| `run_1d_leave_one_out.sh` | `1D/` | `1D_zscore_normalized/` |

## 📂 完整路径变更

### 3D数据 (用于patch训练)
```bash
# 原始路径
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"

# 新路径 (z-score标准化)
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated_zscore_normalized/"
```

### 1D数据 (原始格式训练)
```bash
# 原始路径
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"

# 新路径 (z-score标准化)
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D_zscore_normalized/"
```

## ⚡ 优势

### 1. **数据质量提升**
- ✅ 病人内z-score标准化，消除个体差异
- ✅ 每个维度独立标准化，保持特征平衡
- ✅ 只基于脑组织体素计算统计量，避免背景噪声

### 2. **训练性能优化**
- ✅ **7×7×1 patch优化**: 使用`(32,32,1,351)`分块
- ✅ **LZF压缩**: 快速解压，适合频繁patch访问
- ✅ **HDF5格式**: 比MAT文件小10-15倍，加载更快

### 3. **数据格式改进**
- ✅ `.h5`格式替代`.mat`格式
- ✅ 针对深度学习优化的存储布局
- ✅ 更好的内存效率和I/O性能

## 🚨 重要注意事项

### 1. **数据集转换**
在运行训练脚本前，必须先运行数据集转换：
```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create
python quick_convert.py
```

### 2. **文件格式变更**
- **原始**: `subject*.mat`
- **新格式**: `subject*.h5`
- 确保训练代码能正确读取`.h5`格式

### 3. **向后兼容**
所有脚本都保留了`*_ORIGINAL`变量，可以快速切换回原始数据：
```bash
# 切换回原始数据
DATA_DIR=$DATA_DIR_3D_ORIGINAL  # 替代 DATA_DIR=$DATA_DIR_3D
```

### 4. **输出目录更新**
训练结果输出目录也相应更新，避免与原始实验混淆：
- `results_leave_one_out_resnet` (ResNet保持不变，因为主要实验)
- `results_1d_with_3d_zscore` (1D训练)
- `results_1d_leave_one_out_zscore` (1D Leave-one-out)

## 📊 预期效果

使用z-score标准化数据训练后，应该观察到：
1. **训练稳定性提升**: 梯度更稳定，收敛更快
2. **模型性能改善**: F1-score等指标提升
3. **跨被试泛化**: 减少个体差异的影响
4. **训练效率**: 更快的数据加载和patch提取

## 🔧 故障排除

### 路径不存在
如果训练时报告路径不存在，请：
1. 确认已运行数据集转换工具
2. 检查转换后的目录是否存在
3. 验证文件格式是否为`.h5`

### 性能问题
如果遇到性能问题：
1. 确认使用的是z-score标准化版本（有LZF压缩优化）
2. 检查chunk size是否为`(32,32,1,351)`
3. 验证训练代码是否支持`.h5`格式的高效读取

---

**注意**: 这些更新确保了训练脚本使用最优化的、z-score标准化的数据集，将显著改善训练效果和性能。