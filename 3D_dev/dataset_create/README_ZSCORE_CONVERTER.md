# Z-Score数据集转换工具

## 📋 概述

这个工具专门为您的MRI脑区分割项目创建，可以：

1. **自动读取**您shell脚本中配置的数据集路径
2. **病人内z-score标准化**：对每个病人的351维特征分别标准化
3. **保持空间一致性**：所有体素位置与原数据完全相同
4. **创建patch友好格式**：优化7×7 patch提取性能
5. **保存到新目录**：原数据不受影响

## 🔧 数据集路径配置

工具会自动从这些shell脚本读取数据集路径：
- `training/3D CNN/ResNet/scripts/run_leave_one_out.sh`
- `training/3D CNN/run_all_models_7x7.sh`
- `training/3D CNN/run_all_models_3x3.sh`

当前检测到的路径：
```bash
DATA_DIR_3D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"  # 3D数据
DATA_DIR_1D="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/"              # 1D数据
```

## 🚀 使用方法

### 方法1：快速转换（推荐）
```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create
python quick_convert.py
```

### 方法2：使用配置脚本
```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create
python convert_with_sh_config.py
```

### 方法3：命令行调用
```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create

# 使用默认设置（已优化7×7×1 patch）
python zscore_dataset_converter.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output

# 自定义chunk size（高级用户）
python zscore_dataset_converter.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --chunk-size 16,16,1,351 \
    --log-level DEBUG
```

### 方法4：直接调用转换器
```python
from zscore_dataset_converter import ZScoreDatasetConverter

converter = ZScoreDatasetConverter(
    input_dir="/path/to/your/dataset",
    output_dir="/path/to/output",  # 可选，默认创建 _zscore_normalized 版本
    chunk_size=(32, 32, 1, 351)    # 已优化7×7×1 patch
)

successful, failed = converter.process_all_subjects()
```

## 📂 输出结果

转换后的数据集会保存在：
```
原目录_zscore_normalized/
├── subject1.h5                    # Z-score标准化的数据
├── subject2.h5
├── ...
├── dataset_info.json              # 数据集信息
├── logs/
│   └── zscore_conversion_*.log    # 详细日志
└── zscore_conversion_report_*.json # 转换报告
```

## ✅ 转换内容说明

### 保持不变的数据
- `big_seg`: FreeSurfer完整分割
- `region_mask`: 脑组织掩膜（0/1二值）
- `region_seg_3d`: FreeSurfer标签重构
- `region_labels`: 整数标签(0-101)

### Z-Score标准化的数据
- `data`: **351维多模态特征** ⭐
  - 每个病人独立标准化
  - 每个维度分别计算均值和标准差
  - 公式：`(x - mean) / std`
  - 结果：每个维度均值≈0，标准差≈1

## 🔍 验证机制

转换工具包含完整验证：
```
✅ Z-score标准化效果验证
✅ 空间位置一致性验证
✅ 标签数据完整性验证
✅ 体素数量匹配验证
```

## 📊 性能优化

### 7×7×1 Patch提取优化
- **HDF5分块存储：`(32, 32, 1, 351)`**
- **Z维度=1**: 完美适配单层2D patch提取
- **XY维度=32×32**: 最小化解压开销，每个chunk包含~20个7×7 patch
- **通道维度=351**: 保持所有特征在一起，提高patch读取效率
- **LZF压缩（数据）**: 快速解压，针对频繁patch访问优化
- **GZIP压缩（标签/掩膜）**: 高压缩比，针对非频繁访问优化

### 文件大小
- 原始MAT文件：~3GB
- 转换后H5文件：~200-300MB
- 压缩比：约10-15倍

## ❓ 常见问题

### Q1: 数据集保存在哪里？
**A**: 在原数据集的父目录，例如：
- 原数据：`/path/to/original_dataset/`
- 新数据：`/path/to/original_dataset_zscore_normalized/`

### Q2: 文件名是否一致？
**A**: 基本一致，但扩展名改变：
- 原文件：`subject1.mat`
- 新文件：`subject1.h5`

### Q3: 体素位置是否相同？
**A**: **完全相同！** 每个体素的(x,y,z)位置保持不变

### Q4: 哪些数据被z-score标准化？
**A**: **只有`multidim_data`的351维特征**，其他数据保持原值

### Q5: 如何验证转换正确性？
**A**: 工具自动验证，日志中显示：
```
✅ Subject subject1 processed successfully
   - Z-score normalization: ✅ Passed
   - Spatial consistency: ✅ Passed
```

## 🛠 故障排除

### 路径不存在
如果shell脚本中的路径是服务器路径，工具会提示您输入本地路径。

### 内存不足
工具优化了内存使用，但如果遇到问题：
- 使用测试模式先转换一个文件
- 检查系统内存是否足够（建议8GB+）

### 验证失败
如果验证未通过：
- 检查原始数据完整性
- 查看详细日志文件
- 确认数据格式符合预期

## 📝 使用示例

```bash
# 1. 进入工具目录
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create

# 2. 运行快速转换
python quick_convert.py

# 3. 根据提示选择要转换的数据集类型
# 4. 选择是否先测试一个文件
# 5. 等待转换完成

# 6. 检查结果
ls ../path_to_output_zscore_normalized/
```

## 🎯 下一步

转换完成后，您可以：
1. 更新训练脚本中的数据路径
2. 使用新的z-score标准化数据集训练模型
3. 比较标准化前后的训练效果
4. 享受更好的patch提取性能

---

**注意**: 这个工具专门针对您的数据格式和需求定制，确保了最佳的兼容性和性能。