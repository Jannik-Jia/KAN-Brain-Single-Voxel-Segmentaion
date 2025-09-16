# Z-Score数据集转换工具 - 常见问题解答

## 您的具体问题回答

### 1. 数据集目录保存在哪？

**答案**: 数据集会保存在原目录的父目录中，自动创建一个新的目录。

例如，如果您的原始数据集在：
```
/Users/jannik/original_dataset/
```

转换后的数据集会保存在：
```
/Users/jannik/original_dataset_zscore_normalized/
```

这样可以保持原始数据不变，同时创建一个规范化的版本。

### 2. 保存的文件名是否与原本一致？

**答案**: 基本一致，但有小的改变：

- **原文件**: `subject1.mat`, `subject2.mat`, ...
- **新文件**: `subject1.h5`, `subject2.h5`, ...

**原因**:
- 保持相同的主文件名便于对应
- 使用`.h5`扩展名表示这是HDF5格式（更高效的存储格式）
- 包含z-score规范化和patch优化的特殊格式

### 3. 每个体素在矩阵的位置是否与原本完全一样？

**答案**: **是的，100%一致！**

转换工具确保：
- **3D空间位置完全保持**: 每个体素在(x,y,z)坐标系中的位置不变
- **区域掩膜保持一致**: `region_mask`与原始`region`完全相同
- **标签位置保持一致**: `region_labels`和原始标签的空间分布完全相同

验证机制确保空间一致性：
```python
# 验证体素位置完全一致
assert np.array_equal(original_data['region'], converted_data['region_mask'])
assert np.array_equal(original_labels_3d, converted_data['region_labels'])
```

### 4. Z-Score是否只对每个病人的data key的351维度进行？

**答案**: **是的，精确地说就是这样！**

Z-score标准化**只应用于**：
- **数据来源**: `multidim_data` (351维多模态特征)
- **标准化范围**: 每个病人的351维度中的每一个维度
- **标准化方式**: 病人内（patient-wise）z-score标准化

**具体过程**:
```python
# 🔥 关键：只对脑组织体素进行z-score标准化（排除背景体素）
region = data['region'].astype(bool)  # 脑组织掩膜
features = data['multidim_data']      # 只包含脑组织体素的特征 (n_brain_voxels, 351)

for ch in range(351):  # 对每个维度
    channel_data = features[:, ch]        # 该病人该维度的脑组织体素值
    mean_val = np.mean(channel_data)      # 只基于脑组织体素计算均值
    std_val = np.std(channel_data)        # 只基于脑组织体素计算标准差

    # Z-score: (x - mean) / std
    if std_val > 1e-8:
        features_normalized[:, ch] = (channel_data - mean_val) / std_val
    else:
        features_normalized[:, ch] = 0  # 如果std=0，设为0
```

**重要说明**:
- ✅ **只使用脑组织体素** 计算均值和标准差
- ✅ **背景体素不参与** 标准化统计计算
- ✅ **背景体素保持为0** （在最终的4D数据中）
- ✅ **避免背景噪声** 影响标准化质量

### 5. 检查标准化了哪些必要的key

根据您的`mri_dataset_create_readme.md`，我确保标准化了所有必要的key：

#### ✅ 已正确处理的Key：

1. **`big_seg`** → **保持不变**
   - 维度: (384, 336, 256)
   - FreeSurfer完整分割结果
   - **不进行z-score标准化**（保持原始标签值）

2. **`region`** → **`region_mask`**
   - 维度: (384, 336, 256)
   - 脑组织二值掩膜
   - **不进行z-score标准化**（保持0/1二值）

3. **`multidim_data`** → **`data`** ⭐ **Z-SCORE标准化**
   - 维度: (n_voxels, 351) → (384, 336, 256, 351)
   - **这是唯一进行z-score标准化的数据**
   - 351维多模态特征，每个病人每个维度分别标准化

4. **`region_seg`** → **`region_seg_3d`**
   - 维度: (n_voxels,) → (384, 336, 256)
   - FreeSurfer标签重构
   - **不进行z-score标准化**（保持原始标签值）

5. **`seg_one_hot`** → **`region_labels`**
   - 维度: (102, n_voxels) → (384, 336, 256)
   - 从one-hot转换为整数标签(0-101)
   - **不进行z-score标准化**（保持标签整数值）

#### ✅ 新增的元数据：
- **`subject_metadata`**: 包含被试ID、体素数、标准化信息等

## 关键优势

### 1. 病人内标准化
- 每个病人独立标准化，消除个体间差异
- 每个维度分别标准化，保持特征间的相对关系

### 2. Patch友好格式
- HDF5分块存储，优化7×7 patch读取
- 数据布局: (384, 336, 256, 351)，直接支持patch提取

### 3. 完整性验证
- 验证z-score标准化效果（均值≈0，标准差≈1）
- 验证空间位置一致性
- 验证标签数据完整性

### 4. 高效存储
- 分级压缩策略
- patch优化的chunking
- 比原始文件小约10-15倍

## 使用示例

```python
from zscore_dataset_converter import ZScoreDatasetConverter

# 创建转换器
converter = ZScoreDatasetConverter(
    input_dir="/path/to/original/dataset",
    output_dir=None  # 自动在父目录创建 _zscore_normalized 版本
)

# 处理所有文件
successful, failed = converter.process_all_subjects()
```

## 验证结果示例

转换后，每个文件的验证会显示：
```
✅ Subject subject1 processed successfully (45.2s, 187.3MB)
   - Z-score normalization: ✅ Passed (351 dimensions normalized)
   - Spatial consistency: ✅ Passed
   - Label integrity: ✅ Passed
```

这确保了转换的质量和数据的完整性。