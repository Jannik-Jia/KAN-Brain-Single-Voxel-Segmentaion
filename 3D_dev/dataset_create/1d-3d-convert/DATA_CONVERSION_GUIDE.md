# 3D-1D 数据转换指南

## 概述

本文档详细说明了MRI数据在3D和1D格式之间转换的逻辑、数据结构和维度变化。该转换工具支持降采样和上采样，确保数据在不同维度表示之间的一致性。

## 数据格式说明

### 3D数据格式（转换后）

3D数据为完整的MRI体积数据，默认尺寸为 `(384, 336, 256)`，支持降采样到任意尺寸。

| Key | 维度 | 数据类型 | 说明 |
|-----|------|---------|------|
| `data` | (384, 336, 256, 351) | float32 | 4D特征数据，从`multidim_data`转换而来 |
| `region_mask` | (384, 336, 256) | uint8 | 从`region`复制的ROI掩码 |
| `region_labels` | (384, 336, 256) | uint8 | 从`seg_one_hot`转换的3D标签，值范围0-101 |
| `region_seg_3d` | (384, 336, 256) | uint8 | FreeSurfer原始分割标签【不需要，可忽略】 |
| `big_seg` | (384, 336, 256) | uint8 | 直接从原始数据复制的完整分割 |
| `prob_idx` | (384, 336, 256) | uint8 | 被试索引的3D体积（可选） |

### 1D数据格式（原始MAT文件）

1D数据仅包含ROI内的体素，其他区域被剔除以减少数据量。

| Key | 维度 | 数据类型 | 说明 |
|-----|------|---------|------|
| `multidim_data` | (n_voxels, 351) | float32 | 多模态特征数据 |
| `region` | (384, 336, 256) 或 (x, y, z) | bool | ROI区域掩码，3D布尔数组 |
| `big_seg` | (384, 336, 256) 或 (x, y, z) | uint8 | 3D分割FreeSurfer数据 |
| `seg_one_hot` | (102, n_voxels) | float32 | One-hot编码的标签 |
| `region_seg` | (n_voxels,) | uint8 | FreeSurfer原始分割标签【不需要，可忽略】 |
| `n_voxels` | scalar | int | ROI内的体素数量 |
| `original_shape` | (3,) | tuple | 原始3D形状 |
| `downsampled_shape` | (3,) | tuple | 降采样后的3D形状 |

**注意**：`n_voxels` 是 ROI 内的体素数量，通常远小于总体素数。

## 转换逻辑

### 3D → 1D 转换流程

```
原始3D数据 → [可选降采样] → 提取ROI内体素 → 1D数据
```

1. **降采样（可选）**
   - 输入：`(384, 336, 256)`
   - 输出：`(x, y, z)`，例如 `(192, 168, 128)`
   - 方法：使用scipy.ndimage.zoom进行插值

2. **ROI提取**
   - 使用布尔索引：`data[region_mask]`
   - 保证体素顺序的一致性

3. **数据重组**
   - 4D特征 → 2D矩阵：`data (x,y,z,351)` → `multidim_data (n_voxels, 351)`
   - 3D标签 → one-hot编码：`region_labels (x,y,z)` → `seg_one_hot (102, n_voxels)`
   - 3D FreeSurfer → 1D：`region_seg_3d (x,y,z)` → `region_seg (n_voxels,)`【不需要】

### 1D → 3D 转换流程

```
1D数据 → 填充到3D空间 → [可选上采样] → 3D数据
```

1. **3D重构**
   - 创建零填充的3D数组
   - 使用布尔索引填充：`array_3d[region_mask] = array_1d`
   - Key映射：
     - `multidim_data` → `data`
     - `region` → `region_mask`
     - `seg_one_hot` → `region_labels`（通过argmax）
     - `region_seg` → `region_seg_3d`【FreeSurfer标签，不需要】
     - `big_seg` → `big_seg`（直接复制）

2. **上采样（可选）**
   - 从降采样尺寸恢复到目标尺寸
   - 保持数据的空间连续性

## 维度变化示例

### 标准流程（无降采样）

```python
# 3D → 1D
3D输入:
  data: (384, 336, 256, 351)
  region_mask: (384, 336, 256)
  region_labels: (384, 336, 256)
  region_seg_3d: (384, 336, 256)  # FreeSurfer标签【不需要】
  big_seg: (384, 336, 256)

1D输出:
  multidim_data: (n_voxels, 351)  # 例如 (50000, 351)
  region: (384, 336, 256)  # 保持3D
  seg_one_hot: (102, n_voxels)  # 例如 (102, 50000)
  region_seg: (n_voxels,)  # FreeSurfer标签【不需要】
  big_seg: (384, 336, 256)  # 直接复制
```

### 降采样流程

```python
# 3D → 降采样 → 1D
3D原始:
  data: (384, 336, 256, 351)
  region_mask: (384, 336, 256)

3D降采样:
  data: (192, 168, 128, 351)
  region_mask: (192, 168, 128)

1D输出:
  multidim_data: (n_voxels_downsampled, 351)  # 例如 (12000, 351)
  region: (192, 168, 128)  # 降采样后的3D掩码
  seg_one_hot: (102, n_voxels_downsampled)  # 例如 (102, 12000)
  region_seg: (n_voxels_downsampled,)  # FreeSurfer标签【不需要】
  big_seg: (192, 168, 128)  # 降采样后的分割
```

## MATLAB兼容性

### 索引差异
- Python：0-based索引（0-101）
- MATLAB：1-based索引（1-102）

### 维度顺序
- Python：行优先（C order）
- MATLAB：列优先（Fortran order）

### 保存时的调整
```python
# 4D数据保存为MATLAB格式
Python: (x, y, z, features)
保存时转换: (features, x, y, z)
使用 np.asfortranarray() 确保列优先顺序
```

### 加载时的调整
```python
# 从MATLAB加载
MATLAB中: multidim_data (351, n_voxels)
加载后转置: multidim_data.T → (n_voxels, 351)
```

## 关键函数说明

### `convert_3d_to_1d()`
- **输入**：3D数据字典
- **可选参数**：`downsample_shape` - 降采样目标尺寸
- **输出**：1D数据字典（与原始1D格式一致）
- **适用的keys**：
  - 输入：`data`, `region_mask`, `region_labels`, `region_seg_3d`【不需要】, `big_seg`
  - 输出：`multidim_data`, `region`, `seg_one_hot`, `region_seg`【不需要】, `big_seg`

### `convert_1d_to_3d()`
- **输入**：1D数据字典（原始1D格式）
- **可选参数**：`target_shape` - 上采样目标尺寸, `prob_idx` - 被试索引
- **输出**：3D数据字典（与原始3D格式一致）
- **适用的keys**：
  - 输入：`multidim_data`, `region`, `seg_one_hot`, `region_seg`【不需要】, `big_seg`
  - 输出：`data`, `region_mask`, `region_labels`, `region_seg_3d`【不需要】, `big_seg`, `prob_idx`(可选)

### `map_1d_predictions_to_3d()`
- **输入**：1D预测结果 `(n_voxels,)` 或 `(n_voxels, n_classes)`
- **输出**：3D预测结果 `(x, y, z)` 或 `(x, y, z, n_classes)`
- **用途**：将模型预测映射回3D空间用于可视化

## 使用建议

### 内存优化
- 降采样可显著减少内存使用：
  - 原始：`384×336×256×351×4 bytes ≈ 14.5 GB`
  - 降采样50%：`192×168×128×351×4 bytes ≈ 1.8 GB`
  - 仅ROI：`n_voxels×351×4 bytes`（通常 < 100 MB）

### 处理大批量数据
```python
# 推荐流程
for subject in subjects:
    # 1. 加载3D数据
    data_3d = load_3d_data(subject)

    # 2. 降采样并转换为1D（减少内存）
    data_1d = mapper.convert_3d_to_1d(data_3d, downsample_shape=(192, 168, 128))

    # 3. 保存1D数据（文件更小）
    save_1d_data(data_1d)

    # 4. 清理内存
    del data_3d
```

### 验证数据一致性
```python
# 往返测试
data_3d_recovered = mapper.convert_1d_to_3d(data_1d)
data_1d_recovered = mapper.convert_3d_to_1d(data_3d_recovered)

# 验证
assert np.allclose(data_1d['multidim_data'], data_1d_recovered['multidim_data'])
# 验证one-hot标签
labels_original = np.argmax(data_1d['seg_one_hot'], axis=0)
labels_recovered = np.argmax(data_1d_recovered['seg_one_hot'], axis=0)
assert np.array_equal(labels_original, labels_recovered)
```

## 注意事项

1. **体素顺序**：布尔索引 `array[mask]` 保证了体素提取和填充的顺序一致性
2. **标签范围**：FreeSurfer标签固定为0-101（102个类别）
3. **数据类型**：注意保持正确的数据类型（float32 for features, uint8 for labels）
4. **降采样方法**：
   - 掩码和标签使用最近邻插值（order=0）
   - 连续数据使用线性插值（order=1）
5. **空值处理**：ROI外的体素在3D表示中填充为0
6. **⚠️ MATLAB转置判断**：
   - **原始脚本**：仅当 `multidim_data.shape[0] == 351` 时才转置
   - **新工具**：优先使用相同规则，但会警告异常情况
   - **潜在问题**：
     - 极小ROI（n_voxels < 351）可能导致误判
     - 非标准特征数（不是351）需要手动指定
   - **解决方案**：初始化时指定 `expected_features` 参数
   ```python
   mapper = Data3D1DMapper(expected_features=351)  # 明确指定特征数
   ```

## 常见问题

### Q: 为什么1D数据中还保留3D的region？
A: 因为需要知道每个1D体素在3D空间中的位置，用于将预测结果映射回3D。

### Q: region_seg和region_seg_3d是什么？
A: 这两个都是FreeSurfer的原始分割标签，**不需要使用**。我们使用`seg_one_hot`（one-hot编码）和`region_labels`（从one-hot转换的标签）。这两个FreeSurfer相关的key仅为兼容性保留。

### Q: 降采样会影响模型精度吗？
A: 会有一定影响，但可以显著减少计算量。建议根据具体任务平衡精度和效率。

### Q: 如何处理不同大小的输入数据？
A: 工具自动适应输入维度，不硬编码尺寸，支持任意3D形状。

## 完整示例

```python
from data_3d_1d_mapper import Data3D1DMapper

# 初始化
mapper = Data3D1DMapper()

# 1. 加载3D数据
data_3d = mapper.load_data('input_3d.mat')

# 2. 降采样并转换为1D
data_1d = mapper.convert_3d_to_1d(
    data_3d,
    downsample_shape=(192, 168, 128)
)

# 3. 训练模型（伪代码）
model.fit(data_1d['multidim_data'], data_1d['seg_one_hot'])

# 4. 预测
predictions_1d = model.predict(data_1d['multidim_data'])

# 5. 映射回3D
predictions_3d = mapper.map_1d_predictions_to_3d(
    predictions_1d,
    region_mask=data_1d['region']
)

# 6. 保存结果
mapper.save_data({'predictions': predictions_3d}, 'output_3d.mat')
```