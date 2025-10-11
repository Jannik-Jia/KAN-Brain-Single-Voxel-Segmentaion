# Downsampled数据格式说明

**版本**: v1.3.0
**日期**: 2025-01-11
**文件类型**: NPZ (NumPy压缩格式)

---

## 📦 输出文件结构

批处理脚本运行后，每个被试会生成3个文件：

```
{subject_id}_downsampled.npz     # 主数据文件（包含3D和1D数据）
{subject_id}_metadata.json       # 元数据（处理参数、坐标映射等）
{subject_id}_qa_metrics.json     # QA指标（质量评估）
```

---

## 🔑 NPZ文件包含的Key

### 必需的3D数据（所有模式）

| Key | 维度 | 数据类型 | 坐标系 | 说明 |
|-----|------|---------|--------|------|
| `data_lr` | `(Z', X', Y', 351)` | `float32` | **原始坐标系(Z,X,Y,C)** | 下采样后的多模态特征数据<br>351个通道包含所有成像模态 |
| `proba_labels` | `(Z', X', Y', 102)` | `float32` | **原始坐标系(Z,X,Y,K)** | 概率标签（软标签）<br>每个体素对应102个脑区的概率分布<br>**注意**：不是严格one-hot，而是概率分布 |
| `region_mask_lr` | `(Z', X', Y')` | `uint8` | **原始坐标系(Z,X,Y)** | 下采样后的ROI掩码<br>值为0或1 |

**典型尺寸**（目标分辨率1.8×1.8×3.0 mm³）：
- `Z' ≈ 18` (约54mm厚的slab)
- `X' ≈ 128`
- `Y' ≈ 104`

### 可选的1D数据（`save_axis_order='orig'`模式）

| Key | 维度 | 数据类型 | 说明 |
|-----|------|---------|------|
| `multidim_data` | `(n_voxels, 351)` | `float32` | 1D特征矩阵<br>每行对应一个ROI内的体素<br>按C-order排列 |
| `seg_one_hot` | `(102, n_voxels)` | `float32` | 1D概率标签（软标签）<br>**注意**：实际是概率分布，不是严格one-hot<br>每列对应一个体素的102个区域概率 |
| `region_seg` | `(n_voxels,)` | `uint8` | 1D区域标签（硬标签）<br>每个体素的最可能区域（argmax） |
| `n_voxels` | 标量 | `int` | ROI内体素总数 |

---

## 📊 数据详细说明

### 1. **data_lr** - 多模态特征数据

**维度**: `(Z', X', Y', 351)`
**坐标系**: **原始输入坐标系(Z,X,Y,C)**
**数据范围**: 因模态而异，部分已归一化

#### 351个通道的组成

| 通道范围 | 数量 | 模态 | 说明 |
|---------|------|------|------|
| 0-14 | 15 | QTI参数 | 量化张量成像参数 |
| 15-224 | 210 | DWI | 扩散加权成像<br>(b_lin, b_plan, b_spher) |
| 225-228 | 4 | CEST参数 | 化学交换饱和转移参数 |
| 229 | 1 | M0_LOW_B1 | 低B1场的M0参考 |
| 230-283 | 54 | Z-spectrum (Low B1) | 低B1场Z谱（归一化） |
| 284 | 1 | M0_HIGH_B1_PRIMARY | 高B1场M0参考（主要） |
| 285 | 1 | M0_HIGH_B1_SECONDARY | 高B1场M0参考（次要） |
| 286-339 | 54 | Z-spectrum (High B1) | 高B1场Z谱（归一化） |
| 340 | 1 | M0_FALLBACK | 备用M0参考 |
| 341 | 1 | MPRAGE | T1加权结构成像 |
| 342-346 | 5 | GRE (QSM_TE) | 梯度回波多TE |
| 347 | 1 | TE_avg | TE平均 |
| 348-349 | 2 | SMWI | 磁化率加权成像 |
| 350 | 1 | QSM | 定量磁化率成像 |

**访问示例**：
```python
import numpy as np

# 加载数据
data = np.load('subject001_downsampled.npz')
data_lr = data['data_lr']  # (Z', X', Y', 351)

# 提取特定模态
mprage = data_lr[..., 341]  # (Z', X', Y') - MPRAGE图像
z_spectrum_low = data_lr[..., 230:284]  # (Z', X', Y', 54) - 低B1 Z谱
dwi = data_lr[..., 15:225]  # (Z', X', Y', 210) - DWI数据
```

---

### 2. **proba_labels** - 概率标签

**维度**: `(Z', X', Y', 102)`
**坐标系**: **原始输入坐标系(Z,X,Y,K)**
**数据类型**: `float32`
**数值范围**: [0.0, 1.0]
**概率和**: 每个体素的102个概率之和≈1.0

#### 重要特性

⚠️ **这不是严格的one-hot编码！**

- 每个体素可能属于多个区域（软标签/概率分布）
- 反映了下采样过程中的部分容积效应
- 生成方式：对高分辨率one-hot标签应用MPRAGE PSF平滑 + 线性重采样

**概率解释**：
- `proba_labels[z, x, y, k]` = 位置(z,x,y)的体素属于第k个区域的概率
- 如果某体素横跨多个脑区边界，会有多个非零概率

**访问示例**：
```python
# 获取某个体素的区域概率分布
voxel_probs = proba_labels[5, 10, 15, :]  # (102,)
print(f"概率和: {voxel_probs.sum():.6f}")  # 应该接近1.0

# 找到最可能的区域
most_likely_region = np.argmax(voxel_probs)
confidence = voxel_probs[most_likely_region]
print(f"最可能区域: {most_likely_region}, 置信度: {confidence:.4f}")

# 找到所有可能的区域（概率>5%）
possible_regions = np.where(voxel_probs > 0.05)[0]
print(f"可能的区域: {possible_regions}")
```

#### 102个区域标签

标签值范围：0-101

- **0-101**: 所有都是有效的脑区标签（FreeSurfer DKT atlas + Aseg）
- ⚠️ **注意**: 标签0**不是**背景！它也代表一个有效的脑区
- 背景由 `region_mask_lr` 定义（值为0的位置）

详细区域列表请参考FreeSurfer文档。

---

### 3. **region_mask_lr** - ROI掩码

**维度**: `(Z', X', Y')`
**坐标系**: **原始输入坐标系(Z,X,Y)**
**数据类型**: `uint8`
**数值**: 0 (背景) 或 1 (ROI内)

**用途**：
- 定义感兴趣区域（脑内体素）
- 用于提取1D数据
- 用于忽略背景体素

**访问示例**：
```python
region_mask_lr = data['region_mask_lr']  # (Z', X', Y')

# 统计ROI体素数
n_voxels = np.sum(region_mask_lr)
print(f"ROI内体素数: {n_voxels}")

# 提取ROI内的特征（转换为1D）
features_in_roi = data_lr[region_mask_lr > 0]  # (n_voxels, 351)
```

---

### 4. **multidim_data** - 1D特征矩阵

**维度**: `(n_voxels, 351)`
**数据类型**: `float32`
**排列顺序**: **C-order（行优先）**

**生成方式**：
```python
# 从3D data_lr提取ROI内的体素（按C-order自动展平）
multidim_data = data_lr[region_mask_lr > 0]
```

**重要**：行顺序与3D空间的对应关系

- 第i行对应3D空间中按C-order遍历的第i个ROI体素
- 可以通过以下方式恢复对应关系：

```python
# 获取ROI体素的3D坐标（按C-order）
roi_coords = np.column_stack(np.where(region_mask_lr))  # (n_voxels, 3)

# 第i行特征对应的3D位置
i = 100
z, x, y = roi_coords[i]
feature_1d = multidim_data[i, :]  # (351,)
feature_3d = data_lr[z, x, y, :]  # (351,)
assert np.allclose(feature_1d, feature_3d)  # 应该相等
```

**与3D数据的对应**：
```python
# 验证对应关系
assert np.allclose(multidim_data, data_lr[region_mask_lr > 0])
```

---

### 5. **seg_one_hot** - 1D概率标签

**维度**: `(102, n_voxels)`
**数据类型**: `float32`
**数值范围**: [0.0, 1.0]
**概率和**: 每列（每个体素）的和≈1.0

⚠️ **命名注意事项**：虽然叫"one_hot"，但实际是**概率分布**（软标签），不是严格的one-hot编码！

**生成方式**：
```python
# 从3D proba_labels提取并转置
seg_one_hot = proba_labels[region_mask_lr > 0].T  # (102, n_voxels)
```

**数据组织**：
- 第k行：所有体素属于第k个区域的概率
- 第j列：第j个体素的区域概率分布（102维）

**访问示例**：
```python
seg_one_hot = data['seg_one_hot']  # (102, n_voxels)

# 获取某个体素的区域分布
voxel_idx = 100
voxel_probs = seg_one_hot[:, voxel_idx]  # (102,)
print(f"体素{voxel_idx}的概率分布: 和={voxel_probs.sum():.6f}")

# 获取某个区域在所有体素中的概率
region_idx = 10
region_probs_all_voxels = seg_one_hot[region_idx, :]  # (n_voxels,)
mean_prob = region_probs_all_voxels.mean()
print(f"区域{region_idx}的平均概率: {mean_prob:.4f}")

# 找到属于某个区域的体素（概率>阈值）
threshold = 0.5
voxels_in_region = np.where(seg_one_hot[region_idx, :] > threshold)[0]
print(f"高置信度属于区域{region_idx}的体素: {len(voxels_in_region)}个")
```

**与3D数据的对应**：
```python
# 验证对应关系
proba_in_roi = proba_labels[region_mask_lr > 0]  # (n_voxels, 102)
assert np.allclose(seg_one_hot, proba_in_roi.T)
```

---

### 6. **region_seg** - 1D硬标签

**维度**: `(n_voxels,)`
**数据类型**: `uint8`
**数值范围**: 0-101

**生成方式**：对proba_labels取argmax
```python
# 从概率分布生成硬标签
labels_3d = np.argmax(proba_labels, axis=-1)  # (Z', X', Y')
region_seg = labels_3d[region_mask_lr > 0]  # (n_voxels,)
```

**用途**：
- 适用于需要离散标签的算法
- 分割可视化
- 区域统计

**访问示例**：
```python
region_seg = data['region_seg']  # (n_voxels,)

# 统计每个区域的体素数
from collections import Counter
region_counts = Counter(region_seg)
print("各区域体素统计:")
for region_id, count in sorted(region_counts.items()):
    print(f"  区域{region_id}: {count}个体素")

# 提取特定区域的特征
region_id = 10
mask_region = region_seg == region_id
features_in_region = multidim_data[mask_region]  # (n_region_voxels, 351)
print(f"区域{region_id}包含{features_in_region.shape[0]}个体素")
```

**关系**：
```python
# region_seg是seg_one_hot的argmax
assert np.array_equal(region_seg, np.argmax(seg_one_hot, axis=0))
```

---

### 7. **n_voxels** - 体素数

**数据类型**: `int` (标量)
**含义**: ROI内的总体素数

**关系**：
```python
n_voxels = data['n_voxels']

# 应该等于其他数组的尺寸
assert n_voxels == np.sum(region_mask_lr > 0)
assert n_voxels == multidim_data.shape[0]
assert n_voxels == seg_one_hot.shape[1]
assert n_voxels == region_seg.shape[0]
```

---

## 🔄 3D-1D数据对应关系

### C-order遍历规则

1D数据按**C-order（行优先，row-major）**排列：

```
对于3D数组 A[Z, X, Y]，C-order遍历顺序为：
A[0,0,0], A[0,0,1], ..., A[0,0,Y-1],
A[0,1,0], A[0,1,1], ..., A[0,1,Y-1],
...
A[0,X-1,Y-1],
A[1,0,0], ...
```

### 对应关系验证

```python
import numpy as np

# 加载数据
data = np.load('subject001_downsampled.npz')
data_lr = data['data_lr']
region_mask_lr = data['region_mask_lr']
multidim_data = data['multidim_data']
seg_one_hot = data['seg_one_hot']
proba_labels = data['proba_labels']

# 验证特征对应
features_from_3d = data_lr[region_mask_lr > 0]  # (n_voxels, 351)
assert np.allclose(features_from_3d, multidim_data)

# 验证标签对应
labels_from_3d = proba_labels[region_mask_lr > 0]  # (n_voxels, 102)
assert np.allclose(labels_from_3d.T, seg_one_hot)

# 恢复3D到1D的映射
n_voxels = np.sum(region_mask_lr > 0)
for i in np.random.choice(n_voxels, 10):
    # 找到第i个体素的3D位置
    roi_indices = np.where(region_mask_lr.ravel(order='C'))[0]
    linear_idx = roi_indices[i]
    z, x, y = np.unravel_index(linear_idx, region_mask_lr.shape, order='C')

    # 验证对应
    assert np.allclose(multidim_data[i, :], data_lr[z, x, y, :])
    assert np.allclose(seg_one_hot[:, i], proba_labels[z, x, y, :])

print("✓ 3D-1D对应关系验证通过！")
```

---

## 📝 使用示例

### 示例1：加载并查看数据

```python
import numpy as np

# 加载数据
data = np.load('subject001_downsampled.npz')

# 查看包含的key
print("文件包含的数据:")
for key in data.files:
    if isinstance(data[key], np.ndarray):
        print(f"  {key}: {data[key].shape}")
    else:
        print(f"  {key}: {data[key]}")

# 提取数据
data_lr = data['data_lr']
proba_labels = data['proba_labels']
region_mask_lr = data['region_mask_lr']
multidim_data = data['multidim_data']
seg_one_hot = data['seg_one_hot']
region_seg = data['region_seg']
n_voxels = int(data['n_voxels'])

print(f"\nROI体素数: {n_voxels}")
print(f"3D shape: {data_lr.shape}")
print(f"1D shape: {multidim_data.shape}")
```

### 示例2：提取特定区域的特征

```python
# 选择海马区（假设ID为17）
hippocampus_id = 17

# 方法1：使用1D数据
mask_hippo = region_seg == hippocampus_id
features_hippo_1d = multidim_data[mask_hippo]
print(f"海马体素数: {features_hippo_1d.shape[0]}")

# 方法2：使用3D数据
labels_3d = np.argmax(proba_labels, axis=-1)
mask_hippo_3d = labels_3d == hippocampus_id
features_hippo_3d = data_lr[mask_hippo_3d]
print(f"海马体素数(3D): {features_hippo_3d.shape[0]}")
```

### 示例3：计算区域平均信号

```python
# 计算每个区域的MPRAGE平均信号
mprage_channel = 341

for region_id in range(1, 102):
    # 使用软标签（概率加权平均）
    weights = seg_one_hot[region_id, :]  # (n_voxels,)
    mprage_values = multidim_data[:, mprage_channel]  # (n_voxels,)

    # 加权平均
    if weights.sum() > 0:
        weighted_mean = np.sum(weights * mprage_values) / weights.sum()
        print(f"区域{region_id} MPRAGE加权平均: {weighted_mean:.2f}")
```

### 示例4：使用3D-1D转换工具

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '1d-3d-convert'))

from data_3d_1d_mapper import Data3D1DMapper

# 创建mapper
mapper = Data3D1DMapper()

# 构建3D数据字典
data_3d = {
    'data': data_lr,
    'region_mask': region_mask_lr,
    'proba_labels': proba_labels
}

# 转换为1D
data_1d = mapper.convert_3d_to_1d(data_3d)

# 验证
assert np.allclose(data_1d['multidim_data'], multidim_data)
assert np.allclose(data_1d['seg_one_hot'], seg_one_hot)

print("✓ 转换验证通过")
```

---

## ⚠️ 重要注意事项

### 1. 区域标签0不是背景！

🚨 **非常重要**：区域标签值0是一个**有效的脑区标签**，不是背景！

**区分**：
- **区域标签** (`proba_labels`, `region_seg`): 值范围0-101，**所有都是有效脑区**
  - 标签0代表某个具体的脑区（例如可能是左侧大脑白质）
  - 标签1-101代表其他脑区

- **背景掩码** (`region_mask_lr`): 值为0或1
  - `region_mask_lr == 0`: 表示背景（脑外、头骨等）
  - `region_mask_lr == 1`: 表示ROI内（脑内体素）

**示例**：
```python
# 错误的理解
bad_brain_voxels = (region_seg != 0)  # ✗ 错误！会排除标签0的脑区

# 正确的理解
brain_voxels = (region_mask_lr > 0)  # ✓ 正确！使用mask区分脑内外
label_0_voxels = (region_seg == 0) & (region_mask_lr > 0)  # 标签0的脑区体素

# 统计所有102个区域（包括0）
for label in range(102):  # 0-101，都是有效标签
    count = np.sum((region_seg == label) & (region_mask_lr > 0))
    if count > 0:
        print(f"区域{label}: {count}个体素")
```

### 2. 坐标系统

**3D数据坐标系**: 保持原始输入坐标系 **(Z, X, Y, C)**
- Z: Superior-Inferior (上下)
- X: Anterior-Posterior (前后)
- Y: Left-Right (左右)

**为什么使用原始坐标系？**
- 与MATLAB数据保持一致
- 便于使用现有的1D-3D转换工具
- 支持直接用C-order展平和恢复

### 3. 概率标签 vs One-hot

⚠️ **`seg_one_hot` 和 `proba_labels` 不是严格的one-hot编码！**

**区别**：
- **One-hot**: 每个体素只属于一个类别，其他为0
  - 例如：`[0, 0, 1, 0, 0, ...]` (只有一个1)

- **概率分布（我们的数据）**: 每个体素可能属于多个类别
  - 例如：`[0.05, 0.15, 0.60, 0.10, 0.10, ...]` (多个非零值，和为1)

**为什么使用概率分布？**
1. 反映下采样过程中的不确定性
2. 保留部分容积效应信息
3. 更适合软分类和概率建模

### 4. 内存优化

对于大规模数据处理：

```python
# 只加载需要的数据
with np.load('subject001_downsampled.npz') as data:
    # 只读取1D数据
    multidim_data = data['multidim_data']
    region_seg = data['region_seg']
    # 不加载3D数据，节省内存

# 或使用mmap模式（不加载到内存）
data = np.load('subject001_downsampled.npz', mmap_mode='r')
```

### 5. 数据验证

建议在使用前验证数据完整性：

```bash
# 使用提供的验证脚本
python verify_1d_data.py subject001_downsampled.npz
```

---

## 📚 相关文档

- **下采样方法详细说明**: `DOWNSAMPLING_README.md`
- **快速开始指南**: `QUICKSTART.md`
- **Bug修复记录**: `BUGFIX_v1.3.md`
- **HDF5加载修复**: `HDF5_LOADING_FIX.md`
- **3D-1D转换工具**: `1d-3d-convert/data_3d_1d_mapper.py`

---

## 📞 支持

如有问题，请联系项目维护者或查看相关文档。

**最后更新**: 2025-01-11
**版本**: v1.3.0
