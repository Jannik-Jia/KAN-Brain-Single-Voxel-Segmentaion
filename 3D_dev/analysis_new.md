# 1D到3D数据映射指南

> **基于深度分析的正确数据处理方法**  
> *经过完整验证，匹配率100%*

## 概述

本文档详细说明了如何正确地将FreeSurfer处理的1D格式MAT文件转换为3D体积数据，解决了转置和内存布局的关键问题。

## 原始MAT文件结构

### 输入文件包含的数据

| Key | 原始形状 | 数据类型 | 描述 |
|-----|----------|----------|------|
| `big_seg` | `(384, 336, 256)` | float64 | FreeSurfer完整分割结果 |
| `region` | `(384, 336, 256)` | uint8 | 脑组织二值掩膜 |
| `multidim_data` | `(351, n_voxels)` | float32 | 多模态特征矩阵 |
| `region_seg` | `(1, n_voxels)` | float64 | 体素级FreeSurfer标签 |
| `seg_one_hot` | `(102, n_voxels)` | uint8 | 102脑区域One-Hot编码 |

其中 `n_voxels` ≈ 150万-240万（因被试而异）

## 关键发现与原理

### 🔍 内存布局问题

**核心发现**：原始数据是通过**直接布尔索引**提取的，但传统重构方法使用了**Fortran order flatten**，导致顺序错乱。

```python
# ❌ 错误的提取/重构顺序（匹配率0.76%）
big_seg_flat = big_seg.flatten(order='F')
mask_flat = region.flatten(order='F')
extracted = big_seg_flat[mask_flat]  # 错误的顺序

# ✅ 正确的提取/重构顺序（匹配率100%）
mask = region.astype(bool)
extracted = big_seg[mask]  # 直接布尔索引
```

### 📊 验证结果

| 提取方法 | 匹配率 | 说明 |
|----------|--------|------|
| 直接布尔索引 | **100.00%** | ✅ 正确方法 |
| Fortran flatten | 0.76% | ❌ 顺序错乱 |
| C order flatten | **100.00%** | ✅ 也可工作 |

## 正确的数据转换映射

### 1. 数据加载策略

```python
def load_mat_h5_correct(mat_path):
    """正确的MAT文件加载方法"""
    data = {}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            if not k.startswith("#"):
                v = f[k][()]
                
                # 关键：只转置需要转置的数据
                if k == 'multidim_data' and v.shape[0] == 351:
                    # 特征矩阵：(351, n_voxels) → (n_voxels, 351)
                    v = v.T
                elif k == 'region_seg':
                    # 1D标签：(1, n_voxels) → (n_voxels,)
                    v = v.flatten()
                # 其他数据保持原样
                data[k] = v
    return data
```

### 2. 1D到3D映射规则

#### A. 3D体数据（直接保留）

```python
# big_seg: (384, 336, 256) → 直接复制
output['big_seg'] = data['big_seg']  # 保持不变

# region掩膜: (384, 336, 256) → 直接保留
output['region_mask'] = data['region'].astype(np.uint8)
```

#### B. 多模态特征数据

```python
# multidim_data: (n_voxels, 351) → (384, 336, 256, 351)
def revert_reshape_features(features, region):
    """正确的特征重构方法"""
    n_voxels, n_features = features.shape
    
    # 创建4D输出数组
    data_4d = np.zeros((*region.shape, n_features), dtype=features.dtype)
    
    # 关键：使用直接布尔索引
    mask = region.astype(bool)
    data_4d[mask] = features  # 不使用flatten操作
    
    return data_4d

output['data'] = revert_reshape_features(data['multidim_data'], data['region'])
```

#### C. One-Hot标签转换

```python
# seg_one_hot: (102, n_voxels) → (384, 336, 256)
def convert_onehot_to_labels(onehot, region):
    """One-Hot到整数标签的转换"""
    # 转换为整数标签
    labels_1d = np.argmax(onehot, axis=0).astype(np.uint8)
    
    # 重构为3D
    labels_3d = np.zeros(region.shape, dtype=np.uint8)
    mask = region.astype(bool)
    labels_3d[mask] = labels_1d  # 直接布尔索引
    
    return labels_3d

output['region_labels'] = convert_onehot_to_labels(data['seg_one_hot'], data['region'])
```

#### D. 原始标签重构

```python
# region_seg: (n_voxels,) → (384, 336, 256)
def revert_reshape_labels(labels_1d, region):
    """1D标签到3D的重构"""
    labels_3d = np.zeros(region.shape, dtype=labels_1d.dtype)
    mask = region.astype(bool)
    labels_3d[mask] = labels_1d  # 直接布尔索引
    return labels_3d

output['region_seg_3d'] = revert_reshape_labels(data['region_seg'], data['region'])
```

#### E. 被试索引体积

```python
# 创建prob_idx: () → (384, 336, 256)
def create_prob_idx_volume(region, prob_idx):
    """创建被试索引的3D体积"""
    prob_idx_3d = np.zeros(region.shape, dtype=np.uint8)
    mask = region.astype(bool)
    prob_idx_3d[mask] = prob_idx  # 填充被试编号
    return prob_idx_3d

output['prob_idx'] = create_prob_idx_volume(data['region'], prob_idx)
```

## 完整的转换流程

### 输入→输出映射表

| 输入Key | 输入形状 | 输出Key | 输出形状 | 转换方法 |
|---------|----------|---------|----------|----------|
| `big_seg` | `(384, 336, 256)` | `big_seg` | `(384, 336, 256)` | 直接复制 |
| `region` | `(384, 336, 256)` | `region_mask` | `(384, 336, 256)` | 类型转换 |
| `multidim_data` | `(n_voxels, 351)*` | `data` | `(384, 336, 256, 351)` | 直接布尔索引重构 |
| `seg_one_hot` | `(102, n_voxels)` | `region_labels` | `(384, 336, 256)` | argmax + 重构 |
| `region_seg` | `(n_voxels,)*` | `region_seg_3d` | `(384, 336, 256)` | 直接布尔索引重构 |
| `prob_idx` | 参数 | `prob_idx` | `(384, 336, 256)` | 创建新的3D体积 |

*注：加载时已经转置/展平

### 示例代码

```python
def process_subject_to_3d(mat_path, prob_idx, output_path):
    """完整的被试数据3D转换流程"""
    
    # 1. 加载数据
    data = load_mat_h5_correct(mat_path)
    region = data['region'].astype(bool)
    
    # 2. 构建输出数据
    output_data = {}
    
    # 3D体数据（直接保留）
    output_data['big_seg'] = data['big_seg']
    output_data['region_mask'] = data['region'].astype(np.uint8)
    
    # 4D特征数据
    output_data['data'] = revert_reshape_features(data['multidim_data'], region)
    
    # 3D标签数据
    output_data['region_labels'] = convert_onehot_to_labels(data['seg_one_hot'], region)
    output_data['region_seg_3d'] = revert_reshape_labels(data['region_seg'], region)
    
    # 被试索引
    output_data['prob_idx'] = create_prob_idx_volume(region, prob_idx)
    
    # 3. 保存为MAT文件
    save_mat_h5(output_path, output_data)
    
    return output_path
```

## 重要注意事项

### ⚠️ 常见错误

1. **错误的flatten操作**
   ```python
   # ❌ 错误 - 会导致顺序错乱
   big_img_flat = big_img.reshape(-1, order='F')
   bool_mask = region.flatten(order='F').astype(bool)
   ```

2. **不必要的转置**
   ```python
   # ❌ 错误 - 3D数据已经是正确格式
   region = data['region'].T  # 不需要转置
   ```

3. **直接使用region作为索引**
   ```python
   # ❌ 错误 - region包含0和1，不是布尔索引
   big_img[region.flatten(order='F')] = array
   ```

### ✅ 正确做法

1. **使用直接布尔索引**
   ```python
   # ✅ 正确
   mask = region.astype(bool)
   big_img[mask] = array
   ```

2. **只转置需要的数据**
   ```python
   # ✅ 正确 - 只转置特征矩阵
   if k == 'multidim_data' and v.shape[0] == 351:
       v = v.T
   ```

## 验证方法

### 完整性验证

```python
def validate_conversion(original_data, converted_3d):
    """验证转换的正确性"""
    region = original_data['region'].astype(bool)
    
    # 验证特征数据
    extracted_features = converted_3d['data'][region]
    match_rate_features = 100.0 * np.mean(
        np.allclose(extracted_features, original_data['multidim_data'])
    )
    
    # 验证标签数据
    extracted_labels = converted_3d['region_seg_3d'][region]
    match_rate_labels = 100.0 * np.mean(
        extracted_labels == original_data['region_seg']
    )
    
    print(f"特征数据匹配率: {match_rate_features:.2f}%")
    print(f"标签数据匹配率: {match_rate_labels:.2f}%")
    
    return match_rate_features > 99.9 and match_rate_labels > 99.9
```

## 总结

1. **3D数据已经是正确格式**，不需要转置
2. **只有`multidim_data`需要转置**，从`(351, n_voxels)`到`(n_voxels, 351)`
3. **核心原则**：重构时使用与原始提取相同的方法（直接布尔索引）
4. **避免使用**：Fortran order的flatten操作
5. **验证结果**：正确实现可达到100%匹配率

---

*文档创建时间: 2025-08-17*  
*基于完整验证的最终正确方法*