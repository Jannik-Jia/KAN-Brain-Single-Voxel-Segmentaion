# 自动数据验证功能

**版本**: v1.4.0
**日期**: 2025-01-11

---

## 概述

批处理脚本现在会在保存每个被试数据后**自动进行验证**，确保数据质量和完整性。

---

## 验证内容

每个被试在保存后会自动进行以下验证：

### 1. Key完整性检查
- 检查3D文件包含: `data_lr`, `proba_labels`, `region_mask_lr`
- 检查1D文件包含: `multidim_data`, `seg_one_hot`, `region_seg`, `region`, `n_voxels`

### 2. 体素数一致性
- `region_mask_lr`中的体素数 = `n_voxels`
- `multidim_data`行数 = `n_voxels`
- `seg_one_hot`列数 = `n_voxels`

### 3. 数据形状验证
- 检查所有数组的形状是否符合预期
- 验证3D和1D数据的维度匹配

### 4. 概率和验证
- 3D `proba_labels`: 每个体素的102个概率之和 ≈ 1.0
- 1D `seg_one_hot`: 每列的102个概率之和 ≈ 1.0
- 容差范围: [0.99, 1.01]

### 5. 3D-1D对应关系
- 特征数据对应: `data_lr[mask]` ≈ `multidim_data`
- 标签数据对应: `proba_labels[mask].T` ≈ `seg_one_hot`
- 最大差异阈值: 1e-5

---

## 日志输出

### 验证通过
```
步骤5: 验证保存的数据
验证数据: subject001
✓ 数据验证通过: subject001
✅ 被试 subject001 处理成功（已验证）
   处理时间: 45.2秒
   3D数据: 25.3MB → output_dir/3d/subject001_3d.npz
   3D shape: (18, 128, 104, 351)
   1D数据: 12.1MB → output_dir/1d/subject001_1d.npz
   1D shape: (17523, 351)
   ROI体素数: 17523
   总大小: 37.4MB
   验证状态: ✓ 通过
```

### 验证失败
```
步骤5: 验证保存的数据
验证数据: subject002
⚠ 数据验证发现问题: subject002
  - prob_sum_valid_3d: FAILED
  - correspondence_valid: FAILED
⚠️ 被试 subject002 处理完成，但验证失败: Some checks failed
   处理时间: 43.8秒
   ...
   验证状态: ✗ 失败
```

---

## 批量报告

### 统计信息
批量处理完成后，报告会包含验证统计：

```
================================================================================
生成批量Downsampling报告
================================================================================
处理统计:
  总处理数: 100
  成功处理: 98
  处理失败: 2
  验证通过: 95
  验证失败: 3
  总时间: 125.3分钟
  最终内存: 2458.3 MB
```

### 验证失败详情
如果有验证失败的被试，会显示详细信息：

```
验证失败的被试:
  - subject023
    原因: Some checks failed
      × prob_sum_valid_3d
      × correspondence_valid
  - subject045
    原因: Missing required keys
      × keys_1d
  - subject067
    原因: Some checks failed
      × n_voxels_match
```

---

## CSV报告字段

CSV摘要文件现在包含验证信息：

| 字段 | 说明 |
|------|------|
| `verified` | True/False - 验证是否通过 |
| `verification_reason` | 验证结果说明 |
| `n_voxels` | ROI体素数 |
| `prob_sum_3d` | 3D概率和均值 |
| `prob_sum_1d` | 1D概率和均值 |
| `max_feat_diff` | 3D-1D特征最大差异 |
| `max_label_diff` | 3D-1D标签最大差异 |

---

## JSON报告

完整的验证详情保存在JSON报告中：

```json
{
  "subject_id": "subject001",
  "status": "success",
  "verification": {
    "verified": true,
    "reason": "All checks passed",
    "checks": {
      "keys_3d": true,
      "keys_1d": true,
      "n_voxels_match": true,
      "n_voxels": 17523,
      "multidim_data_shape": true,
      "seg_one_hot_shape": true,
      "prob_sum_3d": 1.0000234,
      "prob_sum_1d": 0.9999876,
      "prob_sum_valid_3d": true,
      "prob_sum_valid_1d": true,
      "max_feat_diff": 2.3841858e-08,
      "max_label_diff": 1.4901161e-08,
      "correspondence_valid": true
    }
  }
}
```

---

## 手动验证

如果需要重新验证某个被试的数据，可以使用独立的验证脚本：

```bash
# 验证单个被试
python verify_1d_data.py subject001

# 验证多个被试
python verify_1d_data.py subject001 subject002 subject003

# 指定数据目录
python verify_1d_data.py subject001 --data-dir /path/to/downsampling
```

---

## 依赖

自动验证功能依赖于 `data_3d_1d_mapper.py`。如果该模块不可用，验证将被跳过，并在日志中显示警告：

```
Warning: data_3d_1d_mapper not available, data verification will be skipped
```

---

## 最佳实践

1. **检查验证报告**: 批处理完成后，检查有多少被试验证失败
2. **排查验证失败**: 对于验证失败的被试，查看详细的检查结果
3. **重新处理**: 如果验证失败是由于数据问题，考虑重新处理该被试
4. **保留日志**: 验证日志对于追溯数据质量问题非常重要

---

## 优势

1. **质量保证**: 每个被试数据保存后立即验证，确保数据正确性
2. **早期发现**: 在批处理过程中就发现问题，而不是在使用数据时
3. **详细报告**: 提供完整的验证统计和失败详情
4. **自动化**: 无需手动运行验证脚本，完全自动化

---

**最后更新**: 2025-01-11
**相关文件**: 
- `batch_downsampling_pipeline.py` (主脚本)
- `verify_1d_data.py` (独立验证脚本)
- `data_3d_1d_mapper.py` (验证依赖)
