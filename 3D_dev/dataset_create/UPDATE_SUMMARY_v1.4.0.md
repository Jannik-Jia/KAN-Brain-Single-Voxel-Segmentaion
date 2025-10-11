# 数据格式更新总结 v1.4.0

**日期**: 2025-01-11
**变更类型**: 重要更新 - 3D和1D数据分离

---

## 主要变更

### 1. 文件结构变更 ⚠️

**之前 (v1.3.0)**:
```
output_dir/
├── subject001_downsampled.npz        # 包含所有3D和1D数据
├── subject001_metadata.json
└── subject001_qa_metrics.json
```

**现在 (v1.4.0)**:
```
output_dir/
├── 3d/
│   └── subject001_3d.npz            # 只包含3D数据
├── 1d/
│   └── subject001_1d.npz            # 只包含1D数据
├── subject001_metadata.json
└── subject001_qa_metrics.json
```

### 2. 数据内容分配

#### 3D文件 (`subject001_3d.npz`)
- `data_lr`: (Z', X', Y', 351) - 多模态特征
- `proba_labels`: (Z', X', Y', 102) - 概率标签
- `region_mask_lr`: (Z', X', Y') - ROI掩码

#### 1D文件 (`subject001_1d.npz`)
- `multidim_data`: (n_voxels, 351) - 1D特征
- `seg_one_hot`: (102, n_voxels) - 1D概率标签
- `region_seg`: (n_voxels,) - 1D硬标签
- `region`: (Z', X', Y') - ROI掩码（**新增**，用于3D重建）
- `n_voxels`: 标量 - 体素数

### 3. 更新的文件列表

#### 代码文件
1. ✅ `batch_downsampling_pipeline.py`
   - 修改 `__init__()` 支持 `output_3d_dir` 和 `output_1d_dir`
   - 重写 `save_downsampled_data()` 分别保存3D和1D数据
   - 更新 `process_all_subjects()` 的skip逻辑
   - 添加命令行参数 `--output-3d-dir` 和 `--output-1d-dir`

2. ✅ `verify_1d_data.py`
   - 更新为 `verify_downsampled_data(data_dir, subject_id)` 接口
   - 支持从分离的3D和1D文件加载
   - 增加region mask一致性检查
   - 更新命令行参数支持多个被试验证

#### 文档文件
3. ✅ `DOWNSAMPLED_DATA_FORMAT.md`
   - 更新版本至 v1.4.0
   - 重写文件结构说明
   - 添加快速参考表
   - 更新所有使用示例以反映新文件结构
   - 添加版本更新说明

---

## 使用变更

### 旧方式 (v1.3.0)
```python
data = np.load('subject001_downsampled.npz')
X = data['multidim_data']
y = data['seg_one_hot']
```

### 新方式 (v1.4.0)
```python
# 分别加载3D和1D数据
data_3d = np.load('output_dir/3d/subject001_3d.npz')
data_1d = np.load('output_dir/1d/subject001_1d.npz')

X = data_1d['multidim_data']
y = data_1d['seg_one_hot']
```

---

## 验证工具更新

### 旧命令
```bash
python verify_1d_data.py subject001_downsampled.npz
```

### 新命令
```bash
# 验证单个被试
python verify_1d_data.py subject001

# 验证多个被试
python verify_1d_data.py subject001 subject002 subject003

# 指定数据目录
python verify_1d_data.py subject001 --data-dir /path/to/downsampling
```

---

## 优势

1. **更好的组织**: 3D和1D数据分开，便于分类管理
2. **独立访问**: 只需要1D数据时不必加载3D数据，节省内存
3. **清晰的结构**: 目录结构更加清晰，易于理解
4. **灵活的存储**: 可以单独备份或迁移3D或1D数据

---

## 兼容性

⚠️ **不向后兼容**: v1.4.0生成的数据与v1.3.0格式不兼容

如果需要使用旧版本数据，需要：
1. 使用旧版本的加载代码
2. 或手动将旧格式数据拆分为新格式

---

## 检查清单

- [x] batch_downsampling_pipeline.py 更新完成
- [x] verify_1d_data.py 更新完成
- [x] DOWNSAMPLED_DATA_FORMAT.md 更新完成
- [x] 所有使用示例已更新
- [x] 命令行参数已添加
- [x] 文档版本号已更新

---

**最后更新**: 2025-01-11
**作者**: Claude Code
