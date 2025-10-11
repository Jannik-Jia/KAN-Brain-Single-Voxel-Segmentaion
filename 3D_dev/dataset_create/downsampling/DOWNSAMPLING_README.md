# MRI Multi-modal Downsampling Pipeline to CEST Resolution

## 概述

这是一个专业的MRI多模态数据下采样流水线，用于将高分辨率（~0.65mm各向同性）的4D多模态MRI数据科学地下采样到CEST采集分辨率（1.8×1.8×3.0 mm³）。该流水线实现了物理空间的抗混叠滤波、通道族特定的PSF匹配，以及Z-谱数据的重新归一化。

## 主要特性

- ✅ **通道族特定PSF匹配**: 根据每个模态的原生采集分辨率进行各向异性高斯滤波
- ✅ **方法D**: 抗混叠高斯 + 线性重采样（用于强度数据）
- ✅ **方法C**: 低分辨率重算（用于比值/参数图，如Z-谱）
- ✅ **自动CEST slab定位**: 基于Z-谱强度的鲁棒性裁剪，带有智能降级策略
- ✅ **概率标签生成**: 使用MPRAGE→CEST PSF生成平滑的概率标签
- ✅ **完整的QA指标**: 数据统计、处理日志和元数据
- ✅ **100%可复现**: 固定随机种子，详细日志记录
- ✅ **轴顺序验证**: 自动检测和转换(Z,X,Y)→(X,Y,Z)顺序
- ✅ **灵活的输出格式** (v1.3+): 支持处理顺序或原始轴顺序输出
- ✅ **1D数据生成** (v1.3+): 自动生成ROI-only的1D格式数据，与外部工具兼容

## 系统要求

### Python环境
```bash
Python 3.7+
```

### 依赖包
```bash
pip install numpy scipy SimpleITK scikit-image h5py
```

### 推荐配置
- 内存: 至少16GB，推荐32GB
- 存储: SSD硬盘可显著提升I/O性能
- CPU: 多核处理器（当前单进程，可扩展为多进程）

## 安装

```bash
# 克隆或下载代码到本地
cd /path/to/your/project

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import SimpleITK; print('SimpleITK version:', SimpleITK.Version.VersionString())"
```

## 快速开始

### 1. 基础使用示例

```python
from pathlib import Path
import scipy.io as sio
from mri_downsampling_pipeline import MRIDownsamplingPipeline

# 加载数据
mat_data = sio.loadmat("subject1_3d_validated.mat")
data = mat_data['data']  # (384, 336, 256, 351) in (Z, X, Y, C)
region_mask = mat_data['region_mask']  # (384, 336, 256)
region_labels = mat_data['region_labels']  # (384, 336, 256)

# 初始化流水线
pipeline = MRIDownsamplingPipeline(
    output_dir=Path("./output"),
    log_level='INFO',
    random_seed=42
)

# 运行下采样
results = pipeline.run(
    data=data,
    region_mask=region_mask,
    region_labels=region_labels
)

# 获取结果
data_lr = results['data_lr']  # (X', Y', Z', 351)
proba_labels = results['proba_labels']  # (X', Y', Z', 102)
mask_lr = results['region_mask_lr']  # (X', Y', Z')
```

### 2. 运行测试

```bash
# 运行完整测试套件
python test_downsampling_pipeline.py

# 运行特定测试
python -m unittest test_downsampling_pipeline.TestAcceptanceCriteria
```

### 3. 查看示例

```bash
# 运行示例1：基础使用
python example_usage.py 1

# 运行示例2：批量处理
python example_usage.py 2

# 查看所有可用示例
python example_usage.py
```

## 数据格式

### 输入数据

**数组维度和顺序**:
- `data`: **(384, 336, 256, 351)** - 轴顺序为 **(Z, X, Y, C)**
  - Z=384: 矢状位切片数（物理上是Z轴，250mm）
  - X=336: 冠状位宽度（物理上是X轴，218mm）
  - Y=256: 轴位高度（物理上是Y轴，166mm）
  - C=351: 通道数
- `region_mask`: **(384, 336, 256)** - 脑组织掩膜（0/1）
- `region_labels`: **(384, 336, 256)** - 区域标签（0-101）

**物理spacing（来自FOV）**:
- X方向: 218mm / 336 ≈ 0.649 mm
- Y方向: 166mm / 256 ≈ 0.648 mm
- Z方向: 250mm / 384 ≈ 0.651 mm

**目标spacing**:
- CEST分辨率: **1.8 × 1.8 × 3.0 mm³**

### 输出数据

**下采样后的数组** (默认: `save_axis_order='proc'`):
- `data_lr`: **(X', Y', Z', 351)** - 轴顺序为 **(X, Y, Z, C)**
  - X' ≈ 128: 取决于裁剪后的实际尺寸
  - Y' ≈ 104
  - Z' ≈ 18 (54mm / 3mm)
  - C = 351: 通道数不变
- `proba_labels`: **(X', Y', Z', 102)** - 概率标签
- `region_mask_lr`: **(X', Y', Z')** - 下采样后的掩膜

**原始轴顺序输出** (v1.3+: `save_axis_order='orig'`):
- `data_lr`: **(Z', X', Y', 351)** - 轴顺序为 **(Z, X, Y, C)**
- `proba_labels`: **(Z', X', Y', 102)** - 概率标签
- `region_mask_lr`: **(Z', X', Y')** - 下采样后的掩膜
- `multidim_data`: **(n_voxels, 351)** - 1D特征数据
- `seg_one_hot`: **(102, n_voxels)** - 1D标签（one-hot）
- `region_seg`: **(n_voxels,)** - 1D标签（整数）
- `n_voxels`: int - ROI内体素数量

## 通道配置

### 通道族定义（0-based Python索引）

| 通道族 | 索引范围 (0-based) | 1-based | 原生分辨率 (mm) | σ_add (mm) | 方法 |
|--------|-------------------|---------|----------------|-----------|------|
| **QTI参数** | 0-14 | 1-15 | 1.5×1.5×3.0 | (0.423, 0.423, 0) | C |
| **DWI** | 15-224 | 16-225 | 1.5×1.5×3.0 | (0.423, 0.423, 0) | D |
| **CEST参数** | 225-228 | 226-229 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **M0 (低B1)** | 229 | 230 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **Z-谱 (低B1)** | 230-283 | 231-284 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **M0 (高B1-1)** | 284 | 285 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **M0 (高B1-2)** | 285 | 286 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **Z-谱 (高B1)** | 286-339 | 287-340 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **M0 (候补)** | 340 | 341 | 1.8×1.8×3.0 | (0, 0, 0) | C |
| **MPRAGE** | 341 | 342 | 0.65×0.65×0.65 | (0.713, 0.713, 1.245) | D |
| **GRE源 (QSM_TE)** | 342-346 | 343-347 | 0.6×0.6×0.6 | (0.721, 0.721, 1.249) | D |
| **TE_avg** | 347 | 348 | 0.65×0.65×0.65 | (0.713, 0.713, 1.245) | D |
| **SMWI** | 348-349 | 349-350 | 0.6×0.6×0.6 | (0.721, 0.721, 1.249) | C |
| **QSM** | 350 | 351 | 0.6×0.6×0.6 | (0.721, 0.721, 1.249) | C |

### σ_add计算公式

对于每个轴（X/Y/Z）：
```
FWHM_add = sqrt(max(FWHM_target² - FWHM_current², 0))
σ_add = FWHM_add / 2.355
```

其中：
- FWHM_target: 目标分辨率的半高全宽（CEST: 1.8/1.8/3.0 mm）
- FWHM_current: 当前模态的原生分辨率

## 核心流程

### Step 0: 轴顺序统一

```
输入: (Z, X, Y, C) = (384, 336, 256, 351)
↓
转置: (1, 2, 0, 3)
↓
输出: (X, Y, Z, C) = (336, 256, 384, 351)
```

**断言验证**:
- ✓ 转置后形状必须为 (336, 256, 384, 351)
- ✓ CEST slab厚度必须在 48-60 mm范围内（在Step 1中检查）

### Step 1: CEST Slab自动定位

**算法流程**:
1. 计算Z-谱强度图：`Zsum = max(Zsum_low, Zsum_high)`
2. 获取对应M0图
3. 自适应阈值：`M = (Zsum > τ1) ∧ (M0 > τ2)`
   - τ1 = P90(Zsum) × 0.3
   - τ2 = P90(M0) × 0.3
4. 3D连通域分析：保留最大连通区域
5. 厚度验证：Z厚度应在 [48mm, 60mm]范围内
6. **降级策略**: 如失败，降低阈值至P80重试；仍失败则使用中心裁剪

**输出**:
- 裁剪后的数据、掩膜、标签
- 包围盒坐标
- Slab厚度（mm和voxels）
- 是否使用降级策略的标志

### Step 2-3: 通道族特定下采样

**方法D（直接下采样）**:
```
1. 各向异性高斯平滑（PSF匹配）
   - 分三轴依次应用RecursiveGaussian
   - σ = σ_add(mm)，在物理空间执行
   - UseImageSpacing=True

2. 线性/B-spline重采样
   - 目标spacing: (1.8, 1.8, 3.0) mm
   - 插值器选择：
     * 强度数据 → Linear
     * 连续参数 → BSpline
   - 边界模式: Clamp
```

**方法C（低分辨率重算）**:

对于**Z-谱**:
```
1. 检测是否已归一化
   - 远离共振通道（|Δω|>5ppm）均值 ∈ [0.9, 1.1] → 已归一化
   - 否则 → 原始S_sat

2. 如已归一化：
   a) 反归一化: S_sat = Z * M0
   b) 分别下采样S_sat和M0（方法D）
   c) 重新归一化: Z' = S_sat' / M0'
   d) 裁剪: Z' = clip(Z', 0, 1)

3. 如未归一化：
   直接下采样S_sat和M0，然后归一化
```

对于**QTI/CEST参数/SMWI/QSM**:
```
理想：从低分辨率上游信号重新拟合/重建
当前：Fallback到方法D（带警告日志）
```

### Step 4: 概率标签生成

```
1. 将整数标签转为one-hot编码 (102类)
2. 对每个类别概率图：
   - 使用MPRAGE家族的σ_add进行平滑
   - Linear插值重采样
3. 归一化：逐体素概率和 = 1
4. 裁剪到 [0, 1]
```

### Step 5: QA指标计算

**自动计算的指标**:
- 输入/输出形状
- 下采样比率
- 脑体素计数（高分辨率vs低分辨率）
- 每个通道族的数据统计（均值、标准差、最小值、最大值）

**输出文件**:
- `qa_report.json`: 详细QA指标
- `pipeline_metadata.json`: 完整处理元数据

## 输出文件结构

```
output_dir/
├── logs/
│   └── downsampling_YYYYMMDD_HHMMSS.log  # 详细日志
├── pipeline_metadata.json                 # 处理元数据
├── qa_report.json                         # QA指标
└── subject1_downsampled.npz              # 下采样结果
```

### 元数据JSON结构

```json
{
  "timestamp": "2025-01-11T10:30:00",
  "random_seed": 42,
  "config": {
    "input_spacing_mm": [0.649, 0.648, 0.651],
    "target_spacing_mm": [1.8, 1.8, 3.0]
  },
  "slab_localization": {
    "bbox": [[x_start, x_end], [y_start, y_end], [z_start, z_end]],
    "slab_thickness_mm": 54.0,
    "slab_thickness_voxels": 18,
    "coverage_fallback": false,
    "thresholds": {"tau1": 1234.5, "tau2": 567.8}
  },
  "channel_processing": [...],
  "output_shape": {
    "data_lr": [128, 104, 18, 351],
    "proba_labels": [128, 104, 18, 102]
  }
}
```

## 高级用法

### 批量处理多个被试（推荐方式）

**🔥 使用 `batch_downsampling_pipeline.py` 自动批处理**

这是最简单和推荐的批量处理方式，会**自动保存3D和1D数据**：

```bash
# 测试模式：处理前3个被试
cd downsampling/
python batch_downsampling_pipeline.py --test-only

# 处理所有被试（自动生成3D+1D数据）
python batch_downsampling_pipeline.py \
  --input-dir /path/to/3D_validated \
  --output-dir /path/to/downsampled_output

# 强制重新处理已存在的文件
python batch_downsampling_pipeline.py --no-skip-existing

# 查看帮助
python batch_downsampling_pipeline.py --help
```

**批处理脚本的特性**：
- ✅ **自动生成3D和1D数据**（v1.3+）
- ✅ 自动遍历输入目录中的所有 `*_3d_validated.mat` 文件
- ✅ 断点续传：意外中断后可继续处理
- ✅ 内存监控和错误处理
- ✅ 详细日志记录
- ✅ 生成QA报告和处理摘要
- ✅ 支持HDF5格式MAT文件（自动fallback）

**生成的文件**（每个被试）：
```
output_dir/
├── subject001_downsampled.npz        # 包含3D+1D数据
├── subject001_metadata.json          # 处理元数据
├── subject001_qa_metrics.json        # QA指标
├── downsampling_dataset_index.json   # 数据集索引
├── downsampling_checkpoint.pkl       # 断点续传检查点
├── downsampling_report_*.json        # 批处理报告
├── downsampling_summary_*.csv        # 批处理摘要
└── logs/
    └── batch_downsampling_*.log      # 详细日志
```

**NPZ文件包含的数据**：

**3D数据**（所有模式，原始轴顺序 Z,X,Y）：
- `data_lr`: (Z', X', Y', 351) - 多模态特征
- `proba_labels`: (Z', X', Y', 102) - 概率标签（软标签）
- `region_mask_lr`: (Z', X', Y') - ROI掩码

**1D数据**（v1.3+，自动生成）：
- `multidim_data`: (n_voxels, 351) - 1D特征矩阵
- `seg_one_hot`: (102, n_voxels) - 1D概率标签
- `region_seg`: (n_voxels,) - 1D硬标签
- `n_voxels`: 标量 - ROI体素数

**加载和使用批处理结果**：
```python
import numpy as np

# 加载批处理生成的数据
data = np.load('subject001_downsampled.npz')

# 查看包含的key
print("包含的数据:")
for key in data.files:
    if isinstance(data[key], np.ndarray):
        print(f"  {key}: {data[key].shape}")
    else:
        print(f"  {key}: {data[key]}")

# 使用3D数据
data_lr = data['data_lr']  # (Z', X', Y', 351)
proba_labels = data['proba_labels']  # (Z', X', Y', 102)
region_mask_lr = data['region_mask_lr']  # (Z', X', Y')

# 使用1D数据
multidim_data = data['multidim_data']  # (n_voxels, 351)
seg_one_hot = data['seg_one_hot']  # (102, n_voxels)
region_seg = data['region_seg']  # (n_voxels,)
n_voxels = int(data['n_voxels'])

print(f"\nROI体素数: {n_voxels}")
print(f"3D shape: {data_lr.shape}")
print(f"1D shape: {multidim_data.shape}")
```

**验证数据完整性**：
```bash
# 使用提供的验证脚本
cd ..
python verify_1d_data.py subject001_downsampled.npz
```

**详细数据格式说明**：
参见 `DOWNSAMPLED_DATA_FORMAT.md` 文档，包含：
- 每个key的详细说明
- 维度和数据类型
- 3D-1D对应关系
- 使用示例

---

### 手动批量处理（编程方式）

如果需要更多自定义控制，可以手动编写批处理循环：

```python
from pathlib import Path
from mri_downsampling_pipeline import MRIDownsamplingPipeline
import scipy.io as sio

# 初始化流水线（复用）
pipeline = MRIDownsamplingPipeline(
    output_dir=Path("./batch_output"),
    log_level='INFO',
    random_seed=42
)

# 处理所有被试
data_dir = Path("/path/to/3d_data")
for subject_file in sorted(data_dir.glob("subject*_3d_validated.mat")):
    print(f"Processing: {subject_file.name}")

    mat_data = sio.loadmat(subject_file)

    # 使用save_axis_order='orig'生成3D+1D数据
    results = pipeline.run(
        data=mat_data['data'],
        region_mask=mat_data['region_mask'],
        region_labels=mat_data['region_labels'],
        save_axis_order='orig'  # 关键参数
    )

    # 保存完整结果（3D+1D）
    output_file = Path("./batch_output") / f"{subject_file.stem}_downsampled.npz"
    import numpy as np
    np.savez_compressed(output_file,
        # 3D数据
        data_lr=results['data_lr'],
        proba_labels=results['proba_labels'],
        region_mask_lr=results['region_mask_lr'],
        # 1D数据
        multidim_data=results['multidim_data'],
        seg_one_hot=results['seg_one_hot'],
        region_seg=results['region_seg'],
        n_voxels=results['n_voxels']
    )
```

### 生成1D格式数据 (v1.3+)

```python
# 启用原始轴顺序+1D数据生成
results = pipeline.run(
    data=data,
    region_mask=mask,
    region_labels=labels,
    save_axis_order='orig'  # 关键参数
)

# 访问3D数据（原始轴顺序）
data_lr = results['data_lr']  # (Z', X', Y', 351)
mask_lr = results['region_mask_lr']  # (Z', X', Y')

# 访问1D数据
multidim_data = results['multidim_data']  # (n_voxels, 351)
seg_one_hot = results['seg_one_hot']  # (102, n_voxels)
region_seg = results['region_seg']  # (n_voxels,)

# 保存3D数据
np.savez_compressed(
    "subject1_3d_orig.npz",
    data_lr=data_lr,
    proba_labels=results['proba_labels'],
    region_mask_lr=mask_lr
)

# 保存1D数据（与data_3d_1d_mapper.py兼容）
np.savez_compressed(
    "subject1_1d.npz",
    multidim_data=multidim_data,
    seg_one_hot=seg_one_hot,
    region_seg=region_seg,
    region=mask_lr,
    n_voxels=results['n_voxels']
)

# 验证1D-3D对应关系
for i in range(min(10, results['n_voxels'])):
    feat_1d = multidim_data[i, :]
    # 找到对应的3D位置（C-order索引）
    idx = np.where(mask_lr.ravel(order="C"))[0][i]
    z, x, y = np.unravel_index(idx, mask_lr.shape, order="C")
    feat_3d = data_lr[z, x, y, :]

    # 应该完美匹配
    assert np.allclose(feat_1d, feat_3d, atol=1e-6)
    print(f"Voxel {i}: Position ({z},{x},{y}) - Match ✓")
```

**用途**:
- 与现有1D↔3D转换工具集成（如 `data_3d_1d_mapper.py`）
- 用于仅ROI的机器学习模型训练
- 保持数据在原始解剖轴顺序以便可视化
- 生成更紧凑的数据格式（只保存ROI内的体素）

### 强制对齐到固定矩阵大小

```python
results = pipeline.run(
    data=data,
    region_mask=region_mask,
    region_labels=region_labels,
    align_to_128x104x18=True  # 强制对齐到128×104×18
)
```

### 提取特定通道族

```python
import numpy as np
from mri_downsampling_pipeline import ChannelConfig

# 加载结果
data = np.load("subject1_downsampled.npz")
data_lr = data['data_lr']

config = ChannelConfig()

# 提取Z-谱（低B1）
z_low = data_lr[..., config.Z_SPECTRUM_LOW_B1]  # Shape: (X, Y, Z, 54)

# 提取MPRAGE
mprage = data_lr[..., config.FAMILIES['MPRAGE'].indices[0]]  # Shape: (X, Y, Z)

# 提取QSM
qsm = data_lr[..., config.FAMILIES['QSM'].indices[0]]  # Shape: (X, Y, Z)
```

## 故障排除

### 问题1: 内存不足

**症状**: `MemoryError` 或系统变慢

**解决方案**:
- 增加系统内存（推荐至少32GB）
- 关闭其他应用程序
- 修改代码以分块处理通道

### 问题2: CEST slab定位失败

**症状**: 日志中显示 "Using fallback: center crop"

**原因**:
- Z-谱信号太弱
- M0值异常
- 数据质量问题

**解决方案**:
- 检查原始数据质量
- 降级策略会自动启用，结果仍然可用
- 检查 `coverage_fallback` 标志

### 问题3: 轴顺序错误

**症状**: `AssertionError` 关于形状不匹配

**解决方案**:
```python
# 检查输入数据形状
print(f"Data shape: {data.shape}")
# 应该是 (384, 336, 256, 351)

# 如果不是，手动转置
if data.shape != (384, 336, 256, 351):
    # 根据实际情况调整转置顺序
    data = np.transpose(data, (desired_order))
```

### 问题4: Z-谱值异常

**症状**: Z-谱值超出 [0, 1] 范围

**原因**:
- M0值接近零
- 原始数据未正确归一化

**解决方案**:
- Pipeline会自动裁剪到 [0, 1]
- 检查 `qa_report.json` 中的统计信息
- 查看日志中的 "Z' range" 信息

### 问题5: SimpleITK安装问题

**症状**: `ImportError: No module named 'SimpleITK'`

**解决方案**:
```bash
# 使用pip安装
pip install SimpleITK

# 或使用conda
conda install -c simpleitk simpleitk

# 验证安装
python -c "import SimpleITK; print(SimpleITK.Version.VersionString())"
```

## 性能优化

### 当前性能

- **单个被试处理时间**: 5-15分钟（取决于硬件）
- **内存峰值**: 8-16 GB
- **CPU使用**: 单核心

### 优化建议

1. **使用SSD存储**: 可提升I/O性能30-50%
2. **增加内存**: 减少内存交换，提升稳定性
3. **多进程并行**: 可修改代码实现多被试并行处理
4. **GPU加速**: 未来可以添加CUDA支持（需要修改代码）

## API参考

### MRIDownsamplingPipeline

```python
class MRIDownsamplingPipeline:
    def __init__(self,
                 output_dir: Path,
                 log_level: str = 'INFO',
                 random_seed: int = 42)

    def run(self,
            data: np.ndarray,
            region_mask: np.ndarray,
            region_labels: np.ndarray,
            align_to_128x104x18: bool = False,
            z_offsets_ppm: Optional[np.ndarray] = None,
            save_axis_order: str = "proc") -> Dict[str, Any]
```

**参数**:
- `output_dir`: 输出目录（日志、元数据、QA报告）
- `log_level`: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `random_seed`: 随机种子（用于可复现性）
- `data`: (384, 336, 256, 351) 输入数据 in (Z, X, Y, C) order
- `region_mask`: (384, 336, 256) 脑组织掩膜
- `region_labels`: (384, 336, 256) 区域标签
- `align_to_128x104x18`: 是否强制对齐到固定矩阵大小
- `z_offsets_ppm`: Z-谱频率偏移（可选）
- `save_axis_order` (v1.3+): 输出轴顺序 {'proc', 'orig'}
  - **'proc'** (默认): 处理顺序 (X,Y,Z,C)，仅3D数据
  - **'orig'**: 原始顺序 (Z,X,Y,C)，含3D+1D数据

**返回值 (save_axis_order='proc', 默认)**:
```python
{
    'data_lr': np.ndarray,          # (X', Y', Z', 351)
    'proba_labels': np.ndarray,     # (X', Y', Z', 102)
    'region_mask_lr': np.ndarray,   # (X', Y', Z')
    'metadata': dict,               # 完整处理元数据
    'qa_metrics': dict              # QA指标
}
```

**返回值 (save_axis_order='orig', v1.3+)**:
```python
{
    # 3D数据（原始轴顺序）
    'data_lr': np.ndarray,          # (Z', X', Y', 351)
    'proba_labels': np.ndarray,     # (Z', X', Y', 102)
    'region_mask_lr': np.ndarray,   # (Z', X', Y')

    # 1D数据（新增）
    'multidim_data': np.ndarray,    # (n_voxels, 351)
    'seg_one_hot': np.ndarray,      # (102, n_voxels)
    'region_seg': np.ndarray,       # (n_voxels,)
    'n_voxels': int,
    'region': np.ndarray,           # (Z', X', Y') - 掩膜别名

    # 标准输出
    'metadata': dict,               # 包含 'mapping' 字段
    'qa_metrics': dict
}
```

**元数据mapping字段 (v1.3+)**:
```python
metadata['mapping'] = {
    'axes': str,                    # 轴转换描述
    'permute': list,                # 前向置换
    'inv_perm': list,               # 逆向置换（仅'orig'模式）
    'save_axis_order': str,         # 'proc' 或 'orig'
    'one_d_reorder_applied': bool,  # 是否应用1D重排序
    'one_d_order_len': int          # 1D数据长度（仅'orig'模式）
}
```

### ChannelConfig

```python
class ChannelConfig:
    TARGET_SPACING_MM = (1.8, 1.8, 3.0)
    INPUT_SPACING_MM = (0.649, 0.648, 0.651)

    FAMILIES = {
        'CEST': ChannelFamily(...),
        'QTI_params': ChannelFamily(...),
        'DWI': ChannelFamily(...),
        # ... 其他通道族
    }

    Z_SPECTRUM_LOW_B1 = list(range(230, 284))
    Z_SPECTRUM_HIGH_B1 = list(range(286, 340))
    # ... 其他通道定义
```

## 测试

### 运行测试套件

```bash
# 完整测试
python test_downsampling_pipeline.py

# 特定测试类
python -m unittest test_downsampling_pipeline.TestAcceptanceCriteria

# 单个测试
python -m unittest test_downsampling_pipeline.TestAcceptanceCriteria.test_acceptance_1_sinusoidal_grid
```

### 验收测试清单

- ✓ 合成正弦栅格频谱抑制测试
- ✓ 体块平均对比测试
- ✓ Z-谱裁剪无负值测试
- ✓ 概率标签和为1测试
- ✓ 往返保真度测试（PSNR/SSIM）

## 引用

如果您在研究中使用此Pipeline，请引用：

```bibtex
@software{mri_downsampling_pipeline,
  title = {MRI Multi-modal Downsampling Pipeline to CEST Resolution},
  author = {KAN-Brain Project Team},
  year = {2025},
  version = {1.0.0}
}
```

## 许可证

本项目仅供研究使用。使用前请确保获得数据集的使用许可。

## 更新日志

### v1.3.0 (2025-01-11)

- ✨ 新增 `save_axis_order` 参数（'proc' / 'orig'）
- ✅ 原始轴顺序输出支持 (Z,X,Y,C)
- ✅ 自动生成1D格式数据（multidim_data, seg_one_hot, region_seg）
- ✅ 1D行重排序确保与3D C-order对应
- ✅ 添加mapping元数据追踪轴转换
- ✅ 新增Example 7和8演示新功能
- ✅ 新增TestSaveAxisOrder测试类（6个测试）
- ✅ 100%向后兼容，默认行为不变

### v1.2.0 (2025-01-11)

- 🐛 修复各向异性高斯实现（使用三个1D滤波器）
- 🐛 修复概率标签插值器（强制Linear避免负值）
- ✅ 边缘校正（归一化卷积）
- ✅ Z-谱远离共振判定改进（基于物理offset）
- ✅ M0自动选择（基于相关系数）

### v1.1.0 (2025-01-11)

- ✅ 实现规范化卷积（mask边界校正）
- ✅ 高斯σ物理单位语义确保
- ✅ Z-谱offset判定基于ppm阈值
- ✅ M0自动选择基于相关系数

### v1.0.0 (2025-01-11)

- ✨ 初始版本发布
- ✅ 完整的通道族特定下采样
- ✅ 自动CEST slab定位
- ✅ Z-谱方法C实现
- ✅ 概率标签生成
- ✅ 完整的测试套件
- ✅ 详细文档和示例

## 技术支持

如遇问题：

1. 查看本文档的"故障排除"章节
2. 检查日志文件（`logs/downsampling_*.log`）
3. 查看QA报告（`qa_report.json`）
4. 运行测试套件验证环境

## 附录

### A. PSF匹配计算示例

对于MPRAGE（原生0.65mm各向同性）到CEST（1.8/1.8/3.0）：

```python
# X/Y轴
FWHM_target = 1.8 mm
FWHM_current = 0.65 mm
FWHM_add = sqrt(1.8² - 0.65²) = sqrt(2.82) ≈ 1.679 mm
σ_add_xy = 1.679 / 2.355 ≈ 0.713 mm

# Z轴
FWHM_target = 3.0 mm
FWHM_current = 0.65 mm
FWHM_add = sqrt(3.0² - 0.65²) = sqrt(8.58) ≈ 2.929 mm
σ_add_z = 2.929 / 2.355 ≈ 1.245 mm
```

### B. 完整通道映射表

参见代码中的 `ChannelConfig` 类定义。

### C. 坐标系说明

- **输入**: (Z, X, Y, C) = (384, 336, 256, 351)
  - Z轴（第0维）: 矢状位方向，384层，物理250mm
  - X轴（第1维）: 冠状位方向，336像素，物理218mm
  - Y轴（第2维）: 轴位方向，256像素，物理166mm

- **处理**: (X, Y, Z, C) = (336, 256, 384, 351)
  - 转置后使用标准解剖坐标系

- **输出**: (X', Y', Z', C) ≈ (128, 104, 18, 351)
  - 下采样到CEST分辨率后的尺寸

### D. 1D-3D数据对应关系 (v1.3+)

当使用 `save_axis_order='orig'` 时，生成的1D数据与3D数据有精确对应关系：

**C-order索引保证**:
```python
# 对于第i个ROI体素
multidim_data[i, :]  # 1D特征向量

# 对应的3D位置
idx = np.where(region_mask_lr.ravel(order="C"))[0][i]
z, x, y = np.unravel_index(idx, region_mask_lr.shape, order="C")

# 对应的3D特征向量
data_lr[z, x, y, :]  # 应该与multidim_data[i, :]完全相等
```

**可逆性保证**:
```python
# 从1D重建3D
data_reconstructed = np.zeros_like(data_lr)
data_reconstructed[region_mask_lr > 0] = multidim_data

# 应该与原始3D完全相等
assert np.allclose(data_reconstructed, data_lr)
```

**与data_3d_1d_mapper.py的兼容性**:
```python
# 该工具使用boolean索引提取1D数据
multidim_data_mapper = data_lr[region_mask_lr > 0]

# 应该与pipeline生成的1D数据相等
assert np.array_equal(multidim_data_mapper, multidim_data)
```

---

**最后更新**: 2025-01-11
**版本**: 1.3.0
**状态**: 生产就绪 ✅

**相关文档**:
- [v1.3新特性详解](./BUGFIX_v1.3.md)
- [v1.2 Bug修复](./BUGFIX_v1.2.md)
- [v1.1 Bug修复](./BUGFIX_v1.1.md)
- [快速开始指南](./QUICKSTART.md)
- [批量处理指南](./BATCH_DOWNSAMPLING_README.md)
