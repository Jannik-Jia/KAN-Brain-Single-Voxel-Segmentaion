# MRI下采样流水线 - 快速开始指南

## 已生成的文件

```
dataset_create/
├── mri_downsampling_pipeline.py          # 主流水线实现（1200+ 行）
├── test_downsampling_pipeline.py         # 完整测试套件（600+ 行）
├── example_usage.py                      # 6个实用示例（400+ 行）
├── DOWNSAMPLING_README.md                # 详细技术文档
├── QUICKSTART.md                         # 本文件
└── requirements_downsampling.txt         # 依赖包列表
```

## 30秒快速开始

### 1. 安装依赖

```bash
pip install -r requirements_downsampling.txt
```

### 2. 运行测试（验证环境）

```bash
python test_downsampling_pipeline.py
```

预期输出：所有测试通过 ✓

### 3. 处理单个被试

```python
from pathlib import Path
import scipy.io as sio
from mri_downsampling_pipeline import MRIDownsamplingPipeline

# 加载数据
mat_data = sio.loadmat("subject1_3d_validated.mat")

# 初始化并运行
pipeline = MRIDownsamplingPipeline(
    output_dir=Path("./output"),
    log_level='INFO'
)

results = pipeline.run(
    data=mat_data['data'],
    region_mask=mat_data['region_mask'],
    region_labels=mat_data['region_labels']
)

# 保存结果
import numpy as np
np.savez_compressed(
    "./output/subject1_lr.npz",
    data_lr=results['data_lr'],
    proba_labels=results['proba_labels'],
    region_mask_lr=results['region_mask_lr']
)
```

## 核心特性一览

| 特性 | 说明 |
|------|------|
| **输入** | (384, 336, 256, 351) in (Z,X,Y,C) |
| **输出** | (~128, ~104, ~18, 351) in (X,Y,Z,C) or (Z,X,Y,C) |
| **目标分辨率** | 1.8×1.8×3.0 mm³ (CEST) |
| **通道族** | 9个独立处理的通道族 |
| **PSF匹配** | 各向异性高斯，mm空间 |
| **Z-谱处理** | 方法C：反归一→下采样→重归一 |
| **概率标签** | 102类，和为1 |
| **QA指标** | 自动计算并保存 |
| **可复现** | 固定随机种子 |
| **轴顺序模式** | 'proc' (默认) 或 'orig' (含1D数据) |
| **1D数据生成** | multidim_data, seg_one_hot, region_seg (v1.3+) |

## 通道族配置速查

```python
from mri_downsampling_pipeline import ChannelConfig
config = ChannelConfig()

# 访问通道索引（0-based）
qti_indices = config.FAMILIES['QTI_params'].indices  # [0-14]
dwi_indices = config.FAMILIES['DWI'].indices         # [15-224]
z_low = config.Z_SPECTRUM_LOW_B1                     # [230-283]
z_high = config.Z_SPECTRUM_HIGH_B1                   # [286-339]
mprage_idx = config.FAMILIES['MPRAGE'].indices[0]   # 341
qsm_idx = config.FAMILIES['QSM'].indices[0]          # 350
```

## 常用命令

### 批量处理

```bash
python example_usage.py 2
```

### 查看结果

```bash
python example_usage.py 3
```

### 提取特定通道

```bash
python example_usage.py 6
```

### 生成1D格式数据（v1.3+）

```bash
python example_usage.py 7
```

### 对比轴顺序模式（v1.3+）

```bash
python example_usage.py 8
```

## 输出文件说明

处理完成后，输出目录包含：

**默认模式 (save_axis_order='proc')**:
```
output/
├── logs/
│   └── downsampling_YYYYMMDD_HHMMSS.log   # 详细日志
├── pipeline_metadata.json                  # 处理元数据
├── qa_report.json                          # QA指标
└── subject1_downsampled.npz               # 下采样结果
    ├── data_lr: (X', Y', Z', 351)
    ├── proba_labels: (X', Y', Z', 102)
    └── region_mask_lr: (X', Y', Z')
```

**原始轴顺序模式 (save_axis_order='orig', v1.3+)**:
```
output/
├── logs/
│   └── downsampling_YYYYMMDD_HHMMSS.log   # 详细日志
├── pipeline_metadata.json                  # 处理元数据（含mapping信息）
├── qa_report.json                          # QA指标
└── subject1_downsampled.npz               # 下采样结果
    ├── data_lr: (Z', X', Y', 351)          # 原始轴顺序
    ├── proba_labels: (Z', X', Y', 102)
    ├── region_mask_lr: (Z', X', Y')
    ├── multidim_data: (n_voxels, 351)      # 1D格式
    ├── seg_one_hot: (102, n_voxels)
    ├── region_seg: (n_voxels,)
    └── n_voxels: scalar
```

## 重要参数说明

### Pipeline初始化

```python
MRIDownsamplingPipeline(
    output_dir=Path("./output"),      # 输出目录
    log_level='INFO',                 # 日志级别: DEBUG/INFO/WARNING/ERROR
    random_seed=42                    # 随机种子（可复现性）
)
```

### 运行参数

```python
pipeline.run(
    data=data,                        # (384, 336, 256, 351)
    region_mask=mask,                 # (384, 336, 256)
    region_labels=labels,             # (384, 336, 256)
    align_to_128x104x18=False,        # 是否强制对齐到固定大小
    save_axis_order='proc'            # 'proc' (默认) 或 'orig' (v1.3+)
)
```

**v1.3新增参数说明**:
- `save_axis_order='proc'`: 输出 (X,Y,Z,C) 处理顺序（默认）
- `save_axis_order='orig'`: 输出 (Z,X,Y,C) 原始顺序 + 生成1D数据

## 故障排除速查

| 问题 | 快速解决 |
|------|---------|
| 内存不足 | 增加RAM或关闭其他程序 |
| Slab定位失败 | 自动降级策略会启用 |
| 轴顺序错误 | 检查输入shape是否为(384,336,256,351) |
| SimpleITK导入错误 | `pip install SimpleITK` |
| 测试失败 | 检查依赖版本是否满足 |

## 处理流程概览

```
输入数据 (Z,X,Y,C)
    ↓
Step 0: 轴重排 → (X,Y,Z,C)
    ↓
Step 1: CEST slab定位与裁剪
    ↓
Step 2-3: 通道族特定下采样
    ├─ DWI: σ=(0.423, 0.423, 0) mm
    ├─ CEST: σ=(0, 0, 0) mm (方法C)
    ├─ MPRAGE: σ=(0.713, 0.713, 1.245) mm
    └─ GRE/QSM: σ=(0.721, 0.721, 1.249) mm
    ↓
Step 4: Z-谱重新归一化
    ↓
Step 5: 概率标签生成
    ↓
输出: data_lr, proba_labels, mask_lr
```

## 数据验证检查清单

运行后检查以下项：

- [ ] 输出shape合理（X'≈128, Y'≈104, Z'≈18）
- [ ] Z-谱值在[0, 1]范围内
- [ ] 概率标签逐体素和≈1.0
- [ ] Slab厚度在[48, 60]mm范围内
- [ ] QA报告无异常统计值
- [ ] 日志无ERROR级别消息

## 性能基准

**参考配置**:
- CPU: Intel i7/i9 或 AMD Ryzen 7/9
- RAM: 16-32 GB
- 存储: SSD

**预期性能**:
- 单被试处理时间: 5-15分钟
- 内存峰值: 8-16 GB
- 磁盘空间: 输入~3GB → 输出~200MB

## 下一步

1. 阅读 `DOWNSAMPLING_README.md` 了解技术细节
2. 查看 `example_usage.py` 学习更多用法
3. 运行 `test_downsampling_pipeline.py` 验证环境
4. 根据需求调整通道配置（修改 `ChannelConfig` 类）

## 技术支持

遇到问题时的检查顺序：

1. 查看日志文件：`output/logs/downsampling_*.log`
2. 检查QA报告：`output/qa_report.json`
3. 运行测试套件：`python test_downsampling_pipeline.py`
4. 参考完整文档：`DOWNSAMPLING_README.md`

## 引用信息

```bibtex
@software{mri_downsampling_pipeline,
  title = {MRI Multi-modal Downsampling Pipeline to CEST Resolution},
  author = {KAN-Brain Project Team},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/your-repo/mri-downsampling-pipeline}
}
```

## v1.3 新增功能速览

### 生成1D格式数据

```python
# 运行pipeline，启用1D数据生成
results = pipeline.run(
    data=data,
    region_mask=mask,
    region_labels=labels,
    save_axis_order='orig'  # 关键：启用原始轴顺序+1D生成
)

# 访问3D数据（原始轴顺序）
data_lr = results['data_lr']  # (Z', X', Y', 351)

# 访问1D数据（与data_3d_1d_mapper.py兼容）
multidim_data = results['multidim_data']  # (n_voxels, 351)
seg_one_hot = results['seg_one_hot']      # (102, n_voxels)
region_seg = results['region_seg']        # (n_voxels,)

# 验证1D-3D对应关系
for i in range(10):
    feat_1d = multidim_data[i, :]
    # 找到对应的3D位置
    idx = np.where(results['region_mask_lr'].ravel('C'))[0][i]
    z, x, y = np.unravel_index(idx, results['region_mask_lr'].shape, 'C')
    feat_3d = data_lr[z, x, y, :]
    assert np.allclose(feat_1d, feat_3d)  # 完美对应
```

**用途**:
- 与现有1D↔3D转换工具兼容（如 `data_3d_1d_mapper.py`）
- 用于仅ROI的模型训练
- 保持数据在原始解剖轴顺序

详见：`python example_usage.py 7` 和 `python example_usage.py 8`

---

**版本**: 1.3.0
**更新日期**: 2025-01-11
**状态**: 生产就绪 ✅

**快速链接**:
- [完整文档](./DOWNSAMPLING_README.md)
- [示例代码](./example_usage.py)
- [测试套件](./test_downsampling_pipeline.py)
- [主程序](./mri_downsampling_pipeline.py)
- [v1.3新特性](./BUGFIX_v1.3.md)
