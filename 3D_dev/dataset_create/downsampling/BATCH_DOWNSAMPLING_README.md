# 批量Downsampling处理 - 使用指南

## 概述

`batch_downsampling_pipeline.py` 是一个自动化批处理脚本，用于将3D_validated数据集下采样到CEST分辨率（1.8×1.8×3.0 mm³）。

**Pipeline版本**: v1.2.0 (包含BUGFIX v1.2 - 修复了各向异性高斯和概率标签插值器的关键bug)

## 数据流

```
输入: /home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/
    ├── subject001_3d_validated.mat  (384, 336, 256, 351)
    ├── subject002_3d_validated.mat
    └── ...

                    ↓ Downsampling Pipeline

输出: /home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling/3d/
    ├── subject001_downsampled.npz      # 下采样数据 (~128, ~104, ~18, 351)
    ├── subject001_metadata.json        # 处理元数据
    ├── subject001_qa_metrics.json      # QA指标
    ├── logs/
    │   └── batch_downsampling_*.log    # 详细日志
    ├── downsampling_report_*.json      # 批量处理报告
    └── downsampling_summary_*.csv      # CSV摘要
```

## 快速开始

### 方法1: 交互式运行（推荐用于首次测试）

```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create
python batch_downsampling_pipeline.py
```

然后选择：
- `1` - 测试模式（只处理前3个被试）
- `2` - 处理所有被试
- `3` - 断点续传（继续之前中断的任务）

### 方法2: 命令行运行（用于自动化）

```bash
# 测试模式
python batch_downsampling_pipeline.py --test-only

# 处理所有被试（默认跳过已存在文件）
python batch_downsampling_pipeline.py

# 强制重新处理所有文件
python batch_downsampling_pipeline.py --no-skip-existing

# 自定义输入输出目录
python batch_downsampling_pipeline.py \
  --input-dir /path/to/3D_validated \
  --output-dir /path/to/downsampling/output
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `/home/jovyan/gpu_space/.../3D_validated` | 输入目录（3D验证数据） |
| `--output-dir` | `/home/jovyan/gpu_space/.../downsampling/3d` | 输出目录 |
| `--no-skip-existing` | False | 不跳过已存在文件（强制重新处理） |
| `--test-only` | False | 测试模式（只处理前3个被试） |
| `--log-level` | INFO | 日志级别：DEBUG/INFO/WARNING/ERROR |
| `--target-spacing` | 1.8,1.8,3.0 | 目标分辨率 (X,Y,Z) in mm |

## 核心功能

### ✅ 自动化处理
- 自动扫描所有`*_3d_validated.mat`文件
- 按顺序处理每个被试
- 生成详细的进度条和状态更新

### ✅ 断点续传
- 自动保存检查点 (`downsampling_checkpoint.pkl`)
- 程序中断后可继续未完成的任务
- 避免重复处理已完成的被试

### ✅ 内存监控
- 实时监控内存使用
- 超过16GB时发出警告
- 每个被试处理后自动垃圾回收

### ✅ 错误处理
- 捕获所有异常并记录详细堆栈
- 单个被试失败不影响其他被试
- 优雅退出（Ctrl+C安全保存进度）

### ✅ 详细日志
- 文件级日志：`logs/batch_downsampling_YYYYMMDD_HHMMSS.log`
- 控制台实时输出
- 记录每个步骤的耗时和内存使用

### ✅ 完整报告
- JSON报告：包含所有处理结果和元数据
- CSV摘要：便于Excel分析
- QA metrics：每个被试的质量指标

## 输出文件说明

### 主要数据文件

#### `{subject_id}_downsampled.npz`
NumPy压缩文件，包含3个数组：

```python
import numpy as np
data = np.load('subject001_downsampled.npz')

# 下采样后的351通道数据
data_lr = data['data_lr']              # (~128, ~104, ~18, 351)

# 概率标签（102类，每体素和=1）
proba_labels = data['proba_labels']    # (~128, ~104, ~18, 102)

# 下采样后的脑掩膜
region_mask_lr = data['region_mask_lr']  # (~128, ~104, ~18)
```

### 元数据文件

#### `{subject_id}_metadata.json`
包含完整处理元数据：
- 时间戳和随机种子
- 输入/输出spacing
- CEST slab定位信息
- 通道家族处理配置
- M0自动选择结果

#### `{subject_id}_qa_metrics.json`
包含QA指标：
- 输入/输出shape
- 下采样比例
- 脑体素计数
- 每个通道家族的统计信息（均值、标准差、min、max）

## 处理流程

每个被试的处理包含4个主要步骤：

### 步骤1: 加载3D验证数据
- 从`.mat`文件读取`data`, `region_mask`, `region_labels`
- 验证shape是否为 (384, 336, 256, 351)
- 记录加载时间和内存使用

### 步骤2: 初始化Pipeline
- 创建被试专属输出目录
- 初始化`MRIDownsamplingPipeline`
- 固定随机种子为42（可复现）

### 步骤3: 运行Downsampling
完整执行6个子步骤：
1. **轴重排**: (Z,X,Y,C) → (X,Y,Z,C)
2. **CEST slab定位**: 自动裁剪到48-60mm厚度
3. **通道家族下采样**: 9个家族分别处理（各向异性高斯PSF匹配）
4. **Z-spectrum处理**: Method C（反归一化→下采样→重归一化）
5. **概率标签生成**: MPRAGE PSF + Linear插值器
6. **QA指标计算**: 自动验证输出质量

### 步骤4: 保存结果
- 压缩保存`.npz`文件
- 保存元数据和QA metrics
- 记录文件大小和处理时间

## 预期性能

**参考配置**:
- CPU: Intel i7/i9 或 AMD Ryzen 7/9
- RAM: 16-32 GB
- 存储: SSD

**处理速度**:
- 单被试处理时间: 5-15分钟
- 内存峰值: 8-16 GB
- 输入文件: ~3GB → 输出文件: ~200MB

**全量处理估计**（38个被试）:
- 总时间: 3-9 小时
- 总输出: ~7.6 GB
- 建议: 在服务器上运行，不要在个人电脑上长时间占用

## 错误处理和恢复

### 常见错误

| 错误类型 | 原因 | 解决方案 |
|---------|------|----------|
| `FileNotFoundError` | 输入目录不存在 | 检查`--input-dir`路径 |
| `ValueError: 缺少必需的key` | MAT文件损坏或格式不对 | 跳过该文件，检查原始数据 |
| `MemoryError` | 内存不足 | 关闭其他程序，或增加系统RAM |
| `AssertionError: shape不匹配` | 输入数据shape错误 | 检查是否使用了正确的3D验证数据 |

### 断点续传使用

如果程序被中断（断电、网络断开、内存不足等）：

```bash
# 查看已完成的被试
python batch_downsampling_pipeline.py
# 选择 "3. 断点续传"

# 或直接命令行继续
python batch_downsampling_pipeline.py  # 自动跳过已完成的
```

检查点文件位置: `{output_dir}/downsampling_checkpoint.pkl`

### 紧急报告

程序异常退出时会自动生成紧急报告：
- 文件: `emergency_report_YYYYMMDD_HHMMSS.json`
- 包含: 已完成被试列表、失败列表、最近10条结果、内存使用

## 质量验证

### 自动验证检查

每个被试处理完成后自动检查：
- ✅ 输出shape合理（X'≈128, Y'≈104, Z'≈18）
- ✅ Z-谱值在[0, 1]范围内
- ✅ 概率标签逐体素和≈1.0
- ✅ Slab厚度在[48, 60]mm范围内

### 手动验证步骤

```python
import numpy as np
import json

# 加载downsampled数据
data = np.load('subject001_downsampled.npz')

# 1. 检查shape
print(f"data_lr shape: {data['data_lr'].shape}")         # 应该约为 (~128, ~104, ~18, 351)
print(f"proba_labels shape: {data['proba_labels'].shape}") # 应该约为 (~128, ~104, ~18, 102)

# 2. 检查概率标签
proba_sum = data['proba_labels'].sum(axis=-1)
print(f"概率和检查 (应该≈1.0): mean={proba_sum.mean():.6f}, std={proba_sum.std():.6f}")

# 3. 检查Z-谱值范围（假设channel 230-283是Z-spectra）
z_channels = data['data_lr'][..., 230:284]
print(f"Z-谱值范围: [{z_channels.min():.4f}, {z_channels.max():.4f}]")  # 应该在[0,1]内

# 4. 加载QA metrics
with open('subject001_qa_metrics.json', 'r') as f:
    qa = json.load(f)
    print(f"输入shape: {qa['input_shape']}")
    print(f"输出shape: {qa['output_shape']}")
    print(f"下采样比例: {qa['downsampling_ratio']}")
```

## 批量报告分析

### 查看处理摘要

```python
import pandas as pd

# 读取CSV摘要
df = pd.read_csv('downsampling_summary_20250111_123456.csv')

# 统计
print(f"总被试数: {len(df)}")
print(f"成功: {(df['status']=='success').sum()}")
print(f"失败: {(df['status']=='failed').sum()}")

# 平均处理时间
print(f"平均处理时间: {df[df['status']=='success']['processing_time'].mean():.1f}秒")

# 查看失败的被试
failed = df[df['status'] == 'failed']
if len(failed) > 0:
    print("\n失败的被试:")
    print(failed[['subject_id', 'error_type', 'error']])

# Slab定位fallback统计
print(f"\n使用fallback的被试数: {df['coverage_fallback'].sum()}")
```

## 高级用法

### 自定义目标分辨率

```bash
# 下采样到2.0×2.0×3.5 mm³
python batch_downsampling_pipeline.py --target-spacing 2.0,2.0,3.5
```

### 调试模式

```bash
# 详细DEBUG日志
python batch_downsampling_pipeline.py --test-only --log-level DEBUG
```

### 强制重新处理

```bash
# 重新处理所有文件（忽略已存在的输出）
python batch_downsampling_pipeline.py --no-skip-existing
```

## 与原始批处理的对比

| 特性 | batch_convert_validated | batch_downsampling_pipeline |
|------|------------------------|----------------------------|
| **输入** | 1D格式 (n_voxels, 351) | 3D格式 (384, 336, 256, 351) |
| **输出** | 3D验证数据 | Downsampled到CEST分辨率 |
| **处理** | 1D→3D转换 + 验证 | PSF匹配 + 多方法下采样 |
| **输出大小** | ~3GB/被试 | ~200MB/被试 |
| **处理时间** | ~30-60秒/被试 | ~5-15分钟/被试 |
| **复杂度** | 低（数组重排） | 高（科学下采样） |

## 故障排除

### Q: 程序运行很慢，内存持续增长？
**A**:
1. 检查是否有其他程序占用内存
2. 确保使用SSD而非HDD
3. 考虑减少并发（当前是串行处理）

### Q: 某些被试CEST slab定位失败？
**A**:
- 查看日志中的`coverage_fallback: True`
- Pipeline会自动使用fallback策略（center crop 54mm）
- 不影响结果质量，只是定位方式不同

### Q: 想要查看某个被试的详细处理日志？
**A**:
```bash
# 查看总体日志
cat logs/batch_downsampling_20250111_123456.log | grep "subject001"

# 查看该被试的pipeline日志
cat subject001/logs/downsampling_*.log
```

### Q: 如何恢复中断的处理？
**A**:
直接重新运行脚本即可，程序会：
1. 自动加载检查点文件
2. 跳过已完成的被试
3. 从下一个未完成的被试继续

### Q: 想要验证downsampling质量？
**A**:
1. 查看QA metrics文件
2. 运行`example_usage.py` 示例3（可视化对比）
3. 检查日志中的警告信息

## 相关文档

- **完整技术文档**: `DOWNSAMPLING_README.md`
- **快速开始**: `QUICKSTART.md`
- **Bug修复记录**: `BUGFIX_v1.2.md`
- **示例代码**: `example_usage.py`
- **Pipeline源码**: `mri_downsampling_pipeline.py`

## 技术支持

遇到问题时的检查顺序：

1. 查看批量处理日志：`logs/batch_downsampling_*.log`
2. 查看个别被试日志：`{subject_id}/logs/downsampling_*.log`
3. 检查QA报告：`{subject_id}_qa_metrics.json`
4. 查看批量摘要：`downsampling_summary_*.csv`

---

**更新日期**: 2025-01-11
**Pipeline版本**: v1.2.0
**状态**: 生产就绪 ✅

**包含修复**:
- ✅ 各向异性高斯实现（三个1D滤波器）
- ✅ 概率标签插值器（强制Linear）
- ✅ 归一化卷积边缘校正
- ✅ Z-谱偏移量检测（|Δω| > 4 ppm）
- ✅ M0自动选择（基于相关系数）
