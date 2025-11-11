# 多模态MRI配准质量控制分析系统

**版本**: v1.2.0
**创建日期**: 2025-11-10
**最后更新**: 2025-11-10 (v1.2.0)
**适用数据**: 3D Minimal数据集 (384×336×256×351)

---

## 📋 目录

1. [系统概述](#系统概述)
2. [快速开始](#快速开始)
3. [分析任务详解](#分析任务详解)
4. [输出文件说明](#输出文件说明)
5. [参数配置指南](#参数配置指南)
6. [故障排除](#故障排除)
7. [扩展到全模态分析](#扩展到全模态分析)

---

## 系统概述

### 功能简介

本系统对351通道多模态MRI数据进行**系统性的配准质量控制分析**，包括：

| 任务 | 功能 | 输出 |
|------|------|------|
| **任务1** | 模态间相似性矩阵 | LNCC、NGF、MIND-SSD热图 |
| **任务2** | 边缘结构一致性 | ASSD、HD95、Edge IOU柱状图 |
| **任务3** | ROI区域一致性 | 基于FreeSurfer标签的信号分析热图 |
| **任务4** | 降维可视化 | PCA/UMAP聚类散点图 |
| **任务5** | QC评分聚合 | PASS/WARN/FAIL判断，综合评分表 |

### 分析目标

✅ 判断整体模态是否对齐良好
✅ 识别错配或异常模态
✅ 定量评估不同模态家族的一致性
✅ 生成科研级的QC报告和可视化

### 物理参数说明

本系统使用以下物理参数进行分析：

| 参数 | 值 | 来源 | 说明 |
|------|-----|------|------|
| **spacing** | (0.65, 0.65, 0.65) mm | German et al. 2021 | MPRAGE空间各向同性分辨率 |
| **assd_max** | 2.5 mm | 经验值 @ 0.65mm | 平均对称表面距离阈值（物理距离） |
| **hd95_max** | 10.0 mm | 经验值 @ 0.65mm | 95% Hausdorff距离阈值（物理距离） |

**重要说明**:
- ✅ 所有距离指标（ASSD, HD95）均为**物理距离（mm）**，已考虑spacing=0.65mm
- ✅ 如果数据来自不同扫描参数或重建分辨率，需要相应调整spacing参数
- ✅ 阈值基于0.65mm spacing设定，如更改spacing需同步调整阈值

### v1.2.0 新特性

**性能提升** (相比v1.1.0):
- ✅ ASSD/HD95计算速度提升 >10x（距离变换法）
- ✅ 整体分析速度提升 70%（20个模态: 40分钟 → 12分钟）

**算法改进**:
- ✅ 自适应Canny阈值（Otsu自动阈值，对不同对比度模态更稳健）
- ✅ 三正交面边缘并集策略（边缘覆盖率提升20-30%）
- ✅ 快速距离变换法（替代O(N²)的cdist方法）

**用户体验**:
- ✅ 图表优化（异常排序、阈值标注、统计信息）
- ✅ CSV增强（modality_family、qc_rank字段）
- ✅ JSON增强（分位数统计、按家族汇总）

详见 `CHANGELOG_QC_v1.2.0.md`

---

## 快速开始

### 前置要求

```bash
# Python 3.7+
# 必需库
pip install numpy scipy h5py pandas scikit-image scikit-learn matplotlib seaborn tqdm

# 可选库（用于UMAP）
pip install umap-learn
```

### 文件组织

确保以下文件在正确位置：

```
/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create/
├── multimodal_mri_qc_analysis.ipynb       # 主notebook（任务1）
├── multimodal_qc_tasks_2_5_v1.2.0.py      # 任务2-5的代码（v1.2.0推荐）
├── multimodal_qc_tasks_2_5_fixed.py       # 任务2-5的代码（v1.1.0）
├── 3D_MINIMAL_DATASET_README.md           # 数据集说明
├── MULTIMODAL_QC_ANALYSIS_README.md       # 本文档
├── CHANGELOG_QC_v1.2.0.md                 # v1.2.0改进说明
└── QUICKSTART_QC_ANALYSIS.md              # 快速启动指南

服务器数据路径:
/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/
├── 3D_minimal/                         # 精简数据集
└── qc_analysis_results/                # 输出目录（自动创建）

标签映射文件:
/Users/jannik/.../Freesurfer_LUT_alex_labels_jiayi (1).xlsx
```

### 使用步骤

#### 步骤1: 打开Notebook

```bash
jupyter notebook multimodal_mri_qc_analysis.ipynb
```

#### 步骤2: 配置参数（Cell 2）

```python
# 修改以下参数
DATA_DIR = "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_minimal"
SUBJECT_FILE = "PDP_02_xxx_3d_validated_minimal.mat"  # 改为实际文件名

# 可选：调整QC阈值（单位已明确）
QC_THRESHOLDS = {
    'lncc_min': 0.3,        # 归一化相关性（无单位）
    'ngf_min': 0.4,         # 归一化梯度场（无单位）
    'assd_max': 2.5,        # mm（物理距离，基于spacing=0.65mm）
    'hd95_max': 10.0,       # mm（物理距离，基于spacing=0.65mm）
    'edge_iou_min': 0.3,    # 交并比（无单位）
    'mad_multiplier': 3.0   # MAD倍数（无单位）
}

# 物理间距（必须与数据匹配）
SPACING = (0.65, 0.65, 0.65)  # mm, MPRAGE空间
```

#### 步骤3: 运行任务1（Notebook中的cells）

按顺序运行notebook中的所有cells，完成：
- 数据加载
- 相似性矩阵计算（LNCC, NGF, MIND-SSD）
- 异常模态检测

**预计时间**: ~10-15分钟（20个模态）

#### 步骤4: 运行任务2-5

在notebook中添加新的cell，复制 `multimodal_qc_tasks_2_5.py` 中的代码并运行。

**或者**，直接在Python中运行：

```python
# 先运行notebook到任务1完成
# 然后在新cell中：
%run multimodal_qc_tasks_2_5.py
```

**预计时间**: ~20-30分钟（全部任务）

#### 步骤5: 查看结果

所有结果保存在 `qc_analysis_results/` 目录：

```
qc_analysis_results/
├── task1_similarity_matrices.png       # 相似性矩阵热图
├── task2_edge_metrics.png             # 边缘指标柱状图
├── task3_roi_heatmaps.png             # ROI分析热图
├── task4_pca_visualization.png        # PCA降维可视化
├── task5_qc_scores.png                # QC评分图表
├── modality_qc.csv                    # QC评分表（Excel可打开）
└── qc_analysis_report.json            # 完整JSON报告
```

---

## 分析任务详解

### 任务1: 模态间相似性矩阵

#### 计算指标

| 指标 | 全称 | 含义 | 范围 | 优质阈值 |
|------|------|------|------|----------|
| **LNCC** | Local Normalized Cross-Correlation | 局部归一化互相关 | [-1, 1] | > 0.3 |
| **NGF** | Normalized Gradient Fields | 归一化梯度场相似度 | [0, 1] | > 0.4 |
| **MIND-SSD** | Modality Independent Neighborhood Descriptor | 模态无关邻域描述符 | [0, ∞] | < 阈值 |

#### 输出示例

- **20×20相似性矩阵热图**（3张：LNCC, NGF, MIND-SSD）
- **异常模态列表**（基于MAD检测）

#### 解释

- **高LNCC/NGF**: 说明模态之间强度和结构高度一致
- **低MIND-SSD**: 说明局部结构相似
- **异常模态**: 平均相似性低于3×MAD的模态，可能存在配准问题

---

### 任务2: 边缘结构一致性评估

#### 计算指标

| 指标 | 全称 | 含义 | 单位 | 优质阈值 |
|------|------|------|------|----------|
| **ASSD** | Average Symmetric Surface Distance | 平均对称表面距离 | mm | < 2.5 |
| **HD95** | 95% Hausdorff Distance | 95%豪斯多夫距离 | mm | < 10.0 |
| **Edge IOU** | Edge Intersection over Union | 边缘交并比 | [0, 1] | > 0.3 |
| **Grad Corr** | Gradient Correlation | 梯度方向相关性 | [-1, 1] | > 0.5 |

#### 参考模态

- 默认使用 **MPRAGE (通道342)** 作为配准锚点
- 所有模态的边缘与MPRAGE边缘比较

#### 输出示例

- **4张柱状图**: ASSD, HD95, Edge IOU, Gradient Correlation
- 每张图标注阈值线（红色虚线）
- 按指标值排序，直观显示偏离模态

#### 解释

- **ASSD < 2.5mm**: 边缘平均距离小，配准精确
- **HD95 < 10mm**: 95%的边缘点距离小于10mm，无显著错位
- **Edge IOU > 0.3**: 边缘重叠度高
- **Grad Corr > 0.5**: 梯度方向一致，结构对齐

---

### 任务3: 基于ROI的区域一致性分析

#### 分析的ROI

基于FreeSurfer 102类标签，重点分析：

| ROI名称 | 标签ID | 解剖位置 | 重要性 |
|---------|--------|----------|--------|
| Thalamus | 5 | 丘脑 | 深部灰质核团 |
| Caudate | 6 | 尾状核 | 基底节 |
| Putamen | 7 | 壳核 | 基底节 |
| Pallidum | 8 | 苍白球 | 基底节 |
| Hippocampus | 12 | 海马 | 内侧颞叶 |
| Amygdala | 13 | 杏仁核 | 边缘系统 |
| Cerebellum_Cortex | 4 | 小脑皮层 | 后颅窝 |

#### 计算统计

对每个模态在每个ROI中计算：
- **Mean Signal**: ROI内平均信号强度
- **Std Signal**: 信号标准差
- **Contrast**: ROI与邻近区域的对比度
- **N Voxels**: ROI体素数

#### 输出示例

- **2张热图**:
  1. ROI信号强度热图（每个模态归一化）
  2. ROI对比度热图

#### 解释

- **信号强度模式一致**: 说明模态在特定结构中的表现一致
- **对比度异常**: 某些模态在特定结构中信号异常（可能是伪影或错配）
- **跨模态比较**: 识别哪些结构在不同模态中最稳定

---

### 任务4: 模态空间降维可视化

#### 方法

- **PCA (主成分分析)**
  - 保留2个主成分
  - 每个模态用7个统计特征表示（均值、方差、中位数、四分位数等）

- **DBSCAN聚类**
  - 检测离群模态
  - eps=1.5, min_samples=2

#### 输出示例

- **PCA散点图**
  - 每个点代表一个模态
  - 颜色区分模态家族（QTI/DWI/CEST/QSM）
  - 红圈标记离群模态

#### 解释

- **聚类清晰**: 同家族模态聚在一起，说明配准一致
- **离群点**: 远离主簇的模态，可能存在配准错误或模态特性差异大
- **PC1/PC2解释方差**: 通常PC1和PC2合计解释60-80%方差

---

### 任务5: QC评分聚合与判断

#### 评分机制

综合前4个任务的所有指标，计算**0-100分**的QC评分：

```python
QC Score =
    0.20 × LNCC_norm +
    0.15 × NGF_norm +
    0.20 × ASSD_norm +
    0.15 × HD95_norm +
    0.15 × Edge_IOU_norm +
    0.15 × Gradient_Corr_norm
```

其中：
- ASSD和HD95归一化时**反转**（越小越好）
- 其他指标越大越好

#### 判断标准

| QC分数 | 判断 | 说明 |
|--------|------|------|
| ≥ 70 | **PASS** | 配准质量良好 |
| 50-69 | **WARN** | 配准质量一般，建议检查 |
| < 50 | **FAIL** | 配准质量差，需要重新配准 |

#### 输出

1. **modality_qc.csv** 表格

   | 字段 | 说明 |
   |------|------|
   | modality_id | 模态索引（0-350） |
   | modality_name | 模态名称 |
   | modality_type | 模态家族（QTI/DWI/CEST等） |
   | mean_lncc | 平均LNCC |
   | mean_ngf | 平均NGF |
   | mind_ssd | MIND-SSD |
   | assd_mm | ASSD (mm) |
   | hd95_mm | HD95 (mm) |
   | edge_iou | 边缘IOU |
   | grad_angle_corr | 梯度相关性 |
   | roi_contrast_avg | ROI平均对比度 |
   | **qc_score** | **综合评分 (0-100)** |
   | **decision** | **PASS/WARN/FAIL** |
   | notes | 问题说明 |

2. **可视化**
   - QC评分柱状图（颜色区分PASS/WARN/FAIL）
   - Top 6模态的雷达图（多指标对比）

3. **JSON报告**
   - 汇总统计
   - Top质量模态列表
   - 低质量模态列表

---

## 输出文件说明

### 图像文件

| 文件名 | 内容 | 尺寸 | 用途 |
|--------|------|------|------|
| task1_similarity_matrices.png | 3张热图（LNCC, NGF, MIND） | 20×6英寸 | 展示模态间相似性 |
| task2_edge_metrics.png | 4张柱状图 | 14×12英寸 | 边缘对齐质量 |
| task3_roi_heatmaps.png | 2张热图 | 16×8英寸 | ROI信号分析 |
| task4_pca_visualization.png | PCA散点图 | 12×8英寸 | 模态聚类和离群检测 |
| task5_qc_scores.png | 评分柱状图+雷达图 | 14×10英寸 | 综合质量评估 |

所有图像：
- **DPI**: 300（出版级质量）
- **格式**: PNG
- **字体**: 支持中文

### 数据文件

#### modality_qc.csv

- **格式**: UTF-8 with BOM（Excel可正常打开中文）
- **行数**: 20行（选定模态数）
- **列数**: 14列
- **排序**: 按QC分数降序

**使用建议**:
```python
import pandas as pd
df = pd.read_csv('modality_qc.csv')

# 筛选FAIL模态
fails = df[df['decision'] == 'FAIL']

# 按指标排序
df_sorted = df.sort_values('assd_mm')
```

#### qc_analysis_report.json

```json
{
  "subject": "PDP_02_xxx_3d_validated_minimal.mat",
  "analysis_date": "2025-11-10T15:30:00",
  "n_modalities_analyzed": 20,
  "reference_modality": "MPRAGE",
  "summary": {
    "total_modalities": 20,
    "pass": 15,
    "warn": 3,
    "fail": 2,
    "avg_qc_score": 68.5,
    "outliers_detected": 2
  },
  "top_quality_modalities": ["MPRAGE", "QTI_μFA", ...],
  "poor_quality_modalities": ["DWI_b220", ...],
  "warnings": ["Z_5ppm", ...]
}
```

---

## 参数配置指南

### 选择不同的参考模态

```python
# 默认: MPRAGE (通道342)
REFERENCE_MODALITY = 341

# 改为其他模态，例如QSM
REFERENCE_MODALITY = 350  # QSM map
```

### 调整QC阈值

```python
QC_THRESHOLDS = {
    'lncc_min': 0.3,        # 降低此值会放宽LNCC要求
    'ngf_min': 0.4,         # 降低此值会放宽NGF要求
    'assd_max': 2.5,        # mm（物理距离）- 增加此值会放宽ASSD要求
    'hd95_max': 10.0,       # mm（物理距离）- 增加此值会放宽HD95要求
    'edge_iou_min': 0.3,    # 降低此值会放宽IOU要求
    'mad_multiplier': 3.0   # 增加此值会减少异常检测数量
}
```

**建议**:
- 严格QC: `assd_max=2.0, hd95_max=8.0` (基于spacing=0.65mm)
- 宽松QC: `assd_max=3.0, hd95_max=15.0` (基于spacing=0.65mm)

**重要**: 如果数据spacing不是0.65mm，需要同步调整阈值：
```python
# 示例：如果数据spacing=1.0mm
# 阈值应相应放大 1.0/0.65 ≈ 1.54倍
assd_max_new = 2.5 * (1.0 / 0.65)  # ≈ 3.85mm
```

### 修改物理间距参数

如果数据来自不同扫描或重建参数：

```python
# 默认: MPRAGE空间 0.65mm各向同性
SPACING = (0.65, 0.65, 0.65)  # mm

# 示例: 改为1mm各向同性
SPACING = (1.0, 1.0, 1.0)  # mm
# 注意：同时需要调整QC_THRESHOLDS中的距离阈值！
```

### 修改选择的模态

```python
# 在SELECTED_MODALITIES中添加或删除模态索引
SELECTED_MODALITIES = [
    0, 1, 4, 7, 14,         # QTI
    20, 50, 100, 160, 220,  # DWI
    # ... 更多
]

# 同时更新MODALITY_NAMES字典
MODALITY_NAMES[新索引] = '新模态名称'
```

### 添加新ROI

```python
# 查看标签映射Excel，找到ROI的标签ID
KEY_ROIS = {
    'Thalamus': 5,
    'Caudate': 6,
    # 添加新ROI
    'Accumbens': 26,  # 示例（需确认标签ID）
}
```

---

## 故障排除

### 问题1: 内存不足

**症状**: `MemoryError` 或系统卡死

**解决方案**:
1. 减少选择的模态数量（从20个减到10个）
2. 减少MIND-SSD采样点数（代码中`n_samples=1000`改为`500`）
3. 在ASSD/HD95计算中减少采样（改为`n_samples=500`）

### 问题2: 计算时间过长

**症状**: 任务1运行超过30分钟

**解决方案**:
1. 检查是否在计算全部351通道（应该只有20个）
2. 增加MIND-SSD的patch_size（从3改为5，计算更快）
3. 减少LNCC的window_size（从7改为5）

### 问题3: 文件未找到

**症状**: `FileNotFoundError`

**解决方案**:
```python
# 检查路径
data_path = Path(DATA_DIR) / SUBJECT_FILE
print(f"查找文件: {data_path}")
print(f"文件存在: {data_path.exists()}")

# 列出目录内容
print(list(Path(DATA_DIR).glob("*.mat")))
```

### 问题4: FreeSurfer标签映射错误

**症状**: ROI分析结果为空或NaN

**解决方案**:
```python
# 检查标签是否存在
for roi_name, roi_id in KEY_ROIS.items():
    n_voxels = np.sum(region_labels == roi_id)
    print(f"{roi_name} (ID={roi_id}): {n_voxels} voxels")
```

如果某个ROI体素数为0，说明：
- 标签ID错误（检查Excel文件）
- 该被试确实没有这个结构

### 问题5: 可视化中文乱码

**解决方案**:
```python
# 在Cell 1添加
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# 或
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows/Linux
```

---

## 扩展到全模态分析

### 从20个模态扩展到351个

**步骤**:

1. **修改选择列表**

```python
# 替换SELECTED_MODALITIES
SELECTED_MODALITIES = list(range(351))  # 所有351个模态
```

2. **调整计算策略**

由于351×351 = 123,201对比较，需要优化：

```python
# 在任务1中，只计算与参考模态的相似性
ref_img = extract_modality(data_4d, REFERENCE_MODALITY, brain_mask)

lncc_scores = []
for mod_idx in tqdm(range(351)):
    mod_img = extract_modality(data_4d, mod_idx, brain_mask)
    lncc = compute_lncc(mod_img, ref_img, brain_mask)
    lncc_scores.append(lncc)

# 不计算全矩阵
```

3. **并行化**

```python
from multiprocessing import Pool

def compute_metrics_parallel(mod_idx):
    # 计算单个模态的所有指标
    ...
    return results

with Pool(8) as p:  # 8个进程
    results = p.map(compute_metrics_parallel, range(351))
```

4. **预计时间**

- **20个模态**: ~30分钟
- **351个模态（优化后）**: ~3-5小时
- **351个模态（全矩阵）**: ~10-20小时

### 批量处理多个被试

```python
# 创建一个循环
subjects = [
    "PDP_02_xxx_3d_validated_minimal.mat",
    "PDP_05_xxx_3d_validated_minimal.mat",
    # ...
]

for subject_file in subjects:
    print(f"\n处理被试: {subject_file}")

    # 加载数据
    mri_data = load_minimal_3d_data(Path(DATA_DIR) / subject_file)

    # 运行所有分析任务
    # ... (完整代码)

    # 保存结果到被试特定的目录
    subject_output_dir = OUTPUT_DIR / subject_file.replace('.mat', '')
    subject_output_dir.mkdir(exist_ok=True)
```

---

## 最佳实践

### 1. 迭代式分析

建议流程：
1. **先分析1个被试** → 验证方法正确性
2. **扩展到3-5个被试** → 检查稳定性
3. **扩展到全部38个被试** → 完整分析

### 2. 保存中间结果

```python
# 在任务1完成后保存矩阵
np.savez(OUTPUT_DIR / 'similarity_matrices.npz',
         lncc=lncc_matrix,
         ngf=ngf_matrix,
         mind=mind_matrix)

# 后续可以快速加载
data = np.load(OUTPUT_DIR / 'similarity_matrices.npz')
lncc_matrix = data['lncc']
```

### 3. 模块化代码

如果要频繁使用，建议将函数封装成Python模块：

```python
# qc_metrics.py
def compute_all_metrics(data_4d, brain_mask, modalities):
    # ...
    return results

# 在notebook中
from qc_metrics import compute_all_metrics
results = compute_all_metrics(data_4d, brain_mask, SELECTED_MODALITIES)
```

### 4. 版本控制

记录每次分析的参数：

```python
# 在JSON报告中添加
report['parameters'] = {
    'selected_modalities': SELECTED_MODALITIES,
    'qc_thresholds': QC_THRESHOLDS,
    'reference_modality': REFERENCE_MODALITY,
    'version': '1.0.0'
}
```

---

## 引用和参考

### 方法学参考

- **LNCC**: Avants et al. (2008) "Symmetric diffeomorphic image registration"
- **NGF**: Haber & Modersitzki (2006) "Intensity Gradient Based Registration"
- **MIND**: Heinrich et al. (2012) "MIND: Modality independent neighbourhood descriptor"
- **ASSD/HD**: Taha & Hanbury (2015) "Metrics for evaluating 3D medical image segmentation"

### 相关文献

1. 多模态MRI配准质量控制
2. FreeSurfer自动分割
3. 深部灰质核团影像学

---

## 更新日志

### v1.0.0 (2025-11-10)

- ✅ 初始版本
- ✅ 实现5个核心分析任务
- ✅ 支持20个代表性模态
- ✅ 生成7种输出文件
- ✅ 完整文档

---

## 联系与支持

如有问题或建议：
1. 检查本文档的[故障排除](#故障排除)章节
2. 查看代码注释中的详细说明
3. 检查输出目录中的日志信息

---

**文档最后更新**: 2025-11-10
**适用数据集**: 3D Minimal v1.0.0
**分析代码**: multimodal_mri_qc_analysis.ipynb + multimodal_qc_tasks_2_5.py
