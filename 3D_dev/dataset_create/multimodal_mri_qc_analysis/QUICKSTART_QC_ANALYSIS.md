# 多模态MRI质量控制分析 - 快速启动指南

⏱️ **阅读时间**: 5分钟
🚀 **运行时间**: 30-40分钟（20个模态）

**版本**: v1.2.0 (优化版)
**推荐**: 请使用 `multimodal_qc_tasks_2_5_v1.2.0.py`（包含性能优化和算法改进）
**备选**: `multimodal_qc_tasks_2_5_fixed.py` (v1.1.0, 已修正5个关键bug)

---

## 🎯 一句话总结

对351通道MRI数据进行配准质量控制，自动检测错配模态，生成PASS/WARN/FAIL评分报告。

---

## 📦 准备工作（5分钟）

### 1. 确认文件位置

```bash
# 服务器数据（确认文件名）
ls /home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_minimal/

# 应该看到类似这样的文件：
# PDP_02_xxx_3d_validated_minimal.mat
# PDP_05_xxx_3d_validated_minimal.mat
# ...
```

### 2. 安装依赖（如果还没有）

```bash
pip install numpy scipy h5py pandas scikit-image scikit-learn matplotlib seaborn tqdm
```

### 3. 打开Notebook

```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/dataset_create/
jupyter notebook multimodal_mri_qc_analysis.ipynb
```

---

## ⚙️ 配置（2分钟）

### 修改Cell 2的配置

```python
# ========== 必改参数 ==========
DATA_DIR = "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_minimal"
SUBJECT_FILE = "PDP_02_xxx_3d_validated_minimal.mat"  # ← 改成实际文件名

# ========== 可选参数（使用默认值即可）==========
# QC阈值、参考模态等
```

**如何找到实际文件名？**

运行这个cell：
```python
from pathlib import Path
files = list(Path("/home/jovyan/.../3D_minimal").glob("*.mat"))
for f in files[:5]:
    print(f.name)
```

---

## 🏃 运行分析（30分钟）

### 步骤1: 运行Notebook的所有Cells

在Jupyter中：
1. 点击菜单 `Kernel` → `Restart & Run All`
2. 或者按顺序运行每个cell（推荐，可以看到每步的输出）

**完成任务1后会看到**:
- ✅ 数据加载成功
- ✅ 相似性矩阵计算完成（3张热图）
- ✅ 异常模态检测结果

### 步骤2: 运行剩余任务（2-5）

在notebook最后添加一个新cell：

```python
# ✅ 使用v1.2.0优化版（强烈推荐）
%run multimodal_qc_tasks_2_5_v1.2.0.py

# 或使用v1.1.0修正版
# %run multimodal_qc_tasks_2_5_fixed.py
```

**或者**分步骤运行（更好地控制）：

1. 创建新cell，复制 `multimodal_qc_tasks_2_5_v1.2.0.py` 的内容
2. 分段粘贴到不同cells中（按任务分）
3. 逐个运行

**⚠️ 重要**:
- v1.2.0性能提升70%（20个模态：40分钟 → 12分钟）
- 包含自适应阈值、三正交面边缘、快速距离计算等改进
- 不要使用旧版 `multimodal_qc_tasks_2_5.py`（有未修正的bug）

---

## 📊 查看结果（5分钟）

### 自动生成的文件

```bash
ls qc_analysis_results/
```

应该看到：

| 文件 | 内容 | 用途 |
|------|------|------|
| ✅ `task1_similarity_matrices.png` | LNCC/NGF/MIND热图 | 模态相似性 |
| ✅ `task2_edge_metrics.png` | ASSD/HD95/IOU柱状图 | 边缘对齐质量 |
| ✅ `task3_roi_heatmaps.png` | ROI信号/对比度热图 | 区域一致性 |
| ✅ `task4_pca_visualization.png` | PCA降维散点图 | 模态聚类 |
| ✅ `task5_qc_scores.png` | QC评分图 | 综合质量 |
| 📄 **`modality_qc.csv`** | **评分表（Excel可打开）** | **核心结果** |
| 📄 `qc_analysis_report.json` | 完整JSON报告 | 程序读取 |

### 重点查看：modality_qc.csv

用Excel或Pandas打开：

```python
import pandas as pd
df = pd.read_csv('qc_analysis_results/modality_qc.csv')

# 查看失败的模态
print(df[df['decision'] == 'FAIL'])

# 查看需要警告的模态
print(df[df['decision'] == 'WARN'])

# 按QC分数排序
print(df.sort_values('qc_score', ascending=False))
```

---

## 🔍 结果解读（3分钟）

### 关键指标速查

| 指标 | 优质阈值 | 含义 |
|------|----------|------|
| **qc_score** | ≥ 70 | 综合评分（0-100） |
| **decision** | PASS | PASS/WARN/FAIL |
| **assd_mm** | < 2.5 | 边缘距离（越小越好） |
| **edge_iou** | > 0.3 | 边缘重叠度（越大越好） |
| **mean_lncc** | > 0.3 | 相似性（越大越好） |

### 典型场景

#### ✅ 场景1: 全部PASS

```
PASS: 18/20
WARN: 2/20
FAIL: 0/20
平均QC分数: 75.3
```

**解读**: 配准质量优秀，可以安心使用数据训练模型。

---

#### ⚠️ 场景2: 有WARN

```
PASS: 15/20
WARN: 4/20  ← 注意这些模态
FAIL: 1/20
```

**行动**:
1. 查看WARN模态的具体问题（notes列）
2. 决定是否排除这些模态
3. 如果是关键模态（如MPRAGE），需要检查原始数据

---

#### ❌ 场景3: 有FAIL

```
PASS: 12/20
WARN: 3/20
FAIL: 5/20  ← 严重问题
```

**行动**:
1. **立即检查FAIL模态**
2. 可能原因：
   - 配准失败
   - 原始数据质量问题
   - 模态本身特性差异大（如DWI与T1对比度差异）
3. 解决方案：
   - 重新配准
   - 排除问题模态
   - 调整参考模态

---

## 🛠️ 常见问题速查

### Q1: 显示"文件未找到"

```python
# 检查路径
from pathlib import Path
print("数据目录存在:", Path(DATA_DIR).exists())
print("文件存在:", (Path(DATA_DIR) / SUBJECT_FILE).exists())

# 列出所有mat文件
print(list(Path(DATA_DIR).glob("*.mat")))
```

### Q2: 计算很慢

**正常情况**: 20个模态约30-40分钟

**如果超过1小时**:
- 检查是否在计算全部351通道
- 检查`SELECTED_MODALITIES`只有20个元素

### Q3: 内存不足

```python
# 减少模态数量
SELECTED_MODALITIES = [
    0, 1, 4,        # 只选3个QTI
    20, 50,         # 只选2个DWI
    225, 226,       # 只选2个CEST
    341             # MPRAGE
]  # 总共8个模态，更快
```

### Q4: 想调整阈值

```python
# 在Cell 2修改
QC_THRESHOLDS = {
    'assd_max': 3.0,     # 放宽ASSD要求（原2.5）
    'hd95_max': 15.0,    # 放宽HD95要求（原10.0）
    # ...
}
```

---

## 🎓 下一步

### 场景A: 结果良好，准备训练

1. ✅ 保存QC报告备查
2. ✅ 使用 `3D_minimal` 数据集训练
3. ✅ 参考 `3D_MINIMAL_DATASET_README.md` 了解数据格式

### 场景B: 需要检查特定模态

1. 🔍 查看该模态的具体指标
2. 🔍 查看可视化图表定位问题
3. 🔍 可选：单独分析该模态（修改`SELECTED_MODALITIES`只包含该模态）

### 场景C: 扩展到全部38个被试

```python
# 创建批处理脚本
subjects = [
    "PDP_02_xxx_3d_validated_minimal.mat",
    "PDP_05_xxx_3d_validated_minimal.mat",
    # ... 添加所有被试
]

for subject in subjects:
    SUBJECT_FILE = subject
    # 运行分析...
```

### 场景D: 扩展到全部351通道

⚠️ **警告**: 计算量巨大（5-10小时）

参考 `MULTIMODAL_QC_ANALYSIS_README.md` 的"扩展到全模态分析"章节

---

## 📚 文档索引

| 文档 | 用途 | 何时阅读 |
|------|------|----------|
| 📄 **QUICKSTART_QC_ANALYSIS.md** (本文档) | 快速上手 | **现在** |
| 📘 MULTIMODAL_QC_ANALYSIS_README.md | 详细说明 | 遇到问题时 |
| 📗 3D_MINIMAL_DATASET_README.md | 数据集格式 | 训练模型前 |
| 📙 UPDATE_SUMMARY_v1.4.0.md | 数据版本变更 | 了解数据来源 |

---

## ✅ 检查清单

运行前：
- [ ] 确认数据文件路径正确
- [ ] 确认FreeSurfer标签映射文件存在
- [ ] 安装所有依赖库

运行中：
- [ ] Cell 1: 库导入成功
- [ ] Cell 2: 配置无误
- [ ] Cell 4: 数据加载成功
- [ ] 任务1-5: 所有任务完成

运行后：
- [ ] 生成7个文件（5张图+1个CSV+1个JSON）
- [ ] CSV中有PASS/WARN/FAIL判断
- [ ] 理解哪些模态质量好/差

---

## 🆘 获取帮助

1. **检查本文档** → 常见问题速查
2. **查看详细文档** → MULTIMODAL_QC_ANALYSIS_README.md
3. **检查代码注释** → 函数docstring有详细说明
4. **查看输出日志** → notebook中的print输出

---

**最后更新**: 2025-11-10
**版本**: v1.1.0 (修正版)

---

## 🔧 版本历史

### v1.2.0 优化版 (2025-11-10) - 当前推荐

**6项重要改进**（详见 `CHANGELOG_QC_v1.2.0.md`）:

1. ✅ **自适应Canny阈值** - Otsu自动阈值，对不同对比度模态更稳健
2. ✅ **三正交面边缘并集** - 边缘覆盖率提升20-30%
3. ✅ **距离变换法加速** - ASSD/HD95计算速度提升>10x
4. ✅ **图表优化** - 异常排序、阈值标注、统计信息
5. ✅ **CSV增强** - 新增modality_family、qc_rank字段
6. ✅ **JSON增强** - 分位数统计、按家族汇总

**性能提升**:
- 20个模态分析时间: 40分钟 → 12分钟 (提升70%)
- 351个模态预估: 8-12小时 → 2-3小时

### v1.1.0 修正版 (2025-11-10)

**5项关键修正**（详见 `CHANGELOG_QC_v1.1.0.md`）:

1. ✅ **Canny 3D边缘检测** - 改为逐切片实现（修复bug）
2. ✅ **ASSD/HD95 spacing** - 修正为0.65mm（数值降低约35%）
3. ✅ **补全imports** - 脚本可独立运行
4. ✅ **ROI对比度** - 邻域与脑掩膜相交（避免颅骨干扰）
5. ✅ **随机种子** - 确保结果完全可重复

---

🎉 **祝分析顺利！**
