# 脑区分割训练流程 - 单轮快跑版 (Single-Split Training)

**版本**: v1.0
**日期**: 2025-01-11
**目标**: 按被试分层的36/1/1划分，支持软标签训练与3D预测还原

---

## 概述

本训练流程实现了基于4×4096全连接网络的脑区分割任务，使用概率标签（软标签）进行训练，并支持将1D预测结果无损还原到3D空间。

### 主要特性

- ✅ **被试分层**: 严格按被试划分36 train / 1 val / 1 test，杜绝数据泄漏
- ✅ **逐被试标准化**: 每个被试独立进行z-score标准化（351维特征）
- ✅ **软标签训练**: 使用概率分布而非one-hot编码，保留不确定性信息
- ✅ **3D预测还原**: 将1D预测按C-order无损映射回3D空间
- ✅ **完整评估指标**: Gross Acc, NLL, Brier, Macro/Micro F1, Top-k等
- ✅ **可视化支持**: 混淆矩阵、3D切片对比图
- ✅ **Jupyter友好**: 提供开箱即用的Notebook快速启动

---

## 快速开始

### 方式1: Jupyter Notebook（推荐）

```bash
# 1. 打开notebook
jupyter notebook notebook_quickstart.ipynb

# 2. 修改DATA_ROOT为你的数据路径
# 3. 运行所有单元格
```

### 方式2: 命令行

```bash
python train_runner.py \
  --data-root /path/to/downsampling \
  --epochs 3 \
  --batch-size 256 \
  --save-dir runs/quickstart
```

---

## 数据格式要求

训练脚本期望的数据目录结构：

```
data_root/
├── 1d/
│   ├── subject001_1d.npz
│   ├── subject002_1d.npz
│   └── ...
├── 3d/
│   ├── subject001_3d.npz
│   ├── subject002_3d.npz
│   └── ...
```

### 1D文件 (`*_1d.npz`)

| Key | 维度 | 类型 | 说明 |
|-----|------|------|------|
| `multidim_data` | `(n_vox, 351)` | float32 | ROI内体素特征 |
| `seg_one_hot` | `(102, n_vox)` | float32 | 软标签（概率分布） |
| `region_seg` | `(n_vox,)` | uint8 | 硬标签（仅对照） |
| `region` | `(Z', X', Y')` | uint8 | ROI掩码（用于3D还原） |
| `n_voxels` | 标量 | int | 体素数 |

### 3D文件 (`*_3d.npz`)

| Key | 维度 | 类型 | 说明 |
|-----|------|------|------|
| `data_lr` | `(Z', X', Y', 351)` | float32 | 多模态特征 |
| `proba_labels` | `(Z', X', Y', 102)` | float32 | 概率标签 |
| `region_mask_lr` | `(Z', X', Y')` | uint8 | ROI掩码 |

**注意**:
- 坐标系为 `(Z, X, Y, *)` 存储顺序
- 1D数据与3D数据通过C-order索引一一对应
- 标签范围: 0-101（102个脑区）

---

## 训练流程详解

### 步骤1: 被试划分

```python
split_subjects(all_subject_ids, test_id=None, val_id=None, seed=42)
```

- **默认规则**: 按 `sorted(all_ids)` 取最后两个为 val/test，其余为 train
- **手动指定**: 可通过 `--val-id` 和 `--test-id` 参数手动指定
- **输出**: `split_summary.json` 记录划分详情

### 步骤2: 数据预处理

#### 逐被试z-score标准化

对每个被试 `multidim_data ∈ ℝ^{n_vox×351}`：

```
μ[c] = mean(x[:, c])  # 第c维特征的均值
σ[c] = std(x[:, c])   # 第c维特征的标准差
x_norm[:, c] = (x[:, c] - μ[c]) / max(σ[c], 1e-8)
```

- ✅ **严禁泄漏**: Val/Test被试的统计量**不进入**训练集
- ✅ **独立标准化**: 每个被试使用自己的均值/标准差
- 📁 **保存统计量**: `norm_stats.json` 记录每个被试的 μ 和 σ

#### 软标签准备

- 将 `seg_one_hot` 转置为 `(n_vox, 102)`
- 验证每行概率和 ≈ 1.0
- 直接用于训练（不转换为硬标签）

### 步骤3: 模型架构

```
输入: (batch, 351)
  ↓
Dense(4096) + ReLU + Dropout(0.5)
  ↓
Dense(4096) + ReLU + Dropout(0.5)
  ↓
Dense(4096) + ReLU + Dropout(0.5)
  ↓
Dense(4096) + ReLU + Dropout(0.5)
  ↓
Dense(102) → Logits
  ↓
Softmax → 概率分布
```

### 步骤4: 损失函数

**软标签交叉熵** (Soft Cross-Entropy):

```
L = -Σ_i p_i log(q_i)
```

其中：
- `p_i`: 软标签（真实概率分布）
- `q_i`: 预测概率（softmax输出）

**可选**: 类别权重

```
w_c = (1 / freq_c)^α
样本权重 = Σ p_i * w_i
```

### 步骤5: 训练与早停

- **优化器**: Adam (lr=1e-5, weight_decay=1e-5)
- **早停指标**: Validation NLL（越低越好）
- **保存策略**: 每个epoch后验证，保存最优模型到 `best.pth`

### 步骤6: 评估指标

#### 1D指标（体素级）

| 指标 | 说明 |
|------|------|
| Gross Accuracy | 硬预测准确率 |
| NLL | 负对数似然（软标签） |
| Brier Score | 概率预测误差 |
| Macro-F1 | 类别平均F1 |
| Micro-F1 | 全局F1 |
| Top-3/5 Acc | 前k预测准确率 |

#### 3D指标（空间级）

- **3D Gross Accuracy**: 将1D预测还原到3D后计算的准确率
- **一致性验证**: 确保1D与3D指标一致

### 步骤7: 3D预测还原

使用 `Data3D1DMapper` 将 `(n_vox, 102)` 预测映射到 `(Z', X', Y', 102)`：

```python
pred_probs_3d = mapper.map_1d_predictions_to_3d(
    pred_probs_1d,
    region_mask=region_mask
)
pred_argmax_3d = pred_probs_3d.argmax(axis=-1)
```

**关键约束**:
- 严格按 1D 的 C-order 顺序映射
- 使用 `region` 掩码定位ROI体素
- 非ROI区域填充为0

### 步骤8: 可视化

- **混淆矩阵**: 行归一化，保存为CSV
- **3D切片图**: 预测 vs 真实 vs 差异（默认2个切片）

---

## 输出文件结构

```
runs/quickstart/
├── split_summary.json          # 被试划分信息
├── norm_stats.json             # 每个被试的标准化统计量
├── class_weights.npy           # 类别权重（如果启用）
├── run_summary.json            # 运行摘要（包含所有指标）
├── checkpoints/
│   └── best.pth                # 最优模型权重
├── logs/
│   └── train.log               # 训练日志
├── metrics_val.json            # 验证集指标
├── metrics_test.json           # 测试集指标
├── confusion_val.csv           # 验证集混淆矩阵
├── confusion_test.csv          # 测试集混淆矩阵
├── pred_3d/
│   ├── val_{subject_id}_pred_softmax_3d.npz   # 验证集3D概率预测
│   ├── val_{subject_id}_argmax_3d.npz         # 验证集3D硬标签
│   ├── test_{subject_id}_pred_softmax_3d.npz  # 测试集3D概率预测
│   └── test_{subject_id}_argmax_3d.npz        # 测试集3D硬标签
└── figs/
    ├── {subject_id}_slice_z6.png    # 3D切片可视化
    └── {subject_id}_slice_z12.png
```

---

## 命令行参数

### 必需参数

- `--data-root`: 数据根目录（包含 `1d/` 和 `3d/` 子目录）

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--val-id` | None | 验证集被试ID（默认取倒数第2个） |
| `--test-id` | None | 测试集被试ID（默认取最后一个） |
| `--seed` | 42 | 随机种子 |
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 256 | 批大小 |
| `--lr` | 1e-5 | 学习率 |
| `--weight-decay` | 1e-5 | L2正则化系数 |
| `--use-class-weights` | False | 是否使用类别权重 |
| `--class-weight-alpha` | 0.5 | 类别权重平衡系数 |
| `--max-vox-per-subject` | None | 每个被试最大体素数（用于限制显存） |
| `--save-dir` | runs/quickstart | 保存目录 |

### 使用示例

```bash
# 基础训练（默认参数）
python train_runner.py --data-root /path/to/downsampling

# 指定val/test被试
python train_runner.py \
  --data-root /path/to/downsampling \
  --val-id subject037 \
  --test-id subject038

# 启用类别权重
python train_runner.py \
  --data-root /path/to/downsampling \
  --use-class-weights \
  --class-weight-alpha 0.5

# 限制显存使用
python train_runner.py \
  --data-root /path/to/downsampling \
  --max-vox-per-subject 50000

# 完整训练（25 epochs）
python train_runner.py \
  --data-root /path/to/downsampling \
  --epochs 25 \
  --batch-size 256 \
  --lr 1e-5 \
  --save-dir runs/full_training
```

---

## Jupyter Notebook使用

### 1. 打开Notebook

```bash
jupyter notebook notebook_quickstart.ipynb
```

### 2. 修改配置

在第一个代码单元格中修改：

```python
DATA_ROOT = "../../dataset_create/downsampling"  # 你的数据路径
EPOCHS = 3              # 快跑模式
USE_CLASS_WEIGHTS = False  # 是否使用类别权重
```

### 3. 运行所有单元格

- 自动完成训练、评估、可视化全流程
- 实时查看训练日志
- 交互式查看结果图表

### 4. 查看结果

Notebook自动显示：
- 运行摘要（指标统计）
- 混淆矩阵热图
- 3D切片可视化
- 文件结构树

---

## 性能优化

### 显存优化

1. **限制每个被试体素数**:
   ```python
   --max-vox-per-subject 50000
   ```

2. **减小批大小**:
   ```python
   --batch-size 128
   ```

3. **使用混合精度**（未实现，可扩展）

### 速度优化

1. **减少验证频率**（未实现，可扩展）
2. **使用DataLoader多进程**（已启用）
3. **预加载所有数据到内存**（当前实现）

---

## 扩展功能（未来版本）

### 即将支持

- [ ] **LOSO交叉验证** (Leave-One-Subject-Out)
- [ ] **K-fold交叉验证** (Stratified by subject)
- [ ] **温度缩放校准** (Temperature Scaling)
- [ ] **类别权重Ablation实验**
- [ ] **Early Stopping控制**
- [ ] **学习率调度器**
- [ ] **混合精度训练** (AMP)
- [ ] **TensorBoard日志**
- [ ] **模型集成** (Ensemble)

### 使用建议

1. **先跑快速版本**（3 epochs）验证数据管线
2. **调整超参数**后再进行完整训练（25 epochs）
3. **启用类别权重**处理类别不平衡
4. **使用LOSO/K-fold**进行完整评估（扩展版本）

---

## 故障排查

### 问题1: 文件找不到

**错误**: `FileNotFoundError: 1D文件不存在`

**解决**:
- 检查 `DATA_ROOT` 路径是否正确
- 确认 `1d/` 和 `3d/` 子目录存在
- 验证文件命名格式: `{subject_id}_1d.npz` 和 `{subject_id}_3d.npz`

### 问题2: 显存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
```python
--max-vox-per-subject 30000  # 限制体素数
--batch-size 128              # 减小批大小
```

### 问题3: 标签维度不匹配

**错误**: `AssertionError: 期望102个类别`

**解决**:
- 检查 `seg_one_hot` 是否为 `(102, n_vox)` 格式
- 确认使用的是批处理脚本v1.4.0生成的数据

### 问题4: 3D还原失败

**错误**: `ValueError: 体素数不匹配`

**解决**:
- 确认 `region` 掩码与 `n_voxels` 一致
- 验证1D数据与3D数据来自同一被试

---

## 依赖项

### Python版本
- Python >= 3.7

### 必需包

```bash
pip install numpy torch scipy scikit-learn matplotlib seaborn h5py jupyter
```

或使用requirements.txt:

```bash
pip install -r requirements.txt
```

### 外部模块

- `data_3d_1d_mapper.py`: 3D-1D数据映射工具（自动加载）
  - 路径: `../dataset_create/1d-3d-convert/data_3d_1d_mapper.py`（若使用3D_dev目录，则为 `../3D_dev/dataset_create/1d-3d-convert/data_3d_1d_mapper.py`）

---

## 版本历史

### v1.0 (2025-01-11)

- ✅ 初始版本发布
- ✅ 支持36/1/1固定划分
- ✅ 逐被试z-score标准化
- ✅ 软标签训练
- ✅ 3D预测还原
- ✅ 完整评估指标
- ✅ Jupyter Notebook快速启动

---

## 引用

如果使用本训练流程，请引用：

```bibtex
@software{brain_segmentation_training_2025,
  title={Brain Region Segmentation Training Pipeline},
  author={Claude Code},
  year={2025},
  version={1.0}
}
```

---

## 联系与支持

- **文档**: 本README + 代码注释
- **示例**: `notebook_quickstart.ipynb`
- **数据格式**: 参见 `../dataset_create/DOWNSAMPLED_DATA_FORMAT.md`

---

**最后更新**: 2025-01-11
**维护者**: Claude Code
**许可**: MIT
