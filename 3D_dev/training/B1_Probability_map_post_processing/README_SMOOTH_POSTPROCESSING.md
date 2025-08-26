# 体素级Softmax概率图轻量平滑后处理和评估

## 概述

该系统对B0_1D_training模块生成的体素级softmax概率图进行轻量平滑后处理，通过2D平均滤波（3×3和7×7核）来改善预测质量，并全面评估平滑前后的性能变化，重点关注不均衡数据下的检测改善效果。

## 系统特点

- **轻量后处理**：仅对softmax概率进行平滑，不改动前端分类器
- **有效像素加权**：使用完整脑组织掩膜进行平滑，在有标签区域评估
- **多核大小对比**：3×3和7×7平滑核的效果对比
- **全面评估指标**：准确率、宏F1、κ系数、AUPRC等
- **不均衡检测优化**：优先使用AUPRC评估不均衡问题下的检测改善
- **可视化对比**：混淆矩阵和指标对比图表
- **HDF5格式支持**：兼容B0训练模块输出的HDF5格式预测文件
- **智能文件匹配**：从预测文件属性自动匹配正确的标签文件
- **大体积支持**：使用HDF5保存平滑结果，支持>2GB的概率体积

## 文件说明

### 核心脚本

1. **`smooth_postprocess_eval.py`** - 主处理和评估脚本
   - 实现2D平滑算法（支持快速和标准版本）
   - 计算多类别AUPRC、宏F1、准确率、κ系数
   - 生成混淆矩阵对比和评估报告
   - 保存平滑后的预测结果

2. **`run_smooth_evaluation.sh`** - 快速运行脚本
   - 自动查找预测文件和真实标签文件
   - 运行平滑评估并显示结果摘要
   - 计算改进效果（Δ指标）

## 算法原理

### 2D平滑算法

对softmax概率体积 `softmax_vol[D,H,W,102]` 进行逐切片2D平滑：

1. **有效像素加权**：
   ```python
   # 对每个像素的邻域进行加权平均
   for 邻域内每个像素:
       if mask[像素] > 0:  # 只考虑有效像素
           累积概率值
   
   平滑值 = 累积概率值 / 有效像素数
   ```

2. **边界处理**：
   - 邻域超出边界时自动截断
   - 邻域内无有效像素时保持原值

3. **快速实现**：
   ```python
   # 使用ndimage.convolve加速
   numerator = convolve(prob_masked, kernel)
   denominator = convolve(mask, kernel) 
   result = numerator / (denominator + epsilon)
   ```

### 评估指标

1. **基础分类指标**：
   - 准确率：正确分类的体素比例
   - 宏F1：所有类别F1的平均值
   - κ系数：考虑随机一致性的分类性能

2. **不均衡检测指标**：
   - **宏AUPRC**：每类别AUPRC的平均（重点指标）
   - 微AUPRC：所有类别合并计算的AUPRC

3. **对比分析**：
   - Δ准确率：平滑后与原始的准确率差异
   - Δ宏F1：宏F1的改进程度
   - ΔAUPRC：AUPRC的改进程度（主要关注）

## 使用方法

### 1. 快速开始

```bash
# 修改数据路径（如需要）
vim run_smooth_evaluation.sh
# 设置 RESULTS_DIR="../predictions"  # HDF5预测文件目录
# 设置 DATA_DIR_3D="/path/to/3d/validated/data"

# 运行平滑评估（自动从预测文件属性匹配GT文件）
bash run_smooth_evaluation.sh
```

**重要更新**：现在脚本会自动从预测文件的HDF5属性中读取`test_file_3d`路径，确保与训练时使用完全相同的标签文件，避免文件错配风险。

### 2. 自定义评估

```bash
python smooth_postprocess_eval.py \
    --pred_file "../predictions/predictions_3d_test38.mat" \
    --gt_file /path/to/3d/data/subject38_3d_validated.mat \
    --output_dir ./smooth_eval_results \
    --fast_smooth
```

### 3. 批量处理多个被试

```bash
# 处理所有Leave-one-out结果
for i in {1..38}; do
    if [ -f "../predictions/predictions_3d_test${i}.mat" ]; then
        python smooth_postprocess_eval.py \
            --pred_file "../predictions/predictions_3d_test${i}.mat" \
            --gt_file auto \
            --output_dir ./smooth_eval_results/test${i} \
            --fast_smooth
    fi
done
```

**注意**: 现在GT文件会自动从预测文件的HDF5属性中读取，无需手动指定。

## 参数说明

### 平滑参数
- **kernel_size**: 平滑核大小（3或7）
- **fast_smooth**: 使用快速算法（推荐）

### 评估参数
- **max_classes**: 混淆矩阵显示的最大类别数（默认20）

## 输出文件

### 评估结果
```
smooth_eval_results/
├── evaluation_results.json           # 详细评估指标
├── confusion_matrices_comparison.png  # 混淆矩阵对比图
├── predictions_smooth_k3_test*.mat    # 3×3平滑预测结果（HDF5格式）
└── predictions_smooth_k7_test*.mat    # 7×7平滑预测结果（HDF5格式）
```

**注意**：平滑后的预测结果现在使用HDF5格式保存，支持大体积文件（>2GB），并包含完整的元数据信息。

### JSON结果格式
```json
{
  "original": {
    "accuracy": 0.7234,
    "macro_f1": 0.6891,
    "kappa": 0.7102,
    "macro_auprc": 0.7456,
    "micro_auprc": 0.7823
  },
  "smooth_k3": {
    "accuracy": 0.7298,
    "macro_f1": 0.6945,
    "kappa": 0.7169,
    "macro_auprc": 0.7521,
    "micro_auprc": 0.7891
  },
  "smooth_k7": {
    "accuracy": 0.7267,
    "macro_f1": 0.6923,
    "kappa": 0.7134,
    "macro_auprc": 0.7498,
    "micro_auprc": 0.7864
  }
}
```

### HDF5文件格式
```matlab
predictions_smooth_k3_test38.mat (HDF5/MAT v7.3格式):

数据集:
  - softmax_probabilities: (384, 336, 256, 102) float32, gzip压缩
  - predicted_labels: (384, 336, 256) uint8, gzip压缩

属性:
  - test_subject: 测试被试编号
  - test_file_1d: 原始1D训练文件路径
  - test_file_3d: 对应3D标签文件路径
  - smooth_kernel_size: 平滑核大小 (3 或 7)
  - shape: 数据形状
  - prob_range: 概率值范围 [min, max]
```

## 预期效果

### 平滑改善机制
1. **噪声抑制**：减少孤立错误分类
2. **区域连续性**：增强解剖结构的连续性
3. **边界平滑**：改善区域边界的预测质量
4. **不确定性降低**：提高预测置信度

### 掩膜使用原理

**为什么平滑时使用完整脑组织掩膜而不是有标签掩膜？**

1. **解剖连续性**：脑组织在解剖上是连续的，即使某些区域没有标签
2. **边界平滑**：标签区域边界处需要利用邻近脑组织信息进行平滑
3. **避免人工边界**：仅使用有标签掩膜会创造人工的硬边界，影响平滑效果
4. **保持生物学意义**：在整个脑组织范围内平滑更符合神经解剖学原理

**评估为什么只在有标签区域？**

1. **客观评估**：只有有标签的区域才有ground truth用于评估
2. **避免虚假改善**：不在无标签区域评估，避免因平滑产生的虚假性能提升

### AUPRC改善预期
- **类别不均衡问题**：脑区域大小差异巨大
- **小区域改善**：平滑有助于小区域检测
- **边界区域**：减少边界处的误分类
- **整体检测性能**：AUPRC通常比F1更敏感

## 算法复杂度

### 时间复杂度
- **标准算法**：O(D × H × W × C × K²)
- **快速算法**：O(D × H × W × C)（推荐使用）

### 空间复杂度
- **内存需求**：约3倍原始softmax体积大小
- **推荐配置**：16GB+ RAM用于处理完整数据

## 故障排除

### 内存不足
```bash
# 使用快速算法减少内存使用
--fast_smooth
```

### 文件格式或路径错误
```bash
# 1. 确保预测文件为HDF5格式（由B0_1D_training生成）
# 2. 检查预测文件是否包含必要属性：test_file_3d 或 test_subject
# 3. 验证3D数据目录路径是否正确
# 4. 确保标签文件形状为 (384,336,256)
```

### 标签对齐问题
```python
# 现在使用0-101标签与102通道对齐，不再有-1偏移
# y_true: 0-101 (背景=0, 脑区=1-101)
# y_scores: 102通道 (通道0=背景, 通道1-101=脑区)
```

### 大文件保存失败
```bash
# 现在使用HDF5格式，支持>2GB文件
# 如果仍有问题，检查磁盘空间（~13GB per file）
```

### GT文件匹配失败
```bash
# 检查错误信息：
# "未能从预测文件属性定位真实标签文件"
# 1. 预测文件是否为正确的HDF5格式
# 2. 预测文件是否包含test_file_3d或test_subject属性  
# 3. 3D数据目录路径是否正确
```

## 技术细节

### 平滑实现选择
- **标准版本**：逐像素邻域遍历，内存友好但较慢
- **快速版本**：矢量化卷积操作，速度快但内存占用高

### 掩膜处理
```python
# 定义两种掩膜
region_mask_binary = (region_mask > 0)           # 脑组织区域，用于平滑
valid_mask = (region_mask > 0) & (region_labels > 0)  # 有标签区域，用于评估

# 平滑时使用完整脑组织掩膜，评估时使用有标签掩膜
```

### 多类别AUPRC
```python
# 宏平均：每类别单独计算后平均
macro_auprc = mean([auprc_class_i for i in range(102)])

# 微平均：所有类别合并计算  
micro_auprc = auprc(all_true_labels, all_predictions)
```

## 与其他后处理方法对比

| 方法 | 计算复杂度 | 参数量 | 保形性 | 改善效果 |
|------|------------|--------|--------|----------|
| **2D平滑** | 低 | 0 | 高 | 中等 |
| 3D平滑 | 高 | 0 | 高 | 高 |
| CRF后处理 | 高 | 多 | 中 | 高 |
| 双边滤波 | 中 | 2-3 | 高 | 中等 |

## 下一步扩展

1. **自适应平滑**：根据局部不确定性调整核大小
2. **形态学后处理**：结合开闭运算进一步优化
3. **集成平滑**：结合多种平滑方法的优势
4. **学习式后处理**：训练小型CNN进行端到端后处理优化

## 重要更新记录

### v1.1 - HDF5格式支持与文件匹配改进
- **HDF5读取**：支持B0_1D_training输出的HDF5格式预测文件
- **智能文件匹配**：从预测文件HDF5属性自动匹配正确的GT文件，避免错配
- **标签对齐修正**：去除-1偏移，实现0-101标签与102通道完美对齐
- **大文件支持**：使用HDF5格式保存平滑结果，支持>2GB概率体积
- **严格形状验证**：替换危险的转置操作为严格的形状断言
- **元数据保存**：保存完整的文件路径和处理参数用于追溯