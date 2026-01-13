# 完整使用指南 - 排除实验与扩展评估指标系统

## 📋 目录

1. [系统概述](#系统概述)
2. [新增评估指标详解](#新增评估指标详解)
3. [完整使用流程](#完整使用流程)
4. [生成的文件说明](#生成的文件说明)
5. [科学解释与应用](#科学解释与应用)
6. [故障排除](#故障排除)

---

## 系统概述

本系统提供了完整的脑部体素分割实验框架，支持：
- ✅ 排除配准问题数据集的对比实验
- ✅ 9大类别、13个详细评估指标
- ✅ 统计显著性检验
- ✅ 多维度可视化分析
- ✅ HTML格式详细报告

### 核心文件

```
B0_1D_training/
├── evaluation_metrics.py                    # 评估指标模块（核心）
├── train_1d_with_3d_dataset.py             # 训练脚本（已扩展）
├── compare_exclude_results_extended.py     # 对比分析脚本（扩展版）
├── generate_detailed_report.py             # 单实验详细报告生成器
├── run_exclude_experiments.sh              # 批量实验脚本
├── exclude_subjects.txt                    # 排除列表配置
└── check_exclude_logic.py                  # 排除逻辑验证工具
```

---

## 新增评估指标详解

### 1. 准确性指标 (Accuracy Metrics)

#### Gross Accuracy (GC)
```python
gross_accuracy = correct_predictions / total_predictions
```
- **定义**: 总体准确率，所有体素的分类准确率
- **范围**: [0, 1]，越高越好
- **用途**: 原始基线对比标准
- **注意**: 受类别不平衡影响较大

#### Top-K Accuracy
```python
top_k_accuracy = samples_with_correct_in_topk / total_samples
```
- **Top-1**: 预测第1名正确的比例（等于Gross Accuracy）
- **Top-3**: 预测前3名包含正确答案的比例
- **Top-5**: 预测前5名包含正确答案的比例
- **用途**: 评估模型置信度和覆盖率
- **解释**:
  - Top-3 ≈ 85-90%: 模型对大部分样本有较高信心
  - Top-5 ≈ 90-95%: 模型覆盖良好
- **应用**: 在102类脑区域分类中，Top-3/5反映模型对相似区域的区分能力

### 2. 类别平衡指标 (Balance Metrics)

#### Macro-F1 (核心指标)
```python
macro_f1 = mean([f1_score_class_i for all classes])
```
- **定义**: 所有类别F1分数的算术平均
- **范围**: [0, 1]，越高越好
- **用途**: 模型Checkpoint选择的核心指标
- **优势**: 不受类别不平衡影响，每个类别权重相同
- **注意**: 少数类别性能差会拉低整体分数

#### Balanced Accuracy
```python
balanced_accuracy = mean([recall_class_i for all classes])
```
- **定义**: 所有类别召回率的平均
- **范围**: [0, 1]，越高越好
- **对比**: 与Gross Accuracy相比，考虑了类别不平衡
- **解释**: 如果Balanced Acc << Gross Acc，说明存在严重类别不平衡

#### Weighted F1
```python
weighted_f1 = sum([f1_class_i * n_samples_class_i]) / total_samples
```
- **定义**: 按样本数量加权的F1分数
- **范围**: [0, 1]，越高越好
- **用途**: 反映大类别的性能
- **对比**: Macro-F1关注小类别，Weighted F1关注大类别

### 3. 一致性指标 (Consistency Metrics)

#### Cohen's Kappa (κ)
```python
kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
```
- **定义**: 校正随机一致性后的一致性系数
- **范围**: [-1, 1]
  - κ > 0.8: 几乎完全一致（Excellent）
  - 0.6 < κ ≤ 0.8: 高度一致（Good）
  - 0.4 < κ ≤ 0.6: 中度一致（Moderate）
  - κ ≤ 0.4: 一致性较差（Fair/Poor）
- **优势**: 考虑了随机猜测的影响
- **应用**: 评估预测的可靠性和一致性

### 4. 分割质量指标 (Segmentation Metrics)

#### Macro 3D Soft Dice
```python
soft_dice_c = (2 * sum(p_c * y_c) + ε) / (sum(p_c) + sum(y_c) + ε)
macro_soft_dice = mean([soft_dice_c for all classes])
```
- **定义**: 使用softmax概率的Dice系数的宏平均
- **范围**: [0, 1]，越高越好
- **优势**:
  - 比hard Dice更稳定（使用概率而非0/1）
  - 宏平均确保每个脑区域权重相同
  - 对小区域更敏感
- **应用**: 评估分割质量，特别适合医学图像

### 5. 风险分析 (Risk Analysis)

#### Risk @ 95% Coverage
```python
risk = error_rate_of_retained_samples
coverage = proportion_of_samples_retained
```
- **定义**: 在保留95%样本时的错误率
- **范围**: [0, 1]，越低越好
- **计算**:
  1. 按预测置信度(max probability)排序
  2. 保留top 95%置信度的样本
  3. 计算这些样本的错误率
- **解释**:
  - Risk@95% = 0.1: 保留95%样本时，错误率10%
  - 低风险+高覆盖率 = 模型校准良好
- **应用**:
  - 临床决策：只对高置信度预测采取行动
  - 质量控制：识别需要人工复查的样本

---

## 完整使用流程

### 步骤1: 配置排除列表

```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/Fullyconnected_exclude/B0_1D_training

# 编辑排除列表
vim exclude_subjects.txt
```

添加你的6个配准问题数据集：
```
# 配准有问题的被试（每行一个）
YHC04
YHC2
PDP01
Subject_XYZ
Patient_ABC
Case_123
```

### 步骤2: 验证排除逻辑（推荐）

```bash
python check_exclude_logic.py \
    --data_dir_1d /path/to/1d/data \
    --exclude_file ./exclude_subjects.txt
```

输出示例：
```
排除列表: {'YHC04', 'YHC2', 'PDP01', 'Subject_XYZ', 'Patient_ABC', 'Case_123'}

================================================================================
✅ 将被包含的数据集 (32个):
================================================================================
  ✓ ODP_01_xxx.mat
  ✓ ODP_02_yyy.mat
  ...

================================================================================
❌ 将被排除的数据集 (6个):
================================================================================
  ✗ ODP_03_YHC04.mat (匹配: ['YHC04'])
  ✗ ODP_05_YHC2.mat (匹配: ['YHC2'])
  ✗ ODP_08_PDP01.mat (匹配: ['PDP01'])
  ...
```

### 步骤3: 配置训练参数

```bash
vim run_exclude_experiments.sh
```

修改关键配置：
```bash
# 数据路径
DATA_DIR_1D="/path/to/1d/data"
DATA_DIR_3D="/path/to/3d/data"

# 固定测试被试（选一个正确的被试名关键字）
FIXED_TEST_SUBJECT="qhlazec"  # 或其他正确被试
```

### 步骤4: 运行批量实验

```bash
chmod +x run_exclude_experiments.sh
./run_exclude_experiments.sh
```

执行流程：
```
实验1/7: 排除 YHC04   (38→37个被试) → 训练25 epochs
实验2/7: 排除 YHC2    (38→37个被试) → 训练25 epochs
实验3/7: 排除 PDP01   (38→37个被试) → 训练25 epochs
实验4/7: 排除 Subject_XYZ ...
实验5/7: 排除 Patient_ABC ...
实验6/7: 排除 Case_123 ...
实验7/7: 排除所有6个  (38→32个被试) → 训练25 epochs
─────────────────────────────────────────────
生成详细汇总报告...
生成可视化图表...
完成！
```

### 步骤5: 查看结果

```bash
cd results_exclude_experiments

# 查看HTML报告（推荐）
open detailed_report.html  # macOS
# 或
xdg-open detailed_report.html  # Linux
# 或直接在浏览器中打开该文件

# 查看控制台输出
cat comparison_report_extended.json | python -m json.tool | less

# 查看可视化图表
open *.png
```

### 步骤6: 生成单个实验的详细报告（可选）

```bash
# 为某个特定实验生成详细报告
python generate_detailed_report.py \
    --result_dir ./results_exclude_experiments/exp1_exclude_YHC04 \
    --output_dir ./results_exclude_experiments/exp1_exclude_YHC04/detailed_report

# 查看HTML报告
open ./results_exclude_experiments/exp1_exclude_YHC04/detailed_report/detailed_report.html
```

---

## 生成的文件说明

### 目录结构

```
results_exclude_experiments/
├── exp1_exclude_YHC04/
│   ├── dense_4x4096_model_test1.pth         # 模型权重+scaler
│   ├── history_test1.json                   # 训练历史（包含所有指标）
│   ├── predictions_3d_test1.mat             # 3D概率体积（~13GB）
│   └── detailed_report/                     # 详细报告目录
│       ├── detailed_report.html             # HTML报告（主文件）
│       ├── training_curves_detailed.png     # 训练曲线分析
│       ├── metrics_radar.png                # 指标雷达图
│       ├── risk_coverage_analysis.png       # 风险-覆盖率曲线
│       └── metrics_summary.csv              # 指标汇总表
├── exp2_exclude_YHC2/
│   └── ... (同上)
├── ...
├── exp7_exclude_all/
│   └── ...
├── comparison_report_extended.json          # 详细对比报告（JSON）
├── comprehensive_metrics_comparison.png     # 9个指标对比图（3×3）
├── topk_accuracy_comparison.png             # Top-K准确率对比
├── metrics_correlation_heatmap.png          # 指标相关性热力图
└── single_vs_all_boxplot.png               # 单个vs所有箱线图
```

### JSON报告结构

```json
{
  "summary": {
    "total_experiments": 7,
    "single_exclusion_experiments": 6,
    "all_exclusion_experiments": 1
  },
  "experiments": [
    {
      "experiment": "exp1_exclude_YHC04",
      "excluded_subject": "YHC04",
      "exclude_type": "single",
      "best_test_f1": 0.7234,
      "gross_accuracy": 0.7234,
      "top1_accuracy": 0.7234,
      "top3_accuracy": 0.8567,
      "top5_accuracy": 0.9123,
      "balanced_accuracy": 0.7189,
      "weighted_f1": 0.7456,
      "cohen_kappa": 0.6892,
      "macro_soft_dice": 0.7312,
      "risk_at_95_coverage": 0.2456,
      ...
    },
    ...
  ],
  "statistics": {
    "best_test_f1": {
      "mean": 0.7256,
      "std": 0.0078,
      "median": 0.7245,
      "min": 0.7189,
      "max": 0.7401,
      "q25": 0.7212,
      "q75": 0.7298
    },
    ...
  },
  "comparison": {
    "best_test_f1": {
      "single_mean": 0.7234,
      "single_std": 0.0065,
      "all_value": 0.7401,
      "difference": 0.0167,
      "relative_change": 2.31,
      "statistical_tests": {
        "t_test_p_value": 0.023,
        "wilcoxon_p_value": 0.031,
        "significant": true
      }
    },
    ...
  }
}
```

---

## 科学解释与应用

### 指标选择原则

| 研究目标 | 推荐指标 | 原因 |
|---------|---------|------|
| **模型选择** | Macro-F1 | 平衡所有类别性能 |
| **临床应用** | Balanced Acc + Risk@95% | 考虑类别平衡+风险控制 |
| **算法对比** | Gross Acc + Top-3 + Kappa | 多角度评估 |
| **分割质量** | Macro Soft Dice | 专门针对分割任务 |
| **可靠性分析** | Cohen's Kappa + Risk@95% | 一致性+风险评估 |

### 统计显著性解释

系统自动进行两种统计检验：

#### 1. 配对t检验（Paired t-test）
```
H0: 排除所有 = 排除单个的均值
H1: 排除所有 ≠ 排除单个的均值

p < 0.05: 拒绝H0，差异显著
p ≥ 0.05: 不能拒绝H0，差异不显著
```

#### 2. Wilcoxon符号秩检验（非参数）
```
用于小样本或数据不满足正态分布的情况
更稳健但功效较低
```

**输出示例**：
```
Metric                    Single (Mean±Std)       All         Diff        p-value
Best Test F1              0.7234±0.0065           0.7401      +0.0167     0.023 ***
Gross Accuracy            0.7189±0.0072           0.7356      +0.0167     0.019 ***
Cohen's Kappa             0.6867±0.0092           0.7012      +0.0145     0.045 ***

注: *** 表示 p < 0.05 (统计显著)
```

### 结果解读示例

#### 场景1: 排除数据集提升显著
```
Best F1: 0.7234 (single) → 0.7401 (all), p=0.023 ***
```
**解释**:
- 排除所有配准问题数据集后，F1提升了0.0167 (2.3%)
- p<0.05，提升具有统计显著性
- **结论**: 这些数据集确实影响了模型性能，建议在后续实验中排除

#### 场景2: 个别数据集影响大
```
排除YHC04: F1=0.7312  (↑ 相对baseline)
排除YHC2:  F1=0.7189  (↓ 相对baseline)
排除PDP01: F1=0.7267  (~ 相对baseline)
```
**解释**:
- YHC04对性能影响最大，排除后提升明显
- YHC2可能包含有用信息，排除反而降低性能
- PDP01影响中性
- **结论**: 需要进一步分析YHC2和YHC04的数据特征

#### 场景3: Top-K分析
```
Top-1: 0.7234 (72.3%)
Top-3: 0.8567 (85.7%)
Top-5: 0.9123 (91.2%)
```
**解释**:
- 85.7%的样本，正确答案在前3预测中
- Top-3 - Top-1 = 13.4%: 模型对13.4%样本不够自信但方向正确
- **应用**: 可以设计两阶段决策系统，对Top-3外的样本进行人工复查

---

## 故障排除

### 问题1: 导入错误

```
ImportError: cannot import name 'compute_all_metrics' from 'evaluation_metrics'
```

**解决方法**:
```bash
# 确保evaluation_metrics.py与训练脚本在同一目录
ls -l evaluation_metrics.py train_1d_with_3d_dataset.py

# 如果不在，复制到正确位置
cp evaluation_metrics.py /path/to/B0_1D_training/
```

### 问题2: 内存不足

```
CUDA out of memory when computing metrics
```

**解决方法**:
```python
# 方法1: 减少batch size
--batch_size 64  # 从128减少到64

# 方法2: 在CPU上计算指标（自动降级，无需配置）

# 方法3: 只在最后epoch计算完整指标（默认已实现）
```

### 问题3: 没有扩展指标

```
警告: 没有扩展指标数据
```

**原因**: 使用了旧版本训练脚本

**解决方法**:
```bash
# 重新训练一次，使用新版脚本
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/data \
    --data_dir_3d /path/to/data \
    --output_dir ./results_new \
    --epochs 25 \
    --save_predictions
```

### 问题4: 统计检验失败

```
Warning: Not enough samples for statistical test
```

**原因**: 单个排除实验少于3个

**解决方法**:
- 至少需要3个单个排除实验才能进行有效的统计检验
- 如果数据集少于3个，结果仅供参考

### 问题5: 图表不显示

```
RuntimeError: Invalid DISPLAY variable
```

**解决方法**:
```bash
# 在服务器上运行时，使用Agg后端
export MPLBACKEND=Agg
python compare_exclude_results_extended.py --results_dir ./results
```

---

## 高级用法

### 自定义指标阈值

在 `evaluation_metrics.py` 中修改：

```python
# 修改风险分析的覆盖率阈值
risk_90 = compute_risk_at_coverage(y_true, y_probs, target_coverage=0.90)
risk_99 = compute_risk_at_coverage(y_true, y_probs, target_coverage=0.99)
```

### 添加自定义指标

```python
# 在 evaluation_metrics.py 中添加
def compute_custom_metric(y_true, y_pred):
    """你的自定义指标"""
    return custom_score

# 在 compute_all_metrics() 中添加
metrics['custom_metric'] = compute_custom_metric(y_true, y_pred)
```

### 批量生成详细报告

```bash
#!/bin/bash
# generate_all_reports.sh

for exp_dir in results_exclude_experiments/exp*/; do
    echo "Generating report for $exp_dir"
    python generate_detailed_report.py \
        --result_dir "$exp_dir" \
        --output_dir "${exp_dir}/detailed_report"
done
```

---

## 引用与参考

如果在论文中使用本系统，建议引用以下指标：

1. **Macro-F1**: Van Asch & Daelemans (2010)
2. **Cohen's Kappa**: Cohen (1960)
3. **Soft Dice**: Milletari et al. (2016) - V-Net
4. **Risk-Coverage**: Geifman & El-Yaniv (2017) - Selective Prediction

---

## 总结

本系统提供了完整的评估框架，包括：
- ✅ 13个详细评估指标
- ✅ 统计显著性检验
- ✅ 多维度可视化
- ✅ 自动化报告生成
- ✅ HTML格式详细报告
- ✅ 科学的结果解读

**推荐工作流程**:
1. 配置排除列表
2. 验证排除逻辑
3. 运行批量实验
4. 查看对比报告
5. 生成详细报告
6. 分析结果并得出结论

祝实验顺利！🎉
