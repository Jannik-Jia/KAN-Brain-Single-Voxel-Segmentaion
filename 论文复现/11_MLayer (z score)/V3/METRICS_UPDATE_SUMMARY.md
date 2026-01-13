# 评估指标系统更新总结

## 新增的评估指标

###  1. 准确性指标 ✅
- **Gross Accuracy (GC)**: 总体准确率
- **Top-1/Top-3/Top-5 Accuracy**: Top-K准确率，评估Top-K覆盖率

### 2. 类别平衡指标 ✅
- **Macro-F1**: 宏平均F1（已有，继续作为核心指标）
- **Balanced Accuracy**: 平衡准确率，考虑类别不平衡
- **Weighted F1**: 加权F1分数

### 3. 一致性指标 ✅
- **Cohen's κ (Kappa)**: Cohen's Kappa系数，衡量预测与真实标签的一致性

### 4. 分割质量指标 ✅
- **Macro 3D Soft Dice**: 宏平均软Dice系数

### 5. 风险分析 ✅
- **Risk & Coverage**: 风险-覆盖率曲线
- **Risk@95% Coverage**: 95%覆盖率下的风险值

## 文件修改说明

### 1. 新增文件

#### `evaluation_metrics.py`
**功能**: 完整的评估指标模块

**包含函数**:
```python
# 准确性指标
- compute_gross_accuracy()
- compute_top_k_accuracy()
- compute_top_k_accuracies()  # 计算Top-1/3/5

# 类别平衡指标
- compute_macro_f1()
- compute_weighted_f1()
- compute_balanced_accuracy()

# 一致性指标
- compute_cohen_kappa()

# 分割质量指标
- compute_soft_dice_per_class()
- compute_macro_soft_dice()
- compute_3d_soft_dice()

# 风险分析
- compute_risk_coverage_curve()
- compute_risk_at_coverage()

# 综合评估
- compute_all_metrics()  # 一次性计算所有指标
- format_metrics()  # 格式化输出
```

### 2. 修改文件

#### `train_1d_with_3d_dataset.py`
**修改内容**:

1. **导入评估模块**:
```python
from evaluation_metrics import compute_all_metrics
```

2. **扩展训练历史**:
```python
self.history = {
    'train_loss': [], 'test_loss': [],
    'train_f1': [], 'test_f1': [],
    'train_metrics': [],  # 新增：完整训练指标
    'test_metrics': []     # 新增：完整测试指标
}
```

3. **修改 `train_epoch()` 方法**:
- 新增参数 `compute_full_metrics`
- 收集softmax概率用于计算扩展指标
- 返回值从 `(loss, f1)` 扩展为 `(loss, f1, metrics)`

4. **修改 `evaluate()` 方法**:
- 新增参数 `compute_full_metrics`
- 收集softmax概率
- 返回完整评估指标

5. **修改 `train()` 方法**:
- 在最后一个epoch计算完整指标
- 在最佳模型上重新评估所有指标
- 保存 `best_test_metrics` 到历史记录
- 打印详细的评估指标

**输出示例**:
```
=== 最终评估指标 ===
Gross Accuracy: 0.7234
Top-1/3/5 Accuracy: 0.7234 / 0.8567 / 0.9123
Balanced Accuracy: 0.7189
Weighted F1: 0.7456
Cohen's Kappa: 0.6892
Macro Soft Dice: 0.7312
Risk@95% Coverage: 0.2456

===== 最佳模型的完整评估 =====
Best Test F1: 0.7312
Gross Accuracy: 0.7234
...
```

#### `compare_exclude_results.py`
**修改内容**:

1. **`load_experiment_results()` 函数**:
- 从history文件中提取 `best_test_metrics`
- 解析所有新指标并添加到结果字典

2. **`generate_comparison_table()` 函数**:
- 创建两个表格：基础指标表 + 扩展指标表
- 添加扩展指标的统计分析
- 对比排除单个vs排除所有的扩展指标差异

**输出示例**:
```
====================================================================================================
实验结果对比表 - 基础指标
====================================================================================================

实验                      排除被试  最佳F1  训练F1  测试F1  训练Loss  测试Loss
exp1_exclude_YHC04        YHC04    0.7234  0.8123  0.7198  0.4521    0.5234
exp2_exclude_YHC2         YHC2     0.7189  0.8098  0.7156  0.4567    0.5301
...

====================================================================================================
实验结果对比表 - 扩展评估指标
====================================================================================================

实验                      排除被试  Gross Acc  Top-1   Top-3   Top-5   Bal. Acc  W-F1    Kappa   Soft Dice  Risk@95
exp1_exclude_YHC04        YHC04    0.7234     0.7234  0.8567  0.9123  0.7189    0.7456  0.6892  0.7312     0.2456
exp2_exclude_YHC2         YHC2     0.7189     0.7189  0.8523  0.9089  0.7145    0.7412  0.6834  0.7278     0.2501
...

====================================================================================================
统计分析
====================================================================================================

【基础指标】
  最佳测试F1 - 平均: 0.7256 ± 0.0078
  最佳测试F1 - 范围: [0.7189, 0.7401]
  最好结果: ALL (F1=0.7401)
  最差结果: YHC2 (F1=0.7189)

【扩展指标】
  Gross Accuracy: 0.7234 ± 0.0082 (range: [0.7156, 0.7389])
  Top-1 Accuracy: 0.7234 ± 0.0082 (range: [0.7156, 0.7389])
  Top-3 Accuracy: 0.8545 ± 0.0065 (range: [0.8489, 0.8623])
  Top-5 Accuracy: 0.9098 ± 0.0045 (range: [0.9045, 0.9156])
  Balanced Accuracy: 0.7189 ± 0.0078 (range: [0.7112, 0.7345])
  Weighted F1: 0.7432 ± 0.0085 (range: [0.7367, 0.7589])
  Cohen's Kappa: 0.6867 ± 0.0092 (range: [0.6789, 0.7012])
  Macro Soft Dice: 0.7289 ± 0.0089 (range: [0.7198, 0.7456])
  Risk@95% Coverage: 0.2478 ± 0.0056 (range: [0.2398, 0.2567])

【排除策略对比】
  排除单个数据集 - 平均F1: 0.7234
  排除所有数据集 - F1: 0.7401
  差异: +0.0167

  扩展指标对比:
    Gross Acc: 0.7234 (single) vs 0.7389 (all) | diff: +0.0155
    Bal. Acc: 0.7189 (single) vs 0.7345 (all) | diff: +0.0156
    Soft Dice: 0.7289 (single) vs 0.7456 (all) | diff: +0.0167
    Kappa: 0.6867 (single) vs 0.7012 (all) | diff: +0.0145
```

## 使用方法

### 1. 运行训练（自动计算所有指标）

```bash
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results \
    --epochs 25 \
    --save_predictions
```

训练脚本会自动：
- 在最后一个epoch计算完整指标
- 在最佳模型上重新评估所有指标
- 保存所有指标到 `history.json`

### 2. 运行排除实验（包含所有指标）

```bash
./run_exclude_experiments.sh
```

会自动：
- 运行7次训练
- 每次训练计算完整指标
- 生成包含扩展指标的对比报告

### 3. 手动对比分析

```bash
python compare_exclude_results.py \
    --results_dir ./results_exclude_experiments \
    --output_file ./comparison_report.json
```

## 指标说明

### Top-K准确率的意义
- **Top-1**: 预测的第1名是否正确（等同于Gross Accuracy）
- **Top-3**: 预测的前3名中是否包含正确答案
- **Top-5**: 预测的前5名中是否包含正确答案

在102类脑区域分类中：
- Top-3 ≈ 85-90% 表示模型较有信心
- Top-5 ≈ 90-95% 表示模型覆盖良好

### Balanced Accuracy vs Gross Accuracy
- **Gross Accuracy**: 所有样本的准确率（受类别不平衡影响）
- **Balanced Accuracy**: 每个类别准确率的平均（不受类别不平衡影响）

如果两者差异大，说明存在类别不平衡问题。

### Cohen's Kappa
- **范围**: [-1, 1]
- **解释**:
  - κ > 0.8: 几乎完全一致
  - 0.6 < κ ≤ 0.8: 高度一致
  - 0.4 < κ ≤ 0.6: 中度一致
  - κ ≤ 0.4: 一致性较差

### Macro Soft Dice
- **范围**: [0, 1]
- 使用softmax概率计算，比hard Dice更稳定
- 宏平均确保每个类别权重相同

### Risk & Coverage
- **Risk**: 被覆盖样本中的错误率
- **Coverage**: 保留的样本比例
- **Risk@95%**: 覆盖95%样本时的错误率

低风险+高覆盖率 = 高质量模型

## 兼容性说明

### 向后兼容
- 旧版本的history文件仍然可以读取
- 如果没有扩展指标，会显示 "N/A"
- 基础指标（F1, Loss）保持不变

### 性能影响
- 扩展指标仅在最后一个epoch和最佳模型上计算
- 对训练速度影响很小（< 5%）
- Top-K计算需要额外内存存储概率

## 故障排除

### 问题1: 导入错误
```
ImportError: cannot import name 'compute_all_metrics'
```

**解决**: 确保 `evaluation_metrics.py` 与训练脚本在同一目录

### 问题2: 内存不足
```
CUDA out of memory when computing metrics
```

**解决**: 在评估时概率数据会占用额外内存。可以：
- 减少batch size
- 在CPU上计算指标（自动处理）

### 问题3: 旧版本history文件
```
KeyError: 'best_test_metrics'
```

**解决**: 这是正常的，对比脚本会自动处理，显示 "N/A"

## 下一步

建议的后续工作：
1. 添加风险-覆盖率曲线可视化
2. 添加混淆矩阵分析
3. 添加类别级别的Dice可视化
4. 生成PDF格式的详细报告

## 参考

- Gross Accuracy: 基线对比标准
- Top-K: 评估Top-K覆盖率
- Macro-F1: 核心指标，用于Checkpoint选择
- Balanced Accuracy & Weighted F1: 类别平衡评估
- Cohen's κ: 一致性评估
- Macro 3D Soft Dice: 分割质量评估
- Risk & Coverage: 风险分析
