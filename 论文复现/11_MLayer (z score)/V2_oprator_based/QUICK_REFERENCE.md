# 快速参考卡片

## 🚀 快速开始（3步）

```bash
# 1. 编辑排除列表
vim exclude_subjects.txt
# 添加你的6个问题数据集名称

# 2. 修改数据路径
vim run_exclude_experiments.sh
# 修改 DATA_DIR_1D 和 DATA_DIR_3D

# 3. 运行
./run_exclude_experiments.sh
```

## 📊 13个评估指标速查

| 类别 | 指标 | 范围 | 越X越好 | 用途 |
|-----|------|-----|---------|------|
| **准确性** | Gross Accuracy | [0,1] | 高 | 基线对比 |
| | Top-1 Accuracy | [0,1] | 高 | 第1名正确率 |
| | Top-3 Accuracy | [0,1] | 高 | 前3名覆盖率 |
| | Top-5 Accuracy | [0,1] | 高 | 前5名覆盖率 |
| **平衡** | Macro-F1 | [0,1] | 高 | **核心指标** |
| | Balanced Accuracy | [0,1] | 高 | 类别平衡 |
| | Weighted F1 | [0,1] | 高 | 样本加权 |
| **一致性** | Cohen's Kappa | [-1,1] | 高 | 预测可靠性 |
| **分割** | Macro Soft Dice | [0,1] | 高 | 分割质量 |
| **风险** | Risk@95% Cov | [0,1] | 低 | 错误率 |

## 🎯 指标选择指南

| 研究目标 | 主要指标 | 次要指标 |
|---------|---------|---------|
| 模型选择 | Macro-F1 | Balanced Acc |
| 临床应用 | Risk@95% | Balanced Acc, Kappa |
| 算法对比 | Gross Acc, Top-3 | Kappa, Soft Dice |
| 分割评估 | Soft Dice | Macro-F1 |

## 📈 结果解读

### Kappa解释
- κ > 0.8: 优秀（Excellent）
- 0.6 < κ ≤ 0.8: 良好（Good）
- 0.4 < κ ≤ 0.6: 中等（Moderate）
- κ ≤ 0.4: 一般（Fair/Poor）

### Top-K解释（102类分类）
- Top-3 > 85%: 模型置信度高
- Top-5 > 90%: 模型覆盖良好
- Top-3 - Top-1 > 15%: 存在混淆类别

### 统计显著性
- p < 0.05: 显著 (***)
- p < 0.01: 非常显著 (****)
- p ≥ 0.05: 不显著 (ns)

## 🛠️ 常用命令

```bash
# 验证排除逻辑
python check_exclude_logic.py \
    --data_dir_1d /path/to/1d/data \
    --exclude_file ./exclude_subjects.txt

# 单次训练（排除单个）
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results \
    --exclude_single YHC04 \
    --fixed_test_subject qhlazec \
    --epochs 25 \
    --save_predictions

# 单次训练（排除所有）
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results \
    --exclude_subjects ./exclude_subjects.txt \
    --fixed_test_subject qhlazec \
    --epochs 25 \
    --save_predictions

# 生成对比报告
python compare_exclude_results_extended.py \
    --results_dir ./results_exclude_experiments \
    --output_file ./comparison_report.json

# 生成详细报告（单个实验）
python generate_detailed_report.py \
    --result_dir ./results_exclude_experiments/exp1_exclude_YHC04 \
    --output_dir ./detailed_report
```

## 📁 关键文件

| 文件 | 说明 |
|-----|------|
| `evaluation_metrics.py` | 评估指标模块 |
| `exclude_subjects.txt` | 排除列表（需编辑） |
| `run_exclude_experiments.sh` | 批量实验脚本 |
| `comparison_report_extended.json` | 对比报告 |
| `detailed_report.html` | 详细HTML报告 |

## 🐛 常见问题

### Q: 如何指定测试被试？
```bash
# 方法1: 使用固定名称（推荐）
--fixed_test_subject qhlazec

# 方法2: 使用索引
--test_subject 1
```

### Q: 内存不足怎么办？
```bash
# 减小batch size
--batch_size 64

# 减少每个被试的采样
--samples_per_subject 25000
```

### Q: 如何只计算指标不训练？
```bash
python train_1d_with_3d_dataset.py \
    --load_model ./model.pth \
    --predict_only \
    --save_predictions \
    ...
```

### Q: 旧模型如何计算新指标？
A: 需要重新训练。新指标需要softmax概率，旧模型可能没有保存。

## 📞 获取帮助

```bash
# 查看完整使用指南
cat COMPLETE_USAGE_GUIDE.md

# 查看指标更新说明
cat METRICS_UPDATE_SUMMARY.md

# 查看排除实验说明
cat README_EXCLUDE_EXPERIMENTS.md
```

## 🎓 学习资源

1. **评估指标理论**: `COMPLETE_USAGE_GUIDE.md` → "科学解释与应用"
2. **指标计算细节**: `evaluation_metrics.py` → 函数文档
3. **统计检验**: `compare_exclude_results_extended.py` → `compute_statistical_tests()`
4. **可视化**: `generate_detailed_report.py` → 绘图函数

---

**提示**: 这是快速参考。完整文档请查看 `COMPLETE_USAGE_GUIDE.md`
