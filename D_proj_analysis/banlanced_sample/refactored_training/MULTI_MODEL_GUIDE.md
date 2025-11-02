# 多模型训练指南

## 概述

`run_multiple_models.sh` 是一个交互式脚本，允许您一次性选择并训练多个模型，每个模型的结果自动保存到独立的文件夹中。

---

## 🚀 快速开始

### 基础使用

```bash
cd refactored_training
./run_multiple_models.sh
```

### 自定义参数

```bash
# 自定义训练参数
EPOCHS=30 BATCH_SIZE=4096 ./run_multiple_models.sh

# 包含背景训练
INCLUDE_BACKGROUND=true ./run_multiple_models.sh

# 完整自定义
EPOCHS=25 \
BATCH_SIZE=8192 \
LEARNING_RATE=0.0001 \
INCLUDE_BACKGROUND=false \
./run_multiple_models.sh
```

---

## 📋 使用流程

### 1. 启动脚本

```bash
./run_multiple_models.sh
```

### 2. 查看可用模型

脚本会显示4个可用模型：

```
可用模型:
  [1] reg_model - 深度全连接神经网络（Alex identical）
  [2] resnet_mlp - 带残差连接的全连接网络
  [3] simple_mlp - 简单多层感知机（轻量级）
  [4] kan - KAN模型（自动类权重）✓
```

### 3. 选择模型

**方式A - 选择特定模型**:
```
您的选择: 1 3 4
```
这将训练 reg_model, simple_mlp, 和 kan

**方式B - 训练所有模型**:
```
您的选择: all
```

### 4. 确认训练

脚本会显示训练计划：
```
将训练以下 3 个模型:
  1. reg_model
  2. simple_mlp
  3. kan

确认开始训练？(y/n):
```

### 5. 等待完成

脚本会按顺序训练每个模型，显示进度和实时日志。

---

## 📁 输出结构

### 目录组织

所有训练结果保存在 `training_runs/` 目录下：

```
training_runs/
├── reg_model_bg_excl_20250102_143022/
│   ├── results/
│   │   ├── reg_model_bg_excl_*.pth                    # 模型权重
│   │   ├── training_history_reg_model_*.png           # 训练曲线
│   │   ├── test_softmax_3d_*_bg_excl_*.nii.gz        # 3D Softmax
│   │   ├── test_softmax_info_*.json                   # Softmax信息
│   │   └── per_class_analysis_reg_model_*/            # Per-class分析
│   │       ├── per_class_detailed_metrics.csv
│   │       ├── comprehensive_per_class_analysis.png
│   │       ├── detailed_performance_ranking_analysis.png
│   │       └── ... (更多可视化)
│   ├── logs/
│   │   └── training_reg_model_*.log                   # 训练日志
│   └── SUMMARY.txt                                     # 汇总报告
│
├── simple_mlp_bg_excl_20250102_143022/
│   └── ... (相同结构)
│
├── kan_bg_excl_20250102_143022/
│   └── ... (相同结构)
│
└── MASTER_SUMMARY_20250102_143022.txt                  # 总汇总报告
```

### 保存的文件内容

每个模型目录包含：

#### 1. 模型权重 (`.pth`)
- 训练好的模型参数
- 包含配置、历史、scaler信息

#### 2. 训练曲线 (`.png`)
- Loss曲线（训练+测试）
- Accuracy曲线
- F1 Score曲线
- 综合性能图

#### 3. 3D Softmax (`.nii.gz`)
- 测试集的完整3D softmax概率分布
- 可用于可视化和后续分析
- 配套的JSON元信息文件

#### 4. Per-class分析目录
**详细指标** (`per_class_detailed_metrics.csv`):
- 每个类别的precision, recall, F1等指标
- CSV格式，可在Excel中打开

**可视化图表**:
- `comprehensive_per_class_analysis.png` - 综合分析（8个子图）
- `detailed_performance_ranking_analysis.png` - 性能排名
- `detailed_correlation_analysis.png` - 相关性分析
- `support_distribution_analysis.png` - 样本分布
- `f1_score_detailed_comparison.png` - F1详细对比
- `dice_coefficient_detailed_comparison.png` - Dice详细对比
- `precision_detailed_comparison.png` - Precision对比
- `recall_detailed_comparison.png` - Recall对比
- `specificity_detailed_comparison.png` - Specificity对比
- `iou_jaccard_detailed_comparison.png` - IoU对比
- `balanced_accuracy_detailed_comparison.png` - Balanced Acc对比

**文本报告** (`per_class_summary_report.txt`):
- 人类友好的分析报告
- Top/Bottom performers
- 性能分级统计

#### 5. 训练日志 (`.log`)
- 完整的训练过程日志
- 包括数据加载、训练进度、指标等

#### 6. 汇总报告 (`SUMMARY.txt`)
- 训练参数
- 文件统计
- 快速概览

---

## ⚙️ 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| EPOCHS | 25 | 训练轮数 |
| BATCH_SIZE | 8192 | 批大小 |
| LEARNING_RATE | 0.00001 | 学习率 |
| INCLUDE_BACKGROUND | false | 是否包含背景 |
| ROOT_DIR | (默认路径) | 数据集根目录 |

### 使用示例

```bash
# 快速测试（10 epochs）
EPOCHS=10 ./run_multiple_models.sh

# 大batch训练
BATCH_SIZE=16384 ./run_multiple_models.sh

# 包含背景的完整训练
INCLUDE_BACKGROUND=true EPOCHS=30 ./run_multiple_models.sh

# 自定义学习率
LEARNING_RATE=0.0001 ./run_multiple_models.sh

# 组合配置
EPOCHS=25 \
BATCH_SIZE=8192 \
LEARNING_RATE=0.00001 \
INCLUDE_BACKGROUND=false \
./run_multiple_models.sh
```

---

## 📊 实用场景

### 场景1: 模型对比实验

训练所有模型进行对比：

```bash
# 一次性训练所有模型
./run_multiple_models.sh
# 输入: all
```

然后对比 `training_runs/` 下各模型的：
- 训练曲线
- 最终F1分数
- Per-class性能

### 场景2: 快速验证

只训练轻量级模型快速验证：

```bash
EPOCHS=5 ./run_multiple_models.sh
# 输入: 3 (simple_mlp)
```

### 场景3: KAN vs 传统模型

对比KAN和传统模型：

```bash
./run_multiple_models.sh
# 输入: 1 4 (reg_model和kan)
```

查看KAN的类权重是否带来性能提升。

### 场景4: 参数搜索

不同参数的实验：

```bash
# 实验1: 小学习率
LEARNING_RATE=0.00001 ./run_multiple_models.sh
# 选择: 1 4

# 实验2: 大学习率
LEARNING_RATE=0.0001 ./run_multiple_models.sh
# 选择: 1 4
```

对比两组结果。

---

## 🔍 查看结果

### 方式1: 查看汇总报告

```bash
# 总汇总
cat training_runs/MASTER_SUMMARY_*.txt

# 单个模型汇总
cat training_runs/kan_bg_excl_*/SUMMARY.txt
```

### 方式2: 查看训练曲线

```bash
# 使用文件浏览器打开
open training_runs/kan_bg_excl_*/results/training_history_*.png
```

### 方式3: 查看Per-class分析

```bash
# 打开分析目录
open training_runs/kan_bg_excl_*/results/per_class_analysis_*/
```

### 方式4: 查看训练日志

```bash
# 实时查看（如果还在训练）
tail -f training_runs/kan_bg_excl_*/logs/training_*.log

# 查看完整日志
less training_runs/kan_bg_excl_*/logs/training_*.log
```

### 方式5: 对比CSV数据

使用Excel或Python对比不同模型的per-class指标：

```python
import pandas as pd

# 加载不同模型的CSV
reg_df = pd.read_csv('training_runs/reg_model_bg_excl_*/results/per_class_analysis_*/per_class_detailed_metrics.csv')
kan_df = pd.read_csv('training_runs/kan_bg_excl_*/results/per_class_analysis_*/per_class_detailed_metrics.csv')

# 对比F1分数
comparison = pd.DataFrame({
    'class_id': reg_df['class_id'],
    'reg_f1': reg_df['f1_score'],
    'kan_f1': kan_df['f1_score'],
    'improvement': kan_df['f1_score'] - reg_df['f1_score']
})

print(comparison.sort_values('improvement', ascending=False).head(10))
```

---

## 💡 最佳实践

### 1. 分阶段训练

**第一阶段 - 快速验证**:
```bash
EPOCHS=5 ./run_multiple_models.sh
# 选择: 3 4 (SimpleMLP和KAN快速测试)
```

**第二阶段 - 完整训练**:
```bash
EPOCHS=25 ./run_multiple_models.sh
# 选择: 1 2 4 (排除SimpleMLP)
```

### 2. 使用后台运行

对于长时间训练：

```bash
# 使用nohup后台运行
nohup ./run_multiple_models.sh > multi_train.log 2>&1 &

# 查看进度
tail -f multi_train.log
```

### 3. 训练前检查

运行测试确保环境正常：

```bash
# 测试KAN
python test_kan.py

# 检查数据
ls $ROOT_DIR/FOR_*/balanced_output/
```

### 4. 定期备份

```bash
# 训练完成后备份结果
tar -czf training_results_$(date +%Y%m%d).tar.gz training_runs/
```

---

## 🐛 常见问题

### Q1: 训练中断了怎么办？

**A**: 脚本会询问是否继续训练下一个模型。如果完全中断，已训练的模型结果仍然保存在 `training_runs/` 中。

### Q2: 如何只重新训练失败的模型？

**A**: 再次运行脚本，只选择失败的模型：

```bash
./run_multiple_models.sh
# 输入: 2 (假设ResNetMLP失败了)
```

### Q3: 磁盘空间不足怎么办？

**A**:
- 训练较少的模型
- 减少epochs
- 训练完成后删除不需要的结果
- 压缩旧的训练结果

### Q4: 如何对比不同时间戳的训练？

**A**: 所有时间戳相同批次的训练共享同一个MASTER_TIMESTAMP，便于对比：

```bash
# 查看同一批次的所有模型
ls -la training_runs/*_20250102_143022/
```

### Q5: 可以中途修改配置吗？

**A**: 不建议。如需不同配置，运行新的训练批次：

```bash
# 第一批次 - 默认配置
./run_multiple_models.sh

# 第二批次 - 大batch
BATCH_SIZE=16384 ./run_multiple_models.sh
```

---

## 📝 与原始流程的对比

### 保存内容完全一致

✅ 模型权重 (.pth)
✅ 训练曲线 (training_history_*.png)
✅ 3D Softmax (test_softmax_3d_*.nii.gz)
✅ Softmax元信息 (.json)
✅ Per-class详细分析目录
✅ Per-class CSV数据
✅ Per-class可视化图表
✅ Per-class文本报告
✅ 训练日志

### 额外功能

✅ 交互式模型选择
✅ 独立的输出目录
✅ 自动汇总报告
✅ 批次总汇总
✅ 训练进度显示
✅ 失败处理机制

---

## 📚 相关文档

- **README.md**: 完整系统文档
- **KAN_SETUP.md**: KAN模型设置
- **INTEGRATION_SUMMARY.md**: KAN集成总结
- **example_usage.sh**: 单模型使用示例

---

## 🎉 总结

`run_multiple_models.sh` 提供了：

1. **便捷性**: 一次性训练多个模型
2. **组织性**: 每个模型独立目录
3. **完整性**: 保存所有训练结果
4. **可比性**: 相同时间戳便于对比
5. **灵活性**: 环境变量自定义配置

开始使用：
```bash
./run_multiple_models.sh
```

祝实验顺利！🚀
