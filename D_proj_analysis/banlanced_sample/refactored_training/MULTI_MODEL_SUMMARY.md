# 多模型训练功能 - 完成总结

## ✅ 已实现的功能

### 核心脚本

创建了 `run_multiple_models.sh` - 交互式多模型训练脚本

**主要特性**:

1. **交互式模型选择**
   - 显示所有可用模型列表
   - 支持多选（用空格分隔）
   - 支持 `all` 快速选择所有模型
   - 自动检查依赖（如KAN的fastkan）

2. **独立输出目录**
   - 每个模型有独立的输出文件夹
   - 命名格式: `模型名_bg_模式_统一时间戳`
   - 同一批次的所有模型共享相同时间戳

3. **完整结果保存**
   - ✅ 模型权重 (.pth)
   - ✅ 训练曲线 (training_history_*.png)
   - ✅ 3D Softmax (test_softmax_3d_*.nii.gz)
   - ✅ Softmax元信息 (.json)
   - ✅ Per-class详细分析目录
   - ✅ Per-class CSV数据
   - ✅ Per-class可视化图表（10+张）
   - ✅ Per-class文本报告
   - ✅ 训练日志

4. **自动汇总报告**
   - 每个模型的 `SUMMARY.txt`
   - 总的 `MASTER_SUMMARY_*.txt`
   - 包含训练参数、文件统计等

5. **环境变量配置**
   - EPOCHS: 训练轮数
   - BATCH_SIZE: 批大小
   - LEARNING_RATE: 学习率
   - INCLUDE_BACKGROUND: 背景模式
   - ROOT_DIR: 数据集路径

---

## 📁 输出目录结构

```
training_runs/
├── reg_model_bg_excl_20250102_143022/
│   ├── results/
│   │   ├── reg_model_bg_excl_*.pth
│   │   ├── training_history_*.png
│   │   ├── test_softmax_3d_*.nii.gz
│   │   ├── test_softmax_info_*.json
│   │   └── per_class_analysis_reg_model_*/
│   │       ├── per_class_detailed_metrics.csv
│   │       ├── per_class_complete_analysis.json
│   │       ├── per_class_summary_report.txt
│   │       ├── comprehensive_per_class_analysis.png
│   │       ├── top_bottom_performers_detailed.png
│   │       ├── detailed_performance_ranking_analysis.png
│   │       ├── detailed_correlation_analysis.png
│   │       ├── support_distribution_analysis.png
│   │       ├── f1_score_detailed_comparison.png
│   │       ├── dice_coefficient_detailed_comparison.png
│   │       ├── precision_detailed_comparison.png
│   │       ├── recall_detailed_comparison.png
│   │       ├── specificity_detailed_comparison.png
│   │       ├── iou_jaccard_detailed_comparison.png
│   │       └── balanced_accuracy_detailed_comparison.png
│   ├── logs/
│   │   └── training_reg_model_*.log
│   └── SUMMARY.txt
│
├── resnet_mlp_bg_excl_20250102_143022/
│   └── ... (相同结构)
│
├── simple_mlp_bg_excl_20250102_143022/
│   └── ... (相同结构)
│
├── kan_bg_excl_20250102_143022/
│   └── ... (相同结构)
│
└── MASTER_SUMMARY_20250102_143022.txt
```

---

## 🚀 使用方法

### 基础使用

```bash
./run_multiple_models.sh
```

脚本会提示：
```
可用模型:
  [1] reg_model - 深度全连接神经网络（Alex identical）
  [2] resnet_mlp - 带残差连接的全连接网络
  [3] simple_mlp - 简单多层感知机（轻量级）
  [4] kan - KAN模型（自动类权重）✓

请选择要训练的模型（可多选）:
输入格式: 用空格分隔，例如 '1 2 4' 表示训练 reg_model, resnet_mlp, 和 kan
输入 'all' 训练所有模型
您的选择:
```

### 快速示例

```bash
# 训练所有模型（默认25 epochs）
./run_multiple_models.sh
# 输入: all

# 快速测试（10 epochs）
EPOCHS=10 ./run_multiple_models.sh
# 输入: 3 4  (SimpleMLP和KAN)

# 对比KAN和RegModel
./run_multiple_models.sh
# 输入: 1 4

# 自定义参数
EPOCHS=30 BATCH_SIZE=4096 LEARNING_RATE=0.0001 ./run_multiple_models.sh
# 输入: 1 2 4
```

---

## 📊 与原始流程对比

### 保存内容 - 100%一致

| 文件类型 | 原始流程 | 多模型脚本 |
|----------|----------|------------|
| 模型权重 | ✅ | ✅ |
| 训练曲线 | ✅ | ✅ |
| 3D Softmax | ✅ | ✅ |
| Softmax Info | ✅ | ✅ |
| Per-class CSV | ✅ | ✅ |
| Per-class 可视化 | ✅ (所有图表) | ✅ (所有图表) |
| Per-class 报告 | ✅ | ✅ |
| 训练日志 | ✅ | ✅ |

### 额外功能

| 功能 | 原始流程 | 多模型脚本 |
|------|----------|------------|
| 交互式选择 | ❌ | ✅ |
| 多模型训练 | ❌ | ✅ |
| 独立输出目录 | ❌ | ✅ |
| 自动汇总 | ❌ | ✅ |
| 批次管理 | ❌ | ✅ |
| 失败处理 | ❌ | ✅ |

---

## 📝 创建的文件

### 主要脚本

1. **run_multiple_models.sh** (核心脚本)
   - 交互式多模型训练
   - 独立输出目录管理
   - 自动汇总报告
   - 约500行代码

2. **quick_start_multi.sh** (快速开始指南)
   - 使用示例展示
   - 命令说明
   - 快速参考

### 文档

3. **MULTI_MODEL_GUIDE.md** (详细使用指南)
   - 完整的功能说明
   - 使用场景示例
   - 故障排除

4. **MULTI_MODEL_SUMMARY.md** (本文档)
   - 功能总结
   - 对比说明

5. **README.md** (已更新)
   - 添加了多模型训练部分
   - 更新了快速开始

---

## 💡 实用场景

### 场景1: 模型性能对比

```bash
# 一次性训练所有模型进行对比
./run_multiple_models.sh
# 输入: all

# 训练完成后对比:
# - training_runs/reg_model_*/results/training_history_*.png
# - training_runs/kan_*/results/training_history_*.png
# - ...
```

### 场景2: 快速验证

```bash
# 用轻量级模型快速验证数据和代码
EPOCHS=5 ./run_multiple_models.sh
# 输入: 3  (SimpleMLP)
```

### 场景3: KAN与传统模型对比

```bash
# 对比KAN的类权重效果
./run_multiple_models.sh
# 输入: 1 4  (RegModel vs KAN)

# 查看per-class分析，重点关注少数类的性能
```

### 场景4: 参数搜索

```bash
# 实验1: 默认参数
./run_multiple_models.sh
# 输入: 1 4

# 实验2: 大学习率
LEARNING_RATE=0.0001 ./run_multiple_models.sh
# 输入: 1 4

# 对比两组结果
```

---

## 🎯 关键特性

### 1. 时间戳管理

所有在同一批次训练的模型共享相同的 `MASTER_TIMESTAMP`：

```
reg_model_bg_excl_20250102_143022/
simple_mlp_bg_excl_20250102_143022/
kan_bg_excl_20250102_143022/
MASTER_SUMMARY_20250102_143022.txt
```

便于识别和对比同一批次的模型。

### 2. 独立输出

每个模型有完全独立的输出目录，包括：
- 独立的 `results/` 子目录
- 独立的 `logs/` 子目录
- 独立的 `SUMMARY.txt`

### 3. 符号链接技术

训练时临时创建符号链接：
```bash
ln -sf training_runs/kan_bg_excl_*/results ./results
# 训练...
# 完成后移除链接
```

确保train.py写入正确的位置。

### 4. 进度显示

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
进度: [2/3]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 开始训练: simple_mlp - 排除背景
```

### 5. 环境检查

训练前自动检查：
- ✅ Python版本
- ✅ PyTorch
- ✅ nibabel
- ✅ sklearn
- ✅ fastkan (可选，KAN需要)

---

## 📈 性能统计

### 文件数量

每个模型的输出文件数：
- 模型权重: 1个 (.pth)
- 训练曲线: 1个 (.png)
- 3D Softmax: 1个 (.nii.gz)
- Softmax Info: 1个 (.json)
- Per-class CSV: 1个
- Per-class JSON: 1个
- Per-class 文本报告: 1个
- Per-class 可视化: 12-15个 (.png)
- 训练日志: 1个 (.log)
- 汇总报告: 1个 (SUMMARY.txt)

**总计**: 约20-25个文件/模型

### 磁盘空间

单个模型的预估空间：
- RegModel: ~500MB (大模型)
- ResNetMLP: ~200MB
- SimpleMLP: ~50MB
- KAN: ~100-200MB (取决于grid_size)

4个模型总计: ~1GB

---

## 🔧 技术实现

### 核心逻辑

1. **模型选择**: 数组存储，循环遍历
2. **目录管理**: 符号链接+备份恢复
3. **日志重定向**: tee命令同时输出到文件和终端
4. **错误处理**: set -e + 条件检查
5. **时间戳**: 统一的MASTER_TIMESTAMP

### 关键代码片段

```bash
# 创建符号链接指向模型专用目录
ln -sf "$temp_results_dir" "$old_results_dir"

# 运行训练
eval "$cmd" 2>&1 | tee "$log_file"

# 恢复原始目录
rm "$old_results_dir"
```

---

## 🎓 最佳实践

### 1. 分阶段训练

```bash
# 第一阶段：快速验证（5 epochs）
EPOCHS=5 ./run_multiple_models.sh
# 输入: all

# 第二阶段：完整训练（25 epochs）
EPOCHS=25 ./run_multiple_models.sh
# 输入: 1 4  (选择表现好的模型)
```

### 2. 后台运行

```bash
nohup ./run_multiple_models.sh > multi_train.log 2>&1 &

# 查看进度
tail -f multi_train.log

# 检查进程
ps aux | grep run_multiple_models
```

### 3. 结果对比

使用Python对比CSV：

```python
import pandas as pd
import glob

# 找到所有per-class CSV
csv_files = glob.glob('training_runs/*/results/per_class_analysis_*/per_class_detailed_metrics.csv')

# 加载并对比
models = {}
for csv_file in csv_files:
    model_name = csv_file.split('/')[1].split('_bg_')[0]
    models[model_name] = pd.read_csv(csv_file)

# 对比F1分数
for model_name, df in models.items():
    print(f"{model_name}: Mean F1 = {df['f1_score'].mean():.4f}")
```

---

## 📚 相关文档

- **MULTI_MODEL_GUIDE.md**: 详细使用指南
- **README.md**: 完整系统文档
- **KAN_SETUP.md**: KAN模型设置
- **INTEGRATION_SUMMARY.md**: KAN集成总结
- **quick_start_multi.sh**: 快速示例

---

## 🎉 总结

### 实现的核心价值

1. **效率**: 一次运行，多个模型
2. **组织**: 独立目录，清晰管理
3. **完整**: 保存所有训练结果
4. **便捷**: 交互式选择，简单易用
5. **兼容**: 100%保持原有输出格式

### 适用对象

- ✅ 需要对比多个模型的研究人员
- ✅ 进行模型选择的工程师
- ✅ 希望批量训练的用户
- ✅ 需要完整记录的实验

### 开始使用

```bash
cd refactored_training
./run_multiple_models.sh
```

就是这么简单！🚀

---

**创建时间**: 2025-01-02
**版本**: 1.0
**维护者**: Training System Team
