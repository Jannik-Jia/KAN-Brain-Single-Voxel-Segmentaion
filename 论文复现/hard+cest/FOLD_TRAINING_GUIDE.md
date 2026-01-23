# Fold训练指南

本指南介绍如何使用改进的fold目录结构进行单次训练和交叉验证训练。

## 更新内容

### 1. Fold子目录支持

`train_runner.py` 现在支持 `--fold-name` 参数，可以为每个fold创建独立的子目录。

**优势：**
- 每个fold的所有文件（模型、日志、指标、可视化等）都在独立的文件夹中
- 更清晰的目录结构，便于管理和查看
- 避免文件名冲突
- 便于并行运行多个fold（如果资源允许）

### 2. 目录结构示例

```
runs/leave_one_out/
├── config.txt                          # 训练配置摘要
├── cross_validation_summary.json       # 交叉验证汇总结果
├── cross_validation_results.csv        # CSV格式结果表
├── fold_1_test_sub100307/              # Fold 1
│   ├── checkpoints/
│   │   └── best.pth                    # 最优模型
│   ├── logs/
│   │   └── train.log                   # 训练日志
│   ├── figs/
│   │   ├── val_reliability_diagram.png
│   │   ├── val_risk_coverage.png
│   │   ├── val_entropy_histogram.png
│   │   └── val_sub100206_slice_*.png
│   ├── pred_3d/
│   │   ├── val_sub100206_pred_softmax_3d.npz
│   │   └── test_sub100307_pred_softmax_3d.npz
│   ├── metrics_val.json                # 验证集指标
│   ├── metrics_test.json               # 测试集指标
│   ├── confusion_val.csv               # 验证集混淆矩阵
│   ├── confusion_test.csv              # 测试集混淆矩阵
│   ├── temperature_scaling.json        # 温度缩放结果
│   └── run_summary.json                # 运行摘要
├── fold_2_test_sub100206/              # Fold 2
│   └── ...
└── fold_3_test_sub100408/              # Fold 3
    └── ...
```

## 使用方法

### 方法1: 单次训练（指定fold名称）

```bash
python train_runner.py \
    --data-root /path/to/data \
    --test-id sub100307 \
    --val-id sub100206 \
    --epochs 3 \
    --batch-size 256 \
    --lr 1e-5 \
    --save-dir runs/my_experiment \
    --fold-name fold_1_test_sub100307
```

结果将保存在 `runs/my_experiment/fold_1_test_sub100307/`

### 方法2: 单次训练（不使用fold目录）

如果不指定 `--fold-name`，行为与之前一致：

```bash
python train_runner.py \
    --data-root /path/to/data \
    --test-id sub100307 \
    --epochs 3 \
    --save-dir runs/single_run
```

结果将直接保存在 `runs/single_run/`

### 方法3: Leave-One-Out交叉验证（推荐）

使用提供的 `run_leave_one_out.sh` 脚本自动运行所有fold：

#### 步骤1: 配置脚本

编辑 `run_leave_one_out.sh`，修改以下参数：

```bash
# 数据路径
DATA_ROOT="/path/to/your/data"  # 修改为实际数据路径

# 输出目录
OUTPUT_DIR="./runs/leave_one_out"

# 训练超参数
EPOCHS=3
BATCH_SIZE=256
LR=1e-5
WEIGHT_DECAY=1e-5

# 是否使用类别权重
USE_CLASS_WEIGHTS=false  # 改为true启用
CLASS_WEIGHT_ALPHA=0.5
```

#### 步骤2: 运行脚本

```bash
./run_leave_one_out.sh
```

脚本将自动：
1. 检测数据目录中的所有被试
2. 对每个被试作为测试集运行训练
3. 为每个fold创建独立的子目录
4. 在所有训练完成后生成汇总报告

#### 步骤3: 查看结果

训练完成后，查看以下文件：

- **配置摘要**: `runs/leave_one_out/config.txt`
- **JSON汇总**: `runs/leave_one_out/cross_validation_summary.json`
- **CSV表格**: `runs/leave_one_out/cross_validation_results.csv`
- **各fold详细结果**: `runs/leave_one_out/fold_*/`

### 方法4: K-Fold交叉验证

如果需要k-fold（而非leave-one-out），可以手动指定val和test被试：

```bash
# Fold 1: test=sub1, val=sub2
python train_runner.py \
    --data-root /path/to/data \
    --test-id sub100307 \
    --val-id sub100206 \
    --save-dir runs/5fold \
    --fold-name fold_1

# Fold 2: test=sub2, val=sub3
python train_runner.py \
    --data-root /path/to/data \
    --test-id sub100206 \
    --val-id sub100408 \
    --save-dir runs/5fold \
    --fold-name fold_2

# ... 以此类推
```

## 命令行参数说明

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fold-name` | str | None | Fold名称，用于创建子目录（例如"fold_1"） |

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-root` | str | **必需** | 数据根目录（包含1d/和3d/子目录） |
| `--val-id` | str | None | 验证集被试ID（未指定时自动选择） |
| `--test-id` | str | None | 测试集被试ID（未指定时自动选择） |
| `--seed` | int | 42 | 随机种子 |
| `--epochs` | int | 3 | 训练轮数 |
| `--batch-size` | int | 256 | 批大小 |
| `--lr` | float | 1e-5 | 学习率 |
| `--weight-decay` | float | 1e-5 | 权重衰减（L2正则化） |
| `--use-class-weights` | flag | False | 是否使用类别权重 |
| `--class-weight-alpha` | float | 0.5 | 类别权重平衡系数 |
| `--max-vox-per-subject` | int | None | 每个被试最大体素数（用于控制显存） |
| `--grad-clip-norm` | float | 1.0 | 梯度裁剪范数（设为0禁用） |
| `--save-dir` | str | runs/quickstart | 保存目录 |
| `--fold-name` | str | None | Fold名称（创建子目录） |

## 类别权重使用建议

### 当前代码的问题

代码在**训练和验证阶段的loss计算中都使用了类别权重**（如果启用）。这可能不太合理，因为：

- **训练时**：使用类别权重可以平衡少数类的学习
- **验证/测试时**：不应使用类别权重，以评估模型在真实分布下的性能

### 建议

1. **默认不使用类别权重**（`USE_CLASS_WEIGHTS=false`）
2. 如果数据严重不平衡，可以尝试启用：
   ```bash
   USE_CLASS_WEIGHTS=true
   CLASS_WEIGHT_ALPHA=0.5  # 0表示不加权，1表示完全逆频率加权
   ```
3. 对比有无类别权重的结果，选择更好的配置

## 输出文件说明

### 每个fold目录包含：

1. **模型文件**
   - `checkpoints/best.pth`: 最优模型权重（基于验证集NLL）

2. **训练日志**
   - `logs/train.log`: 完整训练日志

3. **评估指标**
   - `metrics_val.json`: 验证集详细指标
   - `metrics_test.json`: 测试集详细指标
   - `metrics_val_3d_advanced.json`: 验证集3D高级指标
   - `metrics_test_3d_advanced.json`: 测试集3D高级指标
   - `temperature_scaling.json`: 温度缩放校准结果

4. **混淆矩阵**
   - `confusion_val.csv`: 验证集硬混淆矩阵
   - `confusion_test.csv`: 测试集硬混淆矩阵
   - `soft_confusion_val.csv`: 验证集软混淆矩阵
   - `soft_confusion_test.csv`: 测试集软混淆矩阵

5. **3D预测**
   - `pred_3d/val_*_pred_softmax_3d.npz`: 验证集3D概率预测
   - `pred_3d/val_*_argmax_3d.npz`: 验证集3D硬标签
   - `pred_3d/test_*_pred_softmax_3d.npz`: 测试集3D概率预测
   - `pred_3d/test_*_argmax_3d.npz`: 测试集3D硬标签

6. **可视化图表**
   - `figs/val_reliability_diagram.png`: 验证集可靠性曲线
   - `figs/val_risk_coverage.png`: 验证集Risk-Coverage曲线
   - `figs/val_entropy_histogram.png`: 验证集熵分布
   - `figs/val_*_slice_*.png`: 验证集切片对比
   - `figs/test_*.png`: 测试集对应图表

7. **运行摘要**
   - `run_summary.json`: 包含所有关键指标和配置的汇总文件
   - `split_summary.json`: 数据划分信息
   - `norm_stats.json`: 各被试的归一化统计量
   - `class_weights.npy`: 类别权重（如果启用）

### 交叉验证汇总文件：

- `cross_validation_summary.json`: JSON格式的汇总统计
- `cross_validation_results.csv`: CSV格式的结果表格
- `config.txt`: 训练配置摘要

## 常见问题

### Q1: 如何快速测试脚本？

修改 `run_leave_one_out.sh`，取消注释以下代码（只运行前3个fold）：

```bash
# 在训练循环末尾添加：
if [ ${FOLD_NUM} -eq 3 ]; then
    echo "测试模式：只运行前3个fold"
    break
fi
```

### Q2: 如何并行运行多个fold？

如果有多个GPU，可以手动启动多个训练进程，指定不同的fold_name和CUDA设备：

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python train_runner.py ... --fold-name fold_1 &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python train_runner.py ... --fold-name fold_2 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python train_runner.py ... --fold-name fold_3 &
```

### Q3: 如何查看某个fold的训练进度？

```bash
# 实时查看日志
tail -f runs/leave_one_out/fold_1_test_sub100307/logs/train.log

# 查看摘要
cat runs/leave_one_out/fold_1_test_sub100307/run_summary.json
```

### Q4: 脚本执行失败怎么办？

1. 检查数据路径是否正确：
   ```bash
   ls /path/to/data/1d/*.npz
   ls /path/to/data/3d/*.npz
   ```

2. 检查Python环境：
   ```bash
   python --version
   pip list | grep torch
   ```

3. 查看错误日志：
   ```bash
   cat runs/leave_one_out/fold_*/logs/train.log
   ```

## 下一步

训练完成后，可以：

1. 使用 `visualize_1d_3d_predictions.py` 可视化3D预测结果
2. 分析 `cross_validation_summary.json` 中的指标统计
3. 对比不同超参数配置的结果
4. 进行模型融合或集成学习

## 参考

- `TRAINING_README.md`: 原始训练说明
- `COMPATIBILITY_CHECK.md`: 数据兼容性检查
- `NOTEBOOK_VERIFICATION.md`: Notebook使用指南
