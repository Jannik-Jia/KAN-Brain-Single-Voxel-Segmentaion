# 排除配准问题数据集的对比实验

## 实验目的

对比排除不同配准问题数据集对模型训练的影响，包括：
1. 每次排除1个错误数据集（6次实验）
2. 排除所有错误数据集（1次实验）
3. 对比这7次训练的结果

## 文件说明

### 核心文件

1. **`exclude_subjects.txt`** - 配准问题数据集列表
   - 每行一个被试标识
   - 支持注释（以 `#` 开头）
   - 文件名匹配（包含即排除）

2. **`train_1d_with_3d_dataset.py`** - 修改后的训练脚本
   - 新增 `--exclude_subjects`: 排除列表文件路径
   - 新增 `--exclude_single`: 单次排除一个被试
   - 新增 `--fixed_test_subject`: 固定测试被试

3. **`run_exclude_experiments.sh`** - 批量训练脚本
   - 自动执行7次训练
   - 每次使用不同的排除策略

4. **`compare_exclude_results.py`** - 结果对比分析脚本
   - 生成对比表格
   - 绘制可视化图表
   - 输出统计分析

## 使用步骤

### 第1步：配置排除列表

编辑 `exclude_subjects.txt`，添加配准有问题的被试名：

```bash
vim exclude_subjects.txt
```

添加内容（示例）：
```
# 配准有问题的被试
YHC04
YHC2
PDP01
XXX01
YYY02
ZZZ03
```

**注意**：
- 只需要被试标识的关键字即可（如 `YHC04`）
- 系统会匹配文件名中包含这些字符串的所有文件
- 例如：`YHC04` 会匹配 `ODP_01_YHC04.mat`, `YHC04_xxx.mat` 等

### 第2步：配置训练参数

编辑 `run_exclude_experiments.sh`，修改以下配置：

```bash
vim run_exclude_experiments.sh
```

关键配置项：
```bash
# 数据路径
DATA_DIR_1D="/path/to/1D/data"      # 修改为你的1D数据路径
DATA_DIR_3D="/path/to/3D/data"      # 修改为你的3D数据路径

# 固定测试被试（重要！）
FIXED_TEST_SUBJECT="qhlazec"        # 选一个正确的被试作为测试集
                                    # 或留空使用默认的被试1
```

### 第3步：运行批量训练

赋予执行权限并运行：

```bash
chmod +x run_exclude_experiments.sh
./run_exclude_experiments.sh
```

训练过程：
```
实验1: 排除 YHC04  (38 -> 37个被试)
实验2: 排除 YHC2   (38 -> 37个被试)
实验3: 排除 PDP01  (38 -> 37个被试)
实验4: 排除 XXX01  (38 -> 37个被试)
实验5: 排除 YYY02  (38 -> 37个被试)
实验6: 排除 ZZZ03  (38 -> 37个被试)
实验7: 排除所有    (38 -> 32个被试)
```

### 第4步：查看结果

训练完成后，结果保存在：
```
results_exclude_experiments/
├── exp1_exclude_YHC04/
│   ├── dense_4x4096_model_test1.pth
│   ├── history_test1.json
│   └── predictions_3d_test1.mat
├── exp2_exclude_YHC2/
│   └── ...
├── exp3_exclude_PDP01/
│   └── ...
...
├── exp7_exclude_all/
│   └── ...
├── comparison_report.json          # JSON格式汇总报告
├── comparison_plots.png            # 4合1对比图
└── training_curves.png             # 训练曲线对比图
```

### 第5步：查看对比分析

对比结果会自动生成，包括：

1. **控制台输出**：
   - 对比表格
   - 统计分析
   - 最佳/最差结果

2. **可视化图表**：
   - `comparison_plots.png`: 4个子图
     - 最佳测试F1对比（柱状图）
     - 训练曲线对比
     - 训练Loss vs 测试Loss（散点图）
     - 过拟合分析

   - `training_curves.png`: 2个子图
     - 训练F1曲线
     - 测试Loss曲线

3. **JSON报告**：
   - `comparison_report.json`: 包含所有实验的详细数据

## 手动运行单个实验

如果需要手动运行单个实验，可以使用：

### 实验A：排除单个数据集

```bash
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results_test \
    --exclude_single YHC04 \
    --fixed_test_subject qhlazec \
    --batch_size 128 \
    --epochs 25 \
    --samples_per_subject 50000 \
    --save_predictions
```

### 实验B：排除所有列表中的数据集

```bash
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results_test \
    --exclude_subjects ./exclude_subjects.txt \
    --fixed_test_subject qhlazec \
    --batch_size 128 \
    --epochs 25 \
    --samples_per_subject 50000 \
    --save_predictions
```

### 实验C：不排除任何数据集（基线）

```bash
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results_baseline \
    --fixed_test_subject qhlazec \
    --batch_size 128 \
    --epochs 25 \
    --samples_per_subject 50000 \
    --save_predictions
```

## 手动运行对比分析

如果需要重新生成对比报告：

```bash
python compare_exclude_results.py \
    --results_dir ./results_exclude_experiments \
    --output_file ./results_exclude_experiments/comparison_report.json
```

## 新增参数说明

### `--exclude_subjects <file>`
从文件读取排除列表，排除所有匹配的被试。

**示例**：
```bash
--exclude_subjects ./exclude_subjects.txt
```

### `--exclude_single <name>`
排除单个被试（用于逐个对比实验）。

**示例**：
```bash
--exclude_single YHC04
```

### `--fixed_test_subject <name>`
固定测试被试名（所有实验使用同一个测试集）。

**示例**：
```bash
--fixed_test_subject qhlazec
```

**注意**：
- 如果不指定，使用 `--test_subject` 编号
- 使用被试名的部分匹配（如 `YHC04` 会匹配 `ODP_01_YHC04`）

## 预期结果

通过这7次对比实验，你可以分析：

1. **影响最大的数据集**：哪个配准问题数据集对结果影响最大
2. **排除效果**：排除后是否改善了模型性能
3. **累积效应**：排除所有vs排除单个的性能差异
4. **训练稳定性**：不同排除策略下的训练曲线变化

## 故障排除

### 问题1：找不到匹配的测试被试

```
ValueError: 未找到匹配的固定测试被试: xxx
```

**解决**：
- 检查 `FIXED_TEST_SUBJECT` 拼写
- 使用 `ls` 查看数据目录中的文件名
- 确保测试被试没有被排除

### 问题2：排除后被试数量不对

```
logger.info(f'剩余可用被试: XX个')
```

**检查**：
- 查看日志中的"被排除的被试"列表
- 确认 `exclude_subjects.txt` 中的名称正确
- 检查文件名匹配逻辑

### 问题3：对比脚本报错

```
未找到任何实验结果
```

**解决**：
- 确认训练已完成
- 检查 `results_exclude_experiments/` 目录下是否有 `exp*` 文件夹
- 确认每个实验目录中有 `history_test*.json` 文件

## 注意事项

1. **固定测试集**：所有实验使用同一个测试集才能公平对比
2. **被试选择**：选择一个**没有配准问题**的被试作为测试集
3. **计算时间**：7次训练需要较长时间（约25 epochs × 7 = 175 epochs）
4. **磁盘空间**：每次训练会生成约13GB的预测文件（如果开启 `--save_predictions`）

## 示例输出

### 控制台输出示例

```
===== 实验结果对比表 =====

实验                      排除被试  最佳测试F1  最终训练F1  最终测试F1  最终训练Loss  最终测试Loss
exp1_exclude_YHC04        YHC04    0.7234      0.8123      0.7198      0.4521        0.5234
exp2_exclude_YHC2         YHC2     0.7189      0.8098      0.7156      0.4567        0.5301
exp3_exclude_PDP01        PDP01    0.7312      0.8145      0.7289      0.4489        0.5178
...
exp7_exclude_all          ALL      0.7401      0.8201      0.7385      0.4312        0.5045

===== 统计分析 =====

最佳测试F1 - 平均值: 0.7256
最佳测试F1 - 标准差: 0.0078
最佳测试F1 - 最大值: 0.7401 (exp7_exclude_all)
最佳测试F1 - 最小值: 0.7189 (exp2_exclude_YHC2)

排除单个数据集的平均F1: 0.7234
排除所有数据集的F1: 0.7401
差异: +0.0167
```

## 参考

- 原始训练方法：`README_1D_TRAINING.md`
- 数据集创建：查看数据集文档
