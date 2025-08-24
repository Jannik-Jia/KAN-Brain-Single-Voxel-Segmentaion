# 3×3 和 7×7 基线模型训练

本目录包含用于训练和评估3×3和7×7 patch基线模型的完整代码。

## 文件说明

- `train_baseline_3x3_7x7.py` - 主训练脚本，支持3×3和7×7 patch
- `run_leave_one_out.sh` - Leave-one-out批量训练脚本
- `analyze_results.py` - 结果分析和可视化脚本
- `test_data_loading.py` - 数据加载测试脚本

## 模型架构

基于您提供的`ImprovedConv2D_Baseline`设计：

1. **通道混合层** (kernel=1): 351→128通道，保持空间尺寸
2. **空间聚合层** (kernel=3或7): 将patch收缩到1×1
3. **SE注意力模块**: 通道注意力机制
4. **分类头**: 128→102类

主要特点：
- 直接在2D切片上提取patch（xy平面，z固定）
- 使用GroupNorm代替BatchNorm
- 使用SiLU激活函数
- 可选的SE通道注意力
- 参数量小（~58K参数）

## 数据格式要求

需要3D转换后的MAT文件，包含以下字段：
- `data`: (384, 336, 256, 351) - 4D多模态特征
- `region_labels`: (384, 336, 256) - 标签（1-102）
- `region_mask`: (384, 336, 256) - 脑组织掩膜

## 使用方法

### 1. 测试数据加载

首先验证数据是否能正确加载：

```bash
python test_data_loading.py --mat_file /path/to/subject1_3d_validated.mat --patch_size 3
```

### 2. 训练单个模型

训练3×3模型，使用被试1作为测试集：

```bash
python train_baseline_3x3_7x7.py \
    --data_dir /path/to/3d/mat/files \
    --output_dir ./results \
    --patch_size 3 \
    --test_subject 1 \
    --batch_size 256 \
    --epochs 50 \
    --samples_per_subject 10000 \
    --use_se
```

训练7×7模型：

```bash
python train_baseline_3x3_7x7.py \
    --data_dir /path/to/3d/mat/files \
    --output_dir ./results \
    --patch_size 7 \
    --test_subject 1 \
    --batch_size 256 \
    --epochs 50 \
    --samples_per_subject 10000 \
    --use_se
```

### 3. Leave-One-Out 批量训练

修改`run_leave_one_out.sh`中的`DATA_DIR`变量，然后运行：

```bash
# 修改脚本中的 DATA_DIR 路径
vim run_leave_one_out.sh

# 运行批量训练
bash run_leave_one_out.sh
```

这将对每个被试运行一次测试（共38次），分别训练3×3和7×7模型。

### 4. 分析结果

训练完成后，分析所有结果：

```bash
python analyze_results.py --results_dir ./results_leave_one_out
```

这将生成：
- 统计摘要（平均F1、标准差等）
- 可视化图表（箱线图、散点图等）
- CSV格式的详细结果

## 参数说明

### 训练参数
- `--data_dir`: 3D MAT文件目录
- `--output_dir`: 输出目录
- `--patch_size`: Patch大小（3或7）
- `--test_subject`: 测试被试编号（1-38）
- `--batch_size`: 批次大小（默认256）
- `--epochs`: 训练轮数（默认50）
- `--samples_per_subject`: 每个被试采样数（默认10000）
- `--lr`: 学习率（默认1e-4）
- `--use_se`: 使用SE模块（推荐）
- `--use_refine`: 使用额外的refine层（可选）

### 性能考虑

- **GPU内存**: A6000有48GB显存，可以使用较大的batch_size
- **采样策略**: 每个被试采样10000个体素，平衡训练时间和覆盖率
- **数据缓存**: 默认将数据缓存到内存，加速训练

## 评估指标

- **Loss**: 交叉熵损失
- **Macro F1 Score**: 所有类别F1的平均值，适合处理类别不平衡

## 预期结果

基于文档中的架构设计：
- 3×3模型：参数少，训练快，适合捕获局部特征
- 7×7模型：更大的感受野，可能捕获更多空间上下文

## 注意事项

1. 确保3D MAT文件已经正确生成（使用dataset_create中的脚本）
2. 第一次运行时建议先用少量epoch测试
3. 可以通过修改`samples_per_subject`来控制训练时间
4. 结果会自动保存最佳模型和训练历史

## 故障排除

如果遇到内存问题：
- 减小`batch_size`
- 减小`samples_per_subject`
- 设置`cache_data=False`（会变慢）

如果找不到MAT文件：
- 检查文件命名格式
- 确认路径正确
- 使用`test_data_loading.py`验证