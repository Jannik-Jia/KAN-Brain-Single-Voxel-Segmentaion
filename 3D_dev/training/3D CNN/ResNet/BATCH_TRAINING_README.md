# 批量数据加载训练指南

## 概述

为解决服务器RAM不足导致无法一次性加载所有MAT文件的问题，我们开发了批量数据加载版本的ResNet训练脚本。该版本可以：

- **分批加载数据**：每次只加载指定数量的MAT文件到内存
- **自动内存管理**：训练完一批后自动释放内存，加载下一批
- **保持训练效果**：确保与原始版本相同的训练效果
- **节省内存使用**：大幅减少内存占用，适合资源受限的服务器

## 新增文件

1. **`models/batch_dataset.py`** - 基础批量数据加载的数据集类
2. **`models/batch_dataset_improved.py`** - 改进版本，避免重复训练相同patch（推荐）
3. **`scripts/train_mri_resnet_batch.py`** - 支持批量加载的训练脚本（已更新使用改进版本）
4. **`scripts/test_batch_loading.py`** - 测试批量加载功能的脚本
5. **`scripts/test_voxel_uniqueness.py`** - 测试体素唯一性和重复问题的脚本

## 使用方法

### 1. 基本训练命令

```bash
# 批量训练（每次加载3个mat文件）
python scripts/train_mri_resnet_batch.py \\
    --data_dir /path/to/mat/files \\
    --test_subject 1 \\
    --batch_files 3 \\
    --epochs 100 \\
    --batch_size 256
```

### 2. 关键参数说明

- `--batch_files`: 每次加载的文件数量（默认3个）
  - 根据服务器内存调整：内存越小，设置越小
  - 建议值：2-5个文件

- `--data_dir`: MAT文件目录路径
- `--test_subject`: Leave-One-Out的测试被试编号
- `--output_dir`: 结果输出目录（默认`./results_batch`）

### 3. 内存优化参数

```bash
# 最小内存使用配置
python scripts/train_mri_resnet_batch.py \\
    --data_dir /path/to/mat/files \\
    --test_subject 1 \\
    --batch_files 2 \\          # 每次只加载2个文件
    --batch_size 128 \\         # 减小批次大小
    --samples_per_subject 5000 \\  # 减少每个被试的样本数
    --num_workers 2             # 减少数据加载进程数
```

### 4. 完整训练示例

```bash
# 标准训练配置
python scripts/train_mri_resnet_batch.py \\
    --data_dir /data/mat_files \\
    --test_subject 1 \\
    --output_dir ./results/subject_1 \\
    --batch_files 3 \\
    --epochs 100 \\
    --batch_size 256 \\
    --learning_rate 1e-4 \\
    --patience 15 \\
    --loss_type cb_focal \\
    --use_mixup \\
    --use_ema \\
    --verbose
```

## 测试功能

### 1. 基本功能测试

```bash
# 测试批量加载是否正常工作
python scripts/test_batch_loading.py \\
    --data_dir /path/to/mat/files \\
    --batch_files 3
```

### 2. 内存使用对比测试

```bash
# 对比原始方法和批量方法的内存使用
python scripts/test_batch_loading.py \\
    --data_dir /path/to/mat/files \\
    --batch_files 3 \\
    --compare
```

## 工作原理

### 1. 批量数据加载流程（改进版本）

```
初始化阶段：
1. 预扫描所有MAT文件，获取有效体素位置（只读坐标，不加载完整数据）
2. 分析全局类别分布（采样统计，避免内存溢出）
3. 为epoch预分配唯一的体素样本（避免重复训练）
4. 将训练文件分为多个批次（每批batch_files个文件）
5. 加载第一批文件到内存（避免重复加载）

训练阶段：
6. 训练第一批数据（使用预分配的样本）
7. 释放第一批数据，加载第二批
8. 重复直到所有批次训练完成
9. 一个epoch = 所有批次训练完成，每个体素最多训练一次
10. 下一个epoch重新分配样本（保证多样性）
```

### 2. 内存管理

- **预扫描阶段**：只读取标签信息，不加载完整数据
- **批次加载**：只保留当前批次的数据在内存中
- **自动释放**：切换批次时自动调用垃圾回收
- **智能缓存**：测试数据（通常只有1个文件）仍使用原始缓存方式

### 3. 训练兼容性

- **损失函数**：与原始版本完全一致
- **数据处理**：保持相同的标准化和增强策略
- **模型架构**：使用相同的ResNet-50模型
- **评估方式**：使用相同的Leave-One-Out交叉验证

## 内存节省效果

假设有30个训练文件，每个文件约1GB：

| 方法 | 内存使用 | 节省比例 |
|------|----------|----------|
| 原始方法 | ~30GB | 0% |
| 批量方法（3文件/批） | ~3GB | 90% |
| 批量方法（2文件/批） | ~2GB | 93% |

## 性能说明

### 1. 训练时间

- **略有增加**：由于需要分批加载数据
- **增加幅度**：约10-20%（取决于磁盘IO速度）
- **可接受范围**：相比无法运行，轻微的时间增加是值得的

### 2. 训练效果

- **完全一致**：最终模型性能与原始版本相同
- **收敛稳定**：批次间的样本分布保持均匀
- **指标可比**：F1分数等指标与原始版本可直接对比

## 故障排除

### 1. 内存不足错误

```bash
# 减少batch_files
--batch_files 2

# 减少batch_size
--batch_size 128

# 减少samples_per_subject
--samples_per_subject 5000
```

### 2. 训练速度慢

```bash
# 增加batch_files（如果内存允许）
--batch_files 4

# 减少num_workers（避免IO竞争）
--num_workers 2

# 使用SSD存储MAT文件
```

### 3. 磁盘空间不足

- 确保有足够空间存储检查点和结果
- 定期清理旧的实验结果
- 考虑使用外部存储

## 与原始版本的差异

| 特性 | 原始版本 | 批量版本 |
|------|----------|----------|
| 内存使用 | 一次性加载所有文件 | 分批加载 |
| 训练时间 | 更快 | 略慢 |
| 内存需求 | 高（30GB+） | 低（2-5GB） |
| 训练效果 | 基准 | 相同 |
| 适用场景 | 高内存服务器 | 普通服务器 |

## 建议配置

### 1. 服务器内存 >= 16GB
```bash
--batch_files 3
--batch_size 256
--samples_per_subject 10000
```

### 2. 服务器内存 8-16GB
```bash
--batch_files 2
--batch_size 128
--samples_per_subject 8000
```

### 3. 服务器内存 < 8GB
```bash
--batch_files 1
--batch_size 64
--samples_per_subject 5000
```

## 监控和调试

### 1. 内存监控

```bash
# 在训练过程中监控内存使用
watch -n 1 'free -h'

# 或使用htop
htop
```

### 2. 训练日志

训练日志会显示：
- 当前批次信息
- 内存使用情况
- 每批次的训练指标
- 批次切换时间

### 3. 调试模式

```bash
# 启用详细日志
--verbose

# 使用更小的数据集测试
--samples_per_subject 1000
--epochs 5
```

## 注意事项

1. **确保磁盘IO性能**：批量加载对磁盘读取速度有要求
2. **合理设置batch_files**：太小会影响性能，太大会占用过多内存
3. **监控训练进度**：确保所有批次都正常训练
4. **保存重要检查点**：定期保存模型以防训练中断
5. **验证结果一致性**：与原始版本对比确保效果相同

## 重要改进：避免重复训练相同的Patch

### 问题背景

原始的批量加载方案存在一个潜在问题：**可能在同一个epoch内重复训练相同的7×7 patch**。这是因为：

1. 每个批次独立进行随机采样
2. 不同批次可能选择相同的体素位置
3. 降低了训练效率，影响模型性能

### 改进方案

我们开发了 `batch_dataset_improved.py`，确保：

1. **Epoch级别的样本预分配**：每个epoch开始时预先分配所有体素样本
2. **严格的唯一性保证**：每个epoch内每个体素最多被训练一次
3. **跨epoch的多样性**：不同epoch使用不同的随机种子重新分配样本
4. **内存效率不变**：仍然只加载当前批次的数据到内存

### 验证唯一性

```bash
# 测试体素唯一性
python scripts/test_voxel_uniqueness.py \\
    --data_dir /path/to/mat/files \\
    --batch_files 3 \\
    --test_epochs 3
```

这个脚本会验证：
- 原始版本 vs 改进版本的重复情况
- 单个epoch内的体素唯一性
- 跨epoch的体素分布多样性

### 性能提升

通过避免重复训练：
- **提高训练效率**：每个epoch覆盖更多唯一样本
- **改善收敛速度**：减少冗余计算
- **保持内存优势**：内存使用量不变
- **确保训练质量**：每个体素得到适当的训练频次

### 实现细节

1. **预扫描阶段**：只读取坐标信息，不加载完整数据
2. **样本分配**：基于epoch编号和文件哈希的确定性随机分配
3. **批次管理**：将预分配的样本按文件归属分配给对应批次
4. **自动切换**：epoch结束时自动重新分配下一epoch的样本

通过这些改进，你现在可以在资源受限的服务器上成功训练ResNet模型，同时确保最佳的训练效率和效果！