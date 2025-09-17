# 如何使用批量训练脚本

## 🚀 快速开始

### 单个被试训练
```bash
# 切换到ResNet目录
cd "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/training/3D CNN/ResNet"




  python scripts/test_batch_loading.py \
      --data_dir /home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated \
      --batch_files 3
      
    python scripts/test_voxel_uniqueness.py \
      --data_dir /home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated \
      --batch_files 3 \
      --test_epochs 3
      
# 训练被试1（推荐方法）
python scripts/train_mri_resnet_batch.py \
    --data_dir /path/to/your/mat/files \
    --test_subject 1 \
    --batch_files 3 \
    --epochs 100
```

### 完整Leave-One-Out训练
```bash
# 修改脚本中的数据路径
nano scripts/run_leave_one_out_batch.sh

# 运行完整的38个被试训练
bash scripts/run_leave_one_out_batch.sh
```

## 📁 脚本选择指南

| 脚本名称 | 用途 | 内存需求 | 推荐度 |
|---------|------|----------|--------|
| `train_mri_resnet_batch.py` | 单被试批量训练 | 低（2-5GB） | ⭐⭐⭐⭐⭐ |
| `run_leave_one_out_batch.sh` | 38被试批量训练 | 低（2-5GB） | ⭐⭐⭐⭐⭐ |
| `train_mri_resnet.py` | 单被试原始训练 | 高（30GB+） | ⭐⭐ |
| `run_leave_one_out.sh` | 38被试原始训练 | 高（30GB+） | ⭐⭐ |

## ⚙️ 重要配置

### 根据服务器内存调整 `batch_files`

**内存 >= 16GB**：
```bash
--batch_files 3
--batch_size 256
```

**内存 8-16GB**：
```bash
--batch_files 2
--batch_size 256
```

**内存 < 8GB**：
```bash
--batch_files 2
--batch_size 128
--samples_per_subject 5000
```

### Shell脚本配置

编辑 `scripts/run_leave_one_out_batch.sh`：

```bash
# 1. 修改数据路径
DATA_DIR="/your/actual/path/to/mat/files"

# 2. 根据内存调整批量文件数
BATCH_FILES=3  # 改为 2 如果内存不足

# 3. 根据GPU调整批次大小
BATCH_SIZE=256  # 改为 128 如果GPU内存不足
```

## 🔧 测试和验证

### 1. 基础功能测试
```bash
python scripts/test_batch_loading.py \
    --data_dir /path/to/mat/files \
    --batch_files 3
```

### 2. 体素唯一性验证
```bash
python scripts/test_voxel_uniqueness.py \
    --data_dir /path/to/mat/files \
    --batch_files 3
```

## 📊 监控训练

### 实时内存监控
```bash
# 在另一个终端运行
watch -n 2 'free -h'
# 或
htop
```

### 检查训练日志
```bash
# 查看最新的训练输出
tail -f results_leave_one_out_resnet_batch/training_progress.log
```

## 🚨 常见问题

### 内存不足错误
```bash
# 减少批量文件数
--batch_files 2

# 减少批次大小
--batch_size 128

# 减少样本数
--samples_per_subject 5000
```

### GPU内存不足
```bash
# 减少批次大小
--batch_size 64

# 减少工作进程
--num_workers 2
```

### 训练太慢
```bash
# 增加批量文件数（如果内存允许）
--batch_files 4

# 使用SSD存储MAT文件
# 减少worker数量避免IO竞争
--num_workers 2
```

## 📈 结果分析

训练完成后：

```bash
# 分析单个被试结果
ls results_leave_one_out_resnet_batch/resnet_batch_test_subject_1/

# 查看训练历史图表
results_leave_one_out_resnet_batch/resnet_batch_test_subject_1/training_history.png

# 查看详细指标
cat results_leave_one_out_resnet_batch/resnet_batch_test_subject_1/training_results.json
```

## 💡 最佳实践

1. **首次使用**：先用小数据集测试
2. **监控资源**：训练时监控内存和GPU使用
3. **定期保存**：检查是否生成了检查点文件
4. **备份结果**：重要实验结果及时备份

## 🔄 从原始脚本迁移

如果你之前使用原始脚本：

```bash
# 原始方法
python scripts/train_mri_resnet.py --data_dir /path --test_subject 1

# 改为批量方法
python scripts/train_mri_resnet_batch.py --data_dir /path --test_subject 1 --batch_files 3
```

训练效果完全相同，只是内存使用大幅减少！