# 使用1D训练集进行训练并映射回3D系统

## 概述

这个系统使用1D训练集（38个probanden的1D MAT文件，去除_3d后缀）进行训练，复现了Alex的网络架构，但适配了我们的351维输入（原始是341维）。训练采用Leave-one-out策略，预测后将softmax概率映射回3D体积，便于后续分析。

## 系统特点

- **使用1D训练集**：直接使用multidim_data和seg_one_hot数据
- **Leave-one-out训练**：37个probanden训练，1个测试
- **完全复现Alex的架构**：4层4096神经元的全连接网络
- **相同的超参数**：学习率1e-5，批次128，25个epochs
- **L2正则化**：weight_decay=0.00001（仅对权重）
- **评估指标**：每个epoch记录train/test的loss和macro F1
- **3D映射**：使用测试probanden的3D mask将预测映射回(384,336,256,102)
- **HDF5格式输出**：支持大体积文件保存（>2GB），MAT v7.3格式
- **严格验证系统**：往返一致性检查、概率约束验证、文件匹配安全性
- **智能文件匹配**：基于被试名的键值匹配，避免索引错配
- **完整的可视化**：包括置信度图、不确定性图、类别性能分析

## 文件说明

### 核心脚本

1. **`train_1d_with_3d_dataset.py`** - 主训练脚本
   - 加载1D训练集（multidim_data, seg_one_hot）
   - 使用Alex的网络架构进行1D训练
   - Leave-one-out训练策略
   - 使用3D mask将预测映射回3D体积

2. **`visualize_1d_3d_predictions.py`** - 可视化脚本
   - 加载预测的3D概率体积
   - 生成多种可视化（预测图、置信度图、熵图等）
   - 分析每个类别的性能
   - 计算整体准确率

3. **`run_1d_training.sh`** - 单次训练脚本
   - 运行单个被试作为测试集的训练
   - 自动生成可视化

4. **`run_1d_leave_one_out.sh`** - 完整Leave-one-out脚本
   - 对所有38个被试分别作为测试集进行训练
   - 生成汇总报告

## 网络架构

```python
RegModel(
  input_dim=351,      # 我们的特征维度
  num_classes=102     # 脑区域类别数
)
├── fc1: Linear(351 → 4096)
├── fc2: Linear(4096 → 4096)
├── fc3: Linear(4096 → 4096)
├── fc4: Linear(4096 → 4096)
├── fc5: Linear(4096 → 102)
└── dropout: Dropout(0.5)
```

## 数据流程

```
1D训练集 (38个probanden MAT文件)
    ↓
Leave-one-out: 37个训练 + 1个测试
    ↓
训练集: multidim_data (n_voxels, 351) + seg_one_hot → labels
    ↓
StandardScaler标准化
    ↓
Alex的1D网络训练: (batch, 351) → (batch, 102)
    ↓
测试集预测: Softmax概率 (n_test_voxels, 102)
    ↓
3D mask映射: 使用测试probanden的3D mask
    ↓
3D概率体积: (384×336×256×102)
```

## 使用方法

### 1. 快速开始（训练新模型）

```bash
# 修改数据路径
vim run_1d_training.sh
# 设置 DATA_DIR_1D="/path/to/1d/data"  # 1D训练集目录
# 设置 DATA_DIR_3D="/path/to/3d/data"  # 3D数据集目录（用于mask）

# 运行单次训练（使用被试38作为测试集）
bash run_1d_training.sh

# 或运行完整Leave-one-out（38次训练）
bash run_1d_leave_one_out.sh
```

### 2. 使用已有模型直接预测（推荐）

如果你已有训练好的模型，可以直接进行预测并生成3D softmax概率：

```bash
# 直接预测单个被试
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./predictions \
    --test_subject 38 \
    --load_model /path/to/dense_4x4096_model_test38.pth \
    --predict_only \
    --save_predictions

# 批量预测所有被试（保存为shell脚本）
cat > run_batch_predictions.sh << 'EOF'
#!/bin/bash

# 配置路径
DATA_DIR_1D="/path/to/1d/data"
DATA_DIR_3D="/path/to/3d/data" 
OUTPUT_DIR="./predictions"
MODEL_DIR="/path/to/models"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 批量预测
for i in {1..38}; do
    echo "预测被试 ${i}..."
    python train_1d_with_3d_dataset.py \
        --data_dir_1d ${DATA_DIR_1D} \
        --data_dir_3d ${DATA_DIR_3D} \
        --output_dir ${OUTPUT_DIR} \
        --test_subject ${i} \
        --load_model ${MODEL_DIR}/dense_4x4096_model_test${i}.pth \
        --predict_only \
        --save_predictions
done

echo "批量预测完成！"
EOF

chmod +x run_batch_predictions.sh
bash run_batch_predictions.sh
```

### 3. 自定义训练

```bash
python train_1d_with_3d_dataset.py \
    --data_dir_1d /path/to/1d/data \
    --data_dir_3d /path/to/3d/data \
    --output_dir ./results \
    --test_subject 38 \
    --batch_size 128 \
    --epochs 25 \
    --samples_per_subject 50000 \
    --save_predictions
```

### 4. 可视化结果

```bash
python visualize_1d_3d_predictions.py \
    --pred_file results/predictions_3d_test38.mat \
    --gt_file /path/to/test/subject_3d_validated.mat \
    --slice_idx 128 \
    --output_dir ./visualizations
```

## 参数说明

### 核心参数
- `--data_dir_1d`: 1D训练集目录路径
- `--data_dir_3d`: 3D数据集目录路径（提供mask信息）
- `--output_dir`: 输出目录（默认：./results_1d）
- `--test_subject`: 测试被试编号1-38（默认：38）

### 训练参数（与Alex一致）
- `--batch_size`: 批次大小（默认：128）
- `--epochs`: 训练轮数（默认：25）
- `--samples_per_subject`: 每个被试采样体素数（默认：None，使用全部）
- `--save_predictions`: 是否保存3D预测概率（推荐开启）

### 预测模式参数
- `--load_model`: 预训练模型路径（.pth文件）
- `--predict_only`: 仅预测模式，跳过训练

### 网络参数（固定）
- `input_dim`: 351（我们的特征维度）
- `num_classes`: 102（脑区域数）
- `learning_rate`: 0.00001
- `dropout`: 0.5
- `weight_decay`: 0.00001（L2正则化）
- `optimizer`: Adam

## 输出文件

### 训练模式输出
```
results_1d_with_3d/
├── dense_4x4096_model_test38.pth    # 模型权重和scaler
├── history_test38.json               # 训练历史（train/test loss和F1）
└── predictions_3d_test38.mat        # 3D概率体积（如果开启--save_predictions）
```

### 预测模式输出
```
predictions/
└── predictions_3d_test38.mat        # 3D概率体积（必须开启--save_predictions）
```

### HDF5文件格式
```matlab
predictions_3d_test38.mat (HDF5/MAT v7.3格式):

数据集:
  - softmax_probabilities: (384, 336, 256, 102) float32, gzip压缩

属性:
  - test_subject: 38
  - test_file_1d: 完整1D文件路径（用于追溯）
  - test_file_3d: 完整3D文件路径（用于后处理匹配）
  - shape: (384, 336, 256, 102)
  - prob_range: [min_prob, max_prob]
```

**注意**：现在使用HDF5格式保存大体积概率文件（~13GB），支持MATLAB v7.3加载，并包含完整元数据用于文件匹配和追溯。

### 可视化输出
```
visualizations/
├── slice_128_visualization.png  # 切片可视化（6个子图）
└── class_performance.png        # 类别性能分析（4个子图）
```

## 性能指标

### 训练期间记录
- **每个epoch**：train loss, train macro F1, test loss, test macro F1
- **最佳模型**：基于测试F1选择最佳模型
- **最终报告**：最佳测试F1和最终指标

### 3D可视化分析
- **整体准确率**：所有有效体素的分类准确率
- **类别准确率**：每个脑区域的单独准确率
- **置信度分析**：平均置信度和不确定性
- **错误分析**：哪些类别容易混淆

## 与2D/3D Patch方法的对比

| 方法 | 输入 | 参数量 | 特点 |
|------|------|--------|------|
| **1D (Alex)** | (351,) | ~68M | 纯特征，无空间信息 |
| **2D Patch** | (351,3,3) | ~58K | 利用2D邻域 |
| **3D Patch** | (351,3,3,3) | ~60K | 利用3D邻域 |

## 优势和局限

### 优势
- **简单直接**：每个体素独立处理
- **并行性好**：可以批量处理大量体素
- **基线明确**：纯特征分类的性能基准

### 局限
- **无空间信息**：不利用邻域关系
- **参数量大**：4层4096的网络参数多
- **可能过拟合**：特别是在小数据集上

## 下一步建议

1. **性能对比**：与2D/3D patch方法比较
2. **特征分析**：哪些特征对分类最重要
3. **错误分析**：系统性地分析错误模式
4. **集成学习**：结合1D、2D、3D的预测

## 注意事项

1. **内存需求**：加载所有体素可能需要大量内存
2. **计算时间**：1D网络参数多，训练较慢
3. **数据标准化**：使用StandardScaler，测试时需要相同的scaler
4. **文件格式**：输出HDF5格式，文件较大（~13GB）但支持压缩
5. **文件匹配**：使用被试名进行安全匹配，避免索引错配
6. **形状验证**：严格检查3D数据形状，不允许隐式转置

## 故障排除

### 内存不足
```python
# 减少每个被试的采样数
--samples_per_subject 10000
```

### 训练不收敛
```python
# 调整学习率
--lr 1e-4  # 或 1e-6
```

### 3D数据形状错误
```python
# 现在使用严格断言，形状必须为(384, 336, 256)
# 如果遇到形状不匹配，检查3D数据集的创建过程
# 不再支持自动转置，确保数据集标准化
```

### 文件匹配错误
```python
# 检查1D和3D文件的被试名是否一致
# 确保文件名格式：1D为"ODP_01_qhlazec.mat"，3D为"ODP_01_qhlazec_3d_validated.mat"
# 验证subject_key_1d()和subject_key_3d()函数的提取逻辑
```

### 预测概率验证失败
```python
# 检查往返一致性验证错误
# 1. roundtrip_before_fix: 映射逻辑问题
# 2. 概率约束验证: 概率和不为1或值超出[0,1]
# 3. 掩膜一致性: 背景区域非零
```

### HDF5文件保存失败
```python
# 检查磁盘空间（需要~13GB）
# 确保有写权限
# 如果h5py版本过低，升级：pip install h5py>=3.0
```

### 类别不平衡
考虑使用加权损失函数或平衡采样策略。

## 重要更新记录

### v1.1 - 安全性与验证系统大幅强化

#### 🔒 文件匹配安全性
- **被试名匹配**：替换危险的索引匹配为基于被试名的键值匹配
- **一致性验证**：训练前验证1D与3D文件的被试名完全一致
- **错配防护**：`subject_key_1d()` 和 `subject_key_3d()` 函数确保文件对应关系

#### 🛡️ 数据验证系统
- **严格形状检查**：替换`.T`转置为`assert`断言，防止隐式轴变换
- **往返一致性验证**：映射到3D后立即验证能否完全恢复原始1D数据
- **概率约束验证**：检查softmax概率和为1、值在[0,1]范围内
- **1D-3D标签一致性**：验证同一被试1D和3D数据的标签完全匹配

#### 💾 HDF5格式升级
- **大文件支持**：使用HDF5替代MAT v5，支持>2GB概率体积
- **元数据保存**：保存完整文件路径用于后处理模块的安全匹配
- **压缩存储**：gzip压缩减小文件大小，保持数据精度
- **MATLAB兼容**：HDF5格式等效于MAT v7.3，MATLAB可直接加载

#### 🔧 技术改进
- **体素级统计修正**：按体素聚合统计，避免通道维度放大错误
- **设备兼容性**：修复预测模式下的CPU/GPU设备不匹配问题
- **数据类型一致性**：sklearn变换后强制转换为float32，避免tensor类型冲突
- **验证顺序优化**：往返验证在概率修正前执行，确保验证纯净性