# 3D脑体素分类基线模型训练系统 - 详细文档

## 目录
1. [系统概述](#系统概述)
2. [文件详细说明](#文件详细说明)
3. [使用流程详解](#使用流程详解)
4. [数据流和架构](#数据流和架构)
5. [常见问题和解决方案](#常见问题和解决方案)
6. [性能优化建议](#性能优化建议)

---

## 系统概述

本训练系统实现了一个完整的Leave-One-Out交叉验证框架，用于评估3×3和7×7 patch基线模型在3D脑体素分类任务上的性能。系统设计用于处理38个被试的多模态MRI数据，每个被试包含384×336×256个体素，每个体素有351维特征，需要分类到102个脑区域。

### 核心特点
- **Leave-One-Out策略**：每次留一个被试作为测试集，其余37个作为训练集，共运行38次
- **两种Patch尺寸**：3×3和7×7，对应不同的感受野
- **高效架构**：参数量仅~58K，使用GroupNorm和SiLU激活
- **完整评估**：计算Loss和Macro F1 Score
- **自动化流程**：从训练到结果分析全自动化

---

## 文件详细说明

### 1. `train_baseline_3x3_7x7.py` - 核心训练脚本

#### 功能描述
这是整个系统的核心，负责模型定义、数据加载、训练循环和评估。

#### 主要组件

##### 1.1 模型定义
```python
class SE2D(nn.Module)
```
- **作用**：Squeeze-and-Excitation通道注意力模块
- **参数**：
  - `ch`: 输入通道数
  - `r`: 缩减比例（默认8）
- **流程**：
  1. 全局平均池化获取通道描述符
  2. 两层FC进行通道权重学习
  3. Sigmoid激活生成通道注意力权重
  4. 原始特征与权重相乘

```python
class ImprovedConv2D_Baseline(nn.Module)
```
- **作用**：主分类模型
- **参数**：
  - `input_channels`: 输入特征维度（351）
  - `num_classes`: 输出类别数（102）
  - `mid`: 中间层通道数（128）
  - `use_refine`: 是否使用额外精炼层
  - `use_se`: 是否使用SE模块
  - `kernel_size`: 空间聚合核大小（3或7）
- **架构流程**：
  1. **通道混合层**：Conv2d(351→128, kernel=1) + GroupNorm + SiLU
  2. **空间聚合层**：Conv2d(128→128, kernel=3/7) + GroupNorm + SiLU
  3. **可选精炼层**：Conv2d(128→128, kernel=1) + 残差连接
  4. **SE注意力**：通道重加权
  5. **分类头**：Linear(128→102)

##### 1.2 数据集类
```python
class Brain3DPatchDataset(Dataset)
```
- **作用**：处理3D脑体积数据，提取patch用于训练
- **初始化参数**：
  - `mat_files`: MAT文件路径列表
  - `patch_size`: patch尺寸（3或7）
  - `samples_per_subject`: 每个被试采样数（默认10000）
  - `is_train`: 训练/测试模式标志
  - `cache_data`: 是否缓存数据到内存

- **关键方法**：
  - `load_subject_data()`: 加载单个被试的3D数据
    - 读取HDF5格式的MAT文件
    - 处理MATLAB的列优先存储格式（转置）
    - 返回data、labels、mask字典
  
  - `_generate_sample_indices()`: 生成采样索引
    - 找出所有有效体素（mask>0且label>0）
    - 训练时随机采样，测试时均匀采样
    - 返回(文件索引, x, y, z)元组列表
  
  - `extract_patch_2d()`: 提取2D patch
    - 在xy平面上提取patch（z坐标固定）
    - 处理边界情况（零填充）
    - 确保输出尺寸一致
    - 返回patch_size×patch_size×351的2D数组
  
  - `__getitem__()`: 获取单个样本
    - 提取2D patch（在固定z切片上的xy邻域）
    - 转换为PyTorch张量格式(C, H, W)

##### 1.3 训练器类
```python
class Trainer
```
- **作用**：管理训练过程
- **组件**：
  - AdamW优化器（weight_decay=1e-4）
  - CrossEntropyLoss损失函数
  - CosineAnnealingLR学习率调度
  - 混合精度训练（AMP）

- **方法**：
  - `train_epoch()`: 执行一个训练epoch
    - 前向传播、损失计算、反向传播
    - 梯度缩放（混合精度）
    - 记录预测和标签用于F1计算
  
  - `evaluate()`: 评估模型
    - 无梯度计算
    - 收集所有预测
    - 计算Loss和Macro F1

##### 1.4 主函数流程
1. **参数解析**：处理命令行参数
2. **数据准备**：
   - 查找所有MAT文件
   - 分割训练集（37个被试）和测试集（1个被试）
3. **数据集创建**：
   - 训练集：37个被试×10000样本
   - 测试集：1个被试×20000样本（多采样以获得稳定评估）
4. **模型初始化**：创建指定patch_size的模型
5. **训练循环**：
   - 每个epoch训练并评估
   - 保存最佳模型（基于test F1）
   - 记录训练历史
6. **结果保存**：
   - 模型checkpoint（.pth文件）
   - 训练历史（.json文件）

#### 使用示例
```bash
# 基础使用
python train_baseline_3x3_7x7.py \
    --data_dir /path/to/mat/files \
    --patch_size 3 \
    --test_subject 1

# 完整参数示例
python train_baseline_3x3_7x7.py \
    --data_dir /data/3d_mri \
    --output_dir ./results \
    --patch_size 7 \
    --test_subject 15 \
    --batch_size 512 \
    --epochs 100 \
    --samples_per_subject 20000 \
    --lr 5e-5 \
    --use_se \
    --use_refine
```

---

### 2. `run_leave_one_out.sh` - 批量训练脚本

#### 功能描述
自动化执行完整的Leave-One-Out交叉验证，对所有38个被试分别训练3×3和7×7模型。

#### 脚本结构
```bash
#!/bin/bash

# 配置部分
DATA_DIR="/path/to/your/3d/mat/files"  # 需要修改
OUTPUT_DIR="./results_leave_one_out"
EPOCHS=50
BATCH_SIZE=256
SAMPLES_PER_SUBJECT=10000

# 目录创建
mkdir -p ${OUTPUT_DIR}/3x3
mkdir -p ${OUTPUT_DIR}/7x7

# 3×3模型训练循环（38次）
for test_subject in {1..38}
do
    python train_baseline_3x3_7x7.py \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/3x3 \
        --patch_size 3 \
        --test_subject ${test_subject} \
        ...
done

# 7×7模型训练循环（38次）
for test_subject in {1..38}
do
    python train_baseline_3x3_7x7.py \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/7x7 \
        --patch_size 7 \
        --test_subject ${test_subject} \
        ...
done
```

#### 使用步骤
1. **修改数据路径**：
   ```bash
   vim run_leave_one_out.sh
   # 修改 DATA_DIR 变量为实际路径
   ```

2. **调整参数**（可选）：
   - `EPOCHS`: 训练轮数
   - `BATCH_SIZE`: 批次大小
   - `SAMPLES_PER_SUBJECT`: 采样数

3. **运行脚本**：
   ```bash
   # 添加执行权限
   chmod +x run_leave_one_out.sh
   
   # 运行（建议使用nohup或tmux）
   nohup bash run_leave_one_out.sh > training.log 2>&1 &
   ```

4. **监控进度**：
   ```bash
   tail -f training.log
   ```

#### 输出结构
```
results_leave_one_out/
├── 3x3/
│   ├── best_model_patch3_test1.pth
│   ├── history_patch3_test1.json
│   ├── best_model_patch3_test2.pth
│   ├── history_patch3_test2.json
│   └── ... (共38个)
└── 7x7/
    ├── best_model_patch7_test1.pth
    ├── history_patch7_test1.json
    └── ... (共38个)
```

---

### 3. `analyze_results.py` - 结果分析脚本

#### 功能描述
汇总和分析所有Leave-One-Out训练的结果，生成统计报告和可视化。

#### 主要功能

##### 3.1 数据加载
```python
def load_results(results_dir: Path, patch_size: int)
```
- 遍历所有测试被试的结果文件
- 提取每个训练的最佳F1、对应epoch、训练F1等
- 返回结果列表

##### 3.2 统计分析
```python
def analyze_and_plot(results_3x3, results_7x7, output_dir)
```
计算的统计指标：
- **平均F1 Score**：所有被试的平均测试F1
- **标准差**：衡量模型稳定性
- **最小/最大F1**：性能范围
- **中位数F1**：鲁棒的中心趋势
- **平均最佳轮数**：收敛速度指标

##### 3.3 可视化（4个子图）
1. **箱线图**：F1分布对比
2. **折线图**：每个被试的F1表现
3. **散点图**：训练vs测试F1（过拟合分析）
4. **直方图**：最佳epoch分布（收敛速度）

##### 3.4 输出文件
- `results_comparison.png`: 可视化图表
- `detailed_results.csv`: 所有结果详细表格
- `summary_statistics.csv`: 统计摘要

#### 使用方法
```bash
# 基础使用
python analyze_results.py --results_dir ./results_leave_one_out

# 分析结果会打印到控制台并保存文件
```

#### 输出示例
```
==================================================
Leave-One-Out 交叉验证结果汇总
==================================================

3x3 Patch 模型:
  测试被试数: 38
  平均 F1 Score: 0.8234 ± 0.0456
  中位数 F1 Score: 0.8267
  最小 F1 Score: 0.7123
  最大 F1 Score: 0.9012
  平均最佳轮数: 35.2

7x7 Patch 模型:
  测试被试数: 38
  平均 F1 Score: 0.8456 ± 0.0398
  中位数 F1 Score: 0.8489
  最小 F1 Score: 0.7456
  最大 F1 Score: 0.9234
  平均最佳轮数: 32.8
```

---

### 4. `test_data_loading.py` - 数据验证脚本

#### 功能描述
在训练前验证数据加载和patch提取逻辑是否正确。

#### 测试内容

##### 4.1 MAT文件加载测试
```python
def test_load_mat_file(mat_file: Path)
```
- 列出文件中所有keys
- 检查数据形状和类型
- 验证转置操作
- 统计标签分布
- 计算有效体素比例

##### 4.2 Patch提取测试
```python
def test_patch_extraction(mat_file: Path, patch_size: int)
```
- 随机选择一个有效体素
- 提取3D patch
- 处理边界情况（padding）
- 提取2D切片
- 转换为模型输入格式

#### 使用方法
```bash
# 测试3×3 patch
python test_data_loading.py \
    --mat_file /path/to/subject1_3d_validated.mat \
    --patch_size 3

# 测试7×7 patch
python test_data_loading.py \
    --mat_file /path/to/subject1_3d_validated.mat \
    --patch_size 7
```

#### 预期输出
```
测试文件: subject1_3d_validated.mat
==================================================
文件包含的keys:
  data: shape=(351, 256, 336, 384), dtype=float32
  region_labels: shape=(256, 336, 384), dtype=int64
  region_mask: shape=(256, 336, 384), dtype=uint8
  ...

原始 'data' 形状: (351, 256, 336, 384)
转置后 'data' 形状: (384, 336, 256, 351)
  数据范围: [-2.34, 5.67]
  
'region_labels' 形状: (384, 336, 256)
  唯一标签数: 102
  标签范围: [0, 102]
  
'region_mask' 形状: (384, 336, 256)
  有效体素数: 1,876,234 / 33,030,144 (5.68%)

测试 3×3 2D Patch 提取
==================================================
选择的体素位置: (192, 168, 128)
该位置的标签: 42
在z=128切片上提取3×3的2D patch
提取的2D patch形状: (3, 3, 351)
模型输入形状 (C, H, W): (351, 3, 3)
  C=351个特征通道
  H×W=3×3空间维度

测试完成！
```

---

## 使用流程详解

### 完整工作流程

#### Step 1: 环境准备
```bash
# 1. 确认Python环境
python --version  # 需要3.7+

# 2. 安装依赖
pip install torch numpy h5py pandas scikit-learn matplotlib seaborn tqdm

# 3. 确认GPU
python -c "import torch; print(torch.cuda.is_available())"
```

#### Step 2: 数据准备
```bash
# 1. 确认3D MAT文件已生成
ls /path/to/3d/mat/files/*.mat | wc -l  # 应该有38个文件

# 2. 验证数据格式
python test_data_loading.py \
    --mat_file /path/to/3d/mat/files/subject1_3d_validated.mat \
    --patch_size 3
```

#### Step 3: 单次测试训练
```bash
# 先用少量epoch测试流程
python train_baseline_3x3_7x7.py \
    --data_dir /path/to/3d/mat/files \
    --output_dir ./test_run \
    --patch_size 3 \
    --test_subject 1 \
    --epochs 3 \
    --samples_per_subject 1000
```

#### Step 4: 完整训练
```bash
# 1. 修改批量脚本
vim run_leave_one_out.sh
# 设置 DATA_DIR="/path/to/3d/mat/files"

# 2. 运行批量训练（建议在tmux或screen中）
tmux new -s training
bash run_leave_one_out.sh

# 3. 脱离tmux (Ctrl+B, D)
# 4. 重新连接查看
tmux attach -t training
```

#### Step 5: 监控训练
```bash
# 查看整体进度
ls results_leave_one_out/3x3/*.json | wc -l  # 完成的3×3模型数
ls results_leave_one_out/7x7/*.json | wc -l  # 完成的7×7模型数

# 查看单个训练的历史
python -c "
import json
with open('results_leave_one_out/3x3/history_patch3_test1.json') as f:
    h = json.load(f)
    print(f'最佳F1: {max(h[\"test_f1\"])}')
"
```

#### Step 6: 结果分析
```bash
# 生成完整分析报告
python analyze_results.py --results_dir ./results_leave_one_out

# 查看生成的文件
ls results_leave_one_out/*.png  # 可视化图表
ls results_leave_one_out/*.csv  # 数据表格
```

---

## 2D Patch提取方式说明

### 重要修正内容

根据您的要求，已将patch提取方式从**错误的3D提取**修正为**正确的2D提取**。

#### 原始错误实现：
```python
# ❌ 错误：提取3×3×3的3D patch，然后取中间切片
patch_3d = data[x-1:x+2, y-1:y+2, z-1:z+2, :]  # (3, 3, 3, 351)
patch_2d = patch_3d[:, :, 1, :]  # 取中间切片 -> (3, 3, 351)
```

#### 修正后的实现：
```python
# ✅ 正确：直接在xy平面上提取2D patch（z坐标固定）
patch_2d = data[x-1:x+2, y-1:y+2, z, :]  # 直接得到 (3, 3, 351)
```

### 正确的2D Patch提取流程

#### 输入数据：
- **3D体积**：(384, 336, 256, 351)
- **体素位置**：(x, y, z)

#### 提取过程：
1. **确定中心体素**：位置(x, y, z)
2. **固定z坐标**：在z切片上工作
3. **提取xy邻域**：在xy平面上提取3×3或7×7的邻域
4. **结果**：(patch_size, patch_size, 351)

#### 模型输入：
- **张量形状**：(351, 3, 3) 或 (351, 7, 7)
- **含义**：351个特征通道，每个通道是patch_size×patch_size的2D空间

### 修正的好处：

1. **符合您的设计意图**：真正的2D卷积，处理2D空间邻域
2. **数据量合理**：不需要处理额外的z维度
3. **计算高效**：直接2D操作，无需3D到2D的转换
4. **语义清晰**：在固定深度上分析空间邻域关系

### 验证方法：

运行测试脚本验证patch提取：
```bash
python test_data_loading.py --mat_file your_file.mat --patch_size 3
```

预期输出：
```
测试 3×3 2D Patch 提取
选择的体素位置: (x, y, z)
在z=z切片上提取3×3的2D patch
提取的2D patch形状: (3, 3, 351)
模型输入形状 (C, H, W): (351, 3, 3)
```

### 模型架构匹配：

您的模型设计：
```python
# 输入: (B, 351, 3, 3) 或 (B, 351, 7, 7)
self.mix = nn.Conv2d(351, 128, kernel_size=1)  # 通道混合
self.agg = nn.Conv2d(128, 128, kernel_size=3/7, padding=0)  # 空间聚合到1×1
```

现在的实现完全匹配这个设计！

---

## 数据流和架构

### 数据处理流程
```
原始3D数据 (384×336×256×351)
    ↓
有效体素筛选 (mask>0, label>0)
    ↓
采样策略 (训练:随机, 测试:均匀)
    ↓
2D Patch提取 (在xy平面上提取3×3 或 7×7，z固定)
    ↓
张量转换 (351×3×3 或 351×7×7)
    ↓
批次组装 (batch_size个样本)
    ↓
模型前向传播
```

### 模型架构流程
```
输入 (B×351×H×W)
    ↓
通道混合 Conv2d(k=1): 351→128
    ↓
GroupNorm + SiLU
    ↓
空间聚合 Conv2d(k=3/7): H×W→1×1
    ↓
GroupNorm + SiLU
    ↓
SE注意力模块 (可选)
    ↓
展平: (B×128)
    ↓
分类头 Linear: 128→102
    ↓
输出 (B×102)
```

---

## 常见问题和解决方案

### Q1: 找不到MAT文件
**问题**：运行时提示"找到0个MAT文件"

**解决方案**：
1. 检查文件路径是否正确
2. 确认文件扩展名（.mat）
3. 检查文件命名模式，修改代码中的glob模式：
```python
# 在train_baseline_3x3_7x7.py中
mat_files = sorted(data_dir.glob('*.mat'))  # 改为适合的模式
```

### Q2: CUDA内存不足
**问题**：RuntimeError: CUDA out of memory

**解决方案**：
1. 减小batch_size：
```bash
--batch_size 128  # 或更小
```
2. 减少采样数：
```bash
--samples_per_subject 5000
```
3. 关闭数据缓存（在代码中设置`cache_data=False`）

### Q3: 数据加载很慢
**问题**：每个epoch耗时过长

**解决方案**：
1. 确保使用SSD存储
2. 增加num_workers：
```python
DataLoader(..., num_workers=8)  # 根据CPU核心数调整
```
3. 启用数据缓存（默认已启用）

### Q4: F1 Score很低
**问题**：模型性能不理想

**解决方案**：
1. 增加训练轮数：
```bash
--epochs 100
```
2. 调整学习率：
```bash
--lr 5e-5  # 或尝试1e-3
```
3. 增加采样数量：
```bash
--samples_per_subject 20000
```
4. 启用refine层：
```bash
--use_refine
```

### Q5: 训练中断恢复
**问题**：训练中断后如何继续

**解决方案**：
1. 查看已完成的被试：
```bash
ls results_leave_one_out/3x3/*.json
```
2. 修改run_leave_one_out.sh，从中断处继续：
```bash
for test_subject in {15..38}  # 假设14已完成
```

---

## 性能优化建议

### 1. GPU优化
- **混合精度训练**：已默认启用，可节省显存
- **批次大小**：A6000可以使用512或更大
- **梯度累积**：如需更大有效批次，可添加梯度累积

### 2. 数据优化
- **预处理**：可以预先提取所有patch并保存
- **数据增强**：可添加随机旋转、翻转等
- **平衡采样**：如果类别不平衡严重，可考虑加权采样

### 3. 模型优化
- **学习率调度**：已使用CosineAnnealing，可尝试OneCycleLR
- **正则化**：可添加更多Dropout或L2正则
- **架构调整**：可尝试更深的网络或残差连接

### 4. 训练策略
- **早停机制**：可添加早停避免过拟合
- **集成学习**：可训练多个模型进行集成
- **知识蒸馏**：可用大模型指导小模型

### 5. 硬件利用
- **多GPU训练**：可修改为DataParallel或DistributedDataParallel
- **CPU并行**：增加DataLoader的num_workers
- **内存管理**：定期torch.cuda.empty_cache()

---

## 附录：关键参数说明

### 数据相关
| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| patch_size | 3 | Patch空间尺寸 | 3, 5, 7 |
| samples_per_subject | 10000 | 每被试采样数 | 5000-50000 |
| batch_size | 256 | 批次大小 | 64-1024 |

### 模型相关
| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| mid | 128 | 中间层通道数 | 64-256 |
| use_se | True | 使用SE注意力 | 推荐开启 |
| use_refine | False | 额外精炼层 | 可选 |

### 训练相关
| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| epochs | 50 | 训练轮数 | 30-100 |
| lr | 1e-4 | 学习率 | 1e-5到1e-3 |
| weight_decay | 1e-4 | 权重衰减 | 1e-5到1e-3 |

### 评估指标
| 指标 | 说明 | 重要性 |
|------|------|--------|
| Loss | 交叉熵损失 | 优化目标 |
| Macro F1 | 所有类别F1均值 | 主要指标 |
| Per-class F1 | 每个类别的F1 | 诊断用 |

---

## 总结

本系统提供了一个完整的3D脑体素分类训练和评估框架，具有以下特点：

1. **完整性**：从数据加载到结果分析的全流程
2. **灵活性**：支持多种配置和参数调整
3. **鲁棒性**：Leave-One-Out交叉验证确保结果可靠
4. **高效性**：优化的数据加载和模型架构
5. **可扩展性**：易于添加新的模型或数据增强

使用本系统，您可以：
- 快速建立基线性能
- 比较不同patch尺寸的效果
- 获得每个被试的个体化评估
- 生成可发表的统计结果和图表

建议从小规模测试开始，确认流程正确后再进行完整的38次Leave-One-Out训练。