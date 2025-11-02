# 重构版MRI大脑体素分割训练系统

## 📋 目录
- [系统概述](#系统概述)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [添加新模型](#添加新模型)
- [配置说明](#配置说明)

---

## 系统概述

这是一个重构的MRI大脑体素级别分割训练系统，支持多种深度学习模型。

### 主要特点

✅ **模块化设计**: 数据加载、模型定义、训练评估完全分离
✅ **灵活的模型切换**: 通过参数选择不同的模型，无需修改代码
✅ **易于扩展**: 添加新模型只需创建一个新的py文件
✅ **完整的分析工具**: 集成3D softmax保存和per-class性能分析
✅ **统一的配置**: 数据路径不变，只需改变模型名称

### 内置模型

1. **RegModel** (`reg_model`)
   - 深度全连接神经网络（Alex identical structure）
   - 4层隐藏层，每层4096神经元
   - 无BatchNorm，使用ReLU + Dropout

2. **ResNetMLP** (`resnet_mlp`)
   - 带残差连接的全连接神经网络
   - 使用ResidualBlock和BatchNorm
   - 更容易训练深层网络

3. **SimpleMLP** (`simple_mlp`)
   - 简单多层感知机
   - 3层隐藏层 [512, 256, 128]
   - 参数量少，训练速度快

4. **BrainVoxelKAN** (`kan`) ⭐
   - 基于Kolmogorov-Arnold网络的分类模型
   - 使用FastKAN实现，无需PCA
   - 默认3层隐藏层 [256, 128, 64]
   - 自动使用类权重处理类别不平衡
   - grid_size=8（可配置）

5. **DeepMLP** (`deep_mlp`) ⭐ **新增**
   - 超深度MLP模型，集成多种先进技术
   - 7层深度网络 [2048, 1536, 1536, 1536, 1536, 2048, 2048]
   - 特征交互层（双线性/三线性交互）
   - 残差连接 + 瓶颈结构
   - 多头自注意力机制
   - 混合激活函数（GELU/SiLU）
   - Shake-Shake正则化 + 随机深度
   - 自动使用类权重处理类别不平衡

---

## 项目结构

```
refactored_training/
├── models/                          # 模型定义目录
│   ├── __init__.py                 # 模型注册和管理
│   ├── reg_model.py                # RegModel (原始模型)
│   ├── resnet_model.py             # ResNetMLP (残差网络)
│   ├── simple_mlp.py               # SimpleMLP (简单MLP)
│   ├── kan_model.py                # BrainVoxelKAN (KAN模型) ⭐
│   └── deep_mlp.py                 # DeepMLP (超深度MLP) ⭐
│
├── utils/                           # 工具函数目录
│   ├── __init__.py
│   ├── data_loader.py              # 数据加载和预处理
│   ├── trainer.py                  # 训练和评估函数
│   ├── prediction.py               # 3D预测和重建
│   ├── visualization.py            # 可视化工具
│   └── class_weights.py            # 类权重计算 ⭐
│
├── train.py                         # 主训练脚本
├── run_training.sh                  # Shell训练脚本
├── README.md                        # 本文档
│
├── results/                         # 训练结果输出目录（自动创建）
└── logs/                            # 训练日志目录（自动创建）
```

---

## 快速开始

### 方法1: 多模型训练（推荐）⭐ **新功能**

一次性选择并训练多个模型，结果自动保存到独立文件夹：

```bash
# 进入重构目录
cd refactored_training

# 运行多模型训练脚本
./run_multiple_models.sh

# 脚本会提示您选择模型，例如:
# 您的选择: 1 3 4    # 训练 reg_model, simple_mlp, kan
# 您的选择: all      # 训练所有模型

# 快速测试（10 epochs）
EPOCHS=10 ./run_multiple_models.sh

# 自定义参数
EPOCHS=25 BATCH_SIZE=8192 ./run_multiple_models.sh
```

**特点**:
- 📋 交互式选择模型（可多选）
- 📁 每个模型独立的输出文件夹（模型名+时间戳）
- 📊 自动生成汇总报告
- 🔄 按顺序训练，失败可继续
- 💾 保存完整的训练结果（与原始流程一致）

详细使用说明请查看 [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md)

### 方法2: 单模型训练

使用原有的单模型训练脚本：

```bash
# 给脚本添加执行权限
chmod +x run_training.sh

# 训练RegModel（排除背景）
MODEL_NAME=reg_model ./run_training.sh

# 训练ResNetMLP（排除背景）
MODEL_NAME=resnet_mlp ./run_training.sh

# 训练SimpleMLP（排除背景）
MODEL_NAME=simple_mlp ./run_training.sh

# 训练KAN模型（自动使用类权重）⭐
MODEL_NAME=kan ./run_training.sh

# 训练DeepMLP模型（超深度MLP）⭐
MODEL_NAME=deep_mlp ./run_training.sh

# 训练RegModel（包含背景）
MODEL_NAME=reg_model INCLUDE_BACKGROUND=true ./run_training.sh
```

### 方法3: 直接使用Python脚本

```bash
# 训练RegModel
python train.py --model reg_model --epochs 25 --batch_size 8192

# 训练ResNetMLP
python train.py --model resnet_mlp --epochs 25 --batch_size 8192

# 训练SimpleMLP
python train.py --model simple_mlp --epochs 25 --batch_size 8192

# 训练KAN模型（自动使用类权重）⭐
python train.py --model kan --epochs 25 --batch_size 8192

# 训练DeepMLP模型（超深度MLP）⭐ 新增
python train.py --model deep_mlp --epochs 25 --batch_size 8192

# KAN模型自定义参数
python train.py --model kan --epochs 30 --grid_size 16

# 包含背景训练
python train.py --model reg_model --include_background --epochs 25
```

### 安装KAN模型依赖

KAN模型需要额外安装fastkan库：

```bash
pip install fastkan
```

---

## 使用说明

### 训练脚本参数

#### Python脚本 (train.py)

```bash
python train.py [OPTIONS]

必需参数:
  无 (所有参数都有默认值)

可选参数:
  --model {reg_model,resnet_mlp,simple_mlp}
                        选择模型 (default: reg_model)
  --root_dir PATH       数据集根目录
  --include_background  包含背景体素（默认排除）
  --exclude_features N [N ...]
                        要排除的特征索引（默认[14]）
  --epochs N            训练轮数 (default: 25)
  --batch_size N        批大小 (default: 8192)
  --lr FLOAT            学习率 (default: 0.00001)
  --weight_decay FLOAT  权重衰减 (default: 0.00001)
  --hidden_dim N        隐藏层维度（针对某些模型）
  --dropout_rate FLOAT  Dropout率
  --seed N              随机种子 (default: 42)
```

#### Shell脚本 (run_training.sh)

通过环境变量配置:

```bash
# 模型选择
MODEL_NAME=reg_model          # reg_model, resnet_mlp, simple_mlp

# 训练参数
EPOCHS=25                     # 训练轮数
BATCH_SIZE=8192               # 批大小
LEARNING_RATE=0.00001         # 学习率

# 数据参数
INCLUDE_BACKGROUND=false      # true 或 false

# 执行训练
./run_training.sh
```

### 使用示例

#### 示例1: 快速测试不同模型

```bash
# 测试RegModel（原始模型）
MODEL_NAME=reg_model EPOCHS=10 ./run_training.sh

# 测试ResNetMLP（残差网络）
MODEL_NAME=resnet_mlp EPOCHS=10 ./run_training.sh

# 测试SimpleMLP（轻量级）
MODEL_NAME=simple_mlp EPOCHS=10 ./run_training.sh
```

#### 示例2: 完整训练

```bash
# RegModel完整训练（25 epochs）
MODEL_NAME=reg_model EPOCHS=25 BATCH_SIZE=8192 ./run_training.sh

# ResNetMLP完整训练
MODEL_NAME=resnet_mlp EPOCHS=25 BATCH_SIZE=8192 ./run_training.sh
```

#### 示例3: 两种模式对比

```bash
# 排除背景模式
MODEL_NAME=reg_model INCLUDE_BACKGROUND=false ./run_training.sh

# 包含背景模式
MODEL_NAME=reg_model INCLUDE_BACKGROUND=true ./run_training.sh
```

#### 示例4: 自定义参数

```bash
# 使用Python脚本自定义更多参数
python train.py \
    --model resnet_mlp \
    --epochs 30 \
    --batch_size 4096 \
    --lr 0.0001 \
    --hidden_dim 3072 \
    --dropout_rate 0.4
```

---

## 添加新模型

添加新模型非常简单，只需三步：

### 步骤1: 创建模型文件

在 `models/` 目录下创建新的模型文件，例如 `my_custom_model.py`:

```python
#!/usr/bin/env python
# coding: utf-8

"""
MyCustomModel - 我的自定义模型
"""

import torch.nn as nn
import torch.nn.functional as F


class MyCustomModel(nn.Module):
    """
    自定义模型描述
    """

    def __init__(self, input_dim=42, num_classes=52,
                 hidden_dim=1024, dropout_rate=0.3):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        num_classes : int
            输出类别数
        hidden_dim : int
            隐藏层维度
        dropout_rate : float
            Dropout率
        """
        super(MyCustomModel, self).__init__()

        # 定义你的网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向传播
        """
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def get_model_config():
    """
    获取模型默认配置
    """
    return {
        'model_name': 'MyCustomModel',
        'hidden_dim': 1024,
        'dropout_rate': 0.3,
        'description': '我的自定义模型'
    }
```

### 步骤2: 注册模型

在 `models/__init__.py` 中注册新模型:

```python
# 导入新模型
from .my_custom_model import MyCustomModel, get_model_config as get_my_custom_model_config

# 添加到模型注册表
MODEL_REGISTRY = {
    'reg_model': {...},
    'resnet_mlp': {...},
    'simple_mlp': {...},
    'my_custom': {  # 新增
        'class': MyCustomModel,
        'config_fn': get_my_custom_model_config
    }
}

# 在 get_model() 函数中添加参数处理（如需要）
elif model_name == 'my_custom':
    model_params['hidden_dim'] = kwargs.get('hidden_dim', default_config['hidden_dim'])
    model_params['dropout_rate'] = kwargs.get('dropout_rate', default_config['dropout_rate'])
```

### 步骤3: 使用新模型

```bash
# 使用Shell脚本
MODEL_NAME=my_custom ./run_training.sh

# 或使用Python脚本
python train.py --model my_custom --epochs 25
```

就是这么简单！数据加载、训练、评估、保存等所有流程都会自动处理。

---

## 配置说明

### 数据目录结构

系统期望的数据目录结构:

```
ROOT_DIR/
├── FOR_001/
│   └── balanced_output/
│       ├── balanced_data_4d10000.nii.gz      # 4D特征数据
│       └── balanced_labels_3d10000.nii.gz    # 3D标签数据
├── FOR_002/
│   └── balanced_output/
│       ├── balanced_data_4d10000.nii.gz
│       └── balanced_labels_3d10000.nii.gz
└── ...
```

### 输出文件

训练完成后，会在 `results/` 目录生成:

1. **模型文件**: `{model_name}_bg_{excl/incl}_{timestamp}.pth`
   - 包含模型权重、配置、训练历史等

2. **训练曲线**: `training_history_{model_name}_bg_{excl/incl}_{timestamp}.png`
   - Loss、Accuracy、F1曲线图

3. **3D Softmax**: `test_softmax_3d_{subject}_bg_{excl/incl}_{timestamp}.nii.gz`
   - 测试集的3D softmax概率分布

4. **Per-class分析**: `per_class_analysis_{model_name}_bg_{excl/incl}_{timestamp}/`
   - 详细的per-class性能分析图表和数据

### 模型对比

| 模型 | 参数量 | 训练速度 | 适用场景 |
|------|--------|----------|----------|
| RegModel | ~84M | 中等 | 原始基准模型 |
| ResNetMLP | ~25M | 较快 | 需要更好的梯度流动 |
| SimpleMLP | ~1M | 很快 | 快速实验、资源受限 |
| KAN | ~5-10M | 中等 | 处理类别不平衡、探索性实验 |
| DeepMLP | ~150-200M | 较慢 | 追求最高性能、大规模实验 |

---

## 常见问题

### Q1: 如何只改变数据路径？

**A**: 使用 `--root_dir` 参数:

```bash
python train.py --model reg_model --root_dir /path/to/your/data
```

或在Shell脚本中设置环境变量:

```bash
ROOT_DIR=/path/to/your/data MODEL_NAME=reg_model ./run_training.sh
```

### Q2: 如何对比不同模型的性能？

**A**: 使用相同的参数运行不同模型:

```bash
# 训练所有模型
for model in reg_model resnet_mlp simple_mlp; do
    MODEL_NAME=$model EPOCHS=25 ./run_training.sh
done
```

然后对比 `results/` 目录中的结果。

### Q3: 如何恢复训练？

**A**: 当前版本不支持直接恢复。如需要，可以修改 `train.py` 添加checkpoint加载功能。

### Q4: 如何调整模型特定参数？

**A**: 使用Python脚本的参数:

```bash
# 调整RegModel的隐藏层维度
python train.py --model reg_model --hidden_dim 2048

# 调整Dropout率
python train.py --model resnet_mlp --dropout_rate 0.5
```

---

## 优势总结

### 相比原始代码的改进

1. **模块化**: 代码分离清晰，易于维护
2. **灵活性**: 切换模型只需改变一个参数
3. **可扩展**: 添加新模型无需修改现有代码
4. **统一接口**: 所有模型使用相同的训练流程
5. **易于实验**: 快速对比不同模型的性能

### 保持的功能

✅ 所有原始功能都保留
✅ 数据格式完全兼容
✅ Patient-wise标准化
✅ 3D softmax保存
✅ Per-class性能分析
✅ 训练历史可视化

---

## 依赖环境

```bash
# Python 3.7+
pip install torch
pip install nibabel
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install numpy
pip install pandas
```

---

## 联系与支持

如有问题或建议，请查看原始文档或联系开发团队。

祝训练顺利！ 🚀
