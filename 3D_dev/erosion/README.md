# 38-Fold 交叉验证训练系统

一个完整的38折交叉验证训练流程，包含详细的指标监控、ONNX导出和全面的日志记录。

## 功能特点

- **灵活训练**: 支持训练单个fold或全部38个fold
- **详细指标**: 逐epoch监控训练集/验证集/测试集的loss、accuracy和macro-F1
- **全面评估**: 每个类别的Precision/Recall/F1以及macro/micro/weighted平均
- **ONNX导出**: 模型自动导出为ONNX格式便于部署
- **完整日志**: 所有输出保存到日志文件，确保可复现性
- **分析工具**: 提供可视化和报告生成脚本进行结果分析
- **多GPU支持**: 自动检测并使用DataParallel进行多GPU训练
- **混合精度**: 支持FP16自动混合精度训练（1.5-2x加速）

## 快速开始

### 安装依赖

```bash
# 安装Python依赖包
pip install -r requirements.txt
```

### 路径配置

所有路径都使用相对路径，便于服务器部署：

```json
{
  "paths": {
    "dataset_index": "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/dataset_index_validated.json",  // 数据集索引
    "output_base": "./output"                               // 输出目录
  }
}
```

**重要**：上传到服务器前请确认：
1. `dataset_index_validated.json` 在项目根目录
2. 数据文件路径与JSON中的路径一致
3. 使用相对路径或通过 `--data-dir` 指定本地数据目录

### 训练

#### 训练单个fold（例如fold 5）
```bash
./run_training.sh single 5
```

#### 训练全部38个fold
```bash
./run_training.sh all
```

#### 快速测试（使用fold 38，仅3个epoch）
```bash
./run_training.sh test
```

### 直接使用Python脚本

```bash
# 训练单个fold（基础）
python train_38fold.py --fold 5 --epochs 25

# 训练单个fold（双GPU + 混合精度，推荐）
python train_38fold.py --fold 5 --epochs 25 --use-amp

# 训练单个fold（最优性能，2×A6000）
python train_38fold.py --fold 5 --epochs 25 --use-amp --batch-size 768

# 训练所有fold
python train_38fold.py --epochs 25

# 训练所有fold（双GPU加速）
python train_38fold.py --epochs 25 --use-amp --batch-size 768

# 自定义配置
python train_38fold.py \
    --fold 10 \
    --epochs 30 \
    --batch-size 512 \
    --lr 0.00002 \
    --use-amp \
    --output-dir ./custom_output

# 强制单GPU训练
python train_38fold.py --fold 5 --no-multi-gpu
```

### GPU训练参数说明

| 参数 | 说明 | 推荐值（2×A6000） |
|------|------|------------------|
| `--use-amp` | 启用FP16混合精度训练 | ✅ 推荐启用 |
| `--batch-size` | 批量大小 | 768（双GPU）|
| `--no-multi-gpu` | 禁用多GPU | 仅调试时使用 |
| `--num-workers` | 数据加载线程数 | 4（默认） |

## 输出目录结构

```
output/
├── logs/
│   ├── train_fold1_20240326_143022.log  # 训练日志
│   └── ...
├── models/
│   ├── model_fold1.pth          # PyTorch模型
│   ├── model_fold1.onnx         # ONNX模型
│   └── ...
├── history/
│   ├── history_fold1.json       # 训练历史记录
│   └── ...
└── cross_validation_results.json # 汇总结果
```

## 结果分析

训练完成后，分析结果：

```bash
# 生成分析报告和图表
python analyze_results.py --results-dir ./output/all_folds_20240326_143022

# 绘制特定fold的训练曲线
python analyze_results.py \
    --results-dir ./output/all_folds_20240326_143022 \
    --plot-fold 5
```

## 监控的指标

### 每个Epoch
- **训练集/验证集/测试集**: Loss、Accuracy、Macro-F1
- **测试集详细指标**（每5个epoch）：
  - 每个类别的Precision/Recall/F1
  - Macro/Micro/Weighted平均
  - Gross accuracy（总体准确率）

### 最终输出
- 模型文件（.pth和.onnx格式）
- 完整训练历史
- 分类报告
- 交叉验证统计

## 配置参数

### 默认参数
- **Epochs**: 25
- **Batch Size**: 128
- **学习率**: 1e-5
- **L2正则化**: 1e-5
- **Dropout**: 0.5
- **网络架构**: 4x4096全连接层

### 数据格式
- **输入**: 351维特征
- **输出**: 102个脑区域
- **数据**: 38个被试（1D MRI数据）

## 模型架构

```python
输入 (351) → 全连接(4096) → ReLU → Dropout(0.5)
          → 全连接(4096) → ReLU → Dropout(0.5)
          → 全连接(4096) → ReLU → Dropout(0.5)
          → 全连接(4096) → ReLU → Dropout(0.5)
          → 全连接(102) → Softmax
```

## ONNX导出

模型自动导出为ONNX格式，具有以下特点：
- 支持动态batch size
- 优化推理性能
- 兼容ONNX Runtime

### 使用ONNX模型

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("model_fold1.onnx")

# 准备输入数据 (batch_size, 351)
input_data = np.random.randn(10, 351).astype(np.float32)

# 运行推理
outputs = session.run(None, {"input": input_data})
predictions = outputs[0]  # (10, 102)
```

## 日志记录

所有训练信息同时记录到控制台和文件：
- 训练配置
- 每个epoch的指标
- 详细的测试指标
- 最终结果汇总

日志文件带时间戳并保存在输出目录中。

## 性能说明

### 训练时间对比

| 配置 | 单个Fold | 38个Fold |
|------|----------|----------|
| 单GPU（V100/A6000） | ~10-15分钟 | ~6-10小时 |
| 双GPU（2×A6000） | ~5-8分钟 | ~3-5小时 |
| 双GPU + FP16 | ~3-5分钟 | ~2-3小时 |

### 硬件需求

| 组件 | 最小需求 | 推荐配置 |
|------|----------|----------|
| GPU显存 | 8GB | 48GB×2（A6000） |
| 批量大小 | 128 | 768（双A6000） |
| 系统内存 | 32GB | 64GB |
| 磁盘空间 | 20GB | 50GB |

### 性能优化建议

1. **使用混合精度**：`--use-amp` 可获得1.5-2x加速
2. **增大批量大小**：充分利用A6000的48GB显存
3. **多GPU并行**：自动检测并使用所有可用GPU
4. **数据预加载**：使用`--num-workers 4`优化I/O

## 问题排查

### 内存不足
- 减小batch size：`--batch-size 64`
- 使用CPU：添加`--cpu`参数

### 数据加载问题
- 检查数据集索引路径
- 验证.mat文件可访问性
- 使用`--data-dir`指定本地数据目录

### Fold训练失败
- 查看具体错误日志
- 重试单个fold
- 验证数据完整性

## 引用

基于Alex的PyTorch实现的4x4096全连接网络用于脑区域分割。