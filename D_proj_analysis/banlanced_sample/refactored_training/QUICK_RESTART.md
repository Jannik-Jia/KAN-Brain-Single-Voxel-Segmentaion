# 快速重启指南 - 符号链接问题已修复

## ✅ 最新修复（2025-11-03）

**问题**: `ln: failed to create symbolic link './results': Operation not supported`

**原因**: JupyterHub/Docker环境中的文件系统不支持符号链接

**修复**: 脚本已更新，不再使用符号链接，改用文件移动策略

---

## 🚀 立即开始训练

### 方案1: 快速测试（推荐）

```bash
cd /home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/refactored_training

# 快速测试（5 epochs，约10-15分钟）
EPOCHS=5 ./run_multiple_models.sh
```

**选择模型**: 输入 `5` (DeepMLP)

### 方案2: 完整训练

```bash
# 完整训练（25 epochs）
./run_multiple_models.sh
```

**选择模型**:
- 输入 `5` - 只训练DeepMLP
- 输入 `4 5` - 训练KAN和DeepMLP
- 输入 `all` - 训练所有模型

---

## 📊 预期输出

训练成功后，你应该看到：

```
✅ 模型 deep_mlp 训练完成!
   耗时: XX分XX秒

移动训练结果到目标目录...

调试: 检查目录 ./training_runs/deep_mlp_bg_excl_XXXXXXXX/results
调试: 目录内容:
total XXX
-rw-r--r--  deep_mlp_bg_excl_*.pth
-rw-r--r--  test_softmax_3d_*.nii.gz
-rw-r--r--  training_history_*.png
drwxr-xr-x  per_class_analysis_deep_mlp_*

✓ 模型文件: 1
✓ 3D Softmax: 1
✓ 训练曲线: 1
✓ Per-class分析: 1
```

---

## 🔍 训练过程中会看到

### 1. 模型创建
```
🧠 创建模型: deep_mlp
输入维度: 41
输出类别数: 52
总参数量: 70,460,468
✓ visualization_toolkit loaded successfully  ← 这个很重要！
```

### 2. 类权重计算
```
⚖️ 计算类权重以处理类别不平衡...
⚠️ 以下类别没有训练样本: [0]  ← 这是正常的
   这些类别的权重将被设置为0
✅ 使用加权损失函数 (Weighted CrossEntropyLoss)
```

### 3. 训练进度
```
🚀 开始训练 (25 epochs)...
Epoch 1/25: 100%|████████| 635/635 [08:42<00:00, 1.22it/s]
Train Loss: 0.8234, Acc: 0.7456, F1: 0.7123
Test Loss: 0.9123, Acc: 0.7234, F1: 0.6987
```

### 4. Per-class分析（最后）
```
💾 Per-class分析结果已保存到: results/per_class_analysis_deep_mlp_...
   生成了12-15个可视化图表和详细指标
```

---

## 🐛 如果遇到问题

### 问题1: 还是提示符号链接错误

**不应该发生** - 如果还有这个错误，请检查：

```bash
# 检查脚本是否最新
grep "文件移动策略" run_multiple_models.sh

# 应该看到类似输出，否则请重新下载脚本
```

### 问题2: GPU内存不足

```bash
# 减小batch_size
BATCH_SIZE=4096 ./run_multiple_models.sh

# 或使用更简单的配置
python train.py --model deep_mlp --batch_size 2048 --use_trilinear False
```

### 问题3: visualization_toolkit还是没找到

```bash
# 检查父目录
ls -la ../visualization_toolkit/per_class_analyzer.py

# 如果不存在，需要确保visualization_toolkit在正确位置
```

### 问题4: 类别0有大权重

**已修复** - 现在类别0权重会被设置为0，不会影响训练

---

## 📁 训练结果位置

```
./training_runs/
└── deep_mlp_bg_excl_20251103_XXXXXX/
    ├── results/
    │   ├── deep_mlp_bg_excl_*.pth          # 模型权重
    │   ├── test_softmax_3d_*.nii.gz        # 3D Softmax
    │   ├── test_softmax_info_*.json        # Softmax信息
    │   ├── training_history_*.png          # 训练曲线
    │   └── per_class_analysis_deep_mlp_*/  # Per-class分析
    │       ├── per_class_detailed_metrics.csv
    │       ├── per_class_complete_analysis.json
    │       ├── comprehensive_per_class_analysis.png
    │       └── ... (10+个可视化图表)
    ├── logs/
    │   └── training_deep_mlp_*.log         # 完整日志
    └── SUMMARY.txt                          # 汇总报告
```

---

## ⏱️ 预期训练时间

| 模型 | 5 epochs | 25 epochs | GPU内存 |
|------|----------|-----------|---------|
| SimpleMLP | 2-3分钟 | 10-15分钟 | ~2-3GB |
| KAN | 10-15分钟 | 50-75分钟 | ~4-6GB |
| DeepMLP | 30-45分钟 | 2.5-4小时 | ~12-16GB |

---

## 💡 训练后查看结果

### 查看训练曲线
```bash
# 在Jupyter中查看
from IPython.display import Image
Image('./training_runs/deep_mlp_bg_excl_*/results/training_history_*.png')
```

### 查看Per-class分析
```bash
# 列出所有图表
ls ./training_runs/deep_mlp_bg_excl_*/results/per_class_analysis_*/

# 在Jupyter中查看
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open('./training_runs/deep_mlp_bg_excl_*/results/per_class_analysis_*/comprehensive_per_class_analysis.png')
plt.figure(figsize=(20, 15))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### 查看详细指标
```bash
# CSV格式
import pandas as pd
df = pd.read_csv('./training_runs/deep_mlp_bg_excl_*/results/per_class_analysis_*/per_class_detailed_metrics.csv')
print(df.sort_values('f1_score', ascending=False).head(10))
```

---

## 📝 相关文档

- `ALL_FIXES_SUMMARY.md` - 所有修复总结
- `DEEP_MLP_SETUP.md` - DeepMLP详细配置
- `MULTI_MODEL_GUIDE.md` - 多模型训练指南
- `DEEP_MLP_FIX.md` - DeepMLP修复详情
- `MULTI_MODEL_FIX.md` - 多模型训练修复详情

---

## ✅ 修复清单

所有问题已修复：
- ✅ DeepMLP维度不匹配
- ✅ 类权重异常（类别0）
- ✅ 符号链接不支持（最新修复）
- ✅ visualization_toolkit导入

**现在可以安全地开始训练了！**

---

**最后更新**: 2025-11-03
**状态**: ✅ 所有已知问题已修复，可以开始训练

开始训练吧！🚀
