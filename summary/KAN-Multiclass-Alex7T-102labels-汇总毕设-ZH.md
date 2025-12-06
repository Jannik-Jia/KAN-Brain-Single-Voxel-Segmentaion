# KAN-Multiclass 于 Alex7T-102labels

## 1. 在论文中的角色
- 将 KAN 扩展到 Alex 超多模态 7T 数据集的完整 102 类体素分类。
- 测试基于样条的激活结合 PCA/标准化与类别平衡，是否能超越一对多检测器。
- 为后续多模态架构与多分类混淆趋势的报告提供对比基线。

## 2. 代码文件与入口
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类.ipynb`：主 multiclass FastKAN 流水线，含数据重构、可选 PCA、平衡采样、两层 KAN 隐藏层 `[256,128]`、grid size 8，以及多步学习率调度。
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类_L1.ipynb`：无 PCA（341 维原始特征），隐藏 128，较强 weight decay (1e-3)，并监控 KAN 权重幅度/熵（偏重正则）。
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类_多重尝试/...`：主 notebook 的额外试验副本。
- 支撑单元实现数据集合并/切分（`merge_and_shuffle_datasets`, `split_merged_dataset`）、类别权重计算与特征重要性绘制。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T。
- 输入：341 通道体素签名；PCA 可选（默认开启，自动定组件数），若关闭则用原始特征。
- 标签：102 个解剖类别；背景设为 0，映射为 -1 在 `CrossEntropyLoss` 中忽略。
- 划分：原 train/val 合并后重新 60/20/20 切分到 `restructured/{train,test,val}`；平衡时通过简单复制将少数类增至 `TARGET_SAMPLES`（10000），并限制放大量。

## 4. 预处理流程
- 可选 PCA 后归一化；若无 PCA，则对原始特征做 z-score。
- 数据重构生成按类 `.npy` 块与 `label_index.txt` 供加载；背景体素删除。
- 通过过采样做类别平衡；可选在损失中加入类别权重（notebook 默认注释）。

## 5. 模型结构
- FastKAN，层 `[feature_dim, 256, 128, 102]`，样条网格 8–10（可调）；无 dropout。
- 另一简化变体仅用单隐藏 128（L1 notebook）。
- 通过输入样条权重绘制特征重要性；背景在损失中忽略。

## 6. 训练配置
- 损失：`nn.CrossEntropyLoss`，可选类别权重，`ignore_index=-1` 处理背景。
- 优化器：Adam（`lr=2e-5`, `weight_decay=5e-3` 或 `1e-3`）；batch 128；训练 100 轮；每 3 轮验证。
- 学习率调度：MultiStep 里程碑 [25,50,75]，gamma 0.5（代码中可切换 cosine/plateau）。
- 检查点命名沿用二分类流程；以最高 accuracy/F1 选最佳。

## 7. 评估指标与输出
- 指标：总体 accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1；混淆矩阵。
- 可视化：数据分布、PR/ROC 曲线、主成分/特征的重要性条形图。
- 预期输出目录 `Results/BrainVoxel_102Class/BrainVoxel`（检查点/图表）；实际数值未存于仓库。

## 8. 论文写作解读
- 评估 KAN 以全部 341 模态做端到端 102 类体素分类的可行性，强调类别失衡处理与基于 PCA 的压缩。
- 展示从一对多到完整多分类时性能变化，适合 Methods/Experiments 小节中的“全分区 KAN”对比。
- 特征重要性可用于附录中的模态级讨论。

## 9. 限制与开放问题
- 需外部绝对路径与 `.npy` 数据块；仅凭仓库无法复现。
- 未记录 accuracy/F1/κ；调度与平衡的实际效果 UNKNOWN。
- 正则（L1/熵监控）属于探索性质，报告不完整。
