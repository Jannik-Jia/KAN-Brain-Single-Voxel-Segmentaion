# KAN-RFFeatures 于 Alex7T-102labels

## 1. 在论文中的角色
- 消融：研究随机森林特征选择（77 维）加 KAN 正则是否能提升 Alex 超多模态 7T 数据集的 102 类体素分类。
- 测试在 KAN 之前用传统特征工程降维，与基于 PCA 或全特征模型对比。

## 2. 代码文件与入口
- `1DKAN_随机森林特征选择_完全分离的数据集创建方法_102分类_L1.ipynb`：从 `rf_selected_features.h5` 读取 RF 选特征，做 z-score 归一化，可选平衡训练数据，训练 FastKAN `[77,128,102]` 模型并支持调度器。
- 包含检查模态组索引（diffusion/QTI/CEST）、准备平衡数据集、监控 KAN 权重/熵统计的辅助函数。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T。
- 输入：由先前随机森林挑选的 77 个特征（按 diffusion/QTI/CEST 分组索引）；数据以 HDF5 划分（`all/train`、`all/val`、`all/test`）。
- 标签：102 类分区；整型标签，无软目标；背景处理未明示（假定在 RF 导出中已去除）。
- 划分：HDF5 内预计算 train/val/test；可选平衡到 `TARGET_SAMPLES`（3000）每类，通过下采样（无过采样，`MAX_MULTIPLIER=1`）。

## 4. 预处理流程
- 使用训练集均值/方差做 z-score，应用于 val/test；因已做特征选择，关闭 PCA。
- 可选下采样做类别平衡；增强钩子存在但默认不用。

## 5. 模型结构
- FastKAN，层 `[77, 128, 102]`，样条网格 8；无 dropout 或 batch norm。
- 超参含 L1 与熵正则系数（`LAMBDA_L1=0.005`，`LAMBDA_ENTROPY=2.0`），但其加入损失的实现不明确（效果 UNKNOWN）。

## 6. 训练配置
- 损失：`nn.CrossEntropyLoss(ignore_index=-1)`（背景忽略占位）；默认不使用类别权重。
- 优化器：Adam（`lr=2e-5`, `weight_decay=1e-3`）；batch 128；训练 100 轮；每 3 轮验证。
- 学习率调度与多分类基线相同（MultiStep [25,50,75], gamma 0.5；代码提供 cosine/plateau 备选）。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1 及 train/test/val 混淆矩阵。
- 预期输出目录 `Results/BrainVoxel_102Class_RF/BrainVoxel`（检查点/图表），但训练权重/CSV 未在仓库存档。

## 8. 论文写作解读
- 作为特征选择消融：比较 RF 选的 77 维输入与 341 全维输入，看传统特征过滤 + KAN 正则是否稳定多分类性能或降低计算。
- 适合“传统特征选择 vs PCA”方法/实验小节。

## 9. 限制与开放问题
- RF 选特征文件路径为绝对路径且不在库中；缺数据无法复现。
- 正则项虽定义但未明确加入损失；其实证效果 UNKNOWN。
- 未记录指标；77 维子集相对 PCA/全特征的效果尚待验证。
