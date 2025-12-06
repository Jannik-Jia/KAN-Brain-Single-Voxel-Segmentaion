# KAN 于 Alex7T-102labels

## 1. 在论文中的角色
基于 notebook 的尝试，训练分层 Kolmogorov–Arnold Networks（FastKAN）用于 Alex 7T 数据集的体素级组织分类。提供超越传统模型的首批神经基线，测试按模态分组的专家 KAN 模块的可行性与精度。

## 2. 代码文件与入口
- 102LABEL_1DKAN_brainvoxel_分层分类.ipynb：核心原型，定义 ExpertKAN（基于 FastKAN）、超参、数据加载、特征选择与训练循环。
- V2_102LABEL_1DKAN_brainvoxel_分层分类_多方法尝试.ipynb、V3_102LABEL_1DKAN_brainvoxel_分层分类_多方法尝试_加速.ipynb、V4_102LABEL_数据结构分层分类_多方法尝试_加速.ipynb：迭代版，增加加速、方法与数据结构调整。
- v2_102LABEL_1DKAN_brainvoxel_完全分离的数据集创建方法.ipynb：数据重组/干净划分准备。
- 共享常量与 config.py 对齐（MODEL_NAME、特征索引、SAVE_PATH 模式）。

## 3. 数据集与标签
- Alex 超多模态 7T 数据集；341 通道体素签名（diffusion 0–14，QTI 15–224，CEST 225–340）。
- 102 个硬标签映射到 7 个大类用于粗分类；采样比例常设 0.1 以提速；train/test/val 的 npy 分片。
- 标签作为整型，偶有 one-hot；无软标签处理。

## 4. 预处理流程
- 可选 PCA（APPLY_PCA=True）、归一化标志 NORM，以及按模态组的 SelectKBest F 统计特征选择（常见 10/30/20 特征）。
- 增强预处理使用 Robust 缩放；数据创建 notebook 专注完全分离的 train/val/test 划分。
- 无显式缺失值处理；假设体素有效。

## 5. 模型结构
- ExpertKAN 封装 FastKAN，layers_hidden=[input_dim, hidden_dim, num_classes]，grid_size=10。
- 每组隐藏维：diffusion 64，QTI 128，CEST 64；旨在作为模态专家，可融合（融合/门控策略在 notebook 中未完全明确）。
- 训练超参：EPOCH=50，BATCH_SIZE=640，LR=1e-3，WEIGHT_DECAY=1e-6，RANDOM_SEED=666。

## 6. 训练配置
- 损失：CrossEntropyLoss；优化器 Adam，使用上述 lr/weight_decay。
- 推测使用 DataLoader；部分单元将 USE_SAMPLING=True 以 10% 采样加速。
- 无显式调度/早停；检查点路径由类似 config 的 SAVE_PATH 指定。

## 7. 评估指标与输出
- 分类准确率为主；绘制 sklearn 混淆矩阵；导入 ROC/AUC 用于按类曲线（越高越好）。
- 复用传统流水线的特征选择/聚类代码做对比；当 SAVE_PATH 生效时，图与日志保存到 Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<timestamp>/。

## 8. 论文写作解读
- 建立 FastKAN 体素分类器的可行性，并将模态专家与传统基线比较。可支撑关于超多模态体素签名的神经架构章节，并与 MLP/TabNet 实验形成对照。

## 9. 限制与开放问题
- 专家 KAN 的融合尚未明确实现；依赖 notebook 执行，复现性较弱。
- 数据路径硬编码；由于采样与时长限制，运行可能未覆盖完整 train/val/test。
- 缺少校准指标与软标签处理；与传统基线的对比结果需要进一步汇总。
