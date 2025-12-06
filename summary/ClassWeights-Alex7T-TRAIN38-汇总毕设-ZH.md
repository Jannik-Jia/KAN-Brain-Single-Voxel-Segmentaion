# ClassWeights 于 Alex7T-TRAIN38

## 1. 在论文中的角色
针对 German et al. 2021 的 Alex 超多模态 7T 数据集（TRAIN38.mat 子集）进行类别失衡分析并生成类别权重，为体素级分类器提供平衡、逆频次、sqrt/log、Class-Balanced Effective Number 以及 prob_idx 感知的权重，用于稳定稀有组织的交叉熵或 focal loss。作为下游 FC/MLP/KAN 或 TabNet 的预处理/校准工具，而非独立分类器。

## 2. 代码文件与入口
- data_analysis/train38_weighting.ipynb：端到端加载 TRAIN38.mat，z-score 特征，分析失衡，计算多种权重策略，生成报告/图表，并通过 analyze_mat_file 导出可直接用于 PyTorch 的权重。
- data_analysis/Effective Number权重计算.ipynb：轻量的 Effective Number 工具（EffectiveNumberWeights、compute_class_weights_from_labels、create_pytorch_weighted_loss），演示独立权重计算与 focal-loss 封装。

## 3. 数据集与标签
- 数据集：TRAIN38.mat（代码路径：/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat），假定来源于 German et al. 2021 的 Alex 超多模态 7T；包含 data（特征）、region（标签）、prob_idx（分组索引）。
- 输入维度：特征转置为 [n_samples, n_features]，预期约 341 通道（代码在归一化后打印“341个特征维度”）；具体样本数取决于 MAT 文件（库中未存）。
- 标签空间：来自 region 的硬标签；类别数量由标签范围推断；记录缺失类别；无软标签。
- 划分：分析在完整 MAT 上进行；未编码 train/val/test 划分——用户需对齐所用划分以避免泄漏。

## 4. 预处理流程
- 读取 HDF5/MAT 键 data、region、prob_idx，转置为行主样本。
- 对每个特征做 z-score 归一化（保留原始副本对比）；对零方差特征发出警告并保留原值。
- 质量检查：NaN/Inf、零行、数据范围、prob_idx 覆盖；可加载外部 YAML/JSON/NPY/NPZ/CSV 权重配置。
- 无降维；报告特征相关性（>0.9）与异常值计数供参考。

## 5. 模型结构
- 非网络；提供可嵌入下游分类器的权重策略。
- 策略：sklearn balanced、逆频次、sqrt-balanced、log-balanced、Class-Balanced Effective Number（Cui et al. 2019，自动 β 选择并含 β 变体）、prob_idx 感知平均、外部权重合并。自定义策略做均值归一化；Effective Number 保持原始 1/E_n。
- 损失封装示例：PyTorch CrossEntropyLoss 或小型 FocalLoss（γ=2.0）注入权重。

## 6. 训练配置
- 无优化器/epoch 循环；专注于准备供后续训练使用的 CLASS_WEIGHTS_TENSOR。
- analyze_mat_file 负责加载 → 统计 → 权重计算 → 报告/绘图 → 导出（recommended_class_weights.py，或 weights_<strategy>.json/npy/npz/csv）。
- Effective Number 的 β 由类别计数自动调节（细/粗标签空间的启发式不同）；也计算手动列表 [0.9, 0.99, 0.999, 0.9999]。

## 7. 评估指标与输出
- 数据集统计：样本数、特征数、dtype、内存、标签格式/范围、prob_idx 计数；平衡度通过失衡比接近 1 来衡量。
- 类别分布：每类最小/均值/中位/最大样本、分位数、缺失类列表、失衡等级（轻度→极端）；prob_idx 维度的类别覆盖。
- 特征统计：零/低方差计数，高相关对（>0.9），各特征异常值数，归一化效果诊断。
- 权重诊断：各策略的范围/均值，推荐的 Effective Number β。
- 输出：文本报告 comprehensive_analysis_report.txt，可视化（class_distribution.png, weight_strategies.png, feature_analysis.png, prob_idx_analysis.png，可选 normalization_comparison.png），导出的权重（recommended_class_weights.py, weights_<strategy>.json/.npy/.npz/.csv）。

## 8. 论文写作解读
这些 notebook 记录了 Alex 超多模态 7T 体素数据的失衡特征，并提供稳健的类别权重以稳定体素级分类器。可支撑“类别失衡处理与损失校准”章节，对比未加权基线与 Effective Number/prob_idx 感知加权在 FC/MLP/KAN/TabNet 训练中的差异。生成的报告与图表可作为失衡严重程度与 β 选择依据的证据。

## 9. 限制与开放问题
- MAT 数据未随库提供；精确样本数、类别数及 prob_idx 语义未知。
- 权重在全数据上计算；用户需在训练集上重算以避免泄漏。
- 数据路径硬编码；需根据本地存储调整，并验证 341 特征假设与当前预处理一致。
- 此处无下游训练或校准指标；对 accuracy/校准的影响需另行实验。
