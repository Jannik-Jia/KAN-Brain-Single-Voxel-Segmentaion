# Deep4x4096 于 Alex7T-RegionCLS

## 1. 在论文中的角色
- 作为 Alex 超多模态 7T 数据集（German et al. 2021）的主要体素级基线分类器，并探查是否需要 subject embedding。
- 衡量简单全连接网络能否区分约 101–102 个脑区，以及在跨受试上的泛化程度（随机划分 vs LOSO），为更高级的嵌入策略做铺垫。

## 2. 代码文件与入口
- embedding_project_4_phase/orchestrator.py：运行 Phase 0–4 的端到端或分阶段流程并记录日志。
- embedding_project_4_phase/phase0_data_preparation/main.py 及 data_splitter.py、data_validator.py：加载 TRAIN38_no_label43.mat，执行多受试留出划分、缩放、标签映射、特征/区域 QC，保存 npz/json 统计。
- embedding_project_4_phase/phase1_subject_analysis/main.py 配合 global_analyzer.py、region_analyzer.py、multilabel_patient_analyzer.py、conditional_umap_visualizer.py：分析受试差异、区域特异性、UMAP 聚类。
- embedding_project_4_phase/phase2_separability/main.py 及 baseline_tester.py、loso_evaluator.py、embedding_assessor.py：训练逻辑回归与 Deep4x4096 基线，做 LOSO 评估，区域级嵌入需求评分；可选保存模型。
- embedding_project_4_phase/common/config.py、common/deep_network.py、common/data_io.py、common/visualization.py：超参、4×4096 模型定义、IO 辅助、绘图；run.sh 负责启动与检查点/日志管理。
- 输出位于 embedding_project_4_phase/data_exchange/phase{0..3}_output 及子目录（visualizations, trained_models, *results.json/npz）。

## 3. 数据集与标签
- Alex 超多模态 7T 数据集；默认输入文件 TRAIN38_no_label43.mat。
- 输入特征：341 通道（15 个 QTI 参数、210 个原始 b-tensor 样本、4 个 CEST 参数、112 个 Z-spectrum 点）。
- 标签：脑区 one-hot ID（≈101–102 类，去掉标签 43），源自 FreeSurfer 风格分割；硬标签。
- 划分：按受试的 Multi-Subject-Out（train 1–30，val 31–37，test 38）；LOSO 使用全部 train+val 受试。
- 采样：逐体素训练；区域级分析在每个 (subject, region) ≥50 体素时做均值聚合。

## 4. 预处理流程
- 在训练体素上拟合 StandardScaler，应用于 val/test；Scaler 保存供复用。
- 数据完整性检查：NaN/Inf、恒定特征检测、区域/样本覆盖、类别平衡摘要；跟踪特征组。
- 区域感知数据构建器在 (subject, region) ≥50 样本时做体素均值。
- 输出元数据（data_statistics.json、label_mapping.json、feature_group_analysis.json）供后续阶段使用。

## 5. 模型结构
- Deep4x4096：全连接堆叠 341 → 4096×4 → n_classes，ReLU + dropout 0.5，最终线性 logits；与早期 alex TensorFlow 模型类似。
- LogisticRegression 基线：C=0.1，max_iter=1000；可选区域级模型。
- 区域级深度变体可选更小 batch (64) 与 15 轮训练。
- 超参集中在 embedding_project_4_phase/common/config.py (ALEX_HYPERPARAMS)。

## 6. 训练配置
- 损失：交叉熵，权重带显式 L2/kernel 正则（weight_decay 1e-5）。
- 优化器：Adam；lr 1e-5，batch_size 128，训练 25 轮（全局），dropout 0.5；固定 torch 随机种子以复现。
- 验证：每轮跟踪 accuracy/F1；区域级运行使用 80/20 划分；LOSO 评估留一受试；可选保存训练模型。
- 无数据增强；GPU/CPU 自动选择。

## 7. 评估指标与输出
- 分类：验证集 accuracy、macro/weighted F1；区域级 accuracy/F1；LOSO 平均准确率与泛化差（基线 – LOSO）。
- 嵌入需求探针：区域嵌入必要性得分、关键/高优先级区域数量、深度网络权威分（性能、收敛、泛化）。
- Phase 1 的受试差异指标：PCA 解释方差、距离/相关矩阵、silhouette 分数、受试/区域方差比、UMAP 聚类得分。
- 可视化：性能对比柱状图、泛化差图、嵌入需求热图、受试相似性热图、PCA/UMAP 散点；保存为 .png 至各 phase 输出目录。
- 日志/元数据：baseline_results.json、loso_results.json、deep_network_analysis.json、region_classification_performance.json、region_embedding_needs.json、phase*_scores.json、validation_report.json、npz 特征摘要。

## 8. 论文写作解读
- 检验经典的 4×4096 全连接网络与简单基线能否在 Alex 超多模态 7T 上做体素级脑区分类，以及留出受试时性能下降多少。
- 提供定量证据：(i) 基线精度上限，(ii) 跨受试泛化差距，(iii) 哪些脑区最需要受试特定嵌入。
- 结果可支撑基线性能、泛化局限与 subject embedding/高级架构动机的章节，附模态可分性与嵌入需求热图。

## 9. 限制与开放问题
- 数值结果未写在代码中；需查看实际运行生成的 JSON/NPZ。
- 假设 TRAIN38_no_label43.mat 格式与硬编码受试 ID；其他数据或标签方案需改配置。
- GPU 可选；缺乏加速时长跑成本高。
- 尽管文件夹命名提到 TabNet/KAN，本流程仅 logistic 与 Deep4x4096 可用。
