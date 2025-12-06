# SubjectEmbedding 于 Alex7T-EmbeddingFeasibility

## 1. 在论文中的角色
- 面向 German et al. 2021 的 Alex 超多模态 7T 数据集的端到端脑区感知 subject embedding 可行性分析，含检查点与决策逻辑。
- 评估受试差异、可分性、嵌入设计并给出实现建议，同时保留 alex 4×4096 稠密网络作为神经基线。

## 2. 代码文件与入口
- embedding_project/main.py：CLI 入口，含 checkpoint/恢复；调度数据加载、分析阶段、可视化与报告生成。
- embedding_project/src/analyzer.py：BrainAwareSubjectEmbeddingAnalyzer，实现各阶段（数据准备、受试差异、可分性、嵌入设计、决策）、深度网络封装、区域特定网络、可视化、最终报告。
- embedding_project/src/data_loader.py：加载 TRAIN38_no_label43.mat，做 Multi-Subject-Out 划分与缩放。
- embedding_project/config/settings.py 及 src/settings.py：特征分组、阈值、matplotlib 设置、默认路径/种子。
- embedding_project/checkpoint_manager.py 与 checkpoint_tools.py：检查点保存/恢复、元数据、清理与导出工具。
- embedding_project/src/utils.py：日志、配置持久化、进度跟踪；run.sh 启动并检查环境。

## 3. 数据集与标签
- Alex 超多模态 7T 数据集；默认路径 TRAIN38_no_label43.mat。
- 特征：341 通道（15 QTI，210 b-tensor，4 CEST 参数，112 Z-spectrum）。
- 标签：脑区 one-hot（≈101–102 类，缺 label 43），硬标签。
- 划分：train 1–30，val 31–37，test 38；prob_idx 为受试 ID；为每个划分构建掩膜。
- 采样：逐体素；当 (subject, region) ≥50 体素时构建受试×区域张量（均值特征），并记录缺失区域。

## 4. 预处理流程
- 在训练体素上拟合 StandardScaler，应用于 val/test；保存原始与缩放版本。
- 校验必需键，报告受试计数与缺失区域；记录缺失类别与标签覆盖。
- 构建受试-区域张量，计算覆盖统计，存储 label_info（one-hot 维度、缺失类、标签最小/最大）。
- 强制 matplotlib 使用 Agg 后端便于无头绘图；记录环境检查。

## 5. 模型结构
- 深层 4×4096 稠密网络（ReLU + dropout 0.5），线性 logits；输出维度由标签推断（≥101 类），与 alex 基线一致。
- 区域特定小网络：1024 → 512 → 256 → 输出，dropout 0.3，用于区域级受试识别实验。
- DeepNetworkWrapper 提供 sklearn 式 fit/predict/predict_proba，存训练历史；包含权重 L2 正则。
- 如可用使用 GPU；固定种子保证确定性。

## 6. 训练配置
- 损失：交叉熵 + 显式 L2 权重正则。
- 优化器：Adam，lr 1e-5，weight_decay 1e-5；25 轮；batch 128（区域网：bs 64，10 轮快速测试）；dropout 0.5。
- 数据准备：torch TensorDataset，训练阶段 shuffle；predict_proba 时才做 softmax。
- 每阶段后保存检查点；异常或中断时创建应急检查点。

## 7. 评估指标与输出
- 分类：逐轮跟踪训练准确率；可选区域特定深度精度与 specificity strength；基线分析对比深度与简单模型（logistic/RF 占位）。
- 受试差异：PCA 方差、受试间距离/相关矩阵、ANOVA/Kruskal、silhouette、KMeans/AGG/DBSCAN 聚类。
- 嵌入需求：患者方差比、区域特异性得分、可分性指数、UMAP 聚类质量、综合嵌入必要性得分与分级区域优先级。
- 可视化：相关矩阵、PCA/TSNE/UMAP、降维对比、区域特异性热图；保存于 output_dir/visualizations。
- 报告与日志：brain_aware_analysis_output/outputs/*.json（analysis_config、key_results、阶段结果）、报告 txt、checkpoints/*.ckpt、恢复日志、绘制的 .png、检查点内的训练模型状态。

## 8. 论文写作解读
- 提供是否需要 subject embedding 的结构化诊断，将统计差异分析与神经基线结合。
- 生成可执行建议（统一 vs 选择性 vs 分层嵌入架构、优先区域、实现时间线），可写入关于嵌入策略选择的方法/讨论。
- 用于说明基线可分性不足与受试特异信号主导的部位，从而激励个性化或区域自适应模型。

## 9. 限制与开放问题
- 实际性能取决于运行结果，未硬编码。
- 假定特定 .mat 结构与受试 ID 范围；其他数据需适配。
- 区域阈值（≥50 体素）与缺失类处理可能偏向小结构；需做敏感性分析。
- 虽提及 TabNet/KAN 或校准方法，但本流程未实现。
