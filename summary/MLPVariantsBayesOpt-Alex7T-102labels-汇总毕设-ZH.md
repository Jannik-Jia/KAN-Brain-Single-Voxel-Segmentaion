# MLPVariantsBayesOpt 于 Alex7T-102labels

## 1. 在论文中的角色
- Alex 超多模态 7T 数据集（German et al. 2021）体素级的核心全连接基线，使用 MLP 并通过 BayesOpt 调节深度/宽度/激活。
- 为含背景的 102 类组织/分区标签建立参考性能，兼容 MAT 与目录式数据布局。
- 提供带校准的训练/评估流水线（train/val/test 划分、指标、检查点），用于后续架构基准。

## 2. 代码文件与入口
- main.py：CLI 入口；解析参数，加载配置，建立数据加载器（MAT 或目录），构建模型（base/deep/residual MLP），可选运行贝叶斯优化，训练并评估。
- config.py：默认超参、数据路径、标签格式处理、PCA/归一化开关、患者划分提示；自动检测 MAT vs 目录数据。
- train.py：多分类 MLP 训练循环；记录指标，保存检查点/CSV 日志，以验证 F1 选最佳模型。
- data/mat_loader.py 与 data/dataset.py：MAT 加载器（过滤背景 label 0），患者划分（prob_idx==38 为测试，其余 75/25 训练/验证），StandardScaler；可选 PCA/归一化；BrainVoxelDataset 处理目录数据。
- models/base_mlp.py、models/deep_mlp.py、models/residual_mlp.py、models/__init__.py：4×4096 基线，以及激活/Dropout 可调的深层与残差变体。
- utils/metrics.py、utils/visualization.py、utils/model_io.py、utils/optimization.py：指标（accuracy/F1/balanced acc/kappa，混淆矩阵）、曲线绘制、安全保存/加载架构元数据、Optuna BayesOpt 搜索空间。
- evaluate.py：重载已训检查点并做完整评估。
- predict.py 与 predict_config.json：推理/回写到 3D 网格（支持 DEMO38.mat），处理 scaler，可选 .mat 导出。
- run.sh、predict.sh：可运行预设（训练 MAT/原始；可选 BayesOpt）与带 scaler 的推理。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；TRAIN38.mat/DEMO38.mat 或重构 .npy 目录。
- 输入维度：每体素 341 标量特征（模态汇总成向量；PCA 可选，默认关）。
- 标签：102 类含背景；MAT 加载器先去背景再映射 1–102 → 0–101；目录加载器将 0 视为背景并移到 0–101。
- 划分：MAT 路径将 prob_idx==38 设为测试，其余 75/25 随机分为 train/val。目录路径依赖预分好的 train/test/val；config 也列出患者 ID 模板。Test_size CLI 可覆盖 MAT 划分（默认 1%）。
- 采样/平衡：提供类别权重计算工具，但默认训练用未加权 CE；BayesOpt 为加速使用较少 epoch。

## 4. 预处理流程
- 去背景后再缩放；标签转为连续索引。
- StandardScaler 在训练体素上拟合（MAT），或训练样本 z-score；Scaler 随检查点一起保存（scaler.joblib）。
- 可选 PCA（组件数可配；含解释方差分析辅助），默认 n_pca=0。
- 可选保存训练集均值/方差供 training_curves 绘图并存入检查点。

## 5. 模型结构
- base_mlp：线性/激活/Dropout 堆叠（默认 hidden_units=[4096,4096,4096,4096]，激活 relu/gelu/swish，dropout 0.5）→ 102 logits。
- deep_mlp：可配置深度（BayesOpt 5–8 层），常数宽度（1024/2048/3072），可选跳连。
- residual_mlp：块级残差，含可选 bottleneck（0.5）与多种层规模（如 4×4096、4×2048、1024-2048-2048-1024、4096-2048-2048-4096）。
- BayesOpt 搜索激活、dropout、优化器、调度器，并在 model_type 间选择。

## 6. 训练配置
- 损失：CrossEntropyLoss（无 ignore_index；背景保留为类 0）。类权重计算可用但主循环注释。
- 优化器：默认 AdamW（或 Adam）；lr=1e-5，weight_decay=1e-5；batch 128；训练 30 轮；每 3 轮验证。
- 调度器：cosine（默认）、multistep 或 ReduceLROnPlateau；BayesOpt 可调。
- BayesOpt：Optuna 默认 30 次 trial，搜索期间缩短 epoch（5–15），探索模型深度/宽度、dropout、激活、优化器、调度器、lr/weight_decay。
- 检查点：每次验证保存，文件名含指标；以验证 F1 选最佳；同时保存 JSON 架构。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1；混淆矩阵；训练/验证曲线。
- accuracy/F1/κ/balanced acc 越高越好；混淆矩阵揭示类别混淆。
- 输出：results/<experiment_name>/ 下含 `*_training_log.txt`、`*_metrics.csv`、`training_curves.png`、`evaluation_summary.txt`、`final_report.txt`、最佳检查点（.pth + `*_architecture.json`）、可选 scaler.joblib、`utils.visualization` 生成的图。

## 8. 论文写作解读
- 建立稳健的 FC 基线与超参调优变体，展示深度/宽度/激活/Dropout 与学习率调度对 102 类性能的影响。
- 提供可复现的 train/val/test 指标与报告，为后续 TabNet/KAN 等方法的比较奠定基准。

## 9. 限制与开放问题
- 数据路径硬编码到外部（/home/jovyan/...）；数据未随库提供，需本地更新路径。
- 背景作为可学习类别（无 ignore_index），可能影响与纯前景训练的校准对比。
- 无显式校准指标（ECE/NLL）或不确定性估计；仅报告 accuracy/F1/κ。
- BayesOpt 耗时；搜索阶段采用缩短 epoch，最终训练需完整轮数重跑。
