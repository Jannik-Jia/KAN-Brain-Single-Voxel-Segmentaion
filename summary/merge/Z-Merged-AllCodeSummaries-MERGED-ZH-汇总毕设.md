# 毕设代码实验汇总（中文合并版）

本文件是对 36 个英文 Markdown 实验汇总文件的**完整中文翻译与合并**：

`SUMMARY_DIR = /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/summary`

所有原始的 `*-汇总毕设.md` 文件保持不变。
下面各节按来源文件分组。

## 原始文件列表（按合并顺序）

- 0-Overall-AllModels-AllDatasets-汇总毕设.md
- AttnResMLP-Alex7T-102labels-汇总毕设.md
- AttnResMLP-Alex7T-30labels-汇总毕设.md
- BaseMLP-Alex7T-102labels-汇总毕设.md
- BayesOpt-MLP-Alex7T-101labels-汇总毕设.md
- ClassWeights-Alex7T-TRAIN38-汇总毕设.md
- ClassicalML-Alex7T-102labels-汇总毕设.md
- ClassicalML-GPU-Alex7T-102labels-汇总毕设.md
- ClassicalML-Imbalanced-Alex7T-102labels-汇总毕设.md
- ConvPatch2D-Alex7T-102labels-汇总毕设.md
- DataProfiling-Alex7T-102labels-汇总毕设.md
- Deep4x4096-Alex7T-RegionCLS-汇总毕设.md
- DeepMLP-Advanced-Alex7T-102labels-汇总毕设.md
- DeepMLP-Alex7T-101labels-汇总毕设.md
- DeepMLP-Alex7T-102labels-汇总毕设.md
- DeepMLP6x2048-Alex7T-102labels-汇总毕设.md
- FC4x4096-Alex7T-101labels-汇总毕设.md
- FC4x4096-Alex7T-102labels-patientwiseStd-汇总毕设.md
- FC4x4096-Alex7T-102labels-汇总毕设.md
- FC4x4096-Alex7T-softlabels-汇总毕设.md
- FixedMLP-BackgroundWeighting-Alex7T-102labels-汇总毕设.md
- KAN-Alex7T-102labels-汇总毕设.md
- KAN-Binary-Alex7T-1vRest-汇总毕设.md
- KAN-Multiclass-Alex7T-102labels-汇总毕设.md
- KAN-RFFeatures-Alex7T-102labels-汇总毕设.md
- MLP-PatientSplit-Alex7T-Train38-汇总毕设.md
- MLPVariantsBayesOpt-Alex7T-102labels-汇总毕设.md
- OriginalMLP-CouplingLR-Alex7T-102labels-汇总毕设.md
- ProbSmoothing-Alex7T-102labels-汇总毕设.md
- PseudoInverse-FeatureSelection-Alex7T-102labels-汇总毕设.md
- PseudoInverse-PCA-Alex7T-102labels-汇总毕设.md
- QC-MultimodalRegistration-Alex7T-351modes-汇总毕设.md
- ResNet50Patch-Alex7T-102labels-汇总毕设.md
- ResidualMLP-Alex7T-101labels-汇总毕设.md
- ResidualMLP-Alex7T-102labels-汇总毕设.md
- SubjectEmbedding-Alex7T-EmbeddingFeasibility-汇总毕设.md

---
## 来源文件 (Source file): 0-Overall-AllModels-AllDatasets-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): 0-Overall-AllModels-AllDatasets-汇总毕设.md
---

# 总体概要

该代码库覆盖了针对 Alex 超多模态 7T 数据集的体素级分类全流程：包含硬标签的 MLP 基线、软标签/校准实验、空间补丁 CNN（轻量版与 ResNet）、概率图后处理，以及多模态 QC/下采样工具。软标签实验强调校准与部分体积效应，空间模型验证邻域上下文的收益，QC 工具记录训练前的配准质量。

| ModelShort | DatasetShort | 毕设中的作用 | 关键脚本 | 关键指标 | 输出文件 |
| --- | --- | --- | --- | --- | --- |
| FC4x4096 | Alex7T-102labels | 硬标签基线与概率图生成器 | `training/B0_1D_training/train_1d_with_3d_dataset.py`; `erosion/train_38fold.py` | Macro-F1，整体准确率 | `.pth`，`predictions_3d` `.mat`，history JSON/ONNX |
| FC4x4096 | Alex7T-softlabels | 在下采样数据上的软标签与校准实验 | `training/downsampling/train_runner.py`; `dataset_create/downsampling/mri_downsampling_pipeline.py` | NLL、soft-ECE、Brier、AURC、soft Dice | `best.pth`、metrics JSON、可靠性/风险-覆盖率曲线、3D NPZ 预测 |
| ConvPatch2D | Alex7T-102labels | 轻量级空间基线（3×3/7×7 补丁） | `training/3D CNN/train_baseline_3x3_7x7.py` | Macro-F1 | `best_model_patch*.pth`，history JSON |
| ResNet50Patch | Alex7T-102labels | 使用失衡感知损失的深度空间模型 | `training/3D CNN/ResNet/scripts/train_mri_resnet.py` | Macro-F1，按类别 P/R/F1 | `best_model.pth`，`training_results.json`，`training_history.png` |
| ProbSmoothing | Alex7T-102labels | FC 概率图的后处理 | `training/B1_Probability_map_post_processing/smooth_postprocess_eval.py` | Accuracy、macro-F1、AUPRC、κ | 平滑后的 HDF5 预测、CSV 指标、图表 |
| QC-MultimodalRegistration | Alex7T-351modes | 多模态输入的配准/QC 分析 | `dataset_create/multimodal_mri_qc_analysis/multimodal_mri_qc_analysis.ipynb` | LNCC、NGF、MIND-SSD、ASSD/HD95、边缘 IoU | PNG 图、`modality_qc.csv`、`qc_analysis_report.json` |

论文撰写建议：将 FC4x4096（硬标签）作为主要基线；FC4x4096（软标签）用于软标签/校准分析；ConvPatch2D 与 ResNet50Patch 用于空间上下文消融；ProbSmoothing 用于后处理讨论；QC-MultimodalRegistration 用于数据质量/附录说明。注意：`HSIConvKAN-main` 含 KAN 参考实现，但未接入 MRI 数据流水线（用途未知）。



---
## 来源文件 (Source file): AttnResMLP-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): AttnResMLP-Alex7T-102labels-汇总毕设.md
---

# AttnResMLP 于 Alex7T-102labels

## 1. 在论文中的角色
- 面向 Alex 超多模态 7T 数据集（German et al. 2021）的体素级分类，作为带注意力的残差 MLP 基线。
- 在保持单体素输入的前提下，测试 transformer 式升级（Stage1 基础残差，Stage2 加自注意力，Stage3 加 pre-LN + FFN）相对简单 FC/KAN 方案的效果。
- 聚焦 102 个硬标签，带类别失衡缓解，评估 train/val/test 与 merged 集。

## 2. 代码文件与入口
- ：端到端 notebook，定义数据采样器、dataloader、可选 PCA、模型变体（Stage1–3）、训练循环、评估与可视化。
- 关键组件：、 、 、 、模型类 、 、 、 、 ，训练/评估辅助（ 、 ，绘图）。
- 输出保存在 notebook 相对路径下的 。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；体素特征位于 。
- 输入维度：默认 341 通道（diffusion/QTI、CEST offsets、Amide/Amine/NOE/MT 等），可选 PCA 到 （默认 0 = 不做 PCA）。
- 标签：102 个组织/区域类别（id 1–102）+ 背景 0；背景重映射为 ，在 loss 中忽略；类别映射存到 。
- 划分：由  重构，大约 60/20/20 train/val/test；额外的“merged”集合载入所有标签做 sanity check。
- 采样：可选平衡（, ）并用简单复制增强，最多 ；也可使用全样本（）。

## 4. 预处理流程
- 可选在 train/test/val 拼接后做 PCA，再对每个主成分做 min-max 归一化；默认 （使用原始 341 维特征）。
- PCA 关闭时不做显式归一化；假设上游已做特征缩放。
- 标签处理：背景 -> ；类别权重由训练标签计数计算。
- 数据重构工具合并旧 train/val 并按 0.6/0.2/0.2 切分（上游一次性执行）。
- 无缺失值处理或模态特定缩放。

## 5. 模型结构
- Stage1 ：输入线性层 + GELU + BatchNorm -> 在  上堆叠 （默认 [256,256,256]）-> 线性分类头。
- Stage2：在每个残差块后加入多头 （num_heads=4, dropout=0.1），带 LayerNorm 与残差融合。
- Stage3 Transformer 风格：输入嵌入（Linear -> LayerNorm -> SiLU -> Dropout 重复），若干 pre-LN 多头注意力（num_heads=8, dropout=0.2）+ FFN（4x 扩张，SiLU）+ 投影残差，最终 LayerNorm + 分类器。
- 正则化：dropout 0.1/0.2，前期阶段含 BatchNorm；定义了 L1/entropy 正则标志（, , ），但未在损失中实际使用。

## 6. 训练配置
- 损失： 带逆频次类别权重， 背景忽略。
- 优化器：AdamW (, )；seed 666。
- 学习率调度：可选；默认  里程碑 [15,35,50,75]，gamma 0.6；也实现了 cosine 与 plateau 方案。
- Batch size 256；训练 100 轮；每 3 轮验证一次。
- 通过重采样/复制增强做数据平衡；无 mixup/cutmix；无显式梯度裁剪。
- 检查点每次验证保存到 ；配置保存到 。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1；混淆矩阵（对数尺度热力图）。
- 可视化：类别分布柱状图、按类准确率柱状图、混淆矩阵、多数据集指标对比。
- 输出： 图（如 , , ），文本报告（, ），检查点 , , 。

## 8. 论文写作解读
- 为 Alex 数据集 102 类提供增强注意力和残差的强力 FC/MLP 基线。
- 展示 transformer 式组件如何影响体素级分类与类别失衡，为与 KAN/TabNet/线性基线的对比提供依据。
- 适合放在“基线与注意力增强 MLP”部分，或注意力深度的消融。

## 9. 限制与开放问题
- 精确指标数值未嵌入，需依赖已保存的报告。
- 数据路径硬编码到 ，且依赖预构建的 ，可移植性有限。
- 缺少校准指标（ECE/NLL）和不确定度估计；PCA 关闭时的归一化假设不明确。
- L1/entropy 正则标志未用；Stage1/2/3 对比未在 notebook 中给出。



---
## 来源文件 (Source file): AttnResMLP-Alex7T-30labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): AttnResMLP-Alex7T-30labels-汇总毕设.md
---

# AttnResMLP 于 Alex7T-30labels

## 1. 在论文中的角色
- 对 German et al. 2021 的 Alex 超多模态 7T 数据集进行 30 类子集验证，检验带注意力的残差 MLP。
- 缩减类别空间以加快迭代，并在完整 102 类跑之前测试采样/平衡策略。

## 2. 代码文件与入口
- ：完成数据子集选择、加载器、模型变体（Stage1–3），训练与评估与全类别 notebook 相同。
- 复用的关键助手： （按样本数挑前若干类）、 、 、 、模型类、 ，评估/绘图。
- 输出位于 （相对 notebook）。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；体素特征在 。
- 输入维度：341 通道；可选 PCA（默认关闭）。
- 标签：从 102 类中选出的 30 类子集；要求训练样本 >=1000、测试/验证至少 ~100；标签重映射为 0–29，背景作为  被忽略。
- 划分：与重构版 train/test/val 相同（约 60/20/20）；子集选择在读取标签索引后进行。
- 采样：用简单复制增强将每类平衡到 ；保存了类别映射。

## 4. 预处理流程
- 与全量设置一致：可选 PCA + min-max 归一化；否则使用原始特征；计算类别权重；忽略背景。
- 无额外的模态对齐或缺失值处理。

## 5. 模型结构
- 与全实验相同的 Stage1/2/3 注意力残差 MLP 选项（隐藏层 [256,256,256]，注意力头 4 或 8，dropout 0.1–0.2）。
- 残差块使用 GELU 与 BatchNorm；Stage3 采用 pre-LN 的 transformer 式模块。

## 6. 训练配置
- 损失： 带类别权重， 。
- 优化器 AdamW (, )；训练 100 轮；batch 256；每 3 轮验证。
- 学习率调度同样可选（默认 MultiStep [15,35,50,75]，gamma 0.6）。
- 启用平衡与数据增强；检查点与配置导出到 。

## 7. 评估指标与输出
- 指标同样包括 accuracy、balanced accuracy、macro/weighted F1、Cohen κ、按类指标、混淆矩阵。
- 可视化与文本报告与 102 类 notebook 对齐，仅路径指向子集保存目录。

## 8. 论文写作解读
- 快速迭代实验，检验注意力残差 MLP 在较少类别下的可扩展性，以及平衡/增强流水线在小类空间中的表现。
- 可作为方法附录或完整训练前的先导实验，支撑架构与采样策略的选择。

## 9. 限制与开放问题
- 实际选取的类别依赖数据可用性，仅记录在 。
- 与全实验一样，存在硬编码数据路径、缺少校准指标的问题。
- 性能数值未写入 notebook，需查阅保存的报告。



---
## 来源文件 (Source file): BaseMLP-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): BaseMLP-Alex7T-102labels-汇总毕设.md
---

# BaseMLP 于 Alex7T-102labels

## 1. 在论文中的角色
- 作为 Alex 超多模态 7T 数据集（German et al. 2021）体素级组织分类的主要全连接基线（4x4096 单元）。
- 为后续消融（深层 MLP、残差 MLP、贝叶斯超参搜索）以及校准/推理流程建立参考性能。

## 2. 代码文件与入口
- `config.py`：默认超参（4x4096 ReLU MLP，dropout 0.5，AdamW 1e-5，batch 128，30 轮，cosine LR）；患者 ID 划分定义于此。
- `main.py`：CLI 训练入口；加载按患者划分的数据，可选 Optuna 搜索，训练、评估并写出报告/检查点。
- `train.py`：训练循环，使用类别加权交叉熵，每 `val_epochs` 验证一次，检查点名包含 epoch/acc/F1。
- `eval.py` 与 `evaluate.py`：重新加载已存检查点并重跑评估的封装。
- 数据流水线：`utils/patient_data_adapter.py`（按患者的加载器、StandardScaler 保存）、`BrainVoxel38PatientLoader.py`（按患者加载器，one-hot 标签）、`data/dataset.py` 与 `data/samplers.py`（旧版标签索引加载、可选 PCA），以及记录数据重构的 notebook（`fullyconnected_brainvoxel_完全分离的数据集创建方法_102分类_L1.ipynb`、`数据库再次修改接入38折验证.ipynb` 等）。
- 模型：`models/base_mlp.py`；工厂位于 `models/__init__.py`。
- 指标/可视化：`utils/metrics.py`、`utils/visualization.py`；日志与检查点 I/O 在 `utils/model_io.py`。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，重组为 38 个患者 ID 的体素特征。
- 输入特征：341 通道（CEST/QTI/SMWI/MPRAGE 提取的体素特征）；默认 `feature_dim=341`；PCA 默认关闭但受支持。
- 标签：102 个皮层/皮层下/分割类别；原始标签 0 视为背景并在训练中映射为忽略索引 -1。
- 划分：按患者；默认 train 1–20，val 21–28，test 29–38（可配置）；按患者加载器确保无跨受试体素泄漏。
- 采样：无显式类别平衡；类别权重通过 `utils/metrics.calculate_class_weights` 的逆频次计算。

## 4. 预处理流程
- 使用 `sklearn.StandardScaler` 在训练特征上拟合标准化；Scaler 保存到 `save_dir/scalers/...pkl`，供 val/test 与推理复用。
- 背景体素映射为 -1，在损失/指标中忽略；按患者加载器的 one-hot 标签折叠为索引。
- 可选 PCA 位于 `data/dataset.apply_pca`/`load_multiclass_data`（默认按患者流程不使用）。
- 除可选 PCA 外无额外降维或特征选择；除数据构建外无显式缺失值处理。

## 5. 模型结构
- 基础 MLP：全连接堆叠，隐藏层 `[4096, 4096, 4096, 4096]`，可选 ReLU/GELU/Swish，层后 dropout 0.5，线性输出 102 维 logits。
- 无跳连/残差；纯全连接；保存时通过 `get_model_info` 记录参数量。
- 配置标签：`model_type='base_mlp'`，`model_name='BrainVoxel_102Class_MLP'`。

## 6. 训练配置
- 损失：类别加权 `CrossEntropyLoss`，忽略索引 -1 处理背景。
- 优化器：AdamW（默认）或 Adam；lr 1e-5，weight_decay 1e-5。
- 调度器：可选（默认 30 轮 cosine）；CLI 支持 multistep/plateau。
- Batch size 128；30 轮；每 3 轮验证；无数据增强。
- 贝叶斯搜索（`config.py` 默认开启）可搜索 lr、weight decay、dropout、激活函数、调度器与备选层宽；若需纯基线，可关闭 `--run_bayesian_opt`。
- 类别权重按运行计算；未使用梯度裁剪。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1 与样本数。
- 校准风格产物：softmax 概率保存为 `.npz`；混淆矩阵（对数尺度热力图）与按类柱状图由 `utils/visualization` 生成。
- 输出：训练日志 `*_training_log.txt`，指标 CSV `*_metrics.csv`，各划分 `*_class_metrics.csv`，预测 `.npz`，混淆矩阵 `.png`，训练曲线 `training_curves.png`，`evaluation_summary.txt`，`final_report.txt`，以及包含 epoch/acc/F1 的检查点，位于 `results/<experiment_name>/`。
- 最优模型通过 `utils/metrics.get_best_model` 基于验证 macro F1 选取；控制台与文件均会摘要。

## 8. 论文写作解读
- 测试朴素 4x4096 全连接分类器在 Alex 超多模态 7T 体素特征上的上限，作为主要性能与校准参考。
- 证明患者级划分与标准化足以构成强基线，再去探索结构变化或校准技巧。
- 可填入“基线体素分类器”小节，支撑后续对比。

## 9. 限制与开放问题
- 具体体素数与类别平衡摘要依赖外部 `label_index.txt`/重组数据，未内置；需从运行日志获取。
- 数据路径（`/home/jovyan/.../reorganized_fold_data`）硬编码，迁移需调整。
- 基线训练循环无显式早停；过拟合控制依赖 dropout 与 weight decay。
- 除 accuracy/F1/κ 外未包含显式校准指标（ECE/MCE）。



---
## 来源文件 (Source file): BayesOpt-MLP-Alex7T-101labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): BayesOpt-MLP-Alex7T-101labels-汇总毕设.md
---

# BayesOpt-MLP 于 Alex7T-101labels

## 1. 在论文中的角色
在 Alex 超多模态 7T 数据集上对基础/深层/残差 MLP 进行自动化超参与结构搜索，为对比其他架构前挑选最强的 FC 风格模型提供数据驱动方案。

## 2. 代码文件与入口
- `utils/optimization.py`：Optuna 目标/搜索空间，覆盖学习率/权重衰减/dropout/激活/优化器/调度器/model_type/depth/width/skip/bottleneck 等；含可视化与报告工具。
- `main.py` 中的集成开关（`--run_bayesian_opt`, `--n_trials`）与交互式启动脚本 `run.sh`（自动测试三类 MLP，用户选择标准化/背景模式）。
- 复用加载/训练模块（`train.py`、`data/mat_loader_patientwise.py`、`data/samplers.py`、`utils/metrics.py`、`utils/model_io.py`）。

## 3. 数据集与标签
同样使用 341 维特征、101 类标签（背景可配置）的 Alex 超多模态 7T 设置。大多数搜索假设按患者标准化；固定 prob_idx 划分（val 20，test 38），除非预设覆盖。类别权重由训练标签计算。

## 4. 预处理流程
与其他组一致：按患者/全局标准化，可选背景过滤/忽略，保存 scaler，可选 PCA（默认关），如启用则批次感知患者。

## 5. 模型结构
- 搜索覆盖 `model_type` ∈ {base_mlp, deep_mlp, residual_mlp}。
- 隐层宽度 1024–8192、深度 4–12，支持宽度策略（常数/递减/递增/沙漏/钟形）；可选残差 bottleneck；可切换跳连。
- 激活可选 `relu`/`gelu`/`swish`，dropout 最高 0.8。

## 6. 训练配置
- 试验训练 5–20 轮（由 `epochs` 限制），使用交叉熵 + 类别权重；优化器可选 Adam/AdamW/SGD/RMSprop；调度器支持 cosine/step/plateau/none；可选梯度裁剪。
- 验证集 macro-F1 作为优化目标；使用 MedianPruner 做早停。最优参数保存到 JSON 与 `optimized_config.json`，后续完整训练复用最佳设置。

## 7. 评估指标与输出
- 试验指标由 Optuna 记录；汇总结果导出到 `<study_name>_results.json`，并在 study 目录下生成分析图（`architecture_performance_boxplot.png`、`architecture_trial_counts.png`、`performance_evolution.png`，及参数重要性/历史图如有）。
- 最终模型仍用标准指标集（accuracy、balanced accuracy、macro/weighted F1、kappa）评估，检查点/图表保存方式与其他组一致。

## 8. 论文写作解读
- 展示系统化超参搜索对 FC 风格模型的影响，区分性能提升究竟来自深度/宽度等结构还是优化设置，而非新模型家族。
- 可支撑“自动化模型选择”小节，并提供下游对比使用的选定配置。

## 9. 限制与开放问题
- 搜索使用缩短训练轮数；获胜配置需完整训练以确认提升。
- Optuna 带来计算开销；结果依赖固定验证患者（20），可能对该受试过拟合。
- 需要 Optuna 依赖，较大宽度的试验可能需 GPU；并非所有 trial 都保存中间检查点。



---
## 来源文件 (Source file): ClassWeights-Alex7T-TRAIN38-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ClassWeights-Alex7T-TRAIN38-汇总毕设.md
---

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



---
## 来源文件 (Source file): ClassicalML-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ClassicalML-Alex7T-102labels-汇总毕设.md
---

# ClassicalML 于 Alex7T-102labels

## 1. 在论文中的角色
对 Alex 超多模态 7T 数据集（102 标签，341 特征）进行传统特征组分析与层次聚类/分类。作为基线，评估类别可分性、测试大类映射，并在重型 KAN 之前找出潜在的优质特征子集；给出粗粒度组织分组的特征组合与分类器建议。

## 2. 代码文件与入口
- main.py：验证集端到端流程；加载数据，按特征组预处理，执行可分性分析、聚类、分类，并报告与预定义大类的一致性。
- config.py：全局超参（特征索引、PCA 开关、采样标志、训练 LR/EPOCH 占位）、数据路径、保存目录构造。
- data_loader.py：加载 train/test/val 的 *.npy 体素文件，构建标签数组，可选 PCA+min-max 缩放，定义 7 类大类映射。
- feature_analysis.py：SelectKBest（F 统计）特征选择，PCA/UMAP/TSNE/MDS 可视化，聚类搜索（kmeans/spectral/agglomerative）并用 silhouette/Calinski-Harabasz/Davies-Bouldin 评分，cluster-vs-label 热图，与大类的一致性评分。
- classification.py：robust/standard 缩放，分组特征选择，特征组合搜索，交叉验证比较分类器（KNN/SVM/RF/MLP），输出比较图。
- utils.py：日志设置，PCA 解释方差估计；run_analysis.sh 启动 nohup 任务。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；体素特征重组到 train/test/val 目录。
- 输入：每体素 341 模态（0–14 diffusion，15–224 QTI/b-tensor，225–340 CEST）。体素数量未给；main.py 使用验证划分。
- 标签：102 个类似 FreeSurfer 的解剖类别，硬标签；映射到 7 个大类（脑室、白质、皮层灰质、深部核团、边缘系统、脑干、其他）。
- 划分：train/test/val 目录；主流程仅分析 val；data_loader 记录各划分的类别计数。

## 4. 预处理流程
- 可选 PCA（config 中 APPLY_PCA=True，但 main.py 用 apply_pca=False 读取原始特征）以及 PCA 后的 min–max 归一化。
- 增强预处理：按组 RobustScaler，SelectKBest F 统计特征选择（10 diffusion，30 QTI，20 CEST），并保存重要性图。
- 无显式缺失值处理；假设体素行有效。存在采样标志，但 main.py 关闭采样。

## 5. 模型结构
- 传统基线：KNN(k=5)、RBF SVM(C=1, probability=True)、RandomForest(50–100 树)、小型 MLP(50 隐层) 用于可分性与分类器比较。
- 聚类：kmeans、spectral、agglomerative，针对每个特征组测试 2–7 个簇；以 silhouette 选优。
- 此处无深度 KAN；探索仅限特征 + 传统模型。

## 6. 训练配置
- 使用 StratifiedShuffleSplit（3 折，70/30）或 5 折交叉验证（视函数而定）；指标为 accuracy。
- 无 epoch/优化器；基于 sklearn 的经典训练。config 中 lr/weight decay 未用于主流程。
- 最佳特征组合由交叉验证准确率选取；分类器排名报告均值±方差。

## 7. 评估指标与输出
- 分类：交叉验证准确率（越高越好）及标准差；给出随机猜测基线。本版本未保存混淆矩阵。
- 聚类：silhouette（高优）、Calinski-Harabasz（高优）、Davies-Bouldin（低优）；簇大小分布。
- 一致性：Hungarian 匹配的簇与大类对齐得分（越高越好）。
- 可视化：特征重要性条形图、PCA/UMAP 图、聚类指标曲线、cluster-vs-label 热图；日志与图像保存到 Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<timestamp>/，run_analysis.sh 运行时输出到 nohup_output/*.out。

## 8. 论文写作解读
- 为 Alex 7T 数据的体素级分类与大类验证提供非神经网络基线。展示哪些模态组最可分、哪些特征组合有效，以及预定义的 7 大类是否与数据驱动簇对齐。
- 为“是否需要 KAN/TabNet 的复杂度”提供证据：若传统模型已表现良好，复杂模型的价值需要论证。

## 9. 限制与开放问题
- main.py 仅在验证集上运行；无端到端 train/val/test 评估。
- 数据路径硬编码到 /home/jovyan/...，假设已预打乱的 npy 文件。
- 未处理严重类别失衡或软标签；无校准指标。
- 未集成 KAN；主要用于探索性分析。



---
## 来源文件 (Source file): ClassicalML-GPU-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ClassicalML-GPU-Alex7T-102labels-汇总毕设.md
---

# ClassicalML-GPU 于 Alex7T-102labels

## 1. 在论文中的角色
GPU 加速与跳步变体（V3），用于有选择地重跑耗时分析，同时保留失衡处理工具。支持对 Alex 7T 体素数据更快速地迭代聚类与分类器对比。

## 2. 代码文件与入口
- V3_不均衡分类_GPU加速/main.py：V2 流水线的变体，含缓存占位（check_step_completed）、重新排序的分组处理、跳过已完成步骤的钩子；导入 pickle 以保存中间结果。
- V3_不均衡分类_GPU加速/config/config.py 以及 data/preprocessing.py、analysis/*、utils/*：在预处理、聚类、分类、评估上镜像 V2 结构。
- run_brain_voxel.sh、run_experiments.sh、experiments/feature_selection_exp.py、clustering_exp.py：批量执行助手。
- output/ 目录用于图像、日志、结果、模型。

## 3. 数据集与标签
- 与 Alex 超多模态 7T 体素数据相同，341 通道输入、102 个硬标签；保留到 7 大类的映射。
- 加载 train/test/val 的 npy 分片；提供采样标志但默认全量数据。

## 4. 预处理流程
- Robust/standard 归一化选项，分组 PCA（10/30/20/60），SelectKBest 特征选择；可选采样以提速。
- 处理顺序优先 all_features/qti/cest，再到 diffusion；若已有结果，可跳过 diffusion 聚类。

## 5. 模型结构
- 传统 sklearn 分类器（KNN、SVM、RF、MLP）用于大类预测；聚类用 kmeans/spectral/agglomerative。
- 本分支无神经网络/KAN 实现。

## 6. 训练配置
- 与 V2 类似的交叉验证，指标含 accuracy/balanced accuracy/F1/kappa；聚类测试 2–10 个簇。
- 重点在跳过重复的重负载步骤，而非调整超参；无基于 epoch 的训练。

## 7. 评估指标与输出
- 指标同 V2（accuracy 系指标；silhouette/Calinski-Harabasz/Davies-Bouldin）。
- 可视化与日志保存在 V3_不均衡分类_GPU加速/output/；缓存占位存在但未完全实现。

## 8. 论文写作解读
- 展示为使传统层次分析可扩展所做的工程努力；可用于讨论资源受限场景下的运行时考量与可复现性。

## 9. 限制与开放问题
- 许多流程段仍被注释或依赖外部缓存；GPU 加速仅暗示，并未显式（无 cupy 等）。
- 跳步逻辑与缓存加载不完整，端到端运行可能需要手动调整。



---
## 来源文件 (Source file): ClassicalML-Imbalanced-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ClassicalML-Imbalanced-Alex7T-102labels-汇总毕设.md
---

# ClassicalML-Imbalanced 于 Alex7T-102labels

## 1. 在论文中的角色
失衡感知的传统分析流水线（V2），用于在偏斜的体素计数下评估特征预处理、聚类与分类器。提供可复现的 CLI 与批量脚本，说明重采样和特征选择在进入神经模型前如何影响基线表现。

## 2. 代码文件与入口
- V2_不均衡分类/main.py：CLI 覆盖数据加载、大类映射、预处理（归一化、PCA、特征选择）、聚类、分类器比较、混淆矩阵、映射评估；支持子集选择与跳过绘图。
- V2_不均衡分类/config/config.py：特征组、PCA 维度、特征选择方法、聚类范围、评估指标、路径、随机种子、设备等配置。
- data/data_loader.py 与 data/preprocessing.py：加载 train/test/val 的 npy，记录类别计数，按组应用缩放/PCA/特征选择，可选采样标志。
- analysis/*（feature_analysis.py, dimensionality.py, clustering.py, classification.py）：特征重要性、可分性、可视化（PCA/UMAP）、聚类搜索、分类器评估。
- utils/logging_utils.py, utils/evaluation.py, utils/model_utils.py：日志、指标计算（accuracy、balanced accuracy、f1_weighted、kappa；聚类指标）、结果保存、映射生成/可视化。
- run_brain_voxel.sh, run_experiments.sh, experiments/*.py：shell 与批量实验脚本（特征选择、聚类、组合实验）。
- output/figures、output/logs、output/results 作为生成物占位。

## 3. 数据集与标签
- Alex 超多模态 7T 数据；体素特征 341 维（拆分为 diffusion/QTI/CEST），102 个硬标签。
- 与主分支一致的 7 类大类映射，整型标签。
- 划分通过 train/test/val 目录；可对单一子集或合并集分析；记录类别分布。
- 采样控制：USE_SAMPLING 标志（默认 False）和 SAMPLE_RATIO=0.1 以快速测试；导入 imblearn SMOTE/RandomUnderSampler 以缓解失衡。

## 4. 预处理流程
- Robust/standard/无 归一化选项；按组 PCA（10/30/20/60 组件）可配置；SelectKBest 特征选择（f_classif、mutual_info、chi2）。
- 保留特征分组以比较模态贡献；可选降采样以提速。
- 除 numpy 读取外无显式缺失值处理；假设体素行有效。

## 5. 模型结构
- sklearn 经典分类器：KNN、RBF SVM、RandomForest、MLP；在大类标签上评估。
- 聚类：kmeans、spectral、agglomerative，测试 2–10 簇，记录 silhouette/Calinski-Harabasz/Davies-Bouldin。
- 提供映射评估以对齐聚类与预定义大类。

## 6. 训练配置
- 5 折或分层划分的交叉验证，指标含 accuracy、balanced accuracy、weighted F1、Cohen κ。
- 无基于 epoch 的训练；超参多为默认（树数或核选择除外）；可在批量运行时跳过昂贵图表。

## 7. 评估指标与输出
- 分类指标如上（准确率/F1/κ 越高越好）。
- 聚类指标：silhouette 与 Calinski-Harabasz（越高越好），Davies-Bouldin（越低越好）；稳定性检查位于 utils/evaluation。
- 可视化：特征重要性、降维散点、cluster-vs-label 热图、混淆矩阵；输出存于 V2_不均衡分类/output/ 子目录，带 config 中 RUN_ID 时间戳。
- 日志通过 logging_utils 保存到 output/logs。

## 8. 论文写作解读
- 量化传统模型在类别失衡与不同预处理设置下的表现，为大类粒度与特征有效性提供基线。
- 可用于“不平衡处理与特征选择消融”章节，作为 KAN 等神经模型前的参考。

## 9. 限制与开放问题
- 数据路径硬编码到 /home/jovyan/...；假定预生成的 npy 分片。
- 失衡缓解（SMOTE/欠采样）已接入但默认 CLI 流程未充分使用；效果未知。
- 仍聚焦于大类而非完整的 102 分类；无校准指标。



---
## 来源文件 (Source file): ConvPatch2D-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ConvPatch2D-Alex7T-102labels-汇总毕设.md
---

# ConvPatch2D 于 Alex7T-102labels

## 1. 在论文中的角色
基于补丁的 CNN 基线，在保持轻量的同时引入最小空间上下文（xy 平面 3×3 或 7×7）。作为 1D MLP 与更深 ResNet 实验之间的过渡。

## 2. 代码文件与入口
- `training/3D CNN/train_baseline_3x3_7x7.py`：主训练脚本，包含模型定义（ImprovedConv2D_Baseline 及参数对齐变体）。
- `training/3D CNN/README.md`：3×3/7×7 补丁的使用指南与超参说明。
- `training/3D CNN/run_leave_one_out.sh`：批量留一法运行器；`test_data_loading.py` 用于数据检查。
- `training/3D CNN/analyze_results.py`：汇总留一结果并绘制统计。

## 3. 数据集与标签
- 数据集：Alex 超多模态 7T 的 3D MAT 文件（384×336×256×351），含 `region_labels` 和 `region_mask`。
- 输入：在固定 z 上切取 xy 平面的 2D 补丁（3×3 或 7×7）；每体素 351 通道，从掩膜内体素采样（默认每个训练受试 1 万，测试 2 倍）。
- 标签：102 个硬类别（在加载器内偏移到 0–101）；背景排除。
- 划分：按受试留一（37 训 / 1 测），通过 CLI 选择。

## 4. 预处理流程
- 每个受试的每通道 z-score 归一化（整 3D 体积）。
- 补丁提取在边界处填充以保持固定大小；启用时缓存受试数据到内存。
- 仅在掩膜且标签>0的体素中采样；训练可选 shuffle。

## 5. 模型结构
- Stem：1×1 卷积混合通道（351→mid，默认 mid=128）+ GroupNorm + SiLU。
- 聚合：单个卷积核大小等于补丁大小（3 或 7），用于折叠空间维度 + GroupNorm + SiLU。
- 可选 1×1 残差 refine 块；可选 SE 通道注意力。
- Head：线性分类器或 MLP 头（参数对齐变体，目标约 35M/52M/69M 参数，隐藏层大且带 dropout）。

## 6. 训练配置
- 损失：交叉熵；优化器 AdamW（lr=1e-4, weight_decay=1e-4）；CosineAnnealingLR（T_max=50, eta_min=1e-6）。
- Batch size 256；训练 50 轮；使用 GradScaler 的混合精度；除补丁采样外无显式增强。
- 每轮以测试集 macro-F1 选最佳检查点；历史保存为 JSON。

## 7. 评估指标与输出
- 指标：每轮训练/测试的 macro-F1 与平均 loss；打印参数量。
- 输出：`best_model_patch{3|7}_test{N}.pth`，`history_patch{...}.json`，留一运行的日志。

## 8. 论文写作解读
评估小尺度空间上下文（3×3 vs 7×7）与参数规模对体素级准确率的影响，相对 1D MLP 的改进，为迈向更深 ResNet 架构的消融步骤。

## 9. 限制与开放问题
- 仅在 2D 切片上操作（无完整 3D 卷积）；增强有限。
- 每受试采样固定体素数，使用全数据训练的效果未知。
- 未显式处理类别失衡或校准。



---
## 来源文件 (Source file): DataProfiling-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): DataProfiling-Alex7T-102labels-汇总毕设.md
---

# DataProfiling 于 Alex7T-102labels

## 1. 在论文中的角色
MRI 数据剖析工具包（V5），在训练前量化特征分布、相关性、可分性与特征重要性。用于评估模态价值、指导后续架构选择。

## 2. 代码文件与入口
- V5_data_analysis/main.py：CLI 入口，执行预处理、基础统计、特征分析、降维；支持 GPU 标志。
- V5_data_analysis/data_loader.py：加载脑体素测试数据（npy），可用 one-hot 或整型标签，报告标准化状态，定义特征分组；若可用尝试 cupy/cuml。
- basic_analysis.py、feature_analysis.py、dim_reduction.py：计算摘要统计、相关性、类别可分性、特征重要性（RF、互信息），评估特征选择方法，运行 PCA/TSNE/UMAP/Isomap 降维。
- run_analysis.sh：执行流水线的助手脚本；输出目录建在 analysis_results 下。

## 3. 数据集与标签
- Alex 超多模态 7T 数据集，主要使用 /home/jovyan/.../restructured/test 的测试划分。
- 341 个特征，分组为 diffusion/QTI/CEST/all_features；102 个标签（提供整型与 one-hot 形式）。
- 聚焦测试子集做剖析；类别计数由标签索引文件报告。

## 4. 预处理流程
- preprocess_data 中可选 robust/standard 归一化；检测数据是否已标准化。
- 试用 cupy/cuml 做 GPU 加速；若不可用则回退到 numpy/sklearn。
- 加载时不做 PCA；降维在分析阶段处理。

## 5. 模型结构
- 非训练流水线；使用随机森林特征重要性与互信息评分特征。除可分性评分外无神经/传统分类器训练。

## 6. 训练配置
- 不适用；分析直接作用于给定数据，无基于 epoch 的训练。

## 7. 评估指标与输出
- 基础统计（各特征均值/方差范围）、组内/组间特征相关性、类别可分性比（F 分数）、特征重要性得分（RF/MI）。
- 降维图（PCA/TSNE/UMAP/Isomap）用于可视化类别分布。
- 输出保存到 analysis_results 子目录（basic_analysis、feature_analysis、dim_reduction），包含文本摘要与图像。

## 8. 论文写作解读
- 提供关于哪类模态携带判别信息以及通道相关度的描述性证据，支撑后续分类器在归一化、特征选择上的方法选择。

## 9. 限制与开放问题
- 主要在测试子集上运行；若无 train/val 检查，结论未必可泛化。
- 数据路径硬编码；假设标签索引文件与 npy 分片存在。
- 与下游模型性能没有直接关联；仅作探索性分析。



---
## 来源文件 (Source file): Deep4x4096-Alex7T-RegionCLS-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): Deep4x4096-Alex7T-RegionCLS-汇总毕设.md
---

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



---
## 来源文件 (Source file): DeepMLP-Advanced-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): DeepMLP-Advanced-Alex7T-102labels-汇总毕设.md
---

# DeepMLP-Advanced 于 Alex7T-102labels

## 1. 在论文中的角色
- 深层 MLP 的探索变体，为 German et al. 2021 的 Alex 超多模态 7T 数据集加入更强正则（shake-shake、随机深度）、混合激活、三线性特征池化。
- 作为消融，用于测试更丰富的特征交互与正则化是否能提升 102 类体素分类。

## 2. 代码文件与入口
- ：定义高级模块并运行训练/评估；还包含遗留的 KAN 评估钩子。
- 关键组件： 带可选三线性池化和预定义特征组（anatomical/functional/diffusion）， 带位置嵌入， 支持 shake-shake 与随机深度， 、 、 类。
- 训练/评估辅助与基础 deep MLP 类似（ 、 、 绘图、 ）；输出位于 。

## 3. 数据集与标签
- 相同的 Alex 超多模态 7T 数据集，使用重构的 train/test/val 文件夹中的 341 通道体素特征。
- 标签：102 类，加背景忽略为 ；使用类别权重；仅硬标签。
- 划分：train/test/val（~60/20/20），可选平衡到每类 1 万。

## 4. 预处理流程
- 可选 PCA + min-max 归一化（默认关闭）；通过重采样/增强做平衡；背景移除；配置导出到 。
- 除模型内部的语义特征分组外，无额外模态对齐。

## 5. 模型结构
- 隐层 ；在索引 [1,3,5] 处插入注意力层，16 个头；dropout 列表 ，（注意长度可能与隐藏层不匹配）。
- FeatureInteractionLayer 支持三线性池化，并对解剖（0–120）、功能（120–240）、扩散（240–341）通道做特征组投影。
- 残差块可使用 shake-shake 正则（alpha/beta 随机混合）、随机深度（率 0.2）、混合 GELU/SiLU 激活；自注意力含可学习位置嵌入。
- 分类器为线性头；权重初始化使用 Kaiming 变体。

## 6. 训练配置
- 损失： 带类别权重与 ；背景忽略。
- 优化器 AdamW (, )；batch size 512，使用 ；训练 150 轮。
- 调度器：OneCycleLR (, )；混合精度；梯度裁剪 10；每 3 轮验证。
- 检查点通过  和逐 epoch 保存；包含训练曲线与混淆矩阵绘图。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类指标；混淆矩阵；类别 F1 分布图。
- 输出： 用于训练曲线和混淆矩阵，文本测试报告 ()，检查点 , 。

## 8. 论文写作解读
- 测试强正则与特征池化思路，观察深层 MLP 是否能比基础模型更好利用多模态结构。
- 适合放在稠密架构内部的特征交互/正则化策略消融小节。

## 9. 限制与开放问题
- dropout 列表与隐藏层数量可能需对齐；代码混杂 DeepMLP 与 KAN 训练片段，显示仍在开发中。
- 仍有硬编码数据路径，缺少校准指标。
- 实际性能未写在 notebook 中。



---
## 来源文件 (Source file): DeepMLP-Alex7T-101labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): DeepMLP-Alex7T-101labels-汇总毕设.md
---

# DeepMLP 于 Alex7T-101labels

## 1. 在论文中的角色
更深的全连接变体，测试增加深度/宽度与可选跳连是否能在 Alex 超多模态 7T 数据集上超越 4×4096 基线的体素级分类表现。作为网络容量与梯度流的消融。

## 2. 代码文件与入口
- `models/deep_mlp.py`：可配置深度/宽度的深层 MLP，可选非相邻层之间的跳连。
- 共享训练栈：`main.py`、`config.py`、`train.py`、`data/mat_loader_patientwise.py`、`data/mat_loader.py`、`data/samplers.py`、`utils/metrics.py`、`utils/label_processing.py`、`utils/model_io.py`、`utils/visualization.py`。
- 通过 `utils/optimization.py` 支持超参搜索（深度/宽度策略、激活、调度器），`run.sh` 做交互式运行。
- 评估/推理与基线相同，使用 `evaluate.py`、`predict.py`，以及调参 notebook（`alex版本优化*.ipynb`）。

## 3. 数据集与标签
与基线相同：Alex 超多模态 7T 的 341 特征体素；101 类标签空间（背景可选）。固定 prob_idx 划分（val 20，test 38），并提供替代测试 ID 或随机划分预设。类别权重来自训练标签；可选按患者批次。

## 4. 预处理流程
与基线一致：按患者或全局标准化，可选 PCA（默认关），可配置背景处理，保存 scaler，检查划分完整性。

## 5. 模型结构
- 深度与宽度可配置：hidden_dims 列表可超过四层；BayesOpt 探索 4–12 层及宽度策略（常数、递减、递增、沙漏、钟形）。
- 可选 `use_skip_connections`，在非相邻层间添加加法跳连以改善梯度流。
- 激活可选 `relu`/`gelu`/`swish`，dropout 可调；输出线性层大小为 `num_class`。

## 6. 训练配置
- 损失/优化器/调度器设置与基线相同；默认 lr=1e-5，weight_decay=1e-5，batch size 128，30 轮，cosine 调度。
- 当启用 `--run_bayesian_opt` 时，Optuna 试验（5–20 轮）搜索学习率、dropout、激活、优化器（Adam/AdamW/SGD/RMSprop）、调度器类型、深度/宽度以及跳连；最佳参数保存到 `optimized_config.json`。
- 检查点/日志与基线一致。

## 7. 评估指标与输出
- 指标与产物同基线（accuracy、balanced accuracy、macro/weighted F1、kappa、按类统计、混淆矩阵、训练曲线、预测 npz/mat）。
- 最优模型仍以验证 macro-F1 选取；输出保存在 `results/<experiment_name>/`。

## 8. 论文写作解读
- 探索更深 MLP + 跳连是否能比浅层基线更好捕获 341 模态间的非线性交互。
- 适合放在网络深度/容量的消融小节；帮助论证在 MLP 扩容后是否还需要更复杂的 TabNet/KAN。

## 9. 限制与开放问题
- 最佳深度/宽度未硬编码；需查看保存的配置/日志确认。
- Optuna 试验使用缩短训练，完整训练的表现可能不同；大搜索空间下计算量高。
- 与基线一样，受试划分固定且缺少校准/数据增强。



---
## 来源文件 (Source file): DeepMLP-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): DeepMLP-Alex7T-102labels-汇总毕设.md
---

# DeepMLP 于 Alex7T-102labels

## 1. 在论文中的角色
- 更深的 MLP 变体，测试增加深度/宽度与可选跳连是否能在 German et al. 2021 的 Alex 超多模态 7T 数据集上超越 4×4096 基线的体素分类。
- 主要通过共享的 Optuna 搜索探索；不是默认配置，但可用于容量 vs 过拟合的消融。

## 2. 代码文件与入口
- `models/deep_mlp.py`：构建可变深度的 MLP（宽度 1024–3072，深度 5–8），在非相邻层间可加跳接适配。
- 共享流水线文件：`main.py`、`train.py`、`models/config.py`、`data/mat_loader.py`、`utils/optimization.py`、`utils/metrics.py`、`utils/model_io.py`、`utils/visualization.py`（与 FC 基线一致）。
- 选择方式：在 `main.py` 设 `--model_type deep_mlp`，或让 Optuna（`model_type` 搜索空间）自动选择。

## 3. 数据集与标签
- 相同的 TRAIN38/DEMO38 Alex 超多模态 7T 数据集；feature_dim=341，num_class=102，背景=0 被忽略。
- 按患者划分与基线一致，除非覆盖；类别权重来自训练标签。

## 4. 预处理流程
- 与 FC 基线相同的 StandardScaler 归一化、Scaler 持久化、可选 normalization_params 保存；PCA 标志未用。

## 5. 模型结构
- 输入层 → 线性层堆叠，激活可选 ReLU/GELU/Swish，带 dropout；宽度按 hidden_dims 列表配置（默认常数）。
- 可选跳连：适配器对齐早期层与后期层的维度后相加。
- 最后线性映射到 102 类；无 batch norm。

## 6. 训练配置
- 损失/优化器/调度器与 FC 基线相同；训练轮数/验证频次除非覆盖，否则一致。
- Optuna 搜索涵盖深度（5–8）、width_factor ∈ {1024, 2048, 3072}、use_skip_connections 标志，以及共享超参（lr、weight_decay、dropout、activation、optimizer、scheduler、batch size）。

## 7. 评估指标与输出
- 使用同样的评估栈（accuracy、balanced acc、macro/weighted F1、kappa、按类指标、混淆图），产物命名/位置与 `results/<experiment>` 下的基线一致。

## 8. 论文写作解读
- 提供密集体素分类器在深度/跳连上的消融，观察增加容量/跳连是否实质改变类别 F1 或泛化。
- 若 Optuna 选择 DeepMLP，其表现可揭示纯密集特征混合的上限，再考虑模态感知或注意力模型。

## 9. 限制与开放问题
- DeepMLP 不是默认路径；实证结果取决于 Optuna 是否选中，因此可能需要专门跑。
- get_model_info 的字段（`input_dim`、`activation_name` 等）未在类内显式存储，保存的元数据可能不完整。
- 与基线一样，标签来源与体素数 UNKNOWN，脚本中的硬编码路径需更新。



---
## 来源文件 (Source file): DeepMLP6x2048-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): DeepMLP6x2048-Alex7T-102labels-汇总毕设.md
---

# DeepMLP6x2048 于 Alex7T-102labels

## 1. 在论文中的角色
- 源自贝叶斯搜索的最优 MLP 变体：更深网络 + StandardScaler 归一化，用于提升 Alex 7T 体素签名任务的基线性能。
- 作为强力 FC 参考模型，用于对比更复杂架构并支持下游推理（如 demo38 部署）。

## 2. 代码文件与入口
- `train_best_model.py`：端到端脚本，拟合 StandardScaler，按固定最优超参训练深层 MLP，保存 scaler + 检查点 + 报告。
- `data/dataset.py`：加载重构的 train/test/val 体素特征；此处使用 `norm=False`，以便统一应用外部 StandardScaler。
- `utils/metrics.py`、`utils/visualization.py`：最终模型的评估与绘图。
- `utils/model_io.py`：保存包含架构元数据的检查点及 JSON 辅助文件，并安全加载。
- `predict.py` / `predict.sh`：推理流水线，使用保存的模型与 scaler 处理外部 `.mat` 体积。

## 3. 数据集与标签
- 同 Alex 超多模态 7T 数据（341 通道体素签名，102 类，背景忽略），固定 train/val/test 目录。
- 标签空间与 FC 基线一致；逐体素采样，除损失中的类别权重外无额外平衡。

## 4. 预处理流程
- 在训练样本上拟合 StandardScaler，保存为 `<experiment_name>_scaler.pkl`；同一 scaler 应用于 val/test 并供推理复用。
- 归一化统计（均值/方差范围）写入文本报告；无 PCA 或降维。

## 5. 模型结构
- `DeepMLP`（`models/deep_mlp.py`），6 个 2048 单元的隐藏层，GELU 激活，每层后 dropout ≈0.2567，启用可选跳连。
- 线性头输出 102 logits；无 batch norm；面向稠密表格式体素特征。

## 6. 训练配置
- 损失：加权交叉熵，忽略背景。
- 优化器：Adam，lr ≈1.697e-4，weight_decay ≈1.33e-5；batch 128；30 轮；每轮验证。
- 学习率调度：StepLR，step_size=2，gamma≈0.2216。
- 类别权重由训练标签计算；每次验证保存检查点，记录架构元数据与归一化参数。

## 7. 评估指标与输出
- 指标与基线一致：accuracy、balanced accuracy、macro/weighted F1、kappa、按类 precision/recall/F1、混淆矩阵。
- 输出存于 `./best_model_results/<experiment_name>`，日志在 `./best_model_logs`：scaler `.pkl`、归一化统计文本、训练 CSV/日志、各划分指标/图表、最终 `test_final` 评估文件，以及模型 `.pth` + `_architecture.json`。

## 8. 论文写作解读
- 代表经过调优的 FC/MLP 方案，受益于更深容量与一致的 z-score，可带来更高的 macro-F1 和更稳的按类表现。
- 适合作为对 KAN/TabNet/其他方法的主要对照；保存的 scaler 使在新受试上可复现推理。

## 9. 限制与开放问题
- 超参固定于一次搜索；未做交叉验证或不同划分的鲁棒性分析。
- 仍基于体素级划分；受试级泄漏风险未测试。
- 无校准或不确定性评估；仅记录准确率类指标。
- 依赖已保存的 scaler；若不匹配，将损害推理。



---
## 来源文件 (Source file): FC4x4096-Alex7T-101labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): FC4x4096-Alex7T-101labels-汇总毕设.md
---

# FC4x4096 于 Alex7T-101labels

## 1. 在论文中的角色
German et al. 2021 的 Alex 超多模态 7T 数据集体素级分类的全连接基线。提供可复现的参考（按患者标准化，101 类含背景），以便与更深/残差变体和后续架构比较。

## 2. 代码文件与入口
- `main.py`：CLI，负责配置加载、数据集准备、模型构建、训练、评估、报告。
- `config.py`：默认超参、按患者/全局标准化预设、背景处理预设、固定 prob_idx 划分。
- `models/base_mlp.py`：4×4096 全连接 MLP 定义，可配置激活/Dropout。
- `train.py`：训练循环，考虑背景的 accuracy/F1，CSV + 文本日志，保存检查点。
- `data/mat_loader_patientwise.py`、`data/mat_loader.py`：MAT 加载器，支持按患者或全局缩放、固定或随机划分、背景过滤、Scaler 持久化。
- `data/samplers.py`：按患者的 batch 采样器与平衡采样器。
- `utils/metrics.py`、`utils/visualization.py`：accuracy/F1/balanced-accuracy/kappa、按类统计、混淆矩阵图、训练曲线。
- `utils/label_processing.py`：根据保留/忽略/过滤背景模式映射标签并构建 criterion。
- `utils/model_io.py`：保存包含架构与归一化参数的检查点；安全加载。
- 运行器：`run.sh`（101 类 MAT 的交互式训练）、`evaluate.py`（离线评估）、`predict.py` + `predict.sh`（MAT 推理 + 3D 回写），notebook `alex版本优化*.ipynb`、`train_with_best_hyperparams.ipynb` 用于手动调参。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，文件 `TRAIN38_no_label43.mat`。
- 输入：341 通道体素签名（多对比特征；具体模态列表未写于代码）。
- 标签：默认 101 类（0 背景 + 1–100 组织；去掉标签 43）。可选 100 类变体（过滤或忽略背景）。
- 划分：固定 prob_idx——验证 `[20]`，测试 `[38]`，其余为训练。提供替代方案（如 test `[13,23,38]`、更大测试集）或在缺少固定 ID 时用 `test_size=0.01` 的随机划分。
- 采样：可选按患者 batch，保证每批 ≥3 个患者；类别权重由训练标签计算用于失衡缓解。

## 4. 预处理流程
- 按患者标准化（默认）：每患者均值/方差，epsilon 1e-10；可选全局 `StandardScaler`。
- Scaler 保存为 `scaler.joblib`，在评估/推理中加载；归一化参数也存入检查点。
- PCA 默认关闭（`apply_pca=False`, `n_pca=0`）；虽有接口，但配置未用。
- 背景处理可配置：保留为类别 0、在损失中忽略（映射为 -1）、或直接删除背景样本；标签相应重映射。
- 数据完整性检查报告标签范围与 prob_idx 覆盖，避免划分泄漏。

## 5. 模型结构
- 顺序 MLP：四层 4096 隐藏单元（可配置），激活可选（默认 `relu`，可用 `gelu`/`swish`），每层后 dropout 0.5，线性分类头输出 `num_class`。
- 输入维度取自数据（341 特征）；无跳连/残差。
- 检查点记录架构、隐藏层大小、激活、dropout、归一化参数。

## 6. 训练配置
- 损失：交叉熵 + 类别权重；忽略背景时使用 `ignore_index`。
- 优化器：默认 AdamW（可选 Adam）；lr=1e-5，weight_decay=1e-5；batch 128；30 轮；每 3 轮验证；seed 666。
- 学习率调度：默认 cosine；亦支持 multistep 与 ReduceLROnPlateau。
- 每次验证保存检查点，文件名含 epoch/acc/F1；日志文本与 CSV 存于 `results/<experiment_name>/` 与 `logs/<experiment_name>/`。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1 与样本数。
- 可视化：混淆矩阵（对数尺度）、最佳/最差类别柱状图、训练曲线、数据分布图。
- 输出：`<dataset>_evaluation_report.txt`、`<dataset>_class_metrics.csv`、`<dataset>_metrics.csv`、`<dataset>_predictions.npz`、`_confusion_matrix.png`、`training_curves.png`、`evaluation_summary.txt`，检查点 `.pth` 以及 `_architecture.json` 与 scaler/joblib。
- 最优模型以验证 macro-F1 选取（`utils.metrics.get_best_model`）。
- 推理（`predict.py`）：加载已存 scaler/归一化，输出 `predictions.mat`、`probabilities.mat`、`volume_3d.mat`、可选 3D 可视化、`normalization_info.txt`。

## 8. 论文写作解读
- 确立在 341 通道签名上进行体素级 FC 分类的可行性，结合按患者归一化与显式背景处理。
- 提供混淆矩阵与按类行为，为后续（深层 MLP、残差 MLP、超参搜索或 KAN/TabNet 等架构）对比提供参考。
- 演示从 MAT 导入到 3D 预测导出的端到端流程，可用于“基线模型与数据处理”章节。

## 9. 限制与开放问题
- 数据路径硬编码（`/home/jovyan/...`）需迁移；各划分体素数未在代码中记录。
- 验证/测试默认仅依赖单个 prob_idx（20/38），跨受试泛化不明。
- 无校准指标（ECE/MCE）或不确定性估计；除逐体素特征外无空间上下文或增强。
- PCA/降维关闭；失衡处理仅限静态权重；背景过滤变体没有配套的专用评估脚本。



---
## 来源文件 (Source file): FC4x4096-Alex7T-102labels-patientwiseStd-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): FC4x4096-Alex7T-102labels-patientwiseStd-汇总毕设.md
---

# FC4x4096 于 Alex7T-102labels-patientwiseStd

## 1. 在论文中的角色
PyTorch 版的 4×4096 全连接基线，用于 German et al. 2021 的 Alex 超多模态 7T 数据集体素分类。此 notebook 测试按患者标准化（每受试 z-score），以更好匹配各受试的谱分布，相比全局归一化更贴合个体。作为与原 TensorFlow 基线的校核与可复现性检查，并评估患者特定缩放对单受试验证的影响。

## 2. 代码文件与入口
- `update_standarlize/alex的torch版本_新标准化措施.ipynb`：端到端 notebook，四个单元：(1) 从 `.mat` 读取数据与按患者标准化工具；(2) 4×4096 稠密网络定义与训练循环；(3) 在 `DEMO38.mat` 上推理并应用患者级统计；(4) 对选定验证体素做梯度显著性可视化。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，从 `TRAIN38.mat`（`data`、`region`、`prob_idx`）和 `DEMO38.mat`（`multidim_data`，可选 `prob_idx`）加载。
- 输入维度：341 通道体素签名（逐体素谱/模态特征），跨受试聚合数百万体素（具体数量未列）。
- 标签空间：`region` 中的 102 类 one-hot 标签，可能来自对齐 CEST/原始网格的 FreeSurfer 式分割；为硬 one-hot。
- 划分：训练使用除患者 38 外所有受试（`prob_idx != 38`）；验证仅患者 38（`prob_idx == 38`）。额外 `train_test_split` 留出 1% 非 38 数据，但训练只用 `X_train1`/`y_train1`（小留出未用）。测试/推理单元需要 `DEMO38.mat`，优先使用训练统计。

## 4. 预处理流程
- 按患者标准化：为每个患者计算均值/标准差，并在该患者内部做 z-score；零标准差替换为 1.0。返回并存储 `train_patient_stats` 与 `val_patient_stats`。
- 验证使用验证患者自身统计；推理时若患者 ID 已知则用训练统计，否则 fallback 为测试数据上现算的患者统计。
- 无额外缩放、截断或插补，假设特征稠密。

## 5. 模型结构
- 与 TensorFlow 基线匹配的全连接网络：341 → 4096 → 4096 → 4096 → 4096 → 102。
- 激活：每层 ReLU；每层后 Dropout(p=0.5)。
- 输出：线性 logits（softmax 在推理外部）；无 batchnorm 或残差。

## 6. 训练配置
- 优化器：Adam (lr=1e-5)；仅对权重矩阵手动 L2（weight_decay 1e-5 加入损失）。
- 损失：对 one-hot 取 argmax 的 CrossEntropyLoss。
- Batch size：128；Epochs：25；为 torch/NumPy 设定确定性种子；若可用则用 GPU。
- 每轮记录指标：训练 loss/accuracy/macro F1，验证集（患者 38）loss/accuracy/F1。
- 模型检查点：保存至 `outputs/dense_4x4096_model.pth`。

## 7. 评估指标与输出
- 指标：训练与验证的 macro F1、总体准确率；逐轮打印，末尾报告最佳验证 accuracy/F1。两者越高越好。
- 输出：`outputs/train_patient_stats.npy`、`outputs/val_patient_stats.npy`（按患者均值/方差）；`outputs/dense_4x4096_model.pth`（检查点）；通过 `scipy.io.savemat` 将预测导出为 `dense_4x4096_model_prediction.mat`（需用户自设 `predictpath`）；验证体素的梯度显著性图（显示，默认不保存）。

## 8. 论文写作解读
此实验在引入患者级归一化的同时验证核心 FC 基线的可复现性，缓解体素签名的跨受试分布偏移。展示了 4×4096 稠密网络可在 PyTorch 中训练，与 TensorFlow 版本类似的正则；验证使用单一受试（患者 38）。显著性可视化为光谱通道贡献提供初步可解释性视角。

## 9. 限制与开放问题
- 验证仅依赖单一受试（患者 38）；对其他受试的泛化未知。
- `train_test_split` 留出的少量数据未用，除患者 38 外无专用测试集。
- `predictpath` 为占位，保存预测需用户提供路径。
- 341 通道的类别定义与模态拆分未写入 notebook；无下游校准或更广指标（ECE、混淆矩阵）。
- 验证标准化使用自身统计而非训练统计，可能与部署时行为不符。



---
## 来源文件 (Source file): FC4x4096-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): FC4x4096-Alex7T-102labels-汇总毕设.md
---

# FC4x4096 于 Alex7T-102labels

## 1. 在论文中的角色
复现原始的 Alex 全连接体素分类器（4×4096），建立硬标签基线，生成体素概率体积供后续后处理与与空间/校准模型比较。

## 2. 代码文件与入口
- `training/B0_1D_training/train_1d_with_3d_dataset.py`：主训练/预测脚本，在 1D MAT 上训练并重建 3D 体积。
- `training/B0_1D_training/run_1d_training.sh`、`training/B0_1D_training/run_1d_leave_one_out.sh`：单次或 38 折留一封装。
- `training/B0_1D_training/visualize_1d_3d_predictions.py`：从保存的概率图做切片可视化与快速准确率检查。
- `training/B0_1D_training/README_1D_TRAINING.md`：用法与架构回顾。
- `erosion/train_38fold.py`、`erosion/run_training.sh`、`erosion/analyze_results.py`：扩展交叉验证，含 AMP/DataParallel、ONNX 导出、折内报告。
- `erosion/adjacency_matrix/compute_adjacency_matrices.py`：可选计算 102 区域的邻接矩阵，用于错误/混淆分析。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；38 名受试，体积 (384, 336, 256)。
- 输入：351 模态体素特征（`multidim_data`，转置为 (n_voxels, 351)），与区域掩膜对齐。
- 标签：102 类硬标签，来自 FreeSurfer 分割（`seg_one_hot` → argmax）；背景排除。
- 划分：按受试留一（37 训 / 1 测），可选受试内采样；erosion 脚本支持内部验证。

## 4. 预处理流程
- 每受试在 351 通道上做 z-score 归一化（StandardScaler）。
- 可选每受试下采样以提速；否则覆盖全部 ROI。
- 区域掩膜保证体素有效；预测通过布尔索引映射回 3D (384×336×256×102)。
- 做形状验证与转置处理以保持通道/体素顺序一致。

## 5. 模型结构
- 全连接网络：351 → 4096 → 4096 → 4096 → 4096 → 102，隐藏层后 ReLU + Dropout(0.5)。
- 无 batch/weight norm；仅对权重矩阵施加 1e-5 的手动 L2。
- erosion 流水线支持 ONNX 导出以便部署。

## 6. 训练配置
- 优化器 Adam (lr=1e-5) 加手动 L2；batch 128；训练 25 轮。
- 损失：硬标签的交叉熵；默认无 label smoothing 或类别权重。
- 通过 CLI 选择留一受试；erosion 变体强制使用 GPU，支持 AMP 与多卡 DataParallel。
- 每轮记录 train/test loss 与 macro-F1；以测试 F1 缓存最佳模型。

## 7. 评估指标与输出
- 指标：macro-F1、整体准确率；erosion 流水线额外给出按类 precision/recall/F1，macro/micro/weighted 均值与混淆报告。
- 输出：`.pth` 检查点（每个测试受试），可选 `.onnx`，训练历史 JSON/日志，3D 概率体积（`predictions_3d_test*.mat`），切片/不确定性可视化。

## 8. 论文写作解读
基线体素分类器，贴合已发表的 Alex 设置，展示非空间 MLP 在 351 通道输入上的上限。流水线还验证了概率图可靠回写 3D，为后续平滑与空间消融奠基。38 折结果为软标签、卷积、校准等实验提供锚点。

## 9. 限制与开放问题
- 超参固定，无自动调参；默认无校准或类别失衡处理。
- 依赖固定的 1D MAT 格式；若预处理变化，存在方向不一致风险。
- 采样策略与缺少增强可能限制泛化；仅支持硬标签。



---
## 来源文件 (Source file): FC4x4096-Alex7T-softlabels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): FC4x4096-Alex7T-softlabels-汇总毕设.md
---

# FC4x4096 于 Alex7T-softlabels

## 1. 在论文中的角色
在下采样到 CEST 分辨率的数据上，用概率标签训练 4×4096 网络，利用部分体积信息并评估校准/不确定性处理。

## 2. 代码文件与入口
- `training/downsampling/train_runner.py`：单划分训练器（36 训 / 1 验 / 1 测），带软标签指标与 3D 重建。
- `training/downsampling/notebook_quickstart.ipynb`、`training/downsampling/notebook_loso_37fold.ipynb`、`training/downsampling/NOTEBOOK_VERIFICATION.md`：基于 notebook 的运行与验证步骤。
- `dataset_create/downsampling/mri_downsampling_pipeline.py`、`dataset_create/downsampling/DOWNSAMPLED_DATA_FORMAT.md`：使用模态特定 PSF/间距生成带软标签的 CEST 分辨率 NPZ。
- `dataset_create/1d-3d-convert/data_3d_1d_mapper.py`：通用 3D↔1D 转换器，保持 C 序体素对齐。

## 3. 数据集与标签
- 数据集：Alex 超多模态 7T 数据下采样至 CEST 分辨率（≈1.8×1.8×3.0 mm³），存为成对 1D/3D NPZ。
- 输入：每体素 351 通道特征（`multidim_data`，形状 (n_vox, 351)）。
- 标签：102 类概率标签（`seg_one_hot` / `proba_labels`），保留部分体积信息；`region` 掩膜背景。
- 划分：固定 36/1/1 受试划分（train/val/test）；CLI 可选覆盖。

## 4. 预处理流程
- 每受试在 351 通道上 z-score；可选限制每受试体素数以控内存。
- 下采样流水线应用模态族特定 PSF/重采样并重计算参数图；掩膜保持 ROI 对齐。
- Data3D1DMapper 在保持体素顺序的前提下恢复 3D softmax 体积；记录各受试的归一化统计。

## 5. 模型结构
- 稠密 4×4096 MLP，Dropout(0.5) + ReLU；输出 102 维 logits。
- 对权重施加 1e-5 手动 L2；无 batchnorm 或注意力。

## 6. 训练配置
- 损失：针对概率标签的软交叉熵，可选类别权重（alpha=0.5），含 L2 正则与梯度裁剪（默认 1.0）。
- 优化器 Adam（lr=1e-5，weight_decay 由手动 L2 提供）；默认 batch 256；默认训练 3 轮（快速跑），以验证 NLL 选最佳检查点。
- 训练后做温度缩放以校准；默认随机种子 42。

## 7. 评估指标与输出
- 指标：NLL、整体准确率、macro/micro/weighted F1、top-k 准确率、Brier 分数及 Murphy 分解、class-mass error、soft-ECE（按类与均值）、AURC（风险-覆盖）、熵直方图。
- 校准：可靠性图、温度缩放摘要、软混淆矩阵。
- 3D 指标：soft Dice（macro）、ROI 内 3D NLL/Brier、恢复后的切片图。
- 输出：最佳模型 `best.pth`，`metrics_{val,test}.json`，`temperature_scaling.json`，`run_summary.json`，混淆 CSV，可靠性/风险-覆盖/熵图，3D 预测（`val_*/test_*_pred_softmax_3d.npz`）。

## 8. 论文写作解读
检验概率标签与校准是否能相对硬标签基线提升体素预测与不确定性估计。流水线突出校准指标（soft-ECE、AURC、Brier），并能报告 3D 概率质量（soft Dice），为“软标签与校准”章节提供依据。

## 9. 限制与开放问题
- 默认训练很短（epochs=3）；可能需要更长训练以达峰值表现。
- 假设下采样 NPZ 与一致的 CEST 分辨率掩膜已存在。
- 未做完整留一；结果仅代表一次 36/1/1 划分，除非手动扩展。



---
## 来源文件 (Source file): FixedMLP-BackgroundWeighting-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): FixedMLP-BackgroundWeighting-Alex7T-102labels-汇总毕设.md
---

# FixedMLP-BackgroundWeighting 于 Alex7T-102labels

## 1. 在论文中的角色
- 用于诊断 Alex 超多模态数据集体素分类中的背景处理与类别失衡策略。
- 对比四种情景：有/无背景训练 × 有/无逆频次类别权重，以解决 NaN 损失和性能下降。

## 2. 代码文件与入口
- comparison_alex/4场景对比/comparison_experiment.py：端到端实验驱动，定义四种情景、数据加载、训练、评估与绘图。
- comparison_alex/4场景对比/run_4_senario.sh：nohup 封装，启动实验并记录输出。
- comparison_alex/4场景对比/4场景对比alex代码数据.ipynb：同一比较的探索式 notebook 版本（此处未运行）。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T（TRAIN38.mat）。
- 输入：每体素 341 特征（转置为样本 × 特征）。
- 标签：含背景的 102 类 one-hot；情景要么保留背景作为类别，要么在训练时映射为 ignore_index (-1) 并仅训练 101 前景类。
- 划分：按受试——prob_idx==38 作为验证，其余患者中 1% 为测试，剩余为训练。

## 4. 预处理流程
- 在训练体素上拟合 StandardScaler；应用于 val/test；打印 scaler 统计做 QC。
- 数据集类按情景可选背景过滤；过滤时可移除验证集背景避免 all-ignore 批次。
- 无 PCA 或其他特征选择。

## 5. 模型结构
- FixedMLP：4×4096 ReLU + Dropout(0.5) 的全连接堆叠，Xavier 初始化；可选 L2 正则（1e-5）加到损失。
- 输出维度在忽略背景时为 101，保留背景时为 102。

## 6. 训练配置
- 优化器：Adam，lr=1e-5；batch_size=128；epochs=25（NUM_EPOCHS_DEMO）。
- 损失：CrossEntropyLoss；过滤背景时 ignore_index=-1。类别权重可按逆频次在不同情景下启用。
- 每轮记录指标：验证 accuracy 与 macro F1；训练 loss/acc/F1 亦记录。

## 7. 评估指标与输出
- 指标：验证 accuracy 与 macro F1（越高越好）；包含 NaN 检查防止无效损失。
- 输出：comparison_results_fixed/ 目录，含 final_results_summary_fixed.txt（各情景验证指标）、validation_curves_comparison_fixed.png、training_loss_comparison_fixed.png，以及内存中的各情景历史。

## 8. 论文写作解读
- 检验将背景作为可学习类别或忽略，以及应用类别权重，是否能稳定训练并提升验证表现。旨在为主流水线的背景处理选择提供依据。

## 9. 限制与开放问题
- 路径硬编码到 /home/jovyan/... 的 TRAIN38.mat，复用前需调整。
- 仅报告验证指标；无独立测试或校准分析。
- 使用单一固定架构与学习率；情景仅隔离背景/加权效应，未考察架构依赖行为。



---
## 来源文件 (Source file): KAN-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): KAN-Alex7T-102labels-汇总毕设.md
---

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



---
## 来源文件 (Source file): KAN-Binary-Alex7T-1vRest-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): KAN-Binary-Alex7T-1vRest-汇总毕设.md
---

# KAN-Binary 于 Alex7T-1vRest

## 1. 在论文中的角色
- 在 German et al. 2021 的 Alex 超多模态 7T 数据集上，建立 Kolmogorov-Arnold Networks (KAN) 的一对多体素基线。
- 探索 KAN 是否比简单 MLP 更好处理高维多模态体素签名，结合采样策略与阈值扫描进行类校准分析。
- 作为全 102 类之前的逐标签检测入口。

## 2. 代码文件与入口
- `brain_voxel_kan_project/main.py`：二分类 FastKAN 的 CLI 训练/评估，处理 PCA、采样策略、检查点与数据可视化。
- `brain_voxel_kan_project/train.py`：核心训练循环，跟踪 accuracy/F1/recall/AUC-PR，检查点命名（`epoch_*_acc_*_f1_*_aucpr_*.pth`），评估辅助（ROC/PR 曲线）。
- `brain_voxel_kan_project/datasets.py`：通过 `label_index.txt` 加载按标签切分的 `.npy` 特征，支持平衡/分层/改良分层采样，可选 PCA + min-max 缩放。
- `brain_voxel_kan_project/models.py`：定义 `BrainVoxelKAN`（FastKAN，layers [input_dim, hidden_dim, num_classes]）。
- `brain_voxel_kan_project/evaluate_epochs.py`：对各轮检查点扫评，重算 train/test/val/merged 指标，导出 `.pkl/.csv/.png` 摘要与阈值分析。
- `brain_voxel_kan_project/config.py`：默认超参（341 输入，label_id=1，二分类，PCA 开，neg:pos=5，grid=10，lr=1e-3，batch=500，epochs=100）。
- `brain_voxel_kan_project/run_full_experiment.sh`：自动化 200 轮训练 + 全轮评估（batch 640）。
- Notebook（`1DKAN_brainvoxel.ipynb`、`brain_voxel_fast_kan_step_same_data_adam_*`、`brain_voxel_fast_kan_step_same_data_adam_focalloss*`、`brain_voxel_fast_kan_step_same_data_adam_创建数据集*`）：用于数据重组、focal-loss/早停变体和逐标签数据集创建的交互迭代。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，展平成逐体素特征向量。
- 输入维度：每体素 341 模态/特征（diffusion/QTI/CEST/MPRAGE/QSM/SMWI）；PCA 可自动选取解释约 95% 方差的组件或用户指定 `N_PCA`。
- 标签空间：提供 102 个解剖类别；每次运行选择一个 `label_id` 为正类（默认 1），其余作为负类（一对多）。
- 标签为整型 ID，从 `label_index.txt` 索引的逐标签 `.npy` 文件读取；无软标签或部分体积处理。
- 划分：预生成的 `restructured/{train,test,val,merged}` 目录；采样器按配置的 neg:pos 比例（默认 5:1）做平衡/分层/改良分层批次，可选限制负类子集。

## 4. 预处理流程
- 可选 PCA（sklearn）在 train/test/val 拼接样本上拟合；`N_PCA=0` 时，`analyze_pca_variance` 选出解释约 95% 方差的最小组件。
- PCA 后对所有集合做 min-max 缩放；若关闭 PCA，可保持原始特征或 z-score（notebook 有示例）。
- 数据平衡在采样阶段完成，不通过损失加权；背景体素在数据构造时已排除。

## 5. 模型结构
- FastKAN，层 `[input_dim, 64, 2]`，固定样条网格 10（可调）；无 dropout 或 batch norm。
- FastKAN 提供可学习的激活样条；可通过输入样条权重（`utils.analyze_kan_model`）提取特征重要性。

## 6. 训练配置
- 损失：`nn.CrossEntropyLoss`，2 类。
- 优化器：Adam（`lr=1e-3`, `weight_decay=1e-6`）；batch 500（脚本用 640）；训练 100–200 轮；每轮验证。
- 检查点：每轮评估时保存；`get_best_model` 以 AUC-PR/F1/accuracy 选最佳；可通过 `CHECK_POINT` 重新加载。
- Notebook 变体测试 focal loss、早停、不同负样本比例。

## 7. 评估指标与输出
- 指标：accuracy、precision、recall、F1、AUC-PR（主）、ROC-AUC；混淆矩阵与阈值扫描用于不同决策阈值。
- 输出：`.pth` 检查点、`.pkl/.csv` 多轮摘要、`.png` 的指标轨迹与阈值分析，保存在 `Results/BrainVoxel_1DKAN/BrainVoxel`（以及 `eval_results_label_<id>/`）。
- 解释：accuracy/F1/recall/AUC-PR/ROC-AUC 越高越好；混淆矩阵展示给定阈值下的误报/漏报取舍。

## 8. 论文写作解读
- 展示轻量级 KAN 在逐结构体素检测中的可行性，利用完整 341 通道签名，探索样条激活是否优于传统全连接。
- 改良分层采样 + PCA 流程示范如何在保留多模态方差的同时控制失衡；阈值扫描可用于后续校准图。
- 适合“逐标签体素基线”方法小节，便于报告按标签的灵敏度/特异度或 PR 曲线。

## 9. 限制与开放问题
- 数据路径为 Jupyter 环境的绝对路径且未版本化；真实体素数与训练权重缺失。
- 此处未记录性能数值；focal-loss、早停等变体的可靠性 UNKNOWN。
- 依赖外部 `fastkan` 库与预生成的 `restructured` 数据集；可复现性要求这些资源。



---
## 来源文件 (Source file): KAN-Multiclass-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): KAN-Multiclass-Alex7T-102labels-汇总毕设.md
---

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



---
## 来源文件 (Source file): KAN-RFFeatures-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): KAN-RFFeatures-Alex7T-102labels-汇总毕设.md
---

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



---
## 来源文件 (Source file): MLP-PatientSplit-Alex7T-Train38-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): MLP-PatientSplit-Alex7T-Train38-汇总毕设.md
---

# MLP-PatientSplit 于 Alex7T-Train38

## 1. 在论文中的角色
- 为 `.mat` 数据（如 TRAIN38/DEMO38）提供按患者划分与部署的支撑流程，可在新受试上评估已训 MLP，并重建 3D 标签体积。
- 作为主 FC/DeepMLP 训练的补充，提供外部案例的数据转换、缩放与预测工具。

## 2. 代码文件与入口
- `data/mat_loader.py`：加载 `.mat`（键 `data`/`multidim_data`、`region`、`prob_idx`），按患者 ID 划分，拟合/应用 StandardScaler，通过 `load_and_process_data` 构建 PyTorch dataloader。
- `predict.py`、`predict.sh`：加载包含架构元数据的 MLP 检查点，应用 scaler 或嵌入的归一化参数，预测体素类别，对低置信度阈值化，并将预测映射回 3D；保存 `.mat` 输出与可选 3D 可视化。
- `predict_standardized.py`：简化的 demo38 流水线，在线标准化并写预测/概率体积。
- `predict_config.json`：推理默认路径/键；安全加载检查点使用 `utils/model_io.py`。

## 3. 数据集与标签
- 输入：MATLAB `.mat` 体积；特征通常 341 通道（与 Alex 7T 体素签名一致），标签在 `region`；`prob_idx` 中的患者 ID 用于划分。
- 默认划分逻辑：若未提供 ID，则 `prob_idx != 38` 为训练，`prob_idx == 38` 为验证，随后划出小测试集；也可在 `config['dataset_split']` 中配置患者列表。
- 标签空间预计与 102 类图谱一致；背景处理沿用上游模型；仅硬标签。
- 若提供 ID，则按患者划分；否则使用默认启发式。

## 4. 预处理流程
- 在 `process_train38_data` 内对训练样本拟合 StandardScaler；保存为 `scaler.joblib` 或从检查点加载，供评估/推理。
- 此路径无需 PCA；利用区域掩膜将平坦预测重塑回 3D 体积。

## 5. 模型结构
- 通过检查点元数据复用已存 MLP 架构（基础/深层/残差）；推理脚本根据记录的超参实例化相应模型。
- 面向稠密表格体素特征；激活/Dropout/跳连/bottleneck 取决于加载的检查点。

## 6. 训练配置
- 此模块侧重准备 dataloader 与缩放；训练本身应镜像主循环（交叉熵 + 可选类别权重），但此处无专门的 patient-split 训练脚本。
- 预测脚本聚焦使用现有权重推理；可选概率阈值将不确定体素标为未知 (-1)。

## 7. 评估指标与输出
- 插入主评估工具时，指标与基线一致（accuracy、balanced accuracy、macro/weighted F1、kappa、按类报告）。
- 预测输出：`predictions.mat`、`probabilities.mat`、`volume_3d.mat`、可选 `probability_volume.mat`、`normalization_info.txt`，以及 3D 可视化 PNG；日志保存在结果目录。

## 8. 论文写作解读
- 提供部署路径，可在新的 7T 采集中测试 MLP 分类器，并从体素级预测重建体积分割。
- 可用于展示对留出受试（如患者 38）的泛化，并生成预测组织图的定性图像。

## 9. 限制与开放问题
- 外部 `.mat` 的类别数量与语义必须与已训模型一致，否则评估无意义。
- 未提供 patient-split 数据的端到端训练脚本；假定已有在重构数据上训练的检查点。
- 缩放策略依赖已保存的 scaler 或嵌入的归一化参数；若不匹配会影响性能。



---
## 来源文件 (Source file): MLPVariantsBayesOpt-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): MLPVariantsBayesOpt-Alex7T-102labels-汇总毕设.md
---

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



---
## 来源文件 (Source file): OriginalMLP-CouplingLR-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): OriginalMLP-CouplingLR-Alex7T-102labels-汇总毕设.md
---

# OriginalMLP-CouplingLR 于 Alex7T-102labels

## 1. 在论文中的角色
- 复现原始 notebook 风格的 4×4096 MLP 训练，研究架构宽/深与学习率的耦合对 Alex 超多模态数据集的影响。
- 提供快速扫描以在全量训练前挑选稳定超参。

## 2. 代码文件与入口
- comparison_alex/coupling_test.py：定义 OriginalStyleMLP 并在架构 × 学习率上网格搜索，记录结果并生成分析报告。
- comparison_alex/run_coupling_test.sh：便捷启动（nohup）脚本引用上述 Python。
- comparison_alex/对比.ipynb：相关探索 notebook（此处未运行）。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T（TRAIN38.mat）。
- 输入：341 通道体素向量。
- 标签：含背景的 102 类 one-hot；本复现未去背景。
- 划分：prob_idx!=38 用于训练/测试（1% 测试比例），prob_idx==38 用作验证。

## 4. 预处理流程
- 在训练子集上拟合 StandardScaler；应用于验证与 1% 测试切分。
- 无 PCA 或特征选择；数据转置为 (samples × 341)，标签转为 (samples × 102)。

## 5. 模型结构
- OriginalStyleMLP 基线：4×4096 ReLU + Dropout(0.5)，权重 L2 正则（1e-5）。
- 测试变体：wide_shallow（2×8192）与 narrow_deep（6×2048），激活/Dropout/L2 相同。

## 6. 训练配置
- 优化器：Adam；学习率扫描 {5e-6, 1e-5, 2e-5, 5e-5}。
- Batch 128；快速扫描训练 12 轮（原 notebook 为 25）；损失：CrossEntropyLoss + L2。
- 每次运行指标：验证 accuracy、macro F1、Cohen κ；记录训练轨迹。

## 7. 评估指标与输出
- 指标：验证 accuracy/F1/κ（越高越好）；跟踪每个实验的收敛。
- 输出：coupling_test_results/<timestamp>/ 存储 JSON/分析报告（results.json, analysis_report.txt），总结每种架构的最佳 lr 并给出推荐。

## 8. 论文写作解读
- 快速网格展示 lr 与架构深度/宽度的交互，帮助确定主 MLP 基线的 lr/架构，并指出稳定区间。

## 9. 限制与开放问题
- 训练轮数较少，数值仅供参考非最终基准。
- 包含背景类别；未测试 ignore_index 或软标签。
- 无外部测试或校准指标；关注验证耦合。



---
## 来源文件 (Source file): ProbSmoothing-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ProbSmoothing-Alex7T-102labels-汇总毕设.md
---

# ProbSmoothing 于 Alex7T-102labels

## 1. 在论文中的角色
对 FC 基线的体素概率图进行平滑后处理，测试无需重训分类器即可加入轻量空间正则的效果。

## 2. 代码文件与入口
- `training/B1_Probability_map_post_processing/smooth_postprocess_eval.py`：核心平滑/评估脚本（标准版与 gated 版）。
- `training/B1_Probability_map_post_processing/run_smooth_evaluation.sh`、`training/B1_Probability_map_post_processing/test_gated_smooth.sh`：可运行预设。
- `training/B1_Probability_map_post_processing/README_SMOOTH_POSTPROCESSING.md`：动机、指标与使用说明。

## 3. 数据集与标签
- 输入：来自 FC 基线的 3D 概率体积（`softmax_vol`，约 384×336×256×102），以及 Alex 数据集的对应标签/掩膜。
- 标签：102 个硬类别；评估限制在脑掩膜内。

## 4. 预处理流程
- 对每个切片做 2D 均值平滑（3×3 或 7×7 核），带掩膜归一化；支持矢状/冠状/轴位。
- 可选 gating：按类别门控（仅在预测类别内平滑）或不确定性门控（基于熵/间隔的 Sigmoid 混合）以保护边缘/高置信区域。

## 5. 模型结构
- 无训练模型；对现有概率图应用确定性的平滑算子。

## 6. 训练配置
- 通过 CLI 标志配置：核大小、轴向、快速卷积路径、gating 选项、不确定性参数；无优化循环。

## 7. 评估指标与输出
- 指标：整体准确率、macro-F1、Cohen κ、macro/micro AUPRC、相对原始概率的提升、混淆矩阵。
- 输出：平滑后的概率体积（HDF5）、CSV 指标、图表（混淆对比）、gating 行为日志。

## 8. 论文写作解读
展示轻量空间平滑如何改进噪声较大的体素预测，作为无需重训的低成本替代方案；可用于讨论失衡组织上的后处理影响。

## 9. 限制与开放问题
- 效果依赖于基线概率质量；最优核/gating 可能因受试/类别而异。
- 纯 2D 平滑可能缺乏层间一致性；除平滑外不做概率再校准。



---
## 来源文件 (Source file): PseudoInverse-FeatureSelection-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): PseudoInverse-FeatureSelection-Alex7T-102labels-汇总毕设.md
---

# PseudoInverse-FeatureSelection 于 Alex7T-102labels

## 1. 在论文中的角色
- 伪逆分类器的直接特征选择变体（LASSO/elastic-net），常跳过 PCA 以在原模态空间上选择稀疏特征，测试稀疏性是否提升体素分类精度与可解释性。

## 2. 代码文件与入口
- `proj/src/main.py`：通过 `--focus_on_fs`/`--skip_pca` 启用特征选择模式，控制实验采样与日志。
- `proj/src/feature_selector.py`：实现 LASSO 与基于 torch 的 elastic-net 选择器、稳定性分析（Jaccard），以及选择器的保存/加载。
- `proj/src/utils.py`：`create_feature_selection_without_pca_param_grid` 及针对特征选择扫描的采样策略。
- `proj/src/brain_voxel_dataloader.py`：`preprocess_data_with_feature_selection` 负责缩放、可选 PCA、调用选择器；支持缓存。
- `proj/src/experiment_manager.py`：调度逐实验或全局特征选择，保存选择器，绘制选择结果，并在选定特征上跑伪逆分类器。
- `proj/src/pseudoinverse_model.py`、`proj/src/model_evaluator.py`、`proj/src/visualization_utils.py`、`proj/src/gpu_utils.py`：与 PCA 流水线相同，但作用于选定特征。
- `proj/scripts/run_baseline.sh`：`EXPERIMENT_TYPE=fs` 时的特征选择基线。
- `proj/scripts/run_full_experiments.sh`：默认 fs 模式且 `SKIP_PCA=true`，启动大规模扫描（如 300 组合）。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，经 per-label 体素签名 `.npy` 与 `label_index.txt` 元数据准备到 train/test/val 文件夹。
- 输入维度：原始特征长度未明确（可能 >300 模态）；在 `skip_pca` 时选择器直接作用于原特征，选择 50–300 个特征。
- 标签：102 个硬类别，来自标签索引；假定为 CEST 网格上的 FreeSurfer 风格组织/ROI 标签。
- 划分与采样：沿用目录式 train/val/test；可选 `max_samples_per_label` 限制（脚本最多 2000）与平衡到 `target_samples`；受试级划分 UNKNOWN。

## 4. 预处理流程
- 可选标准或 min-max 缩放；类别平衡与 PCA 流水线一致。
- 特征选择：`FeatureSelector` 支持 `lasso`（L1 逻辑回归 OVR）与 `elastic_net_torch`（PyTorch 回归，L1/L2 惩罚）；`selection_mode` 固定 top-k 或阈值兜底；`max_features` 50–300；`l1_ratio` 控制稀疏度。
- 通过 K 折 Jaccard 与选择频率做稳定性分析；缓存变换后数据与选择器；`skip_pca` 强制在原特征空间选择。

## 5. 模型结构
- 与 PCA 基线相同的伪逆线性分类器（102 输出），但输入为已选特征；正则可选 none/L2/truncated。
- 可从选择器（前置）和伪逆权重（后置）获得特征重要性。

## 6. 训练配置
- 模型为闭式解；特征选择阶段对 LASSO/elastic net 迭代（max_iter 1000/2000，tol 1e-4/1e-5）。
- `create_feature_selection_without_pca_param_grid` 的参数：`apply_pca=False`，归一化 {standard,minmax,None}，类别平衡开关，伪逆正则 {none,l2,truncated} 及 `alpha` ∈ {0.0001,0.001,0.005,0.01,0.05,0.1,0.5,5.0}，feature_selection {lasso, elastic_net_torch}，`selection_mode` 固定，`max_features` {50,100,150,200,250,300}，`l1_ratio` {0.1,0.3,0.5,0.7,0.9,1.0}，`max_iter` {1000,2000}，`tol` {1e-4,1e-5}；`max_experiments` 控制采样（脚本用 300）。
- 基线 fs 跑对比含/不含额外 L2 的 LASSO；GPU 可选（CuPy/cuML）。

## 7. 评估指标与输出
- 指标：train/test/val 的 accuracy、balanced accuracy、macro/weighted F1、Cohen κ（越高越好）；保存按类 F1 与混淆矩阵。
- 额外特征选择可视化：`feature_selection_visualization.png`、`feature_stability_visualization.png`、`feature_selection_impact.png`、`selection_ratio_impact.png`、`l1_ratio_impact.png`、`feature_selection_comparison.png`，以及 summary_report_with_fs 中的最佳实验拷贝。
- 产物：`feature_selector.pkl`、`model.pkl`、参数/状态 JSON、实验日志 `experiment_log_with_fs.csv`、summary_report_with_fs 中排序的 CSV/HTML 报告，启用 global 模式时还有 `global_feature_selection/`。

## 8. 论文写作解读
探究在全模态集合上直接施加稀疏性是否能与 PCA 压缩相匹敌或更佳，用于体素组织分类。所选特征数量、稳定性（Jaccard/频率）及性能差异说明真正需要多少模态，以及哪些选择设置更稳健。

可作为特征选择消融小节：与 PCA 基线对照，论证显式稀疏性的利弊，并突出模态重要性模式。

## 9. 限制与开放问题
- 已选特征索引与性能未包含在库中；结果依赖外部实验输出。
- 受试级划分与泄漏控制未编码；当前设置为逐体素采样加可选平衡。
- 缺少校准/不确定性指标；关注 accuracy/F1。
- 假定配置路径下存在预提取体素签名与 `label_index.txt`。



---
## 来源文件 (Source file): PseudoInverse-PCA-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): PseudoInverse-PCA-Alex7T-102labels-汇总毕设.md
---

# PseudoInverse-PCA 于 Alex7T-102labels

## 1. 在论文中的角色
- 使用 PCA 压缩超多模态体素签名的闭式伪逆线性基线；为更深网络提供快速解析参考，并探查经典降维+正则对体素性能的影响。

## 2. 代码文件与入口
- `proj/src/main.py`：CLI 入口，切换 PCA vs 特征选择运行，记录日志，初始化 GPU，调度基线/全扫描。
- `proj/src/brain_voxel_dataloader.py`：从 `train/`/`test/`/`val/` 目录按标签读取 `.npy` 体素数组（使用 `label_index.txt`），处理归一化、PCA、类别平衡、变换缓存。
- `proj/src/utils.py`：定义以 PCA 为中心的参数网格与采样辅助。
- `proj/src/experiment_manager.py`：运行实验、缓存预处理、训练伪逆模型、记录指标、输出 HTML/CSV 汇总。
- `proj/src/pseudoinverse_model.py`：多类线性分类器，伪逆求解，支持 L2（Tikhonov）或截断 SVD 正则；从权重计算特征重要性。
- `proj/src/model_evaluator.py`：评估 accuracy/F1/Kappa，绘制混淆矩阵与按类图。
- `proj/src/visualization_utils.py`：特征重要性与权重分布可视化。
- `proj/src/gpu_utils.py`：CPU/GPU 抽象，CuPy/cuML 回退。
- `proj/scripts/run_baseline.sh`：`EXPERIMENT_TYPE=pca` 的示例启动。
- `proj/scripts/run_full_experiments.sh`：全量扫描启动器，可切换到 PCA 模式。
- `PseudoInverse_FeatureSelection_BrainVoxel.ipynb`：早期原型，流程相同。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，重组为 train/val/test 目录下的体素签名数组，附 per-label `label_index.txt` 元数据。
- 输入维度：原始特征长度未明（可能数百模态，含 MPRAGE/QTI/CEST/QSM）；PCA 探索 50–300 组件或自动选择解释 95% 方差的组件。
- 标签空间：默认 102 类，硬标签；假定为映射到 CEST 网格的 FreeSurfer 分割；通过 `valid_labels` 跳过无效标签。
- 划分：目录式 train/val/test；可选 `max_samples_per_label` 下采样与平衡到 `target_samples`（过/欠采样）；受试级策略未说明（UNKNOWN）。

## 4. 预处理流程
- 可选 z-score 或 min-max，在 PCA 前或后（`scaling_before_pca`）。
- PCA 降维，固定 `n_components` 或用 `auto_pca_variance`（默认 0.95）自动定组件数；变换通过 `_get_config_key` 与 `precompute_transformations` 缓存。
- 类别平衡对每类过/欠采样到 `target_samples`；可用 GPU 持续存放数据。
- 未显式处理缺失；特征重要性在训练后由权重给出。

## 5. 模型结构
- 解析多类线性模型：加入偏置列，计算设计矩阵的伪逆乘 one-hot 目标。
- 正则：无、L2/Tikhonov（强度 `alpha`），或截断 SVD（将小于 `alpha * max(s)` 的奇异值置零）。
- 输出 102 维 logits；特征重要性为各类权重绝对值均值。

## 6. 训练配置
- 每个实验仅一次闭式求解（无 epoch）；CuPy 可用时用 GPU，加速矩阵运算，否则 CPU。
- 通过 `create_feature_selection_param_grid` + 采样形成参数网格：`n_components` {50,80,100,150,200,250,300}，归一化 {standard,minmax,None}，正则 {none,l2,truncated}，`alpha` ∈ {0.001,0.01,0.05,0.1,0.5,1.0}，类平衡开/关，`max_experiments` 上限；`max_samples_per_label` 限制每类体素数。
- 基线跑比较 PCA vs PCA+L2 vs PCA+平衡；`auto_pca_variance` 可按解释方差覆盖 `n_components`。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ（越高越好）；保留分类报告与按类 F1。
- 可视化：`performance_comparison.png`、`confusion_matrix_{train|test|val}.png`（大矩阵用对数）、`class_f1_sorted.png`、`sample_count_vs_f1.png`、`top_bottom_classes.png`、`feature_importance.png`、`weight_distribution.png`。
- 日志与产物：`experiment_log_with_fs.csv`；每实验 `params.json`、`status.json`、`model.pkl`；可选 `summary_report_with_fs` 目录，含排序 CSV 与 HTML 报告。

## 8. 论文写作解读
本组测试 PCA 压缩后的 Alex 超多模态体素签名在 102 类组织上的线性可分性。对组件数、正则与类别平衡的消融展示降维与经典先验对 accuracy/F1 的影响。

可作为深度网络前的解析基线小节；生成的混淆矩阵与特征重要性图提供可分性与模态影响的快速 sanity check。

## 9. 限制与开放问题
- 原始特征数量与模态映射未在库中编码；假定已有 Alex 数据的体素签名。
- train/val/test 划分策略（逐体素 vs 逐受试）未说明；需外部元数据。
- 无校准或不确定性指标；评估仅关注 accuracy/F1。
- 脚本假设配置路径下存在 `label_index.txt` 与 `.npy`；重跑需这些数据。



---
## 来源文件 (Source file): QC-MultimodalRegistration-Alex7T-351modes-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): QC-MultimodalRegistration-Alex7T-351modes-汇总毕设.md
---

# QC-MultimodalRegistration 于 Alex7T-351modes

## 1. 在论文中的角色
在训练前评估多模态配准/对齐的质量控制套件，确保 351 通道输入空间一致、无明显伪影。

## 2. 代码文件与入口
- `dataset_create/multimodal_mri_qc_analysis/multimodal_mri_qc_analysis.ipynb`：相似性分析主 notebook（任务 1）。
- `dataset_create/multimodal_mri_qc_analysis/multimodal_qc_tasks_extra.py`、`multimodal_mri_qc_analysis1.py`：任务 2–5 的脚本（边缘/ROI/QC 评分）。
- `dataset_create/multimodal_mri_qc_analysis/MULTIMODAL_QC_ANALYSIS_README.md`、`QUICKSTART_QC_ANALYSIS.md`：流程、阈值、数据需求。
- `dataset_create/multimodal_mri_qc_analysis/CHANGELOG_QC_v1.1.0.md`、`CHANGELOG_QC_v1.2.0.md`：算法与性能更新。

## 3. 数据集与标签
- 数据集：Alex 7T 的 3D “minimal” 体积（384×336×256×351），MPRAGE 间距约 0.65 mm；ROI 掩膜/标签与训练数据共用。
- 标签：使用 FreeSurfer 派生区域做 ROI 检查；此处不生成训练标签。

## 4. 预处理流程
- 读取 351 通道，按模态族分组，使用 0.65 mm 的间距敏感指标，可选 CEST slab 定位。
- 支持各向异性 PSF 考虑，并用掩膜限定脑实质体素。

## 5. 模型结构
- 不适用（纯分析/QC）。

## 6. 训练配置
- Notebook/脚本的阈值参数：LNCC/NGF 下限、MIND-SSD、ASSD/HD95 上限（mm）、edge IoU、基于 MAD 的异常检测。
- 可选 ROI 列表与 UMAP/PCA 的降维设置。

## 7. 评估指标与输出
- 相似性矩阵：LNCC、NGF、MIND-SSD（LNCC/NGF 越高越好，MIND-SSD 越低越好），跨模态族。
- 边缘一致性：ASSD、HD95（越低越好），edge IoU（越高越好），并用自适应 Canny 阈值。
- ROI 信号一致性热图；PCA/UMAP 散点；QC PASS/WARN/FAIL 评分与 CSV/JSON 摘要。
- 输出：PNG 图（相似性矩阵、边缘指标、ROI 热图、PCA/UMAP）、`modality_qc.csv`、`qc_analysis_report.json`，位于 `qc_analysis_results/`。

## 8. 论文写作解读
为多模态对齐质量提供客观证据，训练前识别问题通道。支撑数据整理与配准可靠性的章节，尤其在论证剔除质量差病例时。

## 9. 限制与开放问题
- 需要正确的数据路径与针对数据集的参数调优；对 351 通道全部评估时耗时较长。
- QC 阈值具启发性；PASS/WARN/FAIL 的判定可能需要人工核查。



---
## 来源文件 (Source file): ResNet50Patch-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ResNet50Patch-Alex7T-102labels-汇总毕设.md
---

# ResNet50Patch 于 Alex7T-102labels

## 1. 在论文中的角色
更深的空间模型（约 50M 参数），检验丰富的空间处理与失衡感知损失是否能在 7×7 补丁上超越轻量 ConvPatch2D 与 MLP 基线。

## 2. 代码文件与入口
- `training/3D CNN/ResNet/scripts/train_mri_resnet.py`：主训练器，含损失/增强选项、早停与日志。
- `training/3D CNN/ResNet/models/resnet.py`：MRI 优化的 ResNet-50 定义（expand-first stem，base_width=104）。
- `training/3D CNN/ResNet/models/dataset.py`：补丁加载器，支持平衡/增强；`models/losses.py`：focal/类平衡/Logit 调整损失与 mixup 工具。
- `training/3D CNN/ResNet/configs/default_config.json`：默认超参；`scripts/run_leave_one_out.sh` 做批量 CV；`scripts/analyze_resnet_results.py` 汇总结果。

## 3. 数据集与标签
- 数据集：同 Alex 7T MAT（384×336×256×351），默认 7×7 补丁。
- 输入：每补丁 351 通道；可选类别平衡采样或加权采样。
- 标签：102 个硬类别（在加载器中移到 0–101）；掩膜去除背景。
- 划分：按受试留一；可选每受试采样（默认 1 万）或文档描述的“内存友好”全数据模式。

## 4. 预处理流程
- 每受试按通道 z-score；可缓存内存。
- 可选增强：随机翻转/旋转、轻 Gaussian 噪声（`--augmentation` 时）。
- 提供加权采样与类别权重以缓解失衡。

## 5. 模型结构
- ResNet-50（3-4-6-3 bottleneck），expand-first 3×3 stem（351→512），无 max pooling，base_width=104（~50M 参数）。
- 空间流：7→4→2→1；通道流：512→256→512→1024→2048→102 分类器。
- 可选 EMA 跟踪、mixup、梯度裁剪。

## 6. 训练配置
- 损失选项：CE、加权 CE、focal、类平衡 focal（默认 gamma=1.5, beta=0.9999）、logit-adjusted CE (tau)、balanced softmax；可选 label smoothing。
- 优化器 AdamW（lr=1e-4, weight_decay=1e-4）；CosineAnnealingLR（T_max=100, eta_min=1e-6）；梯度裁剪 1.0。
- Batch 256；训练 100 轮；早停 patience=15；可选 mixup (alpha=0.2) 与 EMA (decay=0.999)。

## 7. 评估指标与输出
- 指标：训练/测试 macro-F1，按类 precision/recall/F1/support；学习率与过拟合诊断。
- 输出：`best_model.pth`，每 10 轮检查点，`training_results.json`，`training_history.png`，`training.log`，位于 `resnet_test_subject_*` 目录。

## 8. 论文写作解读
检验更深空间建模与失衡感知目标是否优于简单补丁 CNN 与 MLP。结果为空间上下文、类别失衡处理及高级优化技巧（mixup/EMA）的消融提供依据。

## 9. 限制与开放问题
- 文档描述的“内存友好”全数据模式默认加载器仍采样；71M 体素全量训练的影响未验证。
- 无显式校准指标；损失超参（gamma/beta/tau）需调节。
- 补丁大小默认 7，改动需重训；数据路径需手动配置。



---
## 来源文件 (Source file): ResidualMLP-Alex7T-101labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ResidualMLP-Alex7T-101labels-汇总毕设.md
---

# ResidualMLP 于 Alex7T-101labels

## 1. 在论文中的角色
带残差的 MLP 变体，评估跳连（可选 bottleneck）是否能在 Alex 超多模态 7T 体素特征上相较普通/深层 MLP 稳定训练并提升精度。

## 2. 代码文件与入口
- `models/residual_mlp.py`：残差块实现，块内可选 bottleneck 压缩/展开。
- 训练/评估/推理栈与其他 MLP 共享（`main.py`、`train.py`、`config.py`、数据加载器、采样器、metrics、model_io、可视化）。
- 通过 `utils/optimization.py` 对 bottleneck 使用/宽度做超参搜索；可用 `run.sh --model_type residual_mlp` 或 BayesOpt 运行。

## 3. 数据集与标签
同 Alex 超多模态 7T 设置（341 特征，101 类，背景可配置）。固定 prob_idx 划分（val 20，test 38）除非覆盖；支持背景过滤/忽略与按患者批次/类别加权。

## 4. 预处理流程
与其他组共用：按患者/全局标准化，可选 PCA（关），可配置背景处理，保存 scaler，检查划分。

## 5. 模型结构
- 输入层后接若干残差块；每块包含两层线性 + 激活/Dropout，并加快捷分支（若维度不同则线性投影）。
- 当隐藏宽度>1000 时可启用 bottleneck，以 `bottleneck_factor`（默认 0.5）压缩再展开。
- 激活可选 `relu`/`gelu`/`swish`，Dropout 可调；最后线性头输出 `num_class`。

## 6. 训练配置
- 与基线相同：交叉熵+类别权重，AdamW/Adam，cosine/multistep/plateau 调度，batch 128，30 轮，每 3 轮验证。
- BayesOpt 试验可切换 bottleneck 与块宽；配置随检查点保存。

## 7. 评估指标与输出
- 指标/产物与基线一致（accuracy、balanced accuracy、macro/weighted F1、kappa、按类统计、混淆矩阵、训练曲线、预测导出）。最佳检查点按验证 macro-F1 选。

## 8. 论文写作解读
- 检验残差是否缓解优化难度与类别失衡敏感性，适用于高维体素特征。
- 适合放在架构微调的消融小节（普通 vs 深层 vs 残差），再过渡到新型模型。

## 9. 限制与开放问题
- 残差实现对检查点的元数据（bottleneck 标志）记录有限；需查看保存的配置确定块结构。
- 无空间先验或校准指标；仍依赖单一验证/测试患者。
- 性能提升取决于 BayesOpt 选定宽度；默认可能接近基线容量。



---
## 来源文件 (Source file): ResidualMLP-Alex7T-102labels-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): ResidualMLP-Alex7T-102labels-汇总毕设.md
---

# ResidualMLP 于 Alex7T-102labels

## 1. 在论文中的角色
- 带残差的 MLP 变体，旨在通过残差捷径和可选 bottleneck 缓解超宽层的优化难度，目标仍是 German et al. 2021 的 Alex 超多模态 7T 数据集。
- 通过共享的 Optuna 搜索探索残差/瓶颈是否能比纯 FC 基线带来更稳定的体素表现。

## 2. 代码文件与入口
- `models/residual_mlp.py`：定义残差块，含可选 bottleneck 压缩/展开，层宽变动时用线性捷径。
- 复用其他 MLP 的训练/评估栈：`main.py`、`train.py`、`models/config.py`、`data/mat_loader.py`、`utils/optimization.py`、`utils/metrics.py`、`utils/model_io.py`、`utils/visualization.py`。
- 通过 `--model_type residual_mlp` 选择，或让 Optuna 自动挑选。

## 3. 数据集与标签
- 同 TRAIN38/DEMO38 的 Alex 超多模态 7T；341 通道输入，102 类，背景=0 被忽略；按患者划分与类别加权与基线一致。

## 4. 预处理流程
- 同样的 StandardScaler 归一化与 Scaler 持久化；PCA 标志未用；在损失/指标中忽略背景。

## 5. 模型结构
- 输入线性层 → 残差块序列。每块主路径含两层线性 + 激活/Dropout（当 width>1000 且 use_bottleneck=True 时采用压缩-展开 bottleneck），捷径为恒等或线性以匹配维度，残差相加后激活。
- 最终线性层输出 102 logits。隐藏层选项包括 [4096,4096,4096,4096]、[2048,...]、[1024,2048,2048,1024]、[4096,2048,2048,4096]。

## 6. 训练配置
- 损失、优化器、调度器、epoch、batch 与验证频次同其他 MLP。
- Optuna 在共享超参（lr、weight_decay、dropout、activation、optimizer、scheduler、batch size）之外，探索 use_bottleneck 标志与 bottleneck_factor（0.25–0.5）。

## 7. 评估指标与输出
- 指标与产物位置与其他 MLP 相同（`results/<experiment>` 的日志、CSV、图表、检查点、摘要）。

## 8. 论文写作解读
- 检验残差/bottleneck 结构是否改善 dense 体素特征的优化稳定性与按类 F1，相比纯 FC 基线。
- 可作为消融，讨论在稠密网络内的结构微调是否足够，还是需要更结构化的模型。

## 9. 限制与开放问题
- ResidualMLP 需手动标志或 Optuna 选中；无专门配置保证已运行。
- dropout_rate/use_bottleneck 等元数据未存为属性，`get_model_info` 可能不全；复用前需查看检查点元数据。
- 共性不确定性仍在（图谱来源、体素数量、硬编码数据路径）。



---
## 来源文件 (Source file): SubjectEmbedding-Alex7T-EmbeddingFeasibility-汇总毕设.md
相对路径 (relative to SUMMARY_DIR): SubjectEmbedding-Alex7T-EmbeddingFeasibility-汇总毕设.md
---

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
