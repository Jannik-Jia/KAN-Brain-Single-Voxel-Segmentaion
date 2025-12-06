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
