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
