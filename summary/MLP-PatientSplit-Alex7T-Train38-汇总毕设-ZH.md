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
