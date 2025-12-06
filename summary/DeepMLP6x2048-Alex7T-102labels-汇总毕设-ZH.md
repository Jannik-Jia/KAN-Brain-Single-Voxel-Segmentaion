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
