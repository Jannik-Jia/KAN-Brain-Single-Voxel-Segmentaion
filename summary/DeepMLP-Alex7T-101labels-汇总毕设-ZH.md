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
