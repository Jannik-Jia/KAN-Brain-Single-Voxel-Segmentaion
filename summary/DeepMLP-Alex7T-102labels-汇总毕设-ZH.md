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
