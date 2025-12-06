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
