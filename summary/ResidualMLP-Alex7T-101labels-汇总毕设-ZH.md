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
