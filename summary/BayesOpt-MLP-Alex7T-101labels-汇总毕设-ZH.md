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
