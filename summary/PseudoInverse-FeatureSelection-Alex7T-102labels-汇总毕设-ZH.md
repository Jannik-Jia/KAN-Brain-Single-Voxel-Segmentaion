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
