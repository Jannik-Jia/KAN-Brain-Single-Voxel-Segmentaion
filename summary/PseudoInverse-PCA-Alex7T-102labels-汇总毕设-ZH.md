# PseudoInverse-PCA 于 Alex7T-102labels

## 1. 在论文中的角色
- 使用 PCA 压缩超多模态体素签名的闭式伪逆线性基线；为更深网络提供快速解析参考，并探查经典降维+正则对体素性能的影响。

## 2. 代码文件与入口
- `proj/src/main.py`：CLI 入口，切换 PCA vs 特征选择运行，记录日志，初始化 GPU，调度基线/全扫描。
- `proj/src/brain_voxel_dataloader.py`：从 `train/`/`test/`/`val/` 目录按标签读取 `.npy` 体素数组（使用 `label_index.txt`），处理归一化、PCA、类别平衡、变换缓存。
- `proj/src/utils.py`：定义以 PCA 为中心的参数网格与采样辅助。
- `proj/src/experiment_manager.py`：运行实验、缓存预处理、训练伪逆模型、记录指标、输出 HTML/CSV 汇总。
- `proj/src/pseudoinverse_model.py`：多类线性分类器，伪逆求解，支持 L2（Tikhonov）或截断 SVD 正则；从权重计算特征重要性。
- `proj/src/model_evaluator.py`：评估 accuracy/F1/Kappa，绘制混淆矩阵与按类图。
- `proj/src/visualization_utils.py`：特征重要性与权重分布可视化。
- `proj/src/gpu_utils.py`：CPU/GPU 抽象，CuPy/cuML 回退。
- `proj/scripts/run_baseline.sh`：`EXPERIMENT_TYPE=pca` 的示例启动。
- `proj/scripts/run_full_experiments.sh`：全量扫描启动器，可切换到 PCA 模式。
- `PseudoInverse_FeatureSelection_BrainVoxel.ipynb`：早期原型，流程相同。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，重组为 train/val/test 目录下的体素签名数组，附 per-label `label_index.txt` 元数据。
- 输入维度：原始特征长度未明（可能数百模态，含 MPRAGE/QTI/CEST/QSM）；PCA 探索 50–300 组件或自动选择解释 95% 方差的组件。
- 标签空间：默认 102 类，硬标签；假定为映射到 CEST 网格的 FreeSurfer 分割；通过 `valid_labels` 跳过无效标签。
- 划分：目录式 train/val/test；可选 `max_samples_per_label` 下采样与平衡到 `target_samples`（过/欠采样）；受试级策略未说明（UNKNOWN）。

## 4. 预处理流程
- 可选 z-score 或 min-max，在 PCA 前或后（`scaling_before_pca`）。
- PCA 降维，固定 `n_components` 或用 `auto_pca_variance`（默认 0.95）自动定组件数；变换通过 `_get_config_key` 与 `precompute_transformations` 缓存。
- 类别平衡对每类过/欠采样到 `target_samples`；可用 GPU 持续存放数据。
- 未显式处理缺失；特征重要性在训练后由权重给出。

## 5. 模型结构
- 解析多类线性模型：加入偏置列，计算设计矩阵的伪逆乘 one-hot 目标。
- 正则：无、L2/Tikhonov（强度 `alpha`），或截断 SVD（将小于 `alpha * max(s)` 的奇异值置零）。
- 输出 102 维 logits；特征重要性为各类权重绝对值均值。

## 6. 训练配置
- 每个实验仅一次闭式求解（无 epoch）；CuPy 可用时用 GPU，加速矩阵运算，否则 CPU。
- 通过 `create_feature_selection_param_grid` + 采样形成参数网格：`n_components` {50,80,100,150,200,250,300}，归一化 {standard,minmax,None}，正则 {none,l2,truncated}，`alpha` ∈ {0.001,0.01,0.05,0.1,0.5,1.0}，类平衡开/关，`max_experiments` 上限；`max_samples_per_label` 限制每类体素数。
- 基线跑比较 PCA vs PCA+L2 vs PCA+平衡；`auto_pca_variance` 可按解释方差覆盖 `n_components`。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ（越高越好）；保留分类报告与按类 F1。
- 可视化：`performance_comparison.png`、`confusion_matrix_{train|test|val}.png`（大矩阵用对数）、`class_f1_sorted.png`、`sample_count_vs_f1.png`、`top_bottom_classes.png`、`feature_importance.png`、`weight_distribution.png`。
- 日志与产物：`experiment_log_with_fs.csv`；每实验 `params.json`、`status.json`、`model.pkl`；可选 `summary_report_with_fs` 目录，含排序 CSV 与 HTML 报告。

## 8. 论文写作解读
本组测试 PCA 压缩后的 Alex 超多模态体素签名在 102 类组织上的线性可分性。对组件数、正则与类别平衡的消融展示降维与经典先验对 accuracy/F1 的影响。

可作为深度网络前的解析基线小节；生成的混淆矩阵与特征重要性图提供可分性与模态影响的快速 sanity check。

## 9. 限制与开放问题
- 原始特征数量与模态映射未在库中编码；假定已有 Alex 数据的体素签名。
- train/val/test 划分策略（逐体素 vs 逐受试）未说明；需外部元数据。
- 无校准或不确定性指标；评估仅关注 accuracy/F1。
- 脚本假设配置路径下存在 `label_index.txt` 与 `.npy`；重跑需这些数据。
