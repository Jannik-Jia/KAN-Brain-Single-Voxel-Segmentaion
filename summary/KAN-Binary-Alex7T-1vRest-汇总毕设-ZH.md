# KAN-Binary 于 Alex7T-1vRest

## 1. 在论文中的角色
- 在 German et al. 2021 的 Alex 超多模态 7T 数据集上，建立 Kolmogorov-Arnold Networks (KAN) 的一对多体素基线。
- 探索 KAN 是否比简单 MLP 更好处理高维多模态体素签名，结合采样策略与阈值扫描进行类校准分析。
- 作为全 102 类之前的逐标签检测入口。

## 2. 代码文件与入口
- `brain_voxel_kan_project/main.py`：二分类 FastKAN 的 CLI 训练/评估，处理 PCA、采样策略、检查点与数据可视化。
- `brain_voxel_kan_project/train.py`：核心训练循环，跟踪 accuracy/F1/recall/AUC-PR，检查点命名（`epoch_*_acc_*_f1_*_aucpr_*.pth`），评估辅助（ROC/PR 曲线）。
- `brain_voxel_kan_project/datasets.py`：通过 `label_index.txt` 加载按标签切分的 `.npy` 特征，支持平衡/分层/改良分层采样，可选 PCA + min-max 缩放。
- `brain_voxel_kan_project/models.py`：定义 `BrainVoxelKAN`（FastKAN，layers [input_dim, hidden_dim, num_classes]）。
- `brain_voxel_kan_project/evaluate_epochs.py`：对各轮检查点扫评，重算 train/test/val/merged 指标，导出 `.pkl/.csv/.png` 摘要与阈值分析。
- `brain_voxel_kan_project/config.py`：默认超参（341 输入，label_id=1，二分类，PCA 开，neg:pos=5，grid=10，lr=1e-3，batch=500，epochs=100）。
- `brain_voxel_kan_project/run_full_experiment.sh`：自动化 200 轮训练 + 全轮评估（batch 640）。
- Notebook（`1DKAN_brainvoxel.ipynb`、`brain_voxel_fast_kan_step_same_data_adam_*`、`brain_voxel_fast_kan_step_same_data_adam_focalloss*`、`brain_voxel_fast_kan_step_same_data_adam_创建数据集*`）：用于数据重组、focal-loss/早停变体和逐标签数据集创建的交互迭代。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，展平成逐体素特征向量。
- 输入维度：每体素 341 模态/特征（diffusion/QTI/CEST/MPRAGE/QSM/SMWI）；PCA 可自动选取解释约 95% 方差的组件或用户指定 `N_PCA`。
- 标签空间：提供 102 个解剖类别；每次运行选择一个 `label_id` 为正类（默认 1），其余作为负类（一对多）。
- 标签为整型 ID，从 `label_index.txt` 索引的逐标签 `.npy` 文件读取；无软标签或部分体积处理。
- 划分：预生成的 `restructured/{train,test,val,merged}` 目录；采样器按配置的 neg:pos 比例（默认 5:1）做平衡/分层/改良分层批次，可选限制负类子集。

## 4. 预处理流程
- 可选 PCA（sklearn）在 train/test/val 拼接样本上拟合；`N_PCA=0` 时，`analyze_pca_variance` 选出解释约 95% 方差的最小组件。
- PCA 后对所有集合做 min-max 缩放；若关闭 PCA，可保持原始特征或 z-score（notebook 有示例）。
- 数据平衡在采样阶段完成，不通过损失加权；背景体素在数据构造时已排除。

## 5. 模型结构
- FastKAN，层 `[input_dim, 64, 2]`，固定样条网格 10（可调）；无 dropout 或 batch norm。
- FastKAN 提供可学习的激活样条；可通过输入样条权重（`utils.analyze_kan_model`）提取特征重要性。

## 6. 训练配置
- 损失：`nn.CrossEntropyLoss`，2 类。
- 优化器：Adam（`lr=1e-3`, `weight_decay=1e-6`）；batch 500（脚本用 640）；训练 100–200 轮；每轮验证。
- 检查点：每轮评估时保存；`get_best_model` 以 AUC-PR/F1/accuracy 选最佳；可通过 `CHECK_POINT` 重新加载。
- Notebook 变体测试 focal loss、早停、不同负样本比例。

## 7. 评估指标与输出
- 指标：accuracy、precision、recall、F1、AUC-PR（主）、ROC-AUC；混淆矩阵与阈值扫描用于不同决策阈值。
- 输出：`.pth` 检查点、`.pkl/.csv` 多轮摘要、`.png` 的指标轨迹与阈值分析，保存在 `Results/BrainVoxel_1DKAN/BrainVoxel`（以及 `eval_results_label_<id>/`）。
- 解释：accuracy/F1/recall/AUC-PR/ROC-AUC 越高越好；混淆矩阵展示给定阈值下的误报/漏报取舍。

## 8. 论文写作解读
- 展示轻量级 KAN 在逐结构体素检测中的可行性，利用完整 341 通道签名，探索样条激活是否优于传统全连接。
- 改良分层采样 + PCA 流程示范如何在保留多模态方差的同时控制失衡；阈值扫描可用于后续校准图。
- 适合“逐标签体素基线”方法小节，便于报告按标签的灵敏度/特异度或 PR 曲线。

## 9. 限制与开放问题
- 数据路径为 Jupyter 环境的绝对路径且未版本化；真实体素数与训练权重缺失。
- 此处未记录性能数值；focal-loss、早停等变体的可靠性 UNKNOWN。
- 依赖外部 `fastkan` 库与预生成的 `restructured` 数据集；可复现性要求这些资源。
