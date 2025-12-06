# FC4x4096 于 Alex7T-102labels

## 1. 在论文中的角色
复现原始的 Alex 全连接体素分类器（4×4096），建立硬标签基线，生成体素概率体积供后续后处理与与空间/校准模型比较。

## 2. 代码文件与入口
- `training/B0_1D_training/train_1d_with_3d_dataset.py`：主训练/预测脚本，在 1D MAT 上训练并重建 3D 体积。
- `training/B0_1D_training/run_1d_training.sh`、`training/B0_1D_training/run_1d_leave_one_out.sh`：单次或 38 折留一封装。
- `training/B0_1D_training/visualize_1d_3d_predictions.py`：从保存的概率图做切片可视化与快速准确率检查。
- `training/B0_1D_training/README_1D_TRAINING.md`：用法与架构回顾。
- `erosion/train_38fold.py`、`erosion/run_training.sh`、`erosion/analyze_results.py`：扩展交叉验证，含 AMP/DataParallel、ONNX 导出、折内报告。
- `erosion/adjacency_matrix/compute_adjacency_matrices.py`：可选计算 102 区域的邻接矩阵，用于错误/混淆分析。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；38 名受试，体积 (384, 336, 256)。
- 输入：351 模态体素特征（`multidim_data`，转置为 (n_voxels, 351)），与区域掩膜对齐。
- 标签：102 类硬标签，来自 FreeSurfer 分割（`seg_one_hot` → argmax）；背景排除。
- 划分：按受试留一（37 训 / 1 测），可选受试内采样；erosion 脚本支持内部验证。

## 4. 预处理流程
- 每受试在 351 通道上做 z-score 归一化（StandardScaler）。
- 可选每受试下采样以提速；否则覆盖全部 ROI。
- 区域掩膜保证体素有效；预测通过布尔索引映射回 3D (384×336×256×102)。
- 做形状验证与转置处理以保持通道/体素顺序一致。

## 5. 模型结构
- 全连接网络：351 → 4096 → 4096 → 4096 → 4096 → 102，隐藏层后 ReLU + Dropout(0.5)。
- 无 batch/weight norm；仅对权重矩阵施加 1e-5 的手动 L2。
- erosion 流水线支持 ONNX 导出以便部署。

## 6. 训练配置
- 优化器 Adam (lr=1e-5) 加手动 L2；batch 128；训练 25 轮。
- 损失：硬标签的交叉熵；默认无 label smoothing 或类别权重。
- 通过 CLI 选择留一受试；erosion 变体强制使用 GPU，支持 AMP 与多卡 DataParallel。
- 每轮记录 train/test loss 与 macro-F1；以测试 F1 缓存最佳模型。

## 7. 评估指标与输出
- 指标：macro-F1、整体准确率；erosion 流水线额外给出按类 precision/recall/F1，macro/micro/weighted 均值与混淆报告。
- 输出：`.pth` 检查点（每个测试受试），可选 `.onnx`，训练历史 JSON/日志，3D 概率体积（`predictions_3d_test*.mat`），切片/不确定性可视化。

## 8. 论文写作解读
基线体素分类器，贴合已发表的 Alex 设置，展示非空间 MLP 在 351 通道输入上的上限。流水线还验证了概率图可靠回写 3D，为后续平滑与空间消融奠基。38 折结果为软标签、卷积、校准等实验提供锚点。

## 9. 限制与开放问题
- 超参固定，无自动调参；默认无校准或类别失衡处理。
- 依赖固定的 1D MAT 格式；若预处理变化，存在方向不一致风险。
- 采样策略与缺少增强可能限制泛化；仅支持硬标签。
