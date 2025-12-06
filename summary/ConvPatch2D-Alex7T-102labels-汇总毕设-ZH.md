# ConvPatch2D 于 Alex7T-102labels

## 1. 在论文中的角色
基于补丁的 CNN 基线，在保持轻量的同时引入最小空间上下文（xy 平面 3×3 或 7×7）。作为 1D MLP 与更深 ResNet 实验之间的过渡。

## 2. 代码文件与入口
- `training/3D CNN/train_baseline_3x3_7x7.py`：主训练脚本，包含模型定义（ImprovedConv2D_Baseline 及参数对齐变体）。
- `training/3D CNN/README.md`：3×3/7×7 补丁的使用指南与超参说明。
- `training/3D CNN/run_leave_one_out.sh`：批量留一法运行器；`test_data_loading.py` 用于数据检查。
- `training/3D CNN/analyze_results.py`：汇总留一结果并绘制统计。

## 3. 数据集与标签
- 数据集：Alex 超多模态 7T 的 3D MAT 文件（384×336×256×351），含 `region_labels` 和 `region_mask`。
- 输入：在固定 z 上切取 xy 平面的 2D 补丁（3×3 或 7×7）；每体素 351 通道，从掩膜内体素采样（默认每个训练受试 1 万，测试 2 倍）。
- 标签：102 个硬类别（在加载器内偏移到 0–101）；背景排除。
- 划分：按受试留一（37 训 / 1 测），通过 CLI 选择。

## 4. 预处理流程
- 每个受试的每通道 z-score 归一化（整 3D 体积）。
- 补丁提取在边界处填充以保持固定大小；启用时缓存受试数据到内存。
- 仅在掩膜且标签>0的体素中采样；训练可选 shuffle。

## 5. 模型结构
- Stem：1×1 卷积混合通道（351→mid，默认 mid=128）+ GroupNorm + SiLU。
- 聚合：单个卷积核大小等于补丁大小（3 或 7），用于折叠空间维度 + GroupNorm + SiLU。
- 可选 1×1 残差 refine 块；可选 SE 通道注意力。
- Head：线性分类器或 MLP 头（参数对齐变体，目标约 35M/52M/69M 参数，隐藏层大且带 dropout）。

## 6. 训练配置
- 损失：交叉熵；优化器 AdamW（lr=1e-4, weight_decay=1e-4）；CosineAnnealingLR（T_max=50, eta_min=1e-6）。
- Batch size 256；训练 50 轮；使用 GradScaler 的混合精度；除补丁采样外无显式增强。
- 每轮以测试集 macro-F1 选最佳检查点；历史保存为 JSON。

## 7. 评估指标与输出
- 指标：每轮训练/测试的 macro-F1 与平均 loss；打印参数量。
- 输出：`best_model_patch{3|7}_test{N}.pth`，`history_patch{...}.json`，留一运行的日志。

## 8. 论文写作解读
评估小尺度空间上下文（3×3 vs 7×7）与参数规模对体素级准确率的影响，相对 1D MLP 的改进，为迈向更深 ResNet 架构的消融步骤。

## 9. 限制与开放问题
- 仅在 2D 切片上操作（无完整 3D 卷积）；增强有限。
- 每受试采样固定体素数，使用全数据训练的效果未知。
- 未显式处理类别失衡或校准。
