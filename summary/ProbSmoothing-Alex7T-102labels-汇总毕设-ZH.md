# ProbSmoothing 于 Alex7T-102labels

## 1. 在论文中的角色
对 FC 基线的体素概率图进行平滑后处理，测试无需重训分类器即可加入轻量空间正则的效果。

## 2. 代码文件与入口
- `training/B1_Probability_map_post_processing/smooth_postprocess_eval.py`：核心平滑/评估脚本（标准版与 gated 版）。
- `training/B1_Probability_map_post_processing/run_smooth_evaluation.sh`、`training/B1_Probability_map_post_processing/test_gated_smooth.sh`：可运行预设。
- `training/B1_Probability_map_post_processing/README_SMOOTH_POSTPROCESSING.md`：动机、指标与使用说明。

## 3. 数据集与标签
- 输入：来自 FC 基线的 3D 概率体积（`softmax_vol`，约 384×336×256×102），以及 Alex 数据集的对应标签/掩膜。
- 标签：102 个硬类别；评估限制在脑掩膜内。

## 4. 预处理流程
- 对每个切片做 2D 均值平滑（3×3 或 7×7 核），带掩膜归一化；支持矢状/冠状/轴位。
- 可选 gating：按类别门控（仅在预测类别内平滑）或不确定性门控（基于熵/间隔的 Sigmoid 混合）以保护边缘/高置信区域。

## 5. 模型结构
- 无训练模型；对现有概率图应用确定性的平滑算子。

## 6. 训练配置
- 通过 CLI 标志配置：核大小、轴向、快速卷积路径、gating 选项、不确定性参数；无优化循环。

## 7. 评估指标与输出
- 指标：整体准确率、macro-F1、Cohen κ、macro/micro AUPRC、相对原始概率的提升、混淆矩阵。
- 输出：平滑后的概率体积（HDF5）、CSV 指标、图表（混淆对比）、gating 行为日志。

## 8. 论文写作解读
展示轻量空间平滑如何改进噪声较大的体素预测，作为无需重训的低成本替代方案；可用于讨论失衡组织上的后处理影响。

## 9. 限制与开放问题
- 效果依赖于基线概率质量；最优核/gating 可能因受试/类别而异。
- 纯 2D 平滑可能缺乏层间一致性；除平滑外不做概率再校准。
