# OriginalMLP-CouplingLR 于 Alex7T-102labels

## 1. 在论文中的角色
- 复现原始 notebook 风格的 4×4096 MLP 训练，研究架构宽/深与学习率的耦合对 Alex 超多模态数据集的影响。
- 提供快速扫描以在全量训练前挑选稳定超参。

## 2. 代码文件与入口
- comparison_alex/coupling_test.py：定义 OriginalStyleMLP 并在架构 × 学习率上网格搜索，记录结果并生成分析报告。
- comparison_alex/run_coupling_test.sh：便捷启动（nohup）脚本引用上述 Python。
- comparison_alex/对比.ipynb：相关探索 notebook（此处未运行）。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T（TRAIN38.mat）。
- 输入：341 通道体素向量。
- 标签：含背景的 102 类 one-hot；本复现未去背景。
- 划分：prob_idx!=38 用于训练/测试（1% 测试比例），prob_idx==38 用作验证。

## 4. 预处理流程
- 在训练子集上拟合 StandardScaler；应用于验证与 1% 测试切分。
- 无 PCA 或特征选择；数据转置为 (samples × 341)，标签转为 (samples × 102)。

## 5. 模型结构
- OriginalStyleMLP 基线：4×4096 ReLU + Dropout(0.5)，权重 L2 正则（1e-5）。
- 测试变体：wide_shallow（2×8192）与 narrow_deep（6×2048），激活/Dropout/L2 相同。

## 6. 训练配置
- 优化器：Adam；学习率扫描 {5e-6, 1e-5, 2e-5, 5e-5}。
- Batch 128；快速扫描训练 12 轮（原 notebook 为 25）；损失：CrossEntropyLoss + L2。
- 每次运行指标：验证 accuracy、macro F1、Cohen κ；记录训练轨迹。

## 7. 评估指标与输出
- 指标：验证 accuracy/F1/κ（越高越好）；跟踪每个实验的收敛。
- 输出：coupling_test_results/<timestamp>/ 存储 JSON/分析报告（results.json, analysis_report.txt），总结每种架构的最佳 lr 并给出推荐。

## 8. 论文写作解读
- 快速网格展示 lr 与架构深度/宽度的交互，帮助确定主 MLP 基线的 lr/架构，并指出稳定区间。

## 9. 限制与开放问题
- 训练轮数较少，数值仅供参考非最终基准。
- 包含背景类别；未测试 ignore_index 或软标签。
- 无外部测试或校准指标；关注验证耦合。
