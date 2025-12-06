# FixedMLP-BackgroundWeighting 于 Alex7T-102labels

## 1. 在论文中的角色
- 用于诊断 Alex 超多模态数据集体素分类中的背景处理与类别失衡策略。
- 对比四种情景：有/无背景训练 × 有/无逆频次类别权重，以解决 NaN 损失和性能下降。

## 2. 代码文件与入口
- comparison_alex/4场景对比/comparison_experiment.py：端到端实验驱动，定义四种情景、数据加载、训练、评估与绘图。
- comparison_alex/4场景对比/run_4_senario.sh：nohup 封装，启动实验并记录输出。
- comparison_alex/4场景对比/4场景对比alex代码数据.ipynb：同一比较的探索式 notebook 版本（此处未运行）。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T（TRAIN38.mat）。
- 输入：每体素 341 特征（转置为样本 × 特征）。
- 标签：含背景的 102 类 one-hot；情景要么保留背景作为类别，要么在训练时映射为 ignore_index (-1) 并仅训练 101 前景类。
- 划分：按受试——prob_idx==38 作为验证，其余患者中 1% 为测试，剩余为训练。

## 4. 预处理流程
- 在训练体素上拟合 StandardScaler；应用于 val/test；打印 scaler 统计做 QC。
- 数据集类按情景可选背景过滤；过滤时可移除验证集背景避免 all-ignore 批次。
- 无 PCA 或其他特征选择。

## 5. 模型结构
- FixedMLP：4×4096 ReLU + Dropout(0.5) 的全连接堆叠，Xavier 初始化；可选 L2 正则（1e-5）加到损失。
- 输出维度在忽略背景时为 101，保留背景时为 102。

## 6. 训练配置
- 优化器：Adam，lr=1e-5；batch_size=128；epochs=25（NUM_EPOCHS_DEMO）。
- 损失：CrossEntropyLoss；过滤背景时 ignore_index=-1。类别权重可按逆频次在不同情景下启用。
- 每轮记录指标：验证 accuracy 与 macro F1；训练 loss/acc/F1 亦记录。

## 7. 评估指标与输出
- 指标：验证 accuracy 与 macro F1（越高越好）；包含 NaN 检查防止无效损失。
- 输出：comparison_results_fixed/ 目录，含 final_results_summary_fixed.txt（各情景验证指标）、validation_curves_comparison_fixed.png、training_loss_comparison_fixed.png，以及内存中的各情景历史。

## 8. 论文写作解读
- 检验将背景作为可学习类别或忽略，以及应用类别权重，是否能稳定训练并提升验证表现。旨在为主流水线的背景处理选择提供依据。

## 9. 限制与开放问题
- 路径硬编码到 /home/jovyan/... 的 TRAIN38.mat，复用前需调整。
- 仅报告验证指标；无独立测试或校准分析。
- 使用单一固定架构与学习率；情景仅隔离背景/加权效应，未考察架构依赖行为。
