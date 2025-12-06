# FC4x4096 于 Alex7T-102labels-patientwiseStd

## 1. 在论文中的角色
PyTorch 版的 4×4096 全连接基线，用于 German et al. 2021 的 Alex 超多模态 7T 数据集体素分类。此 notebook 测试按患者标准化（每受试 z-score），以更好匹配各受试的谱分布，相比全局归一化更贴合个体。作为与原 TensorFlow 基线的校核与可复现性检查，并评估患者特定缩放对单受试验证的影响。

## 2. 代码文件与入口
- `update_standarlize/alex的torch版本_新标准化措施.ipynb`：端到端 notebook，四个单元：(1) 从 `.mat` 读取数据与按患者标准化工具；(2) 4×4096 稠密网络定义与训练循环；(3) 在 `DEMO38.mat` 上推理并应用患者级统计；(4) 对选定验证体素做梯度显著性可视化。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T，从 `TRAIN38.mat`（`data`、`region`、`prob_idx`）和 `DEMO38.mat`（`multidim_data`，可选 `prob_idx`）加载。
- 输入维度：341 通道体素签名（逐体素谱/模态特征），跨受试聚合数百万体素（具体数量未列）。
- 标签空间：`region` 中的 102 类 one-hot 标签，可能来自对齐 CEST/原始网格的 FreeSurfer 式分割；为硬 one-hot。
- 划分：训练使用除患者 38 外所有受试（`prob_idx != 38`）；验证仅患者 38（`prob_idx == 38`）。额外 `train_test_split` 留出 1% 非 38 数据，但训练只用 `X_train1`/`y_train1`（小留出未用）。测试/推理单元需要 `DEMO38.mat`，优先使用训练统计。

## 4. 预处理流程
- 按患者标准化：为每个患者计算均值/标准差，并在该患者内部做 z-score；零标准差替换为 1.0。返回并存储 `train_patient_stats` 与 `val_patient_stats`。
- 验证使用验证患者自身统计；推理时若患者 ID 已知则用训练统计，否则 fallback 为测试数据上现算的患者统计。
- 无额外缩放、截断或插补，假设特征稠密。

## 5. 模型结构
- 与 TensorFlow 基线匹配的全连接网络：341 → 4096 → 4096 → 4096 → 4096 → 102。
- 激活：每层 ReLU；每层后 Dropout(p=0.5)。
- 输出：线性 logits（softmax 在推理外部）；无 batchnorm 或残差。

## 6. 训练配置
- 优化器：Adam (lr=1e-5)；仅对权重矩阵手动 L2（weight_decay 1e-5 加入损失）。
- 损失：对 one-hot 取 argmax 的 CrossEntropyLoss。
- Batch size：128；Epochs：25；为 torch/NumPy 设定确定性种子；若可用则用 GPU。
- 每轮记录指标：训练 loss/accuracy/macro F1，验证集（患者 38）loss/accuracy/F1。
- 模型检查点：保存至 `outputs/dense_4x4096_model.pth`。

## 7. 评估指标与输出
- 指标：训练与验证的 macro F1、总体准确率；逐轮打印，末尾报告最佳验证 accuracy/F1。两者越高越好。
- 输出：`outputs/train_patient_stats.npy`、`outputs/val_patient_stats.npy`（按患者均值/方差）；`outputs/dense_4x4096_model.pth`（检查点）；通过 `scipy.io.savemat` 将预测导出为 `dense_4x4096_model_prediction.mat`（需用户自设 `predictpath`）；验证体素的梯度显著性图（显示，默认不保存）。

## 8. 论文写作解读
此实验在引入患者级归一化的同时验证核心 FC 基线的可复现性，缓解体素签名的跨受试分布偏移。展示了 4×4096 稠密网络可在 PyTorch 中训练，与 TensorFlow 版本类似的正则；验证使用单一受试（患者 38）。显著性可视化为光谱通道贡献提供初步可解释性视角。

## 9. 限制与开放问题
- 验证仅依赖单一受试（患者 38）；对其他受试的泛化未知。
- `train_test_split` 留出的少量数据未用，除患者 38 外无专用测试集。
- `predictpath` 为占位，保存预测需用户提供路径。
- 341 通道的类别定义与模态拆分未写入 notebook；无下游校准或更广指标（ECE、混淆矩阵）。
- 验证标准化使用自身统计而非训练统计，可能与部署时行为不符。
