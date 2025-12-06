# ClassicalML-GPU 于 Alex7T-102labels

## 1. 在论文中的角色
GPU 加速与跳步变体（V3），用于有选择地重跑耗时分析，同时保留失衡处理工具。支持对 Alex 7T 体素数据更快速地迭代聚类与分类器对比。

## 2. 代码文件与入口
- V3_不均衡分类_GPU加速/main.py：V2 流水线的变体，含缓存占位（check_step_completed）、重新排序的分组处理、跳过已完成步骤的钩子；导入 pickle 以保存中间结果。
- V3_不均衡分类_GPU加速/config/config.py 以及 data/preprocessing.py、analysis/*、utils/*：在预处理、聚类、分类、评估上镜像 V2 结构。
- run_brain_voxel.sh、run_experiments.sh、experiments/feature_selection_exp.py、clustering_exp.py：批量执行助手。
- output/ 目录用于图像、日志、结果、模型。

## 3. 数据集与标签
- 与 Alex 超多模态 7T 体素数据相同，341 通道输入、102 个硬标签；保留到 7 大类的映射。
- 加载 train/test/val 的 npy 分片；提供采样标志但默认全量数据。

## 4. 预处理流程
- Robust/standard 归一化选项，分组 PCA（10/30/20/60），SelectKBest 特征选择；可选采样以提速。
- 处理顺序优先 all_features/qti/cest，再到 diffusion；若已有结果，可跳过 diffusion 聚类。

## 5. 模型结构
- 传统 sklearn 分类器（KNN、SVM、RF、MLP）用于大类预测；聚类用 kmeans/spectral/agglomerative。
- 本分支无神经网络/KAN 实现。

## 6. 训练配置
- 与 V2 类似的交叉验证，指标含 accuracy/balanced accuracy/F1/kappa；聚类测试 2–10 个簇。
- 重点在跳过重复的重负载步骤，而非调整超参；无基于 epoch 的训练。

## 7. 评估指标与输出
- 指标同 V2（accuracy 系指标；silhouette/Calinski-Harabasz/Davies-Bouldin）。
- 可视化与日志保存在 V3_不均衡分类_GPU加速/output/；缓存占位存在但未完全实现。

## 8. 论文写作解读
- 展示为使传统层次分析可扩展所做的工程努力；可用于讨论资源受限场景下的运行时考量与可复现性。

## 9. 限制与开放问题
- 许多流程段仍被注释或依赖外部缓存；GPU 加速仅暗示，并未显式（无 cupy 等）。
- 跳步逻辑与缓存加载不完整，端到端运行可能需要手动调整。
