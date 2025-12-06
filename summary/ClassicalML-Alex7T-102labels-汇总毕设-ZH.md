# ClassicalML 于 Alex7T-102labels

## 1. 在论文中的角色
对 Alex 超多模态 7T 数据集（102 标签，341 特征）进行传统特征组分析与层次聚类/分类。作为基线，评估类别可分性、测试大类映射，并在重型 KAN 之前找出潜在的优质特征子集；给出粗粒度组织分组的特征组合与分类器建议。

## 2. 代码文件与入口
- main.py：验证集端到端流程；加载数据，按特征组预处理，执行可分性分析、聚类、分类，并报告与预定义大类的一致性。
- config.py：全局超参（特征索引、PCA 开关、采样标志、训练 LR/EPOCH 占位）、数据路径、保存目录构造。
- data_loader.py：加载 train/test/val 的 *.npy 体素文件，构建标签数组，可选 PCA+min-max 缩放，定义 7 类大类映射。
- feature_analysis.py：SelectKBest（F 统计）特征选择，PCA/UMAP/TSNE/MDS 可视化，聚类搜索（kmeans/spectral/agglomerative）并用 silhouette/Calinski-Harabasz/Davies-Bouldin 评分，cluster-vs-label 热图，与大类的一致性评分。
- classification.py：robust/standard 缩放，分组特征选择，特征组合搜索，交叉验证比较分类器（KNN/SVM/RF/MLP），输出比较图。
- utils.py：日志设置，PCA 解释方差估计；run_analysis.sh 启动 nohup 任务。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；体素特征重组到 train/test/val 目录。
- 输入：每体素 341 模态（0–14 diffusion，15–224 QTI/b-tensor，225–340 CEST）。体素数量未给；main.py 使用验证划分。
- 标签：102 个类似 FreeSurfer 的解剖类别，硬标签；映射到 7 个大类（脑室、白质、皮层灰质、深部核团、边缘系统、脑干、其他）。
- 划分：train/test/val 目录；主流程仅分析 val；data_loader 记录各划分的类别计数。

## 4. 预处理流程
- 可选 PCA（config 中 APPLY_PCA=True，但 main.py 用 apply_pca=False 读取原始特征）以及 PCA 后的 min–max 归一化。
- 增强预处理：按组 RobustScaler，SelectKBest F 统计特征选择（10 diffusion，30 QTI，20 CEST），并保存重要性图。
- 无显式缺失值处理；假设体素行有效。存在采样标志，但 main.py 关闭采样。

## 5. 模型结构
- 传统基线：KNN(k=5)、RBF SVM(C=1, probability=True)、RandomForest(50–100 树)、小型 MLP(50 隐层) 用于可分性与分类器比较。
- 聚类：kmeans、spectral、agglomerative，针对每个特征组测试 2–7 个簇；以 silhouette 选优。
- 此处无深度 KAN；探索仅限特征 + 传统模型。

## 6. 训练配置
- 使用 StratifiedShuffleSplit（3 折，70/30）或 5 折交叉验证（视函数而定）；指标为 accuracy。
- 无 epoch/优化器；基于 sklearn 的经典训练。config 中 lr/weight decay 未用于主流程。
- 最佳特征组合由交叉验证准确率选取；分类器排名报告均值±方差。

## 7. 评估指标与输出
- 分类：交叉验证准确率（越高越好）及标准差；给出随机猜测基线。本版本未保存混淆矩阵。
- 聚类：silhouette（高优）、Calinski-Harabasz（高优）、Davies-Bouldin（低优）；簇大小分布。
- 一致性：Hungarian 匹配的簇与大类对齐得分（越高越好）。
- 可视化：特征重要性条形图、PCA/UMAP 图、聚类指标曲线、cluster-vs-label 热图；日志与图像保存到 Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<timestamp>/，run_analysis.sh 运行时输出到 nohup_output/*.out。

## 8. 论文写作解读
- 为 Alex 7T 数据的体素级分类与大类验证提供非神经网络基线。展示哪些模态组最可分、哪些特征组合有效，以及预定义的 7 大类是否与数据驱动簇对齐。
- 为“是否需要 KAN/TabNet 的复杂度”提供证据：若传统模型已表现良好，复杂模型的价值需要论证。

## 9. 限制与开放问题
- main.py 仅在验证集上运行；无端到端 train/val/test 评估。
- 数据路径硬编码到 /home/jovyan/...，假设已预打乱的 npy 文件。
- 未处理严重类别失衡或软标签；无校准指标。
- 未集成 KAN；主要用于探索性分析。
