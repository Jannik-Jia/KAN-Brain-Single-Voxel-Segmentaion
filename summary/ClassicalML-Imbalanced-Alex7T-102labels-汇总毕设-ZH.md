# ClassicalML-Imbalanced 于 Alex7T-102labels

## 1. 在论文中的角色
失衡感知的传统分析流水线（V2），用于在偏斜的体素计数下评估特征预处理、聚类与分类器。提供可复现的 CLI 与批量脚本，说明重采样和特征选择在进入神经模型前如何影响基线表现。

## 2. 代码文件与入口
- V2_不均衡分类/main.py：CLI 覆盖数据加载、大类映射、预处理（归一化、PCA、特征选择）、聚类、分类器比较、混淆矩阵、映射评估；支持子集选择与跳过绘图。
- V2_不均衡分类/config/config.py：特征组、PCA 维度、特征选择方法、聚类范围、评估指标、路径、随机种子、设备等配置。
- data/data_loader.py 与 data/preprocessing.py：加载 train/test/val 的 npy，记录类别计数，按组应用缩放/PCA/特征选择，可选采样标志。
- analysis/*（feature_analysis.py, dimensionality.py, clustering.py, classification.py）：特征重要性、可分性、可视化（PCA/UMAP）、聚类搜索、分类器评估。
- utils/logging_utils.py, utils/evaluation.py, utils/model_utils.py：日志、指标计算（accuracy、balanced accuracy、f1_weighted、kappa；聚类指标）、结果保存、映射生成/可视化。
- run_brain_voxel.sh, run_experiments.sh, experiments/*.py：shell 与批量实验脚本（特征选择、聚类、组合实验）。
- output/figures、output/logs、output/results 作为生成物占位。

## 3. 数据集与标签
- Alex 超多模态 7T 数据；体素特征 341 维（拆分为 diffusion/QTI/CEST），102 个硬标签。
- 与主分支一致的 7 类大类映射，整型标签。
- 划分通过 train/test/val 目录；可对单一子集或合并集分析；记录类别分布。
- 采样控制：USE_SAMPLING 标志（默认 False）和 SAMPLE_RATIO=0.1 以快速测试；导入 imblearn SMOTE/RandomUnderSampler 以缓解失衡。

## 4. 预处理流程
- Robust/standard/无 归一化选项；按组 PCA（10/30/20/60 组件）可配置；SelectKBest 特征选择（f_classif、mutual_info、chi2）。
- 保留特征分组以比较模态贡献；可选降采样以提速。
- 除 numpy 读取外无显式缺失值处理；假设体素行有效。

## 5. 模型结构
- sklearn 经典分类器：KNN、RBF SVM、RandomForest、MLP；在大类标签上评估。
- 聚类：kmeans、spectral、agglomerative，测试 2–10 簇，记录 silhouette/Calinski-Harabasz/Davies-Bouldin。
- 提供映射评估以对齐聚类与预定义大类。

## 6. 训练配置
- 5 折或分层划分的交叉验证，指标含 accuracy、balanced accuracy、weighted F1、Cohen κ。
- 无基于 epoch 的训练；超参多为默认（树数或核选择除外）；可在批量运行时跳过昂贵图表。

## 7. 评估指标与输出
- 分类指标如上（准确率/F1/κ 越高越好）。
- 聚类指标：silhouette 与 Calinski-Harabasz（越高越好），Davies-Bouldin（越低越好）；稳定性检查位于 utils/evaluation。
- 可视化：特征重要性、降维散点、cluster-vs-label 热图、混淆矩阵；输出存于 V2_不均衡分类/output/ 子目录，带 config 中 RUN_ID 时间戳。
- 日志通过 logging_utils 保存到 output/logs。

## 8. 论文写作解读
- 量化传统模型在类别失衡与不同预处理设置下的表现，为大类粒度与特征有效性提供基线。
- 可用于“不平衡处理与特征选择消融”章节，作为 KAN 等神经模型前的参考。

## 9. 限制与开放问题
- 数据路径硬编码到 /home/jovyan/...；假定预生成的 npy 分片。
- 失衡缓解（SMOTE/欠采样）已接入但默认 CLI 流程未充分使用；效果未知。
- 仍聚焦于大类而非完整的 102 分类；无校准指标。
