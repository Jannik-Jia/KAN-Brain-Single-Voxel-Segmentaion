# DataProfiling 于 Alex7T-102labels

## 1. 在论文中的角色
MRI 数据剖析工具包（V5），在训练前量化特征分布、相关性、可分性与特征重要性。用于评估模态价值、指导后续架构选择。

## 2. 代码文件与入口
- V5_data_analysis/main.py：CLI 入口，执行预处理、基础统计、特征分析、降维；支持 GPU 标志。
- V5_data_analysis/data_loader.py：加载脑体素测试数据（npy），可用 one-hot 或整型标签，报告标准化状态，定义特征分组；若可用尝试 cupy/cuml。
- basic_analysis.py、feature_analysis.py、dim_reduction.py：计算摘要统计、相关性、类别可分性、特征重要性（RF、互信息），评估特征选择方法，运行 PCA/TSNE/UMAP/Isomap 降维。
- run_analysis.sh：执行流水线的助手脚本；输出目录建在 analysis_results 下。

## 3. 数据集与标签
- Alex 超多模态 7T 数据集，主要使用 /home/jovyan/.../restructured/test 的测试划分。
- 341 个特征，分组为 diffusion/QTI/CEST/all_features；102 个标签（提供整型与 one-hot 形式）。
- 聚焦测试子集做剖析；类别计数由标签索引文件报告。

## 4. 预处理流程
- preprocess_data 中可选 robust/standard 归一化；检测数据是否已标准化。
- 试用 cupy/cuml 做 GPU 加速；若不可用则回退到 numpy/sklearn。
- 加载时不做 PCA；降维在分析阶段处理。

## 5. 模型结构
- 非训练流水线；使用随机森林特征重要性与互信息评分特征。除可分性评分外无神经/传统分类器训练。

## 6. 训练配置
- 不适用；分析直接作用于给定数据，无基于 epoch 的训练。

## 7. 评估指标与输出
- 基础统计（各特征均值/方差范围）、组内/组间特征相关性、类别可分性比（F 分数）、特征重要性得分（RF/MI）。
- 降维图（PCA/TSNE/UMAP/Isomap）用于可视化类别分布。
- 输出保存到 analysis_results 子目录（basic_analysis、feature_analysis、dim_reduction），包含文本摘要与图像。

## 8. 论文写作解读
- 提供关于哪类模态携带判别信息以及通道相关度的描述性证据，支撑后续分类器在归一化、特征选择上的方法选择。

## 9. 限制与开放问题
- 主要在测试子集上运行；若无 train/val 检查，结论未必可泛化。
- 数据路径硬编码；假设标签索引文件与 npy 分片存在。
- 与下游模型性能没有直接关联；仅作探索性分析。
