# 脑体素分层分类项目

该项目专注于分析脑部磁共振成像的多模态特征数据，并开发分层分类系统，通过先将102个脑结构细分类别归类为更少的大类，然后在大类内进行细分类别识别，提高整体分类效果。

## 项目结构

```
brain_voxel_hierarchical/
├── config/               # 配置文件
│   └── config.py         # 项目配置参数
├── data/                 # 数据处理
│   ├── data_loader.py    # 数据加载
│   └── preprocessing.py  # 特征预处理
├── analysis/             # 分析模块
│   ├── feature_analysis.py  # 特征重要性分析
│   ├── dimensionality.py    # 降维分析
│   ├── clustering.py        # 聚类分析
│   └── classification.py    # 分类评估
├── visualization/        # 可视化
│   ├── feature_viz.py    # 特征可视化
│   ├── cluster_viz.py    # 聚类结果可视化
│   └── comparison_viz.py # 比较分析可视化
├── utils/                # 工具函数
│   ├── logging_utils.py  # 日志工具
│   ├── evaluation.py     # 评估指标
│   └── model_utils.py    # 模型辅助函数
├── experiments/          # 实验脚本
│   ├── feature_selection_exp.py  # 特征选择实验
│   ├── clustering_exp.py         # 聚类实验
│   ├── all_features_exp.py       # 全特征分析实验
│   └── comparison_exp.py         # 不同方法比较实验
├── output/               # 输出目录
│   ├── figures/          # 保存图表
│   ├── logs/             # 保存日志
│   ├── models/           # 保存模型
│   └── results/          # 保存分析结果
├── main.py               # 主执行脚本
├── run_experiments.py    # 批量实验脚本
└── README.md             # 项目说明
```

## 数据说明

该项目使用脑体素数据，每个样本包含341个特征，分为三组：
- 扩散特征 (0-14)：表征水分子扩散特性
- QTI特征 (15-224)：量化张量成像特征
- CEST特征 (225-340)：化学交换饱和转移特征

样本被划分为102个细分类别，代表不同的脑部解剖结构。

## 项目目标

1. 分析特征重要性和特征组合效果
2. 探索最佳聚类方法，确定合理的大类数量
3. 评估现有大类划分与数据驱动聚类的一致性
4. 可能重新定义大类标签映射关系，提高分类效果
5. 为后续的分层分类模型（如KAN）提供最佳特征和分类策略

## 使用方法

### 主程序执行

```bash
python main.py --data_subset=val --normalize=robust --pca --min_clusters=2 --max_clusters=10
```

参数说明：
- `--data_subset`: 指定使用的数据子集 (train, test, val, all)
- `--normalize`: 特征标准化方法 (standard, robust, none)
- `--pca`: 使用PCA降维（`--no_pca`表示不使用）
- `--feature_selection`: 使用特征选择（`--no_feature_selection`表示不使用）
- `--min_clusters`/`--max_clusters`: 聚类数量范围
- `--output_prefix`: 输出文件前缀
- `--skip_plots`: 跳过绘图（用于批处理）

### 批量实验执行

```bash
python run_experiments.py
```

该脚本会自动执行一系列不同参数组合的实验，并比较结果。

### 特定实验执行

```bash
# 特征选择实验
python experiments/feature_selection_exp.py

# 聚类实验
python experiments/clustering_exp.py

# 全特征分析实验
python experiments/all_features_exp.py
```

## 输出结果

项目会生成以下输出：
1. 分析日志：记录详细的分析过程和结果
2. 可视化图表：特征分布、聚类效果、分类性能比较等
3. 最终报告：包含大类划分建议和最佳特征组合
4. 存储模型：可用于后续预测的分类器模型

## 依赖库

- Python 3.8+
- NumPy, SciPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- UMAP-learn
- PyTorch (可选，用于后续的神经网络模型)


## 后续工作

本项目主要完成大类划分和特征分析。后续工作将基于这些结果，继续开发：

1. 层次化的Komogorov-Arnold Network (KAN) 架构
2. 各特征组专家模型
3. 增量学习策略，支持处理新的类别



要使用这些脚本：

首先使它们可执行：

chmod +x run_brain_voxel.sh
chmod +x run_experiments.sh

运行单一分析：

./run_brain_voxel.sh

或者运行综合实验：

./run_experiments.sh