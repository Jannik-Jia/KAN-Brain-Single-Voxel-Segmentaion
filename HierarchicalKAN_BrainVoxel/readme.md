# BrainVoxel分层分类分析脚本

这个脚本执行BrainVoxel数据的分层分类分析，适用于nohup后台提交运行，会自动保存所有输出和图表。

## 功能概述

该脚本实现以下功能：

1. 从磁盘加载脑体素数据
2. 对不同特征组（扩散、QTI、CEST）进行预处理和特征选择
3. 定义和分析大类分类标签
4. 分析特征空间的类别可分性
5. 探索最佳分类策略（包括聚类和分类器评估）
6. 比较聚类结果与预定义大类的一致性
7. 生成详细的分析报告和可视化结果

## 文件结构

代码已模块化为以下几个文件：

- **config.py**: 包含所有配置参数和常量
- **utils.py**: 工具函数，如日志记录和PCA分析
- **data_loader.py**: 负责数据加载和处理
- **feature_analysis.py**: 特征分析、选择和聚类功能
- **classification.py**: 分类模型和策略分析
- **main.py**: 主程序，协调执行整个流程
- **run_analysis.sh**: 后台运行分析的bash脚本

## 使用方法

### 预备工作

1. 确保安装了所有必要的依赖包：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn tqdm h5py
```

2. 确保数据目录结构正确，默认数据路径在config.py中指定：
```
/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/
├── train/
├── test/
├── val/
```

### 运行脚本

1. 给脚本添加执行权限：
```bash
chmod +x main.py
chmod +x run_analysis.sh
```

2. 后台运行分析（使用nohup）：
```bash
./run_analysis.sh
```

3. 监控运行状态：
```bash
# 查看实时输出
tail -f ./nohup_output/brainvoxel_analysis_*.out

# 检查是否仍在运行
ps -ef | grep main.py
```

## 输出结果

所有结果将保存在以下目录（自动创建）：
```
./Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<时间戳>/
```

包含的输出内容：
- 所有特征组的重要性分析图表
- 降维可视化（PCA和UMAP）
- 聚类评估结果
- 特征组合性能比较
- 分类器性能比较
- 聚类与大类一致性分析
- 详细的日志文件

## 定制分析

可以修改config.py中的以下参数来定制分析：

- `SAMPLE_RATIO`: 数据采样比例（加速分析）
- `FEATURE_DIM`: 特征总维度
- `NUM_CLASS`: 类别总数
- `DIFF_FEATURES`, `QTI_FEATURES`, `CEST_FEATURES`: 特征索引范围
- `DATA_DIRS`: 数据目录位置

## 常见问题

1. **内存不足**: 可以在config.py中调整`USE_SAMPLING`为True并设置更小的`SAMPLE_RATIO`（如0.05）减少内存消耗

2. **运行时间过长**: 可以修改classification.py和feature_analysis.py中的聚类方法和降维方法的数量

3. **数据路径错误**: 确保在config.py中正确设置`DATA_DIRS`指向实际数据位置
