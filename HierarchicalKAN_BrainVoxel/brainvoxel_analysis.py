#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BrainVoxel分层分类分析脚本
可使用nohup方式在后台运行，所有输出和图表会保存到指定目录
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，避免需要显示器
import matplotlib.pyplot as plt
import matplotlib.patches as mpts
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from scipy.io import loadmat
from tqdm import tqdm
import glob
import sys
import logging
import h5py
import copy

# 设置日志
def setup_logger(log_dir):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"brainvoxel_analysis_{timestamp}.log")
    
    # 创建一个logger
    logger = logging.getLogger('brainvoxel_analysis')
    logger.setLevel(logging.INFO)
    
    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建一个控制台处理器，用于在控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 超参数和实验设置
# 设置随机种子，确保实验可重复性
RANDOM_SEED = 666

# 数据采样参数
SAMPLE_RATIO = 0.1  # 使用10%的数据进行分析
USE_SAMPLING = True  # 是否使用数据采样

# 数据预处理参数
APPLY_PCA = True   # 是否应用PCA降维
NORM = True        # 是否对数据进行标准化/归一化处理

# 特征分组
DIFF_FEATURES = list(range(0, 15))     # 扩散特征 (1-15)
QTI_FEATURES = list(range(15, 225))    # QTI特征 (16-225)
CEST_FEATURES = list(range(225, 341))  # CEST特征 (226-341)

# 定义模型名称，用于结果保存和模型标识
MODEL_NAME = 'HierarchicalKAN_BrainVoxel'

# 大类定义 (初始设置，可能需要根据实际数据调整)
DEFAULT_NUM_BIG_CLASSES = 5  # 大类数量

# 指定数据集名称
DATASET = 'BrainVoxel'

# 训练参数
EPOCH = 50         # 总训练轮数
VAL_EPOCH = 1      # 每隔多少轮进行一次验证
LR = 0.001         # 学习率
WEIGHT_DECAY = 1e-6  # 权重衰减系数，用于L2正则化
BATCH_SIZE = 640    # 批处理大小，固定不变

# 计算设备选择
DEVICE = 0         # -1表示使用CPU，0表示使用第一块GPU(cuda:0)

# 数据参数
FEATURE_DIM = 341  # 输入特征总维度
NUM_CLASS = 102    # 细分类别数量
FIXED_GRID = 10    # 固定网格大小，不进行网格扩展

# 专家模型参数
DIFF_HIDDEN_DIM = 64    # 扩散专家隐藏层维度
QTI_HIDDEN_DIM = 128    # QTI专家隐藏层维度
CEST_HIDDEN_DIM = 64    # CEST专家隐藏层维度

# PCA参数
DIFF_PCA_COMPONENTS = 10   # 扩散特征PCA组件数
QTI_PCA_COMPONENTS = 30    # QTI特征PCA组件数
CEST_PCA_COMPONENTS = 20   # CEST特征PCA组件数

# 模型检查点路径
CHECK_POINT = None  # 加载预训练模型的路径，None表示从头开始训练

# 结果保存路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"./Results/{MODEL_NAME}/{DATASET}/{timestamp}"

# 数据目录
DATA_DIRS = {
    'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
    'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
    'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
}

def select_discriminative_features(data, labels, group_name, k=20, verbose=True, plot=True, save_path=None, logger=None):
    """
    选择最具区分性的特征
    
    参数：
        data: 输入数据特征
        labels: 类别标签
        group_name: 特征组名称
        k: 选择的顶部特征数量
        verbose: 是否打印信息
        plot: 是否绘制重要性分布图
        save_path: 图表保存路径
        logger: 日志记录器
        
    返回：
        selected_features: 选择后的特征
        feature_indices: 选择的特征索引
        feature_scores: 特征重要性分数
    """
    # 使用F统计量计算特征重要性
    k = min(k, data.shape[1])  # 确保k不超过特征数量
    selector = SelectKBest(f_classif, k=k)
    selected_features = selector.fit_transform(data, labels)
    feature_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    if verbose:
        msg = f"\n{group_name} 特征组选择结果:"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"原始特征维度: {data.shape[1]}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"选择后特征维度: {selected_features.shape[1]}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"均值F分数: {np.mean(feature_scores):.2f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"最大F分数: {np.max(feature_scores):.2f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 打印前10个最重要的特征索引和得分
        sorted_indices = np.argsort(feature_scores)[::-1]
        msg = "\n前10个最重要的特征:"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        for i, idx in enumerate(sorted_indices[:10]):
            msg = f"  特征 {idx}: F分数 = {feature_scores[idx]:.2f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
        
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Left plot: Importance distribution of all features
        plt.subplot(1, 2, 1)
        sorted_indices = np.argsort(feature_scores)[::-1]
        plt.bar(range(len(feature_scores)), feature_scores[sorted_indices])
        plt.title(f'{group_name} Feature Importance (All)')
        plt.xlabel('Feature Rank')
        plt.ylabel('F-score')
        plt.yscale('log')  # Logarithmic scale
        plt.grid(True)
        
        # Right plot: Importance of selected features
        plt.subplot(1, 2, 2)
        selected_scores = feature_scores[feature_indices]
        sorted_selected = np.argsort(selected_scores)[::-1]
        plt.bar(range(len(selected_scores)), selected_scores[sorted_selected])
        plt.title(f'{group_name} Feature Importance (Selected)')
        plt.xlabel('Feature Rank')
        plt.ylabel('F-score')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'{group_name}_feature_importance.png'))
        plt.close()

    return selected_features, feature_indices, feature_scores

def advanced_feature_reduction(data, method='umap', n_components=2, labels=None, plot=True, title=None, save_path=None, logger=None):
    """
    高级特征降维与可视化
    
    参数：
        data: 输入数据
        method: 降维方法，可选 'pca', 'tsne', 'umap', 'mds'
        n_components: 降维后的维度
        labels: 类别标签，用于可视化
        plot: 是否绘制降维结果
        title: 图表标题
        save_path: 图表保存路径
        logger: 日志记录器
        
    返回：
        reduced_data: 降维后的数据
        reducer: 降维模型
    """
    # 确保数据是浮点型
    data = data.astype(np.float32)
    
    # 根据指定方法进行降维
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(data)
        explained_var = reducer.explained_variance_ratio_
        explained_var_str = f"解释方差: {sum(explained_var):.2%}"
    
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, 
                      perplexity=min(30, data.shape[0] // 5), 
                      n_iter=1000, 
                      random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components,
                          n_neighbors=min(30, data.shape[0] // 5),
                          min_dist=0.1,
                          random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    elif method.lower() == 'mds':
        reducer = MDS(n_components=n_components, n_jobs=-1, random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    else:
        msg = f"不支持的降维方法: {method}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    # 可视化降维结果
    if plot and labels is not None and n_components in [2, 3]:
        plt.figure(figsize=(12, 10))
        
        if n_components == 2:
            # 2D可视化
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                           alpha=0.6, label=f'Class {label}')
            
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
        else:
            # 3D可视化
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2],
                         alpha=0.6, label=f'Class {label}')
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        
        # 设置标题
        if title:
            plt.title(f'{title} ({method.upper()} projection) {explained_var_str}')
        else:
            plt.title(f'{method.upper()} projection {explained_var_str}')
        
        plt.grid(True)
        # 仅显示部分标签，避免图例过大
        if len(unique_labels) > 10:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        else:
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'{title}_{method}_visualization.png' if title else f'{method}_visualization.png'))
        plt.close()
    
    return reduced_data, reducer

def analyze_optimal_clusters(data, min_clusters=2, max_clusters=10, methods=['kmeans', 'spectral', 'agglomerative'], save_path=None, logger=None):
    """
    分析最佳聚类数量和方法
    
    参数：
        data: 输入数据
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
        methods: 要尝试的聚类方法列表
        save_path: 图表保存路径
        logger: 日志记录器
        
    返回：
        best_results: 最佳聚类结果的字典
    """
    import time
    
    # 如果数据过大，进行下采样
    if data.shape[0] > 100000:
        from sklearn.model_selection import train_test_split
        _, data = train_test_split(data, test_size=100000/data.shape[0], random_state=RANDOM_SEED)
        msg = f"数据太大，下采样到 {data.shape[0]} 样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 初始化结果存储
    silhouette_scores = {method: [] for method in methods}
    calinski_scores = {method: [] for method in methods}
    davies_scores = {method: [] for method in methods}
    cluster_labels = {method: {} for method in methods}
    
    msg = "分析最佳聚类数量..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 尝试不同的聚类数量和方法
    for n_clusters in range(min_clusters, max_clusters+1):
        msg = f"\n尝试 {n_clusters} 个聚类:"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        for method in methods:
            start_time = time.time()
            msg = f"  方法: {method}..."
            if logger:
                logger.info(msg)
            else:
                print(msg, end="", flush=True)
            
            # 创建聚类模型
            if method == 'kmeans':
                cluster_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=5)
            elif method == 'spectral':
                cluster_model = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED, 
                                                  affinity='nearest_neighbors', n_neighbors=min(30, data.shape[0]//100))
            elif method == 'agglomerative':
                cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                if logger:
                    logger.error(f"不支持的聚类方法: {method}")
                raise ValueError(f"不支持的聚类方法: {method}")
            
            # 执行聚类
            labels = cluster_model.fit_predict(data)
            
            # 存储标签
            cluster_labels[method][n_clusters] = labels
            
            # 计算聚类评估指标
            try:
                if len(np.unique(labels)) > 1:  # 确保至少有两个聚类
                    # 为了加速，使用数据样本计算指标
                    if data.shape[0] > 10000:
                        sample_idx = np.random.choice(data.shape[0], 10000, replace=False)
                        sample_data = data[sample_idx]
                        sample_labels = labels[sample_idx]
                        sil_score = silhouette_score(sample_data, sample_labels)
                        cal_score = calinski_harabasz_score(sample_data, sample_labels)
                        dav_score = davies_bouldin_score(sample_data, sample_labels)
                    else:
                        sil_score = silhouette_score(data, labels)
                        cal_score = calinski_harabasz_score(data, labels)
                        dav_score = davies_bouldin_score(data, labels)
                    
                    silhouette_scores[method].append(sil_score)
                    calinski_scores[method].append(cal_score)
                    davies_scores[method].append(dav_score)
                    
                    elapsed = time.time() - start_time
                    
                    msg = f" 完成! ({elapsed:.1f}秒)"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"  准确率: {mean_score:.4f} ± {std_score:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        model_results.append({
            'classifier': clf_name,
            'accuracy': mean_score,
            'std': std_score
        })
    
    # 对结果排序
    model_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # 可视化分类器比较
    plt.figure(figsize=(10, 6))
    
    # 准备数据
    clf_names = [r['classifier'] for r in model_results]
    accuracies = [r['accuracy'] for r in model_results]
    stds = [r['std'] for r in model_results]
    
    # 绘制条形图
    plt.bar(clf_names, accuracies, yerr=stds, alpha=0.8)
    plt.axhline(y=1/len(np.unique(labels)), color='r', linestyle='--', 
            label=f'Random Guessing ({1/len(np.unique(labels)):.4f})')

    plt.xlabel('Classifiers')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Comparison of Different Classifiers')
    plt.grid(axis='y')
    plt.legend()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'classifier_comparison.png'))
    plt.close()

    # 5. 总结最佳策略
    best_classifier = model_results[0]['classifier']
    best_accuracy = model_results[0]['accuracy']
    
    msg = "\n最佳分类策略:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"1. 特征组合: {' + '.join(best_combination)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"2. 分类器: {best_classifier}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"3. 预期准确率: {best_accuracy:.4f}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 对各特征组最佳聚类的汇总
    msg = "\n各特征组最佳聚类:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for group_name, clusters in clustering_results.items():
        best_method = max(clusters.items(), key=lambda x: x[1]['silhouette'])[0]
        best_n = clusters[best_method]['n_clusters']
        best_score = clusters[best_method]['silhouette']
        
        msg = f"  {group_name}: {best_method} 聚类, {best_n} 个类别, 轮廓系数 = {best_score:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 构建最佳策略字典
    best_strategy = {
        'feature_combination': best_combination,
        'classifier': best_classifier,
        'accuracy': best_accuracy,
        'clustering': {
            group_name: {
                'method': max(clusters.items(), key=lambda x: x[1]['silhouette'])[0],
                'n_clusters': clusters[max(clusters.items(), key=lambda x: x[1]['silhouette'])[0]]['n_clusters'],
                'labels': clusters[max(clusters.items(), key=lambda x: x[1]['silhouette'])[0]]['labels']
            } for group_name, clusters in clustering_results.items()
        }
    }
    
    return best_strategy


def compare_clustering_with_big_classes(cluster_labels, big_class_labels, cluster_method, big_class_names=None, save_path=None, logger=None):
    """
    比较聚类结果与现有大类划分的一致性
    
    参数：
        cluster_labels: 聚类标签
        big_class_labels: 大类标签
        cluster_method: 聚类方法名称
        big_class_names: 大类名称列表
        save_path: 图表保存路径
        logger: 日志记录器
    
    返回：
        consistency_score: 一致性评分
    """
    # 创建混淆矩阵
    unique_clusters = np.unique(cluster_labels)
    unique_big_classes = np.unique(big_class_labels)
    n_clusters = len(unique_clusters)
    n_big_classes = len(unique_big_classes)
    
    matrix = np.zeros((n_big_classes, n_clusters))
    for i, big_class in enumerate(unique_big_classes):
        for j, cluster in enumerate(unique_clusters):
            # 计算同时属于此大类和此聚类的样本数
            matrix[i, j] = np.sum((big_class_labels == big_class) & (cluster_labels == cluster))
    
    # 计算行归一化矩阵（每个大类的分布）
    row_normalized = matrix.copy()
    row_sums = row_normalized.sum(axis=1, keepdims=True)
    row_normalized = np.divide(row_normalized, row_sums, where=row_sums!=0)
    
    # 计算列归一化矩阵（每个聚类的分布）
    col_normalized = matrix.copy()
    col_sums = col_normalized.sum(axis=0, keepdims=True)
    col_normalized = np.divide(col_normalized, col_sums, where=col_sums!=0)
    
    # 绘制热图
    plt.figure(figsize=(15, 12))
    
    # 使用大类名称（如果提供）
    if big_class_names is not None:
        y_labels = big_class_names
    else:
        y_labels = [f'Big Class {l}' for l in unique_big_classes]
    
    # Plot original confusion matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=[f'Cluster {c}' for c in unique_clusters],
            yticklabels=y_labels)
    plt.title(f'{cluster_method} Clustering Results vs. Major Categories')
    plt.xlabel('Clustering Results')
    plt.ylabel('Major Categories')

    # Plot row-normalized confusion matrix
    plt.subplot(2, 2, 2)
    sns.heatmap(row_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[f'Cluster {c}' for c in unique_clusters],
            yticklabels=y_labels)
    plt.title('Row-Normalized - Cluster Distribution per Major Category')
    plt.xlabel('Clustering Results')
    plt.ylabel('Major Categories')

    # Plot column-normalized confusion matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(col_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[f'Cluster {c}' for c in unique_clusters],
            yticklabels=y_labels)
    plt.title('Column-Normalized - Major Category Distribution per Cluster')
    plt.xlabel('Clustering Results')
    plt.ylabel('Major Categories')
    
    # 计算一致性评分
    # 使用匈牙利算法找到最佳匹配
    from scipy.optimize import linear_sum_assignment
    
    # 创建成本矩阵（要最大化一致性，所以取反）
    cost_matrix = -matrix.copy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 计算最佳匹配的一致性得分
    matched_samples = sum(matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
    total_samples = matrix.sum()
    consistency_score = matched_samples / total_samples
    
    # 显示最佳匹配和一致性得分
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.5, 0.9, 'Best Cluster-Major Category Match', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.8, f'Consistency Score: {consistency_score:.4f}', ha='center', fontsize=12)
    
    for i, (r, c) in enumerate(zip(row_ind, col_ind)):
        if i < 10:  # 只显示前10个匹配，避免过度拥挤
            big_class_name = y_labels[r]
            cluster_name = f'Cluster {unique_clusters[c]}'
            match_score = matrix[r, c] / row_sums[r]
            plt.text(0.5, 0.7 - i*0.05, f'{big_class_name} ↔ {cluster_name} ({match_score[0]:.2f})', 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{cluster_method}_vs_big_classes.png'))
    plt.close()
    
    # 打印每个聚类的主要大类
    msg = f"\n{cluster_method} 聚类与大类的对应关系:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for j, cluster in enumerate(unique_clusters):
        # 获取此聚类中各大类的数量
        class_counts = matrix[:, j]
        # 找出主要大类
        dominant_idx = np.argmax(class_counts)
        dominant_big_class = unique_big_classes[dominant_idx]
        dominant_percentage = np.max(class_counts) / np.sum(class_counts) * 100
        dominant_name = y_labels[dominant_idx]
        
        msg = f"聚类 {cluster}: 主要对应大类 = {dominant_name} ({dominant_percentage:.1f}%)"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 列出前3个主要大类
        top_indices = np.argsort(class_counts)[::-1][:3]
        for idx in top_indices:
            if class_counts[idx] > 0:
                big_class = unique_big_classes[idx]
                big_name = y_labels[idx]
                
                msg = f"  {big_name}: {class_counts[idx]} 样本 ({class_counts[idx]/np.sum(class_counts)*100:.1f}%)"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
    
    return consistency_score


def define_big_classes(logger=None):
    """
    定义基于神经解剖学的大类标签映射关系
    将102个细分类别映射到7个大类

    返回:
        fine_to_big: 细分类别到大类的映射字典
        big_to_fine: 大类到细分类别的映射字典
        big_class_names: 大类名称列表
    """
    # 定义大类名称
    big_class_names = [
        "脑室系统 (Ventricular System)",  # 0
        "白质 (White Matter)",            # 1
        "灰质-皮层 (Cortical Gray Matter)", # 2
        "深部灰质核团 (Deep Gray Nuclei)",  # 3
        "边缘系统 (Limbic System)",        # 4
        "脑干 (Brain Stem)",              # 5
        "其他结构 (Other Structures)"      # 6
    ]
    
    # 初始化映射字典
    fine_to_big = {}
    big_to_fine = {i: [] for i in range(len(big_class_names))}
    
    # 1. 脑室系统 (Ventricular System)
    ventricular_system_classes = [1, 2, 9, 10, 14, 19, 34]
    for cls in ventricular_system_classes:
        fine_to_big[cls] = 0
        big_to_fine[0].append(cls)
    
    # 2. 白质 (White Matter)
    white_matter_classes = [
        3, 22, 96, 97, 98, 99, 100, 
        # 胼胝体
        22, 23, 24, 25, 26,
        # 各脑区白质标签(wm-前缀)
        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 101, 102
    ]
    for cls in white_matter_classes:
        fine_to_big[cls] = 1
        if cls not in big_to_fine[1]:
            big_to_fine[1].append(cls)
    
    # 3. 灰质-皮层 (Cortical Gray Matter)
    cortical_gray_matter_classes = [
        4, 23,  # 小脑皮层
        # 大脑皮层标签(ctx-前缀)
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
    ]
    for cls in cortical_gray_matter_classes:
        fine_to_big[cls] = 2
        if cls not in big_to_fine[2]:
            big_to_fine[2].append(cls)
    
    # 4. 深部灰质核团 (Deep Gray Nuclei)
    deep_gray_nuclei_classes = [
        5, 6, 7, 8, 15, 16, 17, 27,  # 丘脑、尾状核、壳核等
        24, 25, 26, 27, 30, 31, 32  # 右侧对应结构
    ]
    for cls in deep_gray_nuclei_classes:
        fine_to_big[cls] = 3
        if cls not in big_to_fine[3]:
            big_to_fine[3].append(cls)
    
    # 5. 边缘系统 (Limbic System)
    limbic_system_classes = [12, 13, 28, 29]  # 海马、杏仁核
    for cls in limbic_system_classes:
        fine_to_big[cls] = 4
        if cls not in big_to_fine[4]:
            big_to_fine[4].append(cls)
    
    # 6. 脑干 (Brain Stem)
    brain_stem_classes = [11]
    for cls in brain_stem_classes:
        fine_to_big[cls] = 5
        big_to_fine[5].append(cls)
    
    # 7. 其他结构 (Other Structures)
    # 包括未包含在上述类别中的所有其他结构
    for cls in range(NUM_CLASS):  # 假设NUM_CLASS已定义为102
        if cls not in fine_to_big:
            fine_to_big[cls] = 6
            big_to_fine[6].append(cls)
    
    # 打印大类统计信息
    msg = "大类划分统计:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for i, name in enumerate(big_class_names):
        msg = f"大类 {i} - {name}: {len(big_to_fine[i])} 个细分类别"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"    包含的细分类别: {big_to_fine[i]}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return fine_to_big, big_to_fine, big_class_names


def map_to_big_classes(fine_labels, fine_to_big, logger=None):
    """
    将细分类别标签映射为大类标签
    
    参数:
        fine_labels (array): 细分类别标签
        fine_to_big (dict): 细分类别到大类的映射字典
        logger: 日志记录器
    
    返回:
        big_labels: 大类标签
    """
    # 使用numpy的vectorize功能提高效率
    mapper = np.vectorize(lambda x: fine_to_big.get(x, 6))  # 默认映射到"其他结构"类别
    big_labels = mapper(fine_labels)
    
    # 打印大类分布
    unique_classes, counts = np.unique(big_labels, return_counts=True)
    
    msg = "\n大类标签分布:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for cls, count in zip(unique_classes, counts):
        msg = f"大类 {cls}: {count} 个样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return big_labels


def load_multiclass_data_from_dirs(data_dirs, apply_pca=True, n_components=24, norm=True, logger=None):
    """
    加载多类别脑体素数据，使用与原始BrainVoxelSampler类似的逻辑
    
    参数:
        data_dirs: 包含训练、测试和验证数据目录的字典
        apply_pca: 是否应用PCA
        n_components: PCA组件数
        norm: 是否归一化
        logger: 日志记录器
    
    返回:
        包含处理后数据的字典
    """
    msg = f"从目录加载多类别数据..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"训练集目录: {data_dirs['train_dir']}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"测试集目录: {data_dirs['test_dir']}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"验证集目录: {data_dirs['val_dir']}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 加载所有标签的数据
    train_data_all = []
    train_labels_all = []
    test_data_all = []
    test_labels_all = []
    val_data_all = []
    val_labels_all = []
    
    # 获取目录中的所有体素文件
    train_files = glob.glob(os.path.join(data_dirs['train_dir'], "label_*_count_*_voxels.npy"))
    test_files = glob.glob(os.path.join(data_dirs['test_dir'], "label_*_count_*_voxels.npy"))
    val_files = glob.glob(os.path.join(data_dirs['val_dir'], "label_*_count_*_voxels.npy"))
    
    msg = f"找到 {len(train_files)} 个训练文件, {len(test_files)} 个测试文件, {len(val_files)} 个验证文件"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 处理训练集
    for file_path in train_files:
        # 从文件名提取标签ID
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'label':
            try:
                label_id = int(parts[1])
                # 确保标签在有效范围内 (0 到 NUM_CLASS-1)
                if 0 <= label_id < NUM_CLASS:
                    msg = f"加载特征数据: {filename}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    voxels = np.load(file_path)
                    labels = np.full(len(voxels), label_id)
                    train_data_all.append(voxels)
                    train_labels_all.append(labels)
                else:
                    msg = f"警告: 跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
            except ValueError:
                msg = f"警告: 无法从 {filename} 提取标签ID"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
    
    # 处理测试集
    for file_path in test_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'label':
            try:
                label_id = int(parts[1])
                if 0 <= label_id < NUM_CLASS:
                    msg = f"加载特征数据: {filename}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    voxels = np.load(file_path)
                    labels = np.full(len(voxels), label_id)
                    test_data_all.append(voxels)
                    test_labels_all.append(labels)
                else:
                    msg = f"警告: 跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
            except ValueError:
                msg = f"警告: 无法从 {filename} 提取标签ID"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
    
    # 处理验证集
    for file_path in val_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'label':
            try:
                label_id = int(parts[1])
                if 0 <= label_id < NUM_CLASS:
                    msg = f"加载特征数据: {filename}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    voxels = np.load(file_path)
                    labels = np.full(len(voxels), label_id)
                    val_data_all.append(voxels)
                    val_labels_all.append(labels)
                else:
                    msg = f"警告: 跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
            except ValueError:
                msg = f"警告: 无法从 {filename} 提取标签ID"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
    
    # 合并所有数据
    if train_data_all:
        train_data = np.vstack(train_data_all)
        train_labels = np.concatenate(train_labels_all)
    else:
        msg = "没有找到有效的训练数据"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    if test_data_all:
        test_data = np.vstack(test_data_all)
        test_labels = np.concatenate(test_labels_all)
    else:
        msg = "没有找到有效的测试数据"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    if val_data_all:
        val_data = np.vstack(val_data_all)
        val_labels = np.concatenate(val_labels_all)
    else:
        msg = "没有找到有效的验证数据"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    # 打印数据范围
    msg = f"训练集数据范围: {np.min(train_data)} 至 {np.max(train_data)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"测试集数据范围: {np.min(test_data)} 至 {np.max(test_data)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"验证集数据范围: {np.min(val_data)} 至 {np.max(val_data)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 检查标签范围
    msg = f"训练集标签范围: {np.min(train_labels)} 至 {np.max(train_labels)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"测试集标签范围: {np.min(test_labels)} 至 {np.max(test_labels)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"验证集标签范围: {np.min(val_labels)} 至 {np.max(val_labels)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 应用PCA
    if apply_pca:
        # 合并所有数据用于PCA拟合
        all_data = np.vstack([train_data, test_data, val_data])
        
        if n_components > 0:
            from sklearn.decomposition import PCA
            pca_model = PCA(n_components=n_components)
            pca_model.fit(all_data)
        else:
            # 自动选择PCA组件数
            from sklearn.decomposition import PCA
            
            # 执行PCA分析和绘图函数
            def analyze_pca_variance(data, threshold=0.95, plot=True, save_path=None):
                # 计算完整PCA
                pca_full = PCA()
                pca_full.fit(data)
                
                # 计算累积解释方差
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                
                # 确定达到阈值所需的组件数
                n_components = np.argmax(cumulative_variance >= threshold) + 1
                
                if plot:
                    plt.figure(figsize=(12, 6))
                    
                    # 绘制方差解释率
                    plt.subplot(1, 2, 1)
                    plt.plot(pca_full.explained_variance_ratio_, 'o-', markersize=4)
                    plt.title('PCA Explained Variance Ratio')
                    plt.xlabel('Principal Component')
                    plt.ylabel('Explained Variance Ratio')
                    plt.grid(True)
                    
                    # 绘制累积方差解释率
                    plt.subplot(1, 2, 2)
                    plt.plot(cumulative_variance, 'o-', markersize=4)
                    plt.axhline(y=threshold, color='r', linestyle='--', 
                             label=f'Threshold: {threshold}')
                    plt.axvline(x=n_components-1, color='g', linestyle='--',
                             label=f'Components: {n_components}')
                    plt.title('PCA Cumulative Explained Variance')
                    plt.xlabel('Number of Components')
                    plt.ylabel('Cumulative Explained Variance')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    
                    if save_path:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)
                        plt.savefig(os.path.join(save_path, 'pca_variance_analysis.png'))
                    plt.close()
                    
                return n_components, pca_full.explained_variance_ratio_, cumulative_variance
            
            n_components, _, _ = analyze_pca_variance(all_data, plot=True, save_path=SAVE_PATH)
            pca_model = PCA(n_components=n_components)
            pca_model.fit(all_data)
            
        # 应用PCA变换
        train_data = pca_model.transform(train_data)
        test_data = pca_model.transform(test_data)
        val_data = pca_model.transform(val_data#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BrainVoxel分层分类分析脚本
可使用nohup方式在后台运行，所有输出和图表会保存到指定目录
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，避免需要显示器
import matplotlib.pyplot as plt
import matplotlib.patches as mpts
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from scipy.io import loadmat
from tqdm import tqdm
import glob
import sys
import logging
import h5py
import copy

# 设置日志
def setup_logger(log_dir):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"brainvoxel_analysis_{timestamp}.log")
    
    # 创建一个logger
    logger = logging.getLogger('brainvoxel_analysis')
    logger.setLevel(logging.INFO)
    
    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建一个控制台处理器，用于在控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 超参数和实验设置
# 设置随机种子，确保实验可重复性
RANDOM_SEED = 666

# 数据采样参数
SAMPLE_RATIO = 0.1  # 使用10%的数据进行分析
USE_SAMPLING = True  # 是否使用数据采样

# 数据预处理参数
APPLY_PCA = True   # 是否应用PCA降维
NORM = True        # 是否对数据进行标准化/归一化处理

# 特征分组
DIFF_FEATURES = list(range(0, 15))     # 扩散特征 (1-15)
QTI_FEATURES = list(range(15, 225))    # QTI特征 (16-225)
CEST_FEATURES = list(range(225, 341))  # CEST特征 (226-341)

# 定义模型名称，用于结果保存和模型标识
MODEL_NAME = 'HierarchicalKAN_BrainVoxel'

# 大类定义 (初始设置，可能需要根据实际数据调整)
DEFAULT_NUM_BIG_CLASSES = 5  # 大类数量

# 指定数据集名称
DATASET = 'BrainVoxel'

# 训练参数
EPOCH = 50         # 总训练轮数
VAL_EPOCH = 1      # 每隔多少轮进行一次验证
LR = 0.001         # 学习率
WEIGHT_DECAY = 1e-6  # 权重衰减系数，用于L2正则化
BATCH_SIZE = 640    # 批处理大小，固定不变

# 计算设备选择
DEVICE = 0         # -1表示使用CPU，0表示使用第一块GPU(cuda:0)

# 数据参数
FEATURE_DIM = 341  # 输入特征总维度
NUM_CLASS = 102    # 细分类别数量
FIXED_GRID = 10    # 固定网格大小，不进行网格扩展

# 专家模型参数
DIFF_HIDDEN_DIM = 64    # 扩散专家隐藏层维度
QTI_HIDDEN_DIM = 128    # QTI专家隐藏层维度
CEST_HIDDEN_DIM = 64    # CEST专家隐藏层维度

# PCA参数
DIFF_PCA_COMPONENTS = 10   # 扩散特征PCA组件数
QTI_PCA_COMPONENTS = 30    # QTI特征PCA组件数
CEST_PCA_COMPONENTS = 20   # CEST特征PCA组件数

# 模型检查点路径
CHECK_POINT = None  # 加载预训练模型的路径，None表示从头开始训练

# 结果保存路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"./Results/{MODEL_NAME}/{DATASET}/{timestamp}"

# 数据目录
DATA_DIRS = {
    'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
    'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
    'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
}

def select_discriminative_features(data, labels, group_name, k=20, verbose=True, plot=True, save_path=None, logger=None):
    """
    选择最具区分性的特征
    
    参数：
        data: 输入数据特征
        labels: 类别标签
        group_name: 特征组名称
        k: 选择的顶部特征数量
        verbose: 是否打印信息
        plot: 是否绘制重要性分布图
        save_path: 图表保存路径
        logger: 日志记录器
        
    返回：
        selected_features: 选择后的特征
        feature_indices: 选择的特征索引
        feature_scores: 特征重要性分数
    """
    # 使用F统计量计算特征重要性
    k = min(k, data.shape[1])  # 确保k不超过特征数量
    selector = SelectKBest(f_classif, k=k)
    selected_features = selector.fit_transform(data, labels)
    feature_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    if verbose:
        msg = f"\n{group_name} 特征组选择结果:"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"原始特征维度: {data.shape[1]}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"选择后特征维度: {selected_features.shape[1]}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"均值F分数: {np.mean(feature_scores):.2f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"最大F分数: {np.max(feature_scores):.2f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 打印前10个最重要的特征索引和得分
        sorted_indices = np.argsort(feature_scores)[::-1]
        msg = "\n前10个最重要的特征:"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        for i, idx in enumerate(sorted_indices[:10]):
            msg = f"  特征 {idx}: F分数 = {feature_scores[idx]:.2f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
        
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Left plot: Importance distribution of all features
        plt.subplot(1, 2, 1)
        sorted_indices = np.argsort(feature_scores)[::-1]
        plt.bar(range(len(feature_scores)), feature_scores[sorted_indices])
        plt.title(f'{group_name} Feature Importance (All)')
        plt.xlabel('Feature Rank')
        plt.ylabel('F-score')
        plt.yscale('log')  # Logarithmic scale
        plt.grid(True)
        
        # Right plot: Importance of selected features
        plt.subplot(1, 2, 2)
        selected_scores = feature_scores[feature_indices]
        sorted_selected = np.argsort(selected_scores)[::-1]
        plt.bar(range(len(selected_scores)), selected_scores[sorted_selected])
        plt.title(f'{group_name} Feature Importance (Selected)')
        plt.xlabel('Feature Rank')
        plt.ylabel('F-score')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'{group_name}_feature_importance.png'))
        plt.close()

    return selected_features, feature_indices, feature_scores

def advanced_feature_reduction(data, method='umap', n_components=2, labels=None, plot=True, title=None, save_path=None, logger=None):
    """
    高级特征降维与可视化
    
    参数：
        data: 输入数据
        method: 降维方法，可选 'pca', 'tsne', 'umap', 'mds'
        n_components: 降维后的维度
        labels: 类别标签，用于可视化
        plot: 是否绘制降维结果
        title: 图表标题
        save_path: 图表保存路径
        logger: 日志记录器
        
    返回：
        reduced_data: 降维后的数据
        reducer: 降维模型
    """
    # 确保数据是浮点型
    data = data.astype(np.float32)
    
    # 根据指定方法进行降维
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(data)
        explained_var = reducer.explained_variance_ratio_
        explained_var_str = f"解释方差: {sum(explained_var):.2%}"
    
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, 
                      perplexity=min(30, data.shape[0] // 5), 
                      n_iter=1000, 
                      random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components,
                          n_neighbors=min(30, data.shape[0] // 5),
                          min_dist=0.1,
                          random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    elif method.lower() == 'mds':
        reducer = MDS(n_components=n_components, n_jobs=-1, random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    else:
        msg = f"不支持的降维方法: {method}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    # 可视化降维结果
    if plot and labels is not None and n_components in [2, 3]:
        plt.figure(figsize=(12, 10))
        
        if n_components == 2:
            # 2D可视化
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                           alpha=0.6, label=f'Class {label}')
            
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
        else:
            # 3D可视化
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2],
                         alpha=0.6, label=f'Class {label}')
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        
        # 设置标题
        if title:
            plt.title(f'{title} ({method.upper()} projection) {explained_var_str}')
        else:
            plt.title(f'{method.upper()} projection {explained_var_str}')
        
        plt.grid(True)
        # 仅显示部分标签，避免图例过大
        if len(unique_labels) > 10:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        else:
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'{title}_{method}_visualization.png' if title else f'{method}_visualization.png'))
        plt.close()
    
    return reduced_data, reducer

def analyze_optimal_clusters(data, min_clusters=2, max_clusters=10, methods=['kmeans', 'spectral', 'agglomerative'], save_path=None, logger=None):
    """
    分析最佳聚类数量和方法
    
    参数：
        data: 输入数据
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
        methods: 要尝试的聚类方法列表
        save_path: 图表保存路径
        logger: 日志记录器
        
    返回：
        best_results: 最佳聚类结果的字典
    """
    import time
    
    # 如果数据过大，进行下采样
    if data.shape[0] > 100000:
        from sklearn.model_selection import train_test_split
        _, data = train_test_split(data, test_size=100000/data.shape[0], random_state=RANDOM_SEED)
        msg = f"数据太大，下采样到 {data.shape[0]} 样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 初始化结果存储
    silhouette_scores = {method: [] for method in methods}
    calinski_scores = {method: [] for method in methods}
    davies_scores = {method: [] for method in methods}
    cluster_labels = {method: {} for method in methods}
    
    msg = "分析最佳聚类数量..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 尝试不同的聚类数量和方法
    for n_clusters in range(min_clusters, max_clusters+1):
        msg = f"\n尝试 {n_clusters} 个聚类:"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        for method in methods:
            start_time = time.time()
            msg = f"  方法: {method}..."
            if logger:
                logger.info(msg)
            else:
                print(msg, end="", flush=True)
            
            # 创建聚类模型
            if method == 'kmeans':
                cluster_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=5)
            elif method == 'spectral':
                cluster_model = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED, 
                                                  affinity='nearest_neighbors', n_neighbors=min(30, data.shape[0]//100))
            elif method == 'agglomerative':
                cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                if logger:
                    logger.error(f"不支持的聚类方法: {method}")
                raise ValueError(f"不支持的聚类方法: {method}")
            
            # 执行聚类
            labels = cluster_model.fit_predict(data)
            
            # 存储标签
            cluster_labels[method][n_clusters] = labels
            
            # 计算聚类评估指标
            try:
                if len(np.unique(labels)) > 1:  # 确保至少有两个聚类
                    # 为了加速，使用数据样本计算指标
                    if data.shape[0] > 10000:
                        sample_idx = np.random.choice(data.shape[0], 10000, replace=False)
                        sample_data = data[sample_idx]
                        sample_labels = labels[sample_idx]
                        sil_score = silhouette_score(sample_data, sample_labels)
                        cal_score = calinski_harabasz_score(sample_data, sample_labels)
                        dav_score = davies_bouldin_score(sample_data, sample_labels)
                    else:
                        sil_score = silhouette_score(data, labels)
                        cal_score = calinski_harabasz_score(data, labels)
                        dav_score = davies_bouldin_score(data, labels)
                    
                    silhouette_scores[method].append(sil_score)
                    calinski_scores[method].append(cal_score)
                    davies_scores[method].append(dav_score)
                    
                    elapsed = time.time() - start_time
                    
                    msg = f" 完成! ({elapsed:.1f}秒)"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    msg = f"    轮廓系数: {sil_score:.4f}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    msg = f"    Calinski-Harabasz指数: {cal_score:.1f}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    msg = f"    Davies-Bouldin指数: {dav_score:.4f}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    
                    # 统计不同类别的样本数
                    unique, counts = np.unique(labels, return_counts=True)
                    dist_str = ", ".join([f"类别{int(u)}:{c}" for u, c in zip(unique, counts)])
                    
                    msg = f"    类别分布: {dist_str}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                else:
                    msg = f" 警告: {method} 产生了单一聚类或空聚类"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                        
                    silhouette_scores[method].append(-1)
                    calinski_scores[method].append(-1)
                    davies_scores[method].append(float('inf'))
            except Exception as e:
                msg = f" 评估指标计算错误: {e}"
                if logger:
                    logger.error(msg)
                else:
                    print(msg)
                    
                silhouette_scores[method].append(-1)
                calinski_scores[method].append(-1)
                davies_scores[method].append(float('inf'))
    
    # 绘制评估指标
    plt.figure(figsize=(15, 12))
    
    # 轮廓系数（越高越好）
    plt.subplot(3, 1, 1)
    for method in methods:
        if len(silhouette_scores[method]) > 0:
            plt.plot(range(min_clusters, min_clusters + len(silhouette_scores[method])),
                    silhouette_scores[method], 'o-', label=method)
    plt.title('Silhouette Score (Higher is Better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.legend()

    # Calinski-Harabasz指数（越高越好）
    plt.subplot(3, 1, 2)
    for method in methods:
        if len(calinski_scores[method]) > 0:
            plt.plot(range(min_clusters, min_clusters + len(calinski_scores[method])),
                    calinski_scores[method], 'o-', label=method)
    plt.title('Calinski-Harabasz Index (Higher is Better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Index')
    plt.grid(True)
    plt.legend()
    
    # Davies-Bouldin指数（越低越好）
    plt.subplot(3, 1, 3)
    for method in methods:
        if len(davies_scores[method]) > 0:
            plt.plot(range(min_clusters, min_clusters + len(davies_scores[method])),
                    davies_scores[method], 'o-', label=method)
    plt.title('Davies-Bouldin Index (Lower is Better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'cluster_analysis.png'))
    plt.close()
    
    # 查找最佳结果
    best_results = {}
    for method in methods:
        if len(silhouette_scores[method]) > 0:
            best_idx = np.argmax(silhouette_scores[method])
            best_n = min_clusters + best_idx
            best_results[method] = {
                'n_clusters': best_n,
                'silhouette': silhouette_scores[method][best_idx],
                'calinski': calinski_scores[method][best_idx],
                'davies': davies_scores[method][best_idx],
                'labels': cluster_labels[method][best_n]
            }
    
    # 打印最佳结果
    msg = "\n最佳聚类结果:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for method, result in best_results.items():
        msg = f"\n{method.upper()} 最佳聚类数量: {result['n_clusters']}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"  轮廓系数: {result['silhouette']:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"  Calinski-Harabasz指数: {result['calinski']:.1f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"  Davies-Bouldin指数: {result['davies']:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 统计类别分布
        unique, counts = np.unique(result['labels'], return_counts=True)
        dist_str = ", ".join([f"类别{int(u)}:{c}" for u, c in zip(unique, counts)])
        
        msg = f"  类别分布: {dist_str}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return best_results

def explore_best_classification_strategy(feature_groups, labels, save_path=None, use_sampling=True, sample_ratio=0.1, logger=None):
    """
    探索最佳分类策略
    
    参数：
        feature_groups: 处理后的特征组
        labels: 类别标签
        save_path: 结果保存路径
        use_sampling: 是否使用采样来加速分析
        sample_ratio: 采样比例
        logger: 日志记录器
        
    返回：
        best_strategy: 最佳分类策略
    """
    import time
    msg = "探索最佳分类策略..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 如果使用采样，创建一个采样子集
    if use_sampling:
        msg = f"使用{sample_ratio*100:.1f}%的数据进行分析"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        from sklearn.model_selection import train_test_split
        
        # 使用分层采样保持类别分布
        indices = np.arange(len(labels))
        _, sampled_indices, _, sampled_labels = train_test_split(
            indices, labels, test_size=sample_ratio, stratify=labels, random_state=RANDOM_SEED
        )
        
        # 打印采样后的样本数量和类别分布
        msg = f"原始数据: {len(labels)} 样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"采样后: {len(sampled_labels)} 样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        unique, counts = np.unique(sampled_labels, return_counts=True)
        msg = f"采样后类别分布: {len(unique)} 个唯一类别"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 对每个特征组进行采样
        sampled_groups = {}
        for group_name, group_data in feature_groups.items():
            sampled_groups[group_name] = group_data[sampled_indices]
        
        # 使用采样后的数据
        analysis_groups = sampled_groups
        analysis_labels = sampled_labels
    else:
        # 使用全部数据
        analysis_groups = feature_groups
        analysis_labels = labels
    
    # 1. 降维分析 - 对每个特征组降维并可视化
    msg = "\n1. 降维分析"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    reduction_results = {}
    
    for group_name, group_data in analysis_groups.items():
        msg = f"\n对 {group_name} 特征组进行降维分析..."
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 对每个特征组应用多种降维方法
        for method in ['pca', 'umap']:  # 只使用两种方法来加速
            start_time = time.time()
            msg = f"  使用 {method} 降维..."
            if logger:
                logger.info(msg)
            else:
                print(msg, end="", flush=True)
                
            reduced_data, _ = advanced_feature_reduction(
                group_data, method=method, n_components=2, labels=analysis_labels,
                plot=True, title=f"{group_name} Feature Space", 
                save_path=save_path, logger=logger
            )
            elapsed = time.time() - start_time
            
            msg = f" 完成! ({elapsed:.1f}秒)"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            
            reduction_results[f"{group_name}_{method}"] = reduced_data
    
    # 2. 聚类分析 - 查找最佳聚类数量和方法
    msg = "\n2. 聚类分析"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    clustering_results = {}
    
    for group_name, group_data in analysis_groups.items():
        msg = f"\n对 {group_name} 特征组进行聚类分析..."
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 对特征组进行聚类分析
        start_time = time.time()
        msg = f"  分析最佳聚类..."
        if logger:
            logger.info(msg)
        else:
            print(msg, end="", flush=True)
            
        best_clusters = analyze_optimal_clusters(
            group_data, min_clusters=2, max_clusters=7,  # 减少聚类数范围
            methods=['kmeans', 'spectral'],  # 只使用两种方法来加速
            save_path=save_path, logger=logger
        )
        elapsed = time.time() - start_time
        
        msg = f" 完成! ({elapsed:.1f}秒)"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        clustering_results[group_name] = best_clusters
        
        # 分析聚类结果与原始标签的对应关系
        for method, result in best_clusters.items():
            visualize_cluster_vs_labels(
                result['labels'], analysis_labels, f"{group_name}_{method}",
                save_path=save_path, logger=logger
            )
    
    # 3. 特征组合分析 - 找出最佳特征组合
    msg = "\n3. 特征组合分析"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    start_time = time.time()
    msg = "  评估特征组合..."
    if logger:
        logger.info(msg)
    else:
        print(msg, end="", flush=True)
        
    combination_results = analyze_feature_group_combinations(
        analysis_groups, analysis_labels, save_path=save_path, logger=logger
    )
    elapsed = time.time() - start_time
    
    msg = f" 完成! ({elapsed:.1f}秒)"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 4. 分类模型比较
    msg = "\n4. 分类模型比较"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # 获取最佳特征组合
    best_combination = combination_results[0]['combination']
    msg = f"使用最佳特征组合: {' + '.join(best_combination)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 合并最佳特征组合
    combined_features = []
    for group in best_combination:
        combined_features.append(analysis_groups[group])
    X = np.hstack(combined_features)
    
    # 比较不同分类器
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', C=1),
        'RF': RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=RANDOM_SEED)
    }
    
    model_results = []
    cv_splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=RANDOM_SEED)
    
    for clf_name, clf in classifiers.items():
        start_time = time.time()
        msg = f"\n评估 {clf_name} 分类器..."
        if logger:
            logger.info(msg)
        else:
            print(msg, end="", flush=True)
        
        # 使用自定义的交叉验证而不是完整的cross_val_score
        scores = []
        for train_idx, test_idx in cv_splitter.split(X, analysis_labels):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = analysis_labels[train_idx], analysis_labels[test_idx]
            
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        elapsed = time.time() - start_time
        
        msg = f" 完成! ({elapsed:.1f}秒)"
    """
    分析不同特征组合的分类性能
    
    参数：
        feature_groups: 特征组字典
        labels: 标签
        group_names: 特征组名称列表
        save_path: 结果保存路径
        logger: 日志记录器
    """
    if group_names is None:
        group_names = list(feature_groups.keys())
    
    # 初始化性能矩阵
    n_groups = len(group_names)
    combinations = 2**n_groups - 1  # 所有可能的非空组合
    
    # 准备结果存储
    results = []
    
    # 测试单个组以及不同组合
    msg = "分析不同特征组合的分类性能..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 首先测试单个特征组
    for i, group in enumerate(group_names):
        msg = f"\n测试单个特征组: {group}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        X = feature_groups[group]
        
        # 使用随机森林评估性能
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # 使用5折交叉验证
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf, X, labels, cv=5, scoring='accuracy')
        
        acc = scores.mean()
        std = scores.std()
        
        msg = f"  准确率: {acc:.4f} ± {std:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 存储结果
        results.append({
            'combination': [group],
            'accuracy': acc,
            'std': std
        })
    
    # 测试组合（最多3个组的组合，以避免组合爆炸）
    if n_groups >= 2:
        # 测试两两组合
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                group_i = group_names[i]
                group_j = group_names[j]
                
                msg = f"\n测试特征组合: {group_i} + {group_j}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                
                # 合并特征
                X = np.hstack([feature_groups[group_i], feature_groups[group_j]])
                
                # 评估性能
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                scores = cross_val_score(rf, X, labels, cv=5, scoring='accuracy')
                
                acc = scores.mean()
                std = scores.std()
                
                msg = f"  准确率: {acc:.4f} ± {std:.4f}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                
                # 存储结果
                results.append({
                    'combination': [group_i, group_j],
                    'accuracy': acc,
                    'std': std
                })
    
    # 测试三组组合（如果有3个或更多组）
    if n_groups >= 3:
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                for k in range(j+1, n_groups):
                    group_i = group_names[i]
                    group_j = group_names[j]
                    group_k = group_names[k]
                    
                    msg = f"\n测试特征组合: {group_i} + {group_j} + {group_k}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    
                    # 合并特征
                    X = np.hstack([feature_groups[group_i], feature_groups[group_j], feature_groups[group_k]])
                    
                    # 评估性能
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    scores = cross_val_score(rf, X, labels, cv=5, scoring='accuracy')
                    
                    acc = scores.mean()
                    std = scores.std()
                    
                    msg = f"  准确率: {acc:.4f} ± {std:.4f}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    
                    # 存储结果
                    results.append({
                        'combination': [group_i, group_j, group_k],
                        'accuracy': acc,
                        'std': std
                    })
    
    # 如果有全部特征组
    if n_groups > 1:
        msg = "\n测试所有特征组合"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        all_features = []
        for group in group_names:
            all_features.append(feature_groups[group])
        
        # 合并特征
        X = np.hstack(all_features)
        
        # 评估性能
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf, X, labels, cv=5, scoring='accuracy')
        
        acc = scores.mean()
        std = scores.std()
        
        msg = f"  准确率: {acc:.4f} ± {std:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 存储结果
        results.append({
            'combination': group_names,
            'accuracy': acc,
            'std': std
        })
    
    # 对结果排序
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # 打印排序后的结果
    msg = "\n所有特征组合的性能排名:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for i, result in enumerate(results):
        combination_str = ' + '.join(result['combination'])
        msg = f"{i+1}. {combination_str}: {result['accuracy']:.4f} ± {result['std']:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 可视化结果
    plt.figure(figsize=(14, 8))
    
    # 准备数据
    combinations = [' + '.join(r['combination']) for r in results]
    accuracies = [r['accuracy'] for r in results]
    stds = [r['std'] for r in results]
    
    # 绘制条形图
    plt.bar(combinations, accuracies, yerr=stds, alpha=0.8)
    plt.axhline(y=1/len(np.unique(labels)), color='r', linestyle='--', 
            label=f'Random Guessing ({1/len(np.unique(labels)):.4f})')

    plt.xlabel('Feature Combinations')
    plt.ylabel('5-Fold Cross-Validation Accuracy')
    plt.title('Classification Performance of Different Feature Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'feature_combination_analysis.png'))
    plt.close()
    
    return results


def enhanced_preprocess_feature_groups(data, feature_indices, labels, normalize_method='robust', apply_feature_selection=True, n_features=None, save_path=None, logger=None):
    """
    增强版特征组预处理
    
    参数：
        data: 输入数据
        feature_indices: 特征索引字典
        labels: 类别标签
        normalize_method: 归一化方法，'standard'或'robust'
        apply_feature_selection: 是否应用特征选择
        n_features: 各特征组选择的特征数量字典
        save_path: 结果保存路径
        logger: 日志记录器
        
    返回：
        processed_groups: 处理后的特征组
        selected_indices: 选择的特征索引
    """
    processed_groups = {}
    selected_indices = {}
    
    # 设置默认特征数量
    if n_features is None:
        n_features = {
            'diffusion': min(10, len(feature_indices['diffusion'])),
            'qti': min(20, len(feature_indices['qti'])),
            'cest': min(15, len(feature_indices['cest']))
        }
    
    # 处理每个特征组
    for group_name, indices in feature_indices.items():
        msg = f"\n处理 {group_name} 特征组..."
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 提取特征
        group_data = data[:, indices]
        
        # 归一化
        if normalize_method.lower() == 'standard':
            msg = f"  使用StandardScaler标准化 {group_name} 特征..."
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
            scaler = StandardScaler()
            group_data = scaler.fit_transform(group_data)
        elif normalize_method.lower() == 'robust':
            msg = f"  使用RobustScaler归一化 {group_name} 特征..."
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
            scaler = RobustScaler()
            group_data = scaler.fit_transform(group_data)
        
        # 特征选择
        if apply_feature_selection:
            k = min(n_features[group_name], group_data.shape[1])
            msg = f"  选择 {group_name} 特征组中的前 {k} 个特征..."
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
            selected_data, indices, scores = select_discriminative_features(
                group_data, labels, group_name, k=k, save_path=save_path, logger=logger
            )
            processed_groups[group_name] = selected_data
            selected_indices[group_name] = indices
        else:
            processed_groups[group_name] = group_data

    # 打印处理后的维度
    for group_name, group_data in processed_groups.items():
        msg = f"  {group_name} 特征组处理后维度: {group_data.shape}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return processed_groups, selected_indices


def analyze_class_separability(feature_groups, labels, save_path=None, use_sampling=True, sample_ratio=0.1, logger=None):
    """
    分析特征空间中的类别可分性
    
    参数：
        feature_groups: 处理后的特征组
        labels: 类别标签
        save_path: 结果保存路径
        use_sampling: 是否使用采样来加速分析
        sample_ratio: 采样比例
        logger: 日志记录器
        
    返回：
        separability_scores: 各特征组的可分性评分
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedShuffleSplit
    import time
    
    separability_scores = {}
    
    msg = "分析特征空间中的类别可分性..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 如果使用采样，创建一个采样子集
    if use_sampling:
        msg = f"使用{sample_ratio*100:.1f}%的数据进行分析"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        from sklearn.model_selection import train_test_split
        
        # 使用分层采样保持类别分布
        indices = np.arange(len(labels))
        _, sampled_indices, _, sampled_labels = train_test_split(
            indices, labels, test_size=sample_ratio, stratify=labels, random_state=RANDOM_SEED
        )
        
        # 打印采样后的样本数量和类别分布
        msg = f"原始数据: {len(labels)} 样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"采样后: {len(sampled_labels)} 样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        unique, counts = np.unique(sampled_labels, return_counts=True)
        msg = f"采样后类别分布: {len(unique)} 个唯一类别"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 对每个特征组进行采样
        sampled_groups = {}
        for group_name, group_data in feature_groups.items():
            sampled_groups[group_name] = group_data[sampled_indices]
        
        # 使用采样后的数据
        analysis_groups = sampled_groups
        analysis_labels = sampled_labels
    else:
        # 使用全部数据
        analysis_groups = feature_groups
        analysis_labels = labels
    
    # 使用交叉验证分割器而不是完整的k折交叉验证
    cv_splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=RANDOM_SEED)
    
    for group_name, group_data in analysis_groups.items():
        msg = f"\n分析 {group_name} 特征组..."
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 使用多个分类器评估可分性
        classifiers = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', C=1, probability=True),
            'RF': RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
        }
        
        group_scores = {}
        
        for clf_name, clf in classifiers.items():
            start_time = time.time()
            msg = f"  计算 {clf_name} 分类器性能..."
            if logger:
                logger.info(msg)
            else:
                print(msg, end="", flush=True)
            
            # 使用自定义的交叉验证而不是完整的cross_val_score
            scores = []
            for train_idx, test_idx in cv_splitter.split(group_data, analysis_labels):
                X_train, X_test = group_data[train_idx], group_data[test_idx]
                y_train, y_test = analysis_labels[train_idx], analysis_labels[test_idx]
                
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            elapsed = time.time() - start_time
            
            msg = f" 完成! ({elapsed:.1f}秒)"
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
            msg = f"    结果: {mean_score:.4f} ± {std_score:.4f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            
            group_scores[clf_name] = {
                'mean': mean_score,
                'std': std_score
            }
        
        # 计算平均得分作为总体可分性评分
        mean_separability = np.mean([s['mean'] for s in group_scores.values()])
        separability_scores[group_name] = {
            'overall': mean_separability,
            'classifiers': group_scores
        }
        
        msg = f"  总体可分性评分: {mean_separability:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    groups = list(separability_scores.keys())
    clf_names = list(separability_scores[groups[0]]['classifiers'].keys())
    
    x = np.arange(len(groups))
    width = 0.25
    offsets = np.linspace(-width, width, len(clf_names))
    
    # 绘制条形图
    for i, clf_name in enumerate(clf_names):
        means = [separability_scores[g]['classifiers'][clf_name]['mean'] for g in groups]
        stds = [separability_scores[g]['classifiers'][clf_name]['std'] for g in groups]
        
        plt.bar(x + offsets[i], means, width, label=clf_name, yerr=stds)
    
    # 绘制随机猜测基线
    plt.axhline(y=1/len(np.unique(labels)), color='r', linestyle='--', 
           label=f'Random Guessing ({1/len(np.unique(labels)):.4f})')

    plt.xlabel('Feature Groups')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Class Separability of Different Feature Groups')
    plt.xticks(x, groups)
    plt.legend()
    plt.grid(axis='y')

    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'class_separability.png'))
    plt.close()
    
    return separability_scores


def visualize_cluster_vs_labels(cluster_labels, original_labels, cluster_method, save_path=None, logger=None):
    """
    可视化聚类结果与原始标签的对应关系
    
    参数：
        cluster_labels: 聚类标签
        original_labels: 原始标签
        cluster_method: 聚类方法名称
        save_path: 图表保存路径
        logger: 日志记录器
    """
    # 创建混淆矩阵
    # 行是原始类别，列是聚类结果
    unique_clusters = np.unique(cluster_labels)
    unique_labels = np.unique(original_labels)
    n_clusters = len(unique_clusters)
    n_labels = len(unique_labels)
    
    matrix = np.zeros((n_labels, n_clusters))
    for i, label in enumerate(unique_labels):
        for j, cluster in enumerate(unique_clusters):
            # 计算同时属于此标签和此聚类的样本数
            matrix[i, j] = np.sum((original_labels == label) & (cluster_labels == cluster))
    
    # 计算行归一化矩阵（每个原始类别的分布）
    row_normalized = matrix.copy()
    row_sums = row_normalized.sum(axis=1, keepdims=True)
    row_normalized = np.divide(row_normalized, row_sums, where=row_sums!=0)
    
    # 绘制热图
    plt.figure(figsize=(15, 10))
    
    # 绘制原始混淆矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=[f'Cluster {c}' for c in unique_clusters],
            yticklabels=[f'Class {l}' for l in unique_labels])
    plt.title(f'{cluster_method.upper()} Clustering Results vs. Original Labels')
    plt.xlabel('Clustering Results')
    plt.ylabel('Original Labels')
    
    # 绘制行归一化混淆矩阵
    plt.subplot(1, 2, 2)
    sns.heatmap(row_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[f'Cluster {c}' for c in unique_clusters],
            yticklabels=[f'Class {l}' for l in unique_labels])
    plt.title(f'{cluster_method.upper()} Clustering Results vs. Original Labels (Normalized)')
    plt.xlabel('Clustering Results')
    plt.ylabel('Original Labels')
    
    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{cluster_method}_cluster_vs_labels.png'))
    plt.close()
    
    # 计算每个聚类的主要类别
    msg = f"\n{cluster_method.upper()} 聚类结果分析:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for j, cluster in enumerate(unique_clusters):
        # 获取此聚类中各原始类别的数量
        class_counts = matrix[:, j]
        # 找出主要类别
        dominant_label = unique_labels[np.argmax(class_counts)]
        dominant_percentage = np.max(class_counts) / np.sum(class_counts) * 100
        
        msg = f"聚类 {cluster}: 主要类别 = {dominant_label} ({dominant_percentage:.1f}%)"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        # 列出前3个主要类别
        top_indices = np.argsort(class_counts)[::-1][:3]
        for idx in top_indices:
            if class_counts[idx] > 0:
                msg = f"  类别 {unique_labels[idx]}: {class_counts[idx]} 样本 ({class_counts[idx]/np.sum(class_counts)*100:.1f}%)"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)