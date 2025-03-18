#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征分析模块 - 实现特征选择、降维、聚类分析
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
import time


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
                os.makedirs(save_path, exist_ok=True)
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
                os.makedirs(save_path, exist_ok=True)
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
    from config import RANDOM_SEED
    
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
            os.makedirs(save_path, exist_ok=True)
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
            os.makedirs(save_path, exist_ok=True)
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
    from config import RANDOM_SEED
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
        msg = f"\\n分析 {group_name} 特征组..."
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
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'class_separability.png'))
    plt.close()
    
    return separability_scores