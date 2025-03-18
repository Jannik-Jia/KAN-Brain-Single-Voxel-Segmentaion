#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聚类实验，评估不同聚类方法和参数的效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FIGURES_DIR, RESULTS_DIR, FEATURE_GROUPS, CLUSTERING, DEFAULT_NUM_BIG_CLASSES
from data.data_loader import load_multiclass_data_from_dirs, define_big_classes, map_to_big_classes
from data.preprocessing import preprocess_feature_groups
from analysis.clustering import visualize_cluster_vs_labels, analyze_cluster_stability
from utils.logging_utils import get_logger, log_section, log_execution_time
from utils.model_utils import save_results

# 获取日志记录器
logger = get_logger("clustering_exp")

def evaluate_clustering_parameters(data, original_labels, cluster_method='kmeans', 
                                  min_clusters=2, max_clusters=10, verbose=True):
    """
    评估聚类参数的影响
    
    参数：
        data: 输入数据
        original_labels: 原始标签
        cluster_method: 聚类方法
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
        verbose: 是否打印详细信息
        
    返回：
        results: 评估结果字典
    """
    if verbose:
        logger.info(f"评估 {cluster_method} 聚类方法的参数...")
    
    # 准备结果存储
    results = {}
    
    # 评估指标
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    # 稳定性评分
    stability_scores = []
    
    # 与原始标签的一致性
    consistency_scores = []
    
    # 每个聚类数量的标签
    all_labels = {}
    
    # 对每个聚类数量进行评估
    for n_clusters in range(min_clusters, max_clusters+1):
        if verbose:
            logger.info(f"\n评估聚类数量: {n_clusters}")
        
        try:
            # 创建并训练聚类模型
            if cluster_method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(data)
            elif cluster_method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                        affinity='nearest_neighbors', 
                                        n_neighbors=min(30, data.shape[0]//100))
                labels = model.fit_predict(data)
            elif cluster_method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(data)
            else:
                if verbose:
                    logger.error(f"不支持的聚类方法: {cluster_method}")
                return None
            
            # 存储聚类标签
            all_labels[n_clusters] = labels
            
            # 计算内部评估指标
            sil = silhouette_score(data, labels)
            cal = calinski_harabasz_score(data, labels)
            dav = davies_bouldin_score(data, labels)
            
            silhouette_scores.append(sil)
            calinski_scores.append(cal)
            davies_scores.append(dav)
            
            if verbose:
                logger.info(f"  轮廓系数: {sil:.4f}")
                logger.info(f"  Calinski-Harabasz指数: {cal:.1f}")
                logger.info(f"  Davies-Bouldin指数: {dav:.4f}")
            
            # 评估聚类稳定性
            stability, _ = analyze_cluster_stability(
                data, n_clusters, method=cluster_method, n_runs=5, 
                verbose=False, plot=False
            )
            stability_scores.append(stability)
            
            if verbose:
                logger.info(f"  稳定性得分: {stability:.4f}")
            
            # 评估与原始标签的一致性
            consistency, _ = visualize_cluster_vs_labels(
                labels, original_labels, f"{cluster_method}_{n_clusters}",
                verbose=False, plot=False
            )
            consistency_scores.append(consistency)
            
            if verbose:
                logger.info(f"  与原始标签的一致性: {consistency:.4f}")
        
        except Exception as e:
            if verbose:
                logger.error(f"  评估聚类数量 {n_clusters} 时出错: {e}")
            # 添加占位值
            silhouette_scores.append(-1)
            calinski_scores.append(-1)
            davies_scores.append(float('inf'))
            stability_scores.append(-1)
            consistency_scores.append(-1)
    
    # 存储结果
    results = {
        'cluster_method': cluster_method,
        'n_clusters_range': list(range(min_clusters, max_clusters+1)),
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_scores': davies_scores,
        'stability_scores': stability_scores,
        'consistency_scores': consistency_scores,
        'all_labels': all_labels
    }
    
    # 找出最佳聚类数量（根据轮廓系数）
    best_idx = np.argmax(silhouette_scores)
    best_n = min_clusters + best_idx
    
    # 找出与原始标签最一致的聚类数量
    best_consistency_idx = np.argmax(consistency_scores)
    best_consistency_n = min_clusters + best_consistency_idx
    
    results['best_n_clusters'] = best_n
    results['best_silhouette'] = silhouette_scores[best_idx]
    results['best_consistency_n'] = best_consistency_n
    results['best_consistency'] = consistency_scores[best_consistency_idx]
    
    if verbose:
        logger.info(f"\n最佳聚类数量(轮廓系数): {best_n}, 轮廓系数: {silhouette_scores[best_idx]:.4f}")
        logger.info(f"最佳聚类数量(一致性): {best_consistency_n}, 一致性: {consistency_scores[best_consistency_idx]:.4f}")
    
    # 绘制评估结果
    plt.figure(figsize=(15, 12))
    
    # 绘制内部评估指标
    plt.subplot(3, 1, 1)
    plt.plot(results['n_clusters_range'], silhouette_scores, 'o-', label='Silhouette Score')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Best n={best_n}')
    plt.axvline(x=DEFAULT_NUM_BIG_CLASSES, color='g', linestyle=':', 
               label=f'Default n={DEFAULT_NUM_BIG_CLASSES}')
    plt.title(f'{cluster_method.capitalize()} Clustering: Internal Evaluation Metrics')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.legend()
    
    # 绘制稳定性得分
    plt.subplot(3, 1, 2)
    plt.plot(results['n_clusters_range'], stability_scores, 'o-', label='Stability Score')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Best n={best_n}')
    plt.axvline(x=DEFAULT_NUM_BIG_CLASSES, color='g', linestyle=':', 
               label=f'Default n={DEFAULT_NUM_BIG_CLASSES}')
    plt.title(f'{cluster_method.capitalize()} Clustering: Stability Evaluation')
    plt.ylabel('Stability Score')
    plt.grid(True)
    plt.legend()
    
    # 绘制一致性得分
    plt.subplot(3, 1, 3)
    plt.plot(results['n_clusters_range'], consistency_scores, 'o-', label='Consistency Score')
    plt.axvline(x=best_consistency_n, color='r', linestyle='--', label=f'Best n={best_consistency_n}')
    plt.axvline(x=DEFAULT_NUM_BIG_CLASSES, color='g', linestyle=':', 
               label=f'Default n={DEFAULT_NUM_BIG_CLASSES}')
    plt.title(f'{cluster_method.capitalize()} Clustering: Consistency with Original Labels')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Consistency Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, f"{cluster_method}_parameter_evaluation.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"聚类参数评估图表已保存至: {save_path}")
    
    plt.close()
    
    return results

def compare_clustering_methods(data, original_labels, methods=None, n_clusters=DEFAULT_NUM_BIG_CLASSES, 
                             verbose=True):
    """
    比较不同的聚类方法
    
    参数:
        data: 输入数据
        original_labels: 原始标签
        methods: 聚类方法列表
        n_clusters: 聚类数量
        verbose: 是否打印详细信息
        
    返回:
        results: 比较结果字典
    """
    if methods is None:
        methods = ['kmeans', 'spectral', 'agglomerative']
    
    if verbose:
        logger.info(f"比较聚类方法 (聚类数量={n_clusters})...")
    
    # 准备结果存储
    results = {}
    
    # 对每种方法进行评估
    for method in methods:
        if verbose:
            logger.info(f"\n评估 {method} 聚类方法...")
        
        try:
            # 创建并训练聚类模型
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(data)
            elif method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                          affinity='nearest_neighbors', 
                                          n_neighbors=min(30, data.shape[0]//100))
                labels = model.fit_predict(data)
            elif method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(data)
            else:
                if verbose:
                    logger.error(f"不支持的聚类方法: {method}")
                continue
            
            # 计算内部评估指标
            sil = silhouette_score(data, labels)
            cal = calinski_harabasz_score(data, labels)
            dav = davies_bouldin_score(data, labels)
            
            # 评估聚类稳定性
            stability, _ = analyze_cluster_stability(
                data, n_clusters, method=method, n_runs=5, 
                verbose=False, plot=False
            )
            
            # 评估与原始标签的一致性
            consistency, _ = visualize_cluster_vs_labels(
                labels, original_labels, f"{method}_{n_clusters}",
                verbose=False, plot=False
            )
            
            if verbose:
                logger.info(f"  轮廓系数: {sil:.4f}")
                logger.info(f"  Calinski-Harabasz指数: {cal:.1f}")
                logger.info(f"  Davies-Bouldin指数: {dav:.4f}")
                logger.info(f"  稳定性得分: {stability:.4f}")
                logger.info(f"  与原始标签的一致性: {consistency:.4f}")
            
            # 存储结果
            results[method] = {
                'silhouette': sil,
                'calinski': cal,
                'davies': dav,
                'stability': stability,
                'consistency': consistency,
                'labels': labels
            }
        
        except Exception as e:
            if verbose:
                logger.error(f"  评估聚类方法 {method} 时出错: {e}")
    
    # 找出最佳方法（根据轮廓系数）
    best_method = max(results.items(), key=lambda x: x[1]['silhouette'])[0]
    best_silhouette = results[best_method]['silhouette']
    
    # 找出最稳定的方法
    most_stable = max(results.items(), key=lambda x: x[1]['stability'])[0]
    best_stability = results[most_stable]['stability']
    
    # 找出与原始标签最一致的方法
    most_consistent = max(results.items(), key=lambda x: x[1]['consistency'])[0]
    best_consistency = results[most_consistent]['consistency']
    
    if verbose:
        logger.info(f"\n最佳聚类方法(轮廓系数): {best_method}, 轮廓系数: {best_silhouette:.4f}")
        logger.info(f"最稳定的聚类方法: {most_stable}, 稳定性: {best_stability:.4f}")
        logger.info(f"最一致的聚类方法: {most_consistent}, 一致性: {best_consistency:.4f}")
    
    # 绘制比较图表
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    methods_list = list(results.keys())
    silhouettes = [results[m]['silhouette'] for m in methods_list]
    stabilities = [results[m]['stability'] for m in methods_list]
    consistencies = [results[m]['consistency'] for m in methods_list]
    
    # 绘制比较图
    x = np.arange(len(methods_list))
    width = 0.25
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width, silhouettes, width, label='Silhouette Score')
    plt.bar(x, stabilities, width, label='Stability Score')
    plt.bar(x + width, consistencies, width, label='Consistency Score')
    
    plt.xticks(x, [m.capitalize() for m in methods_list])
    plt.ylabel('Score')
    plt.title(f'Clustering Methods Comparison (n_clusters={n_clusters})')
    plt.legend()
    plt.grid(axis='y')
    
    # 绘制排名
    plt.subplot(2, 1, 2)
    # 计算排名 (1是最好的)
    silhouette_ranks = np.argsort(np.argsort(silhouettes)[::-1]) + 1
    stability_ranks = np.argsort(np.argsort(stabilities)[::-1]) + 1
    consistency_ranks = np.argsort(np.argsort(consistencies)[::-1]) + 1
    
    # 计算平均排名
    avg_ranks = (silhouette_ranks + stability_ranks + consistency_ranks) / 3
    
    plt.bar(x - width, silhouette_ranks, width, label='Silhouette Rank')
    plt.bar(x, stability_ranks, width, label='Stability Rank')
    plt.bar(x + width, consistency_ranks, width, label='Consistency Rank')
    plt.plot(x, avg_ranks, 'ko-', linewidth=2, label='Average Rank')
    
    plt.xticks(x, [m.capitalize() for m in methods_list])
    plt.ylabel('Rank (lower is better)')
    plt.title('Ranking of Clustering Methods')
    plt.legend()
    plt.grid(axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, f"clustering_methods_comparison_n{n_clusters}.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"聚类方法比较图表已保存至: {save_path}")
    
    plt.close()
    
    # 添加整体评估
    results['summary'] = {
        'best_method': best_method,
        'best_silhouette': best_silhouette,
        'most_stable': most_stable,
        'best_stability': best_stability,
        'most_consistent': most_consistent,
        'best_consistency': best_consistency,
        'average_ranks': {m: avg_ranks[i] for i, m in enumerate(methods_list)}
    }
    
    return results

def evaluate_dim_reduction_impact(data, original_labels, method='kmeans', n_clusters=DEFAULT_NUM_BIG_CLASSES, 
                                verbose=True):
    """
    评估降维对聚类效果的影响
    
    参数:
        data: 输入数据
        original_labels: 原始标签
        method: 聚类方法
        n_clusters: 聚类数量
        verbose: 是否打印详细信息
        
    返回:
        results: 评估结果字典
    """
    if verbose:
        logger.info(f"评估降维对 {method} 聚类方法的影响...")
    
    # 准备结果存储
    results = {}
    
    # 尝试不同的PCA维度
    dims = [5, 10, 20, 30, 50, 100, data.shape[1]]  # 最后一个是原始维度
    
    for dim in dims:
        # 超过原始维度的情况
        if dim > data.shape[1]:
            continue
        
        if verbose:
            logger.info(f"\n评估维度 {dim}...")
        
        try:
            # 应用PCA降维（对于原始维度，跳过此步骤）
            if dim < data.shape[1]:
                pca = PCA(n_components=dim)
                reduced_data = pca.fit_transform(data)
                
                if verbose:
                    explained_var = sum(pca.explained_variance_ratio_)
                    logger.info(f"  PCA解释方差: {explained_var:.4f}")
            else:
                reduced_data = data
                if verbose:
                    logger.info("  使用原始维度 (不降维)")
            
            # 创建并训练聚类模型
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(reduced_data)
            elif method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                         affinity='nearest_neighbors', 
                                         n_neighbors=min(30, reduced_data.shape[0]//100))
                labels = model.fit_predict(reduced_data)
            elif method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(reduced_data)
            else:
                if verbose:
                    logger.error(f"不支持的聚类方法: {method}")
                return None
            
            # 计算内部评估指标
            sil = silhouette_score(reduced_data, labels)
            cal = calinski_harabasz_score(reduced_data, labels)
            dav = davies_bouldin_score(reduced_data, labels)
            
            # 评估与原始标签的一致性
            consistency, _ = visualize_cluster_vs_labels(
                labels, original_labels, f"{method}_dim{dim}",
                verbose=False, plot=False
            )
            
            if verbose:
                logger.info(f"  轮廓系数: {sil:.4f}")
                logger.info(f"  Calinski-Harabasz指数: {cal:.1f}")
                logger.info(f"  Davies-Bouldin指数: {dav:.4f}")
                logger.info(f"  与原始标签的一致性: {consistency:.4f}")
            
            # 存储结果
            results[dim] = {
                'silhouette': sil,
                'calinski': cal,
                'davies': dav,
                'consistency': consistency,
                'labels': labels
            }
        
        except Exception as e:
            if verbose:
                logger.error(f"  评估维度 {dim} 时出错: {e}")
    
    # 找出最佳维度（根据轮廓系数）
    best_dim = max(results.items(), key=lambda x: x[1]['silhouette'])[0]
    best_silhouette = results[best_dim]['silhouette']
    
    # 找出与原始标签最一致的维度
    most_consistent_dim = max(results.items(), key=lambda x: x[1]['consistency'])[0]
    best_consistency = results[most_consistent_dim]['consistency']
    
    if verbose:
        logger.info(f"\n最佳降维维度(轮廓系数): {best_dim}, 轮廓系数: {best_silhouette:.4f}")
        logger.info(f"最一致的降维维度: {most_consistent_dim}, 一致性: {best_consistency:.4f}")
    
    # 绘制比较图表
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    dims_list = sorted(results.keys())
    silhouettes = [results[d]['silhouette'] for d in dims_list]
    consistencies = [results[d]['consistency'] for d in dims_list]
    
    # 绘制比较图
    plt.subplot(2, 1, 1)
    plt.plot(dims_list, silhouettes, 'o-', label='Silhouette Score')
    plt.axvline(x=best_dim, color='r', linestyle='--', label=f'Best dim={best_dim}')
    plt.title(f'Impact of Dimensionality Reduction on {method.capitalize()} Clustering')
    plt.ylabel('Silhouette Score')
    plt.xscale('log')  # 对数刻度更容易看出低维度的差异
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(dims_list, consistencies, 'o-', label='Consistency Score')
    plt.axvline(x=most_consistent_dim, color='g', linestyle='--', 
               label=f'Best dim={most_consistent_dim}')
    plt.title('Consistency with Original Labels')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Consistency Score')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, f"{method}_dim_reduction_impact.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"降维影响图表已保存至: {save_path}")
    
    plt.close()
    
    # 添加整体评估
    results['summary'] = {
        'best_dim': best_dim,
        'best_silhouette': best_silhouette,
        'most_consistent_dim': most_consistent_dim,
        'best_consistency': best_consistency
    }
    
    return results

def main():
    """主执行函数"""
    # 记录开始时间
    start_time = datetime.now()
    
    # 输出配置信息
    log_section(logger, "聚类实验")
    
    # 第1步：加载数据
    logger.info("加载数据...")
    dataset = load_multiclass_data_from_dirs(subset='val')
    
    # 获取验证集数据
    data = dataset['val_samples']
    labels = dataset['val_labels']
    
    # 获取大类标签
    fine_to_big, big_to_fine, big_class_names = define_big_classes()
    big_labels = map_to_big_classes(labels, fine_to_big)
    
    # 第2步：预处理特征
    logger.info("预处理特征...")
    processed_groups, _ = preprocess_feature_groups(
        data, big_labels, apply_normalization=True, normalization_method='robust',
        apply_pca_dict={k: False for k in FEATURE_GROUPS},  # 不使用PCA，在实验中手动控制
        apply_feature_selection=True, verbose=True, plot=False
    )
    
    # 第3步：评估聚类参数
    log_section(logger, "评估聚类参数")
    
    # 选择主要特征组进行实验
    feature_group = 'all_features' if 'all_features' in processed_groups else list(processed_groups.keys())[0]
    logger.info(f"使用 {feature_group} 特征组进行实验")
    data_for_clustering = processed_groups[feature_group]
    
    # 评估KMeans
    kmeans_param_results = evaluate_clustering_parameters(
        data_for_clustering, big_labels, cluster_method='kmeans',
        min_clusters=2, max_clusters=12
    )
    
    # 评估Spectral
    spectral_param_results = evaluate_clustering_parameters(
        data_for_clustering, big_labels, cluster_method='spectral',
        min_clusters=2, max_clusters=12
    )
    
    # 评估Agglomerative
    agg_param_results = evaluate_clustering_parameters(
        data_for_clustering, big_labels, cluster_method='agglomerative',
        min_clusters=2, max_clusters=12
    )
    
    # 第4步：比较聚类方法
    log_section(logger, "比较聚类方法")
    
    # 使用默认的大类数量
    comparison_results_default = compare_clustering_methods(
        data_for_clustering, big_labels, 
        methods=['kmeans', 'spectral', 'agglomerative'],
        n_clusters=DEFAULT_NUM_BIG_CLASSES
    )
    
    # 使用最佳聚类数量（基于KMeans的轮廓系数）
    best_n = kmeans_param_results['best_n_clusters']
    comparison_results_best = compare_clustering_methods(
        data_for_clustering, big_labels, 
        methods=['kmeans', 'spectral', 'agglomerative'],
        n_clusters=best_n
    )
    
    # 第5步：评估降维影响
    log_section(logger, "评估降维影响")
    
    # 评估降维对KMeans的影响
    dim_impact_kmeans = evaluate_dim_reduction_impact(
        data, big_labels, method='kmeans',
        n_clusters=DEFAULT_NUM_BIG_CLASSES
    )
    
    # 评估降维对Spectral的影响
    dim_impact_spectral = evaluate_dim_reduction_impact(
        data, big_labels, method='spectral',
        n_clusters=DEFAULT_NUM_BIG_CLASSES
    )
    
    # 保存结果
    save_results({
        'kmeans_param_results': kmeans_param_results,
        'spectral_param_results': spectral_param_results,
        'agg_param_results': agg_param_results,
        'comparison_results_default': comparison_results_default,
        'comparison_results_best': comparison_results_best,
        'dim_impact_kmeans': dim_impact_kmeans,
        'dim_impact_spectral': dim_impact_spectral
    }, "clustering_experiment", format='pkl')
    
    # 记录结束时间
    end_time = datetime.now()
    log_execution_time(logger, start_time, end_time, "聚类实验总执行时间")
    
    return {
        'kmeans_param_results': kmeans_param_results,
        'spectral_param_results': spectral_param_results,
        'agg_param_results': agg_param_results,
        'comparison_results_default': comparison_results_default,
        'comparison_results_best': comparison_results_best,
        'dim_impact_kmeans': dim_impact_kmeans,
        'dim_impact_spectral': dim_impact_spectral
    }

if __name__ == "__main__":
    main()