#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分类模型模块 - 实现分类模型，特征组合分析，最佳分类策略探索
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from feature_analysis import advanced_feature_reduction, analyze_optimal_clusters, visualize_cluster_vs_labels


def analyze_feature_group_combinations(feature_groups, labels, group_names=None, save_path=None, logger=None):
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
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'feature_combination_analysis.png'))
    plt.close()
    
    return results


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
    from config import RANDOM_SEED
    
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
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'class_separability.png'))
    plt.close()
    
    return separability_scores


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
    from config import RANDOM_SEED
    
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
    from feature_analysis import select_discriminative_features
    
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