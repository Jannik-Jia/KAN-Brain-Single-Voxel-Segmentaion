#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
降维分析模块，实现各种降维方法和可视化
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FIGURES_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def advanced_feature_reduction(data, method='umap', n_components=2, labels=None, 
                              verbose=True, plot=True, title=None, save_name=None):
    """
    高级特征降维与可视化
    
    参数：
        data: 输入数据
        method: 降维方法，可选 'pca', 'tsne', 'umap', 'mds'
        n_components: 降维后的维度
        labels: 类别标签，用于可视化
        verbose: 是否打印详细信息
        plot: 是否绘制降维结果
        title: 图表标题
        save_name: 保存文件名，不含扩展名
        
    返回：
        reduced_data: 降维后的数据
        reducer: 降维模型
    """
    if verbose:
        logger.info(f"使用 {method.upper()} 方法进行{n_components}维降维...")
    
    # 确保数据是浮点型
    data = data.astype(np.float32)
    
    # 根据指定方法进行降维
    if method.lower() == 'pca':
        if verbose:
            logger.info("应用PCA降维...")
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(data)
        explained_var = reducer.explained_variance_ratio_
        explained_var_str = f"解释方差: {sum(explained_var):.2%}"
        if verbose:
            logger.info(f"PCA解释的总方差: {sum(explained_var):.2%}")
            logger.info(f"各主成分贡献: {explained_var}")
    
    elif method.lower() == 'tsne':
        if verbose:
            logger.info("应用t-SNE降维...")
        # 使用合适的参数，较大数据集可使用Barnes-Hut近似
        n_samples = data.shape[0]
        if n_samples > 10000:
            if verbose:
                logger.info(f"样本数 {n_samples} > 10000, 使用Barnes-Hut近似算法")
            reducer = TSNE(n_components=n_components, 
                         perplexity=min(30, data.shape[0] // 5),
                         n_iter=1000, 
                         random_state=42,
                         method='barnes_hut')
        else:
            reducer = TSNE(n_components=n_components, 
                         perplexity=min(30, data.shape[0] // 5),
                         n_iter=1000, 
                         random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    elif method.lower() == 'umap':
        if verbose:
            logger.info("应用UMAP降维...")
        reducer = umap.UMAP(n_components=n_components,
                          n_neighbors=min(30, data.shape[0] // 5),
                          min_dist=0.1,
                          random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    elif method.lower() == 'mds':
        if verbose:
            logger.info("应用MDS降维...")
        reducer = MDS(n_components=n_components, n_jobs=-1, random_state=42)
        reduced_data = reducer.fit_transform(data)
        explained_var_str = ""
    
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    # 可视化降维结果
    if plot and labels is not None and n_components in [2, 3]:
        visualize_embedding(reduced_data, labels, method, title, explained_var_str, save_name)
    
    return reduced_data, reducer

def visualize_embedding(embedding, labels, method, title=None, explained_var_str="", save_name=None):
    """
    可视化降维嵌入结果
    
    参数:
        embedding: 降维后的数据
        labels: 类别标签
        method: 降维方法名称
        title: 图表标题
        explained_var_str: 解释方差信息字符串
        save_name: 保存文件名(不含扩展名)
    """
    plt.figure(figsize=(12, 10))
    
    # 确定降维的维度
    n_components = embedding.shape[1]
    
    if n_components == 2:
        # 2D可视化
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 为每个类别选择不同的颜色和形状
        cmap = plt.cm.get_cmap('viridis', n_classes)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        
        # 使用不同的颜色和形状绘制每个类别
        for i, label in enumerate(unique_labels):
            mask = labels == label
            marker = markers[i % len(markers)]
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[cmap(i/n_classes)], 
                      marker=marker,
                      alpha=0.7, 
                      label=f'Class {label}',
                      edgecolors='w')
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
    elif n_components == 3:
        # 3D可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 为每个类别选择不同的颜色和形状
        cmap = plt.cm.get_cmap('viridis', n_classes)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        
        # 使用不同的颜色和形状绘制每个类别
        for i, label in enumerate(unique_labels):
            mask = labels == label
            marker = markers[i % len(markers)]
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                     c=[cmap(i/n_classes)], 
                     marker=marker,
                     alpha=0.7, 
                     label=f'Class {label}',
                     edgecolors='w')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    
    # 如果类别数量大于10，不显示图例，因为它会占用太多空间
    if len(unique_labels) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        logger.info(f"类别数量 ({len(unique_labels)}) > 10, 不显示图例")
    
    # 设置标题
    if title:
        plt.title(f'{title} ({method.upper()} projection) {explained_var_str}')
    else:
        plt.title(f'{method.upper()} projection {explained_var_str}')
    
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    if save_name:
        save_path = os.path.join(FIGURES_DIR, f'{save_name}_{method.lower()}.png')
        plt.savefig(save_path, dpi=300)
        logger.info(f"降维可视化图表已保存至: {save_path}")
    
    plt.close()

def compare_dimensionality_reduction_methods(data, labels, methods=None, n_components=2, 
                                           verbose=True, save_prefix=None):
    """
    比较不同降维方法的效果
    
    参数:
        data: 输入数据
        labels: 类别标签
        methods: 要比较的降维方法列表
        n_components: 降维后的维度
        verbose: 是否打印详细信息
        save_prefix: 保存文件名前缀
        
    返回:
        embeddings: 各方法的降维结果字典
    """
    if methods is None:
        methods = ['pca', 'tsne', 'umap']
    
    embeddings = {}
    
    for method in methods:
        logger.info(f"\n使用 {method.upper()} 进行降维...")
        save_name = f"{save_prefix}_{method}" if save_prefix else method
        embedding, _ = advanced_feature_reduction(
            data, method=method, n_components=n_components, labels=labels,
            verbose=verbose, plot=True, title=f"{save_prefix if save_prefix else 'Data'} Embedding",
            save_name=save_name
        )
        embeddings[method] = embedding
    
    return embeddings

def visualize_feature_space_by_groups(feature_groups, labels, n_components=2, 
                                     methods=None, verbose=True):
    """
    按特征组可视化特征空间
    
    参数:
        feature_groups: 特征组字典
        labels: 类别标签
        n_components: 降维后的维度
        methods: 降维方法列表
        verbose: 是否打印详细信息
    
    返回:
        embeddings: 各特征组各方法的降维结果嵌套字典
    """
    if methods is None:
        methods = ['pca', 'umap']  # 默认只使用PCA和UMAP，因为t-SNE和MDS计算成本高
    
    embeddings = {}
    
    for group_name, group_data in feature_groups.items():
        if verbose:
            logger.info(f"\n分析 {group_name} 特征组的特征空间...")
        
        group_embeddings = compare_dimensionality_reduction_methods(
            group_data, labels, methods=methods, n_components=n_components,
            verbose=verbose, save_prefix=group_name
        )
        
        embeddings[group_name] = group_embeddings
    
    # 如果有'all_features'组，也可以将所有特征组合并进行一次分析
    if 'all_features' not in feature_groups and len(feature_groups) > 1:
        if verbose:
            logger.info("\n分析所有特征组合并后的特征空间...")
        
        # 合并所有特征
        all_features = []
        for group_data in feature_groups.values():
            all_features.append(group_data)
        combined_data = np.hstack(all_features)
        
        combined_embeddings = compare_dimensionality_reduction_methods(
            combined_data, labels, methods=methods, n_components=n_components,
            verbose=verbose, save_prefix='combined_features'
        )
        
        embeddings['combined_features'] = combined_embeddings
    
    return embeddings

if __name__ == "__main__":
    # 测试降维功能
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_multiclass_data_from_dirs, define_big_classes, map_to_big_classes
    from data.preprocessing import preprocess_feature_groups
    
    # 加载验证集数据
    dataset = load_multiclass_data_from_dirs(subset='val')
    val_data, val_labels = dataset['val_samples'], dataset['val_labels']
    
    # 获取大类标签
    fine_to_big, big_to_fine, big_class_names = define_big_classes()
    big_labels = map_to_big_classes(val_labels, fine_to_big)
    
    # 预处理特征
    processed_groups, _ = preprocess_feature_groups(val_data, big_labels)
    
    # 测试降维和可视化
    visualize_feature_space_by_groups(processed_groups, big_labels)
    
    print("降维分析测试完成")