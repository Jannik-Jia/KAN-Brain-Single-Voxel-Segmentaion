#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
降维分析模块
用于MRI数据的降维可视化和分析
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dim_reduction')

# 检测是否可以使用GPU
try:
    import cupy as cp
    import cudf
    import cuml
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold import UMAP as cuUMAP
    HAS_GPU = True
    logger.info("GPU加速可用")
except ImportError:
    HAS_GPU = False
    logger.warning("未检测到GPU加速库，将使用CPU进行计算")
    try:
        import umap
    except ImportError:
        logger.warning("UMAP库未安装，将无法使用UMAP降维")

def reduce_dimensions(data, method='pca', n_components=2, use_gpu=False, random_state=42):
    """
    使用指定方法对数据进行降维
    
    参数:
        data: 输入数据
        method: 降维方法 ('pca', 'tsne', 'umap', 'isomap')
        n_components: 降维后的维度
        use_gpu: 是否使用GPU加速
        random_state: 随机种子
    
    返回:
        reduced_data: 降维后的数据
        model: 降维模型
    """
    start_time = time.time()
    logger.info(f"使用{method.upper()}方法降至{n_components}维...")
    
    try:
        # 根据方法选择实现
        if method.lower() == 'pca':
            if use_gpu and HAS_GPU:
                # GPU版PCA
                model = cuPCA(n_components=n_components)
                reduced_data = model.fit_transform(data)
                # 转回CPU
                reduced_data = reduced_data.get() if hasattr(reduced_data, 'get') else reduced_data
                
                # 获取解释方差比
                explained_variance = model.explained_variance_ratio_.get() if hasattr(model.explained_variance_ratio_, 'get') else model.explained_variance_ratio_
                logger.info(f"  解释方差比: {sum(explained_variance):.4f}")
            else:
                # CPU版PCA
                model = PCA(n_components=n_components, random_state=random_state)
                reduced_data = model.fit_transform(data)
                logger.info(f"  解释方差比: {sum(model.explained_variance_ratio_):.4f}")
        
        elif method.lower() == 'tsne':
            if use_gpu and HAS_GPU:
                # GPU版t-SNE
                model = cuTSNE(n_components=n_components, perplexity=min(30, data.shape[0]//5))
                reduced_data = model.fit_transform(data)
                # 转回CPU
                reduced_data = reduced_data.get() if hasattr(reduced_data, 'get') else reduced_data
            else:
                # CPU版t-SNE
                model = TSNE(n_components=n_components, 
                            perplexity=min(30, data.shape[0]//5), 
                            random_state=random_state, 
                            n_jobs=-1)
                reduced_data = model.fit_transform(data)
        
        elif method.lower() == 'umap':
            if use_gpu and HAS_GPU:
                # GPU版UMAP
                model = cuUMAP(n_components=n_components, n_neighbors=min(15, data.shape[0]//5))
                reduced_data = model.fit_transform(data)
                # 转回CPU
                reduced_data = reduced_data.get() if hasattr(reduced_data, 'get') else reduced_data
            else:
                # CPU版UMAP
                try:
                    model = umap.UMAP(n_components=n_components, 
                                    n_neighbors=min(15, data.shape[0]//5), 
                                    random_state=random_state)
                    reduced_data = model.fit_transform(data)
                except NameError:
                    logger.error("UMAP未安装，无法使用此方法")
                    return None, None
        
        elif method.lower() == 'isomap':
            # Isomap目前没有GPU实现
            model = Isomap(n_components=n_components, n_neighbors=min(15, data.shape[0]//5))
            reduced_data = model.fit_transform(data)
        
        else:
            logger.error(f"不支持的降维方法: {method}")
            return None, None
        
        elapsed = time.time() - start_time
        logger.info(f"  降维完成，耗时: {elapsed:.2f}秒")
        logger.info(f"  原始维度: {data.shape[1]}，降维后维度: {reduced_data.shape[1]}")
        
        return reduced_data, model
    
    except Exception as e:
        logger.error(f"降维失败: {str(e)}")
        return None, None

def visualize_reduced_data(reduced_data, labels, method='pca', class_names=None, 
                         title=None, save_path=None):
    """
    可视化降维后的数据
    
    参数:
        reduced_data: 降维后的数据
        labels: 类别标签
        method: 使用的降维方法
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径
    """
    if reduced_data is None or reduced_data.shape[1] not in [2, 3]:
        logger.error(f"不支持的维度: {reduced_data.shape[1] if reduced_data is not None else None}")
        return
    
    # 确保labels与数据长度匹配
    if len(labels) > reduced_data.shape[0]:
        labels = labels[:reduced_data.shape[0]]
    elif len(labels) < reduced_data.shape[0]:
        reduced_data = reduced_data[:len(labels)]
    
    # 2D或3D可视化
    if reduced_data.shape[1] == 2:
        plt.figure(figsize=(12, 10))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 如果类别太多，可能需要采样
        if n_classes > 20:
            logger.warning(f"类别数量({n_classes})太多，可能导致图形拥挤")
            
            # 选择出现次数最多的20个类别
            label_counts = [np.sum(labels == label) for label in unique_labels]
            top_indices = np.argsort(label_counts)[-20:]
            top_labels = unique_labels[top_indices]
            
            mask = np.isin(labels, top_labels)
            reduced_data = reduced_data[mask]
            labels = labels[mask]
            unique_labels = top_labels
            n_classes = len(unique_labels)
            
            logger.info(f"筛选后只显示出现次数最多的20个类别")
        
        # 使用不同颜色和形状
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        
        # 绘制散点图
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.sum(mask) == 0:
                continue
                
            marker = markers[i % len(markers)]
            label_name = class_names[label] if class_names is not None and label < len(class_names) else f"Class {label}"
            plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                       c=[colors[i]], marker=marker, alpha=0.7, label=label_name)
        
        # 设置图表属性
        if title:
            plt.title(f"{title} ({method.upper()} visualization)")
        else:
            plt.title(f"{method.upper()} visualization")
            
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # 如果类别太多，使用紧凑图例
        if n_classes > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            plt.legend()
        
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"可视化结果已保存至: {save_path}")
        
        plt.close()
    
    elif reduced_data.shape[1] == 3:
        # 3D可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 如果类别太多，可能需要采样
        if n_classes > 20:
            logger.warning(f"类别数量({n_classes})太多，可能导致图形拥挤")
            
            # 选择出现次数最多的20个类别
            label_counts = [np.sum(labels == label) for label in unique_labels]
            top_indices = np.argsort(label_counts)[-20:]
            top_labels = unique_labels[top_indices]
            
            mask = np.isin(labels, top_labels)
            reduced_data = reduced_data[mask]
            labels = labels[mask]
            unique_labels = top_labels
            n_classes = len(unique_labels)
            
            logger.info(f"筛选后只显示出现次数最多的20个类别")
        
        # 使用不同颜色和形状
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        
        # 绘制散点图
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.sum(mask) == 0:
                continue
                
            marker = markers[i % len(markers)]
            label_name = class_names[label] if class_names is not None and label < len(class_names) else f"Class {label}"
            ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2],
                      c=[colors[i]], marker=marker, alpha=0.7, label=label_name)
        
        # 设置图表属性
        if title:
            ax.set_title(f"{title} ({method.upper()} visualization)")
        else:
            ax.set_title(f"{method.upper()} visualization")
            
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        
        # 如果类别太多，使用紧凑图例
        if n_classes > 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            ax.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"可视化结果已保存至: {save_path}")
        
        plt.close()

def perform_comprehensive_reduction(data, labels, feature_groups=None, 
                                  methods=None, n_components=2, use_gpu=False, 
                                  class_names=None, sample_ratio=1.0,
                                  save_dir=None):
    """
    对特征组执行全面的降维分析
    
    参数:
        data: 输入数据
        labels: 类别标签
        feature_groups: 特征组字典
        methods: 降维方法列表
        n_components: 降维后的维度
        use_gpu: 是否使用GPU加速
        class_names: 类别名称列表
        sample_ratio: 采样比例，用于减少计算量
        save_dir: 结果保存目录
    
    返回:
        results_dict: 降维结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if methods is None:
        methods = ['pca', 'tsne', 'umap', 'isomap']
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 如果数据量很大，采样以加快计算
    if sample_ratio < 1.0:
        n_samples = int(data.shape[0] * sample_ratio)
        logger.info(f"从{data.shape[0]}个样本中采样{n_samples}个样本进行降维分析")
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
        sampled_data = data[indices]
        sampled_labels = labels[indices]
    else:
        sampled_data = data
        sampled_labels = labels
    
    results_dict = {}
    
    # 对每个特征组执行降维
    for group_name, indices in tqdm(feature_groups.items(), desc="特征组降维分析"):
        logger.info(f"分析特征组: {group_name} ({len(indices)}个特征)")
        
        # 提取特征组数据
        group_data = sampled_data[:, indices]
        
        # 对数据进行标准化，使不同特征具有相同的尺度
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(group_data)
        
        group_results = {}
        
        # 对每种方法执行降维和可视化
        for method in methods:
            logger.info(f"  使用{method}方法进行降维")
            
            reduced_data, model = reduce_dimensions(
                scaled_data, method=method, n_components=n_components, 
                use_gpu=use_gpu, random_state=42
            )
            
            if reduced_data is not None:
                # 存储结果
                group_results[method] = {
                    'data': reduced_data,
                    'model': model
                }
                
                # 可视化结果
                if save_dir:
                    save_path = os.path.join(save_dir, f"{group_name}_{method}_{n_components}d.png")
                    visualize_reduced_data(
                        reduced_data, sampled_labels, method=method, 
                        class_names=class_names, title=f"{group_name}",
                        save_path=save_path
                    )
        
        results_dict[group_name] = group_results
    
    # 比较不同降维方法对特征组的降维效果
    if save_dir and len(methods) > 1 and len(feature_groups) > 1:
        # 比较不同方法对同一特征组的效果
        for group_name, group_results in results_dict.items():
            if len(group_results) < 2:
                continue
                
            # 创建比较图
            plt.figure(figsize=(15 * min(len(methods), 2), 10 * ((len(methods) + 1) // 2)))
            
            for i, (method, result) in enumerate(group_results.items()):
                reduced_data = result['data']
                if reduced_data.shape[1] != 2:
                    continue
                    
                plt.subplot(1, len(methods), i + 1)
                
                # 获取唯一标签
                unique_labels = np.unique(sampled_labels)
                n_classes = len(unique_labels)
                
                # 使用不同颜色
                colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
                
                # 绘制散点图（简化，不使用形状区分以提高可读性）
                for j, label in enumerate(unique_labels):
                    mask = sampled_labels == label
                    plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                               c=[colors[j]], alpha=0.7, s=20)
                
                plt.title(f"{method.upper()}")
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
            
            plt.suptitle(f"{group_name} Feature Group - Different Dimension Reduction Methods")
            plt.tight_layout()
            
            # 保存比较图
            save_path = os.path.join(save_dir, f"{group_name}_methods_comparison.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
        
        # 比较不同特征组在同一降维方法下的效果
        for method in methods:
            method_results = {}
            for group_name, group_results in results_dict.items():
                if method in group_results:
                    method_results[group_name] = group_results[method]['data']
            
            if len(method_results) < 2:
                continue
                
            # 创建比较图
            plt.figure(figsize=(15 * min(len(method_results), 2), 10 * ((len(method_results) + 1) // 2)))
            
            for i, (group_name, reduced_data) in enumerate(method_results.items()):
                if reduced_data.shape[1] != 2:
                    continue
                    
                plt.subplot(1, len(method_results), i + 1)
                
                # 获取唯一标签
                unique_labels = np.unique(sampled_labels)
                n_classes = len(unique_labels)
                
                # 使用不同颜色
                colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
                
                # 绘制散点图（简化，不使用形状区分以提高可读性）
                for j, label in enumerate(unique_labels):
                    mask = sampled_labels == label
                    plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                               c=[colors[j]], alpha=0.7, s=20)
                
                plt.title(f"{group_name}")
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
            
            plt.suptitle(f"{method.upper()} - Different Feature Groups")
            plt.tight_layout()
            
            # 保存比较图
            save_path = os.path.join(save_dir, f"{method}_groups_comparison.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
    
    return results_dict

def extract_pca_components(data, feature_groups=None, n_components=10, use_gpu=False, save_dir=None):
    """
    提取每个特征组的主成分
    
    参数:
        data: 输入数据
        feature_groups: 特征组字典
        n_components: 要提取的主成分数量
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
    
    返回:
        pca_dict: PCA结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pca_dict = {}
    
    for group_name, indices in tqdm(feature_groups.items(), desc="PCA特征提取"):
        logger.info(f"分析特征组: {group_name} ({len(indices)}个特征)")
        
        # 提取特征组数据
        group_data = data[:, indices]
        
        # 对数据进行标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(group_data)
        
        # 确定实际主成分数量
        actual_n = min(n_components, min(scaled_data.shape[0], scaled_data.shape[1]))
        logger.info(f"  提取{actual_n}个主成分")
        
        # 执行PCA
        if use_gpu and HAS_GPU:
            try:
                pca = cuPCA(n_components=actual_n)
                components = pca.fit_transform(scaled_data)
                explained_variance_ratio = pca.explained_variance_ratio_.get() if hasattr(pca.explained_variance_ratio_, 'get') else pca.explained_variance_ratio_
                
                # 转回CPU
                components = components.get() if hasattr(components, 'get') else components
                
                # 获取组件矩阵
                components_matrix = pca.components_.get() if hasattr(pca.components_, 'get') else pca.components_
            except Exception as e:
                logger.error(f"GPU PCA失败: {str(e)}")
                pca = PCA(n_components=actual_n)
                components = pca.fit_transform(scaled_data)
                explained_variance_ratio = pca.explained_variance_ratio_
                components_matrix = pca.components_
        else:
            pca = PCA(n_components=actual_n)
            components = pca.fit_transform(scaled_data)
            explained_variance_ratio = pca.explained_variance_ratio_
            components_matrix = pca.components_
        
        # 计算累积解释方差
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        logger.info(f"  第一主成分解释方差: {explained_variance_ratio[0]:.4f}")
        logger.info(f"  前{actual_n}个主成分累积解释方差: {cumulative_variance[-1]:.4f}")
        
        # 保存结果
        pca_dict[group_name] = {
            'components': components,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'components_matrix': components_matrix,
            'feature_indices': indices,
            'scaler': scaler
        }
        
        # 可视化解释方差
        if save_dir:
            plt.figure(figsize=(12, 5))
            
            # 绘制解释方差
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title(f'{group_name} Explained Variance')
            plt.xticks(range(1, len(explained_variance_ratio) + 1, max(1, actual_n // 10)))
            plt.grid(True)
            
            # 绘制累积解释方差
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
            
            # 添加参考线
            thresholds = [0.8, 0.9, 0.95, 0.99]
            for threshold in thresholds:
                if threshold <= cumulative_variance[-1]:
                    # 找到第一个超过阈值的主成分数
                    n_comp = np.argmax(cumulative_variance >= threshold) + 1
                    plt.axhline(y=threshold, color='r', linestyle='--', 
                               alpha=0.5, label=f'{threshold*100:.0f}% ({n_comp} components)')
            
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title(f'{group_name} Cumulative Variance')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_pca_variance.png"), dpi=300)
            plt.close()
            
            # 可视化前两个主成分的成分矩阵（特征权重）
            if components_matrix.shape[0] >= 2:
                plt.figure(figsize=(14, 6))
                
                # 选择要显示的特征数量
                n_features = min(30, len(indices))
                feature_indices = np.arange(len(indices))
                
                if len(indices) > n_features:
                    # 选择前15个最重要的特征
                    weights = np.abs(components_matrix[0, :]) + np.abs(components_matrix[1, :])
                    top_indices = np.argsort(weights)[-n_features:]
                    feature_indices = feature_indices[top_indices]
                
                # 第一主成分
                plt.subplot(1, 2, 1)
                plt.bar(range(len(feature_indices)), components_matrix[0, feature_indices])
                plt.xlabel('Feature Index')
                plt.ylabel('Weight')
                plt.title(f'{group_name} First Principal Component')
                plt.xticks(range(len(feature_indices)), [indices[i] for i in feature_indices], rotation=90)
                plt.grid(True)
                
                # 第二主成分
                plt.subplot(1, 2, 2)
                plt.bar(range(len(feature_indices)), components_matrix[1, feature_indices])
                plt.xlabel('Feature Index')
                plt.ylabel('Weight')
                plt.title(f'{group_name} Second Principal Component')
                plt.xticks(range(len(feature_indices)), [indices[i] for i in feature_indices], rotation=90)
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{group_name}_pca_components.png"), dpi=300)
                plt.close()
                
                # 保存前5个主成分的特征权重
                n_save_components = min(5, components_matrix.shape[0])
                weights_df = pd.DataFrame({
                    'feature_idx': indices,
                    **{f'PC{i+1}': components_matrix[i, :] for i in range(n_save_components)}
                })
                weights_df.to_csv(os.path.join(save_dir, f"{group_name}_pca_weights.csv"), index=False)
    
    return pca_dict

if __name__ == "__main__":
    # 测试降维分析功能
    print("降维分析模块测试")
    
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.randn(1000, 100)
    test_labels = np.random.randint(0, 5, 1000)
    
    test_groups = {
        'group1': list(range(0, 30)),
        'group2': list(range(30, 70)),
        'group3': list(range(70, 100))
    }
    
    # 创建测试输出目录
    test_save_dir = "test_results/dim_reduction"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # 测试降维
    reduced = perform_comprehensive_reduction(
        test_data, test_labels, test_groups, 
        methods=['pca', 'tsne', 'isomap'], 
        use_gpu=False, save_dir=test_save_dir
    )
    
    # 测试PCA组件提取
    pca_results = extract_pca_components(
        test_data, test_groups, n_components=10, 
        use_gpu=False, save_dir=test_save_dir
    )
    
    print("测试完成")