#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征预处理模块，负责特征标准化、降维和特征选择
"""

import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FEATURE_GROUPS, PCA_CONFIG, FEATURE_SELECTION, FIGURES_DIR, NORMALIZATION_METHOD
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def normalize_features(data, method='robust', verbose=True):
    """
    特征标准化
    
    参数:
        data: 输入数据
        method: 标准化方法，可选 'standard', 'robust', 或 'none'
        verbose: 是否打印详细信息
    
    返回:
        normalized_data: 标准化后的数据
        scaler: 标准化器，用于后续转换
    """
    if method.lower() == 'none':
        if verbose:
            logger.info("跳过特征标准化")
        return data, None
    
    if verbose:
        logger.info(f"使用 {method} 方法进行特征标准化")
    
    if method.lower() == 'standard':
        scaler = StandardScaler()
    elif method.lower() == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    normalized_data = scaler.fit_transform(data)
    
    if verbose:
        logger.info(f"标准化前数据范围: {np.min(data)} 至 {np.max(data)}")
        logger.info(f"标准化后数据范围: {np.min(normalized_data)} 至 {np.max(normalized_data)}")
    
    return normalized_data, scaler

def apply_pca(data, n_components=None, variance_threshold=0.95, verbose=True, plot=False, group_name=None):
    """
    应用PCA降维
    
    参数:
        data: 输入数据
        n_components: PCA组件数量，如果为None则使用方差阈值自动确定
        variance_threshold: 方差阈值，当n_components为None时使用
        verbose: 是否打印详细信息
        plot: 是否绘制方差解释图
        group_name: 特征组名称，用于绘图标题和文件名
    
    返回:
        pca_data: PCA降维后的数据
        pca_model: PCA模型，用于后续转换
        n_components: 使用的组件数量
    """
    if verbose:
        logger.info("应用PCA降维")
    
    # 如果n_components未指定，使用方差阈值自动确定
    if n_components is None:
        # 先拟合一个完整的PCA模型
        full_pca = PCA()
        full_pca.fit(data)
        
        # 计算累积方差比例
        cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
        
        # 找到满足方差阈值的最小组件数
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        if verbose:
            logger.info(f"根据方差阈值 {variance_threshold} 自动确定PCA组件数: {n_components}")
    else:
        # 确保组件数不超过特征数量或样本数量
        n_components = min(n_components, data.shape[1], data.shape[0])
        if verbose:
            logger.info(f"使用指定的PCA组件数: {n_components}")
    
    # 创建并拟合PCA模型
    pca_model = PCA(n_components=n_components)
    pca_data = pca_model.fit_transform(data)
    
    if verbose:
        explained_variance = pca_model.explained_variance_ratio_.sum()
        logger.info(f"PCA降维后解释的方差比例: {explained_variance:.4f}")
        logger.info(f"PCA降维前维度: {data.shape[1]}, 降维后维度: {pca_data.shape[1]}")
    
    # 绘制方差解释图
    if plot:
        plt.figure(figsize=(10, 6))
        
        # 绘制各主成分解释的方差
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
                pca_model.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f'Explained Variance by Component{" - " + group_name if group_name else ""}')
        plt.grid(True)
        
        # 绘制累积方差
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(pca_model.explained_variance_ratio_) + 1), 
                np.cumsum(pca_model.explained_variance_ratio_), marker='o')
        plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({variance_threshold})')
        plt.axvline(x=n_components, color='g', linestyle='--', 
                   label=f'Selected Components ({n_components})')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'Cumulative Explained Variance{" - " + group_name if group_name else ""}')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if group_name:
            save_path = os.path.join(FIGURES_DIR, f'pca_variance_{group_name}.png')
            plt.savefig(save_path, dpi=300)
            if verbose:
                logger.info(f"PCA方差图表已保存至: {save_path}")
        
        plt.close()
    
    return pca_data, pca_model, n_components

def select_discriminative_features(data, labels, group_name, k=20, method='f_classif', verbose=True, plot=True):
    """
    选择最具区分性的特征
    
    参数：
        data: 输入数据特征
        labels: 类别标签
        group_name: 特征组名称
        k: 选择的顶部特征数量
        method: 特征选择方法，可选 'f_classif', 'mutual_info', 'chi2'
        verbose: 是否打印信息
        plot: 是否绘制重要性分布图
        
    返回：
        selected_features: 选择后的特征
        feature_indices: 选择的特征索引
        feature_scores: 特征重要性分数
    """
    # 确保k不超过特征数量
    k = min(k, data.shape[1])
    
    if verbose:
        logger.info(f"为 {group_name} 特征组选择 {k} 个最具区分性的特征，使用 {method} 方法")
    
    # 选择评分函数
    if method == 'f_classif':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    elif method == 'chi2':
        # 确保数据非负，chi2需要非负数据
        if np.min(data) < 0:
            if verbose:
                logger.warning("chi2方法需要非负数据，将进行数据平移")
            data = data - np.min(data) + 1e-6
        score_func = chi2
    else:
        raise ValueError(f"不支持的特征选择方法: {method}")
    
    # 使用F统计量计算特征重要性
    selector = SelectKBest(score_func, k=k)
    selected_features = selector.fit_transform(data, labels)
    feature_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    if verbose:
        logger.info(f"\n{group_name} 特征组选择结果:")
        logger.info(f"原始特征维度: {data.shape[1]}")
        logger.info(f"选择后特征维度: {selected_features.shape[1]}")
        logger.info(f"均值分数: {np.mean(feature_scores):.2f}")
        logger.info(f"最大分数: {np.max(feature_scores):.2f}")
        
        # 打印前10个最重要的特征索引和得分
        sorted_indices = np.argsort(feature_scores)[::-1]
        logger.info("\n前10个最重要的特征:")
        for i, idx in enumerate(sorted_indices[:10]):
            logger.info(f"  特征 {idx}: 分数 = {feature_scores[idx]:.2f}")
    
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Left plot: Importance distribution of all features
        plt.subplot(1, 2, 1)
        plt.bar(range(len(feature_scores)), feature_scores[sorted_indices])
        plt.title(f'{group_name} Feature Importance (All)')
        plt.xlabel('Feature Rank')
        plt.ylabel('Score')
        plt.yscale('log')  # Logarithmic scale
        plt.grid(True)
        
        # Right plot: Importance of selected features
        plt.subplot(1, 2, 2)
        selected_scores = feature_scores[feature_indices]
        sorted_selected = np.argsort(selected_scores)[::-1]
        plt.bar(range(len(selected_scores)), selected_scores[sorted_selected])
        plt.title(f'{group_name} Feature Importance (Selected)')
        plt.xlabel('Feature Rank')
        plt.ylabel('Score')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(FIGURES_DIR, f'{group_name}_feature_importance.png')
        plt.savefig(save_path, dpi=300)
        if verbose:
            logger.info(f"特征重要性图表已保存至: {save_path}")
        
        plt.close()

    return selected_features, feature_indices, feature_scores

def preprocess_feature_groups(data, labels, feature_groups=None, apply_normalization=True, 
                             normalization_method=NORMALIZATION_METHOD, apply_pca_dict=None, 
                             apply_feature_selection=True, n_features=None, verbose=True, plot=True):
    """
    处理特征组
    
    参数：
        data: 输入数据
        labels: 类别标签
        feature_groups: 特征索引字典，默认使用config中的FEATURE_GROUPS
        apply_normalization: 是否应用标准化
        normalization_method: 标准化方法
        apply_pca_dict: 每个特征组是否应用PCA的字典，默认使用config中的PCA_CONFIG
        apply_feature_selection: 是否应用特征选择
        n_features: 各特征组选择的特征数量字典
        verbose: 是否打印详细信息
        plot: 是否绘制图表
        
    返回：
        processed_groups: 处理后的特征组
        preprocessing_info: 预处理信息字典，包含标准化器、PCA模型等
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS
    
    if apply_pca_dict is None:
        apply_pca_dict = {group: PCA_CONFIG[group]['apply'] for group in feature_groups}
    
    if n_features is None and apply_feature_selection:
        n_features = FEATURE_SELECTION['n_features']
    
    processed_groups = {}
    preprocessing_info = {
        'scalers': {},
        'pca_models': {},
        'feature_indices': {},
        'feature_scores': {}
    }
    
    for group_name, indices in tqdm(feature_groups.items(), desc="处理特征组"):
        if verbose:
            logger.info(f"\n处理 {group_name} 特征组...")
        
        # 提取特征
        group_data = data[:, indices]
        
        # 标准化
        if apply_normalization:
            normalized_data, scaler = normalize_features(
                group_data, method=normalization_method, verbose=verbose
            )
            preprocessing_info['scalers'][group_name] = scaler
        else:
            normalized_data = group_data
            preprocessing_info['scalers'][group_name] = None
        
        # 应用PCA
        if apply_pca_dict.get(group_name, False):
            n_components = PCA_CONFIG[group_name]['n_components'] if group_name in PCA_CONFIG else None
            pca_data, pca_model, used_components = apply_pca(
                normalized_data, n_components=n_components, verbose=verbose, 
                plot=plot, group_name=group_name
            )
            preprocessing_info['pca_models'][group_name] = pca_model
            current_data = pca_data
        else:
            current_data = normalized_data
            preprocessing_info['pca_models'][group_name] = None
        
        # 特征选择
        if apply_feature_selection:
            group_n_features = n_features[group_name] if group_name in n_features else min(20, current_data.shape[1])
            selected_data, indices, scores = select_discriminative_features(
                current_data, labels, group_name, k=group_n_features, 
                method=FEATURE_SELECTION['method'], verbose=verbose, plot=plot
            )
            preprocessing_info['feature_indices'][group_name] = indices
            preprocessing_info['feature_scores'][group_name] = scores
            processed_groups[group_name] = selected_data
        else:
            processed_groups[group_name] = current_data
            preprocessing_info['feature_indices'][group_name] = None
            preprocessing_info['feature_scores'][group_name] = None
    
    # 打印处理后的维度
    if verbose:
        logger.info("\n处理后的特征组维度:")
        for group_name, group_data in processed_groups.items():
            logger.info(f"  {group_name} 特征组: {group_data.shape}")
    
    return processed_groups, preprocessing_info

if __name__ == "__main__":
    # 测试特征预处理
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_multiclass_data_from_dirs
    
    # 加载验证集数据
    dataset = load_multiclass_data_from_dirs(subset='val')
    val_data, val_labels = dataset['val_samples'], dataset['val_labels']
    
    # 测试特征组预处理
    processed_groups, preprocessing_info = preprocess_feature_groups(
        val_data, val_labels, verbose=True, plot=True
    )
    
    print("特征预处理完成")