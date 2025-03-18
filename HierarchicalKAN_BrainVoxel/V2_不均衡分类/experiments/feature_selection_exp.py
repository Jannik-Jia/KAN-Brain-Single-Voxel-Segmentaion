#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征选择实验，评估不同特征组和选择方法的效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FIGURES_DIR, RESULTS_DIR, FEATURE_GROUPS, FEATURE_SELECTION
from data.data_loader import load_multiclass_data_from_dirs, define_big_classes, map_to_big_classes
from data.preprocessing import normalize_features, select_discriminative_features
from utils.logging_utils import get_logger, log_section, log_execution_time
from utils.model_utils import save_results

# 获取日志记录器
logger = get_logger("feature_selection_exp")

def compare_feature_selection_methods(data, labels, feature_groups=None, methods=None, 
                                    n_features_range=None, verbose=True):
    """
    比较不同特征选择方法的效果
    
    参数:
        data: 输入数据
        labels: 目标标签
        feature_groups: 特征组字典
        methods: 特征选择方法列表
        n_features_range: 特征数量范围列表
        verbose: 是否打印详细信息
        
    返回:
        results: 实验结果字典
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS
    
    if methods is None:
        methods = ['f_classif', 'mutual_info', 'chi2']
    
    if n_features_range is None:
        n_features_range = [5, 10, 20, 30, 50]
    
    # 对数据进行标准化
    normalized_data, _ = normalize_features(data, method='robust', verbose=verbose)
    
    # 准备结果存储
    results = {}
    
    # 对每个特征组进行实验
    for group_name, indices in feature_groups.items():
        if verbose:
            logger.info(f"\n评估 {group_name} 特征组的特征选择方法...")
        
        # 提取组特征
        group_data = normalized_data[:, indices]
        
        # 记录基线性能（使用全部特征）
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        baseline_scores = cross_val_score(rf, group_data, labels, cv=5, scoring='balanced_accuracy')
        baseline_acc = baseline_scores.mean()
        
        if verbose:
            logger.info(f"  基线准确率(全部特征): {baseline_acc:.4f} ± {baseline_scores.std():.4f}")
        
        group_results = {
            'baseline': {
                'accuracy': baseline_acc,
                'std': baseline_scores.std(),
                'n_features': group_data.shape[1]
            }
        }
        
        # 对每种方法进行评估
        for method in methods:
            if verbose:
                logger.info(f"\n  使用 {method} 方法:")
            
            method_results = {}
            
            # 对每个特征数量进行评估
            for n_features in n_features_range:
                # 确保特征数量不超过组特征总数
                if n_features >= group_data.shape[1]:
                    continue
                
                # 选择特征
                selected_features, _, _ = select_discriminative_features(
                    group_data, labels, group_name, k=n_features, 
                    method=method, verbose=False, plot=False
                )
                
                # 评估性能
                rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                scores = cross_val_score(rf, selected_features, labels, cv=5, scoring='balanced_accuracy')
                
                acc = scores.mean()
                std = scores.std()
                
                if verbose:
                    logger.info(f"    特征数量 {n_features}: {acc:.4f} ± {std:.4f}")
                
                method_results[n_features] = {
                    'accuracy': acc,
                    'std': std,
                    'change': acc - baseline_acc
                }
            
            group_results[method] = method_results
        
        results[group_name] = group_results
    
    # 绘制比较图
    plt.figure(figsize=(15, 12))
    
    # 为每个特征组绘制一个子图
    for i, (group_name, group_results) in enumerate(results.items()):
        plt.subplot(len(results), 1, i+1)
        
        # 对于每种方法绘制一条线
        for method, method_results in group_results.items():
            if method == 'baseline':
                # 绘制基线
                plt.axhline(y=method_results['accuracy'], color='r', linestyle='--', 
                           label=f'Baseline ({method_results["n_features"]} features)')
                continue
            
            # 提取数据
            n_features = [n for n in method_results.keys()]
            accuracies = [method_results[n]['accuracy'] for n in n_features]
            stds = [method_results[n]['std'] for n in n_features]
            
            # 绘制线图
            plt.errorbar(n_features, accuracies, yerr=stds, label=method, marker='o')
        
        plt.title(f'{group_name} Feature Selection Methods Comparison')
        plt.xlabel('Number of Features')
        plt.ylabel('Balanced Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, "feature_selection_comparison.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"特征选择方法比较图表已保存至: {save_path}")
    
    plt.close()
    
    return results

def find_optimal_features(data, labels, feature_groups=None, method='f_classif', verbose=True):
    """
    找出每个特征组的最佳特征数量
    
    参数:
        data: 输入数据
        labels: 目标标签
        feature_groups: 特征组字典
        method: 特征选择方法
        verbose: 是否打印详细信息
        
    返回:
        optimal_features: 最佳特征配置
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS
    
    # 对数据进行标准化
    normalized_data, _ = normalize_features(data, method='robust', verbose=verbose)
    
    # 准备结果存储
    optimal_features = {}
    
    # 对每个特征组进行实验
    for group_name, indices in feature_groups.items():
        if verbose:
            logger.info(f"\n查找 {group_name} 特征组的最佳特征数量...")
        
        # 提取组特征
        group_data = normalized_data[:, indices]
        
        # 尝试不同的特征数量
        best_acc = 0
        best_n = 0
        
        max_features = min(group_data.shape[1], 50)  # 最多尝试50个特征
        feature_range = list(range(5, max_features + 1, 5))  # 步长为5
        
        for n_features in feature_range:
            # 选择特征
            selected_features, selected_indices, _ = select_discriminative_features(
                group_data, labels, group_name, k=n_features, 
                method=method, verbose=False, plot=False
            )
            
            # 评估性能
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            scores = cross_val_score(rf, selected_features, labels, cv=5, scoring='balanced_accuracy')
            
            acc = scores.mean()
            
            if verbose:
                logger.info(f"  特征数量 {n_features}: {acc:.4f} ± {scores.std():.4f}")
            
            # 更新最佳结果
            if acc > best_acc:
                best_acc = acc
                best_n = n_features
                best_indices = selected_indices
        
        if verbose:
            logger.info(f"  最佳特征数量: {best_n}, 准确率: {best_acc:.4f}")
        
        optimal_features[group_name] = {
            'n_features': best_n,
            'accuracy': best_acc,
            'indices': best_indices.tolist() if hasattr(best_indices, 'tolist') else best_indices
        }
    
    # 绘制最佳特征数量图
    plt.figure(figsize=(10, 6))
    
    groups = list(optimal_features.keys())
    n_features = [optimal_features[g]['n_features'] for g in groups]
    accuracies = [optimal_features[g]['accuracy'] for g in groups]
    
    # 绘制条形图
    plt.subplot(2, 1, 1)
    plt.bar(groups, n_features)
    plt.title('Optimal Number of Features by Group')
    plt.ylabel('Number of Features')
    plt.grid(axis='y')
    
    plt.subplot(2, 1, 2)
    plt.bar(groups, accuracies)
    plt.title('Optimal Accuracy by Group')
    plt.ylabel('Balanced Accuracy')
    plt.grid(axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, "optimal_features.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"最佳特征数量图表已保存至: {save_path}")
    
    plt.close()
    
    return optimal_features

def main():
    """主执行函数"""
    # 记录开始时间
    start_time = datetime.now()
    
    # 输出配置信息
    log_section(logger, "特征选择实验")
    
    # 第1步：加载数据
    logger.info("加载数据...")
    dataset = load_multiclass_data_from_dirs(subset='val')
    
    # 获取验证集数据
    data = dataset['val_samples']
    labels = dataset['val_labels']
    
    # 获取大类标签
    fine_to_big, big_to_fine, big_class_names = define_big_classes()
    big_labels = map_to_big_classes(labels, fine_to_big)
    
    # 第2步：比较特征选择方法
    log_section(logger, "比较特征选择方法")
    
    fs_results = compare_feature_selection_methods(
        data, big_labels, methods=['f_classif', 'mutual_info', 'chi2'],
        n_features_range=[5, 10, 15, 20, 30, 40, 50]
    )
    
    # 第3步：找出最佳特征配置
    log_section(logger, "寻找最佳特征配置")
    
    optimal_features = find_optimal_features(
        data, big_labels, method='f_classif'
    )
    
    # 保存结果
    save_results({
        'feature_selection_comparison': fs_results,
        'optimal_features': optimal_features
    }, "feature_selection_experiment", format='json')
    
    # 记录结束时间
    end_time = datetime.now()
    log_execution_time(logger, start_time, end_time, "特征选择实验总执行时间")
    
    return fs_results, optimal_features

if __name__ == "__main__":
    main()