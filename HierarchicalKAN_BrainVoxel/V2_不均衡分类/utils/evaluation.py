#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估工具模块，提供评估指标计算和结果分析功能
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import json
from imblearn.metrics import classification_report_imbalanced
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RESULTS_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def evaluate_clustering_stability(clustering_results):
    """
    评估聚类结果的稳定性
    
    参数:
        clustering_results: 聚类结果字典，包含多个特征组的聚类结果
        
    返回:
        stability_scores: 各特征组的稳定性评分
    """
    stability_scores = {}
    
    for group_name, results in clustering_results.items():
        method_scores = {}
        
        for method, result in results.items():
            # 聚类结果的稳定性由轮廓系数表示
            method_scores[method] = result['silhouette']
        
        # 找出最稳定的方法
        best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        best_score = method_scores[best_method]
        
        stability_scores[group_name] = {
            'scores': method_scores,
            'best_method': best_method,
            'best_score': best_score
        }
    
    return stability_scores

def evaluate_classification_performance(classification_results):
    """
    评估分类性能
    
    参数:
        classification_results: 分类结果字典，包含多个分类器的性能
        
    返回:
        performance_summary: 分类性能总结
    """
    performance_summary = {}
    
    for clf_name, results in classification_results.items():
        # 提取关键指标
        metrics = {
            'balanced_accuracy': results['mean']['balanced_accuracy'],
            'f1_macro': results['mean']['f1_macro'],
            'kappa': results['mean']['kappa']
        }
        
        # 计算性能分数（加权平均）
        weights = {'balanced_accuracy': 0.5, 'f1_macro': 0.3, 'kappa': 0.2}
        weighted_score = sum(metrics[k] * weights[k] for k in metrics)
        
        performance_summary[clf_name] = {
            'metrics': metrics,
            'weighted_score': weighted_score
        }
    
    # 找出最佳分类器
    best_clf = max(performance_summary.items(), key=lambda x: x[1]['weighted_score'])[0]
    best_score = performance_summary[best_clf]['weighted_score']
    
    performance_summary['best_classifier'] = {
        'name': best_clf,
        'score': best_score,
        'metrics': performance_summary[best_clf]['metrics']
    }
    
    return performance_summary

def evaluate_feature_importance(feature_groups, feature_importances, feature_indices=None):
    """
    评估特征重要性
    
    参数:
        feature_groups: 特征组字典，指定每个组包含的特征索引
        feature_importances: 特征重要性数组
        feature_indices: 特征索引，如果进行了特征选择，则提供用于映射回原始特征空间
        
    返回:
        group_importances: 各特征组的重要性评分
    """
    group_importances = {}
    
    # 如果提供了特征索引映射，则将重要性映射回原始特征空间
    if feature_indices is not None:
        # 创建一个全零数组表示原始特征空间的重要性
        original_importances = np.zeros(max(idx for group_indices in feature_indices.values() 
                                         for idx in group_indices) + 1)
        
        # 使用特征索引将重要性值填充回原始特征空间
        for group, indices in feature_indices.items():
            if group in feature_importances:
                for i, idx in enumerate(indices):
                    original_importances[idx] = feature_importances[group][i]
        
        importances = original_importances
    else:
        importances = feature_importances
    
    # 计算每个特征组的总重要性和平均重要性
    for group_name, indices in feature_groups.items():
        # 提取该特征组的重要性值
        group_imp = importances[indices]
        
        # 计算总重要性和平均重要性
        total_importance = np.sum(group_imp)
        mean_importance = np.mean(group_imp)
        
        # 找出组内最重要的特征
        sorted_indices = np.argsort(group_imp)[::-1]
        top_indices = [indices[i] for i in sorted_indices[:min(5, len(sorted_indices))]]
        top_importances = group_imp[sorted_indices[:min(5, len(sorted_indices))]]
        
        group_importances[group_name] = {
            'total_importance': float(total_importance),
            'mean_importance': float(mean_importance),
            'top_indices': top_indices,
            'top_importances': top_importances.tolist()
        }
    
    # 归一化组重要性
    total_importance = sum(info['total_importance'] for info in group_importances.values())
    for group in group_importances:
        group_importances[group]['normalized_importance'] = group_importances[group]['total_importance'] / total_importance
    
    return group_importances

def evaluate_bigclass_mapping(clustering_results, original_labels, big_class_names=None):
    """
    评估聚类结果与预定义大类的映射关系
    
    参数:
        clustering_results: 聚类结果字典
        original_labels: 原始标签
        big_class_names: 大类名称列表
        
    返回:
        mapping_evaluation: 映射评估结果
    """
    mapping_evaluation = {}
    
    for group_name, results in clustering_results.items():
        # 找出最佳聚类方法（基于轮廓系数）
        best_method = max(results.items(), key=lambda x: x[1]['silhouette'])[0]
        best_result = results[best_method]
        
        # 获取聚类标签
        cluster_labels = best_result['labels']
        
        # 计算聚类标签与原始标签的混淆矩阵
        unique_clusters = np.unique(cluster_labels)
        unique_labels = np.unique(original_labels)
        n_clusters = len(unique_clusters)
        n_labels = len(unique_labels)
        
        matrix = np.zeros((n_labels, n_clusters))
        for i, label in enumerate(unique_labels):
            for j, cluster in enumerate(unique_clusters):
                matrix[i, j] = np.sum((original_labels == label) & (cluster_labels == cluster))
        
        # 使用匈牙利算法找到最佳匹配
        from scipy.optimize import linear_sum_assignment
        cost_matrix = -matrix.copy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 计算一致性得分
        matched_samples = sum(matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
        total_samples = matrix.sum()
        consistency_score = matched_samples / total_samples
        
        # 创建映射关系
        mapping = {}
        for i, (r, c) in enumerate(zip(row_ind, col_ind)):
            original_label = int(unique_labels[r])
            cluster_label = int(unique_clusters[c])
            original_name = big_class_names[original_label] if big_class_names and original_label < len(big_class_names) else f"Class {original_label}"
            mapping[original_label] = {
                'cluster': cluster_label,
                'name': original_name,
                'confidence': float(matrix[r, c] / np.sum(matrix[r])) if np.sum(matrix[r]) > 0 else 0
            }
        
        mapping_evaluation[f"{group_name}_{best_method}"] = {
            'consistency_score': float(consistency_score),
            'mapping': mapping,
            'confusion_matrix': matrix.tolist()
        }
    
    # 找出最佳映射（基于一致性得分）
    best_mapping = max(mapping_evaluation.items(), key=lambda x: x[1]['consistency_score'])
    mapping_evaluation['best_mapping'] = {
        'name': best_mapping[0],
        'consistency_score': best_mapping[1]['consistency_score'],
        'mapping': best_mapping[1]['mapping']
    }
    
    return mapping_evaluation

def generate_final_report(clustering_analysis, classification_analysis, feature_importance_analysis, 
                        mapping_evaluation, output_file=None):
    """
    生成最终分析报告
    
    参数:
        clustering_analysis: 聚类分析结果
        classification_analysis: 分类分析结果
        feature_importance_analysis: 特征重要性分析结果
        mapping_evaluation: 映射评估结果
        output_file: 输出文件路径
        
    返回:
        report: 完整报告字典
    """
    report = {
        'clustering_analysis': clustering_analysis,
        'classification_analysis': classification_analysis,
        'feature_importance_analysis': feature_importance_analysis,
        'mapping_evaluation': mapping_evaluation,
        'recommendations': {}
    }
    
    # 生成建议
    recommendations = {}
    
    # 1. 最佳大类数量建议
    best_mapping = mapping_evaluation['best_mapping']
    mapping_source = best_mapping['name']
    consistency_score = best_mapping['consistency_score']
    
    if consistency_score > 0.7:
        recommendations['bigclass_strategy'] = "保持当前预定义的大类划分，与数据自然聚类高度一致。"
    elif consistency_score > 0.5:
        recommendations['bigclass_strategy'] = "当前预定义的大类划分与数据聚类有一定一致性，但可考虑调整部分类别的归属。"
    else:
        recommendations['bigclass_strategy'] = f"当前预定义的大类划分与数据聚类差异较大，建议使用{mapping_source}的聚类结果重新定义大类。"
    
    # 2. 特征组建议
    if feature_importance_analysis:
        sorted_groups = sorted(feature_importance_analysis.items(), 
                             key=lambda x: x[1]['normalized_importance'], reverse=True)
        top_group = sorted_groups[0][0]
        bottom_group = sorted_groups[-1][0]
        
        recommendations['feature_group_strategy'] = f"{top_group}特征组对类别区分最有效，{bottom_group}特征组效果最差。"
        
        # 建议最佳特征组合
        best_combination = classification_analysis.get('best_combination', [])
        if best_combination:
            if len(best_combination) == len(feature_importance_analysis):
                recommendations['feature_combination'] = "组合所有特征组能达到最佳分类效果，建议采用全特征模型。"
            else:
                recommendations['feature_combination'] = f"使用{'+'.join(best_combination)}特征组合能达到最佳分类效果，可以简化模型。"
    
    # 3. 分类器建议
    best_classifier = classification_analysis.get('best_classifier', {}).get('name', "")
    if best_classifier:
        recommendations['classifier'] = f"推荐使用{best_classifier}作为分类器，提供最佳的平衡准确率和宏平均F1分数。"
    
    # 将建议添加到报告中
    report['recommendations'] = recommendations
    
    # 如果指定了输出文件，保存报告
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析报告已保存至: {output_file}")
    
    return report

def visualize_bigclass_distribution(big_labels, big_class_names=None, save_path=None):
    """
    可视化大类分布
    
    参数:
        big_labels: 大类标签
        big_class_names: 大类名称列表
        save_path: 图表保存路径
    """
    # 统计各大类的样本数量
    unique_labels, counts = np.unique(big_labels, return_counts=True)
    
    # 使用类别名称（如果提供）
    if big_class_names is not None:
        labels = [big_class_names[i] if i < len(big_class_names) else f"Class {i}" 
                for i in unique_labels]
    else:
        labels = [f"Class {i}" for i in unique_labels]
    
    # 绘制饼图
    plt.figure(figsize=(12, 8))
    
    # 添加百分比标签
    total = np.sum(counts)
    autopct = lambda p: f'{p:.1f}%\n({int(p*total/100)})'
    
    # 排序大小
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # 绘制饼图
    plt.pie(sorted_counts, labels=sorted_labels, autopct=autopct, 
           startangle=140, shadow=True)
    plt.axis('equal')  # 确保饼图是圆的
    plt.title('Distribution of Major Categories')
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Major category distribution chart saved to: {save_path}")
    
    plt.close()
    
    # 打印统计信息
    logger.info("\nMajor category distribution statistics:")
    for label, count in zip(sorted_labels, sorted_counts):
        percentage = count / total * 100
        logger.info(f"  {label}: {count} samples ({percentage:.1f}%)")
    return unique_labels, counts

def plot_feature_group_importances(group_importances, save_path=None):
    """
    绘制特征组重要性
    
    参数:
        group_importances: 特征组重要性字典
        save_path: 图表保存路径
    """
    if len(group_importances) == 0:
        logger.warning("没有数据可视化！")
        return
    
    # 提取数据
    groups = list(group_importances.keys())
    importances = [group_importances[g]['normalized_importance'] for g in groups]
    
    # 按重要性排序
    sorted_indices = np.argsort(importances)
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    
    y_pos = np.arange(len(sorted_groups))
    plt.barh(y_pos, sorted_importances, align='center', alpha=0.7)
    plt.yticks(y_pos, sorted_groups)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Group Importance')
    plt.grid(axis='x')
    
    # 添加数值标签
    for i, imp in enumerate(sorted_importances):
        plt.text(imp + 0.01, i, f'{imp:.2f}', va='center')
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"特征组重要性图表已保存至: {save_path}")
    
    plt.close()
    
    return sorted_groups, sorted_importances

if __name__ == "__main__":
    # 测试评估工具
    import numpy as np
    
    # 模拟数据
    feature_groups = {
        'group1': [0, 1, 2, 3, 4],
        'group2': [5, 6, 7, 8, 9]
    }
    
    feature_importances = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1])
    
    group_importances = evaluate_feature_importance(feature_groups, feature_importances)
    print(group_importances)
    
    # 绘制特征组重要性
    plot_feature_group_importances(group_importances, save_path=os.path.join(RESULTS_DIR, 'test_group_importance.png'))
    
    print("评估工具测试完成")