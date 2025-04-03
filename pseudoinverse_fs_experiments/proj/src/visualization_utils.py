# src/visualization_utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

def visualize_feature_selection(selector, feature_names=None, save_path=None):
    """
    可视化特征选择结果
    
    参数:
        selector: 已拟合的特征选择器
        feature_names: 特征名称列表
        save_path: 图表保存路径
    """
    if selector.feature_importance is None:
        print("特征选择器尚未拟合或没有特征重要性信息")
        return
    
    # 获取特征重要性和选择掩码并确保为NumPy数组
    from src.gpu_utils import ensure_numpy
    importance = ensure_numpy(selector.get_feature_importance())
    selected = selector.get_support()
    
    # 如果没有提供特征名称，使用索引
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # 确保特征名称长度匹配
    if len(feature_names) != len(importance):
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # 创建一个包含3个子图的图表
    plt.figure(figsize=(18, 12))
    
    # 1. 特征重要性分布图
    plt.subplot(2, 2, 1)
    plt.hist(importance, bins=50, alpha=0.7)
    
    # 转换为NumPy数组以确保兼容性
    threshold = selector.selection_threshold if selector.selection_mode == 'threshold' else \
               importance[np.argsort(importance)[-int(selector.max_features)]]  # 确保索引是整数
               
    plt.axvline(x=threshold, color='r', linestyle='--', 
               label=f"Selection Threshold: {selector.selection_threshold}" if selector.selection_mode == 'threshold' else 
               f"Top {selector.max_features} Features")
    plt.xlabel('Feature Importance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Importance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 所选特征在原始特征空间中的位置
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(importance)), importance, alpha=0.5, 
               c=['blue' if s else 'gray' for s in selected])
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('Selected Features in Feature Space')
    # 如果特征太多，只显示部分刻度
    if len(importance) > 20:
        tick_step = len(importance) // 10
        plt.xticks(range(0, len(importance), tick_step))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 前N个重要特征
    top_n = min(20, np.sum(selected))
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_importance = importance[top_indices]
    top_names = [feature_names[int(i)] for i in top_indices]  # 确保索引是整数
    
    plt.subplot(2, 2, 3)
    plt.barh(range(len(top_names)), top_importance, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 4. 选择频率分布（如果有）
    if hasattr(selector, 'selection_frequency') and selector.selection_frequency is not None:
        selection_frequency = ensure_numpy(selector.selection_frequency)  # 确保为NumPy数组
        plt.subplot(2, 2, 4)
        plt.hist(selection_frequency, bins=10, alpha=0.7)
        plt.xlabel('Selection Frequency')
        plt.ylabel('Number of Features')
        plt.title('Feature Selection Stability')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        # 或者替代图：选择前后的特征数量
        plt.subplot(2, 2, 4)
        plt.bar(['Original', 'Selected'], [len(importance), np.sum(selected)])
        plt.ylabel('Number of Features')
        plt.title('Feature Reduction')
        for i, v in enumerate([len(importance), np.sum(selected)]):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_feature_stability(stability_metrics, save_path=None):
    """
    可视化特征选择稳定性
    
    参数:
        stability_metrics: 包含稳定性指标的字典
        save_path: 图表保存路径
    """
    # 确保转换为NumPy数组
    from src.gpu_utils import ensure_numpy
    jaccard_matrix = ensure_numpy(stability_metrics['jaccard_matrix'])
    selection_frequency = ensure_numpy(stability_metrics['selection_frequency'])
    
    plt.figure(figsize=(15, 6))
    
    # 1. Jaccard相似度热图
    plt.subplot(1, 2, 1)
    sns.heatmap(jaccard_matrix, annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.title('Jaccard Similarity Between Folds')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    
    # 2. 特征选择频率分布
    plt.subplot(1, 2, 2)
    plt.hist(selection_frequency, bins=10, alpha=0.7)
    plt.xlabel('Selection Frequency Across Folds')
    plt.ylabel('Number of Features')
    plt.title('Feature Selection Stability')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_selected_feature_performance(results_before, results_after, save_path=None):
    """
    可视化特征选择前后的性能对比
    
    参数:
        results_before: 特征选择前的评估结果
        results_after: 特征选择后的评估结果
        save_path: 图表保存路径
    """
    # 提取数据
    datasets = list(results_before.keys())
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'kappa']
    metric_names = ['Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', 'Kappa']
    
    plt.figure(figsize=(15, 10))
    
    # 为每个数据集绘制性能对比图
    for i, dataset in enumerate(datasets):
        if dataset in ['train_time', 'evaluation_time']:
            continue
            
        plt.subplot(len(datasets), 1, i+1)
        
        # 准备数据
        before_values = [results_before[dataset][m] for m in metrics]
        after_values = [results_after[dataset][m] for m in metrics]
        
        # 绘制柱状图
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, before_values, width, label='Before Feature Selection')
        plt.bar(x + width/2, after_values, width, label='After Feature Selection')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Performance Comparison on {dataset.capitalize()} Set')
        plt.xticks(x, metric_names)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for j, (before, after) in enumerate(zip(before_values, after_values)):
            plt.text(j - width/2, before + 0.01, f"{before:.3f}", ha='center', va='bottom', fontsize=8)
            plt.text(j + width/2, after + 0.01, f"{after:.3f}", ha='center', va='bottom', fontsize=8)
            
            # 计算变化百分比
            if before > 0:
                change_pct = (after - before) / before * 100
                color = 'green' if change_pct >= 0 else 'red'
                plt.text(j, max(before, after) + 0.05, f"{change_pct:+.1f}%", 
                        ha='center', va='bottom', fontsize=9, color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_feature_importance(model, feature_names=None, top_n=20, save_path=None):
    """
    可视化特征重要性
    
    参数:
        model: 已拟合的伪逆模型
        feature_names: 特征名称列表
        top_n: 要显示的顶部特征数量
        save_path: 图表保存路径
    """
    if model.feature_importance is None:
        print("模型尚未拟合或没有特征重要性信息")
        return
    
    # 获取特征重要性并确保为NumPy数组
    from src.gpu_utils import ensure_numpy
    importance = ensure_numpy(model.feature_importance)
    
    # 如果没有提供特征名称，使用索引
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # 确保特征名称长度匹配
    if len(feature_names) != len(importance):
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # 获取前N个重要特征
    if top_n > len(importance):
        top_n = len(importance)
    
    # 注意这里需要转换为整数
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_importance = importance[top_indices]
    top_names = [feature_names[int(i)] for i in top_indices]  # 确保i是整数
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_names)), top_importance, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance (Mean Absolute Weight)')
    plt.title(f'Top {top_n} Most Important Features')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_weight_distribution(model, save_path=None):
    """
    可视化权重分布
    
    参数:
        model: 已拟合的伪逆模型
        save_path: 图表保存路径
    """
    if model.weights is None:
        print("模型尚未拟合")
        return
    
    # 将权重矩阵展平
    from src.gpu_utils import ensure_numpy

    weights = ensure_numpy(model.weights.flatten())

    
    plt.figure(figsize=(10, 6))
    
    # 绘制权重直方图
    plt.subplot(1, 2, 1)
    plt.hist(weights, bins=50, alpha=0.7)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制权重箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(weights)
    plt.ylabel('Weight Value')
    plt.title('Weight Boxplot')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

