#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化函数
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def visualize_dataset_distribution(dataset_dict, save_path=None):
    """
    可视化数据集的分布情况，支持one-hot和索引形式的标签
    
    参数:
        dataset_dict: 数据集字典
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 函数来处理标签，将one-hot转换为索引
    def process_labels(labels):
        if labels is None:
            return np.array([])
        elif len(labels.shape) > 1 and labels.shape[1] > 1:
            # one-hot编码，转换为索引
            return np.argmax(labels, axis=1)
        else:
            # 已经是索引形式
            return labels
    
    # 处理标签
    train_labels = process_labels(dataset_dict.get('train_labels'))
    test_labels = process_labels(dataset_dict.get('test_labels'))
    val_labels = process_labels(dataset_dict.get('val_labels'))
    
    # 1. 训练、验证、测试集的样本数
    plt.subplot(1, 3, 1)
    datasets = ['Training Set', 'Test Set', 'Validation Set']
    counts = [
        len(dataset_dict.get('train_samples', [])),
        len(dataset_dict.get('test_samples', [])),
        len(dataset_dict.get('val_samples', []))
    ]
    
    plt.bar(datasets, counts)
    plt.xlabel('Dataset')
    plt.ylabel('Sample Count')
    plt.title('Dataset Size Distribution')
    plt.grid(True, axis='y')
    
    # 2. 类别分布柱状图
    plt.subplot(1, 3, 2)
    # 统计每个类别的样本数
    all_labels = np.concatenate([
        train_labels,
        test_labels,
        val_labels
    ]) if len(train_labels) > 0 and len(test_labels) > 0 and len(val_labels) > 0 else np.array([])
    
    if len(all_labels) > 0:
        unique_labels = np.unique(all_labels)
        
        # 取前20个类别
        top_classes = min(20, len(unique_labels))
        sorted_labels = sorted(unique_labels)[:top_classes]
        
        class_counts = []
        for label in sorted_labels:
            train_count = np.sum(train_labels == label) if len(train_labels) > 0 else 0
            test_count = np.sum(test_labels == label) if len(test_labels) > 0 else 0
            val_count = np.sum(val_labels == label) if len(val_labels) > 0 else 0
            class_counts.append([train_count, test_count, val_count])
        
        class_counts = np.array(class_counts).T
        
        x = np.arange(len(sorted_labels))
        width = 0.25
        
        plt.bar(x - width, class_counts[0], width, label='Train')
        plt.bar(x, class_counts[1], width, label='Test')
        plt.bar(x + width, class_counts[2], width, label='Val')
        
        plt.xlabel('Class')
        plt.ylabel('Sample Count')
        plt.title(f'Top {top_classes} Classes Sample Distribution')
        plt.xticks(x, sorted_labels)
        plt.legend()
        plt.grid(True, axis='y')
        
        # 3. 不平衡程度分析
        plt.subplot(1, 3, 3)
        all_counts = {}
        for label in unique_labels:
            train_count = np.sum(train_labels == label) if len(train_labels) > 0 else 0
            test_count = np.sum(test_labels == label) if len(test_labels) > 0 else 0
            val_count = np.sum(val_labels == label) if len(val_labels) > 0 else 0
            all_counts[label] = train_count + test_count + val_count
        
        sorted_counts = sorted(all_counts.values(), reverse=True)
        plt.plot(sorted_counts, 'bo-')
        plt.xlabel('Class Rank')
        plt.ylabel('Sample Count')
        plt.title('Class Imbalance Distribution')
        plt.grid(True)
        plt.yscale('log')
    else:
        # 如果没有有效的标签数据
        plt.subplot(1, 3, 2)
        plt.text(0.5, 0.5, 'No valid label data', horizontalalignment='center', verticalalignment='center')
        plt.title('Class Distribution')
        
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, 'No valid label data', horizontalalignment='center', verticalalignment='center')
        plt.title('Class Imbalance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_training_curves(training_results, save_path=None):
    """
    可视化训练曲线
    
    参数:
        training_results: 训练结果字典
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(training_results['loss_list'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(training_results['acc_list'], label='Training')
    plt.plot(training_results['val_epoch_list'], training_results['val_acc_list'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1和Kappa曲线
    plt.subplot(1, 3, 3)
    plt.plot(training_results['val_epoch_list'], training_results['val_f1_macro_list'], 'g-', label='F1 Macro')
    plt.plot(training_results['val_epoch_list'], training_results['val_kappa_list'], 'r--', label='Kappa')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_confusion_matrix(conf_matrix, save_path=None, log_scale=True):
    """
    可视化混淆矩阵
    
    参数:
        conf_matrix: 混淆矩阵
        save_path: 保存路径
        log_scale: 是否使用对数缩放
    """
    plt.figure(figsize=(10, 8))
    
    if log_scale:
        # 使用对数缩放
        conf_mat_log = np.log1p(conf_matrix)  # log(1+x)
        mask = conf_matrix == 0
        sns.heatmap(conf_mat_log, annot=False, fmt='d', cmap='Blues', mask=mask)
        plt.title('Confusion Matrix (log scale)')
    else:
        sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()