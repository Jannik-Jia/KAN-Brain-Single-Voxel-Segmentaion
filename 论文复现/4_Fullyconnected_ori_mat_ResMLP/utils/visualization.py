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
    可视化数据集的分布情况
    
    参数:
        dataset_dict: 数据集字典
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 1. 训练、验证、测试集的样本数
    plt.subplot(1, 3, 1)
    datasets = ['Training Set', 'Test Set', 'Validation Set']
    counts = [
        len(dataset_dict['train_labels']),
        len(dataset_dict['test_labels']),
        len(dataset_dict['val_labels'])
    ]
    
    plt.bar(datasets, counts)
    plt.xlabel('Dataset')
    plt.ylabel('Sample Count')
    plt.title('Dataset Size Distribution')
    plt.grid(True, axis='y')
    
    # 2. 类别分布柱状图
    plt.subplot(1, 3, 2)
    # 统计每个类别的样本数
    unique_labels = np.unique(np.concatenate([
        dataset_dict['train_labels'],
        dataset_dict['test_labels'],
        dataset_dict['val_labels']
    ]))
    
    # 取前20个类别
    top_classes = 20
    sorted_labels = sorted(unique_labels)[:top_classes]
    
    class_counts = []
    for label in sorted_labels:
        train_count = np.sum(dataset_dict['train_labels'] == label)
        test_count = np.sum(dataset_dict['test_labels'] == label)
        val_count = np.sum(dataset_dict['val_labels'] == label)
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
        train_count = np.sum(dataset_dict['train_labels'] == label)
        test_count = np.sum(dataset_dict['test_labels'] == label)
        val_count = np.sum(dataset_dict['val_labels'] == label)
        all_counts[label] = train_count + test_count + val_count
    
    sorted_counts = sorted(all_counts.values(), reverse=True)
    plt.plot(sorted_counts, 'bo-')
    plt.xlabel('Class Rank')
    plt.ylabel('Sample Count')
    plt.title('Class Imbalance Distribution')
    plt.grid(True)
    plt.yscale('log')
    
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