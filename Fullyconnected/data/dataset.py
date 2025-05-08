#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集类和数据加载函数实现
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from .samplers import BrainVoxelSampler

class BrainVoxelDataset(Dataset):
    """
    脑体素数据集类，多分类版本
    """
    def __init__(self, data, labels):
        """
        初始化数据集
        
        参数:
            data: 特征数据，形状为(n_samples, feature_dim)
            labels: 标签数据，形状为(n_samples,)
        """
        super(BrainVoxelDataset, self).__init__()
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.FloatTensor(x)
        
        y = self.labels[idx]
        # 对于多分类问题，标签从0开始到101
        if y == 0:  # 背景像素
            y = -1  # 设置为忽略索引
        else:
            y = y - 1  # 将1-102转为0-101
            
        y = torch.LongTensor([int(y)])[0]
        return x, y

def apply_pca(X, num_components=15, norm=True, pca_model=None):
    """
    对数据进行PCA降维和标准化处理
    
    参数:
        X (ndarray): 需要降维的数据
        num_components (int): 保留的主成分数量，0表示不进行PCA
        norm (bool): 是否进行标准化处理
        pca_model: 预先训练好的PCA模型，None表示需要重新拟合
    
    返回:
        new_X: 处理后的数据
        num_components: 最终的特征维度
        pca_model: 使用或训练的PCA模型
    """
    if num_components == 0:
        # 不进行PCA，但可能进行标准化
        if norm:
            # 对每个特征进行标准化
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            # 避免除以0
            std[std == 0] = 1
            new_X = (X - mean) / std
        else:
            new_X = X.copy()
        return new_X, X.shape[1], None
    else:
        # 进行PCA降维
        if pca_model is None:
            # 如果没有提供PCA模型，则训练一个新的
            pca_model = PCA(n_components=num_components)
            new_X = pca_model.fit_transform(X)
        else:
            # 使用提供的PCA模型转换数据
            new_X = pca_model.transform(X)
        
        # 可选的标准化
        if norm:
            # 对PCA后的特征进行归一化
            new_X = (new_X - np.min(new_X, axis=0)) / (np.max(new_X, axis=0) - np.min(new_X, axis=0) + 1e-10)
        
        return new_X, new_X.shape[1], pca_model

def analyze_pca_variance(X, max_components=None, plot=True, save_path=None):
    """
    分析PCA的方差解释率，找到合适的降维维度
    
    参数:
        X (ndarray): 输入数据
        max_components (int): 最大考虑的主成分数，None表示使用特征维度
        plot (bool): 是否绘制解释方差曲线
        save_path (str): 保存图像的路径，None表示不保存
        
    返回:
        optimal_n_components: 建议的主成分数量
    """
    # 确定最大主成分数
    if max_components is None:
        max_components = min(X.shape[0], X.shape[1])
    else:
        max_components = min(max_components, X.shape[0], X.shape[1])
    
    # 计算所有可能的主成分
    pca = PCA(n_components=max_components)
    pca.fit(X)
    
    # 计算累积解释方差
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 寻找方差解释率达到95%的拐点
    threshold = 0.95
    optimal_n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
    
    # 寻找拐点（斜率变化最大的点）
    gradient = np.gradient(explained_variance_ratio)
    gradient_of_gradient = np.gradient(gradient)
    elbow_index = np.argmax(np.abs(gradient_of_gradient))
    elbow_n_components = elbow_index + 1
    
    if plot:
        plt.figure(figsize=(12, 6))
    
        # Plot Explained Variance Ratio
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(explained_variance_ratio) + 1), 
                 explained_variance_ratio, 'bo-', markersize=4)
        plt.axvline(x=elbow_n_components, color='r', linestyle='--', 
                    label=f'Elbow Point: {elbow_n_components} Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio per Principal Component')
        plt.grid(True)
        plt.legend()
    
        # Plot Cumulative Explained Variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
                 cumulative_variance_ratio, 'ro-', markersize=4)
        plt.axhline(y=threshold, color='g', linestyle='--', 
                    label=f'{threshold*100}% Variance')
        plt.axvline(x=optimal_n_components, color='b', linestyle='--', 
                    label=f'Threshold Components: {optimal_n_components}')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio')
        plt.grid(True)
        plt.legend()
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    print(f"方差拐点对应的主成分数量: {elbow_n_components}")
    print(f"达到{threshold*100}%方差解释率需要的主成分数量: {optimal_n_components}")
    print(f"前{optimal_n_components}个主成分解释了总方差的{cumulative_variance_ratio[optimal_n_components-1]*100:.2f}%")
    
    # 修改为使用95%阈值点
    suggested_components = optimal_n_components  # 使用保留95%信息的维度
    return suggested_components, explained_variance_ratio, cumulative_variance_ratio

def load_multiclass_data(data_dirs, apply_pca_flag=True, n_components=24, norm=True):
    """
    载入所有类别的数据用于多分类训练 - 不进行类别平衡处理
    
    参数:
        data_dirs: 包含训练、测试和验证数据目录的字典
        apply_pca_flag: 是否应用PCA降维
        n_components: PCA保留的主成分数量，0表示自动选择
        norm: 是否进行标准化处理
        
    返回:
        dataset_dict: 包含训练、测试和验证集数据的字典
    """
    # 创建数据采样器
    train_sampler = BrainVoxelSampler(data_dirs['train_dir'])
    test_sampler = BrainVoxelSampler(data_dirs['test_dir'])
    val_sampler = BrainVoxelSampler(data_dirs['val_dir'])
    
    # 收集所有有效标签
    valid_labels = sorted(list(set(
        train_sampler.valid_labels + 
        test_sampler.valid_labels + 
        val_sampler.valid_labels
    )))
    
    print(f"找到 {len(valid_labels)} 个有效标签")
    
    # 收集所有训练集样本和标签
    train_samples = []
    train_labels = []
    
    for label_id in tqdm(valid_labels, desc="加载训练集数据"):
        file_path = train_sampler.get_file_path(label_id)
        if file_path and os.path.exists(file_path):
            samples = np.load(file_path)
            labels = np.ones(len(samples)) * label_id
            train_samples.append(samples)
            train_labels.append(labels)
    
    # 收集所有测试集样本和标签
    test_samples = []
    test_labels = []
    
    for label_id in tqdm(valid_labels, desc="加载测试集数据"):
        file_path = test_sampler.get_file_path(label_id)
        if file_path and os.path.exists(file_path):
            samples = np.load(file_path)
            labels = np.ones(len(samples)) * label_id
            test_samples.append(samples)
            test_labels.append(labels)
    
    # 收集所有验证集样本和标签
    val_samples = []
    val_labels = []
    
    for label_id in tqdm(valid_labels, desc="加载验证集数据"):
        file_path = val_sampler.get_file_path(label_id)
        if file_path and os.path.exists(file_path):
            samples = np.load(file_path)
            labels = np.ones(len(samples)) * label_id
            val_samples.append(samples)
            val_labels.append(labels)
    
    # 合并各自的数据
    train_samples = np.vstack(train_samples) if train_samples else np.array([])
    train_labels = np.concatenate(train_labels) if train_labels else np.array([])
    test_samples = np.vstack(test_samples) if test_samples else np.array([])
    test_labels = np.concatenate(test_labels) if test_labels else np.array([])
    val_samples = np.vstack(val_samples) if val_samples else np.array([])
    val_labels = np.concatenate(val_labels) if val_labels else np.array([])
    
    # 打印数据集统计信息
    print(f"\n{'='*60}\n多分类数据集统计信息\n{'='*60}")
    print(f"训练集: {len(train_labels)} 个样本")
    print(f"测试集: {len(test_labels)} 个样本")
    print(f"验证集: {len(val_labels)} 个样本")
    
    # 应用PCA（如果需要）
    if apply_pca_flag:
        # 合并所有数据进行PCA拟合
        all_samples = np.vstack([train_samples, test_samples, val_samples])
        
        if n_components == 0:
            # 自动选择主成分数量
            n_components, _, _ = analyze_pca_variance(all_samples, plot=True)
        
        # 创建并拟合PCA模型
        pca_model = PCA(n_components=n_components)
        pca_model.fit(all_samples)
        
        # 应用PCA变换
        train_samples = pca_model.transform(train_samples)
        test_samples = pca_model.transform(test_samples)
        val_samples = pca_model.transform(val_samples)
        
        # 应用标准化（如果需要）
        if norm:
            # 基于所有样本计算标准化参数 - 这里修改为使用标准化而非最小-最大归一化
            all_transformed = np.vstack([train_samples, test_samples, val_samples])
            means = np.mean(all_transformed, axis=0)
            stds = np.std(all_transformed, axis=0)
            stds[stds == 0] = 1e-10  # 避免除零
            
            # 应用标准化
            train_samples = (train_samples - means) / stds
            test_samples = (test_samples - means) / stds
            val_samples = (val_samples - means) / stds
        
        feature_dim = train_samples.shape[1]
        print(f"\n特征维度: {feature_dim} (PCA降维后)")
    else:
        feature_dim = train_samples.shape[1]
        print(f"\n特征维度: {feature_dim} (原始特征)")
        pca_model = None
    
    # 类别分布统计
    print("\n类别分布:")
    class_counts = {}
    for label_id in valid_labels:
        train_count = np.sum(train_labels == label_id)
        test_count = np.sum(test_labels == label_id)
        val_count = np.sum(val_labels == label_id)
        total_count = train_count + test_count + val_count
        print(f"标签 {label_id}: 训练集 {train_count}, 测试集 {test_count}, 验证集 {val_count}, 总计 {total_count}")
        class_counts[label_id] = {
            'train': train_count,
            'test': test_count,
            'val': val_count,
            'total': total_count
        }
    
    # 显示某些关键的类别统计
    print("\n类别统计摘要:")
    min_label = min(class_counts.items(), key=lambda x: x[1]['total'])[0]
    max_label = max(class_counts.items(), key=lambda x: x[1]['total'])[0]
    print(f"样本最少的类别: 标签 {min_label}, 共 {class_counts[min_label]['total']} 个样本")
    print(f"样本最多的类别: 标签 {max_label}, 共 {class_counts[max_label]['total']} 个样本")
    print(f"类别不平衡比例: {class_counts[max_label]['total'] / class_counts[min_label]['total']:.2f} : 1")
    
    # 统计小样本类别
    small_classes = [k for k, v in class_counts.items() if v['total'] < 10]
    if small_classes:
        print(f"\n样本数少于10的类别: {len(small_classes)} 个")
        for label in small_classes:
            print(f"标签 {label}: {class_counts[label]['total']} 个样本")
    
    return {
        'train_samples': train_samples,
        'train_labels': train_labels,
        'test_samples': test_samples,
        'test_labels': test_labels,
        'val_samples': val_samples,
        'val_labels': val_labels,
        'feature_dim': feature_dim,
        'pca_model': pca_model,
        'valid_labels': valid_labels,
        'class_counts': class_counts
    }