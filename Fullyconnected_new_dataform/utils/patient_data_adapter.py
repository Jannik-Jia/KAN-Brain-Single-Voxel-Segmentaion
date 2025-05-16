#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于患者ID的数据加载适配器
用于将BrainVoxel38PatientLoader的输出格式适配到现有贝叶斯优化框架
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys

# 添加项目根目录到路径，确保可以导入BrainVoxel38PatientLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BrainVoxel38PatientLoader import create_data_loaders as patient_create_data_loaders

class IndexLabelAdapter(Dataset):
    """
    将one-hot标签转换为类别索引的数据集适配器
    """
    def __init__(self, dataset):
        """
        初始化适配器
        
        参数:
            dataset: 原始数据集对象，具有__getitem__方法，返回(features, one_hot_labels)
        """
        self.dataset = dataset
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        获取数据项并转换标签格式
        
        参数:
            idx: 索引
            
        返回:
            features: 特征向量
            label_idx: 类别索引（转换自one-hot标签）
        """
        features, one_hot_label = self.dataset[idx]
        
        # 将one-hot标签转换为类别索引
        # one_hot_label形状为(102,)，表示102个类别
        if isinstance(one_hot_label, torch.Tensor):
            label_idx = torch.argmax(one_hot_label).long()
        else:
            label_idx = np.argmax(one_hot_label)
            label_idx = torch.tensor(label_idx, dtype=torch.long)
        
        return features, label_idx


class PatientDataLoaderWrapper:
    """
    包装原始DataLoader，在迭代过程中转换标签格式
    """
    def __init__(self, dataloader):
        """
        初始化包装器
        
        参数:
            dataloader: 原始数据加载器对象
        """
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
    
    def __len__(self):
        """返回数据加载器长度"""
        return len(self.dataloader)
    
    def __iter__(self):
        """迭代器，转换批次标签格式"""
        for features, one_hot_labels in self.dataloader:
            # 将one-hot批次标签转换为类别索引
            # one_hot_labels形状为(batch_size, 102)
            label_indices = torch.argmax(one_hot_labels, dim=1).long()
            yield features, label_indices


def load_patient_based_data(base_dir, batch_size=32, test_patient_id=38, seed=42,
                          train_patient_ids=None, valid_patient_ids=None, test_patient_ids=None):
    """
    加载基于患者ID的数据，并将其格式适配到现有框架
    
    参数:
        base_dir: 重组数据的基础目录
        batch_size: 批次大小
        test_patient_id: 当未指定测试集患者ID时，必须放入测试集的患者ID
        seed: 随机种子
        train_patient_ids: 可选，指定训练集的患者ID列表
        valid_patient_ids: 可选，指定验证集的患者ID列表
        test_patient_ids: 可选，指定测试集的患者ID列表
        
    返回:
        dataset_dict: 包含数据集信息的字典
        train_loader: 训练数据加载器（已适配格式）
        val_loader: 验证数据加载器（已适配格式）
        test_loader: 测试数据加载器（已适配格式）
    """
    print("使用基于患者ID的数据加载...")
    
    # 调用原始的create_data_loaders函数
    train_loader_orig, valid_loader_orig, test_loader_orig = patient_create_data_loaders(
        base_dir=base_dir,
        batch_size=batch_size,
        test_patient_id=test_patient_id,
        seed=seed,
        train_patient_ids=train_patient_ids,
        valid_patient_ids=valid_patient_ids,
        test_patient_ids=test_patient_ids
    )
    
    # 创建适配后的数据加载器
    train_loader = PatientDataLoaderWrapper(train_loader_orig)
    val_loader = PatientDataLoaderWrapper(valid_loader_orig)
    test_loader = PatientDataLoaderWrapper(test_loader_orig)
    
    # 获取特征维度
    sample_features, _ = next(iter(train_loader_orig))
    feature_dim = sample_features.shape[1]
    
    # 创建与原有框架兼容的数据集字典
    dataset_dict = {
        'feature_dim': feature_dim,  # 特征维度
        'train_samples': None,  # 这里不提供原始样本数组，因为已经包装在DataLoader中
        'train_labels': None,   # 同上
        'test_samples': None,
        'test_labels': None,
        'val_samples': None,
        'val_labels': None,
        'pca_model': None,      # 没有使用PCA
        'valid_labels': list(range(102)),  # 假设有102个类别
    }
    
    print(f"基于患者ID的数据加载完成!")
    print(f"特征维度: {feature_dim}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    return dataset_dict, train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 设置重组数据的基础目录
    base_dir = '/path/to/reorganized_fold_data'
    
    # 测试数据加载
    dataset_dict, train_loader, val_loader, test_loader = load_patient_based_data(
        base_dir=base_dir,
        batch_size=32,
        test_patient_id=38,
        seed=666
    )
    
    # 检查数据格式
    for features, labels in train_loader:
        print(f"特征形状: {features.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签类型: {labels.dtype}")
        print(f"标签范围: {labels.min().item()} - {labels.max().item()}")
        break