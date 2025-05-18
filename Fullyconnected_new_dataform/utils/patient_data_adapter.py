#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于患者ID的数据加载适配器
集成了标准化处理并支持保存/加载标准化参数
"""

import os
import numpy as np
import torch
import pickle
import datetime
import shutil
from torch.utils.data import Dataset, DataLoader
import sys
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BrainVoxel38PatientLoader import create_data_loaders as patient_create_data_loaders


class IndexLabelAdapter(Dataset):
    """将one-hot标签转换为类别索引的数据集适配器"""
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
        if isinstance(one_hot_label, torch.Tensor):
            label_idx = torch.argmax(one_hot_label).long()
        else:
            label_idx = np.argmax(one_hot_label)
            label_idx = torch.tensor(label_idx, dtype=torch.long)
        
        return features, label_idx


class StandardizedDataset(Dataset):
    """应用标准化的数据集"""
    def __init__(self, features, labels, transform=None):
        """
        初始化数据集
        
        参数:
            features: 特征数据
            labels: 标签数据
            transform: 转换函数（可选）
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        参数:
            idx: 索引
            
        返回:
            x: 特征
            y: 标签
        """
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


class PatientDataLoaderWrapper:
    """包装原始DataLoader，在迭代过程中转换标签格式"""
    def __init__(self, dataloader):
        """
        初始化包装器
        
        参数:
            dataloader: 原始数据加载器对象
        """
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        # 添加这一行来传递batch_size属性
        self.batch_size = dataloader.batch_size
    
    def __len__(self):
        """返回数据加载器长度"""
        return len(self.dataloader)
    
    def __iter__(self):
        """迭代器，转换批次标签格式"""
        for features, one_hot_labels in self.dataloader:
            # 将one-hot批次标签转换为类别索引
            label_indices = torch.argmax(one_hot_labels, dim=1).long()
            yield features, label_indices


def extract_raw_data(data_loader):
    """
    从数据加载器中提取原始数据
    
    参数:
        data_loader: 数据加载器
        
    返回:
        features: 特征数据
        labels: 标签数据
    """
    features = []
    labels = []
    
    for inputs, lbls in data_loader:
        features.append(inputs.numpy())
        labels.append(lbls.numpy())
    
    features = np.vstack(features)
    labels = np.vstack(labels)
    
    return features, labels


def load_patient_based_data(base_dir, batch_size=32, test_patient_id=38, seed=42,
                           train_patient_ids=None, valid_patient_ids=None, test_patient_ids=None,
                           apply_normalization=True, scaler_path=None, save_scaler=True,
                           save_dir="./results", config=None):
    """
    加载基于患者ID的数据，并将其格式适配到现有框架，支持标准化处理
    
    参数:
        base_dir: 重组数据的基础目录
        batch_size: 批次大小
        test_patient_id: 当未指定测试集患者ID时，必须放入测试集的患者ID
        seed: 随机种子
        train_patient_ids: 可选，指定训练集的患者ID列表
        valid_patient_ids: 可选，指定验证集的患者ID列表
        test_patient_ids: 可选，指定测试集的患者ID列表
        apply_normalization: 是否应用标准化
        scaler_path: 预先训练好的标准化器路径（如果提供，将直接加载而不是重新拟合）
        save_scaler: 是否保存拟合的标准化器
        save_dir: 保存标准化器的目录
        config: 配置字典（可选）
        
    返回:
        dataset_dict: 包含数据集信息的字典
        train_loader: 训练数据加载器（已适配格式）
        val_loader: 验证数据加载器（已适配格式）
        test_loader: 测试数据加载器（已适配格式）
    """
    print("使用基于患者ID的数据加载...")
    
    # 如果提供了config，从中读取参数
    if config is not None:
        apply_normalization = config.get('norm', apply_normalization)
        scaler_path = config.get('scaler_path', scaler_path)
        save_scaler = config.get('save_scaler', save_scaler)
        save_dir = config.get('save_dir', save_dir)
    
    # 调用原始的create_data_loaders函数获取原始数据加载器
    train_loader_orig, valid_loader_orig, test_loader_orig = patient_create_data_loaders(
        base_dir=base_dir,
        batch_size=batch_size,
        test_patient_id=test_patient_id,
        seed=seed,
        train_patient_ids=train_patient_ids,
        valid_patient_ids=valid_patient_ids,
        test_patient_ids=test_patient_ids
    )
    
    # 提取原始特征和标签
    print("提取原始数据...")
    train_features_raw, train_labels_raw = extract_raw_data(train_loader_orig)
    valid_features_raw, valid_labels_raw = extract_raw_data(valid_loader_orig)
    test_features_raw, test_labels_raw = extract_raw_data(test_loader_orig)
    
    # 获取特征维度
    feature_dim = train_features_raw.shape[1]
    print(f"特征维度: {feature_dim}")
    
    # 标准化处理
    normalization_params = None
    scaled_scaler_path = None
    
    if apply_normalization:
        print("应用标准化处理...")
        
        if scaler_path and os.path.exists(scaler_path):
            # 加载预训练的标准化器
            print(f"加载预训练的标准化器: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            scaled_scaler_path = scaler_path
        else:
            # 在训练集上拟合新的标准化器
            print("在训练集上拟合标准化器...")
            scaler = StandardScaler()
            scaler.fit(train_features_raw)
            
            # 保存标准化器
            if save_scaler:
                # 创建保存目录
                scaler_dir = os.path.join(save_dir, 'scalers')
                os.makedirs(scaler_dir, exist_ok=True)
                
                # 生成文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = config.get('experiment_name', 'exp') if config else 'exp'
                scaler_filename = os.path.join(scaler_dir, f"scaler_{timestamp}_{experiment_name}.pkl")
                
                # 保存标准化器
                with open(scaler_filename, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"标准化器已保存至: {scaler_filename}")
                scaled_scaler_path = scaler_filename
                
                # 如果提供了config，更新标准化器路径
                if config is not None:
                    config['scaler_path'] = scaler_filename
        
        # 应用标准化
        print("转换数据集...")
        train_features = scaler.transform(train_features_raw)
        valid_features = scaler.transform(valid_features_raw)
        test_features = scaler.transform(test_features_raw)
        
        # 保存标准化参数
        normalization_params = {
            'mean': scaler.mean_.tolist(),
            'std': np.sqrt(scaler.var_).tolist(),
            'scaler_path': scaled_scaler_path
        }
        
        # 打印标准化的效果
        print("标准化前后的数据统计:")
        print(f"训练集 - 原始: 均值 = {np.mean(train_features_raw):.4f}, 标准差 = {np.std(train_features_raw):.4f}")
        print(f"训练集 - 标准化后: 均值 = {np.mean(train_features):.4f}, 标准差 = {np.std(train_features):.4f}")
    else:
        # 不应用标准化，使用原始特征
        print("不应用标准化，使用原始数据...")
        train_features = train_features_raw
        valid_features = valid_features_raw
        test_features = test_features_raw
    
    # 创建使用标准化数据的数据集
    train_dataset = StandardizedDataset(train_features, train_labels_raw)
    valid_dataset = StandardizedDataset(valid_features, valid_labels_raw)
    test_dataset = StandardizedDataset(test_features, test_labels_raw)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 转换为索引标签适配器
    train_loader = PatientDataLoaderWrapper(train_loader)
    val_loader = PatientDataLoaderWrapper(val_loader)
    test_loader = PatientDataLoaderWrapper(test_loader)
    
    # 创建与原有框架兼容的数据集字典
    dataset_dict = {
        'feature_dim': feature_dim,           # 特征维度
        'train_samples': train_features,      # 提供标准化后的样本
        'train_labels': np.argmax(train_labels_raw, axis=1),   # 转换为索引标签
        'test_samples': test_features,
        'test_labels': np.argmax(test_labels_raw, axis=1),
        'val_samples': valid_features,
        'val_labels': np.argmax(valid_labels_raw, axis=1),
        'normalization_params': normalization_params,  # 添加标准化参数
        'valid_labels': list(range(102)),     # 假设有102个类别
        'scaler_path': scaled_scaler_path     # 标准化器路径
    }
    
    print(f"数据加载完成!")
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