#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAT文件数据加载和处理函数
"""

import os
import h5py
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class BrainVoxelMatDataset(Dataset):
    """
    基于.mat文件的脑体素数据集类
    """
    def __init__(self, data, labels):
        """
        初始化数据集
        
        参数:
            data: 特征数据，形状为(n_samples, feature_dim)
            labels: 标签数据，形状为(n_samples, num_classes)，one-hot编码或索引
        """
        super(BrainVoxelMatDataset, self).__init__()
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.FloatTensor(x)
        
        # 处理标签 - 根据输入类型判断处理方式
        if len(self.labels.shape) > 1 and self.labels.shape[1] > 1:
            # one-hot编码，转换为索引
            y = np.argmax(self.labels[idx])
        else:
            # 已经是索引形式
            y = self.labels[idx]
            
        y = torch.LongTensor([int(y)])[0]
        return x, y

def load_mat_data(mat_file_path):
    """
    从.mat文件加载数据
    
    参数:
        mat_file_path: .mat文件路径
    
    返回:
        加载的数据字典
    """
    print(f"从 {mat_file_path} 加载数据...")
    arrays = {}
    with h5py.File(mat_file_path, 'r') as f:
        for k, v in f.items():
            arrays[k] = np.array(v)
    
    # 转置数据，确保格式正确
    if 'data' in arrays:
        arrays['data'] = arrays['data'].transpose()
    if 'region' in arrays:
        arrays['region'] = arrays['region'].transpose()
    if 'prob_idx' in arrays:
        arrays['prob_idx'] = arrays['prob_idx'].transpose()
    if 'multidim_data' in arrays:
        arrays['multidim_data'] = arrays['multidim_data'].transpose()
    
    return arrays

def process_train38_data(mat_file_path, test_size=0.01, random_state=666, scaler_save_path=None):
    """
    处理TRAIN38.mat数据，分割为训练集、验证集和测试集
    
    参数:
        mat_file_path: TRAIN38.mat文件路径
        test_size: 测试集比例
        random_state: 随机种子
        scaler_save_path: scaler保存路径，None表示不保存
    
    返回:
        dataset_dict: 包含数据集和处理信息的字典
    """
    print("处理TRAIN38.mat数据...")
    
    # 加载数据
    arrays = load_mat_data(mat_file_path)
    train_data = arrays['data']
    train_region = arrays['region']
    prob_idx = arrays['prob_idx'].flatten()  # 确保是一维数组
    
    # 根据prob_idx分割数据
    train_set_idx = np.where(prob_idx != 38)[0]
    val_set_idx = np.where(prob_idx == 38)[0]
    
    # 提取训练集和验证集
    train_set_data = train_data[train_set_idx, :]
    train_set_region = train_region[train_set_idx, :]
    val_data = train_data[val_set_idx, :]
    val_label = train_region[val_set_idx, :]
    
    print(f"训练集数据形状: {train_set_data.shape}")
    print(f"验证集数据形状: {val_data.shape}")
    
    # 进一步分割训练集，留出一小部分作为测试集
    X_train, X_test, y_train, y_test = train_test_split(
        train_set_data, train_set_region, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 应用StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    val_data_scaled = scaler.transform(val_data)
    
    # 保存scaler
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler已保存到: {scaler_save_path}")
    
    # 创建数据集字典
    dataset_dict = {
        'train_samples': X_train_scaled,
        'train_labels': y_train,
        'test_samples': X_test_scaled,
        'test_labels': y_test,
        'val_samples': val_data_scaled,
        'val_labels': val_label,
        'feature_dim': X_train.shape[1],
        'scaler': scaler,
        'num_classes': y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))
    }
    
    return dataset_dict

def load_external_mat_data(mat_file_path, scaler=None, scaler_path=None):
    """
    加载并处理外部.mat数据文件
    
    参数:
        mat_file_path: .mat文件路径
        scaler: 预先训练好的StandardScaler对象
        scaler_path: scaler保存的路径，如果scaler为None则从此路径加载
    
    返回:
        处理后的数据
    """
    # 加载数据
    arrays = load_mat_data(mat_file_path)
    
    # 判断数据键名
    if 'multidim_data' in arrays:
        data = arrays['multidim_data']
    elif 'data' in arrays:
        data = arrays['data']
    else:
        raise KeyError("未在.mat文件中找到有效的数据键('data'或'multidim_data')")
    
    # 获取scaler
    if scaler is None and scaler_path:
        scaler = joblib.load(scaler_path)
        print(f"从 {scaler_path} 加载scaler")
    
    if scaler:
        # 应用scaler
        data_scaled = scaler.transform(data)
    else:
        data_scaled = data
        print("警告: 未提供scaler，使用原始数据")
    
    # 如果数据文件包含标签，也返回标签
    labels = None
    if 'region' in arrays:
        labels = arrays['region']
    
    return {
        'data': data_scaled,
        'labels': labels,
        'original_data': data
    }

def create_dataloaders_from_mat(dataset_dict, batch_size=128, shuffle_train=True):
    """
    从处理好的数据创建PyTorch DataLoader
    
    参数:
        dataset_dict: 数据集字典，包含train_samples, train_labels等
        batch_size: 批处理大小
        shuffle_train: 是否打乱训练数据
    
    返回:
        包含DataLoader的字典
    """
    # 创建数据集
    train_dataset = BrainVoxelMatDataset(dataset_dict['train_samples'], dataset_dict['train_labels'])
    test_dataset = BrainVoxelMatDataset(dataset_dict['test_samples'], dataset_dict['test_labels'])
    val_dataset = BrainVoxelMatDataset(dataset_dict['val_samples'], dataset_dict['val_labels'])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'test': test_loader,
        'val': val_loader
    }