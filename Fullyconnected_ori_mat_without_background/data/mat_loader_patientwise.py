#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAT文件数据加载和处理函数 - 支持Patientwise标准化
"""

import os
import h5py
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from utils.label_processing import should_filter_background_samples, process_labels_for_training, get_effective_num_classes


class BrainVoxelMatDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.FloatTensor(x)
        
        # 标签已经在数据加载阶段处理完成，直接使用
        y = self.labels[idx]
        
        # 确保是整数类型
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


def patientwise_standardize(data, patient_idx, epsilon=1e-10):
    """
    对每个patient的体素分别进行标准化
    
    参数:
    data: shape (N, 341) 的数据
    patient_idx: shape (N,) 的patient索引
    epsilon: 避免除零的小值
    
    返回:
    standardized_data: 标准化后的数据，shape不变
    """
    # 确保patient_idx是一维的
    if patient_idx.ndim > 1:
        patient_idx = patient_idx.ravel()
    
    # 确保数据类型是float，避免整数除法问题
    data = data.astype(np.float64)
    standardized_data = np.zeros_like(data)
    
    # 获取所有唯一的patient索引
    unique_patients = np.unique(patient_idx)
    
    print(f"Patientwise标准化 - 数据形状: {data.shape}")
    print(f"Patient索引形状: {patient_idx.shape}")
    print(f"唯一的patient数量: {len(unique_patients)}")
    
    for patient_id in unique_patients:
        # 找到属于当前patient的所有体素
        patient_mask = (patient_idx == patient_id)
        patient_voxels = data[patient_mask]
        
        print(f"  Patient {patient_id}: {patient_voxels.shape[0]} 个体素", end='')
        
        # 计算该patient每个feature的均值和标准差
        patient_mean = np.mean(patient_voxels, axis=0)
        patient_std = np.std(patient_voxels, axis=0)
        
        # 避免除以0
        patient_std[patient_std < epsilon] = 1.0
        
        # 标准化该patient的数据
        standardized_data[patient_mask] = (patient_voxels - patient_mean) / patient_std
        
        # 检查标准化后的统计量
        standardized_patient = standardized_data[patient_mask]
        print(f" → 标准化后均值: {np.abs(np.mean(standardized_patient)).max():.2e}, "
              f"标准差: {np.std(standardized_patient).mean():.2f}")
    
    return standardized_data


def validate_filtered_data(train_data, train_labels, description=""):
    """验证过滤后数据的完整性"""
    print(f"\n=== {description} 数据验证 ===")
    
    if len(train_labels.shape) > 1:
        labels = np.argmax(train_labels, axis=1)
    else:
        labels = train_labels.flatten()
    
    unique_labels = np.unique(labels)
    print(f"标签范围: {np.min(unique_labels)} - {np.max(unique_labels)}")
    print(f"标签数量: {len(unique_labels)}")
    print(f"样本总数: {len(train_data)}")
    
    # 检查是否还有标签0
    if 0 in unique_labels:
        print("  警告: 仍存在背景标签0")
    else:
        print(" 确认: 已成功过滤背景标签")
    
    # 检查标签连续性
    expected_range = set(range(1, 103))  # 1-102
    actual_labels = set(unique_labels)
    missing_labels = expected_range - actual_labels
    if missing_labels:
        print(f"  缺失的标签: {sorted(missing_labels)}")
    
    return True


def process_train38_data_with_patientwise(mat_file_path, config, random_state=666):
    """
    处理TRAIN38.mat数据，支持Patientwise标准化和固定患者验证/测试集
    
    参数:
        mat_file_path: .mat文件路径
        config: 配置字典
        random_state: 随机种子
    
    返回:
        dataset_dict: 包含训练、验证、测试数据的字典
    """
    print(" 处理TRAIN38.mat数据（Patientwise标准化）...")
    
    # 获取配置
    standardization_method = config.get('standardization_method', 'patientwise')
    epsilon = config.get('standardization_epsilon', 1e-10)
    filter_background = should_filter_background_samples(config)
    
    # 获取固定的验证集和测试集患者ID
    val_prob_idx = config.get('val_prob_idx', [20])
    test_prob_idx = config.get('test_prob_idx', [38])
    
    print(f" 配置参数:")
    print(f"  标准化方法: {standardization_method}")
    print(f"  验证集prob_idx: {val_prob_idx}")
    print(f"  测试集prob_idx: {test_prob_idx}")
    print(f"  背景处理模式: {'过滤背景' if filter_background else '保留背景'}")
    
    # 加载数据
    arrays = load_mat_data(mat_file_path)
    train_data = arrays['data']
    train_region = arrays['region']
    prob_idx = arrays['prob_idx'].flatten().astype(int)
    
    print(f"原始数据形状: data={train_data.shape}, region={train_region.shape}, prob_idx={prob_idx.shape}")
    print(f"prob_idx唯一值: {sorted(np.unique(prob_idx))}")
    
    # 根据配置决定是否过滤背景像素
    if filter_background:
        print("\n 过滤背景像素...")
        if len(train_region.shape) > 1 and train_region.shape[1] > 1:
            label_indices = np.argmax(train_region, axis=1)
        else:
            label_indices = train_region.flatten()

        background_mask = (label_indices == 0)
        valid_mask = ~background_mask

        print(f"总样本数: {len(train_data)}")
        print(f"背景像素数: {np.sum(background_mask)} ({np.sum(background_mask)/len(train_data)*100:.2f}%)")
        print(f"有效像素数: {np.sum(valid_mask)} ({np.sum(valid_mask)/len(train_data)*100:.2f}%)")

        # 过滤掉背景像素
        train_data = train_data[valid_mask]
        train_region = train_region[valid_mask]
        prob_idx = prob_idx[valid_mask]

        print(f"过滤后数据形状: data={train_data.shape}, region={train_region.shape}, prob_idx={prob_idx.shape}")
        validate_filtered_data(train_data, train_region, "过滤背景后的完整数据")
    else:
        print("\n 保留背景像素，所有样本参与训练")
        validate_filtered_data(train_data, train_region, "包含背景的完整数据")
    
    # 按固定prob_idx划分数据集
    print(f"\n 按固定prob_idx划分数据集...")
    
    # 创建掩码
    val_mask = np.isin(prob_idx, val_prob_idx)
    test_mask = np.isin(prob_idx, test_prob_idx)
    train_mask = ~(val_mask | test_mask)
    
    # 分离数据集
    X_train = train_data[train_mask]
    y_train = train_region[train_mask]
    train_prob_idx_actual = prob_idx[train_mask]
    
    X_val = train_data[val_mask]
    y_val = train_region[val_mask]
    val_prob_idx_actual = prob_idx[val_mask]
    
    X_test = train_data[test_mask]
    y_test = train_region[test_mask]
    test_prob_idx_actual = prob_idx[test_mask]
    
    print(f"数据集划分结果:")
    print(f"  训练集样本数: {len(X_train)} (患者: {sorted(np.unique(train_prob_idx_actual))})")
    print(f"  验证集样本数: {len(X_val)} (患者: {sorted(np.unique(val_prob_idx_actual))})")
    print(f"  测试集样本数: {len(X_test)} (患者: {sorted(np.unique(test_prob_idx_actual))})")
    
    # 验证数据集之间无交叉
    train_patients = set(np.unique(train_prob_idx_actual))
    val_patients = set(np.unique(val_prob_idx_actual))
    test_patients = set(np.unique(test_prob_idx_actual))
    
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise ValueError(" 数据集交叉检测失败！患者ID有重叠")
    else:
        print(" 数据集交叉检测通过：各数据集患者ID无重叠")
    
    # 根据标准化方法进行处理
    if standardization_method == 'patientwise':
        print("\n 应用Patientwise标准化...")
        
        # 对每个数据集分别进行patientwise标准化
        X_train_scaled = patientwise_standardize(X_train, train_prob_idx_actual, epsilon)
        X_val_scaled = patientwise_standardize(X_val, val_prob_idx_actual, epsilon)
        X_test_scaled = patientwise_standardize(X_test, test_prob_idx_actual, epsilon)
        
        print(" Patientwise标准化完成")
        
    else:  # global standardization
        print("\n 应用全局标准化...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(" 全局标准化完成")
        print(f"  训练集均值范围: [{X_train_scaled.mean(axis=0).min():.4f}, {X_train_scaled.mean(axis=0).max():.4f}]")
        print(f"  训练集标准差范围: [{X_train_scaled.std(axis=0).min():.4f}, {X_train_scaled.std(axis=0).max():.4f}]")
    
    # 处理标签格式
    print(f"\n 处理标签格式...")
    processed_train_labels = process_labels_for_training(y_train, config)
    processed_val_labels = process_labels_for_training(y_val, config)
    processed_test_labels = process_labels_for_training(y_test, config)
    
    # 获取实际类别数量
    effective_num_classes = get_effective_num_classes(config)
    
    print(f"标签处理完成:")
    print(f"  训练集标签范围: {processed_train_labels.min()} - {processed_train_labels.max()}")
    print(f"  验证集标签范围: {processed_val_labels.min()} - {processed_val_labels.max()}")
    print(f"  测试集标签范围: {processed_test_labels.min()} - {processed_test_labels.max()}")
    print(f"  实际类别数量: {effective_num_classes}")
    
    # 创建数据集字典
    dataset_dict = {
        'train_samples': X_train_scaled,
        'train_labels': processed_train_labels,
        'val_samples': X_val_scaled,
        'val_labels': processed_val_labels,
        'test_samples': X_test_scaled,
        'test_labels': processed_test_labels,
        'feature_dim': X_train.shape[1],
        'num_classes': effective_num_classes,
        # 额外信息
        'val_prob_idx_used': val_prob_idx,
        'test_prob_idx_used': test_prob_idx,
        'val_prob_idx_actual': sorted(np.unique(val_prob_idx_actual)),
        'test_prob_idx_actual': sorted(np.unique(test_prob_idx_actual)),
        'train_prob_idx': sorted(np.unique(train_prob_idx_actual)),
        # 标准化信息
        'standardization_method': standardization_method,
        'background_filtered': filter_background,
        'background_config': {
            'filter_background': config.get('filter_background', True),
            'include_background_in_classes': config.get('include_background_in_classes', False),
            'background_label_target': config.get('background_label_target', -1)
        }
    }
    
    # 如果使用全局标准化，保存scaler
    if standardization_method == 'global' and 'scaler' in locals():
        dataset_dict['scaler'] = scaler
    
    print(f"\n 数据处理完成！")
    print(f"  特征维度: {dataset_dict['feature_dim']}")
    print(f"  类别数量: {dataset_dict['num_classes']}")
    print(f"  标准化方法: {standardization_method}")
    print(f"  验证集患者: {dataset_dict['val_prob_idx_actual']}")
    print(f"  测试集患者: {dataset_dict['test_prob_idx_actual']}")
    
    return dataset_dict


def create_dataloaders_from_mat(dataset_dict, batch_size=128, shuffle_train=True):
    """
    从处理好的数据创建PyTorch DataLoader
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


def load_external_mat_data_patientwise(mat_file_path, config):
    """
    加载并处理外部.mat数据文件，支持patientwise标准化
    
    参数:
        mat_file_path: .mat文件路径
        config: 配置字典
    
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
    
    # 获取patient索引
    if 'prob_idx' in arrays:
        prob_idx = arrays['prob_idx'].flatten()
    else:
        # 如果没有prob_idx，假设所有数据来自同一个patient
        print("警告: 没有找到prob_idx，假设所有数据来自同一个patient")
        prob_idx = np.zeros(data.shape[0], dtype=int)
    
    # 根据标准化方法处理
    standardization_method = config.get('standardization_method', 'patientwise')
    
    if standardization_method == 'patientwise':
        # 应用patientwise标准化
        data_scaled = patientwise_standardize(data, prob_idx, config.get('standardization_epsilon', 1e-10))
    else:
        # 使用全局标准化
        if 'scaler' in config:
            data_scaled = config['scaler'].transform(data)
        else:
            print("警告: 未提供scaler，使用原始数据")
            data_scaled = data
    
    # 如果数据文件包含标签，也返回标签
    labels = None
    if 'region' in arrays:
        labels = arrays['region']
    
    return {
        'data': data_scaled,
        'labels': labels,
        'original_data': data,
        'prob_idx': prob_idx
    }


def load_and_process_data_with_patientwise(config, mode='train', model_path=None):
    """
    统一的数据加载与处理函数，支持patientwise标准化
    
    参数:
        config: 配置字典
        mode: 'train'表示训练模式，'eval'表示评估模式
        model_path: 在'eval'模式下，可以提供模型路径
    
    返回:
        dataset_dict: 数据集字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    print(f" 开始加载数据集（模式: {mode}，标准化: {config.get('standardization_method', 'patientwise')}）...")
    
    # 检查配置
    if 'mat_file_path' not in config or not config['mat_file_path']:
        raise ValueError("配置中缺少'mat_file_path'，请在配置中指定TRAIN38.mat文件路径")
    
    # 设置默认的验证集和测试集
    if 'val_prob_idx' not in config:
        config['val_prob_idx'] = [20]
        print(f" 未设置val_prob_idx，使用默认值: {config['val_prob_idx']}")
    
    if 'test_prob_idx' not in config:
        config['test_prob_idx'] = [38]
        print(f" 未设置test_prob_idx，使用默认值: {config['test_prob_idx']}")
    
    # 处理数据
    dataset_dict = process_train38_data_with_patientwise(
        mat_file_path=config['mat_file_path'],
        config=config,
        random_state=config.get('random_seed', 666)
    )
    
    # 创建数据加载器
    shuffle_train = True if mode == 'train' else False
    dataloaders = create_dataloaders_from_mat(
        dataset_dict,
        batch_size=config.get('batch_size', 128),
        shuffle_train=shuffle_train
    )
    
    # 更新配置
    config['feature_dim'] = dataset_dict['feature_dim']
    config['num_class'] = dataset_dict['num_classes']
    
    print(f"\n数据加载完成！")
    print(f"  训练样本: {len(dataset_dict['train_samples'])} (患者数: {len(dataset_dict['train_prob_idx'])})")
    print(f"  验证样本: {len(dataset_dict['val_samples'])} (患者: {dataset_dict['val_prob_idx_actual']})")
    print(f"  测试样本: {len(dataset_dict['test_samples'])} (患者: {dataset_dict['test_prob_idx_actual']})")
    print(f"  特征维度: {dataset_dict['feature_dim']}")
    print(f"  类别数量: {dataset_dict['num_classes']}")
    print(f"  标准化方法: {dataset_dict['standardization_method']}")
    
    return dataset_dict, dataloaders['train'], dataloaders['val'], dataloaders['test']