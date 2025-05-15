#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载辅助函数，用于加载方法1的数据结构
"""

import numpy as np
import os
import scipy.io
import h5py
import glob
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

class BrainVoxelDataset(Dataset):
    """脑体素数据集类，用于PyTorch数据加载"""
    def __init__(self, features, labels):
        """
        初始化数据集
        
        Args:
            features (np.ndarray): 特征数据，形状为 [n_samples, n_features]
            labels (np.ndarray): 标签数据，形状为 [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_scaler(base_dir):
    """加载保存的StandardScaler"""
    scaler_path = os.path.join(base_dir, 'data_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"无法找到scaler: {scaler_path}")

def load_region_data(directory, region_id, format='mat'):
    """
    加载单个区域的数据
    
    Args:
        directory (str): 数据目录路径
        region_id (int): 区域ID
        format (str): 数据格式，'mat'或'npy'
        
    Returns:
        dict: 包含data, region, prob_idx和可能的age的字典
    """
    result = {}
    
    if format.lower() == 'mat':
        # Mat格式加载
        file_path = os.path.join(directory, 'mat', f"region_{region_id}.mat")
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        try:
            # 尝试使用scipy.io.loadmat加载
            mat_data = scipy.io.loadmat(file_path)
            for key in mat_data:
                if key in ['data', 'region', 'prob_idx', 'age']:
                    result[key] = mat_data[key]
        except:
            # 如果失败，尝试使用h5py加载
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    if key in ['data', 'region', 'prob_idx', 'age']:
                        result[key] = np.array(f[key])
                        # 如果需要转置
                        if result[key].ndim > 1:
                            result[key] = result[key].transpose()
    
    elif format.lower() == 'npy':
        # Npy格式加载
        npy_dir = os.path.join(directory, 'npy')
        
        # 加载数据
        data_path = os.path.join(npy_dir, f"region_{region_id}_data.npy")
        if os.path.exists(data_path):
            result['data'] = np.load(data_path)
        else:
            print(f"文件不存在: {data_path}")
            return None
        
        # 加载区域标签
        region_path = os.path.join(npy_dir, f"region_{region_id}_region.npy")
        if os.path.exists(region_path):
            result['region'] = np.load(region_path)
        
        # 加载病人ID
        prob_idx_path = os.path.join(npy_dir, f"region_{region_id}_prob_idx.npy")
        if os.path.exists(prob_idx_path):
            result['prob_idx'] = np.load(prob_idx_path)
        
        # 加载年龄数据（如果有）
        age_path = os.path.join(npy_dir, f"region_{region_id}_age.npy")
        if os.path.exists(age_path):
            result['age'] = np.load(age_path)
    
    else:
        print(f"不支持的格式: {format}")
        return None
    
    return result

def get_active_regions(directory, format='mat'):
    """
    获取目录中的所有活跃区域ID
    
    Args:
        directory (str): 数据目录路径
        format (str): 数据格式，'mat'或'npy'
        
    Returns:
        list: 活跃区域ID列表
    """
    active_regions = []
    
    if format.lower() == 'mat':
        # Mat格式
        mat_dir = os.path.join(directory, 'mat')
        if os.path.exists(mat_dir):
            files = glob.glob(os.path.join(mat_dir, "region_*.mat"))
            for file in files:
                try:
                    region_id = int(os.path.basename(file).split('_')[1].split('.')[0])
                    active_regions.append(region_id)
                except:
                    pass
    
    elif format.lower() == 'npy':
        # Npy格式
        npy_dir = os.path.join(directory, 'npy')
        if os.path.exists(npy_dir):
            files = glob.glob(os.path.join(npy_dir, "region_*_data.npy"))
            for file in files:
                try:
                    region_id = int(os.path.basename(file).split('_')[1])
                    active_regions.append(region_id)
                except:
                    pass
    
    return sorted(active_regions)

def load_all_regions(directory, region_ids=None, format='mat'):
    """
    加载指定目录下的所有区域数据或指定区域数据
    
    Args:
        directory (str): 数据目录路径
        region_ids (list): 要加载的区域ID列表，如果为None则加载所有区域
        format (str): 数据格式，'mat'或'npy'
        
    Returns:
        dict: 包含所有区域数据的字典，键为区域ID
    """
    if region_ids is None:
        region_ids = get_active_regions(directory, format)
    
    if not region_ids:
        print(f"未找到活跃区域，请检查目录: {directory}")
        return {}
    
    result = {}
    for region_id in tqdm(region_ids, desc=f"加载区域数据"):
        region_data = load_region_data(directory, region_id, format)
        if region_data is not None:
            result[region_id] = region_data
    
    return result
def load_brain_voxel_data(base_dir, split='train', format='mat', shuffle=True, seed=666):
    """
    从方法1的数据结构中加载脑体素数据
    
    Args:
        base_dir (str): 数据基础目录路径
        split (str): 数据集划分，'train', 'val'或'test'
        format (str): 数据格式，'mat'或'npy'
        shuffle (bool): 是否打乱数据
        seed (int): 随机种子
        
    Returns:
        dict: 包含样本和标签的数据字典
    """
    print(f"从 {base_dir} 加载 {split} 数据集...")
    
    # 设置目标目录
    target_dir = os.path.join(base_dir, split)
    
    # 加载所有区域数据
    print(f"加载所有区域数据...")
    regions_data = load_all_regions(target_dir, format=format)
    
    if not regions_data:
        raise ValueError(f"未找到区域数据，请检查目录: {target_dir}")
    
    # 初始化存储所有样本和标签的列表
    all_features = []
    all_labels = []
    
    print(f"处理区域数据...")
    for region_id, region_data in tqdm(regions_data.items(), desc=f"处理{split}区域"):
        if 'data' not in region_data or 'region' not in region_data:
            print(f"警告: 区域 {region_id} 缺少必要的数据字段，跳过")
            continue
        
        # 获取该区域的特征数据
        features = region_data['data']  # 这已经是经过StandardScaler处理的数据
        
        # 获取该区域的标签数据
        region_labels = region_data['region']
        
        # 确定每个样本的类别索引
        # 在方法1中，region是one-hot编码的，需要转换为类别索引
        # 检查region_labels的形状确定是否为one-hot
        if region_labels.shape[1] > 1:  # 如果是one-hot编码
            sample_labels = np.argmax(region_labels, axis=1)
        else:  # 如果已经是类别索引
            sample_labels = region_labels.flatten()
        
        # 添加到总列表
        all_features.append(features)
        all_labels.append(sample_labels)
    
    # 合并所有区域的数据
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"{split} 数据集总样本数: {len(all_features)}")
    
    # 全局打乱数据（如果需要）
    if shuffle:
        print(f"全局打乱 {split} 数据集...")
        np.random.seed(seed)
        indices = np.random.permutation(len(all_features))
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        print(f"完成全局打乱，确保341维特征、体素和标签的对应关系")
    
    return {
        'features': all_features,
        'labels': all_labels,
        'feature_dim': all_features.shape[1],
        'num_classes': 102  # 根据方法1中102维region标签
    }

def create_data_loaders(dataset_dict, batch_size=128, shuffle_train=True):
    """
    从数据字典创建PyTorch数据加载器
    
    Args:
        dataset_dict (dict): 包含features和labels的数据字典
        batch_size (int): 批次大小
        shuffle_train (bool): 是否打乱训练数据
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = BrainVoxelDataset(
        dataset_dict['features'],
        dataset_dict['labels']
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle_train
    )

def load_and_prepare_data(base_dir, format='mat', batch_size=128, shuffle=True, seed=666):
    """
    加载方法1的数据并创建数据加载器
    
    Args:
        base_dir (str): 数据基础目录路径
        format (str): 数据格式，'mat'或'npy'
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        seed (int): 随机种子
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_dim, num_classes)
    """
    # 加载scaler
    scaler = load_scaler(base_dir)
    print(f"成功加载StandardScaler，特征数量: {len(scaler.mean_)}")
    
    # 加载训练集
    train_data = load_brain_voxel_data(
        base_dir=base_dir, 
        split='train', 
        format=format,
        shuffle=shuffle,
        seed=seed
    )
    
    # 加载验证集
    val_data = load_brain_voxel_data(
        base_dir=base_dir, 
        split='val', 
        format=format,
        shuffle=shuffle,
        seed=seed
    )
    
    # 加载测试集
    test_data = load_brain_voxel_data(
        base_dir=base_dir, 
        split='test', 
        format=format,
        shuffle=shuffle,
        seed=seed
    )
    
    # 创建数据加载器
    train_loader = create_data_loaders(train_data, batch_size, shuffle_train=True)
    val_loader = create_data_loaders(val_data, batch_size, shuffle_train=False)
    test_loader = create_data_loaders(test_data, batch_size, shuffle_train=False)
    
    return train_loader, val_loader, test_loader, train_data['feature_dim'], train_data['num_classes'], scaler

if __name__ == "__main__":
    # 测试代码
    base_dir = "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/processed_data"
    format = 'mat'  # 或 'npy'
    
    try:
        print("测试加载scaler...")
        scaler = load_scaler(base_dir)
        print(f"成功加载scaler，特征数量: {len(scaler.mean_)}")
        
        print("\n测试加载单个区域数据...")
        region_id = 0  # 测试第一个区域
        region_data = load_region_data(os.path.join(base_dir, 'train'), region_id, format)
        if region_data:
            print(f"成功加载区域 {region_id} 数据:")
            for key, value in region_data.items():
                print(f"  - {key}: 形状 {value.shape}")
        
        print("\n测试加载所有区域ID...")
        active_regions = get_active_regions(os.path.join(base_dir, 'train'), format)
        print(f"活跃区域列表: {active_regions[:10]}... (共 {len(active_regions)} 个)")
        
        print("\n测试加载训练集体素数据...")
        train_data = load_brain_voxel_data(base_dir, 'train', format, shuffle=True)
        print(f"训练集样本数: {len(train_data['features'])}")
        print(f"特征维度: {train_data['feature_dim']}")
        print(f"标签分布: {np.bincount(train_data['labels'].astype(int))}")
        
        print("\n测试创建数据加载器...")
        train_loader = create_data_loaders(train_data, batch_size=128)
        print(f"数据加载器批次数: {len(train_loader)}")
        
        # 测试获取一个批次
        features, labels = next(iter(train_loader))
        print(f"批次特征形状: {features.shape}")
        print(f"批次标签形状: {labels.shape}")
        
        print("\n所有测试通过!")
        
    except Exception as e:
        print(f"测试出错: {str(e)}")