import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class BrainVoxelDataset(Dataset):
    def __init__(self, data_features, data_labels):
        """
        初始化脑部体素数据集
        
        Args:
            data_features (numpy.ndarray): 体素特征数据 (N, 341)
            data_labels (numpy.ndarray): 区域标签数据 (N, 102)
        """
        self.features = torch.FloatTensor(data_features)
        self.labels = torch.FloatTensor(data_labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_reorganized_data(base_dir):
    """
    从重组后的数据目录加载数据
    
    Args:
        base_dir (str): 重组数据的基础目录
        
    Returns:
        dict: 按患者ID组织的数据字典
    """
    dataset_index_file = os.path.join(base_dir, "dataset_index.csv")
    
    if not os.path.exists(dataset_index_file):
        raise FileNotFoundError(f"索引文件未找到: {dataset_index_file}")
    
    # 加载索引文件
    index_df = pd.read_csv(dataset_index_file)
    
    # 创建按患者ID组织的数据字典
    patient_data = {}
    
    for patient_id in index_df['patient_id'].unique():
        patient_id = int(patient_id)
        patient_records = index_df[index_df['patient_id'] == patient_id]
        
        features_list = []
        labels_list = []
        
        for _, record in patient_records.iterrows():
            file_path = os.path.join(base_dir, record['file_path'])
            region_id = record['region_id']
            
            if os.path.exists(file_path):
                # 加载体素数据
                voxel_data = np.load(file_path)
                
                # 为每个体素创建标签 (one-hot 编码)
                labels = np.zeros((voxel_data.shape[0], 102))
                labels[:, region_id] = 1
                
                # 添加到列表
                features_list.append(voxel_data)
                labels_list.append(labels)
        
        if features_list and labels_list:
            # 合并该患者的所有数据
            patient_features = np.vstack(features_list)
            patient_labels = np.vstack(labels_list)
            
            # 存储到字典
            patient_data[patient_id] = {
                'features': patient_features,
                'labels': patient_labels
            }
    
    return patient_data

def create_data_loaders(base_dir, batch_size=32, test_patient_id=38, seed=42):
    """
    创建训练集、验证集和测试集的数据加载器
    
    Args:
        base_dir (str): 重组数据的基础目录
        batch_size (int): 批次大小
        test_patient_id (int): 必须放入测试集的患者ID
        seed (int): 随机种子
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 加载按患者ID组织的数据
    patient_data = load_reorganized_data(base_dir)
    
    # 提取所有患者ID
    all_patient_ids = list(patient_data.keys())
    
    # 确保测试患者ID在列表中
    if test_patient_id not in all_patient_ids:
        raise ValueError(f"测试患者ID {test_patient_id} 不在数据集中")
    
    # 从所有患者ID中移除测试患者ID
    remaining_patient_ids = [pid for pid in all_patient_ids if pid != test_patient_id]
    
    # 随机打乱剩余患者ID
    random.shuffle(remaining_patient_ids)
    
    # 计算训练集和验证集的大小
    total_remaining = len(remaining_patient_ids)
    train_size = int(total_remaining * 0.75)  # 6/8 = 0.75 (因为剩余患者占总数的80%)
    
    # 分割患者ID
    train_patient_ids = remaining_patient_ids[:train_size]
    valid_patient_ids = remaining_patient_ids[train_size:]
    test_patient_ids = [test_patient_id]
    
    print(f"训练集患者数: {len(train_patient_ids)}")
    print(f"验证集患者数: {len(valid_patient_ids)}")
    print(f"测试集患者数: {len(test_patient_ids)}")
    
    # 收集每个数据集的特征和标签
    train_features, train_labels = [], []
    valid_features, valid_labels = [], []
    test_features, test_labels = [], []
    
    # 收集训练集数据
    for patient_id in train_patient_ids:
        train_features.append(patient_data[patient_id]['features'])
        train_labels.append(patient_data[patient_id]['labels'])
    
    # 收集验证集数据
    for patient_id in valid_patient_ids:
        valid_features.append(patient_data[patient_id]['features'])
        valid_labels.append(patient_data[patient_id]['labels'])
    
    # 收集测试集数据
    for patient_id in test_patient_ids:
        test_features.append(patient_data[patient_id]['features'])
        test_labels.append(patient_data[patient_id]['labels'])
    
    # 合并数据
    train_features = np.vstack(train_features)
    train_labels = np.vstack(train_labels)
    valid_features = np.vstack(valid_features)
    valid_labels = np.vstack(valid_labels)
    test_features = np.vstack(test_features)
    test_labels = np.vstack(test_labels)
    
    print(f"训练集样本数: {len(train_features)}")
    print(f"验证集样本数: {len(valid_features)}")
    print(f"测试集样本数: {len(test_features)}")
    
    # 特征标准化 (只使用训练集来拟合标准化器)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)
    test_features = scaler.transform(test_features)
    
    # 为训练集和验证集创建索引
    train_indices = np.arange(len(train_features))
    valid_indices = np.arange(len(valid_features))
    
    # 随机打乱训练集和验证集的索引
    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    
    # 使用打乱的索引重新排列数据
    train_features = train_features[train_indices]
    train_labels = train_labels[train_indices]
    valid_features = valid_features[valid_indices]
    valid_labels = valid_labels[valid_indices]
    
    # 创建数据集
    train_dataset = BrainVoxelDataset(train_features, train_labels)
    valid_dataset = BrainVoxelDataset(valid_features, valid_labels)
    test_dataset = BrainVoxelDataset(test_features, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 设置重组数据的基础目录
    base_dir = '/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/reorganized_data'
    
    # 创建数据加载器
    train_loader, valid_loader, test_loader = create_data_loaders(
        base_dir=base_dir,
        batch_size=32,
        test_patient_id=38,
        seed=42
    )
    
    # 检查数据加载器
    print("\n数据加载器信息:")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(valid_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 获取一批数据样本并查看形状
    train_features, train_labels = next(iter(train_loader))
    print(f"\n训练批次样本形状: {train_features.shape}")
    print(f"训练批次标签形状: {train_labels.shape}")