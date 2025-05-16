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


def create_data_loaders(base_dir, batch_size=32, test_patient_id=38, seed=42, 
                        train_patient_ids=None, valid_patient_ids=None, test_patient_ids=None):
    """
    创建训练集、验证集和测试集的数据加载器
    
    Args:
        base_dir (str): 重组数据的基础目录
        batch_size (int): 批次大小
        test_patient_id (int): 当未指定测试集患者ID时，必须放入测试集的患者ID
        seed (int): 随机种子
        train_patient_ids (list): 可选，指定训练集的患者ID列表
        valid_patient_ids (list): 可选，指定验证集的患者ID列表
        test_patient_ids (list): 可选，指定测试集的患者ID列表
        
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
    
    # 判断是使用手动指定的患者ID还是自动分配
    if train_patient_ids is not None and valid_patient_ids is not None and test_patient_ids is not None:
        # 手动指定模式
        print("使用手动指定的患者ID分配")
        
        # 验证指定的患者ID是否有效
        all_specified_ids = set(train_patient_ids + valid_patient_ids + test_patient_ids)
        available_ids = set(all_patient_ids)
        
        if not all_specified_ids.issubset(available_ids):
            missing_ids = all_specified_ids - available_ids
            raise ValueError(f"以下患者ID在数据集中不存在: {missing_ids}")
        
        # 检查是否有重复
        if len(all_specified_ids) != len(train_patient_ids) + len(valid_patient_ids) + len(test_patient_ids):
            raise ValueError("患者ID在不同集合之间有重复，请确保每个患者只分配到一个数据集")
        
    else:
        # 自动分配模式
        print("使用自动分配的患者ID (6:2:2比例)")
        
        # 确保测试患者ID在列表中
        if test_patient_id not in all_patient_ids:
            raise ValueError(f"测试患者ID {test_patient_id} 不在数据集中")
        
        # 从所有患者ID中移除测试患者ID
        remaining_patient_ids = [pid for pid in all_patient_ids if pid != test_patient_id]
        
        # 随机打乱剩余患者ID
        random.shuffle(remaining_patient_ids)
        
        # 计算6:2:2比例下应有的患者数
        total_patients = len(all_patient_ids)
        train_count = int(total_patients * 0.6)  # 60%
        valid_count = int(total_patients * 0.2)  # 20%
        
        # 测试集已经包含了指定的患者，计算还需要多少患者
        test_count_needed = int(total_patients * 0.2) - 1  # 20% - 已有的1个患者
        
        # 如果测试集需要的患者数为负数，则调整为0（极端情况下可能发生）
        test_count_needed = max(0, test_count_needed)
        
        # 从剩余患者中分配到测试集
        test_patient_ids = [test_patient_id] + remaining_patient_ids[:test_count_needed]
        
        # 剩余的患者按照训练集和验证集的比例分配
        # 剩余患者应该分配到训练集和验证集，比例为3:1（因为总比例是6:2，所以60:20 = 3:1）
        remaining_for_train_valid = remaining_patient_ids[test_count_needed:]
        train_valid_split = int(len(remaining_for_train_valid) * 0.75)  # 3/(3+1) = 0.75
        
        train_patient_ids = remaining_for_train_valid[:train_valid_split]
        valid_patient_ids = remaining_for_train_valid[train_valid_split:]
    
    print(f"训练集患者数: {len(train_patient_ids)}")
    print(f"验证集患者数: {len(valid_patient_ids)}")
    print(f"测试集患者数: {len(test_patient_ids)}")
    
    print(f"训练集患者ID: {train_patient_ids}")
    print(f"验证集患者ID: {valid_patient_ids}")
    print(f"测试集患者ID: {test_patient_ids}")
    
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
    train_features = np.vstack(train_features) if train_features else np.array([]).reshape(0, 341)
    train_labels = np.vstack(train_labels) if train_labels else np.array([]).reshape(0, 102)
    valid_features = np.vstack(valid_features) if valid_features else np.array([]).reshape(0, 341)
    valid_labels = np.vstack(valid_labels) if valid_labels else np.array([]).reshape(0, 102)
    test_features = np.vstack(test_features) if test_features else np.array([]).reshape(0, 341)
    test_labels = np.vstack(test_labels) if test_labels else np.array([]).reshape(0, 102)
    
    print(f"训练集样本数: {len(train_features)}")
    print(f"验证集样本数: {len(valid_features)}")
    print(f"测试集样本数: {len(test_features)}")
    
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
    base_dir = '/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/reorganized_fold_data'
    
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


    # 自动模式（保持原有行为）
    # train_loader, valid_loader, test_loader = create_data_loaders(
    #     base_dir=base_dir,
    #     batch_size=32,
    #     test_patient_id=38,
    #     seed=42
    # )

    # # 手动模式（指定患者ID分配）
    # train_loader, valid_loader, test_loader = create_data_loaders(
    #     base_dir=base_dir,
    #     batch_size=32,
    #     train_patient_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    #     valid_patient_ids=[21, 22, 23, 24, 25, 26, 27, 28],
    #     test_patient_ids=[29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    # )