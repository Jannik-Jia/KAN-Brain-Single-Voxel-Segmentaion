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

def process_train38_data(mat_file_path, config, random_state=666, scaler_save_path=None):
    """
    处理TRAIN38.mat数据，根据指定的患者ID分割为训练集、验证集和测试集
    
    参数:
        mat_file_path: TRAIN38.mat文件路径
        config: 配置字典，包含患者ID的配置
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
    
    # 确保prob_idx是整数类型，与配置中的患者ID匹配
    prob_idx = arrays['prob_idx'].flatten()
    # 转换为整数类型 - 这是关键修改点
    prob_idx = prob_idx.astype(int)
    
    # 获取配置中的患者ID
    dataset_split = config.get('dataset_split', {})
    train_patients = dataset_split.get('train_patients', [])
    val_patients = dataset_split.get('val_patients', [])
    test_patients = dataset_split.get('test_patients', [])
    
    # 如果未指定患者ID，使用旧的按比例分割方法
    if not train_patients and not val_patients and not test_patients:
        print("未指定患者ID，使用默认的比例分割方法")
        # 按原来的比例分割
        train_set_idx = np.where(prob_idx != 38)[0]
        val_set_idx = np.where(prob_idx == 38)[0]
        
        # 提取训练集和验证集
        train_set_data = train_data[train_set_idx, :]
        train_set_region = train_region[train_set_idx, :]
        val_data = train_data[val_set_idx, :]
        val_label = train_region[val_set_idx, :]
        
        # 分割训练集，留出一小部分作为测试集
        X_train, X_test, y_train, y_test = train_test_split(
            train_set_data, train_set_region, 
            test_size=config.get('test_size', 0.01), 
            random_state=random_state
        )
    else:
        print(f"使用指定的患者ID分割数据集:")
        print(f"  - 训练集患者ID: {train_patients}")
        print(f"  - 验证集患者ID: {val_patients}")
        print(f"  - 测试集患者ID: {test_patients}")
        
        # 根据患者ID分割数据
        # 这里使用已转换为整数的prob_idx进行比较
        train_set_idx = np.array([i for i, p in enumerate(prob_idx) if p in train_patients])
        val_set_idx = np.array([i for i, p in enumerate(prob_idx) if p in val_patients])
        test_set_idx = np.array([i for i, p in enumerate(prob_idx) if p in test_patients])
        
        # 打印分割信息
        print(f"数据分割情况:")
        print(f"  - 训练集样本索引数量: {len(train_set_idx)}")
        print(f"  - 验证集样本索引数量: {len(val_set_idx)}")
        print(f"  - 测试集样本索引数量: {len(test_set_idx)}")
        
        # 如果任何集合为空，发出警告
        if len(train_set_idx) == 0:
            print("警告: 训练集为空！请检查训练集患者ID是否正确。")
        if len(val_set_idx) == 0:
            print("警告: 验证集为空！请检查验证集患者ID是否正确。")
        if len(test_set_idx) == 0:
            print("警告: 测试集为空！请检查测试集患者ID是否正确。")
        
        # 提取各个数据集
        X_train = train_data[train_set_idx, :]
        y_train = train_region[train_set_idx, :]
        
        val_data = train_data[val_set_idx, :]
        val_label = train_region[val_set_idx, :]
        
        X_test = train_data[test_set_idx, :]
        y_test = train_region[test_set_idx, :]
        
        print(f"数据集分割完成:")
        print(f"  - 训练集样本数: {len(X_train)}")
        print(f"  - 验证集样本数: {len(val_data)}")
        print(f"  - 测试集样本数: {len(X_test)}")
        
        # 记录为空的集合
        if len(X_train) == 0 or len(val_data) == 0 or len(X_test) == 0:
            print("错误：至少有一个数据集为空。请检查患者ID配置。")
            # 可以在这里选择抛出异常或使用备选方案
            raise ValueError("数据集分割失败：至少有一个数据集为空")
    
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



def load_and_process_data(config, mode='train', model_path=None):
    """
    统一的数据加载与处理函数，用于训练和评估
    
    参数:
        config: 配置字典，至少包含mat_file_path和患者ID配置
        mode: 'train'表示训练模式，会拟合scaler; 'eval'表示评估模式，会加载已有scaler
        model_path: 在'eval'模式下，可以提供模型路径以加载与之关联的scaler
    
    返回:
        dataset_dict: 数据集字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    print(f"开始加载数据集 (模式: {mode})...")
    
    # 检查配置中是否有mat_file_path
    if 'mat_file_path' not in config or not config['mat_file_path']:
        raise ValueError("配置中缺少'mat_file_path'，请在配置中指定TRAIN38.mat文件路径")
    
    scaler = None
    
    # 评估模式：尝试加载现有scaler
    if mode == 'eval' and model_path:
        try:
            # 尝试从模型加载scaler
            import joblib
            from utils.model_io import load_model_with_architecture
            
            _, checkpoint, loaded_scaler = load_model_with_architecture(
                model_path=model_path,
                device='cpu',  # 只需要scaler，不需要模型加载到GPU
                load_scaler=True
            )
            
            if loaded_scaler:
                scaler = loaded_scaler
                print(f"从模型加载了scaler: {model_path}")
            else:
                # 尝试从模型目录加载scaler.joblib
                scaler_path = os.path.join(os.path.dirname(model_path), "scaler.joblib")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    print(f"从目录加载了scaler: {scaler_path}")
        except Exception as e:
            print(f"加载scaler时出错: {e}")
            print("将重新创建scaler")
    
    # 训练模式：处理数据并创建新scaler
    if mode == 'train' or scaler is None:
        # 处理TRAIN38.mat数据
        scaler_save_path = None
        if 'save_dir' in config:
            scaler_save_path = os.path.join(config['save_dir'], "scaler.joblib")
        
        # 修改：传递完整配置
        dataset_dict = process_train38_data(
            mat_file_path=config['mat_file_path'],
            config=config,  # 传递完整配置
            random_state=config.get('random_seed', 666),
            scaler_save_path=scaler_save_path
        )
        
        # 保存scaler到配置
        if 'scaler' not in config:
            config['scaler'] = dataset_dict['scaler']
    
    # 评估模式且scaler已加载：使用现有scaler处理数据
    elif mode == 'eval' and scaler:
        # 加载数据
        arrays = load_mat_data(config['mat_file_path'])
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
        
        # 进一步分割训练集，留出一小部分作为测试集
        X_train, X_test, y_train, y_test = train_test_split(
            train_set_data, train_set_region, 
            test_size=config.get('test_size', 0.01), 
            random_state=config.get('random_seed', 666)
        )
        
        # 使用加载的scaler进行变换
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        val_data_scaled = scaler.transform(val_data)
        
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
        
        # 保存scaler到配置
        if 'scaler' not in config:
            config['scaler'] = scaler
    
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
    
    print(f"数据加载完成! 共载入 {len(dataset_dict['train_samples'])} 个训练样本，"
          f"{len(dataset_dict['val_samples'])} 个验证样本，"
          f"{len(dataset_dict['test_samples'])} 个测试样本")
    print(f"特征维度: {dataset_dict['feature_dim']}")
    
    return dataset_dict, dataloaders['train'], dataloaders['val'], dataloaders['test']

