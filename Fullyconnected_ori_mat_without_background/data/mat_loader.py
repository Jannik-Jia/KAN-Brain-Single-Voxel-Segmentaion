#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAT文件数据加载和处理函数 - 支持固定prob_idx测试集划分
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
        print("⚠️  警告: 仍存在背景标签0")
    else:
        print("✅ 确认: 已成功过滤背景标签")
    
    # 检查标签连续性
    expected_range = set(range(1, 103))  # 1-102
    actual_labels = set(unique_labels)
    missing_labels = expected_range - actual_labels
    if missing_labels:
        print(f"⚠️  缺失的标签: {sorted(missing_labels)}")
    
    return True

def process_train38_data_with_fixed_test(mat_file_path, config, random_state=666, scaler_save_path=None):
    """
    处理TRAIN38.mat数据，支持固定prob_idx测试集划分和可配置的背景处理
    
    参数:
        mat_file_path: .mat文件路径
        config: 配置字典，应包含'test_prob_idx'列表和背景处理配置
        random_state: 随机种子
        scaler_save_path: scaler保存路径
    
    返回:
        dataset_dict: 包含训练、验证、测试数据的字典
    """
    print("🔄 处理TRAIN38.mat数据（固定prob_idx测试集划分，可配置背景处理）...")
    
    # 获取背景处理配置
    from utils.label_processing import should_filter_background_samples
    filter_background = should_filter_background_samples(config)
    
    # 🔧 从配置中读取测试集prob_idx列表
    test_prob_idx = config.get('test_prob_idx', [13, 23, 38])
    train_val_ratio = config.get('train_val_ratio', 0.75)  # 训练集在训练+验证中的比例
    
    print(f"📋 配置参数:")
    print(f"  测试集prob_idx: {test_prob_idx}")
    print(f"  训练/验证比例: {train_val_ratio:.2f}/{1-train_val_ratio:.2f}")
    print(f"  背景处理模式: {'过滤背景' if filter_background else '保留背景'}")
    
    # 🔧 第1步：加载数据
    arrays = load_mat_data(mat_file_path)
    train_data = arrays['data']
    train_region = arrays['region']
    prob_idx = arrays['prob_idx'].flatten().astype(int)
    
    print(f"原始数据形状: data={train_data.shape}, region={train_region.shape}, prob_idx={prob_idx.shape}")
    print(f"prob_idx范围: {prob_idx.min()} - {prob_idx.max()}")
    print(f"prob_idx唯一值: {sorted(np.unique(prob_idx))}")
    
    # 🔧 第2步：根据配置决定是否过滤背景像素
    if filter_background:
        print("\n🔍 过滤背景像素...")
        if len(train_region.shape) > 1 and train_region.shape[1] > 1:
            # one-hot格式，获取标签索引
            label_indices = np.argmax(train_region, axis=1)
        else:
            # 已经是索引格式
            label_indices = train_region.flatten()

        # 找出背景像素（标签0）
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
        
        # 🔧 第3步：验证过滤后的数据
        validate_filtered_data(train_data, train_region, "过滤背景后的完整数据")
    else:
        print("\n📋 保留背景像素，所有样本参与训练")
        print(f"保留数据形状: data={train_data.shape}, region={train_region.shape}, prob_idx={prob_idx.shape}")
        
        # 🔧 第3步：验证保留背景的数据
        validate_filtered_data(train_data, train_region, "包含背景的完整数据")
    
    # 🔧 第4步：按固定prob_idx划分测试集
    print(f"\n📊 按固定prob_idx划分数据集...")
    
    # 创建测试集掩码
    test_mask = np.isin(prob_idx, test_prob_idx)
    train_val_mask = ~test_mask
    
    # 分离测试集
    X_test = train_data[test_mask]
    y_test = train_region[test_mask]
    test_prob_idx_actual = prob_idx[test_mask]
    
    # 分离训练+验证数据
    train_val_data = train_data[train_val_mask]
    train_val_region = train_region[train_val_mask]
    train_val_prob_idx = prob_idx[train_val_mask]
    
    print(f"数据集划分结果:")
    print(f"  测试集样本数: {len(X_test)}")
    print(f"  训练+验证集样本数: {len(train_val_data)}")
    print(f"  测试集实际包含的prob_idx: {sorted(np.unique(test_prob_idx_actual))}")
    print(f"  训练+验证集包含的prob_idx: {sorted(np.unique(train_val_prob_idx))}")
    
    # 🔧 验证数据集之间无交叉
    test_prob_set = set(test_prob_idx_actual)
    train_val_prob_set = set(train_val_prob_idx)
    intersection = test_prob_set.intersection(train_val_prob_set)
    
    if intersection:
        raise ValueError(f"❌ 数据集交叉检测失败！交叉的prob_idx: {intersection}")
    else:
        print("✅ 数据集交叉检测通过：测试集与训练+验证集无prob_idx交叉")
    
    # 🔧 第5步：将训练+验证数据随机划分为训练集和验证集
    print(f"\n🎲 随机划分训练集和验证集（比例: {train_val_ratio:.2f}:{1-train_val_ratio:.2f}）...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_val_data, train_val_region,
        test_size=1-train_val_ratio,  # 验证集比例
        random_state=random_state,
        shuffle=True,
        stratify=None  # 可以考虑按标签分层，但由于类别较多可能会有问题
    )
    
    print(f"最终数据集大小:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本") 
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  总计: {len(X_train) + len(X_val) + len(X_test)} 样本")
    
    # 🔧 第6步：验证分割后的各数据集
    print("\n🔍 验证分割后的数据集...")
    validate_filtered_data(X_train, y_train, "训练集")
    validate_filtered_data(X_val, y_val, "验证集")  
    validate_filtered_data(X_test, y_test, "测试集")
    
    # 🔧 第7步：仅在训练集上拟合StandardScaler
    print("\n📊 在训练集上拟合StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 使用训练集的scaler变换验证集和测试集
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ StandardScaler拟合和变换完成")
    print(f"  训练集均值范围: [{X_train_scaled.mean(axis=0).min():.4f}, {X_train_scaled.mean(axis=0).max():.4f}]")
    print(f"  训练集标准差范围: [{X_train_scaled.std(axis=0).min():.4f}, {X_train_scaled.std(axis=0).max():.4f}]")
    
    # 🔧 第8步：保存scaler
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"✅ Scaler已保存到: {scaler_save_path}")
    
    # 🔧 第9步：处理标签格式（新增）
    print(f"\n🏷️ 处理标签格式...")
    
    # 使用统一的标签处理函数
    from utils.label_processing import process_labels_for_training, get_effective_num_classes
    
    # 处理标签 - 注意这里传入的是原始标签（未scaled的）
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
    
    # 检查是否有ignore标签
    ignore_count_train = np.sum(processed_train_labels == -1) if not filter_background else 0
    ignore_count_val = np.sum(processed_val_labels == -1) if not filter_background else 0
    ignore_count_test = np.sum(processed_test_labels == -1) if not filter_background else 0
    
    if not filter_background:
        print(f"  训练集忽略样本数: {ignore_count_train}")
        print(f"  验证集忽略样本数: {ignore_count_val}")
        print(f"  测试集忽略样本数: {ignore_count_test}")
    
    # 🔧 第10步：创建数据集字典（保留所有原有功能）
    dataset_dict = {
        'train_samples': X_train_scaled,
        'train_labels': processed_train_labels,  # 使用处理后的标签
        'val_samples': X_val_scaled,  # 注意：这里是验证集，不是测试集
        'val_labels': processed_val_labels,      # 使用处理后的标签
        'test_samples': X_test_scaled,
        'test_labels': processed_test_labels,    # 使用处理后的标签
        'feature_dim': X_train.shape[1],
        'scaler': scaler,
        'num_classes': effective_num_classes,    # 使用实际类别数（原来是这样计算的，现在改为配置驱动）
        # 额外信息（保留所有原有信息）
        'test_prob_idx_used': test_prob_idx,
        'test_prob_idx_actual': sorted(np.unique(test_prob_idx_actual)),
        'train_val_prob_idx': sorted(np.unique(train_val_prob_idx)),
        # 新增：背景处理信息
        'background_filtered': filter_background,
        'background_config': {
            'filter_background': config.get('filter_background', True),
            'include_background_in_classes': config.get('include_background_in_classes', False),
            'background_label_target': config.get('background_label_target', -1)
        }
    }
    
    print(f"\n✅ 数据处理完成！")
    print(f"  特征维度: {dataset_dict['feature_dim']}")
    print(f"  类别数量: {dataset_dict['num_classes']}")
    print(f"  测试集使用的prob_idx: {dataset_dict['test_prob_idx_used']}")
    print(f"  测试集实际包含的prob_idx: {dataset_dict['test_prob_idx_actual']}")
    print(f"  背景处理模式: {'过滤背景' if filter_background else '保留背景'}")
    
    return dataset_dict

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

def load_and_process_data_with_fixed_test(config, mode='train', model_path=None):
    """
    统一的数据加载与处理函数，支持固定prob_idx测试集划分
    
    参数:
        config: 配置字典，应包含mat_file_path和test_prob_idx
        mode: 'train'表示训练模式，会拟合scaler; 'eval'表示评估模式，会加载已有scaler
        model_path: 在'eval'模式下，可以提供模型路径以加载与之关联的scaler
    
    返回:
        dataset_dict: 数据集字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    print(f"🚀 开始加载数据集（固定测试集模式: {mode}）...")
    
    # 检查配置
    if 'mat_file_path' not in config or not config['mat_file_path']:
        raise ValueError("配置中缺少'mat_file_path'，请在配置中指定TRAIN38.mat文件路径")
    
    # 设置默认的test_prob_idx
    if 'test_prob_idx' not in config:
        config['test_prob_idx'] = [13, 23, 38]  # 默认测试集prob_idx
        print(f"⚠️ 未设置test_prob_idx，使用默认值: {config['test_prob_idx']}")
    
    scaler = None
    
    # 评估模式：尝试加载现有scaler
    if mode == 'eval' and model_path:
        try:
            import joblib
            from utils.model_io import load_model_with_architecture
            
            _, checkpoint, loaded_scaler = load_model_with_architecture(
                model_path=model_path,
                device='cpu',
                load_scaler=True
            )
            
            if loaded_scaler:
                scaler = loaded_scaler
                print(f"✅ 从模型加载了scaler: {model_path}")
            else:
                # 尝试从模型目录加载scaler.joblib
                scaler_path = os.path.join(os.path.dirname(model_path), "scaler.joblib")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    print(f"✅ 从目录加载了scaler: {scaler_path}")
        except Exception as e:
            print(f"⚠️ 加载scaler时出错: {e}")
            print("将重新创建scaler")
    
    # 训练模式：使用新的固定测试集处理函数
    if mode == 'train' or scaler is None:
        scaler_save_path = None
        if 'save_dir' in config:
            scaler_save_path = os.path.join(config['save_dir'], "scaler.joblib")
        
        # 🔧 使用新的固定测试集处理函数
        dataset_dict = process_train38_data_with_fixed_test(
            mat_file_path=config['mat_file_path'],
            config=config,
            random_state=config.get('random_seed', 666),
            scaler_save_path=scaler_save_path
        )
        
        # 保存scaler到配置
        if 'scaler' not in config:
            config['scaler'] = dataset_dict['scaler']
    
    # 评估模式且scaler已加载：使用现有scaler重新处理数据
    elif mode == 'eval' and scaler:
        print("⚠️ 评估模式下使用现有scaler重新处理数据...")
        # 为了保持一致性，建议在评估模式下也使用相同的数据划分逻辑
        # 但使用已加载的scaler
        temp_config = config.copy()
        temp_config['scaler'] = scaler
        
        # 重新处理数据但使用现有scaler
        dataset_dict = process_train38_data_with_fixed_test(
            mat_file_path=config['mat_file_path'],
            config=temp_config,
            random_state=config.get('random_seed', 666),
            scaler_save_path=None  # 不保存，因为使用的是现有scaler
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
    
    print(f"\n🎉 数据加载完成！")
    print(f"  📊 训练样本: {len(dataset_dict['train_samples'])}")
    print(f"  📊 验证样本: {len(dataset_dict['val_samples'])}")
    print(f"  📊 测试样本: {len(dataset_dict['test_samples'])}")
    print(f"  🔢 特征维度: {dataset_dict['feature_dim']}")
    print(f"  🏷️ 类别数量: {dataset_dict['num_classes']}")
    print(f"  🎯 测试集prob_idx: {dataset_dict['test_prob_idx_actual']}")
    
    return dataset_dict, dataloaders['train'], dataloaders['val'], dataloaders['test']

# 为了保持向后兼容性，保留原有函数名
def load_and_process_data(config, mode='train', model_path=None):
    """
    向后兼容的数据加载函数
    
    如果config中设置了test_prob_idx，则使用固定测试集模式
    否则使用原有的随机划分模式
    """
    if 'test_prob_idx' in config and config['test_prob_idx']:
        print("🔧 检测到test_prob_idx配置，使用固定测试集划分模式")
        return load_and_process_data_with_fixed_test(config, mode, model_path)
    else:
        print("🔧 未检测到test_prob_idx配置，使用原有随机划分模式")
        # 这里需要调用原有的process_train38_data函数
        # 为了简化，我们将其重命名为process_train38_data_random
        return load_and_process_data_random(config, mode, model_path)

def process_train38_data(mat_file_path, config, random_state=666, scaler_save_path=None):
    """
    向后兼容的process_train38_data函数
    根据config中是否有test_prob_idx来决定使用哪种划分方式
    """
    if 'test_prob_idx' in config and config['test_prob_idx']:
        return process_train38_data_with_fixed_test(mat_file_path, config, random_state, scaler_save_path)
    else:
        return process_train38_data_random(mat_file_path, config, random_state, scaler_save_path)

def process_train38_data_random(mat_file_path, config, random_state=666, scaler_save_path=None):
    """
    原有的随机划分逻辑（保持向后兼容）
    """
    print("处理TRAIN38.mat数据（随机划分模式）...")
    
    # 加载数据
    arrays = load_mat_data(mat_file_path)
    train_data = arrays['data']
    train_region = arrays['region']
    
    # 【第1步：添加背景过滤逻辑】
    prob_idx = arrays['prob_idx'].flatten()
    prob_idx = prob_idx.astype(int)
    
    # 新增：过滤背景像素
    print("过滤背景像素...")
    if len(train_region.shape) > 1 and train_region.shape[1] > 1:
        # one-hot格式，获取标签索引
        label_indices = np.argmax(train_region, axis=1)
    else:
        # 已经是索引格式
        label_indices = train_region.flatten()

    # 找出背景像素（标签0）
    background_mask = (label_indices == 0)
    valid_mask = ~background_mask

    print(f"总样本数: {len(train_data)}")
    print(f"背景像素数: {np.sum(background_mask)} ({np.sum(background_mask)/len(train_data)*100:.2f}%)")
    print(f"有效像素数: {np.sum(valid_mask)} ({np.sum(valid_mask)/len(train_data)*100:.2f}%)")

    # 过滤掉背景像素
    train_data = train_data[valid_mask]
    train_region = train_region[valid_mask]
    prob_idx = prob_idx[valid_mask]  # 同时过滤患者ID

    print(f"过滤后数据形状: {train_data.shape}, {train_region.shape}")
    
    # 【第2步：在这里添加过滤后的验证】
    validate_filtered_data(train_data, train_region, "过滤后的完整数据")
    
    # 🔧 修改：新的数据分割逻辑
    print("使用新的数据分割策略:")
    print("  - 测试集: prob_idx == 38")
    print("  - 训练集和验证集: prob_idx != 38，按6:2随机分割")
    
    # 分离测试集（prob_idx == 38）
    test_set_idx = np.where(prob_idx == 38)[0]
    X_test = train_data[test_set_idx, :]
    y_test = train_region[test_set_idx, :]
    
    # 获取非38号的数据用于训练和验证
    train_val_set_idx = np.where(prob_idx != 38)[0]
    train_val_data = train_data[train_val_set_idx, :]
    train_val_region = train_region[train_val_set_idx, :]
    
    # 将非38号数据按6:2随机分割为训练集和验证集
    # test_size=0.25 表示验证集占25%，即2/(6+2)=0.25
    X_train, val_data, y_train, val_label = train_test_split(
        train_val_data, train_val_region,
        test_size=0.25,  # 验证集占25% (2/8)
        random_state=random_state,
        shuffle=True  # 确保打乱
    )
    
    print(f"数据集分割完成:")
    print(f"  - 训练集样本数: {len(X_train)} (约75%)")
    print(f"  - 验证集样本数: {len(val_data)} (约25%)")
    print(f"  - 测试集样本数: {len(X_test)} (prob_idx==38)")
    
    # 【第3步：在数据分割完成后添加验证】
    print("\n验证分割后的数据集...")
    validate_filtered_data(X_train, y_train, "训练集")
    validate_filtered_data(val_data, val_label, "验证集")  
    validate_filtered_data(X_test, y_test, "测试集")
    
    # 应用StandardScaler（原有逻辑保持不变）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    val_data_scaled = scaler.transform(val_data)
    
    # 保存scaler
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler已保存到: {scaler_save_path}")
    
    # 创建数据集字典（原有逻辑保持不变）
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

def load_and_process_data_random(config, mode='train', model_path=None):
    """
    原有的随机划分数据加载函数（保持向后兼容）
    """
    print(f"开始加载数据集（随机划分模式: {mode}）...")
    
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
        
        # 使用随机划分函数
        dataset_dict = process_train38_data_random(
            mat_file_path=config['mat_file_path'],
            config=config,
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