import os
import json
import numpy as np
import h5py
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

class DataLoader:  
    def __init__(self, config_path):
        """
        初始化数据加载器
        
        参数:
            config_path: 配置文件路径，包含数据路径和分组信息
        """
        self.config = self.load_config(config_path)
        self.feature_groups = self.config.get('feature_groups', {})
        
        # 使用配置文件中指定的具体路径
        self.train_dir = self.config.get('train_dir', '')
        self.val_dir = self.config.get('val_dir', '')
        self.test_dir = self.config.get('test_dir', '')
        
        # 检查数据目录是否存在
        for dir_name, dir_path in [('train_dir', self.train_dir), 
                                ('val_dir', self.val_dir), 
                                ('test_dir', self.test_dir)]:
            if not os.path.exists(dir_path):
                print(f"警告: {dir_name} 目录不存在: {dir_path}")
                
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return json.load(f)
    

    def load_data(self, split='all'):
        """
        加载原始数据，返回特征矩阵和标签
        
        参数:
            split: 'train', 'val', 'test' 或 'all'
            
        返回:
            features_dict: 包含各集合特征的字典
            labels_dict: 包含各集合标签的字典
        """
        features_dict = {}
        labels_dict = {}
        
        if split in ['train', 'all']:
            train_sampler = BrainVoxelSampler(self.train_dir)
            train_features, train_labels = self._load_split_data(train_sampler, 'train')
            features_dict['train'] = train_features
            labels_dict['train'] = train_labels
            
        if split in ['val', 'all']:
            val_sampler = BrainVoxelSampler(self.val_dir)
            val_features, val_labels = self._load_split_data(val_sampler, 'validation')
            features_dict['val'] = val_features
            labels_dict['val'] = val_labels
            
        if split in ['test', 'all']:
            test_sampler = BrainVoxelSampler(self.test_dir)
            test_features, test_labels = self._load_split_data(test_sampler, 'test')
            features_dict['test'] = test_features
            labels_dict['test'] = test_labels
            
        return features_dict, labels_dict



    def _load_split_data(self, sampler, split_name):
        """从采样器加载特定分割的数据"""
        features = []
        labels = []
        
        for label_id in tqdm(sampler.valid_labels, desc=f"加载{split_name}数据"):
            file_path = sampler.get_file_path(label_id)
            if file_path:
                samples = np.load(file_path)
                labels_array = np.ones(len(samples)) * label_id
                features.append(samples)
                labels.append(labels_array)
        
        return np.vstack(features), np.concatenate(labels)
    
    def split_feature_groups(self, features_dict):
        """
        按配置分离特征组，返回字典形式的分组数据
        
        参数:
            features_dict: 包含各数据集特征的字典
            
        返回:
            grouped_features: 包含各特征组的字典
        """
        grouped_features = defaultdict(dict)
        
        for split, features in features_dict.items():
            for group_name, indices in self.feature_groups.items():
                # 转换索引范围为实际索引列表
                if isinstance(indices, list) and len(indices) == 2:
                    # 假设为范围[start, end]
                    start, end = indices
                    indices = list(range(start, end + 1))
                
                # 提取该组特征
                group_features = features[:, indices]
                grouped_features[split][group_name] = group_features
        
        return grouped_features
    
    def validate_data(self, features_dict, labels_dict):
        """
        验证数据完整性，返回验证报告
        
        参数:
            features_dict: 包含各数据集特征的字典
            labels_dict: 包含各数据集标签的字典
            
        返回:
            validation_report: 数据验证报告字典
        """
        validation_report = {
            "dataset_sizes": {},
            "feature_statistics": {},
            "label_statistics": {},
            "issues": []
        }
        
        for split in features_dict.keys():
            features = features_dict[split]
            labels = labels_dict[split]
            
            # 记录数据集大小
            validation_report["dataset_sizes"][split] = len(labels)
            
            # 检查特征统计
            feature_stats = {
                "shape": features.shape,
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "nan_count": int(np.isnan(features).sum()),
                "inf_count": int(np.isinf(features).sum()),
                "zero_count": int((features == 0).sum())
            }
            validation_report["feature_statistics"][split] = feature_stats
            
            # 检查标签统计
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            label_stats = {
                "unique_labels": len(unique_labels),
                "min_label": int(np.min(unique_labels)),
                "max_label": int(np.max(unique_labels)),
                "most_common": int(unique_labels[np.argmax(label_counts)]),
                "least_common": int(unique_labels[np.argmin(label_counts)]),
                "max_count": int(np.max(label_counts)),
                "min_count": int(np.min(label_counts)),
                "imbalance_ratio": float(np.max(label_counts) / np.min(label_counts))
            }
            validation_report["label_statistics"][split] = label_stats
            
            # 检查潜在问题
            if feature_stats["nan_count"] > 0:
                validation_report["issues"].append(f"{split} 数据中包含 {feature_stats['nan_count']} 个NaN值")
            if feature_stats["inf_count"] > 0:
                validation_report["issues"].append(f"{split} 数据中包含 {feature_stats['inf_count']} 个Inf值")
            if label_stats["imbalance_ratio"] > 10:
                validation_report["issues"].append(f"{split} 数据类别严重不平衡，比例为 {label_stats['imbalance_ratio']:.2f}:1")
        
        return validation_report
    
    def save_processed_data(self, features_dict, labels_dict, grouped_features, output_path):
        """
        将处理后的数据保存到HDF5文件
        
        参数:
            features_dict: 原始特征字典
            labels_dict: 标签字典
            grouped_features: 分组后的特征字典
            output_path: 输出文件路径
        """
        with h5py.File(output_path, 'w') as f:
            # 保存原始特征和标签
            for split in features_dict.keys():
                f.create_group(f'original/{split}')
                f[f'original/{split}'].create_dataset('features', data=features_dict[split])
                f[f'original/{split}'].create_dataset('labels', data=labels_dict[split])
            
            # 保存分组特征
            for split in grouped_features.keys():
                for group_name, group_features in grouped_features[split].items():
                    f.create_group(f'grouped/{split}/{group_name}')
                    f[f'grouped/{split}/{group_name}'].create_dataset('features', data=group_features)
                    f[f'grouped/{split}/{group_name}'].attrs['description'] = self.feature_groups.get(f"{group_name}_description", "")

class BrainVoxelSampler:
    """脑体素数据采样器，与您现有的代码兼容"""
    
    def __init__(self, data_dir):
        """
        初始化采样器
        
        参数:
            data_dir: 数据集目录
        """
        self.data_dir = data_dir
        self.label_info = self._load_label_index()
        self.valid_labels = [label for label, info in self.label_info.items() if info['count'] > 0]
    
    def _load_label_index(self):
        """加载标签索引文件"""
        index_file = os.path.join(self.data_dir, "label_index.txt")
        label_info = {}
        
        with open(index_file, 'r') as f:
            # 跳过表头
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    label_id = int(parts[0])
                    voxel_count = int(parts[1])
                    filename = parts[2] if parts[2] else None
                    label_info[label_id] = {'count': voxel_count, 'filename': filename}
        
        return label_info
    
    def get_file_path(self, label_id):
        """获取指定标签的文件路径"""
        if label_id not in self.label_info:
            return None
        
        filename = self.label_info[label_id]['filename']
        if not filename:
            return None
            
        return os.path.join(self.data_dir, filename)