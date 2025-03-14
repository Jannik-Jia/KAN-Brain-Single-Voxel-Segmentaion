"""
数据集和采样器模块
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

class BrainVoxelDataset(Dataset):
    """
    脑体素数据集类，简化版
    """
    def __init__(self, data, labels, is_inference=False):
        """
        初始化数据集
        
        参数:
            data: 特征数据，形状为(n_samples, feature_dim)
            labels: 标签数据，形状为(n_samples,)
            is_inference: 是否为推理模式（不返回标签）
        """
        super(BrainVoxelDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.is_inference = is_inference
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.FloatTensor(x)
        
        if self.is_inference:
            return x
        else:
            y = self.labels[idx]
            y = torch.LongTensor([int(y)])[0]  # 显式转换为整数
            return x, y

class BrainVoxelSampler:
    """脑体素数据采样器，提供多种采样策略"""
    
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
    
    def balanced_sampling(self, target_label, sample_count=None, pos_neg_ratio=1.0):
        """
        平衡采样策略
        
        参数:
            target_label: 目标标签（正类）
            sample_count: 采样数量，None表示使用所有可用样本
            pos_neg_ratio: 正负样本比例，默认1.0（平衡）
            
        返回:
            samples: 特征数据
            labels: 对应的标签
        """
        if target_label not in self.valid_labels:
            raise ValueError(f"标签 {target_label} 不在有效标签列表中")
        
        # 加载正样本
        pos_file = self.get_file_path(target_label)
        if not pos_file:
            raise ValueError(f"找不到标签 {target_label} 的数据文件")
            
        pos_samples = np.load(pos_file)
        pos_count = len(pos_samples)
        
        # 确定采样数量
        if sample_count is None:
            # 使用所有正样本
            target_pos_count = pos_count
        else:
            # 根据pos_neg_ratio计算正样本数量
            target_pos_count = min(pos_count, int(sample_count * pos_neg_ratio / (1 + pos_neg_ratio)))
        
        # 如果需要，随机选取正样本子集
        if target_pos_count < pos_count:
            pos_indices = np.random.choice(pos_count, target_pos_count, replace=False)
            pos_samples = pos_samples[pos_indices]
        
        # 计算需要的负样本数量
        neg_count_needed = int(target_pos_count / pos_neg_ratio)
        
        # 收集负样本（从其他标签）
        other_labels = [l for l in self.valid_labels if l != target_label]
        np.random.shuffle(other_labels)
        
        neg_samples = []
        current_neg_count = 0
        
        for other_label in other_labels:
            if current_neg_count >= neg_count_needed:
                break
                
            neg_file = self.get_file_path(other_label)
            if not neg_file:
                continue
                
            other_samples = np.load(neg_file)
            samples_needed = min(len(other_samples), neg_count_needed - current_neg_count)
            
            if samples_needed < len(other_samples):
                # 随机选择子集
                indices = np.random.choice(len(other_samples), samples_needed, replace=False)
                selected_samples = other_samples[indices]
            else:
                selected_samples = other_samples
            
            neg_samples.append(selected_samples)
            current_neg_count += len(selected_samples)
        
        # 合并所有负样本
        if neg_samples:
            all_neg_samples = np.vstack(neg_samples)
        else:
            all_neg_samples = np.array([]).reshape(0, pos_samples.shape[1])
        
        # 创建标签
        pos_labels = np.ones(len(pos_samples))
        neg_labels = np.zeros(len(all_neg_samples))
        
        # 合并样本和标签
        X = np.vstack([pos_samples, all_neg_samples])
        y = np.concatenate([pos_labels, neg_labels])
        
        # 随机打乱
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def stratified_sampling(self, target_label, sample_count=None, neg_label_count=None):
        """
        分层采样策略 - 从每个负类标签中平均采样
        
        参数:
            target_label: 目标标签（正类）
            sample_count: 总采样数量，None表示尽可能多
            neg_label_count: 使用的负类标签数量，None表示使用所有
            
        返回:
            samples: 特征数据
            labels: 对应的标签
        """
        if target_label not in self.valid_labels:
            raise ValueError(f"标签 {target_label} 不在有效标签列表中")
        
        # 加载正样本
        pos_file = self.get_file_path(target_label)
        if not pos_file:
            raise ValueError(f"找不到标签 {target_label} 的数据文件")
            
        pos_samples = np.load(pos_file)
        pos_count = len(pos_samples)
        
        # 确定采样数量
        if sample_count is None:
            # 使用所有正样本
            target_pos_count = pos_count
        else:
            # 使用指定数量，但不超过可用数量
            target_pos_count = min(pos_count, sample_count // 2)
        
        # 如果需要，随机选取正样本子集
        if target_pos_count < pos_count:
            pos_indices = np.random.choice(pos_count, target_pos_count, replace=False)
            pos_samples = pos_samples[pos_indices]
        
        # 收集负样本（从其他标签）
        other_labels = [l for l in self.valid_labels if l != target_label]
        if neg_label_count is not None:
            if neg_label_count < len(other_labels):
                other_labels = np.random.choice(other_labels, neg_label_count, replace=False)
        
        neg_count_per_label = target_pos_count // max(1, len(other_labels))
        
        neg_samples = []
        
        for other_label in other_labels:
            neg_file = self.get_file_path(other_label)
            if not neg_file:
                continue
                
            other_samples = np.load(neg_file)
            samples_needed = min(len(other_samples), neg_count_per_label)
            
            if samples_needed < len(other_samples):
                # 随机选择子集
                indices = np.random.choice(len(other_samples), samples_needed, replace=False)
                selected_samples = other_samples[indices]
            else:
                selected_samples = other_samples
            
            neg_samples.append(selected_samples)
        
        # 合并所有负样本
        if neg_samples:
            all_neg_samples = np.vstack(neg_samples)
        else:
            all_neg_samples = np.array([]).reshape(0, pos_samples.shape[1])
        
        # 确保负样本总数与正样本相同
        if len(all_neg_samples) > target_pos_count:
            neg_indices = np.random.choice(len(all_neg_samples), target_pos_count, replace=False)
            all_neg_samples = all_neg_samples[neg_indices]
        
        # 创建标签
        pos_labels = np.ones(len(pos_samples))
        neg_labels = np.zeros(len(all_neg_samples))
        
        # 合并样本和标签
        X = np.vstack([pos_samples, all_neg_samples])
        y = np.concatenate([pos_labels, neg_labels])
        
        # 随机打乱
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def modified_stratified_sampling(self, target_label, neg_pos_ratio=3.0, neg_label_count=None, verbose=True):
        """
        修改版分层采样策略 - 总体正负比例1:3，且各负类样本均匀分布
        
        参数:
            target_label: 目标标签（正类）
            neg_pos_ratio: 负样本与正样本的总体比例，默认3.0
            neg_label_count: 使用的负类标签数量，None表示使用所有
            verbose: 是否打印详细统计信息
            
        返回:
            samples: 特征数据
            labels: 对应的标签
        """
        if target_label not in self.valid_labels:
            raise ValueError(f"标签 {target_label} 不在有效标签列表中")
        
        # 加载正样本
        pos_file = self.get_file_path(target_label)
        if not pos_file:
            raise ValueError(f"找不到标签 {target_label} 的数据文件")
            
        pos_samples = np.load(pos_file)
        pos_count = len(pos_samples)
        
        # 计算需要的总负样本数量
        total_neg_count_needed = int(pos_count * neg_pos_ratio)
        
        # 收集所有可用的负类标签
        other_labels = [l for l in self.valid_labels if l != target_label]
        
        # 如果指定了负类标签数量，随机选择指定数量
        if neg_label_count is not None and neg_label_count < len(other_labels):
            other_labels = np.random.choice(other_labels, neg_label_count, replace=False)
        
        # 计算每个负类标签应该贡献的样本数量
        neg_count_per_label = total_neg_count_needed // len(other_labels)
        
        # 处理可能的余数
        remainder = total_neg_count_needed % len(other_labels)
        
        neg_samples = []
        label_sample_counts = {}  # 用于记录每个负类标签的样本数
        available_counts = {}     # 用于记录每个负类标签的可用样本数
        
        # 从每个负类标签中均匀采样
        for i, other_label in enumerate(other_labels):
            neg_file = self.get_file_path(other_label)
            if not neg_file:
                continue
                
            other_samples = np.load(neg_file)
            available_counts[other_label] = len(other_samples)
            
            # 计算本标签需要的样本数（考虑余数分配）
            if i < remainder:
                samples_needed = min(len(other_samples), neg_count_per_label + 1)
            else:
                samples_needed = min(len(other_samples), neg_count_per_label)
            
            if samples_needed < len(other_samples):
                # 随机选择子集
                indices = np.random.choice(len(other_samples), samples_needed, replace=False)
                selected_samples = other_samples[indices]
            else:
                selected_samples = other_samples
            
            neg_samples.append(selected_samples)
            label_sample_counts[other_label] = len(selected_samples)
        
        # 合并所有负样本
        if neg_samples:
            all_neg_samples = np.vstack(neg_samples)
        else:
            all_neg_samples = np.array([]).reshape(0, pos_samples.shape[1])
        
        # 创建标签
        pos_labels = np.ones(len(pos_samples))
        neg_labels = np.zeros(len(all_neg_samples))
        
        # 合并样本和标签
        X = np.vstack([pos_samples, all_neg_samples])
        y = np.concatenate([pos_labels, neg_labels])
        
        # 随机打乱
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 打印详细统计信息
        if verbose:
            print(f"\n{'='*50}")
            print(f"分层采样统计 - 标签 {target_label} (正类)")
            print(f"{'='*50}")
            print(f"正样本数量: {len(pos_samples)}")
            print(f"负样本总数: {len(all_neg_samples)}")
            print(f"实际正负比例: 1:{len(all_neg_samples)/len(pos_samples):.2f}")
            print(f"目标负样本总数: {total_neg_count_needed} (正负比例 1:{neg_pos_ratio})")
            print(f"使用的负类标签数量: {len(label_sample_counts)}")
            print(f"每个标签目标样本数: {neg_count_per_label} (余数: {remainder})")
            
            print("\n负类标签采样明细:")
            if label_sample_counts:
                # 按样本数量排序输出
                sorted_labels = sorted(label_sample_counts.items(), key=lambda x: x[1], reverse=True)
                for label, count in sorted_labels:
                    available = available_counts.get(label, 0)
                    usage_percent = (count / available * 100) if available > 0 else 0
                    print(f"  - 标签 {label}: {count} 样本 (可用: {available}, 使用率: {usage_percent:.1f}%)")
                
                # 计算统计数据
                counts = list(label_sample_counts.values())
                print(f"\n负类标签样本统计:")
                print(f"  - 平均每个标签: {np.mean(counts):.1f} 样本")
                print(f"  - 中位数: {np.median(counts):.1f}")
                print(f"  - 标准差: {np.std(counts):.1f}")
                print(f"  - 最小值: {min(counts)} (标签 {min(label_sample_counts, key=lambda k: label_sample_counts[k])})")
                print(f"  - 最大值: {max(counts)} (标签 {max(label_sample_counts, key=lambda k: label_sample_counts[k])})")
                print(f"  - 最大/最小比例: {max(counts)/min(counts):.2f}")
            
            print(f"{'='*50}")
        
        return X, y

class BrainVoxelDataManager:
    """脑体素数据管理器，集成训练、测试和验证集的访问"""
    
    def __init__(self, train_dir, test_dir, val_dir):
        """
        初始化数据管理器
        
        参数:
            train_dir: 训练集目录
            test_dir: 测试集目录
            val_dir: 验证集目录
        """
        self.train_sampler = BrainVoxelSampler(train_dir)
        self.test_sampler = BrainVoxelSampler(test_dir)
        self.val_sampler = BrainVoxelSampler(val_dir)
        
        # 收集所有有效标签
        self.valid_labels = sorted(list(set(
            self.train_sampler.valid_labels + 
            self.test_sampler.valid_labels + 
            self.val_sampler.valid_labels
        )))
    
    def get_datasets(self, target_label, sampling_strategy='modified_stratified', apply_pca_flag=True, 
                    n_components=0, norm=True, pca_model=None, **kwargs):
        """
        获取完整的训练、测试和验证数据集
        
        参数:
            target_label: 目标标签ID
            sampling_strategy: 采样策略，可选'balanced'、'stratified'、'modified_stratified'、'hard_negative'
            apply_pca_flag: 是否应用PCA降维
            n_components: PCA保留的主成分数量，0表示自动选择
            norm: 是否进行标准化处理
            pca_model: 预训练的PCA模型
            **kwargs: 传递给采样器的额外参数
            
        返回:
            datasets: 包含训练、测试和验证集的字典
        """
        # 选择采样方法
        if sampling_strategy == 'balanced':
            train_samples, train_labels = self.train_sampler.balanced_sampling(target_label, **kwargs)
            test_samples, test_labels = self.test_sampler.balanced_sampling(target_label, **kwargs)
            val_samples, val_labels = self.val_sampler.balanced_sampling(target_label, **kwargs)
        elif sampling_strategy == 'stratified':
            train_samples, train_labels = self.train_sampler.stratified_sampling(target_label, **kwargs)
            test_samples, test_labels = self.test_sampler.stratified_sampling(target_label, **kwargs)
            val_samples, val_labels = self.val_sampler.stratified_sampling(target_label, **kwargs)
        elif sampling_strategy == 'modified_stratified':
            train_samples, train_labels = self.train_sampler.modified_stratified_sampling(target_label, **kwargs)
            test_samples, test_labels = self.test_sampler.modified_stratified_sampling(target_label, **kwargs)
            val_samples, val_labels = self.val_sampler.modified_stratified_sampling(target_label, **kwargs)
        elif sampling_strategy == 'hard_negative':
            # 注意：难例采样需要已训练的模型
            if 'model' not in kwargs or 'device' not in kwargs:
                raise ValueError("难例采样需要提供model和device参数")
                
            train_samples, train_labels = self.train_sampler.hard_negative_mining(target_label, **kwargs)
            test_samples, test_labels = self.test_sampler.balanced_sampling(target_label)  # 测试集通常使用平衡采样
            val_samples, val_labels = self.val_sampler.balanced_sampling(target_label)     # 验证集通常使用平衡采样
        else:
            raise ValueError(f"不支持的采样策略: {sampling_strategy}")
            
        # 应用PCA（如果需要）
        if apply_pca_flag:
            # 导入PCA
            from sklearn.decomposition import PCA
            
            # 合并所有数据进行PCA拟合
            all_samples = np.vstack([train_samples, test_samples, val_samples])
            
            if pca_model is None and n_components > 0:
                # 如果没有提供PCA模型，且指定了主成分数量，则训练一个新的
                pca_model = PCA(n_components=n_components)
                pca_model.fit(all_samples)
            elif pca_model is None and n_components == 0:
                # 自动选择主成分数量
                from utils import analyze_pca_variance
                n_components, _, _ = analyze_pca_variance(all_samples, plot=True)
                pca_model = PCA(n_components=n_components)
                pca_model.fit(all_samples)
                
            # 应用PCA变换
            train_samples = pca_model.transform(train_samples)
            test_samples = pca_model.transform(test_samples)
            val_samples = pca_model.transform(val_samples)
            
            # 应用标准化（如果需要）
            if norm:
                # 基于所有样本计算归一化参数
                all_transformed = np.vstack([train_samples, test_samples, val_samples])
                mins = np.min(all_transformed, axis=0)
                maxs = np.max(all_transformed, axis=0)
                ranges = maxs - mins + 1e-10  # 避免除零
                
                # 应用归一化
                train_samples = (train_samples - mins) / ranges
                test_samples = (test_samples - mins) / ranges
                val_samples = (val_samples - mins) / ranges
            
            feature_dim = train_samples.shape[1]
        else:
            feature_dim = train_samples.shape[1]
        
        return {
            'train_samples': train_samples,
            'train_labels': train_labels,
            'test_samples': test_samples,
            'test_labels': test_labels,
            'val_samples': val_samples,
            'val_labels': val_labels,
            'feature_dim': feature_dim,
            'pca_model': pca_model
        }