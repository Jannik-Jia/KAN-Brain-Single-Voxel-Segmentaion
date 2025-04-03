# src/brain_voxel_dataloader.py

import os
import numpy as np
import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# To:
from src.bilingual_logger import BilingualLogger
from src.feature_selector import FeatureSelector


class BrainVoxelDataLoader:
    """脑体素数据加载类，处理训练、测试和验证数据"""
    
    def __init__(self, train_dir, test_dir, val_dir, logger=None):
        """
        初始化数据加载器
        
        参数:
            train_dir: 训练数据目录
            test_dir: 测试数据目录
            val_dir: 验证数据目录
            logger: 日志记录器
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.logger = logger if logger else BilingualLogger()
        
        # 存储加载的数据
        self.train_samples = None
        self.train_labels = None
        self.test_samples = None
        self.test_labels = None
        self.val_samples = None
        self.val_labels = None
        
        # 预处理模型
        self.pca_model = None
        self.scaler = None
        
        self.logger.info("数据加载器初始化完成", "Data loader initialized")
    
    def _load_data_from_dir(self, directory, desc="加载数据"):
        """从指定目录加载数据"""
        # 创建采样器
        sampler = BrainVoxelSampler(directory)
        valid_labels = sampler.valid_labels
        
        # 加载每个标签的数据
        all_samples = []
        all_labels = []
        
        for label_id in tqdm(valid_labels, desc=desc):
            file_path = sampler.get_file_path(label_id)
            if file_path:
                samples = np.load(file_path)
                labels = np.ones(len(samples)) * label_id
                all_samples.append(samples)
                all_labels.append(labels)
        
        # 合并数据
        if all_samples:
            all_samples = np.vstack(all_samples)
            all_labels = np.concatenate(all_labels)
            return all_samples, all_labels
        else:
            return np.array([]), np.array([])
    
    def load_all_data(self):
        """加载所有数据集"""
        self.logger.info("开始加载所有数据集", "Start loading all datasets")
        
        # 加载训练集
        self.train_samples, self.train_labels = self._load_data_from_dir(
            self.train_dir, "加载训练集数据")
        
        # 加载测试集
        self.test_samples, self.test_labels = self._load_data_from_dir(
            self.test_dir, "加载测试集数据")
        
        # 加载验证集
        self.val_samples, self.val_labels = self._load_data_from_dir(
            self.val_dir, "加载验证集数据")
        
        # 打印数据集统计信息
        self._print_dataset_stats()
        
        return {
            'train_samples': self.train_samples,
            'train_labels': self.train_labels,
            'test_samples': self.test_samples,
            'test_labels': self.test_labels,
            'val_samples': self.val_samples,
            'val_labels': self.val_labels
        }
    
    def _print_dataset_stats(self):
        """打印数据集统计信息"""
        self.logger.info(
            f"数据集统计信息:\n"
            f"训练集: {len(self.train_labels)} 个样本\n"
            f"测试集: {len(self.test_labels)} 个样本\n"
            f"验证集: {len(self.val_labels)} 个样本",
            
            f"Dataset statistics:\n"
            f"Training set: {len(self.train_labels)} samples\n"
            f"Test set: {len(self.test_labels)} samples\n"
            f"Validation set: {len(self.val_labels)} samples"
        )
    
    def preprocess_data(self, apply_pca=False, n_components=50, 
                       normalization='standard', class_balance=False,
                       target_samples=1000, random_state=42):
        """
        预处理数据
        
        参数:
            apply_pca: 是否应用PCA降维
            n_components: PCA组件数量
            normalization: 标准化方法，'standard'或'minmax'或None
            class_balance: 是否进行类别平衡
            target_samples: 每个类别的目标样本数
            random_state: 随机种子
        """
        processed_data = {}
        
        # 复制原始数据
        train_X = self.train_samples.copy()
        test_X = self.test_samples.copy()
        val_X = self.val_samples.copy()
        
        train_y = self.train_labels.copy()
        test_y = self.test_labels.copy()
        val_y = self.val_labels.copy()
        
        # 1. 应用PCA降维
        if apply_pca and n_components > 0:
            self.logger.info(f"应用PCA降维至{n_components}个组件", 
                            f"Applying PCA reduction to {n_components} components")
            
            # 创建并拟合PCA模型
            self.pca_model = PCA(n_components=n_components, random_state=random_state)
            train_X = self.pca_model.fit_transform(train_X)
            
            # 转换测试集和验证集
            test_X = self.pca_model.transform(test_X)
            val_X = self.pca_model.transform(val_X)
            
            # 记录解释方差
            explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
            self.logger.info(f"PCA解释方差: {explained_variance:.4f}",
                            f"PCA explained variance: {explained_variance:.4f}")
        
        # 2. 应用标准化
        if normalization == 'standard':
            self.logger.info("应用标准化(Z-score)", "Applying standardization (Z-score)")
            self.scaler = StandardScaler()
            train_X = self.scaler.fit_transform(train_X)
            test_X = self.scaler.transform(test_X)
            val_X = self.scaler.transform(val_X)
            
        elif normalization == 'minmax':
            self.logger.info("应用归一化(MinMax)", "Applying normalization (MinMax)")
            self.scaler = MinMaxScaler()
            train_X = self.scaler.fit_transform(train_X)
            test_X = self.scaler.transform(test_X)
            val_X = self.scaler.transform(val_X)
        
        # 3. 类别平衡（仅针对训练集）
        if class_balance:
            self.logger.info(f"应用类别平衡，目标每类{target_samples}个样本",
                           f"Applying class balancing, target {target_samples} samples per class")
            
            balanced_X = []
            balanced_y = []
            
            # 获取唯一类别
            unique_classes = np.unique(train_y)
            
            # 对每个类别进行平衡
            for cls in unique_classes:
                cls_idx = np.where(train_y == cls)[0]
                cls_samples = train_X[cls_idx]
                cls_labels = train_y[cls_idx]
                
                # 如果样本数量超过目标，随机下采样
                if len(cls_idx) > target_samples:
                    np.random.seed(random_state)
                    select_idx = np.random.choice(
                        len(cls_idx), target_samples, replace=False)
                    cls_samples = cls_samples[select_idx]
                    cls_labels = cls_labels[select_idx]
                
                # 如果样本数量不足目标，过采样
                elif len(cls_idx) < target_samples and len(cls_idx) > 0:
                    # 简单的复制采样
                    n_copies = target_samples // len(cls_idx)
                    remainder = target_samples % len(cls_idx)
                    
                    # 复制整数倍
                    if n_copies > 0:
                        cls_samples = np.repeat(cls_samples, n_copies, axis=0)
                        cls_labels = np.repeat(cls_labels, n_copies)
                    
                    # 处理余数
                    if remainder > 0:
                        np.random.seed(random_state)
                        extra_idx = np.random.choice(
                            len(cls_idx), remainder, replace=False)
                        cls_samples = np.vstack([cls_samples, train_X[cls_idx[extra_idx]]])
                        cls_labels = np.concatenate([cls_labels, train_y[cls_idx[extra_idx]]])
                
                balanced_X.append(cls_samples)
                balanced_y.append(cls_labels)
            
            # 合并所有平衡后的类别
            train_X = np.vstack(balanced_X)
            train_y = np.concatenate(balanced_y)
            
            # 打乱数据
            np.random.seed(random_state)
            shuffle_idx = np.random.permutation(len(train_y))
            train_X = train_X[shuffle_idx]
            train_y = train_y[shuffle_idx]
            
            self.logger.info(f"类别平衡后训练集大小: {len(train_y)}", 
                           f"Training set size after balancing: {len(train_y)}")
        
        # 保存处理后的数据
        processed_data['train_X'] = train_X
        processed_data['train_y'] = train_y
        processed_data['test_X'] = test_X
        processed_data['test_y'] = test_y
        processed_data['val_X'] = val_X
        processed_data['val_y'] = val_y
        
        return processed_data
    
    def preprocess_data_with_feature_selection(self, apply_pca=False, n_components=50, 
                                              normalization='standard', class_balance=False,
                                              target_samples=1000, random_state=42,
                                              feature_selection=None, selection_mode='threshold',
                                              selection_threshold=0.01, max_features=100,
                                              l1_ratio=1.0, cv_folds=5,
                                              scaling_before_selection=True,
                                              selection_metric='coefficient'):
        """
        预处理数据，包括可选的特征选择
        
        参数:
            apply_pca: 是否应用PCA降维
            n_components: PCA组件数量
            normalization: 标准化方法，'standard'或'minmax'或None
            class_balance: 是否进行类别平衡
            target_samples: 每个类别的目标样本数
            random_state: 随机种子
            feature_selection: 特征选择方法，None, 'lasso'或'elastic_net'
            selection_mode: 特征选择模式，'threshold'或'fixed'
            selection_threshold: 特征选择阈值
            max_features: 最大特征数量
            l1_ratio: 弹性网络的L1比例
            cv_folds: 交叉验证折数
            scaling_before_selection: 是否在特征选择前进行标准化
            selection_metric: 特征重要性度量
            
        返回:
            预处理后的数据字典
        """
        # 首先进行常规预处理
        processed_data = self.preprocess_data(
            apply_pca=apply_pca,
            n_components=n_components,
            normalization=normalization,
            class_balance=class_balance,
            target_samples=target_samples,
            random_state=random_state
        )
        
        # 如果不需要特征选择，直接返回
        if feature_selection is None:
            return processed_data
        
        self.logger.info(f"应用特征选择: 方法={feature_selection}, 模式={selection_mode}",
                      f"Applying feature selection: method={feature_selection}, mode={selection_mode}")
        
        # 创建特征选择器
        selector = FeatureSelector(
            method=feature_selection,
            selection_mode=selection_mode,
            selection_threshold=selection_threshold,
            max_features=max_features,
            l1_ratio=l1_ratio,
            cv_folds=cv_folds,
            random_state=random_state,
            scaling_before_selection=scaling_before_selection,
            selection_metric=selection_metric,
            logger=self.logger
        )
        
        # 拟合特征选择器并转换训练数据
        train_X_selected = selector.fit_transform(
            processed_data['train_X'], 
            processed_data['train_y']
        )
        
        # 转换测试和验证数据
        test_X_selected = selector.transform(processed_data['test_X'])
        val_X_selected = selector.transform(processed_data['val_X'])
        
        # 更新处理后的数据
        processed_data['train_X'] = train_X_selected
        processed_data['test_X'] = test_X_selected
        processed_data['val_X'] = val_X_selected
        processed_data['feature_selector'] = selector
        
        # 获取所选特征的索引和数量
        selected_indices = selector.get_selected_indices()
        self.logger.info(f"特征选择完成，选择了 {len(selected_indices)} 个特征",
                      f"Feature selection completed, selected {len(selected_indices)} features")
        
        return processed_data


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