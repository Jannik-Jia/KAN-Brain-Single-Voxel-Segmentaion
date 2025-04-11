# src/brain_voxel_dataloader.py

import os
import numpy as np
import time
from tqdm import tqdm

# 根据可用性选择合适的实现
from src.gpu_utils import xp, to_gpu, to_cpu, ensure_numpy, USE_GPU, has_cuda_ml

# 根据GPU可用性选择实现
if USE_GPU and has_cuda_ml():
    from cuml.decomposition import PCA
    from cuml.preprocessing import StandardScaler, MinMaxScaler
else:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.bilingual_logger import BilingualLogger
from src.feature_selector import FeatureSelector


class BrainVoxelDataLoader:
    """脑体素数据加载类，处理训练、测试和验证数据，支持GPU加速"""
    
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
        
        gpu_status = "启用" if USE_GPU else "未启用"
        self.logger.info(f"数据加载器初始化完成，GPU加速：{gpu_status}",
                        f"Data loader initialized, GPU acceleration: {gpu_status}")
    



    def _load_data_from_dir(self, directory, desc="加载数据", max_samples_per_label=None):
        """从指定目录加载数据，可限制每个标签的最大样本数，修改以使用所有可用样本"""
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
                actual_samples = len(samples)
                
                # 限制每个标签的样本数
                if max_samples_per_label and len(samples) > max_samples_per_label:
                    # 随机选择max_samples_per_label个样本
                    np.random.seed(42)  # 保持随机结果一致
                    indices = np.random.choice(len(samples), max_samples_per_label, replace=False)
                    samples = samples[indices]
                    self.logger.info(f"标签{label_id}使用了{max_samples_per_label}/{actual_samples}个样本",
                                f"Label {label_id} using {max_samples_per_label}/{actual_samples} samples")
                else:
                    if max_samples_per_label:
                        self.logger.info(f"标签{label_id}所有样本都被使用：{actual_samples}个",
                                    f"Label {label_id} using all available samples: {actual_samples}")
                
                labels = np.ones(len(samples)) * label_id
                all_samples.append(samples)
                all_labels.append(labels)
        
        # 合并数据
        if all_samples:
            all_samples = np.vstack(all_samples)
            all_labels = np.concatenate(all_labels)
            
            # 转换到GPU（如果启用）
            all_samples = to_gpu(all_samples)
            all_labels = to_gpu(all_labels)
            
            return all_samples, all_labels
        else:
            # 返回空数组，也转换到GPU
            return to_gpu(np.array([])), to_gpu(np.array([]))

    def precompute_transformations(self, preprocess_configs):
        """
        预计算并缓存不同的数据变换，避免重复计算
        
        参数:
            preprocess_configs: 预处理配置列表，每个配置是一个字典
        
        返回:
            cached_data: 缓存的预处理数据字典
        """
        self.logger.info("开始预计算数据变换", "Starting precomputation of data transformations")
        
        # 创建缓存字典
        self.cached_data = {}
        
        # 确保原始数据已加载
        if self.train_samples is None:
            self.logger.warning("原始数据未加载，正在加载数据", "Raw data not loaded, loading now")
            self.load_all_data()
        
        # 为每种预处理配置计算变换
        for config in preprocess_configs:
            config_key = self._get_config_key(config)
            self.logger.info(f"计算配置: {config_key}", f"Computing for config: {config_key}")
            
            # 使用原有的预处理函数，但结果存储在缓存中
            if config.get('feature_selection') is None:
                processed_data = self.preprocess_data(
                    apply_pca=config.get('apply_pca', False),
                    n_components=config.get('n_components', 50),
                    normalization=config.get('normalization', 'standard'),
                    class_balance=config.get('class_balance', False),
                    target_samples=config.get('target_samples', 1000),
                    auto_pca_variance=config.get('auto_pca_variance', None),
                    scaling_before_pca=config.get('scaling_before_pca', True)
                )
            else:
                processed_data = self.preprocess_data_with_feature_selection(
                    apply_pca=config.get('apply_pca', False),
                    n_components=config.get('n_components', 50),
                    normalization=config.get('normalization', 'standard'),
                    class_balance=config.get('class_balance', False),
                    target_samples=config.get('target_samples', 1000),
                    feature_selection=config.get('feature_selection'),
                    selection_mode=config.get('selection_mode', 'threshold'),
                    selection_threshold=config.get('selection_threshold', 0.01),
                    max_features=config.get('max_features', 100),
                    l1_ratio=config.get('l1_ratio', 1.0),
                    scaling_before_selection=config.get('scaling_before_selection', True),
                    selection_metric=config.get('selection_metric', 'coefficient'),
                    auto_pca_variance=config.get('auto_pca_variance', None),
                    scaling_before_pca=config.get('scaling_before_pca', True)
                )
            
            # 存储到缓存
            self.cached_data[config_key] = processed_data
        
        self.logger.info(f"预计算完成，缓存了 {len(self.cached_data)} 种变换", 
                    f"Precomputation completed, cached {len(self.cached_data)} transformations")
        
        return self.cached_data

    def _get_config_key(self, config):
        """生成配置的唯一键"""
        key_parts = []
        # 添加关键配置参数到键
        key_parts.append(f"pca={config.get('apply_pca', False)}")
        
        if config.get('apply_pca', False):
            key_parts.append(f"comp={config.get('n_components', 0)}")
        
        key_parts.append(f"norm={config.get('normalization', 'none')}")
        
        if config.get('feature_selection') is not None:
            key_parts.append(f"fs={config.get('feature_selection')}")
        
        return "_".join(key_parts)

    def get_cached_data(self, config):
        """获取缓存的预处理数据，如果没有则计算"""
        config_key = self._get_config_key(config)
        
        if hasattr(self, 'cached_data') and config_key in self.cached_data:
            self.logger.info(f"使用缓存的数据变换: {config_key}", f"Using cached transformation: {config_key}")
            return self.cached_data[config_key]
        
        # 如果没有缓存，执行计算
        self.logger.info(f"缓存未命中，计算数据变换: {config_key}", 
                    f"Cache miss, computing transformation: {config_key}")
        
        if config.get('feature_selection') is None:
            processed_data = self.preprocess_data(
                apply_pca=config.get('apply_pca', False),
                n_components=config.get('n_components', 50),
                normalization=config.get('normalization', 'standard'),
                class_balance=config.get('class_balance', False),
                target_samples=config.get('target_samples', 1000),
                auto_pca_variance=config.get('auto_pca_variance', None),
                scaling_before_pca=config.get('scaling_before_pca', True)
            )
        else:
            processed_data = self.preprocess_data_with_feature_selection(
                apply_pca=config.get('apply_pca', False),
                n_components=config.get('n_components', 50),
                normalization=config.get('normalization', 'standard'),
                class_balance=config.get('class_balance', False),
                target_samples=config.get('target_samples', 1000),
                feature_selection=config.get('feature_selection'),
                selection_mode=config.get('selection_mode', 'threshold'),
                selection_threshold=config.get('selection_threshold', 0.01),
                max_features=config.get('max_features', 100),
                l1_ratio=config.get('l1_ratio', 1.0),
                scaling_before_selection=config.get('scaling_before_selection', True),
                selection_metric=config.get('selection_metric', 'coefficient'),
                auto_pca_variance=config.get('auto_pca_variance', None),
                scaling_before_pca=config.get('scaling_before_pca', True)
            )
        
        # 动态创建缓存字典
        if not hasattr(self, 'cached_data'):
            self.cached_data = {}
        
        # 存储到缓存
        self.cached_data[config_key] = processed_data
        
        return processed_data

    
    # 修改 brain_voxel_dataloader.py 中的 load_all_data 方法
    def load_all_data(self, max_samples_per_label=None):
        """加载所有数据集"""
        self.logger.info(f"开始加载所有数据集，每个标签最多 {max_samples_per_label} 个样本", 
                    f"Start loading all datasets, max {max_samples_per_label} samples per label")
        
        # 加载训练集
        self.train_samples, self.train_labels = self._load_data_from_dir(
            self.train_dir, "加载训练集数据", max_samples_per_label)
        
        # 加载测试集
        self.test_samples, self.test_labels = self._load_data_from_dir(
            self.test_dir, "加载测试集数据", max_samples_per_label)
        
        # 加载验证集
        self.val_samples, self.val_labels = self._load_data_from_dir(
            self.val_dir, "加载验证集数据", max_samples_per_label)
        
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
        # 确保计算长度时使用CPU上的数据
        train_len = len(ensure_numpy(self.train_labels))
        test_len = len(ensure_numpy(self.test_labels))
        val_len = len(ensure_numpy(self.val_labels))
        
        self.logger.info(
            f"数据集统计信息:\n"
            f"训练集: {train_len} 个样本\n"
            f"测试集: {test_len} 个样本\n"
            f"验证集: {val_len} 个样本",
            
            f"Dataset statistics:\n"
            f"Training set: {train_len} samples\n"
            f"Test set: {test_len} samples\n"
            f"Validation set: {val_len} samples"
        )
    

    def preprocess_data(self, apply_pca=False, n_components=50,
                normalization='standard', class_balance=False,
                target_samples=1000, random_state=42,
                auto_pca_variance=0.95,
                scaling_before_pca=True,
                max_iter=1000,  # 新增参数
                tol=1e-4):  # 新增参数

        """
        预处理数据，优化版本避免重复标准化
        
        参数:
            apply_pca: 是否应用PCA降维
            n_components: PCA组件数量 (如果auto_pca_variance为None时使用)
            normalization: 标准化方法，'standard'或'minmax'或None
            class_balance: 是否进行类别平衡
            target_samples: 每个类别的目标样本数
            random_state: 随机种子
            auto_pca_variance: 如果不为None，自动寻找解释这一比例方差所需的组件数量
            scaling_before_pca: 是否在PCA前进行标准化 (避免重复标准化)
            max_iter: 特征选择算法的最大迭代次数
            tol: 特征选择算法的收敛阈值
        """
        processed_data = {}
        
        # 复制原始数据
        train_X = self.train_samples.copy() if hasattr(self.train_samples, 'copy') else self.train_samples
        test_X = self.test_samples.copy() if hasattr(self.test_samples, 'copy') else self.test_samples
        val_X = self.val_samples.copy() if hasattr(self.val_samples, 'copy') else self.val_samples
        
        train_y = self.train_labels.copy() if hasattr(self.train_labels, 'copy') else self.train_labels
        test_y = self.test_labels.copy() if hasattr(self.test_labels, 'copy') else self.test_labels
        val_y = self.val_labels.copy() if hasattr(self.val_labels, 'copy') else self.val_labels
        
        # 获取CPU数据用于处理
        train_X_cpu = ensure_numpy(train_X)
        test_X_cpu = ensure_numpy(test_X)
        val_X_cpu = ensure_numpy(val_X)
        
        # 优化：只标准化一次，根据需要在PCA前或PCA后进行
        already_normalized = False
        
        # 如果需要在PCA前标准化
        if scaling_before_pca and apply_pca and normalization:
            already_normalized = True
            if normalization == 'standard':
                self.logger.info("在PCA前应用标准化(Z-score)", "Applying standardization (Z-score) before PCA")
                self.scaler = StandardScaler()
                train_X_cpu = self.scaler.fit_transform(train_X_cpu)
                test_X_cpu = self.scaler.transform(test_X_cpu)
                val_X_cpu = self.scaler.transform(val_X_cpu)
            elif normalization == 'minmax':
                self.logger.info("在PCA前应用归一化(MinMax)", "Applying normalization (MinMax) before PCA")
                self.scaler = MinMaxScaler()
                train_X_cpu = self.scaler.fit_transform(train_X_cpu)
                test_X_cpu = self.scaler.transform(test_X_cpu)
                val_X_cpu = self.scaler.transform(val_X_cpu)
        
        # 应用PCA降维
        if apply_pca:
            # 自动寻找解释目标方差比例所需的组件数量
            if auto_pca_variance is not None:
                from sklearn.decomposition import PCA
                
                # 创建临时PCA对象用于分析方差解释率
                temp_pca = PCA(n_components=min(train_X_cpu.shape[1], 300))  # 使用较大的组件数先拟合
                temp_pca.fit(train_X_cpu)
                
                # 计算累积方差贡献率
                cumulative_variance_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
                
                # 找到第一个满足方差目标的组件数量
                n_components = np.argmax(cumulative_variance_ratio >= auto_pca_variance) + 1
                
                self.logger.info(f"自动选择了 {n_components} 个PCA组件，可解释 {auto_pca_variance*100:.1f}% 的方差",
                                f"Automatically selected {n_components} PCA components to explain {auto_pca_variance*100:.1f}% variance")
            
            self.logger.info(f"应用PCA降维至{n_components}个组件",
                            f"Applying PCA reduction to {n_components} components")
            
            # 创建并拟合PCA模型
            from sklearn.decomposition import PCA
            self.pca_model = PCA(n_components=n_components, random_state=random_state)
            train_X_cpu = self.pca_model.fit_transform(train_X_cpu)
            test_X_cpu = self.pca_model.transform(test_X_cpu)
            val_X_cpu = self.pca_model.transform(val_X_cpu)
            
            # 记录解释方差
            explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
            self.logger.info(f"PCA解释方差: {explained_variance:.4f}",
                            f"PCA explained variance: {explained_variance:.4f}")
        
        # 如果还没有标准化且需要标准化，在PCA后进行
        if not already_normalized and normalization:
            if normalization == 'standard':
                self.logger.info("应用标准化(Z-score)", "Applying standardization (Z-score)")
                self.scaler = StandardScaler()
                train_X_cpu = self.scaler.fit_transform(train_X_cpu)
                test_X_cpu = self.scaler.transform(test_X_cpu)
                val_X_cpu = self.scaler.transform(val_X_cpu)
            elif normalization == 'minmax':
                self.logger.info("应用归一化(MinMax)", "Applying normalization (MinMax)")
                self.scaler = MinMaxScaler()
                train_X_cpu = self.scaler.fit_transform(train_X_cpu)
                test_X_cpu = self.scaler.transform(test_X_cpu)
                val_X_cpu = self.scaler.transform(val_X_cpu)
        
        # 转回GPU
        train_X = to_gpu(train_X_cpu)
        test_X = to_gpu(test_X_cpu)
        val_X = to_gpu(val_X_cpu)
        
        # 3. 类别平衡（仅针对训练集）
        if class_balance:
            self.logger.info(f"应用类别平衡，目标每类{target_samples}个样本",
                        f"Applying class balancing, target {target_samples} samples per class")
            
            # 转换到CPU进行类别平衡
            train_X_cpu = ensure_numpy(train_X)
            train_y_cpu = ensure_numpy(train_y)
            
            balanced_X = []
            balanced_y = []
            
            # 获取唯一类别
            unique_classes = np.unique(train_y_cpu)
            
            # 对每个类别进行平衡
            for cls in unique_classes:
                cls_idx = np.where(train_y_cpu == cls)[0]
                cls_samples = train_X_cpu[cls_idx]
                cls_labels = train_y_cpu[cls_idx]
                
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
                        cls_samples = np.vstack([cls_samples, train_X_cpu[cls_idx[extra_idx]]])
                        cls_labels = np.concatenate([cls_labels, train_y_cpu[cls_idx[extra_idx]]])
                
                balanced_X.append(cls_samples)
                balanced_y.append(cls_labels)
            
            # 合并所有平衡后的类别
            train_X_cpu = np.vstack(balanced_X)
            train_y_cpu = np.concatenate(balanced_y)
            
            # 打乱数据
            np.random.seed(random_state)
            shuffle_idx = np.random.permutation(len(train_y_cpu))
            train_X_cpu = train_X_cpu[shuffle_idx]
            train_y_cpu = train_y_cpu[shuffle_idx]
            
            # 转回GPU
            train_X = to_gpu(train_X_cpu)
            train_y = to_gpu(train_y_cpu)
            
            self.logger.info(f"类别平衡后训练集大小: {len(train_y_cpu)}",
                        f"Training set size after balancing: {len(train_y_cpu)}")
        
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
                                            feature_selection='lasso', selection_mode='threshold',
                                            selection_threshold=0.01, max_features=100,
                                            l1_ratio=1.0, cv_folds=5,
                                            scaling_before_selection=True,
                                            selection_metric='coefficient',
                                            auto_pca_variance=0.95,
                                            scaling_before_pca=True,
                                            skip_pca=False):  # 新增参数，是否完全跳过PCA
        """
        预处理数据，包括可选的特征选择，优化版本避免重复标准化
        
        参数:
            apply_pca: 是否应用PCA降维
            n_components: PCA组件数量
            normalization: 标准化方法，'standard'或'minmax'或None
            class_balance: 是否进行类别平衡
            target_samples: 每个类别的目标样本数
            random_state: 随机种子
            feature_selection: 特征选择方法，'lasso'或'elastic_net'
            selection_mode: 特征选择模式，'threshold'或'fixed'
            selection_threshold: 特征选择阈值
            max_features: 最大特征数量
            l1_ratio: 弹性网络的L1比例
            cv_folds: 交叉验证折数
            scaling_before_selection: 是否在特征选择前进行标准化
            selection_metric: 特征重要性度量
            auto_pca_variance: 自动PCA方差比例
            scaling_before_pca: 是否在PCA前标准化
            skip_pca: 是否完全跳过PCA，即在特征选择前不进行PCA降维，直接对原始特征进行选择
            
        返回:
            预处理后的数据字典
        """
        # 如果skip_pca为True，强制将apply_pca设为False
        if skip_pca:
            apply_pca = False
            self.logger.info("完全跳过PCA，直接在原始特征上进行选择", 
                        "Completely skipping PCA, performing selection on original features")

        # 使用优化后的预处理方法，避免重复标准化
        processed_data = self.preprocess_data(
            apply_pca=apply_pca,
            n_components=n_components,
            normalization=normalization if not scaling_before_selection else None,  # 如果特征选择前会标准化，这里就不重复
            class_balance=class_balance,
            target_samples=target_samples,
            random_state=random_state,
            auto_pca_variance=auto_pca_variance,
            scaling_before_pca=scaling_before_pca
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
