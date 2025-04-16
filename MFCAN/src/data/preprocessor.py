import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import h5py
import logging
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config_path):
        """
        初始化预处理器
        
        参数:
            config_path: 配置文件路径
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 配置日志
        self._setup_logging()
        
        logger.info(f"初始化预处理器，配置文件: {config_path}")
        
        # 加载配置
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("成功加载配置文件")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
            
        # 验证必要的配置参数
        self._validate_config()
            
        self.normalization = self.config.get('normalization', 'standard')  # 'standard', 'minmax', 'robust'
        self.apply_pca = self.config.get('apply_pca', False)
        self.n_components = self.config.get('n_components', 0)  # 0表示自动选择
        self.output_dir = self.config.get('output_dir', 'results/preprocessed')
        
        # 初始化转换器
        self.scalers = {}
        self.pca_models = {}
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"输出目录: {self.output_dir}")
        
        # 特征统计信息
        self.feature_stats = {}
        
    def _setup_logging(self):
        """配置日志记录"""
        log_dir = os.path.join("logs", "preprocessing")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"preprocessor_{self.timestamp}.log")
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到Logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        
    def _validate_config(self):
        """验证配置是否有效"""
        required_fields = ["output_dir"]
        for field in required_fields:
            if field not in self.config:
                error_msg = f"配置文件缺少必要字段: {field}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        # 验证normalization参数
        if 'normalization' in self.config and self.config['normalization'] not in ['standard', 'minmax', 'robust']:
            error_msg = f"无效的归一化方法: {self.config['normalization']}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("配置文件有效")
        
    def fit_transform(self, data_dict, feature_groups=None):
        """
        对数据进行拟合和转换
        
        参数:
            data_dict: 包含各集合特征的字典 {'train': X_train, 'val': X_val, 'test': X_test}
            feature_groups: 特征组字典，格式为 {'group_name': feature_array}
            
        返回:
            transformed_data: 转换后的数据字典
        """
        logger.info("开始特征转换")
        transformed_data = {}
        
        # 处理原始特征
        for split, features in data_dict.items():
            logger.info(f"处理 {split} 数据集: {features.shape}")
            if split == 'train':
                # 在训练集上拟合并转换
                transformed_data[split] = self._fit_transform_split(features, 'all')
                # 计算特征统计信息
                self._compute_feature_stats(features, split, 'all')
            else:
                # 在验证集和测试集上只进行转换
                transformed_data[split] = self._transform_split(features, 'all')
        
        # 处理特征组
        if feature_groups:
            for group_name, group_data in feature_groups.items():
                logger.info(f"处理特征组 {group_name}")
                group_transformed = {}
                for split, features in group_data.items():
                    logger.info(f"  - {split} 数据: {features.shape}")
                    if split == 'train':
                        group_transformed[split] = self._fit_transform_split(features, group_name)
                        # 计算特征组统计信息
                        self._compute_feature_stats(features, split, group_name)
                    else:
                        group_transformed[split] = self._transform_split(features, group_name)
                transformed_data[f"group_{group_name}"] = group_transformed
        
        logger.info("特征转换完成")
        return transformed_data
    
    def _fit_transform_split(self, features, group_name):
        """对单个数据集进行拟合和转换"""
        logger.debug(f"拟合并转换 {group_name} 组 {features.shape}")
        
        # 标准化
        scaler = self._get_scaler()
        try:
            scaled_features = scaler.fit_transform(features)
            self.scalers[group_name] = scaler
            logger.debug(f"标准化完成: {scaled_features.shape}")
        except Exception as e:
            logger.error(f"标准化失败: {e}")
            raise
        
        # PCA降维(如果需要)
        if self.apply_pca:
            try:
                if self.n_components == 0:
                    # 自动选择主成分数量
                    optimal_n, _, _ = self.analyze_pca_variance(
                        scaled_features, 
                        save_path=os.path.join(self.output_dir, f"{group_name}_pca_analysis.png")
                    )
                    n_components = optimal_n
                    logger.info(f"自动选择 {n_components} 个主成分 (组: {group_name})")
                else:
                    n_components = min(self.n_components, scaled_features.shape[1], scaled_features.shape[0])
                    logger.info(f"使用配置的 {n_components} 个主成分 (组: {group_name})")
                
                pca = PCA(n_components=n_components)
                transformed_features = pca.fit_transform(scaled_features)
                self.pca_models[group_name] = pca
                
                # 保存PCA解释方差比
                self._save_pca_variance_explained(pca, group_name)
                logger.debug(f"PCA降维完成: {transformed_features.shape}")
                
                return transformed_features
            except Exception as e:
                logger.error(f"PCA降维失败: {e}")
                raise
        else:
            return scaled_features
    
    def _transform_split(self, features, group_name):
        """使用已拟合的转换器对数据进行转换"""
        logger.debug(f"转换 {group_name} 组 {features.shape}")
        
        if group_name not in self.scalers:
            error_msg = f"Scaler for group '{group_name}' has not been fitted yet"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 标准化
        try:
            scaled_features = self.scalers[group_name].transform(features)
        except Exception as e:
            logger.error(f"标准化转换失败: {e}")
            raise
        
        # PCA降维(如果已拟合)
        if self.apply_pca and group_name in self.pca_models:
            try:
                transformed_features = self.pca_models[group_name].transform(scaled_features)
                logger.debug(f"PCA转换完成: {transformed_features.shape}")
                return transformed_features
            except Exception as e:
                logger.error(f"PCA转换失败: {e}")
                raise
        else:
            return scaled_features
    
    def _compute_feature_stats(self, features, split, group_name):
        """计算特征统计信息"""
        logger.debug(f"计算特征统计信息: {split}/{group_name}")
        
        if group_name not in self.feature_stats:
            self.feature_stats[group_name] = {}
            
        stats = {
            "count": features.shape[0],
            "feature_dim": features.shape[1],
            "min": np.min(features, axis=0).tolist() if features.size > 0 else [],
            "max": np.max(features, axis=0).tolist() if features.size > 0 else [],
            "mean": np.mean(features, axis=0).tolist() if features.size > 0 else [],
            "std": np.std(features, axis=0).tolist() if features.size > 0 else [],
            "missing_values": int(np.isnan(features).sum()),
            "zeros_count": int((features == 0).sum()),
            "timestamp": self.timestamp
        }
        
        self.feature_stats[group_name][split] = stats
        logger.debug(f"特征统计完成: {len(stats)} 项")
    
    def _get_scaler(self):
        """根据配置选择合适的标准化器"""
        if self.normalization == 'minmax':
            logger.debug("使用MinMaxScaler")
            return MinMaxScaler()
        elif self.normalization == 'robust':
            logger.debug("使用RobustScaler")
            return RobustScaler()
        else:  # 默认使用StandardScaler
            logger.debug("使用StandardScaler")
            return StandardScaler()
    
    def analyze_pca_variance(self, X, max_components=None, plot=True, save_path=None):
        """分析PCA方差解释率，找到合适的降维维度"""
        logger.info("分析PCA方差解释率")
        
        # 确定最大主成分数
        if max_components is None:
            max_components = min(X.shape[0], X.shape[1])
        else:
            max_components = min(max_components, X.shape[0], X.shape[1])
        
        # 计算所有可能的主成分
        try:
            pca = PCA(n_components=max_components)
            pca.fit(X)
        except Exception as e:
            logger.error(f"PCA拟合失败: {e}")
            raise
        
        # 计算累积解释方差
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 寻找方差解释率达到95%的拐点
        threshold = 0.95
        optimal_n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
        
        # 寻找拐点（斜率变化最大的点）
        gradient = np.gradient(explained_variance_ratio)
        gradient_of_gradient = np.gradient(gradient)
        elbow_index = np.argmax(np.abs(gradient_of_gradient))
        elbow_n_components = elbow_index + 1
        
        if plot:
            try:
                plt.figure(figsize=(12, 6))
            
                # Plot Explained Variance Ratio
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(explained_variance_ratio) + 1), 
                        explained_variance_ratio, 'bo-', markersize=4)
                plt.axvline(x=elbow_n_components, color='r', linestyle='--', 
                            label=f'Elbow Point: {elbow_n_components} Components')
                plt.xlabel('Number of Principal Components')
                plt.ylabel('Explained Variance Ratio')
                plt.title('Explained Variance Ratio per Principal Component')
                plt.grid(True)
                plt.legend()
            
                # Plot Cumulative Explained Variance
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
                        cumulative_variance_ratio, 'ro-', markersize=4)
                plt.axhline(y=threshold, color='g', linestyle='--', 
                            label=f'{threshold*100}% Variance')
                plt.axvline(x=optimal_n_components, color='b', linestyle='--', 
                            label=f'Threshold Components: {optimal_n_components}')
                plt.xlabel('Number of Principal Components')
                plt.ylabel('Cumulative Explained Variance Ratio')
                plt.title('Cumulative Explained Variance Ratio')
                plt.grid(True)
                plt.legend()
            
                plt.tight_layout()
            
                if save_path:
                    plt.savefig(save_path)
                    plt.close()
                    logger.info(f"PCA方差分析图保存至 {save_path}")
                else:
                    plt.show()
            except Exception as e:
                logger.error(f"创建PCA方差可视化失败: {e}")
                
        logger.info(f"方差拐点对应的主成分数量: {elbow_n_components}")
        logger.info(f"达到{threshold*100}%方差解释率需要的主成分数量: {optimal_n_components}")
        logger.info(f"前{optimal_n_components}个主成分解释了总方差的{cumulative_variance_ratio[optimal_n_components-1]*100:.2f}%")
        
        # 使用保留95%信息的维度
        return optimal_n_components, explained_variance_ratio, cumulative_variance_ratio
    
    def _save_pca_variance_explained(self, pca_model, group_name):
        """保存PCA方差解释比例信息"""
        variance_explained = {
            'n_components': pca_model.n_components_,
            'explained_variance_ratio': pca_model.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca_model.explained_variance_ratio_).tolist(),
            'total_variance_explained': float(sum(pca_model.explained_variance_ratio_))
        }
        
        try:
            output_path = os.path.join(self.output_dir, f"{group_name}_pca_variance.json")
            with open(output_path, 'w') as f:
                json.dump(variance_explained, f, indent=4)
            logger.info(f"PCA方差解释信息已保存至 {output_path}")
        except Exception as e:
            logger.error(f"保存PCA方差信息失败: {e}")
    
    def save_transformers(self, output_path):
        """保存所有变换器的状态信息"""
        transformer_info = {
            'timestamp': self.timestamp,
            'normalization': self.normalization,
            'apply_pca': self.apply_pca,
            'scalers': {name: type(scaler).__name__ for name, scaler in self.scalers.items()},
            'pca_models': {
                name: {
                    'n_components': model.n_components_,
                    'variance_explained': float(sum(model.explained_variance_ratio_))
                } for name, model in self.pca_models.items()
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(transformer_info, f, indent=4)
            logger.info(f"转换器信息已保存至 {output_path}")
        except Exception as e:
            logger.error(f"保存转换器信息失败: {e}")
            
    def save_feature_stats(self, output_path):
        """保存特征统计信息"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.feature_stats, f, indent=4)
            logger.info(f"特征统计信息已保存至 {output_path}")
        except Exception as e:
            logger.error(f"保存特征统计信息失败: {e}")
    
    def save_transformed_data(self, transformed_data, labels_dict, output_path):
        """将转换后的数据保存到HDF5文件"""
        logger.info(f"保存转换后的数据到 {output_path}")
        
        try:
            with h5py.File(output_path, 'w') as f:
                # 添加元数据
                f.attrs['timestamp'] = self.timestamp
                f.attrs['normalization'] = self.normalization
                f.attrs['apply_pca'] = str(self.apply_pca)
                
                # 保存处理后的特征和标签
                for key, data in transformed_data.items():
                    if isinstance(data, dict):  # 特征组
                        logger.debug(f"保存特征组: {key}")
                        for split, features in data.items():
                            # 为每个split创建一个组
                            group = f.create_group(f'{key}/{split}')
                            group.create_dataset('features', data=features)
                            # 只有当split在labels_dict中时才保存标签
                            if split in labels_dict:
                                group.create_dataset('labels', data=labels_dict[split])
                            else:
                                logger.warning(f"未找到 {split} 的标签数据")
                    else:  # 主数据集
                        # 找出key对应的split
                        if key in ['train', 'val', 'test']:
                            split = key
                            logger.debug(f"保存主数据集: {split}")
                            group = f.create_group(split)
                            group.create_dataset('features', data=data)
                            if split in labels_dict:
                                group.create_dataset('labels', data=labels_dict[split])
                            else:
                                logger.warning(f"未找到 {split} 的标签数据")
                        else:
                            logger.warning(f"未知的数据键: {key}")
            
            logger.info("数据保存完成")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise
    
    def visualize_features(self, output_dir=None):
        """可视化特征统计信息"""
        if not output_dir:
            output_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始特征可视化, 输出到 {output_dir}")
        
        # 对每个特征组进行统计可视化
        for group_name, splits_stats in self.feature_stats.items():
            logger.debug(f"可视化特征组: {group_name}")
            
            for split, stats in splits_stats.items():
                # 绘制特征分布
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # 均值分布
                    plt.subplot(2, 2, 1)
                    plt.hist(stats['mean'], bins=30, alpha=0.7)
                    plt.title(f'{split} - {group_name} 均值分布')
                    plt.xlabel('均值')
                    plt.ylabel('频率')
                    plt.grid(True, alpha=0.3)
                    
                    # 标准差分布
                    plt.subplot(2, 2, 2)
                    plt.hist(stats['std'], bins=30, alpha=0.7)
                    plt.title(f'{split} - {group_name} 标准差分布')
                    plt.xlabel('标准差')
                    plt.ylabel('频率')
                    plt.grid(True, alpha=0.3)
                    
                    # 最小值分布
                    plt.subplot(2, 2, 3)
                    plt.hist(stats['min'], bins=30, alpha=0.7)
                    plt.title(f'{split} - {group_name} 最小值分布')
                    plt.xlabel('最小值')
                    plt.ylabel('频率')
                    plt.grid(True, alpha=0.3)
                    
                    # 最大值分布
                    plt.subplot(2, 2, 4)
                    plt.hist(stats['max'], bins=30, alpha=0.7)
                    plt.title(f'{split} - {group_name} 最大值分布')
                    plt.xlabel('最大值')
                    plt.ylabel('频率')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # 保存图表
                    viz_path = os.path.join(output_dir, f"{group_name}_{split}_distribution.png")
                    plt.savefig(viz_path)
                    plt.close()
                    logger.info(f"特征分布可视化已保存至 {viz_path}")
                    
                except Exception as e:
                    logger.error(f"生成特征分布可视化失败: {e}")
                    continue
                
                # 只为少量特征绘制直方图 (前10个)
                try:
                    if len(stats['mean']) > 0:
                        n_features = min(10, len(stats['mean']))
                        
                        plt.figure(figsize=(15, n_features * 2))
                        
                        for i in range(n_features):
                            plt.subplot(n_features, 1, i+1)
                            values = np.linspace(stats['min'][i], stats['max'][i], 1000)
                            plt.axvline(stats['mean'][i], color='r', linestyle='-', label='均值')
                            plt.axvline(stats['mean'][i] - stats['std'][i], color='g', linestyle='--', label='均值-标准差')
                            plt.axvline(stats['mean'][i] + stats['std'][i], color='g', linestyle='--', label='均值+标准差')
                            plt.title(f'特征 #{i} 统计信息')
                            plt.xlabel('值')
                            plt.grid(True, alpha=0.3)
                            if i == 0:
                                plt.legend()
                        
                        plt.tight_layout()
                        
                        # 保存图表
                        viz_path = os.path.join(output_dir, f"{group_name}_{split}_feature_details.png")
                        plt.savefig(viz_path)
                        plt.close()
                        logger.info(f"特征详细可视化已保存至 {viz_path}")
                        
                except Exception as e:
                    logger.error(f"生成特征详细可视化失败: {e}")
        
        logger.info("特征可视化完成")