import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import h5py
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.logging_utils import Logger

class FeatureAnalyzer:
    """特征分析工具，提供详细的特征统计和可视化"""
    
    def __init__(self, config_path=None, output_dir=None, logger=None):
        """
        初始化特征分析器
        
        参数:
            config_path: 配置文件路径
            output_dir: 输出目录
            logger: 日志记录器
        """
        # 加载配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # 设置输出目录
        self.output_dir = output_dir or self.config.get('output_dir', 'results/analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        if logger:
            self.logger = logger
        else:
            log_manager = Logger("FeatureAnalyzer", log_dir="logs/analysis")
            self.logger = log_manager.get_logger()
            
        self.logger.info(f"特征分析器初始化完成，输出目录: {self.output_dir}")
    
    def load_data(self, data_path):
        """
        加载HDF5格式的数据
        
        参数:
            data_path: 数据文件路径
            
        返回:
            data_dict: 包含各集合特征和标签的字典
        """
        self.logger.info(f"从 {data_path} 加载数据")
        data_dict = {}
        
        try:
            with h5py.File(data_path, 'r') as f:
                # 读取所有组
                def visit_group(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # 将数据集路径解析为键
                        parts = name.split('/')
                        
                        # 创建嵌套字典
                        current_dict = data_dict
                        for i, part in enumerate(parts[:-1]):
                            if part not in current_dict:
                                current_dict[part] = {}
                            current_dict = current_dict[part]
                        
                        # 添加数据集
                        current_dict[parts[-1]] = obj[()]
                        self.logger.debug(f"加载数据集: {name}, 形状: {obj[()].shape}")
                
                # 遍历所有组和数据集
                f.visititems(visit_group)
                
            self.logger.info("数据加载完成")
            return data_dict
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise
    
    def analyze_basic_stats(self, features, feature_names=None, group_name="all", save_prefix=None):
        """
        计算基本统计量
        
        参数:
            features: 特征矩阵
            feature_names: 特征名称列表
            group_name: 特征组名称
            save_prefix: 保存文件的前缀
            
        返回:
            stats_df: 统计数据框
        """
        self.logger.info(f"计算 {group_name} 组的基本统计量")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        # 计算基本统计量
        stats = {
            'Mean': np.mean(features, axis=0),
            'Std': np.std(features, axis=0),
            'Min': np.min(features, axis=0),
            'Max': np.max(features, axis=0),
            'Median': np.median(features, axis=0),
            '25%': np.percentile(features, 25, axis=0),
            '75%': np.percentile(features, 75, axis=0),
            'Skewness': stats.skew(features, axis=0),
            'Kurtosis': stats.kurtosis(features, axis=0),
            'Missing': np.isnan(features).sum(axis=0),
            'Zeros': (features == 0).sum(axis=0)
        }
        
        # 创建统计数据框
        stats_df = pd.DataFrame(stats, index=feature_names)
        
        # 保存统计结果
        if save_prefix:
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_basic_stats.csv")
            stats_df.to_csv(save_path)
            self.logger.info(f"基本统计量已保存至 {save_path}")
            
            # 可视化
            self._visualize_basic_stats(stats_df, group_name, save_prefix)
        
        return stats_df
    
    def _visualize_basic_stats(self, stats_df, group_name, save_prefix):
        """可视化基本统计量"""
        self.logger.info(f"可视化 {group_name} 组的基本统计量")
        
        try:
            # 创建统计量分布图
            plt.figure(figsize=(20, 15))
            
            # 均值分布
            plt.subplot(3, 3, 1)
            plt.hist(stats_df['Mean'], bins=30, alpha=0.7)
            plt.title(f'{group_name} 均值分布')
            plt.xlabel('均值')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 标准差分布
            plt.subplot(3, 3, 2)
            plt.hist(stats_df['Std'], bins=30, alpha=0.7)
            plt.title(f'{group_name} 标准差分布')
            plt.xlabel('标准差')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 中位数分布
            plt.subplot(3, 3, 3)
            plt.hist(stats_df['Median'], bins=30, alpha=0.7)
            plt.title(f'{group_name} 中位数分布')
            plt.xlabel('中位数')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 偏度分布
            plt.subplot(3, 3, 4)
            plt.hist(stats_df['Skewness'], bins=30, alpha=0.7)
            plt.title(f'{group_name} 偏度分布')
            plt.xlabel('偏度')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 峰度分布
            plt.subplot(3, 3, 5)
            plt.hist(stats_df['Kurtosis'], bins=30, alpha=0.7)
            plt.title(f'{group_name} 峰度分布')
            plt.xlabel('峰度')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 范围分布
            plt.subplot(3, 3, 6)
            ranges = stats_df['Max'] - stats_df['Min']
            plt.hist(ranges, bins=30, alpha=0.7)
            plt.title(f'{group_name} 范围分布')
            plt.xlabel('范围')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 异常值比例
            plt.subplot(3, 3, 7)
            z_scores = (stats_df['Mean'] - stats_df['Median']) / stats_df['Std']
            plt.hist(z_scores, bins=30, alpha=0.7)
            plt.title(f'{group_name} 均值-中位数偏差 (z-score)')
            plt.xlabel('Z-score')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 零值比例
            plt.subplot(3, 3, 8)
            zero_ratio = stats_df['Zeros'] / len(stats_df)
            plt.hist(zero_ratio, bins=30, alpha=0.7)
            plt.title(f'{group_name} 零值比例')
            plt.xlabel('零值比例')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 缺失值比例
            plt.subplot(3, 3, 9)
            missing_ratio = stats_df['Missing'] / len(stats_df)
            plt.hist(missing_ratio, bins=30, alpha=0.7)
            plt.title(f'{group_name} 缺失值比例')
            plt.xlabel('缺失值比例')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_stats_distribution.png")
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"统计量分布图已保存至 {save_path}")
            
        except Exception as e:
            self.logger.error(f"生成统计量可视化失败: {e}")
    
    def analyze_correlations(self, features, feature_names=None, group_name="all", method='pearson', save_prefix=None):
        """
        分析特征相关性
        
        参数:
            features: 特征矩阵
            feature_names: 特征名称列表
            group_name: 特征组名称
            method: 相关性计算方法，'pearson'、'spearman'或'kendall'
            save_prefix: 保存文件的前缀
            
        返回:
            corr_df: 相关性矩阵数据框
        """
        self.logger.info(f"计算 {group_name} 组的特征相关性 (方法: {method})")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        # 计算相关性矩阵
        features_df = pd.DataFrame(features, columns=feature_names)
        corr_df = features_df.corr(method=method)
        
        # 保存相关性矩阵
        if save_prefix:
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_correlation_{method}.csv")
            corr_df.to_csv(save_path)
            self.logger.info(f"相关性矩阵已保存至 {save_path}")
            
            # 可视化相关性矩阵
            self._visualize_correlation(corr_df, group_name, method, save_prefix)
        
        return corr_df
    
    def _visualize_correlation(self, corr_df, group_name, method, save_prefix):
        """可视化相关性矩阵"""
        self.logger.info(f"可视化 {group_name} 组的相关性矩阵")
        
        try:
            # 绘制相关性热图
            plt.figure(figsize=(16, 14))
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            
            # 生成样式
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            
            # 绘制热图
            sns.heatmap(corr_df, mask=mask, cmap=cmap, annot=False,
                       vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5)
            
            plt.title(f'{group_name} 特征相关性矩阵 ({method})')
            
            # 保存图表
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_correlation_{method}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"相关性热图已保存至 {save_path}")
            
            # 提取高相关特征对
            high_corr = self._extract_high_correlations(corr_df, threshold=0.8)
            if high_corr:
                high_corr_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_high_correlations.csv")
                pd.DataFrame(high_corr).to_csv(high_corr_path, index=False)
                self.logger.info(f"高相关特征对已保存至 {high_corr_path}")
            
        except Exception as e:
            self.logger.error(f"生成相关性可视化失败: {e}")
    
    def _extract_high_correlations(self, corr_df, threshold=0.8):
        """提取高相关的特征对"""
        high_corr = []
        
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                if abs(corr_df.iloc[i, j]) >= threshold:
                    high_corr.append({
                        'feature1': corr_df.columns[i],
                        'feature2': corr_df.columns[j],
                        'correlation': corr_df.iloc[i, j]
                    })
        
        return high_corr
    
    def analyze_class_separability(self, features, labels, feature_names=None, group_name="all", save_prefix=None):
        """
        分析特征对类别的区分能力
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_names: 特征名称列表
            group_name: 特征组名称
            save_prefix: 保存文件的前缀
            
        返回:
            separability_df: 可分性指标数据框
        """
        self.logger.info(f"分析 {group_name} 组特征的类别区分能力")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        # 计算ANOVA F值
        f_values, p_values = f_classif(features, labels)
        
        # 计算互信息
        try:
            mi_values = mutual_info_classif(features, labels)
        except Exception as e:
            self.logger.warning(f"计算互信息失败: {e}, 使用零值替代")
            mi_values = np.zeros(features.shape[1])
        
        # 创建结果数据框
        separability_df = pd.DataFrame({
            'Feature': feature_names,
            'F_value': f_values,
            'P_value': p_values,
            'Mutual_Info': mi_values
        })
        
        # 添加显著性标记
        separability_df['Significant'] = separability_df['P_value'] < 0.05
        
        # 按F值排序
        separability_df = separability_df.sort_values('F_value', ascending=False)
        
        # 保存结果
        if save_prefix:
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_class_separability.csv")
            separability_df.to_csv(save_path, index=False)
            self.logger.info(f"类别区分能力分析已保存至 {save_path}")
            
            # 可视化
            self._visualize_separability(separability_df, group_name, save_prefix)
        
        return separability_df
    
    def _visualize_separability(self, separability_df, group_name, save_prefix):
        """可视化类别区分能力"""
        self.logger.info(f"可视化 {group_name} 组的类别区分能力")
        
        try:
            # 显示前20个最具区分性的特征
            top_n = min(20, len(separability_df))
            top_features = separability_df.head(top_n)
            
            plt.figure(figsize=(12, 8))
            
            # F值柱状图
            plt.subplot(2, 1, 1)
            bars = plt.barh(top_features['Feature'][::-1], top_features['F_value'][::-1])
            
            # 为显著性特征添加不同颜色
            for i, significant in enumerate(top_features['Significant'][::-1]):
                if significant:
                    bars[i].set_color('green')
                else:
                    bars[i].set_color('red')
                    
            plt.title(f'{group_name} 组前{top_n}个特征的F值 (绿色=显著)')
            plt.xlabel('F值')
            plt.tight_layout()
            
            # 互信息柱状图
            plt.subplot(2, 1, 2)
            plt.barh(top_features['Feature'][::-1], top_features['Mutual_Info'][::-1])
            plt.title(f'{group_name} 组前{top_n}个特征的互信息值')
            plt.xlabel('互信息')
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_top_features.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"类别区分能力可视化已保存至 {save_path}")
            
            # 绘制P值直方图
            plt.figure(figsize=(10, 6))
            plt.hist(separability_df['P_value'], bins=50, alpha=0.7)
            plt.axvline(x=0.05, color='r', linestyle='--', label='p=0.05显著性阈值')
            plt.title(f'{group_name} 组P值分布')
            plt.xlabel('P值')
            plt.ylabel('频率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_pvalue_distribution.png")
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"P值分布图已保存至 {save_path}")
            
        except Exception as e:
            self.logger.error(f"生成类别区分能力可视化失败: {e}")
    
    def visualize_feature_distribution(self, features, labels, feature_ids=None, feature_names=None, 
                                       group_name="all", save_prefix=None):
        """
        可视化特征分布
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_ids: 要可视化的特征ID列表
            feature_names: 特征名称列表
            group_name: 特征组名称
            save_prefix: 保存文件的前缀
        """
        self.logger.info(f"可视化 {group_name} 组的特征分布")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        if feature_ids is None:
            # 默认可视化前10个特征
            feature_ids = range(min(10, features.shape[1]))
        
        # 获取唯一类别标签
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 为每个类别选择不同的颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        try:
            for feature_id in feature_ids:
                feature_name = feature_names[feature_id]
                plt.figure(figsize=(12, 6))
                
                # 绘制不同类别的直方图
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    plt.hist(features[mask, feature_id], bins=30, alpha=0.5, 
                            color=colors[i], label=f'Class {label}')
                
                plt.title(f'{feature_name} 在不同类别中的分布')
                plt.xlabel('特征值')
                plt.ylabel('频率')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存图表
                if save_prefix:
                    save_path = os.path.join(self.output_dir, 
                                            f"{save_prefix}_{group_name}_{feature_name.replace(' ', '_')}_distribution.png")
                    plt.savefig(save_path)
                    self.logger.info(f"特征分布图已保存至 {save_path}")
                
                plt.close()
            
        except Exception as e:
            self.logger.error(f"生成特征分布可视化失败: {e}")
    
    def visualize_pca_projection(self, features, labels, n_components=2, group_name="all", save_prefix=None):
        """
        PCA降维可视化

        参数:
            features: 特征矩阵
            labels: 类别标签
            n_components: 主成分数量
            group_name: 特征组名称
            save_prefix: 保存文件的前缀
        """
        self.logger.info(f"使用PCA对 {group_name} 组进行降维可视化")

        try:
            # 标准化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # PCA降维
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)

            # 获取唯一类别标签
            unique_labels = np.unique(labels)

            # 绘制散点图
            plt.figure(figsize=(12, 10))

            for label in unique_labels:
                mask = labels == label
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                            alpha=0.7, s=30, label=f'Class {label}')

            plt.title(f'{group_name} 组的PCA降维可视化')
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.grid(True, alpha=0.3)

            # 保存图表
            if save_prefix:
                save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_pca_visualization.png")
                plt.savefig(save_path, bbox_inches='tight')
                self.logger.info(f"PCA可视化已保存至 {save_path}")

            plt.close()

        except Exception as e:
            self.logger.error(f"生成PCA可视化失败: {e}")

        
    def analyze_group_importance(self, features_dict, labels, save_prefix=None):
        """
        分析不同特征组的重要性
        
        参数:
            features_dict: 特征组字典，键为组名，值为特征矩阵
            labels: 类别标签
            save_prefix: 保存文件的前缀
            
        返回:
            importance_df: 特征组重要性数据框
        """
        self.logger.info("分析不同特征组的重要性")
        
        group_metrics = []
        
        for group_name, features in features_dict.items():
            self.logger.info(f"分析特征组 {group_name}")
            
            try:
                # 计算F值
                f_values, p_values = f_classif(features, labels)
                
                # 计算平均F值和显著性特征比例
                avg_f_value = np.mean(f_values)
                significant_ratio = np.mean(p_values < 0.05)
                
                # 计算互信息
                mi_values = mutual_info_classif(features, labels)
                avg_mi = np.mean(mi_values)
                
                # 记录指标
                group_metrics.append({
                    'Group': group_name,
                    'Feature_Count': features.shape[1],
                    'Avg_F_Value': avg_f_value,
                    'Significant_Ratio': significant_ratio,
                    'Avg_Mutual_Info': avg_mi,
                    'Importance_Score': avg_f_value * significant_ratio  # 简单加权得分
                })
                
            except Exception as e:
                self.logger.error(f"分析特征组 {group_name} 时出错: {e}")
        
        # 创建结果数据框
        importance_df = pd.DataFrame(group_metrics)
        
        # 按重要性得分排序
        if not importance_df.empty:
            importance_df = importance_df.sort_values('Importance_Score', ascending=False)
        
        # 保存结果
        if save_prefix and not importance_df.empty:
            save_path = os.path.join(self.output_dir, f"{save_prefix}_group_importance.csv")
            importance_df.to_csv(save_path, index=False)
            self.logger.info(f"特征组重要性分析已保存至 {save_path}")
            
            # 可视化
            self._visualize_group_importance(importance_df, save_prefix)
        
        return importance_df
    
    def _visualize_group_importance(self, importance_df, save_prefix):
        """可视化特征组重要性"""
        self.logger.info("可视化特征组重要性")
        
        try:
            # 可视化每个指标
            plt.figure(figsize=(14, 10))
            
            # 重要性得分柱状图
            plt.subplot(2, 2, 1)
            plt.barh(importance_df['Group'], importance_df['Importance_Score'])
            plt.title('特征组重要性得分')
            plt.xlabel('重要性得分')
            plt.grid(True, alpha=0.3)
            
            # 平均F值柱状图
            plt.subplot(2, 2, 2)
            plt.barh(importance_df['Group'], importance_df['Avg_F_Value'])
            plt.title('特征组平均F值')
            plt.xlabel('平均F值')
            plt.grid(True, alpha=0.3)
            
            # 显著性特征比例柱状图
            plt.subplot(2, 2, 3)
            plt.barh(importance_df['Group'], importance_df['Significant_Ratio'])
            plt.title('特征组显著性特征比例')
            plt.xlabel('显著性特征比例')
            plt.grid(True, alpha=0.3)
            
            # 平均互信息柱状图
            plt.subplot(2, 2, 4)
            plt.barh(importance_df['Group'], importance_df['Avg_Mutual_Info'])
            plt.title('特征组平均互信息')
            plt.xlabel('平均互信息')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(self.output_dir, f"{save_prefix}_group_importance.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"特征组重要性可视化已保存至 {save_path}")
            
        except Exception as e:
            self.logger.error(f"生成特征组重要性可视化失败: {e}")
    

    def run_complete_analysis(self, data_path, output_prefix=None):
        """
        运行完整的特征分析流程
        
        参数:
            data_path: 数据文件路径
            output_prefix: 输出文件前缀
        """
        if output_prefix is None:
            output_prefix = "analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"开始完整特征分析, 输出前缀: {output_prefix}")
        
        # 加载数据
        data_dict = self.load_data(data_path)
        
        # 分析每个特征组
        group_features = {}
        
        # 收集训练集特征和标签
        train_features = None
        train_labels = None
        
        # 遍历数据字典提取特征组
        for key, value in data_dict.items():
            if isinstance(value, dict) and 'train' in value:
                # 特征组
                group_name = key
                if 'features' in value['train'] and 'labels' in value['train']:
                    group_features[group_name] = value['train']['features']
                    
                    # 保存训练集标签
                    if train_labels is None:
                        train_labels = value['train']['labels']
            
            # 主数据集
            elif key == 'train' and isinstance(value, dict):
                if 'features' in value and 'labels' in value:
                    train_features = value['features']
                    train_labels = value['labels']
        
        if train_features is None or train_labels is None:
            self.logger.error("未找到训练集特征或标签数据")
            return
        
        # 分析全部特征
        self.logger.info("分析全部特征...")
        self.analyze_basic_stats(train_features, group_name="all", save_prefix=output_prefix)
        self.analyze_correlations(train_features, group_name="all", save_prefix=output_prefix)
        self.analyze_class_separability(train_features, train_labels, group_name="all", save_prefix=output_prefix)
        self.visualize_pca_projection(train_features, train_labels, group_name="all", save_prefix=output_prefix)
        self.visualize_tsne_projection(train_features, train_labels, group_name="all", save_prefix=output_prefix)
        
        # 分析每个特征组
        for group_name, features in group_features.items():
            self.logger.info(f"分析特征组 {group_name}...")
            self.analyze_basic_stats(features, group_name=group_name, save_prefix=output_prefix)
            self.analyze_correlations(features, group_name=group_name, save_prefix=output_prefix)
            self.analyze_class_separability(features, train_labels, group_name=group_name, save_prefix=output_prefix)
            self.visualize_pca_projection(features, train_labels, group_name=group_name, save_prefix=output_prefix)
            self.visualize_tsne_projection(features, train_labels, group_name=group_name, save_prefix=output_prefix)
        
        # 分析特征组重要性
        if len(group_features) > 1:
            self.analyze_group_importance(group_features, train_labels, save_prefix=output_prefix)
        
        self.logger.info("完整特征分析完成")



    def visualize_tsne_projection(self, features, labels, perplexity=30, group_name="all", save_prefix=None):
        """
        t-SNE降维可视化
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            perplexity: t-SNE复杂度参数
            group_name: 特征组名称
            save_prefix: 保存文件的前缀
        """
        self.logger.info(f"使用t-SNE对 {group_name} 组进行降维可视化")
        
        try:
            # 获取特征子集（如果特征太多，只使用一个子集）
            if features.shape[1] > 50:
                # 使用PCA先降维
                self.logger.info(f"特征维度过高 ({features.shape[1]}), 先使用PCA降至50维")
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                pca = PCA(n_components=50)
                features_subset = pca.fit_transform(features_scaled)
            else:
                # 标准化
                scaler = StandardScaler()
                features_subset = scaler.fit_transform(features)
            
            # 如果数据量太大，随机选择子集
            if features.shape[0] > 5000:
                self.logger.info(f"样本数量过多 ({features.shape[0]}), 随机选择5000个样本")
                indices = np.random.choice(features.shape[0], 5000, replace=False)
                features_subset = features_subset[indices]
                labels_subset = labels[indices]
            else:
                labels_subset = labels
            
            # t-SNE降维
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            features_tsne = tsne.fit_transform(features_subset)
            
            # 获取唯一类别标签
            unique_labels = np.unique(labels_subset)
            n_classes = len(unique_labels)
            
            # 为每个类别选择不同的颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            # 绘制散点图
            plt.figure(figsize=(12, 10))
            
            for i, label in enumerate(unique_labels):
                mask = labels_subset == label
                plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                        alpha=0.7, s=30, color=colors[i], label=f'Class {label}')
            
            plt.title(f'{group_name} 组的t-SNE降维可视化 (perplexity={perplexity})')
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            if save_prefix:
                save_path = os.path.join(self.output_dir, f"{save_prefix}_{group_name}_tsne_visualization.png")
                plt.savefig(save_path, bbox_inches='tight')
                self.logger.info(f"t-SNE可视化已保存至 {save_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"生成t-SNE可视化失败: {e}")
        self.logger.error(f"错误详情: {str(e)}")


