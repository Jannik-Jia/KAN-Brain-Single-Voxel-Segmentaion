import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


import json
from datetime import datetime

class BasicAnalyzer:
    """基础统计分析工具，提供详细的特征统计和可视化"""
    
    def __init__(self, config_path=None, output_dir=None, logger=None):
        """
        初始化基础统计分析器
        
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
        self.output_dir = output_dir or self.config.get('output_dir', 'results/basic_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logger
        if logger:
            self.logger.info(f"基础统计分析器初始化完成，输出目录: {self.output_dir}")
            
        # 存储分析结果
        self.stats_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def compute_basic_stats(self, features, feature_names=None, feature_group="all"):
        """
        计算基本统计量
        
        参数:
            features: 特征矩阵
            feature_names: 特征名称列表
            feature_group: 特征组名称
            
        返回:
            stats_df: 统计数据框
        """
        if self.logger:
            self.logger.info(f"计算 {feature_group} 组的基本统计量")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]

        stats = {}

        # 计算基本统计量
        stats = {
            'Mean': np.mean(features, axis=0),
            'Std': np.std(features, axis=0),
            'Min': np.min(features, axis=0),
            'Max': np.max(features, axis=0),
            'Median': np.median(features, axis=0),
            '25%': np.percentile(features, 25, axis=0),
            '75%': np.percentile(features, 75, axis=0),
            'Skewness': scipy_stats.skew(features, axis=0),
            'Kurtosis': scipy_stats.kurtosis(features, axis=0),
            'Missing': np.isnan(features).sum(axis=0),
            'Zeros': (features == 0).sum(axis=0)
        }
        
        # 创建统计数据框
        stats_df = pd.DataFrame(stats, index=feature_names)
        
        # 存储结果
        self.stats_results[feature_group] = {
            'stats_df': stats_df,
            'features_shape': features.shape,
            'timestamp': self.timestamp
        }
        
        return stats_df
    
    def generate_stats_report(self, output_dir=None, prefix="all"):
        """
        生成统计分析报告
        
        参数:
            output_dir: 输出目录
            prefix: 文件前缀
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        if prefix not in self.stats_results:
            if self.logger:
                self.logger.warning(f"未找到 {prefix} 组的统计结果")
            return
        
        stats_df = self.stats_results[prefix]['stats_df']
        
        # 创建输出子目录
        basic_stats_dir = os.path.join(output_dir, 'basic_stats')
        os.makedirs(basic_stats_dir, exist_ok=True)
        
        # 保存统计结果
        csv_path = os.path.join(basic_stats_dir, f"{prefix}_basic_stats.csv")
        stats_df.to_csv(csv_path)
        
        if self.logger:
            self.logger.info(f"基本统计量已保存至 {csv_path}")
        
        # 生成可视化
        self._visualize_basic_stats(stats_df, prefix, basic_stats_dir)
        
        # 生成报告
        self._generate_stats_md_report(stats_df, prefix, basic_stats_dir)
    
    def _visualize_basic_stats(self, stats_df, group_name, output_dir):
        """可视化基本统计量"""
        if self.logger:
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
            z_scores = (stats_df['Mean'] - stats_df['Median']) / stats_df['Std'].replace(0, 1)  # 避免除以零
            plt.hist(z_scores, bins=30, alpha=0.7)
            plt.title(f'{group_name} 均值-中位数偏差 (z-score)')
            plt.xlabel('Z-score')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 零值比例
            plt.subplot(3, 3, 8)
            zero_ratio = stats_df['Zeros'] / self.stats_results[group_name]['features_shape'][0]
            plt.hist(zero_ratio, bins=30, alpha=0.7)
            plt.title(f'{group_name} 零值比例')
            plt.xlabel('零值比例')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 缺失值比例
            plt.subplot(3, 3, 9)
            missing_ratio = stats_df['Missing'] / self.stats_results[group_name]['features_shape'][0]
            plt.hist(missing_ratio, bins=30, alpha=0.7)
            plt.title(f'{group_name} 缺失值比例')
            plt.xlabel('缺失值比例')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(output_dir, f"{group_name}_stats_distribution.png")
            plt.savefig(save_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"统计量分布图已保存至 {save_path}")
                
            # 创建箱线图
            self._create_boxplots(stats_df, group_name, output_dir)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成统计量可视化失败: {e}")
    
    def _create_boxplots(self, stats_df, group_name, output_dir):
        """创建特征统计的箱线图"""
        try:
            # 选择一部分特征进行可视化(最多30个)
            num_features = min(30, len(stats_df))
            features_to_plot = stats_df.index[:num_features]
            
            # 绘制标准差箱线图
            plt.figure(figsize=(12, 8))
            std_data = stats_df.loc[features_to_plot, 'Std'].sort_values(ascending=False)
            plt.boxplot(std_data)
            plt.title(f'{group_name} 特征标准差分布')
            plt.xticks([1], ['标准差'])
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{group_name}_std_boxplot.png"))
            plt.close()
            
            # 绘制偏度和峰度箱线图
            plt.figure(figsize=(12, 8))
            skew_kurt_data = [
                stats_df.loc[features_to_plot, 'Skewness'],
                stats_df.loc[features_to_plot, 'Kurtosis']
            ]
            plt.boxplot(skew_kurt_data)
            plt.title(f'{group_name} 特征偏度和峰度分布')
            plt.xticks([1, 2], ['偏度', '峰度'])
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{group_name}_skew_kurt_boxplot.png"))
            plt.close()
            
            if self.logger:
                self.logger.info(f"特征箱线图已保存至 {output_dir}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成箱线图失败: {e}")
    
    def _generate_stats_md_report(self, stats_df, group_name, output_dir):
        """生成Markdown统计报告"""
        try:
            report_path = os.path.join(output_dir, f"{group_name}_stats_report.md")
            
            with open(report_path, 'w') as f:
                f.write(f"# {group_name} 特征组 - 基础统计分析报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                f.write("## 数据概况\n\n")
                f.write(f"- 特征数量: {len(stats_df)}\n")
                f.write(f"- 样本数量: {self.stats_results[group_name]['features_shape'][0]}\n\n")
                
                f.write("## 统计摘要\n\n")
                
                # 范围信息
                min_val = stats_df['Min'].min()
                max_val = stats_df['Max'].max()
                f.write(f"- 整体数值范围: [{min_val:.4f}, {max_val:.4f}]\n")
                
                # 均值和标准差
                mean_mean = stats_df['Mean'].mean()
                mean_std = stats_df['Std'].mean()
                f.write(f"- 平均均值: {mean_mean:.4f}\n")
                f.write(f"- 平均标准差: {mean_std:.4f}\n")
                
                # 偏度和峰度
                mean_skew = stats_df['Skewness'].mean()
                mean_kurt = stats_df['Kurtosis'].mean()
                f.write(f"- 平均偏度: {mean_skew:.4f}\n")
                f.write(f"- 平均峰度: {mean_kurt:.4f}\n")
                
                # 零值和缺失值
                total_zeros = stats_df['Zeros'].sum()
                total_samples = self.stats_results[group_name]['features_shape'][0] * len(stats_df)
                zero_percentage = (total_zeros / total_samples) * 100 if total_samples > 0 else 0
                
                total_missing = stats_df['Missing'].sum()
                missing_percentage = (total_missing / total_samples) * 100 if total_samples > 0 else 0
                
                f.write(f"- 零值占比: {zero_percentage:.2f}%\n")
                f.write(f"- 缺失值占比: {missing_percentage:.2f}%\n\n")
                
                # 统计量分布图引用
                f.write("## 统计量分布\n\n")
                f.write(f"![统计量分布图]({group_name}_stats_distribution.png)\n\n")
                
                f.write("## 标准差分布\n\n")
                f.write(f"![标准差箱线图]({group_name}_std_boxplot.png)\n\n")
                
                f.write("## 偏度和峰度分布\n\n")
                f.write(f"![偏度和峰度箱线图]({group_name}_skew_kurt_boxplot.png)\n\n")
                
                # 前10个特征统计详情
                f.write("## 主要特征统计详情\n\n")
                f.write("| 特征 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 | 偏度 | 峰度 | 零值比例(%) |\n")
                f.write("|------|------|--------|--------|--------|--------|------|------|------------|\n")
                
                for i, feature in enumerate(stats_df.index[:10]):  # 只展示前10个特征
                    row = stats_df.loc[feature]
                    zero_ratio = (row['Zeros'] / self.stats_results[group_name]['features_shape'][0]) * 100
                    f.write(f"| {feature} | {row['Mean']:.4f} | {row['Std']:.4f} | {row['Min']:.4f} | "
                           f"{row['Max']:.4f} | {row['Median']:.4f} | {row['Skewness']:.4f} | "
                           f"{row['Kurtosis']:.4f} | {zero_ratio:.2f} |\n")
                
                f.write("\n\n")
                
                # 异常特征分析
                f.write("## 异常特征分析\n\n")
                
                # 高偏度特征
                high_skew = stats_df[abs(stats_df['Skewness']) > 3]
                if not high_skew.empty:
                    f.write(f"### 高偏度特征 (|偏度| > 3): {len(high_skew)} 个\n\n")
                    for feature in high_skew.index[:5]:  # 只展示前5个
                        f.write(f"- {feature}: 偏度 = {high_skew.loc[feature, 'Skewness']:.4f}\n")
                    if len(high_skew) > 5:
                        f.write(f"- ... 以及 {len(high_skew) - 5} 个其他特征\n")
                    f.write("\n")
                
                # 高零值比例特征
                zero_ratio = stats_df['Zeros'] / self.stats_results[group_name]['features_shape'][0]
                high_zero = stats_df[zero_ratio > 0.5]
                if not high_zero.empty:
                    f.write(f"### 高零值比例特征 (零值比例 > 50%): {len(high_zero)} 个\n\n")
                    for feature in high_zero.index[:5]:  # 只展示前5个
                        feature_zero_ratio = zero_ratio.loc[feature] * 100
                        f.write(f"- {feature}: 零值比例 = {feature_zero_ratio:.2f}%\n")
                    if len(high_zero) > 5:
                        f.write(f"- ... 以及 {len(high_zero) - 5} 个其他特征\n")
                    f.write("\n")
                    
                # 存在缺失值的特征
                missing_features = stats_df[stats_df['Missing'] > 0]
                if not missing_features.empty:
                    f.write(f"### 存在缺失值的特征: {len(missing_features)} 个\n\n")
                    for feature in missing_features.index[:5]:  # 只展示前5个
                        missing_count = missing_features.loc[feature, 'Missing']
                        missing_ratio = (missing_count / self.stats_results[group_name]['features_shape'][0]) * 100
                        f.write(f"- {feature}: 缺失值数量 = {missing_count}, 比例 = {missing_ratio:.2f}%\n")
                    if len(missing_features) > 5:
                        f.write(f"- ... 以及 {len(missing_features) - 5} 个其他特征\n")
                
            if self.logger:
                self.logger.info(f"统计报告已保存至 {report_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成统计报告失败: {e}")
    
    def analyze_feature_distribution(self, features, labels=None, feature_indices=None, 
                                   feature_names=None, group_name="all", n_samples=1000):
        """
        分析特征分布
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_indices: 要分析的特征索引列表
            feature_names: 特征名称列表
            group_name: 特征组名称
            n_samples: 抽样数量，避免图像过于密集
        """
        if self.logger:
            self.logger.info(f"分析 {group_name} 组的特征分布")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
            
        if feature_indices is None:
            # 默认分析前10个特征
            feature_indices = list(range(min(10, features.shape[1])))
        
        # 创建输出子目录
        dist_dir = os.path.join(self.output_dir, 'distributions', group_name)
        os.makedirs(dist_dir, exist_ok=True)
        
        # 抽样，避免图像过于密集
        if features.shape[0] > n_samples:
            indices = np.random.choice(features.shape[0], n_samples, replace=False)
            sampled_features = features[indices]
            if labels is not None:
                sampled_labels = labels[indices]
            else:
                sampled_labels = None
        else:
            sampled_features = features
            sampled_labels = labels
        
        # 分析特征分布
        for i, idx in enumerate(feature_indices):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            
            try:
                plt.figure(figsize=(12, 6))
                
                if sampled_labels is not None:
                    # 不同类别分布
                    unique_labels = np.unique(sampled_labels)
                    for label in unique_labels:
                        label_mask = sampled_labels == label
                        plt.hist(sampled_features[label_mask, idx], bins=30, alpha=0.3, 
                                label=f'Class {label}')
                    plt.legend()
                    plt.title(f'{feature_name} 按类别的分布')
                else:
                    # 整体分布
                    plt.hist(sampled_features[:, idx], bins=30, alpha=0.7)
                    plt.title(f'{feature_name} 分布')
                
                plt.xlabel('特征值')
                plt.ylabel('频率')
                plt.grid(True, alpha=0.3)
                
                # 添加正态性检验结果
                if sampled_features.shape[0] > 8:  # 最小样本大小要求
                    k2, p = scipy_stats.normaltest(sampled_features[:, idx])
                    plt.figtext(0.01, 0.01, f'正态性检验 p值: {p:.4f}', 
                               wrap=True, fontsize=10)
                
                save_path = os.path.join(dist_dir, f"{feature_name.replace(' ', '_')}_distribution.png")
                plt.savefig(save_path)
                plt.close()
                
                if i < 5 and self.logger:  # 只记录前几个特征的日志，避免日志过长
                    self.logger.info(f"特征 {feature_name} 分布图已保存至 {save_path}")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"生成特征 {feature_name} 分布图失败: {e}")
        
        if self.logger:
            self.logger.info(f"{group_name} 组特征分布分析完成")
    
    def generate_summary_report(self, output_path=None):
        """
        生成基础统计分析的汇总报告
        
        参数:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"basic_analysis_summary_{self.timestamp}.md")
            
        if not self.stats_results:
            if self.logger:
                self.logger.warning("没有可供汇总的统计结果")
            return
            
        try:
            with open(output_path, 'w') as f:
                f.write("# 基础统计分析汇总报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                f.write("## 特征组概况\n\n")
                f.write("| 特征组 | 特征数量 | 样本数量 | 平均均值 | 平均标准差 | 平均偏度 | 平均峰度 | 零值比例(%) | 缺失值比例(%) |\n")
                f.write("|--------|----------|----------|----------|------------|----------|----------|-------------|---------------|\n")
                
                for group_name, result in self.stats_results.items():
                    stats_df = result['stats_df']
                    feature_count = len(stats_df)
                    sample_count = result['features_shape'][0]
                    
                    mean_mean = stats_df['Mean'].mean()
                    mean_std = stats_df['Std'].mean()
                    mean_skew = stats_df['Skewness'].mean()
                    mean_kurt = stats_df['Kurtosis'].mean()
                    
                    total_zeros = stats_df['Zeros'].sum()
                    total_samples = sample_count * feature_count
                    zero_percentage = (total_zeros / total_samples) * 100 if total_samples > 0 else 0
                    
                    total_missing = stats_df['Missing'].sum()
                    missing_percentage = (total_missing / total_samples) * 100 if total_samples > 0 else 0
                    
                    f.write(f"| {group_name} | {feature_count} | {sample_count} | {mean_mean:.4f} | {mean_std:.4f} | "
                           f"{mean_skew:.4f} | {mean_kurt:.4f} | {zero_percentage:.2f} | {missing_percentage:.2f} |\n")
                
                f.write("\n\n")
                
                # 异常特征汇总
                f.write("## 异常特征汇总\n\n")
                
                for group_name, result in self.stats_results.items():
                    stats_df = result['stats_df']
                    sample_count = result['features_shape'][0]
                    
                    f.write(f"### {group_name} 组异常特征\n\n")
                    
                    # 高偏度特征
                    high_skew = stats_df[abs(stats_df['Skewness']) > 3]
                    if not high_skew.empty:
                        f.write(f"- 高偏度特征 (|偏度| > 3): {len(high_skew)} 个 ({len(high_skew)/len(stats_df)*100:.1f}%)\n")
                    
                    # 高零值比例特征
                    zero_ratio = stats_df['Zeros'] / sample_count
                    high_zero = stats_df[zero_ratio > 0.5]
                    if not high_zero.empty:
                        f.write(f"- 高零值比例特征 (零值比例 > 50%): {len(high_zero)} 个 ({len(high_zero)/len(stats_df)*100:.1f}%)\n")
                    
                    # 存在缺失值的特征
                    missing_features = stats_df[stats_df['Missing'] > 0]
                    if not missing_features.empty:
                        f.write(f"- 存在缺失值的特征: {len(missing_features)} 个 ({len(missing_features)/len(stats_df)*100:.1f}%)\n")
                    
                    f.write("\n")
                
                f.write("## 建议\n\n")
                
                # 根据分析结果给出建议
                f.write("根据基础统计分析结果，提出以下建议：\n\n")
                
                # 检查是否需要标准化
                std_variation = False
                for group_name, result in self.stats_results.items():
                    stats_df = result['stats_df']
                    std_ratio = stats_df['Std'].max() / stats_df['Std'].min() if stats_df['Std'].min() > 0 else float('inf')
                    if std_ratio > 10:
                        std_variation = True
                        break
                
                if std_variation:
                    f.write("1. **特征标准化**：特征间标准差差异较大，建议进行标准化处理。\n")
                else:
                    f.write("1. **特征标准化**：特征间标准差差异相对较小，但标准化处理通常仍然有益。\n")
                
                # 检查是否需要处理异常值
                outlier_issue = False
                for group_name, result in self.stats_results.items():
                    stats_df = result['stats_df']
                    if (abs(stats_df['Skewness']) > 3).any():
                        outlier_issue = True
                        break
                
                if outlier_issue:
                    f.write("2. **异常值处理**：多个特征存在较高偏度，可能包含异常值，建议使用鲁棒缩放或异常值检测方法。\n")
                else:
                    f.write("2. **数据分布**：大部分特征分布相对正常，常规标准化方法应该适用。\n")
                
                # 检查是否需要处理缺失值和零值
                missing_issue = False
                zero_issue = False
                for group_name, result in self.stats_results.items():
                    stats_df = result['stats_df']
                    sample_count = result['features_shape'][0]
                    
                    if (stats_df['Missing'] > 0).any():
                        missing_issue = True
                    
                    zero_ratio = stats_df['Zeros'] / sample_count
                    if (zero_ratio > 0.5).any():
                        zero_issue = True
                
                if missing_issue:
                    f.write("3. **缺失值处理**：数据中存在缺失值，建议使用插补方法处理。\n")
                
                if zero_issue:
                    f.write("4. **零值处理**：部分特征存在高比例零值，可能需要特殊处理或考虑是否为稀疏特征。\n")
                
            if self.logger:
                self.logger.info(f"基础统计分析汇总报告已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成基础统计分析汇总报告失败: {e}")