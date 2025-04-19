import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import json
from datetime import datetime

class CorrelationAnalyzer:
    """特征相关性分析工具，用于分析特征组内部和组间相关性"""
    
    def __init__(self, config_path=None, output_dir=None, logger=None, gpu_enabled=False):
        """
        初始化相关性分析器
        
        参数:
            config_path: 配置文件路径
            output_dir: 输出目录
            logger: 日志记录器
            gpu_enabled: 是否启用GPU加速
        """
        # 加载配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # 设置输出目录
        self.output_dir = output_dir or self.config.get('output_dir', 'results/correlation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logger
        if logger:
            self.logger.info(f"相关性分析器初始化完成，输出目录: {self.output_dir}")
            
        # 设置GPU加速
        self.gpu_enabled = gpu_enabled
        
        # 存储分析结果
        self.correlation_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def analyze_intra_group_correlation(self, features, feature_names=None, method='pearson'):
        """
        分析特征组内部相关性
        
        参数:
            features: 特征矩阵
            feature_names: 特征名称列表
            method: 相关性计算方法，'pearson'、'spearman'或'kendall'
            
        返回:
            corr_df: 相关性矩阵数据框
        """
        if self.logger:
            self.logger.info(f"分析特征组内部相关性 (方法: {method})")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        # 创建特征数据框
        df = pd.DataFrame(features, columns=feature_names)
        
        # 计算相关性矩阵
        try:
            corr_df = df.corr(method=method)
            
            # 存储结果
            result_key = f"intra_{method}"
            self.correlation_results[result_key] = {
                'corr_df': corr_df,
                'feature_names': feature_names,
                'method': method,
                'timestamp': self.timestamp
            }
            
            if self.logger:
                self.logger.info(f"特征组内部相关性分析完成，相关系数矩阵形状: {corr_df.shape}")
                
            return corr_df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"计算相关性矩阵失败: {e}")
            raise
    
    def analyze_inter_group_correlation(self, group_features, method='pearson'):
        """
        分析特征组间相关性
        
        参数:
            group_features: 特征组字典，键为组名，值为特征矩阵
            method: 相关性计算方法，'pearson'、'spearman'或'kendall'
            
        返回:
            group_corr: 特征组间相关性字典
        """
        if self.logger:
            self.logger.info(f"分析特征组间相关性 (方法: {method})")
        
        # 检查输入
        if not isinstance(group_features, dict) or len(group_features) < 2:
            if self.logger:
                self.logger.warning("分析特征组间相关性需要至少两个特征组")
            return {}
        
        # 计算每对特征组之间的相关性
        group_corr = {}
        group_names = list(group_features.keys())
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names):
                if i >= j:  # 只计算上三角矩阵
                    continue
                    
                features1 = group_features[group1]
                features2 = group_features[group2]
                
                # 检查样本数是否匹配
                if features1.shape[0] != features2.shape[0]:
                    if self.logger:
                        self.logger.warning(f"特征组 {group1} 和 {group2} 样本数不匹配，跳过")
                    continue
                
                # 计算两组特征间的相关性
                try:
                    # 创建特征数据框
                    df1 = pd.DataFrame(features1, columns=[f"{group1}_{i}" for i in range(features1.shape[1])])
                    df2 = pd.DataFrame(features2, columns=[f"{group2}_{i}" for i in range(features2.shape[1])])
                    
                    # 计算相关性
                    corr_df = pd.concat([df1, df2], axis=1).corr(method=method)
                    
                    # 提取组间相关性
                    inter_corr = corr_df.loc[df1.columns, df2.columns]
                    
                    # 存储结果
                    group_pair = f"{group1}_{group2}"
                    group_corr[group_pair] = {
                        'corr_df': inter_corr,
                        'group1': group1,
                        'group2': group2,
                        'method': method
                    }
                    
                    if self.logger:
                        self.logger.info(f"特征组 {group1} 和 {group2} 间相关性分析完成，"
                                        f"相关系数矩阵形状: {inter_corr.shape}")
                        
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"计算特征组 {group1} 和 {group2} 间相关性失败: {e}")
        
        # 存储结果
        self.correlation_results['inter_group'] = {
            'group_corr': group_corr,
            'method': method,
            'timestamp': self.timestamp
        }
        
        return group_corr
    
    def identify_correlation_clusters(self, corr_df, threshold=0.8, method='single'):
        """
        识别高相关性特征集合
        
        参数:
            corr_df: 相关性矩阵数据框
            threshold: 相关性阈值
            method: 层次聚类方法
            
        返回:
            clusters: 特征集群列表
        """
        if self.logger:
            self.logger.info(f"识别高相关性特征集合 (阈值: {threshold})")
        
        # 将相关系数转换为距离矩阵
        distance = 1 - abs(corr_df)
        
        # 提取上三角矩阵的距离值
        condensed_distance = squareform(distance)
        
        # 执行层次聚类
        try:
            z = hierarchy.linkage(condensed_distance, method=method)
            clusters = hierarchy.fcluster(z, t=1-threshold, criterion='distance')
            
            # 将聚类结果与特征名称关联
            feature_clusters = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in feature_clusters:
                    feature_clusters[cluster_id] = []
                feature_clusters[cluster_id].append(corr_df.index[i])
            
            # 只保留包含多个特征的集群
            multi_feature_clusters = {k: v for k, v in feature_clusters.items() if len(v) > 1}
            
            # 转换为列表格式
            clusters_list = list(multi_feature_clusters.values())
            
            # 存储结果
            self.correlation_results['clusters'] = {
                'clusters': clusters_list,
                'threshold': threshold,
                'method': method,
                'timestamp': self.timestamp
            }
            
            if self.logger:
                self.logger.info(f"识别出 {len(clusters_list)} 个高相关性特征集合")
                
            return clusters_list
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"识别高相关性特征集合失败: {e}")
            return []
    
    def extract_high_correlations(self, corr_df, threshold=0.8):
        """
        提取高相关特征对
        
        参数:
            corr_df: 相关性矩阵
            threshold: 相关性阈值
            
        返回:
            high_corr: 高相关特征对列表
        """
        high_corr = []
        
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                if abs(corr_df.iloc[i, j]) >= threshold:
                    high_corr.append({
                        'feature1': corr_df.columns[i],
                        'feature2': corr_df.columns[j],
                        'correlation': float(corr_df.iloc[i, j])
                    })
        
        # 按相关系数绝对值降序排序
        high_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # 存储结果
        self.correlation_results['high_corr_pairs'] = {
            'pairs': high_corr,
            'threshold': threshold,
            'timestamp': self.timestamp
        }
        
        return high_corr
    
    def save_high_correlations(self, high_corr_pairs, prefix='all'):
        """
        保存高相关特征对
        
        参数:
            high_corr_pairs: 高相关特征对列表
            prefix: 文件前缀
        """
        if not high_corr_pairs:
            if self.logger:
                self.logger.info(f"没有找到高相关特征对")
            return
            
        # 创建输出子目录
        corr_dir = os.path.join(self.output_dir, 'high_correlations')
        os.makedirs(corr_dir, exist_ok=True)
        
        # 保存为CSV文件
        csv_path = os.path.join(corr_dir, f"{prefix}_high_correlations.csv")
        pd.DataFrame(high_corr_pairs).to_csv(csv_path, index=False)
        
        # 保存为JSON文件
        json_path = os.path.join(corr_dir, f"{prefix}_high_correlations.json")
        with open(json_path, 'w') as f:
            json.dump(high_corr_pairs, f, indent=2)
        
        if self.logger:
            self.logger.info(f"高相关特征对已保存至 {csv_path} 和 {json_path}")
    
    def visualize_correlation_matrix(self, corr_df, prefix='all', max_features=100):
        """
        可视化相关性矩阵
        
        参数:
            corr_df: 相关性矩阵
            prefix: 文件前缀
            max_features: 可视化的最大特征数
        """
        if self.logger:
            self.logger.info(f"可视化相关性矩阵 (特征数: {len(corr_df)})")
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 如果特征数量太多，只可视化部分特征
        if len(corr_df) > max_features:
            if self.logger:
                self.logger.info(f"特征数量超过 {max_features}，只可视化部分特征")
                
            # 计算平均绝对相关系数
            mean_abs_corr = abs(corr_df).mean(axis=1)
            
            # 选择平均相关性最高的特征
            selected_features = mean_abs_corr.nlargest(max_features).index
            corr_df_subset = corr_df.loc[selected_features, selected_features]
            
            # 保存所选特征
            selected_features_path = os.path.join(vis_dir, f"{prefix}_selected_features.txt")
            with open(selected_features_path, 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
                    
            if self.logger:
                self.logger.info(f"所选特征已保存至 {selected_features_path}")
        else:
            corr_df_subset = corr_df
        
        try:
            # 绘制相关性热图
            plt.figure(figsize=(20, 16))
            mask = np.triu(np.ones_like(corr_df_subset, dtype=bool))
            
            # 使用发散色图
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            
            # 绘制热图
            sns.heatmap(corr_df_subset, mask=mask, cmap=cmap, annot=False,
                    vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5)
            
            plt.title(f'{prefix} Feature Correlation Matrix', fontsize=16)
            
            # 保存图表
            heatmap_path = os.path.join(vis_dir, f"{prefix}_correlation_heatmap.png")
            plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            if self.logger:
                self.logger.info(f"相关性热图已保存至 {heatmap_path}")
            
            # 绘制相关性分布直方图
            plt.figure(figsize=(12, 8))
            
            # 提取上三角矩阵的相关系数（不包括对角线）
            triu_indices = np.triu_indices_from(corr_df.values, k=1)
            correlations = corr_df.values[triu_indices]
            
            plt.hist(correlations, bins=50, alpha=0.75)
            plt.axvline(x=0, color='r', linestyle='--')
            
            # 添加垂直线表示高相关阈值
            plt.axvline(x=0.8, color='g', linestyle='--', label='High Positive Corr (0.8)')
            plt.axvline(x=-0.8, color='orange', linestyle='--', label='High Negative Corr (-0.8)')
            
            plt.title(f'{prefix} Feature Correlation Coefficient Distribution', fontsize=16)
            plt.xlabel('Correlation Coefficient', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            hist_path = os.path.join(vis_dir, f"{prefix}_correlation_histogram.png")
            plt.savefig(hist_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"相关系数分布直方图已保存至 {hist_path}")
                
            # 绘制聚类热图
            if len(corr_df_subset) <= 200:  # 只为较小的矩阵绘制聚类热图
                plt.figure(figsize=(20, 16))
                
                # 计算特征聚类
                row_linkage = hierarchy.linkage(distance.pdist(1 - abs(corr_df_subset)), method='average')
                col_linkage = hierarchy.linkage(distance.pdist(1 - abs(corr_df_subset.T)), method='average')
                
                # 绘制聚类热图
                sns.clustermap(corr_df_subset, figsize=(20, 16), cmap=cmap,
                            center=0, vmin=-1, vmax=1,
                            row_linkage=row_linkage, col_linkage=col_linkage,
                            linewidths=.5, annot=False)
                
                # 保存图表
                cluster_path = os.path.join(vis_dir, f"{prefix}_correlation_clustermap.png")
                plt.savefig(cluster_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                if self.logger:
                    self.logger.info(f"相关性聚类热图已保存至 {cluster_path}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"可视化相关性矩阵失败: {e}")

            

    def visualize_inter_group_correlation(self, group_corr):
        """
        可视化特征组间相关性

        参数:
            group_corr: 特征组间相关性字典
        """
        if not group_corr:
            if self.logger:
                self.logger.warning("没有特征组间相关性数据可视化")
            return
            
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 计算每对特征组之间的平均绝对相关系数
        group_pairs = []
        for pair_key, corr_data in group_corr.items():
            corr_df = corr_data['corr_df']
            group1 = corr_data['group1']
            group2 = corr_data['group2']
            
            # 计算平均绝对相关系数
            mean_abs_corr = abs(corr_df).mean().mean()
            
            group_pairs.append({
                'group1': group1,
                'group2': group2,
                'mean_abs_corr': mean_abs_corr
            })
        
        try:
            # 绘制组间相关性条形图
            plt.figure(figsize=(12, 8))
            
            pairs = [f"{p['group1']} - {p['group2']}" for p in group_pairs]
            mean_corrs = [p['mean_abs_corr'] for p in group_pairs]
            
            plt.barh(pairs, mean_corrs, color='skyblue')
            plt.xlabel('Mean Absolute Correlation')
            plt.title('Mean Absolute Correlation Between Feature Groups')
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            bar_path = os.path.join(vis_dir, "inter_group_correlation_bar.png")
            plt.savefig(bar_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"特征组间相关性条形图已保存至 {bar_path}")
            
            # 绘制组间相关性热图
            if len(group_pairs) > 1:
                # 创建组间相关性矩阵
                groups = sorted(list(set([p['group1'] for p in group_pairs] + [p['group2'] for p in group_pairs])))
                n_groups = len(groups)
                
                group_corr_matrix = np.zeros((n_groups, n_groups))
                
                # 填充相关性矩阵
                for p in group_pairs:
                    i = groups.index(p['group1'])
                    j = groups.index(p['group2'])
                    group_corr_matrix[i, j] = p['mean_abs_corr']
                    group_corr_matrix[j, i] = p['mean_abs_corr']  # 对称矩阵
                
                # 对角线设为1
                np.fill_diagonal(group_corr_matrix, 1.0)
                
                # 绘制热图
                plt.figure(figsize=(10, 8))
                sns.heatmap(group_corr_matrix, annot=True, cmap='YlGnBu',
                            xticklabels=groups, yticklabels=groups, vmin=0, vmax=1)
                
                plt.title('Inter-Group Feature Correlation Heatmap')
                
                # 保存图表
                heatmap_path = os.path.join(vis_dir, "inter_group_correlation_heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()
                
                if self.logger:
                    self.logger.info(f"特征组间相关性热图已保存至 {heatmap_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"可视化特征组间相关性失败: {e}")
