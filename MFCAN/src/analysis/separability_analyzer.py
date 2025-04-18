import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
from datetime import datetime

class ClassSeparabilityAnalyzer:
    """类别可分性分析工具，用于分析特征对类别区分的贡献度"""
    
    def __init__(self, config_path=None, output_dir=None, logger=None):
        """
        初始化类别可分性分析器
        
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
        self.output_dir = output_dir or self.config.get('output_dir', 'results/separability')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logger
        if logger:
            self.logger.info(f"类别可分性分析器初始化完成，输出目录: {self.output_dir}")
            
        # 存储分析结果
        self.separability_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def compute_feature_significance(self, features, labels, feature_names=None, feature_group="all"):
        """
        计算特征的统计显著性和F值
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_names: 特征名称列表
            feature_group: 特征组名称
            
        返回:
            significance_df: 特征显著性数据框
        """
        if self.logger:
            self.logger.info(f"计算 {feature_group} 组的特征显著性")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        # 确保特征和标签的形状正确
        if features.shape[0] != labels.shape[0]:
            error_msg = f"特征和标签样本数不匹配: {features.shape[0]} vs {labels.shape[0]}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 计算ANOVA F值
        try:
            f_values, p_values = f_classif(features, labels)
            
            # 计算互信息
            mi_values = mutual_info_classif(features, labels, random_state=42)
            
            # 创建结果数据框
            significance_df = pd.DataFrame({
                'Feature': feature_names,
                'F_value': f_values,
                'P_value': p_values,
                'Mutual_Info': mi_values,
                'Significant': p_values < 0.05,
                'Normalized_F': f_values / np.max(f_values) if np.max(f_values) > 0 else f_values,
                'Normalized_MI': mi_values / np.max(mi_values) if np.max(mi_values) > 0 else mi_values,
            })
            
            # 计算综合得分 (F值和互信息的加权平均)
            significance_df['Combined_Score'] = 0.6 * significance_df['Normalized_F'] + 0.4 * significance_df['Normalized_MI']
            
            # 按综合得分排序
            significance_df = significance_df.sort_values('Combined_Score', ascending=False)
            
            # 存储结果
            self.separability_results[f"{feature_group}_significance"] = {
                'significance_df': significance_df,
                'feature_names': feature_names,
                'n_features': features.shape[1],
                'n_samples': features.shape[0],
                'n_classes': len(np.unique(labels)),
                'timestamp': self.timestamp
            }
            
            if self.logger:
                significant_count = significance_df['Significant'].sum()
                significant_ratio = significant_count / len(significance_df) * 100
                self.logger.info(f"找到 {significant_count} 个显著特征 ({significant_ratio:.1f}%)")
                
            return significance_df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"计算特征显著性失败: {e}")
            raise
    
    def identify_discriminative_features(self, features, labels, significance_df=None, 
                                       feature_group="all", top_n=10):
        """
        识别每个类别的最佳区分特征
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            significance_df: 特征显著性数据框，如果没有提供，将重新计算
            feature_group: 特征组名称
            top_n: 每个类别返回的顶级特征数量
            
        返回:
            discriminative_features: 类别区分特征字典
        """
        if self.logger:
            self.logger.info(f"识别 {feature_group} 组的类别区分特征")
        
        # 如果没有提供显著性数据框，先计算
        if significance_df is None:
            significance_df = self.compute_feature_significance(features, labels, feature_group=feature_group)
        
        feature_names = significance_df['Feature'].values
        
        # 获取唯一类别
        unique_labels = np.unique(labels)
        
        # 存储每个类别的区分特征
        discriminative_features = {}
        
        for class_label in unique_labels:
            # 创建二分类标签 (当前类别 vs 其他类别)
            binary_labels = (labels == class_label).astype(int)
            
            # 计算每个特征的类别区分能力
            try:
                # 计算F值和互信息
                f_values, _ = f_classif(features, binary_labels)
                mi_values = mutual_info_classif(features, binary_labels, random_state=42)
                
                # 创建类别特定的显著性数据框
                class_significance = pd.DataFrame({
                    'Feature': feature_names,
                    'F_value': f_values,
                    'Mutual_Info': mi_values,
                    'Normalized_F': f_values / np.max(f_values) if np.max(f_values) > 0 else f_values,
                    'Normalized_MI': mi_values / np.max(mi_values) if np.max(mi_values) > 0 else mi_values,
                })
                
                # 计算综合得分
                class_significance['Combined_Score'] = 0.6 * class_significance['Normalized_F'] + 0.4 * class_significance['Normalized_MI']
                
                # 按综合得分排序并选择前N个特征
                top_features = class_significance.sort_values('Combined_Score', ascending=False).head(top_n)
                
                discriminative_features[int(class_label)] = top_features.to_dict('records')
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"计算类别 {class_label} 的区分特征失败: {e}")
                discriminative_features[int(class_label)] = []
        
        # 存储结果
        self.separability_results[f"{feature_group}_discriminative"] = {
            'discriminative_features': discriminative_features,
            'top_n': top_n,
            'n_classes': len(unique_labels),
            'timestamp': self.timestamp
        }
        
        if self.logger:
            self.logger.info(f"完成 {len(unique_labels)} 个类别的区分特征识别")
            
        return discriminative_features
    
    def analyze_class_similarity(self, features, labels, feature_group="all", n_components=50, metric='euclidean'):
        """
        分析类别间的相似性
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_group: 特征组名称
            n_components: 用于计算类别中心的主成分数，如果特征维度较低可设为None
            metric: 距离度量方法
            
        返回:
            similarity_matrix: 类别相似度矩阵
        """
        if self.logger:
            self.logger.info(f"分析 {feature_group} 组的类别相似性")
        
        # 获取唯一类别
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 如果特征维度很高，先使用PCA降维
        if n_components is not None and features.shape[1] > n_components:
            try:
                pca = PCA(n_components=n_components)
                features_reduced = pca.fit_transform(features)
                if self.logger:
                    explained_var = sum(pca.explained_variance_ratio_) * 100
                    self.logger.info(f"使用PCA降至{n_components}维，保留{explained_var:.1f}%的方差")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"PCA降维失败，使用原始特征: {e}")
                features_reduced = features
        else:
            features_reduced = features
        
        # 计算每个类别的中心
        class_centers = np.zeros((n_classes, features_reduced.shape[1]))
        class_counts = np.zeros(n_classes)
        
        for i, class_label in enumerate(unique_labels):
            class_mask = (labels == class_label)
            class_features = features_reduced[class_mask]
            class_centers[i] = np.mean(class_features, axis=0)
            class_counts[i] = np.sum(class_mask)
        
        # 计算类别中心之间的距离
        try:
            distance_matrix = pairwise_distances(class_centers, metric=metric)
            
            # 转换为相似度矩阵 (反转并归一化距离)
            max_distance = np.max(distance_matrix)
            if max_distance > 0:
                similarity_matrix = 1 - (distance_matrix / max_distance)
            else:
                similarity_matrix = 1 - distance_matrix
            
            # 创建相似度数据框
            similarity_df = pd.DataFrame(similarity_matrix, 
                                        index=unique_labels, 
                                        columns=unique_labels)
            
            # 存储结果
            self.separability_results[f"{feature_group}_similarity"] = {
                'similarity_matrix': similarity_matrix,
                'class_labels': unique_labels,
                'class_counts': class_counts,
                'metric': metric,
                'timestamp': self.timestamp
            }
            
            if self.logger:
                avg_similarity = np.mean(similarity_matrix - np.eye(n_classes))  # 排除对角线
                self.logger.info(f"类别平均相似度: {avg_similarity:.4f}")
                
            return similarity_df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"计算类别相似度失败: {e}")
            raise
    
    
    def save_analysis_results(self, significance_df, discriminative_features, 
                            similarity_df, prefix='all'):
        """
        保存分析结果
        
        参数:
            significance_df: 特征显著性数据框
            discriminative_features: 类别区分特征字典
            similarity_df: 类别相似度数据框
            prefix: 文件前缀
        """
        if self.logger:
            self.logger.info(f"保存 {prefix} 组的分析结果")
        
        # 创建输出子目录
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # 保存特征显著性
            if significance_df is not None:
                csv_path = os.path.join(results_dir, f"{prefix}_feature_significance.csv")
                significance_df.to_csv(csv_path, index=False)
                
                if self.logger:
                    self.logger.info(f"特征显著性已保存至 {csv_path}")
            
            # 保存类别区分特征
            if discriminative_features:
                json_path = os.path.join(results_dir, f"{prefix}_class_discriminative_features.json")
                
                # 转换为可JSON序列化的格式
                json_data = {}
                for class_label, features in discriminative_features.items():
                    json_data[str(class_label)] = []
                    for feature in features:
                        json_data[str(class_label)].append({
                            'Feature': feature['Feature'],
                            'F_value': float(feature['F_value']),
                            'Mutual_Info': float(feature['Mutual_Info']),
                            'Combined_Score': float(feature['Combined_Score'])
                        })
                
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
                if self.logger:
                    self.logger.info(f"类别区分特征已保存至 {json_path}")
            
            # 保存类别相似度
            if similarity_df is not None:
                csv_path = os.path.join(results_dir, f"{prefix}_class_similarity.csv")
                similarity_df.to_csv(csv_path)
                
                if self.logger:
                    self.logger.info(f"类别相似度已保存至 {csv_path}")
                    
            # 生成Markdown报告
            self._generate_markdown_report(significance_df, discriminative_features, 
                                         similarity_df, prefix, results_dir)
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存分析结果失败: {e}")
    
    def _generate_markdown_report(self, significance_df, discriminative_features, 
                                similarity_df, prefix, output_dir):
        """生成Markdown格式的分析报告"""
        try:
            report_path = os.path.join(output_dir, f"{prefix}_separability_report.md")
            
            with open(report_path, 'w') as f:
                f.write(f"# {prefix} 特征组 - 类别可分性分析报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                # 特征显著性
                if significance_df is not None:
                    significant_count = significance_df['Significant'].sum()
                    total_features = len(significance_df)
                    significant_ratio = significant_count / total_features * 100
                    
                    f.write("## 特征显著性分析\n\n")
                    f.write(f"- 总特征数: {total_features}\n")
                    f.write(f"- 显著特征数 (p < 0.05): {significant_count} ({significant_ratio:.1f}%)\n\n")
                    
                    f.write("### 前10个最具区分性的特征\n\n")
                    f.write("| 特征 | F值 | P值 | 互信息 | 综合得分 | 显著性 |\n")
                    f.write("|------|-----|-----|--------|----------|--------|\n")
                    
                    for i, row in significance_df.head(10).iterrows():
                        significant = "是" if row['Significant'] else "否"
                        f.write(f"| {row['Feature']} | {row['F_value']:.4f} | {row['P_value']:.4e} | "
                               f"{row['Mutual_Info']:.4f} | {row['Combined_Score']:.4f} | {significant} |\n")
                    
                    f.write("\n")
                
                # 类别区分特征
                if discriminative_features:
                    f.write("## 类别区分特征\n\n")
                    
                    class_labels = sorted(discriminative_features.keys())
                    for i, class_label in enumerate(class_labels):
                        if i >= 5:  # 只展示前5个类别
                            f.write(f"*以及其他 {len(class_labels) - 5} 个类别...*\n\n")
                            break
                            
                        features = discriminative_features[class_label]
                        f.write(f"### 类别 {class_label} 的区分特征\n\n")
                        
                        if features:
                            f.write("| 特征 | F值 | 互信息 | 综合得分 |\n")
                            f.write("|------|-----|--------|----------|\n")
                            
                            for feature in features[:5]:  # 只展示前5个特征
                                f.write(f"| {feature['Feature']} | {feature['F_value']:.4f} | "
                                       f"{feature['Mutual_Info']:.4f} | {feature['Combined_Score']:.4f} |\n")
                                
                            if len(features) > 5:
                                f.write(f"*以及其他 {len(features) - 5} 个特征...*\n")
                                
                        else:
                            f.write("*未找到显著区分特征*\n")
                            
                        f.write("\n")
                
                # 类别相似性
                if similarity_df is not None:
                    f.write("## 类别相似性分析\n\n")
                    
                    # 计算平均相似度
                    similarity_array = similarity_df.values
                    n_classes = len(similarity_df)
                    
                    # 计算除对角线外的平均相似度
                    mask = ~np.eye(n_classes, dtype=bool)
                    avg_similarity = similarity_array[mask].mean()
                    
                    f.write(f"- 类别数量: {n_classes}\n")
                    f.write(f"- 类别间平均相似度: {avg_similarity:.4f}\n\n")
                    
                    # 找出最相似的类别对
                    if n_classes > 1:
                        max_sim = 0
                        max_pair = (0, 0)
                        
                        for i in range(n_classes):
                            for j in range(i+1, n_classes):
                                if similarity_array[i, j] > max_sim:
                                    max_sim = similarity_array[i, j]
                                    max_pair = (similarity_df.index[i], similarity_df.index[j])
                        
                        f.write(f"- 最相似的类别对: {max_pair[0]} 和 {max_pair[1]} (相似度 = {max_sim:.4f})\n\n")
                
                # 可视化引用
                f.write("## 可视化\n\n")
                f.write(f"![顶级特征]({os.path.join('..', 'visualizations', f'{prefix}_top_features.png')})\n\n")
                
                if similarity_df is not None:
                    f.write(f"![类别相似度]({os.path.join('..', 'visualizations', f'{prefix}_class_similarity.png')})\n\n")
                    
                    if len(similarity_df) <= 20:
                        network_path = os.path.join('..', 'visualizations', f'{prefix}_class_similarity_network.png')
                        f.write(f"![类别相似度网络]({network_path})\n\n")
                
                # 总结与发现
                f.write("## 总结与发现\n\n")
                
                if significance_df is not None:
                    top_features = significance_df.head(3)['Feature'].values
                    f.write(f"1. **最具区分力的特征**: 前三名是 {', '.join(top_features)}，表明这些特征对类别区分最有帮助。\n\n")
                    
                    # 根据显著特征比例给出评价
                    significant_ratio = significant_count / total_features * 100
                    if significant_ratio > 50:
                        f.write("2. **特征质量**: 大部分特征对类别区分显著有效，数据集包含大量有用信息。\n\n")
                    elif significant_ratio > 25:
                        f.write("2. **特征质量**: 中等比例的特征对类别区分有显著作用，可考虑进一步特征选择。\n\n")
                    else:
                        f.write("2. **特征质量**: 仅少数特征对类别区分有显著作用，建议进行严格的特征选择或考虑获取更多有区分力的特征。\n\n")
                
                if similarity_df is not None:
                    if avg_similarity > 0.7:
                        f.write("3. **类别区分挑战**: 类别间平均相似度较高，表明类别区分可能较困难，建议使用更复杂的分类器。\n\n")
                    elif avg_similarity > 0.4:
                        f.write("3. **类别可分性**: 类别间有一定相似度，但总体上类别边界应该可以分辨。\n\n")
                    else:
                        f.write("3. **类别可分性**: 类别间相似度较低，类别边界清晰，分类任务相对容易。\n\n")
                
            if self.logger:
                self.logger.info(f"类别可分性分析报告已保存至 {report_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成类别可分性分析报告失败: {e}")
    
    def generate_summary_report(self, output_path=None):
        """
        生成类别可分性分析的汇总报告
        
        参数:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"separability_summary_{self.timestamp}.md")
            
        if not self.separability_results:
            if self.logger:
                self.logger.warning("没有可供汇总的类别可分性分析结果")
            return
            
        try:
            with open(output_path, 'w') as f:
                f.write("# 类别可分性分析汇总报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                # 特征组统计
                f.write("## 特征组可分性统计\n\n")
                f.write("| 特征组 | 特征数量 | 显著特征比例(%) | 最高F值 | 最高互信息 | 类别平均相似度 |\n")
                f.write("|--------|----------|-----------------|---------|------------|----------------|\n")
                
                group_prefixes = set()
                for key in self.separability_results.keys():
                    if key.endswith('_significance'):
                        group_prefix = key.replace('_significance', '')
                        group_prefixes.add(group_prefix)
                
                for prefix in sorted(group_prefixes):
                    sig_key = f"{prefix}_significance"
                    sim_key = f"{prefix}_similarity"
                    
                    if sig_key in self.separability_results:
                        sig_data = self.separability_results[sig_key]
                        sig_df = sig_data['significance_df']
                        
                        n_features = sig_data['n_features']
                        significant_count = sum(sig_df['Significant'])
                        significant_ratio = significant_count / n_features * 100
                        max_f = sig_df['F_value'].max()
                        max_mi = sig_df['Mutual_Info'].max()
                        
                        # 获取类别相似度
                        avg_similarity = '-'
                        if sim_key in self.separability_results:
                            sim_data = self.separability_results[sim_key]
                            sim_matrix = sim_data['similarity_matrix']
                            n_classes = len(sim_data['class_labels'])
                            
                            # 计算除对角线外的平均相似度
                            mask = ~np.eye(n_classes, dtype=bool)
                            avg_similarity = f"{sim_matrix[mask].mean():.4f}"
                        
                        f.write(f"| {prefix} | {n_features} | {significant_ratio:.1f} | {max_f:.4f} | "
                               f"{max_mi:.4f} | {avg_similarity} |\n")
                
                f.write("\n")
                
                # 特征组比较
                f.write("## 特征组比较\n\n")
                
                # 统计每个特征组的前十名特征并比较
                f.write("### 各特征组前5名特征\n\n")
                
                for prefix in sorted(group_prefixes):
                    sig_key = f"{prefix}_significance"
                    
                    if sig_key in self.separability_results:
                        sig_data = self.separability_results[sig_key]
                        sig_df = sig_data['significance_df']
                        
                        f.write(f"**{prefix} 组**:\n\n")
                        for i, row in sig_df.head(5).iterrows():
                            f.write(f"- {row['Feature']} (F值={row['F_value']:.4f}, MI={row['Mutual_Info']:.4f})\n")
                        f.write("\n")
                
                # 类别可分性分析总结
                f.write("## 类别可分性分析总结\n\n")
                
                # 根据分析结果给出建议
                # 1. 特征组比较
                f.write("### 特征组比较\n\n")
                
                # 收集每个特征组的显著性比例
                group_significance = {}
                for prefix in group_prefixes:
                    sig_key = f"{prefix}_significance"
                    
                    if sig_key in self.separability_results:
                        sig_data = self.separability_results[sig_key]
                        sig_df = sig_data['significance_df']
                        
                        n_features = sig_data['n_features']
                        significant_count = sum(sig_df['Significant'])
                        significant_ratio = significant_count / n_features * 100
                        group_significance[prefix] = significant_ratio
                
                if group_significance:
                    best_group = max(group_significance.items(), key=lambda x: x[1])[0]
                    worst_group = min(group_significance.items(), key=lambda x: x[1])[0]
                    
                    f.write(f"- **最具区分力的特征组**: {best_group} 组，显著特征比例为 {group_significance[best_group]:.1f}%\n")
                    f.write(f"- **最缺乏区分力的特征组**: {worst_group} 组，显著特征比例为 {group_significance[worst_group]:.1f}%\n\n")
                
                # 2. 类别可分性评估
                f.write("### 类别可分性评估\n\n")
                
                # 检查是否有类别相似度信息
                has_similarity = False
                high_similarity = False
                for key in self.separability_results.keys():
                    if key.endswith('_similarity'):
                        has_similarity = True
                        sim_data = self.separability_results[key]
                        sim_matrix = sim_data['similarity_matrix']
                        n_classes = len(sim_data['class_labels'])
                        
                        # 计算除对角线外的平均相似度
                        mask = ~np.eye(n_classes, dtype=bool)
                        avg_similarity = sim_matrix[mask].mean()
                        
                        if avg_similarity > 0.6:
                            high_similarity = True
                            break
                
                if has_similarity:
                    if high_similarity:
                        f.write("- **类别区分挑战**: 存在高度相似的类别，表明分类任务具有一定挑战性，建议选择更复杂的分类器并重点关注高区分力的特征。\n\n")
                    else:
                        f.write("- **良好的类别可分性**: 类别之间区分较为明显，使用常规分类器应能达到良好效果。\n\n")
                
                # 3. 特征选择建议
                f.write("### 特征选择建议\n\n")
                
                # 统计所有特征组的显著特征占比
                all_significant_ratio = 0
                total_features = 0
                
                for prefix in group_prefixes:
                    sig_key = f"{prefix}_significance"
                    
                    if sig_key in self.separability_results:
                        sig_data = self.separability_results[sig_key]
                        sig_df = sig_data['significance_df']
                        
                        n_features = sig_data['n_features']
                        significant_count = sum(sig_df['Significant'])
                        total_features += n_features
                        all_significant_ratio += significant_count
                
                if total_features > 0:
                    all_significant_ratio = (all_significant_ratio / total_features) * 100
                    
                    if all_significant_ratio < 25:
                        f.write("- **严格特征选择**: 显著特征比例较低，建议进行严格的特征选择，只保留最具区分力的特征。\n")
                        f.write("- **考虑使用特征重要性重排序**: 可采用随机森林或梯度提升树等方法对特征重要性进行评估。\n\n")
                    elif all_significant_ratio < 50:
                        f.write("- **中等特征选择**: 显著特征比例中等，建议保留F值或互信息较高的前50%特征。\n")
                        f.write("- **特征组合考虑**: 不同特征组的信息可能互补，建议在特征选择时综合考虑各组的贡献。\n\n")
                    else:
                        f.write("- **轻度特征选择**: 大部分特征具有区分力，可以只移除最不显著的特征。\n")
                        f.write("- **考虑特征加权**: 可根据特征重要性为不同特征赋予不同权重。\n\n")
                
                # 4. 模型选择建议
                f.write("### 模型选择建议\n\n")
                
                if has_similarity:
                    if high_similarity:
                        f.write("- **复杂分类器**: 由于类别相似度较高，建议使用非线性分类器如深度神经网络、SVM(RBF核)或集成学习方法。\n")
                        f.write("- **考虑分层分类**: 对于高度相似的类别组，可考虑先分大类再细分的层次分类策略。\n\n")
                    else:
                        f.write("- **适中复杂度**: 类别可分性良好，常规机器学习模型如随机森林、SVM或中等深度神经网络应该足够。\n\n")
                else:
                    f.write("- **基于特征分析**: 根据特征显著性分析，建议选择能充分利用重要特征的模型结构。\n\n")
                
            if self.logger:
                self.logger.info(f"类别可分性分析汇总报告已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成类别可分性分析汇总报告失败: {e}")

    def visualize_class_separability(self, significance_df, prefix='all'):
        """
        可视化类别可分性（基于特征显著性）
        
        参数:
            significance_df: 特征显著性数据框
            prefix: 文件前缀
        """
        if self.logger:
            self.logger.info(f"可视化 {prefix} 组的类别可分性")
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # 显示前20个最具区分性的特征
            top_n = min(20, len(significance_df))
            top_features = significance_df.head(top_n)
            
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
                    
            plt.title(f'{prefix} 组前{top_n}个特征的F值 (绿色=显著)')
            plt.xlabel('F值')
            plt.tight_layout()
            
            # 互信息柱状图
            plt.subplot(2, 1, 2)
            plt.barh(top_features['Feature'][::-1], top_features['Mutual_Info'][::-1])
            plt.title(f'{prefix} 组前{top_n}个特征的互信息值')
            plt.xlabel('互信息')
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_top_features.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"类别可分性可视化已保存至 {save_path}")
            
            # 绘制P值直方图
            plt.figure(figsize=(10, 6))
            plt.hist(significance_df['P_value'], bins=50, alpha=0.7)
            plt.axvline(x=0.05, color='r', linestyle='--', label='p=0.05显著性阈值')
            plt.title(f'{prefix} 组P值分布')
            plt.xlabel('P值')
            plt.ylabel('频率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_pvalue_distribution.png")
            plt.savefig(save_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"P值分布图已保存至 {save_path}")
                
            # 绘制互信息与F值关系散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(significance_df['Normalized_F'], significance_df['Normalized_MI'], 
                    alpha=0.7, c=significance_df['Combined_Score'], cmap='viridis')
            
            # 添加前10个特征的标签
            for i, row in significance_df.head(10).iterrows():
                plt.annotate(row['Feature'], 
                        (row['Normalized_F'], row['Normalized_MI']),
                        fontsize=9)
            
            plt.colorbar(label='综合得分')
            plt.title(f'{prefix} 组特征的F值与互信息关系')
            plt.xlabel('归一化F值')
            plt.ylabel('归一化互信息')
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_f_vs_mi.png")
            plt.savefig(save_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"F值与互信息关系图已保存至 {save_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成类别可分性可视化失败: {e}")

    def visualize_class_similarity(self, similarity_df, prefix='all'):
        """
        可视化类别相似性
        
        参数:
            similarity_df: 类别相似度数据框
            prefix: 文件前缀
        """
        if self.logger:
            self.logger.info(f"可视化 {prefix} 组的类别相似性")
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # 绘制类别相似度热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(similarity_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
            plt.title(f'{prefix} 组类别相似度矩阵')
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_class_similarity.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"类别相似度热图已保存至 {save_path}")
                
            # 如果类别数量小于等于20，绘制类别相似度网络图
            if len(similarity_df) <= 20:
                plt.figure(figsize=(14, 12))
                
                # 使用多维缩放将相似度嵌入到二维空间
                from sklearn.manifold import MDS
                similarity_array = similarity_df.values
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                # 转换为距离矩阵
                distances = 1 - similarity_array
                pos = mds.fit_transform(distances)
                
                # 绘制节点
                plt.scatter(pos[:, 0], pos[:, 1], s=200, c='skyblue', edgecolors='black')
                
                # 添加类别标签
                for i, label in enumerate(similarity_df.index):
                    plt.annotate(label, (pos[i, 0], pos[i, 1]), fontsize=12, ha='center', va='center')
                
                # 绘制连接线，只有相似度高于0.5的类别对之间绘制连线
                threshold = 0.5
                for i in range(len(similarity_df)):
                    for j in range(i+1, len(similarity_df)):
                        if similarity_array[i, j] > threshold:
                            plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                                'gray', alpha=similarity_array[i, j], linewidth=similarity_array[i, j]*3)
                
                plt.title(f'{prefix} 组类别相似度网络图 (相似度 > {threshold})')
                plt.axis('off')
                
                # 保存图表
                save_path = os.path.join(vis_dir, f"{prefix}_class_similarity_network.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
                if self.logger:
                    self.logger.info(f"类别相似度网络图已保存至 {save_path}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成类别相似性可视化失败: {e}")