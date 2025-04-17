import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
import json
from datetime import datetime
import joblib

class FeatureImportanceAnalyzer:
    """特征重要性分析工具，使用随机森林和互信息评估特征重要性"""
    
    def __init__(self, config_path=None, output_dir=None, logger=None):
        """
        初始化特征重要性分析器
        
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
        self.output_dir = output_dir or self.config.get('output_dir', 'results/feature_importance')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logger
        if logger:
            self.logger.info(f"特征重要性分析器初始化完成，输出目录: {self.output_dir}")
            
        # 存储分析结果
        self.importance_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置随机森林参数
        self.rf_params = self.config.get('rf_params', {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        })
    
    def compute_importance_scores(self, features, labels, feature_names=None, 
                                method='all', feature_group="all"):
        """
        计算特征重要性分数
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_names: 特征名称列表
            method: 重要性计算方法，'rf'(随机森林)、'mi'(互信息)或'all'(所有方法)
            feature_group: 特征组名称
            
        返回:
            importance_df: 特征重要性数据框
        """
        if self.logger:
            self.logger.info(f"计算 {feature_group} 组的特征重要性 (方法: {method})")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
        
        importance_scores = {}
        
        # 随机森林重要性
        if method in ['rf', 'all']:
            try:
                rf_importance = self._compute_rf_importance(features, labels)
                importance_scores['rf'] = rf_importance
                
                if self.logger:
                    self.logger.info(f"随机森林重要性计算完成，特征数: {len(rf_importance)}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"随机森林重要性计算失败: {e}")
                importance_scores['rf'] = np.zeros(features.shape[1])
        
        # 互信息
        if method in ['mi', 'all']:
            try:
                mi_importance = self._compute_mi_importance(features, labels)
                importance_scores['mi'] = mi_importance
                
                if self.logger:
                    self.logger.info(f"互信息计算完成，特征数: {len(mi_importance)}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"互信息计算失败: {e}")
                importance_scores['mi'] = np.zeros(features.shape[1])
        
        # 排列重要性
        if method in ['permutation', 'all']:
            try:
                perm_importance = self._compute_permutation_importance(features, labels)
                importance_scores['permutation'] = perm_importance
                
                if self.logger:
                    self.logger.info(f"排列重要性计算完成，特征数: {len(perm_importance)}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"排列重要性计算失败: {e}")
                importance_scores['permutation'] = np.zeros(features.shape[1])
        
        # 组合所有方法的重要性
        try:
            # 归一化每种方法的重要性
            normalized_scores = {}
            for method_name, scores in importance_scores.items():
                max_score = np.max(scores)
                if max_score > 0:
                    normalized_scores[method_name] = scores / max_score
                else:
                    normalized_scores[method_name] = scores
            
            # 计算综合重要性
            if len(normalized_scores) > 0:
                # 默认权重
                weights = {
                    'rf': 0.4,
                    'mi': 0.3,
                    'permutation': 0.3
                }
                
                # 计算加权平均
                combined_importance = np.zeros(features.shape[1])
                weight_sum = 0
                
                for method_name, scores in normalized_scores.items():
                    method_weight = weights.get(method_name, 1.0 / len(normalized_scores))
                    combined_importance += scores * method_weight
                    weight_sum += method_weight
                
                if weight_sum > 0:
                    combined_importance /= weight_sum
            else:
                combined_importance = np.zeros(features.shape[1])
                
            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': combined_importance
            })
            
            # 添加各种方法的重要性
            for method_name, scores in importance_scores.items():
                importance_df[f'{method_name}_importance'] = scores
                
            # 按综合重要性降序排序
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # 存储结果
            self.importance_results[feature_group] = {
                'importance_df': importance_df,
                'methods': list(importance_scores.keys()),
                'feature_names': feature_names,
                'timestamp': self.timestamp
            }
            
            if self.logger:
                top_features = importance_df.head(5)['Feature'].tolist()
                self.logger.info(f"特征重要性分析完成，前5名特征: {', '.join(top_features)}")
                
            return importance_df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"计算综合重要性失败: {e}")
            return pd.DataFrame({'Feature': feature_names, 'Importance': np.zeros(len(feature_names))})
    
    def _compute_rf_importance(self, features, labels):
        """使用随机森林计算特征重要性"""
        # 创建和训练随机森林
        rf = RandomForestClassifier(**self.rf_params)
        rf.fit(features, labels)
        
        # 获取特征重要性
        importance = rf.feature_importances_
        
        # 保存模型
        model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(rf, os.path.join(model_dir, f'rf_importance_{self.timestamp}.joblib'))
        
        return importance
    
    def _compute_mi_importance(self, features, labels):
        """使用互信息计算特征重要性"""
        # 计算每个特征与类别之间的互信息
        mi_scores = mutual_info_classif(features, labels, random_state=42)
        return mi_scores
    
    def _compute_permutation_importance(self, features, labels):
        """使用排列重要性计算特征重要性"""
        # 创建和训练随机森林
        rf = RandomForestClassifier(**self.rf_params)
        rf.fit(features, labels)
        
        # 计算排列重要性
        result = permutation_importance(rf, features, labels, n_repeats=10, 
                                      random_state=42, n_jobs=-1)
        
        return result.importances_mean
    
    def rank_features(self, importance_df=None, feature_group="all"):
        """
        根据重要性对特征进行排序
        
        参数:
            importance_df: 特征重要性数据框，如果没有提供，将使用已存储的结果
            feature_group: 特征组名称
            
        返回:
            ranked_features: 排序后的特征列表
        """
        if self.logger:
            self.logger.info(f"对 {feature_group} 组的特征进行排序")
        
        if importance_df is None:
            if feature_group in self.importance_results:
                importance_df = self.importance_results[feature_group]['importance_df']
            else:
                if self.logger:
                    self.logger.warning(f"未找到 {feature_group} 组的重要性结果")
                return []
        
        # 提取特征名称和重要性
        ranked_features = importance_df[['Feature', 'Importance']].to_dict('records')
        
        return ranked_features
    
    def generate_importance_thresholds(self, ranked_features, coverage_levels=None):
        """
        生成不同覆盖率的特征数阈值
        
        参数:
            ranked_features: 排序后的特征列表
            coverage_levels: 覆盖率水平列表，例如[0.8, 0.9, 0.95, 0.99]
            
        返回:
            thresholds: 阈值字典
        """
        if self.logger:
            self.logger.info("生成特征重要性阈值")
        
        if not ranked_features:
            if self.logger:
                self.logger.warning("没有特征排名可用于计算阈值")
            return {}
        
        if coverage_levels is None:
            coverage_levels = [0.8, 0.9, 0.95, 0.99]
        
        # 提取重要性分数
        importance_values = [feature['Importance'] for feature in ranked_features]
        total_importance = sum(importance_values)
        
        thresholds = {}
        if total_importance > 0:
            # 计算累积重要性
            cumulative_importance = np.cumsum(importance_values) / total_importance
            
            # 为每个覆盖率计算阈值
            for coverage in coverage_levels:
                # 找到第一个超过覆盖率的索引
                idx = np.argmax(cumulative_importance >= coverage)
                feature_count = idx + 1
                min_importance = importance_values[idx] if idx < len(importance_values) else 0
                
                thresholds[str(coverage)] = {
                    'feature_count': feature_count,
                    'min_importance': min_importance,
                    'coverage': coverage
                }
        
        return thresholds
    
    def visualize_importance(self, ranked_features, prefix='all'):
        """
        可视化特征重要性
        
        参数:
            ranked_features: 排序后的特征列表
            prefix: 文件前缀
        """
        if self.logger:
            self.logger.info(f"可视化 {prefix} 组的特征重要性")
        
        if not ranked_features:
            if self.logger:
                self.logger.warning("没有特征排名可用于可视化")
            return
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # 提取特征名称和重要性
            feature_names = [feature['Feature'] for feature in ranked_features]
            importance_values = [feature['Importance'] for feature in ranked_features]
            
            # 只显示前20个特征
            top_n = min(20, len(ranked_features))
            top_features = feature_names[:top_n]
            top_importance = importance_values[:top_n]
            
            # 创建条形图
            plt.figure(figsize=(12, 10))
            bars = plt.barh(top_features[::-1], top_importance[::-1])
            
            # 为每个条形添加值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}', ha='left', va='center')
            
            plt.title(f'{prefix} 组前{top_n}个最重要特征')
            plt.xlabel('重要性分数')
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_top_features_importance.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"特征重要性条形图已保存至 {save_path}")
            
            # 创建累积重要性图
            plt.figure(figsize=(12, 8))
            # 计算累积重要性
            cumulative_importance = np.cumsum(importance_values) / sum(importance_values)
            
            plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-')
            
            # 标记80%, 90%, 95%的点
            for coverage in [0.8, 0.9, 0.95]:
                idx = np.argmax(cumulative_importance >= coverage)
                feature_count = idx + 1
                plt.axvline(x=feature_count, color='r', linestyle='--')
                plt.text(feature_count + 0.5, coverage, f'特征数: {feature_count}\n覆盖率: {coverage:.2f}', 
                       ha='left', va='center')
            
            plt.title(f'{prefix} 组特征重要性累积分布')
            plt.xlabel('特征数量')
            plt.ylabel('累积重要性')
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_cumulative_importance.png")
            plt.savefig(save_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"累积重要性图已保存至 {save_path}")
            
            # 可视化特征重要性分布
            plt.figure(figsize=(12, 8))
            
            plt.hist(importance_values, bins=50, alpha=0.7)
            plt.title(f'{prefix} 组特征重要性分布')
            plt.xlabel('重要性分数')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            
            # 计算并标记重要性阈值
            for coverage in [0.8, 0.9, 0.95]:
                idx = np.argmax(cumulative_importance >= coverage)
                if idx < len(importance_values):
                    threshold = importance_values[idx]
                    plt.axvline(x=threshold, color='r', linestyle='--', 
                               label=f'{coverage:.0%} 覆盖率阈值: {threshold:.4f}')
            
            plt.legend()
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{prefix}_importance_distribution.png")
            plt.savefig(save_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"特征重要性分布图已保存至 {save_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成特征重要性可视化失败: {e}")
    
    def save_results(self, ranked_features, thresholds, output_dir=None, prefix='all'):
        """
        保存分析结果
        
        参数:
            ranked_features: 排序后的特征列表
            thresholds: 阈值字典
            output_dir: 输出目录
            prefix: 文件前缀
        """
        if self.logger:
            self.logger.info(f"保存 {prefix} 组的特征重要性结果")
        
        if not ranked_features:
            if self.logger:
                self.logger.warning("没有特征排名可用于保存")
            return
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 保存特征排名
            ranking_path = os.path.join(output_dir, f"{prefix}_feature_ranking.csv")
            pd.DataFrame(ranked_features).to_csv(ranking_path, index=False)
            
            if self.logger:
                self.logger.info(f"特征排名已保存至 {ranking_path}")
            
            # 保存重要性阈值
            thresholds_path = os.path.join(output_dir, f"{prefix}_importance_thresholds.json")
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds, f, indent=2)
            
            if self.logger:
                self.logger.info(f"重要性阈值已保存至 {thresholds_path}")
                
            # 生成Markdown报告
            self._generate_markdown_report(ranked_features, thresholds, output_dir, prefix)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存特征重要性结果失败: {e}")
    
    def _generate_markdown_report(self, ranked_features, thresholds, output_dir, prefix):
        """生成Markdown格式的分析报告"""
        try:
            report_path = os.path.join(output_dir, f"{prefix}_importance_report.md")
            
            with open(report_path, 'w') as f:
                f.write(f"# {prefix} 特征组 - 特征重要性分析报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                # 特征概况
                f.write("## 特征重要性概况\n\n")
                f.write(f"- 总特征数: {len(ranked_features)}\n")
                
                # 重要性阈值表格
                f.write("## 重要性阈值\n\n")
                f.write("| 覆盖率 | 特征数量 | 最低重要性阈值 |\n")
                f.write("|--------|----------|----------------|\n")
                
                for coverage, threshold_data in thresholds.items():
                    coverage_percent = float(coverage) * 100
                    f.write(f"| {coverage_percent:.0f}% | {threshold_data['feature_count']} | {threshold_data['min_importance']:.6f} |\n")
                
                f.write("\n")
                
                # 前20个最重要特征
                top_n = min(20, len(ranked_features))
                
                f.write(f"## 前{top_n}个最重要特征\n\n")
                f.write("| 排名 | 特征 | 重要性 |\n")
                f.write("|------|------|--------|\n")
                
                for i, feature in enumerate(ranked_features[:top_n]):
                    f.write(f"| {i+1} | {feature['Feature']} | {feature['Importance']:.6f} |\n")
                
                f.write("\n")
                
                # 可视化引用
                f.write("## 可视化\n\n")
                
                importance_plot = f"../visualizations/{prefix}_top_features_importance.png"
                f.write(f"![特征重要性条形图]({importance_plot})\n\n")
                
                cumulative_plot = f"../visualizations/{prefix}_cumulative_importance.png"
                f.write(f"![累积重要性图]({cumulative_plot})\n\n")
                
                distribution_plot = f"../visualizations/{prefix}_importance_distribution.png"
                f.write(f"![重要性分布图]({distribution_plot})\n\n")
                
                # 分析结论
                f.write("## 分析结论\n\n")
                
                # 根据重要性分布给出结论
                if len(ranked_features) > 0:
                    top_5_importance = sum(feature['Importance'] for feature in ranked_features[:5])
                    total_importance = sum(feature['Importance'] for feature in ranked_features)
                    top_5_ratio = top_5_importance / total_importance if total_importance > 0 else 0
                    
                    # 特征集中度
                    if top_5_ratio > 0.5:
                        f.write("1. **高特征集中度**: 前5个特征贡献了超过50%的重要性，表明大部分有用信息集中在少量特征中。\n\n")
                    elif top_5_ratio > 0.3:
                        f.write("1. **中等特征集中度**: 前5个特征贡献了约30-50%的重要性，信息分布相对均衡。\n\n")
                    else:
                        f.write("1. **低特征集中度**: 前5个特征贡献的重要性不到30%，信息分散在众多特征中。\n\n")
                    
                    # 覆盖率建议
                    coverage_90_count = thresholds.get('0.9', {}).get('feature_count', len(ranked_features))
                    coverage_ratio = coverage_90_count / len(ranked_features) if len(ranked_features) > 0 else 1
                    
                    if coverage_ratio < 0.2:
                        f.write("2. **特征选择建议**: 仅需保留约{:.0f}%的特征即可捕获90%的重要信息，建议进行强度较高的特征选择。\n\n".format(coverage_ratio * 100))
                    elif coverage_ratio < 0.5:
                        f.write("2. **特征选择建议**: 需要保留约{:.0f}%的特征才能捕获90%的重要信息，建议进行中等强度的特征选择。\n\n".format(coverage_ratio * 100))
                    else:
                        f.write("2. **特征选择建议**: 需要保留大部分特征({:.0f}%)才能捕获90%的重要信息，建议进行轻度的特征选择。\n\n".format(coverage_ratio * 100))
                
            if self.logger:
                self.logger.info(f"特征重要性分析报告已保存至 {report_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成特征重要性分析报告失败: {e}")
    
    def generate_summary_report(self, output_path=None):
        """
        生成特征重要性分析的汇总报告
        
        参数:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"importance_summary_{self.timestamp}.md")
            
        if not self.importance_results:
            if self.logger:
                self.logger.warning("没有可供汇总的特征重要性分析结果")
            return
            
        try:
            with open(output_path, 'w') as f:
                f.write("# 特征重要性分析汇总报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                # 特征组统计
                f.write("## 特征组重要性统计\n\n")
                f.write("| 特征组 | 特征数量 | 90%覆盖所需特征数 | 90%覆盖比例 | 前5特征重要性占比 |\n")
                f.write("|--------|----------|-------------------|-------------|-------------------|\n")
                
                all_top_features = []
                
                for group_name, result in self.importance_results.items():
                    importance_df = result['importance_df']
                    feature_count = len(importance_df)
                    
                    # 计算90%覆盖所需特征数
                    importance_values = importance_df['Importance'].values
                    cumulative_importance = np.cumsum(importance_values) / sum(importance_values)
                    
                    coverage_90_idx = np.argmax(cumulative_importance >= 0.9)
                    coverage_90_count = coverage_90_idx + 1
                    coverage_90_ratio = coverage_90_count / feature_count
                    
                    # 计算前5特征重要性占比
                    top_5_importance = importance_df.head(5)['Importance'].sum()
                    total_importance = importance_df['Importance'].sum()
                    top_5_ratio = top_5_importance / total_importance if total_importance > 0 else 0
                    
                    f.write(f"| {group_name} | {feature_count} | {coverage_90_count} | {coverage_90_ratio:.2f} | {top_5_ratio:.2f} |\n")
                    
                    # 收集前5名特征
                    top_features = importance_df.head(5)[['Feature', 'Importance']].to_dict('records')
                    for feature in top_features:
                        all_top_features.append({
                            'Feature': feature['Feature'],
                            'Importance': feature['Importance'],
                            'Group': group_name
                        })
                
                f.write("\n")
                
                # 所有特征组的顶级特征
                f.write("## 各特征组顶级特征\n\n")
                
                all_top_df = pd.DataFrame(all_top_features)
                if not all_top_df.empty:
                    # 按重要性排序
                    all_top_df = all_top_df.sort_values('Importance', ascending=False)
                    
                    f.write("| 特征组 | 特征 | 重要性 |\n")
                    f.write("|--------|------|--------|\n")
                    
                    for i, row in all_top_df.head(20).iterrows():
                        f.write(f"| {row['Group']} | {row['Feature']} | {row['Importance']:.6f} |\n")
                
                f.write("\n")
                
                # 特征重要性分析总结
                f.write("## 特征重要性分析总结\n\n")
                
                # 特征选择建议
                f.write("### 特征选择建议\n\n")
                
                # 分析覆盖率
                avg_coverage_ratio = 0
                count = 0
                
                for group_name, result in self.importance_results.items():
                    importance_df = result['importance_df']
                    feature_count = len(importance_df)
                    
                    # 计算90%覆盖所需特征数
                    importance_values = importance_df['Importance'].values
                    cumulative_importance = np.cumsum(importance_values) / sum(importance_values)
                    
                    coverage_90_idx = np.argmax(cumulative_importance >= 0.9)
                    coverage_90_count = coverage_90_idx + 1
                    coverage_90_ratio = coverage_90_count / feature_count
                    
                    avg_coverage_ratio += coverage_90_ratio
                    count += 1
                
                if count > 0:
                    avg_coverage_ratio /= count
                    
                    if avg_coverage_ratio < 0.2:
                        f.write("1. **强特征选择**: 分析表明，平均只需约{:.0f}%的顶级特征即可覆盖90%的重要信息。建议采用强度较高的特征选择。\n\n".format(avg_coverage_ratio * 100))
                    elif avg_coverage_ratio < 0.5:
                        f.write("1. **中等特征选择**: 分析表明，平均需要约{:.0f}%的特征才能覆盖90%的重要信息。建议采用中等强度的特征选择。\n\n".format(avg_coverage_ratio * 100))
                    else:
                        f.write("1. **轻度特征选择**: 分析表明，平均需要{:.0f}%以上的特征才能覆盖90%的重要信息。建议采用较轻的特征选择或考虑使用降维技术。\n\n".format(avg_coverage_ratio * 100))
                
                # 特征组对比
                f.write("### 特征组对比\n\n")
                
                if len(self.importance_results) > 1:
                    # 比较不同特征组的信息浓缩程度
                    group_concentration = {}
                    
                    for group_name, result in self.importance_results.items():
                        importance_df = result['importance_df']
                        
                        # 计算前5特征重要性占比
                        top_5_importance = importance_df.head(5)['Importance'].sum()
                        total_importance = importance_df['Importance'].sum()
                        top_5_ratio = top_5_importance / total_importance if total_importance > 0 else 0
                        
                        group_concentration[group_name] = top_5_ratio
                    
                    # 找出信息最集中和最分散的特征组
                    if group_concentration:
                        most_concentrated = max(group_concentration.items(), key=lambda x: x[1])
                        least_concentrated = min(group_concentration.items(), key=lambda x: x[1])
                        
                        f.write(f"- **信息最集中的特征组**: {most_concentrated[0]} (前5特征重要性占比: {most_concentrated[1]:.2f})\n")
                        f.write(f"- **信息最分散的特征组**: {least_concentrated[0]} (前5特征重要性占比: {least_concentrated[1]:.2f})\n\n")
                
                # 模型训练建议
                f.write("### 模型训练建议\n\n")
                f.write("基于特征重要性分析，在模型训练中可考虑以下策略：\n\n")
                f.write("1. **分阶段特征引入**: 先使用最重要的特征子集训练基线模型，再逐步引入更多特征，观察性能变化。\n")
                f.write("2. **特征加权**: 在模型中对特征按重要性进行加权，或使用L1正则化自动筛选重要特征。\n")
                f.write("3. **特征组合测试**: 尝试仅使用每个特征组中的最重要特征，以减少计算成本并可能提高泛化能力。\n")
                
            if self.logger:
                self.logger.info(f"特征重要性分析汇总报告已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成特征重要性分析汇总报告失败: {e}")