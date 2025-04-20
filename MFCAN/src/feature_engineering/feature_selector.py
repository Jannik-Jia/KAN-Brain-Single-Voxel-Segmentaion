import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import h5py
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    """特征选择工具，用于基于重要性分数选择特征子集"""
    
    def __init__(self, config_path=None, importance_path=None, output_dir=None, logger=None):
        """
        初始化特征选择器
        
        参数:
            config_path: 配置文件路径
            importance_path: 特征重要性结果目录
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
        self.output_dir = output_dir or self.config.get('output_dir', 'results/feature_selection')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置特征重要性路径
        self.importance_path = importance_path
        
        # 设置日志记录器
        self.logger = logger
        if logger:
            self.logger.info(f"特征选择器初始化完成，输出目录: {self.output_dir}")
            
        # 存储分析结果
        self.selection_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def select_by_importance(self, features, ranked_features=None, coverage=0.95, 
                           importance_file=None, feature_group="all"):
        """
        按重要性覆盖率选择特征
        
        参数:
            features: 特征矩阵
            ranked_features: 排序后的特征列表，每项包含'Feature'和'Importance'
            coverage: 重要性覆盖率，默认0.95表示保留能解释95%重要性的特征
            importance_file: 特征重要性文件路径，如果ranked_features未提供
            feature_group: 特征组名称
            
        返回:
            selected_indices: 选中的特征索引
        """
        if self.logger:
            self.logger.info(f"根据重要性选择 {feature_group} 组的特征子集 (覆盖率: {coverage})")
        
        # 如果未提供ranked_features但提供了importance_file，加载排名
        if ranked_features is None and importance_file:
            try:
                ranked_df = pd.read_csv(importance_file)
                ranked_features = ranked_df[['Feature', 'Importance']].to_dict('records')
                if self.logger:
                    self.logger.info(f"从 {importance_file} 加载了特征排名")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"加载特征排名失败: {e}")
                return np.arange(features.shape[1])  # 返回所有特征
        
        # 如果未提供ranked_features且未提供importance_file，尝试从默认路径加载
        elif ranked_features is None and self.importance_path:
            try:
                default_file = os.path.join(self.importance_path, f"{feature_group}_feature_ranking.csv")
                if os.path.exists(default_file):
                    ranked_df = pd.read_csv(default_file)
                    ranked_features = ranked_df[['Feature', 'Importance']].to_dict('records')
                    if self.logger:
                        self.logger.info(f"从 {default_file} 加载了特征排名")
                else:
                    if self.logger:
                        self.logger.warning(f"默认特征排名文件不存在: {default_file}，将返回所有特征")
                    return np.arange(features.shape[1])  # 返回所有特征
            except Exception as e:
                if self.logger:
                    self.logger.error(f"加载特征排名失败: {e}")
                return np.arange(features.shape[1])  # 返回所有特征
                
        # 如果仍未获取到特征排名，返回所有特征
        if not ranked_features:
            if self.logger:
                self.logger.warning(f"未提供特征排名，将返回所有特征")
            return np.arange(features.shape[1])
        
        # 计算累积重要性
        total_importance = sum(feature['Importance'] for feature in ranked_features)
        
        if total_importance == 0:
            if self.logger:
                self.logger.warning(f"特征重要性总和为0，将返回所有特征")
            return np.arange(features.shape[1])
            
        cumulative_importance = 0
        selected_features = []
        
        for feature in ranked_features:
            feature_name = feature['Feature']
            importance = feature['Importance']
            
            cumulative_importance += importance
            selected_features.append(feature_name)
            
            # 达到覆盖率阈值后停止
            if cumulative_importance / total_importance >= coverage:
                break
        
        # 获取特征索引
        selected_indices = []
        for feature_name in selected_features:
            # 处理特征名称可能是数字索引的情况
            if feature_name.startswith('Feature_'):
                try:
                    idx = int(feature_name.split('_')[1])
                    selected_indices.append(idx)
                except:
                    # 如果无法解析索引，跳过
                    if self.logger:
                        self.logger.warning(f"无法解析特征索引: {feature_name}")
                    continue
            else:
                # 如果是自定义特征名，假设它与列索引一一对应
                try:
                    if isinstance(feature_name, str):
                        idx = int(feature_name)
                    else:
                        idx = feature_name
                    selected_indices.append(idx)
                except:
                    # 如果无法解析索引，跳过
                    if self.logger:
                        self.logger.warning(f"无法解析特征索引: {feature_name}")
                    continue
        
        if not selected_indices:
            if self.logger:
                self.logger.warning(f"未能选择任何特征，将返回所有特征")
            return np.arange(features.shape[1])
        
        # 确保索引在有效范围内
        selected_indices = [idx for idx in selected_indices if 0 <= idx < features.shape[1]]
        
        if self.logger:
            selection_ratio = len(selected_indices) / features.shape[1] * 100
            self.logger.info(f"已选择 {len(selected_indices)} 个特征 ({selection_ratio:.1f}% 的总特征)")
        
        # 存储选择结果
        self.selection_results[f"{feature_group}_importance"] = {
            'method': 'importance',
            'coverage': coverage,
            'selected_indices': selected_indices,
            'feature_count': len(selected_indices),
            'timestamp': self.timestamp
        }
        
        return np.array(selected_indices)
    
    def remove_redundancy(self, features, selected_indices, correlation_threshold=0.9, feature_group="all"):
        """
        从选中的特征中移除高度冗余的特征
        
        参数:
            features: 特征矩阵
            selected_indices: 预先选择的特征索引
            correlation_threshold: 相关性阈值，高于此值的特征被视为冗余
            feature_group: 特征组名称
            
        返回:
            non_redundant_indices: 移除冗余后的特征索引
        """
        if self.logger:
            self.logger.info(f"从 {feature_group} 组移除冗余特征 (相关性阈值: {correlation_threshold})")
        
        if len(selected_indices) <= 1:
            if self.logger:
                self.logger.info(f"选中的特征数 <= 1，不需要移除冗余")
            return selected_indices
        
        # 提取选定的特征
        selected_features = features[:, selected_indices]
        
        # 计算相关性矩阵
        try:
            # 标准化特征以获得更准确的相关性
            scaler = StandardScaler()
            selected_features_scaled = scaler.fit_transform(selected_features)
            
            # 计算相关系数矩阵
            correlation_matrix = np.corrcoef(selected_features_scaled.T)
            
            # 处理NaN值
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            # 创建一个布尔掩码，标记要保留的特征
            # 初始时所有特征都被标记为保留
            keep_mask = np.ones(len(selected_indices), dtype=bool)
            
            # 按重要性顺序检查冗余性
            for i in range(len(selected_indices)):
                # 如果当前特征已被标记为移除，则跳过
                if not keep_mask[i]:
                    continue
                    
                # 检查与当前特征高度相关的其他特征
                for j in range(i+1, len(selected_indices)):
                    # 如果特征j已被标记为移除，则跳过
                    if not keep_mask[j]:
                        continue
                        
                    # 如果相关性高于阈值，则标记特征j为移除
                    if abs(correlation_matrix[i, j]) >= correlation_threshold:
                        keep_mask[j] = False
            
            # 获取保留的特征索引
            kept_indices = np.where(keep_mask)[0]
            non_redundant_indices = selected_indices[kept_indices]
            
            if self.logger:
                removed_count = len(selected_indices) - len(non_redundant_indices)
                removed_ratio = removed_count / len(selected_indices) * 100
                self.logger.info(f"移除了 {removed_count} 个冗余特征 ({removed_ratio:.1f}% 的选中特征)")
            
            # 存储选择结果
            self.selection_results[f"{feature_group}_nonredundant"] = {
                'method': 'nonredundant',
                'correlation_threshold': correlation_threshold,
                'selected_indices': non_redundant_indices.tolist(),
                'feature_count': len(non_redundant_indices),
                'timestamp': self.timestamp
            }
            
            return non_redundant_indices
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"移除冗余特征失败: {e}")
            return selected_indices
    
    def generate_feature_subsets(self, features, ranked_features, prefix='all'):
        """
        生成多个特征子集供后续实验
        
        参数:
            features: 特征矩阵
            ranked_features: 排序后的特征列表
            prefix: 文件前缀
            
        返回:
            feature_subsets: 特征子集字典
        """
        if self.logger:
            self.logger.info(f"为 {prefix} 组生成特征子集")
        
        feature_subsets = {}
        
        try:
            # 基于重要性覆盖率的子集
            for coverage in [0.8, 0.9, 0.95, 0.98, 0.99]:
                coverage_str = f"{int(coverage*100)}pct"
                indices = self.select_by_importance(features, ranked_features, coverage, 
                                                 feature_group=f"{prefix}_{coverage_str}")
                feature_subsets[f"importance_{coverage_str}"] = indices
                
                if self.logger:
                    feature_ratio = len(indices) / features.shape[1] * 100
                    self.logger.info(f"{coverage*100}%覆盖率子集: {len(indices)}个特征 ({feature_ratio:.1f}%)")
            
            # 基于固定特征数量的子集
            for feature_count in [10, 20, 50, 100]:
                if feature_count < features.shape[1]:
                    # 只选择前N个特征
                    if len(ranked_features) >= feature_count:
                        top_features = ranked_features[:feature_count]
                        feature_names = [f['Feature'] for f in top_features]
                        
                        # 获取特征索引
                        indices = []
                        for feature_name in feature_names:
                            if feature_name.startswith('Feature_'):
                                try:
                                    idx = int(feature_name.split('_')[1])
                                    indices.append(idx)
                                except:
                                    continue
                            else:
                                try:
                                    if isinstance(feature_name, str):
                                        idx = int(feature_name)
                                    else:
                                        idx = feature_name
                                    indices.append(idx)
                                except:
                                    continue
                        
                        # 确保索引在有效范围内
                        indices = [idx for idx in indices if 0 <= idx < features.shape[1]]
                        
                        feature_subsets[f"top_{feature_count}"] = np.array(indices)
                        
                        if self.logger:
                            feature_ratio = len(indices) / features.shape[1] * 100
                            self.logger.info(f"前{feature_count}特征子集: {len(indices)}个特征 ({feature_ratio:.1f}%)")
            
            # 基于模型选择的子集
            # RandomForest
            try:
                if self.logger:
                    self.logger.info(f"使用RandomForest进行特征选择")
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # 准备数据
                # 假设标签在最后一列（如果需要，可以根据实际情况修改）
                if features.shape[0] > 10000:  # 如果样本太多，随机抽样
                    indices = np.random.choice(features.shape[0], 10000, replace=False)
                    X = features[indices]
                    # 由于没有标签，这里我们不能进行训练，所以注释掉相关代码
                    # y = labels[indices]
                    # 训练模型
                    # rf.fit(X, y)
                    # selector = SelectFromModel(rf, prefit=True)
                    # selected_mask = selector.get_support()
                    # rf_indices = np.where(selected_mask)[0]
                    # feature_subsets["random_forest"] = rf_indices
                    
                    # 因为没有标签，我们退回到使用重要性阈值选择
                    threshold = 0.9  # 作为替代
                    rf_indices = self.select_by_importance(features, ranked_features, threshold, 
                                                     feature_group=f"{prefix}_rf")
                    feature_subsets["random_forest"] = rf_indices
                    
                    if self.logger:
                        rf_ratio = len(rf_indices) / features.shape[1] * 100
                        self.logger.info(f"RandomForest子集: {len(rf_indices)}个特征 ({rf_ratio:.1f}%)")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"RandomForest特征选择失败: {e}")
            
            # 存储选择结果
            self.selection_results[f"{prefix}_subsets"] = {
                'method': 'multiple_subsets',
                'subsets': {name: indices.tolist() for name, indices in feature_subsets.items()},
                'timestamp': self.timestamp
            }
            
            return feature_subsets
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成特征子集失败: {e}")
            return {}
    
    def apply_selection(self, features, selected_indices):
        """
        应用特征选择，返回选中的特征子集
        
        参数:
            features: 特征矩阵
            selected_indices: 选中的特征索引
            
        返回:
            selected_features: 选中的特征子集
        """
        # 确保indices是有效范围内
        valid_indices = [idx for idx in selected_indices if 0 <= idx < features.shape[1]]
        
        if not valid_indices:
            if self.logger:
                self.logger.warning("没有有效的特征索引，返回原始特征")
            return features
        
        return features[:, valid_indices]
    
    def save_feature_subsets(self, selected_data, train_labels=None, val_labels=None, 
                           test_labels=None, output_path=None):
        """
        保存所选特征子集
        
        参数:
            selected_data: 所选特征数据字典，格式为{'group_name': {'train': X_train, 'val': X_val, ...}}
            train_labels: 训练集标签
            val_labels: 验证集标签
            test_labels: 测试集标签
            output_path: 输出文件路径
        """
        if not selected_data:
            if self.logger:
                self.logger.warning("没有特征子集可保存")
            return
            
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"selected_features_{self.timestamp}.h5")
            
        if self.logger:
            self.logger.info(f"保存特征子集到 {output_path}")
            
        try:
            with h5py.File(output_path, 'w') as f:
                # 保存元数据
                f.attrs['timestamp'] = self.timestamp
                f.attrs['feature_selection_method'] = 'importance_based'
                
                # 保存每个特征子集
                for group_name, group_data in selected_data.items():
                    # 创建组
                    group = f.create_group(group_name)
                    
                    # 保存选中的特征索引
                    if 'selected_indices' in group_data:
                        group.create_dataset('selected_indices', data=group_data['selected_indices'])
                    
                    # 保存特征数据
                    if 'train' in group_data:
                        train_group = group.create_group('train')
                        train_group.create_dataset('features', data=group_data['train'])
                        if train_labels is not None:
                            train_group.create_dataset('labels', data=train_labels)
                    
                    if 'val' in group_data:
                        val_group = group.create_group('val')
                        val_group.create_dataset('features', data=group_data['val'])
                        if val_labels is not None:
                            val_group.create_dataset('labels', data=val_labels)
                    
                    if 'test' in group_data:
                        test_group = group.create_group('test')
                        test_group.create_dataset('features', data=group_data['test'])
                        if test_labels is not None:
                            test_group.create_dataset('labels', data=test_labels)
                    
                    if self.logger:
                        self.logger.info(f"保存了 {group_name} 组的特征子集")
            
            if self.logger:
                self.logger.info(f"特征子集已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存特征子集失败: {e}")
    
    def generate_summary_report(self, output_path=None):
        """
        生成特征选择的汇总报告
        
        参数:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"feature_selection_summary_{self.timestamp}.md")
            
        if not self.selection_results:
            if self.logger:
                self.logger.warning("没有可供汇总的特征选择结果")
            return
            
        try:
            with open(output_path, 'w') as f:
                f.write("# 特征选择汇总报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                # 特征选择统计
                f.write("## 特征选择统计\n\n")
                f.write("| 特征集 | 选择方法 | 选择特征数 | 选择比例 |\n")
                f.write("|--------|----------|------------|----------|\n")
                
                for key, result in self.selection_results.items():
                    if 'selected_indices' in result:
                        feature_count = result['feature_count']
                        # 特征总数无法从结果中获取，因此不计算比例
                        # 如果需要比例，需要在结果中添加total_features字段
                        selection_ratio = "N/A"
                        
                        f.write(f"| {key} | {result['method']} | {feature_count} | {selection_ratio} |\n")
                
                f.write("\n")
                
                # 冗余特征分析
                redundancy_results = {k: v for k, v in self.selection_results.items() if 'nonredundant' in k}
                
                if redundancy_results:
                    f.write("## 冗余特征分析\n\n")
                    f.write("| 特征集 | 选择前特征数 | 选择后特征数 | 冗余比例 |\n")
                    f.write("|--------|--------------|--------------|----------|\n")
                    
                    for key, result in redundancy_results.items():
                        group_name = key.replace('_nonredundant', '')
                        importance_key = f"{group_name}_importance"
                        
                        if importance_key in self.selection_results:
                            importance_result = self.selection_results[importance_key]
                            before_count = importance_result.get('feature_count', "N/A")
                            after_count = result.get('feature_count', "N/A")
                            
                            if isinstance(before_count, int) and isinstance(after_count, int):
                                redundancy_ratio = (before_count - after_count) / before_count * 100
                                ratio_str = f"{redundancy_ratio:.1f}%"
                            else:
                                ratio_str = "N/A"
                                
                            f.write(f"| {group_name} | {before_count} | {after_count} | {ratio_str} |\n")
                    
                    f.write("\n")
                
                # 特征选择建议
                f.write("## 特征选择建议\n\n")
                
                # 基于结果提出建议
                has_high_redundancy = False
                for key, result in redundancy_results.items():
                    group_name = key.replace('_nonredundant', '')
                    importance_key = f"{group_name}_importance"
                    
                    if importance_key in self.selection_results:
                        importance_result = self.selection_results[importance_key]
                        before_count = importance_result.get('feature_count', 0)
                        after_count = result.get('feature_count', 0)
                        
                        if before_count > 0:
                            redundancy_ratio = (before_count - after_count) / before_count
                            if redundancy_ratio > 0.3:  # 冗余比例超过30%
                                has_high_redundancy = True
                                break
                
                if has_high_redundancy:
                    f.write("1. **移除冗余特征**: 分析显示特征中存在高度冗余，建议使用经过冗余移除的特征子集进行训练。\n\n")
                else:
                    f.write("1. **特征相关性**: 特征集合中冗余性不高，可以使用基于重要性的特征子集，无需额外的冗余移除。\n\n")
                
                # 特征数量建议
                coverage_results = {}
                for key, result in self.selection_results.items():
                    if '_subsets' in key and 'subsets' in result:
                        for subset_name, indices in result['subsets'].items():
                            if 'importance_' in subset_name:
                                coverage = subset_name.replace('importance_', '').replace('pct', '')
                                coverage_results[coverage] = len(indices)
                
                if coverage_results:
                    recommended_coverage = None
                    for coverage in ['90', '95']:
                        if coverage in coverage_results:
                            recommended_coverage = coverage
                            break
                    
                    if recommended_coverage:
                        f.write(f"2. **推荐特征子集**: 建议使用{recommended_coverage}%重要性覆盖率的特征子集 ({coverage_results[recommended_coverage]}个特征)，提供良好的准确性和计算效率平衡。\n\n")
                
                # 模型评估建议
                f.write("3. **模型训练策略**: 建议比较以下特征子集的模型性能:\n")
                f.write("   - 重要性覆盖率90%的特征子集\n")
                f.write("   - 移除冗余后的特征子集\n")
                f.write("   - 固定数量的前N个最重要特征(例如前50或100个特征)\n\n")
                
                f.write("4. **进一步优化**: 可考虑以下进一步优化策略:\n")
                f.write("   - 尝试不同特征组之间的组合\n")
                f.write("   - 对特征进行交互或多项式扩展，创建新特征\n")
                f.write("   - 应用主成分分析(PCA)或自编码器进行非线性降维\n")
                
            if self.logger:
                self.logger.info(f"特征选择汇总报告已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成特征选择汇总报告失败: {e}")