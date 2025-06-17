#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑区感知Subject Embedding分析器
核心分析功能模块
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, KFold
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway, kruskal, spearmanr
from pathlib import Path

try:
    from .utils import ensure_directory, save_analysis_results, cleanup_memory
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append('.')
    from src.utils import ensure_directory, save_analysis_results, cleanup_memory

logger = logging.getLogger(__name__)



class BrainAwareSubjectEmbeddingAnalyzer:
    """
    脑区感知的Subject Embedding 可行性分析器
    
    增强功能：
    - 保留原有的四Phase分析框架
    - 新增脑区维度的受试者差异分析
    - 扩展可视化系统支持脑区特异性分析
    - 增强决策框架提供分脑区建议
    """
    
    def __init__(self, save_path='./subject_embedding_analysis_brain_aware/'):
        self.save_path = Path(save_path)  # 使用Path对象
        ensure_directory(self.save_path)  # 使用utils中的函数
        ensure_directory(self.save_path / 'visualizations')
        
        # 分析结果存储
        self.analysis_results = {}
        self.decision_scores = {}
        
        logger.info("🧠 脑区感知Subject Embedding分析器初始化完成")  # 改为logger
        logger.info(f"📁 结果保存路径: {self.save_path}")

    def prepare_data_with_subjects_enhanced(self, data_dict):
        """
        增强版数据准备 - 保留原有功能 + 脑区张量构建
        
        Args:
            data_dict: load_and_prepare_data_multi_subject_out()的返回结果
            
        Returns:
            dict: 包含完整受试者映射和脑区分析数据的字典
        """
        logger.info("\n" + "="*80)
        logger.info("📊 Phase 0: 增强版数据准备（原有功能 + 脑区感知）")
        logger.info("="*80)
        
        # 🔄 保留原有数据验证逻辑
        required_keys = ['X_train_scaled', 'y_train', 'subjects_train',
                        'X_val_scaled', 'y_val', 'subjects_val', 
                        'X_test_scaled', 'y_test', 'subjects_test']
        
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(f"数据字典缺少必要字段: {missing_keys}")
        
        # 提取数据
        X_train = data_dict['X_train_scaled']
        y_train = data_dict['y_train'] 
        subjects_train = data_dict['subjects_train']
        
        X_val = data_dict['X_val_scaled']
        y_val = data_dict['y_val']
        subjects_val = data_dict['subjects_val']
        
        X_test = data_dict['X_test_scaled'] 
        y_test = data_dict['y_test']
        subjects_test = data_dict['subjects_test']
        
        # 获取所有可用受试者
        all_subjects = np.concatenate([subjects_train, subjects_val, subjects_test])
        available_subjects = np.unique(all_subjects)
        
        logger.info(f"✅ 原有数据验证通过")
        logger.info(f"  - 训练集: {len(subjects_train):,} 样本, 受试者 {sorted(np.unique(subjects_train))}")
        logger.info(f"  - 验证集: {len(subjects_val):,} 样本, 受试者 {sorted(np.unique(subjects_val))}")
        logger.info(f"  - 测试集: {len(subjects_test):,} 样本, 受试者 {sorted(np.unique(subjects_test))}")
        logger.info(f"  - 总受试者数: {len(available_subjects)}")
        
        # 🔥 新增：构建脑区感知分析数据
        logger.info(f"\n🧠 构建脑区感知分析数据...")
        
        # 解析one-hot标签到脑区ID  
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_regions = np.argmax(y_train, axis=1)
        else:
            y_train_regions = y_train.flatten()
        
        # 构建subject×region张量
        brain_region_analysis = self._build_subject_region_tensor(
            X_train, y_train_regions, subjects_train
        )
        
        # 计算受试者样本统计
        subject_counts = {}
        for subject_id in available_subjects:
            train_count = np.sum(subjects_train == subject_id)
            val_count = np.sum(subjects_val == subject_id) 
            test_count = np.sum(subjects_test == subject_id)
            total_count = train_count + val_count + test_count
            
            subject_counts[subject_id] = {
                'train': train_count,
                'val': val_count, 
                'test': test_count,
                'total': total_count
            }
        
        # 存储完整数据
        self.data = {
            # 使用标准化数据作为主要数据源
            'X_train': X_train,
            'y_train': y_train, 
            'subjects_train': subjects_train,
            'X_val': X_val,
            'y_val': y_val,
            'subjects_val': subjects_val,
            'X_test': X_test,
            'y_test': y_test,
            'subjects_test': subjects_test,
            
            # 🔥 关键修复：同时保存scaled版本的键名
            'X_train_scaled': X_train,
            'X_val_scaled': X_val, 
            'X_test_scaled': X_test,
            
            # 其他数据...
            'available_subjects': available_subjects,
            'subject_counts': subject_counts,
            'brain_region_analysis': brain_region_analysis,
            'y_train_regions': y_train_regions
        }
        
        
        logger.info(f"✅ 增强版数据准备完成")
        logger.info(f"  - 数据映射方法: 精确Multi-Subject-Out + 脑区感知")
        logger.info(f"  - 有效脑区×受试者组合: {len(brain_region_analysis['subject_region_features'])}")
        
        return self.data

    def _build_subject_region_tensor(self, X, regions, subjects):
        """
        构建三维分析张量: [受试者 × 脑区 × 特征]
        
        Args:
            X: 特征矩阵 (N_voxels, 341)
            regions: 脑区标签 (N_voxels,)
            subjects: 受试者标签 (N_voxels,)
            
        Returns:
            dict: 脑区感知分析数据结构
        """
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(regions)
        
        subject_region_features = {}
        sample_counts = {}
        
        logger.info(f"    🔍 分析 {len(unique_subjects)} 个受试者 × {len(unique_regions)} 个脑区...")
        
        valid_combinations = 0
        total_combinations = len(unique_subjects) * len(unique_regions)
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects == subject_id) & (regions == region_id)
                n_samples = np.sum(mask)
                
                if n_samples >= 50:  # 最小样本阈值，确保统计可靠性
                    region_features = np.mean(X[mask], axis=0)
                    subject_region_features[(subject_id, region_id)] = region_features
                    sample_counts[(subject_id, region_id)] = n_samples
                    valid_combinations += 1
        
        logger.info(f"    ✅ 有效组合: {valid_combinations} / {total_combinations} ({valid_combinations/total_combinations*100:.1f}%)")
        
        # 统计每个脑区的受试者覆盖情况
        region_subject_counts = {}
        for region_id in unique_regions:
            region_subjects = [s for s, r in subject_region_features.keys() if r == region_id]
            region_subject_counts[region_id] = len(region_subjects)
        
        logger.info(f"    📊 脑区统计:")
        logger.info(f"      - 平均每脑区覆盖受试者数: {np.mean(list(region_subject_counts.values())):.1f}")
        logger.info(f"      - 最大覆盖受试者数: {np.max(list(region_subject_counts.values()))}")
        logger.info(f"      - 最小覆盖受试者数: {np.min(list(region_subject_counts.values()))}")
        
        return {
            'subject_region_features': subject_region_features,
            'sample_counts': sample_counts,
            'unique_subjects': unique_subjects,
            'unique_regions': unique_regions,
            'region_subject_counts': region_subject_counts,
            'valid_combinations': valid_combinations,
            'total_combinations': total_combinations
        }

    def phase1_subject_differences_analysis(self):
        """
        Phase 1: 受试者间差异分析 - 保留原有全局分析 + 新增分脑区分析
        """
        logger.info("\n" + "="*80)
        logger.info("📊 Phase 1: 受试者间差异本质分析 (增强版)")
        logger.info("="*80)
        
        # 🔄 Phase 1A: 保留原有全局分析逻辑
        logger.info("\n📊 Phase 1A: 全局受试者差异分析 (保持原有逻辑)")
        self._phase1a_global_subject_analysis()
        
        # 🔥 Phase 1B: 新增分脑区分析
        logger.info("\n📊 Phase 1B: 分脑区受试者差异分析 (新增)")
        self._phase1b_region_wise_analysis()
        
        # 综合决策得分计算
        self._compute_phase1_combined_scores()

    def _phase1a_global_subject_analysis(self):
        """Phase 1A: 保留原有的全局受试者差异分析"""
        
        # 1.1 统计分布差异诊断（保持原有逻辑）
        logger.info("🔍 1.1A 计算全局受试者统计特征...")
        
        subjects = self.data['available_subjects']
        n_features = self.data['X_train'].shape[1]
        
        # 为每个受试者计算统计量
        subject_stats = {
            'means': np.zeros((len(subjects), n_features)),
            'stds': np.zeros((len(subjects), n_features)),
            'skews': np.zeros((len(subjects), n_features)),
            'kurts': np.zeros((len(subjects), n_features)),
            'sample_counts': np.zeros(len(subjects))
        }
        
        for i, subject_id in enumerate(subjects):
            # 训练集中该受试者的数据
            subject_mask = self.data['subjects_train'] == subject_id
            if np.sum(subject_mask) > 0:
                subject_data = self.data['X_train'][subject_mask]
                
                subject_stats['means'][i] = np.mean(subject_data, axis=0)
                subject_stats['stds'][i] = np.std(subject_data, axis=0)
                subject_stats['skews'][i] = stats.skew(subject_data, axis=0)
                subject_stats['kurts'][i] = stats.kurtosis(subject_data, axis=0)
                subject_stats['sample_counts'][i] = len(subject_data)
            
            # 处理验证集
            val_mask = self.data['subjects_val'] == subject_id
            if np.sum(val_mask) > 0:
                val_data = self.data['X_val'][val_mask]
                subject_stats['sample_counts'][i] += len(val_data)
        
        self.analysis_results['subject_stats'] = subject_stats
        
        logger.info(f"✅ 全局受试者统计特征计算完成")
        logger.info(f"  - 平均每受试者样本量: {np.mean(subject_stats['sample_counts']):.0f}")
        logger.info(f"  - 样本量范围: [{np.min(subject_stats['sample_counts']):.0f}, {np.max(subject_stats['sample_counts']):.0f}]")
        
        # 1.2 受试者间相似性分析（保持原有逻辑，增强数值稳定性）
        logger.info("🔍 1.2A 分析全局受试者间相似性...")
        
        subject_means = subject_stats['means']
        
        # 数值稳定性检查
        feature_variance = np.var(subject_means, axis=0)
        constant_features = np.sum(feature_variance < 1e-12)
        valid_features_mask = feature_variance >= 1e-12
        
        logger.info(f"    - 总特征数: {n_features}")
        logger.info(f"    - 常数特征数: {constant_features}")
        logger.info(f"    - 有效特征数: {np.sum(valid_features_mask)}")
        
        if constant_features > 0:
            subject_means_filtered = subject_means[:, valid_features_mask]
        else:
            subject_means_filtered = subject_means
        
        # 距离矩阵计算
        try:
            distance_matrix = squareform(pdist(subject_means_filtered, metric='euclidean'))
            upper_triangle_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            logger.info(f"    - 平均受试者间距离: {np.mean(upper_triangle_distances):.6f}")
            logger.info(f"    - 距离标准差: {np.std(upper_triangle_distances):.6f}")
        except Exception as e:
            logger.info(f"    ❌ 距离计算失败: {e}")
            distance_matrix = np.zeros((len(subjects), len(subjects)))
        
        # 相关性矩阵计算
        try:
            correlation_matrix = np.corrcoef(subject_means_filtered)
            if np.all(np.isfinite(correlation_matrix)):
                upper_triangle_corr = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                valid_correlations = upper_triangle_corr[np.isfinite(upper_triangle_corr)]
                mean_correlation = np.mean(valid_correlations) if len(valid_correlations) > 0 else 0.0
                logger.info(f"    - 平均受试者间相关性: {mean_correlation:.6f}")
            else:
                raise ValueError("相关性矩阵包含无效值")
        except Exception as e:
            logger.info(f"    ⚠️ 标准相关性计算失败，使用备用方法")
            correlation_matrix = np.eye(len(subjects))
            mean_correlation = 0.0
        
        # 层次聚类
        try:
            linkage_matrix = linkage(subject_means_filtered, method='ward')
        except Exception as e:
            logger.info(f"    ⚠️ 层次聚类失败: {e}")
            linkage_matrix = np.zeros((len(subjects)-1, 4))
        
        # 保存全局相似性分析结果
        self.analysis_results['global_subject_similarity'] = {
            'distance_matrix': distance_matrix,
            'correlation_matrix': correlation_matrix,
            'linkage_matrix': linkage_matrix,
            'mean_correlation': mean_correlation,
            'constant_features_count': constant_features,
            'valid_features_count': np.sum(valid_features_mask)
        }
        
        # 1.3 特征变异模式分析（保持原有逻辑）
        logger.info("🔍 1.3A 分析全局特征变异模式...")
        
        feature_f_stats = np.var(subject_means, axis=0)
        feature_f_stats_norm = feature_f_stats / (np.mean(feature_f_stats) + 1e-8)
        
        high_variation_threshold = np.percentile(feature_f_stats_norm, 90)
        low_variation_threshold = np.percentile(feature_f_stats_norm, 10)
        
        high_variation_features = np.where(feature_f_stats_norm > high_variation_threshold)[0]
        low_variation_features = np.where(feature_f_stats_norm < low_variation_threshold)[0]
        
        # PCA分析
        pca = PCA()
        subject_pca_result = pca.fit_transform(subject_means)
        
        self.analysis_results['global_feature_variation'] = {
            'f_stats': feature_f_stats_norm,
            'high_variation_features': high_variation_features,
            'low_variation_features': low_variation_features,
            'pca_result': subject_pca_result,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'pca_cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
        
        logger.info(f"✅ 全局特征变异分析完成")
        logger.info(f"  - 高变异特征数量: {len(high_variation_features)} ({len(high_variation_features)/n_features*100:.1f}%)")
        logger.info(f"  - 低变异特征数量: {len(low_variation_features)} ({len(low_variation_features)/n_features*100:.1f}%)")
        logger.info(f"  - 前3个主成分解释方差: {np.sum(pca.explained_variance_ratio_[:3])*100:.1f}%")

    def _phase1b_region_wise_analysis(self):
        """Phase 1B: 新增的分脑区受试者差异分析"""
        
        brain_data = self.data['brain_region_analysis']
        subject_region_features = brain_data['subject_region_features']
        unique_regions = brain_data['unique_regions']
        
        region_wise_results = {}
        
        logger.info(f"🔍 1.1B 分析每个脑区的受试者特异性...")
        logger.info(f"    - 总脑区数: {len(unique_regions)}")
        logger.info(f"    - 有效组合数: {len(subject_region_features)}")
        
        processed_regions = 0
        
        for region_id in unique_regions:
            # 提取该脑区所有受试者的特征
            region_subject_features = []
            valid_subjects = []
            
            for subject_id in brain_data['unique_subjects']:
                if (subject_id, region_id) in subject_region_features:
                    region_subject_features.append(subject_region_features[(subject_id, region_id)])
                    valid_subjects.append(subject_id)
            
            if len(region_subject_features) >= 5:  # 至少5个受试者有该脑区
                region_features_array = np.array(region_subject_features)
                
                # 受试者间距离分析
                try:
                    distances = squareform(pdist(region_features_array))
                    mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                    std_distance = np.std(distances[np.triu_indices_from(distances, k=1)])
                except:
                    mean_distance = 0.0
                    std_distance = 1.0
                
                # 受试者间相关性
                try:
                    correlations = np.corrcoef(region_features_array)
                    if np.all(np.isfinite(correlations)):
                        mean_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
                    else:
                        mean_correlation = 0.0
                except:
                    mean_correlation = 0.0
                
                # PCA分析该脑区的受试者差异模式
                try:
                    pca = PCA()
                    pca_result = pca.fit_transform(region_features_array)
                    pca_3pc_variance = np.sum(pca.explained_variance_ratio_[:3]) if len(pca.explained_variance_ratio_) >= 3 else np.sum(pca.explained_variance_ratio_)
                except:
                    pca = None
                    pca_3pc_variance = 0.0
                
                # 受试者特异性得分 (距离相对于变异度)
                specificity_score = mean_distance / (std_distance + 1e-8)
                
                region_wise_results[region_id] = {
                    'n_subjects': len(valid_subjects),
                    'mean_inter_subject_distance': mean_distance,
                    'std_inter_subject_distance': std_distance,
                    'mean_inter_subject_correlation': abs(mean_correlation),
                    'pca_explained_variance': pca.explained_variance_ratio_ if pca else np.array([0]),
                    'pca_3pc_variance': pca_3pc_variance,
                    'subject_specificity_score': specificity_score,
                    'valid_subjects': valid_subjects,
                    'sample_sizes': [brain_data['sample_counts'][(s, region_id)] for s in valid_subjects]
                }
                
                processed_regions += 1
        
        self.analysis_results['region_wise_subject_analysis'] = region_wise_results
        
        # 计算脑区特异性统计
        if region_wise_results:
            specificity_scores = [r['subject_specificity_score'] for r in region_wise_results.values()]
            distances = [r['mean_inter_subject_distance'] for r in region_wise_results.values()]
            correlations = [r['mean_inter_subject_correlation'] for r in region_wise_results.values()]
            
            logger.info(f"✅ 分脑区分析完成")
            logger.info(f"  - 成功分析脑区数量: {processed_regions}")
            logger.info(f"  - 特异性得分范围: [{np.min(specificity_scores):.3f}, {np.max(specificity_scores):.3f}]")
            logger.info(f"  - 平均特异性得分: {np.mean(specificity_scores):.3f}")
            logger.info(f"  - 平均受试者间距离: {np.mean(distances):.3f}")
            logger.info(f"  - 平均受试者间相关性: {np.mean(correlations):.3f}")
            
            # 识别高/低特异性脑区
            high_threshold = np.percentile(specificity_scores, 75)
            low_threshold = np.percentile(specificity_scores, 25)
            
            high_specificity_regions = [r_id for r_id, r_data in region_wise_results.items() 
                                       if r_data['subject_specificity_score'] > high_threshold]
            low_specificity_regions = [r_id for r_id, r_data in region_wise_results.items() 
                                      if r_data['subject_specificity_score'] < low_threshold]
            
            logger.info(f"  - 高特异性脑区 (>75th): {len(high_specificity_regions)} 个")
            logger.info(f"  - 低特异性脑区 (<25th): {len(low_specificity_regions)} 个")
            
            # 显示极端脑区
            if high_specificity_regions:
                top_region = max(high_specificity_regions, 
                               key=lambda x: region_wise_results[x]['subject_specificity_score'])
                logger.info(f"  - 最高特异性脑区: {top_region} (得分: {region_wise_results[top_region]['subject_specificity_score']:.3f})")
            
            if low_specificity_regions:
                bottom_region = min(low_specificity_regions,
                                  key=lambda x: region_wise_results[x]['subject_specificity_score'])
                logger.info(f"  - 最低特异性脑区: {bottom_region} (得分: {region_wise_results[bottom_region]['subject_specificity_score']:.3f})")
        
        else:
            logger.info(f"❌ 没有足够数据进行分脑区分析")

    def _compute_phase1_combined_scores(self):
        """计算Phase 1的综合决策得分"""
        
        # 原有全局得分
        if 'global_feature_variation' in self.analysis_results:
            global_pca = self.analysis_results['global_feature_variation']
            variance_explained_3pc = np.sum(global_pca['pca_explained_variance'][:3])
            
            global_correlation = 0.0
            if 'global_subject_similarity' in self.analysis_results:
                global_correlation = abs(self.analysis_results['global_subject_similarity']['mean_correlation'])
        else:
            variance_explained_3pc = 0.5
            global_correlation = 0.0
        
        # 🔥 新增：分脑区得分
        region_specificity_diversity = 0.0
        high_specificity_ratio = 0.0
        
        if 'region_wise_subject_analysis' in self.analysis_results:
            region_results = self.analysis_results['region_wise_subject_analysis']
            if region_results:
                specificity_scores = [r['subject_specificity_score'] for r in region_results.values()]
                
                # 脑区特异性多样性：标准差/均值 (值越大，脑区间差异越大)
                region_specificity_diversity = np.std(specificity_scores) / (np.mean(specificity_scores) + 1e-8)
                
                # 高特异性脑区比例
                high_threshold = np.percentile(specificity_scores, 75)
                high_specificity_count = np.sum(np.array(specificity_scores) > high_threshold)
                high_specificity_ratio = high_specificity_count / len(specificity_scores)
        
        # Phase 1 综合决策得分
        self.decision_scores['phase1'] = {
            # 原有得分
            'difference_significance': 1.0 if variance_explained_3pc > 0.6 else 0.8 if variance_explained_3pc > 0.4 else 0.5,
            'pattern_linearity': variance_explained_3pc,
            'subject_similarity': global_correlation,
            'feature_heterogeneity': len(self.analysis_results.get('global_feature_variation', {}).get('high_variation_features', [])) / self.data['X_train'].shape[1],
            
            # 🔥 新增得分
            'region_specificity_diversity': region_specificity_diversity,
            'high_specificity_ratio': high_specificity_ratio,
            'region_analysis_success': 1.0 if 'region_wise_subject_analysis' in self.analysis_results else 0.0
        }
        
        logger.info(f"\n📈 Phase 1 综合决策指标:")
        logger.info(f"  - 差异显著性得分: {self.decision_scores['phase1']['difference_significance']:.3f}")
        logger.info(f"  - 模式线性度: {self.decision_scores['phase1']['pattern_linearity']:.3f}")
        logger.info(f"  - 受试者相似性: {self.decision_scores['phase1']['subject_similarity']:.3f}")
        logger.info(f"  - 特征异质性: {self.decision_scores['phase1']['feature_heterogeneity']:.3f}")
        logger.info(f"  🔥 脑区特异性多样性: {self.decision_scores['phase1']['region_specificity_diversity']:.3f}")
        logger.info(f"  🔥 高特异性脑区比例: {self.decision_scores['phase1']['high_specificity_ratio']:.3f}")

    def phase2_subject_separability_analysis(self):
        """
        Phase 2: 受试者可分离性评估 - 保留原有全局分析 + 新增分脑区分析
        """
        logger.info("\n" + "="*80)
        logger.info("📊 Phase 2: 受试者可分离性评估 (增强版)")
        logger.info("="*80)
        
        # 🔄 Phase 2A: 修正版全局受试者可分离性分析
        logger.info("\n📊 Phase 2A: 全局受试者可分离性分析 (修正版)")
        self._phase2a_global_separability_corrected()
        
        # 🔥 Phase 2B: 新增分脑区可分离性分析
        logger.info("\n📊 Phase 2B: 分脑区受试者可分离性分析 (新增)")
        self._phase2b_region_wise_separability()
        
        # Phase 2C: 保留原有的脑区分类一致性分析
        logger.info("\n📊 Phase 2C: 脑区分类一致性分析 (保持原有)")
        self._phase2c_class_consistency_analysis()
        
        # 综合决策得分计算
        self._compute_phase2_combined_scores()

    def _phase2a_global_separability_corrected(self):
        """修正版：神经网络baseline性能分析"""
        
        logger.info("🔍 2.1A 神经网络baseline性能分析...")
        
        X_train = self.data['X_train_scaled']
        y_train = self.data['y_train']
        subjects_train = self.data['subjects_train']
        
        # 转换标签
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_classes = np.argmax(y_train, axis=1)
        else:
            y_classes = y_train.flatten()
        
        logger.info(f"    - 体素数量: {len(X_train):,}")
        logger.info(f"    - 特征维度: {X_train.shape[1]}")
        logger.info(f"    - 脑区类别数: {len(np.unique(y_classes))}")
        
        # 1. Baseline性能：随机分割（模拟你的训练方式）
        baseline_results = self._test_random_split_performance(X_train, y_classes)
        
        # 2. 关键测试：Leave-One-Subject-Out性能（真实泛化性能）
        loso_results = self._test_leave_one_subject_out_performance(
            X_train, y_classes, subjects_train)
        
        # 3. Subject Embedding潜在价值评估
        embedding_value_assessment = self._assess_subject_embedding_value(
            baseline_results, loso_results)
        
        # 保存结果
        self.analysis_results['neural_network_baseline_analysis'] = {
            'baseline_random_split': baseline_results,
            'leave_one_subject_out': loso_results,
            'embedding_value_assessment': embedding_value_assessment
        }
        
        return {
            'baseline_random_split': baseline_results,
            'leave_one_subject_out': loso_results,
            'embedding_value_assessment': embedding_value_assessment
        }

    def _phase2b_region_wise_separability(self):
        """修正版分脑区Subject Embedding需求分析"""
        
        logger.info("🔍 2.1B 分脑区Subject Embedding需求分析...")
        
        if 'region_wise_subject_analysis' not in self.analysis_results:
            logger.info("❌ 需要先运行Phase 1B")
            return
        
        region_wise_results = self.analysis_results['region_wise_subject_analysis']
        region_embedding_analysis = {}
        
        logger.info(f"    - 待分析脑区数: {len(region_wise_results)}")
        
        # 🎯 核心分析：每个脑区的Subject Embedding需求评估
        for region_id, region_data in region_wise_results.items():
            if region_data['n_subjects'] >= 5:  # 确保足够受试者
                
                logger.info(f"    🧠 分析脑区{region_id}的Subject Embedding需求...")
                
                # 1. 脑区分类一致性分析
                consistency_analysis = self._analyze_region_classification_consistency(region_id)
                
                # 2. 交叉受试者分类测试
                cross_subject_analysis = self._cross_subject_region_classification_test(region_id)
                
                # 3. 受试者特异性影响评估
                specificity_impact = self._assess_subject_specificity_impact(region_id)
                
                # 4. Subject Embedding需求评级
                embedding_necessity = self._compute_embedding_necessity_score(
                    consistency_analysis, cross_subject_analysis, specificity_impact
                )
                
                region_embedding_analysis[region_id] = {
                    'consistency_analysis': consistency_analysis,
                    'cross_subject_analysis': cross_subject_analysis,
                    'specificity_impact': specificity_impact,
                    'embedding_necessity_score': embedding_necessity['score'],
                    'embedding_necessity_level': embedding_necessity['level'],
                    'recommended_embedding_dim': embedding_necessity['recommended_dim'],
                    'implementation_priority': embedding_necessity['priority']
                }
                
                logger.info(f"      ✅ 脑区{region_id}: {embedding_necessity['level']} "
                    f"(得分: {embedding_necessity['score']:.3f}, "
                    f"推荐维度: {embedding_necessity['recommended_dim']})")
        
        self.analysis_results['region_wise_separability'] = region_embedding_analysis
        
        # 🔥 保留原有的完整结果分析，但重新解释含义
        if region_embedding_analysis:
            necessity_scores = [r['embedding_necessity_score'] for r in region_embedding_analysis.values()]
            recommended_dims = [r['recommended_embedding_dim'] for r in region_embedding_analysis.values()]
            
            logger.info(f"✅ 分脑区Subject Embedding需求分析完成")
            logger.info(f"  - 成功分析脑区数量: {len(region_embedding_analysis)}")
            logger.info(f"  - 平均embedding需求得分: {np.mean(necessity_scores):.3f}")
            logger.info(f"  - 需求得分范围: [{np.min(necessity_scores):.3f}, {np.max(necessity_scores):.3f}]")
            logger.info(f"  - 平均推荐embedding维度: {np.mean(recommended_dims):.1f}")
            
            # 按需求等级分类
            critical_regions = [rid for rid, data in region_embedding_analysis.items() 
                            if data['embedding_necessity_level'] == 'CRITICAL']
            high_regions = [rid for rid, data in region_embedding_analysis.items() 
                        if data['embedding_necessity_level'] == 'HIGH']
            medium_regions = [rid for rid, data in region_embedding_analysis.items() 
                            if data['embedding_necessity_level'] == 'MEDIUM']
            low_regions = [rid for rid, data in region_embedding_analysis.items() 
                        if data['embedding_necessity_level'] == 'LOW']
            
            logger.info(f"  - CRITICAL级别脑区: {len(critical_regions)} 个")
            logger.info(f"  - HIGH级别脑区: {len(high_regions)} 个") 
            logger.info(f"  - MEDIUM级别脑区: {len(medium_regions)} 个")
            logger.info(f"  - LOW级别脑区: {len(low_regions)} 个")
            
            # 识别最需要和最不需要embedding的脑区
            if len(region_embedding_analysis) > 0:
                most_needed_region = max(region_embedding_analysis.keys(), 
                                    key=lambda x: region_embedding_analysis[x]['embedding_necessity_score'])
                least_needed_region = min(region_embedding_analysis.keys(), 
                                        key=lambda x: region_embedding_analysis[x]['embedding_necessity_score'])
                
                logger.info(f"  - 最需要Subject Embedding的脑区: {most_needed_region} "
                    f"({region_embedding_analysis[most_needed_region]['embedding_necessity_score']:.3f})")
                logger.info(f"  - 最不需要Subject Embedding的脑区: {least_needed_region} "
                    f"({region_embedding_analysis[least_needed_region]['embedding_necessity_score']:.3f})")
        else:
            logger.info(f"❌ 没有成功的分脑区Subject Embedding分析")

    def _analyze_region_classification_consistency(self, region_id):
        """分析单个脑区的分类一致性"""
        
        # 获取该脑区的所有体素
        y_labels = np.argmax(self.data['y_train'], axis=1)
        region_mask = y_labels == region_id
        
        if np.sum(region_mask) < 100:  # 样本量太少
            return {'status': 'insufficient_data', 'n_voxels': np.sum(region_mask)}
        
        region_features = self.data['X_train'][region_mask]
        region_subjects = self.data['subjects_train'][region_mask]
        
        # 计算每个受试者该脑区的特征统计
        subject_stats = {}
        for subject_id in np.unique(region_subjects):
            subject_mask = region_subjects == subject_id
            if np.sum(subject_mask) >= 10:  # 至少10个体素
                subject_data = region_features[subject_mask]
                subject_stats[subject_id] = {
                    'mean': np.mean(subject_data, axis=0),
                    'std': np.std(subject_data, axis=0),
                    'n_voxels': len(subject_data)
                }
        
        if len(subject_stats) < 3:  # 受试者数量不足
            return {'status': 'insufficient_subjects', 'n_subjects': len(subject_stats)}
        
        # 计算受试者间变异
        all_means = np.array([stats['mean'] for stats in subject_stats.values()])
        feature_cv = np.std(all_means, axis=0) / (np.abs(np.mean(all_means, axis=0)) + 1e-8)
        mean_cv = np.mean(feature_cv)
        
        # 计算受试者间距离
        from scipy.spatial.distance import pdist
        try:
            distances = pdist(all_means)
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
        except:
            mean_distance = 0
            std_distance = 0
        
        return {
            'status': 'success',
            'n_subjects': len(subject_stats),
            'n_voxels': np.sum(region_mask),
            'mean_cv': mean_cv,
            'mean_inter_subject_distance': mean_distance,
            'std_inter_subject_distance': std_distance,
            'consistency_score': 1.0 / (1.0 + mean_cv)  # 越一致分数越高
        }

    def _cross_subject_region_classification_test(self, region_id):
        """交叉受试者脑区分类测试"""
        
        # 准备该脑区 vs 其他脑区的二分类数据
        y_labels = np.argmax(self.data['y_train'], axis=1)
        
        # 创建二分类标签：该脑区=1，其他脑区=0
        binary_labels = (y_labels == region_id).astype(int)
        
        # 确保正负样本平衡
        positive_indices = np.where(binary_labels == 1)[0]
        negative_indices = np.where(binary_labels == 0)[0]
        
        if len(positive_indices) < 100 or len(negative_indices) < 100:
            return {'status': 'insufficient_data'}
        
        # 随机采样保持平衡
        n_samples = min(len(positive_indices), len(negative_indices), 5000)  # 限制样本量提高效率
        
        np.random.seed(42)
        selected_positive = np.random.choice(positive_indices, n_samples, replace=False)
        selected_negative = np.random.choice(negative_indices, n_samples, replace=False)
        
        selected_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(selected_indices)
        
        X_balanced = self.data['X_train'][selected_indices]
        y_balanced = binary_labels[selected_indices]
        subjects_balanced = self.data['subjects_train'][selected_indices]
        
        # 留一受试者交叉验证
        unique_subjects = np.unique(subjects_balanced)
        if len(unique_subjects) < 5:
            return {'status': 'insufficient_subjects'}
        
        cross_subject_scores = []
        
        # 随机选择5个受试者进行测试
        test_subjects = np.random.choice(unique_subjects, min(5, len(unique_subjects)), replace=False)
        
        for test_subject in test_subjects:
            train_mask = subjects_balanced != test_subject
            test_mask = subjects_balanced == test_subject
            
            if np.sum(test_mask) < 10:  # 测试样本太少
                continue
                
            X_train_cv = X_balanced[train_mask]
            y_train_cv = y_balanced[train_mask]
            X_test_cv = X_balanced[test_mask]
            y_test_cv = y_balanced[test_mask]
            
            # 训练简单分类器
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train_cv, y_train_cv)
            
            score = clf.score(X_test_cv, y_test_cv)
            cross_subject_scores.append(score)
        
        if len(cross_subject_scores) == 0:
            return {'status': 'no_valid_tests'}
        
        # 计算与随机分类器的对比
        random_baseline = 0.5  # 二分类随机基线
        mean_accuracy = np.mean(cross_subject_scores)
        improvement_over_random = (mean_accuracy - random_baseline) / random_baseline
        
        return {
            'status': 'success',
            'cross_subject_accuracies': cross_subject_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': np.std(cross_subject_scores),
            'random_baseline': random_baseline,
            'improvement_over_random': improvement_over_random,
            'n_tests': len(cross_subject_scores),
            'generalization_quality': mean_accuracy  # 泛化质量
        }

    def _assess_subject_specificity_impact(self, region_id):
        """评估受试者特异性对该脑区分类的影响"""
        
        y_labels = np.argmax(self.data['y_train'], axis=1)
        binary_labels = (y_labels == region_id).astype(int)
        
        # 同样的数据平衡处理
        positive_indices = np.where(binary_labels == 1)[0]
        negative_indices = np.where(binary_labels == 0)[0]
        
        if len(positive_indices) < 100 or len(negative_indices) < 100:
            return {'status': 'insufficient_data'}
        
        n_samples = min(len(positive_indices), len(negative_indices), 3000)
        
        np.random.seed(42)
        selected_positive = np.random.choice(positive_indices, n_samples, replace=False)
        selected_negative = np.random.choice(negative_indices, n_samples, replace=False)
        
        selected_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(selected_indices)
        
        X_balanced = self.data['X_train'][selected_indices]
        y_balanced = binary_labels[selected_indices]
        subjects_balanced = self.data['subjects_train'][selected_indices]
        
        # 1. 随机分割基线性能（忽略受试者）
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42
        )
        
        clf_random = LogisticRegression(random_state=42, max_iter=1000)
        clf_random.fit(X_train_random, y_train_random)
        random_split_accuracy = clf_random.score(X_test_random, y_test_random)
        
        # 2. 受试者分割性能（考虑受试者）
        unique_subjects = np.unique(subjects_balanced)
        if len(unique_subjects) < 4:
            return {'status': 'insufficient_subjects'}
        
        # 选择一个受试者作为测试
        test_subject = np.random.choice(unique_subjects)
        train_mask = subjects_balanced != test_subject
        test_mask = subjects_balanced == test_subject
        
        if np.sum(test_mask) < 20:
            return {'status': 'insufficient_test_data'}
        
        X_train_subject = X_balanced[train_mask]
        y_train_subject = y_balanced[train_mask]
        X_test_subject = X_balanced[test_mask]
        y_test_subject = y_balanced[test_mask]
        
        clf_subject = LogisticRegression(random_state=42, max_iter=1000)
        clf_subject.fit(X_train_subject, y_train_subject)
        subject_split_accuracy = clf_subject.score(X_test_subject, y_test_subject)
        
        # 3. 计算受试者特异性影响
        subject_impact = random_split_accuracy - subject_split_accuracy
        impact_ratio = subject_impact / random_split_accuracy if random_split_accuracy > 0 else 0
        
        return {
            'status': 'success',
            'random_split_accuracy': random_split_accuracy,
            'subject_split_accuracy': subject_split_accuracy,
            'subject_specificity_impact': subject_impact,
            'impact_ratio': impact_ratio,
            'embedding_benefit_potential': max(0, impact_ratio)  # 潜在收益
        }

    def _compute_embedding_necessity_score(self, consistency_analysis, cross_subject_analysis, specificity_impact):
        """计算Subject Embedding必要性评分"""
        
        # 检查所有分析是否成功
        if (consistency_analysis.get('status') != 'success' or 
            cross_subject_analysis.get('status') != 'success' or 
            specificity_impact.get('status') != 'success'):
            return {
                'score': 0.0,
                'level': 'INSUFFICIENT_DATA',
                'recommended_dim': 0,
                'priority': 'SKIP'
            }
        
        # 计算综合评分 (0-1之间)
        
        # 1. 一致性评分 (越不一致越需要embedding)
        consistency_score = 1.0 - consistency_analysis['consistency_score']
        consistency_score = max(0, min(1, consistency_score))
        
        # 2. 泛化质量评分 (泛化性能越差越需要embedding)
        generalization_score = 1.0 - cross_subject_analysis['generalization_quality']
        generalization_score = max(0, min(1, generalization_score))
        
        # 3. 受试者影响评分
        impact_score = specificity_impact['embedding_benefit_potential']
        impact_score = max(0, min(1, impact_score))
        
        # 综合评分 (加权平均)
        weights = {'consistency': 0.4, 'generalization': 0.3, 'impact': 0.3}
        
        overall_score = (
            consistency_score * weights['consistency'] +
            generalization_score * weights['generalization'] + 
            impact_score * weights['impact']
        )
        
        # 确定等级和推荐维度
        if overall_score > 0.7:
            level = 'CRITICAL'
            recommended_dim = 128
            priority = 'HIGH'
        elif overall_score > 0.5:
            level = 'HIGH'
            recommended_dim = 64
            priority = 'MEDIUM-HIGH'
        elif overall_score > 0.3:
            level = 'MEDIUM'
            recommended_dim = 32
            priority = 'MEDIUM'
        else:
            level = 'LOW'
            recommended_dim = 16
            priority = 'LOW'
        
        return {
            'score': overall_score,
            'level': level,
            'recommended_dim': recommended_dim,
            'priority': priority,
            'component_scores': {
                'consistency': consistency_score,
                'generalization': generalization_score,
                'impact': impact_score
            }
        }


    def _test_random_split_performance(self, X, y):
        """测试随机分割的baseline性能 (增强版：全局 + 分脑区分析)"""
        
        logger.info("    🔍 增强版Baseline随机分割性能测试...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        classifiers = {
            'complex_rf': RandomForestClassifier(
                n_estimators=200, max_depth=20, 
                min_samples_split=2, random_state=42),
            'simple_lr': LogisticRegression(
                max_iter=1000, C=0.1, random_state=42)
        }
        
        results = {
            'global_analysis': {},
            'region_wise_analysis': {},
            'comparative_analysis': {}
        }
        
        # ========================================================================
        # 1. 保留原有全局分析
        # ========================================================================
        logger.info("      🌐 全局随机分割性能...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        for name, clf in classifiers.items():
            logger.info(f"        🔍 训练全局{name}...")
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            
            results['global_analysis'][name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_val),
                'n_classes': len(np.unique(y))
            }
            
            logger.info(f"          ✅ 全局{name}: Acc={accuracy:.3f}, F1_macro={f1_macro:.3f}")
        
        # ========================================================================
        # 2. 新增分脑区分析
        # ========================================================================
        logger.info("      🧠 分脑区随机分割性能...")
        
        region_dataset = self._build_region_aware_dataset()
        
        if len(region_dataset['features']) > 200:  # 确保有足够样本
            
            X_region = region_dataset['features']
            subject_labels_region = region_dataset['subject_labels']
            
            # 重新映射受试者标签
            unique_subjects = np.unique(subject_labels_region)
            subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
            y_region = np.array([subject_mapping[s] for s in subject_labels_region])
            
            logger.info(f"        📊 分脑区数据集: {len(X_region)} 样本, {len(unique_subjects)} 受试者")
            
            try:
                X_train_region, X_val_region, y_train_region, y_val_region = train_test_split(
                    X_region, y_region, test_size=0.2, stratify=y_region, random_state=42)
                
                for name, clf in classifiers.items():
                    logger.info(f"        🔍 训练分脑区{name}...")
                    
                    # 使用新的分类器实例
                    clf_region = type(clf)(**clf.get_params())
                    clf_region.fit(X_train_region, y_train_region)
                    
                    y_pred_region = clf_region.predict(X_val_region)
                    accuracy_region = accuracy_score(y_val_region, y_pred_region)
                    f1_macro_region = f1_score(y_val_region, y_pred_region, average='macro')
                    f1_weighted_region = f1_score(y_val_region, y_pred_region, average='weighted')
                    
                    results['region_wise_analysis'][name] = {
                        'accuracy': accuracy_region,
                        'f1_macro': f1_macro_region,
                        'f1_weighted': f1_weighted_region,
                        'n_train_samples': len(X_train_region),
                        'n_test_samples': len(X_val_region),
                        'n_classes': len(unique_subjects),
                        'n_regions': len(np.unique(region_dataset['region_labels']))
                    }
                    
                    logger.info(f"          ✅ 分脑区{name}: Acc={accuracy_region:.3f}, F1_macro={f1_macro_region:.3f}")
                    
            except Exception as e:
                logger.info(f"        ❌ 分脑区分割失败: {e}")
                results['region_wise_analysis'] = {'error': str(e)}
        
        else:
            logger.info("        ⚠️ 分脑区样本不足，跳过分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # ========================================================================
        # 3. 性能对比分析
        # ========================================================================
        logger.info("      📈 Baseline性能对比分析...")
        
        for name in classifiers.keys():
            if (name in results['global_analysis'] and 
                name in results['region_wise_analysis'] and
                'accuracy' in results['global_analysis'][name] and
                'accuracy' in results['region_wise_analysis'][name]):
                
                global_metrics = results['global_analysis'][name]
                region_metrics = results['region_wise_analysis'][name]
                
                # 计算各指标的提升
                acc_improvement = region_metrics['accuracy'] - global_metrics['accuracy']
                f1_improvement = region_metrics['f1_macro'] - global_metrics['f1_macro']
                
                results['comparative_analysis'][name] = {
                    'accuracy_improvement': acc_improvement,
                    'f1_macro_improvement': f1_improvement,
                    'accuracy_improvement_percent': acc_improvement / global_metrics['accuracy'] * 100,
                    'f1_improvement_percent': f1_improvement / global_metrics['f1_macro'] * 100,
                    'sample_size_ratio': region_metrics['n_train_samples'] / global_metrics['n_train_samples'],
                    'interpretation': self._interpret_baseline_improvement(acc_improvement, f1_improvement)
                }
                
                logger.info(f"        📊 {name}对比:")
                logger.info(f"          准确率: {global_metrics['accuracy']:.3f} → {region_metrics['accuracy']:.3f} "
                    f"({acc_improvement:+.3f})")
                logger.info(f"          F1分数: {global_metrics['f1_macro']:.3f} → {region_metrics['f1_macro']:.3f} "
                    f"({f1_improvement:+.3f})")
        
        return results

    def _interpret_baseline_improvement(self, acc_improvement, f1_improvement):
        """解释baseline性能改进"""
        if acc_improvement > 0.05 and f1_improvement > 0.05:
            return "显著改善：分脑区方法在准确率和F1分数上都有明显提升"
        elif acc_improvement > 0.02 or f1_improvement > 0.02:
            return "中等改善：分脑区方法在某些指标上有所提升"
        elif acc_improvement > -0.02 and f1_improvement > -0.02:
            return "性能相当：两种方法表现相似"
        else:
            return "性能下降：全局方法可能更适合baseline任务"
        


    def _test_leave_one_subject_out_performance(self, X, y, subjects):
        """测试Leave-One-Subject-Out性能 (增强版：全局 + 分脑区分析)"""
        
        logger.info("    🔍 增强版Leave-One-Subject-Out性能测试...")
        
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        
        results = {
            'global_analysis': {
                'per_subject_scores': {},
                'overall_performance': {}
            },
            'region_wise_analysis': {
                'per_subject_scores': {},
                'overall_performance': {},
                'per_region_performance': {}
            },
            'comparative_analysis': {}
        }
        
        # ========================================================================
        # 1. 保留原有全局LOSO分析
        # ========================================================================
        logger.info("      🌐 全局LOSO性能分析...")
        
        unique_subjects = np.unique(subjects)
        all_accuracies_global = []
        all_f1_scores_global = []
        
        # 只测试部分受试者以节省时间
        test_subjects = unique_subjects[:8]
        
        for test_subject in test_subjects:
            logger.info(f"        🔍 全局测试受试者{test_subject}...")
            
            train_mask = subjects != test_subject
            test_mask = subjects == test_subject
            
            if np.sum(test_mask) < 1000:
                logger.info(f"          ⚠️ 受试者{test_subject}样本不足，跳过")
                continue
            
            X_train_global, X_test_global = X[train_mask], X[test_mask]
            y_train_global, y_test_global = y[train_mask], y[test_mask]
            
            # 检查类别平衡
            train_classes = len(np.unique(y_train_global))
            test_classes = len(np.unique(y_test_global))
            
            if train_classes < 50 or test_classes < 20:
                logger.info(f"          ⚠️ 受试者{test_subject}类别不足，跳过")
                continue
            
            # 训练全局分类器
            clf_global = RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=42)
            clf_global.fit(X_train_global, y_train_global)
            
            # 测试全局性能
            y_pred_global = clf_global.predict(X_test_global)
            accuracy_global = accuracy_score(y_test_global, y_pred_global)
            f1_macro_global = f1_score(y_test_global, y_pred_global, average='macro')
            
            all_accuracies_global.append(accuracy_global)
            all_f1_scores_global.append(f1_macro_global)
            
            results['global_analysis']['per_subject_scores'][test_subject] = {
                'accuracy': accuracy_global,
                'f1_macro': f1_macro_global,
                'n_test_samples': len(X_test_global),
                'n_test_classes': test_classes
            }
            
            logger.info(f"          ✅ 全局Acc={accuracy_global:.3f}, F1={f1_macro_global:.3f}")
        
        # 全局总体统计
        if all_accuracies_global:
            results['global_analysis']['overall_performance'] = {
                'mean_accuracy': np.mean(all_accuracies_global),
                'std_accuracy': np.std(all_accuracies_global),
                'mean_f1_macro': np.mean(all_f1_scores_global),
                'std_f1_macro': np.std(all_f1_scores_global),
                'n_tested_subjects': len(all_accuracies_global)
            }
            
            logger.info(f"      📊 全局LOSO总体性能:")
            logger.info(f"        - 平均准确率: {np.mean(all_accuracies_global):.3f} ± {np.std(all_accuracies_global):.3f}")
            logger.info(f"        - 平均F1: {np.mean(all_f1_scores_global):.3f} ± {np.std(all_f1_scores_global):.3f}")
        
        # ========================================================================
        # 2. 新增分脑区LOSO分析
        # ========================================================================
        logger.info("      🧠 分脑区LOSO性能分析...")
        
        region_dataset = self._build_region_aware_dataset()
        
        if len(region_dataset['features']) > 500:  # 确保有足够样本
            
            X_region_full = region_dataset['features']
            subject_labels_region_full = region_dataset['subject_labels']
            region_labels_region_full = region_dataset['region_labels']
            
            # 重新映射受试者标签
            unique_subjects_region = np.unique(subject_labels_region_full)
            subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects_region)}
            y_region_full = np.array([subject_mapping[s] for s in subject_labels_region_full])
            
            all_accuracies_region = []
            all_f1_scores_region = []
            region_performance_summary = {}
            
            # 测试部分受试者
            test_subjects_region = [s for s in test_subjects if s in unique_subjects_region][:5]
            
            for test_subject in test_subjects_region:
                logger.info(f"        🔍 分脑区测试受试者{test_subject}...")
                
                # 找到该受试者在分脑区数据中的所有样本
                test_mask_region = subject_labels_region_full == test_subject
                train_mask_region = subject_labels_region_full != test_subject
                
                if np.sum(test_mask_region) < 10:
                    logger.info(f"          ⚠️ 受试者{test_subject}分脑区样本不足，跳过")
                    continue
                
                X_train_region = X_region_full[train_mask_region]
                y_train_region = y_region_full[train_mask_region]
                X_test_region = X_region_full[test_mask_region]
                y_test_region = y_region_full[test_mask_region]
                
                # 检查类别平衡
                train_classes_region = len(np.unique(y_train_region))
                test_classes_region = len(np.unique(y_test_region))
                
                if train_classes_region < 5:
                    logger.info(f"          ⚠️ 受试者{test_subject}训练类别不足，跳过")
                    continue
                
                # 训练分脑区分类器
                clf_region = RandomForestClassifier(
                    n_estimators=100, max_depth=15, random_state=42)
                clf_region.fit(X_train_region, y_train_region)
                
                # 测试分脑区性能
                y_pred_region = clf_region.predict(X_test_region)
                accuracy_region = accuracy_score(y_test_region, y_pred_region)
                f1_macro_region = f1_score(y_test_region, y_pred_region, average='macro')
                
                all_accuracies_region.append(accuracy_region)
                all_f1_scores_region.append(f1_macro_region)
                
                # 分析该受试者每个脑区的表现
                test_regions = region_labels_region_full[test_mask_region]
                region_performance = {}
                
                for region_id in np.unique(test_regions):
                    region_test_mask = test_regions == region_id
                    if np.sum(region_test_mask) >= 3:  # 至少3个样本
                        region_acc = accuracy_score(
                            y_test_region[region_test_mask], 
                            y_pred_region[region_test_mask]
                        )
                        region_performance[region_id] = {
                            'accuracy': region_acc,
                            'n_samples': np.sum(region_test_mask)
                        }
                
                results['region_wise_analysis']['per_subject_scores'][test_subject] = {
                    'accuracy': accuracy_region,
                    'f1_macro': f1_macro_region,
                    'n_test_samples': len(X_test_region),
                    'n_test_regions': len(np.unique(test_regions)),
                    'per_region_performance': region_performance
                }
                
                logger.info(f"          ✅ 分脑区Acc={accuracy_region:.3f}, F1={f1_macro_region:.3f}")
                logger.info(f"          📊 测试脑区数: {len(np.unique(test_regions))}")
            
            # 分脑区总体统计
            if all_accuracies_region:
                results['region_wise_analysis']['overall_performance'] = {
                    'mean_accuracy': np.mean(all_accuracies_region),
                    'std_accuracy': np.std(all_accuracies_region),
                    'mean_f1_macro': np.mean(all_f1_scores_region),
                    'std_f1_macro': np.std(all_f1_scores_region),
                    'n_tested_subjects': len(all_accuracies_region)
                }
                
                logger.info(f"      📊 分脑区LOSO总体性能:")
                logger.info(f"        - 平均准确率: {np.mean(all_accuracies_region):.3f} ± {np.std(all_accuracies_region):.3f}")
                logger.info(f"        - 平均F1: {np.mean(all_f1_scores_region):.3f} ± {np.std(all_f1_scores_region):.3f}")
        
        else:
            logger.info("        ⚠️ 分脑区样本不足，跳过LOSO分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # ========================================================================
        # 3. LOSO对比分析
        # ========================================================================
        logger.info("      📈 LOSO性能对比分析...")
        
        if (all_accuracies_global and all_accuracies_region and
            'overall_performance' in results['global_analysis'] and
            'overall_performance' in results['region_wise_analysis']):
            
            global_perf = results['global_analysis']['overall_performance']
            region_perf = results['region_wise_analysis']['overall_performance']
            
            acc_improvement = region_perf['mean_accuracy'] - global_perf['mean_accuracy']
            f1_improvement = region_perf['mean_f1_macro'] - global_perf['mean_f1_macro']
            
            # 计算泛化稳定性（标准差比较）
            acc_stability_change = global_perf['std_accuracy'] - region_perf['std_accuracy']
            f1_stability_change = global_perf['std_f1_macro'] - region_perf['std_f1_macro']
            
            results['comparative_analysis'] = {
                'accuracy_improvement': acc_improvement,
                'f1_improvement': f1_improvement,
                'accuracy_stability_improvement': acc_stability_change,
                'f1_stability_improvement': f1_stability_change,
                'generalization_assessment': self._assess_generalization_improvement(
                    acc_improvement, f1_improvement, acc_stability_change, f1_stability_change),
                'subject_embedding_value': self._assess_subject_embedding_loso_value(
                    global_perf, region_perf)
            }
            
            logger.info(f"        📊 LOSO性能对比:")
            logger.info(f"          准确率提升: {acc_improvement:+.3f} "
                f"(稳定性变化: {acc_stability_change:+.3f})")
            logger.info(f"          F1分数提升: {f1_improvement:+.3f} "
                f"(稳定性变化: {f1_stability_change:+.3f})")
            logger.info(f"          🎯 {results['comparative_analysis']['generalization_assessment']}")
        
        return results

    def _assess_generalization_improvement(self, acc_imp, f1_imp, acc_stab, f1_stab):
        """评估泛化性能改进"""
        if acc_imp > 0.03 and f1_imp > 0.03 and acc_stab > 0 and f1_stab > 0:
            return "优秀：分脑区方法在准确率、F1和稳定性上都有显著提升"
        elif acc_imp > 0.02 or f1_imp > 0.02:
            if acc_stab > 0 or f1_stab > 0:
                return "良好：分脑区方法在性能或稳定性上有所改善"
            else:
                return "中等：分脑区方法性能有提升但稳定性略有下降"
        elif acc_imp > -0.01 and f1_imp > -0.01:
            return "相当：两种方法的泛化性能相似"
        else:
            return "较差：全局方法的泛化性能可能更好"

    def _assess_subject_embedding_loso_value(self, global_perf, region_perf):
        """评估Subject Embedding在LOSO任务中的价值"""
        global_f1 = global_perf['mean_f1_macro']
        region_f1 = region_perf['mean_f1_macro']
        
        improvement = region_f1 - global_f1
        relative_improvement = improvement / global_f1 if global_f1 > 0 else 0
        
        if relative_improvement > 0.15:
            return {
                'recommendation': "强烈推荐Subject Embedding",
                'expected_improvement': f"预期LOSO性能提升{improvement:.3f}",
                'priority': "HIGH"
            }
        elif relative_improvement > 0.08:
            return {
                'recommendation': "建议使用Subject Embedding",
                'expected_improvement': f"预期LOSO性能提升{improvement:.3f}",
                'priority': "MEDIUM-HIGH"
            }
        elif relative_improvement > 0.02:
            return {
                'recommendation': "可以尝试Subject Embedding",
                'expected_improvement': f"预期LOSO性能提升{improvement:.3f}",
                'priority': "MEDIUM"
            }
        else:
            return {
                'recommendation': "Subject Embedding价值有限",
                'expected_improvement': f"预期提升仅{improvement:.3f}",
                'priority': "LOW"
            }


    def _assess_subject_embedding_value(self, baseline_results, loso_results):
        """评估Subject Embedding的潜在价值"""
        
        logger.info("    🔍 评估Subject Embedding潜在价值...")
        
        # 获取性能数据
        baseline_f1 = baseline_results.get('complex_rf', {}).get('f1_macro', 0.0)
        loso_f1 = loso_results.get('overall_performance', {}).get('mean_f1_macro', 0.0)
        
        if baseline_f1 == 0 or loso_f1 == 0:
            logger.info("    ⚠️ 性能数据不足，无法评估")
            return {}
        
        # 计算性能差距
        generalization_gap = baseline_f1 - loso_f1
        relative_gap = generalization_gap / baseline_f1 if baseline_f1 > 0 else 0
        
        # Subject Embedding价值评估
        if relative_gap > 0.15:  # 差距>15%
            embedding_recommendation = "强烈推荐Subject Embedding"
            expected_improvement = f"预期提升{generalization_gap*0.5:.3f}F1分数"
            priority = "HIGH"
        elif relative_gap > 0.08:  # 差距>8%
            embedding_recommendation = "建议使用Subject Embedding"  
            expected_improvement = f"预期提升{generalization_gap*0.3:.3f}F1分数"
            priority = "MEDIUM"
        else:
            embedding_recommendation = "Subject Embedding价值有限"
            expected_improvement = f"预期提升{generalization_gap*0.1:.3f}F1分数"
            priority = "LOW"
        
        assessment = {
            'baseline_f1': baseline_f1,
            'loso_f1': loso_f1,
            'generalization_gap': generalization_gap,
            'relative_gap_percent': relative_gap * 100,
            'embedding_recommendation': embedding_recommendation,
            'expected_improvement': expected_improvement,
            'priority': priority
        }
        
        logger.info(f"    📊 Subject Embedding价值评估:")
        logger.info(f"      - Baseline F1 (随机分割): {baseline_f1:.3f}")
        logger.info(f"      - LOSO F1 (受试者泛化): {loso_f1:.3f}")
        logger.info(f"      - 泛化差距: {generalization_gap:.3f} ({relative_gap*100:.1f}%)")
        logger.info(f"      - 推荐: {embedding_recommendation}")
        logger.info(f"      - {expected_improvement}")
        
        return assessment

                
    def _phase2c_class_consistency_analysis(self):
        """脑区分类一致性分析 (保持原有逻辑)"""
        
        logger.info("🔍 2.2C 脑区分类一致性分析...")
        
        # 转换one-hot标签到类别索引
        if len(self.data['y_train'].shape) > 1 and self.data['y_train'].shape[1] > 1:
            y_train_classes = np.argmax(self.data['y_train'], axis=1)
        else:
            y_train_classes = self.data['y_train'].flatten()
        
        unique_classes = np.unique(y_train_classes)
        n_classes = len(unique_classes)
        
        logger.info(f"  - 脑区类别数量: {n_classes}")
        logger.info(f"  - 分析前20个脑区的一致性...")
        
        # 计算每个脑区在不同受试者间的一致性
        class_consistency = {}
        
        subjects_train_int = self.data['subjects_train'].astype(int)
        
        for class_id in unique_classes[:20]:  # 只分析前20个类别
            class_mask = y_train_classes == class_id
            
            if np.sum(class_mask) > 100:  # 确保有足够样本
                class_data = self.data['X_train'][class_mask]
                class_subjects = subjects_train_int[class_mask]
                
                # 计算每个受试者该脑区的均值特征
                subject_means_for_class = []
                subjects_with_class = []
                
                for subject_id in np.unique(class_subjects):
                    subject_class_mask = class_subjects == subject_id
                    if np.sum(subject_class_mask) >= 10:  # 至少10个样本
                        subject_mean = np.mean(class_data[subject_class_mask], axis=0)
                        subject_means_for_class.append(subject_mean)
                        subjects_with_class.append(subject_id)
                
                if len(subject_means_for_class) >= 3:  # 至少3个受试者
                    subject_means_array = np.array(subject_means_for_class)
                    
                    # 计算受试者间的变异系数
                    feature_cv = np.std(subject_means_array, axis=0) / (np.abs(np.mean(subject_means_array, axis=0)) + 1e-8)
                    mean_cv = np.mean(feature_cv)
                    
                    # 计算受试者间距离
                    distances = pdist(subject_means_array)
                    mean_distance = np.mean(distances)
                    
                    class_consistency[class_id] = {
                        'mean_cv': mean_cv,
                        'mean_distance': mean_distance,
                        'n_subjects': len(subjects_with_class),
                        'n_samples': np.sum(class_mask),
                        'subjects_with_class': subjects_with_class
                    }
        
        self.analysis_results['class_consistency'] = class_consistency
        
        if class_consistency:
            avg_consistency = np.mean([info['mean_cv'] for info in class_consistency.values()])
            avg_distance = np.mean([info['mean_distance'] for info in class_consistency.values()])
            
            logger.info(f"✅ 脑区一致性分析完成")
            logger.info(f"  - 分析脑区数量: {len(class_consistency)}")
            logger.info(f"  - 平均变异系数: {avg_consistency:.3f} (越低越一致)")
            logger.info(f"  - 平均受试者间距离: {avg_distance:.3f}")
            
            # 找出最一致和最不一致的脑区
            if len(class_consistency) > 0:
                most_consistent_class = min(class_consistency.keys(), 
                                        key=lambda x: class_consistency[x]['mean_cv'])
                least_consistent_class = max(class_consistency.keys(), 
                                        key=lambda x: class_consistency[x]['mean_cv'])
                
                logger.info(f"  - 最一致脑区: {most_consistent_class} (CV: {class_consistency[most_consistent_class]['mean_cv']:.3f})")
                logger.info(f"  - 最不一致脑区: {least_consistent_class} (CV: {class_consistency[least_consistent_class]['mean_cv']:.3f})")
        else:
            logger.info("❌ 脑区一致性分析失败")


    def _compute_phase2_combined_scores(self):
        """计算Phase 2的综合决策得分（修正版）"""
        
        # 全局可分离性得分（保持原逻辑，但重新解释含义）
        global_separability = 0.0
        if 'global_subject_identification' in self.analysis_results:
            global_id = self.analysis_results['global_subject_identification']
            global_accuracy = global_id['accuracy']
            random_baseline = global_id['random_baseline']
            # 这里实际反映的是全局受试者差异强度
            global_separability = max(0, min(1, (global_accuracy - random_baseline) / (1 - random_baseline + 1e-8)))
        
        # 脑区一致性得分（保持原逻辑）
        consistency_score = 0.5
        if 'class_consistency' in self.analysis_results and self.analysis_results['class_consistency']:
            avg_consistency = np.mean([info['mean_cv'] for info in self.analysis_results['class_consistency'].values()])
            consistency_score = max(0, min(1, 1 / (1 + avg_consistency)))
        
        # 🔥 新增：基于正确分析的Subject Embedding需求评分
        embedding_necessity_score = 0.0
        high_necessity_ratio = 0.0
        
        if 'region_wise_separability' in self.analysis_results and self.analysis_results['region_wise_separability']:
            region_analysis = self.analysis_results['region_wise_separability']
            
            # 只分析成功的脑区
            successful_regions = [rid for rid, data in region_analysis.items() 
                                if 'embedding_necessity_score' in data]
            
            if successful_regions:
                necessity_scores = [region_analysis[rid]['embedding_necessity_score'] 
                                for rid in successful_regions]
                
                embedding_necessity_score = np.mean(necessity_scores)
                
                # 计算高需求脑区比例
                high_necessity_count = sum(1 for rid in successful_regions 
                                        if region_analysis[rid]['embedding_necessity_level'] in ['CRITICAL', 'HIGH'])
                high_necessity_ratio = high_necessity_count / len(successful_regions)
        
        # Phase 2 综合决策得分（更新含义）
        self.decision_scores['phase2'] = {
            # 重新解释的原有得分
            'global_subject_variability': global_separability,  # 全局受试者变异性
            'class_consistency': consistency_score,  # 脑区分类一致性
            'identification_accuracy': global_separability,  # 保持向后兼容
            'feature_competition_risk': 1.0 - consistency_score,  # 特征竞争风险
            
            # 🔥 新增：正确的Subject Embedding评估指标
            'embedding_necessity_average': embedding_necessity_score,  # 平均embedding需求强度
            'high_necessity_ratio': high_necessity_ratio,  # 高需求脑区比例
            'embedding_recommended': embedding_necessity_score > 0.4,  # 是否推荐使用embedding
            'analysis_coverage': len(self.analysis_results.get('region_wise_separability', {})) / 
                            len(self.analysis_results.get('region_wise_subject_analysis', {})) 
                            if self.analysis_results.get('region_wise_subject_analysis') else 0.0
        }
        
        logger.info(f"\n📈 Phase 2 综合决策指标 (修正版含义):")
        logger.info(f"  - 全局受试者变异性: {self.decision_scores['phase2']['global_subject_variability']:.3f}")
        logger.info(f"  - 脑区分类一致性: {self.decision_scores['phase2']['class_consistency']:.3f}")
        logger.info(f"  🔥 Subject Embedding平均需求强度: {self.decision_scores['phase2']['embedding_necessity_average']:.3f}")
        logger.info(f"  🔥 高需求脑区比例: {self.decision_scores['phase2']['high_necessity_ratio']:.3f}")
        logger.info(f"  🔥 推荐使用Embedding: {'是' if self.decision_scores['phase2']['embedding_recommended'] else '否'}")
        logger.info(f"  🔥 分析覆盖率: {self.decision_scores['phase2']['analysis_coverage']:.3f}")

    def phase3_embedding_adaptability_analysis(self):
        """
        Phase 3: Embedding适配性评估 - 保留原有分析 + 新增脑区感知建议
        """
        logger.info("\n" + "="*80)
        logger.info("📊 Phase 3: Embedding适配性评估 (增强版)")
        logger.info("="*80)
        
        # 🔄 保留原有的增强版降维分析
        logger.info("\n📊 Phase 3A: 增强版降维适配性分析 (保持原有)")
        self._phase3a_enhanced_dimensionality_analysis()
        
        # 🔥 新增：脑区感知的embedding设计分析
        logger.info("\n📊 Phase 3B: 脑区感知embedding设计分析 (新增)")
        self._phase3b_brain_aware_embedding_design()
        
        # 综合决策得分计算
        self._compute_phase3_combined_scores()

    def _phase3a_enhanced_dimensionality_analysis(self):
        """Phase 3A: 保留原有的增强版降维分析"""
        
        if 'subject_stats' not in self.analysis_results:
            logger.info("❌ 需要先运行Phase 1")
            return
        
        subject_means = self.analysis_results['subject_stats']['means']
        
        # 保留原有的多种降维方法对比分析
        logger.info("🔍 3.1A 多种降维方法对比分析...")
        dimensionality_results = self._comprehensive_dimensionality_analysis(subject_means)
        
        # 保留原有的高维空间直接分析
        logger.info("🔍 3.2A 高维空间直接分析...")
        high_dim_results = self._high_dimensional_direct_analysis(subject_means)
        
        # 保留原有的特征分组分析
        logger.info("🔍 3.3A 特征分组分析...")
        group_results = self._feature_group_analysis(subject_means)
        
        # 保留原有的样本量充足性评估
        logger.info("🔍 3.4A 样本量充足性评估...")
        sample_adequacy = self._sample_adequacy_assessment()
        
        # 保留原有的综合适配性评估
        logger.info("🔍 3.5A 综合适配性评估...")
        comprehensive_assessment = self._comprehensive_embedding_assessment(
            dimensionality_results, high_dim_results, group_results, sample_adequacy
        )
        
        # 存储原有结果
        self.analysis_results['enhanced_dimensionality'] = {
            'dimensionality_comparison': dimensionality_results,
            'high_dimensional_analysis': high_dim_results,
            'feature_group_analysis': group_results,
            'sample_adequacy': sample_adequacy,
            'comprehensive_assessment': comprehensive_assessment
        }
        
        logger.info(f"✅ 增强版降维分析完成")

    def _phase3b_brain_aware_embedding_design(self):
        """Phase 3B: 新增的脑区感知embedding设计分析"""
        
        logger.info("🔍 3.1B 脑区embedding需求分层分析...")
        
        if 'region_wise_subject_analysis' not in self.analysis_results:
            logger.info("❌ 需要先运行Phase 1B")
            return
        
        region_results = self.analysis_results['region_wise_subject_analysis']
        
        # 1. 脑区分层策略
        brain_aware_design = self._analyze_region_embedding_requirements(region_results)
        
        # 2. 脑区间相似性分析
        logger.info("🔍 3.2B 脑区间embedding相似性分析...")
        region_similarity_analysis = self._analyze_region_similarity_patterns(region_results)
        
        # 3. 脑区特异的embedding维度推荐
        logger.info("🔍 3.3B 脑区特异embedding维度推荐...")
        region_embedding_dims = self._recommend_region_specific_dimensions(region_results)
        
        # 4. 脑区聚类和共享策略
        logger.info("🔍 3.4B 脑区聚类和embedding共享策略...")
        region_clustering_strategy = self._design_region_clustering_strategy(region_results)
        
        # 存储脑区感知分析结果
        self.analysis_results['brain_aware_embedding_design'] = {
            'region_requirements': brain_aware_design,
            'region_similarity': region_similarity_analysis,
            'region_embedding_dims': region_embedding_dims,
            'region_clustering_strategy': region_clustering_strategy
        }
        
        logger.info(f"✅ 脑区感知embedding设计分析完成")

    def _analyze_region_embedding_requirements(self, region_results):
        """分析每个脑区的embedding需求"""
        
        # 按特异性分组脑区
        specificity_scores = {rid: data['subject_specificity_score'] 
                             for rid, data in region_results.items()}
        
        sorted_regions = sorted(specificity_scores.keys(), 
                               key=lambda x: specificity_scores[x], reverse=True)
        
        n_regions = len(sorted_regions)
        
        # 分层策略
        tier1_high = sorted_regions[:n_regions//4]        # Top 25%
        tier2_medium_high = sorted_regions[n_regions//4:n_regions//2]  # 25%-50%
        tier3_medium_low = sorted_regions[n_regions//2:3*n_regions//4]  # 50%-75%
        tier4_low = sorted_regions[3*n_regions//4:]       # Bottom 25%
        
        requirements = {
            'tier1_high_specificity': {
                'region_ids': tier1_high,
                'embedding_necessity': 'CRITICAL',
                'recommended_dim': 128,
                'adaptation_samples_needed': 1000,
                'priority': 1,
                'description': '极高特异性，强烈需要Subject Embedding'
            },
            'tier2_medium_high_specificity': {
                'region_ids': tier2_medium_high,
                'embedding_necessity': 'HIGH',
                'recommended_dim': 64,
                'adaptation_samples_needed': 500,
                'priority': 2,
                'description': '高特异性，建议使用Subject Embedding'
            },
            'tier3_medium_low_specificity': {
                'region_ids': tier3_medium_low,
                'embedding_necessity': 'MODERATE',
                'recommended_dim': 32,
                'adaptation_samples_needed': 200,
                'priority': 3,
                'description': '中等特异性，可选择性使用Subject Embedding'
            },
            'tier4_low_specificity': {
                'region_ids': tier4_low,
                'embedding_necessity': 'LOW',
                'recommended_dim': 16,
                'adaptation_samples_needed': 0,
                'priority': 4,
                'description': '低特异性，可能不需要Subject Embedding'
            }
        }
        
        logger.info(f"    - Tier 1 (极高特异性): {len(tier1_high)} 个脑区")
        logger.info(f"    - Tier 2 (高特异性): {len(tier2_medium_high)} 个脑区")
        logger.info(f"    - Tier 3 (中等特异性): {len(tier3_medium_low)} 个脑区")
        logger.info(f"    - Tier 4 (低特异性): {len(tier4_low)} 个脑区")
        
        return requirements

    def _analyze_region_similarity_patterns(self, region_results):
        """分析脑区间的特异性模式相似性"""
        
        # 构建脑区特征矩阵 (脑区 × 特异性特征)
        region_ids = list(region_results.keys())
        n_regions = len(region_ids)
        
        if n_regions < 2:
            return {}
        
        # 每个脑区的特异性特征向量
        region_feature_matrix = np.zeros((n_regions, 4))  # 4个特异性特征
        
        for i, region_id in enumerate(region_ids):
            data = region_results[region_id]
            region_feature_matrix[i] = [
                data['subject_specificity_score'],
                data['mean_inter_subject_distance'], 
                data['mean_inter_subject_correlation'],
                data['pca_3pc_variance']
            ]
        
        # 脑区间相似性矩阵
        try:
            region_distances = squareform(pdist(region_feature_matrix))
            region_correlations = np.corrcoef(region_feature_matrix)
        except:
            region_distances = np.zeros((n_regions, n_regions))
            region_correlations = np.eye(n_regions)
        
        # 层次聚类分析
        try:
            region_linkage = linkage(region_feature_matrix, method='ward')
        except:
            region_linkage = np.zeros((n_regions-1, 4))
        
        similarity_analysis = {
            'region_ids': region_ids,
            'similarity_matrix': region_correlations,
            'distance_matrix': region_distances,
            'linkage_matrix': region_linkage,
            'feature_matrix': region_feature_matrix
        }
        
        logger.info(f"    - 脑区间相似性分析完成: {n_regions} 个脑区")
        
        return similarity_analysis

    def _recommend_region_specific_dimensions(self, region_results):
        """为每个脑区推荐特异的embedding维度"""
        
        dimension_recommendations = {}
        
        for region_id, data in region_results.items():
            specificity_score = data['subject_specificity_score']
            n_subjects = data['n_subjects']
            pca_variance = data['pca_3pc_variance']
            
            # 基于特异性得分和PCA方差推荐维度
            if specificity_score > 2.0 and pca_variance < 0.8:
                # 高特异性，低线性度 -> 需要大维度
                recommended_dim = 128
                embedding_type = 'nonlinear'
            elif specificity_score > 1.5 and pca_variance < 0.9:
                # 中高特异性 -> 中等维度
                recommended_dim = 64
                embedding_type = 'mixed'
            elif specificity_score > 1.0:
                # 中等特异性 -> 小维度
                recommended_dim = 32
                embedding_type = 'linear'
            else:
                # 低特异性 -> 最小维度或不需要
                recommended_dim = 16
                embedding_type = 'minimal'
            
            # 考虑受试者数量限制
            max_reasonable_dim = min(recommended_dim, n_subjects * 2)
            
            dimension_recommendations[region_id] = {
                'recommended_dim': max_reasonable_dim,
                'embedding_type': embedding_type,
                'specificity_score': specificity_score,
                'n_subjects': n_subjects,
                'justification': f"基于特异性{specificity_score:.3f}和{n_subjects}个受试者"
            }
        
        logger.info(f"    - 维度推荐完成: {len(dimension_recommendations)} 个脑区")
        
        return dimension_recommendations

    def _design_region_clustering_strategy(self, region_results):
        """设计脑区聚类和embedding共享策略"""
        
        if len(region_results) < 3:
            return {}
        
        # 构建脑区特征用于聚类
        region_ids = list(region_results.keys())
        features = []
        
        for region_id in region_ids:
            data = region_results[region_id]
            features.append([
                data['subject_specificity_score'],
                data['mean_inter_subject_distance'],
                data['pca_3pc_variance']
            ])
        
        features = np.array(features)
        
        # K-means聚类，确定最优聚类数
        best_k = 3  # 默认值
        best_silhouette = -1
        
        for k in range(2, min(8, len(region_ids))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(features)
                silhouette = silhouette_score(features, cluster_labels)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
            except:
                continue
        
        # 使用最优K进行聚类
        try:
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
        except:
            cluster_labels = np.zeros(len(region_ids))
        
        # 组织聚类结果
        clusters = {}
        for i, region_id in enumerate(region_ids):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(region_id)
        
        # 分析每个聚类的特征
        cluster_analysis = {}
        for cluster_id, cluster_regions in clusters.items():
            cluster_specificities = [region_results[rid]['subject_specificity_score'] for rid in cluster_regions]
            cluster_analysis[cluster_id] = {
                'regions': cluster_regions,
                'n_regions': len(cluster_regions),
                'avg_specificity': np.mean(cluster_specificities),
                'specificity_std': np.std(cluster_specificities),
                'shared_embedding_recommended': len(cluster_regions) > 1 and np.std(cluster_specificities) < 0.5
            }
        
        clustering_strategy = {
            'n_clusters': best_k,
            'silhouette_score': best_silhouette,
            'clusters': cluster_analysis,
            'cluster_labels': dict(zip(region_ids, cluster_labels)),
            'sharing_strategy': 'cluster_based' if best_k < len(region_ids) else 'individual'
        }
        
        logger.info(f"    - 脑区聚类完成: {best_k} 个聚类，轮廓系数 {best_silhouette:.3f}")
        
        return clustering_strategy

    def _compute_phase3_combined_scores(self):
        """计算Phase 3的综合决策得分"""
        
        # 原有得分
        linearity_score = 0.5
        clustering_quality = 0.5
        sample_adequacy = 0.5
        embedding_feasibility = 0.5
        intrinsic_dim_score = 0.5
        high_dim_score = 0.5
        
        if 'enhanced_dimensionality' in self.analysis_results:
            enhanced_results = self.analysis_results['enhanced_dimensionality']
            
            if 'comprehensive_assessment' in enhanced_results:
                assessment = enhanced_results['comprehensive_assessment']
                linearity_score = assessment.get('linearity_score', 0.5)
                clustering_quality = assessment.get('clustering_score', 0.5)
                embedding_feasibility = assessment.get('overall_feasibility', 0.5)
                intrinsic_dim_score = assessment.get('intrinsic_dim_score', 0.5)
                high_dim_score = assessment.get('high_dim_score', 0.5)
            
            if 'sample_adequacy' in enhanced_results:
                sample_adequacy = enhanced_results['sample_adequacy'].get('adequacy_score', 0.5)
        
        # 🔥 新增：脑区感知得分
        brain_aware_feasibility = 0.0
        region_diversity_score = 0.0
        clustering_effectiveness = 0.0
        
        if 'brain_aware_embedding_design' in self.analysis_results:
            brain_design = self.analysis_results['brain_aware_embedding_design']
            
            # 脑区embedding可行性：基于成功分析的脑区比例
            if 'region_requirements' in brain_design:
                requirements = brain_design['region_requirements']
                total_regions = sum(len(tier['region_ids']) for tier in requirements.values())
                high_priority_regions = len(requirements.get('tier1_high_specificity', {}).get('region_ids', []))
                
                if total_regions > 0:
                    brain_aware_feasibility = min(1.0, total_regions / 50)  # 假设50个脑区是理想数量
                    region_diversity_score = high_priority_regions / total_regions if total_regions > 0 else 0
            
            # 聚类效果：基于轮廓系数
            if 'region_clustering_strategy' in brain_design:
                clustering_strategy = brain_design['region_clustering_strategy']
                silhouette = clustering_strategy.get('silhouette_score', 0)
                clustering_effectiveness = max(0, min(1, (silhouette + 1) / 2))  # 将[-1,1]映射到[0,1]
        
        # Phase 3 综合决策得分
        self.decision_scores['phase3'] = {
            # 原有得分
            'linearity': linearity_score,
            'clustering_quality': clustering_quality,
            'sample_adequacy': sample_adequacy,
            'embedding_feasibility': embedding_feasibility,
            'intrinsic_dimensionality': intrinsic_dim_score,
            'high_dim_performance': high_dim_score,
            
            # 🔥 新增得分
            'brain_aware_feasibility': brain_aware_feasibility,
            'region_diversity': region_diversity_score,
            'region_clustering_effectiveness': clustering_effectiveness
        }
        
        logger.info(f"\n📈 Phase 3 综合决策指标:")
        logger.info(f"  - 线性度: {self.decision_scores['phase3']['linearity']:.3f}")
        logger.info(f"  - 聚类质量: {self.decision_scores['phase3']['clustering_quality']:.3f}")
        logger.info(f"  - 样本充足性: {self.decision_scores['phase3']['sample_adequacy']:.3f}")
        logger.info(f"  - 内在维度得分: {self.decision_scores['phase3']['intrinsic_dimensionality']:.3f}")
        logger.info(f"  - 高维性能得分: {self.decision_scores['phase3']['high_dim_performance']:.3f}")
        logger.info(f"  🔥 脑区感知可行性: {self.decision_scores['phase3']['brain_aware_feasibility']:.3f}")
        logger.info(f"  🔥 脑区多样性得分: {self.decision_scores['phase3']['region_diversity']:.3f}")
        logger.info(f"  🔥 脑区聚类效果: {self.decision_scores['phase3']['region_clustering_effectiveness']:.3f}")
        logger.info(f"  - 整体可行性: {self.decision_scores['phase3']['embedding_feasibility']:.3f}")

    def _comprehensive_dimensionality_analysis(self, subject_means):
        """多种降维方法对比分析 (保持原有逻辑)"""
        results = {}
        
        # 1. PCA (线性)
        pca = PCA()
        pca_result = pca.fit_transform(subject_means)
        pca_2d = PCA(n_components=2).fit_transform(subject_means)
        
        results['pca'] = {
            'full_result': pca_result,
            '2d_result': pca_2d,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'method_type': 'linear'
        }
        
        # 2. t-SNE (非线性流形)
        try:
            perplexity = min(30, len(subject_means)-1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                    learning_rate='auto', init='random')
            tsne_result = tsne.fit_transform(subject_means)
            
            results['tsne'] = {
                '2d_result': tsne_result,
                'perplexity': perplexity,
                'method_type': 'nonlinear_manifold'
            }
            logger.info(f"    ✅ t-SNE分析完成 (perplexity={perplexity})")
        except Exception as e:
            logger.info(f"    ⚠️ t-SNE分析失败: {e}")
            results['tsne'] = None
        
        # 3. 比较不同方法的结构保持能力
        structure_preservation = self._compare_structure_preservation(subject_means, results)
        results['structure_preservation'] = structure_preservation
        
        return results

    def _high_dimensional_direct_analysis(self, subject_means):
        """高维空间直接分析 (保持原有逻辑)"""
        results = {}
        
        # 1. 内在维度估计
        logger.info("    🔍 估计内在维度...")
        intrinsic_dim = self._estimate_intrinsic_dimensionality(subject_means)
        results['intrinsic_dimensionality'] = intrinsic_dim
        
        # 2. 高维分类器性能测试
        logger.info("    🔍 高维分类器性能测试...")
        separability_scores = self._test_high_dim_classifiers(subject_means)
        results['separability_scores'] = separability_scores
        
        logger.info(f"    ✅ 高维分析完成")
        logger.info(f"      - 估计内在维度: {intrinsic_dim:.1f}")
        logger.info(f"      - 测试分类器: {len(separability_scores)}")
        
        return results

    def _feature_group_analysis(self, subject_means):
        """特征分组分析 (保持原有逻辑)"""
        feature_groups = {
            'qti_params': list(range(0, 15)),           # QTI参数 (15个)
            'raw_b_tensors': list(range(15, 225)),      # 原始b-tensor值 (210个)
            'cest_params': list(range(225, 229)),       # CEST参数 (4个)
            'z_spectrum': list(range(229, 341))         # Z-spectrum值 (112个)
        }
        
        group_analysis = {}
        
        for group_name, feature_indices in feature_groups.items():
            if len(feature_indices) == 0:
                continue
                
            logger.info(f"    🔍 分析 {group_name} ({len(feature_indices)} 特征)...")
            
            try:
                group_data = subject_means[:, feature_indices]
                
                if len(feature_indices) >= 2:
                    # PCA分析
                    pca = PCA()
                    pca_result = pca.fit_transform(group_data)
                    
                    group_analysis[group_name] = {
                        'pca_explained_variance': pca.explained_variance_ratio_,
                        'variance_in_3pc': np.sum(pca.explained_variance_ratio_[:3]) if len(pca.explained_variance_ratio_) >= 3 else np.sum(pca.explained_variance_ratio_),
                        'feature_count': len(feature_indices),
                        'feature_range': (min(feature_indices), max(feature_indices))
                    }
                    
                    logger.info(f"      ✅ {group_name}: 前3PC解释方差 {group_analysis[group_name]['variance_in_3pc']:.3f}")
                    
            except Exception as e:
                logger.info(f"      ⚠️ {group_name} 分析失败: {e}")
                group_analysis[group_name] = {'error': str(e)}
        
        return group_analysis

    def _sample_adequacy_assessment(self):
        """样本量充足性评估 (保持原有逻辑)"""
        sample_counts = self.analysis_results['subject_stats']['sample_counts']
        total_samples = np.sum(sample_counts)
        n_subjects = len(sample_counts)
        avg_samples_per_subject = np.mean(sample_counts)
        
        sample_adequacy_score = 0
        
        if avg_samples_per_subject >= 50000:
            sample_adequacy_score += 0.4
        elif avg_samples_per_subject >= 10000:
            sample_adequacy_score += 0.2
        
        if n_subjects >= 30:
            sample_adequacy_score += 0.3
        elif n_subjects >= 20:
            sample_adequacy_score += 0.2
        
        if total_samples >= 1000000:
            sample_adequacy_score += 0.3
        elif total_samples >= 500000:
            sample_adequacy_score += 0.2
        
        return {
            'total_samples': total_samples,
            'n_subjects': n_subjects,
            'avg_samples_per_subject': avg_samples_per_subject,
            'adequacy_score': sample_adequacy_score
        }

    def _comprehensive_embedding_assessment(self, dimensionality_results, high_dim_results, group_results, sample_adequacy):
        """综合embedding适配性评估 (保持原有逻辑)"""
        assessment = {
            'linearity_score': 0.0,
            'clustering_score': 0.0,
            'intrinsic_dim_score': 0.0,
            'high_dim_score': 0.0,
            'overall_feasibility': 0.0
        }
        
        # 线性度评估
        if 'pca' in dimensionality_results:
            pca_3pc_variance = np.sum(dimensionality_results['pca']['explained_variance'][:3])
            assessment['linearity_score'] = min(1.0, pca_3pc_variance * 1.25)
        
        # 内在维度评估
        if 'intrinsic_dimensionality' in high_dim_results:
            intrinsic_dim = high_dim_results['intrinsic_dimensionality']
            if intrinsic_dim <= 20:
                assessment['intrinsic_dim_score'] = 1.0
            elif intrinsic_dim <= 50:
                assessment['intrinsic_dim_score'] = 0.8
            elif intrinsic_dim <= 100:
                assessment['intrinsic_dim_score'] = 0.6
            else:
                assessment['intrinsic_dim_score'] = 0.3
        
        # 高维性能评估
        if 'separability_scores' in high_dim_results:
            high_dim_scores = []
            for clf_name, clf_results in high_dim_results['separability_scores'].items():
                high_dim_scores.append(clf_results['mean_score'])
            
            if high_dim_scores:
                avg_high_dim_performance = np.mean(high_dim_scores)
                n_subjects = len(self.data['available_subjects'])
                random_baseline = 1.0 / n_subjects
                normalized_score = (avg_high_dim_performance - random_baseline) / (1 - random_baseline)
                assessment['high_dim_score'] = max(0, min(1, normalized_score))
        
        # 综合可行性评估
        weights = {
            'linearity': 0.3,
            'intrinsic_dim': 0.3,
            'high_dim': 0.2,
            'sample_adequacy': 0.2
        }
        
        assessment['overall_feasibility'] = (
            assessment['linearity_score'] * weights['linearity'] +
            assessment['intrinsic_dim_score'] * weights['intrinsic_dim'] +
            assessment['high_dim_score'] * weights['high_dim'] +
            sample_adequacy['adequacy_score'] * weights['sample_adequacy']
        )
        
        return assessment

    def _estimate_intrinsic_dimensionality(self, data):
        """估计数据的内在维度 (保持原有逻辑)"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            k = min(10, len(data) - 1)
            if k < 2:
                return data.shape[1]
                
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
            distances, indices = nbrs.kneighbors(data)
            
            distances = distances[:, 1:]  # 排除自身距离
            ratios = distances[:, -1] / (distances[:, 0] + 1e-10)
            ratios = ratios[ratios > 1]
            ratios = ratios[ratios < 1000]
            
            if len(ratios) > 0:
                log_ratios = np.log(ratios)
                intrinsic_dim = k / np.mean(log_ratios)
                return min(max(1, intrinsic_dim), data.shape[1])
            else:
                return data.shape[1]
                
        except Exception as e:
            logger.info(f"      ⚠️ 内在维度估计失败: {e}")
            return data.shape[1]


    def _test_high_dim_classifiers(self, subject_means):
        """测试多种高维分类器的性能 (增强版：全局 + 分脑区分析)"""
        
        logger.info("    🔍 增强版高维分类器性能测试...")
        logger.info("    📊 分析维度：全局受试者识别 + 分脑区受试者识别")
        
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1)
        }
        
        results = {
            'global_analysis': {},
            'region_wise_analysis': {},
            'comparative_analysis': {}
        }
        
        # ========================================================================
        # 1. 保留原有全局分析
        # ========================================================================
        logger.info("      🌐 全局受试者识别分析...")
        subject_labels = np.arange(len(subject_means))
        
        for clf_name, clf in classifiers.items():
            try:
                n_subjects = len(subject_means)
                if n_subjects >= 5:
                    cv_folds = min(5, n_subjects)
                    
                    if cv_folds < 3:
                        # 样本太少，使用train_test_split
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            subject_means, subject_labels, test_size=0.3, random_state=42
                        )
                        clf.fit(X_train, y_train)
                        score = clf.score(X_test, y_test)
                        results['global_analysis'][clf_name] = {
                            'mean_score': score,
                            'std_score': 0.0,
                            'scores': [score],
                            'method': 'train_test_split'
                        }
                    else:
                        # 使用KFold交叉验证
                        from sklearn.model_selection import KFold
                        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        scores = cross_val_score(clf, subject_means, subject_labels, 
                                            cv=cv_strategy, scoring='accuracy')
                        results['global_analysis'][clf_name] = {
                            'mean_score': np.mean(scores),
                            'std_score': np.std(scores),
                            'scores': scores,
                            'method': 'cross_validation'
                        }
                        
                logger.info(f"        ✅ 全局{clf_name}: {results['global_analysis'][clf_name]['mean_score']:.3f}")
                
            except Exception as e:
                logger.info(f"        ❌ 全局{clf_name} 失败: {e}")
                results['global_analysis'][clf_name] = {'error': str(e)}
        
        # ========================================================================
        # 2. 新增分脑区分析
        # ========================================================================
        logger.info("      🧠 分脑区受试者识别分析...")
        
        region_dataset = self._build_region_aware_dataset()
        
        if len(region_dataset['features']) > 100:  # 确保有足够样本
            
            X_region = region_dataset['features']
            subject_labels_region = region_dataset['subject_labels']
            region_labels_region = region_dataset['region_labels']
            
            # 重新映射受试者标签为连续编号
            unique_subjects = np.unique(subject_labels_region)
            subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
            mapped_subject_labels = np.array([subject_mapping[s] for s in subject_labels_region])
            
            logger.info(f"        📊 分脑区数据集: {len(X_region)} 样本, {len(unique_subjects)} 受试者, {len(np.unique(region_labels_region))} 脑区")
            
            for clf_name, clf in classifiers.items():
                try:
                    # 使用StratifiedKFold，现在可以正常工作了
                    from sklearn.model_selection import StratifiedKFold
                    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(clf, X_region, mapped_subject_labels, 
                                        cv=cv_strategy, scoring='accuracy')
                    
                    results['region_wise_analysis'][clf_name] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'scores': scores,
                        'n_samples': len(X_region),
                        'n_subjects': len(unique_subjects),
                        'n_regions': len(np.unique(region_labels_region)),
                        'method': 'stratified_cv'
                    }
                    
                    logger.info(f"        ✅ 分脑区{clf_name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                    
                except Exception as e:
                    logger.info(f"        ❌ 分脑区{clf_name} 失败: {e}")
                    results['region_wise_analysis'][clf_name] = {'error': str(e)}
        
        else:
            logger.info("        ⚠️ 分脑区样本不足，跳过分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # ========================================================================
        # 3. 对比分析
        # ========================================================================
        logger.info("      📈 全局 vs 分脑区性能对比...")
        
        for clf_name in classifiers.keys():
            if (clf_name in results['global_analysis'] and 
                clf_name in results['region_wise_analysis'] and
                'mean_score' in results['global_analysis'][clf_name] and
                'mean_score' in results['region_wise_analysis'][clf_name]):
                
                global_score = results['global_analysis'][clf_name]['mean_score']
                region_score = results['region_wise_analysis'][clf_name]['mean_score']
                
                improvement = region_score - global_score
                relative_improvement = improvement / global_score if global_score > 0 else 0
                
                results['comparative_analysis'][clf_name] = {
                    'global_score': global_score,
                    'region_wise_score': region_score,
                    'absolute_improvement': improvement,
                    'relative_improvement_percent': relative_improvement * 100,
                    'conclusion': self._interpret_improvement(improvement, relative_improvement)
                }
                
                logger.info(f"        📊 {clf_name}对比: 全局{global_score:.3f} → 分脑区{region_score:.3f} "
                    f"(提升{improvement:+.3f}, {relative_improvement*100:+.1f}%)")
        
        return results

    def _build_region_aware_dataset(self):
        """构建脑区感知数据集：每个样本 = 一个受试者在一个脑区的平均特征"""
        
        logger.info("        🔧 构建分脑区数据集...")
        
        # 获取脑区标签
        if len(self.data['y_train'].shape) > 1 and self.data['y_train'].shape[1] > 1:
            y_classes = np.argmax(self.data['y_train'], axis=1)
        else:
            y_classes = self.data['y_train'].flatten()
        
        X_list = []
        subject_labels = []
        region_labels = []
        sample_counts = []
        
        available_subjects = self.data['available_subjects']
        unique_regions = np.unique(y_classes)
        
        valid_combinations = 0
        total_combinations = len(available_subjects) * len(unique_regions)
        
        for subject_id in available_subjects:
            for region_id in unique_regions:
                # 找到该受试者在该脑区的所有体素
                mask = (self.data['subjects_train'] == subject_id) & (y_classes == region_id)
                n_voxels = np.sum(mask)
                
                if n_voxels >= 50:  # 最小体素数阈值
                    # 计算该受试者在该脑区的平均特征
                    region_features = np.mean(self.data['X_train'][mask], axis=0)
                    
                    X_list.append(region_features)
                    subject_labels.append(subject_id)
                    region_labels.append(region_id)
                    sample_counts.append(n_voxels)
                    valid_combinations += 1
        
        logger.info(f"        ✅ 有效组合: {valid_combinations}/{total_combinations} "
            f"({valid_combinations/total_combinations*100:.1f}%)")
        logger.info(f"        📊 平均每组合体素数: {np.mean(sample_counts):.0f}")
        
        return {
            'features': np.array(X_list),
            'subject_labels': np.array(subject_labels),
            'region_labels': np.array(region_labels),
            'sample_counts': np.array(sample_counts)
        }

    def _interpret_improvement(self, improvement, relative_improvement):
        """解释性能改进的意义"""
        if relative_improvement > 0.1:  # 10%以上提升
            return "显著提升：分脑区分析明显优于全局分析"
        elif relative_improvement > 0.05:  # 5-10%提升
            return "中等提升：分脑区分析有一定优势"
        elif relative_improvement > 0.01:  # 1-5%提升
            return "轻微提升：分脑区分析略有优势"
        elif relative_improvement > -0.01:  # ±1%以内
            return "性能相当：两种方法差异不大"
        else:  # 下降
            return "性能下降：全局分析可能更适合"
        

    def _compare_structure_preservation(self, original_data, dimensionality_results):
        """比较不同降维方法的结构保持能力 (保持原有逻辑)"""
        original_distances = squareform(pdist(original_data))
        preservation_scores = {}
        
        for method_name, method_data in dimensionality_results.items():
            if method_data is None or method_name == 'structure_preservation':
                continue
                
            try:
                if '2d_result' in method_data:
                    reduced_data = method_data['2d_result']
                elif method_name == 'pca' and 'full_result' in method_data:
                    reduced_data = method_data['full_result'][:, :2]
                else:
                    continue
                
                reduced_distances = squareform(pdist(reduced_data))
                
                from scipy.stats import spearmanr
                correlation, p_value = spearmanr(
                    original_distances.flatten(), 
                    reduced_distances.flatten()
                )
                
                preservation_scores[method_name] = {
                    'spearman_correlation': correlation,
                    'p_value': p_value
                }
                
            except Exception as e:
                logger.info(f"      ⚠️ {method_name} 结构保持分析失败: {e}")
        
        return preservation_scores

    def generate_visualizations(self):
        """生成可视化图表 - 保留原有可视化 + 新增脑区特异性可视化"""
        logger.info("\n" + "="*80)
        logger.info("📊 生成增强版可视化图表")
        logger.info("="*80)
        
        try:
            # 🔄 生成原有可视化 (简化版，保留主要图表)
            logger.info("\n📊 生成原有分析可视化...")
            self._generate_original_visualizations()
            
            # 🔥 生成脑区特异性可视化
            logger.info("\n📊 生成脑区特异性分析图表...")
            self._generate_brain_region_specificity_visualizations()
            
            # 🔥 生成脑区embedding设计可视化
            logger.info("\n📊 生成脑区embedding设计图表...")
            self._generate_brain_embedding_design_visualizations()
            
            logger.info(f"✅ 增强版可视化图表生成完成")
            
        except Exception as e:
            logger.error(f"可视化生成过程中出现错误: {e}")
            logger.exception("详细错误信息:")

    def _generate_original_visualizations(self):
        """生成原有的可视化图表 (简化版)"""
        
        if 'subject_stats' not in self.analysis_results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 受试者相似性热图
        ax = axes[0, 0]
        if 'global_subject_similarity' in self.analysis_results:
            correlation_matrix = self.analysis_results['global_subject_similarity']['correlation_matrix']
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('全局受试者间相关性矩阵', fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 2. PCA累积方差解释
        ax = axes[0, 1]
        if 'global_feature_variation' in self.analysis_results:
            cumulative_variance = self.analysis_results['global_feature_variation']['pca_cumulative_variance']
            ax.plot(range(1, min(21, len(cumulative_variance)+1)), 
                   cumulative_variance[:20], 'o-', linewidth=2)
            ax.axhline(0.8, color='red', linestyle='--', label='80%解释阈值')
            ax.set_title('PCA累积方差解释', fontweight='bold')
            ax.set_xlabel('主成分数量')
            ax.set_ylabel('累积方差解释比例')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. 全局受试者识别准确率
        ax = axes[0, 2]
        if 'global_subject_identification' in self.analysis_results:
            global_id = self.analysis_results['global_subject_identification']
            accuracy = global_id['accuracy']
            baseline = global_id['random_baseline']
            
            ax.bar(['随机基线', '最佳分类器'], [baseline, accuracy], 
                  color=['gray', 'lightcoral'])
            ax.set_title('全局受试者识别准确率', fontweight='bold')
            ax.set_ylabel('准确率')
            
            ax.text(0, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            ax.text(1, accuracy + 0.01, f'{accuracy:.3f}', ha='center', va='bottom')
        
        # 4. 特征变异分布
        ax = axes[1, 0]
        if 'global_feature_variation' in self.analysis_results:
            f_stats = self.analysis_results['global_feature_variation']['f_stats']
            ax.hist(f_stats, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.percentile(f_stats, 90), color='red', linestyle='--', 
                      label=f'90th: {np.percentile(f_stats, 90):.2f}')
            ax.set_title('特征受试者间变异分布', fontweight='bold')
            ax.set_xlabel('标准化F统计量')
            ax.set_ylabel('特征数量')
            ax.legend()
        
        # 5. 综合决策雷达图
        ax = axes[1, 1]
        categories = ['差异显著性', '模式线性度', '受试者相似性', '可分离性', '样本充足性']
        
        scores = [
            self.decision_scores.get('phase1', {}).get('difference_significance', 0),
            self.decision_scores.get('phase1', {}).get('pattern_linearity', 0),
            self.decision_scores.get('phase1', {}).get('subject_similarity', 0),
            self.decision_scores.get('phase2', {}).get('subject_separability', 0),
            self.decision_scores.get('phase3', {}).get('sample_adequacy', 0)
        ]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores_plot = scores + [scores[0]]
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='blue')
        ax.fill(angles_plot, scores_plot, alpha=0.25, color='blue')
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Subject Embedding 可行性雷达图', fontweight='bold')
        ax.grid(True)
        
        # 6. 降维对比 (如果有)
        ax = axes[1, 2]
        if 'enhanced_dimensionality' in self.analysis_results:
            dim_results = self.analysis_results['enhanced_dimensionality']
            if 'dimensionality_comparison' in dim_results and 'pca' in dim_results['dimensionality_comparison']:
                pca_data = dim_results['dimensionality_comparison']['pca']
                pca_2d = pca_data['2d_result']
                scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                                   c=range(len(pca_2d)), cmap='viridis', s=50, alpha=0.7)
                ax.set_title('PCA - 受试者2D分布', fontweight='bold')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                plt.colorbar(scatter, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        original_viz_path = self.save_path / 'visualizations' / 'original_analysis.png'
        plt.savefig(original_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 重要：关闭图形避免内存泄漏
            
        logger.info(f"  ✅ 原有分析图表已保存: {original_viz_path}")
        
    def _generate_brain_region_specificity_visualizations(self):
        """生成脑区Subject Embedding需求分析图表（修正版）"""
        
        if 'region_wise_separability' not in self.analysis_results:
            logger.info("  ⚠️ 没有脑区Subject Embedding分析结果，跳过可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        region_analysis = self.analysis_results['region_wise_separability']
        
        # 过滤出成功分析的脑区
        successful_regions = {rid: data for rid, data in region_analysis.items() 
                            if 'embedding_necessity_score' in data}
        
        if len(successful_regions) == 0:
            logger.info("  ⚠️ 没有成功的Subject Embedding分析结果")
            return
        
        region_ids = list(successful_regions.keys())
        
        # 1. Subject Embedding需求得分分布
        ax = axes[0, 0]
        necessity_scores = [successful_regions[rid]['embedding_necessity_score'] for rid in region_ids]
        
        ax.hist(necessity_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(np.mean(necessity_scores), color='red', linestyle='--',
                label=f'平均: {np.mean(necessity_scores):.3f}')
        ax.axvline(0.5, color='orange', linestyle='--', label='推荐阈值: 0.5')
        ax.set_title('脑区Subject Embedding需求得分分布', fontweight='bold', fontsize=14)
        ax.set_xlabel('需求得分')
        ax.set_ylabel('脑区数量')
        ax.legend()
        
        # 2. Top 20 高需求脑区
        ax = axes[0, 1]
        sorted_regions = sorted(region_ids, key=lambda x: successful_regions[x]['embedding_necessity_score'], reverse=True)
        top_20_regions = sorted_regions[:20]
        top_20_scores = [successful_regions[rid]['embedding_necessity_score'] for rid in top_20_regions]
        
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(top_20_regions)))
        bars = ax.bar(range(len(top_20_regions)), top_20_scores, color=colors, alpha=0.8)
        ax.set_title('Top 20 最需要Subject Embedding的脑区', fontweight='bold', fontsize=14)
        ax.set_xlabel('脑区排名')
        ax.set_ylabel('需求得分')
        ax.set_xticks(range(0, len(top_20_regions), max(1, len(top_20_regions)//5)))
        
        # 3. 需求等级分布饼图
        ax = axes[0, 2]
        necessity_levels = [successful_regions[rid]['embedding_necessity_level'] for rid in region_ids]
        level_counts = {}
        for level in necessity_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            labels = list(level_counts.keys())
            sizes = list(level_counts.values())
            colors_pie = ['red', 'orange', 'yellow', 'lightblue'][:len(labels)]
            
            ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax.set_title('Subject Embedding需求等级分布', fontweight='bold', fontsize=14)
        
        # 4. 一致性 vs 泛化性能散点图
        ax = axes[1, 0]
        consistency_scores = []
        generalization_scores = []
        
        for rid in region_ids:
            data = successful_regions[rid]
            if 'component_scores' in data:
                consistency_scores.append(data['component_scores']['consistency'])
                generalization_scores.append(data['component_scores']['generalization'])
        
        if consistency_scores and generalization_scores:
            scatter = ax.scatter(consistency_scores, generalization_scores, 
                            c=necessity_scores, cmap='Reds', alpha=0.7, s=60)
            ax.set_xlabel('一致性问题得分')
            ax.set_ylabel('泛化问题得分')
            ax.set_title('脑区问题类型分析', fontweight='bold', fontsize=14)
            plt.colorbar(scatter, ax=ax, label='总需求得分')
        
        # 5. 推荐embedding维度分布
        ax = axes[1, 1]
        recommended_dims = [successful_regions[rid]['recommended_embedding_dim'] for rid in region_ids]
        dim_counts = {}
        for dim in recommended_dims:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        if dim_counts:
            dims = list(dim_counts.keys())
            counts = list(dim_counts.values())
            
            colors_bar = plt.cm.viridis(np.linspace(0, 1, len(dims)))
            ax.bar(dims, counts, color=colors_bar, alpha=0.7)
            ax.set_title('推荐Embedding维度分布', fontweight='bold', fontsize=14)
            ax.set_xlabel('推荐维度')
            ax.set_ylabel('脑区数量')
        
        # 6. 实施优先级分布
        ax = axes[1, 2]
        priorities = [successful_regions[rid]['implementation_priority'] for rid in region_ids]
        priority_counts = {}
        for priority in priorities:
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        if priority_counts:
            priority_labels = list(priority_counts.keys())
            priority_sizes = list(priority_counts.values())
            
            x = range(len(priority_labels))
            colors_priority = ['red', 'orange', 'yellow', 'lightgreen'][:len(priority_labels)]
            
            bars = ax.bar(x, priority_sizes, color=colors_priority, alpha=0.7)
            ax.set_title('实施优先级分布', fontweight='bold', fontsize=14)
            ax.set_xlabel('优先级')
            ax.set_ylabel('脑区数量')
            ax.set_xticks(x)
            ax.set_xticklabels(priority_labels, rotation=45)
            
            # 添加数值标签
            for bar, size in zip(bars, priority_sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        brain_viz_path = self.save_path / 'visualizations' / 'subject_embedding_necessity_analysis.png'
        plt.savefig(brain_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 添加plt.close()

    def _generate_brain_embedding_design_visualizations(self):
        """生成脑区embedding设计图表（修复版）"""
        
        if 'brain_aware_embedding_design' not in self.analysis_results:
            logger.info("  ⚠️ 没有脑区embedding设计结果，跳过相关可视化")
            return
        
        brain_design = self.analysis_results['brain_aware_embedding_design']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 脑区分层需求柱状图
        ax = axes[0, 0]
        if 'region_requirements' in brain_design:
            requirements = brain_design['region_requirements']
            
            tier_names = []
            tier_counts = []
            tier_dims = []
            
            for tier_name, tier_data in requirements.items():
                if tier_data['region_ids']:  # 只包含非空的层次
                    tier_names.append(tier_name.replace('_', ' ').title())
                    tier_counts.append(len(tier_data['region_ids']))
                    tier_dims.append(tier_data['recommended_dim'])
            
            if tier_names:
                # 修复：使用matplotlib.cm而不是错误的plt.cm.viridis()调用
                import matplotlib.cm as cm
                colors_bar = [cm.viridis(i/len(tier_names)) for i in range(len(tier_names))]
                
                bars = ax.bar(tier_names, tier_counts, color=colors_bar, alpha=0.7)
                ax.set_title('脑区分层需求分布', fontweight='bold', fontsize=14)
                ax.set_xlabel('需求层次')
                ax.set_ylabel('脑区数量')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加推荐维度标签
                for bar, dim in zip(bars, tier_dims):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{dim}D', ha='center', va='bottom', fontweight='bold')
        
        # 2. 脑区embedding维度推荐分布
        ax = axes[0, 1]
        if 'region_embedding_dims' in brain_design:
            region_dims = brain_design['region_embedding_dims']
            dims = [data['recommended_dim'] for data in region_dims.values()]
            
            ax.hist(dims, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(dims), color='red', linestyle='--', 
                    label=f'平均: {np.mean(dims):.1f}D')
            ax.set_title('脑区embedding维度分布', fontweight='bold', fontsize=14)
            ax.set_xlabel('推荐embedding维度')
            ax.set_ylabel('脑区数量')
            ax.legend()
        
        # 3. 脑区聚类结果
        ax = axes[1, 0]
        if 'region_clustering_strategy' in brain_design:
            clustering = brain_design['region_clustering_strategy']
            
            if 'clusters' in clustering:
                cluster_ids = list(clustering['clusters'].keys())
                cluster_sizes = [clustering['clusters'][cid]['n_regions'] for cid in cluster_ids]
                cluster_specificities = [clustering['clusters'][cid]['avg_specificity'] for cid in cluster_ids]
                
                bars = ax.bar(cluster_ids, cluster_sizes, color='lightgreen', alpha=0.7)
                ax.set_title(f'脑区聚类结果 ({len(cluster_ids)} 个聚类)', fontweight='bold', fontsize=14)
                ax.set_xlabel('聚类ID')
                ax.set_ylabel('脑区数量')
                
                # 添加平均特异性标签
                for bar, spec in zip(bars, cluster_specificities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{spec:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 4. embedding策略总结饼图
        ax = axes[1, 1]
        if 'region_requirements' in brain_design:
            requirements = brain_design['region_requirements']
            
            necessity_counts = {}
            for tier_data in requirements.values():
                necessity = tier_data['embedding_necessity']
                necessity_counts[necessity] = necessity_counts.get(necessity, 0) + len(tier_data['region_ids'])
            
            if necessity_counts:
                labels = list(necessity_counts.keys())
                sizes = list(necessity_counts.values())
                colors = ['red', 'orange', 'yellow', 'lightblue'][:len(labels)]
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('脑区embedding需求分布', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        embedding_viz_path = self.save_path / 'visualizations' / 'brain_embedding_design.png'
        plt.savefig(embedding_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 添加plt.close()
        
        logger.info(f"  ✅ 脑区embedding设计图表已保存: {embedding_viz_path}")

    def phase4_decision_generation(self):
        """
        Phase 4: 决策建议生成 - 保留原有决策 + 新增脑区感知建议
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 Phase 4: 决策建议生成 (脑区感知增强版)")
        logger.info("="*80)
        
        # 🔄 保留原有决策逻辑
        logger.info("\n📊 Phase 4A: 全局决策生成...")
        global_decision = self._generate_global_decision()
        
        # 🔥 新增脑区感知决策
        logger.info("\n📊 Phase 4B: 脑区感知决策生成...")
        brain_aware_decision = self._generate_brain_aware_decision()
        
        # 综合决策
        logger.info("\n📊 Phase 4C: 综合决策整合...")
        comprehensive_decision = self._integrate_comprehensive_decision(global_decision, brain_aware_decision)
        
        # 生成实施建议
        logger.info("\n📊 Phase 4D: 实施建议生成...")
        implementation_plan = self._generate_implementation_plan(comprehensive_decision)
        
        # 最终决策结果
        final_decision = {
            'global_analysis': global_decision,
            'brain_aware_analysis': brain_aware_decision,
            'comprehensive_recommendation': comprehensive_decision,
            'implementation_plan': implementation_plan,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_version': 'Brain-Aware v1.0'
        }
        
        self.analysis_results['final_comprehensive_decision'] = final_decision
        
        # 打印决策报告
        self._print_comprehensive_decision_report(final_decision) 
    
        
        return final_decision

    def _generate_global_decision(self):
        """生成全局决策 (基于原有逻辑)"""
        
        # 综合所有决策得分
        all_scores = {}
        for phase in self.decision_scores:
            all_scores.update(self.decision_scores[phase])
        
        # 原有权重
        weights = {
            'difference_significance': 0.15,
            'pattern_linearity': 0.15,
            'subject_separability': 0.15,
            'class_consistency': 0.10,
            'sample_adequacy': 0.10,
            'embedding_feasibility': 0.15,
            'intrinsic_dimensionality': 0.10,
            'high_dim_performance': 0.10
        }
        
        # 计算加权总分
        weighted_score = sum(all_scores.get(key, 0) * weight 
                            for key, weight in weights.items())
        
        # 全局决策
        if weighted_score > 0.75:
            global_recommendation = "强烈推荐使用Subject Embedding"
            confidence = "高"
        elif weighted_score > 0.6:
            global_recommendation = "建议使用Subject Embedding"
            confidence = "中高"
        elif weighted_score > 0.45:
            global_recommendation = "可以尝试Subject Embedding"
            confidence = "中"
        else:
            global_recommendation = "Subject Embedding效果可能有限"
            confidence = "低"
        
        return {
            'weighted_score': weighted_score,
            'recommendation': global_recommendation,
            'confidence': confidence,
            'key_scores': all_scores
        }

    def _generate_brain_aware_decision(self):
        """生成脑区感知决策"""
        
        brain_decision = {
            'region_tier_analysis': {},
            'embedding_architecture': 'uniform',
            'priority_regions': [],
            'implementation_strategy': 'global_first'
        }
        
        if 'brain_aware_embedding_design' in self.analysis_results:
            brain_design = self.analysis_results['brain_aware_embedding_design']
            
            # 1. 脑区分层分析
            if 'region_requirements' in brain_design:
                requirements = brain_design['region_requirements']
                
                total_regions = sum(len(tier['region_ids']) for tier in requirements.values())
                critical_regions = len(requirements.get('tier1_high_specificity', {}).get('region_ids', []))
                high_regions = len(requirements.get('tier2_medium_high_specificity', {}).get('region_ids', []))
                
                brain_decision['region_tier_analysis'] = {
                    'total_analyzed_regions': total_regions,
                    'critical_priority_regions': critical_regions,
                    'high_priority_regions': high_regions,
                    'priority_ratio': (critical_regions + high_regions) / total_regions if total_regions > 0 else 0
                }
                
                # 优先脑区列表
                brain_decision['priority_regions'] = (
                    requirements.get('tier1_high_specificity', {}).get('region_ids', []) +
                    requirements.get('tier2_medium_high_specificity', {}).get('region_ids', [])
                )
            
            # 2. 推荐embedding架构
            if brain_decision['region_tier_analysis'].get('priority_ratio', 0) > 0.3:
                brain_decision['embedding_architecture'] = 'hierarchical'
                brain_decision['implementation_strategy'] = 'tier_based'
            elif brain_decision['region_tier_analysis'].get('critical_priority_regions', 0) > 0:
                brain_decision['embedding_architecture'] = 'selective'
                brain_decision['implementation_strategy'] = 'critical_first'
            else:
                brain_decision['embedding_architecture'] = 'uniform'
                brain_decision['implementation_strategy'] = 'global_only'
        
        return brain_decision

    def _integrate_comprehensive_decision(self, global_decision, brain_aware_decision):
        """整合综合决策"""
        
        # 综合推荐强度
        global_score = global_decision['weighted_score']
        priority_ratio = brain_aware_decision['region_tier_analysis'].get('priority_ratio', 0)
        
        # 调整后的综合得分
        brain_aware_boost = min(0.2, priority_ratio * 0.4)  # 脑区感知可以提升最多0.2分
        comprehensive_score = min(1.0, global_score + brain_aware_boost)
        
        # 综合推荐
        if comprehensive_score > 0.8:
            final_recommendation = "强烈推荐脑区感知Subject Embedding"
            implementation_priority = "HIGH"
        elif comprehensive_score > 0.65:
            final_recommendation = "建议使用分层Subject Embedding"
            implementation_priority = "MEDIUM-HIGH"
        elif comprehensive_score > 0.5:
            final_recommendation = "可选择性使用Subject Embedding"
            implementation_priority = "MEDIUM"
        else:
            final_recommendation = "Subject Embedding收益可能有限"
            implementation_priority = "LOW"
        
        return {
            'final_recommendation': final_recommendation,
            'comprehensive_score': comprehensive_score,
            'implementation_priority': implementation_priority,
            'architecture_type': brain_aware_decision['embedding_architecture'],
            'implementation_strategy': brain_aware_decision['implementation_strategy'],
            'global_component': global_decision,
            'brain_aware_component': brain_aware_decision
        }

    def _generate_implementation_plan(self, comprehensive_decision):
        """生成具体实施计划"""
        
        implementation_plan = {
            'phase1_preparation': [],
            'phase2_development': [],
            'phase3_deployment': [],
            'estimated_timeline': '',
            'resource_requirements': {},
            'success_metrics': []
        }
        
        architecture_type = comprehensive_decision['architecture_type']
        
        # Phase 1: 准备阶段
        implementation_plan['phase1_preparation'] = [
            "完成详细的脑区特异性分析验证",
            "确定最终的脑区分层策略",
            "设计数据预处理pipeline"
        ]
        
        # Phase 2: 开发阶段
        if architecture_type == 'hierarchical':
            implementation_plan['phase2_development'] = [
                "实现层次化Subject Embedding架构",
                "开发分脑区的embedding训练策略",
                "实现脑区特异的新受试者适应机制",
                "建立脑区间embedding共享机制"
            ]
            implementation_plan['estimated_timeline'] = "4-6周"
            
        elif architecture_type == 'selective':
            implementation_plan['phase2_development'] = [
                "实现选择性Subject Embedding系统",
                "为优先脑区开发专门的embedding策略",
                "建立脑区重要性动态评估机制"
            ]
            implementation_plan['estimated_timeline'] = "3-4周"
            
        else:  # uniform
            implementation_plan['phase2_development'] = [
                "实现统一的Subject Embedding系统",
                "优化全局embedding维度和架构",
                "开发通用的新受试者适应策略"
            ]
            implementation_plan['estimated_timeline'] = "2-3周"
        
        # Phase 3: 部署阶段
        implementation_plan['phase3_deployment'] = [
            "在验证集上测试embedding效果",
            "与baseline模型进行对比评估",
            "优化hyperparameters",
            "部署到测试集进行最终评估"
        ]
        
        # 资源需求
        implementation_plan['resource_requirements'] = {
            'computational': 'GPU训练资源，建议V100或以上',
            'memory': '至少32GB RAM用于大规模数据处理',
            'storage': '充足存储空间保存embedding和中间结果',
            'development_time': implementation_plan['estimated_timeline']
        }
        
        # 成功指标
        implementation_plan['success_metrics'] = [
            "验证集F1分数提升 > 5%",
            "测试集泛化性能提升 > 3%",
            "优先脑区分类准确率显著提升",
            "新受试者适应速度和效果评估"
        ]
        
        return implementation_plan


    def _generate_basic_visualizations(self, axes):
        """生成所有分析结果的可视化图表"""
        logger.info("\n" + "="*80)
        logger.info("📊 生成可视化图表")
        logger.info("="*80)
        
        # 创建大图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # 1. 受试者相似性热图
        ax = axes[0, 0]
        correlation_matrix = self.analysis_results['subject_similarity']['correlation_matrix']
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('受试者间相关性矩阵', fontweight='bold')
        ax.set_xlabel('受试者 ID')
        ax.set_ylabel('受试者 ID')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 2. 层次聚类树状图
        ax = axes[0, 1]
        linkage_matrix = self.analysis_results['subject_similarity']['linkage_matrix']
        dendrogram(linkage_matrix, ax=ax, leaf_rotation=90)
        ax.set_title('受试者层次聚类', fontweight='bold')
        ax.set_xlabel('受试者 ID')
        ax.set_ylabel('距离')
        
        # 3. 特征变异分布
        ax = axes[0, 2]
        f_stats = self.analysis_results['feature_variation']['f_stats']
        ax.hist(f_stats, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.percentile(f_stats, 90), color='red', linestyle='--', 
                  label=f'90th percentile: {np.percentile(f_stats, 90):.2f}')
        ax.axvline(np.percentile(f_stats, 10), color='green', linestyle='--',
                  label=f'10th percentile: {np.percentile(f_stats, 10):.2f}')
        ax.set_title('特征受试者间变异分布', fontweight='bold')
        ax.set_xlabel('标准化F统计量')
        ax.set_ylabel('特征数量')
        ax.legend()
        
        # 4. PCA累积方差解释
        ax = axes[0, 3]
        cumulative_variance = self.analysis_results['feature_variation']['pca_cumulative_variance']
        ax.plot(range(1, min(21, len(cumulative_variance)+1)), 
               cumulative_variance[:20], 'o-', linewidth=2, markersize=6)
        ax.axhline(0.8, color='red', linestyle='--', label='80%解释阈值')
        ax.axhline(0.9, color='orange', linestyle='--', label='90%解释阈值')
        ax.set_title('PCA累积方差解释', fontweight='bold')
        ax.set_xlabel('主成分数量')
        ax.set_ylabel('累积方差解释比例')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 受试者识别准确率
        ax = axes[1, 0]
        if 'subject_identification' in self.analysis_results:
            cv_scores = self.analysis_results['subject_identification'].get('cv_scores', [0])
            accuracy = self.analysis_results['subject_identification'].get('accuracy', 0)
            n_subjects = self.analysis_results['subject_identification'].get('n_valid_subjects', 1)
            random_baseline = 1 / n_subjects
            
            ax.bar(['随机基线', '逻辑回归'], [random_baseline, accuracy], 
                  color=['gray', 'lightcoral'])
            ax.set_title('受试者识别准确率对比', fontweight='bold')
            ax.set_ylabel('准确率')
            
            # 添加数值标签
            ax.text(0, random_baseline + 0.01, f'{random_baseline:.3f}', 
                   ha='center', va='bottom')
            ax.text(1, accuracy + 0.01, f'{accuracy:.3f}', 
                   ha='center', va='bottom')
        
        # 6. 特征重要性（Top 20）
        ax = axes[1, 1]
        if 'subject_identification' in self.analysis_results and 'feature_importance' in self.analysis_results['subject_identification']:
            feature_importance = self.analysis_results['subject_identification']['feature_importance']
            top_features = np.argsort(feature_importance)[-20:]
            ax.barh(range(20), feature_importance[top_features], color='lightgreen')
            ax.set_title('Top 20 受试者识别特征重要性', fontweight='bold')
            ax.set_xlabel('重要性得分')
            ax.set_ylabel('特征索引')
            ax.set_yticks(range(20))
            ax.set_yticklabels(top_features)
        
        # 7. PCA vs t-SNE受试者分布对比
        ax = axes[1, 2]
        if 'linearity_analysis' in self.analysis_results:
            pca_2d = self.analysis_results['linearity_analysis']['pca_2d']
            scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                               c=range(len(pca_2d)), cmap='viridis', s=50, alpha=0.7)
            ax.set_title('PCA - 受试者2D分布', fontweight='bold')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, 3]
        if 'linearity_analysis' in self.analysis_results:
            tsne_2d = self.analysis_results['linearity_analysis']['tsne_2d']
            scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], 
                               c=range(len(tsne_2d)), cmap='viridis', s=50, alpha=0.7)
            ax.set_title('t-SNE - 受试者2D分布', fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        # 8. 聚类质量评估
        ax = axes[2, 0]
        if 'linearity_analysis' in self.analysis_results and 'silhouette_scores' in self.analysis_results['linearity_analysis']:
            silhouette_scores = self.analysis_results['linearity_analysis']['silhouette_scores']
            n_clusters_range = range(2, 2 + len(silhouette_scores))
            ax.plot(n_clusters_range, silhouette_scores, 'o-', linewidth=2, markersize=8)
            best_n = self.analysis_results['linearity_analysis']['best_n_clusters']
            best_score = self.analysis_results['linearity_analysis']['best_silhouette']
            ax.axvline(best_n, color='red', linestyle='--', 
                      label=f'最优聚类数: {best_n}')
            ax.set_title('聚类质量评估 (轮廓系数)', fontweight='bold')
            ax.set_xlabel('聚类数量')
            ax.set_ylabel('轮廓系数')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 9. 脑区一致性分析
        ax = axes[2, 1]
        if 'class_consistency' in self.analysis_results and self.analysis_results['class_consistency']:
            class_consistency = self.analysis_results['class_consistency']
            class_ids = list(class_consistency.keys())
            cv_values = [class_consistency[cid]['mean_cv'] for cid in class_ids]
            
            ax.bar(range(len(class_ids)), cv_values, color='lightblue', alpha=0.7)
            ax.set_title('脑区受试者间一致性', fontweight='bold')
            ax.set_xlabel('脑区 ID')
            ax.set_ylabel('变异系数 (越低越一致)')
            ax.set_xticks(range(len(class_ids)))
            ax.set_xticklabels(class_ids, rotation=45)
        
        # 10. 样本量分布
        ax = axes[2, 2]
        sample_counts = self.analysis_results['subject_stats']['sample_counts']
        ax.hist(sample_counts, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(np.mean(sample_counts), color='red', linestyle='--',
                  label=f'平均: {np.mean(sample_counts):.0f}')
        ax.axvline(np.median(sample_counts), color='green', linestyle='--',
                  label=f'中位数: {np.median(sample_counts):.0f}')
        ax.set_title('受试者样本量分布', fontweight='bold')
        ax.set_xlabel('样本量')
        ax.set_ylabel('受试者数量')
        ax.legend()
        
        # 11. 综合决策雷达图
        ax = axes[2, 3]
        categories = ['差异显著性', '模式线性度', '受试者相似性', '特征异质性', 
                     '可分离性', '类别一致性', '样本充足性']
        
        scores = [
            self.decision_scores['phase1']['difference_significance'],
            self.decision_scores['phase1']['pattern_linearity'], 
            self.decision_scores['phase1']['subject_similarity'],
            self.decision_scores['phase1']['feature_heterogeneity'],
            self.decision_scores['phase2']['subject_separability'],
            self.decision_scores['phase2']['class_consistency'],
            self.decision_scores['phase3']['sample_adequacy']
        ]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores_plot = scores + [scores[0]]  # 闭合图形
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='blue')
        ax.fill(angles_plot, scores_plot, alpha=0.25, color='blue')
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Subject Embedding 可行性雷达图', fontweight='bold')
        ax.grid(True)
        
        plt.tight_layout()
        viz_path = self.save_path / 'visualizations' / 'comprehensive_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 可视化图表已保存: {viz_path}")
    
    def _generate_implementation_suggestions(self, embedding_suggestions, all_scores, enhanced_results):
        """生成具体的实施建议"""
        suggestions = []
        
        # 基础实施建议
        embedding_dim = embedding_suggestions.get('embedding_dim', 64)
        suggestions.append(f"推荐embedding维度: {embedding_dim}")
        
        # Embedding类型建议
        embedding_type = embedding_suggestions.get('embedding_type', 'linear')
        if embedding_type == 'linear':
            suggestions.extend([
                "使用简单的nn.Embedding层",
                "学习率建议: 1e-3到1e-4",
                "添加L2正则化防止过拟合"
            ])
        elif embedding_type == 'mixed_linear':
            suggestions.extend([
                "使用nn.Embedding + 小型MLP",
                "MLP建议: embedding_dim -> embedding_dim*2 -> embedding_dim",
                "学习率建议: 1e-4到1e-5"
            ])
        else:  # nonlinear
            suggestions.extend([
                "使用深度MLP进行非线性embedding",
                "MLP建议: embedding_dim -> embedding_dim*4 -> embedding_dim*2 -> embedding_dim",
                "使用批归一化和Dropout"
            ])
        
        # 特征处理建议
        if embedding_suggestions.get('feature_selection_needed', False):
            suggestions.append("强烈建议进行特征选择，保留最重要的50-70%特征")
        
        if embedding_suggestions.get('use_feature_groups', False):
            best_group = embedding_suggestions.get('best_feature_group', '')
            suggestions.append(f"可优先使用{best_group}特征组进行embedding")
        
        # 去偏差建议
        if embedding_suggestions.get('use_debiasing', False):
            suggestions.extend([
                "实施特征去偏差策略",
                "考虑对抗训练: 最大化分类性能，最小化受试者识别",
                "使用梯度反转层(Gradient Reversal Layer)"
            ])
        
        # 混合专家建议
        if embedding_suggestions.get('use_mixture_experts', False):
            n_experts = embedding_suggestions.get('n_expert_clusters', 3)
            suggestions.extend([
                f"考虑Mixture of Experts架构，使用{n_experts}个专家",
                "先对受试者进行聚类，再训练每个专家",
                "使用门控网络动态选择专家"
            ])
        
        # 训练策略建议
        complexity = embedding_suggestions.get('complexity_level', 'medium')
        if complexity == 'simple':
            suggestions.extend([
                "使用较小的batch size (64-128)",
                "早停patience建议: 5-10 epochs",
                "使用简单的学习率调度"
            ])
        elif complexity == 'medium':
            suggestions.extend([
                "使用中等batch size (128-256)",
                "早停patience建议: 10-15 epochs", 
                "可尝试余弦退火学习率调度"
            ])
        else:  # complex
            suggestions.extend([
                "使用较大batch size (256-512)",
                "早停patience建议: 15-20 epochs",
                "使用warmup + 余弦退火策略"
            ])
        
        return suggestions

    def _suggest_adaptation_strategy(self, all_scores, enhanced_results, embedding_suggestions):
        """建议新受试者适应策略"""
        strategy = {
            'primary_method': '',
            'backup_methods': [],
            'required_samples': 0,
            'expected_performance': '',
            'implementation_notes': []
        }
        
        # 基于受试者可分离性选择主要策略
        separability = all_scores.get('subject_separability', 0.5)
        
        if separability > 0.7:
            # 高可分离性：需要快速适应
            strategy['primary_method'] = '少样本快速适应'
            strategy['required_samples'] = 500
            strategy['backup_methods'] = ['相似性迁移', '零样本泛化']
            strategy['expected_performance'] = '85-95%的最优性能'
            strategy['implementation_notes'] = [
                "冻结主模型参数，只训练新受试者embedding",
                "使用较高学习率(1e-3)快速收敛",
                "监控过拟合，通常10-50轮即可"
            ]
        
        elif separability > 0.4:
            # 中等可分离性：相似性迁移为主
            strategy['primary_method'] = '相似性迁移 + 微调'
            strategy['required_samples'] = 200
            strategy['backup_methods'] = ['少样本适应', '平均embedding初始化']
            strategy['expected_performance'] = '75-85%的最优性能'
            strategy['implementation_notes'] = [
                "先找最相似的受试者继承embedding",
                "用少量样本进行微调",
                "相似性可基于特征统计量判断"
            ]
        
        else:
            # 低可分离性：零样本泛化
            strategy['primary_method'] = '零样本泛化'
            strategy['required_samples'] = 0
            strategy['backup_methods'] = ['平均embedding', '回归预测embedding']
            strategy['expected_performance'] = '60-75%的最优性能'
            strategy['implementation_notes'] = [
                "使用所有训练受试者embedding的均值",
                "或基于特征统计量回归预测embedding",
                "无需新受试者的标注数据"
            ]
        
        # 基于聚类结果调整策略
        if embedding_suggestions.get('use_mixture_experts', False):
            strategy['implementation_notes'].append(
                "如果使用Mixture of Experts，需要先确定新受试者属于哪个专家组"
            )
        
        # 基于内在维度调整样本需求
        if 'high_dimensional_analysis' in enhanced_results:
            intrinsic_dim = enhanced_results['high_dimensional_analysis'].get('intrinsic_dimensionality', 100)
            if intrinsic_dim < 20:
                strategy['required_samples'] = max(50, strategy['required_samples'] // 2)
                strategy['implementation_notes'].append("内在维度低，需要样本量可以减半")
            elif intrinsic_dim > 100:
                strategy['required_samples'] = strategy['required_samples'] * 2
                strategy['implementation_notes'].append("内在维度高，建议增加样本量")
        
        return strategy

    def _generate_dimensionality_comparison_plot(self):
        """生成专门的降维方法对比图"""
        if 'enhanced_dimensionality' not in self.analysis_results:
            return
        
        enhanced_results = self.analysis_results['enhanced_dimensionality']
        
        if 'dimensionality_comparison' not in enhanced_results:
            return
        
        dim_results = enhanced_results['dimensionality_comparison']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # PCA结果
        if 'pca' in dim_results and dim_results['pca'] is not None:
            ax = axes[0, 0]
            pca_2d = dim_results['pca']['2d_result']
            scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                            c=range(len(pca_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('PCA降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax)
        
        # Kernel PCA结果  
        if 'kernel_pca' in dim_results and dim_results['kernel_pca'] is not None:
            ax = axes[0, 1]
            kpca_2d = dim_results['kernel_pca']['rbf_2d_result']
            scatter = ax.scatter(kpca_2d[:, 0], kpca_2d[:, 1],
                            c=range(len(kpca_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('Kernel PCA (RBF)降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('KPC1')
            ax.set_ylabel('KPC2')
            plt.colorbar(scatter, ax=ax)
        
        # t-SNE结果
        if 'tsne' in dim_results and dim_results['tsne'] is not None:
            ax = axes[0, 2]
            tsne_2d = dim_results['tsne']['2d_result']
            scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1],
                            c=range(len(tsne_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('t-SNE降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax)
        
        # UMAP结果
        if 'umap' in dim_results and dim_results['umap'] is not None:
            ax = axes[1, 0]
            umap_2d = dim_results['umap']['2d_result']
            scatter = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                            c=range(len(umap_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('UMAP降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=ax)
        
        # PCA解释方差
        if 'pca' in dim_results:
            ax = axes[1, 1]
            explained_var = dim_results['pca']['explained_variance'][:20]  # 前20个
            ax.plot(range(1, len(explained_var)+1), explained_var, 'o-', linewidth=2, markersize=6)
            ax.set_title('PCA解释方差', fontweight='bold', fontsize=14)
            ax.set_xlabel('主成分')
            ax.set_ylabel('解释方差比例')
            ax.grid(True, alpha=0.3)
        
        # 结构保持能力对比
        if 'structure_preservation' in dim_results:
            ax = axes[1, 2]
            methods = []
            correlations = []
            
            for method, scores in dim_results['structure_preservation'].items():
                if 'spearman_correlation' in scores:
                    methods.append(method)
                    correlations.append(abs(scores['spearman_correlation']))
            
            if methods:
                bars = ax.bar(methods, correlations, color='lightcoral', alpha=0.7)
                ax.set_title('结构保持能力对比', fontweight='bold', fontsize=14)
                ax.set_ylabel('Spearman相关系数')
                ax.set_xticklabels(methods, rotation=45)
                
                # 添加数值标签
                for bar, corr in zip(bars, correlations):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        dim_viz_path = self.save_path / 'visualizations'/ 'dimensionality_comparison.png'
        plt.savefig(dim_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ 降维对比图已保存: {dim_viz_path}")

    def _print_comprehensive_decision_report(self, final_decision):
        """打印综合决策报告"""
        
        logger.info(f"\n📋 脑区感知Subject Embedding 综合分析报告")
        logger.info("=" * 80)
        
        comp_decision = final_decision['comprehensive_recommendation']
        global_decision = final_decision['global_analysis']
        brain_decision = final_decision['brain_aware_analysis']
        impl_plan = final_decision['implementation_plan']
        
        logger.info(f"🎯 最终推荐: {final_decision['final_recommendation']}")
        logger.info(f"📊 综合得分: {comp_decision['comprehensive_score']:.3f} / 1.0")
        logger.info(f"⚡ 实施优先级: {comp_decision['implementation_priority']}")
        logger.info(f"🏗️ 推荐架构: {comp_decision['architecture_type'].upper()}")
        
        logger.info(f"\n📈 全局分析结果:")
        logger.info(f"  - 加权得分: {global_decision['weighted_score']:.3f}")
        logger.info(f"  - 置信度: {global_decision['confidence']}")
        logger.info(f"  - 基础推荐: {global_decision['recommendation']}")
        
        logger.info(f"\n🧠 脑区感知分析结果:")
        tier_analysis = brain_decision['region_tier_analysis']
        logger.info(f"  - 分析脑区总数: {tier_analysis.get('total_analyzed_regions', 0)}")
        logger.info(f"  - 极高优先级脑区: {tier_analysis.get('critical_priority_regions', 0)}")
        logger.info(f"  - 高优先级脑区: {tier_analysis.get('high_priority_regions', 0)}")
        logger.info(f"  - 优先级脑区比例: {tier_analysis.get('priority_ratio', 0):.1%}")
        
        logger.info(f"\n🛠️ 实施计划:")
        logger.info(f"  - 预估时间线: {impl_plan['estimated_timeline']}")
        logger.info(f"  - 实施策略: {comp_decision['implementation_strategy']}")
        logger.info(f"  - 主要阶段: 准备 → 开发 → 部署")
        
        # 详细的实施建议
        logger.info(f"\n📋 详细实施阶段:")
        logger.info(f"  📍 Phase 1 - 准备阶段:")
        for item in impl_plan['phase1_preparation']:
            logger.info(f"    • {item}")
        
        logger.info(f"  📍 Phase 2 - 开发阶段:")
        for item in impl_plan['phase2_development']:
            logger.info(f"    • {item}")
        
        logger.info(f"  📍 Phase 3 - 部署阶段:")
        for item in impl_plan['phase3_deployment']:
            logger.info(f"    • {item}")
        
        logger.info(f"\n📊 成功指标:")
        for metric in impl_plan['success_metrics']:
            logger.info(f"  📈 {metric}")
        
        logger.info(f"\n💻 资源需求:")
        resources = impl_plan['resource_requirements']
        for key, value in resources.items():
            logger.info(f"  🔧 {key.replace('_', ' ').title()}: {value}")
        
        logger.info(f"\n🎯 关键建议:")
        if comp_decision['architecture_type'] == 'hierarchical':
            logger.info("  ✅ 实施层次化架构，为不同特异性的脑区使用不同embedding策略")
            logger.info("  ✅ 优先处理高特异性脑区，预期获得最大性能提升")
            logger.info("  ✅ 建立脑区间embedding共享机制，提高效率")
        elif comp_decision['architecture_type'] == 'selective':
            logger.info("  ✅ 专注于关键脑区的embedding优化")
            logger.info("  ✅ 为优先脑区分配更多计算资源")
            logger.info("  ✅ 保持其他脑区的简单处理策略")
        else:
            logger.info("  ✅ 使用统一的全局Subject Embedding策略")
            logger.info("  ✅ 专注于优化全局embedding质量")
            logger.info("  ✅ 简化实施复杂度，快速验证效果")
        
        logger.info(f"\n📊 预期效果:")
        if comp_decision['comprehensive_score'] > 0.8:
            logger.info("  🚀 预期显著性能提升 (5-15%)")
            logger.info("  🚀 强烈建议立即实施")
        elif comp_decision['comprehensive_score'] > 0.65:
            logger.info("  📈 预期中等性能提升 (3-8%)")
            logger.info("  📈 建议优先考虑实施")
        else:
            logger.info("  📊 预期小幅性能提升 (1-5%)")
            logger.info("  📊 可作为优化方向之一")
        
        # 优先脑区信息
        if brain_decision['priority_regions']:
            logger.info(f"\n🔥 优先处理脑区:")
            priority_regions = brain_decision['priority_regions'][:10]  # 显示前10个
            logger.info(f"  📍 前10个优先脑区: {priority_regions}")
            if len(brain_decision['priority_regions']) > 10:
                logger.info(f"  📍 总共{len(brain_decision['priority_regions'])}个优先脑区")
        
        logger.info(f"\n⏱️ 分析完成时间: {final_decision['analysis_timestamp']}")
        logger.info(f"🔬 分析版本: {final_decision['analysis_version']}")

    def generate_report(self):
        """生成完整的脑区感知分析报告"""
        report_path = self.save_path / 'brain_aware_subject_embedding_analysis_report.txt'
    

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("脑区感知Subject Embedding 可行性分析报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析版本: 脑区感知增强版 v1.0\n\n")
            
            # 数据概况
            f.write("📊 数据概况\n")
            f.write("-" * 40 + "\n")
            f.write(f"受试者数量: {len(self.data['available_subjects'])}\n")
            f.write(f"特征维度: {self.data['X_train'].shape[1]}\n")
            f.write(f"训练样本量: {len(self.data['X_train']):,}\n")
            f.write(f"验证样本量: {len(self.data['X_val']):,}\n")
            f.write(f"测试样本量: {len(self.data['X_test']):,}\n\n")
            
            # 脑区分析概况
            if 'brain_region_analysis' in self.data:
                brain_data = self.data['brain_region_analysis']
                f.write("🧠 脑区分析概况\n")
                f.write("-" * 40 + "\n")
                f.write(f"有效脑区×受试者组合: {brain_data['valid_combinations']}\n")
                f.write(f"总可能组合: {brain_data['total_combinations']}\n")
                f.write(f"覆盖率: {brain_data['valid_combinations']/brain_data['total_combinations']*100:.1f}%\n\n")
            
            # 全局分析结果
            f.write("📈 全局分析结果\n")
            f.write("-" * 40 + "\n")
            if 'global_feature_variation' in self.analysis_results:
                global_analysis = self.analysis_results['global_feature_variation']
                f.write(f"PCA前3PC解释方差: {np.sum(global_analysis['pca_explained_variance'][:3]):.3f}\n")
                f.write(f"PCA前10PC解释方差: {np.sum(global_analysis['pca_explained_variance'][:10]):.3f}\n")
            
            if 'global_subject_identification' in self.analysis_results:
                global_id = self.analysis_results['global_subject_identification']
                f.write(f"全局受试者识别准确率: {global_id['accuracy']:.3f}\n")
                f.write(f"随机基线: {global_id['random_baseline']:.3f}\n\n")
            
            # 脑区感知分析结果
            f.write("🧠 脑区感知分析结果\n")
            f.write("-" * 40 + "\n")
            if 'region_wise_subject_analysis' in self.analysis_results:
                region_analysis = self.analysis_results['region_wise_subject_analysis']
                f.write(f"成功分析脑区数: {len(region_analysis)}\n")
                
                if region_analysis:
                    specificity_scores = [r['subject_specificity_score'] for r in region_analysis.values()]
                    f.write(f"平均特异性得分: {np.mean(specificity_scores):.3f}\n")
                    f.write(f"特异性得分范围: [{np.min(specificity_scores):.3f}, {np.max(specificity_scores):.3f}]\n")
            
            if 'region_wise_separability' in self.analysis_results:
                region_sep = self.analysis_results['region_wise_separability']
                if region_sep:
                    sep_scores = [r['accuracy'] for r in region_sep.values()]
                    f.write(f"平均脑区识别准确率: {np.mean(sep_scores):.3f}\n\n")
            
            # 最终决策
            f.write("🎯 最终综合决策\n")
            f.write("-" * 40 + "\n")
            if 'final_comprehensive_decision' in self.analysis_results:
                final_decision = self.analysis_results['final_comprehensive_decision']
                comp_rec = final_decision['comprehensive_recommendation']
                
                f.write(f"最终推荐: {comp_rec['final_recommendation']}\n")
                f.write(f"综合得分: {comp_rec['comprehensive_score']:.3f}\n")
                f.write(f"实施优先级: {comp_rec['implementation_priority']}\n")
                f.write(f"推荐架构: {comp_rec['architecture_type']}\n")
                f.write(f"实施策略: {comp_rec['implementation_strategy']}\n\n")
                
                # 实施计划
                impl_plan = final_decision['implementation_plan']
                f.write("实施计划:\n")
                f.write(f"  预估时间: {impl_plan['estimated_timeline']}\n")
                f.write("  主要阶段:\n")
                for phase in impl_plan['phase2_development'][:3]:
                    f.write(f"    • {phase}\n")
        
        logger.info(f"✅ 脑区感知完整报告已保存: {report_path}")
        return report_path
