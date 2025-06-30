#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局受试者分析器
分析受试者间的整体差异和相似性
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from common.metrics import Metrics

logger = logging.getLogger(__name__)


class GlobalSubjectAnalyzer:
    """全局受试者分析器"""
    
    def __init__(self, high_variation_percentile: int = 90,
                 low_variation_percentile: int = 10):
        """
        初始化分析器
        
        Args:
            high_variation_percentile: 高变异特征的百分位数阈值
            low_variation_percentile: 低变异特征的百分位数阈值
        """
        self.high_variation_percentile = high_variation_percentile
        self.low_variation_percentile = low_variation_percentile
        self.metrics = Metrics()
    
    def analyze(self, X: np.ndarray, subjects: np.ndarray) -> Dict[str, Any]:
        """
        执行全局受试者差异分析
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            subjects: 受试者标签 (n_samples,)
            
        Returns:
            分析结果字典
        """
        logger.info("开始全局受试者差异分析...")
        
        # 1. 计算受试者统计特征
        logger.info("1. 计算受试者统计特征...")
        subject_stats = self._compute_subject_statistics(X, subjects)
        
        # 2. 分析受试者间相似性
        logger.info("2. 分析受试者间相似性...")
        similarity_results = self._analyze_subject_similarity(subject_stats['means'])
        
        # 3. 分析特征变异模式
        logger.info("3. 分析特征变异模式...")
        variation_results = self._analyze_feature_variation(
            subject_stats['means'], X, subjects
        )
        
        # 整合结果
        results = {
            'subject_stats': subject_stats,
            'similarity': similarity_results,
            'feature_variation': variation_results
        }
        
        # 打印分析摘要
        self._print_analysis_summary(results)
        
        return results
    
    def _compute_subject_statistics(self, X: np.ndarray, subjects: np.ndarray) -> Dict[str, np.ndarray]:
        """计算每个受试者的统计特征"""
        subject_stats = self.metrics.compute_subject_statistics(X, subjects)
        
        logger.info(f"  - 分析了 {len(subject_stats['subject_ids'])} 个受试者")
        logger.info(f"  - 平均每受试者样本量: {np.mean(subject_stats['sample_counts']):.0f}")
        logger.info(f"  - 样本量范围: [{np.min(subject_stats['sample_counts']):.0f}, "
                   f"{np.max(subject_stats['sample_counts']):.0f}]")
        
        return subject_stats
    
    def _analyze_subject_similarity(self, subject_means: np.ndarray) -> Dict[str, Any]:
        """分析受试者间的相似性"""
        n_subjects = len(subject_means)
        n_features = subject_means.shape[1]
        
        # 数值稳定性检查
        feature_variance = np.var(subject_means, axis=0)
        constant_features = np.sum(feature_variance < 1e-12)
        valid_features_mask = feature_variance >= 1e-12
        
        logger.info(f"  - 总特征数: {n_features}")
        logger.info(f"  - 常数特征数: {constant_features}")
        logger.info(f"  - 有效特征数: {np.sum(valid_features_mask)}")
        
        # 过滤常数特征
        if constant_features > 0:
            subject_means_filtered = subject_means[:, valid_features_mask]
        else:
            subject_means_filtered = subject_means
        
        # 计算距离和相关性矩阵
        distance_matrix, correlation_matrix = self.metrics.compute_similarity_matrix(
            subject_means_filtered
        )
        
        # 计算平均距离和相关性
        upper_triangle_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        upper_triangle_corr = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        mean_distance = np.mean(upper_triangle_distances)
        std_distance = np.std(upper_triangle_distances)
        mean_correlation = np.mean(upper_triangle_corr)
        
        logger.info(f"  - 平均受试者间距离: {mean_distance:.6f} (±{std_distance:.6f})")
        logger.info(f"  - 平均受试者间相关性: {mean_correlation:.6f}")
        
        # 层次聚类
        try:
            linkage_matrix = linkage(subject_means_filtered, method='ward')
            logger.info("  - 层次聚类分析完成")
        except Exception as e:
            logger.warning(f"  - 层次聚类失败: {e}")
            linkage_matrix = np.zeros((n_subjects-1, 4))
        
        return {
            'distance_matrix': distance_matrix,
            'correlation_matrix': correlation_matrix,
            'linkage_matrix': linkage_matrix,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'mean_correlation': mean_correlation,
            'constant_features_count': constant_features,
            'valid_features_count': np.sum(valid_features_mask)
        }
    
    def _analyze_feature_variation(self, subject_means: np.ndarray, 
                                  X: np.ndarray, subjects: np.ndarray) -> Dict[str, Any]:
        """分析特征的受试者间变异模式"""
        
        # 计算特征变异
        variation_results = self.metrics.compute_feature_variation(X, subjects)
        
        # F统计量
        f_stats = variation_results['f_stats']
        
        # 识别高/低变异特征
        high_threshold = np.percentile(f_stats, self.high_variation_percentile)
        low_threshold = np.percentile(f_stats, self.low_variation_percentile)
        
        high_variation_features = np.where(f_stats > high_threshold)[0]
        low_variation_features = np.where(f_stats < low_threshold)[0]
        
        logger.info(f"  - 高变异特征数量: {len(high_variation_features)} "
                   f"({len(high_variation_features)/len(f_stats)*100:.1f}%)")
        logger.info(f"  - 低变异特征数量: {len(low_variation_features)} "
                   f"({len(low_variation_features)/len(f_stats)*100:.1f}%)")
        
        # PCA分析
        pca = PCA()
        subject_pca_result = pca.fit_transform(subject_means)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        
        logger.info(f"  - 前3个主成分解释方差: {np.sum(pca.explained_variance_ratio_[:3])*100:.1f}%")
        logger.info(f"  - 需要 {n_components_80} 个主成分达到80%方差")
        
        return {
            'f_stats': f_stats,
            'high_variation_features': high_variation_features,
            'low_variation_features': low_variation_features,
            'mean_f_stat': variation_results['mean_variation'],
            'std_f_stat': variation_results['std_variation'],
            'pca_result': subject_pca_result,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'pca_cumulative_variance': cumulative_variance,
            'n_components_80': n_components_80
        }
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """打印分析摘要"""
        logger.info("\n全局受试者分析摘要:")
        logger.info("="*50)
        
        # 受试者统计
        n_subjects = len(results['subject_stats']['subject_ids'])
        logger.info(f"受试者数量: {n_subjects}")
        
        # 相似性
        similarity = results['similarity']
        logger.info(f"受试者间平均距离: {similarity['mean_distance']:.3f}")
        logger.info(f"受试者间平均相关性: {similarity['mean_correlation']:.3f}")
        
        # 特征变异
        variation = results['feature_variation']
        pca_3pc = np.sum(variation['pca_explained_variance'][:3])
        logger.info(f"高变异特征比例: {len(variation['high_variation_features'])/len(variation['f_stats'])*100:.1f}%")
        logger.info(f"PCA前3成分方差: {pca_3pc*100:.1f}%")
        
        # 建议
        if pca_3pc < 0.5:
            logger.info("\n💡 建议: 低线性度表明可能需要非线性embedding")
        if similarity['mean_correlation'] < 0.5:
            logger.info("💡 建议: 低相关性表明受试者差异较大，Subject Embedding可能有益")