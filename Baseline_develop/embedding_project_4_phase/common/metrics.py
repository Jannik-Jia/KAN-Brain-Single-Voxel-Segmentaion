#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块
提供各种统计和评估指标的计算
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """评估指标计算器"""
    
    @staticmethod
    def compute_subject_statistics(X: np.ndarray, subjects: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算每个受试者的统计特征
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            subjects: 受试者标签 (n_samples,)
            
        Returns:
            包含各种统计量的字典
        """
        unique_subjects = np.unique(subjects)
        n_features = X.shape[1]
        
        subject_stats = {
            'means': np.zeros((len(unique_subjects), n_features)),
            'stds': np.zeros((len(unique_subjects), n_features)),
            'skews': np.zeros((len(unique_subjects), n_features)),
            'kurts': np.zeros((len(unique_subjects), n_features)),
            'sample_counts': np.zeros(len(unique_subjects))
        }
        
        for i, subject_id in enumerate(unique_subjects):
            subject_mask = subjects == subject_id
            subject_data = X[subject_mask]
            
            if len(subject_data) > 0:
                subject_stats['means'][i] = np.mean(subject_data, axis=0)
                subject_stats['stds'][i] = np.std(subject_data, axis=0)
                
                if len(subject_data) > 3:  # 需要足够样本计算偏度和峰度
                    subject_stats['skews'][i] = stats.skew(subject_data, axis=0)
                    subject_stats['kurts'][i] = stats.kurtosis(subject_data, axis=0)
                
                subject_stats['sample_counts'][i] = len(subject_data)
        
        subject_stats['subject_ids'] = unique_subjects
        return subject_stats
    
    @staticmethod
    def compute_similarity_matrix(features: np.ndarray, metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
        """
        计算特征间的相似性/距离矩阵
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            metric: 距离度量方法
            
        Returns:
            距离矩阵和相关性矩阵
        """
        # 距离矩阵
        distance_matrix = squareform(pdist(features, metric=metric))
        
        # 相关性矩阵
        correlation_matrix = np.corrcoef(features)
        
        return distance_matrix, correlation_matrix
    
    @staticmethod
    def compute_feature_variation(X: np.ndarray, subjects: np.ndarray) -> Dict[str, Any]:
        """
        计算特征的受试者间变异
        
        Args:
            X: 特征矩阵
            subjects: 受试者标签
            
        Returns:
            特征变异分析结果
        """
        unique_subjects = np.unique(subjects)
        n_features = X.shape[1]
        
        # 计算每个受试者的均值
        subject_means = np.zeros((len(unique_subjects), n_features))
        for i, subject_id in enumerate(unique_subjects):
            subject_mask = subjects == subject_id
            subject_means[i] = np.mean(X[subject_mask], axis=0)
        
        # 计算特征的F统计量（受试者间方差/受试者内方差）
        feature_f_stats = np.var(subject_means, axis=0)
        feature_f_stats_norm = feature_f_stats / (np.mean(feature_f_stats) + 1e-8)
        
        # 识别高/低变异特征
        high_variation_threshold = np.percentile(feature_f_stats_norm, 90)
        low_variation_threshold = np.percentile(feature_f_stats_norm, 10)
        
        high_variation_features = np.where(feature_f_stats_norm > high_variation_threshold)[0]
        low_variation_features = np.where(feature_f_stats_norm < low_variation_threshold)[0]
        
        return {
            'f_stats': feature_f_stats_norm,
            'high_variation_features': high_variation_features,
            'low_variation_features': low_variation_features,
            'mean_variation': np.mean(feature_f_stats_norm),
            'std_variation': np.std(feature_f_stats_norm)
        }
    
    @staticmethod
    def compute_region_statistics(X: np.ndarray, regions: np.ndarray, subjects: np.ndarray) -> Dict[int, Dict]:
        """
        计算每个脑区的统计信息
        
        Args:
            X: 特征矩阵
            regions: 脑区标签
            subjects: 受试者标签
            
        Returns:
            每个脑区的统计信息
        """
        unique_regions = np.unique(regions)
        region_stats = {}
        
        for region_id in unique_regions:
            region_mask = regions == region_id
            region_data = X[region_mask]
            region_subjects = subjects[region_mask]
            
            unique_subjects_in_region = np.unique(region_subjects)
            
            region_stats[int(region_id)] = {
                'n_voxels': np.sum(region_mask),
                'n_subjects': len(unique_subjects_in_region),
                'mean_features': np.mean(region_data, axis=0),
                'std_features': np.std(region_data, axis=0),
                'subject_list': unique_subjects_in_region.tolist()
            }
        
        return region_stats
    
    @staticmethod
    def compute_subject_region_matrix(X: np.ndarray, regions: np.ndarray, 
                                    subjects: np.ndarray, min_samples: int = 50) -> Dict[str, Any]:
        """
        构建受试者×脑区特征矩阵
        
        Args:
            X: 特征矩阵
            regions: 脑区标签
            subjects: 受试者标签
            min_samples: 最小样本数阈值
            
        Returns:
            受试者-脑区分析数据
        """
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(regions)
        
        subject_region_features = {}
        sample_counts = {}
        
        valid_combinations = 0
        total_combinations = len(unique_subjects) * len(unique_regions)
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects == subject_id) & (regions == region_id)
                n_samples = np.sum(mask)
                
                if n_samples >= min_samples:
                    region_features = np.mean(X[mask], axis=0)
                    subject_region_features[(int(subject_id), int(region_id))] = region_features
                    sample_counts[(int(subject_id), int(region_id))] = n_samples
                    valid_combinations += 1
        
        # 统计每个脑区的受试者覆盖
        region_subject_counts = {}
        for region_id in unique_regions:
            region_subjects = [s for s, r in subject_region_features.keys() if r == region_id]
            region_subject_counts[int(region_id)] = len(region_subjects)
        
        return {
            'subject_region_features': subject_region_features,
            'sample_counts': sample_counts,
            'unique_subjects': unique_subjects,
            'unique_regions': unique_regions,
            'valid_combinations': valid_combinations,
            'total_combinations': total_combinations,
            'region_subject_counts': region_subject_counts
        }