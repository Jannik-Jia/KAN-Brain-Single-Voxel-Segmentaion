#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑区特异性分析器
分析每个脑区的受试者特异性模式
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from common.metrics import Metrics
from common.config import Config

logger = logging.getLogger(__name__)


class RegionSpecificityAnalyzer:
    """脑区特异性分析器"""
    
    def __init__(self, min_samples_per_combination: int = 50):
        """
        初始化分析器
        
        Args:
            min_samples_per_combination: 每个受试者-脑区组合的最小样本数
        """
        self.min_samples = min_samples_per_combination
        self.metrics = Metrics()
    
    def analyze(self, X: np.ndarray, regions: np.ndarray, 
                subjects: np.ndarray) -> Dict[str, Any]:
        """
        执行脑区特异性分析
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            regions: 脑区标签 (n_samples,)
            subjects: 受试者标签 (n_samples,)
            
        Returns:
            分析结果字典
        """
        logger.info("开始脑区特异性分析...")
        
        # 1. 构建受试者-脑区矩阵
        logger.info("1. 构建受试者-脑区特征矩阵...")
        subject_region_data = self._build_subject_region_matrix(X, regions, subjects)
        
        # 2. 计算每个脑区的特异性得分
        logger.info("2. 计算脑区特异性得分...")
        region_specificity_scores = self._compute_region_specificity(subject_region_data)
        
        # 3. 识别高/低特异性脑区
        logger.info("3. 识别高/低特异性脑区...")
        high_regions, low_regions = self._identify_extreme_regions(region_specificity_scores)
        
        # 4. 生成特异性矩阵（用于可视化）
        logger.info("4. 生成特异性矩阵...")
        specificity_matrix = self._create_specificity_matrix(
            subject_region_data, region_specificity_scores
        )
        
        # 整合结果
        results = {
            'subject_region_data': subject_region_data,
            'region_specificity_scores': region_specificity_scores,
            'high_specificity_regions': high_regions,
            'low_specificity_regions': low_regions,
            'specificity_matrix': specificity_matrix,
            'n_regions_analyzed': len(region_specificity_scores),
            'n_valid_combinations': subject_region_data['valid_combinations'],
            'mean_specificity_score': np.mean([s['specificity_score'] 
                                             for s in region_specificity_scores.values()]),
            'specificity_scores': [s['specificity_score'] 
                                 for s in region_specificity_scores.values()]
        }
        
        # 打印分析摘要
        self._print_analysis_summary(results)
        
        return results
    
    def _build_subject_region_matrix(self, X: np.ndarray, regions: np.ndarray,
                                   subjects: np.ndarray) -> Dict[str, Any]:
        """构建受试者×脑区特征矩阵"""
        
        # 使用Metrics模块的功能
        matrix_data = self.metrics.compute_subject_region_matrix(
            X, regions, subjects, self.min_samples
        )
        
        # 添加额外的统计信息
        unique_regions = matrix_data['unique_regions']
        unique_subjects = matrix_data['unique_subjects']
        
        logger.info(f"  - 总脑区数: {len(unique_regions)}")
        logger.info(f"  - 总受试者数: {len(unique_subjects)}")
        logger.info(f"  - 有效组合数: {matrix_data['valid_combinations']} / "
                   f"{matrix_data['total_combinations']} "
                   f"({matrix_data['valid_combinations']/matrix_data['total_combinations']*100:.1f}%)")
        
        # 计算缺失的脑区
        all_possible_regions = set(range(int(np.max(regions)) + 1))
        actual_regions = set(unique_regions.astype(int))
        missing_regions = sorted(all_possible_regions - actual_regions)
        
        if missing_regions:
            logger.info(f"  - 缺失数据的脑区数: {len(missing_regions)}")
            logger.info(f"    前10个缺失脑区: {missing_regions[:10]}")
        
        matrix_data['missing_regions'] = missing_regions
        
        return matrix_data
    
    def _compute_region_specificity(self, subject_region_data: Dict[str, Any]) -> Dict[int, Dict]:
        """计算每个脑区的特异性得分"""
        
        subject_region_features = subject_region_data['subject_region_features']
        unique_regions = subject_region_data['unique_regions']
        
        region_specificity_scores = {}
        
        for region_id in unique_regions:
            region_id = int(region_id)
            
            # 提取该脑区所有受试者的特征
            region_features = []
            valid_subjects = []
            
            for (subject_id, rid), features in subject_region_features.items():
                if rid == region_id:
                    region_features.append(features)
                    valid_subjects.append(subject_id)
            
            if len(region_features) < 3:  # 至少需要3个受试者
                continue
            
            region_features_array = np.array(region_features)
            
            # 计算特异性指标
            specificity_metrics = self._calculate_specificity_metrics(
                region_features_array, valid_subjects
            )
            
            region_specificity_scores[region_id] = specificity_metrics
        
        logger.info(f"  - 成功分析 {len(region_specificity_scores)} 个脑区")
        
        return region_specificity_scores
    
    def _calculate_specificity_metrics(self, region_features: np.ndarray,
                                     valid_subjects: List[int]) -> Dict[str, Any]:
        """计算单个脑区的特异性指标"""
        
        n_subjects = len(valid_subjects)
        
        # 1. 受试者间距离分析
        try:
            distances = squareform(pdist(region_features))
            mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
            std_distance = np.std(distances[np.triu_indices_from(distances, k=1)])
        except:
            mean_distance = 0.0
            std_distance = 1.0
        
        # 2. 受试者间相关性
        try:
            correlations = np.corrcoef(region_features)
            if np.all(np.isfinite(correlations)):
                mean_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
            else:
                mean_correlation = 0.0
        except:
            mean_correlation = 0.0
        
        # 3. PCA分析该脑区的受试者差异模式
        try:
            pca = PCA()
            pca_result = pca.fit_transform(region_features)
            pca_3pc_variance = (np.sum(pca.explained_variance_ratio_[:3]) 
                               if len(pca.explained_variance_ratio_) >= 3 
                               else np.sum(pca.explained_variance_ratio_))
        except:
            pca_3pc_variance = 0.0
        
        # 4. 受试者特异性得分（综合指标）
        # 距离相对于变异度的比值，值越大表示受试者间差异越明显
        specificity_score = mean_distance / (std_distance + 1e-8)
        
        # 5. 判别能力评估（基于距离矩阵的可分性）
        discriminability = self._compute_discriminability(distances) if n_subjects > 2 else 0.0
        
        return {
            'n_subjects': n_subjects,
            'mean_inter_subject_distance': float(mean_distance),
            'std_inter_subject_distance': float(std_distance),
            'mean_inter_subject_correlation': float(abs(mean_correlation)),
            'pca_3pc_variance': float(pca_3pc_variance),
            'subject_specificity_score': float(specificity_score),
            'discriminability': float(discriminability),
            'specificity_score': float(specificity_score),  # 主要指标
            'valid_subjects': valid_subjects
        }
    
    def _compute_discriminability(self, distance_matrix: np.ndarray) -> float:
        """计算判别能力（基于距离矩阵）"""
        n = len(distance_matrix)
        if n < 3:
            return 0.0
        
        # 计算每个样本到最近邻和次近邻的距离比
        ratios = []
        for i in range(n):
            distances = distance_matrix[i, :]
            # 排除自身（距离为0）
            sorted_distances = np.sort(distances[distances > 0])
            if len(sorted_distances) >= 2:
                ratio = sorted_distances[0] / (sorted_distances[1] + 1e-8)
                ratios.append(ratio)
        
        # 返回平均比值的倒数（值越大，判别能力越强）
        return 1.0 / (np.mean(ratios) + 1e-8) if ratios else 0.0
    
    def _identify_extreme_regions(self, region_scores: Dict[int, Dict]) -> Tuple[List[int], List[int]]:
        """识别高/低特异性脑区"""
        
        if not region_scores:
            return [], []
        
        # 提取特异性得分
        scores = [(rid, data['specificity_score']) 
                 for rid, data in region_scores.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 计算阈值
        all_scores = [s[1] for s in scores]
        high_threshold = np.percentile(all_scores, 75)
        low_threshold = np.percentile(all_scores, 25)
        
        # 识别极端脑区
        high_regions = [rid for rid, score in scores if score > high_threshold]
        low_regions = [rid for rid, score in scores if score < low_threshold]
        
        logger.info(f"  - 高特异性脑区 (>75%): {len(high_regions)} 个")
        logger.info(f"  - 低特异性脑区 (<25%): {len(low_regions)} 个")
        
        # 显示极端例子
        if high_regions:
            top_region = scores[0]
            logger.info(f"  - 最高特异性: 脑区{top_region[0]} (得分: {top_region[1]:.3f})")
        
        if low_regions:
            bottom_region = scores[-1]
            logger.info(f"  - 最低特异性: 脑区{bottom_region[0]} (得分: {bottom_region[1]:.3f})")
        
        return high_regions, low_regions
    
    def _create_specificity_matrix(self, subject_region_data: Dict[str, Any],
                                 region_scores: Dict[int, Dict]) -> Optional[Dict[int, np.ndarray]]:
        """创建特异性矩阵用于可视化"""
        
        if not region_scores:
            return None
        
        # 为每个脑区创建一个特征向量用于可视化
        specificity_matrix = {}
        
        for region_id, scores in region_scores.items():
            feature_vector = np.array([
                scores['mean_inter_subject_distance'],
                scores['std_inter_subject_distance'],
                scores['mean_inter_subject_correlation'],
                scores['pca_3pc_variance'],
                scores['specificity_score'],
                scores['discriminability']
            ])
            specificity_matrix[region_id] = feature_vector
        
        return specificity_matrix
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """打印分析摘要"""
        logger.info("\n脑区特异性分析摘要:")
        logger.info("="*50)
        
        logger.info(f"分析脑区数: {results['n_regions_analyzed']}")
        logger.info(f"有效组合数: {results['n_valid_combinations']}")
        
        if results['specificity_scores']:
            scores = results['specificity_scores']
            logger.info(f"特异性得分统计:")
            logger.info(f"  - 平均值: {np.mean(scores):.3f}")
            logger.info(f"  - 标准差: {np.std(scores):.3f}")
            logger.info(f"  - 范围: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
        
        logger.info(f"高特异性脑区数: {len(results['high_specificity_regions'])}")
        logger.info(f"低特异性脑区数: {len(results['low_specificity_regions'])}")
        
        # 建议
        mean_score = results['mean_specificity_score']
        if mean_score > 1.5:
            logger.info("\n💡 建议: 高脑区特异性表明分层Subject Embedding可能有益")
        elif len(results['high_specificity_regions']) > results['n_regions_analyzed'] * 0.3:
            logger.info("\n💡 建议: 存在较多高特异性脑区，考虑选择性Subject Embedding")