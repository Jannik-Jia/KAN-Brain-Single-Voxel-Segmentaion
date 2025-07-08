#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding需求评估器
评估每个脑区的Subject Embedding必要性
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist  # 添加缺失的导入
import torch

from common.config import Config
from common.deep_network import DeepNetworkUtils

logger = logging.getLogger(__name__)


class EmbeddingNeedAssessor:
    """Embedding需求评估器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.deep_utils = DeepNetworkUtils()
    

    def assess_region_embedding_needs(self, X: np.ndarray, regions: np.ndarray,
                                subjects: np.ndarray,
                                region_specificity: Dict[str, Any],
                                baseline_results: Dict[str, Any],
                                loso_results: Dict[str, Any],
                                region_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估每个脑区的Subject Embedding需求
        
        Args:
            X: 特征数据
            regions: 脑区标签（就是y的类别索引）
            subjects: 受试者标签
            region_specificity: Phase 1的脑区特异性结果
            baseline_results: baseline测试结果
            loso_results: LOSO评估结果
            
        Returns:
            脑区embedding需求评估结果
        """
        logger.info("开始评估分脑区Subject Embedding需求...")
        
        unique_regions = np.unique(regions)
        region_scores = {}
        region_analyses = {}
        
        # 批量处理提高效率
        batch_size = 10
        for batch_start in range(0, len(unique_regions), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_regions))
            batch_regions = unique_regions[batch_start:batch_end]
            
            logger.info(f"  处理脑区批次: {batch_start//batch_size + 1}/{(len(unique_regions) + batch_size - 1)//batch_size}")
            
            for region_id in batch_regions:
                # 分析单个脑区
                region_analysis = self._analyze_single_region(
                    region_id, X, regions, subjects,
                    region_specificity.get(str(int(region_id)), {})
                )
                
                if region_analysis['status'] == 'success':
                    # 计算embedding需求得分
                    necessity_score = self._compute_necessity_score(
                        region_analysis,
                        baseline_results,
                        loso_results,
                        region_performance  # 添加这个参数
                    )

                    
                    region_scores[int(region_id)] = necessity_score
                    region_analyses[int(region_id)] = region_analysis
        
        # 确定需求等级
        critical_regions = []
        high_priority_regions = []
        medium_priority_regions = []
        low_priority_regions = []
        
        for region_id, score in region_scores.items():
            if score > 0.8:
                critical_regions.append(region_id)
            elif score > 0.6:
                high_priority_regions.append(region_id)
            elif score > 0.4:
                medium_priority_regions.append(region_id)
            else:
                low_priority_regions.append(region_id)
        
        results = {
            'region_scores': region_scores,
            'region_analyses': region_analyses,
            'critical_regions': sorted(critical_regions),
            'high_priority_regions': sorted(high_priority_regions),
            'medium_priority_regions': sorted(medium_priority_regions),
            'low_priority_regions': sorted(low_priority_regions),
            'all_regions': sorted(list(region_scores.keys())),
            'summary': {
                'total_regions': len(region_scores),
                'critical_count': len(critical_regions),
                'high_count': len(high_priority_regions),
                'medium_count': len(medium_priority_regions),
                'low_count': len(low_priority_regions),
                'mean_necessity_score': np.mean(list(region_scores.values())) if region_scores else 0
            }
        }
        
        # 打印摘要
        self._print_assessment_summary(results)
        
        return results
    
    def _analyze_single_region(self, region_id: int, X: np.ndarray,
                             regions: np.ndarray, subjects: np.ndarray,
                             phase1_specificity: Dict) -> Dict[str, Any]:
        """分析单个脑区"""
        region_mask = regions == region_id
        
        if np.sum(region_mask) < Config.MIN_SAMPLES_FOR_ANALYSIS:
            return {
                'status': 'insufficient_data', 
                'region_id': int(region_id),
                'n_voxels': int(np.sum(region_mask))
            }
        
        region_X = X[region_mask]
        region_subjects = subjects[region_mask]
        
        # 1. 分析一致性
        consistency_analysis = self._analyze_consistency(region_X, region_subjects)
        
        # 2. 交叉受试者测试
        cross_subject_test = self._cross_subject_test(region_X, region_subjects)
        
        # 3. 受试者特异性影响
        specificity_impact = self._analyze_specificity_impact(
            region_X, region_subjects, phase1_specificity
        )
        
        # 4. 深度网络评估（简化版）
        deep_network_assessment = self._deep_network_assessment(region_X, region_subjects)
        
        return {
            'status': 'success',
            'region_id': int(region_id),
            'n_voxels': int(np.sum(region_mask)),
            'n_subjects': int(len(np.unique(region_subjects))),
            'consistency': consistency_analysis,
            'cross_subject': cross_subject_test,
            'specificity_impact': specificity_impact,
            'deep_network': deep_network_assessment,
            'phase1_specificity_score': float(phase1_specificity.get('specificity_score', 0))
        }
    
    def evaluate_region_classification_performance(self, X_train: np.ndarray, y_train: np.ndarray,
                                             subjects_train: np.ndarray,
                                             X_val: np.ndarray, y_val: np.ndarray,
                                             subjects_val: np.ndarray) -> Dict[str, Any]:
        """
        评估分脑区分类在验证集上的性能
        
        Args:
            X_train, y_train, subjects_train: 训练数据
            X_val, y_val, subjects_val: 验证数据
            
        Returns:
            包含F1分数的性能评估结果
        """
        logger.info("评估分脑区Subject-specific分类性能...")
        
        # 将y转换为脑区标签（如果是one-hot编码）
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            regions_train = np.argmax(y_train, axis=1)
            regions_val = np.argmax(y_val, axis=1)
        else:
            regions_train = y_train.flatten().astype(int)
            regions_val = y_val.flatten().astype(int)
        
        unique_regions = np.unique(regions_train)
        region_performances = {}
        
        # 对每个脑区评估性能
        for region_id in unique_regions:
            # 该脑区的训练和验证数据
            train_mask = regions_train == region_id
            val_mask = regions_val == region_id
            
            if np.sum(train_mask) < Config.MIN_SAMPLES_FOR_ANALYSIS or np.sum(val_mask) < 10:
                continue
            
            # 使用简单的逻辑回归评估受试者分类性能
            X_region_train = X_train[train_mask]
            subjects_region_train = subjects_train[train_mask]
            X_region_val = X_val[val_mask]
            subjects_region_val = subjects_val[val_mask]
            
            # 确保验证集有足够的受试者
            unique_val_subjects = np.unique(subjects_region_val)
            if len(unique_val_subjects) < 2:
                continue
            
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import f1_score, accuracy_score
                
                # 训练受试者分类器
                clf = LogisticRegression(max_iter=500, C=0.1, random_state=42)
                clf.fit(X_region_train, subjects_region_train)
                
                # 在验证集上预测
                y_pred = clf.predict(X_region_val)
                
                # 计算指标
                accuracy = accuracy_score(subjects_region_val, y_pred)
                f1_macro = f1_score(subjects_region_val, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(subjects_region_val, y_pred, average='weighted', zero_division=0)
                
                region_performances[int(region_id)] = {
                    'accuracy': float(accuracy),
                    'f1_macro': float(f1_macro),
                    'f1_weighted': float(f1_weighted),
                    'n_train_samples': int(np.sum(train_mask)),
                    'n_val_samples': int(np.sum(val_mask)),
                    'n_unique_subjects_train': int(len(np.unique(subjects_region_train))),
                    'n_unique_subjects_val': int(len(unique_val_subjects))
                }
                
            except Exception as e:
                logger.warning(f"脑区 {region_id} 分类评估失败: {e}")
                continue
        
        # 计算整体性能
        if region_performances:
            mean_f1_macro = np.mean([r['f1_macro'] for r in region_performances.values()])
            mean_accuracy = np.mean([r['accuracy'] for r in region_performances.values()])
            
            return {
                'region_performances': region_performances,
                'overall_mean_f1_macro': float(mean_f1_macro),
                'overall_mean_accuracy': float(mean_accuracy),
                'n_evaluated_regions': len(region_performances),
                'total_regions': len(unique_regions)
            }
        else:
            return {
                'region_performances': {},
                'overall_mean_f1_macro': 0.0,
                'overall_mean_accuracy': 0.0,
                'n_evaluated_regions': 0,
                'total_regions': len(unique_regions)
            }
    
    
    def _analyze_consistency(self, X: np.ndarray, subjects: np.ndarray) -> Dict[str, float]:
        """分析脑区内受试者一致性"""
        unique_subjects = np.unique(subjects)
        
        if len(unique_subjects) < Config.MIN_SUBJECTS_FOR_ANALYSIS:
            return {
                'consistency_score': 0.5, 
                'n_subjects': int(len(unique_subjects)),
                'insufficient_subjects': True
            }
        
        # 计算每个受试者的平均特征
        subject_means = []
        for subject_id in unique_subjects:
            subject_mask = subjects == subject_id
            if np.sum(subject_mask) >= 10:
                subject_means.append(np.mean(X[subject_mask], axis=0))
        
        if len(subject_means) < 3:
            return {
                'consistency_score': 0.5, 
                'n_subjects': int(len(subject_means)),
                'insufficient_samples': True
            }
        
        subject_means = np.array(subject_means)
        
        # 计算变异系数
        with np.errstate(divide='ignore', invalid='ignore'):
            feature_cv = np.std(subject_means, axis=0) / (np.abs(np.mean(subject_means, axis=0)) + 1e-8)
            feature_cv = np.nan_to_num(feature_cv, nan=0.0, posinf=1.0, neginf=0.0)
        
        mean_cv = np.mean(feature_cv)
        
        # 转换为一致性得分（CV越低，一致性越高）
        consistency_score = 1.0 / (1.0 + mean_cv)
        
        return {
            'consistency_score': float(consistency_score),
            'mean_cv': float(mean_cv),
            'n_subjects': int(len(subject_means))
        }
    
    def _cross_subject_test(self, X: np.ndarray, subjects: np.ndarray) -> Dict[str, float]:
        """交叉受试者测试"""
        unique_subjects = np.unique(subjects)
        
        if len(unique_subjects) < 4:
            return {
                'generalization_quality': 0.5, 
                'tested': False,
                'reason': 'insufficient_subjects'
            }
        
        # 简单的留一受试者验证
        accuracies = []
        max_tests = min(5, len(unique_subjects))  # 最多测试5个受试者
        
        for i, test_subject in enumerate(unique_subjects[:max_tests]):
            train_mask = subjects != test_subject
            test_mask = subjects == test_subject
            
            if np.sum(test_mask) < 5 or np.sum(train_mask) < 20:
                continue
            
            # 创建二分类问题：该受试者 vs 其他
            X_train = X[train_mask]
            y_train = (subjects[train_mask] == test_subject).astype(int)
            X_test = X[test_mask]
            
            # 简单分类器
            clf = LogisticRegression(max_iter=100, random_state=42)
            
            try:
                # 创建平衡的训练集
                pos_indices = np.where(y_train == 1)[0]
                neg_indices = np.where(y_train == 0)[0]
                
                if len(pos_indices) == 0:
                    # 为其他受试者创建正样本
                    other_subject = unique_subjects[unique_subjects != test_subject][0]
                    y_train = (subjects[train_mask] == other_subject).astype(int)
                    pos_indices = np.where(y_train == 1)[0]
                    neg_indices = np.where(y_train == 0)[0]
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    n_samples = min(len(pos_indices), len(neg_indices), 50)
                    balanced_indices = np.concatenate([
                        np.random.choice(pos_indices, n_samples, replace=False),
                        np.random.choice(neg_indices, n_samples, replace=False)
                    ])
                    
                    clf.fit(X_train[balanced_indices], y_train[balanced_indices])
                    
                    # 预测测试受试者
                    y_pred = clf.predict(X_test)
                    accuracy = np.mean(y_pred == 1)  # 应该都预测为该受试者
                    accuracies.append(accuracy)
                    
            except Exception as e:
                logger.warning(f"交叉受试者测试失败: {e}")
                continue
        
        if accuracies:
            generalization_quality = 1.0 - np.mean(accuracies)  # 越难识别，泛化越好
        else:
            generalization_quality = 0.5
        
        return {
            'generalization_quality': float(generalization_quality),
            'tested': len(accuracies) > 0,
            'n_tests': int(len(accuracies))
        }
    
    def _analyze_specificity_impact(self, X: np.ndarray, subjects: np.ndarray,
                                  phase1_specificity: Dict) -> Dict[str, float]:
        """分析受试者特异性影响"""
        # 使用Phase 1的特异性得分
        phase1_score = phase1_specificity.get('specificity_score', 0.5)
        
        # 计算受试者间距离的变异性
        unique_subjects = np.unique(subjects)
        if len(unique_subjects) >= 3:
            subject_means = []
            for subject_id in unique_subjects:
                subject_mask = subjects == subject_id
                if np.sum(subject_mask) >= 10:
                    subject_means.append(np.mean(X[subject_mask], axis=0))
            
            if len(subject_means) >= 3:
                distances = pdist(np.array(subject_means))
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    distance_cv = np.std(distances) / (np.mean(distances) + 1e-8)
                    distance_cv = np.nan_to_num(distance_cv, nan=0.0, posinf=1.0, neginf=0.0)
                
                # 高变异性意味着某些受试者很特殊
                specificity_impact = min(1.0, distance_cv)
            else:
                specificity_impact = phase1_score
        else:
            specificity_impact = phase1_score
        
        return {
            'specificity_impact': float(specificity_impact),
            'embedding_benefit_potential': float(specificity_impact),
            'phase1_score': float(phase1_score)
        }
    
    def _deep_network_assessment(self, X: np.ndarray, subjects: np.ndarray) -> Dict[str, Any]:
        """深度网络评估（简化版）"""
        unique_subjects = np.unique(subjects)
        
        if len(unique_subjects) < 5 or len(X) < 500:
            return {
                'tested': False, 
                'discrimination_strength': 1.0,
                'reason': 'insufficient_data'
            }
        
        # 这里可以添加深度网络的快速测试
        # 为了效率，Phase 2中可以跳过或使用简化版本
        
        return {
            'tested': False,
            'discrimination_strength': 1.0,
            'reason': 'Simplified assessment in Phase 2'
        }
    
    def _compute_necessity_score(self, region_analysis: Dict[str, Any],
                            baseline_results: Dict[str, Any],
                            loso_results: Dict[str, Any],
                            region_performance: Optional[Dict[str, Any]] = None) -> float:


        """计算embedding必要性得分"""
        # 基础权重
        # weights = {
        #     'consistency': 0.3,
        #     'generalization': 0.3,
        #     'specificity': 0.2,
        #     'baseline_gap': 0.2
        # }
        weights = {
            'consistency': 0.25,      # 降低一点
            'generalization': 0.25,   # 降低一点
            'specificity': 0.2,
            'baseline_gap': 0.15,     # 降低一点
            'classification_difficulty': 0.15  # 新增
        }


        
        # 1. 一致性得分（越不一致越需要embedding）
        consistency_score = 1.0 - region_analysis['consistency']['consistency_score']
        
        # 2. 泛化得分（泛化越差越需要embedding）
        generalization_score = 1.0 - region_analysis['cross_subject']['generalization_quality']
        
        # 3. 特异性得分
        specificity_score = region_analysis['specificity_impact']['specificity_impact']
        
        # 4. Baseline差距得分
        baseline_gap = loso_results.get('mean_generalization_gap', 0.1)
        baseline_gap_score = min(1.0, baseline_gap * 2)  # 放大差距影响

        # 5. 分类难度得分（新增）
        classification_difficulty_score = 0.5  # 默认中等
        if region_performance and region_analysis['region_id'] in region_performance:
            region_f1 = region_performance[region_analysis['region_id']].get('f1_macro', 0)
            # F1越低，分类越困难，越需要embedding
            classification_difficulty_score = 1.0 - region_f1
        
        # 综合得分
        necessity_score = (
            consistency_score * weights['consistency'] +
            generalization_score * weights['generalization'] +
            specificity_score * weights['specificity'] +
            baseline_gap_score * weights['baseline_gap'] +
            classification_difficulty_score * weights['classification_difficulty']
        )


        
        # 根据Phase 1的特异性得分进行调整
        phase1_boost = region_analysis.get('phase1_specificity_score', 0) * 0.1
        necessity_score = min(1.0, necessity_score + phase1_boost)
        
        # 如果该脑区在LOSO中表现特别差，增加权重
        if 'region_wise_details' in loso_results:
            region_id = region_analysis['region_id']
            region_loso = loso_results['region_wise_details'].get(region_id, {})
            
            if 'generalization_score' in region_loso:
                poor_generalization = 1.0 - region_loso['generalization_score']
                necessity_score = min(1.0, necessity_score + poor_generalization * 0.1)
        
        return float(necessity_score)
    
    def _print_assessment_summary(self, results: Dict):
        """打印评估摘要"""
        summary = results['summary']
        
        logger.info("\nSubject Embedding需求评估摘要:")
        logger.info("="*50)
        logger.info(f"总脑区数: {summary['total_regions']}")
        logger.info(f"平均需求得分: {summary['mean_necessity_score']:.3f}")
        logger.info(f"\n需求等级分布:")
        logger.info(f"  - CRITICAL (>0.8): {summary['critical_count']} 个脑区")
        logger.info(f"  - HIGH (0.6-0.8): {summary['high_count']} 个脑区")
        logger.info(f"  - MEDIUM (0.4-0.6): {summary['medium_count']} 个脑区")
        logger.info(f"  - LOW (<0.4): {summary['low_count']} 个脑区")
        
        # 显示前5个最需要embedding的脑区
        if results['critical_regions']:
            logger.info(f"\n最需要Subject Embedding的脑区 (前5个):")
            top_regions = sorted(results['critical_regions'], 
                               key=lambda x: results['region_scores'][x], 
                               reverse=True)[:5]
            for region_id in top_regions:
                score = results['region_scores'][region_id]
                logger.info(f"  - 脑区 {region_id}: 得分 {score:.3f}")