#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证器
验证数据的完整性和质量，包含脑区统计分析
"""

import numpy as np
import logging
from typing import Dict, List, Any, Union
from collections import Counter

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.validation_results = {}
    
    def validate_split_data(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        验证分割后的数据
        
        Args:
            split_data: 分割后的数据字典
            
        Returns:
            验证报告
        """
        logger.info("开始数据验证...")
        
        validation_report = {
            'label_analysis': self._validate_labels(split_data),
            'feature_analysis': self._validate_features(split_data),
            'subject_analysis': self._validate_subjects(split_data),
            'balance_analysis': self._validate_balance(split_data),
            'brain_region_analysis': self._analyze_brain_regions(split_data),  # 新增
            'overall_status': {}
        }
        
        # 计算总体评分
        validation_report['overall_status'] = self._compute_overall_status(validation_report)
        
        # 打印验证摘要
        self._print_validation_summary(validation_report)
        
        return validation_report
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {self._convert_numpy_types(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, Counter):
            return dict(Counter({self._convert_numpy_types(k): v for k, v in obj.items()}))
        return obj
    
    def _validate_labels(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """验证标签完整性"""
        logger.info("验证标签...")
        
        y_train = split_data['y_train']
        y_val = split_data['y_val']
        y_test = split_data['y_test']
        
        # 检查是否为one-hot编码
        is_one_hot = len(y_train.shape) > 1 and y_train.shape[1] > 1
        
        if is_one_hot:
            # 转换为类别索引
            y_train_classes = np.argmax(y_train, axis=1)
            y_val_classes = np.argmax(y_val, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            one_hot_dim = y_train.shape[1]
        else:
            y_train_classes = y_train.flatten()
            y_val_classes = y_val.flatten()
            y_test_classes = y_test.flatten()
            one_hot_dim = int(np.max(y_train_classes)) + 1
        
        # 统计各数据集的类别分布
        train_classes = np.unique(y_train_classes)
        val_classes = np.unique(y_val_classes)
        test_classes = np.unique(y_test_classes)
        
        all_classes = np.unique(np.concatenate([train_classes, val_classes, test_classes]))
        
        # 检查缺失的类别
        expected_classes = set(range(one_hot_dim))
        actual_classes = set(all_classes.astype(int))
        missing_classes = sorted(expected_classes - actual_classes)
        
        # 统计每个类别的样本数 - 使用原生Python类型
        train_class_counts = {int(k): int(v) for k, v in Counter(y_train_classes.astype(int)).items()}
        val_class_counts = {int(k): int(v) for k, v in Counter(y_val_classes.astype(int)).items()}
        test_class_counts = {int(k): int(v) for k, v in Counter(y_test_classes.astype(int)).items()}
        
        label_analysis = {
            'is_one_hot': bool(is_one_hot),
            'one_hot_dim': int(one_hot_dim),
            'n_classes_expected': int(one_hot_dim),
            'n_classes_actual': int(len(all_classes)),
            'missing_classes': [int(x) for x in missing_classes],
            'train_classes': [int(x) for x in sorted(train_classes)],
            'val_classes': [int(x) for x in sorted(val_classes)],
            'test_classes': [int(x) for x in sorted(test_classes)],
            'class_distribution': {
                'train': train_class_counts,
                'val': val_class_counts,
                'test': test_class_counts
            }
        }
        
        if missing_classes:
            logger.warning(f"发现缺失的类别: {missing_classes[:10]}...")
            logger.warning(f"总共缺失 {len(missing_classes)} 个类别")
        
        return self._convert_numpy_types(label_analysis)
    
    def _validate_features(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """验证特征质量"""
        logger.info("验证特征...")
        
        X_train = split_data['X_train']
        X_train_scaled = split_data['X_train_scaled']
        
        # 检查NaN和Inf
        nan_count = np.sum(np.isnan(X_train))
        inf_count = np.sum(np.isinf(X_train))
        
        # 检查常数特征
        feature_vars = np.var(X_train, axis=0)
        constant_features = np.where(feature_vars < 1e-10)[0]
        
        # 检查标准化效果
        scaled_mean = np.mean(X_train_scaled, axis=0)
        scaled_std = np.std(X_train_scaled, axis=0)
        
        # 特征组统计
        from common.config import Config
        feature_groups = Config.get_feature_groups()
        feature_group_stats = {}
        
        for group_name, indices in feature_groups.items():
            group_features = X_train[:, indices]
            feature_group_stats[group_name] = {
                'n_features': len(indices),
                'mean_value': float(np.mean(group_features)),
                'std_value': float(np.std(group_features)),
                'min_value': float(np.min(group_features)),
                'max_value': float(np.max(group_features)),
                'n_constant': int(np.sum(np.isin(constant_features, indices)))
            }
        
        feature_analysis = {
            'n_features': int(X_train.shape[1]),
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'constant_features': [int(x) for x in constant_features],
            'n_constant_features': int(len(constant_features)),
            'feature_value_range': {
                'min': float(np.min(X_train)),
                'max': float(np.max(X_train)),
                'mean': float(np.mean(X_train)),
                'std': float(np.std(X_train))
            },
            'scaled_stats': {
                'mean_range': [float(np.min(scaled_mean)), float(np.max(scaled_mean))],
                'std_range': [float(np.min(scaled_std)), float(np.max(scaled_std))]
            },
            'feature_group_stats': feature_group_stats
        }
        
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"发现异常值: {nan_count} 个NaN, {inf_count} 个Inf")
        
        if len(constant_features) > 0:
            logger.warning(f"发现 {len(constant_features)} 个常数特征")
        
        return self._convert_numpy_types(feature_analysis)
    
    def _validate_subjects(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """验证受试者分布"""
        logger.info("验证受试者分布...")
        
        train_subjects = split_data['subjects_train']
        val_subjects = split_data['subjects_val']
        test_subjects = split_data['subjects_test']
        
        # 检查受试者是否有重叠
        train_unique = set(train_subjects.astype(int))
        val_unique = set(val_subjects.astype(int))
        test_unique = set(test_subjects.astype(int))
        
        train_val_overlap = train_unique & val_unique
        train_test_overlap = train_unique & test_unique
        val_test_overlap = val_unique & test_unique
        
        # 检查受试者ID连续性
        all_subject_ids = sorted(train_unique | val_unique | test_unique)
        expected_ids = list(range(min(all_subject_ids), max(all_subject_ids) + 1))
        missing_ids = sorted(set(expected_ids) - set(all_subject_ids))
        
        # 统计每个受试者的样本数
        all_subjects = np.concatenate([train_subjects, val_subjects, test_subjects])
        subject_counts = Counter(all_subjects.astype(int))
        
        # 详细的受试者样本统计
        subject_sample_stats = {
            'train': {int(s): int(np.sum(train_subjects == s)) for s in train_unique},
            'val': {int(s): int(np.sum(val_subjects == s)) for s in val_unique},
            'test': {int(s): int(np.sum(test_subjects == s)) for s in test_unique}
        }
        
        subject_analysis = {
            'n_train_subjects': len(train_unique),
            'n_val_subjects': len(val_unique),
            'n_test_subjects': len(test_unique),
            'train_subject_ids': sorted([int(x) for x in train_unique]),
            'val_subject_ids': sorted([int(x) for x in val_unique]),
            'test_subject_ids': sorted([int(x) for x in test_unique]),
            'subject_overlap': {
                'train_val': len(train_val_overlap),
                'train_test': len(train_test_overlap),
                'val_test': len(val_test_overlap)
            },
            'subject_id_continuity': {
                'all_ids': all_subject_ids,
                'missing_ids': missing_ids,
                'is_continuous': len(missing_ids) == 0
            },
            'samples_per_subject': {
                'mean': float(np.mean(list(subject_counts.values()))),
                'std': float(np.std(list(subject_counts.values()))),
                'min': int(min(subject_counts.values())),
                'max': int(max(subject_counts.values()))
            },
            'subject_sample_distribution': subject_sample_stats
        }
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.error("发现受试者重叠!")
            logger.error(f"训练-验证重叠: {train_val_overlap}")
            logger.error(f"训练-测试重叠: {train_test_overlap}")
            logger.error(f"验证-测试重叠: {val_test_overlap}")
        
        if missing_ids:
            logger.warning(f"受试者ID不连续，缺失: {missing_ids}")
        
        return self._convert_numpy_types(subject_analysis)
    
    def _analyze_brain_regions(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """分析脑区分布和统计信息"""
        logger.info("分析脑区分布...")
        
        # 获取脑区标签
        y_train = split_data['y_train']
        y_val = split_data['y_val']
        y_test = split_data['y_test']
        
        # 转换为脑区ID
        is_one_hot = len(y_train.shape) > 1 and y_train.shape[1] > 1
        if is_one_hot:
            regions_train = np.argmax(y_train, axis=1)
            regions_val = np.argmax(y_val, axis=1)
            regions_test = np.argmax(y_test, axis=1)
        else:
            regions_train = y_train.flatten().astype(int)
            regions_val = y_val.flatten().astype(int)
            regions_test = y_test.flatten().astype(int)
        
        # 统计每个脑区的样本数
        region_counts_train = Counter(regions_train.astype(int))
        region_counts_val = Counter(regions_val.astype(int))
        region_counts_test = Counter(regions_test.astype(int))
        
        # 计算每个脑区在不同受试者中的分布
        region_subject_distribution = {}
        subjects_train = split_data['subjects_train']
        subjects_val = split_data['subjects_val']
        subjects_test = split_data['subjects_test']
        
        all_regions = sorted(set(regions_train) | set(regions_val) | set(regions_test))
        
        for region_id in all_regions:
            # 训练集中的分布
            train_mask = regions_train == region_id
            subjects_in_region_train = subjects_train[train_mask] if np.any(train_mask) else np.array([])
            
            # 验证集中的分布
            val_mask = regions_val == region_id
            subjects_in_region_val = subjects_val[val_mask] if np.any(val_mask) else np.array([])
            
            # 测试集中的分布
            test_mask = regions_test == region_id
            subjects_in_region_test = subjects_test[test_mask] if np.any(test_mask) else np.array([])
            
            # 合并所有数据集的信息
            all_subjects_in_region = np.concatenate([
                subjects_in_region_train,
                subjects_in_region_val,
                subjects_in_region_test
            ])
            
            if len(all_subjects_in_region) > 0:
                unique_subjects = np.unique(all_subjects_in_region)
                subject_sample_counts = {
                    int(s): int(np.sum(all_subjects_in_region == s)) 
                    for s in unique_subjects
                }
            else:
                unique_subjects = np.array([])
                subject_sample_counts = {}
            
            region_subject_distribution[int(region_id)] = {
                'n_subjects_total': len(unique_subjects),
                'n_subjects_train': len(np.unique(subjects_in_region_train)) if len(subjects_in_region_train) > 0 else 0,
                'n_subjects_val': len(np.unique(subjects_in_region_val)) if len(subjects_in_region_val) > 0 else 0,
                'n_subjects_test': len(np.unique(subjects_in_region_test)) if len(subjects_in_region_test) > 0 else 0,
                'n_samples_train': int(np.sum(train_mask)),
                'n_samples_val': int(np.sum(val_mask)),
                'n_samples_test': int(np.sum(test_mask)),
                'n_samples_total': len(all_subjects_in_region),
                'samples_per_subject': subject_sample_counts,
                'subject_coverage': float(len(unique_subjects) / len(np.unique(np.concatenate([subjects_train, subjects_val, subjects_test]))))
            }
        
        # 识别小脑区和大脑区
        region_sizes = {rid: info['n_samples_total'] for rid, info in region_subject_distribution.items()}
        size_threshold_small = np.percentile(list(region_sizes.values()), 10)
        size_threshold_large = np.percentile(list(region_sizes.values()), 90)
        
        small_regions = [rid for rid, size in region_sizes.items() if size <= size_threshold_small]
        large_regions = [rid for rid, size in region_sizes.items() if size >= size_threshold_large]
        
        # 计算脑区平衡性
        region_balance_scores = {}
        for region_id, info in region_subject_distribution.items():
            n_train = info['n_samples_train']
            n_val = info['n_samples_val']
            n_test = info['n_samples_test']
            n_total = info['n_samples_total']
            
            if n_total > 0:
                # 理想比例：训练70%，验证20%，测试10%
                ideal_train = 0.7 * n_total
                ideal_val = 0.2 * n_total
                ideal_test = 0.1 * n_total
                
                balance_score = 1.0 - (
                    abs(n_train - ideal_train) / n_total * 0.5 +
                    abs(n_val - ideal_val) / n_total * 0.3 +
                    abs(n_test - ideal_test) / n_total * 0.2
                )
                region_balance_scores[region_id] = float(max(0, balance_score))
            else:
                region_balance_scores[region_id] = 0.0
        
        brain_region_analysis = {
            'n_regions_total': len(all_regions),
            'n_regions_with_samples': len([r for r, info in region_subject_distribution.items() if info['n_samples_total'] > 0]),
            'region_counts': {
                'train': {int(k): int(v) for k, v in region_counts_train.items()},
                'val': {int(k): int(v) for k, v in region_counts_val.items()},
                'test': {int(k): int(v) for k, v in region_counts_test.items()}
            },
            'region_subject_distribution': region_subject_distribution,
            'region_size_analysis': {
                'small_regions': small_regions,
                'large_regions': large_regions,
                'size_threshold_small': float(size_threshold_small),
                'size_threshold_large': float(size_threshold_large),
                'mean_region_size': float(np.mean(list(region_sizes.values()))),
                'std_region_size': float(np.std(list(region_sizes.values())))
            },
            'region_balance_scores': region_balance_scores,
            'poorly_balanced_regions': [
                rid for rid, score in region_balance_scores.items() if score < 0.5
            ],
            'well_balanced_regions': [
                rid for rid, score in region_balance_scores.items() if score > 0.8
            ]
        }
        
        # 打印一些关键统计
        logger.info(f"脑区总数: {brain_region_analysis['n_regions_total']}")
        logger.info(f"有样本的脑区数: {brain_region_analysis['n_regions_with_samples']}")
        logger.info(f"小脑区数量: {len(small_regions)}")
        logger.info(f"大脑区数量: {len(large_regions)}")
        logger.info(f"平衡性差的脑区数: {len(brain_region_analysis['poorly_balanced_regions'])}")
        
        return self._convert_numpy_types(brain_region_analysis)
    
    def _validate_balance(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """验证数据平衡性"""
        logger.info("验证数据平衡性...")
        
        # 样本数量平衡
        n_train = len(split_data['X_train'])
        n_val = len(split_data['X_val'])
        n_test = len(split_data['X_test'])
        n_total = n_train + n_val + n_test
        
        balance_analysis = {
            'sample_distribution': {
                'train': n_train,
                'val': n_val,
                'test': n_test,
                'total': n_total,
                'train_ratio': float(n_train / n_total),
                'val_ratio': float(n_val / n_total),
                'test_ratio': float(n_test / n_total)
            }
        }
        
        return self._convert_numpy_types(balance_analysis)
    
    def _compute_overall_status(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体验证状态"""
        
        # 数据完整性评分
        label_analysis = validation_report['label_analysis']
        completeness_score = (label_analysis['n_classes_actual'] / 
                            label_analysis['n_classes_expected'])
        
        # 特征质量评分
        feature_analysis = validation_report['feature_analysis']
        feature_score = 1.0
        if feature_analysis['nan_count'] > 0 or feature_analysis['inf_count'] > 0:
            feature_score -= 0.5
        if feature_analysis['n_constant_features'] > 0:
            feature_score -= 0.1 * min(1.0, feature_analysis['n_constant_features'] / feature_analysis['n_features'])
        
        # 受试者分离评分
        subject_analysis = validation_report['subject_analysis']
        separation_score = 1.0
        if (subject_analysis['subject_overlap']['train_val'] > 0 or
            subject_analysis['subject_overlap']['train_test'] > 0 or
            subject_analysis['subject_overlap']['val_test'] > 0):
            separation_score = 0.0
        
        # 数据平衡评分
        balance_analysis = validation_report['balance_analysis']
        train_ratio = balance_analysis['sample_distribution']['train_ratio']
        balance_score = 1.0 - abs(train_ratio - 0.7)  # 理想训练集比例为70%
        
        # 脑区覆盖评分
        brain_region_analysis = validation_report['brain_region_analysis']
        region_coverage_score = (brain_region_analysis['n_regions_with_samples'] / 
                               brain_region_analysis['n_regions_total'])
        
        # 总体状态
        overall_score = (completeness_score * 0.25 + 
                        feature_score * 0.15 + 
                        separation_score * 0.25 + 
                        balance_score * 0.15 +
                        region_coverage_score * 0.20)
        
        overall_status = {
            'completeness_score': float(completeness_score),
            'feature_quality_score': float(feature_score),
            'subject_separation_score': float(separation_score),
            'balance_score': float(balance_score),
            'region_coverage_score': float(region_coverage_score),
            'overall_score': float(overall_score),
            'status': 'PASS' if overall_score > 0.8 else 'WARNING' if overall_score > 0.6 else 'FAIL',
            'label_coverage': float(completeness_score),
            'has_data_issues': feature_analysis['nan_count'] > 0 or feature_analysis['inf_count'] > 0,
            'has_overlap_issues': separation_score < 1.0,
            'has_region_balance_issues': len(brain_region_analysis['poorly_balanced_regions']) > 5
        }
        
        return self._convert_numpy_types(overall_status)
    
    def _print_validation_summary(self, validation_report: Dict[str, Any]):
        """打印验证摘要"""
        logger.info("\n" + "="*60)
        logger.info("数据验证摘要")
        logger.info("="*60)
        
        overall = validation_report['overall_status']
        
        logger.info(f"总体状态: {overall['status']}")
        logger.info(f"总体评分: {overall['overall_score']:.3f}")
        logger.info(f"  - 数据完整性: {overall['completeness_score']:.3f}")
        logger.info(f"  - 特征质量: {overall['feature_quality_score']:.3f}")
        logger.info(f"  - 受试者分离: {overall['subject_separation_score']:.3f}")
        logger.info(f"  - 数据平衡: {overall['balance_score']:.3f}")
        logger.info(f"  - 脑区覆盖: {overall['region_coverage_score']:.3f}")
        
        # 警告信息
        warnings = []
        label_analysis = validation_report['label_analysis']
        if label_analysis['missing_classes']:
            warnings.append(f"缺失 {len(label_analysis['missing_classes'])} 个类别")
        
        feature_analysis = validation_report['feature_analysis']
        if feature_analysis['nan_count'] > 0:
            warnings.append(f"发现 {feature_analysis['nan_count']} 个NaN值")
        if feature_analysis['inf_count'] > 0:
            warnings.append(f"发现 {feature_analysis['inf_count']} 个无穷值")
        if feature_analysis['n_constant_features'] > 0:
            warnings.append(f"发现 {feature_analysis['n_constant_features']} 个常数特征")
        
        if overall['has_overlap_issues']:
            warnings.append("存在受试者重叠问题")
        
        if overall['has_region_balance_issues']:
            warnings.append("多个脑区存在样本平衡问题")
        
        if warnings:
            logger.warning("\n警告:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("\n✅ 未发现数据质量问题")
    
    def extract_label_mapping(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """提取标签映射信息"""
        y_train = split_data['y_train']
        
        # 检查是否为one-hot编码
        is_one_hot = len(y_train.shape) > 1 and y_train.shape[1] > 1
        
        if is_one_hot:
            one_hot_dim = y_train.shape[1]
            y_train_classes = np.argmax(y_train, axis=1)
        else:
            y_train_classes = y_train.flatten()
            one_hot_dim = int(np.max(y_train_classes)) + 1
        
        # 收集所有数据集中的类别
        all_y = []
        for key in ['y_train', 'y_val', 'y_test']:
            if key in split_data:
                y = split_data[key]
                if is_one_hot:
                    all_y.extend(np.argmax(y, axis=1))
                else:
                    all_y.extend(y.flatten())
        
        unique_classes = sorted(np.unique(all_y).astype(int).tolist())
        
        # 检查缺失的类别
        expected_classes = list(range(one_hot_dim))
        missing_classes = sorted(set(expected_classes) - set(unique_classes))
        
        # 为每个脑区生成默认名称（可以后续替换为真实名称）
        region_names = {i: f"Region_{i:03d}" for i in range(one_hot_dim)}
        
        label_mapping = {
            'is_one_hot': bool(is_one_hot),
            'one_hot_dim': int(one_hot_dim),
            'min_label': int(np.min(unique_classes)),
            'max_label': int(np.max(unique_classes)),
            'n_classes_expected': int(one_hot_dim),
            'n_classes_actual': len(unique_classes),
            'unique_classes': unique_classes,
            'missing_classes': [int(x) for x in missing_classes],
            'label_type': 'brain_region',
            'label_description': '脑区标签，每个标签代表一个特定的脑区',
            'region_names': region_names
        }
        
        if missing_classes:
            label_mapping['missing_classes_info'] = {
                'count': len(missing_classes),
                'description': '这些脑区在数据中没有样本，可能是成像质量问题或脑区太小',
                'recommendation': '模型输出保持完整维度，但这些类别的预测可能不可靠'
            }
        
        return self._convert_numpy_types(label_mapping)
    
    def create_phase1_index(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """创建Phase 1快速访问索引"""
        from common.config import Config
        
        index = {
            'subject_ranges': {
                'train': [int(np.min(split_data['subjects_train'])), 
                         int(np.max(split_data['subjects_train']))],
                'val': [int(np.min(split_data['subjects_val'])), 
                       int(np.max(split_data['subjects_val']))],
                'test': [int(split_data['test_subject_id']), 
                        int(split_data['test_subject_id'])]
            },
            'quick_stats': {
                'total_samples': len(split_data['X_train']) + len(split_data['X_val']) + len(split_data['X_test']),
                'total_subjects': len(np.unique(np.concatenate([
                    split_data['subjects_train'],
                    split_data['subjects_val'],
                    split_data['subjects_test']
                ]))),
                'features_per_group': {name: len(indices) for name, indices in Config.get_feature_groups().items()},
                'feature_groups': Config.get_feature_groups()
            },
            'data_files': {
                'train': str(Config.TRAIN_DATA_FILE),
                'val': str(Config.VAL_DATA_FILE),
                'test': str(Config.TEST_DATA_FILE),
                'scaler': str(Config.SCALER_FILE)
            }
        }
        
        return self._convert_numpy_types(index)