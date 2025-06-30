#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证器
验证数据的完整性和质量
"""

import numpy as np
import logging
from typing import Dict, List, Any
from collections import Counter

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
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
            'overall_status': {}
        }
        
        # 计算总体评分
        validation_report['overall_status'] = self._compute_overall_status(validation_report)
        
        # 打印验证摘要
        self._print_validation_summary(validation_report)
        
        return validation_report
    
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
        
        # 统计每个类别的样本数
        train_class_counts = Counter(y_train_classes)
        val_class_counts = Counter(y_val_classes)
        test_class_counts = Counter(y_test_classes)
        
        label_analysis = {
            'is_one_hot': is_one_hot,
            'one_hot_dim': one_hot_dim,
            'n_classes_expected': one_hot_dim,
            'n_classes_actual': len(all_classes),
            'missing_classes': missing_classes,
            'train_classes': sorted(train_classes.tolist()),
            'val_classes': sorted(val_classes.tolist()),
            'test_classes': sorted(test_classes.tolist()),
            'class_distribution': {
                'train': dict(train_class_counts),
                'val': dict(val_class_counts),
                'test': dict(test_class_counts)
            }
        }
        
        if missing_classes:
            logger.warning(f"发现缺失的类别: {missing_classes[:10]}...")
            logger.warning(f"总共缺失 {len(missing_classes)} 个类别")
        
        return label_analysis
    
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
        
        feature_analysis = {
            'n_features': X_train.shape[1],
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'constant_features': constant_features.tolist(),
            'n_constant_features': len(constant_features),
            'feature_value_range': {
                'min': float(np.min(X_train)),
                'max': float(np.max(X_train)),
                'mean': float(np.mean(X_train)),
                'std': float(np.std(X_train))
            },
            'scaled_stats': {
                'mean_range': [float(np.min(scaled_mean)), float(np.max(scaled_mean))],
                'std_range': [float(np.min(scaled_std)), float(np.max(scaled_std))]
            }
        }
        
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"发现异常值: {nan_count} 个NaN, {inf_count} 个Inf")
        
        if len(constant_features) > 0:
            logger.warning(f"发现 {len(constant_features)} 个常数特征")
        
        return feature_analysis
    
    def _validate_subjects(self, split_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """验证受试者分布"""
        logger.info("验证受试者分布...")
        
        train_subjects = split_data['subjects_train']
        val_subjects = split_data['subjects_val']
        test_subjects = split_data['subjects_test']
        
        # 检查受试者是否有重叠
        train_unique = set(train_subjects)
        val_unique = set(val_subjects)
        test_unique = set(test_subjects)
        
        train_val_overlap = train_unique & val_unique
        train_test_overlap = train_unique & test_unique
        val_test_overlap = val_unique & test_unique
        
        # 统计每个受试者的样本数
        all_subjects = np.concatenate([train_subjects, val_subjects, test_subjects])
        subject_counts = Counter(all_subjects)
        
        subject_analysis = {
            'n_train_subjects': len(train_unique),
            'n_val_subjects': len(val_unique),
            'n_test_subjects': len(test_unique),
            'subject_overlap': {
                'train_val': len(train_val_overlap),
                'train_test': len(train_test_overlap),
                'val_test': len(val_test_overlap)
            },
            'samples_per_subject': {
                'mean': float(np.mean(list(subject_counts.values()))),
                'std': float(np.std(list(subject_counts.values()))),
                'min': int(min(subject_counts.values())),
                'max': int(max(subject_counts.values()))
            }
        }
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.error("发现受试者重叠!")
            logger.error(f"训练-验证重叠: {train_val_overlap}")
            logger.error(f"训练-测试重叠: {train_test_overlap}")
            logger.error(f"验证-测试重叠: {val_test_overlap}")
        
        return subject_analysis
    
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
                'train_ratio': n_train / n_total,
                'val_ratio': n_val / n_total,
                'test_ratio': n_test / n_total
            }
        }
        
        return balance_analysis
    
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
        
        # 总体状态
        overall_score = (completeness_score * 0.3 + 
                        feature_score * 0.2 + 
                        separation_score * 0.3 + 
                        balance_score * 0.2)
        
        overall_status = {
            'completeness_score': float(completeness_score),
            'feature_quality_score': float(feature_score),
            'subject_separation_score': float(separation_score),
            'balance_score': float(balance_score),
            'overall_score': float(overall_score),
            'status': 'PASS' if overall_score > 0.8 else 'WARNING' if overall_score > 0.6 else 'FAIL',
            'label_coverage': float(completeness_score),
            'has_data_issues': feature_analysis['nan_count'] > 0 or feature_analysis['inf_count'] > 0,
            'has_overlap_issues': separation_score < 1.0
        }
        
        return overall_status
    
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
        
        label_mapping = {
            'is_one_hot': is_one_hot,
            'one_hot_dim': one_hot_dim,
            'min_label': int(np.min(unique_classes)),
            'max_label': int(np.max(unique_classes)),
            'n_classes_expected': one_hot_dim,
            'n_classes_actual': len(unique_classes),
            'unique_classes': unique_classes,
            'missing_classes': missing_classes,
            'label_type': 'brain_region',
            'label_description': '脑区标签，每个标签代表一个特定的脑区'
        }
        
        if missing_classes:
            label_mapping['missing_classes_info'] = {
                'count': len(missing_classes),
                'description': '这些脑区在数据中没有样本，可能是成像质量问题或脑区太小',
                'recommendation': '模型输出保持完整维度，但这些类别的预测可能不可靠'
            }
        
        return label_mapping





        