#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-One-Subject-Out评估器
评估跨受试者的泛化性能
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch

from common.deep_network import DeepNetworkUtils
from common.config import Config

logger = logging.getLogger(__name__)


class LOSOEvaluator:
    """LOSO评估器"""
    
    def __init__(self, model_types: List[str] = None,
                 device: str = 'cpu',
                 max_subjects: int = 8):
        """
        初始化评估器
        
        Args:
            model_types: 要评估的模型类型
            device: 计算设备
            max_subjects: 最大测试受试者数（节省时间）
        """
        self.model_types = model_types or ['rf', 'lr', 'deep']
        self.device = device
        self.max_subjects = max_subjects
        self.deep_utils = DeepNetworkUtils()
        
        # 模型配置（与BaselineTester保持一致）
        self.model_configs = {
            'rf': {
                'name': 'RandomForest',
                'creator': lambda: RandomForestClassifier(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                )
            },
            'lr': {
                'name': 'LogisticRegression', 
                'creator': lambda: LogisticRegression(
                    max_iter=1000, C=0.1, random_state=42
                )
            },
            'deep': {
                'name': 'Deep4x4096',
                'creator': lambda: self._create_deep_classifier(Config.ALEX_HYPERPARAMS)
            },
            'deep_lightweight': {
                'name': 'DeepLightweight',
                'creator': lambda: self._create_deep_classifier({
                    'batch_size': 64,
                    'no_epochs': 15,
                    'learning_rate': 0.0001,
                    'weight_decay': 0.00001,
                    'dropout_rate': 0.5
                })
            }
        }
    
    def evaluate_loso_performance(self, X: np.ndarray, y: np.ndarray,
                                subjects: np.ndarray,
                                label_mapping: Dict) -> Dict[str, Any]:
        """
        执行LOSO评估
        
        Args:
            X: 特征矩阵
            y: 标签
            subjects: 受试者标签
            label_mapping: 标签映射信息
            
        Returns:
            评估结果
        """
        logger.info("开始Leave-One-Subject-Out评估...")
        
        # 获取唯一受试者
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # 限制测试受试者数量
        if n_subjects > self.max_subjects:
            logger.info(f"受试者总数 {n_subjects}，限制为 {self.max_subjects} 个进行测试")
            test_subjects = np.random.choice(unique_subjects, self.max_subjects, replace=False)
        else:
            test_subjects = unique_subjects
        
        results = {
            'model_results': {},
            'per_subject_results': {},
            'mean_generalization_gap': 0.0,
            'n_subjects_tested': len(test_subjects)
        }
        
        # 1. 全局LOSO评估
        logger.info(f"\n1. 执行全局LOSO评估 ({len(test_subjects)} 个受试者)...")
        global_loso_results = self._evaluate_global_loso(
            X, y, subjects, test_subjects, label_mapping
        )
        
        # 2. 分脑区LOSO评估
        logger.info("\n2. 执行分脑区LOSO评估...")
        region_loso_results = self._evaluate_region_loso(
            X, y, subjects, test_subjects, label_mapping
        )
        
        # 3. 计算泛化差距
        logger.info("\n3. 计算泛化差距...")
        generalization_analysis = self._analyze_generalization_gap(
            global_loso_results, region_loso_results
        )
        
        # 整合结果
        for model_type in self.model_types:
            if model_type in global_loso_results:
                results['model_results'][model_type] = {
                    'mean_accuracy': global_loso_results[model_type]['mean_accuracy'],
                    'std_accuracy': global_loso_results[model_type]['std_accuracy'],
                    'per_subject_scores': global_loso_results[model_type]['per_subject_scores'],
                    'region_mean_accuracy': region_loso_results.get(model_type, {}).get('mean_accuracy', 0),
                    'generalization_gap': generalization_analysis.get(model_type, {}).get('gap', 0)
                }
        
        results['mean_generalization_gap'] = generalization_analysis['overall_gap']
        results['deep_network_advantage'] = generalization_analysis.get('deep_advantage', 0)
        
        # 保存每个受试者的详细结果
        results['per_subject_results'] = self._collect_per_subject_results(
            global_loso_results, test_subjects
        )
        
        # 打印摘要
        self._print_loso_summary(results)
        
        return results
    
    def _evaluate_global_loso(self, X: np.ndarray, y: np.ndarray,
                            subjects: np.ndarray, test_subjects: np.ndarray,
                            label_mapping: Dict) -> Dict[str, Any]:
        """执行全局LOSO评估"""
        
        # 处理标签
        if label_mapping['is_one_hot']:
            y_classes = np.argmax(y, axis=1)
        else:
            y_classes = y.flatten()
        
        results = {}
        
        for model_type in self.model_types:
            if model_type not in self.model_configs:
                continue
            
            logger.info(f"  评估 {self.model_configs[model_type]['name']}...")
            
            per_subject_scores = []
            per_subject_f1 = []
            training_times = []
            
            for i, test_subject in enumerate(test_subjects):
                # 分割数据
                train_mask = subjects != test_subject
                test_mask = subjects == test_subject
                
                if np.sum(test_mask) < 100:  # 测试样本太少
                    continue
                
                X_train = X[train_mask]
                y_train = y[train_mask] if label_mapping['is_one_hot'] else y_classes[train_mask]
                X_test = X[test_mask]
                y_test = y_classes[test_mask]
                
                try:
                    start_time = time.time()
                    
                    # 创建并训练模型
                    model = self.model_configs[model_type]['creator']()
                    
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    else:
                        logger.error(f"模型 {model_type} 没有fit方法")
                        continue
                    
                    training_time = time.time() - start_time
                    
                    # 计算指标
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    
                    per_subject_scores.append(accuracy)
                    per_subject_f1.append(f1)
                    training_times.append(training_time)
                    
                    if i < 3:  # 只打印前3个受试者的结果
                        logger.info(f"    受试者 {int(test_subject)}: Acc={accuracy:.3f}, F1={f1:.3f}")
                    
                except Exception as e:
                    logger.error(f"    受试者 {int(test_subject)} 评估失败: {e}")
                    continue
            
            if per_subject_scores:
                results[model_type] = {
                    'mean_accuracy': np.mean(per_subject_scores),
                    'std_accuracy': np.std(per_subject_scores),
                    'mean_f1': np.mean(per_subject_f1),
                    'std_f1': np.std(per_subject_f1),
                    'per_subject_scores': per_subject_scores,
                    'mean_training_time': np.mean(training_times),
                    'n_subjects_evaluated': len(per_subject_scores)
                }
                
                logger.info(f"    平均准确率: {results[model_type]['mean_accuracy']:.3f} "
                          f"(±{results[model_type]['std_accuracy']:.3f})")
        
        return results
    
    def _evaluate_region_loso(self, X: np.ndarray, y: np.ndarray,
                            subjects: np.ndarray, test_subjects: np.ndarray,
                            label_mapping: Dict) -> Dict[str, Any]:
        """执行分脑区LOSO评估"""
        
        # 构建分脑区数据集
        region_data = self._build_region_dataset_for_loso(X, y, subjects, label_mapping)
        
        if not region_data:
            logger.warning("  分脑区数据集构建失败")
            return {}
        
        X_region = region_data['features']
        subject_labels = region_data['subject_labels']
        
        # 重新映射测试受试者
        unique_subjects_region = np.unique(subject_labels)
        test_subjects_region = [s for s in test_subjects if s in unique_subjects_region]
        
        if len(test_subjects_region) < 3:
            logger.warning("  分脑区测试受试者数量不足")
            return {}
        
        logger.info(f"  分脑区数据: {len(X_region)} 样本, {len(unique_subjects_region)} 受试者")
        
        results = {}
        
        for model_type in self.model_types[:2]:  # 只测试RF和LR，深度网络太慢
            if model_type not in self.model_configs:
                continue
            
            logger.info(f"  评估 {self.model_configs[model_type]['name']} (分脑区)...")
            
            per_subject_scores = []
            
            for test_subject in test_subjects_region[:5]:  # 限制测试数量
                train_mask = subject_labels != test_subject
                test_mask = subject_labels == test_subject
                
                if np.sum(test_mask) < 10:
                    continue
                
                X_train = X_region[train_mask]
                y_train = subject_labels[train_mask]
                X_test = X_region[test_mask]
                y_test = subject_labels[test_mask]
                
                # 重新映射标签
                unique_train_subjects = np.unique(y_train)
                subject_mapping = {sid: i for i, sid in enumerate(unique_train_subjects)}
                
                y_train_mapped = np.array([subject_mapping[sid] for sid in y_train])
                
                # 测试集标签映射
                y_test_mapped = []
                for sid in y_test:
                    if sid in subject_mapping:
                        y_test_mapped.append(subject_mapping[sid])
                    else:
                        y_test_mapped.append(-1)  # 未见过的受试者
                
                if -1 in y_test_mapped:
                    continue  # 跳过包含未见过受试者的测试
                
                y_test_mapped = np.array(y_test_mapped)
                
                try:
                    model = self.model_configs[model_type]['creator']()
                    model.fit(X_train, y_train_mapped)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test_mapped, y_pred)
                    per_subject_scores.append(accuracy)
                    
                except Exception as e:
                    logger.error(f"    分脑区评估失败: {e}")
                    continue
            
            if per_subject_scores:
                results[model_type] = {
                    'mean_accuracy': np.mean(per_subject_scores),
                    'std_accuracy': np.std(per_subject_scores),
                    'n_subjects_evaluated': len(per_subject_scores)
                }
                
                logger.info(f"    平均准确率: {results[model_type]['mean_accuracy']:.3f}")
        
        return results
    
    def _build_region_dataset_for_loso(self, X: np.ndarray, y: np.ndarray,
                                     subjects: np.ndarray, label_mapping: Dict) -> Optional[Dict]:
        """为LOSO构建分脑区数据集"""
        
        # 处理标签
        if label_mapping['is_one_hot']:
            y_regions = np.argmax(y, axis=1)
        else:
            y_regions = y.flatten()
        
        # 构建数据集
        features_list = []
        subject_labels = []
        region_labels = []
        
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(y_regions)
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects == subject_id) & (y_regions == region_id)
                if np.sum(mask) >= Config.MIN_SAMPLES_FOR_SUBJECT_REGION:
                    region_features = np.mean(X[mask], axis=0)
                    features_list.append(region_features)
                    subject_labels.append(subject_id)
                    region_labels.append(region_id)
        
        if len(features_list) < 200:
            return None
        
        return {
            'features': np.array(features_list),
            'subject_labels': np.array(subject_labels),
            'region_labels': np.array(region_labels)
        }
    
    def _analyze_generalization_gap(self, global_results: Dict, 
                                  region_results: Dict) -> Dict[str, Any]:
        """分析泛化差距"""
        
        gaps = []
        analysis = {}
        
        for model_type in self.model_types:
            if model_type in global_results:
                loso_acc = global_results[model_type]['mean_accuracy']
                
                # 这里需要baseline准确率，暂时使用LOSO准确率的1.2倍作为估计
                estimated_baseline = min(0.95, loso_acc * 1.2)
                gap = estimated_baseline - loso_acc
                
                analysis[model_type] = {
                    'loso_accuracy': loso_acc,
                    'estimated_baseline': estimated_baseline,
                    'gap': gap,
                    'relative_gap': gap / estimated_baseline if estimated_baseline > 0 else 0
                }
                
                gaps.append(gap)
        
        analysis['overall_gap'] = np.mean(gaps) if gaps else 0
        
        # 计算深度网络优势
        if 'deep' in analysis:
            traditional_gaps = [analysis[m]['gap'] for m in ['rf', 'lr'] if m in analysis]
            if traditional_gaps:
                deep_gap = analysis['deep']['gap']
                avg_traditional_gap = np.mean(traditional_gaps)
                analysis['deep_advantage'] = avg_traditional_gap - deep_gap
        
        return analysis
    
    def _collect_per_subject_results(self, global_results: Dict, 
                                   test_subjects: np.ndarray) -> Dict:
        """收集每个受试者的详细结果"""
        per_subject = {}
        
        for i, subject_id in enumerate(test_subjects):
            subject_results = {
                'subject_id': int(subject_id),
                'model_scores': {}
            }
            
            for model_type, results in global_results.items():
                if 'per_subject_scores' in results and i < len(results['per_subject_scores']):
                    subject_results['model_scores'][model_type] = results['per_subject_scores'][i]
            
            per_subject[int(subject_id)] = subject_results
        
        return per_subject
    
    def _create_deep_classifier(self, params: Dict) -> Any:
        """创建深度网络分类器（简化版）"""
        
        class SimpleDeepWrapper:
            def __init__(self, deep_utils, params):
                self.deep_utils = deep_utils
                self.params = params
                self.network = None