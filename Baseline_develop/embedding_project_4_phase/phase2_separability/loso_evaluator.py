#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOSO评估器
实现Leave-One-Subject-Out交叉验证
评估脑区分类任务在不同受试者上的泛化能力
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.config import Config
from common.deep_network import DeepNetworkUtils
from phase2_separability.baseline_tester import DeepClassifierWrapper  # 重用已定义的类

logger = logging.getLogger(__name__)


class LOSOEvaluator:
    """LOSO评估器"""
    
    def __init__(self, model_types: List[str] = None,
                 device: str = 'cpu',
                 max_subjects: int = 8,
                 save_best_models: bool = False):
        """
        初始化LOSO评估器
        
        Args:
            model_types: 要测试的模型类型
            device: 计算设备
            max_subjects: 最大测试受试者数（节省时间）
            save_best_models: 是否保存每个LOSO折的最佳模型
        """
        self.model_types = model_types or ['lr', 'deep']
        self.device = device
        self.max_subjects = max_subjects
        self.save_best_models = save_best_models
        self.deep_utils = DeepNetworkUtils()
    
    def evaluate_loso_performance(self, X: np.ndarray, y: np.ndarray, 
                                subjects: np.ndarray,
                                label_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行LOSO评估
        评估脑区分类模型在留一受试者情况下的泛化性能
        
        Args:
            X: 特征数据
            y: 标签数据（脑区标签）
            subjects: 受试者标签
            label_mapping: 标签映射信息
            
        Returns:
            LOSO评估结果
        """
        logger.info("开始Leave-One-Subject-Out评估...")
        logger.info("评估脑区分类任务在不同受试者上的泛化能力")
        
        # 转换标签
        if label_mapping['is_one_hot']:
            y_classes = np.argmax(y, axis=1)
        else:
            y_classes = y.flatten().astype(int)
        
        results = {
            'model_results': {},
            'subject_generalization_gaps': {},
            'mean_generalization_gap': 0.0,
            'deep_network_advantage': 0.0,
            'region_wise_summary': {},
            'deep_network_epoch_wise_results': {}  # 新增：存储深度网络每个epoch的结果
        }
        
        # 1. 全局LOSO分析
        logger.info("\n1. 全局LOSO性能分析...")
        global_loso_results = self._evaluate_global_loso(
            X, y_classes, subjects
        )
        
        # 2. 分脑区LOSO分析
        logger.info("\n2. 分脑区LOSO性能分析...")
        region_loso_results = self._evaluate_region_wise_loso(
            X, y_classes, subjects
        )
        
        # 3. 整合结果
        for model_type in self.model_types:
            model_key = self._get_model_key(model_type)
            
            if model_key in global_loso_results:
                results['model_results'][model_type] = {
                    'mean_accuracy': global_loso_results[model_key]['mean_accuracy'],
                    'std_accuracy': global_loso_results[model_key]['std_accuracy'],
                    'per_subject_accuracies': global_loso_results[model_key]['accuracies'],
                    'generalization_gap': global_loso_results[model_key].get('generalization_gap', 0)
                }
                
                # 添加深度网络的详细结果
                if model_type == 'deep' and 'epoch_wise_results' in global_loso_results[model_key]:
                    results['model_results'][model_type]['epoch_wise_results'] = \
                        global_loso_results[model_key]['epoch_wise_results']
                    results['model_results'][model_type]['best_epoch_info'] = \
                        global_loso_results[model_key].get('best_epoch_info', {})
                    results['model_results'][model_type]['overfitting_analysis'] = \
                        global_loso_results[model_key].get('overfitting_analysis', {})
                
                # 添加分脑区汇总信息
                if 'summary' in region_loso_results and model_key in region_loso_results['summary']:
                    results['model_results'][model_type]['region_wise_accuracy'] = \
                        region_loso_results['summary'][model_key].get('mean_accuracy', 0)
        
        # 保存分脑区详细结果
        if 'region_results' in region_loso_results:
            results['region_wise_details'] = region_loso_results['region_results']
            results['region_wise_summary'] = region_loso_results.get('summary', {})
        
        # 计算平均泛化差距
        gaps = [r['generalization_gap'] for r in results['model_results'].values() 
                if 'generalization_gap' in r]
        results['mean_generalization_gap'] = np.mean(gaps) if gaps else 0
        
        # 计算深度网络优势
        if 'deep' in results['model_results']:
            deep_acc = results['model_results']['deep']['mean_accuracy']
            traditional_accs = []
            
            for model in ['rf', 'lr']:
                if model in results['model_results']:
                    traditional_accs.append(results['model_results'][model]['mean_accuracy'])
            
            if traditional_accs:
                results['deep_network_advantage'] = deep_acc - np.mean(traditional_accs)
        
        # 打印摘要
        self._print_loso_summary(results)
        
        return results
    
    def _get_model_key(self, model_type: str) -> str:
        """获取模型键名"""
        model_mapping = {
            'rf': 'RandomForest',
            'lr': 'LogisticRegression',
            'deep': 'Deep4x4096'
        }
        return model_mapping.get(model_type, model_type)
    
    def _evaluate_global_loso(self, X: np.ndarray, y: np.ndarray, 
                            subjects: np.ndarray) -> Dict[str, Any]:
        """
        评估全局LOSO性能
        留一受试者的脑区分类泛化能力评估
        """
        unique_subjects = np.unique(subjects)
        
        # 限制测试的受试者数量
        test_subjects = unique_subjects[:self.max_subjects]
        logger.info(f"测试 {len(test_subjects)} 个受试者的LOSO性能")
        
        # 初始化结果存储
        model_results = {}
        for model_type in self.model_types:
            model_key = self._get_model_key(model_type)
            model_results[model_key] = {
                'accuracies': [],
                'f1_scores': [],
                'training_times': [],
                'epoch_wise_results': [] if model_type == 'deep' else None,  # 深度网络存储每个epoch结果
                'per_subject_epoch_results': {} if model_type == 'deep' else None  # 每个受试者的epoch结果
            }
        
        # 对每个测试受试者执行LOSO
        for i, test_subject in enumerate(test_subjects):
            logger.info(f"  测试受试者 {int(test_subject)} ({i+1}/{len(test_subjects)})...")
            
            # 创建训练和测试掩码
            train_mask = subjects != test_subject
            test_mask = subjects == test_subject
            
            if np.sum(test_mask) < 100:
                logger.warning(f"    受试者 {int(test_subject)} 样本不足，跳过")
                continue
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # 测试每个模型
            for model_type in self.model_types:
                model_key = self._get_model_key(model_type)
                start_time = time.time()
                
                try:
                    # if model_type == 'rf':
                    #     model = RandomForestClassifier(
                    #         n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                    #     )
                    #     model.fit(X_train, y_train)
                    #     y_pred = model.predict(X_test)
                        
                    # elif model_type == 'lr':
                    #     model = LogisticRegression(
                    #         max_iter=1000, C=0.1, random_state=42
                    #     )
                    #     model.fit(X_train, y_train)
                    #     y_pred = model.predict(X_test)
                        
                    # elif model_type == 'deep':
                    #     # 深度网络
                    #     model = DeepClassifierWrapper(self.deep_utils)
                    #     model.fit(X_train, y_train)
                    #     y_pred = model.predict(X_test)


                    
                    if model_type == 'lr':
                        model = LogisticRegression(
                            max_iter=1000, C=0.1, random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='macro')
                        training_time = time.time() - start_time
                        
                        model_results[model_key]['accuracies'].append(accuracy)
                        model_results[model_key]['f1_scores'].append(f1)
                        model_results[model_key]['training_times'].append(training_time)
                        
                        logger.info(f"    {model_key}: Acc={accuracy:.3f}, F1={f1:.3f}")
                        
                    elif model_type == 'deep':
                        # 深度网络 - 使用留出的测试集作为验证集
                        model = DeepClassifierWrapper(self.deep_utils)
                        # 关键修改：将测试集传入作为验证集，在每个epoch评估
                        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
                        
                        # 获取最终预测
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='macro')
                        training_time = time.time() - start_time
                        
                        model_results[model_key]['accuracies'].append(accuracy)
                        model_results[model_key]['f1_scores'].append(f1)
                        model_results[model_key]['training_times'].append(training_time)
                        
                        # 保存每个epoch的验证结果
                        epoch_results = {
                            'subject_id': int(test_subject),
                            'train_losses': model.training_history['train_loss'],
                            'train_accuracies': model.training_history['train_accuracy'],
                            'train_f1s': model.training_history['train_f1'],
                            'val_losses': model.training_history['val_loss'],
                            'val_accuracies': model.training_history['val_accuracy'],
                            'val_f1s': model.training_history['val_f1'],
                            'final_test_accuracy': accuracy,
                            'final_test_f1': f1
                        }
                        
                        # 分析最佳epoch
                        if model.training_history['val_f1']:
                            best_epoch_idx = np.argmax(model.training_history['val_f1'])
                            best_val_f1 = model.training_history['val_f1'][best_epoch_idx]
                            best_val_acc = model.training_history['val_accuracy'][best_epoch_idx]
                            
                            epoch_results['best_epoch'] = best_epoch_idx + 1
                            epoch_results['best_epoch_val_f1'] = best_val_f1
                            epoch_results['best_epoch_val_accuracy'] = best_val_acc
                            
                            # 检测过拟合
                            final_epoch_val_f1 = model.training_history['val_f1'][-1]
                            overfitting_degree = best_val_f1 - final_epoch_val_f1
                            epoch_results['overfitting_degree'] = overfitting_degree
                            epoch_results['overfitting_detected'] = overfitting_degree > 0.05
                            
                            logger.info(f"    {model_key}: Final Acc={accuracy:.3f}, F1={f1:.3f}")
                            logger.info(f"              Best Epoch={best_epoch_idx+1}, Best Val F1={best_val_f1:.3f}")
                            if epoch_results['overfitting_detected']:
                                logger.warning(f"              过拟合检测: F1下降{overfitting_degree:.3f}")
                        
                        model_results[model_key]['epoch_wise_results'].append(epoch_results)
                        model_results[model_key]['per_subject_epoch_results'][int(test_subject)] = epoch_results
                
                except Exception as e:
                    logger.error(f"    {model_key} 失败: {e}")
                    logger.exception("详细错误信息:")
        
        # 计算统计信息
        final_results = {}
        for model_key, results in model_results.items():
            if results['accuracies']:
                final_results[model_key] = {
                    'mean_accuracy': np.mean(results['accuracies']),
                    'std_accuracy': np.std(results['accuracies']),
                    'mean_f1': np.mean(results['f1_scores']),
                    'std_f1': np.std(results['f1_scores']),
                    'mean_training_time': np.mean(results['training_times']),
                    'accuracies': results['accuracies'],
                    'n_subjects_tested': len(results['accuracies']),
                    'generalization_gap': 0  # 将在后续计算
                }
                
                # 添加深度网络的epoch分析
                if model_key == 'Deep4x4096' and results['epoch_wise_results']:
                    # 计算所有受试者的平均epoch性能
                    all_val_f1s = []
                    all_val_accs = []
                    overfitting_subjects = []
                    
                    for epoch_result in results['epoch_wise_results']:
                        all_val_f1s.append(epoch_result['val_f1s'])
                        all_val_accs.append(epoch_result['val_accuracies'])
                        if epoch_result.get('overfitting_detected', False):
                            overfitting_subjects.append(epoch_result['subject_id'])
                    
                    # 计算每个epoch的平均性能
                    n_epochs = len(all_val_f1s[0]) if all_val_f1s else 0
                    mean_val_f1_per_epoch = []
                    mean_val_acc_per_epoch = []
                    
                    for epoch_idx in range(n_epochs):
                        epoch_f1s = [f1s[epoch_idx] for f1s in all_val_f1s if epoch_idx < len(f1s)]
                        epoch_accs = [accs[epoch_idx] for accs in all_val_accs if epoch_idx < len(accs)]
                        
                        if epoch_f1s:
                            mean_val_f1_per_epoch.append(np.mean(epoch_f1s))
                            mean_val_acc_per_epoch.append(np.mean(epoch_accs))
                    
                    # 找到平均最佳epoch
                    if mean_val_f1_per_epoch:
                        best_avg_epoch = np.argmax(mean_val_f1_per_epoch)
                        best_avg_f1 = mean_val_f1_per_epoch[best_avg_epoch]
                        
                        final_results[model_key]['best_epoch_info'] = {
                            'average_best_epoch': best_avg_epoch + 1,
                            'average_best_f1': float(best_avg_f1),
                            'mean_val_f1_per_epoch': mean_val_f1_per_epoch,
                            'mean_val_acc_per_epoch': mean_val_acc_per_epoch
                        }
                    
                    final_results[model_key]['overfitting_analysis'] = {
                        'n_subjects_overfitted': len(overfitting_subjects),
                        'overfitting_rate': len(overfitting_subjects) / len(results['epoch_wise_results']),
                        'overfitted_subjects': overfitting_subjects
                    }
                    
                    final_results[model_key]['epoch_wise_results'] = results['epoch_wise_results']
                    final_results[model_key]['per_subject_epoch_results'] = results['per_subject_epoch_results']
        
        return final_results
    
    def _evaluate_region_wise_loso(self, X: np.ndarray, y: np.ndarray,
                                subjects: np.ndarray) -> Dict[str, Any]:
        """
        评估分脑区LOSO性能
        
        对每个脑区评估：
        1. 该脑区 vs 其他脑区的二分类LOSO性能
        2. 该脑区在多分类中的LOSO性能
        
        Args:
            X: 特征数据
            y: 脑区标签（每个体素属于哪个脑区）
            subjects: 受试者标签（每个体素来自哪个受试者）
            
        Returns:
            分脑区LOSO评估结果
        """
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(y)
        
        # 限制测试受试者数量
        test_subjects = unique_subjects[:min(self.max_subjects, len(unique_subjects))]
        
        logger.info(f"开始分脑区LOSO评估，共{len(unique_regions)}个脑区")
        
        # 存储每个脑区的评估结果
        region_results = {}
        
        # 1. 先进行全脑区多分类LOSO，收集每个脑区的性能
        logger.info("执行多分类LOSO以评估每个脑区的性能...")
        multiclass_region_scores = self._evaluate_multiclass_region_loso(
            X, y, subjects, test_subjects
        )
        
        # 2. 对每个脑区进行二分类LOSO评估（可选，更细致的分析）
        logger.info("执行分脑区二分类LOSO评估...")
        binary_region_scores = self._evaluate_binary_region_loso(
            X, y, subjects, test_subjects, unique_regions
        )
        
        # 3. 整合结果
        for region_id in unique_regions:
            region_id_int = int(region_id)
            
            region_results[region_id_int] = {
                'multiclass_performance': multiclass_region_scores.get(region_id_int, {}),
                'binary_performance': binary_region_scores.get(region_id_int, {}),
                'n_samples': int(np.sum(y == region_id)),
                'n_subjects': int(len(np.unique(subjects[y == region_id])))
            }
            
            # 计算综合泛化得分
            multi_f1 = multiclass_region_scores.get(region_id_int, {}).get('mean_f1', 0)
            binary_acc = binary_region_scores.get(region_id_int, {}).get('mean_accuracy', 0)
            
            # 综合得分（可以调整权重）
            region_results[region_id_int]['generalization_score'] = (multi_f1 + binary_acc) / 2
        
        # 4. 汇总统计
        summary_results = self._summarize_region_loso_results(region_results)
        
        # 5. 为每个模型类型计算分脑区平均性能
        model_summary = {}
        for model_type in self.model_types:
            model_key = self._get_model_key(model_type)
            # 这里简化处理，使用多分类的平均性能作为分脑区性能
            region_f1s = [r['multiclass_performance'].get('mean_f1', 0) 
                         for r in region_results.values() 
                         if 'mean_f1' in r.get('multiclass_performance', {})]
            
            if region_f1s:
                model_summary[model_key] = {
                    'mean_accuracy': np.mean(region_f1s),  # 使用F1作为准确率的代理
                    'std_accuracy': np.std(region_f1s)
                }
        
        return {
            'region_results': region_results,
            'summary': model_summary,
            'detailed_summary': summary_results,
            'test_subjects': test_subjects.tolist()
        }

    def _evaluate_multiclass_region_loso(self, X: np.ndarray, y: np.ndarray,
                                    subjects: np.ndarray, 
                                    test_subjects: np.ndarray) -> Dict[int, Dict]:
        """
        多分类LOSO：评估每个脑区在101类分类任务中的表现
        修改版：对深度网络使用epoch-wise验证
        """
        from sklearn.metrics import classification_report, confusion_matrix
        
        # 初始化每个脑区的性能记录
        unique_regions = np.unique(y)
        region_scores = {int(r): {'precisions': [], 'recalls': [], 'f1s': []} 
                        for r in unique_regions}
        
        # 对每个测试受试者执行LOSO
        for test_subject in test_subjects:
            train_mask = subjects != test_subject
            test_mask = subjects == test_subject
            
            if np.sum(test_mask) < 50:  # 样本太少，跳过
                continue
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            # 使用最佳模型（根据baseline结果选择）
            if 'deep' in self.model_types:
                model = DeepClassifierWrapper(self.deep_utils)
                # 使用测试集作为验证集
                model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            else:
                model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
                model.fit(X_train, y_train)
                
            try:
                y_pred = model.predict(X_test)
                
                # 计算每个类别的性能
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # 记录每个脑区的性能
                for region_id in unique_regions:
                    region_key = str(int(region_id))
                    if region_key in report:
                        metrics = report[region_key]
                        region_scores[int(region_id)]['precisions'].append(metrics['precision'])
                        region_scores[int(region_id)]['recalls'].append(metrics['recall'])
                        region_scores[int(region_id)]['f1s'].append(metrics['f1-score'])
                        
            except Exception as e:
                logger.warning(f"多分类LOSO失败于受试者{test_subject}: {e}")
                continue
        
        # 计算每个脑区的平均性能
        final_scores = {}
        for region_id, scores in region_scores.items():
            if scores['f1s']:  # 如果有有效的评分
                final_scores[region_id] = {
                    'mean_precision': float(np.mean(scores['precisions'])),
                    'mean_recall': float(np.mean(scores['recalls'])),
                    'mean_f1': float(np.mean(scores['f1s'])),
                    'std_f1': float(np.std(scores['f1s'])),
                    'n_evaluations': len(scores['f1s'])
                }
        
        return final_scores

    def _evaluate_binary_region_loso(self, X: np.ndarray, y: np.ndarray,
                                    subjects: np.ndarray, test_subjects: np.ndarray,
                                    unique_regions: np.ndarray) -> Dict[int, Dict]:
        """
        二分类LOSO：对每个脑区评估"该脑区 vs 其他所有脑区"的分类性能
        """
        region_binary_scores = {}
        
        # 只评估样本充足的脑区
        for region_id in unique_regions:
            region_mask = y == region_id
            n_region_samples = np.sum(region_mask)
            
            if n_region_samples < 500:  # 该脑区样本太少
                continue
                
            region_subjects = subjects[region_mask]
            n_region_subjects = len(np.unique(region_subjects))
            
            if n_region_subjects < 5:  # 该脑区涉及的受试者太少
                continue
            
            logger.info(f"  评估脑区 {int(region_id)} (样本数: {n_region_samples}, "
                    f"受试者数: {n_region_subjects})")
            
            # 对该脑区执行二分类LOSO
            accuracies = []
            f1_scores = []
            
            for test_subject in test_subjects:
                train_mask = subjects != test_subject
                test_mask = subjects == test_subject
                
                if np.sum(test_mask) < 20:
                    continue
                
                # 创建二分类标签
                y_binary_train = (y[train_mask] == region_id).astype(int)
                y_binary_test = (y[test_mask] == region_id).astype(int)
                
                # 检查测试集中是否有该脑区的样本
                if np.sum(y_binary_test) == 0 or np.sum(y_binary_test) == len(y_binary_test):
                    continue  # 全是正样本或全是负样本，跳过
                
                X_train = X[train_mask]
                X_test = X[test_mask]
                
                # 简单的逻辑回归
                clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, class_weight='balanced')
                
                try:
                    clf.fit(X_train, y_binary_train)
                    y_pred = clf.predict(X_test)
                    
                    acc = accuracy_score(y_binary_test, y_pred)
                    f1 = f1_score(y_binary_test, y_pred)
                    
                    accuracies.append(acc)
                    f1_scores.append(f1)
                    
                except Exception as e:
                    logger.debug(f"二分类失败于脑区{region_id}和受试者{test_subject}: {e}")
                    continue
            
            # 记录该脑区的二分类性能
            if accuracies:
                region_binary_scores[int(region_id)] = {
                    'mean_accuracy': float(np.mean(accuracies)),
                    'std_accuracy': float(np.std(accuracies)),
                    'mean_f1': float(np.mean(f1_scores)),
                    'std_f1': float(np.std(f1_scores)),
                    'n_evaluations': len(accuracies)
                }
        
        return region_binary_scores

    def _summarize_region_loso_results(self, region_results: Dict[int, Dict]) -> Dict[str, Any]:
        """
        汇总分脑区LOSO结果
        """
        # 提取所有脑区的泛化得分
        generalization_scores = []
        multiclass_f1s = []
        binary_accs = []
        
        for region_id, results in region_results.items():
            if 'generalization_score' in results:
                generalization_scores.append(results['generalization_score'])
            
            multi_perf = results.get('multiclass_performance', {})
            if 'mean_f1' in multi_perf:
                multiclass_f1s.append(multi_perf['mean_f1'])
            
            binary_perf = results.get('binary_performance', {})
            if 'mean_accuracy' in binary_perf:
                binary_accs.append(binary_perf['mean_accuracy'])
        
        # 识别泛化性能最差的脑区（最需要embedding的候选）
        sorted_regions = sorted(region_results.items(), 
                            key=lambda x: x[1].get('generalization_score', 1.0))
        
        poor_generalization_regions = [r[0] for r in sorted_regions[:10]]  # 最差的10个
        good_generalization_regions = [r[0] for r in sorted_regions[-10:]]  # 最好的10个
        
        return {
            'mean_generalization_score': float(np.mean(generalization_scores)) if generalization_scores else 0,
            'std_generalization_score': float(np.std(generalization_scores)) if generalization_scores else 0,
            'mean_multiclass_f1': float(np.mean(multiclass_f1s)) if multiclass_f1s else 0,
            'mean_binary_accuracy': float(np.mean(binary_accs)) if binary_accs else 0,
            'n_evaluated_regions': len(region_results),
            'poor_generalization_regions': poor_generalization_regions,
            'good_generalization_regions': good_generalization_regions
        }
    

    def _build_region_dataset(self, X: np.ndarray, regions: np.ndarray,
                            subjects: np.ndarray) -> Dict[str, np.ndarray]:
        """
        构建分脑区数据集
        
        修正说明：
        - regions: 脑区标签（体素属于哪个脑区）
        - subjects: 受试者标签（体素来自哪个受试者）
        - 返回：每个受试者-脑区组合的平均特征
        """
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(regions)
        
        X_list = []
        subject_labels = []  # 受试者标签
        region_labels = []   # 脑区标签
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                # 找到属于特定受试者和脑区的体素
                mask = (subjects == subject_id) & (regions == region_id)
                n_voxels = np.sum(mask)
                
                if n_voxels >= Config.MIN_VOXELS_PER_REGION:
                    # 计算该受试者在该脑区的平均特征
                    region_features = np.mean(X[mask], axis=0)
                    X_list.append(region_features)
                    subject_labels.append(subject_id)  # 记录受试者ID
                    region_labels.append(region_id)    # 记录脑区ID
        
        logger.info(f"构建了 {len(X_list)} 个受试者-脑区特征向量")
        
        return {
            'features': np.array(X_list),
            'subject_labels': np.array(subject_labels),  # 受试者标签
            'region_labels': np.array(region_labels)     # 脑区标签
        }
    
    def _print_loso_summary(self, results: Dict):
        """打印LOSO评估摘要（增强版，包含epoch分析）"""
        logger.info("\nLOSO评估摘要:")
        logger.info("="*50)
        
        for model_type, model_results in results['model_results'].items():
            logger.info(f"\n{model_type}:")
            logger.info(f"  平均准确率: {model_results['mean_accuracy']:.3f} ± {model_results['std_accuracy']:.3f}")
            logger.info(f"  泛化差距: {model_results['generalization_gap']:.3f}")
            
            if 'region_wise_accuracy' in model_results:
                logger.info(f"  分脑区准确率: {model_results['region_wise_accuracy']:.3f}")
            
            # 打印深度网络的epoch分析
            if model_type == 'deep' and 'best_epoch_info' in model_results:
                best_info = model_results['best_epoch_info']
                logger.info(f"\n  深度网络Epoch分析:")
                logger.info(f"    平均最佳Epoch: {best_info['average_best_epoch']}")
                logger.info(f"    平均最佳F1: {best_info['average_best_f1']:.3f}")
                
                if 'overfitting_analysis' in model_results:
                    overfit_info = model_results['overfitting_analysis']
                    logger.info(f"    过拟合率: {overfit_info['overfitting_rate']:.2%} "
                              f"({overfit_info['n_subjects_overfitted']}/{len(model_results['per_subject_accuracies'])} 个受试者)")
                
                # 打印每个epoch的平均性能趋势
                if 'mean_val_f1_per_epoch' in best_info:
                    logger.info(f"\n    各Epoch平均验证F1趋势:")
                    for epoch_idx, f1 in enumerate(best_info['mean_val_f1_per_epoch'][:5]):  # 只显示前5个epoch
                        logger.info(f"      Epoch {epoch_idx+1}: {f1:.3f}")
                    if len(best_info['mean_val_f1_per_epoch']) > 5:
                        logger.info(f"      ... (共{len(best_info['mean_val_f1_per_epoch'])}个epoch)")
        
        logger.info(f"\n平均泛化差距: {results['mean_generalization_gap']:.3f}")
        logger.info(f"深度网络优势: {results['deep_network_advantage']:.3f}")
        
        # 打印分脑区汇总
        if 'detailed_summary' in results.get('region_wise_summary', {}):
            summary = results['region_wise_summary']['detailed_summary']
            logger.info(f"\n分脑区评估汇总:")
            logger.info(f"  评估脑区数: {summary['n_evaluated_regions']}")
            logger.info(f"  平均泛化得分: {summary['mean_generalization_score']:.3f}")
            logger.info(f"  泛化最差的脑区: {summary['poor_generalization_regions'][:5]}")
            logger.info(f"  泛化最好的脑区: {summary['good_generalization_regions'][:5]}")