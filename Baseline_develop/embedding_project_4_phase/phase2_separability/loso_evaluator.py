#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOSO评估器
实现Leave-One-Subject-Out交叉验证
"""

import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.config import Config
from common.deep_network import DeepNetworkUtils

logger = logging.getLogger(__name__)


class LOSOEvaluator:
    """LOSO评估器"""
    
    def __init__(self, model_types: List[str] = None,
                 device: str = 'cpu',
                 max_subjects: int = 8):
        """
        初始化LOSO评估器
        
        Args:
            model_types: 要测试的模型类型
            device: 计算设备
            max_subjects: 最大测试受试者数（节省时间）
        """
        self.model_types = model_types or ['rf', 'lr', 'deep']
        self.device = device
        self.max_subjects = max_subjects
        self.deep_utils = DeepNetworkUtils()
    
    def evaluate_loso_performance(self, X: np.ndarray, y: np.ndarray, 
                                subjects: np.ndarray,
                                label_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行LOSO评估
        
        Args:
            X: 特征数据
            y: 标签数据
            subjects: 受试者标签
            label_mapping: 标签映射信息
            
        Returns:
            LOSO评估结果
        """
        logger.info("开始Leave-One-Subject-Out评估...")
        
        # 转换标签
        if label_mapping['is_one_hot']:
            y_classes = np.argmax(y, axis=1)
        else:
            y_classes = y.flatten().astype(int)
        
        results = {
            'model_results': {},
            'subject_generalization_gaps': {},
            'mean_generalization_gap': 0.0,
            'deep_network_advantage': 0.0
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
            if model_type in global_loso_results and model_type in region_loso_results:
                results['model_results'][model_type] = {
                    'mean_accuracy': global_loso_results[model_type]['mean_accuracy'],
                    'std_accuracy': global_loso_results[model_type]['std_accuracy'],
                    'per_subject_accuracies': global_loso_results[model_type]['accuracies'],
                    'region_wise_accuracy': region_loso_results[model_type]['mean_accuracy'],
                    'generalization_gap': global_loso_results[model_type]['generalization_gap']
                }
        
        # 计算平均泛化差距
        gaps = [r['generalization_gap'] for r in results['model_results'].values()]
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
    
    def _evaluate_global_loso(self, X: np.ndarray, y: np.ndarray, 
                            subjects: np.ndarray) -> Dict[str, Any]:
        """
        评估全局LOSO性能
        注意：这里的y是脑区标签，不是受试者标签
        LOSO是在不同受试者上测试脑区分类的泛化能力
        """
        unique_subjects = np.unique(subjects)
        
        # 限制测试的受试者数量
        test_subjects = unique_subjects[:self.max_subjects]
        logger.info(f"测试 {len(test_subjects)} 个受试者的LOSO性能")
        
        # 初始化结果存储
        model_results = {}
        for model_type in self.model_types:
            model_results[model_type] = {
                'accuracies': [],
                'f1_scores': [],
                'training_times': []
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
                start_time = time.time()
                
                try:
                    if model_type == 'rf':
                        model = RandomForestClassifier(
                            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                    elif model_type == 'lr':
                        model = LogisticRegression(
                            max_iter=1000, C=0.1, random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                    elif model_type == 'deep':
                        # 深度网络
                        model = self._create_deep_classifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    training_time = time.time() - start_time
                    
                    model_results[model_type]['accuracies'].append(accuracy)
                    model_results[model_type]['f1_scores'].append(f1)
                    model_results[model_type]['training_times'].append(training_time)
                    
                    logger.info(f"    {model_type}: Acc={accuracy:.3f}, F1={f1:.3f}")
                    
                except Exception as e:
                    logger.error(f"    {model_type} 失败: {e}")
        
        # 计算统计信息
        final_results = {}
        for model_type, results in model_results.items():
            if results['accuracies']:
                final_results[model_type] = {
                    'mean_accuracy': np.mean(results['accuracies']),
                    'std_accuracy': np.std(results['accuracies']),
                    'mean_f1': np.mean(results['f1_scores']),
                    'std_f1': np.std(results['f1_scores']),
                    'mean_training_time': np.mean(results['training_times']),
                    'accuracies': results['accuracies'],
                    'n_subjects_tested': len(results['accuracies']),
                    'generalization_gap': 0  # 将在后续计算
                }
        
        return final_results
    
    def _evaluate_region_wise_loso(self, X: np.ndarray, y: np.ndarray,
                                 subjects: np.ndarray) -> Dict[str, Any]:
        """评估分脑区LOSO性能"""
        # 构建分脑区数据集
        region_dataset = self._build_region_dataset(X, y, subjects)
        
        if len(region_dataset['features']) < 500:
            logger.warning("分脑区数据不足，跳过LOSO分析")
            return {}
        
        X_region = region_dataset['features']
        subject_labels = region_dataset['subject_labels']
        
        # 重新映射受试者标签
        unique_subjects = np.unique(subject_labels)
        subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
        y_region = np.array([subject_mapping[s] for s in subject_labels])
        
        logger.info(f"分脑区数据集: {len(X_region)} 样本, {len(unique_subjects)} 受试者")
        
        # 限制测试受试者
        test_subjects = unique_subjects[:min(self.max_subjects, len(unique_subjects))]
        
        # 初始化结果
        model_results = {}
        for model_type in self.model_types:
            model_results[model_type] = {
                'accuracies': [],
                'f1_scores': []
            }
        
        # LOSO评估
        for test_subject in test_subjects:
            test_subject_mapped = subject_mapping[test_subject]
            
            train_mask = y_region != test_subject_mapped
            test_mask = y_region == test_subject_mapped
            
            if np.sum(test_mask) < 10:
                continue
            
            X_train = X_region[train_mask]
            y_train = y_region[train_mask]
            X_test = X_region[test_mask]
            y_test = y_region[test_mask]
            
            for model_type in self.model_types:
                try:
                    if model_type == 'rf':
                        model = RandomForestClassifier(
                            n_estimators=50, max_depth=10, random_state=42
                        )
                    elif model_type == 'lr':
                        model = LogisticRegression(
                            max_iter=500, C=0.1, random_state=42
                        )
                    elif model_type == 'deep':
                        model = self._create_deep_classifier(
                            config_override={'batch_size': 32, 'no_epochs': 15}
                        )
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    
                    model_results[model_type]['accuracies'].append(accuracy)
                    model_results[model_type]['f1_scores'].append(f1)
                    
                except Exception as e:
                    logger.error(f"分脑区 {model_type} 失败: {e}")
        
        # 计算统计
        final_results = {}
        for model_type, results in model_results.items():
            if results['accuracies']:
                final_results[model_type] = {
                    'mean_accuracy': np.mean(results['accuracies']),
                    'std_accuracy': np.std(results['accuracies']),
                    'mean_f1': np.mean(results['f1_scores']),
                    'std_f1': np.std(results['f1_scores'])
                }
        
        return final_results
    
    def _build_region_dataset(self, X: np.ndarray, regions: np.ndarray,
                            subjects: np.ndarray) -> Dict[str, np.ndarray]:
        """构建分脑区数据集"""
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(regions)
        
        X_list = []
        subject_labels = []
        region_labels = []
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects == subject_id) & (regions == region_id)
                n_voxels = np.sum(mask)
                
                if n_voxels >= Config.MIN_VOXELS_PER_REGION:
                    region_features = np.mean(X[mask], axis=0)
                    X_list.append(region_features)
                    subject_labels.append(subject_id)
                    region_labels.append(region_id)
        
        return {
            'features': np.array(X_list),
            'subject_labels': np.array(subject_labels),
            'region_labels': np.array(region_labels)
        }
    
    def _create_deep_classifier(self, config_override: Optional[Dict] = None):
        """创建深度网络分类器包装器"""
        
        class DeepClassifierWrapper:
            def __init__(self, deep_utils, config=None):
                self.deep_utils = deep_utils
                self.config = Config.ALEX_HYPERPARAMS.copy()
                if config:
                    self.config.update(config)
                self.network = None
            
            def fit(self, X, y):
                # 确定类别数
                num_classes = len(np.unique(y))
                
                # 创建网络
                self.network = self.deep_utils.create_network(
                    input_dim=X.shape[1],
                    num_classes=num_classes
                )
                
                # 准备数据
                X_tensor, y_tensor = self.deep_utils.prepare_data(X, y)
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=True
                )
                
                # 训练
                optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.config['learning_rate']
                )
                criterion = torch.nn.CrossEntropyLoss()
                
                self.network.train()
                for epoch in range(self.config['no_epochs']):
                    for batch_x, batch_y in dataloader:
                        optimizer.zero_grad()
                        
                        outputs = self.network(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        # L2正则化
                        l2_reg = self.deep_utils.kernel_l2_regularization(
                            self.network, self.config['weight_decay']
                        )
                        total_loss = loss + l2_reg
                        
                        total_loss.backward()
                        optimizer.step()
            
            def predict(self, X):
                self.network.eval()
                X_tensor, _ = self.deep_utils.prepare_data(X, None, is_training=False)
                
                with torch.no_grad():
                    outputs = self.network(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                
                return predicted.cpu().numpy()
        
        return DeepClassifierWrapper(self.deep_utils, config_override)
    
    def _print_loso_summary(self, results: Dict):
        """打印LOSO评估摘要"""
        logger.info("\nLOSO评估摘要:")
        logger.info("="*50)
        
        for model_type, model_results in results['model_results'].items():
            logger.info(f"\n{model_type}:")
            logger.info(f"  平均准确率: {model_results['mean_accuracy']:.3f} ± {model_results['std_accuracy']:.3f}")
            logger.info(f"  泛化差距: {model_results['generalization_gap']:.3f}")
        
        logger.info(f"\n平均泛化差距: {results['mean_generalization_gap']:.3f}")
        logger.info(f"深度网络优势: {results['deep_network_advantage']:.3f}")