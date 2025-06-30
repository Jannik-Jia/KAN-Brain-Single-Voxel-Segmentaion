#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline性能测试器
测试各种分类器的基准性能
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

from common.deep_network import DeepNetworkUtils
from common.config import Config
from common.data_io import DataIO

logger = logging.getLogger(__name__)


class BaselineTester:
    """Baseline性能测试器"""
    
    def __init__(self, model_types: List[str] = None,
                 device: str = 'cpu',
                 save_models: bool = False,
                 models_dir: Optional[Path] = None):
        """
        初始化测试器
        
        Args:
            model_types: 要测试的模型类型列表
            device: 计算设备
            save_models: 是否保存训练好的模型
            models_dir: 模型保存目录
        """
        self.model_types = model_types or ['rf', 'lr', 'deep']
        self.device = device
        self.save_models = save_models
        self.models_dir = models_dir
        self.deep_utils = DeepNetworkUtils()
        
        # 模型配置
        self.model_configs = {
            'rf': {
                'name': 'RandomForest',
                'model_class': RandomForestClassifier,
                'params': {'n_estimators': 200, 'max_depth': 20, 
                          'min_samples_split': 2, 'random_state': 42, 'n_jobs': -1}
            },
            'lr': {
                'name': 'LogisticRegression',
                'model_class': LogisticRegression,
                'params': {'max_iter': 1000, 'C': 0.1, 'random_state': 42}
            },
            'deep': {
                'name': 'Deep4x4096',
                'model_class': 'deep_network',
                'params': Config.ALEX_HYPERPARAMS.copy()
            },
            'deep_lightweight': {
                'name': 'DeepLightweight',
                'model_class': 'deep_network',
                'params': {
                    'batch_size': 64,
                    'no_epochs': 15,
                    'learning_rate': 0.0001,
                    'weight_decay': 0.00001,
                    'dropout_rate': 0.5
                }
            }
        }
    
    def test_baseline_performance(self, X_train: np.ndarray, y_train: np.ndarray,
                                subjects_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                subjects_val: np.ndarray,
                                label_mapping: Dict) -> Dict[str, Any]:
        """
        测试baseline性能
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            subjects_train: 训练受试者
            X_val: 验证特征
            y_val: 验证标签
            subjects_val: 验证受试者
            label_mapping: 标签映射信息
            
        Returns:
            测试结果
        """
        logger.info("开始Baseline性能测试...")
        
        results = {
            'model_results': {},
            'best_model': None,
            'best_accuracy': 0.0,
            'n_classes': label_mapping['n_classes_expected']
        }
        
        # 1. 全局性能测试
        logger.info("\n1. 测试全局分类性能...")
        global_results = self._test_global_performance(
            X_train, y_train, X_val, y_val, label_mapping
        )
        
        # 2. 分脑区性能测试
        logger.info("\n2. 测试分脑区受试者识别性能...")
        region_results = self._test_region_aware_performance(
            X_train, y_train, subjects_train,
            X_val, y_val, subjects_val,
            label_mapping
        )
        
        # 3. 性能对比分析
        logger.info("\n3. 分析全局vs分脑区性能...")
        comparison_results = self._compare_performances(global_results, region_results)
        
        # 整合结果
        for model_type in self.model_types:
            if model_type in global_results and model_type in region_results:
                results['model_results'][model_type] = {
                    'global_accuracy': global_results[model_type]['accuracy'],
                    'global_f1': global_results[model_type]['f1_macro'],
                    'region_accuracy': region_results[model_type]['accuracy'],
                    'region_f1': region_results[model_type]['f1_macro'],
                    'accuracy_improvement': comparison_results[model_type]['accuracy_improvement'],
                    'training_time': global_results[model_type].get('training_time', 0)
                }
                
                # 更新最佳模型
                if global_results[model_type]['accuracy'] > results['best_accuracy']:
                    results['best_accuracy'] = global_results[model_type]['accuracy']
                    results['best_model'] = model_type
        
        results['global_vs_region_improvement'] = comparison_results.get('overall_improvement', 0)
        
        # 打印摘要
        self._print_baseline_summary(results)
        
        return results
    
    def _test_global_performance(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               label_mapping: Dict) -> Dict[str, Any]:
        """测试全局分类性能"""
        results = {}
        
        # 处理标签
        if label_mapping['is_one_hot']:
            y_train_classes = np.argmax(y_train, axis=1)
            y_val_classes = np.argmax(y_val, axis=1)
        else:
            y_train_classes = y_train.flatten()
            y_val_classes = y_val.flatten()
        
        for model_type in self.model_types:
            if model_type not in self.model_configs:
                continue
            
            logger.info(f"  测试 {self.model_configs[model_type]['name']}...")
            
            try:
                start_time = time.time()
                
                if self.model_configs[model_type]['model_class'] == 'deep_network':
                    # 深度网络
                    model = self._create_deep_classifier(
                        self.model_configs[model_type]['params']
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # 处理深度网络的预测结果
                    if label_mapping['is_one_hot']:
                        y_val_for_eval = y_val_classes
                    else:
                        y_val_for_eval = y_val_classes
                    
                    accuracy = accuracy_score(y_val_for_eval, y_pred)
                    f1_macro = f1_score(y_val_for_eval, y_pred, average='macro')
                    
                else:
                    # 传统模型
                    model_class = self.model_configs[model_type]['model_class']
                    model = model_class(**self.model_configs[model_type]['params'])
                    model.fit(X_train, y_train_classes)
                    y_pred = model.predict(X_val)
                    
                    accuracy = accuracy_score(y_val_classes, y_pred)
                    f1_macro = f1_score(y_val_classes, y_pred, average='macro')
                
                training_time = time.time() - start_time
                
                results[model_type] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'training_time': training_time,
                    'model': model if self.save_models else None
                }
                
                logger.info(f"    准确率: {accuracy:.3f}, F1: {f1_macro:.3f}, 训练时间: {training_time:.1f}s")
                
                # 保存模型
                if self.save_models and self.models_dir:
                    self._save_model(model, model_type, 'global')
                
            except Exception as e:
                logger.error(f"    {model_type} 测试失败: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def _test_region_aware_performance(self, X_train: np.ndarray, y_train: np.ndarray,
                                     subjects_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray,
                                     subjects_val: np.ndarray,
                                     label_mapping: Dict) -> Dict[str, Any]:
        """测试分脑区受试者识别性能"""
        
        # 构建分脑区数据集
        region_data = self._build_region_aware_dataset(
            X_train, y_train, subjects_train,
            X_val, y_val, subjects_val,
            label_mapping
        )
        
        if not region_data:
            logger.warning("  分脑区数据集构建失败")
            return {}
        
        X_train_region = region_data['X_train']
        y_train_region = region_data['y_train']
        X_val_region = region_data['X_val']
        y_val_region = region_data['y_val']
        
        logger.info(f"  分脑区数据集: 训练{len(X_train_region)}样本, 验证{len(X_val_region)}样本")
        logger.info(f"  受试者类别数: {len(np.unique(y_train_region))}")
        
        results = {}
        
        for model_type in self.model_types:
            if model_type not in self.model_configs:
                continue
            
            logger.info(f"  测试 {self.model_configs[model_type]['name']} (分脑区)...")
            
            try:
                start_time = time.time()
                
                if self.model_configs[model_type]['model_class'] == 'deep_network':
                    # 深度网络 - 为分脑区调整参数
                    params = self.model_configs[model_type]['params'].copy()
                    if model_type == 'deep':
                        params['batch_size'] = 64  # 分脑区数据较少
                        params['no_epochs'] = 20
                    
                    model = self._create_deep_classifier(params)
                    model.fit(X_train_region, y_train_region)
                    y_pred = model.predict(X_val_region)
                    
                else:
                    # 传统模型
                    model_class = self.model_configs[model_type]['model_class']
                    model = model_class(**self.model_configs[model_type]['params'])
                    model.fit(X_train_region, y_train_region)
                    y_pred = model.predict(X_val_region)
                
                accuracy = accuracy_score(y_val_region, y_pred)
                f1_macro = f1_score(y_val_region, y_pred, average='macro')
                
                training_time = time.time() - start_time
                
                results[model_type] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'training_time': training_time,
                    'n_subjects': len(np.unique(y_train_region)),
                    'n_regions': region_data['n_regions']
                }
                
                logger.info(f"    准确率: {accuracy:.3f}, F1: {f1_macro:.3f}, 训练时间: {training_time:.1f}s")
                
                # 保存模型
                if self.save_models and self.models_dir:
                    self._save_model(model, model_type, 'region')
                
            except Exception as e:
                logger.error(f"    {model_type} 测试失败: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def _build_region_aware_dataset(self, X_train: np.ndarray, y_train: np.ndarray,
                                  subjects_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  subjects_val: np.ndarray,
                                  label_mapping: Dict) -> Optional[Dict]:
        """构建分脑区数据集"""
        
        # 处理标签
        if label_mapping['is_one_hot']:
            y_train_regions = np.argmax(y_train, axis=1)
            y_val_regions = np.argmax(y_val, axis=1)
        else:
            y_train_regions = y_train.flatten()
            y_val_regions = y_val.flatten()
        
        # 构建训练集
        X_train_list = []
        y_train_list = []
        
        unique_regions = np.unique(y_train_regions)
        unique_subjects = np.unique(subjects_train)
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects_train == subject_id) & (y_train_regions == region_id)
                if np.sum(mask) >= Config.MIN_SAMPLES_FOR_SUBJECT_REGION:
                    region_features = np.mean(X_train[mask], axis=0)
                    X_train_list.append(region_features)
                    y_train_list.append(subject_id)
        
        # 构建验证集
        X_val_list = []
        y_val_list = []
        
        unique_subjects_val = np.unique(subjects_val)
        
        for subject_id in unique_subjects_val:
            for region_id in unique_regions:
                mask = (subjects_val == subject_id) & (y_val_regions == region_id)
                if np.sum(mask) >= Config.MIN_SAMPLES_FOR_SUBJECT_REGION:
                    region_features = np.mean(X_val[mask], axis=0)
                    X_val_list.append(region_features)
                    y_val_list.append(subject_id)
        
        if len(X_train_list) < 100 or len(X_val_list) < 20:
            return None
        
        # 重新映射受试者标签
        all_subjects = sorted(list(set(y_train_list + y_val_list)))
        subject_mapping = {sid: i for i, sid in enumerate(all_subjects)}
        
        y_train_mapped = [subject_mapping[sid] for sid in y_train_list]
        y_val_mapped = [subject_mapping[sid] for sid in y_val_list]
        
        return {
            'X_train': np.array(X_train_list),
            'y_train': np.array(y_train_mapped),
            'X_val': np.array(X_val_list),
            'y_val': np.array(y_val_mapped),
            'n_subjects': len(all_subjects),
            'n_regions': len(unique_regions),
            'subject_mapping': subject_mapping
        }
    
    def _compare_performances(self, global_results: Dict, region_results: Dict) -> Dict:
        """比较全局和分脑区性能"""
        comparison = {}
        
        improvements = []
        
        for model_type in self.model_types:
            if model_type in global_results and model_type in region_results:
                if 'accuracy' in global_results[model_type] and 'accuracy' in region_results[model_type]:
                    global_acc = global_results[model_type]['accuracy']
                    region_acc = region_results[model_type]['accuracy']
                    
                    improvement = region_acc - global_acc
                    relative_improvement = improvement / global_acc if global_acc > 0 else 0
                    
                    comparison[model_type] = {
                        'accuracy_improvement': improvement,
                        'relative_improvement': relative_improvement,
                        'interpretation': self._interpret_improvement(improvement, relative_improvement)
                    }
                    
                    improvements.append(improvement)
        
        comparison['overall_improvement'] = np.mean(improvements) if improvements else 0
        
        return comparison
    
    def _interpret_improvement(self, improvement: float, relative_improvement: float) -> str:
        """解释性能改进"""
        if relative_improvement > 0.1:
            return "显著提升：分脑区方法明显优于全局方法"
        elif relative_improvement > 0.05:
            return "中等提升：分脑区方法有一定优势"
        elif relative_improvement > 0.01:
            return "轻微提升：分脑区方法略有优势"
        elif relative_improvement > -0.01:
            return "性能相当：两种方法差异不大"
        else:
            return "性能下降：全局方法可能更适合"
    
    def _create_deep_classifier(self, params: Dict) -> Any:
        """创建深度网络分类器包装器"""
        
        class DeepClassifierWrapper:
            def __init__(self, deep_utils, params):
                self.deep_utils = deep_utils
                self.params = params
                self.network = None
                self.training_history = {'loss': [], 'accuracy': []}
            
            def fit(self, X, y):
                # 确定类别数
                if len(y.shape) > 1 and y.shape[1] > 1:
                    num_classes = y.shape[1]
                else:
                    num_classes = len(np.unique(y))
                
                # 创建网络
                self.network = self.deep_utils.create_network(
                    input_dim=X.shape[1],
                    num_classes=num_classes
                )
                
                # 准备数据
                X_tensor, y_tensor = self.deep_utils.prepare_data(X, y, is_training=True)
                
                # 创建数据加载器
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.params['batch_size'],
                    shuffle=True
                )
                
                # 训练
                optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.params['learning_rate']
                )
                criterion = torch.nn.CrossEntropyLoss()
                
                self.network.train()
                for epoch in range(self.params['no_epochs']):
                    epoch_loss = 0
                    correct = 0
                    total = 0
                    
                    for batch_x, batch_y in dataloader:
                        optimizer.zero_grad()
                        
                        outputs = self.network(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        # L2正则化
                        l2_reg = self.deep_utils.kernel_l2_regularization(
                            self.network, self.params['weight_decay']
                        )
                        total_loss = loss + l2_reg
                        
                        total_loss.backward()
                        optimizer.step()
                        
                        epoch_loss += total_loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                    
                    accuracy = correct / total
                    self.training_history['loss'].append(epoch_loss / len(dataloader))
                    self.training_history['accuracy'].append(accuracy)
                    
                    if epoch % 5 == 0:
                        logger.info(f"      Epoch {epoch}/{self.params['no_epochs']}, "
                                  f"Loss: {epoch_loss/len(dataloader):.4f}, "
                                  f"Acc: {accuracy:.3f}")
            
            def predict(self, X):
                self.network.eval()
                X_tensor, _ = self.deep_utils.prepare_data(X, None, is_training=False)
                
                with torch.no_grad():
                    outputs = self.network(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                
                return predicted.cpu().numpy()
        
        return DeepClassifierWrapper(self.deep_utils, params)
    
    def _save_model(self, model, model_type: str, mode: str):
        """保存模型"""
        if not self.models_dir:
            return
        
        filename = self.models_dir / f"{model_type}_{mode}_model.pkl"
        
        try:
            if hasattr(model, 'network'):
                # 深度网络 - 保存state dict
                torch.save(model.network.state_dict(), 
                          self.models_dir / f"{model_type}_{mode}_model.pth")
            else:
                # sklearn模型
                DataIO.save_pickle(model, filename)
            
            logger.info(f"    模型已保存: {filename}")
        except Exception as e:
            logger.warning(f"    模型保存失败: {e}")
    
    def _print_baseline_summary(self, results: Dict):
        """打印baseline测试摘要"""
        logger.info("\nBaseline测试摘要:")
        logger.info("="*50)
        
        for model_type, model_results in results['model_results'].items():
            logger.info(f"\n{model_type}:")
            logger.info(f"  全局准确率: {model_results['global_accuracy']:.3f}")
            logger.info(f"  分脑区准确率: {model_results['region_accuracy']:.3f}")
            logger.info(f"  性能提升: {model_results['accuracy_improvement']:.3f}")
        
        logger.info(f"\n最佳模型: {results['best_model']} (准确率: {results['best_accuracy']:.3f})")
        logger.info(f"整体性能提升: {results['global_vs_region_improvement']:.3f}")