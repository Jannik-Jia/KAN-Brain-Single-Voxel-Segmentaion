#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline测试器
测试随机分割下的分类性能
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.deep_network import DeepNetworkUtils
from common.config import Config

logger = logging.getLogger(__name__)

# 将DeepClassifierWrapper提升为模块级别，避免重复定义
class DeepClassifierWrapper:
    """深度网络分类器包装器"""
    
    def __init__(self, deep_utils, config=None):
        self.deep_utils = deep_utils
        self.config = Config.ALEX_HYPERPARAMS.copy()
        if config:
            self.config.update(config)
        self.network = None
        # 保留原有的training_history结构，同时扩展
        self.training_history = {
            'loss': [], 'accuracy': [],  
            'train_loss': [], 'train_accuracy': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_f1': []
        }
        self.device = deep_utils.device
        self.X_val = None  # 存储验证集
        self.y_val = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        训练模型（扩展版本，向后兼容）
        
        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
        """
        # 存储验证集供后续使用
        self.X_val = X_val
        self.y_val = y_val
        
        # 确定类别数
        if len(np.unique(y)) > 50:  # 脑区分类
            num_classes = 101
        else:  # 受试者分类
            num_classes = len(np.unique(y))
        
        # 创建网络
        self.network = self.deep_utils.create_network(
            input_dim=X.shape[1],
            num_classes=num_classes
        )
        
        # 准备训练数据
        X_tensor, y_tensor = self.deep_utils.prepare_data(X, y)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # 准备验证数据（如果提供）
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self.deep_utils.prepare_data(X_val, y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
        
        # 训练设置
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate']
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # 训练循环
        self.network.train()
        for epoch in range(self.config['no_epochs']):
            epoch_loss = 0
            correct = 0
            total = 0
            all_train_predictions = []
            all_train_labels = []
            
            # 训练阶段
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
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
                
                epoch_loss += total_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # 收集预测结果用于F1计算
                all_train_predictions.extend(predicted.cpu().numpy())
                all_train_labels.extend(batch_y.cpu().numpy())
            
            # 计算训练集指标
            train_accuracy = correct / total
            train_loss_avg = epoch_loss / len(train_loader)
            train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro', zero_division=0)
            
            # 保留原有的历史记录（向后兼容）
            self.training_history['loss'].append(train_loss_avg)
            self.training_history['accuracy'].append(train_accuracy)
            
            # 新的详细历史记录
            self.training_history['train_loss'].append(train_loss_avg)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['train_f1'].append(train_f1)
            
            # 验证阶段（如果有验证集）
            val_loss_avg = 0
            val_accuracy = 0
            val_f1 = 0
            
            if val_loader is not None:
                self.network.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                all_val_predictions = []
                all_val_labels = []
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.network(batch_x)
                        loss = criterion(outputs, batch_y)
                        l2_reg = self.deep_utils.kernel_l2_regularization(
                            self.network, self.config['weight_decay']
                        )
                        total_loss = loss + l2_reg
                        
                        val_loss += total_loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
                        all_val_predictions.extend(predicted.cpu().numpy())
                        all_val_labels.extend(batch_y.cpu().numpy())
                
                val_loss_avg = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                val_f1 = f1_score(all_val_labels, all_val_predictions, average='macro', zero_division=0)
                
                self.network.train()  # 切换回训练模式
            
            # 记录验证集历史
            self.training_history['val_loss'].append(val_loss_avg)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_f1'].append(val_f1)
            
            # 详细的日志输出
            if epoch % 1 == 0:  # 每个epoch都打印
                if val_loader is not None:
                    logger.info(
                        f"    Epoch {epoch+1}/{self.config['no_epochs']} | "
                        f"Train: Loss={train_loss_avg:.4f}, Acc={train_accuracy:.3f}, F1={train_f1:.3f} | "
                        f"Val: Loss={val_loss_avg:.4f}, Acc={val_accuracy:.3f}, F1={val_f1:.3f}"
                    )
                else:
                    logger.info(
                        f"    Epoch {epoch+1}/{self.config['no_epochs']} | "
                        f"Train: Loss={train_loss_avg:.4f}, Acc={train_accuracy:.3f}, F1={train_f1:.3f}"
                    )
    
    def predict(self, X):
        """预测"""
        self.network.eval()
        
        # 分批预测
        batch_size = 1024
        n_samples = len(X)
        predictions = []
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_X = X[start_idx:end_idx]
                
                X_tensor, _ = self.deep_utils.prepare_data(batch_X, None, is_training=False)
                outputs = self.network(X_tensor)
                _, predicted = torch.max(outputs, 1)
                
                predictions.append(predicted.cpu().numpy())
                
                del X_tensor, outputs, predicted
                torch.cuda.empty_cache()
        
        return np.concatenate(predictions)


class BaselineTester:
    """Baseline性能测试器"""
    
    def __init__(self, model_types: List[str] = None, 
                 device: str = 'cpu',
                 save_models: bool = False,
                 models_dir: Path = None):
        self.model_types = model_types or ['rf', 'lr', 'deep']
        self.device = device
        self.save_models = save_models
        self.models_dir = models_dir
        self.skip_deep_network = 'deep' not in self.model_types
        self.deep_utils = DeepNetworkUtils() if not self.skip_deep_network else None
        
    def test_baseline_performance(self, train_data: Dict[str, np.ndarray],
                                val_data: Dict[str, np.ndarray],
                                label_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试baseline性能
        """
        logger.info("开始Baseline性能测试...")
        
        # 准备数据
        X_train = train_data['X_scaled']
        y_train = train_data['y']
        subjects_train = train_data['subjects']
        
        X_val = val_data['X_scaled']
        y_val = val_data['y']
        
        # 转换标签
        y_train_classes = self._convert_labels(y_train, label_mapping)
        y_val_classes = self._convert_labels(y_val, label_mapping)
        
        results = {
            'global_analysis': {},
            'region_wise_analysis': {},
            'comparative_analysis': {},
            'deep_network_detailed_analysis': {}
        }
        
        # 1. 全局分析
        logger.info("\n1. 全局受试者识别分析...")
        results['global_analysis'] = self._test_global_performance(
            X_train, y_train_classes, X_val, y_val_classes, label_mapping,
            X_val_raw=X_val, y_val_raw=y_val_classes  
        )
        
        # 2. 分脑区分析
        logger.info("\n2. 分脑区受试者识别分析...")
        region_dataset = self._build_region_aware_dataset(
            X_train, y_train_classes, subjects_train
        )
        
        if len(region_dataset['features']) > 200:
            results['region_wise_analysis'] = self._test_region_wise_performance(
                region_dataset, label_mapping
            )
        else:
            logger.warning("分脑区样本不足，跳过分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # 3. 对比分析
        logger.info("\n3. 性能对比分析...")
        results['comparative_analysis'] = self._compare_performances(
            results['global_analysis'],
            results['region_wise_analysis']
        )
        
        # 4. 深度网络权威性分析
        if 'Deep4x4096' in results['global_analysis']:
            results['deep_network_detailed_analysis'] = self.analyze_deep_network_authority(
                results, {}
            )

        # 找出最佳模型
        best_model = None
        best_accuracy = 0
        n_classes = label_mapping.get('n_classes_expected', 101)
        
        for model_name, model_results in results['global_analysis'].items():
            if isinstance(model_results, dict) and 'accuracy' in model_results:
                if model_results['accuracy'] > best_accuracy:
                    best_accuracy = model_results['accuracy']
                    best_model = model_name
        
        results['best_model'] = best_model
        results['best_accuracy'] = best_accuracy
        results['n_classes'] = n_classes
        
        # 计算全局vs分脑区的改进
        if (best_model and 
            'region_wise_analysis' in results and 
            best_model in results['region_wise_analysis'] and
            isinstance(results['region_wise_analysis'][best_model], dict)):
            global_acc = results['global_analysis'][best_model]['accuracy']
            region_acc = results['region_wise_analysis'][best_model]['accuracy']
            results['global_vs_region_improvement'] = region_acc - global_acc
        else:
            results['global_vs_region_improvement'] = 0.0
        
        return results

    def _convert_labels(self, y: np.ndarray, label_mapping: Dict[str, Any]) -> np.ndarray:
        """转换标签格式"""
        if label_mapping['is_one_hot']:
            return np.argmax(y, axis=1)
        else:
            return y.flatten().astype(int)
    
    def _test_global_performance(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            label_mapping: Dict[str, Any],
                            X_val_raw: np.ndarray = None,
                            y_val_raw: np.ndarray = None) -> Dict[str, Any]:
        """测试全局性能"""
        
        classifiers = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, C=0.1, random_state=42
            )
        }
        
        if not self.skip_deep_network:
            classifiers['Deep4x4096'] = 'deep'
        
        results = {}
        
        for name, clf in classifiers.items():
            logger.info(f"训练 {name}...")
            start_time = time.time()
            
            try:
                if name == 'Deep4x4096':
                    # 深度网络特殊处理 - 使用验证集
                    clf_wrapper = self._create_deep_classifier_wrapper()
                    # 传递验证集进行训练时监控
                    clf_wrapper.fit(X_train, y_train, X_val_raw, y_val_raw)
                    y_pred = clf_wrapper.predict(X_val)
                    
                    training_history = clf_wrapper.training_history
                    
                    accuracy = accuracy_score(y_val, y_pred)
                    f1_macro = f1_score(y_val, y_pred, average='macro')
                    f1_weighted = f1_score(y_val, y_pred, average='weighted')
                    
                    # 保持原有的输出结构
                    results[name] = {
                        'accuracy': float(accuracy),
                        'f1_macro': float(f1_macro),
                        'f1_weighted': float(f1_weighted),
                        'n_train_samples': len(X_train),
                        'n_test_samples': len(X_val),
                        'n_classes': label_mapping['n_classes_expected'],
                        'training_time': time.time() - start_time,
                        'final_train_accuracy': training_history['accuracy'][-1] if training_history['accuracy'] else 0,
                        'is_deep_network': True,
                        'overfitting_indicator': (
                            training_history['accuracy'][-1] - accuracy 
                            if training_history['accuracy'] else 0
                        )
                    }
                    
                else:
                    # 传统分类器保持完全不变
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    f1_macro = f1_score(y_val, y_pred, average='macro')
                    f1_weighted = f1_score(y_val, y_pred, average='weighted')
                    
                    results[name] = {
                        'accuracy': float(accuracy),
                        'f1_macro': float(f1_macro),
                        'f1_weighted': float(f1_weighted),
                        'n_train_samples': len(X_train),
                        'n_test_samples': len(X_val),
                        'n_classes': label_mapping['n_classes_expected'],
                        'training_time': time.time() - start_time,
                        'is_deep_network': False
                    }
                
                logger.info(f"  {name}: Acc={accuracy:.3f}, F1_macro={f1_macro:.3f}, "
                        f"训练时间={results[name]['training_time']/60:.1f}分钟")
                
            except Exception as e:
                logger.error(f"训练 {name} 失败: {e}")
                logger.exception("详细错误信息:")
                results[name] = {'error': str(e)}
        
        return results
    
    def _build_region_aware_dataset(self, X: np.ndarray, regions: np.ndarray,
                                  subjects: np.ndarray) -> Dict[str, np.ndarray]:
        """
        构建脑区感知数据集
        
        Args:
            X: 特征数据
            regions: 脑区标签（y的类别索引）
            subjects: 受试者标签
            
        Returns:
            脑区感知数据集
        """
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(regions)
        
        X_list = []
        subject_labels = []
        region_labels = []
        sample_counts = []
        
        valid_combinations = 0
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects == subject_id) & (regions == region_id)
                n_voxels = np.sum(mask)
                
                if n_voxels >= Config.MIN_VOXELS_PER_REGION:
                    region_features = np.mean(X[mask], axis=0)
                    X_list.append(region_features)
                    subject_labels.append(subject_id)
                    region_labels.append(region_id)
                    sample_counts.append(n_voxels)
                    valid_combinations += 1
        
        logger.info(f"构建分脑区数据集: {valid_combinations} 个有效组合")
        
        return {
            'features': np.array(X_list),
            'subject_labels': np.array(subject_labels),
            'region_labels': np.array(region_labels),
            'sample_counts': np.array(sample_counts)
        }
    
    def _test_region_wise_performance(self, region_dataset: Dict[str, np.ndarray],
                                    label_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """测试分脑区性能"""
        X_region = region_dataset['features']
        subject_labels = region_dataset['subject_labels']
        
        # 重新映射受试者标签
        unique_subjects = np.unique(subject_labnvidiaels)
        subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
        y_region = np.array([subject_mapping[s] for s in subject_labels])
        
        logger.info(f"分脑区数据集: {len(X_region)} 样本, {len(unique_subjects)} 受试者")
        
        # 训练测试分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_region, y_region, test_size=0.2, stratify=y_region, random_state=42
        )
        
        classifiers = {
            # 'RandomForest': RandomForestClassifier(
            #     n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            # ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, C=0.1, random_state=42
            )
        }
        
        if not self.skip_deep_network:
            classifiers['Deep4x4096'] = 'deep'
        
        results = {}
        
        for name, clf in classifiers.items():
            logger.info(f"训练分脑区 {name}...")
            start_time = time.time()
            
            try:
                if name == 'Deep4x4096':
                    clf_wrapper = self._create_deep_classifier_wrapper(
                        config_override={'batch_size': 64, 'no_epochs': 15}
                    )
                    clf_wrapper.fit(X_train, y_train)
                    y_pred = clf_wrapper.predict(X_test)
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                
                results[name] = {
                    'accuracy': float(accuracy),
                    'f1_macro': float(f1_macro),
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_test),
                    'n_subjects': len(unique_subjects),
                    'n_regions': len(np.unique(region_dataset['region_labels'])),
                    'training_time': time.time() - start_time
                }
                
                logger.info(f"  分脑区 {name}: Acc={accuracy:.3f}, F1={f1_macro:.3f}")
                
            except Exception as e:
                logger.error(f"训练分脑区 {name} 失败: {e}")
                logger.exception("详细错误信息:")
                results[name] = {'error': str(e)}
        
        return results
    
    def _compare_performances(self, global_results: Dict[str, Any],
                            region_results: Dict[str, Any]) -> Dict[str, Any]:
        """对比全局和分脑区性能"""
        comparative_analysis = {}
        
        if region_results.get('insufficient_data'):
            return {'error': 'insufficient_region_data'}
        
        for method in ['RandomForest', 'LogisticRegression', 'Deep4x4096']:
            if method in global_results and method in region_results:
                global_result = global_results[method]
                region_result = region_results[method]
                
                # 检查是否有错误
                if 'error' in global_result or 'error' in region_result:
                    continue
                
                global_acc = global_result['accuracy']
                region_acc = region_result['accuracy']
                
                improvement = region_acc - global_acc
                improvement_percent = improvement / global_acc * 100 if global_acc > 0 else 0
                
                comparative_analysis[method] = {
                    'global_accuracy': global_acc,
                    'region_accuracy': region_acc,
                    'accuracy_improvement': improvement,
                    'improvement_percent': improvement_percent,
                    'interpretation': self._interpret_improvement(improvement_percent)
                }
        
        return comparative_analysis
    
    def _interpret_improvement(self, improvement_percent: float) -> str:
        """解释性能改进"""
        if improvement_percent > 10:
            return "显著提升"
        elif improvement_percent > 5:
            return "中等提升"
        elif improvement_percent > 1:
            return "轻微提升"
        elif improvement_percent > -1:
            return "性能相当"
        else:
            return "性能下降"
    
    def _create_deep_classifier_wrapper(self, config_override: Optional[Dict] = None):
        """创建深度网络分类器包装器"""
        return DeepClassifierWrapper(self.deep_utils, config_override)
    
    def analyze_deep_network_authority(self, baseline_results: Dict[str, Any],
                                     loso_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析深度网络权威性"""
        
        authority_factors = {
            'performance_superiority': 0.0,
            'convergence_quality': 0.0,
            'generalization_ability': 0.0,
            'consistency': 0.0
        }
        
        # 1. 性能优越性
        if 'global_analysis' in baseline_results:
            deep_result = baseline_results['global_analysis'].get('Deep4x4096', {})
            if 'accuracy' in deep_result:
                deep_acc = deep_result['accuracy']
                rf_acc = baseline_results['global_analysis'].get('RandomForest', {}).get('accuracy', 0)
                lr_acc = baseline_results['global_analysis'].get('LogisticRegression', {}).get('accuracy', 0)
                
                if rf_acc > 0 and lr_acc > 0:
                    avg_traditional = (rf_acc + lr_acc) / 2
                    authority_factors['performance_superiority'] = max(0, min(1, 
                        (deep_acc - avg_traditional) / avg_traditional * 2))
        
        # 2. 收敛质量
        if 'Deep4x4096' in baseline_results.get('global_analysis', {}):
            deep_results = baseline_results['global_analysis']['Deep4x4096']
            overfitting = deep_results.get('overfitting_indicator', 0)
            authority_factors['convergence_quality'] = max(0, 1 - abs(overfitting) * 2)
        
        # 3. 泛化能力（如果有LOSO结果）
        if loso_results and 'model_results' in loso_results:
            deep_loso = loso_results['model_results'].get('deep', {})
            if deep_loso and 'mean_accuracy' in deep_loso:
                baseline_deep = baseline_results['global_analysis'].get('Deep4x4096', {}).get('accuracy', 0)
                loso_deep_acc = deep_loso['mean_accuracy']
                
                if baseline_deep > 0:
                    gen_gap = abs(baseline_deep - loso_deep_acc) / baseline_deep
                    authority_factors['generalization_ability'] = max(0, 1 - gen_gap * 2)
        
        # 4. 一致性
        if ('global_analysis' in baseline_results and 
            'region_wise_analysis' in baseline_results):
            
            global_deep = baseline_results['global_analysis'].get('Deep4x4096', {}).get('accuracy', 0)
            region_deep = baseline_results['region_wise_analysis'].get('Deep4x4096', {}).get('accuracy', 0)
            
            if global_deep > 0 and region_deep > 0:
                consistency = 1 - abs(global_deep - region_deep) / max(global_deep, region_deep)
                authority_factors['consistency'] = max(0, consistency)
        
        # 综合权威性评分
        weights = {
            'performance_superiority': 0.3,
            'convergence_quality': 0.2,
            'generalization_ability': 0.3,
            'consistency': 0.2
        }
        
        overall_authority = sum(
            authority_factors[factor] * weights[factor] 
            for factor in authority_factors
        )
        
        # 权威性等级
        if overall_authority > 0.8:
            authority_level = "AUTHORITATIVE"
            recommendation = "深度网络结果具有决定性权威"
        elif overall_authority > 0.6:
            authority_level = "HIGHLY_CREDIBLE"
            recommendation = "深度网络结果高度可信"
        elif overall_authority > 0.4:
            authority_level = "MODERATELY_CREDIBLE"
            recommendation = "深度网络结果中等可信"
        else:
            authority_level = "LIMITED_CREDIBILITY"
            recommendation = "深度网络结果可信度有限"
        
        return {
            'overall_authority': float(overall_authority),
            'authority_level': authority_level,
            'recommendation': recommendation,
            'factor_breakdown': authority_factors
        }