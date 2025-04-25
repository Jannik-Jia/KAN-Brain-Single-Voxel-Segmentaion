import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import h5py
import sys
from typing import Dict, List, Tuple, Any, Optional
import time
from itertools import combinations
from copy import deepcopy
from models.classifiers.classification_head import ClassificationHead
# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.logging_utils import Logger
from utils.input_validator import InputValidator

class GroupEvaluator:
    """特征组评估工具，用于分析不同特征组对模型性能的贡献"""
    
    def __init__(self, model_template, config_path=None, output_dir=None, logger=None, device=None):
        """
        初始化特征组评估器
        
        参数:
            model_template: 模板模型，用于创建不同特征组的模型
            config_path: 配置文件路径
            output_dir: 输出目录
            logger: 日志记录器
            device: 训练设备
        """
        # 加载配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # 设置输出目录
        self.output_dir = output_dir or self.config.get('output_dir', 'results/evaluation/groups')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        if logger:
            self.logger = logger
        else:
            log_manager = Logger("GroupEvaluator", log_dir="logs/evaluation")
            self.logger = log_manager.get_logger()
        
        # 设置设备
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存模型模板
        self.model_template = model_template
        
        # 初始化输入验证器
        self.validator = InputValidator(logger=self.logger)
        
        # 训练参数
        self.num_epochs = self.config.get('num_epochs', 30)
        self.batch_size = self.config.get('batch_size', 64)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.early_stopping = self.config.get('early_stopping', 5)
        
        # 记录训练结果
        self.evaluation_results = {}
        
        self.logger.info(f"特征组评估器初始化完成，设备: {self.device}, 输出目录: {self.output_dir}")
    
    def load_data(self, data_path, split='all'):
        """
        加载HDF5格式的数据
        
        参数:
            data_path: 数据文件路径
            split: 'train', 'val', 'test'或'all'
            
        返回:
            data_dict: 包含各集合特征和标签的字典
        """
        self.logger.info(f"从 {data_path} 加载数据 (split={split})")
        data_dict = {}
        
        try:
            with h5py.File(data_path, 'r') as f:
                # 读取所有组
                def visit_group(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # 排除不需要的split
                        if split != 'all' and split not in name:
                            return
                            
                        # 将数据集路径解析为键
                        parts = name.split('/')
                        
                        # 创建嵌套字典
                        current_dict = data_dict
                        for i, part in enumerate(parts[:-1]):
                            if part not in current_dict:
                                current_dict[part] = {}
                            current_dict = current_dict[part]
                        
                        # 添加数据集
                        current_dict[parts[-1]] = obj[()]
                        self.logger.debug(f"加载数据集: {name}, 形状: {obj[()].shape}")
                
                # 遍历所有组和数据集
                f.visititems(visit_group)
                
            self.logger.info("数据加载完成")
            return data_dict
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise
    
    def prepare_data_loaders(self, data_dict, include_groups=None, exclude_groups=None):
        """
        准备数据加载器
        
        参数:
            data_dict: 数据字典
            include_groups: 要包含的特征组列表，None表示包含所有
            exclude_groups: 要排除的特征组列表
            
        返回:
            data_loaders: 包含训练、验证和测试数据加载器的字典
            feature_dims: 特征维度字典
        """
        self.logger.info("准备数据加载器")
        
        # 提取特征和标签
        train_features, train_labels = {}, None
        val_features, val_labels = {}, None
        test_features, test_labels = {}, None
        
        # 处理包含和排除的特征组
        if include_groups is None:
            include_groups = []
            # 自动检测所有特征组
            for key in data_dict.keys():
                if key.startswith('group_'):
                    group_name = key.replace('group_', '')
                    include_groups.append(group_name)
            
            if not include_groups:
                # 如果没有找到特征组，尝试使用全部特征
                include_groups = ['all']
        
        exclude_groups = exclude_groups or []
        
        # 过滤特征组
        active_groups = [group for group in include_groups if group not in exclude_groups]
        self.logger.info(f"使用特征组: {active_groups}")
        
        # 提取特征
        for group in active_groups:
            if group == 'all':
                # 使用全部特征
                if 'train' in data_dict and 'features' in data_dict['train']:
                    train_features['all'] = data_dict['train']['features']
                if 'val' in data_dict and 'features' in data_dict['val']:
                    val_features['all'] = data_dict['val']['features']
                if 'test' in data_dict and 'features' in data_dict['test']:
                    test_features['all'] = data_dict['test']['features']
            else:
                # 使用特定特征组
                group_key = f'group_{group}'
                if group_key in data_dict:
                    if 'train' in data_dict[group_key] and 'features' in data_dict[group_key]['train']:
                        train_features[group] = data_dict[group_key]['train']['features']
                    if 'val' in data_dict[group_key] and 'features' in data_dict[group_key]['val']:
                        val_features[group] = data_dict[group_key]['val']['features']
                    if 'test' in data_dict[group_key] and 'features' in data_dict[group_key]['test']:
                        test_features[group] = data_dict[group_key]['test']['features']
        
        # 提取标签
        if 'train' in data_dict and 'labels' in data_dict['train']:
            train_labels = data_dict['train']['labels']
        if 'val' in data_dict and 'labels' in data_dict['val']:
            val_labels = data_dict['val']['labels']
        if 'test' in data_dict and 'labels' in data_dict['test']:
            test_labels = data_dict['test']['labels']
        
        # 验证数据
        if not train_features or train_labels is None:
            error_msg = "无法找到训练数据"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 创建数据加载器和记录特征维度
        data_loaders = {}
        feature_dims = {}
        
        # 训练集
        train_dataset = self._create_dataset(train_features, train_labels)
        data_loaders['train'] = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # 记录特征维度
        for group, features in train_features.items():
            feature_dims[group] = features.shape[1]
        
        # 验证集
        if val_features and val_labels is not None:
            val_dataset = self._create_dataset(val_features, val_labels)
            data_loaders['val'] = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
        
        # 测试集
        if test_features and test_labels is not None:
            test_dataset = self._create_dataset(test_features, test_labels)
            data_loaders['test'] = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
        
        self.logger.info(f"数据加载器准备完成, 特征维度: {feature_dims}")
        return data_loaders, feature_dims
    
    def _create_dataset(self, features_dict, labels):
        """创建PyTorch数据集"""
        # 转换为PyTorch张量
        features_tensors = {group: torch.FloatTensor(features) 
                          for group, features in features_dict.items()}
        labels_tensor = torch.LongTensor(labels)
        
        # 创建数据集
        return FeatureGroupDataset(features_tensors, labels_tensor)
    
    def train_and_evaluate(self, data_loaders, feature_dims, active_groups):
        """
        训练和评估使用特定特征组的模型
        
        参数:
            data_loaders: 数据加载器字典
            feature_dims: 特征维度字典
            active_groups: 当前活跃的特征组
            
        返回:
            metrics: 评估指标字典
        """
        self.logger.info(f"训练和评估模型 (特征组: {active_groups})")
        
        # 创建模型
        model = self._create_model(feature_dims, active_groups)
        
        # 训练模型
        model, history = self._train_model(model, data_loaders)
        
        # 评估模型
        metrics = self._evaluate_model(model, data_loaders.get('test', data_loaders.get('val')), 
                                    active_groups)
        
        # 保存训练历史
        metrics['history'] = history
        
        # 组合名称
        group_key = '_'.join(sorted(active_groups))
        self.evaluation_results[group_key] = metrics
        
        # 保存评估结果
        self._save_evaluation_result(group_key, metrics)
        
        return metrics
    
    def _create_model(self, feature_dims, active_groups):
        """创建模型"""
        self.logger.info(f"创建模型，特征组: {active_groups}")
        
        # 复制模板模型
        model = deepcopy(self.model_template)
        
        # 设置活跃特征组的维度
        active_dims = {group: feature_dims[group] for group in active_groups}
        
        # 模型特定的初始化逻辑
        if hasattr(model, 'input_dims'):
            model.input_dims = active_dims
        
        # 确保模型在正确的设备上
        model = model.to(self.device)
        
        return model
    
    def _train_model(self, model, data_loaders):
        """训练模型"""
        self.logger.info(f"开始训练模型，Epochs: {self.num_epochs}")
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1_macro': [],  # 添加宏平均F1记录
            'val_f1_macro': [],    # 添加宏平均F1记录
        }
        
        # 用于早停的变量 - 修改为使用F1分数
        best_val_f1 = 0.0  # 改为F1分数，初始值设为0
        best_model_state = None
        patience_counter = 0
        
        # 训练循环
        for epoch in range(self.num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in data_loaders['train']:
                # 提取特征和标签
                features, labels = batch
                
                # 特征字典移到GPU
                features = {group: tensor.to(self.device) for group, tensor in features.items()}
                labels = labels.to(self.device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                
                # 收集预测和标签用于计算F1
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # 计算训练指标
            train_total = len(train_labels)
            avg_train_loss = train_loss / train_total
            train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
            train_f1 = f1_score(train_labels, train_preds, average='macro')  # 计算宏平均F1
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            if 'val' in data_loaders:
                with torch.no_grad():
                    for batch in data_loaders['val']:
                        # 提取特征和标签
                        features, labels = batch
                        
                        # 特征字典移到GPU
                        features = {group: tensor.to(self.device) for group, tensor in features.items()}
                        labels = labels.to(self.device)
                        
                        # 前向传播
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        
                        # 统计
                        val_loss += loss.item() * labels.size(0)
                        _, predicted = outputs.max(1)
                        
                        # 收集预测和标签
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                # 计算验证指标
                val_total = len(val_labels)
                avg_val_loss = val_loss / val_total
                val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
                val_f1 = f1_score(val_labels, val_preds, average='macro')  # 计算宏平均F1
                
                # 记录历史
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['train_f1_macro'].append(train_f1)
                history['val_f1_macro'].append(val_f1)
                
                # 打印进度 - 添加F1分数信息
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                                f"Train Loss: {avg_train_loss:.4f}, "
                                f"Train Acc: {train_acc:.4f}, "
                                f"Train F1: {train_f1:.4f}, "
                                f"Val Loss: {avg_val_loss:.4f}, "
                                f"Val Acc: {val_acc:.4f}, "
                                f"Val F1: {val_f1:.4f}")
                
                # 早停检查 - 使用F1分数代替损失作为指标
                if val_f1 > best_val_f1:  # 变为 > 而不是 <
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        self.logger.info(f"早停: {patience_counter} 轮F1无改善")
                        break
            else:
                # 没有验证集时，只记录训练指标
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                history['train_f1_macro'].append(train_f1)
                
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                                f"Train Loss: {avg_train_loss:.4f}, "
                                f"Train Acc: {train_acc:.4f}, "
                                f"Train F1: {train_f1:.4f}")
        
        # 恢复最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        self.logger.info("模型训练完成")
        return model, history
        
    def _evaluate_model(self, model, data_loader, active_groups):
        """评估模型"""
        if data_loader is None:
            self.logger.warning("未提供评估数据集")
            return {}
                
        self.logger.info(f"评估模型，特征组: {active_groups}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 提取特征和标签
                features, labels = batch
                
                # 特征字典移到GPU
                features = {group: tensor.to(self.device) for group, tensor in features.items()}
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 收集预测结果
                _, predicted = outputs.max(1)
                all_targets.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 转换为NumPy数组
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # 计算指标
        try:
            accuracy = accuracy_score(all_targets, all_predictions)
            f1_macro = f1_score(all_targets, all_predictions, average='macro')
            f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
            
            metrics = {
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'n_samples': len(all_targets),
                'active_groups': active_groups
            }
            
            # 修改日志，突出宏平均F1分数
            self.logger.info(f"评估结果: 宏平均F1={f1_macro:.4f}, 准确率={accuracy:.4f}")
            
            return metrics
        except Exception as e:
            self.logger.error(f"计算评估指标失败: {e}")
            return {'error': str(e)}

    
    def _save_evaluation_result(self, group_key, metrics):
        """保存评估结果"""
        try:
            # 创建用于保存的字典（移除不必要的大数据）
            save_metrics = metrics.copy()
            if 'history' in save_metrics:
                # 转换训练历史为列表
                history = save_metrics['history']
                save_metrics['history'] = {
                    'train_loss': [float(x) for x in history.get('train_loss', [])],
                    'val_loss': [float(x) for x in history.get('val_loss', [])],
                    'train_acc': [float(x) for x in history.get('train_acc', [])],
                    'val_acc': [float(x) for x in history.get('val_acc', [])]
                }
            
            # 保存结果
            save_path = os.path.join(self.output_dir, f"{group_key}_evaluation.json")
            with open(save_path, 'w') as f:
                json.dump(save_metrics, f, indent=4)
            
            self.logger.info(f"评估结果已保存到 {save_path}")
        except Exception as e:
            self.logger.error(f"保存评估结果失败: {e}")
    
    def evaluate_all_group_combinations(self, data_path, max_combination_size=None):
        """
        评估所有特征组的组合
        
        参数:
            data_path: 数据文件路径
            max_combination_size: 最大组合大小
            
        返回:
            results: 所有评估结果
        """
        self.logger.info(f"评估所有特征组组合")
        
        # 加载数据
        data_dict = self.load_data(data_path)
        
        # 检测所有特征组
        all_groups = []
        for key in data_dict.keys():
            if key.startswith('group_'):
                group_name = key.replace('group_', '')
                all_groups.append(group_name)
        
        if not all_groups:
            self.logger.warning("未检测到特征组")
            return {}
        
        self.logger.info(f"检测到 {len(all_groups)} 个特征组: {all_groups}")
        
        # 设置最大组合大小
        if max_combination_size is None:
            max_combination_size = len(all_groups)
        else:
            max_combination_size = min(max_combination_size, len(all_groups))
        
        # 准备所有组合
        all_combinations = []
        for size in range(1, max_combination_size + 1):
            for combo in combinations(all_groups, size):
                all_combinations.append(list(combo))
        
        self.logger.info(f"将评估 {len(all_combinations)} 个特征组组合")
        
        # 评估每个组合
        for combo in all_combinations:
            self.logger.info(f"评估特征组组合: {combo}")
            
            # 准备数据
            data_loaders, feature_dims = self.prepare_data_loaders(data_dict, include_groups=combo)
            
            # 训练和评估
            self.train_and_evaluate(data_loaders, feature_dims, combo)
        
        # 汇总结果
        self._summarize_evaluations()
        
        return self.evaluation_results
    
    def _summarize_evaluations(self):
        """汇总所有评估结果"""
        self.logger.info("汇总评估结果")
        
        if not self.evaluation_results:
            self.logger.warning("没有评估结果可汇总")
            return
        
        # 提取关键指标
        summary = []
        for group_key, metrics in self.evaluation_results.items():
            summary.append({
                'group_key': group_key,
                'active_groups': metrics.get('active_groups', []),
                'group_count': len(metrics.get('active_groups', [])),
                'accuracy': metrics.get('accuracy', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'f1_weighted': metrics.get('f1_weighted', 0)
            })
        
        # 创建汇总数据框
        summary_df = pd.DataFrame(summary)
        
        # 按宏平均F1分数排序，而不是准确率
        summary_df = summary_df.sort_values('f1_macro', ascending=False)
        
        # 保存汇总结果
        summary_path = os.path.join(self.output_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"评估汇总已保存到 {summary_path}")
        
        # 可视化结果
        self._visualize_evaluations(summary_df)
    
    def _visualize_evaluations(self, summary_df):
        """可视化评估结果"""
        self.logger.info("可视化评估结果")
        
        try:
            # 绘制不同组大小的性能比较
            plt.figure(figsize=(12, 12))  # 增加图形高度
            
            # 按组大小分组
            group_sizes = sorted(summary_df['group_count'].unique())
            
            # 宏平均F1对比 - 将此图从原来的第二位置调到第一位置，表明其重要性
            plt.subplot(3, 1, 1)  # 修改为3行1列的第1个子图
            data = []
            for size in group_sizes:
                size_df = summary_df[summary_df['group_count'] == size]
                data.append(size_df['f1_macro'].values)
            
            plt.boxplot(data, labels=group_sizes)
            plt.title('Macro-averaged F1 score distribution for different feature group sizes')
            plt.xlabel('Number of feature groups')
            plt.ylabel('Macro-averaged F1')
            plt.grid(True, alpha=0.3)
            
            # 准确率对比 - 移到第二位置
            plt.subplot(3, 1, 2)  # 修改为3行1列的第2个子图
            data = []
            for size in group_sizes:
                size_df = summary_df[summary_df['group_count'] == size]
                data.append(size_df['accuracy'].values)
            
            plt.boxplot(data, labels=group_sizes)
            plt.title('Accuracy distribution for different feature group sizes')
            plt.xlabel('Number of feature groups')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            
            # 加权F1分数对比 - 新增子图
            plt.subplot(3, 1, 3)  # 添加为3行1列的第3个子图
            data = []
            for size in group_sizes:
                size_df = summary_df[summary_df['group_count'] == size]
                data.append(size_df['f1_weighted'].values)
            
            plt.boxplot(data, labels=group_sizes)
            plt.title('Weighted F1 score distribution for different feature group sizes')
            plt.xlabel('Number of feature groups')
            plt.ylabel('Weighted F1')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            viz_path = os.path.join(self.output_dir, "group_size_performance.png")
            plt.savefig(viz_path)
            plt.close()
            self.logger.info(f"特征组大小性能对比图已保存到 {viz_path}")
            
            # 绘制热图显示每个特征组的贡献
            self._visualize_group_contributions(summary_df)
            
        except Exception as e:
            self.logger.error(f"生成评估可视化失败: {e}")
    
    def _visualize_group_contributions(self, summary_df):
        """可视化每个特征组的贡献"""
        self.logger.info("可视化特征组贡献")
        
        try:
            # 提取所有唯一的特征组
            all_groups = set()
            for groups in summary_df['active_groups']:
                all_groups.update(groups)
            all_groups = sorted(list(all_groups))
            
            if not all_groups:
                self.logger.warning("未找到特征组")
                return
                
            # 创建贡献矩阵
            contribution_matrix = np.zeros((len(all_groups), len(all_groups)))
            
            # 填充矩阵
            for _, row in summary_df.iterrows():
                group_indices = [all_groups.index(group) for group in row['active_groups']]
                for i in group_indices:
                    for j in group_indices:
                        contribution_matrix[i, j] += row['accuracy']
            
            # 归一化
            for i in range(len(all_groups)):
                if contribution_matrix[i, i] > 0:
                    contribution_matrix[i, :] /= contribution_matrix[i, i]
            
            # 绘制热图
            plt.figure(figsize=(10, 8))
            sns.heatmap(contribution_matrix, annot=True, cmap='YlGnBu', 
                       xticklabels=all_groups, yticklabels=all_groups)
            
            plt.title('Feature group contribution heatmap')
            plt.tight_layout()
            
            # 保存图表
            viz_path = os.path.join(self.output_dir, "group_contributions.png")
            plt.savefig(viz_path)
            plt.close()
            self.logger.info(f"特征组贡献热图已保存到 {viz_path}")
            
            # 绘制每个组的单独影响
            self._visualize_individual_contributions(summary_df, all_groups)
            
        except Exception as e:
            self.logger.error(f"生成特征组贡献可视化失败: {e}")
    
    def _visualize_individual_contributions(self, summary_df, all_groups):
        """可视化每个特征组的单独影响"""
        self.logger.info("可视化单个特征组贡献")
        
        try:
            # 提取单组的性能
            single_group_df = summary_df[summary_df['group_count'] == 1].copy()
            
            if len(single_group_df) == 0:
                self.logger.warning("未找到单个特征组的评估结果")
                return
                    
            # 添加组名列
            single_group_df['group_name'] = single_group_df['active_groups'].apply(lambda x: x[0] if x else None)
            
            # 按宏平均F1分数排序，而不是准确率
            single_group_df = single_group_df.sort_values('f1_macro', ascending=False)
            
            # 绘制条形图
            plt.figure(figsize=(12, 12))  # 增加图形高度以容纳3个子图
            
            # 宏平均F1条形图 - 放在第一位置，表明其重要性
            plt.subplot(3, 1, 1)
            plt.bar(single_group_df['group_name'], single_group_df['f1_macro'], color='forestgreen')
            plt.title('Macro-averaged F1 of individual feature groups')
            plt.xlabel('Feature group')
            plt.ylabel('Macro-averaged F1')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # 准确率条形图 - 移到第二位置
            plt.subplot(3, 1, 2)
            plt.bar(single_group_df['group_name'], single_group_df['accuracy'], color='steelblue')
            plt.title('Accuracy of individual feature groups')
            plt.xlabel('Feature group')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # 加权F1条形图 - 添加为第三个子图
            plt.subplot(3, 1, 3)
            plt.bar(single_group_df['group_name'], single_group_df['f1_weighted'], color='darkorange')
            plt.title('Weighted F1 of individual feature groups')
            plt.xlabel('Feature group')
            plt.ylabel('Weighted F1')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            viz_path = os.path.join(self.output_dir, "individual_group_performance.png")
            plt.savefig(viz_path)
            plt.close()
            self.logger.info(f"单个特征组性能图已保存到 {viz_path}")
            
        except Exception as e:
            self.logger.error(f"生成单个特征组贡献可视化失败: {e}")




class FeatureGroupDataset(torch.utils.data.Dataset):
    """特征组数据集，支持多个特征组"""
    
    def __init__(self, features_dict, labels):
        """
        初始化数据集
        
        参数:
            features_dict: 特征字典，键为组名，值为特征张量
            labels: 标签张量
        """
        self.features_dict = features_dict
        self.labels = labels
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx: 样本索引
            
        返回:
            features: 特征字典，键为组名，值为特征向量
            label: 对应的标签
        """
        # 构建特征字典
        features = {group_name: features[idx] for group_name, features in self.features_dict.items()}
        label = self.labels[idx]
        
        return features, label