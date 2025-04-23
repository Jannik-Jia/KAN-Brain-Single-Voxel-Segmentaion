import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.logging_utils import Logger
import h5py

class BaselineTrainer:
    """基线模型训练器"""
    
    def __init__(self, model, config_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        参数:
            model: PyTorch模型
            config_path: 配置文件路径
            device: 训练设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 设置日志记录器
        log_manager = Logger("BaselineTrainer", log_dir="logs/training")
        self.logger = log_manager.get_logger()
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 训练参数
        self.num_epochs = self.config.get('num_epochs', 50)
        self.batch_size = self.config.get('batch_size', 128)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.optimizer_type = self.config.get('optimizer', 'adam')
        self.scheduler_type = self.config.get('scheduler', 'cosine')
        self.criterion_type = self.config.get('criterion', 'cross_entropy')
        self.label_smoothing = self.config.get('label_smoothing', 0.1)
        self.use_class_weights = self.config.get('use_class_weights', True)
        self.early_stopping = self.config.get('early_stopping', 10)
        self.save_dir = self.config.get('save_dir', 'results/baseline')
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化优化器、损失函数和调度器
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # 将在prepare_data后初始化，因为可能需要知道数据集大小
        
        # 记录训练历史 - 移除测试集相关的历史记录项
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1_macro': [],
            'val_f1_macro': [],
            'learning_rates': []
            # 测试集指标将在最终评估后添加为单一条目，而不是每个epoch都记录
        }
        
        # 训练状态
        self.best_val_f1 = 0.0
        self.best_epoch = -1
        self.best_model_path = None
        
        self.logger.info(f"BaselineTrainer初始化完成，模型将保存在 {self.save_dir}")
        
    def _create_optimizer(self):
        """创建优化器"""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def _create_scheduler(self, num_steps):
        """创建学习率调度器"""
        if self.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif self.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_type.lower() == 'plateau':
            # 修改为监控验证集F1分数，因为我们现在使用F1作为主要指标
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5)
        elif self.scheduler_type.lower() == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.learning_rate * 10,
                total_steps=num_steps,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=10000.0
            )
        else:
            return None
    
    def prepare_data(self, data_path):
        """
        准备训练数据
        
        参数:
            data_path: 处理后数据的路径
            
        返回:
            data_loaders: 包含训练、验证和测试数据加载器的字典
        """
        with h5py.File(data_path, 'r') as f:
            # 加载特征和标签
            train_features = f['train/features'][()]
            train_labels = f['train/labels'][()]
            val_features = f['val/features'][()]
            val_labels = f['val/labels'][()]
            test_features = f['test/features'][()]
            test_labels = f['test/labels'][()]
        
        # 转换为PyTorch张量
        train_features = torch.FloatTensor(train_features)
        train_labels = torch.LongTensor(train_labels)
        val_features = torch.FloatTensor(val_features)
        val_labels = torch.LongTensor(val_labels)
        test_features = torch.FloatTensor(test_features)
        test_labels = torch.LongTensor(test_labels)
        
        # 对类别标签进行预处理（假设原始标签从1开始）
        train_labels = train_labels - 1
        val_labels = val_labels - 1
        test_labels = test_labels - 1
        
        # 创建数据集
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 为每个类别计算权重
        if self.use_class_weights:
            class_weights = self._calculate_class_weights(train_labels)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            class_weights = None
        
        # 初始化损失函数
        if self.criterion_type.lower() == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, 
                label_smoothing=self.label_smoothing
            )
        elif self.criterion_type.lower() == 'focal':
            from torch.nn import functional as F
            # 简单实现的Focal Loss
            class FocalLoss(nn.Module):
                def __init__(self, weight=None, gamma=2.0, reduction='mean'):
                    super(FocalLoss, self).__init__()
                    self.weight = weight
                    self.gamma = gamma
                    self.reduction = reduction
                
                def forward(self, input, target):
                    ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
                    pt = torch.exp(-ce_loss)
                    focal_loss = (1 - pt) ** self.gamma * ce_loss
                    
                    if self.reduction == 'mean':
                        return focal_loss.mean()
                    elif self.reduction == 'sum':
                        return focal_loss.sum()
                    else:
                        return focal_loss
            
            self.criterion = FocalLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported criterion type: {self.criterion_type}")
        
        # 初始化学习率调度器
        num_steps = self.num_epochs * len(train_loader)
        self.scheduler = self._create_scheduler(num_steps)
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def _calculate_class_weights(self, labels):
        """
        计算类别权重，解决不平衡问题
        
        参数:
            labels: 标签张量
            
        返回:
            weights: 类别权重数组
        """
        # 转换为NumPy数组
        labels_np = labels.cpu().numpy()
        
        # 统计每个类别的样本数
        num_classes = len(np.unique(labels_np))
        class_counts = np.bincount(labels_np, minlength=num_classes)
        
        # 计算权重（反比于频率）
        weights = 1.0 / np.maximum(class_counts, 1)  # 避免除零
        
        # 归一化权重
        weights = weights / weights.sum() * len(weights)
        
        return weights

    
    def train(self, data_loaders):
        """
        训练模型
        
        参数:
            data_loaders: 包含训练、验证和测试数据加载器的字典
            
        返回:
            history: 训练历史记录
        """
        self.logger.info(f"开始训练，共 {self.num_epochs} 轮")
        
        if self.criterion is None:
            self.logger.warning("损失函数未初始化，使用默认CrossEntropyLoss")
            self.criterion = nn.CrossEntropyLoss()
        
        # 修改为使用宏平均F1作为主要指标
        best_val_f1 = 0.0  # 初始化为最小值，因为我们想要最大化F1
        best_epoch = 0
        best_model_path = None
        
        for epoch in range(self.num_epochs):
            # 训练一个轮次
            train_loss, train_acc, train_f1 = self._train_epoch(data_loaders['train'])
            
            # 验证
            val_loss, val_acc, val_f1 = self._validate(data_loaders['val'])
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # 使用宏平均F1分数来调整学习率
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # 保存历史记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1_macro'].append(train_f1)
            self.history['val_f1_macro'].append(val_f1)
            
            # 打印进度 - 强调宏平均F1
            self.logger.info(f"第 {epoch+1}/{self.num_epochs} 轮 - "
                            f"训练: 宏平均F1={train_f1:.4f}, 准确率={train_acc:.4f}, 损失={train_loss:.4f}, "
                            f"验证: 宏平均F1={val_f1:.4f}, 准确率={val_acc:.4f}, 损失={val_loss:.4f}, "
                            f"学习率: {current_lr:.8f}")
            
            # 保存最佳模型 - 使用宏平均F1作为判断标准
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                
                # 删除之前的最佳模型
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # 保存新的最佳模型
                best_model_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1_macro': val_f1
                }, best_model_path)
                
                self.logger.info(f"已保存最佳模型到第 {epoch+1} 轮，验证集宏平均F1: {val_f1:.4f}")
            
            # 早停 - 基于F1分数
            if epoch - best_epoch >= self.early_stopping:
                self.logger.info(f"早停：连续 {self.early_stopping} 轮未提升F1分数，在第 {epoch+1} 轮停止训练")
                break
        
        # 训练结束时的总结
        self.logger.info("训练完成")
        self.logger.info(f"最佳模型来自第 {best_epoch+1} 轮")
        self.logger.info(f"最佳验证集宏平均F1: {best_val_f1:.4f}")
        
        # 可视化训练历史
        self._plot_training_history()
        
        # 加载最佳模型
        if best_model_path is not None:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"已加载最佳模型（第 {best_epoch+1} 轮）")
        
        # 在最佳模型上进行测试集评估
        self.logger.info("在测试集上进行最终评估...")
        test_metrics = self.evaluate(data_loaders['test'])
        
        # 记录测试集性能
        self.logger.info(f"测试集性能 - 宏平均F1: {test_metrics['f1_macro']:.4f}, 准确率: {test_metrics['accuracy']:.4f}")
        
        # 将测试集指标添加到历史记录中
        self.history['test_metrics'] = test_metrics
        
        return self.history


    def _train_epoch(self, train_loader):
        """
        训练一个轮次
        
        参数:
            train_loader: 训练数据加载器
            
        返回:
            avg_loss: 平均损失
            accuracy: 分类准确率
            f1_macro: 宏平均F1分数
        """
        self.model.train()
        total_loss = 0
        all_targets = []
        all_predictions = []
        
        # 记录批次总数和当前进度
        batch_count = len(train_loader)
        log_interval = max(1, batch_count // 10)  # 每10%记录一次日志
        
        self.logger.info(f"开始训练，共 {batch_count} 个批次")
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 添加标签验证和处理
            num_classes = self.model.classifier.out_features
            
            # 检查标签是否在有效范围
            if torch.max(targets) >= num_classes:
                self.logger.warning(f"警告: 训练中发现标签值 {torch.max(targets).item()} 超出类别数 {num_classes}")
                # 截断超出范围的标签
                targets = torch.clamp(targets, 0, num_classes - 1)
            if torch.min(targets) < 0:
                self.logger.warning(f"警告: 训练中发现负标签值 {torch.min(targets).item()}")
                # 将负值标签设为0
                targets = torch.clamp(targets, 0, None)

            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            # 收集预测和目标用于计算F1
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # 减少日志频率，只在开始、结束和每10%的时候记录
            if i == 0 or (i+1) % log_interval == 0 or i == batch_count - 1:
                self.logger.info(f"训练进度: {i+1}/{batch_count} 批次 ({(i+1)/batch_count*100:.1f}%), 当前批次损失: {loss.item():.4f}")
        
        # 计算整体指标
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        
        return avg_loss, accuracy, f1_macro


    def _validate(self, val_loader):
        """
        在验证集上评估模型
        
        参数:
            val_loader: 验证数据加载器
            
        返回:
            avg_loss: 平均损失
            accuracy: 分类准确率
            f1_macro: 宏平均F1分数
        """
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []
        
        # 记录批次总数
        batch_count = len(val_loader)
        self.logger.debug(f"开始验证，共 {batch_count} 个批次")
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                
                # 收集预测和目标用于计算F1
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # 计算整体指标
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        
        return avg_loss, accuracy, f1_macro
    

    def evaluate(self, data_loader):
        """
        在测试集上全面评估模型
        
        参数:
            data_loader: 测试数据加载器
            
        返回:
            metrics: 评估指标字典
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        total_loss = 0
        
        # 记录批次总数
        batch_count = len(data_loader)
        self.logger.info(f"开始最终评估，共 {batch_count} 个批次")
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                probabilities = torch.softmax(outputs, dim=1)
                
                # 收集预测结果
                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # 记录进度
                if (i+1) % max(1, batch_count // 5) == 0:
                    self.logger.info(f"评估进度: {i+1}/{batch_count} 批次 ({(i+1)/batch_count*100:.1f}%)")
        
        # 转换为NumPy数组
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # 计算损失
        avg_loss = total_loss / len(all_targets)
        
        # 计算性能指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # 获取实际使用的类别数
        unique_classes = np.unique(np.concatenate([all_targets, all_predictions]))
        num_classes = len(unique_classes)
        
        # 类别别名
        class_names = [f"Class {i}" for i in range(num_classes)]
        
        # 计算每个类别的精确率、召回率、F1分数
        classification_rep = classification_report(
            all_targets, 
            all_predictions, 
            labels=range(num_classes),
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for i in range(num_classes):
            if i in unique_classes:
                # 计算第i类的准确率
                mask = (all_targets == i)
                if np.sum(mask) > 0:  # 确保有这个类别的样本
                    class_accuracies[f"Class {i}"] = accuracy_score(
                        all_targets[mask] == i, 
                        all_predictions[mask] == i
                    )
                else:
                    class_accuracies[f"Class {i}"] = 0.0
        
        # 可视化混淆矩阵
        self._plot_confusion_matrix(conf_matrix, class_names[:min(20, num_classes)], "Confusion Matrix")
        
        # 计算前k个准确率（如果有概率输出）
        top_k_accuracies = {}
        if len(all_probabilities) > 0:
            for k in [1, 3, 5]:
                if k <= num_classes:
                    # 获取前k个预测
                    top_k_indices = np.argsort(-all_probabilities, axis=1)[:, :k]
                    # 计算每个样本的真实标签是否在前k个预测中
                    top_k_correct = [target in predictions for target, predictions in zip(all_targets, top_k_indices)]
                    top_k_accuracies[f"top_{k}_accuracy"] = np.mean(top_k_correct)
        
        # 返回完整的评估指标
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_rep,
            'class_accuracies': class_accuracies,
            'top_k_accuracies': top_k_accuracies
        }
        
        # 保存评估结果
        self._save_evaluation_results(metrics)
        
        # 打印主要指标
        self.logger.info("=" * 50)
        self.logger.info("最终测试集评估结果:")
        self.logger.info(f"损失: {avg_loss:.4f}")
        self.logger.info(f"准确率: {accuracy:.4f}")
        self.logger.info(f"宏平均F1: {f1_macro:.4f}")
        self.logger.info(f"加权平均F1: {f1_weighted:.4f}")
        if top_k_accuracies:
            for k, acc in top_k_accuracies.items():
                self.logger.info(f"{k}: {acc:.4f}")
        self.logger.info("=" * 50)
        
        return metrics


    def _plot_training_history(self):
        """可视化训练历史"""
        plt.figure(figsize=(15, 15))  # 增加图形大小以容纳额外的F1曲线图
        
        # Plot loss curve
        plt.subplot(3, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curve
        plt.subplot(3, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot macro F1 curve - 新增F1曲线图
        plt.subplot(3, 2, 3)
        plt.plot(self.history['train_f1_macro'], label='Train Macro F1')
        plt.plot(self.history['val_f1_macro'], label='Val Macro F1')
        plt.title('Training and Validation Macro F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate curve
        plt.subplot(3, 2, 4)
        plt.plot(self.history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Early stopping visualization
        plt.subplot(3, 2, 5)
        best_epoch = self.history['val_f1_macro'].index(max(self.history['val_f1_macro']))
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
        
        # 绘制验证集F1曲线（重复）以便标记最佳点
        plt.plot(self.history['val_f1_macro'], label='Val Macro F1')
        
        # 标记最佳F1
        plt.plot(best_epoch, self.history['val_f1_macro'][best_epoch], 'ro', 
                label=f'Best F1 ({self.history["val_f1_macro"][best_epoch]:.4f})')
        
        plt.title('Best Model Selection')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Macro F1')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()

    def _plot_confusion_matrix(self, conf_matrix, class_names, title):
        """
        可视化混淆矩阵

        参数:
            conf_matrix: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
        """
        plt.figure(figsize=(12, 10))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        # Set axis ticks
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add value labels
        thresh = conf_matrix.max() / 2
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.save_dir, f"confusion_matrix_{title.replace(' ', '_')}.png"))
        plt.close()



    def _save_evaluation_results(self, metrics):
        """
        保存评估结果
        
        参数:
            metrics: 评估指标字典
        """
        try:
            # 创建评估结果目录
            eval_dir = os.path.join(self.save_dir, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # 保存可序列化的指标
            serializable_metrics = metrics.copy()
            # 移除不可序列化的NumPy数组
            if 'confusion_matrix' in serializable_metrics:
                serializable_metrics['confusion_matrix'] = serializable_metrics['confusion_matrix'].tolist()
            
            # 保存为JSON文件
            with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            
            # 生成可读性报告 - 突出显示宏平均F1分数
            report = f"""
    # 模型评估报告

    ## 基本信息
    - 模型名称: {self.model.__class__.__name__}
    - 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

    ## 整体性能
    - **宏平均F1分数**: {metrics['f1_macro']:.4f}  <!-- 主要指标 -->
    - 加权F1分数: {metrics['f1_weighted']:.4f}
    - 准确率: {metrics['accuracy']:.4f}

    ## Top-K 准确率
    """
            # 添加Top-K准确率（如果存在）
            if 'top_k_accuracies' in metrics and metrics['top_k_accuracies']:
                for k, acc in metrics['top_k_accuracies'].items():
                    report += f"- {k}: {acc:.4f}\n"
            else:
                report += "- 未计算Top-K准确率\n"
                
            report += "\n## 每类性能\n| 类别 | 精确率 | 召回率 | F1分数 | 支持度 |\n|------|--------|--------|--------|--------|\n"
            
            # 添加每个类别的性能指标
            for class_name, metrics_dict in metrics['classification_report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report += f"| {class_name} | {metrics_dict['precision']:.4f} | {metrics_dict['recall']:.4f} | {metrics_dict['f1-score']:.4f} | {metrics_dict['support']} |\n"
            
            # 添加总结 - 突出显示宏平均F1
            report += f"""
    ## 摘要
    - **Macro Avg**: 精确率={metrics['classification_report']['macro avg']['precision']:.4f}, 召回率={metrics['classification_report']['macro avg']['recall']:.4f}, F1={metrics['classification_report']['macro avg']['f1-score']:.4f}
    - Weighted Avg: 精确率={metrics['classification_report']['weighted avg']['precision']:.4f}, 召回率={metrics['classification_report']['weighted avg']['recall']:.4f}, F1={metrics['classification_report']['weighted avg']['f1-score']:.4f}

    ## 注意事项
    - 报告基于测试集生成，该测试集在训练过程中未被使用
    - 宏平均F1是主要评估指标，因为它对每个类别赋予相同的权重，适合类别不平衡的数据集
    - 混淆矩阵图表已保存在evaluation目录下
    """
            
            # 保存报告
            with open(os.path.join(eval_dir, 'evaluation_report.md'), 'w') as f:
                f.write(report)
            
            # 保存更详细的类别性能分析
            self._save_detailed_class_performance(metrics, eval_dir)
            
            self.logger.info(f"评估结果已保存至 {eval_dir}")
            
        except Exception as e:
            self.logger.error(f"保存评估结果失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _save_detailed_class_performance(self, metrics, eval_dir):
        """
        保存详细的类别性能分析
        
        参数:
            metrics: 评估指标字典
            eval_dir: 评估结果目录
        """
        try:
            # 提取分类报告
            class_report = metrics['classification_report']
            
            # 创建类别性能图表
            plt.figure(figsize=(15, 10))
            
            # 提取类别名称和性能指标（排除汇总行）
            classes = []
            precision = []
            recall = []
            f1 = []
            support = []
            
            for class_name, values in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    classes.append(class_name)
                    precision.append(values['precision'])
                    recall.append(values['recall'])
                    f1.append(values['f1-score'])
                    support.append(values['support'])
            
            # 限制显示的类别数量，避免图表过于拥挤
            max_classes = 20
            if len(classes) > max_classes:
                # 根据支持度选择前N个类别
                indices = np.argsort(support)[-max_classes:]
                classes = [classes[i] for i in indices]
                precision = [precision[i] for i in indices]
                recall = [recall[i] for i in indices]
                f1 = [f1[i] for i in indices]
                support = [support[i] for i in indices]
            
            # 创建图表
            x = np.arange(len(classes))
            width = 0.2
            
            fig, ax1 = plt.subplots(figsize=(15, 8))
            ax2 = ax1.twinx()
            
            # 绘制性能指标
            ax1.bar(x - width, precision, width, label='Precision', color='blue', alpha=0.7)
            ax1.bar(x, recall, width, label='Recall', color='green', alpha=0.7)
            ax1.bar(x + width, f1, width, label='F1', color='red', alpha=0.7)
            
            # 绘制支持度
            ax2.plot(x, support, 'o-', color='purple', label='Support')
            
            # 设置标签和标题
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Score')
            ax2.set_ylabel('Support (Number of Samples)')
            plt.title('Performance Metrics by Class')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes, rotation=45, ha='right')
            
            # 添加图例
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, 'class_performance.png'))
            plt.close()
            
            # 创建混淆矩阵热图（限制类别数量）
            if 'confusion_matrix' in metrics:
                conf_matrix = metrics['confusion_matrix']
                # 如果类别太多，选择支持度最高的N个类别
                if len(classes) > max_classes:
                    top_indices = np.argsort(support)[-max_classes:]
                    conf_matrix_subset = conf_matrix[top_indices][:, top_indices]
                    classes_subset = [classes[i] for i in top_indices]
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(conf_matrix_subset, annot=True, fmt='d', cmap='Blues',
                            xticklabels=classes_subset, yticklabels=classes_subset)
                    plt.title('Confusion Matrix (Top Classes by Support)')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    plt.savefig(os.path.join(eval_dir, 'confusion_matrix_top_classes.png'))
                    plt.close()
                
            self.logger.info(f"详细类别性能分析已保存至 {eval_dir}")
            
        except Exception as e:
            self.logger.error(f"保存详细类别性能分析失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())