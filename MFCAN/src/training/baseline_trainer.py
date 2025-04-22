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
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1_macro': [],
            'val_f1_macro': [],
            # 添加测试集指标记录
            'test_loss': [],
            'test_acc': [],
            'test_f1_macro': [],
            'learning_rates': []
        }
    
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
        best_test_f1 = 0.0  # 记录最佳测试集性能
        
        # 设置评估间隔
        eval_interval = max(1, self.num_epochs // 10)  # 默认每10%的轮次评估一次
        if self.num_epochs <= 10:
            eval_interval = 1  # 如果总轮次很少，每轮都评估
        
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
            
            # 定期在测试集上评估
            if (epoch + 1) % eval_interval == 0 or epoch == self.num_epochs - 1:
                self.logger.info(f"在测试集上评估（第 {epoch+1} 轮）...")
                test_loss, test_acc, test_f1 = self._validate(data_loaders['test'])
                self.logger.info(f"测试集性能 - 宏平均F1: {test_f1:.4f}, 准确率: {test_acc:.4f}, 损失: {test_loss:.4f}")
                
                # 如果您想保存测试集指标历史
                if 'test_loss' not in self.history:
                    self.history['test_loss'] = [None] * epoch
                    self.history['test_acc'] = [None] * epoch
                    self.history['test_f1_macro'] = [None] * epoch
                
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
                self.history['test_f1_macro'].append(test_f1)
            else:
                # 填充空值保持历史记录长度一致
                if 'test_loss' in self.history:
                    self.history['test_loss'].append(None)
                    self.history['test_acc'].append(None)
                    self.history['test_f1_macro'].append(None)
            
            # 打印进度 - 强调宏平均F1
            self.logger.info(f"第 {epoch+1}/{self.num_epochs} 轮 - "
                            f"训练: 宏平均F1={train_f1:.4f}, 准确率={train_acc:.4f}, 损失={train_loss:.4f}, "
                            f"验证: 宏平均F1={val_f1:.4f}, 准确率={val_acc:.4f}, 损失={val_loss:.4f}, "
                            f"学习率: {current_lr:.8f}")
            
            # 保存最佳模型 - 使用宏平均F1作为判断标准
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                
                # 在最佳验证集性能时评估测试集
                test_loss, test_acc, test_f1 = self._validate(data_loaders['test'])
                self.logger.info(f"【新的最佳模型】测试集性能 - 宏平均F1: {test_f1:.4f}, 准确率: {test_acc:.4f}")
                best_test_f1 = test_f1  # 记录最佳测试集性能
                
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
                    'val_f1_macro': val_f1,
                    'test_f1_macro': test_f1,
                    'test_acc': test_acc,
                    'test_loss': test_loss
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
        self.logger.info(f"最佳模型测试集宏平均F1: {best_test_f1:.4f}")
        
        # 可视化训练历史
        self._plot_training_history()
        
        # 加载最佳模型
        if best_model_path is not None:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"已加载最佳模型（第 {best_epoch+1} 轮）")
        
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
        
        # 记录批次总数
        batch_count = len(data_loader)
        self.logger.info(f"开始评估，共 {batch_count} 个批次")
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 收集预测结果
                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # 可选：记录进度
                if (i+1) % (max(1, batch_count // 10)) == 0:
                    self.logger.debug(f"评估进度: {i+1}/{batch_count} 批次 ({(i+1)/batch_count*100:.1f}%)")
        
        # 转换为NumPy数组
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # 获取实际使用的类别数
        unique_classes = np.unique(np.concatenate([all_targets, all_predictions]))
        num_classes = len(unique_classes)
        
        # 类别别名 - 修改这里，确保类别名称数量与实际类别数量一致
        class_names = [f"Class {i}" for i in range(np.max(unique_classes) + 1)]
        
        # 明确指定标签参数
        classification_rep = classification_report(
            all_targets, 
            all_predictions, 
            labels=range(len(class_names)),  # 显式指定标签范围
            target_names=class_names, 
            output_dict=True
        )
        
        # 可视化混淆矩阵(只显示部分类别，避免过于复杂)
        self._plot_confusion_matrix(conf_matrix, class_names[:10], "Top 10 Classes Confusion Matrix")
        
        # 返回评估指标
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_rep
        }
        
        # 保存评估结果
        self._save_evaluation_results(metrics)
        
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
        # 提取需要保存的指标
        results = {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'classification_report': metrics['classification_report']
        }
        
        # 保存为JSON文件
        with open(os.path.join(self.save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
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

## 每类性能
| 类别 | 精确率 | 召回率 | F1分数 | 支持度 |
|------|--------|--------|--------|--------|
"""
        # 添加每个类别的性能指标
        for class_name, metrics_dict in metrics['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report += f"| {class_name} | {metrics_dict['precision']:.4f} | {metrics_dict['recall']:.4f} | {metrics_dict['f1-score']:.4f} | {metrics_dict['support']} |\n"
        
        # 添加总结 - 突出显示宏平均F1
        report += f"""
## 摘要
- **Macro Avg**: 精确率={metrics['classification_report']['macro avg']['precision']:.4f}, 召回率={metrics['classification_report']['macro avg']['recall']:.4f}, F1={metrics['classification_report']['macro avg']['f1-score']:.4f}
- Weighted Avg: 精确率={metrics['classification_report']['weighted avg']['precision']:.4f}, 召回率={metrics['classification_report']['weighted avg']['recall']:.4f}, F1={metrics['classification_report']['weighted avg']['f1-score']:.4f}
"""
        
        # 保存报告
        with open(os.path.join(self.save_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)