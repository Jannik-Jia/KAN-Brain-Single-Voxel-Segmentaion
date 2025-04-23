import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import logging

from models.encoders.diffusion_encoder import DiffusionEncoder
from models.encoders.qti_encoder import QTIEncoder
from models.encoders.cest_encoder import CESTEncoder
from utils.model_io import ModelIO

class EncoderTrainer:
    """特征编码器预训练器"""
    
    def __init__(self, modality, config_path, device=None, logger=None):
        """
        初始化编码器训练器
        
        参数:
            modality: 特征模态 ('diffusion', 'qti', 'cest')
            config_path: 配置文件路径
            device: 训练设备
            logger: 日志记录器
        """
        self.modality = modality
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 设置日志记录器
        self.logger = logger or logging.getLogger(__name__)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 训练参数
        self.num_epochs = self.config.get('num_epochs', 50)
        self.batch_size = self.config.get('batch_size', 256)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.early_stopping = self.config.get('early_stopping', 10)
        
        # 从配置中读取评估相关配置
        eval_config = self.config.get('evaluation', {})
        self.metrics = eval_config.get('metrics', ['accuracy', 'f1_macro'])
        self.primary_metric = eval_config.get('primary_metric', 'f1_macro')
        self.eval_interval = eval_config.get('eval_interval', max(1, self.num_epochs // 10))
        
        # 保存目录
        self.save_dir = self.config.get('save_dir', f'models/encoders/{modality}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化ModelIO工具
        from utils.model_io import ModelIO
        self.model_io = ModelIO(logger=self.logger, config=self.config)
        
        # 创建编码器模型
        self.model = self._create_encoder()
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],  # 添加F1历史记录
            'learning_rates': []
        }
        
        self.logger.info(f"初始化 {modality} 编码器预训练器完成")


    def _create_encoder(self, input_dim=None):
        """根据模态创建对应的编码器模型，可以自动检测输入维度"""
        num_classes = self.config.get('num_classes', 102)
        
        # 如果外部提供了输入维度，则使用提供的值
        if input_dim is not None:
            self.logger.info(f"使用提供的输入维度: {input_dim}")
        
        if self.modality == 'diffusion':
            # 使用配置中的值，如果未提供则使用默认值
            input_dim = input_dim or self.config.get('diffusion_dim', 15)
            hidden_dim = self.config.get('diffusion_hidden_dim', 64)
            output_dim = self.config.get('diffusion_output_dim', 32)
            dropout = self.config.get('diffusion_dropout', 0.1)
            
            self.logger.info(f"创建DiffusionEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
            return DiffusionEncoder(input_dim, hidden_dim, output_dim, dropout)
            
        elif self.modality == 'qti':
            input_dim = input_dim or self.config.get('qti_dim', 210)
            hidden_dims = self.config.get('qti_hidden_dims', [512, 256])
            output_dim = self.config.get('qti_output_dim', 128)
            dropout = self.config.get('qti_dropout', 0.3)
            
            self.logger.info(f"创建QTIEncoder: input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}")
            return QTIEncoder(input_dim, hidden_dims, output_dim, dropout)
            
        elif self.modality == 'cest':
            input_dim = input_dim or self.config.get('cest_dim', 116)
            hidden_dim = self.config.get('cest_hidden_dim', 256)
            output_dim = self.config.get('cest_output_dim', 128)
            dropout = self.config.get('cest_dropout', 0.5)
            
            self.logger.info(f"创建CESTEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
            return CESTEncoder(input_dim, hidden_dim, output_dim, dropout)
            
        else:
            raise ValueError(f"不支持的模态: {self.modality}")
    
    def load_data(self, data_path):
        """
        加载特定模态的特征数据 - 自动调整特征维度
        
        参数:
            data_path: 数据文件路径
                
        返回:
            data_loaders: 包含训练集和验证集的数据加载器
        """
        self.logger.info(f"加载 {self.modality} 模态数据: {data_path}")
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件 {data_path} 不存在")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # 首先尝试访问所有可能的路径并记录存在的路径
                possible_paths = [
                    f'grouped/train/{self.modality}/features',
                    f'group_{self.modality}/train/features',
                    f'{self.modality}/train/features',
                    f'{self.modality}_selected/train/features'
                ]
                
                # 查找特征数据
                train_features = None
                for path in possible_paths:
                    if path in f:
                        train_features = f[path][()]
                        train_path = path
                        val_path = train_path.replace('train', 'val')
                        if val_path in f:
                            val_features = f[val_path][()]
                        else:
                            raise KeyError(f"在数据文件中未找到验证集路径 {val_path}")
                        
                        self.logger.info(f"成功从路径 {train_path} 加载特征数据")
                        break
                
                # 如果没有找到特征数据，抛出错误
                if train_features is None:
                    raise KeyError(f"在数据文件中未找到 {self.modality} 模态的特征")
                
                # 尝试查找对应的标签
                train_labels_path = None
                label_candidates = [
                    'original/train/labels',
                    f'grouped/train/{self.modality}/labels',
                    f'group_{self.modality}/train/labels',
                    f'{self.modality}/train/labels',
                    'train/labels'
                ]
                
                for path in label_candidates:
                    if path in f:
                        train_labels = f[path][()]
                        train_labels_path = path
                        val_labels_path = train_labels_path.replace('train', 'val')
                        if val_labels_path in f:
                            val_labels = f[val_labels_path][()]
                        else:
                            raise KeyError(f"在数据文件中未找到验证集标签路径 {val_labels_path}")
                        
                        self.logger.info(f"成功从路径 {train_labels_path} 加载标签数据")
                        break
                
                if train_labels_path is None:
                    raise KeyError(f"在数据文件中未找到训练标签")
                    
                self.logger.info(f"加载数据成功 - 训练集: {train_features.shape}, 验证集: {val_features.shape}")
                
                # 获取特征维度并重新创建编码器
                input_dim = train_features.shape[1]
                self.logger.info(f"从数据中检测到 {self.modality} 模态特征维度: {input_dim}")
                
                # 使用实际维度重新创建编码器
                self.model = self._create_encoder(input_dim=input_dim)
                self.model.to(self.device)
                
                # 重新初始化优化器 (因为模型参数已更新)
                self.optimizer = optim.AdamW(
                    self.model.parameters(), 
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
                
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise
        
        # 类别标签从0开始(如果原始标签从1开始，需要减1)
        if np.min(train_labels) == 1:
            train_labels = train_labels - 1
            val_labels = val_labels - 1
            self.logger.info("标签已从1开始调整为从0开始")
        
        # 转换为PyTorch张量
        train_features = torch.FloatTensor(train_features)
        train_labels = torch.LongTensor(train_labels)
        val_features = torch.FloatTensor(val_features)
        val_labels = torch.LongTensor(val_labels)
        
        # 创建数据集
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader
        }
        
    def train(self, data_loaders):
        """
        训练编码器
        
        参数:
            data_loaders: 包含训练集和验证集的数据加载器
            
        返回:
            model: 训练后的模型
            history: 训练历史
        """
        # 引入ModelIO工具类
        from utils.model_io import ModelIO
        model_io = ModelIO(logger=self.logger)

        self.logger.info(f"开始训练 {self.modality} 编码器，共 {self.num_epochs} 轮")
        
        # 改为跟踪最佳F1
        best_val_f1 = 0.0
        best_epoch = 0
        best_model_path = None
        
        eval_interval = self.eval_interval
        
        for epoch in range(self.num_epochs):
            # 训练一个轮次
            train_loss, train_acc = self._train_epoch(data_loaders['train'])
            
            # 验证
            val_loss, val_acc, val_f1 = self._validate(data_loaders['val'])
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            # 添加F1历史记录
            if 'val_f1' not in self.history:
                self.history['val_f1'] = []
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            
            # 打印进度信息 - 强调宏平均F1
            self.logger.info(f"轮次 {epoch+1}/{self.num_epochs}: "
                            f"训练损失={train_loss:.4f}, 训练准确率={train_acc:.4f}, "
                            f"验证损失={val_loss:.4f}, 验证准确率={val_acc:.4f}, "
                            f"验证宏平均F1={val_f1:.4f}, 学习率={current_lr:.8f}")
            
            # 定期在测试集上评估（如果有测试集）
            test_metrics = {}
            if 'test' in data_loaders and ((epoch + 1) % eval_interval == 0 or epoch == self.num_epochs - 1):
                self.logger.info(f"在测试集上评估（第 {epoch+1} 轮）...")
                test_loss, test_acc, test_f1 = self._validate(data_loaders['test'])
                test_metrics = {
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_f1': test_f1
                }
                self.logger.info(f"测试集性能 - 宏平均F1: {test_f1:.4f}, 准确率: {test_acc:.4f}, 损失: {test_loss:.4f}")
                
                # 可以选择将测试指标也添加到历史记录中
                if 'test_f1' not in self.history:
                    self.history['test_f1'] = [None] * epoch
                self.history['test_f1'].append(test_f1)
            elif 'test' in data_loaders:
                # 填充空值以保持历史记录长度一致
                if 'test_f1' in self.history:
                    self.history['test_f1'].append(None)
            
            # 保存最佳模型 - 使用F1作为判断标准
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                
                # 删除之前的最佳模型文件（如果存在）
                if best_model_path is not None:
                    for ext in ['', '.ckpt', '.full']:
                        path = best_model_path + ext
                        if os.path.exists(path):
                            os.remove(path)
                    # 也删除配置文件
                    config_path = best_model_path.replace('.pth', '_config.json')
                    if os.path.exists(config_path):
                        os.remove(config_path)
                
                # 在最佳模型时评估测试集（如果有）
                if 'test' in data_loaders and not test_metrics:
                    test_loss, test_acc, test_f1 = self._validate(data_loaders['test'])
                    test_metrics = {
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'test_f1': test_f1
                    }
                    self.logger.info(f"【新的最佳模型】测试集性能 - 宏平均F1: {test_f1:.4f}, 准确率: {test_acc:.4f}")
                
                # 保存新的最佳模型
                best_model_path = os.path.join(self.save_dir, f"{self.modality}_encoder_best_epoch_{epoch+1}.pth")
                
                # 准备元数据
                metadata = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'learning_rate': current_lr,
                    'date_saved': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 添加测试集指标（如果有）
                if test_metrics:
                    metadata.update(test_metrics)
                
                # 使用ModelIO保存模型
                model_io.save_encoder(self.model, best_model_path, self.modality, self.config, metadata)
                
                self.logger.info(f"已保存最佳模型到第 {epoch+1} 轮，验证集宏平均F1: {val_f1:.4f}")
            
            # 早停 - 基于F1
            if epoch - best_epoch >= self.early_stopping:
                self.logger.info(f"早停：连续 {self.early_stopping} 轮未提升F1分数，在第 {epoch+1} 轮停止训练")
                break
        
        # 训练结束时的总结
        self.logger.info("训练完成")
        self.logger.info(f"最佳模型来自第 {best_epoch+1} 轮")
        self.logger.info(f"最佳验证集宏平均F1: {best_val_f1:.4f}")
        
        # 可视化训练历史 - 添加F1曲线
        self._plot_training_history()
        
        # 加载最佳模型
        if best_model_path is not None:
            try:
                # 使用ModelIO加载模型
                loaded_model, config = model_io.load_encoder(best_model_path, device=self.device)
                if loaded_model is not None:
                    self.model = loaded_model
                    self.logger.info(f"加载最佳模型 (轮次 {config.get('epoch', '?')+1})")
                else:
                    self.logger.warning("无法加载最佳模型，将使用当前模型")
            except Exception as e:
                self.logger.error(f"加载最佳模型失败: {e}")
                self.logger.warning("将使用最后一轮的模型状态")
        
        # 保存最终模型
        final_model_path = os.path.join(self.save_dir, f"{self.modality}_encoder_final.pth")
        metadata = {
            'total_epochs': self.num_epochs,
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'date_saved': time.strftime('%Y-%m-%d %H:%M:%S'),
            'is_final_model': True
        }
        model_io.save_encoder(self.model, final_model_path, self.modality, self.config, metadata)
        self.logger.info(f"保存最终模型: {final_model_path}")
        
        return self.model, self.history
    
    def _train_epoch(self, train_loader):
        """训练一个轮次 - 移除进度条"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 替换进度条为简单的批次计数
        batch_count = len(train_loader)
        log_interval = max(1, batch_count // 5)  # 每20%记录一次日志
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
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
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 减少日志频率，只在开始、结束和每20%的时候记录
            if i == 0 or (i+1) % log_interval == 0 or i == batch_count - 1:
                self.logger.info(f"训练进度: {i+1}/{batch_count} 批次 ({(i+1)/batch_count*100:.1f}%)")
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    

    
    def _validate(self, data_loader):
        """验证模型并计算F1分数"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="验证中", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集预测和真实标签用于计算F1分数
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算准确率和损失
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # 计算宏平均F1分数
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1


    def _plot_training_history(self):
        """可视化训练历史"""
        plt.figure(figsize=(15, 12))
        
        # Plot loss curve
        plt.subplot(3, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title(f'{self.modality} Encoder Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curve
        plt.subplot(3, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.title(f'{self.modality} Encoder Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 curve
        plt.subplot(3, 2, 3)
        plt.plot(self.history.get('val_f1', []), label='Val F1')
        if 'test_f1' in self.history:
            # 过滤掉None值
            epochs = range(len(self.history['test_f1']))
            test_f1 = [f1 for f1 in self.history['test_f1'] if f1 is not None]
            epochs_with_test = [e for e, f1 in zip(epochs, self.history['test_f1']) if f1 is not None]
            plt.plot(epochs_with_test, test_f1, 'o-', label='Test F1')
        plt.title(f'{self.modality} Encoder Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
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
        plt.savefig(os.path.join(self.save_dir, f"{self.modality}_encoder_training_history.png"))
        plt.close()
        
        self.logger.info(f"保存训练历史图表: {os.path.join(self.save_dir, f'{self.modality}_encoder_training_history.png')}")