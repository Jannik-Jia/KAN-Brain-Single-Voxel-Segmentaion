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
import h5py
import sys
from torch.cuda import amp
import math
import torch.nn.functional as F
from evaluation.group_evaluator import GroupEvaluator
from torch.utils.data import Sampler
from collections import defaultdict
import random

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logging_utils import Logger
from models.mfcan import MFCAN
class BalancedBatchSampler(Sampler):
    """
    按类别平衡的批次采样器
    每个epoch从每个类别中采样指定数量的样本
    """
    
    def __init__(self, dataset, labels, samples_per_class=1000, shuffle=True):
        """
        初始化平衡批次采样器
        
        参数:
            dataset: PyTorch数据集
            labels: 样本标签数组 [num_samples]
            samples_per_class: 每个类别每个epoch采样的样本数量
            shuffle: 是否打乱样本顺序
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.samples_per_class = samples_per_class
        
        # 按类别索引样本
        self.label_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.label_indices[label.item()].append(idx)
        
        # 计算类别数和每个类别的样本数
        self.num_classes = len(self.label_indices)
        self.class_sizes = {label: len(indices) for label, indices in self.label_indices.items()}
        
        # 确定每个类别的实际采样数量
        self.actual_samples_per_class = {
            label: min(samples_per_class, size) 
            for label, size in self.class_sizes.items()
        }
        
        # 计算一个epoch的总样本数
        self.total_samples = sum(self.actual_samples_per_class.values())
    
    def __iter__(self):
        # 每个epoch重新采样
        indices = []
        
        for label, label_indices in self.label_indices.items():
            # 确定要采样的数量
            n_samples = self.actual_samples_per_class[label]
            
            # 如果类别样本数超过要采样的数量，随机采样
            if len(label_indices) > n_samples:
                sampled_indices = random.sample(label_indices, n_samples)
            else:
                # 否则全部使用，并可能重复采样以达到要求的数量
                if n_samples > len(label_indices):
                    # 完整使用所有样本
                    sampled_indices = label_indices.copy()
                    # 添加额外样本以达到要求的数量
                    extra_samples = random.choices(label_indices, k=n_samples-len(label_indices))
                    sampled_indices.extend(extra_samples)
                else:
                    sampled_indices = label_indices.copy()
            
            indices.extend(sampled_indices)
        
        # 打乱所有采样的索引
        if self.shuffle:
            random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.total_samples



class MFCANTrainer:
    """MFCAN模型训练器"""
    

    def __init__(self, config_path, data_path=None, output_dir=None, device=None, logger=None):
        """
        初始化MFCAN训练器
        
        参数:
            config_path: 配置文件路径
            data_path: 数据文件路径
            output_dir: 输出目录，若不指定则使用配置文件中的设置
            device: 训练设备
            logger: 日志记录器
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 设置日志记录器
        if logger:
            self.logger = logger
        else:
            log_manager = Logger("MFCANTrainer", log_dir="logs/training")
            self.logger = log_manager.get_logger()
        
        # 加载配置 - 添加健壮的错误处理
        try:
            # 检查文件是否存在
            if not os.path.exists(config_path):
                self.logger.error(f"配置文件不存在: {config_path}")
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            # 读取配置文件
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # 验证配置是否为空
            if not self.config:
                self.logger.error(f"配置文件为空或格式错误: {config_path}")
                raise ValueError(f"配置文件为空或格式错误: {config_path}")
                
            self.logger.info(f"成功加载配置文件: {config_path}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件JSON格式错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
        
        # 设置输出目录
        self.output_dir = output_dir or self.config.get('save_dir', 'results/mfcan')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置到输出目录
        output_config_path = os.path.join(self.output_dir, 'config.json')
        try:
            with open(output_config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"配置已保存到: {output_config_path}")
        except Exception as e:
            self.logger.warning(f"保存配置文件失败: {e}")
        
        # 训练参数
        training_config = self.config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 100)
        self.initial_lr = training_config.get('initial_lr', 1e-4)
        self.finetune_lr = training_config.get('finetune_lr', 5e-5)
        self.batch_size = training_config.get('batch_size', 256)
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.aux_weight = training_config.get('aux_loss_weight', 0.3)
        self.pretrain_epochs = training_config.get('pretrain_epochs', 30)
        self.finetune_epochs = training_config.get('finetune_epochs', 70)
        self.early_stopping = training_config.get('early_stopping', 10)
        
        self.logger.info(f"MFCAN Trainer initialized: device={self.device}, output_dir={self.output_dir}")
        
        # 创建模型
        self.model = self._build_model()

        loss_config = self.config.get('loss', {})
        self.criterion = MFCANLoss(
        num_classes=self.config.get('main_classifier', {}).get('num_classes', 102),
        aux_weight=loss_config.get('aux_weight', 0.3),
        attn_reg_weight=loss_config.get('attn_reg_weight', 0.01),
        modal_balance_weight=loss_config.get('modal_balance_weight', 0.01),
        class_balance_method=loss_config.get('class_balance_method', 'effective_samples'),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        cb_beta=loss_config.get('cb_beta', 0.9999),
        cb_samples_per_class=None  # 初始为None，稍后计算
    )
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'test_acc': [],
            'test_f1': [],
            'learning_rates': [],
            'attention_weights': []
        }
        
        # 数据加载器
        self.data_loaders = None
        if data_path:
            self.load_data(data_path)

    
    def _build_model(self):
        """创建MFCAN模型"""
        self.logger.info("Building MFCAN model...")
        
        # 检查是否有预训练编码器路径
        pretrained_encoders = {}
        if 'pretrained_encoders' in self.config:
            pretrained_encoders = self.config['pretrained_encoders']
            
            # 检查路径是否存在
            for modality, path in pretrained_encoders.items():
                if not os.path.exists(path) and not (os.path.exists(path + ".ckpt") or os.path.exists(path + ".full")):
                    self.logger.warning(f"Pretrained encoder path for {modality} does not exist: {path}")
        
        # 创建模型，传递logger参数
        model = MFCAN(self.config, pretrained_encoders, logger=self.logger)
        model = model.to(self.device)
        self.logger.info(f"MFCAN model created with config: {self.config}")
        
        return model

    def load_data(self, data_path):
        """
        加载数据
        
        参数:
            data_path: 数据文件路径
        """
        self.logger.info(f"Loading data from {data_path}...")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # 检查数据文件的结构
                keys = list(f.keys())
                self.logger.info(f"Data file structure: {keys}")
                
                # 提取每个模态的数据
                # 这里假设数据文件的结构为 group_modality/split/features 和 /split/labels
                # 实际使用时可能需要根据数据文件结构进行调整
                
                # 提取标签
                if 'train' in f and 'labels' in f['train']:
                    train_labels = f['train/labels'][()]
                    val_labels = f['val/labels'][()]
                    test_labels = f['test/labels'][()]
                    self.logger.info(f"Labels loaded: train={train_labels.shape}, val={val_labels.shape}, test={test_labels.shape}")
                else:
                    raise KeyError("Cannot find labels in the data file")
                
                # 提取各模态特征
                modalities = ['diffusion', 'qti', 'cest']
                train_features = {}
                val_features = {}
                test_features = {}
                
                for modality in modalities:
                    # 尝试不同的数据路径格式
                    possible_paths = [
                        f'group_{modality}/train/features',
                        f'{modality}/train/features',
                        f'original/{modality}/train'
                    ]
                    
                    found = False
                    for path in possible_paths:
                        if path in f:
                            train_features[modality] = f[path][()]
                            val_path = path.replace('train', 'val')
                            test_path = path.replace('train', 'test')
                            
                            if val_path in f:
                                val_features[modality] = f[val_path][()]
                            if test_path in f:
                                test_features[modality] = f[test_path][()]
                                
                            self.logger.info(f"{modality} features loaded from {path}: train={train_features[modality].shape}")
                            found = True
                            break
                    
                    if not found:
                        raise KeyError(f"Cannot find {modality} features in the data file")
            
            # 创建张量数据集
            self.logger.info("Creating tensor datasets...")
            
            # 处理标签（如果需要从1开始调整为从0开始）
            if np.min(train_labels) == 1:
                train_labels = train_labels - 1
                val_labels = val_labels - 1
                test_labels = test_labels - 1
                self.logger.info("Labels adjusted from 1-based to 0-based")
            
            # 创建PyTorch张量
            train_tensors = {modality: torch.FloatTensor(features) for modality, features in train_features.items()}
            val_tensors = {modality: torch.FloatTensor(features) for modality, features in val_features.items()}
            test_tensors = {modality: torch.FloatTensor(features) for modality, features in test_features.items()}
            
            train_labels_tensor = torch.LongTensor(train_labels)
            val_labels_tensor = torch.LongTensor(val_labels)
            test_labels_tensor = torch.LongTensor(test_labels)
            
            # 创建数据集
            train_dataset = TensorMultiModalDataset(train_tensors, train_labels_tensor)
            val_dataset = TensorMultiModalDataset(val_tensors, val_labels_tensor)
            test_dataset = TensorMultiModalDataset(test_tensors, test_labels_tensor)
            
            # 创建数据加载器
            self.data_loaders = {
                'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            }
            
            self.logger.info(f"Data loaders created: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
            
            # 保存类别数量
            self.num_classes = len(np.unique(train_labels))
            self.logger.info(f"Number of classes: {self.num_classes}")
        
            class_counts = self.compute_class_counts(self.data_loaders['train'])
            if class_counts is not None and hasattr(self, 'criterion') and isinstance(self.criterion, MFCANLoss):
                self.criterion.update_class_counts(class_counts)
                self.logger.info("已更新损失函数中的类别样本计数")
            
            return self.data_loaders
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    

    def analyze_feature_group_contributions(self, data_path=None, max_combination_size=3, output_dir=None):
        """
        分析不同特征组对模型性能的贡献
        
        参数:
            data_path: 数据文件路径，若不指定则使用当前加载的数据
            max_combination_size: 最大组合大小，默认为3以减少计算量
            output_dir: 输出目录，若不指定则使用默认目录
            
        返回:
            evaluation_results: 特征组评估结果
        """

        
        self.logger.info("开始分析特征组贡献...")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "feature_groups")
        
        # 创建模型模板 - 使用与MFCAN相似的更简单模型作为模板
        model_template = self._create_group_evaluator_model_template()
        
        # 创建评估器
        evaluator = GroupEvaluator(
            model_template=model_template,
            config_path=None,  # 使用内部配置
            output_dir=output_dir,
            logger=self.logger,
            device=self.device
        )
        
        # 设置评估参数
        evaluator.num_epochs = 20  # 减少训练轮次以加快评估
        evaluator.batch_size = self.batch_size
        evaluator.learning_rate = self.initial_lr
        evaluator.weight_decay = self.weight_decay
        evaluator.early_stopping = 5
        
        # 使用提供的数据路径或当前数据
        if data_path is None:
            if hasattr(self, 'data_path') and self.data_path:
                data_path = self.data_path
            else:
                self.logger.error("未提供数据路径且当前未加载数据")
                return None
        
        # 运行评估
        evaluation_results = evaluator.evaluate_all_group_combinations(
            data_path=data_path,
            max_combination_size=max_combination_size
        )
        
        self.logger.info("特征组贡献分析完成")
        return evaluation_results

    def _create_group_evaluator_model_template(self):
        """创建用于特征组评估的简化模型模板"""
        # 这里我们创建一个简化版的MFCAN，只包含必要组件
        # 可以是一个简单的MLP或者类似结构
        
        # 导入必要组件
        from models.classifiers.classification_head import ClassificationHead
        
        class GroupEvaluationModel(nn.Module):
            """特征组评估专用的简化模型"""
            
            def __init__(self):
                super(GroupEvaluationModel, self).__init__()
                # 初始为空，稍后动态填充
                self.input_dims = {}
                self.group_projections = nn.ModuleDict()
                self.main_classifier = None
                self.num_classes = 102  # 默认值，会在创建时更新
            
            def forward(self, features):
                """前向传播，输入为特征字典"""
                if not self.group_projections:
                    self._initialize_layers()
                
                # 处理每个特征组
                projected_features = []
                for group_name, feature in features.items():
                    if group_name in self.group_projections:
                        proj = self.group_projections[group_name](feature)
                        projected_features.append(proj)
                
                # 如果没有有效特征，返回空
                if not projected_features:
                    raise ValueError("没有匹配的特征组")
                
                # 连接所有特征
                concatenated = torch.cat(projected_features, dim=1)
                
                # 通过分类器
                output = self.main_classifier(concatenated)
                return output
            
            def _initialize_layers(self):
                """根据输入维度初始化层"""
                total_dim = 0
                projection_dim = 64  # 每个组的投影维度
                
                # 为每个特征组创建投影层
                for group_name, dim in self.input_dims.items():
                    self.group_projections[group_name] = nn.Sequential(
                        nn.Linear(dim, projection_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    )
                    total_dim += projection_dim
                
                # 创建主分类器
                self.main_classifier = ClassificationHead(
                    input_dim=total_dim,
                    hidden_dim=512,
                    num_classes=self.num_classes,
                    dropout=0.4
                )
        
        # 创建模型实例
        model_template = GroupEvaluationModel()
        model_template.num_classes = self.config.get('main_classifier', {}).get('num_classes', 102)
        
        return model_template


    def load_data_with_balanced_sampling(self, data_path, samples_per_class=1000, use_balanced_sampler=False):
        """
        加载数据，可选择使用平衡采样
        
        参数:
            data_path: 数据文件路径
            samples_per_class: 每个类别每个epoch采样的样本数量
            use_balanced_sampler: 是否使用平衡采样器
        """
        # 首先加载原始数据
        data_loaders = self.load_data(data_path)
        
        # 如果不使用平衡采样器，直接返回标准数据加载器
        if not use_balanced_sampler:
            return data_loaders
        
        self.logger.info(f"创建平衡采样器，每个类别采样 {samples_per_class} 个样本")
        
        # 获取训练集
        train_dataset = self.data_loaders['train'].dataset
        
        # 收集所有标签
        all_labels = []
        for _, label in train_dataset:
            all_labels.append(label.item())
        
        # 创建平衡采样器
        balanced_sampler = BalancedBatchSampler(
            train_dataset, 
            all_labels, 
            samples_per_class=samples_per_class,
            shuffle=True
        )
        
        # 创建使用平衡采样器的数据加载器
        balanced_train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=balanced_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 替换训练数据加载器
        self.data_loaders['train'] = balanced_train_loader
        
        # 记录平衡采样信息
        class_counts = np.zeros(self.num_classes, dtype=np.int32)
        for label in all_labels:
            class_counts[label] += 1
        
        # 记录原始样本数和平衡后样本数
        self.logger.info("类别分布情况:")
        for c in range(self.num_classes):
            actual_samples = min(samples_per_class, class_counts[c])
            self.logger.info(f"类别 {c}: 原始 {class_counts[c]} 个样本，平衡后 {actual_samples} 个样本")
        
        # 更新损失函数中的类别样本计数（使用平衡后的计数）
        balanced_counts = np.array([min(samples_per_class, count) for count in class_counts])
        if hasattr(self, 'criterion') and isinstance(self.criterion, MFCANLoss):
            self.criterion.update_class_counts(balanced_counts)
            self.logger.info("已更新损失函数中的类别样本计数为平衡后的数量")
        
        return self.data_loaders


    def _create_lr_scheduler(self, optimizer, num_epochs, num_training_steps=None):
        """
        创建学习率调度器，支持预热阶段
        
        参数:
            optimizer: 优化器
            num_epochs: 总轮次数
            num_training_steps: 训练步数 (用于OneCycleLR)
            
        返回:
            scheduler: 学习率调度器
        """
        scheduler_type = self.config.get('training', {}).get('scheduler', 'cosine')
        warmup_ratio = self.config.get('training', {}).get('warmup_ratio', 0.1)
        
        if scheduler_type == 'cosine_warmup':
            # 带预热的余弦退火
            def lr_lambda(current_step):
                # 计算预热步数
                warmup_steps = int(num_epochs * warmup_ratio)
                
                # 预热阶段
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                
                # 余弦退火阶段
                progress = float(current_step - warmup_steps) / float(max(1, num_epochs - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
                
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_type == 'onecycle':
            assert num_training_steps is not None, "需要指定训练步数用于OneCycleLR"
            return optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.initial_lr * 10,
                total_steps=num_training_steps,
                pct_start=warmup_ratio,
                div_factor=25.0,
                final_div_factor=1000.0
            )
        
        elif scheduler_type == 'cosine':
            # 原有的余弦退火
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        
        elif scheduler_type == 'step':
            step_size = int(num_epochs * 0.3)  # 默认在30%处下降
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        
        else:
            self.logger.warning(f"未知的学习率调度器类型: {scheduler_type}，使用默认的余弦退火")
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        


    def compute_class_counts(self, data_loader=None):
        """
        计算数据集中每个类别的样本数量
        
        参数:
            data_loader: 数据加载器，若不指定则使用训练集
            
        返回:
            class_counts: 每个类别的样本计数列表
        """
        if data_loader is None:
            if self.data_loaders is None or 'train' not in self.data_loaders:
                self.logger.error("训练数据加载器不可用")
                return None
            data_loader = self.data_loaders['train']
        
        self.logger.info("计算类别样本数量...")
        
        # 类别数量
        num_classes = self.config.get('main_classifier', {}).get('num_classes', 102)
        class_counts = np.zeros(num_classes, dtype=np.int32)
        
        # 遍历数据集计算每个类别的样本数
        for _, labels in data_loader:
            for label in labels.cpu().numpy():
                class_counts[label] += 1
        
        # 记录类别分布情况
        for c in range(num_classes):
            self.logger.info(f"类别 {c}: {class_counts[c]} 个样本")
        
        # 统计信息
        min_count = np.min(class_counts)
        max_count = np.max(class_counts)
        imbalance_ratio = max_count / max(min_count, 1)
        self.logger.info(f"类别分布: 最少 {min_count}, 最多 {max_count}, 不平衡比例 {imbalance_ratio:.2f}:1")
        
        return class_counts


    def compute_loss(self, outputs, targets, aux_weight=None):
        """
        计算多任务损失函数
        
        参数:
            outputs: 模型输出字典
            targets: 目标类别标签
            aux_weight: 辅助损失的权重，若不指定则使用self.aux_weight
                
        返回:
            total_loss: 总损失
            loss_info: 各组件损失的字典
        """
        # 使用MFCANLoss计算损失
        if hasattr(self, 'criterion') and isinstance(self.criterion, MFCANLoss):
            return self.criterion(outputs, targets)
        else:
            # 后备方案：使用简单的交叉熵损失
            self.logger.warning("未找到MFCANLoss实例，使用简单交叉熵损失")
            aux_weight = aux_weight or self.aux_weight
            
            # 主分类损失
            criterion = nn.CrossEntropyLoss()
            
            # 如果只有辅助输出
            if 'main_output' not in outputs:
                aux_losses = {}
                for modal, aux_output in outputs['aux_outputs'].items():
                    aux_losses[modal] = criterion(aux_output, targets)
                
                # 计算平均辅助损失
                avg_aux_loss = sum(aux_losses.values()) / len(aux_losses)
                
                # 返回损失信息
                loss_info = {
                    'total_loss': avg_aux_loss.item(),
                    'avg_aux_loss': avg_aux_loss.item(),
                    'aux_losses': {k: v.item() for k, v in aux_losses.items()}
                }
                
                return avg_aux_loss, loss_info
            
            # 主分类损失
            main_loss = criterion(outputs['main_output'], targets)
            
            # 辅助分类损失
            aux_losses = {}
            if 'aux_outputs' in outputs:
                for modal, aux_output in outputs['aux_outputs'].items():
                    aux_losses[modal] = criterion(aux_output, targets)
                
                # 计算平均辅助损失
                avg_aux_loss = sum(aux_losses.values()) / len(aux_losses)
                
                # 计算总损失
                total_loss = main_loss + aux_weight * avg_aux_loss
            else:
                # 如果没有辅助输出，只使用主损失
                total_loss = main_loss
                avg_aux_loss = torch.tensor(0.0, device=main_loss.device)
            
            # 返回损失信息
            loss_info = {
                'total_loss': total_loss.item(),
                'main_loss': main_loss.item(),
                'avg_aux_loss': avg_aux_loss.item() if isinstance(avg_aux_loss, torch.Tensor) else avg_aux_loss,
                'aux_losses': {k: v.item() for k, v in aux_losses.items()} if aux_losses else {}
            }
            
            return total_loss, loss_info
        

    def pretrain_encoders(self):
        """
        预训练各模态编码器，只训练辅助分类器
        
        返回:
            history: 训练历史记录
        """
        if self.data_loaders is None:
            self.logger.error("Data loaders not initialized. Call load_data() first.")
            return
        
        self.logger.info("Pretraining encoders...")
        
        # 设置编码器训练模式
        self.model.train()
        
        # 将所有参数设为需要梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 设置优化器
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器 - 余弦退火
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.pretrain_epochs
        )
        
        # 记录最佳模型
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        
        # 训练循环
        for epoch in range(self.pretrain_epochs):
            # 训练一个轮次
            train_loss, train_acc, train_f1 = self._train_epoch_encoders_only(
                self.data_loaders['train'], 
                optimizer
            )
            
            # 验证
            val_loss, val_acc, val_f1 = self._validate(
                self.data_loaders['val'], 
                training_stage='encoders_only'
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            
            # 记录测试集性能
            if (epoch + 1) % 5 == 0 or epoch == self.pretrain_epochs - 1:
                test_loss, test_acc, test_f1 = self._validate(
                    self.data_loaders['test'], 
                    training_stage='encoders_only'
                )
                self.logger.info(f"Test performance: F1={test_f1:.4f}, Acc={test_acc:.4f}, Loss={test_loss:.4f}")
                
                # 记录测试集性能
                if len(self.history['test_acc']) <= epoch:
                    # 填充之前的轮次
                    self.history['test_acc'].extend([None] * (epoch - len(self.history['test_acc']) + 1))
                    self.history['test_f1'].extend([None] * (epoch - len(self.history['test_f1']) + 1))
                else:
                    self.history['test_acc'].append(test_acc)
                    self.history['test_f1'].append(test_f1)
            
            # 打印进度
            self.logger.info(f"Pretrain Epoch {epoch+1}/{self.pretrain_epochs}: "
                        f"Train Loss={train_loss:.4f}, F1={train_f1:.4f}, Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}, "
                        f"LR={current_lr:.8f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # 保存检查点
                checkpoint_path = os.path.join(self.output_dir, f"encoder_pretrain_best_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
                self.logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Restored best model with val F1={best_val_f1:.4f}")
        
        # 保存预训练历史图表
        self._plot_training_history(os.path.join(self.output_dir, "encoder_pretrain_history.png"))
        
        return self.history
    
    def train_fusion_classifier(self):
        """
        训练融合机制和分类器，冻结编码器参数
        
        返回:
            history: 训练历史记录
        """
        if self.data_loaders is None:
            self.logger.error("Data loaders not initialized. Call load_data() first.")
            return
        
        self.logger.info("Training fusion mechanism and classifiers...")
        
        # 冻结编码器参数
        self.model.freeze_encoders()
        
        # 设置优化器
        # 只优化未冻结的参数
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器 - 余弦退火
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.pretrain_epochs
        )
        
        # 记录最佳模型
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        
        # 清空历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'test_acc': [],
            'test_f1': [],
            'learning_rates': [],
            'attention_weights': []
        }
        
        # 训练循环
        for epoch in range(self.pretrain_epochs):
            # 训练一个轮次
            train_loss, train_acc, train_f1, attn_weights = self._train_epoch(
                self.data_loaders['train'], 
                optimizer,
                training_stage='fusion_only'
            )
            
            # 验证
            val_loss, val_acc, val_f1 = self._validate(
                self.data_loaders['val']
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            self.history['attention_weights'].append(attn_weights.cpu().numpy())
            
            # 记录测试集性能
            if (epoch + 1) % 5 == 0 or epoch == self.pretrain_epochs - 1:
                test_loss, test_acc, test_f1 = self._validate(
                    self.data_loaders['test']
                )
                self.logger.info(f"Test performance: F1={test_f1:.4f}, Acc={test_acc:.4f}, Loss={test_loss:.4f}")
                
                # 记录测试集性能
                if len(self.history['test_acc']) <= epoch:
                    # 填充之前的轮次
                    self.history['test_acc'].extend([None] * (epoch - len(self.history['test_acc']) + 1))
                    self.history['test_f1'].extend([None] * (epoch - len(self.history['test_f1']) + 1))
                else:
                    self.history['test_acc'].append(test_acc)
                    self.history['test_f1'].append(test_f1)
            
            # 打印进度
            self.logger.info(f"Fusion Epoch {epoch+1}/{self.pretrain_epochs}: "
                        f"Train Loss={train_loss:.4f}, F1={train_f1:.4f}, Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}, "
                        f"LR={current_lr:.8f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # 保存检查点
                checkpoint_path = os.path.join(self.output_dir, f"fusion_best_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
                self.logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Restored best model with val F1={best_val_f1:.4f}")
        
        # 保存训练历史图表
        self._plot_training_history(os.path.join(self.output_dir, "fusion_training_history.png"))
        
        # 保存注意力权重图表
        self._plot_attention_weights(os.path.join(self.output_dir, "attention_weights.png"))
        
        return self.history
    
    def finetune_full_model(self):
        """
        微调完整模型，解冻编码器参数
        
        返回:
            history: 训练历史记录
        """
        if self.data_loaders is None:
            self.logger.error("Data loaders not initialized. Call load_data() first.")
            return
        
        self.logger.info("Fine-tuning full model...")
        
        # 解冻编码器参数
        self.model.unfreeze_encoders()
        
        # 获取参数分组，为不同组件设置不同学习率
        parameter_groups = self.model.get_parameter_groups()
        optimizer = optim.AdamW([
            {'params': parameter_groups[0]['params'], 'lr': self.finetune_lr},  # 编码器
            {'params': parameter_groups[1]['params'], 'lr': self.initial_lr},   # 融合
            {'params': parameter_groups[2]['params'], 'lr': self.initial_lr}    # 辅助
        ], weight_decay=self.weight_decay)

        # 计算总训练步数（用于OneCycleLR）
        total_steps = len(self.data_loaders['train']) * self.finetune_epochs

        # 使用带预热的学习率调度器
        scheduler = self._create_lr_scheduler(optimizer, self.finetune_epochs, num_training_steps=total_steps)

        
        # 记录最佳模型
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        
        # 清空历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'test_acc': [],
            'test_f1': [],
            'learning_rates': [],
            'attention_weights': []
        }
        
        # 训练循环
        for epoch in range(self.finetune_epochs):
            # 训练一个轮次
            train_loss, train_acc, train_f1, attn_weights = self._train_epoch(
                self.data_loaders['train'], 
                optimizer,
                training_stage='all'
            )
            
            # 验证
            val_loss, val_acc, val_f1 = self._validate(
                self.data_loaders['val']
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            self.history['attention_weights'].append(attn_weights.cpu().numpy())
            
            # 记录测试集性能
            if (epoch + 1) % 5 == 0 or epoch == self.finetune_epochs - 1:
                test_loss, test_acc, test_f1 = self._validate(
                    self.data_loaders['test']
                )
                self.logger.info(f"Test performance: F1={test_f1:.4f}, Acc={test_acc:.4f}, Loss={test_loss:.4f}")
                
                # 记录测试集性能
                if len(self.history['test_acc']) <= epoch:
                    # 填充之前的轮次
                    self.history['test_acc'].extend([None] * (epoch - len(self.history['test_acc']) + 1))
                    self.history['test_f1'].extend([None] * (epoch - len(self.history['test_f1']) + 1))
                else:
                    self.history['test_acc'].append(test_acc)
                    self.history['test_f1'].append(test_f1)
            
            # 打印进度
            self.logger.info(f"Finetune Epoch {epoch+1}/{self.finetune_epochs}: "
                        f"Train Loss={train_loss:.4f}, F1={train_f1:.4f}, Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}, "
                        f"LR={current_lr:.8f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # 保存检查点
                checkpoint_path = os.path.join(self.output_dir, f"finetune_best_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
                self.logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Restored best model with val F1={best_val_f1:.4f}")
        
        # 保存训练历史图表
        self._plot_training_history(os.path.join(self.output_dir, "finetune_training_history.png"))
        
        # 保存注意力权重图表
        self._plot_attention_weights(os.path.join(self.output_dir, "finetune_attention_weights.png"))
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "mfcan_final.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_f1': best_val_f1
        }, final_model_path)
        self.logger.info(f"Saved final model to {final_model_path}")
        
        return self.history
    
    def train_full_pipeline(self):
        """
        执行完整的训练流程
        
        返回:
            history: 训练历史记录
        """
        self.logger.info("Starting full training pipeline...")
        
        # 1. 预训练编码器
        self.pretrain_encoders()
        
        # 2. 训练融合机制和分类器
        self.train_fusion_classifier()
        
        # 3. 微调完整模型
        self.finetune_full_model()
        
        self.logger.info("Full training pipeline completed!")
        
        return self.history
    
    def _train_epoch_encoders_only(self, train_loader, optimizer):
        """
        训练一个轮次 - 只训练编码器部分
        
        参数:
            train_loader: 训练数据加载器
            optimizer: 优化器
            
        返回:
            avg_loss: 平均损失
            accuracy: 分类准确率
            f1_score: 宏平均F1分数
        """
        self.model.train()
        total_loss = 0
        all_targets = []
        all_predictions = {}  # 每个模态的预测
        batch_count = 0
        total_samples = 0
        
        # 记录开始时间
        start_time = time.time()
        self.logger.info("开始编码器训练阶段")
        
        for features, targets in train_loader:
            batch_count += 1
            # 将数据移到设备
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)
            batch_size = targets.size(0)
            total_samples += batch_size
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播 - 只通过编码器
            outputs = self.model(features, training_stage='encoders_only')
            
            # 计算损失
            loss, loss_info = self.compute_loss(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss_info['total_loss'] * batch_size
            
            # 收集预测和标签
            for modal, aux_output in outputs['aux_outputs'].items():
                _, predicted = aux_output.max(1)
                if modal not in all_predictions:
                    all_predictions[modal] = []
                all_predictions[modal].extend(predicted.cpu().numpy())
            
            all_targets.extend(targets.cpu().numpy())
            
            # 每处理10个批次记录一次进度
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                self.logger.info(f"编码器训练中... 已处理 {batch_count}/{len(train_loader)} 批次, "
                            f"耗时: {elapsed:.2f}秒, "
                            f"批次损失: {loss_info['total_loss']:.4f}")
        
        # 计算统计量
        avg_loss = total_loss / total_samples
        
        # 计算平均准确率和F1分数
        accuracies = []
        f1_scores = []
        for modal, predictions in all_predictions.items():
            acc = accuracy_score(all_targets, predictions)
            f1 = f1_score(all_targets, predictions, average='macro')
            accuracies.append(acc)
            f1_scores.append(f1)
            
            # 为每个模态记录性能
            self.logger.info(f"模态 {modal} 性能 - 准确率: {acc:.4f}, 宏平均F1: {f1:.4f}")
        
        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)
        
        # 记录训练统计信息
        elapsed = time.time() - start_time
        self.logger.info(f"编码器训练阶段完成 - "
                    f"样本数: {total_samples}, "
                    f"耗时: {elapsed:.2f}秒, "
                    f"平均损失: {avg_loss:.4f}, "
                    f"平均准确率: {avg_accuracy:.4f}, "
                    f"平均宏平均F1: {avg_f1:.4f}")
        
        return avg_loss, avg_accuracy, avg_f1
    
    def _train_epoch(self, train_loader, optimizer, training_stage='all'):
        """
        训练一个轮次
        
        参数:
            train_loader: 训练数据加载器
            optimizer: 优化器
            training_stage: 训练阶段 'encoders_only', 'fusion_only', 'all'
            
        返回:
            avg_loss: 平均损失
            accuracy: 分类准确率
            f1_score: 宏平均F1分数
            attention_weights: 注意力权重
        """
        self.model.train()
        total_loss = 0
        all_targets = []
        all_predictions = []
        batch_attention_weights = []
        batch_count = 0
        total_samples = 0
        
        # 记录开始时间
        start_time = time.time()
        self.logger.info(f"开始训练阶段: {training_stage}")
        
        for features, targets in train_loader:
            batch_count += 1
            # 将数据移到设备
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)
            batch_size = targets.size(0)
            total_samples += batch_size
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(features, training_stage=training_stage)
            
            # 计算损失
            loss, loss_info = self.compute_loss(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss_info['total_loss'] * batch_size
            
            # 收集预测和标签
            if 'main_output' in outputs:
                _, predicted = outputs['main_output'].max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            # 收集注意力权重
            if 'attention_weights' in outputs:
                batch_attention_weights.append(outputs['attention_weights'].detach().mean(0))
            
            # 每处理10个批次记录一次进度
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                self.logger.info(f"训练中... 已处理 {batch_count}/{len(train_loader)} 批次, "
                            f"耗时: {elapsed:.2f}秒, "
                            f"批次损失: {loss_info['total_loss']:.4f}")
        
        # 计算统计量
        avg_loss = total_loss / total_samples
        
        # 计算准确率和F1分数
        if all_predictions:
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average='macro')
        else:
            accuracy = 0.0
            f1 = 0.0
        
        # 计算平均注意力权重
        if batch_attention_weights:
            avg_attention_weights = torch.stack(batch_attention_weights).mean(0)
        else:
            avg_attention_weights = torch.zeros(3, device=self.device)
        
        # 记录训练统计信息
        elapsed = time.time() - start_time
        self.logger.info(f"训练阶段 {training_stage} 完成 - "
                    f"样本数: {total_samples}, "
                    f"耗时: {elapsed:.2f}秒, "
                    f"平均损失: {avg_loss:.4f}, "
                    f"准确率: {accuracy:.4f}, "
                    f"宏平均F1: {f1:.4f}")
        
        return avg_loss, accuracy, f1, avg_attention_weights


    def _validate(self, val_loader, training_stage='all'):
        """
        验证模型
        
        参数:
            val_loader: 验证数据加载器
            training_stage: 训练阶段 'encoders_only', 'fusion_only', 'all'
            
        返回:
            avg_loss: 平均损失
            accuracy: 分类准确率
            f1_score: 宏平均F1分数
        """
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []
        total_samples = 0
        
        # 记录开始时间
        start_time = time.time()
        self.logger.info(f"开始验证...")
        
        with torch.no_grad():
            for features, targets in val_loader:
                # 将数据移到设备
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                batch_size = targets.size(0)
                total_samples += batch_size
                
                # 前向传播
                outputs = self.model(features, training_stage=training_stage)
                
                # 计算损失
                loss, loss_info = self.compute_loss(outputs, targets)
                
                # 统计
                total_loss += loss_info['total_loss'] * batch_size
                
                # 收集预测和标签
                if training_stage == 'encoders_only':
                    # 使用辅助分类器的预测
                    aux_predictions = []
                    aux_confidences = []
                    for modal, aux_output in outputs['aux_outputs'].items():
                        probs = F.softmax(aux_output, dim=1)
                        confidence, predicted = torch.max(probs, dim=1)
                        aux_predictions.append(predicted)
                        aux_confidences.append(confidence)
                    
                    # 改进的投票策略:
                    # 1. 先尝试多数投票
                    stacked_preds = torch.stack(aux_predictions)  # [n_modalities, batch_size]
                    
                    # 计算每个类别的票数
                    num_classes = self.model.auxiliary_classifiers['diffusion'].classifier[-1].out_features
                    votes = torch.zeros(batch_size, num_classes, device=self.device)
                    for preds in aux_predictions:
                        votes = votes.to(torch.float)
                        index_tensor = preds.unsqueeze(1).to(torch.long)  # 索引必须是整数类型
                        value_tensor = torch.ones_like(preds, dtype=torch.float, device=self.device).unsqueeze(1)
                        votes.scatter_add_(1, index_tensor, value_tensor)
                    # 找出得票最多的类别
                    max_votes, modal_votes = torch.max(votes, dim=1)
                    
                    # 检查是否存在票数相同的情况
                    max_vote_counts = (votes == max_votes.unsqueeze(1)).sum(dim=1)
                    tie_mask = max_vote_counts > 1  # 标记存在票数相同的样本
                    
                    # 对于票数相同的情况，使用置信度最高的预测
                    if tie_mask.any():
                        # 合并置信度
                        stacked_conf = torch.stack(aux_confidences)  # [n_modalities, batch_size]
                        
                        # 对于每个样本，找出每个类别的最高置信度
                        for i in range(batch_size):
                            if tie_mask[i]:
                                # 找出该样本的所有预测及对应置信度
                                sample_preds = stacked_preds[:, i]
                                sample_confs = stacked_conf[:, i]
                                
                                # 只考虑得票最多的类别
                                tied_classes = []
                                for c in range(num_classes):
                                    if votes[i, c] == max_votes[i]:
                                        tied_classes.append(c)
                                
                                # 找出这些类别中置信度最高的一个
                                best_conf = -1
                                best_class = -1
                                for c in tied_classes:
                                    # 找出预测为类别c的所有模态中，置信度最高的
                                    mask = sample_preds == c
                                    if mask.any():
                                        conf = torch.max(sample_confs[mask])
                                        if conf > best_conf:
                                            best_conf = conf
                                            best_class = c
                                
                                # 更新预测
                                if best_class != -1:
                                    modal_votes[i] = best_class
                    
                    all_predictions.extend(modal_votes.cpu().numpy())
                else:
                    _, predicted = outputs['main_output'].max(1)
                    all_predictions.extend(predicted.cpu().numpy())
                
                all_targets.extend(targets.cpu().numpy())
        
        # 计算统计量
        avg_loss = total_loss / total_samples
        
        # 计算准确率和F1分数
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='macro')
        
        # 计算各类别的准确率和F1分数
        # 使用classification_report获取详细的分类报告
        report = classification_report(all_targets, all_predictions, output_dict=True)
        
        # 记录验证统计信息
        elapsed = time.time() - start_time
        self.logger.info(f"验证完成 - "
                    f"样本数: {total_samples}, "
                    f"耗时: {elapsed:.2f}秒, "
                    f"验证损失: {avg_loss:.4f}, "
                    f"准确率: {accuracy:.4f}, "
                    f"宏平均F1: {f1:.4f}")
        
        # 记录更详细的分类性能
        self.logger.info(f"分类详情 - "
                    f"加权F1: {report['weighted avg']['f1-score']:.4f}, "
                    f"精确率: {report['weighted avg']['precision']:.4f}, "
                    f"召回率: {report['weighted avg']['recall']:.4f}")
        
        return avg_loss, accuracy, f1

    
    def evaluate(self, test_loader=None):
        """
        评估模型
        
        参数:
            test_loader: 测试数据加载器，若不指定则使用self.data_loaders['test']
            
        返回:
            metrics: 评估指标字典
        """
        if test_loader is None:
            if self.data_loaders is None or 'test' not in self.data_loaders:
                self.logger.error("Test data loader not available.")
                return
            test_loader = self.data_loaders['test']
        
        self.logger.info("Evaluating model...")
        self.model.eval()
        
        all_targets = []
        all_predictions = []
        all_probabilities = []
        attention_weights_list = []
        
        with torch.no_grad():
            for features, targets in tqdm(test_loader, desc="Evaluating"):
                # 将数据移到设备
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                
                # 获取预测和概率
                probabilities = F.softmax(outputs['main_output'], dim=1)
                _, predicted = outputs['main_output'].max(1)
                
                # 收集注意力权重
                if 'attention_weights' in outputs:
                    attention_weights_list.append(outputs['attention_weights'].cpu().numpy())
                
                # 收集结果
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 转换为NumPy数组
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # 分析各类别性能
        classes = np.unique(np.concatenate([all_targets, all_predictions]))
        class_report = classification_report(all_targets, all_predictions, output_dict=True)
        
        # 分析注意力权重
        if attention_weights_list:
            attention_weights = np.concatenate(attention_weights_list, axis=0)
            mean_attention = np.mean(attention_weights, axis=0)  # [3]
            
            # 按类别分析注意力权重
            class_attention = {}
            for cls in classes:
                class_mask = all_targets == cls
                if np.any(class_mask):
                    class_attention[int(cls)] = np.mean(attention_weights[class_mask], axis=0)
        else:
            mean_attention = None
            class_attention = None
        
        # 生成评估报告
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'class_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'mean_attention': mean_attention.tolist() if mean_attention is not None else None,
            'class_attention': {str(k): v.tolist() for k, v in class_attention.items()} if class_attention else None
        }
        
        # 保存评估结果
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 生成图表
        self._plot_confusion_matrix(conf_matrix, classes[:20], "Top 20 Classes")
        
        # 生成注意力权重图表
        if mean_attention is not None:
            self._plot_class_attention(class_attention, classes[:20])
        
        self.logger.info(f"Evaluation results: Accuracy={accuracy:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}")
        self.logger.info(f"Evaluation results saved to {results_path}")
        
        return metrics
    
    def save_model(self, path=None):
        """
        保存模型
        
        参数:
            path: 保存路径，若不指定则使用默认路径
        """
        if path is None:
            path = os.path.join(self.output_dir, "mfcan_model.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        
        self.logger.info(f"Model saved to {path}")
        
        return path
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
            path: 模型路径
        """
        if not os.path.exists(path):
            self.logger.error(f"Model file not found: {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # 重新创建模型
        self.model = self._build_model()
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.logger.info(f"Model loaded from {path}")
        
        return True
    
    def _plot_training_history(self, save_path=None):
        """
        绘制训练历史图表
        
        参数:
            save_path: 保存路径，若不指定则使用默认路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_history.png")
        
        plt.figure(figsize=(15, 15))
        
        # 绘制损失曲线
        plt.subplot(3, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率曲线
        plt.subplot(3, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        if 'test_acc' in self.history and any(x is not None for x in self.history['test_acc']):
            valid_indices = [i for i, x in enumerate(self.history['test_acc']) if x is not None]
            test_values = [self.history['test_acc'][i] for i in valid_indices]
            plt.plot(valid_indices, test_values, 'o-', label='Test Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 绘制F1曲线
        plt.subplot(3, 2, 3)
        plt.plot(self.history['train_f1'], label='Train F1')
        plt.plot(self.history['val_f1'], label='Val F1')
        if 'test_f1' in self.history and any(x is not None for x in self.history['test_f1']):
            valid_indices = [i for i, x in enumerate(self.history['test_f1']) if x is not None]
            test_values = [self.history['test_f1'][i] for i in valid_indices]
            plt.plot(valid_indices, test_values, 'o-', label='Test F1')
        plt.title('F1 Score Curves')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # 绘制学习率曲线
        plt.subplot(3, 2, 4)
        plt.plot(self.history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # 如果有注意力权重历史，绘制注意力权重
        if 'attention_weights' in self.history and self.history['attention_weights']:
            plt.subplot(3, 2, 5)
            attention_weights = np.array(self.history['attention_weights'])
            for i, modality in enumerate(['Diffusion', 'QTI', 'CEST']):
                plt.plot(attention_weights[:, i], label=modality)
            plt.title('Attention Weights')
            plt.xlabel('Epoch')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Training history saved to {save_path}")
    
    def _plot_attention_weights(self, save_path=None):
        """
        绘制注意力权重图表
        
        参数:
            save_path: 保存路径，若不指定则使用默认路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "attention_weights.png")
        
        if 'attention_weights' not in self.history or not self.history['attention_weights']:
            self.logger.warning("No attention weights history available")
            return
        
        plt.figure(figsize=(10, 6))
        
        attention_weights = np.array(self.history['attention_weights'])
        for i, modality in enumerate(['Diffusion', 'QTI', 'CEST']):
            plt.plot(attention_weights[:, i], label=modality)
        
        plt.title('Modal Attention Weights Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Attention Weight')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Attention weights visualization saved to {save_path}")
    
    def _plot_confusion_matrix(self, conf_matrix, classes, title):
        """
        绘制混淆矩阵
        
        参数:
            conf_matrix: 混淆矩阵
            classes: 类别列表
            title: 图表标题
        """
        save_path = os.path.join(self.output_dir, f"confusion_matrix_{title.replace(' ', '_')}.png")
        
        plt.figure(figsize=(12, 10))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 添加数值标签
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                       horizontalalignment="center",
                       color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_class_attention(self, class_attention, classes):
        """
        绘制类别注意力权重
        
        参数:
            class_attention: 类别注意力权重字典
            classes: 类别列表
        """
        save_path = os.path.join(self.output_dir, "class_attention_weights.png")
        
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        diffusion_weights = []
        qti_weights = []
        cest_weights = []
        
        class_labels = []
        for cls in classes:
            if str(cls) in class_attention:
                weights = class_attention[str(cls)]
                diffusion_weights.append(weights[0])
                qti_weights.append(weights[1])
                cest_weights.append(weights[2])
                class_labels.append(f"Class {cls}")
        
        # 绘制柱状图
        x = np.arange(len(class_labels))
        width = 0.25
        
        plt.bar(x - width, diffusion_weights, width, label='Diffusion')
        plt.bar(x, qti_weights, width, label='QTI')
        plt.bar(x + width, cest_weights, width, label='CEST')
        
        plt.xlabel('Class')
        plt.ylabel('Attention Weight')
        plt.title('Modal Attention Weights by Class')
        plt.xticks(x, class_labels, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Class attention weights visualization saved to {save_path}")






    def analyze_feature_importance(self, data_loader=None, num_samples=1000, method='permutation'):
        """
        分析特征重要性
        
        参数:
            data_loader: 数据加载器，若不指定则使用测试集
            num_samples: 用于分析的样本数量
            method: 分析方法，可选 'permutation', 'shap', 'integrated_gradients'
            
        返回:
            importance: 特征重要性字典
        """
        if data_loader is None:
            if self.data_loaders is None or 'test' not in self.data_loaders:
                self.logger.error("Test data loader not available.")
                return
            data_loader = self.data_loaders['test']
        
        self.logger.info(f"Analyzing feature importance using {method} method...")
        self.model.eval()
        
        if method == 'permutation':
            return self._analyze_permutation_importance(data_loader, num_samples)
        elif method == 'integrated_gradients':
            return self._analyze_integrated_gradients(data_loader, num_samples)
        else:
            self.logger.error(f"Unknown importance analysis method: {method}")
            return None

    def _analyze_permutation_importance(self, data_loader, num_samples):
        """使用排列重要性方法分析特征重要性"""
        # 收集基准性能
        self.logger.info("Collecting baseline performance...")
        baseline_performance = self._evaluate_subset(data_loader, num_samples)
        self.logger.info(f"Baseline accuracy: {baseline_performance['accuracy']:.4f}")
        
        importance = {'diffusion': {}, 'qti': {}, 'cest': {}}
        
        # 对每个模态的特征进行排列重要性分析
        for modality in ['diffusion', 'qti', 'cest']:
            feature_dim = self.config[f'{modality}_encoder']['input_dim']
            self.logger.info(f"Analyzing {modality} features ({feature_dim} dims)...")
            
            # 如果特征维度太高，可以分组分析
            group_size = min(5, feature_dim)
            num_groups = feature_dim // group_size + (1 if feature_dim % group_size > 0 else 0)
            
            for group in range(num_groups):
                start_idx = group * group_size
                end_idx = min((group + 1) * group_size, feature_dim)
                
                feature_indices = list(range(start_idx, end_idx))
                self.logger.info(f"Group {group+1}/{num_groups}: Features {start_idx}-{end_idx-1}")
                
                # 排列这些特征并评估性能下降
                performance = self._evaluate_with_permuted_features(
                    data_loader, num_samples, modality, feature_indices)
                
                # 记录重要性（性能下降幅度）
                importance_score = baseline_performance['accuracy'] - performance['accuracy']
                
                # 保存结果
                group_name = f"f{start_idx}-{end_idx-1}"
                importance[modality][group_name] = float(importance_score)
                
                self.logger.info(f"Group {group_name} importance: {importance_score:.4f}")
        
        # 保存和可视化重要性分数
        self._visualize_feature_importance(importance)
        
        return importance

    def _evaluate_subset(self, data_loader, num_samples):
        """评估一个数据子集的性能"""
        all_targets = []
        all_predictions = []
        samples_seen = 0
        
        with torch.no_grad():
            for features, targets in data_loader:
                if samples_seen >= num_samples:
                    break
                    
                # 将数据移到设备
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                
                # 收集预测
                _, predicted = outputs['main_output'].max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                samples_seen += targets.size(0)
        
        # 计算性能指标
        all_targets = np.array(all_targets[:num_samples])
        all_predictions = np.array(all_predictions[:num_samples])
        
        return {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'f1_macro': f1_score(all_targets, all_predictions, average='macro')
        }

    def _evaluate_with_permuted_features(self, data_loader, num_samples, modality, feature_indices):
        """评估特定特征被排列后的性能"""
        all_targets = []
        all_predictions = []
        samples_seen = 0
        
        with torch.no_grad():
            for features, targets in data_loader:
                if samples_seen >= num_samples:
                    break
                    
                # 将数据移到设备
                features = {k: v.to(self.device) for k, v in features.items()}
                batch_size = targets.size(0)
                
                # 随机排列指定模态的指定特征
                perm_idx = torch.randperm(batch_size)
                features[modality][:, feature_indices] = features[modality][perm_idx][:, feature_indices]
                
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                
                # 收集预测
                _, predicted = outputs['main_output'].max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                samples_seen += batch_size
        
        # 计算性能指标
        all_targets = np.array(all_targets[:num_samples])
        all_predictions = np.array(all_predictions[:num_samples])
        
        return {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'f1_macro': f1_score(all_targets, all_predictions, average='macro')
        }

    def _visualize_feature_importance(self, importance):
        """可视化特征重要性"""
        save_path = os.path.join(self.output_dir, "feature_importance.png")
        
        # 准备数据
        modalities = []
        groups = []
        scores = []
        
        for modality, group_scores in importance.items():
            for group, score in group_scores.items():
                modalities.append(modality)
                groups.append(group)
                scores.append(score)
        
        # 创建数据框
        import pandas as pd
        df = pd.DataFrame({
            'Modality': modalities,
            'Feature Group': groups,
            'Importance': scores
        })
        
        # 按模态分组绘制
        plt.figure(figsize=(12, 8))
        
        for i, modality in enumerate(['diffusion', 'qti', 'cest']):
            if modality in df['Modality'].values:
                modal_df = df[df['Modality'] == modality]
                plt.subplot(3, 1, i+1)
                plt.bar(modal_df['Feature Group'], modal_df['Importance'])
                plt.title(f'{modality.upper()} Feature Importance')
                plt.xlabel('Feature Group')
                plt.ylabel('Importance Score')
                plt.xticks(rotation=45)
                plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # 保存CSV文件
        csv_path = os.path.join(self.output_dir, "feature_importance.csv")
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Feature importance visualization saved to {save_path}")
        self.logger.info(f"Feature importance data saved to {csv_path}")


class TensorMultiModalDataset(Dataset):
    """多模态张量数据集"""
    
    def __init__(self, features, labels):
        """
        初始化数据集
        
        参数:
            features: 特征字典，键为模态名称，值为特征张量
            labels: 标签张量
        """
        self.features = features
        self.labels = labels
        
        # 确保所有特征张量长度一致
        length = len(labels)
        required_modalities = ['diffusion', 'qti', 'cest']  # 添加这行
        for modality in required_modalities:  # 修改这行
            if modality not in features:  # 添加这行
                raise ValueError(f"缺少必要的模态特征: {modality}")  # 添加这行
            tensor = features[modality]
            assert len(tensor) == length, f"Feature length mismatch for {modality}: {len(tensor)} != {length}"
            
    def __len__(self):
        """返回数据集大小""" 
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取样本
        
        参数:
            idx: 索引
            
        返回:
            features: 特征字典
            label: 标签
        """
        # 返回特征字典和标签
        features = {modality: tensor[idx] for modality, tensor in self.features.items()}
        label = self.labels[idx]
        
        return features, label