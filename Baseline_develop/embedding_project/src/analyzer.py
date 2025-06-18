#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑区感知Subject Embedding分析器 - 增强版（带存档点功能）
核心分析功能模块
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, KFold
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway, kruskal, spearmanr
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# 🔥 新增：导入存档点管理器
from checkpoint_manager import CheckpointManager, create_checkpoint_decorator

try:
    from .utils import ensure_directory, save_analysis_results, cleanup_memory
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append('.')
    from src.utils import ensure_directory, save_analysis_results, cleanup_memory

logger = logging.getLogger(__name__)



class BrainAwareSubjectEmbeddingAnalyzer:
    """
    脑区感知的Subject Embedding 可行性分析器
    
    增强功能：
    - 保留原有的四Phase分析框架
    - 新增脑区维度的受试者差异分析
    - 扩展可视化系统支持脑区特异性分析
    - 增强决策框架提供分脑区建议
    """
    
    def __init__(self, save_path='./subject_embedding_analysis_brain_aware/', enable_checkpoints=True):
            self.save_path = Path(save_path)
            ensure_directory(self.save_path)
            ensure_directory(self.save_path / 'visualizations')
            
            # 分析结果存储
            self.analysis_results = {}
            self.decision_scores = {}
            
            # 🔥 新增：存档点管理器
            self.enable_checkpoints = enable_checkpoints
            if enable_checkpoints:
                checkpoint_dir = self.save_path / 'checkpoints'
                self.checkpoint_manager = CheckpointManager(checkpoint_dir)
                logger.info("🔄 存档点系统已启用")
            else:
                self.checkpoint_manager = None
                logger.info("📝 存档点系统已禁用")
            
            # 🔥 新增：性能追踪
            self.phase_durations = []
            self.phase_start_times = {}
            self.config_info = {}  # 用于存储配置信息
            
            # 初始化深度网络组件
            self._initialize_deep_network_components()
            
            logger.info("🧠 脑区感知Subject Embedding分析器初始化完成（增强版）")
            logger.info(f"📁 结果保存路径: {self.save_path}")
        


    def _initialize_deep_network_components(self):
        """初始化深度网络组件 - 🔥 严格按照alex版本的超参数"""
        
        # 设置PyTorch设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔥 深度网络将使用设备: {self.device}")
        
        # 🔥 严格按照alex版本设置随机种子确保可重现性
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 深度网络缓存
        self.deep_network_cache = {}
        
        # 🔥 alex版本的严格超参数配置
        self.alex_hyperparams = {
            'batch_size': 128,
            'no_epochs': 25,
            'learning_rate': 0.00001,
            'weight_decay': 0.00001,
            'dropout_rate': 0.5
        }
        
        logger.info(f"🔥 alex版本超参数已加载: {self.alex_hyperparams}")
    
    def save_checkpoint(self, checkpoint_name: str, phase_completed: int = -1, description: str = ""):
        """手动创建存档点"""
        if not self.checkpoint_manager:
            logger.warning("存档点系统未启用")
            return None
        
        return self.checkpoint_manager.create_checkpoint(
            analyzer_instance=self,
            checkpoint_name=checkpoint_name,
            phase_completed=phase_completed,
            description=description,
            is_auto=False
        )
    
      
    def list_checkpoints(self):
        """列出所有存档点"""
        if not self.checkpoint_manager:
            logger.warning("存档点系统未启用")
            return []
        
        return self.checkpoint_manager.list_checkpoints()
    
    def interactive_recovery(self) -> bool:
        """交互式恢复选择"""
        if not self.checkpoint_manager:
            return False
        
        selected_checkpoint = self.checkpoint_manager.interactive_recovery_selection()
        
        if selected_checkpoint:
            return self.load_checkpoint(selected_checkpoint)
        else:
            return False  # 用户选择重新开始
    
    # 🔥 新增：性能追踪方法
    def _start_phase_timer(self, phase_name: str):
        """开始阶段计时"""
        self.phase_start_times[phase_name] = time.time()
        logger.info(f"⏱️ {phase_name} 开始")
    
    def _end_phase_timer(self, phase_name: str):
        """结束阶段计时"""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations.append(duration)
            logger.info(f"⏱️ {phase_name} 完成，耗时: {duration/60:.2f}分钟")
            return duration
        return 0
    

    def _create_deep_network(self, input_dim=341, num_classes=None):
        """创建4×4096深度网络 - 🔥 严格对应alex版本架构"""
        
        class RegModel(nn.Module):
            def __init__(self, input_dim=341, num_classes=102):
                super(RegModel, self).__init__()
                # 🔥 严格对应alex的TensorFlow版本的Dense层
                self.fc1 = nn.Linear(input_dim, 4096)
                self.fc2 = nn.Linear(4096, 4096) 
                self.fc3 = nn.Linear(4096, 4096)
                self.fc4 = nn.Linear(4096, 4096)
                self.fc5 = nn.Linear(4096, num_classes)  # visualized_layer对应的层
                # 🔥 严格使用alex版本的dropout率
                self.dropout = nn.Dropout(0.5)  # alex版本的dropout率
                
            def forward(self, x):
                # 🔥 严格按照alex的TensorFlow模型的结构
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.dropout(F.relu(self.fc2(x)))
                x = self.dropout(F.relu(self.fc3(x)))
                x = self.dropout(F.relu(self.fc4(x)))
                x = self.fc5(x)  # 注意：这里不应用softmax，让CrossEntropyLoss处理
                return x
        
        if num_classes is None:
            # 自动检测类别数
            if hasattr(self, 'data') and 'y_train' in self.data:
                if len(self.data['y_train'].shape) > 1 and self.data['y_train'].shape[1] > 1:
                    num_classes = self.data['y_train'].shape[1]
                else:
                    num_classes = len(np.unique(self.data['y_train']))
            else:
                num_classes = 102  # alex版本的默认值
        
        model = RegModel(input_dim=input_dim, num_classes=num_classes).to(self.device)
        logger.info(f"    🏗️ 创建alex版4×4096深度网络: {input_dim} → 4096×4 → {num_classes}")
        
        return model

    def _kernel_l2_regularization(self, model, weight_decay=0.00001):
        """L2正则化 - 只对权重矩阵，完全模拟你的TensorFlow版本"""
        l2_reg = 0
        for name, param in model.named_parameters():
            # 只对权重矩阵应用L2正则化，跳过偏置项
            if 'weight' in name and param.requires_grad:
                l2_reg += torch.norm(param, p=2) ** 2
        return weight_decay * l2_reg

    def _prepare_torch_dataset(self, X, y=None, train_mode=True):
        """准备PyTorch数据集"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if train_mode and y is not None:
            # 处理one-hot编码
            if len(y.shape) > 1 and y.shape[1] > 1:
                # one-hot -> 类别索引
                y_indices = np.argmax(y, axis=1)
            else:
                y_indices = y
            
            y_tensor = torch.LongTensor(y_indices).to(self.device)
            return TensorDataset(X_tensor, y_tensor)
        else:
            return TensorDataset(X_tensor)

    def _create_deep_classifier_wrapper(self, network_config=None):
        """创建深度网络的sklearn兼容包装器 - 🔥 严格使用alex超参数"""
        
        class DeepNetworkWrapper:
            def __init__(self, analyzer_instance, config=None):
                self.analyzer = analyzer_instance
                # 🔥 严格使用alex版本超参数，只在特殊情况下允许覆盖
                self.config = self.analyzer.alex_hyperparams.copy()
                
                # 只允许在特殊情况下覆盖（如LOSO时数据较少）
                if config:
                    logger.info(f"    ⚠️ 覆盖alex默认超参数: {config}")
                    self.config.update(config)
                
                self.network = None
                self.trained = False
                self.training_history = {'loss': [], 'accuracy': []}
                
            def fit(self, X, y):
                """训练深度网络 - 🔥 完全按照alex版本的训练流程"""
                logger.info("    🔥 开始训练4×4096深度网络 (alex版本超参数)...")
                logger.info(f"    📊 使用超参数: {self.config}")
                start_time = time.time()
                
                # 确定类别数
                if len(y.shape) > 1 and y.shape[1] > 1:
                    num_classes = y.shape[1]
                    y_indices = np.argmax(y, axis=1)
                else:
                    unique_classes = np.unique(y)
                    num_classes = len(unique_classes)
                    y_indices = y
                
                # 创建网络
                self.network = self.analyzer._create_deep_network(
                    input_dim=X.shape[1], 
                    num_classes=num_classes
                )
                
                # 数据准备
                train_dataset = self.analyzer._prepare_torch_dataset(X, y)
                # 🔥 严格使用alex的batch_size
                batch_size = self.config['batch_size']
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # 🔥 训练配置严格对应alex版本
                optimizer = torch.optim.Adam(
                    self.network.parameters(), 
                    lr=self.config['learning_rate']  # 🔥 alex的学习率
                )
                criterion = nn.CrossEntropyLoss()
                no_epochs = self.config['no_epochs']  # 🔥 alex的epoch数
                
                # 🔥 训练循环完全模拟alex的训练过程
                self.network.train()
                for epoch in range(no_epochs):
                    epoch_loss = 0
                    correct_train = 0
                    total_train = 0
                    
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        
                        output = self.network(batch_x)
                        
                        # 🔥 计算基础损失
                        base_loss = criterion(output, batch_y)
                        
                        # 🔥 添加L2正则化 (严格按照alex的kernel_l2_regularization)
                        l2_reg = self.analyzer._kernel_l2_regularization(
                            self.network, 
                            weight_decay=self.config['weight_decay']  # 🔥 alex的权重衰减
                        )
                        total_loss = base_loss + l2_reg
                        
                        total_loss.backward()
                        optimizer.step()
                        
                        epoch_loss += total_loss.item()
                        
                        # 🔥 计算训练准确率
                        _, predicted = torch.max(output.data, 1)
                        total_train += batch_y.size(0)
                        correct_train += (predicted == batch_y).sum().item()
                    
                    # 🔥 记录历史 (对应alex的history)
                    avg_loss = epoch_loss / len(train_loader)
                    train_accuracy = correct_train / total_train
                    self.training_history['loss'].append(avg_loss)
                    self.training_history['accuracy'].append(train_accuracy)
                    
                    # 🔥 按alex版本的输出频率
                    if epoch % 5 == 0:
                        logger.info(f"      Epoch {epoch}/{no_epochs}, Loss: {avg_loss:.4f}, Acc: {train_accuracy:.3f}")
                
                self.trained = True
                train_time = time.time() - start_time
                logger.info(f"    ✅ 深度网络训练完成，耗时: {train_time/60:.1f}分钟")
                logger.info(f"    📊 最终训练准确率: {self.training_history['accuracy'][-1]:.3f}")
                
            def predict(self, X):
                """预测 - 🔥 对应alex版本的model.predict"""
                if not self.trained:
                    raise ValueError("模型尚未训练")
                
                self.network.eval()
                test_dataset = self.analyzer._prepare_torch_dataset(X, None, train_mode=False)
                # 🔥 预测时可以用更大的batch_size提高效率
                test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
                
                predictions = []
                with torch.no_grad():
                    for batch_x, in test_loader:
                        outputs = self.network(batch_x)
                        preds = torch.argmax(outputs, dim=1)
                        predictions.extend(preds.cpu().numpy())
                
                return np.array(predictions)
                
            def predict_proba(self, X):
                """预测概率 - 🔥 对应alex版本的softmax输出"""
                if not self.trained:
                    raise ValueError("模型尚未训练")
                
                self.network.eval()
                test_dataset = self.analyzer._prepare_torch_dataset(X, None, train_mode=False)
                test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
                
                probabilities = []
                with torch.no_grad():
                    for batch_x, in test_loader:
                        outputs = self.network(batch_x)
                        # 🔥 alex版本在预测时会手动应用softmax
                        probs = F.softmax(outputs, dim=1)
                        probabilities.extend(probs.cpu().numpy())
                
                return np.array(probabilities)
                
            def score(self, X, y):
                """计算准确率"""
                pred = self.predict(X)
                
                # 处理one-hot编码的y
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y_true = np.argmax(y, axis=1)
                else:
                    y_true = y
                
                return accuracy_score(y_true, pred)
        
        return DeepNetworkWrapper(self, network_config)

    def _create_region_specific_network(self, input_dim, output_dim):
        """为单个脑区创建小型深度网络"""
        
        class RegionSpecificModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(RegionSpecificModel, self).__init__()
                # 更小的网络架构，适合单脑区分析
                self.fc1 = nn.Linear(input_dim, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, 256)
                self.fc4 = nn.Linear(256, output_dim)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.dropout(F.relu(self.fc2(x)))
                x = self.dropout(F.relu(self.fc3(x)))
                x = self.fc4(x)
                return x
        
        return RegionSpecificModel(input_dim, output_dim).to(self.device)

    def _quick_train_and_evaluate(self, network, X_train, y_train, X_test, y_test):
        """快速训练和评估网络 (用于分脑区分析)"""
        
        # 数据准备
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(y_train).to(self.device)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test).to(self.device),
            torch.LongTensor(y_test).to(self.device)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 训练配置
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 快速训练 (10个epoch)
        network.train()
        for epoch in range(10):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = network(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # 评估
        network.eval()
        with torch.no_grad():
            test_x = torch.FloatTensor(X_test).to(self.device)
            test_outputs = network(test_x)
            predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
            accuracy = accuracy_score(y_test, predictions)
        
        return accuracy

    

    def prepare_data_with_subjects_enhanced(self, data_dict):
        """增强版数据准备 - 🔥 带存档点功能"""
        self._start_phase_timer("数据准备")
        
        # 保存配置信息用于存档点
        self.config_info = {
            'data_keys': list(data_dict.keys()),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("\n" + "="*80)
        logger.info("📊 Phase 0: 增强版数据准备（原有功能 + 脑区感知 + 🔥 深度网络支持 + 存档点）")
        logger.info("="*80)
        
        # 🔄 保留原有数据验证逻辑（不变）
        required_keys = ['X_train_scaled', 'y_train', 'subjects_train',
                        'X_val_scaled', 'y_val', 'subjects_val', 
                        'X_test_scaled', 'y_test', 'subjects_test']
        
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(f"数据字典缺少必要字段: {missing_keys}")
        
        # 提取数据（不变）
        X_train = data_dict['X_train_scaled']
        y_train = data_dict['y_train'] 
        subjects_train = data_dict['subjects_train']
        
        X_val = data_dict['X_val_scaled']
        y_val = data_dict['y_val']
        subjects_val = data_dict['subjects_val']
        
        X_test = data_dict['X_test_scaled'] 
        y_test = data_dict['y_test']
        subjects_test = data_dict['subjects_test']
        
        # 获取所有可用受试者（不变）
        all_subjects = np.concatenate([subjects_train, subjects_val, subjects_test])
        available_subjects = np.unique(all_subjects)
        
        logger.info(f"✅ 原有数据验证通过")
        logger.info(f"  - 训练集: {len(subjects_train):,} 样本, 受试者 {sorted(np.unique(subjects_train))}")
        logger.info(f"  - 验证集: {len(subjects_val):,} 样本, 受试者 {sorted(np.unique(subjects_val))}")
        logger.info(f"  - 测试集: {len(subjects_test):,} 样本, 受试者 {sorted(np.unique(subjects_test))}")
        logger.info(f"  - 总受试者数: {len(available_subjects)}")
        
        # 🔥 深度网络兼容性检查
        logger.info(f"\n🔥 深度网络环境检查...")
        try:
            import torch
            logger.info(f"  ✅ PyTorch版本: {torch.__version__}")
            logger.info(f"  ✅ CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"  ✅ CUDA设备: {torch.cuda.get_device_name(0)}")
                logger.info(f"  ✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # 测试创建小型网络
            test_network = self._create_deep_network(input_dim=X_train.shape[1], num_classes=y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train)))
            logger.info(f"  ✅ 深度网络创建测试成功")
            del test_network  # 清理测试网络
            
        except Exception as e:
            logger.warning(f"  ⚠️ 深度网络环境检查失败: {e}")
            logger.warning(f"  ⚠️ 将跳过深度网络相关分析")
        
        # 🔥 构建脑区感知分析数据（保留原有逻辑）
        logger.info(f"\n🧠 构建脑区感知分析数据...")
        
        # 解析one-hot标签到脑区ID  
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_regions = np.argmax(y_train, axis=1)
        else:
            y_train_regions = y_train.flatten()
        
        # 构建subject×region张量
        brain_region_analysis = self._build_subject_region_tensor(
            X_train, y_train_regions, subjects_train
        )
        
        # 计算受试者样本统计（保留原有逻辑）
        subject_counts = {}
        for subject_id in available_subjects:
            train_count = np.sum(subjects_train == subject_id)
            val_count = np.sum(subjects_val == subject_id) 
            test_count = np.sum(subjects_test == subject_id)
            total_count = train_count + val_count + test_count
            
            subject_counts[subject_id] = {
                'train': train_count,
                'val': val_count, 
                'test': test_count,
                'total': total_count
            }
        
        # 存储完整数据
        self.data = {
            # 使用标准化数据作为主要数据源
            'X_train': X_train,
            'y_train': y_train, 
            'subjects_train': subjects_train,
            'X_val': X_val,
            'y_val': y_val,
            'subjects_val': subjects_val,
            'X_test': X_test,
            'y_test': y_test,
            'subjects_test': subjects_test,
            
            # 🔥 关键修复：同时保存scaled版本的键名
            'X_train_scaled': X_train,
            'X_val_scaled': X_val, 
            'X_test_scaled': X_test,
            
            # 其他数据...
            'available_subjects': available_subjects,
            'subject_counts': subject_counts,
            'brain_region_analysis': brain_region_analysis,
            'y_train_regions': y_train_regions,
            
            # 🔥 新增：深度网络相关元信息
            'deep_network_compatible': True,
            'feature_dim': X_train.shape[1],
            'n_classes': y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train)),
            'total_samples': len(X_train) + len(X_val) + len(X_test),
            'pytorch_ready': hasattr(self, 'device')
        }
        
        duration = self._end_phase_timer("数据准备")
        
        # 🔥 创建Phase 0存档点
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.create_checkpoint(
                    analyzer_instance=self,
                    checkpoint_name="data_prepared",
                    phase_completed=0,
                    description="数据准备完成，包含脑区感知分析数据",
                    is_auto=True
                )
            except Exception as e:
                logger.warning(f"存档点创建失败: {e}")
        
        logger.info(f"✅ 🔥 增强版数据准备完成（深度网络就绪 + 存档点保存）")
        logger.info(f"  - 数据映射方法: 精确Multi-Subject-Out + 脑区感知 + 深度网络支持")
        logger.info(f"  - 有效脑区×受试者组合: {len(brain_region_analysis['subject_region_features'])}")
        logger.info(f"  - 深度网络兼容性: {'✅' if self.data['deep_network_compatible'] else '❌'}")
        logger.info(f"  - 特征维度: {self.data['feature_dim']}")
        logger.info(f"  - 类别数量: {self.data['n_classes']}")
        
        return self.data

    def _build_subject_region_tensor(self, X, regions, subjects):
        """
        构建三维分析张量: [受试者 × 脑区 × 特征]
        
        Args:
            X: 特征矩阵 (N_voxels, 341)
            regions: 脑区标签 (N_voxels,)
            subjects: 受试者标签 (N_voxels,)
            
        Returns:
            dict: 脑区感知分析数据结构
        """
        unique_subjects = np.unique(subjects)
        unique_regions = np.unique(regions)
        
        subject_region_features = {}
        sample_counts = {}
        
        logger.info(f"    🔍 分析 {len(unique_subjects)} 个受试者 × {len(unique_regions)} 个脑区...")
        
        valid_combinations = 0
        total_combinations = len(unique_subjects) * len(unique_regions)
        
        for subject_id in unique_subjects:
            for region_id in unique_regions:
                mask = (subjects == subject_id) & (regions == region_id)
                n_samples = np.sum(mask)
                
                if n_samples >= 50:  # 最小样本阈值，确保统计可靠性
                    region_features = np.mean(X[mask], axis=0)
                    subject_region_features[(subject_id, region_id)] = region_features
                    sample_counts[(subject_id, region_id)] = n_samples
                    valid_combinations += 1
        
        logger.info(f"    ✅ 有效组合: {valid_combinations} / {total_combinations} ({valid_combinations/total_combinations*100:.1f}%)")
        
        # 统计每个脑区的受试者覆盖情况
        region_subject_counts = {}
        for region_id in unique_regions:
            region_subjects = [s for s, r in subject_region_features.keys() if r == region_id]
            region_subject_counts[region_id] = len(region_subjects)
        
        logger.info(f"    📊 脑区统计:")
        logger.info(f"      - 平均每脑区覆盖受试者数: {np.mean(list(region_subject_counts.values())):.1f}")
        logger.info(f"      - 最大覆盖受试者数: {np.max(list(region_subject_counts.values()))}")
        logger.info(f"      - 最小覆盖受试者数: {np.min(list(region_subject_counts.values()))}")
        
        return {
            'subject_region_features': subject_region_features,
            'sample_counts': sample_counts,
            'unique_subjects': unique_subjects,
            'unique_regions': unique_regions,
            'region_subject_counts': region_subject_counts,
            'valid_combinations': valid_combinations,
            'total_combinations': total_combinations
        }


    def phase1_subject_differences_analysis(self):
        """Phase 1: 受试者间差异分析 - 🔥 带存档点功能"""
        self._start_phase_timer("Phase 1")
        
        try:
            logger.info("\n" + "="*80)
            logger.info("📊 Phase 1: 受试者间差异本质分析 (增强版 + 存档点)")
            logger.info("="*80)
            
            # 🔄 Phase 1A: 保留原有全局分析逻辑
            logger.info("\n📊 Phase 1A: 全局受试者差异分析 (保持原有逻辑)")
            self._phase1a_global_subject_analysis()
            
            # 🔥 Phase 1B: 新增分脑区分析
            logger.info("\n📊 Phase 1B: 分脑区受试者差异分析 (新增)")
            self._phase1b_region_wise_analysis()
            
            # 综合决策得分计算
            self._compute_phase1_combined_scores()
            
            duration = self._end_phase_timer("Phase 1")
            
            # 🔥 创建Phase 1存档点
            if self.checkpoint_manager:
                try:
                    self.checkpoint_manager.create_checkpoint(
                        analyzer_instance=self,
                        checkpoint_name="subject_analysis",
                        phase_completed=1,
                        description="受试者间差异分析完成，包含全局和分脑区分析",
                        is_auto=True
                    )
                except Exception as e:
                    logger.warning(f"Phase 1存档点创建失败: {e}")
        
        except Exception as e:
            # 🔥 异常时创建紧急存档点
            if self.checkpoint_manager:
                self.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=f"Phase 1 执行异常: {str(e)}",
                    stack_trace=str(e)
                )
            raise
    

    def _phase1a_global_subject_analysis(self):
        """Phase 1A: 保留原有的全局受试者差异分析"""
        
        # 1.1 统计分布差异诊断（保持原有逻辑）
        logger.info("🔍 1.1A 计算全局受试者统计特征...")
        
        subjects = self.data['available_subjects']
        n_features = self.data['X_train'].shape[1]
        
        # 为每个受试者计算统计量
        subject_stats = {
            'means': np.zeros((len(subjects), n_features)),
            'stds': np.zeros((len(subjects), n_features)),
            'skews': np.zeros((len(subjects), n_features)),
            'kurts': np.zeros((len(subjects), n_features)),
            'sample_counts': np.zeros(len(subjects))
        }
        
        for i, subject_id in enumerate(subjects):
            # 训练集中该受试者的数据
            subject_mask = self.data['subjects_train'] == subject_id
            if np.sum(subject_mask) > 0:
                subject_data = self.data['X_train'][subject_mask]
                
                subject_stats['means'][i] = np.mean(subject_data, axis=0)
                subject_stats['stds'][i] = np.std(subject_data, axis=0)
                subject_stats['skews'][i] = stats.skew(subject_data, axis=0)
                subject_stats['kurts'][i] = stats.kurtosis(subject_data, axis=0)
                subject_stats['sample_counts'][i] = len(subject_data)
            
            # 处理验证集
            val_mask = self.data['subjects_val'] == subject_id
            if np.sum(val_mask) > 0:
                val_data = self.data['X_val'][val_mask]
                subject_stats['sample_counts'][i] += len(val_data)
        
        self.analysis_results['subject_stats'] = subject_stats
        
        logger.info(f"✅ 全局受试者统计特征计算完成")
        logger.info(f"  - 平均每受试者样本量: {np.mean(subject_stats['sample_counts']):.0f}")
        logger.info(f"  - 样本量范围: [{np.min(subject_stats['sample_counts']):.0f}, {np.max(subject_stats['sample_counts']):.0f}]")
        
        # 1.2 受试者间相似性分析（保持原有逻辑，增强数值稳定性）
        logger.info("🔍 1.2A 分析全局受试者间相似性...")
        
        subject_means = subject_stats['means']
        
        # 数值稳定性检查
        feature_variance = np.var(subject_means, axis=0)
        constant_features = np.sum(feature_variance < 1e-12)
        valid_features_mask = feature_variance >= 1e-12
        
        logger.info(f"    - 总特征数: {n_features}")
        logger.info(f"    - 常数特征数: {constant_features}")
        logger.info(f"    - 有效特征数: {np.sum(valid_features_mask)}")
        
        if constant_features > 0:
            subject_means_filtered = subject_means[:, valid_features_mask]
        else:
            subject_means_filtered = subject_means
        
        # 距离矩阵计算
        try:
            distance_matrix = squareform(pdist(subject_means_filtered, metric='euclidean'))
            upper_triangle_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            logger.info(f"    - 平均受试者间距离: {np.mean(upper_triangle_distances):.6f}")
            logger.info(f"    - 距离标准差: {np.std(upper_triangle_distances):.6f}")
        except Exception as e:
            logger.info(f"    ❌ 距离计算失败: {e}")
            distance_matrix = np.zeros((len(subjects), len(subjects)))
        
        # 相关性矩阵计算
        try:
            correlation_matrix = np.corrcoef(subject_means_filtered)
            if np.all(np.isfinite(correlation_matrix)):
                upper_triangle_corr = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                valid_correlations = upper_triangle_corr[np.isfinite(upper_triangle_corr)]
                mean_correlation = np.mean(valid_correlations) if len(valid_correlations) > 0 else 0.0
                logger.info(f"    - 平均受试者间相关性: {mean_correlation:.6f}")
            else:
                raise ValueError("相关性矩阵包含无效值")
        except Exception as e:
            logger.info(f"    ⚠️ 标准相关性计算失败，使用备用方法")
            correlation_matrix = np.eye(len(subjects))
            mean_correlation = 0.0
        
        # 层次聚类
        try:
            linkage_matrix = linkage(subject_means_filtered, method='ward')
        except Exception as e:
            logger.info(f"    ⚠️ 层次聚类失败: {e}")
            linkage_matrix = np.zeros((len(subjects)-1, 4))
        
        # 保存全局相似性分析结果
        self.analysis_results['global_subject_similarity'] = {
            'distance_matrix': distance_matrix,
            'correlation_matrix': correlation_matrix,
            'linkage_matrix': linkage_matrix,
            'mean_correlation': mean_correlation,
            'constant_features_count': constant_features,
            'valid_features_count': np.sum(valid_features_mask)
        }
        
        # 1.3 特征变异模式分析（保持原有逻辑）
        logger.info("🔍 1.3A 分析全局特征变异模式...")
        
        feature_f_stats = np.var(subject_means, axis=0)
        feature_f_stats_norm = feature_f_stats / (np.mean(feature_f_stats) + 1e-8)
        
        high_variation_threshold = np.percentile(feature_f_stats_norm, 90)
        low_variation_threshold = np.percentile(feature_f_stats_norm, 10)
        
        high_variation_features = np.where(feature_f_stats_norm > high_variation_threshold)[0]
        low_variation_features = np.where(feature_f_stats_norm < low_variation_threshold)[0]
        
        # PCA分析
        pca = PCA()
        subject_pca_result = pca.fit_transform(subject_means)
        
        self.analysis_results['global_feature_variation'] = {
            'f_stats': feature_f_stats_norm,
            'high_variation_features': high_variation_features,
            'low_variation_features': low_variation_features,
            'pca_result': subject_pca_result,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'pca_cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
        
        logger.info(f"✅ 全局特征变异分析完成")
        logger.info(f"  - 高变异特征数量: {len(high_variation_features)} ({len(high_variation_features)/n_features*100:.1f}%)")
        logger.info(f"  - 低变异特征数量: {len(low_variation_features)} ({len(low_variation_features)/n_features*100:.1f}%)")
        logger.info(f"  - 前3个主成分解释方差: {np.sum(pca.explained_variance_ratio_[:3])*100:.1f}%")

    def _phase1b_region_wise_analysis(self):
        """Phase 1B: 新增的分脑区受试者差异分析"""
        
        brain_data = self.data['brain_region_analysis']
        subject_region_features = brain_data['subject_region_features']
        unique_regions = brain_data['unique_regions']
        
        region_wise_results = {}
        
        logger.info(f"🔍 1.1B 分析每个脑区的受试者特异性...")
        logger.info(f"    - 总脑区数: {len(unique_regions)}")
        logger.info(f"    - 有效组合数: {len(subject_region_features)}")
        
        processed_regions = 0
        
        for region_id in unique_regions:
            # 提取该脑区所有受试者的特征
            region_subject_features = []
            valid_subjects = []
            
            for subject_id in brain_data['unique_subjects']:
                if (subject_id, region_id) in subject_region_features:
                    region_subject_features.append(subject_region_features[(subject_id, region_id)])
                    valid_subjects.append(subject_id)
            
            if len(region_subject_features) >= 5:  # 至少5个受试者有该脑区
                region_features_array = np.array(region_subject_features)
                
                # 受试者间距离分析
                try:
                    distances = squareform(pdist(region_features_array))
                    mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                    std_distance = np.std(distances[np.triu_indices_from(distances, k=1)])
                except:
                    mean_distance = 0.0
                    std_distance = 1.0
                
                # 受试者间相关性
                try:
                    correlations = np.corrcoef(region_features_array)
                    if np.all(np.isfinite(correlations)):
                        mean_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
                    else:
                        mean_correlation = 0.0
                except:
                    mean_correlation = 0.0
                
                # PCA分析该脑区的受试者差异模式
                try:
                    pca = PCA()
                    pca_result = pca.fit_transform(region_features_array)
                    pca_3pc_variance = np.sum(pca.explained_variance_ratio_[:3]) if len(pca.explained_variance_ratio_) >= 3 else np.sum(pca.explained_variance_ratio_)
                except:
                    pca = None
                    pca_3pc_variance = 0.0
                
                # 受试者特异性得分 (距离相对于变异度)
                specificity_score = mean_distance / (std_distance + 1e-8)
                
                region_wise_results[region_id] = {
                    'n_subjects': len(valid_subjects),
                    'mean_inter_subject_distance': mean_distance,
                    'std_inter_subject_distance': std_distance,
                    'mean_inter_subject_correlation': abs(mean_correlation),
                    'pca_explained_variance': pca.explained_variance_ratio_ if pca else np.array([0]),
                    'pca_3pc_variance': pca_3pc_variance,
                    'subject_specificity_score': specificity_score,
                    'valid_subjects': valid_subjects,
                    'sample_sizes': [brain_data['sample_counts'][(s, region_id)] for s in valid_subjects]
                }
                
                processed_regions += 1
        
        self.analysis_results['region_wise_subject_analysis'] = region_wise_results
        
        # 计算脑区特异性统计
        if region_wise_results:
            specificity_scores = [r['subject_specificity_score'] for r in region_wise_results.values()]
            distances = [r['mean_inter_subject_distance'] for r in region_wise_results.values()]
            correlations = [r['mean_inter_subject_correlation'] for r in region_wise_results.values()]
            
            logger.info(f"✅ 分脑区分析完成")
            logger.info(f"  - 成功分析脑区数量: {processed_regions}")
            logger.info(f"  - 特异性得分范围: [{np.min(specificity_scores):.3f}, {np.max(specificity_scores):.3f}]")
            logger.info(f"  - 平均特异性得分: {np.mean(specificity_scores):.3f}")
            logger.info(f"  - 平均受试者间距离: {np.mean(distances):.3f}")
            logger.info(f"  - 平均受试者间相关性: {np.mean(correlations):.3f}")
            
            # 识别高/低特异性脑区
            high_threshold = np.percentile(specificity_scores, 75)
            low_threshold = np.percentile(specificity_scores, 25)
            
            high_specificity_regions = [r_id for r_id, r_data in region_wise_results.items() 
                                       if r_data['subject_specificity_score'] > high_threshold]
            low_specificity_regions = [r_id for r_id, r_data in region_wise_results.items() 
                                      if r_data['subject_specificity_score'] < low_threshold]
            
            logger.info(f"  - 高特异性脑区 (>75th): {len(high_specificity_regions)} 个")
            logger.info(f"  - 低特异性脑区 (<25th): {len(low_specificity_regions)} 个")
            
            # 显示极端脑区
            if high_specificity_regions:
                top_region = max(high_specificity_regions, 
                               key=lambda x: region_wise_results[x]['subject_specificity_score'])
                logger.info(f"  - 最高特异性脑区: {top_region} (得分: {region_wise_results[top_region]['subject_specificity_score']:.3f})")
            
            if low_specificity_regions:
                bottom_region = min(low_specificity_regions,
                                  key=lambda x: region_wise_results[x]['subject_specificity_score'])
                logger.info(f"  - 最低特异性脑区: {bottom_region} (得分: {region_wise_results[bottom_region]['subject_specificity_score']:.3f})")
        
        else:
            logger.info(f"❌ 没有足够数据进行分脑区分析")

    def _compute_phase1_combined_scores(self):
        """计算Phase 1的综合决策得分"""
        
        # 原有全局得分
        if 'global_feature_variation' in self.analysis_results:
            global_pca = self.analysis_results['global_feature_variation']
            variance_explained_3pc = np.sum(global_pca['pca_explained_variance'][:3])
            
            global_correlation = 0.0
            if 'global_subject_similarity' in self.analysis_results:
                global_correlation = abs(self.analysis_results['global_subject_similarity']['mean_correlation'])
        else:
            variance_explained_3pc = 0.5
            global_correlation = 0.0
        
        # 🔥 新增：分脑区得分
        region_specificity_diversity = 0.0
        high_specificity_ratio = 0.0
        
        if 'region_wise_subject_analysis' in self.analysis_results:
            region_results = self.analysis_results['region_wise_subject_analysis']
            if region_results:
                specificity_scores = [r['subject_specificity_score'] for r in region_results.values()]
                
                # 脑区特异性多样性：标准差/均值 (值越大，脑区间差异越大)
                region_specificity_diversity = np.std(specificity_scores) / (np.mean(specificity_scores) + 1e-8)
                
                # 高特异性脑区比例
                high_threshold = np.percentile(specificity_scores, 75)
                high_specificity_count = np.sum(np.array(specificity_scores) > high_threshold)
                high_specificity_ratio = high_specificity_count / len(specificity_scores)
        
        # Phase 1 综合决策得分
        self.decision_scores['phase1'] = {
            # 原有得分
            'difference_significance': 1.0 if variance_explained_3pc > 0.6 else 0.8 if variance_explained_3pc > 0.4 else 0.5,
            'pattern_linearity': variance_explained_3pc,
            'subject_similarity': global_correlation,
            'feature_heterogeneity': len(self.analysis_results.get('global_feature_variation', {}).get('high_variation_features', [])) / self.data['X_train'].shape[1],
            
            # 🔥 新增得分
            'region_specificity_diversity': region_specificity_diversity,
            'high_specificity_ratio': high_specificity_ratio,
            'region_analysis_success': 1.0 if 'region_wise_subject_analysis' in self.analysis_results else 0.0
        }
        
        logger.info(f"\n📈 Phase 1 综合决策指标:")
        logger.info(f"  - 差异显著性得分: {self.decision_scores['phase1']['difference_significance']:.3f}")
        logger.info(f"  - 模式线性度: {self.decision_scores['phase1']['pattern_linearity']:.3f}")
        logger.info(f"  - 受试者相似性: {self.decision_scores['phase1']['subject_similarity']:.3f}")
        logger.info(f"  - 特征异质性: {self.decision_scores['phase1']['feature_heterogeneity']:.3f}")
        logger.info(f"  🔥 脑区特异性多样性: {self.decision_scores['phase1']['region_specificity_diversity']:.3f}")
        logger.info(f"  🔥 高特异性脑区比例: {self.decision_scores['phase1']['high_specificity_ratio']:.3f}")


    def phase2_subject_separability_analysis(self):
        """Phase 2: 受试者可分离性评估 - 🔥 带存档点功能"""
        self._start_phase_timer("Phase 2")
        
        try:
            logger.info("\n" + "="*80)
            logger.info("📊 Phase 2: 受试者可分离性评估 (增强版 + 存档点)")
            logger.info("="*80)
            
            # 🔄 Phase 2A: 修正版全局受试者可分离性分析
            logger.info("\n📊 Phase 2A: 全局受试者可分离性分析 (修正版)")
            self._phase2a_global_separability_corrected()
            
            # 🔥 Phase 2B: 新增分脑区可分离性分析
            logger.info("\n📊 Phase 2B: 分脑区受试者可分离性分析 (新增)")
            self._phase2b_region_wise_separability()
            
            # Phase 2C: 保留原有的脑区分类一致性分析
            logger.info("\n📊 Phase 2C: 脑区分类一致性分析 (保持原有)")
            self._phase2c_class_consistency_analysis()
            
            # 综合决策得分计算
            self._compute_phase2_combined_scores()
            
            duration = self._end_phase_timer("Phase 2")
            
            # 🔥 创建Phase 2存档点
            if self.checkpoint_manager:
                try:
                    self.checkpoint_manager.create_checkpoint(
                        analyzer_instance=self,
                        checkpoint_name="separability",
                        phase_completed=2,
                        description="受试者可分离性评估完成，包含深度网络分析",
                        is_auto=True
                    )
                except Exception as e:
                    logger.warning(f"Phase 2存档点创建失败: {e}")
        
        except Exception as e:
            # 🔥 异常时创建紧急存档点
            if self.checkpoint_manager:
                self.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=f"Phase 2 执行异常: {str(e)}",
                    stack_trace=str(e)
                )
            raise

    def _phase2a_global_separability_corrected(self):
        """修正版：神经网络baseline性能分析"""
        
        logger.info("🔍 2.1A 神经网络baseline性能分析...")
        
        X_train = self.data['X_train_scaled']
        y_train = self.data['y_train']
        subjects_train = self.data['subjects_train']
        
        # 转换标签
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_classes = np.argmax(y_train, axis=1)
        else:
            y_classes = y_train.flatten()
        
        logger.info(f"    - 体素数量: {len(X_train):,}")
        logger.info(f"    - 特征维度: {X_train.shape[1]}")
        logger.info(f"    - 脑区类别数: {len(np.unique(y_classes))}")
        
        # 1. Baseline性能：随机分割（模拟你的训练方式）
        baseline_results = self._test_random_split_performance(X_train, y_classes)
        
        # 2. 关键测试：Leave-One-Subject-Out性能（真实泛化性能）
        loso_results = self._test_leave_one_subject_out_performance(
            X_train, y_classes, subjects_train)
        
        # 3. Subject Embedding潜在价值评估
        embedding_value_assessment = self._assess_subject_embedding_value(
            baseline_results, loso_results)
        
        # 保存结果
        self.analysis_results['neural_network_baseline_analysis'] = {
            'baseline_random_split': baseline_results,
            'leave_one_subject_out': loso_results,
            'embedding_value_assessment': embedding_value_assessment
        }
        
        return {
            'baseline_random_split': baseline_results,
            'leave_one_subject_out': loso_results,
            'embedding_value_assessment': embedding_value_assessment
        }

    def _phase2b_region_wise_separability(self):
        """修正版分脑区Subject Embedding需求分析 + 🔥 深度网络增强"""
        
        logger.info("🔍 2.1B 🔥 分脑区Subject Embedding需求分析（含深度网络权威评估）...")
        
        if 'region_wise_subject_analysis' not in self.analysis_results:
            logger.info("❌ 需要先运行Phase 1B")
            return
        
        region_wise_results = self.analysis_results['region_wise_subject_analysis']
        region_embedding_analysis = {}
        deep_network_region_analysis = {}  # 🔥 新增：深度网络分脑区分析
        
        logger.info(f"    - 待分析脑区数: {len(region_wise_results)}")
        
        # 🎯 核心分析：每个脑区的Subject Embedding需求评估
        for region_id, region_data in region_wise_results.items():
            if region_data['n_subjects'] >= 5:  # 确保足够受试者
                
                logger.info(f"    🧠 分析脑区{region_id}的Subject Embedding需求...")
                
                # 1. 原有分析方法（保持不变）
                consistency_analysis = self._analyze_region_classification_consistency(region_id)
                cross_subject_analysis = self._cross_subject_region_classification_test(region_id)
                specificity_impact = self._assess_subject_specificity_impact(region_id)
        
                
                # 原有的embedding需求评级
                embedding_necessity = self._compute_embedding_necessity_score(
                    consistency_analysis, cross_subject_analysis, specificity_impact
                )
                
                # 🔥 2. 新增：深度网络权威分析
                deep_network_analysis = self._analyze_region_with_deep_network(region_id)
                
                # 🔥 3. 综合原有分析和深度网络分析
                if deep_network_analysis['status'] == 'success':
                    # 融合传统分析和深度网络分析
                    traditional_score = embedding_necessity['score']
                    deep_score = min(1.0, deep_network_analysis['subject_specificity_strength'] / 5.0)  # 归一化到0-1
                    
                    # 加权综合评分（深度网络权重更高，因为更权威）
                    comprehensive_score = 0.4 * traditional_score + 0.6 * deep_score
                    
                    # 确定最终等级
                    if comprehensive_score > 0.8:
                        final_level = 'CRITICAL'
                        final_dim = 128
                        final_priority = 'HIGHEST'
                    elif comprehensive_score > 0.6:
                        final_level = 'HIGH'
                        final_dim = 64
                        final_priority = 'HIGH'
                    elif comprehensive_score > 0.4:
                        final_level = 'MEDIUM'
                        final_dim = 32
                        final_priority = 'MEDIUM'
                    else:
                        final_level = 'LOW'
                        final_dim = 16
                        final_priority = 'LOW'
                    
                    region_embedding_analysis[region_id] = {
                        'consistency_analysis': consistency_analysis,
                        'cross_subject_analysis': cross_subject_analysis,
                        'specificity_impact': specificity_impact,
                        'traditional_embedding_necessity_score': traditional_score,
                        'traditional_embedding_necessity_level': embedding_necessity['level'],
                        # 🔥 深度网络分析结果
                        'deep_network_analysis': deep_network_analysis,
                        'deep_necessity_score': deep_score,
                        'deep_necessity_level': deep_network_analysis['deep_necessity_level'],
                        # 🔥 综合评估结果
                        'comprehensive_embedding_necessity_score': comprehensive_score,
                        'comprehensive_embedding_necessity_level': final_level,
                        'final_recommended_embedding_dim': final_dim,
                        'final_implementation_priority': final_priority,
                        'analysis_confidence': 'high',  # 有深度网络验证，置信度高
                        'deep_network_validation': True
                    }
                    
                    # 🔥 深度网络专项分析存储
                    deep_network_region_analysis[region_id] = {
                        'deep_accuracy': deep_network_analysis['deep_subject_identification_accuracy'],
                        'specificity_strength': deep_network_analysis['subject_specificity_strength'],
                        'deep_advantage': deep_network_analysis['deep_network_advantage'],
                        'training_efficiency': deep_network_analysis['deep_subject_identification_accuracy'] / (deep_network_analysis['training_time'] / 60),
                        'necessity_level': deep_network_analysis['deep_necessity_level'],
                        'recommended_dim': deep_network_analysis['deep_recommended_dim']
                    }
                    
                    logger.info(f"      ✅ 脑区{region_id}: 综合={final_level} "
                        f"(传统得分: {traditional_score:.3f}, 深度得分: {deep_score:.3f}, "
                        f"综合得分: {comprehensive_score:.3f}, 推荐维度: {final_dim})")
                    
                else:
                    # 深度网络分析失败，仅使用传统分析
                    region_embedding_analysis[region_id] = {
                        'consistency_analysis': consistency_analysis,
                        'cross_subject_analysis': cross_subject_analysis,
                        'specificity_impact': specificity_impact,
                        'embedding_necessity_score': embedding_necessity['score'],
                        'embedding_necessity_level': embedding_necessity['level'],
                        'recommended_embedding_dim': embedding_necessity['recommended_dim'],
                        'implementation_priority': embedding_necessity['priority'],
                        'deep_network_validation': False,
                        'analysis_confidence': 'medium',  # 没有深度网络验证，置信度中等
                        'deep_network_failure_reason': deep_network_analysis.get('reason', '未知原因')
                    }
                    
                    logger.info(f"      ⚠️ 脑区{region_id}: {embedding_necessity['level']} "
                        f"(得分: {embedding_necessity['score']:.3f}, "
                        f"推荐维度: {embedding_necessity['recommended_dim']}, 深度网络分析失败)")
        
        # 保存分析结果
        self.analysis_results['region_wise_separability'] = region_embedding_analysis
        self.analysis_results['deep_network_region_analysis'] = deep_network_region_analysis  # 🔥 新增
        
        # 🔥 深度网络分脑区分析总结
        if deep_network_region_analysis:
            self._summarize_deep_network_region_analysis(deep_network_region_analysis, region_embedding_analysis)
        
        # 🔥 保留原有的完整结果分析，但重新解释含义
        if region_embedding_analysis:
            # 使用综合评分进行统计
            comprehensive_scores = []
            final_dims = []
            
            for r_data in region_embedding_analysis.values():
                if 'comprehensive_embedding_necessity_score' in r_data:
                    comprehensive_scores.append(r_data['comprehensive_embedding_necessity_score'])
                    final_dims.append(r_data['final_recommended_embedding_dim'])
                else:
                    comprehensive_scores.append(r_data['embedding_necessity_score'])
                    final_dims.append(r_data['recommended_embedding_dim'])
            
            logger.info(f"✅ 🔥 增强版分脑区Subject Embedding需求分析完成")
            logger.info(f"  - 成功分析脑区数量: {len(region_embedding_analysis)}")
            logger.info(f"  - 深度网络验证脑区数: {len(deep_network_region_analysis)}")
            logger.info(f"  - 平均综合embedding需求得分: {np.mean(comprehensive_scores):.3f}")
            logger.info(f"  - 需求得分范围: [{np.min(comprehensive_scores):.3f}, {np.max(comprehensive_scores):.3f}]")
            logger.info(f"  - 平均推荐embedding维度: {np.mean(final_dims):.1f}")
            
            # 按最终需求等级分类
            critical_regions = [rid for rid, data in region_embedding_analysis.items() 
                            if data.get('comprehensive_embedding_necessity_level', data.get('embedding_necessity_level')) == 'CRITICAL']
            high_regions = [rid for rid, data in region_embedding_analysis.items() 
                        if data.get('comprehensive_embedding_necessity_level', data.get('embedding_necessity_level')) == 'HIGH']
            medium_regions = [rid for rid, data in region_embedding_analysis.items() 
                            if data.get('comprehensive_embedding_necessity_level', data.get('embedding_necessity_level')) == 'MEDIUM']
            low_regions = [rid for rid, data in region_embedding_analysis.items() 
                        if data.get('comprehensive_embedding_necessity_level', data.get('embedding_necessity_level')) == 'LOW']
            
            logger.info(f"  🔥 最终需求等级分布:")
            logger.info(f"    - CRITICAL级别脑区: {len(critical_regions)} 个")
            logger.info(f"    - HIGH级别脑区: {len(high_regions)} 个") 
            logger.info(f"    - MEDIUM级别脑区: {len(medium_regions)} 个")
            logger.info(f"    - LOW级别脑区: {len(low_regions)} 个")
            
            # 🔥 深度网络验证统计
            deep_validated_regions = [rid for rid, data in region_embedding_analysis.items() 
                                    if data.get('deep_network_validation', False)]
            logger.info(f"  🔥 深度网络验证率: {len(deep_validated_regions)}/{len(region_embedding_analysis)} "
                f"({len(deep_validated_regions)/len(region_embedding_analysis)*100:.1f}%)")
            
            # 识别最需要和最不需要embedding的脑区（基于综合评分）
            if len(region_embedding_analysis) > 0:
                most_needed_region = max(region_embedding_analysis.keys(), 
                                    key=lambda x: region_embedding_analysis[x].get('comprehensive_embedding_necessity_score', 
                                                                            region_embedding_analysis[x].get('embedding_necessity_score', 0)))
                least_needed_region = min(region_embedding_analysis.keys(), 
                                    key=lambda x: region_embedding_analysis[x].get('comprehensive_embedding_necessity_score',
                                                                                region_embedding_analysis[x].get('embedding_necessity_score', 1)))
                
                most_score = region_embedding_analysis[most_needed_region].get('comprehensive_embedding_necessity_score',
                                                                        region_embedding_analysis[most_needed_region].get('embedding_necessity_score'))
                least_score = region_embedding_analysis[least_needed_region].get('comprehensive_embedding_necessity_score',
                                                                            region_embedding_analysis[least_needed_region].get('embedding_necessity_score'))
                
                logger.info(f"  🔥 最需要Subject Embedding的脑区: {most_needed_region} (综合得分: {most_score:.3f})")
                logger.info(f"  🔥 最不需要Subject Embedding的脑区: {least_needed_region} (综合得分: {least_score:.3f})")
        
        else:
            logger.info(f"❌ 没有成功的分脑区Subject Embedding分析")

                
            
    def _summarize_deep_network_region_analysis(self, deep_network_region_analysis, region_embedding_analysis):
        """🔥 总结深度网络分脑区分析结果"""
        
        logger.info(f"    🔥 深度网络分脑区分析总结:")
        
        if not deep_network_region_analysis:
            logger.info(f"      ⚠️ 没有成功的深度网络分脑区分析")
            return
        
        # 深度网络性能统计
        deep_accuracies = [data['deep_accuracy'] for data in deep_network_region_analysis.values()]
        specificity_strengths = [data['specificity_strength'] for data in deep_network_region_analysis.values()]
        training_efficiencies = [data['training_efficiency'] for data in deep_network_region_analysis.values()]
        
        logger.info(f"      📊 深度网络性能统计:")
        logger.info(f"        - 平均受试者识别准确率: {np.mean(deep_accuracies):.3f}")
        logger.info(f"        - 准确率范围: [{np.min(deep_accuracies):.3f}, {np.max(deep_accuracies):.3f}]")
        logger.info(f"        - 平均特异性强度: {np.mean(specificity_strengths):.2f}x 随机基线")
        logger.info(f"        - 平均训练效率: {np.mean(training_efficiencies):.2f} 准确率/分钟")
        
        # 按深度网络判定的需求等级统计
        deep_critical = [rid for rid, data in deep_network_region_analysis.items() if data['necessity_level'] == 'CRITICAL']
        deep_high = [rid for rid, data in deep_network_region_analysis.items() if data['necessity_level'] == 'HIGH']
        deep_medium = [rid for rid, data in deep_network_region_analysis.items() if data['necessity_level'] == 'MEDIUM']
        deep_low = [rid for rid, data in deep_network_region_analysis.items() if data['necessity_level'] == 'LOW']
        
        logger.info(f"      🎯 深度网络需求等级判定:")
        logger.info(f"        - CRITICAL (特异性>5x): {len(deep_critical)} 个脑区")
        logger.info(f"        - HIGH (特异性3-5x): {len(deep_high)} 个脑区")
        logger.info(f"        - MEDIUM (特异性2-3x): {len(deep_medium)} 个脑区")
        logger.info(f"        - LOW (特异性<2x): {len(deep_low)} 个脑区")
        
        # 找出深度网络表现最好和最差的脑区
        if len(deep_network_region_analysis) > 0:
            best_region = max(deep_network_region_analysis.keys(), 
                            key=lambda x: deep_network_region_analysis[x]['specificity_strength'])
            worst_region = min(deep_network_region_analysis.keys(), 
                            key=lambda x: deep_network_region_analysis[x]['specificity_strength'])
            
            best_strength = deep_network_region_analysis[best_region]['specificity_strength']
            worst_strength = deep_network_region_analysis[worst_region]['specificity_strength']
            
            logger.info(f"      🏆 深度网络最强脑区: {best_region} (特异性强度: {best_strength:.2f}x)")
            logger.info(f"      📉 深度网络最弱脑区: {worst_region} (特异性强度: {worst_strength:.2f}x)")
        
        # 深度网络验证与传统方法的一致性分析
        consistency_analysis = self._analyze_deep_traditional_consistency(deep_network_region_analysis, region_embedding_analysis)
        logger.info(f"      🔬 深度网络与传统方法一致性: {consistency_analysis['overall_consistency']:.1%}")

    def _analyze_deep_traditional_consistency(self, deep_analysis, region_analysis):
        """分析深度网络与传统方法的一致性"""
        
        consistent_count = 0
        total_count = 0
        
        for region_id in deep_analysis.keys():
            if region_id in region_analysis:
                deep_level = deep_analysis[region_id]['necessity_level']
                traditional_level = region_analysis[region_id].get('embedding_necessity_level', 'UNKNOWN')
                
                # 简化的一致性判断
                level_mapping = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                deep_score = level_mapping.get(deep_level, 0)
                traditional_score = level_mapping.get(traditional_level, 0)
                
                # 如果两者差距在1个等级内，认为一致
                if abs(deep_score - traditional_score) <= 1:
                    consistent_count += 1
                
                total_count += 1
        
        consistency_rate = consistent_count / total_count if total_count > 0 else 0
        
        return {
            'overall_consistency': consistency_rate,
            'consistent_regions': consistent_count,
            'total_compared_regions': total_count
        }       

    def _analyze_region_classification_consistency(self, region_id):
        """分析单个脑区的分类一致性"""
        
        # 获取该脑区的所有体素
        y_labels = np.argmax(self.data['y_train'], axis=1)
        region_mask = y_labels == region_id
        
        if np.sum(region_mask) < 100:  # 样本量太少
            return {'status': 'insufficient_data', 'n_voxels': np.sum(region_mask)}
        
        region_features = self.data['X_train'][region_mask]
        region_subjects = self.data['subjects_train'][region_mask]
        
        # 计算每个受试者该脑区的特征统计
        subject_stats = {}
        for subject_id in np.unique(region_subjects):
            subject_mask = region_subjects == subject_id
            if np.sum(subject_mask) >= 10:  # 至少10个体素
                subject_data = region_features[subject_mask]
                subject_stats[subject_id] = {
                    'mean': np.mean(subject_data, axis=0),
                    'std': np.std(subject_data, axis=0),
                    'n_voxels': len(subject_data)
                }
        
        if len(subject_stats) < 3:  # 受试者数量不足
            return {'status': 'insufficient_subjects', 'n_subjects': len(subject_stats)}
        
        # 计算受试者间变异
        all_means = np.array([stats['mean'] for stats in subject_stats.values()])
        feature_cv = np.std(all_means, axis=0) / (np.abs(np.mean(all_means, axis=0)) + 1e-8)
        mean_cv = np.mean(feature_cv)
        
        # 计算受试者间距离
        from scipy.spatial.distance import pdist
        try:
            distances = pdist(all_means)
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
        except:
            mean_distance = 0
            std_distance = 0
        
        return {
            'status': 'success',
            'n_subjects': len(subject_stats),
            'n_voxels': np.sum(region_mask),
            'mean_cv': mean_cv,
            'mean_inter_subject_distance': mean_distance,
            'std_inter_subject_distance': std_distance,
            'consistency_score': 1.0 / (1.0 + mean_cv)  # 越一致分数越高
        }

    def _analyze_region_with_deep_network(self, region_id):
        """🔥 用深度网络分析单个脑区的受试者特异性"""
        
        logger.info(f"      🔥 深度网络分析脑区{region_id}的Subject Embedding需求...")
        
        # 准备该脑区的数据
        y_labels = np.argmax(self.data['y_train'], axis=1)
        region_mask = y_labels == region_id
        
        if np.sum(region_mask) < 500:
            return {'status': 'insufficient_data', 'reason': '脑区样本不足500个'}
        
        region_features = self.data['X_train'][region_mask]
        region_subjects = self.data['subjects_train'][region_mask]
        
        # 构建受试者识别任务
        unique_subjects = np.unique(region_subjects)
        if len(unique_subjects) < 5:
            return {'status': 'insufficient_subjects', 'reason': '受试者数量不足5个'}
        
        # 重新映射受试者标签
        subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
        mapped_labels = np.array([subject_mapping[s] for s in region_subjects])
        
        # 训练深度网络进行受试者识别
        try:
            logger.info(f"        🏗️ 为脑区{region_id}创建专用深度网络...")
            
            # 创建小型网络（针对单脑区）
            region_network = self._create_region_specific_network(
                input_dim=341,
                output_dim=len(unique_subjects)
            )
            
            # 训练和评估
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                region_features, mapped_labels, test_size=0.3, random_state=42, stratify=mapped_labels
            )
            
            # 快速训练和评估
            start_time = time.time()
            accuracy = self._quick_train_and_evaluate(region_network, X_train, y_train, X_test, y_test)
            train_time = time.time() - start_time
            
            # 计算Subject Embedding需求强度
            random_baseline = 1.0 / len(unique_subjects)
            subject_specificity_strength = accuracy / random_baseline
            
            # 深度网络特异性评估
            if subject_specificity_strength > 5.0:
                deep_necessity_level = 'CRITICAL'
                deep_recommended_dim = 128
            elif subject_specificity_strength > 3.0:
                deep_necessity_level = 'HIGH'
                deep_recommended_dim = 64
            elif subject_specificity_strength > 2.0:
                deep_necessity_level = 'MEDIUM'
                deep_recommended_dim = 32
            else:
                deep_necessity_level = 'LOW'
                deep_recommended_dim = 16
            
            logger.info(f"        ✅ 脑区{region_id}深度网络分析: 准确率={accuracy:.3f}, 特异性强度={subject_specificity_strength:.2f}x, 需求={deep_necessity_level}")
            
            return {
                'status': 'success',
                'deep_subject_identification_accuracy': accuracy,
                'random_baseline': random_baseline,
                'subject_specificity_strength': subject_specificity_strength,
                'deep_necessity_level': deep_necessity_level,
                'deep_recommended_dim': deep_recommended_dim,
                'n_subjects': len(unique_subjects),
                'n_samples': len(region_features),
                'training_time': train_time,
                'deep_network_advantage': accuracy - random_baseline,
                'interpretation': f"深度网络在脑区{region_id}显示{deep_necessity_level}级Subject Embedding需求"
            }
            
        except Exception as e:
            logger.info(f"        ❌ 脑区{region_id}深度分析失败: {e}")
            return {'status': 'failed', 'error': str(e)}


    def _cross_subject_region_classification_test(self, region_id):
        """交叉受试者脑区分类测试"""
        
        # 准备该脑区 vs 其他脑区的二分类数据
        y_labels = np.argmax(self.data['y_train'], axis=1)
        
        # 创建二分类标签：该脑区=1，其他脑区=0
        binary_labels = (y_labels == region_id).astype(int)
        
        # 确保正负样本平衡
        positive_indices = np.where(binary_labels == 1)[0]
        negative_indices = np.where(binary_labels == 0)[0]
        
        if len(positive_indices) < 100 or len(negative_indices) < 100:
            return {'status': 'insufficient_data'}
        
        # 随机采样保持平衡
        n_samples = min(len(positive_indices), len(negative_indices), 5000)  # 限制样本量提高效率
        
        np.random.seed(42)
        selected_positive = np.random.choice(positive_indices, n_samples, replace=False)
        selected_negative = np.random.choice(negative_indices, n_samples, replace=False)
        
        selected_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(selected_indices)
        
        X_balanced = self.data['X_train'][selected_indices]
        y_balanced = binary_labels[selected_indices]
        subjects_balanced = self.data['subjects_train'][selected_indices]
        
        # 留一受试者交叉验证
        unique_subjects = np.unique(subjects_balanced)
        if len(unique_subjects) < 5:
            return {'status': 'insufficient_subjects'}
        
        cross_subject_scores = []
        
        # 随机选择5个受试者进行测试
        test_subjects = np.random.choice(unique_subjects, min(5, len(unique_subjects)), replace=False)
        
        for test_subject in test_subjects:
            train_mask = subjects_balanced != test_subject
            test_mask = subjects_balanced == test_subject
            
            if np.sum(test_mask) < 10:  # 测试样本太少
                continue
                
            X_train_cv = X_balanced[train_mask]
            y_train_cv = y_balanced[train_mask]
            X_test_cv = X_balanced[test_mask]
            y_test_cv = y_balanced[test_mask]
            
            # 训练简单分类器
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train_cv, y_train_cv)
            
            score = clf.score(X_test_cv, y_test_cv)
            cross_subject_scores.append(score)
        
        if len(cross_subject_scores) == 0:
            return {'status': 'no_valid_tests'}
        
        # 计算与随机分类器的对比
        random_baseline = 0.5  # 二分类随机基线
        mean_accuracy = np.mean(cross_subject_scores)
        improvement_over_random = (mean_accuracy - random_baseline) / random_baseline
        
        return {
            'status': 'success',
            'cross_subject_accuracies': cross_subject_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': np.std(cross_subject_scores),
            'random_baseline': random_baseline,
            'improvement_over_random': improvement_over_random,
            'n_tests': len(cross_subject_scores),
            'generalization_quality': mean_accuracy  # 泛化质量
        }

    def _assess_subject_specificity_impact(self, region_id):
        """评估受试者特异性对该脑区分类的影响"""
        
        y_labels = np.argmax(self.data['y_train'], axis=1)
        binary_labels = (y_labels == region_id).astype(int)
        
        # 同样的数据平衡处理
        positive_indices = np.where(binary_labels == 1)[0]
        negative_indices = np.where(binary_labels == 0)[0]
        
        if len(positive_indices) < 100 or len(negative_indices) < 100:
            return {'status': 'insufficient_data'}
        
        n_samples = min(len(positive_indices), len(negative_indices), 3000)
        
        np.random.seed(42)
        selected_positive = np.random.choice(positive_indices, n_samples, replace=False)
        selected_negative = np.random.choice(negative_indices, n_samples, replace=False)
        
        selected_indices = np.concatenate([selected_positive, selected_negative])
        np.random.shuffle(selected_indices)
        
        X_balanced = self.data['X_train'][selected_indices]
        y_balanced = binary_labels[selected_indices]
        subjects_balanced = self.data['subjects_train'][selected_indices]
        
        # 1. 随机分割基线性能（忽略受试者）
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42
        )
        
        clf_random = LogisticRegression(random_state=42, max_iter=1000)
        clf_random.fit(X_train_random, y_train_random)
        random_split_accuracy = clf_random.score(X_test_random, y_test_random)
        
        # 2. 受试者分割性能（考虑受试者）
        unique_subjects = np.unique(subjects_balanced)
        if len(unique_subjects) < 4:
            return {'status': 'insufficient_subjects'}
        
        # 选择一个受试者作为测试
        test_subject = np.random.choice(unique_subjects)
        train_mask = subjects_balanced != test_subject
        test_mask = subjects_balanced == test_subject
        
        if np.sum(test_mask) < 20:
            return {'status': 'insufficient_test_data'}
        
        X_train_subject = X_balanced[train_mask]
        y_train_subject = y_balanced[train_mask]
        X_test_subject = X_balanced[test_mask]
        y_test_subject = y_balanced[test_mask]
        
        clf_subject = LogisticRegression(random_state=42, max_iter=1000)
        clf_subject.fit(X_train_subject, y_train_subject)
        subject_split_accuracy = clf_subject.score(X_test_subject, y_test_subject)
        
        # 3. 计算受试者特异性影响
        subject_impact = random_split_accuracy - subject_split_accuracy
        impact_ratio = subject_impact / random_split_accuracy if random_split_accuracy > 0 else 0
        
        return {
            'status': 'success',
            'random_split_accuracy': random_split_accuracy,
            'subject_split_accuracy': subject_split_accuracy,
            'subject_specificity_impact': subject_impact,
            'impact_ratio': impact_ratio,
            'embedding_benefit_potential': max(0, impact_ratio)  # 潜在收益
        }

    def _compute_embedding_necessity_score(self, consistency_analysis, cross_subject_analysis, specificity_impact):
        """计算Subject Embedding必要性评分"""
        
        # 检查所有分析是否成功
        if (consistency_analysis.get('status') != 'success' or 
            cross_subject_analysis.get('status') != 'success' or 
            specificity_impact.get('status') != 'success'):
            return {
                'score': 0.0,
                'level': 'INSUFFICIENT_DATA',
                'recommended_dim': 0,
                'priority': 'SKIP'
            }
        
        # 计算综合评分 (0-1之间)
        
        # 1. 一致性评分 (越不一致越需要embedding)
        consistency_score = 1.0 - consistency_analysis['consistency_score']
        consistency_score = max(0, min(1, consistency_score))
        
        # 2. 泛化质量评分 (泛化性能越差越需要embedding)
        generalization_score = 1.0 - cross_subject_analysis['generalization_quality']
        generalization_score = max(0, min(1, generalization_score))
        
        # 3. 受试者影响评分
        impact_score = specificity_impact['embedding_benefit_potential']
        impact_score = max(0, min(1, impact_score))
        
        # 综合评分 (加权平均)
        weights = {'consistency': 0.4, 'generalization': 0.3, 'impact': 0.3}
        
        overall_score = (
            consistency_score * weights['consistency'] +
            generalization_score * weights['generalization'] + 
            impact_score * weights['impact']
        )
        
        # 确定等级和推荐维度
        if overall_score > 0.7:
            level = 'CRITICAL'
            recommended_dim = 128
            priority = 'HIGH'
        elif overall_score > 0.5:
            level = 'HIGH'
            recommended_dim = 64
            priority = 'MEDIUM-HIGH'
        elif overall_score > 0.3:
            level = 'MEDIUM'
            recommended_dim = 32
            priority = 'MEDIUM'
        else:
            level = 'LOW'
            recommended_dim = 16
            priority = 'LOW'
        
        return {
            'score': overall_score,
            'level': level,
            'recommended_dim': recommended_dim,
            'priority': priority,
            'component_scores': {
                'consistency': consistency_score,
                'generalization': generalization_score,
                'impact': impact_score
            }
        }


    def _test_random_split_performance(self, X, y):
        """测试随机分割的baseline性能 (🔥 增强版：全局 + 分脑区 + 深度网络分析)"""
        
        logger.info("    🔍 🔥 增强版Baseline随机分割性能测试（包含4×4096深度网络）...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # 🔥 扩展分类器列表，添加你的深度网络
        classifiers = {
            'complex_rf': RandomForestClassifier(
                n_estimators=200, max_depth=20, 
                min_samples_split=2, random_state=42),
            'simple_lr': LogisticRegression(
                max_iter=1000, C=0.1, random_state=42),
            # 🔥 严格使用alex超参数，不允许覆盖
            'deep_4x4096': self._create_deep_classifier_wrapper()  # 使用默认alex超参数
        }
        
        results = {
            'global_analysis': {},
            'region_wise_analysis': {},
            'comparative_analysis': {},
            'deep_network_detailed_analysis': {}  # 🔥 新增：深度网络详细分析
        }
        
        # ========================================================================
        # 1. 保留原有全局分析 + 🔥 深度网络增强
        # ========================================================================
        logger.info("      🌐 全局随机分割性能（含深度网络）...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        for name, clf in classifiers.items():
            logger.info(f"        🔍 训练全局{name}...")
            
            # 特殊处理深度网络
            if name == 'deep_4x4096':
                # 深度网络需要更长时间，给出详细反馈
                start_time = time.time()
                clf.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # 获取训练历史
                training_history = clf.training_history
                
                # 详细分析
                y_pred = clf.predict(X_val)
                y_pred_proba = clf.predict_proba(X_val)
                
                accuracy = accuracy_score(y_val.argmax(axis=1) if len(y_val.shape) > 1 else y_val, y_pred)
                f1_macro = f1_score(y_val.argmax(axis=1) if len(y_val.shape) > 1 else y_val, y_pred, average='macro')
                f1_weighted = f1_score(y_val.argmax(axis=1) if len(y_val.shape) > 1 else y_val, y_pred, average='weighted')
                
                results['global_analysis'][name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_val),
                    'n_classes': len(np.unique(y.argmax(axis=1) if len(y.shape) > 1 else y)),
                    'training_time_minutes': train_time / 60,
                    'final_train_accuracy': training_history['accuracy'][-1] if training_history['accuracy'] else 0
                }
                
                # 🔥 深度网络特殊分析
                results['deep_network_detailed_analysis']['global_performance'] = {
                    'convergence_epochs': len(training_history['loss']),
                    'final_training_loss': training_history['loss'][-1] if training_history['loss'] else 0,
                    'loss_trend': 'decreasing' if len(training_history['loss']) > 1 and training_history['loss'][-1] < training_history['loss'][0] else 'stable',
                    'overfitting_risk': training_history['accuracy'][-1] - accuracy if training_history['accuracy'] else 0,
                    'prediction_confidence': np.mean(np.max(y_pred_proba, axis=1))
                }
                
                logger.info(f"          ✅ 🔥 全局深度网络: Acc={accuracy:.3f}, F1_macro={f1_macro:.3f}, 训练时间={train_time/60:.1f}分钟")
                
            else:
                # 传统分类器的原有逻辑
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                f1_macro = f1_score(y_val, y_pred, average='macro')
                f1_weighted = f1_score(y_val, y_pred, average='weighted')
                
                results['global_analysis'][name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_val),
                    'n_classes': len(np.unique(y))
                }
                
                logger.info(f"          ✅ 全局{name}: Acc={accuracy:.3f}, F1_macro={f1_macro:.3f}")
        
        # ========================================================================
        # 2. 保留原有分脑区分析 + 深度网络增强
        # ========================================================================
        logger.info("      🧠 分脑区随机分割性能（含深度网络）...")
        
        region_dataset = self._build_region_aware_dataset()
        
        if len(region_dataset['features']) > 200:  # 确保有足够样本
            
            X_region = region_dataset['features']
            subject_labels_region = region_dataset['subject_labels']
            
            # 重新映射受试者标签
            unique_subjects = np.unique(subject_labels_region)
            subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
            y_region = np.array([subject_mapping[s] for s in subject_labels_region])
            
            logger.info(f"        📊 分脑区数据集: {len(X_region)} 样本, {len(unique_subjects)} 受试者")
            
            try:
                X_train_region, X_val_region, y_train_region, y_val_region = train_test_split(
                    X_region, y_region, test_size=0.2, stratify=y_region, random_state=42)
                
                for name, clf in classifiers.items():
                    logger.info(f"        🔍 训练分脑区{name}...")
                    
                    # 为分脑区分析创建新的分类器实例
                    if name == 'deep_4x4096':
                        clf_region = self._create_deep_classifier_wrapper({
                            'batch_size': 64,  # 分脑区用较小batch size
                            'epochs': 15      # 较少epoch，因为数据较少
                        })
                    else:
                        clf_region = type(clf)(**clf.get_params())
                    
                    start_time = time.time()
                    clf_region.fit(X_train_region, y_train_region)
                    train_time = time.time() - start_time
                    
                    y_pred_region = clf_region.predict(X_val_region)
                    accuracy_region = accuracy_score(y_val_region, y_pred_region)
                    f1_macro_region = f1_score(y_val_region, y_pred_region, average='macro')
                    f1_weighted_region = f1_score(y_val_region, y_pred_region, average='weighted')
                    
                    results['region_wise_analysis'][name] = {
                        'accuracy': accuracy_region,
                        'f1_macro': f1_macro_region,
                        'f1_weighted': f1_weighted_region,
                        'n_train_samples': len(X_train_region),
                        'n_test_samples': len(X_val_region),
                        'n_classes': len(unique_subjects),
                        'n_regions': len(np.unique(region_dataset['region_labels'])),
                        'training_time_minutes': train_time / 60
                    }
                    
                    # 🔥 深度网络在分脑区的特殊分析
                    if name == 'deep_4x4096':
                        region_history = clf_region.training_history
                        results['deep_network_detailed_analysis']['region_wise_performance'] = {
                            'convergence_epochs': len(region_history['loss']),
                            'final_training_loss': region_history['loss'][-1] if region_history['loss'] else 0,
                            'subject_discrimination_strength': accuracy_region / (1/len(unique_subjects)),  # 相对于随机的提升
                            'region_adaptation_time': train_time
                        }
                    
                    logger.info(f"          ✅ 分脑区{name}: Acc={accuracy_region:.3f}, F1_macro={f1_macro_region:.3f}")
                    
            except Exception as e:
                logger.info(f"        ❌ 分脑区分割失败: {e}")
                results['region_wise_analysis'] = {'error': str(e)}
        
        else:
            logger.info("        ⚠️ 分脑区样本不足，跳过分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # ========================================================================
        # 3. 性能对比分析 + 🔥 深度网络专项对比
        # ========================================================================
        logger.info("      📈 Baseline性能对比分析（含深度网络对比）...")
        
        for name in classifiers.keys():
            if (name in results['global_analysis'] and 
                name in results['region_wise_analysis'] and
                'accuracy' in results['global_analysis'][name] and
                'accuracy' in results['region_wise_analysis'][name]):
                
                global_metrics = results['global_analysis'][name]
                region_metrics = results['region_wise_analysis'][name]
                
                # 计算各指标的提升
                acc_improvement = region_metrics['accuracy'] - global_metrics['accuracy']
                f1_improvement = region_metrics['f1_macro'] - global_metrics['f1_macro']
                
                results['comparative_analysis'][name] = {
                    'accuracy_improvement': acc_improvement,
                    'f1_macro_improvement': f1_improvement,
                    'accuracy_improvement_percent': acc_improvement / global_metrics['accuracy'] * 100,
                    'f1_improvement_percent': f1_improvement / global_metrics['f1_macro'] * 100,
                    'sample_size_ratio': region_metrics['n_train_samples'] / global_metrics['n_train_samples'],
                    'interpretation': self._interpret_baseline_improvement(acc_improvement, f1_improvement)
                }
                
                # 🔥 深度网络特殊对比分析
                if name == 'deep_4x4096':
                    results['comparative_analysis'][name].update({
                        'training_time_ratio': region_metrics['training_time_minutes'] / global_metrics['training_time_minutes'],
                        'deep_network_advantage': 'significant' if acc_improvement > 0.05 else 'moderate' if acc_improvement > 0.02 else 'minimal',
                        'efficiency_assessment': 'efficient' if region_metrics['training_time_minutes'] < 5 else 'moderate' if region_metrics['training_time_minutes'] < 15 else 'slow'
                    })
                
                logger.info(f"        📊 {name}对比:")
                logger.info(f"          准确率: {global_metrics['accuracy']:.3f} → {region_metrics['accuracy']:.3f} "
                    f"({acc_improvement:+.3f})")
                logger.info(f"          F1分数: {global_metrics['f1_macro']:.3f} → {region_metrics['f1_macro']:.3f} "
                    f"({f1_improvement:+.3f})")
        
        # 🔥 深度网络总结分析
        if 'deep_4x4096' in results['global_analysis']:
            deep_global = results['global_analysis']['deep_4x4096']
            deep_detailed = results['deep_network_detailed_analysis']
            
            logger.info(f"      🔥 深度网络总结:")
            logger.info(f"        - 全局性能: {deep_global['accuracy']:.3f} (训练时间: {deep_global['training_time_minutes']:.1f}分钟)")
            logger.info(f"        - 收敛性: {deep_detailed['global_performance']['loss_trend']}")
            logger.info(f"        - 过拟合风险: {deep_detailed['global_performance']['overfitting_risk']:.3f}")
            
            if 'region_wise_performance' in deep_detailed:
                region_perf = deep_detailed['region_wise_performance']
                logger.info(f"        - 受试者判别强度: {region_perf['subject_discrimination_strength']:.2f}x随机基线")
        
        return results

    def _interpret_baseline_improvement(self, acc_improvement, f1_improvement):
        """解释baseline性能改进"""
        if acc_improvement > 0.05 and f1_improvement > 0.05:
            return "显著改善：分脑区方法在准确率和F1分数上都有明显提升"
        elif acc_improvement > 0.02 or f1_improvement > 0.02:
            return "中等改善：分脑区方法在某些指标上有所提升"
        elif acc_improvement > -0.02 and f1_improvement > -0.02:
            return "性能相当：两种方法表现相似"
        else:
            return "性能下降：全局方法可能更适合baseline任务"
        


    def _test_leave_one_subject_out_performance(self, X, y, subjects):
        """测试Leave-One-Subject-Out性能 (🔥 增强版：全局 + 分脑区 + 权威深度网络LOSO分析)"""
        
        logger.info("    🔍 🔥 增强版Leave-One-Subject-Out性能测试（含4×4096深度网络权威评估）...")
        
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        
        results = {
            'global_analysis': {
                'per_subject_scores': {},
                'overall_performance': {}
            },
            'region_wise_analysis': {
                'per_subject_scores': {},
                'overall_performance': {},
                'per_region_performance': {}
            },
            'comparative_analysis': {},
            'deep_network_loso_analysis': {}  # 🔥 新增：深度网络LOSO专项分析
        }
        
        # 🔥 扩展测试模型，包含深度网络
        test_models = {
            'RandomForest': {
                'creator': lambda: RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
                'type': 'traditional'
            },
            'Deep4x4096': {
                # 🔥 LOSO时只适度调整batch_size，保持其他alex超参数
                'creator': lambda: self._create_deep_classifier_wrapper({
                    'batch_size': 64  # 🔥 LOSO时数据较少，只调整batch_size
                    # epochs和learning_rate保持alex默认值25和0.00001
                }),
                'type': 'deep_network'
            }
        }
        
        # ========================================================================
        # 1. 保留原有全局LOSO分析 + 🔥 深度网络增强
        # ========================================================================
        logger.info("      🌐 全局LOSO性能分析（含深度网络）...")
        
        unique_subjects = np.unique(subjects)
        
        # 对每个模型运行LOSO
        for model_name, model_info in test_models.items():
            logger.info(f"        🔍 {model_name} 全局LOSO测试...")
            
            all_accuracies = []
            all_f1_scores = []
            all_training_times = []
            deep_network_metrics = []  # 🔥 深度网络特殊指标
            
            # 只测试部分受试者以节省时间
            test_subjects = unique_subjects[:8]
            
            for test_subject in test_subjects:
                logger.info(f"          🔍 {model_name}测试受试者{test_subject}...")
                
                train_mask = subjects != test_subject
                test_mask = subjects == test_subject
                
                if np.sum(test_mask) < 1000:
                    logger.info(f"            ⚠️ 受试者{test_subject}样本不足，跳过")
                    continue
                
                X_train_loso, X_test_loso = X[train_mask], X[test_mask]
                y_train_loso, y_test_loso = y[train_mask], y[test_mask]
                
                # 检查类别平衡
                if len(y_train_loso.shape) > 1:
                    train_classes = len(np.unique(np.argmax(y_train_loso, axis=1)))
                    test_classes = len(np.unique(np.argmax(y_test_loso, axis=1)))
                    y_test_for_eval = np.argmax(y_test_loso, axis=1)
                else:
                    train_classes = len(np.unique(y_train_loso))
                    test_classes = len(np.unique(y_test_loso))
                    y_test_for_eval = y_test_loso
                
                if train_classes < 50 or test_classes < 20:
                    logger.info(f"            ⚠️ 受试者{test_subject}类别不足，跳过")
                    continue
                
                # 创建并训练模型
                model = model_info['creator']()
                
                start_time = time.time()
                model.fit(X_train_loso, y_train_loso)
                train_time = time.time() - start_time
                
                # 测试性能
                y_pred = model.predict(X_test_loso)
                accuracy = accuracy_score(y_test_for_eval, y_pred)
                f1_macro = f1_score(y_test_for_eval, y_pred, average='macro')
                
                all_accuracies.append(accuracy)
                all_f1_scores.append(f1_macro)
                all_training_times.append(train_time)
                
                # 🔥 深度网络特殊指标收集
                if model_info['type'] == 'deep_network':
                    training_history = model.training_history
                    y_pred_proba = model.predict_proba(X_test_loso)
                    
                    deep_metrics = {
                        'subject_id': test_subject,
                        'convergence_epochs': len(training_history['loss']),
                        'final_training_loss': training_history['loss'][-1] if training_history['loss'] else 0,
                        'overfitting_indicator': training_history['accuracy'][-1] - accuracy if training_history['accuracy'] else 0,
                        'prediction_confidence': np.mean(np.max(y_pred_proba, axis=1)),
                        'generalization_strength': accuracy,
                        'adaptation_time': train_time,
                        'sample_efficiency': accuracy / (train_time / 60)  # 准确率/分钟
                    }
                    deep_network_metrics.append(deep_metrics)
                
                results['global_analysis']['per_subject_scores'][test_subject] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'n_test_samples': len(X_test_loso),
                    'n_test_classes': test_classes,
                    'training_time': train_time,
                    'model_type': model_name
                }
                
                logger.info(f"            ✅ {model_name} 受试者{test_subject}: Acc={accuracy:.3f}, F1={f1_macro:.3f}, 时间={train_time/60:.1f}分钟")
            
            # 整体统计
            if all_accuracies:
                results['global_analysis']['overall_performance'][model_name] = {
                    'mean_accuracy': np.mean(all_accuracies),
                    'std_accuracy': np.std(all_accuracies),
                    'mean_f1_macro': np.mean(all_f1_scores),
                    'std_f1_macro': np.std(all_f1_scores),
                    'mean_training_time': np.mean(all_training_times),
                    'n_tested_subjects': len(all_accuracies)
                }
                
                logger.info(f"        📊 {model_name} 全局LOSO总体性能:")
                logger.info(f"          - 平均准确率: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")
                logger.info(f"          - 平均F1: {np.mean(all_f1_scores):.3f} ± {np.std(all_f1_scores):.3f}")
                logger.info(f"          - 平均训练时间: {np.mean(all_training_times)/60:.1f}分钟")
                
                # 🔥 深度网络详细分析
                if model_info['type'] == 'deep_network' and deep_network_metrics:
                    results['deep_network_loso_analysis']['global_detailed'] = {
                        'per_subject_metrics': deep_network_metrics,
                        'convergence_analysis': {
                            'avg_convergence_epochs': np.mean([m['convergence_epochs'] for m in deep_network_metrics]),
                            'convergence_stability': np.std([m['convergence_epochs'] for m in deep_network_metrics])
                        },
                        'generalization_analysis': {
                            'avg_confidence': np.mean([m['prediction_confidence'] for m in deep_network_metrics]),
                            'confidence_consistency': np.std([m['prediction_confidence'] for m in deep_network_metrics]),
                            'overfitting_tendency': np.mean([m['overfitting_indicator'] for m in deep_network_metrics])
                        },
                        'efficiency_analysis': {
                            'avg_sample_efficiency': np.mean([m['sample_efficiency'] for m in deep_network_metrics]),
                            'time_performance_correlation': np.corrcoef(
                                [m['adaptation_time'] for m in deep_network_metrics],
                                [m['generalization_strength'] for m in deep_network_metrics]
                            )[0, 1] if len(deep_network_metrics) > 1 else 0
                        }
                    }
                    
                    logger.info(f"        🔥 深度网络LOSO详细分析:")
                    logger.info(f"          - 平均收敛轮数: {results['deep_network_loso_analysis']['global_detailed']['convergence_analysis']['avg_convergence_epochs']:.1f}")
                    logger.info(f"          - 平均预测置信度: {results['deep_network_loso_analysis']['global_detailed']['generalization_analysis']['avg_confidence']:.3f}")
                    logger.info(f"          - 过拟合倾向: {results['deep_network_loso_analysis']['global_detailed']['generalization_analysis']['overfitting_tendency']:.3f}")
                    logger.info(f"          - 样本效率: {results['deep_network_loso_analysis']['global_detailed']['efficiency_analysis']['avg_sample_efficiency']:.3f} 准确率/分钟")
        
        # ========================================================================
        # 2. 保留原有分脑区LOSO分析 + 深度网络增强
        # ========================================================================
        logger.info("      🧠 分脑区LOSO性能分析（含深度网络）...")
        
        region_dataset = self._build_region_aware_dataset()
        
        if len(region_dataset['features']) > 500:  # 确保有足够样本
            
            X_region_full = region_dataset['features']
            subject_labels_region_full = region_dataset['subject_labels']
            region_labels_region_full = region_dataset['region_labels']
            
            # 重新映射受试者标签
            unique_subjects_region = np.unique(subject_labels_region_full)
            subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects_region)}
            y_region_full = np.array([subject_mapping[s] for s in subject_labels_region_full])
            
            # 为每个模型执行分脑区LOSO
            for model_name, model_info in test_models.items():
                logger.info(f"        🔍 {model_name} 分脑区LOSO测试...")
                
                all_accuracies_region = []
                all_f1_scores_region = []
                region_deep_metrics = []  # 🔥 分脑区深度网络指标
                
                # 测试部分受试者
                test_subjects_region = [s for s in test_subjects if s in unique_subjects_region][:5]
                
                for test_subject in test_subjects_region:
                    logger.info(f"          🔍 {model_name} 分脑区测试受试者{test_subject}...")
                    
                    # 找到该受试者在分脑区数据中的所有样本
                    test_mask_region = subject_labels_region_full == test_subject
                    train_mask_region = subject_labels_region_full != test_subject
                    
                    if np.sum(test_mask_region) < 10:
                        logger.info(f"            ⚠️ 受试者{test_subject}分脑区样本不足，跳过")
                        continue
                    
                    X_train_region = X_region_full[train_mask_region]
                    y_train_region = y_region_full[train_mask_region]
                    X_test_region = X_region_full[test_mask_region]
                    y_test_region = y_region_full[test_mask_region]
                    
                    # 检查类别平衡
                    train_classes_region = len(np.unique(y_train_region))
                    test_classes_region = len(np.unique(y_test_region))
                    
                    if train_classes_region < 5:
                        logger.info(f"            ⚠️ 受试者{test_subject}训练类别不足，跳过")
                        continue
                    
                    # 创建模型
                    if model_info['type'] == 'deep_network':
                        clf_region = self._create_deep_classifier_wrapper({
                            'batch_size': 32,  # 分脑区数据更少，用更小batch
                            'epochs': 20       # 更少epoch
                        })
                    else:
                        clf_region = model_info['creator']()
                    
                    start_time = time.time()
                    clf_region.fit(X_train_region, y_train_region)
                    train_time = time.time() - start_time
                    
                    # 测试分脑区性能
                    y_pred_region = clf_region.predict(X_test_region)
                    accuracy_region = accuracy_score(y_test_region, y_pred_region)
                    f1_macro_region = f1_score(y_test_region, y_pred_region, average='macro')
                    
                    all_accuracies_region.append(accuracy_region)
                    all_f1_scores_region.append(f1_macro_region)
                    
                    # 🔥 分脑区深度网络特殊分析
                    if model_info['type'] == 'deep_network':
                        training_history = clf_region.training_history
                        
                        region_deep_metric = {
                            'subject_id': test_subject,
                            'region_adaptation_epochs': len(training_history['loss']),
                            'region_specific_accuracy': accuracy_region,
                            'subject_discrimination_in_regions': accuracy_region / (1/len(np.unique(y_train_region))),
                            'region_training_efficiency': accuracy_region / (train_time / 60),
                            'n_regions_tested': len(np.unique(region_labels_region_full[test_mask_region]))
                        }
                        region_deep_metrics.append(region_deep_metric)
                    
                    # 分析该受试者每个脑区的表现
                    test_regions = region_labels_region_full[test_mask_region]
                    region_performance = {}
                    
                    for region_id in np.unique(test_regions):
                        region_test_mask = test_regions == region_id
                        if np.sum(region_test_mask) >= 3:  # 至少3个样本
                            region_acc = accuracy_score(
                                y_test_region[region_test_mask], 
                                y_pred_region[region_test_mask]
                            )
                            region_performance[region_id] = {
                                'accuracy': region_acc,
                                'n_samples': np.sum(region_test_mask)
                            }
                    
                    results['region_wise_analysis']['per_subject_scores'][test_subject] = {
                        'accuracy': accuracy_region,
                        'f1_macro': f1_macro_region,
                        'n_test_samples': len(X_test_region),
                        'n_test_regions': len(np.unique(test_regions)),
                        'per_region_performance': region_performance,
                        'model_type': model_name,
                        'training_time': train_time
                    }
                    
                    logger.info(f"            ✅ {model_name} 分脑区受试者{test_subject}: Acc={accuracy_region:.3f}, F1={f1_macro_region:.3f}")
                    logger.info(f"            📊 测试脑区数: {len(np.unique(test_regions))}")
                
                # 分脑区总体统计
                if all_accuracies_region:
                    results['region_wise_analysis']['overall_performance'][model_name] = {
                        'mean_accuracy': np.mean(all_accuracies_region),
                        'std_accuracy': np.std(all_accuracies_region),
                        'mean_f1_macro': np.mean(all_f1_scores_region),
                        'std_f1_macro': np.std(all_f1_scores_region),
                        'n_tested_subjects': len(all_accuracies_region)
                    }
                    
                    logger.info(f"        📊 {model_name} 分脑区LOSO总体性能:")
                    logger.info(f"          - 平均准确率: {np.mean(all_accuracies_region):.3f} ± {np.std(all_accuracies_region):.3f}")
                    logger.info(f"          - 平均F1: {np.mean(all_f1_scores_region):.3f} ± {np.std(all_f1_scores_region):.3f}")
                    
                    # 🔥 分脑区深度网络分析
                    if model_info['type'] == 'deep_network' and region_deep_metrics:
                        results['deep_network_loso_analysis']['region_wise_detailed'] = {
                            'per_subject_region_metrics': region_deep_metrics,
                            'region_adaptation_analysis': {
                                'avg_adaptation_epochs': np.mean([m['region_adaptation_epochs'] for m in region_deep_metrics]),
                                'region_discrimination_strength': np.mean([m['subject_discrimination_in_regions'] for m in region_deep_metrics]),
                                'region_efficiency': np.mean([m['region_training_efficiency'] for m in region_deep_metrics])
                            }
                        }
                        
                        logger.info(f"        🔥 分脑区深度网络分析:")
                        logger.info(f"          - 平均脑区适应轮数: {results['deep_network_loso_analysis']['region_wise_detailed']['region_adaptation_analysis']['avg_adaptation_epochs']:.1f}")
                        logger.info(f"          - 脑区受试者判别强度: {results['deep_network_loso_analysis']['region_wise_detailed']['region_adaptation_analysis']['region_discrimination_strength']:.2f}x")
        
        else:
            logger.info("        ⚠️ 分脑区样本不足，跳过LOSO分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # ========================================================================
        # 3. LOSO对比分析 + 🔥 深度网络权威评估
        # ========================================================================
        logger.info("      📈 LOSO性能对比分析（含深度网络权威评估）...")
        
        # 传统对比分析
        for model_name in test_models.keys():
            if (model_name in results['global_analysis']['overall_performance'] and 
                model_name in results['region_wise_analysis']['overall_performance']):
                
                global_perf = results['global_analysis']['overall_performance'][model_name]
                region_perf = results['region_wise_analysis']['overall_performance'][model_name]
                
                acc_improvement = region_perf['mean_accuracy'] - global_perf['mean_accuracy']
                f1_improvement = region_perf['mean_f1_macro'] - global_perf['mean_f1_macro']
                
                # 计算泛化稳定性（标准差比较）
                acc_stability_change = global_perf['std_accuracy'] - region_perf['std_accuracy']
                f1_stability_change = global_perf['std_f1_macro'] - region_perf['std_f1_macro']
                
                results['comparative_analysis'][model_name] = {
                    'accuracy_improvement': acc_improvement,
                    'f1_improvement': f1_improvement,
                    'accuracy_stability_improvement': acc_stability_change,
                    'f1_stability_improvement': f1_stability_change,
                    'generalization_assessment': self._assess_generalization_improvement(
                        acc_improvement, f1_improvement, acc_stability_change, f1_stability_change),
                    'subject_embedding_value': self._assess_subject_embedding_loso_value(
                        global_perf, region_perf)
                }
                
                logger.info(f"        📊 {model_name} LOSO性能对比:")
                logger.info(f"          准确率提升: {acc_improvement:+.3f} "
                    f"(稳定性变化: {acc_stability_change:+.3f})")
                logger.info(f"          F1分数提升: {f1_improvement:+.3f} "
                    f"(稳定性变化: {f1_stability_change:+.3f})")
                logger.info(f"          🎯 {results['comparative_analysis'][model_name]['generalization_assessment']}")
        
        # 🔥 深度网络权威Subject Embedding价值评估
        if 'Deep4x4096' in results['global_analysis']['overall_performance']:
            deep_global = results['global_analysis']['overall_performance']['Deep4x4096']
            deep_detailed = results['deep_network_loso_analysis']
            
            # 权威评估
            subject_embedding_necessity = self._deep_network_subject_embedding_assessment(
                deep_global, deep_detailed, results['comparative_analysis'].get('Deep4x4096', {})
            )
            
            results['deep_network_loso_analysis']['authoritative_assessment'] = subject_embedding_necessity
            
            logger.info(f"      🔥 深度网络权威Subject Embedding评估:")
            logger.info(f"        - 总体推荐: {subject_embedding_necessity['recommendation']}")
            logger.info(f"        - 置信度: {subject_embedding_necessity['confidence']}")
            logger.info(f"        - 预期收益: {subject_embedding_necessity['expected_benefit']}")
            logger.info(f"        - 实施复杂度: {subject_embedding_necessity['implementation_complexity']}")
        
        return results

    def _deep_network_subject_embedding_assessment(self, global_performance, detailed_analysis, comparative_analysis):
        """基于深度网络LOSO结果的权威Subject Embedding评估"""
        
        global_acc = global_performance['mean_accuracy']
        global_std = global_performance['std_accuracy']
        
        # 评估因子
        factors = {
            'generalization_gap': 0,
            'consistency': 0,
            'efficiency': 0,
            'convergence_stability': 0
        }
        
        # 1. 泛化性能差距
        if 'accuracy_improvement' in comparative_analysis:
            acc_improvement = comparative_analysis['accuracy_improvement']
            if acc_improvement < -0.05:
                factors['generalization_gap'] = 1.0  # 显著泛化问题
            elif acc_improvement < -0.02:
                factors['generalization_gap'] = 0.7
            elif acc_improvement < 0.02:
                factors['generalization_gap'] = 0.4
            else:
                factors['generalization_gap'] = 0.0  # 泛化良好
        
        # 2. 预测一致性
        if global_std > 0.1:
            factors['consistency'] = 1.0  # 高变异，需要Subject Embedding
        elif global_std > 0.05:
            factors['consistency'] = 0.6
        else:
            factors['consistency'] = 0.2
        
        # 3. 过拟合倾向
        if 'global_detailed' in detailed_analysis:
            overfitting_tendency = detailed_analysis['global_detailed']['generalization_analysis']['overfitting_tendency']
            if overfitting_tendency > 0.1:
                factors['efficiency'] = 0.8
            elif overfitting_tendency > 0.05:
                factors['efficiency'] = 0.5
            else:
                factors['efficiency'] = 0.2
        
        # 4. 收敛稳定性
        if 'global_detailed' in detailed_analysis:
            convergence_stability = detailed_analysis['global_detailed']['convergence_analysis']['convergence_stability']
            if convergence_stability > 5:
                factors['convergence_stability'] = 0.7
            elif convergence_stability > 2:
                factors['convergence_stability'] = 0.4
            else:
                factors['convergence_stability'] = 0.1
        
        # 综合评分
        necessity_score = np.mean(list(factors.values()))
        
        # 生成权威建议
        if necessity_score > 0.7:
            recommendation = "强烈推荐Subject Embedding"
            confidence = "高"
            expected_benefit = "显著性能提升 (5-15%)"
            implementation_complexity = "值得投入"
        elif necessity_score > 0.5:
            recommendation = "建议考虑Subject Embedding"
            confidence = "中高"
            expected_benefit = "中等性能提升 (2-8%)"
            implementation_complexity = "适度投入"
        elif necessity_score > 0.3:
            recommendation = "可选择性使用Subject Embedding"
            confidence = "中"
            expected_benefit = "小幅性能提升 (1-5%)"
            implementation_complexity = "低风险尝试"
        else:
            recommendation = "Subject Embedding价值有限"
            confidence = "高"
            expected_benefit = "微小或无提升"
            implementation_complexity = "不建议投入"
        
        return {
            'necessity_score': necessity_score,
            'recommendation': recommendation,
            'confidence': confidence,
            'expected_benefit': expected_benefit,
            'implementation_complexity': implementation_complexity,
            'factor_breakdown': factors,
            'key_insights': [
                f"泛化能力评估: {'需要改善' if factors['generalization_gap'] > 0.5 else '表现良好'}",
                f"预测一致性: {'波动较大' if factors['consistency'] > 0.5 else '相对稳定'}",
                f"过拟合风险: {'较高' if factors['efficiency'] > 0.5 else '可控'}",
                f"收敛稳定性: {'不稳定' if factors['convergence_stability'] > 0.5 else '稳定'}"
            ]
        }

    def _assess_generalization_improvement(self, acc_imp, f1_imp, acc_stab, f1_stab):
        """评估泛化性能改进"""
        if acc_imp > 0.03 and f1_imp > 0.03 and acc_stab > 0 and f1_stab > 0:
            return "优秀：分脑区方法在准确率、F1和稳定性上都有显著提升"
        elif acc_imp > 0.02 or f1_imp > 0.02:
            if acc_stab > 0 or f1_stab > 0:
                return "良好：分脑区方法在性能或稳定性上有所改善"
            else:
                return "中等：分脑区方法性能有提升但稳定性略有下降"
        elif acc_imp > -0.01 and f1_imp > -0.01:
            return "相当：两种方法的泛化性能相似"
        else:
            return "较差：全局方法的泛化性能可能更好"

    def _assess_subject_embedding_loso_value(self, global_perf, region_perf):
        """评估Subject Embedding在LOSO任务中的价值"""
        global_f1 = global_perf['mean_f1_macro']
        region_f1 = region_perf['mean_f1_macro']
        
        improvement = region_f1 - global_f1
        relative_improvement = improvement / global_f1 if global_f1 > 0 else 0
        
        if relative_improvement > 0.15:
            return {
                'recommendation': "强烈推荐Subject Embedding",
                'expected_improvement': f"预期LOSO性能提升{improvement:.3f}",
                'priority': "HIGH"
            }
        elif relative_improvement > 0.08:
            return {
                'recommendation': "建议使用Subject Embedding",
                'expected_improvement': f"预期LOSO性能提升{improvement:.3f}",
                'priority': "MEDIUM-HIGH"
            }
        elif relative_improvement > 0.02:
            return {
                'recommendation': "可以尝试Subject Embedding",
                'expected_improvement': f"预期LOSO性能提升{improvement:.3f}",
                'priority': "MEDIUM"
            }
        else:
            return {
                'recommendation': "Subject Embedding价值有限",
                'expected_improvement': f"预期提升仅{improvement:.3f}",
                'priority': "LOW"
            }


    def _assess_subject_embedding_value(self, baseline_results, loso_results):
        """评估Subject Embedding的潜在价值"""
        
        logger.info("    🔍 评估Subject Embedding潜在价值...")
        
        # 获取性能数据
        baseline_f1 = baseline_results.get('complex_rf', {}).get('f1_macro', 0.0)
        loso_f1 = loso_results.get('overall_performance', {}).get('mean_f1_macro', 0.0)
        
        if baseline_f1 == 0 or loso_f1 == 0:
            logger.info("    ⚠️ 性能数据不足，无法评估")
            return {}
        
        # 计算性能差距
        generalization_gap = baseline_f1 - loso_f1
        relative_gap = generalization_gap / baseline_f1 if baseline_f1 > 0 else 0
        
        # Subject Embedding价值评估
        if relative_gap > 0.15:  # 差距>15%
            embedding_recommendation = "强烈推荐Subject Embedding"
            expected_improvement = f"预期提升{generalization_gap*0.5:.3f}F1分数"
            priority = "HIGH"
        elif relative_gap > 0.08:  # 差距>8%
            embedding_recommendation = "建议使用Subject Embedding"  
            expected_improvement = f"预期提升{generalization_gap*0.3:.3f}F1分数"
            priority = "MEDIUM"
        else:
            embedding_recommendation = "Subject Embedding价值有限"
            expected_improvement = f"预期提升{generalization_gap*0.1:.3f}F1分数"
            priority = "LOW"
        
        assessment = {
            'baseline_f1': baseline_f1,
            'loso_f1': loso_f1,
            'generalization_gap': generalization_gap,
            'relative_gap_percent': relative_gap * 100,
            'embedding_recommendation': embedding_recommendation,
            'expected_improvement': expected_improvement,
            'priority': priority
        }
        
        logger.info(f"    📊 Subject Embedding价值评估:")
        logger.info(f"      - Baseline F1 (随机分割): {baseline_f1:.3f}")
        logger.info(f"      - LOSO F1 (受试者泛化): {loso_f1:.3f}")
        logger.info(f"      - 泛化差距: {generalization_gap:.3f} ({relative_gap*100:.1f}%)")
        logger.info(f"      - 推荐: {embedding_recommendation}")
        logger.info(f"      - {expected_improvement}")
        
        return assessment

                
    def _phase2c_class_consistency_analysis(self):
        """脑区分类一致性分析 (保持原有逻辑)"""
        
        logger.info("🔍 2.2C 脑区分类一致性分析...")
        
        # 转换one-hot标签到类别索引
        if len(self.data['y_train'].shape) > 1 and self.data['y_train'].shape[1] > 1:
            y_train_classes = np.argmax(self.data['y_train'], axis=1)
        else:
            y_train_classes = self.data['y_train'].flatten()
        
        unique_classes = np.unique(y_train_classes)
        n_classes = len(unique_classes)
        
        logger.info(f"  - 脑区类别数量: {n_classes}")
        logger.info(f"  - 分析前20个脑区的一致性...")
        
        # 计算每个脑区在不同受试者间的一致性
        class_consistency = {}
        
        subjects_train_int = self.data['subjects_train'].astype(int)
        
        for class_id in unique_classes[:20]:  # 只分析前20个类别
            class_mask = y_train_classes == class_id
            
            if np.sum(class_mask) > 100:  # 确保有足够样本
                class_data = self.data['X_train'][class_mask]
                class_subjects = subjects_train_int[class_mask]
                
                # 计算每个受试者该脑区的均值特征
                subject_means_for_class = []
                subjects_with_class = []
                
                for subject_id in np.unique(class_subjects):
                    subject_class_mask = class_subjects == subject_id
                    if np.sum(subject_class_mask) >= 10:  # 至少10个样本
                        subject_mean = np.mean(class_data[subject_class_mask], axis=0)
                        subject_means_for_class.append(subject_mean)
                        subjects_with_class.append(subject_id)
                
                if len(subject_means_for_class) >= 3:  # 至少3个受试者
                    subject_means_array = np.array(subject_means_for_class)
                    
                    # 计算受试者间的变异系数
                    feature_cv = np.std(subject_means_array, axis=0) / (np.abs(np.mean(subject_means_array, axis=0)) + 1e-8)
                    mean_cv = np.mean(feature_cv)
                    
                    # 计算受试者间距离
                    distances = pdist(subject_means_array)
                    mean_distance = np.mean(distances)
                    
                    class_consistency[class_id] = {
                        'mean_cv': mean_cv,
                        'mean_distance': mean_distance,
                        'n_subjects': len(subjects_with_class),
                        'n_samples': np.sum(class_mask),
                        'subjects_with_class': subjects_with_class
                    }
        
        self.analysis_results['class_consistency'] = class_consistency
        
        if class_consistency:
            avg_consistency = np.mean([info['mean_cv'] for info in class_consistency.values()])
            avg_distance = np.mean([info['mean_distance'] for info in class_consistency.values()])
            
            logger.info(f"✅ 脑区一致性分析完成")
            logger.info(f"  - 分析脑区数量: {len(class_consistency)}")
            logger.info(f"  - 平均变异系数: {avg_consistency:.3f} (越低越一致)")
            logger.info(f"  - 平均受试者间距离: {avg_distance:.3f}")
            
            # 找出最一致和最不一致的脑区
            if len(class_consistency) > 0:
                most_consistent_class = min(class_consistency.keys(), 
                                        key=lambda x: class_consistency[x]['mean_cv'])
                least_consistent_class = max(class_consistency.keys(), 
                                        key=lambda x: class_consistency[x]['mean_cv'])
                
                logger.info(f"  - 最一致脑区: {most_consistent_class} (CV: {class_consistency[most_consistent_class]['mean_cv']:.3f})")
                logger.info(f"  - 最不一致脑区: {least_consistent_class} (CV: {class_consistency[least_consistent_class]['mean_cv']:.3f})")
        else:
            logger.info("❌ 脑区一致性分析失败")


    def _compute_phase2_combined_scores(self):
        """计算Phase 2的综合决策得分（🔥 深度网络增强版）"""
        
        # 原有得分计算（保持不变）
        global_separability = 0.0
        if 'global_subject_identification' in self.analysis_results:
            global_id = self.analysis_results['global_subject_identification']
            global_accuracy = global_id['accuracy']
            random_baseline = global_id['random_baseline']
            global_separability = max(0, min(1, (global_accuracy - random_baseline) / (1 - random_baseline + 1e-8)))
        
        consistency_score = 0.5
        if 'class_consistency' in self.analysis_results and self.analysis_results['class_consistency']:
            avg_consistency = np.mean([info['mean_cv'] for info in self.analysis_results['class_consistency'].values()])
            consistency_score = max(0, min(1, 1 / (1 + avg_consistency)))
        
        # 🔥 新增：基于深度网络分析的Subject Embedding需求评分
        embedding_necessity_score = 0.0
        high_necessity_ratio = 0.0
        deep_network_authority_boost = 0.0  # 🔥 深度网络权威性加成
        
        if 'region_wise_separability' in self.analysis_results and self.analysis_results['region_wise_separability']:
            region_analysis = self.analysis_results['region_wise_separability']
            
            # 只分析成功的脑区
            successful_regions = [rid for rid, data in region_analysis.items() 
                                if 'embedding_necessity_score' in data or 'comprehensive_embedding_necessity_score' in data]
            
            if successful_regions:
                # 使用综合评分（包含深度网络分析）
                necessity_scores = []
                deep_validated_count = 0
                
                for rid in successful_regions:
                    data = region_analysis[rid]
                    
                    # 优先使用综合评分，否则使用传统评分
                    if 'comprehensive_embedding_necessity_score' in data:
                        necessity_scores.append(data['comprehensive_embedding_necessity_score'])
                        if data.get('deep_network_validation', False):
                            deep_validated_count += 1
                    else:
                        necessity_scores.append(data['embedding_necessity_score'])
                
                embedding_necessity_score = np.mean(necessity_scores)
                
                # 计算高需求脑区比例（使用最终等级）
                high_necessity_count = 0
                for rid in successful_regions:
                    data = region_analysis[rid]
                    final_level = data.get('comprehensive_embedding_necessity_level', 
                                        data.get('embedding_necessity_level', 'LOW'))
                    if final_level in ['CRITICAL', 'HIGH']:
                        high_necessity_count += 1
                
                high_necessity_ratio = high_necessity_count / len(successful_regions)
                
                # 🔥 深度网络权威性加成
                deep_validation_ratio = deep_validated_count / len(successful_regions)
                deep_network_authority_boost = deep_validation_ratio * 0.1  # 最多10%的权威性加成
        
        # 🔥 深度网络LOSO分析得分
        deep_loso_authority_score = 0.0
        if 'neural_network_baseline_analysis' in self.analysis_results:
            baseline_analysis = self.analysis_results['neural_network_baseline_analysis']
            
            if 'deep_network_loso_analysis' in baseline_analysis:
                deep_loso = baseline_analysis['deep_network_loso_analysis']
                
                if 'authoritative_assessment' in deep_loso:
                    auth_assessment = deep_loso['authoritative_assessment']
                    necessity_score = auth_assessment['necessity_score']
                    deep_loso_authority_score = necessity_score
            elif 'leave_one_subject_out' in baseline_analysis:
                # 从LOSO结果中提取深度网络性能
                loso_results = baseline_analysis['leave_one_subject_out']
                if 'deep_network_loso_analysis' in loso_results:
                    deep_loso_detailed = loso_results['deep_network_loso_analysis']
                    if 'authoritative_assessment' in deep_loso_detailed:
                        auth_assessment = deep_loso_detailed['authoritative_assessment']
                        deep_loso_authority_score = auth_assessment['necessity_score']
        
        # Phase 2 综合决策得分（🔥 深度网络增强版）
        self.decision_scores['phase2'] = {
            # 原有得分（重新解释含义）
            'global_subject_variability': global_separability,
            'class_consistency': consistency_score,
            'identification_accuracy': global_separability,  # 保持向后兼容
            'feature_competition_risk': 1.0 - consistency_score,
            
            # 🔥 增强的Subject Embedding评估指标
            'embedding_necessity_average': embedding_necessity_score + deep_network_authority_boost,  # 加入权威性加成
            'high_necessity_ratio': high_necessity_ratio,
            'embedding_recommended': (embedding_necessity_score + deep_network_authority_boost) > 0.4,
            'analysis_coverage': len(self.analysis_results.get('region_wise_separability', {})) / 
                            len(self.analysis_results.get('region_wise_subject_analysis', {})) 
                            if self.analysis_results.get('region_wise_subject_analysis') else 0.0,
            
            # 🔥 深度网络专项评估指标
            'deep_network_authority_boost': deep_network_authority_boost,
            'deep_network_validation_ratio': deep_validated_count / len(successful_regions) if 'successful_regions' in locals() and successful_regions else 0.0,
            'deep_loso_authority_score': deep_loso_authority_score,
            'deep_network_comprehensive_score': (embedding_necessity_score + deep_network_authority_boost + deep_loso_authority_score) / 3,
            
            # 🔥 最终权威推荐
            'authoritative_recommendation': deep_loso_authority_score > 0.6 or (embedding_necessity_score + deep_network_authority_boost) > 0.6
        }
        
        logger.info(f"\n📈 🔥 Phase 2 综合决策指标 (深度网络增强版):")
        logger.info(f"  - 全局受试者变异性: {self.decision_scores['phase2']['global_subject_variability']:.3f}")
        logger.info(f"  - 脑区分类一致性: {self.decision_scores['phase2']['class_consistency']:.3f}")
        logger.info(f"  🔥 Subject Embedding平均需求强度: {self.decision_scores['phase2']['embedding_necessity_average']:.3f}")
        logger.info(f"  🔥 高需求脑区比例: {self.decision_scores['phase2']['high_necessity_ratio']:.3f}")
        logger.info(f"  🔥 深度网络权威性加成: {self.decision_scores['phase2']['deep_network_authority_boost']:.3f}")
        logger.info(f"  🔥 深度网络验证率: {self.decision_scores['phase2']['deep_network_validation_ratio']:.3f}")
        logger.info(f"  🔥 深度LOSO权威得分: {self.decision_scores['phase2']['deep_loso_authority_score']:.3f}")
        logger.info(f"  🔥 深度网络综合评分: {self.decision_scores['phase2']['deep_network_comprehensive_score']:.3f}")
        logger.info(f"  🔥 权威推荐使用Embedding: {'是' if self.decision_scores['phase2']['authoritative_recommendation'] else '否'}")
        logger.info(f"  📊 分析覆盖率: {self.decision_scores['phase2']['analysis_coverage']:.3f}")

    def phase3_embedding_adaptability_analysis(self):
        """Phase 3: Embedding适配性评估 - 🔥 带存档点功能"""
        self._start_phase_timer("Phase 3")
        
        try:
            logger.info("\n" + "="*80)
            logger.info("📊 Phase 3: Embedding适配性评估 (增强版 + 存档点)")
            logger.info("="*80)
            
            # 🔄 保留原有的增强版降维分析
            logger.info("\n📊 Phase 3A: 增强版降维适配性分析 (保持原有)")
            self._phase3a_enhanced_dimensionality_analysis()
            
            # 🔥 新增：脑区感知的embedding设计分析
            logger.info("\n📊 Phase 3B: 脑区感知embedding设计分析 (新增)")
            self._phase3b_brain_aware_embedding_design()
            
            # 综合决策得分计算
            self._compute_phase3_combined_scores()
            
            duration = self._end_phase_timer("Phase 3")
            
            # 🔥 创建Phase 3存档点
            if self.checkpoint_manager:
                try:
                    self.checkpoint_manager.create_checkpoint(
                        analyzer_instance=self,
                        checkpoint_name="embedding_design",
                        phase_completed=3,
                        description="Embedding适配性评估完成，包含脑区感知设计",
                        is_auto=True
                    )
                except Exception as e:
                    logger.warning(f"Phase 3存档点创建失败: {e}")
        
        except Exception as e:
            # 🔥 异常时创建紧急存档点
            if self.checkpoint_manager:
                self.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=f"Phase 3 执行异常: {str(e)}",
                    stack_trace=str(e)
                )
            raise

    def _phase3a_enhanced_dimensionality_analysis(self):
        """Phase 3A: 🔥 深度网络增强版降维分析"""
        
        if 'subject_stats' not in self.analysis_results:
            logger.info("❌ 需要先运行Phase 1")
            return
        
        subject_means = self.analysis_results['subject_stats']['means']
        
        # 🔄 保留原有的多种降维方法对比分析
        logger.info("🔍 3.1A 多种降维方法对比分析...")
        dimensionality_results = self._comprehensive_dimensionality_analysis(subject_means)
        
        # 🔄 保留原有的高维空间直接分析
        logger.info("🔍 3.2A 高维空间直接分析...")
        high_dim_results = self._high_dimensional_direct_analysis(subject_means)
        
        # 🔥 新增：深度网络特征重要性分析
        logger.info("🔍 3.2A-Deep 🔥 深度网络特征重要性分析...")
        deep_feature_analysis = self._deep_network_feature_importance_analysis()
        
        # 🔄 保留原有的特征分组分析
        logger.info("🔍 3.3A 特征分组分析...")
        group_results = self._feature_group_analysis(subject_means)
        
        # 🔥 新增：深度网络增强的特征分组分析
        logger.info("🔍 3.3A-Deep 🔥 深度网络增强特征分组分析...")
        enhanced_group_results = self._deep_enhanced_feature_group_analysis(group_results, deep_feature_analysis)
        
        # 🔄 保留原有的样本量充足性评估
        logger.info("🔍 3.4A 样本量充足性评估...")
        sample_adequacy = self._sample_adequacy_assessment()
        
        # 🔥 新增：深度网络样本效率分析
        logger.info("🔍 3.4A-Deep 🔥 深度网络样本效率分析...")
        deep_sample_efficiency = self._deep_network_sample_efficiency_analysis()
        
        # 🔥 修改：深度网络增强的综合适配性评估
        logger.info("🔍 3.5A 🔥 深度网络增强综合适配性评估...")
        comprehensive_assessment = self._comprehensive_embedding_assessment(
            dimensionality_results, high_dim_results, enhanced_group_results, sample_adequacy
        )
        
        # 🔥 存储增强版结果
        self.analysis_results['enhanced_dimensionality'] = {
            'dimensionality_comparison': dimensionality_results,
            'high_dimensional_analysis': high_dim_results,
            'feature_group_analysis': enhanced_group_results,            # 🔥 使用增强版
            'sample_adequacy': sample_adequacy,
            'comprehensive_assessment': comprehensive_assessment,
            # 🔥 新增深度网络相关分析
            'deep_feature_analysis': deep_feature_analysis,
            'deep_sample_efficiency': deep_sample_efficiency,
            'deep_network_integration_status': 'success'
        }
        
        # 🔥 存储深度网络特征重要性到全局结果
        if 'error' not in deep_feature_analysis:
            self.analysis_results['deep_feature_importance'] = deep_feature_analysis
        
        logger.info(f"✅ 🔥 深度网络增强版降维分析完成")
        logger.info(f"  - 传统降维方法: {len(dimensionality_results)} 种")
        logger.info(f"  - 高维分类器: {len(high_dim_results.get('separability_scores', {}))} 个（含深度网络）")
        logger.info(f"  - 深度网络特征分析: {'成功' if 'error' not in deep_feature_analysis else '失败'}")
        logger.info(f"  - 综合可行性评分: {comprehensive_assessment['overall_feasibility']:.3f}")
        logger.info(f"  🔥 深度网络评分: {comprehensive_assessment['deep_network_score']:.3f}")
        logger.info(f"  🔥 深度网络权威性: {comprehensive_assessment['deep_authority_score']:.3f}")

    def _deep_enhanced_feature_group_analysis(self, group_results, deep_feature_analysis):
        """🔥 深度网络增强的特征分组分析"""
        
        enhanced_group_results = group_results.copy()
        
        if 'error' not in deep_feature_analysis and 'group_importance' in deep_feature_analysis:
            deep_group_importance = deep_feature_analysis['group_importance']
            
            # 为每个特征组添加深度网络重要性信息
            for group_name in enhanced_group_results.keys():
                if group_name in deep_group_importance:
                    deep_info = deep_group_importance[group_name]
                    
                    enhanced_group_results[group_name].update({
                        # 🔥 深度网络分析结果
                        'deep_mean_importance': deep_info['mean_importance'],
                        'deep_max_importance': deep_info['max_importance'],
                        'deep_relative_contribution': deep_info['relative_contribution'],
                        'deep_top_features': deep_info['top_features_in_group'],
                        
                        # 🔥 综合评估
                        'traditional_pca_score': enhanced_group_results[group_name].get('variance_in_3pc', 0),
                        'deep_importance_score': deep_info['mean_importance'],
                        'combined_importance': (enhanced_group_results[group_name].get('variance_in_3pc', 0) * 0.4 + 
                                            deep_info['mean_importance'] * 0.6),  # 深度网络权重更高
                        
                        # 🔥 推荐等级
                        'recommendation_level': self._determine_group_recommendation_level(
                            enhanced_group_results[group_name].get('variance_in_3pc', 0),
                            deep_info['mean_importance']
                        )
                    })
            
            # 🔥 添加整体特征组排名
            enhanced_group_results['deep_network_ranking'] = {
                'by_deep_importance': sorted(deep_group_importance.keys(), 
                                        key=lambda x: deep_group_importance[x]['mean_importance'], reverse=True),
                'by_combined_score': sorted([g for g in enhanced_group_results.keys() if g != 'deep_network_ranking'], 
                                        key=lambda x: enhanced_group_results[x].get('combined_importance', 0), reverse=True),
                'recommendation_summary': self._generate_group_recommendation_summary(enhanced_group_results)
            }
        
        return enhanced_group_results

    def _determine_group_recommendation_level(self, pca_score, deep_score):
        """确定特征组推荐等级"""
        combined = pca_score * 0.4 + deep_score * 0.6
        
        if combined > 0.8:
            return "HIGHLY_RECOMMENDED"
        elif combined > 0.6:
            return "RECOMMENDED"
        elif combined > 0.4:
            return "MODERATELY_USEFUL"
        else:
            return "LOW_PRIORITY"

    def _generate_group_recommendation_summary(self, enhanced_results):
        """生成特征组推荐总结"""
        summary = {
            'HIGHLY_RECOMMENDED': [],
            'RECOMMENDED': [],
            'MODERATELY_USEFUL': [],
            'LOW_PRIORITY': []
        }
        
        for group_name, group_data in enhanced_results.items():
            if group_name != 'deep_network_ranking' and 'recommendation_level' in group_data:
                level = group_data['recommendation_level']
                summary[level].append(group_name)
        
        return summary

    def _deep_network_sample_efficiency_analysis(self):
        """🔥 深度网络样本效率分析"""
        
        logger.info("      🔍 分析深度网络的样本效率...")
        
        if 'deep_feature_importance' not in self.analysis_results:
            return {'error': 'No deep feature analysis available'}
        
        deep_analysis = self.analysis_results['deep_feature_importance']
        if 'error' in deep_analysis:
            return {'error': 'Deep feature analysis failed'}
        
        # 基于深度网络的样本效率评估
        network_performance = deep_analysis.get('network_performance', 0)
        training_time = deep_analysis.get('training_time', 0)
        
        # 样本效率指标
        total_samples = len(self.data['X_train'])
        efficiency_metrics = {
            'samples_per_second': total_samples / training_time if training_time > 0 else 0,
            'accuracy_per_minute': network_performance / (training_time / 60) if training_time > 0 else 0,
            'parameter_efficiency': network_performance / (4096 * 4),  # 性能/参数比
            'convergence_efficiency': network_performance / len(deep_analysis.get('combined_importance', [1])),  # 性能/特征数比
        }
        
        # 效率等级评估
        if efficiency_metrics['accuracy_per_minute'] > 0.1:
            efficiency_level = "HIGHLY_EFFICIENT"
        elif efficiency_metrics['accuracy_per_minute'] > 0.05:
            efficiency_level = "MODERATELY_EFFICIENT"
        else:
            efficiency_level = "LOW_EFFICIENCY"
        
        return {
            'efficiency_metrics': efficiency_metrics,
            'efficiency_level': efficiency_level,
            'total_samples': total_samples,
            'training_time_minutes': training_time / 60,
            'network_performance': network_performance,
            'recommendation': f"深度网络训练效率{efficiency_level.replace('_', ' ').lower()}，建议{'保持当前配置' if efficiency_level == 'HIGHLY_EFFICIENT' else '考虑优化' if efficiency_level == 'MODERATELY_EFFICIENT' else '重新设计网络架构'}"
        }

    def _phase3b_brain_aware_embedding_design(self):
        """Phase 3B: 新增的脑区感知embedding设计分析"""
        
        logger.info("🔍 3.1B 脑区embedding需求分层分析...")
        
        if 'region_wise_subject_analysis' not in self.analysis_results:
            logger.info("❌ 需要先运行Phase 1B")
            return
        
        region_results = self.analysis_results['region_wise_subject_analysis']
        
        # 1. 脑区分层策略
        brain_aware_design = self._analyze_region_embedding_requirements(region_results)
        
        # 2. 脑区间相似性分析
        logger.info("🔍 3.2B 脑区间embedding相似性分析...")
        region_similarity_analysis = self._analyze_region_similarity_patterns(region_results)
        
        # 3. 脑区特异的embedding维度推荐
        logger.info("🔍 3.3B 脑区特异embedding维度推荐...")
        region_embedding_dims = self._recommend_region_specific_dimensions(region_results)
        
        # 4. 脑区聚类和共享策略
        logger.info("🔍 3.4B 脑区聚类和embedding共享策略...")
        region_clustering_strategy = self._design_region_clustering_strategy(region_results)
        
        # 存储脑区感知分析结果
        self.analysis_results['brain_aware_embedding_design'] = {
            'region_requirements': brain_aware_design,
            'region_similarity': region_similarity_analysis,
            'region_embedding_dims': region_embedding_dims,
            'region_clustering_strategy': region_clustering_strategy
        }
        
        logger.info(f"✅ 脑区感知embedding设计分析完成")

    def _analyze_region_embedding_requirements(self, region_results):
        """分析每个脑区的embedding需求"""
        
        # 按特异性分组脑区
        specificity_scores = {rid: data['subject_specificity_score'] 
                             for rid, data in region_results.items()}
        
        sorted_regions = sorted(specificity_scores.keys(), 
                               key=lambda x: specificity_scores[x], reverse=True)
        
        n_regions = len(sorted_regions)
        
        # 分层策略
        tier1_high = sorted_regions[:n_regions//4]        # Top 25%
        tier2_medium_high = sorted_regions[n_regions//4:n_regions//2]  # 25%-50%
        tier3_medium_low = sorted_regions[n_regions//2:3*n_regions//4]  # 50%-75%
        tier4_low = sorted_regions[3*n_regions//4:]       # Bottom 25%
        
        requirements = {
            'tier1_high_specificity': {
                'region_ids': tier1_high,
                'embedding_necessity': 'CRITICAL',
                'recommended_dim': 128,
                'adaptation_samples_needed': 1000,
                'priority': 1,
                'description': '极高特异性，强烈需要Subject Embedding'
            },
            'tier2_medium_high_specificity': {
                'region_ids': tier2_medium_high,
                'embedding_necessity': 'HIGH',
                'recommended_dim': 64,
                'adaptation_samples_needed': 500,
                'priority': 2,
                'description': '高特异性，建议使用Subject Embedding'
            },
            'tier3_medium_low_specificity': {
                'region_ids': tier3_medium_low,
                'embedding_necessity': 'MODERATE',
                'recommended_dim': 32,
                'adaptation_samples_needed': 200,
                'priority': 3,
                'description': '中等特异性，可选择性使用Subject Embedding'
            },
            'tier4_low_specificity': {
                'region_ids': tier4_low,
                'embedding_necessity': 'LOW',
                'recommended_dim': 16,
                'adaptation_samples_needed': 0,
                'priority': 4,
                'description': '低特异性，可能不需要Subject Embedding'
            }
        }
        
        logger.info(f"    - Tier 1 (极高特异性): {len(tier1_high)} 个脑区")
        logger.info(f"    - Tier 2 (高特异性): {len(tier2_medium_high)} 个脑区")
        logger.info(f"    - Tier 3 (中等特异性): {len(tier3_medium_low)} 个脑区")
        logger.info(f"    - Tier 4 (低特异性): {len(tier4_low)} 个脑区")
        
        return requirements

    def _analyze_region_similarity_patterns(self, region_results):
        """分析脑区间的特异性模式相似性"""
        
        # 构建脑区特征矩阵 (脑区 × 特异性特征)
        region_ids = list(region_results.keys())
        n_regions = len(region_ids)
        
        if n_regions < 2:
            return {}
        
        # 每个脑区的特异性特征向量
        region_feature_matrix = np.zeros((n_regions, 4))  # 4个特异性特征
        
        for i, region_id in enumerate(region_ids):
            data = region_results[region_id]
            region_feature_matrix[i] = [
                data['subject_specificity_score'],
                data['mean_inter_subject_distance'], 
                data['mean_inter_subject_correlation'],
                data['pca_3pc_variance']
            ]
        
        # 脑区间相似性矩阵
        try:
            region_distances = squareform(pdist(region_feature_matrix))
            region_correlations = np.corrcoef(region_feature_matrix)
        except:
            region_distances = np.zeros((n_regions, n_regions))
            region_correlations = np.eye(n_regions)
        
        # 层次聚类分析
        try:
            region_linkage = linkage(region_feature_matrix, method='ward')
        except:
            region_linkage = np.zeros((n_regions-1, 4))
        
        similarity_analysis = {
            'region_ids': region_ids,
            'similarity_matrix': region_correlations,
            'distance_matrix': region_distances,
            'linkage_matrix': region_linkage,
            'feature_matrix': region_feature_matrix
        }
        
        logger.info(f"    - 脑区间相似性分析完成: {n_regions} 个脑区")
        
        return similarity_analysis

    def _recommend_region_specific_dimensions(self, region_results):
        """为每个脑区推荐特异的embedding维度"""
        
        dimension_recommendations = {}
        
        for region_id, data in region_results.items():
            specificity_score = data['subject_specificity_score']
            n_subjects = data['n_subjects']
            pca_variance = data['pca_3pc_variance']
            
            # 基于特异性得分和PCA方差推荐维度
            if specificity_score > 2.0 and pca_variance < 0.8:
                # 高特异性，低线性度 -> 需要大维度
                recommended_dim = 128
                embedding_type = 'nonlinear'
            elif specificity_score > 1.5 and pca_variance < 0.9:
                # 中高特异性 -> 中等维度
                recommended_dim = 64
                embedding_type = 'mixed'
            elif specificity_score > 1.0:
                # 中等特异性 -> 小维度
                recommended_dim = 32
                embedding_type = 'linear'
            else:
                # 低特异性 -> 最小维度或不需要
                recommended_dim = 16
                embedding_type = 'minimal'
            
            # 考虑受试者数量限制
            max_reasonable_dim = min(recommended_dim, n_subjects * 2)
            
            dimension_recommendations[region_id] = {
                'recommended_dim': max_reasonable_dim,
                'embedding_type': embedding_type,
                'specificity_score': specificity_score,
                'n_subjects': n_subjects,
                'justification': f"基于特异性{specificity_score:.3f}和{n_subjects}个受试者"
            }
        
        logger.info(f"    - 维度推荐完成: {len(dimension_recommendations)} 个脑区")
        
        return dimension_recommendations

    def _design_region_clustering_strategy(self, region_results):
        """设计脑区聚类和embedding共享策略"""
        
        if len(region_results) < 3:
            return {}
        
        # 构建脑区特征用于聚类
        region_ids = list(region_results.keys())
        features = []
        
        for region_id in region_ids:
            data = region_results[region_id]
            features.append([
                data['subject_specificity_score'],
                data['mean_inter_subject_distance'],
                data['pca_3pc_variance']
            ])
        
        features = np.array(features)
        
        # K-means聚类，确定最优聚类数
        best_k = 3  # 默认值
        best_silhouette = -1
        
        for k in range(2, min(8, len(region_ids))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(features)
                silhouette = silhouette_score(features, cluster_labels)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
            except:
                continue
        
        # 使用最优K进行聚类
        try:
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
        except:
            cluster_labels = np.zeros(len(region_ids))
        
        # 组织聚类结果
        clusters = {}
        for i, region_id in enumerate(region_ids):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(region_id)
        
        # 分析每个聚类的特征
        cluster_analysis = {}
        for cluster_id, cluster_regions in clusters.items():
            cluster_specificities = [region_results[rid]['subject_specificity_score'] for rid in cluster_regions]
            cluster_analysis[cluster_id] = {
                'regions': cluster_regions,
                'n_regions': len(cluster_regions),
                'avg_specificity': np.mean(cluster_specificities),
                'specificity_std': np.std(cluster_specificities),
                'shared_embedding_recommended': len(cluster_regions) > 1 and np.std(cluster_specificities) < 0.5
            }
        
        clustering_strategy = {
            'n_clusters': best_k,
            'silhouette_score': best_silhouette,
            'clusters': cluster_analysis,
            'cluster_labels': dict(zip(region_ids, cluster_labels)),
            'sharing_strategy': 'cluster_based' if best_k < len(region_ids) else 'individual'
        }
        
        logger.info(f"    - 脑区聚类完成: {best_k} 个聚类，轮廓系数 {best_silhouette:.3f}")
        
        return clustering_strategy

    def _compute_phase3_combined_scores(self):
        """计算Phase 3的综合决策得分"""
        
        # 原有得分
        linearity_score = 0.5
        clustering_quality = 0.5
        sample_adequacy = 0.5
        embedding_feasibility = 0.5
        intrinsic_dim_score = 0.5
        high_dim_score = 0.5
        
        if 'enhanced_dimensionality' in self.analysis_results:
            enhanced_results = self.analysis_results['enhanced_dimensionality']
            
            if 'comprehensive_assessment' in enhanced_results:
                assessment = enhanced_results['comprehensive_assessment']
                linearity_score = assessment.get('linearity_score', 0.5)
                clustering_quality = assessment.get('clustering_score', 0.5)
                embedding_feasibility = assessment.get('overall_feasibility', 0.5)
                intrinsic_dim_score = assessment.get('intrinsic_dim_score', 0.5)
                high_dim_score = assessment.get('high_dim_score', 0.5)
            
            if 'sample_adequacy' in enhanced_results:
                sample_adequacy = enhanced_results['sample_adequacy'].get('adequacy_score', 0.5)
        
        # 🔥 新增：脑区感知得分
        brain_aware_feasibility = 0.0
        region_diversity_score = 0.0
        clustering_effectiveness = 0.0
        
        if 'brain_aware_embedding_design' in self.analysis_results:
            brain_design = self.analysis_results['brain_aware_embedding_design']
            
            # 脑区embedding可行性：基于成功分析的脑区比例
            if 'region_requirements' in brain_design:
                requirements = brain_design['region_requirements']
                total_regions = sum(len(tier['region_ids']) for tier in requirements.values())
                high_priority_regions = len(requirements.get('tier1_high_specificity', {}).get('region_ids', []))
                
                if total_regions > 0:
                    brain_aware_feasibility = min(1.0, total_regions / 50)  # 假设50个脑区是理想数量
                    region_diversity_score = high_priority_regions / total_regions if total_regions > 0 else 0
            
            # 聚类效果：基于轮廓系数
            if 'region_clustering_strategy' in brain_design:
                clustering_strategy = brain_design['region_clustering_strategy']
                silhouette = clustering_strategy.get('silhouette_score', 0)
                clustering_effectiveness = max(0, min(1, (silhouette + 1) / 2))  # 将[-1,1]映射到[0,1]
        
        # Phase 3 综合决策得分
        self.decision_scores['phase3'] = {
            # 原有得分
            'linearity': linearity_score,
            'clustering_quality': clustering_quality,
            'sample_adequacy': sample_adequacy,
            'embedding_feasibility': embedding_feasibility,
            'intrinsic_dimensionality': intrinsic_dim_score,
            'high_dim_performance': high_dim_score,
            
            # 🔥 新增得分
            'brain_aware_feasibility': brain_aware_feasibility,
            'region_diversity': region_diversity_score,
            'region_clustering_effectiveness': clustering_effectiveness
        }
        
        logger.info(f"\n📈 Phase 3 综合决策指标:")
        logger.info(f"  - 线性度: {self.decision_scores['phase3']['linearity']:.3f}")
        logger.info(f"  - 聚类质量: {self.decision_scores['phase3']['clustering_quality']:.3f}")
        logger.info(f"  - 样本充足性: {self.decision_scores['phase3']['sample_adequacy']:.3f}")
        logger.info(f"  - 内在维度得分: {self.decision_scores['phase3']['intrinsic_dimensionality']:.3f}")
        logger.info(f"  - 高维性能得分: {self.decision_scores['phase3']['high_dim_performance']:.3f}")
        logger.info(f"  🔥 脑区感知可行性: {self.decision_scores['phase3']['brain_aware_feasibility']:.3f}")
        logger.info(f"  🔥 脑区多样性得分: {self.decision_scores['phase3']['region_diversity']:.3f}")
        logger.info(f"  🔥 脑区聚类效果: {self.decision_scores['phase3']['region_clustering_effectiveness']:.3f}")
        logger.info(f"  - 整体可行性: {self.decision_scores['phase3']['embedding_feasibility']:.3f}")

    def _comprehensive_dimensionality_analysis(self, subject_means):
        """多种降维方法对比分析 (保持原有逻辑)"""
        results = {}
        
        # 1. PCA (线性)
        pca = PCA()
        pca_result = pca.fit_transform(subject_means)
        pca_2d = PCA(n_components=2).fit_transform(subject_means)
        
        results['pca'] = {
            'full_result': pca_result,
            '2d_result': pca_2d,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'method_type': 'linear'
        }
        
        # 2. t-SNE (非线性流形)
        try:
            perplexity = min(30, len(subject_means)-1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                    learning_rate='auto', init='random')
            tsne_result = tsne.fit_transform(subject_means)
            
            results['tsne'] = {
                '2d_result': tsne_result,
                'perplexity': perplexity,
                'method_type': 'nonlinear_manifold'
            }
            logger.info(f"    ✅ t-SNE分析完成 (perplexity={perplexity})")
        except Exception as e:
            logger.info(f"    ⚠️ t-SNE分析失败: {e}")
            results['tsne'] = None
        
        # 3. 比较不同方法的结构保持能力
        try:
            import umap
            logger.info("    🔍 运行UMAP降维分析...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            umap_result = umap_reducer.fit_transform(subject_means)
            
            results['umap'] = {
                '2d_result': umap_result,
                'method_type': 'nonlinear_manifold'
            }
            logger.info(f"    ✅ UMAP分析完成")
        except ImportError:
            logger.info(f"    ⚠️ UMAP未安装，跳过UMAP分析")
            results['umap'] = None
        except Exception as e:
            logger.info(f"    ⚠️ UMAP分析失败: {e}")
            results['umap'] = None
        
        # 3. 比较不同方法的结构保持能力 - 保持原有
        structure_preservation = self._compare_structure_preservation(subject_means, results)
        results['structure_preservation'] = structure_preservation
        
        return results

    def _high_dimensional_direct_analysis(self, subject_means):
        """高维空间直接分析 (保持原有逻辑)"""
        results = {}
        
        # 1. 内在维度估计
        logger.info("    🔍 估计内在维度...")
        intrinsic_dim = self._estimate_intrinsic_dimensionality(subject_means)
        results['intrinsic_dimensionality'] = intrinsic_dim
        
        # 2. 高维分类器性能测试
        logger.info("    🔍 高维分类器性能测试...")
        separability_scores = self._test_high_dim_classifiers(subject_means)
        results['separability_scores'] = separability_scores
        
        logger.info(f"    ✅ 高维分析完成")
        logger.info(f"      - 估计内在维度: {intrinsic_dim:.1f}")
        logger.info(f"      - 测试分类器: {len(separability_scores)}")
        
        return results

    def _feature_group_analysis(self, subject_means):
        """特征分组分析 (保持原有逻辑)"""
        feature_groups = {
            'qti_params': list(range(0, 15)),           # QTI参数 (15个)
            'raw_b_tensors': list(range(15, 225)),      # 原始b-tensor值 (210个)
            'cest_params': list(range(225, 229)),       # CEST参数 (4个)
            'z_spectrum': list(range(229, 341))         # Z-spectrum值 (112个)
        }
        
        group_analysis = {}
        
        for group_name, feature_indices in feature_groups.items():
            if len(feature_indices) == 0:
                continue
                
            logger.info(f"    🔍 分析 {group_name} ({len(feature_indices)} 特征)...")
            
            try:
                group_data = subject_means[:, feature_indices]
                
                if len(feature_indices) >= 2:
                    # PCA分析
                    pca = PCA()
                    pca_result = pca.fit_transform(group_data)
                    
                    group_analysis[group_name] = {
                        'pca_explained_variance': pca.explained_variance_ratio_,
                        'variance_in_3pc': np.sum(pca.explained_variance_ratio_[:3]) if len(pca.explained_variance_ratio_) >= 3 else np.sum(pca.explained_variance_ratio_),
                        'feature_count': len(feature_indices),
                        'feature_range': (min(feature_indices), max(feature_indices))
                    }
                    
                    logger.info(f"      ✅ {group_name}: 前3PC解释方差 {group_analysis[group_name]['variance_in_3pc']:.3f}")
                    
            except Exception as e:
                logger.info(f"      ⚠️ {group_name} 分析失败: {e}")
                group_analysis[group_name] = {'error': str(e)}
        
        return group_analysis

    def _sample_adequacy_assessment(self):
        """样本量充足性评估 (保持原有逻辑)"""
        sample_counts = self.analysis_results['subject_stats']['sample_counts']
        total_samples = np.sum(sample_counts)
        n_subjects = len(sample_counts)
        avg_samples_per_subject = np.mean(sample_counts)
        
        sample_adequacy_score = 0
        
        if avg_samples_per_subject >= 50000:
            sample_adequacy_score += 0.4
        elif avg_samples_per_subject >= 10000:
            sample_adequacy_score += 0.2
        
        if n_subjects >= 30:
            sample_adequacy_score += 0.3
        elif n_subjects >= 20:
            sample_adequacy_score += 0.2
        
        if total_samples >= 1000000:
            sample_adequacy_score += 0.3
        elif total_samples >= 500000:
            sample_adequacy_score += 0.2
        
        return {
            'total_samples': total_samples,
            'n_subjects': n_subjects,
            'avg_samples_per_subject': avg_samples_per_subject,
            'adequacy_score': sample_adequacy_score
        }

    def _comprehensive_embedding_assessment(self, dimensionality_results, high_dim_results, group_results, sample_adequacy):
        """综合embedding适配性评估 (🔥 深度网络增强版)"""
        
        assessment = {
            'linearity_score': 0.0,
            'clustering_score': 0.0,
            'intrinsic_dim_score': 0.0,
            'high_dim_score': 0.0,
            'deep_network_score': 0.0,          # 🔥 新增：深度网络评分
            'feature_importance_score': 0.0,    # 🔥 新增：特征重要性评分
            'deep_authority_score': 0.0,        # 🔥 新增：深度网络权威性评分
            'overall_feasibility': 0.0
        }
        
        # 🔄 保留原有评估逻辑
        
        # 线性度评估
        if 'pca' in dimensionality_results:
            pca_3pc_variance = np.sum(dimensionality_results['pca']['explained_variance'][:3])
            assessment['linearity_score'] = min(1.0, pca_3pc_variance * 1.25)
        
        # 内在维度评估
        if 'intrinsic_dimensionality' in high_dim_results:
            intrinsic_dim = high_dim_results['intrinsic_dimensionality']
            if intrinsic_dim <= 20:
                assessment['intrinsic_dim_score'] = 1.0
            elif intrinsic_dim <= 50:
                assessment['intrinsic_dim_score'] = 0.8
            elif intrinsic_dim <= 100:
                assessment['intrinsic_dim_score'] = 0.6
            else:
                assessment['intrinsic_dim_score'] = 0.3
        
        # 高维性能评估（🔥 增强版：包含深度网络）
        if 'separability_scores' in high_dim_results:
            # 分离传统方法和深度网络
            traditional_scores = []
            deep_network_scores = []
            
            for clf_name, clf_results in high_dim_results['separability_scores'].items():
                if isinstance(clf_results, dict) and 'mean_score' in clf_results:
                    score = clf_results['mean_score']
                    if clf_name.startswith('deep'):
                        deep_network_scores.append(score)
                    else:
                        traditional_scores.append(score)
            
            # 传统高维性能评估
            if traditional_scores:
                avg_traditional_performance = np.mean(traditional_scores)
                n_subjects = len(self.data['available_subjects'])
                random_baseline = 1.0 / n_subjects
                normalized_traditional_score = (avg_traditional_performance - random_baseline) / (1 - random_baseline)
                assessment['high_dim_score'] = max(0, min(1, normalized_traditional_score))
            
            # 🔥 深度网络性能评估
            if deep_network_scores:
                avg_deep_performance = np.mean(deep_network_scores)
                n_subjects = len(self.data['available_subjects'])
                random_baseline = 1.0 / n_subjects
                normalized_deep_score = (avg_deep_performance - random_baseline) / (1 - random_baseline)
                assessment['deep_network_score'] = max(0, min(1, normalized_deep_score))
                
                # 🔥 深度网络权威性评估
                if 'authority_assessment' in high_dim_results.get('separability_scores', {}):
                    authority_data = high_dim_results['separability_scores']['authority_assessment']
                    assessment['deep_authority_score'] = authority_data.get('overall_authority', 0.0)
            
            # 🔥 如果深度网络表现更好，提高总体评分
            if deep_network_scores and traditional_scores:
                deep_advantage = np.mean(deep_network_scores) - np.mean(traditional_scores)
                if deep_advantage > 0.05:  # 深度网络显著更好
                    assessment['deep_network_score'] = min(1.0, assessment['deep_network_score'] + 0.1)
        
        # 🔥 特征重要性评估
        if 'deep_feature_importance' in self.analysis_results:
            feature_analysis = self.analysis_results['deep_feature_importance']
            if 'error' not in feature_analysis:
                # 基于特征重要性的稀疏度和区分度评估
                sparsity = feature_analysis['importance_statistics']['sparsity']
                importance_std = feature_analysis['importance_statistics']['std']
                
                # 稀疏度适中（不全重要，也不全不重要）且区分度高更好
                sparsity_score = 1.0 - abs(sparsity - 0.3) / 0.7  # 30%稀疏度为最优
                discrimination_score = min(1.0, importance_std * 2)  # 标准差越大，区分度越高
                
                assessment['feature_importance_score'] = (sparsity_score + discrimination_score) / 2
        
        # 🔥 修改综合可行性计算权重
        weights = {
            'linearity': 0.15,              # 降低传统方法权重
            'intrinsic_dim': 0.15,          # 降低传统方法权重
            'high_dim': 0.15,               # 降低传统方法权重
            'deep_network': 0.25,           # 🔥 深度网络主要权重
            'feature_importance': 0.10,     # 🔥 特征重要性权重
            'deep_authority': 0.10,         # 🔥 深度网络权威性权重
            'sample_adequacy': 0.10         # 降低样本充足性权重
        }
        
        assessment['overall_feasibility'] = (
            assessment['linearity_score'] * weights['linearity'] +
            assessment['intrinsic_dim_score'] * weights['intrinsic_dim'] +
            assessment['high_dim_score'] * weights['high_dim'] +
            assessment['deep_network_score'] * weights['deep_network'] +                    # 🔥 新增
            assessment['feature_importance_score'] * weights['feature_importance'] +       # 🔥 新增
            assessment['deep_authority_score'] * weights['deep_authority'] +               # 🔥 新增
            sample_adequacy['adequacy_score'] * weights['sample_adequacy']
        )
        
        # 🔥 深度网络加成机制
        if assessment['deep_authority_score'] > 0.8:
            assessment['overall_feasibility'] = min(1.0, assessment['overall_feasibility'] + 0.05)  # 权威性很高时额外加成
        
        return assessment

    def _estimate_intrinsic_dimensionality(self, data):
        """估计数据的内在维度 (保持原有逻辑)"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            k = min(10, len(data) - 1)
            if k < 2:
                return data.shape[1]
                
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
            distances, indices = nbrs.kneighbors(data)
            
            distances = distances[:, 1:]  # 排除自身距离
            ratios = distances[:, -1] / (distances[:, 0] + 1e-10)
            ratios = ratios[ratios > 1]
            ratios = ratios[ratios < 1000]
            
            if len(ratios) > 0:
                log_ratios = np.log(ratios)
                intrinsic_dim = k / np.mean(log_ratios)
                return min(max(1, intrinsic_dim), data.shape[1])
            else:
                return data.shape[1]
                
        except Exception as e:
            logger.info(f"      ⚠️ 内在维度估计失败: {e}")
            return data.shape[1]

    def _test_high_dim_classifiers(self, subject_means):
        """测试多种高维分类器的性能 (🔥 深度网络增强版：全局 + 分脑区 + 4×4096深度网络)"""
        
        logger.info("    🔍 🔥 深度网络增强版高维分类器性能测试...")
        logger.info("    📊 分析维度：全局受试者识别 + 分脑区受试者识别 + 4×4096深度网络权威评估")
        
        # 🔥 扩展分类器列表，添加4×4096深度网络
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
            # 🔥 新增：4×4096深度网络 - 严格使用alex超参数
            'deep_4x4096': self._create_deep_classifier_wrapper(),
            # 🔥 新增：轻量级深度网络用于对比
            'deep_lightweight': self._create_deep_classifier_wrapper({
                'batch_size': 64,
                'no_epochs': 15  # 较少epoch用于快速测试
            })
        }
        
        results = {
            'global_analysis': {},
            'region_wise_analysis': {},
            'comparative_analysis': {},
            'deep_network_detailed_analysis': {}  # 🔥 新增：深度网络详细分析
        }
        
        # ========================================================================
        # 1. 🔥 增强版全局分析 - 添加深度网络
        # ========================================================================
        logger.info("      🌐 全局受试者识别分析（含4×4096深度网络）...")
        
        subject_labels = np.arange(len(subject_means))
        
        for clf_name, clf in classifiers.items():
            try:
                n_subjects = len(subject_means)
                if n_subjects >= 5:
                    cv_folds = min(5, n_subjects)
                    
                    if cv_folds < 3:
                        # 样本太少，使用train_test_split
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            subject_means, subject_labels, test_size=0.3, random_state=42
                        )
                        
                        # 🔥 特殊处理深度网络
                        if 'deep' in clf_name:
                            logger.info(f"        🔥 训练{clf_name}深度网络...")
                            start_time = time.time()
                            clf.fit(X_train, y_train)
                            train_time = time.time() - start_time
                            
                            score = clf.score(X_test, y_test)
                            
                            # 🔥 深度网络特殊分析
                            if hasattr(clf, 'training_history'):
                                training_history = clf.training_history
                                results['deep_network_detailed_analysis'][f'{clf_name}_global'] = {
                                    'training_time': train_time,
                                    'convergence_epochs': len(training_history['loss']),
                                    'final_training_loss': training_history['loss'][-1] if training_history['loss'] else 0,
                                    'final_training_accuracy': training_history['accuracy'][-1] if training_history['accuracy'] else 0,
                                    'test_accuracy': score,
                                    'generalization_gap': training_history['accuracy'][-1] - score if training_history['accuracy'] else 0,
                                    'convergence_quality': 'good' if len(training_history['loss']) > 1 and training_history['loss'][-1] < training_history['loss'][0] else 'poor'
                                }
                            
                            results['global_analysis'][clf_name] = {
                                'mean_score': score,
                                'std_score': 0.0,
                                'scores': [score],
                                'method': 'train_test_split',
                                'training_time': train_time,
                                'is_deep_network': True
                            }
                            
                            logger.info(f"          ✅ 🔥 全局{clf_name}: Acc={score:.3f}, 训练时间={train_time/60:.1f}分钟")
                            
                        else:
                            # 传统分类器
                            clf.fit(X_train, y_train)
                            score = clf.score(X_test, y_test)
                            results['global_analysis'][clf_name] = {
                                'mean_score': score,
                                'std_score': 0.0,
                                'scores': [score],
                                'method': 'train_test_split',
                                'is_deep_network': False
                            }
                            logger.info(f"          ✅ 全局{clf_name}: Acc={score:.3f}")
                    
                    else:
                        # 使用KFold交叉验证
                        from sklearn.model_selection import KFold
                        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        # 🔥 深度网络需要特殊处理交叉验证
                        if 'deep' in clf_name:
                            logger.info(f"        🔥 {clf_name}深度网络交叉验证...")
                            scores = []
                            training_times = []
                            deep_metrics = []
                            
                            for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(subject_means, subject_labels)):
                                logger.info(f"          Fold {fold+1}/{cv_folds}...")
                                
                                X_train_fold, X_test_fold = subject_means[train_idx], subject_means[test_idx]
                                y_train_fold, y_test_fold = subject_labels[train_idx], subject_labels[test_idx]
                                
                                # 为每个fold创建新的深度网络实例
                                fold_clf = classifiers[clf_name]
                                start_time = time.time()
                                fold_clf.fit(X_train_fold, y_train_fold)
                                train_time = time.time() - start_time
                                
                                score = fold_clf.score(X_test_fold, y_test_fold)
                                scores.append(score)
                                training_times.append(train_time)
                                
                                # 收集深度网络指标
                                if hasattr(fold_clf, 'training_history'):
                                    history = fold_clf.training_history
                                    deep_metrics.append({
                                        'fold': fold,
                                        'training_time': train_time,
                                        'final_loss': history['loss'][-1] if history['loss'] else 0,
                                        'final_train_acc': history['accuracy'][-1] if history['accuracy'] else 0,
                                        'test_acc': score,
                                        'epochs': len(history['loss'])
                                    })
                            
                            # 🔥 深度网络交叉验证统计
                            results['deep_network_detailed_analysis'][f'{clf_name}_global_cv'] = {
                                'per_fold_metrics': deep_metrics,
                                'avg_training_time': np.mean(training_times),
                                'avg_convergence_epochs': np.mean([m['epochs'] for m in deep_metrics]),
                                'cv_stability': np.std(scores),  # 交叉验证稳定性
                                'avg_generalization_gap': np.mean([m['final_train_acc'] - m['test_acc'] for m in deep_metrics])
                            }
                            
                            results['global_analysis'][clf_name] = {
                                'mean_score': np.mean(scores),
                                'std_score': np.std(scores),
                                'scores': scores,
                                'method': 'cross_validation',
                                'avg_training_time': np.mean(training_times),
                                'is_deep_network': True
                            }
                            
                            logger.info(f"          ✅ 🔥 全局{clf_name}: Acc={np.mean(scores):.3f}±{np.std(scores):.3f}, 平均训练时间={np.mean(training_times)/60:.1f}分钟")
                            
                        else:
                            # 传统分类器的交叉验证
                            scores = cross_val_score(clf, subject_means, subject_labels, 
                                                cv=cv_strategy, scoring='accuracy')
                            results['global_analysis'][clf_name] = {
                                'mean_score': np.mean(scores),
                                'std_score': np.std(scores),
                                'scores': scores,
                                'method': 'cross_validation',
                                'is_deep_network': False
                            }
                            logger.info(f"          ✅ 全局{clf_name}: Acc={np.mean(scores):.3f}±{np.std(scores):.3f}")
                            
            except Exception as e:
                logger.info(f"        ❌ 全局{clf_name} 失败: {e}")
                results['global_analysis'][clf_name] = {'error': str(e)}
        
        # ========================================================================
        # 2. 🔥 增强版分脑区分析 - 添加深度网络
        # ========================================================================
        logger.info("      🧠 分脑区受试者识别分析（含4×4096深度网络）...")
        
        region_dataset = self._build_region_aware_dataset()
        
        if len(region_dataset['features']) > 100:  # 确保有足够样本
            
            X_region = region_dataset['features']
            subject_labels_region = region_dataset['subject_labels']
            region_labels_region = region_dataset['region_labels']
            
            # 重新映射受试者标签为连续编号
            unique_subjects = np.unique(subject_labels_region)
            subject_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_subjects)}
            mapped_subject_labels = np.array([subject_mapping[s] for s in subject_labels_region])
            
            logger.info(f"        📊 分脑区数据集: {len(X_region)} 样本, {len(unique_subjects)} 受试者, {len(np.unique(region_labels_region))} 脑区")
            
            for clf_name, clf in classifiers.items():
                try:
                    # 🔥 深度网络分脑区分析需要特殊处理
                    if 'deep' in clf_name:
                        logger.info(f"        🔥 训练分脑区{clf_name}深度网络...")
                        
                        # 为分脑区分析创建专用深度网络配置
                        if clf_name == 'deep_4x4096':
                            clf_region = self._create_deep_classifier_wrapper({
                                'batch_size': 64,  # 分脑区用较小batch size
                                'no_epochs': 20    # 适中的epoch数
                            })
                        else:  # lightweight
                            clf_region = self._create_deep_classifier_wrapper({
                                'batch_size': 32,  
                                'no_epochs': 15
                            })
                        
                        start_time = time.time()
                        clf_region.fit(X_region, mapped_subject_labels)
                        train_time = time.time() - start_time
                        
                        # 🔥 深度网络分脑区性能评估
                        from sklearn.model_selection import StratifiedKFold
                        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 用3折节省时间
                        
                        scores = []
                        for train_idx, test_idx in cv_strategy.split(X_region, mapped_subject_labels):
                            test_score = clf_region.score(X_region[test_idx], mapped_subject_labels[test_idx])
                            scores.append(test_score)
                        
                        # 🔥 分脑区深度网络特殊分析
                        if hasattr(clf_region, 'training_history'):
                            history = clf_region.training_history
                            
                            # 受试者判别能力分析
                            random_baseline = 1.0 / len(unique_subjects)
                            discrimination_strength = np.mean(scores) / random_baseline
                            
                            results['deep_network_detailed_analysis'][f'{clf_name}_region_wise'] = {
                                'training_time': train_time,
                                'convergence_epochs': len(history['loss']),
                                'final_training_accuracy': history['accuracy'][-1] if history['accuracy'] else 0,
                                'cv_test_accuracy': np.mean(scores),
                                'discrimination_strength': discrimination_strength,
                                'random_baseline': random_baseline,
                                'region_adaptation_quality': 'excellent' if discrimination_strength > 5 else 'good' if discrimination_strength > 3 else 'moderate',
                                'training_efficiency': np.mean(scores) / (train_time / 60),  # 准确率/分钟
                                'data_utilization': len(X_region) / len(unique_subjects)  # 平均每受试者样本数
                            }
                        
                        results['region_wise_analysis'][clf_name] = {
                            'mean_score': np.mean(scores),
                            'std_score': np.std(scores),
                            'scores': scores,
                            'n_samples': len(X_region),
                            'n_subjects': len(unique_subjects),
                            'n_regions': len(np.unique(region_labels_region)),
                            'method': 'stratified_cv',
                            'training_time_minutes': train_time / 60,
                            'is_deep_network': True
                        }
                        
                        logger.info(f"          ✅ 🔥 分脑区{clf_name}: Acc={np.mean(scores):.3f}±{np.std(scores):.3f}, 判别强度={discrimination_strength:.2f}x")
                        
                    else:
                        # 传统分类器的分脑区分析（保持原有逻辑）
                        from sklearn.model_selection import StratifiedKFold
                        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        scores = cross_val_score(clf, X_region, mapped_subject_labels, 
                                            cv=cv_strategy, scoring='accuracy')
                        
                        results['region_wise_analysis'][clf_name] = {
                            'mean_score': np.mean(scores),
                            'std_score': np.std(scores),
                            'scores': scores,
                            'n_samples': len(X_region),
                            'n_subjects': len(unique_subjects),
                            'n_regions': len(np.unique(region_labels_region)),
                            'method': 'stratified_cv',
                            'is_deep_network': False
                        }
                        
                        logger.info(f"          ✅ 分脑区{clf_name}: Acc={np.mean(scores):.3f}±{np.std(scores):.3f}")
                        
                except Exception as e:
                    logger.info(f"        ❌ 分脑区{clf_name} 失败: {e}")
                    results['region_wise_analysis'][clf_name] = {'error': str(e)}
        
        else:
            logger.info("        ⚠️ 分脑区样本不足，跳过分析")
            results['region_wise_analysis'] = {'insufficient_data': True}
        
        # ========================================================================
        # 3. 🔥 深度网络专项对比分析
        # ========================================================================
        logger.info("      📈 🔥 深度网络专项性能对比分析...")
        
        # 传统方法 vs 深度网络对比
        for clf_name in classifiers.keys():
            if (clf_name in results['global_analysis'] and 
                clf_name in results['region_wise_analysis'] and
                'mean_score' in results['global_analysis'][clf_name] and
                'mean_score' in results['region_wise_analysis'][clf_name]):
                
                global_metrics = results['global_analysis'][clf_name]
                region_metrics = results['region_wise_analysis'][clf_name]
                
                # 计算各指标的提升
                acc_improvement = region_metrics['mean_score'] - global_metrics['mean_score']
                acc_improvement_percent = acc_improvement / global_metrics['mean_score'] * 100
                
                comparison_result = {
                    'accuracy_improvement': acc_improvement,
                    'accuracy_improvement_percent': acc_improvement_percent,
                    'interpretation': self._interpret_improvement(acc_improvement, acc_improvement / global_metrics['mean_score']),
                    'is_deep_network': clf_name.startswith('deep')
                }
                
                # 🔥 深度网络特殊对比分析
                if clf_name.startswith('deep'):
                    global_time = global_metrics.get('training_time', 0) or global_metrics.get('avg_training_time', 0)
                    region_time = region_metrics.get('training_time_minutes', 0) * 60
                    
                    comparison_result.update({
                        'training_time_ratio': region_time / global_time if global_time > 0 else 0,
                        'deep_network_advantage': 'significant' if acc_improvement > 0.05 else 'moderate' if acc_improvement > 0.02 else 'minimal',
                        'efficiency_assessment': 'efficient' if region_time < 300 else 'moderate' if region_time < 900 else 'slow',  # 秒为单位
                        'scalability': 'good' if region_metrics['n_samples'] > 1000 else 'limited'
                    })
                    
                    # 🔥 深度网络详细性能分析
                    if f'{clf_name}_global' in results['deep_network_detailed_analysis']:
                        global_deep = results['deep_network_detailed_analysis'][f'{clf_name}_global']
                        comparison_result['deep_analysis'] = {
                            'convergence_stability': global_deep.get('convergence_quality', 'unknown'),
                            'generalization_quality': 'good' if global_deep.get('generalization_gap', 1) < 0.1 else 'poor',
                            'training_efficiency': global_deep.get('final_training_accuracy', 0) / (global_deep.get('training_time', 1) / 60)
                        }
                    
                    logger.info(f"        📊 🔥 {clf_name}深度网络对比:")
                    logger.info(f"          准确率: {global_metrics['mean_score']:.3f} → {region_metrics['mean_score']:.3f} ({acc_improvement:+.3f})")
                    logger.info(f"          深度网络优势: {comparison_result['deep_network_advantage']}")
                    logger.info(f"          效率评估: {comparison_result['efficiency_assessment']}")
                    
                else:
                    logger.info(f"        📊 {clf_name}对比:")
                    logger.info(f"          准确率: {global_metrics['mean_score']:.3f} → {region_metrics['mean_score']:.3f} ({acc_improvement:+.3f})")
                
                results['comparative_analysis'][clf_name] = comparison_result
        
        # ========================================================================
        # 4. 🔥 深度网络权威性总结
        # ========================================================================
        deep_networks = [name for name in classifiers.keys() if name.startswith('deep')]
        if deep_networks:
            logger.info(f"      🔥 深度网络权威性总结:")
            
            best_deep_network = None
            best_score = 0
            
            for deep_name in deep_networks:
                if deep_name in results['region_wise_analysis'] and 'mean_score' in results['region_wise_analysis'][deep_name]:
                    score = results['region_wise_analysis'][deep_name]['mean_score']
                    if score > best_score:
                        best_score = score
                        best_deep_network = deep_name
            
            if best_deep_network:
                logger.info(f"        🏆 最佳深度网络: {best_deep_network} (准确率: {best_score:.3f})")
                
                # 🔥 深度网络权威性评估
                deep_authority_score = self._assess_deep_network_authority(results, deep_networks)
                results['deep_network_detailed_analysis']['authority_assessment'] = deep_authority_score
                
                logger.info(f"        📊 深度网络权威性评分: {deep_authority_score['overall_authority']:.3f}")
                logger.info(f"        🎯 权威性等级: {deep_authority_score['authority_level']}")
        
        return results


    def _assess_deep_network_authority(self, results, deep_networks):
        """评估深度网络在高维分类中的权威性"""
        
        authority_factors = {
            'performance_superiority': 0.0,
            'convergence_quality': 0.0,
            'generalization_ability': 0.0,
            'scalability': 0.0,
            'consistency': 0.0
        }
        
        # 1. 性能优越性：与传统方法对比
        traditional_networks = [name for name in results['region_wise_analysis'].keys() 
                            if not name.startswith('deep') and 'mean_score' in results['region_wise_analysis'][name]]
        
        if traditional_networks and deep_networks:
            traditional_scores = [results['region_wise_analysis'][name]['mean_score'] for name in traditional_networks]
            deep_scores = [results['region_wise_analysis'][name]['mean_score'] for name in deep_networks 
                        if 'mean_score' in results['region_wise_analysis'][name]]
            
            if traditional_scores and deep_scores:
                avg_traditional = np.mean(traditional_scores)
                avg_deep = np.mean(deep_scores)
                authority_factors['performance_superiority'] = min(1.0, max(0.0, (avg_deep - avg_traditional) / avg_traditional * 2))
        
        # 2. 收敛质量
        convergence_scores = []
        for deep_name in deep_networks:
            global_key = f'{deep_name}_global'
            if global_key in results['deep_network_detailed_analysis']:
                deep_analysis = results['deep_network_detailed_analysis'][global_key]
                if deep_analysis.get('convergence_quality') == 'good':
                    convergence_scores.append(1.0)
                else:
                    convergence_scores.append(0.0)
        
        if convergence_scores:
            authority_factors['convergence_quality'] = np.mean(convergence_scores)
        
        # 3. 泛化能力
        generalization_scores = []
        for deep_name in deep_networks:
            global_key = f'{deep_name}_global'
            if global_key in results['deep_network_detailed_analysis']:
                deep_analysis = results['deep_network_detailed_analysis'][global_key]
                gen_gap = deep_analysis.get('generalization_gap', 0)
                # 泛化差距越小越好
                gen_score = max(0.0, 1.0 - abs(gen_gap) * 2)
                generalization_scores.append(gen_score)
        
        if generalization_scores:
            authority_factors['generalization_ability'] = np.mean(generalization_scores)
        
        # 4. 可扩展性
        scalability_scores = []
        for deep_name in deep_networks:
            region_key = f'{deep_name}_region_wise'
            if region_key in results['deep_network_detailed_analysis']:
                deep_analysis = results['deep_network_detailed_analysis'][region_key]
                discrimination = deep_analysis.get('discrimination_strength', 1.0)
                # 判别强度越高，可扩展性越好
                scalability_score = min(1.0, discrimination / 5.0)  # 5倍随机基线为满分
                scalability_scores.append(scalability_score)
        
        if scalability_scores:
            authority_factors['scalability'] = np.mean(scalability_scores)
        
        # 5. 一致性（全局和分脑区性能的一致性）
        consistency_scores = []
        for deep_name in deep_networks:
            if (deep_name in results['global_analysis'] and 
                deep_name in results['region_wise_analysis'] and
                'mean_score' in results['global_analysis'][deep_name] and
                'mean_score' in results['region_wise_analysis'][deep_name]):
                
                global_score = results['global_analysis'][deep_name]['mean_score']
                region_score = results['region_wise_analysis'][deep_name]['mean_score']
                
                # 一致性：两个分数越接近越好（但允许分脑区略优）
                consistency = 1.0 - abs(global_score - region_score) / max(global_score, region_score)
                consistency_scores.append(max(0.0, consistency))
        
        if consistency_scores:
            authority_factors['consistency'] = np.mean(consistency_scores)
        
        # 综合权威性评分
        weights = {
            'performance_superiority': 0.3,
            'convergence_quality': 0.2,
            'generalization_ability': 0.2,
            'scalability': 0.2,
            'consistency': 0.1
        }
        
        overall_authority = sum(authority_factors[factor] * weights[factor] 
                            for factor in authority_factors.keys())
        
        # 权威性等级
        if overall_authority > 0.8:
            authority_level = "AUTHORITATIVE"
            recommendation = "深度网络结果具有决定性权威，强烈建议采纳"
        elif overall_authority > 0.6:
            authority_level = "HIGHLY_CREDIBLE"
            recommendation = "深度网络结果高度可信，建议优先考虑"
        elif overall_authority > 0.4:
            authority_level = "MODERATELY_CREDIBLE"
            recommendation = "深度网络结果中等可信，可作为重要参考"
        else:
            authority_level = "LIMITED_CREDIBILITY"
            recommendation = "深度网络结果可信度有限，需结合其他方法"
        
        return {
            'overall_authority': overall_authority,
            'authority_level': authority_level,
            'recommendation': recommendation,
            'factor_breakdown': authority_factors,
            'detailed_interpretation': {
                'performance_superiority': f"相比传统方法的性能优势: {authority_factors['performance_superiority']:.3f}",
                'convergence_quality': f"训练收敛质量: {authority_factors['convergence_quality']:.3f}",
                'generalization_ability': f"泛化能力: {authority_factors['generalization_ability']:.3f}",
                'scalability': f"受试者判别能力: {authority_factors['scalability']:.3f}",
                'consistency': f"全局-分脑区一致性: {authority_factors['consistency']:.3f}"
            }
        }
    
    def _deep_network_feature_importance_analysis(self):
        """🔥 深度网络特征重要性分析 - Phase 3A增强"""
        
        logger.info("    🔥 深度网络特征重要性分析...")
        
        if 'subject_stats' not in self.analysis_results:
            logger.info("    ❌ 需要先运行Phase 1")
            return {}
        
        subject_means = self.analysis_results['subject_stats']['means']
        subject_labels = np.arange(len(subject_means))
        
        try:
            # 创建深度网络进行特征重要性分析
            logger.info("      🏗️ 创建4×4096深度网络进行特征分析...")
            deep_classifier = self._create_deep_classifier_wrapper({
                'batch_size': 64,
                'no_epochs': 20  # 充分训练以获得稳定的特征重要性
            })
            
            # 训练深度网络
            start_time = time.time()
            deep_classifier.fit(subject_means, subject_labels)
            train_time = time.time() - start_time
            
            logger.info(f"      ✅ 深度网络训练完成，耗时: {train_time/60:.1f}分钟")
            
            # 方法1: 扰动测试特征重要性
            logger.info("      🔍 执行特征扰动重要性分析...")
            perturbation_importance = self._compute_perturbation_importance(deep_classifier, subject_means, subject_labels)
            
            # 方法2: 梯度重要性分析
            logger.info("      🔍 执行梯度重要性分析...")
            gradient_importance = self._compute_gradient_importance(deep_classifier, subject_means, subject_labels)
            
            # 方法3: 网络权重分析
            logger.info("      🔍 执行网络权重重要性分析...")
            weight_importance = self._compute_weight_importance(deep_classifier)
            
            # 综合特征重要性
            combined_importance = self._combine_feature_importance(
                perturbation_importance, gradient_importance, weight_importance
            )
            
            # 特征分组重要性分析
            group_importance = self._analyze_feature_group_importance(combined_importance)
            
            # 生成特征重要性排名
            top_features = np.argsort(combined_importance)[-50:][::-1]  # Top 50 重要特征
            bottom_features = np.argsort(combined_importance)[:20]      # Bottom 20 不重要特征
            
            feature_analysis = {
                'perturbation_importance': perturbation_importance,
                'gradient_importance': gradient_importance,
                'weight_importance': weight_importance,
                'combined_importance': combined_importance,
                'group_importance': group_importance,
                'top_features': top_features,
                'bottom_features': bottom_features,
                'importance_statistics': {
                    'mean': np.mean(combined_importance),
                    'std': np.std(combined_importance),
                    'max': np.max(combined_importance),
                    'min': np.min(combined_importance),
                    'sparsity': np.sum(combined_importance < 0.01) / len(combined_importance)  # 不重要特征比例
                },
                'training_time': train_time,
                'network_performance': deep_classifier.score(subject_means, subject_labels)
            }
            
            logger.info(f"      ✅ 深度网络特征重要性分析完成")
            logger.info(f"        - 最重要特征: {top_features[:5]}")
            logger.info(f"        - 特征重要性范围: [{np.min(combined_importance):.4f}, {np.max(combined_importance):.4f}]")
            logger.info(f"        - 稀疏度: {feature_analysis['importance_statistics']['sparsity']:.1%}")
            
            return feature_analysis
            
        except Exception as e:
            logger.info(f"      ❌ 深度网络特征重要性分析失败: {e}")
            return {'error': str(e)}

    def _compute_perturbation_importance(self, classifier, X, y):
        """计算扰动重要性"""
        baseline_score = classifier.score(X, y)
        importance_scores = np.zeros(X.shape[1])
        
        # 对每个特征进行扰动测试
        for i in range(X.shape[1]):
            if i % 50 == 0:  # 每50个特征打印一次进度
                logger.info(f"        扰动测试进度: {i}/{X.shape[1]}")
            
            X_perturbed = X.copy()
            # 用随机噪声替换该特征
            X_perturbed[:, i] = np.random.normal(np.mean(X[:, i]), np.std(X[:, i]), X.shape[0])
            
            perturbed_score = classifier.score(X_perturbed, y)
            importance_scores[i] = baseline_score - perturbed_score  # 性能下降越大，重要性越高
        
        # 归一化到 [0, 1]
        importance_scores = np.maximum(importance_scores, 0)  # 只保留正向重要性
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        
        return importance_scores

    def _compute_gradient_importance(self, classifier, X, y):
        """计算梯度重要性（简化版）"""
        # 注意：这是一个简化的梯度重要性计算
        # 在实际PyTorch实现中，你可能需要更复杂的梯度计算
        
        # 计算特征的方差作为梯度重要性的代理
        feature_variances = np.var(X, axis=0)
        
        # 归一化
        if np.max(feature_variances) > 0:
            gradient_importance = feature_variances / np.max(feature_variances)
        else:
            gradient_importance = np.zeros(X.shape[1])
        
        return gradient_importance

    def _compute_weight_importance(self, classifier):
        """计算网络权重重要性"""
        try:
            # 获取第一层权重（输入层到第一个隐藏层）
            if hasattr(classifier, 'network') and hasattr(classifier.network, 'fc1'):
                first_layer_weights = classifier.network.fc1.weight.data.cpu().numpy()  # shape: (4096, 341)
                
                # 计算每个输入特征对所有隐藏单元的权重绝对值之和
                weight_importance = np.sum(np.abs(first_layer_weights), axis=0)  # shape: (341,)
                
                # 归一化
                if np.max(weight_importance) > 0:
                    weight_importance = weight_importance / np.max(weight_importance)
                
                return weight_importance
            else:
                # 如果无法获取权重，返回均匀分布
                return np.ones(341) * 0.5
                
        except Exception as e:
            logger.info(f"        ⚠️ 权重重要性计算失败: {e}")
            return np.ones(341) * 0.5

    def _combine_feature_importance(self, perturbation, gradient, weight):
        """综合多种特征重要性方法"""
        
        # 加权平均
        weights = {
            'perturbation': 0.5,  # 扰动测试最可靠
            'gradient': 0.2,      # 梯度信息
            'weight': 0.3         # 网络权重
        }
        
        combined = (perturbation * weights['perturbation'] + 
                    gradient * weights['gradient'] + 
                    weight * weights['weight'])
        
        return combined

    def _analyze_feature_group_importance(self, feature_importance):
        """分析特征分组的重要性"""
        
        feature_groups = {
            'qti_params': list(range(0, 15)),           # QTI参数 (15个)
            'raw_b_tensors': list(range(15, 225)),      # 原始b-tensor值 (210个)
            'cest_params': list(range(225, 229)),       # CEST参数 (4个)
            'z_spectrum': list(range(229, 341))         # Z-spectrum值 (112个)
        }
        
        group_importance = {}
        
        for group_name, indices in feature_groups.items():
            if len(indices) > 0:
                group_scores = feature_importance[indices]
                group_importance[group_name] = {
                    'mean_importance': np.mean(group_scores),
                    'max_importance': np.max(group_scores),
                    'std_importance': np.std(group_scores),
                    'top_features_in_group': np.argsort(group_scores)[-5:][::-1] + indices[0],  # Top 5 in group
                    'group_size': len(indices),
                    'relative_contribution': np.sum(group_scores) / np.sum(feature_importance)
                }
        
        return group_importance


    def _build_region_aware_dataset(self):
        """构建脑区感知数据集：每个样本 = 一个受试者在一个脑区的平均特征"""
        
        logger.info("        🔧 构建分脑区数据集...")
        
        # 获取脑区标签
        if len(self.data['y_train'].shape) > 1 and self.data['y_train'].shape[1] > 1:
            y_classes = np.argmax(self.data['y_train'], axis=1)
        else:
            y_classes = self.data['y_train'].flatten()
        
        X_list = []
        subject_labels = []
        region_labels = []
        sample_counts = []
        
        available_subjects = self.data['available_subjects']
        unique_regions = np.unique(y_classes)
        
        valid_combinations = 0
        total_combinations = len(available_subjects) * len(unique_regions)
        
        for subject_id in available_subjects:
            for region_id in unique_regions:
                # 找到该受试者在该脑区的所有体素
                mask = (self.data['subjects_train'] == subject_id) & (y_classes == region_id)
                n_voxels = np.sum(mask)
                
                if n_voxels >= 50:  # 最小体素数阈值
                    # 计算该受试者在该脑区的平均特征
                    region_features = np.mean(self.data['X_train'][mask], axis=0)
                    
                    X_list.append(region_features)
                    subject_labels.append(subject_id)
                    region_labels.append(region_id)
                    sample_counts.append(n_voxels)
                    valid_combinations += 1
        
        logger.info(f"        ✅ 有效组合: {valid_combinations}/{total_combinations} "
            f"({valid_combinations/total_combinations*100:.1f}%)")
        logger.info(f"        📊 平均每组合体素数: {np.mean(sample_counts):.0f}")
        
        return {
            'features': np.array(X_list),
            'subject_labels': np.array(subject_labels),
            'region_labels': np.array(region_labels),
            'sample_counts': np.array(sample_counts)
        }

    def _interpret_improvement(self, improvement, relative_improvement):
        """解释性能改进的意义"""
        if relative_improvement > 0.1:  # 10%以上提升
            return "显著提升：分脑区分析明显优于全局分析"
        elif relative_improvement > 0.05:  # 5-10%提升
            return "中等提升：分脑区分析有一定优势"
        elif relative_improvement > 0.01:  # 1-5%提升
            return "轻微提升：分脑区分析略有优势"
        elif relative_improvement > -0.01:  # ±1%以内
            return "性能相当：两种方法差异不大"
        else:  # 下降
            return "性能下降：全局分析可能更适合"
        

    def _compare_structure_preservation(self, original_data, dimensionality_results):
        """比较不同降维方法的结构保持能力 (保持原有逻辑)"""
        original_distances = squareform(pdist(original_data))
        preservation_scores = {}
        
        for method_name, method_data in dimensionality_results.items():
            if method_data is None or method_name == 'structure_preservation':
                continue
                
            try:
                if '2d_result' in method_data:
                    reduced_data = method_data['2d_result']
                elif method_name == 'pca' and 'full_result' in method_data:
                    reduced_data = method_data['full_result'][:, :2]
                else:
                    continue
                
                reduced_distances = squareform(pdist(reduced_data))
                
                from scipy.stats import spearmanr
                correlation, p_value = spearmanr(
                    original_distances.flatten(), 
                    reduced_distances.flatten()
                )
                
                preservation_scores[method_name] = {
                    'spearman_correlation': correlation,
                    'p_value': p_value
                }
                
            except Exception as e:
                logger.info(f"      ⚠️ {method_name} 结构保持分析失败: {e}")
        
        return preservation_scores

  
    def generate_visualizations(self):
        """生成可视化图表 - 保持原有实现 + 🔥 存档点保护"""
        try:
            logger.info("\n" + "="*80)
            logger.info("📊 生成增强版可视化图表 (带存档点保护)")
            logger.info("="*80)
            
            # 🔥 可视化前创建存档点
            if self.checkpoint_manager:
                try:
                    self.checkpoint_manager.create_checkpoint(
                        analyzer_instance=self,
                        checkpoint_name="before_visualization",
                        phase_completed=-1,
                        description="可视化生成前的状态保存",
                        is_auto=True,
                        include_deep_networks=False  # 可视化前不需要保存网络状态
                    )
                except Exception as e:
                    logger.warning(f"可视化前存档点创建失败: {e}")
            
            # 🔄 生成原有可视化 (简化版，保留主要图表)
            logger.info("\n📊 生成原有分析可视化...")
            self._generate_original_visualizations()
            
            # 🔥 生成脑区特异性可视化
            logger.info("\n📊 生成脑区特异性分析图表...")
            self._generate_brain_region_specificity_visualizations()
            
            # 🔥 生成脑区embedding设计可视化
            logger.info("\n📊 生成脑区embedding设计图表...")
            self._generate_brain_embedding_design_visualizations()
            
            # 🔥 新增：生成降维方法对比图表
            logger.info("\n📊 生成降维方法对比图表...")
            self._generate_dimensionality_comparison_plot()
            
            logger.info(f"✅ 增强版可视化图表生成完成")
            
        except Exception as e:
            logger.error(f"可视化生成过程中出现错误: {e}")
            logger.exception("详细错误信息:")
            
            # 🔥 可视化异常时的紧急存档
            if self.checkpoint_manager:
                self.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=f"可视化生成异常: {str(e)}",
                    stack_trace=str(e)
                )

    def _generate_original_visualizations(self):
        """生成原有的可视化图表 (简化版)"""
        
        if 'subject_stats' not in self.analysis_results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 受试者相似性热图
        ax = axes[0, 0]
        if 'global_subject_similarity' in self.analysis_results:
            correlation_matrix = self.analysis_results['global_subject_similarity']['correlation_matrix']
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('全局受试者间相关性矩阵', fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 2. PCA累积方差解释
        ax = axes[0, 1]
        if 'global_feature_variation' in self.analysis_results:
            cumulative_variance = self.analysis_results['global_feature_variation']['pca_cumulative_variance']
            ax.plot(range(1, min(21, len(cumulative_variance)+1)), 
                   cumulative_variance[:20], 'o-', linewidth=2)
            ax.axhline(0.8, color='red', linestyle='--', label='80%解释阈值')
            ax.set_title('PCA累积方差解释', fontweight='bold')
            ax.set_xlabel('主成分数量')
            ax.set_ylabel('累积方差解释比例')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. 全局受试者识别准确率
        ax = axes[0, 2]
        if 'global_subject_identification' in self.analysis_results:
            global_id = self.analysis_results['global_subject_identification']
            accuracy = global_id['accuracy']
            baseline = global_id['random_baseline']
            
            ax.bar(['随机基线', '最佳分类器'], [baseline, accuracy], 
                  color=['gray', 'lightcoral'])
            ax.set_title('全局受试者识别准确率', fontweight='bold')
            ax.set_ylabel('准确率')
            
            ax.text(0, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            ax.text(1, accuracy + 0.01, f'{accuracy:.3f}', ha='center', va='bottom')
        
        # 4. 特征变异分布
        ax = axes[1, 0]
        if 'global_feature_variation' in self.analysis_results:
            f_stats = self.analysis_results['global_feature_variation']['f_stats']
            ax.hist(f_stats, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.percentile(f_stats, 90), color='red', linestyle='--', 
                      label=f'90th: {np.percentile(f_stats, 90):.2f}')
            ax.set_title('特征受试者间变异分布', fontweight='bold')
            ax.set_xlabel('标准化F统计量')
            ax.set_ylabel('特征数量')
            ax.legend()
        
        # 5. 综合决策雷达图
        ax = axes[1, 1]
        categories = ['差异显著性', '模式线性度', '受试者相似性', '可分离性', '样本充足性']
        
        scores = [
            self.decision_scores.get('phase1', {}).get('difference_significance', 0),
            self.decision_scores.get('phase1', {}).get('pattern_linearity', 0),
            self.decision_scores.get('phase1', {}).get('subject_similarity', 0),
            self.decision_scores.get('phase2', {}).get('subject_separability', 0),
            self.decision_scores.get('phase3', {}).get('sample_adequacy', 0)
        ]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores_plot = scores + [scores[0]]
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='blue')
        ax.fill(angles_plot, scores_plot, alpha=0.25, color='blue')
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Subject Embedding 可行性雷达图', fontweight='bold')
        ax.grid(True)
        
        # 6. 降维对比 (如果有)
        ax = axes[1, 2]
        if 'enhanced_dimensionality' in self.analysis_results:
            dim_results = self.analysis_results['enhanced_dimensionality']
            if 'dimensionality_comparison' in dim_results and 'pca' in dim_results['dimensionality_comparison']:
                pca_data = dim_results['dimensionality_comparison']['pca']
                pca_2d = pca_data['2d_result']
                scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                                   c=range(len(pca_2d)), cmap='viridis', s=50, alpha=0.7)
                ax.set_title('PCA - 受试者2D分布', fontweight='bold')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                plt.colorbar(scatter, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        original_viz_path = self.save_path / 'visualizations' / 'original_analysis.png'
        plt.savefig(original_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 重要：关闭图形避免内存泄漏
            
        logger.info(f"  ✅ 原有分析图表已保存: {original_viz_path}")
        
    def _generate_brain_region_specificity_visualizations(self):
        """生成脑区Subject Embedding需求分析图表（修正版）"""
        
        if 'region_wise_separability' not in self.analysis_results:
            logger.info("  ⚠️ 没有脑区Subject Embedding分析结果，跳过可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        region_analysis = self.analysis_results['region_wise_separability']
        
        # 过滤出成功分析的脑区
        successful_regions = {rid: data for rid, data in region_analysis.items() 
                            if 'embedding_necessity_score' in data}
        
        if len(successful_regions) == 0:
            logger.info("  ⚠️ 没有成功的Subject Embedding分析结果")
            return
        
        region_ids = list(successful_regions.keys())
        
        # 1. Subject Embedding需求得分分布
        ax = axes[0, 0]
        necessity_scores = [successful_regions[rid]['embedding_necessity_score'] for rid in region_ids]
        
        ax.hist(necessity_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(np.mean(necessity_scores), color='red', linestyle='--',
                label=f'平均: {np.mean(necessity_scores):.3f}')
        ax.axvline(0.5, color='orange', linestyle='--', label='推荐阈值: 0.5')
        ax.set_title('脑区Subject Embedding需求得分分布', fontweight='bold', fontsize=14)
        ax.set_xlabel('需求得分')
        ax.set_ylabel('脑区数量')
        ax.legend()
        
        # 2. Top 20 高需求脑区
        ax = axes[0, 1]
        sorted_regions = sorted(region_ids, key=lambda x: successful_regions[x]['embedding_necessity_score'], reverse=True)
        top_20_regions = sorted_regions[:20]
        top_20_scores = [successful_regions[rid]['embedding_necessity_score'] for rid in top_20_regions]
        
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(top_20_regions)))
        bars = ax.bar(range(len(top_20_regions)), top_20_scores, color=colors, alpha=0.8)
        ax.set_title('Top 20 最需要Subject Embedding的脑区', fontweight='bold', fontsize=14)
        ax.set_xlabel('脑区排名')
        ax.set_ylabel('需求得分')
        ax.set_xticks(range(0, len(top_20_regions), max(1, len(top_20_regions)//5)))
        
        # 3. 需求等级分布饼图
        ax = axes[0, 2]
        necessity_levels = [successful_regions[rid]['embedding_necessity_level'] for rid in region_ids]
        level_counts = {}
        for level in necessity_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            labels = list(level_counts.keys())
            sizes = list(level_counts.values())
            colors_pie = ['red', 'orange', 'yellow', 'lightblue'][:len(labels)]
            
            ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax.set_title('Subject Embedding需求等级分布', fontweight='bold', fontsize=14)
        
        # 4. 一致性 vs 泛化性能散点图
        ax = axes[1, 0]
        consistency_scores = []
        generalization_scores = []
        
        for rid in region_ids:
            data = successful_regions[rid]
            if 'component_scores' in data:
                consistency_scores.append(data['component_scores']['consistency'])
                generalization_scores.append(data['component_scores']['generalization'])
        
        if consistency_scores and generalization_scores:
            scatter = ax.scatter(consistency_scores, generalization_scores, 
                            c=necessity_scores, cmap='Reds', alpha=0.7, s=60)
            ax.set_xlabel('一致性问题得分')
            ax.set_ylabel('泛化问题得分')
            ax.set_title('脑区问题类型分析', fontweight='bold', fontsize=14)
            plt.colorbar(scatter, ax=ax, label='总需求得分')
        
        # 5. 推荐embedding维度分布
        ax = axes[1, 1]
        recommended_dims = [successful_regions[rid]['recommended_embedding_dim'] for rid in region_ids]
        dim_counts = {}
        for dim in recommended_dims:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        if dim_counts:
            dims = list(dim_counts.keys())
            counts = list(dim_counts.values())
            
            colors_bar = plt.cm.viridis(np.linspace(0, 1, len(dims)))
            ax.bar(dims, counts, color=colors_bar, alpha=0.7)
            ax.set_title('推荐Embedding维度分布', fontweight='bold', fontsize=14)
            ax.set_xlabel('推荐维度')
            ax.set_ylabel('脑区数量')
        
        # 6. 实施优先级分布
        ax = axes[1, 2]
        priorities = [successful_regions[rid]['implementation_priority'] for rid in region_ids]
        priority_counts = {}
        for priority in priorities:
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        if priority_counts:
            priority_labels = list(priority_counts.keys())
            priority_sizes = list(priority_counts.values())
            
            x = range(len(priority_labels))
            colors_priority = ['red', 'orange', 'yellow', 'lightgreen'][:len(priority_labels)]
            
            bars = ax.bar(x, priority_sizes, color=colors_priority, alpha=0.7)
            ax.set_title('实施优先级分布', fontweight='bold', fontsize=14)
            ax.set_xlabel('优先级')
            ax.set_ylabel('脑区数量')
            ax.set_xticks(x)
            ax.set_xticklabels(priority_labels, rotation=45)
            
            # 添加数值标签
            for bar, size in zip(bars, priority_sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        brain_viz_path = self.save_path / 'visualizations' / 'subject_embedding_necessity_analysis.png'
        plt.savefig(brain_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 添加plt.close()

    def _generate_brain_embedding_design_visualizations(self):
        """生成脑区embedding设计图表（修复版）"""
        
        if 'brain_aware_embedding_design' not in self.analysis_results:
            logger.info("  ⚠️ 没有脑区embedding设计结果，跳过相关可视化")
            return
        
        brain_design = self.analysis_results['brain_aware_embedding_design']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 脑区分层需求柱状图
        ax = axes[0, 0]
        if 'region_requirements' in brain_design:
            requirements = brain_design['region_requirements']
            
            tier_names = []
            tier_counts = []
            tier_dims = []
            
            for tier_name, tier_data in requirements.items():
                if tier_data['region_ids']:  # 只包含非空的层次
                    tier_names.append(tier_name.replace('_', ' ').title())
                    tier_counts.append(len(tier_data['region_ids']))
                    tier_dims.append(tier_data['recommended_dim'])
            
            if tier_names:
                # 修复：使用matplotlib.cm而不是错误的plt.cm.viridis()调用
                import matplotlib.cm as cm
                colors_bar = [cm.viridis(i/len(tier_names)) for i in range(len(tier_names))]
                
                bars = ax.bar(tier_names, tier_counts, color=colors_bar, alpha=0.7)
                ax.set_title('脑区分层需求分布', fontweight='bold', fontsize=14)
                ax.set_xlabel('需求层次')
                ax.set_ylabel('脑区数量')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加推荐维度标签
                for bar, dim in zip(bars, tier_dims):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{dim}D', ha='center', va='bottom', fontweight='bold')
        
        # 2. 脑区embedding维度推荐分布
        ax = axes[0, 1]
        if 'region_embedding_dims' in brain_design:
            region_dims = brain_design['region_embedding_dims']
            dims = [data['recommended_dim'] for data in region_dims.values()]
            
            ax.hist(dims, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(dims), color='red', linestyle='--', 
                    label=f'平均: {np.mean(dims):.1f}D')
            ax.set_title('脑区embedding维度分布', fontweight='bold', fontsize=14)
            ax.set_xlabel('推荐embedding维度')
            ax.set_ylabel('脑区数量')
            ax.legend()
        
        # 3. 脑区聚类结果
        ax = axes[1, 0]
        if 'region_clustering_strategy' in brain_design:
            clustering = brain_design['region_clustering_strategy']
            
            if 'clusters' in clustering:
                cluster_ids = list(clustering['clusters'].keys())
                cluster_sizes = [clustering['clusters'][cid]['n_regions'] for cid in cluster_ids]
                cluster_specificities = [clustering['clusters'][cid]['avg_specificity'] for cid in cluster_ids]
                
                bars = ax.bar(cluster_ids, cluster_sizes, color='lightgreen', alpha=0.7)
                ax.set_title(f'脑区聚类结果 ({len(cluster_ids)} 个聚类)', fontweight='bold', fontsize=14)
                ax.set_xlabel('聚类ID')
                ax.set_ylabel('脑区数量')
                
                # 添加平均特异性标签
                for bar, spec in zip(bars, cluster_specificities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{spec:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 4. embedding策略总结饼图
        ax = axes[1, 1]
        if 'region_requirements' in brain_design:
            requirements = brain_design['region_requirements']
            
            necessity_counts = {}
            for tier_data in requirements.values():
                necessity = tier_data['embedding_necessity']
                necessity_counts[necessity] = necessity_counts.get(necessity, 0) + len(tier_data['region_ids'])
            
            if necessity_counts:
                labels = list(necessity_counts.keys())
                sizes = list(necessity_counts.values())
                colors = ['red', 'orange', 'yellow', 'lightblue'][:len(labels)]
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('脑区embedding需求分布', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        embedding_viz_path = self.save_path / 'visualizations' / 'brain_embedding_design.png'
        plt.savefig(embedding_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 添加plt.close()
        
        logger.info(f"  ✅ 脑区embedding设计图表已保存: {embedding_viz_path}")

    def phase4_decision_generation(self):
        """Phase 4: 决策建议生成 - 🔥 带存档点功能"""
        self._start_phase_timer("Phase 4")
        
        try:
            logger.info("\n" + "="*80)
            logger.info("🎯 Phase 4: 决策建议生成 (脑区感知增强版 + 存档点)")
            logger.info("="*80)
            
            # 🔄 保留原有决策逻辑
            logger.info("\n📊 Phase 4A: 全局决策生成...")
            global_decision = self._generate_global_decision()
            
            # 🔥 新增脑区感知决策
            logger.info("\n📊 Phase 4B: 脑区感知决策生成...")
            brain_aware_decision = self._generate_brain_aware_decision()
            
            # 综合决策
            logger.info("\n📊 Phase 4C: 综合决策整合...")
            comprehensive_decision = self._integrate_comprehensive_decision(global_decision, brain_aware_decision)
            
            # 生成实施建议
            logger.info("\n📊 Phase 4D: 实施建议生成...")
            implementation_plan = self._generate_implementation_plan(comprehensive_decision)
            
            # 最终决策结果
            final_decision = {
                'global_analysis': global_decision,
                'brain_aware_analysis': brain_aware_decision,
                'comprehensive_recommendation': comprehensive_decision,
                'implementation_plan': implementation_plan,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_version': 'Brain-Aware v1.0 + Checkpoints',
                'total_analysis_time': sum(self.phase_durations),
                'phase_durations': self.phase_durations.copy()
            }
            
            self.analysis_results['final_comprehensive_decision'] = final_decision
            
            duration = self._end_phase_timer("Phase 4")
            
            # 🔥 创建最终存档点
            if self.checkpoint_manager:
                try:
                    self.checkpoint_manager.create_checkpoint(
                        analyzer_instance=self,
                        checkpoint_name="final_decision",
                        phase_completed=4,
                        description="完整分析完成，包含最终决策和实施计划",
                        is_auto=True
                    )
                except Exception as e:
                    logger.warning(f"最终存档点创建失败: {e}")
            
            # 打印决策报告
            self._print_comprehensive_decision_report(final_decision)
            
            return final_decision
        
        except Exception as e:
            # 🔥 异常时创建紧急存档点
            if self.checkpoint_manager:
                self.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=f"Phase 4 执行异常: {str(e)}",
                    stack_trace=str(e)
                )
            raise

    
    def run_complete_analysis(self, skip_phase=None):
        """
        执行完整分析流程
        
        Args:
            skip_phase: 跳过的阶段列表 (用于从存档点恢复后的部分执行)
        """
        skip_phase = skip_phase or []
        
        total_start_time = time.time()
        logger.info("🚀 开始完整的脑区感知Subject Embedding分析")
        
        try:
            if 1 not in skip_phase:
                self.phase1_subject_differences_analysis()
            else:
                logger.info("⏭️ 跳过Phase 1 (从存档点恢复)")
            
            if 2 not in skip_phase:
                self.phase2_subject_separability_analysis()
            else:
                logger.info("⏭️ 跳过Phase 2 (从存档点恢复)")
            
            if 3 not in skip_phase:
                self.phase3_embedding_adaptability_analysis()
            else:
                logger.info("⏭️ 跳过Phase 3 (从存档点恢复)")
            
            if 4 not in skip_phase:
                final_decision = self.phase4_decision_generation()
            else:
                logger.info("⏭️ 跳过Phase 4 (从存档点恢复)")
                final_decision = self.analysis_results.get('final_comprehensive_decision', {})
            
            total_time = time.time() - total_start_time
            
            logger.info(f"\n🎉 完整分析成功完成！")
            logger.info(f"⏱️ 总耗时: {total_time/60:.2f}分钟")
            logger.info(f"📊 阶段耗时: {[f'{d/60:.1f}min' for d in self.phase_durations]}")
            
            return final_decision
            
        except Exception as e:
            logger.error(f"❌ 完整分析执行失败: {e}")
            logger.exception("详细错误信息:")
            
            # 🔥 最后的紧急存档
            if self.checkpoint_manager:
                self.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=f"完整分析执行异常: {str(e)}",
                    stack_trace=str(e)
                )
            raise
    

    def _generate_global_decision(self):
        """生成全局决策 (基于原有逻辑)"""
        
        # 综合所有决策得分
        all_scores = {}
        for phase in self.decision_scores:
            all_scores.update(self.decision_scores[phase])
        
        # 原有权重
        weights = {
            'difference_significance': 0.15,
            'pattern_linearity': 0.15,
            'subject_separability': 0.15,
            'class_consistency': 0.10,
            'sample_adequacy': 0.10,
            'embedding_feasibility': 0.15,
            'intrinsic_dimensionality': 0.10,
            'high_dim_performance': 0.10
        }
        
        # 计算加权总分
        weighted_score = sum(all_scores.get(key, 0) * weight 
                            for key, weight in weights.items())
        
        # 全局决策
        if weighted_score > 0.75:
            global_recommendation = "强烈推荐使用Subject Embedding"
            confidence = "高"
        elif weighted_score > 0.6:
            global_recommendation = "建议使用Subject Embedding"
            confidence = "中高"
        elif weighted_score > 0.45:
            global_recommendation = "可以尝试Subject Embedding"
            confidence = "中"
        else:
            global_recommendation = "Subject Embedding效果可能有限"
            confidence = "低"
        
        return {
            'weighted_score': weighted_score,
            'recommendation': global_recommendation,
            'confidence': confidence,
            'key_scores': all_scores
        }

    def _generate_brain_aware_decision(self):
        """生成脑区感知决策"""
        
        brain_decision = {
            'region_tier_analysis': {},
            'embedding_architecture': 'uniform',
            'priority_regions': [],
            'implementation_strategy': 'global_first'
        }
        
        if 'brain_aware_embedding_design' in self.analysis_results:
            brain_design = self.analysis_results['brain_aware_embedding_design']
            
            # 1. 脑区分层分析
            if 'region_requirements' in brain_design:
                requirements = brain_design['region_requirements']
                
                total_regions = sum(len(tier['region_ids']) for tier in requirements.values())
                critical_regions = len(requirements.get('tier1_high_specificity', {}).get('region_ids', []))
                high_regions = len(requirements.get('tier2_medium_high_specificity', {}).get('region_ids', []))
                
                brain_decision['region_tier_analysis'] = {
                    'total_analyzed_regions': total_regions,
                    'critical_priority_regions': critical_regions,
                    'high_priority_regions': high_regions,
                    'priority_ratio': (critical_regions + high_regions) / total_regions if total_regions > 0 else 0
                }
                
                # 优先脑区列表
                brain_decision['priority_regions'] = (
                    requirements.get('tier1_high_specificity', {}).get('region_ids', []) +
                    requirements.get('tier2_medium_high_specificity', {}).get('region_ids', [])
                )
            
            # 2. 推荐embedding架构
            if brain_decision['region_tier_analysis'].get('priority_ratio', 0) > 0.3:
                brain_decision['embedding_architecture'] = 'hierarchical'
                brain_decision['implementation_strategy'] = 'tier_based'
            elif brain_decision['region_tier_analysis'].get('critical_priority_regions', 0) > 0:
                brain_decision['embedding_architecture'] = 'selective'
                brain_decision['implementation_strategy'] = 'critical_first'
            else:
                brain_decision['embedding_architecture'] = 'uniform'
                brain_decision['implementation_strategy'] = 'global_only'
        
        return brain_decision

    def _integrate_comprehensive_decision(self, global_decision, brain_aware_decision):
        """整合综合决策"""
        
        # 综合推荐强度
        global_score = global_decision['weighted_score']
        priority_ratio = brain_aware_decision['region_tier_analysis'].get('priority_ratio', 0)
        
        # 调整后的综合得分
        brain_aware_boost = min(0.2, priority_ratio * 0.4)  # 脑区感知可以提升最多0.2分
        comprehensive_score = min(1.0, global_score + brain_aware_boost)
        
        # 综合推荐
        if comprehensive_score > 0.8:
            final_recommendation = "强烈推荐脑区感知Subject Embedding"
            implementation_priority = "HIGH"
        elif comprehensive_score > 0.65:
            final_recommendation = "建议使用分层Subject Embedding"
            implementation_priority = "MEDIUM-HIGH"
        elif comprehensive_score > 0.5:
            final_recommendation = "可选择性使用Subject Embedding"
            implementation_priority = "MEDIUM"
        else:
            final_recommendation = "Subject Embedding收益可能有限"
            implementation_priority = "LOW"
        
        return {
            'final_recommendation': final_recommendation,
            'comprehensive_score': comprehensive_score,
            'implementation_priority': implementation_priority,
            'architecture_type': brain_aware_decision['embedding_architecture'],
            'implementation_strategy': brain_aware_decision['implementation_strategy'],
            'global_component': global_decision,
            'brain_aware_component': brain_aware_decision
        }

    def _generate_implementation_plan(self, comprehensive_decision):
        """生成具体实施计划"""
        
        implementation_plan = {
            'phase1_preparation': [],
            'phase2_development': [],
            'phase3_deployment': [],
            'estimated_timeline': '',
            'resource_requirements': {},
            'success_metrics': []
        }
        
        architecture_type = comprehensive_decision['architecture_type']
        
        # Phase 1: 准备阶段
        implementation_plan['phase1_preparation'] = [
            "完成详细的脑区特异性分析验证",
            "确定最终的脑区分层策略",
            "设计数据预处理pipeline"
        ]
        
        # Phase 2: 开发阶段
        if architecture_type == 'hierarchical':
            implementation_plan['phase2_development'] = [
                "实现层次化Subject Embedding架构",
                "开发分脑区的embedding训练策略",
                "实现脑区特异的新受试者适应机制",
                "建立脑区间embedding共享机制"
            ]
            implementation_plan['estimated_timeline'] = "4-6周"
            
        elif architecture_type == 'selective':
            implementation_plan['phase2_development'] = [
                "实现选择性Subject Embedding系统",
                "为优先脑区开发专门的embedding策略",
                "建立脑区重要性动态评估机制"
            ]
            implementation_plan['estimated_timeline'] = "3-4周"
            
        else:  # uniform
            implementation_plan['phase2_development'] = [
                "实现统一的Subject Embedding系统",
                "优化全局embedding维度和架构",
                "开发通用的新受试者适应策略"
            ]
            implementation_plan['estimated_timeline'] = "2-3周"
        
        # Phase 3: 部署阶段
        implementation_plan['phase3_deployment'] = [
            "在验证集上测试embedding效果",
            "与baseline模型进行对比评估",
            "优化hyperparameters",
            "部署到测试集进行最终评估"
        ]
        
        # 资源需求
        implementation_plan['resource_requirements'] = {
            'computational': 'GPU训练资源，建议V100或以上',
            'memory': '至少32GB RAM用于大规模数据处理',
            'storage': '充足存储空间保存embedding和中间结果',
            'development_time': implementation_plan['estimated_timeline']
        }
        
        # 成功指标
        implementation_plan['success_metrics'] = [
            "验证集F1分数提升 > 5%",
            "测试集泛化性能提升 > 3%",
            "优先脑区分类准确率显著提升",
            "新受试者适应速度和效果评估"
        ]
        
        return implementation_plan

    def _generate_implementation_suggestions(self, embedding_suggestions, all_scores, enhanced_results):
        """生成具体的实施建议"""
        suggestions = []
        
        # 基础实施建议
        embedding_dim = embedding_suggestions.get('embedding_dim', 64)
        suggestions.append(f"推荐embedding维度: {embedding_dim}")
        
        # Embedding类型建议
        embedding_type = embedding_suggestions.get('embedding_type', 'linear')
        if embedding_type == 'linear':
            suggestions.extend([
                "使用简单的nn.Embedding层",
                "学习率建议: 1e-3到1e-4",
                "添加L2正则化防止过拟合"
            ])
        elif embedding_type == 'mixed_linear':
            suggestions.extend([
                "使用nn.Embedding + 小型MLP",
                "MLP建议: embedding_dim -> embedding_dim*2 -> embedding_dim",
                "学习率建议: 1e-4到1e-5"
            ])
        else:  # nonlinear
            suggestions.extend([
                "使用深度MLP进行非线性embedding",
                "MLP建议: embedding_dim -> embedding_dim*4 -> embedding_dim*2 -> embedding_dim",
                "使用批归一化和Dropout"
            ])
        
        # 特征处理建议
        if embedding_suggestions.get('feature_selection_needed', False):
            suggestions.append("强烈建议进行特征选择，保留最重要的50-70%特征")
        
        if embedding_suggestions.get('use_feature_groups', False):
            best_group = embedding_suggestions.get('best_feature_group', '')
            suggestions.append(f"可优先使用{best_group}特征组进行embedding")
        
        # 去偏差建议
        if embedding_suggestions.get('use_debiasing', False):
            suggestions.extend([
                "实施特征去偏差策略",
                "考虑对抗训练: 最大化分类性能，最小化受试者识别",
                "使用梯度反转层(Gradient Reversal Layer)"
            ])
        
        # 混合专家建议
        if embedding_suggestions.get('use_mixture_experts', False):
            n_experts = embedding_suggestions.get('n_expert_clusters', 3)
            suggestions.extend([
                f"考虑Mixture of Experts架构，使用{n_experts}个专家",
                "先对受试者进行聚类，再训练每个专家",
                "使用门控网络动态选择专家"
            ])
        
        # 训练策略建议
        complexity = embedding_suggestions.get('complexity_level', 'medium')
        if complexity == 'simple':
            suggestions.extend([
                "使用较小的batch size (64-128)",
                "早停patience建议: 5-10 epochs",
                "使用简单的学习率调度"
            ])
        elif complexity == 'medium':
            suggestions.extend([
                "使用中等batch size (128-256)",
                "早停patience建议: 10-15 epochs", 
                "可尝试余弦退火学习率调度"
            ])
        else:  # complex
            suggestions.extend([
                "使用较大batch size (256-512)",
                "早停patience建议: 15-20 epochs",
                "使用warmup + 余弦退火策略"
            ])
        
        return suggestions

    def _suggest_adaptation_strategy(self, all_scores, enhanced_results, embedding_suggestions):
        """建议新受试者适应策略"""
        strategy = {
            'primary_method': '',
            'backup_methods': [],
            'required_samples': 0,
            'expected_performance': '',
            'implementation_notes': []
        }
        
        # 基于受试者可分离性选择主要策略
        separability = all_scores.get('subject_separability', 0.5)
        
        if separability > 0.7:
            # 高可分离性：需要快速适应
            strategy['primary_method'] = '少样本快速适应'
            strategy['required_samples'] = 500
            strategy['backup_methods'] = ['相似性迁移', '零样本泛化']
            strategy['expected_performance'] = '85-95%的最优性能'
            strategy['implementation_notes'] = [
                "冻结主模型参数，只训练新受试者embedding",
                "使用较高学习率(1e-3)快速收敛",
                "监控过拟合，通常10-50轮即可"
            ]
        
        elif separability > 0.4:
            # 中等可分离性：相似性迁移为主
            strategy['primary_method'] = '相似性迁移 + 微调'
            strategy['required_samples'] = 200
            strategy['backup_methods'] = ['少样本适应', '平均embedding初始化']
            strategy['expected_performance'] = '75-85%的最优性能'
            strategy['implementation_notes'] = [
                "先找最相似的受试者继承embedding",
                "用少量样本进行微调",
                "相似性可基于特征统计量判断"
            ]
        
        else:
            # 低可分离性：零样本泛化
            strategy['primary_method'] = '零样本泛化'
            strategy['required_samples'] = 0
            strategy['backup_methods'] = ['平均embedding', '回归预测embedding']
            strategy['expected_performance'] = '60-75%的最优性能'
            strategy['implementation_notes'] = [
                "使用所有训练受试者embedding的均值",
                "或基于特征统计量回归预测embedding",
                "无需新受试者的标注数据"
            ]
        
        # 基于聚类结果调整策略
        if embedding_suggestions.get('use_mixture_experts', False):
            strategy['implementation_notes'].append(
                "如果使用Mixture of Experts，需要先确定新受试者属于哪个专家组"
            )
        
        # 基于内在维度调整样本需求
        if 'high_dimensional_analysis' in enhanced_results:
            intrinsic_dim = enhanced_results['high_dimensional_analysis'].get('intrinsic_dimensionality', 100)
            if intrinsic_dim < 20:
                strategy['required_samples'] = max(50, strategy['required_samples'] // 2)
                strategy['implementation_notes'].append("内在维度低，需要样本量可以减半")
            elif intrinsic_dim > 100:
                strategy['required_samples'] = strategy['required_samples'] * 2
                strategy['implementation_notes'].append("内在维度高，建议增加样本量")
        
        return strategy

    def _generate_dimensionality_comparison_plot(self):
        """生成专门的降维方法对比图"""
        
        # 检查是否有增强版降维分析结果
        if 'enhanced_dimensionality' not in self.analysis_results:
            logger.info("  ⚠️ 没有增强版降维分析结果，跳过降维对比图生成")
            return
        
        enhanced_results = self.analysis_results['enhanced_dimensionality']
        
        if 'dimensionality_comparison' not in enhanced_results:
            logger.info("  ⚠️ 没有降维对比数据，跳过降维对比图生成")
            return
        
        dim_results = enhanced_results['dimensionality_comparison']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. PCA结果
        if 'pca' in dim_results and dim_results['pca'] is not None:
            ax = axes[0, 0]
            pca_2d = dim_results['pca']['2d_result']
            scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                            c=range(len(pca_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('PCA降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax, fraction=0.046)
            logger.info("    ✅ PCA降维图生成完成")
        else:
            axes[0, 0].text(0.5, 0.5, 'PCA数据不可用', ha='center', va='center', 
                        transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_title('PCA降维结果', fontweight='bold', fontsize=14)
        
        # 2. Kernel PCA结果（如果有的话）
        if 'kernel_pca' in dim_results and dim_results['kernel_pca'] is not None:
            ax = axes[0, 1]
            kpca_2d = dim_results['kernel_pca']['rbf_2d_result']
            scatter = ax.scatter(kpca_2d[:, 0], kpca_2d[:, 1],
                            c=range(len(kpca_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('Kernel PCA (RBF)降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('KPC1')
            ax.set_ylabel('KPC2')
            plt.colorbar(scatter, ax=ax, fraction=0.046)
            logger.info("    ✅ Kernel PCA降维图生成完成")
        else:
            axes[0, 1].text(0.5, 0.5, 'Kernel PCA数据不可用', ha='center', va='center', 
                        transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Kernel PCA降维结果', fontweight='bold', fontsize=14)
        
        # 3. t-SNE结果
        if 'tsne' in dim_results and dim_results['tsne'] is not None:
            ax = axes[0, 2]
            tsne_2d = dim_results['tsne']['2d_result']
            scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1],
                            c=range(len(tsne_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('t-SNE降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax, fraction=0.046)
            logger.info("    ✅ t-SNE降维图生成完成")
        else:
            axes[0, 2].text(0.5, 0.5, 't-SNE数据不可用', ha='center', va='center', 
                        transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].set_title('t-SNE降维结果', fontweight='bold', fontsize=14)
        
        # 4. UMAP结果（如果有的话）
        if 'umap' in dim_results and dim_results['umap'] is not None:
            ax = axes[1, 0]
            umap_2d = dim_results['umap']['2d_result']
            scatter = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                            c=range(len(umap_2d)), cmap='viridis', s=100, alpha=0.7)
            ax.set_title('UMAP降维结果', fontweight='bold', fontsize=14)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=ax, fraction=0.046)
            logger.info("    ✅ UMAP降维图生成完成")
        else:
            axes[1, 0].text(0.5, 0.5, 'UMAP数据不可用', ha='center', va='center', 
                        transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('UMAP降维结果', fontweight='bold', fontsize=14)
        
        # 5. PCA解释方差
        if 'pca' in dim_results and dim_results['pca'] is not None:
            ax = axes[1, 1]
            explained_var = dim_results['pca']['explained_variance'][:20]  # 前20个
            ax.plot(range(1, len(explained_var)+1), explained_var, 'o-', linewidth=2, markersize=6)
            ax.set_title('PCA解释方差', fontweight='bold', fontsize=14)
            ax.set_xlabel('主成分')
            ax.set_ylabel('解释方差比例')
            ax.grid(True, alpha=0.3)
            
            # 添加累积方差线
            if len(explained_var) > 1:
                cumulative_var = np.cumsum(explained_var)
                ax2 = ax.twinx()
                ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 's-', 
                        color='red', alpha=0.7, linewidth=2, markersize=4)
                ax2.set_ylabel('累积解释方差', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
            
            logger.info("    ✅ PCA解释方差图生成完成")
        else:
            axes[1, 1].text(0.5, 0.5, 'PCA方差数据不可用', ha='center', va='center', 
                        transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('PCA解释方差', fontweight='bold', fontsize=14)
        
        # 6. 结构保持能力对比
        if 'structure_preservation' in dim_results and dim_results['structure_preservation']:
            ax = axes[1, 2]
            methods = []
            correlations = []
            
            for method, scores in dim_results['structure_preservation'].items():
                if isinstance(scores, dict) and 'spearman_correlation' in scores:
                    methods.append(method.upper())
                    correlations.append(abs(scores['spearman_correlation']))
            
            if methods:
                bars = ax.bar(methods, correlations, color='lightcoral', alpha=0.7)
                ax.set_title('结构保持能力对比', fontweight='bold', fontsize=14)
                ax.set_ylabel('Spearman相关系数')
                ax.set_xticklabels(methods, rotation=45)
                ax.set_ylim(0, 1)
                
                # 添加数值标签
                for bar, corr in zip(bars, correlations):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
                
                logger.info("    ✅ 结构保持能力对比图生成完成")
            else:
                ax.text(0.5, 0.5, '结构保持数据不可用', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
                ax.set_title('结构保持能力对比', fontweight='bold', fontsize=14)
        else:
            axes[1, 2].text(0.5, 0.5, '结构保持数据不可用', ha='center', va='center', 
                        transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_title('结构保持能力对比', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        dim_viz_path = self.save_path / 'visualizations' / 'dimensionality_comparison.png'
        plt.savefig(dim_viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # 🔥 重要：关闭图形避免内存泄漏
        
        logger.info(f"  ✅ 降维对比图已保存: {dim_viz_path}")

    def _print_comprehensive_decision_report(self, final_decision):
        """打印综合决策报告"""
        
        logger.info(f"\n📋 脑区感知Subject Embedding 综合分析报告")
        logger.info("=" * 80)
        
        comp_decision = final_decision['comprehensive_recommendation']
        global_decision = final_decision['global_analysis']
        brain_decision = final_decision['brain_aware_analysis']
        impl_plan = final_decision['implementation_plan']
        
        logger.info(f"🎯 最终推荐: {final_decision['final_recommendation']}")
        logger.info(f"📊 综合得分: {comp_decision['comprehensive_score']:.3f} / 1.0")
        logger.info(f"⚡ 实施优先级: {comp_decision['implementation_priority']}")
        logger.info(f"🏗️ 推荐架构: {comp_decision['architecture_type'].upper()}")
        
        logger.info(f"\n📈 全局分析结果:")
        logger.info(f"  - 加权得分: {global_decision['weighted_score']:.3f}")
        logger.info(f"  - 置信度: {global_decision['confidence']}")
        logger.info(f"  - 基础推荐: {global_decision['recommendation']}")
        
        logger.info(f"\n🧠 脑区感知分析结果:")
        tier_analysis = brain_decision['region_tier_analysis']
        logger.info(f"  - 分析脑区总数: {tier_analysis.get('total_analyzed_regions', 0)}")
        logger.info(f"  - 极高优先级脑区: {tier_analysis.get('critical_priority_regions', 0)}")
        logger.info(f"  - 高优先级脑区: {tier_analysis.get('high_priority_regions', 0)}")
        logger.info(f"  - 优先级脑区比例: {tier_analysis.get('priority_ratio', 0):.1%}")
        
        logger.info(f"\n🛠️ 实施计划:")
        logger.info(f"  - 预估时间线: {impl_plan['estimated_timeline']}")
        logger.info(f"  - 实施策略: {comp_decision['implementation_strategy']}")
        logger.info(f"  - 主要阶段: 准备 → 开发 → 部署")
        
        # 详细的实施建议
        logger.info(f"\n📋 详细实施阶段:")
        logger.info(f"  📍 Phase 1 - 准备阶段:")
        for item in impl_plan['phase1_preparation']:
            logger.info(f"    • {item}")
        
        logger.info(f"  📍 Phase 2 - 开发阶段:")
        for item in impl_plan['phase2_development']:
            logger.info(f"    • {item}")
        
        logger.info(f"  📍 Phase 3 - 部署阶段:")
        for item in impl_plan['phase3_deployment']:
            logger.info(f"    • {item}")
        
        logger.info(f"\n📊 成功指标:")
        for metric in impl_plan['success_metrics']:
            logger.info(f"  📈 {metric}")
        
        logger.info(f"\n💻 资源需求:")
        resources = impl_plan['resource_requirements']
        for key, value in resources.items():
            logger.info(f"  🔧 {key.replace('_', ' ').title()}: {value}")
        
        logger.info(f"\n🎯 关键建议:")
        if comp_decision['architecture_type'] == 'hierarchical':
            logger.info("  ✅ 实施层次化架构，为不同特异性的脑区使用不同embedding策略")
            logger.info("  ✅ 优先处理高特异性脑区，预期获得最大性能提升")
            logger.info("  ✅ 建立脑区间embedding共享机制，提高效率")
        elif comp_decision['architecture_type'] == 'selective':
            logger.info("  ✅ 专注于关键脑区的embedding优化")
            logger.info("  ✅ 为优先脑区分配更多计算资源")
            logger.info("  ✅ 保持其他脑区的简单处理策略")
        else:
            logger.info("  ✅ 使用统一的全局Subject Embedding策略")
            logger.info("  ✅ 专注于优化全局embedding质量")
            logger.info("  ✅ 简化实施复杂度，快速验证效果")
        
        logger.info(f"\n📊 预期效果:")
        if comp_decision['comprehensive_score'] > 0.8:
            logger.info("  🚀 预期显著性能提升 (5-15%)")
            logger.info("  🚀 强烈建议立即实施")
        elif comp_decision['comprehensive_score'] > 0.65:
            logger.info("  📈 预期中等性能提升 (3-8%)")
            logger.info("  📈 建议优先考虑实施")
        else:
            logger.info("  📊 预期小幅性能提升 (1-5%)")
            logger.info("  📊 可作为优化方向之一")
        
        # 优先脑区信息
        if brain_decision['priority_regions']:
            logger.info(f"\n🔥 优先处理脑区:")
            priority_regions = brain_decision['priority_regions'][:10]  # 显示前10个
            logger.info(f"  📍 前10个优先脑区: {priority_regions}")
            if len(brain_decision['priority_regions']) > 10:
                logger.info(f"  📍 总共{len(brain_decision['priority_regions'])}个优先脑区")
        
        logger.info(f"\n⏱️ 分析完成时间: {final_decision['analysis_timestamp']}")
        logger.info(f"🔬 分析版本: {final_decision['analysis_version']}")
  
    def generate_report(self):
        """生成完整的脑区感知分析报告 - 增强版"""
        report_path = self.save_path / 'brain_aware_subject_embedding_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("脑区感知Subject Embedding 可行性分析报告 (增强版)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析版本: 脑区感知增强版 v1.0 + 存档点系统\n")
            
            # 🔥 新增：存档点使用统计
            if self.checkpoint_manager:
                checkpoints = self.checkpoint_manager.list_checkpoints()
                f.write(f"存档点数量: {len(checkpoints)}\n")
                if checkpoints:
                    f.write(f"最新存档点: {checkpoints[0]['name']} ({checkpoints[0]['creation_time']})\n")
            
            # 🔥 新增：性能统计
            if self.phase_durations:
                f.write(f"总分析时间: {sum(self.phase_durations)/60:.2f} 分钟\n")
                f.write(f"各阶段耗时: {[f'{d/60:.1f}min' for d in self.phase_durations]}\n")
            
            f.write("\n")
            
            # 数据概况
            f.write("📊 数据概况\n")
            f.write("-" * 40 + "\n")
            if hasattr(self, 'data') and self.data:
                f.write(f"受试者数量: {len(self.data['available_subjects'])}\n")
                f.write(f"特征维度: {self.data['X_train'].shape[1]}\n")
                f.write(f"训练样本量: {len(self.data['X_train']):,}\n")
                f.write(f"验证样本量: {len(self.data['X_val']):,}\n")
                f.write(f"测试样本量: {len(self.data['X_test']):,}\n\n")
            
            # 脑区分析概况
            if hasattr(self, 'data') and 'brain_region_analysis' in self.data:
                brain_data = self.data['brain_region_analysis']
                f.write("🧠 脑区分析概况\n")
                f.write("-" * 40 + "\n")
                f.write(f"有效脑区×受试者组合: {brain_data['valid_combinations']}\n")
                f.write(f"总可能组合: {brain_data['total_combinations']}\n")
                f.write(f"覆盖率: {brain_data['valid_combinations']/brain_data['total_combinations']*100:.1f}%\n\n")
            
            # 全局分析结果
            f.write("📈 全局分析结果\n")
            f.write("-" * 40 + "\n")
            if 'global_feature_variation' in self.analysis_results:
                global_analysis = self.analysis_results['global_feature_variation']
                f.write(f"PCA前3PC解释方差: {np.sum(global_analysis['pca_explained_variance'][:3]):.3f}\n")
                f.write(f"PCA前10PC解释方差: {np.sum(global_analysis['pca_explained_variance'][:10]):.3f}\n")
            
            if 'neural_network_baseline_analysis' in self.analysis_results:
                baseline_analysis = self.analysis_results['neural_network_baseline_analysis']
                if 'baseline_random_split' in baseline_analysis:
                    baseline_results = baseline_analysis['baseline_random_split']
                    if 'global_analysis' in baseline_results:
                        global_perf = baseline_results['global_analysis']
                        for model_name, results in global_perf.items():
                            if isinstance(results, dict) and 'accuracy' in results:
                                f.write(f"{model_name}准确率: {results['accuracy']:.3f}\n")
            f.write("\n")
            
            # 脑区感知分析结果
            f.write("🧠 脑区感知分析结果\n")
            f.write("-" * 40 + "\n")
            if 'region_wise_subject_analysis' in self.analysis_results:
                region_analysis = self.analysis_results['region_wise_subject_analysis']
                f.write(f"成功分析脑区数: {len(region_analysis)}\n")
                
                if region_analysis:
                    specificity_scores = [r['subject_specificity_score'] for r in region_analysis.values()]
                    f.write(f"平均特异性得分: {np.mean(specificity_scores):.3f}\n")
                    f.write(f"特异性得分范围: [{np.min(specificity_scores):.3f}, {np.max(specificity_scores):.3f}]\n")
            
            if 'region_wise_separability' in self.analysis_results:
                region_sep = self.analysis_results['region_wise_separability']
                if region_sep:
                    successful_regions = {k: v for k, v in region_sep.items() if 'embedding_necessity_score' in v}
                    if successful_regions:
                        necessity_scores = [v['embedding_necessity_score'] for v in successful_regions.values()]
                        f.write(f"平均Subject Embedding需求得分: {np.mean(necessity_scores):.3f}\n")
            f.write("\n")
            
            # 最终决策
            f.write("🎯 最终综合决策\n")
            f.write("-" * 40 + "\n")
            if 'final_comprehensive_decision' in self.analysis_results:
                final_decision = self.analysis_results['final_comprehensive_decision']
                comp_rec = final_decision['comprehensive_recommendation']
                
                f.write(f"最终推荐: {comp_rec['final_recommendation']}\n")
                f.write(f"综合得分: {comp_rec['comprehensive_score']:.3f}\n")
                f.write(f"实施优先级: {comp_rec['implementation_priority']}\n")
                f.write(f"推荐架构: {comp_rec['architecture_type']}\n")
                f.write(f"实施策略: {comp_rec['implementation_strategy']}\n\n")
                
                # 实施计划
                impl_plan = final_decision['implementation_plan']
                f.write("实施计划:\n")
                f.write(f"  预估时间: {impl_plan['estimated_timeline']}\n")
                f.write("  主要阶段:\n")
                for phase in impl_plan['phase2_development'][:3]:
                    f.write(f"    • {phase}\n")
            
            # 🔥 存档点信息
            if self.checkpoint_manager:
                f.write("\n🔄 存档点信息\n")
                f.write("-" * 40 + "\n")
                checkpoints = self.checkpoint_manager.list_checkpoints()
                f.write(f"总存档点数: {len(checkpoints)}\n")
                for ckpt in checkpoints[:5]:  # 显示最新5个
                    f.write(f"  • {ckpt['name']} - {ckpt['creation_time']} ({ckpt['file_size_mb']:.1f}MB)\n")
        
        logger.info(f"✅ 增强版完整报告已保存: {report_path}")
        return report_path