
# src/feature_selector.py

import numpy as np
import os
import time
import pickle
import warnings

# 根据GPU可用性选择实现
from src.gpu_utils import xp, to_gpu, to_cpu, ensure_numpy, USE_GPU, has_cuda_ml

# 标准scikit-learn库
from sklearn.metrics import jaccard_score
from sklearn.model_selection import KFold

# 根据GPU可用性选择线性模型实现
if USE_GPU and has_cuda_ml():
    from cuml.linear_model import LogisticRegression
    # CuML还没有完全实现这些模型
    from sklearn.linear_model import Lasso, ElasticNet
    from sklearn.multiclass import OneVsRestClassifier
else:
    from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

# 根据GPU可用性选择预处理实现
if USE_GPU and has_cuda_ml():
    from cuml.preprocessing import StandardScaler
else:
    from sklearn.preprocessing import StandardScaler

from src.bilingual_logger import BilingualLogger

# 在feature_selector.py中添加PyTorch实现

import torch
import torch.nn as nn
import torch.optim as optim

class TorchElasticNet:
    """使用PyTorch实现的弹性网络"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # 初始化权重和偏置
        n_features = X.shape[1]
        weights = nn.Parameter(torch.zeros(n_features, 1, requires_grad=True))
        bias = nn.Parameter(torch.zeros(1, requires_grad=True))
        
        # 定义优化器
        optimizer = optim.Adam([weights, bias], lr=0.01)
        
        # 训练模型
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            
            # 前向传播
            y_pred = torch.matmul(X_tensor, weights) + bias
            
            # 计算MSE损失
            mse_loss = torch.mean((y_pred - y_tensor) ** 2)
            
            # 添加正则化项
            l1_penalty = self.alpha * self.l1_ratio * torch.sum(torch.abs(weights))
            l2_penalty = self.alpha * (1 - self.l1_ratio) * torch.sum(weights ** 2)
            
            # 总损失
            loss = mse_loss + l1_penalty + l2_penalty
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 检查收敛
            if epoch > 0 and abs(prev_loss - loss.item()) < self.tol:
                break
                
            prev_loss = loss.item()
        
        # 保存系数和截距
        self.coef_ = weights.detach().numpy()
        self.intercept_ = bias.detach().item()
        
        return self

    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        return (torch.matmul(X_tensor, torch.FloatTensor(self.coef_)) + self.intercept_).numpy()


class FeatureSelector:
    """使用LASSO或弹性网络进行特征选择的类，支持GPU加速"""
    
    def __init__(self, method='lasso', selection_mode='threshold', selection_threshold=0.01,
                max_features=100, l1_ratio=1.0, cv_folds=5, random_state=42,
                scaling_before_selection=True, selection_metric='coefficient',
                max_iter=1000, tol=1e-4, logger=None):

        self.method = method
        self.selection_mode = selection_mode
        self.selection_threshold = selection_threshold
        self.max_features = max_features
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scaling_before_selection = scaling_before_selection
        self.selection_metric = selection_metric
        self.max_iter = max_iter  # 添加最大迭代次数参数
        self.tol = tol  # 添加收敛阈值参数
        self.logger = logger if logger else BilingualLogger()
        self.selection_frequency = None
        self.jaccard_matrix = None
        
        # 特征选择模型
        self.model = None
        # 所选特征的索引
        self.selected_indices = None
        # 特征重要性
        self.feature_importance = None
        # 标准化器
        self.scaler = None
        
        gpu_status = "启用" if USE_GPU else "未启用"
        self.logger.info(f"特征选择器初始化: 方法={method}, 选择模式={selection_mode}, "
                    f"阈值/最大特征数={selection_threshold if selection_mode=='threshold' else max_features}, "
                    f"最大迭代次数={max_iter}, 收敛阈值={tol}, GPU加速: {gpu_status}",
                    f"Feature selector initialized: method={method}, selection_mode={selection_mode}, "
                    f"threshold/max_features={selection_threshold if selection_mode=='threshold' else max_features}, "
                    f"max_iter={max_iter}, tol={tol}, GPU acceleration: {gpu_status}")
                    
    def fit(self, X, y, class_weight=None):
        """
        拟合特征选择模型并选择特征
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签，形状为(n_samples,)
            class_weight: 类别权重，用于多分类问题
            
        返回:
            self: 特征选择器自身
        """
        start_time = time.time()
        self.logger.info(f"开始特征选择, 输入特征维度: {X.shape[1]}", 
                    f"Starting feature selection, input dimension: {X.shape[1]}")
        
        # 将数据移到CPU，因为sklearn不支持GPU
        X_cpu = ensure_numpy(X)
        y_cpu = ensure_numpy(y)
        
        # 标准化特征
        if self.scaling_before_selection:
            self.logger.info("应用标准化预处理", "Applying standardization preprocessing")
            self.scaler = StandardScaler()
            X_cpu = self.scaler.fit_transform(X_cpu)
        
        # 创建模型 - 修改这部分使用PyTorch实现
        if self.method == 'elastic_net_torch':
            # 使用PyTorch实现的弹性网络
            self.logger.info(f"使用PyTorch实现的弹性网络，l1_ratio={self.l1_ratio}, max_iter={self.max_iter}", 
                        f"Using PyTorch implementation of ElasticNet, l1_ratio={self.l1_ratio}, max_iter={self.max_iter}")
            
            # 创建多分类模型
            n_classes = len(np.unique(y_cpu))
            
            if n_classes > 2:
                # 多分类问题 - 一对多方法
                models = []
                coefs = []
                
                for i in range(n_classes):
                    # 创建二分类标签 (当前类别 vs 其他)
                    binary_y = np.where(y_cpu == i, 1, 0)
                    
                    # 拟合模型
                    model = TorchElasticNet(
                        alpha=self.selection_threshold,
                        l1_ratio=self.l1_ratio,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=self.random_state
                    )
                    model.fit(X_cpu, binary_y)
                    models.append(model)
                    coefs.append(model.coef_)
                
                # 合并系数
                self.model = models
                coefs = np.vstack([c.flatten() for c in coefs])
                
            else:
                # 二分类问题
                self.model = TorchElasticNet(
                    alpha=self.selection_threshold,
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state
                )
                self.model.fit(X_cpu, y_cpu)
                coefs = self.model.coef_
        
        # 保留原有的LASSO和elastic_net实现，但添加迭代次数和收敛参数
        elif self.method == 'lasso':
            # 对于多分类问题，使用OneVsRestClassifier包装LogisticRegression
            if len(np.unique(y_cpu)) > 2:
                # 根据是否使用GPU选择不同的实现
                if USE_GPU and has_cuda_ml():
                    # cuML版本 - 使用qn solver
                    base_model = LogisticRegression(
                        penalty='l1', solver='qn', C=1.0/self.selection_threshold,
                        random_state=self.random_state, max_iter=self.max_iter, tol=self.tol)
                else:
                    # sklearn版本 - 使用liblinear solver
                    base_model = sklearn.linear_model.LogisticRegression(
                        penalty='l1', solver='liblinear', C=1.0/self.selection_threshold,
                        random_state=self.random_state, class_weight=class_weight,
                        max_iter=self.max_iter, tol=self.tol)
                    
                self.model = OneVsRestClassifier(base_model)
            else:
                # 二分类问题
                if USE_GPU and has_cuda_ml():
                    # cuML版本
                    self.model = LogisticRegression(
                        penalty='l1', solver='qn', C=1.0/self.selection_threshold,
                        random_state=self.random_state, max_iter=self.max_iter, tol=self.tol)
                else:
                    # sklearn版本
                    self.model = sklearn.linear_model.LogisticRegression(
                        penalty='l1', solver='liblinear', C=1.0/self.selection_threshold,
                        random_state=self.random_state, class_weight=class_weight,
                        max_iter=self.max_iter, tol=self.tol)
        
        elif self.method == 'elastic_net':
            # 对于多分类问题
            if len(np.unique(y_cpu)) > 2:
                if USE_GPU and has_cuda_ml():
                    # cuML不支持ElasticNet的LogisticRegression，所以我们使用PyTorch实现
                    
                    self.method = 'elastic_net_torch'  # 修改方法名
                else:
                    # sklearn版本使用saga solver
                    base_model = sklearn.linear_model.LogisticRegression(
                        penalty='elasticnet', solver='saga', C=1.0/self.selection_threshold,
                        l1_ratio=self.l1_ratio, random_state=self.random_state, 
                        class_weight=class_weight, max_iter=self.max_iter, tol=self.tol)
                    
                self.model = OneVsRestClassifier(base_model)
            else:
                # 二分类问题
                if USE_GPU and has_cuda_ml():
                    # cuML不支持ElasticNet的LogisticRegression，使用PyTorch实现
                    self.logger.info("cuML不支持ElasticNet, 使用PyTorch实现", 
                                "ElasticNet not supported in cuML, using PyTorch implementation")
                    return self.fit(X, y, class_weight)  # 递归调用使用PyTorch实现
                else:
                    # sklearn版本
                    self.model = sklearn.linear_model.LogisticRegression(
                        penalty='elasticnet', solver='saga', C=1.0/self.selection_threshold,
                        l1_ratio=self.l1_ratio, random_state=self.random_state,
                        class_weight=class_weight, max_iter=self.max_iter, tol=self.tol)
        
        # 拟合模型
        if self.method != 'elastic_net_torch':  # 非PyTorch实现才需要拟合
            self.model.fit(X_cpu, y_cpu)

        # 获取特征重要性
        if self.method == 'elastic_net_torch':
            if isinstance(self.model, list):
                coefs = np.vstack([model.coef_.flatten() for model in self.model])
                self.feature_importance = np.mean(np.abs(coefs), axis=0)
            else:
                self.feature_importance = np.abs(self.model.coef_).flatten()
        else:
            # 原始代码逻辑
            if hasattr(self.model, 'coef_'):
                coefs = self.model.coef_
            else:
                # 对于OneVsRestClassifier，特征系数在estimators_中
                coefs = np.vstack([est.coef_ for est in self.model.estimators_])
            
            # 计算特征重要性
            if self.selection_metric == 'coefficient':
                # 使用系数绝对值的平均值作为特征重要性
                self.feature_importance = np.mean(np.abs(coefs), axis=0)
            else:
                # 其他特征重要性度量
                self.feature_importance = np.mean(np.abs(coefs), axis=0)
        
        # 选择特征
        if self.selection_mode == 'threshold':
            # 根据阈值选择特征
            self.selected_indices = np.where(self.feature_importance > self.selection_threshold)[0]
        else:
            # 固定数量模式
            if self.max_features < X_cpu.shape[1]:
                # 选择top-k特征
                self.selected_indices = np.argsort(self.feature_importance)[-self.max_features:]
            else:
                # 如果max_features大于等于特征数，保留所有特征
                self.selected_indices = np.arange(X_cpu.shape[1])
        
        # 确保索引已排序
        self.selected_indices = np.sort(self.selected_indices)
        
        # 转换到GPU以便后续操作
        self.feature_importance = to_gpu(self.feature_importance)
        self.selected_indices = to_gpu(self.selected_indices)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"特征选择完成，选择了 {len(self.selected_indices)}/{X.shape[1]} 个特征，"
                      f"耗时 {elapsed_time:.2f} 秒",
                      f"Feature selection completed, selected {len(self.selected_indices)}/{X.shape[1]} "
                      f"features in {elapsed_time:.2f} seconds")
        
        return self
    
    def transform(self, X):
        """
        使用选定的特征转换数据
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            
        返回:
            X_selected: 选择特征后的数据，形状为(n_samples, n_selected_features)
        """
        if self.selected_indices is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 先移到CPU进行处理
        X_cpu = ensure_numpy(X)
        
        # 应用相同的标准化
        if self.scaling_before_selection and self.scaler is not None:
            X_cpu = self.scaler.transform(X_cpu)
        
        # 获取所选特征的索引
        selected_indices_cpu = ensure_numpy(self.selected_indices)
        
        # 选择特征
        X_selected = X_cpu[:, selected_indices_cpu]
        
        # 结果转回GPU
        return to_gpu(X_selected)
    
    def fit_transform(self, X, y, class_weight=None):
        """
        拟合特征选择模型并转换数据
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签，形状为(n_samples,)
            class_weight: 类别权重，用于多分类问题
            
        返回:
            X_selected: 选择特征后的数据，形状为(n_samples, n_selected_features)
        """
        self.fit(X, y, class_weight)
        return self.transform(X)
    
    def get_support(self):
        """
        获取所选特征的布尔掩码
        
        返回:
            support: 布尔掩码，指示每个特征是否被选择
        """
        if self.selected_indices is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 将索引转到CPU
        selected_indices_cpu = ensure_numpy(self.selected_indices)
        feature_importance_cpu = ensure_numpy(self.feature_importance)
        
        support = np.zeros(feature_importance_cpu.shape[0], dtype=bool)
        support[selected_indices_cpu] = True
        return support
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
            importance: 特征重要性向量
        """
        if self.feature_importance is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        return ensure_numpy(self.feature_importance)
    
    def get_selected_indices(self):
        """
        获取所选特征的索引
        
        返回:
            indices: 所选特征的索引
        """
        if self.selected_indices is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        return ensure_numpy(self.selected_indices)
    
    def evaluate_stability(self, X, y, n_splits=None):
        """
        评估特征选择的稳定性
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签，形状为(n_samples,)
            n_splits: 交叉验证折数，如果为None则使用初始化时设置的值
            
        返回:
            stability_metrics: 包含稳定性指标的字典
        """
        if n_splits is None:
            n_splits = self.cv_folds
        
        self.logger.info(f"评估特征选择稳定性，使用{n_splits}折交叉验证", 
                      f"Evaluating feature selection stability using {n_splits}-fold CV")
        
        # 确保数据在CPU上
        X_cpu = ensure_numpy(X)
        y_cpu = ensure_numpy(y)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        selected_features_masks = []
        
        # 在每个折上进行特征选择
        for train_idx, _ in kf.split(X_cpu):
            X_train, y_train = X_cpu[train_idx], y_cpu[train_idx]
            
            # 创建相同参数的选择器
            selector = FeatureSelector(
                method=self.method,
                selection_mode=self.selection_mode,
                selection_threshold=self.selection_threshold,
                max_features=self.max_features,
                l1_ratio=self.l1_ratio
            )
            
            # 拟合选择器
            selector.fit(X_train, y_train)
            
            # 获取所选特征的掩码
            mask = selector.get_support()
            selected_features_masks.append(mask)
        
        # 计算Jaccard相似度矩阵
        n_folds = len(selected_features_masks)
        jaccard_matrix = np.zeros((n_folds, n_folds))
        
        for i in range(n_folds):
            for j in range(i, n_folds):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                else:
                    similarity = jaccard_score(
                        selected_features_masks[i], 
                        selected_features_masks[j], 
                        average='binary'
                    )
                    jaccard_matrix[i, j] = similarity
                    jaccard_matrix[j, i] = similarity
        
        # 计算平均Jaccard相似度
        avg_jaccard = np.mean(jaccard_matrix[np.triu_indices(n_folds, k=1)])
        
        # 计算特征选择频率
        selection_frequency = np.mean(selected_features_masks, axis=0)
        
        # 计算稳定性指标
        stability_metrics = {
            'avg_jaccard_similarity': avg_jaccard,
            'selection_frequency': selection_frequency,
            'jaccard_matrix': jaccard_matrix
        }
        
        self.selection_frequency = to_gpu(selection_frequency)
        self.jaccard_matrix = to_gpu(jaccard_matrix)
        
        self.logger.info(f"特征选择稳定性评估完成，平均Jaccard相似度: {avg_jaccard:.4f}",
                      f"Feature selection stability evaluation completed, "
                      f"average Jaccard similarity: {avg_jaccard:.4f}")
        
        return stability_metrics
    
    def save(self, filepath):
        """
        保存特征选择器到文件
        
        参数:
            filepath: 保存路径
        """
        # 确保数据在CPU上
        selected_indices = ensure_numpy(self.selected_indices) if self.selected_indices is not None else None
        feature_importance = ensure_numpy(self.feature_importance) if self.feature_importance is not None else None
        selection_frequency = ensure_numpy(self.selection_frequency) if self.selection_frequency is not None else None
        jaccard_matrix = ensure_numpy(self.jaccard_matrix) if self.jaccard_matrix is not None else None
        
        model_data = {
            'method': self.method,
            'selection_mode': self.selection_mode,
            'selection_threshold': self.selection_threshold,
            'max_features': self.max_features,
            'l1_ratio': self.l1_ratio,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'scaling_before_selection': self.scaling_before_selection,
            'selection_metric': self.selection_metric,
            'selected_indices': selected_indices,
            'feature_importance': feature_importance,
            'selection_frequency': selection_frequency,
            'jaccard_matrix': jaccard_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"特征选择器已保存到: {filepath}", f"Feature selector saved to: {filepath}")
    
    def load(self, filepath):
        """
        从文件加载特征选择器
        
        参数:
            filepath: 加载路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.method = model_data['method']
        self.selection_mode = model_data['selection_mode']
        self.selection_threshold = model_data['selection_threshold']
        self.max_features = model_data['max_features']
        self.l1_ratio = model_data['l1_ratio']
        self.cv_folds = model_data['cv_folds']
        self.random_state = model_data['random_state']
        self.scaling_before_selection = model_data['scaling_before_selection']
        self.selection_metric = model_data['selection_metric']
        
        # 转移数据到GPU
        self.selected_indices = to_gpu(model_data['selected_indices'])
        self.feature_importance = to_gpu(model_data['feature_importance'])
        
        # 加载稳定性评估结果（如果有的话）
        if 'selection_frequency' in model_data and model_data['selection_frequency'] is not None:
            self.selection_frequency = to_gpu(model_data['selection_frequency'])
        
        if 'jaccard_matrix' in model_data and model_data['jaccard_matrix'] is not None:
            self.jaccard_matrix = to_gpu(model_data['jaccard_matrix'])
        
        self.logger.info(f"特征选择器已从 {filepath} 加载", f"Feature selector loaded from: {filepath}")
        
        return self