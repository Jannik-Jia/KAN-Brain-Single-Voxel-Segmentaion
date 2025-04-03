# src/pseudoinverse_model.py

import numpy as np
import time
import pickle
import warnings

from src.bilingual_logger import BilingualLogger


class PseudoInverseModel:
    """使用伪逆方法的线性模型类"""
    
    def __init__(self, num_classes=102, logger=None):
        """
        初始化伪逆模型
        
        参数:
            num_classes: 类别数量
            logger: 日志记录器
        """
        self.num_classes = num_classes
        self.weights = None  # 权重矩阵
        self.bias = None     # 偏置向量
        self.feature_importance = None  # 特征重要性
        self.logger = logger if logger else BilingualLogger()
        
        self.logger.info(f"初始化伪逆模型，类别数：{num_classes}", 
                       f"Initialized pseudo-inverse model with {num_classes} classes")
    
    def fit(self, X, y, regularization=None, alpha=0.0):
        """
        使用伪逆方法拟合线性模型
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 标签，形状为(n_samples,)
            regularization: 正则化方法，None, 'l2'或'truncated'
            alpha: 正则化强度，用于L2正则化
        """
        self.logger.info("开始拟合伪逆模型", "Starting to fit pseudo-inverse model")
        start_time = time.time()
        
        # 将标签转换为独热编码
        y_one_hot = self._to_one_hot(y)
        
        # 添加偏置项
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        
        # 根据正则化方法计算伪逆
        if regularization == 'l2':
            # Tikhonov正则化（L2）
            self.logger.info(f"应用L2正则化，alpha={alpha}", 
                           f"Applying L2 regularization with alpha={alpha}")
            n_features = X_bias.shape[1]
            identity = np.eye(n_features)
            identity[-1, -1] = 0  # 不对偏置项正则化
            
            # 解析解：W = (X^T X + alpha*I)^(-1) X^T y
            XTX = X_bias.T @ X_bias
            XTX_reg = XTX + alpha * identity
            weights = np.linalg.solve(XTX_reg, X_bias.T @ y_one_hot)
        
        elif regularization == 'truncated':
            # 截断SVD伪逆
            self.logger.info(f"应用截断SVD伪逆，alpha={alpha}", 
                           f"Applying truncated SVD with threshold={alpha}")
            
            # 使用SVD计算伪逆
            U, s, Vh = np.linalg.svd(X_bias, full_matrices=False)
            
            # 截断小于阈值的奇异值
            s_threshold = alpha * max(s)
            s_inv = np.array([1/si if si > s_threshold else 0 for si in s])
            
            # 计算伪逆
            pinv_X = (Vh.T * s_inv) @ U.T
            weights = pinv_X @ y_one_hot
        
        else:
            # 标准伪逆
            self.logger.info("应用标准伪逆", "Applying standard pseudo-inverse")
            pinv_X = np.linalg.pinv(X_bias)
            weights = pinv_X @ y_one_hot
        
        # 分离权重和偏置
        self.weights = weights[:-1, :]
        self.bias = weights[-1, :]
        
        # 计算特征重要性（每个特征权重的平均绝对值）
        self.feature_importance = np.mean(np.abs(self.weights), axis=1)
        
        # 记录拟合时间
        elapsed_time = time.time() - start_time
        self.logger.info(f"伪逆模型拟合完成，耗时 {elapsed_time:.2f} 秒", 
                       f"Pseudo-inverse model fitting completed in {elapsed_time:.2f} seconds")
        
        # 计算训练误差
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        self.logger.info(f"训练集准确率: {accuracy:.4f}", f"Training accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        """
        使用拟合的模型进行预测
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            
        返回:
            预测的类别，形状为(n_samples,)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 计算线性输出
        logits = X @ self.weights + self.bias
        
        # 返回最大概率的类别
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            
        返回:
            类别概率，形状为(n_samples, n_classes)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 计算线性输出
        logits = X @ self.weights + self.bias
        
        # 应用softmax函数获取概率
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs
    
    def _to_one_hot(self, y):
        """将整数标签转换为独热编码"""
        n_samples = len(y)
        y_one_hot = np.zeros((n_samples, self.num_classes))
        
        for i, label in enumerate(y):
            label_idx = int(label) % self.num_classes  # 防止标签超出范围
            y_one_hot[i, label_idx] = 1
        
        return y_one_hot
    
    def save(self, filepath):
        """
        保存模型到文件
        
        参数:
            filepath: 保存路径
        """
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'feature_importance': self.feature_importance,
            'num_classes': self.num_classes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"模型已保存到: {filepath}", f"Model saved to: {filepath}")
    
    def load(self, filepath):
        """
        从文件加载模型
        
        参数:
            filepath: 加载路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.feature_importance = model_data['feature_importance']
        self.num_classes = model_data['num_classes']
        
        self.logger.info(f"模型已从 {filepath} 加载", f"Model loaded from: {filepath}")
        
        return self