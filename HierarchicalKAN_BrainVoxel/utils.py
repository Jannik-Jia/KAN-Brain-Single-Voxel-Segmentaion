#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块 - 包含一般性的工具函数和日志设置
"""

import os
import logging
from datetime import datetime

def setup_logger(log_dir):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"brainvoxel_analysis_{timestamp}.log")
    
    # 创建一个logger
    logger = logging.getLogger('brainvoxel_analysis')
    logger.setLevel(logging.INFO)
    
    # 清除已存在的处理器
    if logger.handlers:
        logger.handlers = []
    
    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建一个控制台处理器，用于在控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def analyze_pca_variance(data, threshold=0.95, plot=True, save_path=None, logger=None):
    """
    分析PCA方差解释率，确定最佳组件数
    
    参数：
        data: 输入数据
        threshold: 累积方差阈值
        plot: 是否绘制图表
        save_path: 保存图表的路径
        logger: 日志记录器
        
    返回：
        n_components: 达到阈值所需的组件数
        explained_variance_ratio: 各组件的方差解释率
        cumulative_variance: 累积方差解释率
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # 计算完整PCA
    pca_full = PCA()
    pca_full.fit(data)
    
    # 计算累积解释方差
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # 确定达到阈值所需的组件数
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    
    msg = f"PCA方差分析: 达到{threshold*100:.1f}%解释率需要{n_components}个组件"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    if plot:
        plt.figure(figsize=(12, 6))
        
        # 绘制方差解释率
        plt.subplot(1, 2, 1)
        plt.plot(pca_full.explained_variance_ratio_, 'o-', markersize=4)
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        
        # 绘制累积方差解释率
        plt.subplot(1, 2, 2)
        plt.plot(cumulative_variance, 'o-', markersize=4)
        plt.axhline(y=threshold, color='r', linestyle='--', 
                 label=f'Threshold: {threshold}')
        plt.axvline(x=n_components-1, color='g', linestyle='--',
                 label=f'Components: {n_components}')
        plt.title('PCA Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'pca_variance_analysis.png'))
        plt.close()
        
    return n_components, pca_full.explained_variance_ratio_, cumulative_variance