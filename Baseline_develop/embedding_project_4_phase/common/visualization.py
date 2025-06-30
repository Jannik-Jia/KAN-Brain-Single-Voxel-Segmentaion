#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化基础模块
提供通用的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """基础可视化器"""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_correlation_matrix(self, matrix: np.ndarray, title: str = "Correlation Matrix",
                               labels: Optional[List[str]] = None, 
                               save_name: Optional[str] = None, 
                               figsize: Tuple[int, int] = (10, 8)):
        """绘制相关性矩阵热图"""
        plt.figure(figsize=figsize)
        
        sns.heatmap(matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Correlation'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()
    
    def plot_distribution(self, data: np.ndarray, title: str = "Distribution",
                         xlabel: str = "Value", ylabel: str = "Frequency",
                         bins: int = 50, save_name: Optional[str] = None):
        """绘制分布直方图"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加统计线
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.3f}')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()
    
    def plot_scatter_2d(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       title: str = "2D Scatter Plot", 
                       xlabel: str = "Component 1", ylabel: str = "Component 2",
                       save_name: Optional[str] = None):
        """绘制2D散点图"""
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                                alpha=0.6, s=50)
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()
    
    def plot_bar_chart(self, values: List[float], labels: List[str],
                      title: str = "Bar Chart", ylabel: str = "Value",
                      save_name: Optional[str] = None, colors: Optional[List[str]] = None):
        """绘制柱状图"""
        plt.figure(figsize=(12, 6))
        
        x = range(len(values))
        
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        
        bars = plt.bar(x, values, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()
    
    def plot_line_chart(self, x: np.ndarray, y: np.ndarray, 
                       title: str = "Line Chart",
                       xlabel: str = "X", ylabel: str = "Y",
                       save_name: Optional[str] = None,
                       multiple_lines: Optional[Dict[str, np.ndarray]] = None):
        """绘制折线图"""
        plt.figure(figsize=(10, 6))
        
        if multiple_lines:
            for label, data in multiple_lines.items():
                plt.plot(x, data, marker='o', label=label, linewidth=2)
        else:
            plt.plot(x, y, marker='o', linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if multiple_lines:
            plt.legend()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()