"""
类别级性能分析工具
针对每个解剖标签计算详细的性能指标
"""

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ClassPerformanceAnalyzer:
    """类别级性能深度分析工具"""
    
    def __init__(self, softmax_path: str, info_path: str, label_path: str):
        """
        初始化分析器
        
        Args:
            softmax_path: softmax预测文件路径
            info_path: 信息JSON文件路径
            label_path: 真实标签文件路径
        """
        # 加载数据
        print("Loading data...")
        self.softmax_nii = nib.load(softmax_path)
        self.softmax = self.softmax_nii.get_fdata()
        
        self.label_nii = nib.load(label_path)
        self.labels = self.label_nii.get_fdata().astype(int)
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.n_classes = self.info['n_classes']
        self.class_names = self.info.get('class_names', list(range(self.n_classes)))
        self.include_background = self.info['include_background']
        
        # 预测结果
        self.predictions = np.argmax(self.softmax, axis=-1)
        
        # 识别有效体素（被采样的体素）
        self._identify_valid_voxels()
        
        # 计算性能指标
        self._compute_metrics()
        
        print(f"Analysis ready: {len(self.valid_indices)} valid voxels, {self.n_classes} classes")
    
    def _identify_valid_voxels(self):
        """识别有效的采样体素"""
        # 有softmax预测的体素
        max_prob = np.max(self.softmax, axis=-1)
        self.valid_mask = max_prob > 0
        
        # 如果排除了背景，需要额外处理
        if not self.include_background:
            # 假设背景标签是0或14（根据你的代码）
            self.valid_mask = self.valid_mask & (self.labels != 14)
        
        # 获取有效体素的索引
        self.valid_indices = np.where(self.valid_mask.flatten())[0]
        
        # 展平的有效数据
        self.valid_predictions = self.predictions.flatten()[self.valid_indices]
        self.valid_labels = self.labels.flatten()[self.valid_indices]
        self.valid_softmax = self.softmax.reshape(-1, self.n_classes)[self.valid_indices]
    
    def _compute_metrics(self):
        """计算所有性能指标"""
        print("Computing performance metrics...")
        
        # 总体指标
        self.overall_accuracy = (self.valid_predictions == self.valid_labels).mean()
        
        # 类别级指标
        self.precision, self.recall, self.f1, self.support = precision_recall_fscore_support(
            self.valid_labels, self.valid_predictions, 
            labels=range(self.n_classes), zero_division=0
        )
        
        # 混淆矩阵
        self.conf_matrix = confusion_matrix(
            self.valid_labels, self.valid_predictions,
            labels=range(self.n_classes)
        )
        
        # 类别级准确率
        self.class_accuracy = []
        for i in range(self.n_classes):
            mask = self.valid_labels == i
            if mask.sum() > 0:
                acc = (self.valid_predictions[mask] == i).mean()
            else:
                acc = np.nan
            self.class_accuracy.append(acc)
        
        # IoU (Intersection over Union) / Dice系数
        self.iou_scores = []
        self.dice_scores = []
        
        for i in range(self.n_classes):
            pred_i = self.valid_predictions == i
            true_i = self.valid_labels == i
            
            intersection = (pred_i & true_i).sum()
            union = (pred_i | true_i).sum()
            
            if union > 0:
                iou = intersection / union
                dice = 2 * intersection / (pred_i.sum() + true_i.sum())
            else:
                iou = np.nan
                dice = np.nan
            
            self.iou_scores.append(iou)
            self.dice_scores.append(dice)
    
    def generate_performance_report(self, save_path: Optional[str] = None):
        """
        生成综合性能报告
        
        Args:
            save_path: 保存路径（CSV格式）
        """
        # 创建性能DataFrame
        metrics_df = pd.DataFrame({
            'Class_ID': range(self.n_classes),
            'Class_Name': [self.class_names[i] if i < len(self.class_names) else f'Class_{i}' 
                          for i in range(self.n_classes)],
            'Support': self.support,
            'Accuracy': self.class_accuracy,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1_Score': self.f1,
            'IoU': self.iou_scores,
            'Dice': self.dice_scores
        })
        
        # 添加排名
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice']:
            metrics_df[f'{metric}_Rank'] = metrics_df[metric].rank(ascending=False, method='min')
        
        # 按F1分数排序
        metrics_df = metrics_df.sort_values('F1_Score', ascending=False)
        
        if save_path:
            metrics_df.to_csv(save_path, index=False)
            print(f"Performance report saved to {save_path}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Overall Accuracy: {self.overall_accuracy:.4f}")
        print(f"Mean F1 Score: {np.nanmean(self.f1):.4f}")
        print(f"Mean IoU: {np.nanmean(self.iou_scores):.4f}")
        print(f"Mean Dice: {np.nanmean(self.dice_scores):.4f}")
        
        print("\nTop 5 Best Performing Classes:")
        print(metrics_df[['Class_Name', 'F1_Score', 'Support']].head())
        
        print("\nTop 5 Worst Performing Classes (with support > 100):")
        worst_df = metrics_df[metrics_df['Support'] > 100].tail()
        print(worst_df[['Class_Name', 'F1_Score', 'Support']])
        
        return metrics_df
    
    def visualize_confusion_matrix(self, top_n: int = 20, 
                                  save_path: Optional[str] = None):
        """
        可视化混淆矩阵
        
        Args:
            top_n: 显示前N个最常见的类别
            save_path: 保存路径
        """
        # 选择top_n个最常见的类别
        class_counts = self.support
        top_classes = np.argsort(class_counts)[-top_n:][::-1]
        
        # 提取子混淆矩阵
        sub_conf_matrix = self.conf_matrix[np.ix_(top_classes, top_classes)]
        
        # 创建标签
        labels = [f"{self.class_names[i] if i < len(self.class_names) else f'C{i}'}" 
                 for i in top_classes]
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 原始计数
        ax = axes[0]
        sns.heatmap(sub_conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title(f'Confusion Matrix - Top {top_n} Classes (Counts)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # 2. 归一化（按行）
        ax = axes[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_matrix = sub_conf_matrix / sub_conf_matrix.sum(axis=1, keepdims=True)
            normalized_matrix = np.nan_to_num(normalized_matrix)
        
        sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        ax.set_title(f'Normalized Confusion Matrix - Top {top_n} Classes')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        plt.suptitle('Confusion Matrix Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_class_performance(self, save_path: Optional[str] = None):
        """
        可视化类别性能对比
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 过滤掉没有样本的类别
        valid_classes = [i for i in range(self.n_classes) if self.support[i] > 0]
        
        # 1. Precision vs Recall散点图
        ax = axes[0, 0]
        scatter = ax.scatter(self.recall[valid_classes], 
                           self.precision[valid_classes],
                           s=np.log10(self.support[valid_classes] + 1) * 50,
                           c=self.f1[valid_classes], cmap='viridis',
                           alpha=0.6)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall (size=log(support))')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        plt.colorbar(scatter, ax=ax, label='F1 Score')
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # 2. F1 Score分布
        ax = axes[0, 1]
        f1_valid = [self.f1[i] for i in valid_classes]
        ax.hist(f1_valid, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(f1_valid), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(f1_valid):.3f}')
        ax.axvline(np.median(f1_valid), color='blue', linestyle='--', 
                  label=f'Median: {np.median(f1_valid):.3f}')
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Number of Classes')
        ax.set_title('F1 Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. IoU vs Dice相关性
        ax = axes[0, 2]
        iou_valid = [self.iou_scores[i] for i in valid_classes 
                    if not np.isnan(self.iou_scores[i])]
        dice_valid = [self.dice_scores[i] for i in valid_classes 
                     if not np.isnan(self.dice_scores[i])]
        
        if len(iou_valid) > 0:
            ax.scatter(iou_valid, dice_valid, alpha=0.6)
            ax.set_xlabel('IoU Score')
            ax.set_ylabel('Dice Score')
            ax.set_title('IoU vs Dice Correlation')
            ax.grid(True, alpha=0.3)
            
            # 添加理论曲线
            iou_theory = np.linspace(0, 1, 100)
            dice_theory = 2 * iou_theory / (1 + iou_theory)
            ax.plot(iou_theory, dice_theory, 'r--', alpha=0.5, label='Theoretical')
            ax.legend()
        
        # 4. 类别性能热图
        ax = axes[1, 0]
        # 选择有足够支持的类别
        significant_classes = [i for i in valid_classes if self.support[i] > 100]
        
        if len(significant_classes) > 0:
            perf_matrix = np.array([
                [self.precision[i] for i in significant_classes],
                [self.recall[i] for i in significant_classes],
                [self.f1[i] for i in significant_classes],
                [self.iou_scores[i] if not np.isnan(self.iou_scores[i]) else 0 
                 for i in significant_classes]
            ])
            
            im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_yticks(range(4))
            ax.set_yticklabels(['Precision', 'Recall', 'F1', 'IoU'])
            ax.set_xticks(range(len(significant_classes)))
            ax.set_xticklabels([f'C{i}' for i in significant_classes], rotation=45)
            ax.set_title('Performance Metrics Heatmap (classes with >100 samples)')
            plt.colorbar(im, ax=ax)
        
        # 5. Support vs Performance
        ax = axes[1, 1]
        support_log = np.log10(self.support[valid_classes] + 1)
        ax.scatter(support_log, self.f1[valid_classes], alpha=0.6)
        ax.set_xlabel('Log10(Support + 1)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Sample Size Effect on Performance')
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(support_log, self.f1[valid_classes], 1)
        p = np.poly1d(z)
        ax.plot(support_log, p(support_log), "r--", alpha=0.5, 
               label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
        ax.legend()
        
        # 6. 类别错误分析
        ax = axes[1, 2]
        # 计算每个类别最常被误分类为哪个类别
        misclass_analysis = []
        for i in valid_classes[:10]:  # Top 10 classes
            if self.support[i] > 0:
                row = self.conf_matrix[i, :]
                row[i] = 0  # 排除正确预测
                if row.sum() > 0:
                    most_confused = np.argmax(row)
                    confusion_rate = row[most_confused] / self.support[i]
                    misclass_analysis.append({
                        'True': i,
                        'Confused_with': most_confused,
                        'Rate': confusion_rate
                    })
        
        if misclass_analysis:
            df_misclass = pd.DataFrame(misclass_analysis)
            bars = ax.bar(range(len(df_misclass)), df_misclass['Rate'])
            ax.set_xticks(range(len(df_misclass)))
            ax.set_xticklabels([f"{r['True']}→{r['Confused_with']}" 
                               for _, r in df_misclass.iterrows()], rotation=45)
            ax.set_ylabel('Confusion Rate')
            ax.set_title('Most Common Misclassifications')
            
            # 着色
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.Reds(df_misclass.iloc[i]['Rate']))
        
        plt.suptitle('Class Performance Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_difficult_regions(self, save_path: Optional[str] = None):
        """
        分析困难区域（容易出错的区域）
        
        Args:
            save_path: 保存路径
        """
        # 创建错误掩码
        error_mask_flat = self.valid_predictions != self.valid_labels
        
        # 还原到3D
        error_volume = np.zeros(self.predictions.shape, dtype=bool)
        error_volume_flat = error_volume.flatten()
        error_volume_flat[self.valid_indices] = error_mask_flat
        error_volume = error_volume_flat.reshape(self.predictions.shape)
        
        # 计算不确定性
        entropy = -np.sum(self.softmax * np.log(self.softmax + 1e-10), axis=-1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 选择代表性切片
        slice_idx = error_volume.shape[2] // 2
        
        # 1. 错误分布热图
        ax = axes[0, 0]
        im = ax.imshow(error_volume[:, :, slice_idx].T, cmap='Reds', 
                      origin='lower', alpha=0.7)
        ax.set_title('Error Distribution')
        ax.axis('off')
        
        # 2. 错误密度图（使用高斯滤波平滑）
        from scipy.ndimage import gaussian_filter
        ax = axes[0, 1]
        error_density = gaussian_filter(error_volume.astype(float), sigma=5)
        im = ax.imshow(error_density[:, :, slice_idx].T, cmap='hot', 
                      origin='lower')
        ax.set_title('Error Density (smoothed)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 3. 错误与不确定性的关系
        ax = axes[0, 2]
        # 分bin统计
        uncertainty_bins = np.percentile(entropy[self.valid_mask], 
                                       np.linspace(0, 100, 11))
        bin_errors = []
        bin_centers = []
        
        for i in range(len(uncertainty_bins)-1):
            mask = (entropy.flatten()[self.valid_indices] >= uncertainty_bins[i]) & \
                   (entropy.flatten()[self.valid_indices] < uncertainty_bins[i+1])
            if mask.sum() > 0:
                bin_errors.append(error_mask_flat[mask].mean())
                bin_centers.append((uncertainty_bins[i] + uncertainty_bins[i+1]) / 2)
        
        ax.plot(bin_centers, bin_errors, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Uncertainty (Entropy)')
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Rate vs Uncertainty')
        ax.grid(True, alpha=0.3)
        
        # 4. 类别特定错误率
        ax = axes[1, 0]
        class_error_rates = []
        for i in range(self.n_classes):
            mask = self.valid_labels == i
            if mask.sum() > 0:
                error_rate = error_mask_flat[mask].mean()
                class_error_rates.append(error_rate)
            else:
                class_error_rates.append(0)
        
        # 显示错误率最高的10个类别
        top_error_classes = np.argsort(class_error_rates)[-10:][::-1]
        ax.barh(range(10), [class_error_rates[i] for i in top_error_classes])
        ax.set_yticks(range(10))
        ax.set_yticklabels([f'Class {i}' for i in top_error_classes])
        ax.set_xlabel('Error Rate')
        ax.set_title('Top 10 Classes by Error Rate')
        
        # 5. 空间错误聚类
        ax = axes[1, 1]
        # 计算局部错误率
        from scipy.ndimage import uniform_filter
        local_error_rate = uniform_filter(error_volume.astype(float), size=10)
        
        im = ax.imshow(local_error_rate[:, :, slice_idx].T, cmap='RdYlGn_r', 
                      origin='lower', vmin=0, vmax=1)
        ax.set_title('Local Error Rate (10x10x10 window)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 6. 边界vs非边界错误
        ax = axes[1, 2]
        from scipy.ndimage import sobel
        
        # 检测边界
        grad_x = sobel(self.labels.astype(float), axis=0)
        grad_y = sobel(self.labels.astype(float), axis=1)
        grad_z = sobel(self.labels.astype(float), axis=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # 定义边界体素
        boundary_threshold = np.percentile(gradient_magnitude[self.valid_mask], 90)
        is_boundary = gradient_magnitude.flatten()[self.valid_indices] > boundary_threshold
        
        # 计算边界和非边界的错误率
        boundary_error_rate = error_mask_flat[is_boundary].mean() if is_boundary.sum() > 0 else 0
        non_boundary_error_rate = error_mask_flat[~is_boundary].mean() if (~is_boundary).sum() > 0 else 0
        
        categories = ['Boundary\nRegions', 'Non-boundary\nRegions']
        error_rates = [boundary_error_rate, non_boundary_error_rate]
        colors = ['red', 'green']
        
        bars = ax.bar(categories, error_rates, color=colors, alpha=0.7)
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Rate: Boundary vs Non-boundary')
        ax.set_ylim([0, max(error_rates) * 1.2])
        
        # 添加数值标签
        for bar, rate in zip(bars, error_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        # 添加样本数
        ax.text(0, -0.1, f'n={is_boundary.sum()}', ha='center', transform=ax.transData)
        ax.text(1, -0.1, f'n={(~is_boundary).sum()}', ha='center', transform=ax.transData)
        
        plt.suptitle('Difficult Regions Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # 打印统计
        print("\nDifficult Regions Statistics:")
        print(f"Overall Error Rate: {error_mask_flat.mean():.3f}")
        print(f"Boundary Error Rate: {boundary_error_rate:.3f}")
        print(f"Non-boundary Error Rate: {non_boundary_error_rate:.3f}")
        print(f"Error concentration (std): {error_density.std():.3f}")
    
    def create_interactive_performance_dashboard(self, save_path: Optional[str] = None):
        """
        创建交互式性能仪表板
        
        Args:
            save_path: HTML文件保存路径
        """
        # 准备数据
        metrics_df = pd.DataFrame({
            'Class_ID': range(self.n_classes),
            'Class_Name': [self.class_names[i] if i < len(self.class_names) else f'Class_{i}' 
                          for i in range(self.n_classes)],
            'Support': self.support,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1_Score': self.f1,
            'IoU': self.iou_scores,
            'Dice': self.dice_scores
        })
        
        # 过滤有效类别
        metrics_df = metrics_df[metrics_df['Support'] > 0]
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Precision-Recall Scatter', 'F1 Score by Class', 
                          'Performance Metrics Heatmap',
                          'Confusion Matrix (Top 10)', 'Support Distribution',
                          'IoU vs Dice Correlation'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Precision-Recall散点图
        fig.add_trace(
            go.Scatter(
                x=metrics_df['Recall'],
                y=metrics_df['Precision'],
                mode='markers+text',
                marker=dict(
                    size=np.log10(metrics_df['Support'] + 1) * 5,
                    color=metrics_df['F1_Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="F1", x=0.35, y=0.85, len=0.3)
                ),
                text=metrics_df['Class_Name'],
                textposition="top center",
                hovertemplate='%{text}<br>Precision: %{y:.3f}<br>Recall: %{x:.3f}<br>F1: %{marker.color:.3f}',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. F1 Score条形图
        top_classes = metrics_df.nlargest(15, 'Support')
        fig.add_trace(
            go.Bar(
                x=top_classes['Class_Name'],
                y=top_classes['F1_Score'],
                marker_color=top_classes['F1_Score'],
                marker_colorscale='RdYlGn',
                text=top_classes['F1_Score'].round(3),
                textposition='outside',
                hovertemplate='%{x}<br>F1: %{y:.3f}<br>Support: %{customdata}',
                customdata=top_classes['Support'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. 性能指标热图
        perf_matrix = top_classes[['Precision', 'Recall', 'F1_Score', 'IoU', 'Dice']].T
        fig.add_trace(
            go.Heatmap(
                z=perf_matrix.values,
                x=top_classes['Class_Name'].values,
                y=['Precision', 'Recall', 'F1', 'IoU', 'Dice'],
                colorscale='RdYlGn',
                text=np.round(perf_matrix.values, 2),
                texttemplate='%{text}',
                showscale=True,
                colorbar=dict(x=1.02, y=0.85, len=0.3)
            ),
            row=1, col=3
        )
        
        # 4. 混淆矩阵（前10个类别）
        top10_idx = metrics_df.nlargest(10, 'Support')['Class_ID'].values
        conf_sub = self.conf_matrix[np.ix_(top10_idx, top10_idx)]
        
        # 归一化
        conf_norm = conf_sub / conf_sub.sum(axis=1, keepdims=True)
        
        fig.add_trace(
            go.Heatmap(
                z=conf_norm,
                x=[f'C{i}' for i in top10_idx],
                y=[f'C{i}' for i in top10_idx],
                colorscale='Blues',
                text=np.round(conf_norm, 2),
                texttemplate='%{text}',
                showscale=True,
                colorbar=dict(x=0.35, y=0.35, len=0.3)
            ),
            row=2, col=1
        )
        
        # 5. Support分布
        fig.add_trace(
            go.Bar(
                x=top_classes['Class_Name'],
                y=top_classes['Support'],
                marker_color='lightblue',
                text=top_classes['Support'],
                textposition='outside',
                hovertemplate='%{x}<br>Support: %{y}',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. IoU vs Dice
        valid_iou_dice = metrics_df.dropna(subset=['IoU', 'Dice'])
        fig.add_trace(
            go.Scatter(
                x=valid_iou_dice['IoU'],
                y=valid_iou_dice['Dice'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=valid_iou_dice['F1_Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="F1", x=1.02, y=0.35, len=0.3)
                ),
                text=valid_iou_dice['Class_Name'],
                hovertemplate='%{text}<br>IoU: %{x:.3f}<br>Dice: %{y:.3f}',
                showlegend=False
            ),
            row=2, col=3
        )
        
        # 添加理论线
        iou_theory = np.linspace(0, 1, 100)
        dice_theory = 2 * iou_theory / (1 + iou_theory)
        fig.add_trace(
            go.Scatter(
                x=iou_theory, y=dice_theory,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Theoretical',
                showlegend=True
            ),
            row=2, col=3
        )
        
        # 更新布局
        fig.update_layout(
            title_text=f"Performance Dashboard - Overall Accuracy: {self.overall_accuracy:.3f}",
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # 更新轴标签
        fig.update_xaxes(title_text="Recall", row=1, col=1)
        fig.update_yaxes(title_text="Precision", row=1, col=1)
        fig.update_xaxes(title_text="Class", row=1, col=2)
        fig.update_yaxes(title_text="F1 Score", row=1, col=2)
        fig.update_xaxes(title_text="True", row=2, col=1)
        fig.update_yaxes(title_text="Predicted", row=2, col=1)
        fig.update_xaxes(title_text="Class", row=2, col=2)
        fig.update_yaxes(title_text="Support", row=2, col=2)
        fig.update_xaxes(title_text="IoU", row=2, col=3)
        fig.update_yaxes(title_text="Dice", row=2, col=3)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
        
        return fig
    
    def export_detailed_results(self, output_dir: str = "performance_analysis"):
        """
        导出详细的分析结果
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Exporting detailed analysis to {output_path}...")
        
        # 1. 性能报告CSV
        metrics_df = self.generate_performance_report(
            save_path=output_path / "performance_metrics.csv"
        )
        
        # 2. 混淆矩阵
        self.visualize_confusion_matrix(
            save_path=output_path / "confusion_matrix.png"
        )
        
        # 3. 类别性能可视化
        self.visualize_class_performance(
            save_path=output_path / "class_performance.png"
        )
        
        # 4. 困难区域分析
        self.analyze_difficult_regions(
            save_path=output_path / "difficult_regions.png"
        )
        
        # 5. 交互式仪表板
        self.create_interactive_performance_dashboard(
            save_path=output_path / "interactive_dashboard.html"
        )
        
        # 6. 导出详细的混淆矩阵
        np.save(output_path / "confusion_matrix.npy", self.conf_matrix)
        
        # 7. 生成摘要报告
        summary = {
            'overall_accuracy': float(self.overall_accuracy),
            'mean_f1': float(np.nanmean(self.f1)),
            'mean_iou': float(np.nanmean(self.iou_scores)),
            'mean_dice': float(np.nanmean(self.dice_scores)),
            'n_classes': self.n_classes,
            'n_valid_voxels': len(self.valid_indices),
            'best_classes': metrics_df.head(5)[['Class_Name', 'F1_Score']].to_dict('records'),
            'worst_classes': metrics_df[metrics_df['Support'] > 100].tail(5)[['Class_Name', 'F1_Score']].to_dict('records')
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis complete! Results saved to {output_path}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    softmax_path = "results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.nii.gz"
    info_path = "results/test_softmax_info_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.json"
    label_path = "path/to/balanced_labels_3d10000.nii.gz"  # 你的标签文件路径
    
    # 创建分析器
    analyzer = ClassPerformanceAnalyzer(softmax_path, info_path, label_path)
    
    # 生成完整分析
    analyzer.export_detailed_results("performance_analysis_output")
    
    # 或者单独运行各个分析
    # df = analyzer.generate_performance_report()
    # analyzer.visualize_confusion_matrix()
    # analyzer.visualize_class_performance()
    # analyzer.analyze_difficult_regions()
    # analyzer.create_interactive_performance_dashboard()