"""
多维度脑部分割结果可视化框架
针对cluster采样后的softmax预测结果
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BrainSegmentationVisualizer:
    """脑部分割结果多维度可视化工具"""
    
    def __init__(self, softmax_path: str, info_path: str, label_path: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            softmax_path: softmax nifti文件路径
            info_path: JSON信息文件路径
            label_path: 可选的真实标签文件路径
        """
        self.softmax_nii = nib.load(softmax_path)
        self.softmax = self.softmax_nii.get_fdata()
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.n_classes = self.info['n_classes']
        self.class_names = self.info['class_names']
        
        if label_path:
            self.label_nii = nib.load(label_path)
            self.labels = self.label_nii.get_fdata().astype(int)
        else:
            self.labels = None
            
        # 预计算常用指标
        self._precompute_metrics()
    
    def _precompute_metrics(self):
        """预计算常用指标以加速可视化"""
        # 预测类别
        self.predictions = np.argmax(self.softmax, axis=-1)
        
        # 最大概率
        self.max_prob = np.max(self.softmax, axis=-1)
        
        # 不确定性（熵）
        self.uncertainty = self._compute_uncertainty()
        
        # Top-2概率差
        self.top2_diff = self._compute_top2_difference()
        
        logger.info(f"预计算完成: 预测形状={self.predictions.shape}, "
                   f"不确定性范围=[{self.uncertainty.min():.3f}, {self.uncertainty.max():.3f}]")
    
    def _compute_uncertainty(self) -> np.ndarray:
        """计算基于熵的不确定性"""
        # 避免log(0)
        eps = 1e-10
        return -np.sum(self.softmax * np.log(self.softmax + eps), axis=-1)
    
    def _compute_top2_difference(self) -> np.ndarray:
        """计算最高和第二高概率的差值"""
        # 沿着类别轴排序
        sorted_probs = np.sort(self.softmax, axis=-1)
        return sorted_probs[..., -1] - sorted_probs[..., -2]
    
    def visualize_uncertainty_slices(self, slice_indices: Optional[Dict] = None, 
                                    save_path: Optional[str] = None):
        """
        可视化不确定性切片
        
        Args:
            slice_indices: 指定切片索引 {'axial': 100, 'coronal': 150, 'sagittal': 120}
            save_path: 保存路径
        """
        if slice_indices is None:
            # 使用中心切片
            shape = self.uncertainty.shape
            slice_indices = {
                'axial': shape[2] // 2,
                'sagittal': shape[0] // 2,
                'coronal': shape[1] // 2
            }
        
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 三个正交切片
        slices = {
            'Axial': self.uncertainty[:, :, slice_indices['axial']],
            'Coronal': self.uncertainty[:, slice_indices['coronal'], :],
            'Sagittal': self.uncertainty[slice_indices['sagittal'], :, :]
        }
        
        for idx, (title, slice_data) in enumerate(slices.items()):
            ax = fig.add_subplot(gs[0, idx])
            im = ax.imshow(slice_data.T, cmap='hot', origin='lower')
            ax.set_title(f'{title} - Uncertainty (Entropy)')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 对应的预测切片
        pred_slices = {
            'Axial': self.predictions[:, :, slice_indices['axial']],
            'Coronal': self.predictions[:, slice_indices['coronal'], :],
            'Sagittal': self.predictions[slice_indices['sagittal'], :, :]
        }
        
        for idx, (title, slice_data) in enumerate(pred_slices.items()):
            ax = fig.add_subplot(gs[1, idx])
            im = ax.imshow(slice_data.T, cmap='tab20', origin='lower')
            ax.set_title(f'{title} - Predictions')
            ax.axis('off')
        
        plt.suptitle('Uncertainty Analysis - Brain Segmentation', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_class_confidence_distribution(self, top_n: int = 10, 
                                          save_path: Optional[str] = None):
        """
        绘制类别置信度分布
        
        Args:
            top_n: 显示前N个最常见的类别
            save_path: 保存路径
        """
        # 计算每个类别的平均概率和出现频率
        class_stats = []
        
        for class_id in range(self.n_classes):
            class_probs = self.softmax[..., class_id]
            mask = class_probs > 0.1  # 只考虑概率>0.1的体素
            
            if np.any(mask):
                stats = {
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'Class_{class_id}',
                    'mean_prob': np.mean(class_probs[mask]),
                    'max_prob': np.max(class_probs),
                    'coverage': np.sum(mask) / mask.size,  # 覆盖率
                    'voxel_count': np.sum(self.predictions == class_id)
                }
                class_stats.append(stats)
        
        # 按体素数量排序
        class_stats.sort(key=lambda x: x['voxel_count'], reverse=True)
        top_classes = class_stats[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 体素数量条形图
        ax = axes[0, 0]
        class_names = [c['class_name'] for c in top_classes]
        voxel_counts = [c['voxel_count'] for c in top_classes]
        bars = ax.bar(range(len(top_classes)), voxel_counts)
        ax.set_xticks(range(len(top_classes)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('Voxel Count')
        ax.set_title('Top Classes by Voxel Count')
        
        # 着色条形图
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.tab20(i))
        
        # 2. 平均置信度
        ax = axes[0, 1]
        mean_probs = [c['mean_prob'] for c in top_classes]
        bars = ax.bar(range(len(top_classes)), mean_probs)
        ax.set_xticks(range(len(top_classes)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Confidence')
        ax.set_title('Mean Confidence by Class')
        ax.set_ylim([0, 1])
        
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.tab20(i))
        
        # 3. 置信度分布箱线图
        ax = axes[1, 0]
        box_data = []
        valid_class_names = []
        for c in top_classes[:5]:  # 只显示前5个类别
            class_probs = self.softmax[..., c['class_id']]
            mask = self.predictions == c['class_id']
            if np.any(mask):
                box_data.append(class_probs[mask])
                valid_class_names.append(c['class_name'])
        
        if len(box_data) > 0:
            bp = ax.boxplot(box_data)
            ax.set_ylabel('Probability')
            ax.set_title('Confidence Distribution (Top Classes)')
            ax.set_xticklabels(valid_class_names, rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No valid data for boxplot', ha='center', va='center', transform=ax.transAxes)
        
        # 4. 不确定性分布
        ax = axes[1, 1]
        uncertainty_by_class = []
        valid_uncertainty_names = []
        for c in top_classes[:5]:
            mask = self.predictions == c['class_id']
            if np.any(mask):
                uncertainty_by_class.append(self.uncertainty[mask])
                valid_uncertainty_names.append(c['class_name'])
        
        if len(uncertainty_by_class) > 0:
            bp = ax.boxplot(uncertainty_by_class)
            ax.set_ylabel('Uncertainty (Entropy)')
            ax.set_title('Uncertainty Distribution by Class')
            ax.set_xticklabels(valid_uncertainty_names, rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No valid data for boxplot', ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Class Confidence Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_interactive_3d_browser(self, save_path: Optional[str] = None):
        """
        创建交互式3D浏览器
        
        Args:
            save_path: HTML保存路径
        """
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Axial View', 'Coronal View', 'Sagittal View', 'Uncertainty'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        shape = self.predictions.shape
        
        # 初始切片位置（中心）
        init_axial = shape[2] // 2
        init_coronal = shape[1] // 2
        init_sagittal = shape[0] // 2
        
        # 添加切片
        # Axial
        fig.add_trace(
            go.Heatmap(
                z=self.predictions[:, :, init_axial].T,
                colorscale='Viridis',
                showscale=True,
                name='Axial'
            ),
            row=1, col=1
        )
        
        # Coronal
        fig.add_trace(
            go.Heatmap(
                z=self.predictions[:, init_coronal, :].T,
                colorscale='Viridis',
                showscale=False,
                name='Coronal'
            ),
            row=1, col=2
        )
        
        # Sagittal
        fig.add_trace(
            go.Heatmap(
                z=self.predictions[init_sagittal, :, :].T,
                colorscale='Viridis',
                showscale=False,
                name='Sagittal'
            ),
            row=2, col=1
        )
        
        # Uncertainty
        fig.add_trace(
            go.Heatmap(
                z=self.uncertainty[:, :, init_axial].T,
                colorscale='Hot',
                showscale=True,
                name='Uncertainty'
            ),
            row=2, col=2
        )
        
        # 创建滑块
        steps = []
        for i in range(0, shape[2], 5):  # 每5个切片一个步骤
            step = dict(
                method="update",
                args=[{"z": [self.predictions[:, :, i].T,
                           self.predictions[:, shape[1]//2, :].T,
                           self.predictions[shape[0]//2, :, :].T,
                           self.uncertainty[:, :, i].T]}],
                label=f"Slice {i}"
            )
            steps.append(step)
        
        sliders = [dict(
            active=init_axial // 5,
            currentvalue={"prefix": "Axial Slice: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            title="Interactive Brain Segmentation Browser",
            sliders=sliders,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def analyze_spatial_consistency(self, window_size: int = 3, 
                                   save_path: Optional[str] = None):
        """
        分析空间一致性
        
        Args:
            window_size: 邻域窗口大小
            save_path: 保存路径
        """
        from scipy.ndimage import generic_filter
        
        def consistency_metric(values):
            """计算邻域内的一致性"""
            unique, counts = np.unique(values, return_counts=True)
            # 返回最常见类别的比例
            return counts.max() / len(values)
        
        # 计算空间一致性
        consistency_map = generic_filter(
            self.predictions, 
            consistency_metric, 
            size=window_size,
            mode='constant',
            cval=0
        )
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 选择代表性切片
        slice_idx = self.predictions.shape[2] // 2
        
        # 预测
        ax = axes[0]
        im = ax.imshow(self.predictions[:, :, slice_idx].T, cmap='tab20', origin='lower')
        ax.set_title('Predictions')
        ax.axis('off')
        
        # 一致性
        ax = axes[1]
        im = ax.imshow(consistency_map[:, :, slice_idx].T, cmap='RdYlGn', 
                      origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Spatial Consistency (window={window_size})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 不一致区域（低一致性）
        ax = axes[2]
        inconsistent = consistency_map < 0.7  # 阈值
        im = ax.imshow(inconsistent[:, :, slice_idx].T, cmap='Reds', 
                      origin='lower', alpha=0.7)
        ax.set_title('Inconsistent Regions (< 0.7)')
        ax.axis('off')
        
        plt.suptitle('Spatial Consistency Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # 统计信息
        print(f"Consistency Statistics:")
        print(f"  Mean: {consistency_map.mean():.3f}")
        print(f"  Std:  {consistency_map.std():.3f}")
        print(f"  Min:  {consistency_map.min():.3f}")
        print(f"  Max:  {consistency_map.max():.3f}")
        print(f"  Inconsistent voxels (<0.7): {(consistency_map < 0.7).sum()} "
              f"({(consistency_map < 0.7).mean() * 100:.1f}%)")
    
    def compare_with_ground_truth(self, save_path: Optional[str] = None):
        """
        与真实标签对比（如果可用）
        
        Args:
            save_path: 保存路径
        """
        if self.labels is None:
            print("No ground truth labels available")
            return
        
        # 计算混淆区域
        confusion_mask = self.predictions != self.labels
        
        # 高不确定性区域
        high_uncertainty = self.uncertainty > np.percentile(self.uncertainty, 90)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        slice_idx = self.predictions.shape[2] // 2
        
        # 第一行：预测、真值、差异、不确定性
        titles1 = ['Predictions', 'Ground Truth', 'Errors', 'Uncertainty']
        data1 = [
            self.predictions[:, :, slice_idx],
            self.labels[:, :, slice_idx],
            confusion_mask[:, :, slice_idx],
            self.uncertainty[:, :, slice_idx]
        ]
        cmaps1 = ['tab20', 'tab20', 'Reds', 'hot']
        
        for ax, title, data, cmap in zip(axes[0], titles1, data1, cmaps1):
            im = ax.imshow(data.T, cmap=cmap, origin='lower')
            ax.set_title(title)
            ax.axis('off')
            if title in ['Errors', 'Uncertainty']:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 第二行：置信度、Top-2差异、高不确定性、错误+不确定
        titles2 = ['Max Probability', 'Top-2 Difference', 
                  'High Uncertainty Areas', 'Errors + Uncertainty']
        
        # 错误且高不确定性
        error_uncertain = confusion_mask & high_uncertainty
        
        data2 = [
            self.max_prob[:, :, slice_idx],
            self.top2_diff[:, :, slice_idx],
            high_uncertainty[:, :, slice_idx],
            error_uncertain[:, :, slice_idx]
        ]
        cmaps2 = ['viridis', 'coolwarm', 'Reds', 'hot']
        
        for ax, title, data, cmap in zip(axes[1], titles2, data2, cmaps2):
            im = ax.imshow(data.T, cmap=cmap, origin='lower')
            ax.set_title(title)
            ax.axis('off')
            if title != 'High Uncertainty Areas':
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('Prediction vs Ground Truth Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # 计算统计
        accuracy = (self.predictions == self.labels).mean()
        uncertain_accuracy = (self.predictions[high_uncertainty] == 
                            self.labels[high_uncertainty]).mean()
        
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Accuracy in high uncertainty regions: {uncertain_accuracy:.3f}")
        print(f"Error rate: {confusion_mask.mean():.3f}")
        print(f"Proportion of high uncertainty: {high_uncertainty.mean():.3f}")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    softmax_path = "results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.nii.gz"
    info_path = "results/test_softmax_info_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.json"
    
    # 创建可视化器
    visualizer = BrainSegmentationVisualizer(softmax_path, info_path)
    
    # 生成各种可视化
    visualizer.visualize_uncertainty_slices(save_path="uncertainty_analysis.png")
    visualizer.plot_class_confidence_distribution(save_path="confidence_distribution.png")
    visualizer.create_interactive_3d_browser(save_path="interactive_browser.html")
    visualizer.analyze_spatial_consistency(save_path="spatial_consistency.png")
    
    # 如果有真实标签
    # label_path = "path/to/labels.nii.gz"
    # visualizer = BrainSegmentationVisualizer(softmax_path, info_path, label_path)
    # visualizer.compare_with_ground_truth(save_path="comparison.png")