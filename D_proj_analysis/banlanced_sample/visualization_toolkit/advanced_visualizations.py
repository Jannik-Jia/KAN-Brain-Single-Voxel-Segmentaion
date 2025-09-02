"""
高级可视化工具：针对cluster采样特性的专门分析
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedSegmentationAnalyzer:
    """高级分割分析工具，专注于采样特性和类别间关系"""
    
    def __init__(self, softmax_path: str, info_path: str):
        """初始化分析器"""
        self.softmax_nii = nib.load(softmax_path)
        self.softmax = self.softmax_nii.get_fdata()
        self.affine = self.softmax_nii.affine
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.n_classes = self.info['n_classes']
        self.class_names = self.info['class_names']
        self.include_background = self.info['include_background']
        
        # 获取非零体素（被采样的体素）
        self._identify_sampled_voxels()
    
    def _identify_sampled_voxels(self):
        """识别被采样的体素"""
        # 对于每个体素，如果至少有一个类别的概率>0，则认为被采样
        max_prob = np.max(self.softmax, axis=-1)
        self.sampled_mask = max_prob > 0
        self.sampled_coords = np.argwhere(self.sampled_mask)
        
        print(f"Identified {len(self.sampled_coords)} sampled voxels "
              f"({len(self.sampled_coords)/np.prod(self.softmax.shape[:-1])*100:.2f}% of total)")
    
    def visualize_sampling_distribution(self, save_path: Optional[str] = None):
        """
        可视化采样分布的空间特性
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 3D采样密度热图（投影到三个平面）
        for idx, (ax, axis_name) in enumerate(zip(axes[0], ['X-Y', 'X-Z', 'Y-Z'])):
            if axis_name == 'X-Y':
                density, xedges, yedges = np.histogram2d(
                    self.sampled_coords[:, 0], 
                    self.sampled_coords[:, 1], 
                    bins=50
                )
            elif axis_name == 'X-Z':
                density, xedges, yedges = np.histogram2d(
                    self.sampled_coords[:, 0], 
                    self.sampled_coords[:, 2], 
                    bins=50
                )
            else:  # Y-Z
                density, xedges, yedges = np.histogram2d(
                    self.sampled_coords[:, 1], 
                    self.sampled_coords[:, 2], 
                    bins=50
                )
            
            im = ax.imshow(density.T, origin='lower', cmap='YlOrRd', aspect='auto')
            ax.set_title(f'Sampling Density - {axis_name} Plane')
            ax.set_xlabel(axis_name.split('-')[0])
            ax.set_ylabel(axis_name.split('-')[1])
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 2. 采样率随切片位置的变化
        for idx, (ax, axis) in enumerate(zip(axes[1], [2, 1, 0])):  # Z, Y, X
            axis_names = ['X', 'Y', 'Z']
            slice_sampling_rate = []
            
            n_slices = self.softmax.shape[axis]
            for i in range(n_slices):
                if axis == 2:  # Z轴
                    slice_mask = self.sampled_mask[:, :, i]
                elif axis == 1:  # Y轴
                    slice_mask = self.sampled_mask[:, i, :]
                else:  # X轴
                    slice_mask = self.sampled_mask[i, :, :]
                
                rate = np.mean(slice_mask)
                slice_sampling_rate.append(rate)
            
            ax.plot(slice_sampling_rate, linewidth=2)
            ax.fill_between(range(len(slice_sampling_rate)), 
                           slice_sampling_rate, alpha=0.3)
            ax.set_xlabel(f'{axis_names[axis]} Slice Index')
            ax.set_ylabel('Sampling Rate')
            ax.set_title(f'Sampling Rate along {axis_names[axis]} axis')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Spatial Distribution of Sampled Voxels', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_class_relationships(self, save_path: Optional[str] = None):
        """
        分析类别间的关系（基于概率分布的相似性）
        """
        # 计算类别间的平均概率共现矩阵
        n_classes = self.n_classes
        cooccurrence = np.zeros((n_classes, n_classes))
        
        # 对每个采样体素，计算类别对的概率乘积
        for coord in self.sampled_coords[:1000]:  # 使用子集以加速
            i, j, k = coord
            probs = self.softmax[i, j, k, :]
            cooccurrence += np.outer(probs, probs)
        
        cooccurrence /= len(self.sampled_coords[:1000])
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 共现矩阵热图
        ax = axes[0]
        im = ax.imshow(cooccurrence, cmap='YlOrRd', aspect='auto')
        ax.set_title('Class Co-occurrence Matrix')
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Class ID')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 2. 类别层次聚类
        ax = axes[1]
        # 使用共现矩阵计算距离
        distance_matrix = 1 - cooccurrence
        np.fill_diagonal(distance_matrix, 0)
        
        # 执行层次聚类
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        dendrogram(linkage_matrix, ax=ax, orientation='left', 
                  labels=[str(i) for i in range(n_classes)])
        ax.set_title('Class Hierarchy (based on co-occurrence)')
        ax.set_xlabel('Distance')
        
        # 3. 类别间的平均概率转移
        ax = axes[2]
        # 计算相邻体素的类别转移
        transition_matrix = self._compute_transition_matrix()
        im = ax.imshow(transition_matrix, cmap='Blues', aspect='auto')
        ax.set_title('Class Transition Probability')
        ax.set_xlabel('To Class')
        ax.set_ylabel('From Class')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('Class Relationship Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def _compute_transition_matrix(self) -> np.ndarray:
        """计算类别转移矩阵"""
        predictions = np.argmax(self.softmax, axis=-1)
        n_classes = self.n_classes
        transition = np.zeros((n_classes, n_classes))
        
        # 检查相邻体素的类别
        for coord in self.sampled_coords[:5000]:  # 使用子集
            i, j, k = coord
            current_class = predictions[i, j, k]
            
            # 检查6个相邻体素
            neighbors = [
                (i+1, j, k), (i-1, j, k),
                (i, j+1, k), (i, j-1, k),
                (i, j, k+1), (i, j, k-1)
            ]
            
            for ni, nj, nk in neighbors:
                if (0 <= ni < predictions.shape[0] and 
                    0 <= nj < predictions.shape[1] and 
                    0 <= nk < predictions.shape[2]):
                    neighbor_class = predictions[ni, nj, nk]
                    transition[current_class, neighbor_class] += 1
        
        # 归一化
        row_sums = transition.sum(axis=1, keepdims=True)
        transition = np.divide(transition, row_sums, 
                              where=row_sums != 0, out=transition)
        
        return transition
    
    def visualize_probability_landscape(self, class_id: int, 
                                       save_path: Optional[str] = None):
        """
        可视化特定类别的概率景观
        
        Args:
            class_id: 要可视化的类别ID
        """
        class_prob = self.softmax[..., class_id]
        
        fig = plt.figure(figsize=(18, 12))
        
        # 创建3D子图
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # 采样一部分高概率点进行3D散点图
        high_prob_mask = class_prob > 0.5
        high_prob_coords = np.argwhere(high_prob_mask)
        
        if len(high_prob_coords) > 0:
            # 随机采样最多1000个点
            sample_idx = np.random.choice(len(high_prob_coords), 
                                        min(1000, len(high_prob_coords)), 
                                        replace=False)
            sample_coords = high_prob_coords[sample_idx]
            sample_probs = class_prob[high_prob_mask][sample_idx]
            
            scatter = ax1.scatter(sample_coords[:, 0], 
                                sample_coords[:, 1], 
                                sample_coords[:, 2],
                                c=sample_probs, cmap='hot', 
                                s=sample_probs*50, alpha=0.6)
            ax1.set_title(f'3D Probability Cloud - Class {class_id}')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            plt.colorbar(scatter, ax=ax1, fraction=0.046)
        
        # 2D投影
        for idx, (subplot_idx, axis_name) in enumerate([(2, 'Axial'), 
                                                        (3, 'Coronal'), 
                                                        (4, 'Sagittal')]):
            ax = fig.add_subplot(2, 3, subplot_idx)
            
            # 最大强度投影
            if axis_name == 'Axial':
                projection = np.max(class_prob, axis=2)
            elif axis_name == 'Coronal':
                projection = np.max(class_prob, axis=1)
            else:  # Sagittal
                projection = np.max(class_prob, axis=0)
            
            im = ax.imshow(projection.T, cmap='hot', origin='lower')
            ax.set_title(f'{axis_name} MIP - Class {class_id}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 概率分布直方图
        ax = fig.add_subplot(2, 3, 5)
        probs_nonzero = class_prob[class_prob > 0.01]
        if len(probs_nonzero) > 0:
            ax.hist(probs_nonzero, bins=50, color='orange', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(probs_nonzero), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(probs_nonzero):.3f}')
            ax.axvline(np.median(probs_nonzero), color='blue', linestyle='--', 
                      label=f'Median: {np.median(probs_nonzero):.3f}')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Count')
            ax.set_title(f'Probability Distribution - Class {class_id}')
            ax.legend()
        
        # 空间聚集度分析
        ax = fig.add_subplot(2, 3, 6)
        # 计算局部密度
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter((class_prob > 0.3).astype(float), sigma=3)
        
        # 显示中间切片的密度
        slice_idx = density.shape[2] // 2
        im = ax.imshow(density[:, :, slice_idx].T, cmap='YlOrRd', origin='lower')
        ax.set_title(f'Spatial Clustering - Class {class_id}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        class_name = (self.class_names[class_id] 
                     if class_id < len(self.class_names) 
                     else f'Class_{class_id}')
        plt.suptitle(f'Probability Landscape Analysis - {class_name}', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_uncertainty_regions_analysis(self, save_path: Optional[str] = None):
        """
        分析和可视化不确定性区域的特征
        """
        # 计算不确定性
        eps = 1e-10
        uncertainty = -np.sum(self.softmax * np.log(self.softmax + eps), axis=-1)
        
        # 定义不确定性级别
        low_threshold = np.percentile(uncertainty[self.sampled_mask], 25)
        high_threshold = np.percentile(uncertainty[self.sampled_mask], 75)
        
        low_uncertainty = uncertainty < low_threshold
        medium_uncertainty = (uncertainty >= low_threshold) & (uncertainty < high_threshold)
        high_uncertainty = uncertainty >= high_threshold
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 可视化不同不确定性级别的区域
        slice_idx = uncertainty.shape[2] // 2
        
        # 1. 不确定性级别图
        ax = axes[0, 0]
        uncertainty_levels = np.zeros_like(uncertainty[:, :, slice_idx])
        uncertainty_levels[low_uncertainty[:, :, slice_idx]] = 1
        uncertainty_levels[medium_uncertainty[:, :, slice_idx]] = 2
        uncertainty_levels[high_uncertainty[:, :, slice_idx]] = 3
        
        im = ax.imshow(uncertainty_levels.T, cmap='RdYlGn_r', origin='lower', vmin=0, vmax=3)
        ax.set_title('Uncertainty Levels')
        ax.axis('off')
        
        # 添加图例
        colors = ['white', 'green', 'yellow', 'red']
        labels = ['Not sampled', 'Low', 'Medium', 'High']
        patches = [Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(4)]
        ax.legend(patches, labels, loc='upper right')
        
        # 2. 高不确定性区域的类别分布
        ax = axes[0, 1]
        predictions = np.argmax(self.softmax, axis=-1)
        high_uncertain_classes = predictions[high_uncertainty & self.sampled_mask]
        
        if len(high_uncertain_classes) > 0:
            unique, counts = np.unique(high_uncertain_classes, return_counts=True)
            ax.bar(unique[:10], counts[:10])  # 显示前10个
            ax.set_xlabel('Class ID')
            ax.set_ylabel('Count')
            ax.set_title('Classes in High Uncertainty Regions')
        
        # 3. 不确定性与概率的关系
        ax = axes[0, 2]
        max_prob = np.max(self.softmax, axis=-1)
        sampled_uncertainty = uncertainty[self.sampled_mask]
        sampled_max_prob = max_prob[self.sampled_mask]
        
        # 采样一部分点以避免过度绘制
        sample_size = min(10000, len(sampled_uncertainty))
        sample_idx = np.random.choice(len(sampled_uncertainty), sample_size, replace=False)
        
        ax.scatter(sampled_max_prob[sample_idx], sampled_uncertainty[sample_idx], 
                  alpha=0.3, s=1)
        ax.set_xlabel('Max Probability')
        ax.set_ylabel('Uncertainty (Entropy)')
        ax.set_title('Uncertainty vs Confidence')
        
        # 4. 边界检测（高不确定性often发生在边界）
        ax = axes[1, 0]
        from scipy.ndimage import sobel
        
        # 计算梯度幅度
        grad_x = sobel(predictions.astype(float), axis=0)
        grad_y = sobel(predictions.astype(float), axis=1)
        grad_z = sobel(predictions.astype(float), axis=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        im = ax.imshow(gradient_magnitude[:, :, slice_idx].T, cmap='hot', origin='lower')
        ax.set_title('Boundary Detection (Gradient Magnitude)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 5. 不确定性的空间自相关
        ax = axes[1, 1]
        from scipy.ndimage import correlate
        
        # 计算局部不确定性的变异系数
        kernel = np.ones((3, 3, 3)) / 27
        local_mean = correlate(uncertainty, kernel)
        local_var = correlate(uncertainty**2, kernel) - local_mean**2
        local_cv = np.sqrt(local_var) / (local_mean + 1e-10)
        
        im = ax.imshow(local_cv[:, :, slice_idx].T, cmap='viridis', origin='lower')
        ax.set_title('Local Uncertainty Variation')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 6. 统计摘要
        ax = axes[1, 2]
        ax.axis('off')
        
        stats_text = f"""Uncertainty Statistics:
        
        Low uncertainty (<25%): {np.sum(low_uncertainty & self.sampled_mask)} voxels
        Medium uncertainty: {np.sum(medium_uncertainty & self.sampled_mask)} voxels
        High uncertainty (>75%): {np.sum(high_uncertainty & self.sampled_mask)} voxels
        
        Mean uncertainty: {sampled_uncertainty.mean():.3f}
        Std uncertainty: {sampled_uncertainty.std():.3f}
        
        Correlation with boundaries: {np.corrcoef(uncertainty.flatten(), gradient_magnitude.flatten())[0,1]:.3f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
               fontfamily='monospace')
        ax.set_title('Summary Statistics')
        
        plt.suptitle('Uncertainty Regions Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, output_dir: str = "visualization_report"):
        """
        生成完整的可视化报告
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating comprehensive visualization report...")
        
        # 1. 采样分布分析
        print("  1. Analyzing sampling distribution...")
        self.visualize_sampling_distribution(
            save_path=output_path / "sampling_distribution.png"
        )
        
        # 2. 类别关系分析
        print("  2. Analyzing class relationships...")
        self.analyze_class_relationships(
            save_path=output_path / "class_relationships.png"
        )
        
        # 3. 不确定性区域分析
        print("  3. Analyzing uncertainty regions...")
        self.create_uncertainty_regions_analysis(
            save_path=output_path / "uncertainty_analysis.png"
        )
        
        # 4. 为主要类别生成概率景观
        print("  4. Generating probability landscapes for major classes...")
        predictions = np.argmax(self.softmax, axis=-1)
        unique, counts = np.unique(predictions[self.sampled_mask], return_counts=True)
        top_classes = unique[np.argsort(counts)[-5:]]  # Top 5 classes
        
        for class_id in top_classes:
            self.visualize_probability_landscape(
                class_id,
                save_path=output_path / f"probability_landscape_class_{class_id}.png"
            )
        
        # 5. 生成HTML报告
        print("  5. Generating HTML report...")
        self._generate_html_report(output_path)
        
        print(f"Report generated successfully in {output_path}")
    
    def _generate_html_report(self, output_path: Path):
        """生成HTML报告"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain Segmentation Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                .section { margin-bottom: 40px; }
                .stats { background: #f5f5f5; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Brain Segmentation Visualization Report</h1>
            
            <div class="section">
                <h2>1. Sampling Distribution Analysis</h2>
                <p>Analysis of spatial distribution of sampled voxels from clustering</p>
                <img src="sampling_distribution.png" alt="Sampling Distribution">
            </div>
            
            <div class="section">
                <h2>2. Class Relationships</h2>
                <p>Inter-class relationships based on co-occurrence and transitions</p>
                <img src="class_relationships.png" alt="Class Relationships">
            </div>
            
            <div class="section">
                <h2>3. Uncertainty Analysis</h2>
                <p>Comprehensive analysis of prediction uncertainty regions</p>
                <img src="uncertainty_analysis.png" alt="Uncertainty Analysis">
            </div>
            
            <div class="section">
                <h2>4. Class-Specific Probability Landscapes</h2>
                <p>Detailed probability distribution for major classes</p>
        """
        
        # 添加概率景观图片
        for img_path in output_path.glob("probability_landscape_*.png"):
            html_content += f'        <img src="{img_path.name}" alt="{img_path.stem}">\n'
        
        html_content += """
            </div>
            
            <div class="section stats">
                <h2>Summary Statistics</h2>
                <ul>
                    <li>Total sampled voxels: """ + str(len(self.sampled_coords)) + """</li>
                    <li>Number of classes: """ + str(self.n_classes) + """</li>
                    <li>Include background: """ + str(self.include_background) + """</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path / "report.html", 'w') as f:
            f.write(html_content)

# 使用示例
if __name__ == "__main__":
    # 设置路径
    softmax_path = "results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.nii.gz"
    info_path = "results/test_softmax_info_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.json"
    
    # 创建分析器
    analyzer = AdvancedSegmentationAnalyzer(softmax_path, info_path)
    
    # 生成完整报告
    analyzer.generate_comprehensive_report("visualization_report")
    
    # 或者单独运行各个分析
    # analyzer.visualize_sampling_distribution()
    # analyzer.analyze_class_relationships()
    # analyzer.visualize_probability_landscape(class_id=10)
    # analyzer.create_uncertainty_regions_analysis()