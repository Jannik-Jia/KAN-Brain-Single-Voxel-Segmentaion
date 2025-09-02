#!/usr/bin/env python3
"""
混合数据分析器
专门处理背景排除模式产生的混合真实/人工数据
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy import stats

class HybridDataAnalyzer:
    """混合数据分析器 - 正确处理背景排除模式"""
    
    def __init__(self, softmax_path: str, info_path: str, labels_path: str):
        """
        初始化分析器
        
        Args:
            softmax_path: softmax预测文件路径
            info_path: 预测信息JSON文件路径  
            labels_path: 真实标签文件路径
        """
        print("🔍 Loading hybrid data for analysis...")
        
        # 加载数据
        self.softmax_nii = nib.load(softmax_path)
        self.softmax = self.softmax_nii.get_fdata()
        
        self.labels_nii = nib.load(labels_path)
        self.labels = self.labels_nii.get_fdata().astype(np.int32)
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.include_background = self.info['include_background']
        self.n_classes = self.info['n_classes']
        self.class_names = self.info.get('class_names', list(range(self.n_classes)))
        
        # 🔑 关键：先映射标签（和训练时一致），然后识别区域
        self.labels_continuous = self._apply_label_mapping_3d(self.labels)
        
        # 识别真实预测区域 vs 人工填充区域
        self.real_prediction_mask, self.artificial_background_mask = self._identify_data_regions()
        
        print(f"✅ Data loaded: {self.softmax.shape}")
        print(f"📋 Background mode: {'Included' if self.include_background else 'Excluded'}")
        print(f"🎯 Real prediction voxels: {np.sum(self.real_prediction_mask):,}")
        print(f"🤖 Artificial background voxels: {np.sum(self.artificial_background_mask):,}")
    
    def _identify_data_regions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        识别真实预测区域和人工背景区域
        
        Returns:
            real_mask: 真实预测区域的3D布尔掩码
            artificial_mask: 人工背景区域的3D布尔掩码
        """
        if self.include_background:
            # 背景包含模式：所有区域都是真实预测
            real_mask = np.ones(self.labels.shape, dtype=bool)
            artificial_mask = np.zeros(self.labels.shape, dtype=bool)
            
        else:
            # 背景排除模式：需要区分真实和人工区域
            # 人工背景的特征：背景类概率=1.0，其他类概率=0.0
            bg_prob = self.softmax[..., 0]  # 假设背景是类别0
            other_classes_sum = np.sum(self.softmax[..., 1:], axis=-1)
            
            # 人工背景特征：背景概率≈1.0且其他类概率≈0.0
            artificial_mask = (bg_prob > 0.999) & (other_classes_sum < 0.001)
            real_mask = ~artificial_mask
            
            # 验证：人工区域应该对应原始标签的背景(0)
            artificial_should_be_bg_original = self.labels[artificial_mask] 
            artificial_should_be_bg_continuous = self.labels_continuous[artificial_mask]
            if len(artificial_should_be_bg_original) > 0:
                bg_ratio_orig = np.mean(artificial_should_be_bg_original == 0)
                bg_ratio_cont = np.mean(artificial_should_be_bg_continuous == 0)
                print(f"🔍 Validation: {bg_ratio_orig:.1%} original background, {bg_ratio_cont:.1%} continuous background")
        
        return real_mask, artificial_mask
    
    def _apply_label_mapping_3d(self, labels):
        """将3D原始标签映射到连续索引（和训练时保持一致）"""
        print("🔄 Applying 3D label mapping (same as training)...")
        
        # 创建映射字典
        forward_mapping = {original: continuous for continuous, original in enumerate(self.class_names)}
        
        # 应用映射
        labels_flat = labels.flatten()
        labels_mapped = np.zeros_like(labels_flat)
        
        unique_orig = np.unique(labels_flat)
        print(f"  Original unique labels: {sorted(unique_orig)}")
        
        for orig_label in unique_orig:
            if orig_label in forward_mapping:
                mask = labels_flat == orig_label
                labels_mapped[mask] = forward_mapping[orig_label]
                count = np.sum(mask)
                if orig_label != 0:  # 不显示背景映射信息
                    print(f"    {orig_label:3d} → {forward_mapping[orig_label]:2d} ({count:,} 体素)")
        
        # 恢复3D形状
        labels_continuous = labels_mapped.reshape(labels.shape)
        
        mapped_unique = np.unique(labels_continuous)
        print(f"  Mapped unique labels: {sorted(mapped_unique)}")
        
        return labels_continuous
    
    def _apply_label_mapping(self, labels):
        """将原始标签映射到连续索引（和训练时保持一致）"""
        # 创建映射字典
        forward_mapping = {original: continuous for continuous, original in enumerate(self.class_names)}
        
        # 应用映射
        labels_mapped = np.zeros_like(labels)
        for orig_label in np.unique(labels):
            if orig_label in forward_mapping:
                labels_mapped[labels == orig_label] = forward_mapping[orig_label]
        
        return labels_mapped
    
    def analyze_foreground_performance(self, save_path: Optional[str] = None):
        """
        分析前景区域的性能（排除人工背景区域）
        """
        print("📊 Analyzing foreground performance (excluding artificial background)...")
        
        # 获取前景区域数据 (🔑 修复：使用映射后的标签定义前景，和训练时一致)
        fg_mask = self.real_prediction_mask & (self.labels_continuous != 0)
        
        if np.sum(fg_mask) == 0:
            print("⚠️ No foreground voxels found for analysis")
            return None
        
        fg_softmax = self.softmax[fg_mask]  # (n_fg_voxels, n_classes)
        fg_labels = self.labels_continuous[fg_mask]  # (n_fg_voxels,) 已映射的连续标签
        
        # 预测直接使用连续索引
        fg_predictions = np.argmax(fg_softmax, axis=1)
        
        print(f"  Analyzing {len(fg_softmax):,} foreground voxels")
        print(f"  Classes present: {len(np.unique(fg_labels))}")
        
        # 计算每类性能
        class_stats = []
        for class_id in range(self.n_classes):
            if class_id >= len(self.class_names):
                continue
                
            original_label = self.class_names[class_id]
            
            # 真实为该类的体素 (现在使用连续索引比较)
            true_mask = fg_labels == class_id
            n_true = np.sum(true_mask)
            
            if n_true == 0:
                continue
            
            # 预测为该类的体素 (现在使用连续索引比较)
            pred_mask = fg_predictions == class_id
            n_pred = np.sum(pred_mask)
            
            # 计算指标
            tp = np.sum(true_mask & pred_mask)
            fp = np.sum(~true_mask & pred_mask)
            fn = np.sum(true_mask & ~pred_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 概率分布统计
            if n_true > 0:
                class_probs = fg_softmax[true_mask, class_id]  # 使用连续索引访问softmax
                confidence = np.mean(class_probs)
                prob_std = np.std(class_probs)
                entropy = -np.sum(fg_softmax[true_mask] * np.log(fg_softmax[true_mask] + 1e-10), axis=1)
                mean_entropy = np.mean(entropy)
            else:
                confidence = 0.0
                prob_std = 0.0
                mean_entropy = 0.0
            
            class_stats.append({
                'class_id': class_id,
                'class_name': original_label,
                'n_true_voxels': n_true,
                'n_pred_voxels': n_pred,
                'precision': precision,
                'recall': recall, 
                'f1_score': f1,
                'mean_confidence': confidence,
                'confidence_std': prob_std,
                'mean_entropy': mean_entropy,
                'coverage_ratio': n_true / len(fg_labels)
            })
        
        df = pd.DataFrame(class_stats)
        df = df.sort_values('n_true_voxels', ascending=False)
        
        # 可视化
        self._visualize_foreground_performance(df, save_path)
        
        return df
    
    def _visualize_foreground_performance(self, df: pd.DataFrame, save_path: Optional[str]):
        """可视化前景性能分析"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. 每类F1分数
        ax = axes[0]
        top_classes = df.head(15)
        bars = ax.bar(range(len(top_classes)), top_classes['f1_score'])
        ax.set_title('F1 Score by Class (Top 15)')
        ax.set_xlabel('Class Rank')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        for i, (_, row) in enumerate(top_classes.iterrows()):
            ax.text(i, row['f1_score'] + 0.01, f"{row['class_name']}", 
                   rotation=45, ha='left', fontsize=8)
        
        # 2. Precision vs Recall
        ax = axes[1]
        scatter = ax.scatter(df['recall'], df['precision'], 
                           s=np.log10(df['n_true_voxels'] + 1) * 20,
                           c=df['f1_score'], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='F1 Score')
        ax.grid(True, alpha=0.3)
        
        # 3. 置信度分析
        ax = axes[2]
        ax.hist(df['mean_confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(df['mean_confidence'].mean(), color='red', linestyle='--',
                  label=f'Mean: {df["mean_confidence"].mean():.3f}')
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 不确定性分析 
        ax = axes[3]
        ax.scatter(df['mean_confidence'], df['mean_entropy'],
                  s=60, alpha=0.7, c=df['f1_score'], cmap='RdYlBu_r')
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Mean Entropy (Uncertainty)')
        ax.set_title('Confidence vs Uncertainty')
        ax.grid(True, alpha=0.3)
        
        # 5. 体素数量分布
        ax = axes[4]
        ax.hist(np.log10(df['n_true_voxels'] + 1), bins=15, 
               alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('log10(Number of Voxels)')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Class Size Distribution')
        ax.grid(True, alpha=0.3)
        
        # 6. 摘要统计
        ax = axes[5]
        ax.axis('off')
        
        # 计算整体指标
        overall_f1 = df['f1_score'].mean()
        overall_precision = df['precision'].mean()
        overall_recall = df['recall'].mean()
        total_voxels = df['n_true_voxels'].sum()
        
        summary_text = f"""Foreground Performance Summary:

📊 Analysis Scope: Real predictions only
   (Excluding artificial background regions)

🎯 Overall Performance:
   Mean F1 Score: {overall_f1:.3f}
   Mean Precision: {overall_precision:.3f}
   Mean Recall: {overall_recall:.3f}
   
📈 Data Statistics:
   Total Foreground Voxels: {total_voxels:,}
   Classes Analyzed: {len(df)}
   Mean Confidence: {df['mean_confidence'].mean():.3f}
   Mean Uncertainty: {df['mean_entropy'].mean():.3f}

🏆 Top 3 Performing Classes:
   {df.iloc[0]['class_name']}: F1={df.iloc[0]['f1_score']:.3f}
   {df.iloc[1]['class_name']}: F1={df.iloc[1]['f1_score']:.3f}
   {df.iloc[2]['class_name']}: F1={df.iloc[2]['f1_score']:.3f}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        mode_str = "included" if self.include_background else "excluded"
        plt.suptitle(f'Foreground Performance Analysis - Background {mode_str.title()}\n'
                    f'Real Prediction Regions Only', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Foreground analysis saved: {save_path}")
        
        plt.show()
    
    def compare_real_vs_artificial_regions(self, save_path: Optional[str] = None):
        """
        比较真实预测区域和人工区域的差异（仅适用于背景排除模式）
        """
        if self.include_background:
            print("⚠️ This analysis is only applicable to background excluded mode")
            return None
        
        print("🔍 Comparing real prediction regions vs artificial background regions...")
        
        # 获取两个区域的数据
        real_softmax = self.softmax[self.real_prediction_mask]
        artificial_softmax = self.softmax[self.artificial_background_mask]
        
        print(f"  Real regions: {len(real_softmax):,} voxels")
        print(f"  Artificial regions: {len(artificial_softmax):,} voxels")
        
        # 计算统计差异
        real_entropy = -np.sum(real_softmax * np.log(real_softmax + 1e-10), axis=1)
        artificial_entropy = -np.sum(artificial_softmax * np.log(artificial_softmax + 1e-10), axis=1)
        
        real_max_prob = np.max(real_softmax, axis=1)
        artificial_max_prob = np.max(artificial_softmax, axis=1)
        
        # 可视化对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. 熵分布对比
        ax = axes[0]
        ax.hist(real_entropy, bins=50, alpha=0.7, label='Real Predictions', color='blue', density=True)
        ax.hist(artificial_entropy, bins=50, alpha=0.7, label='Artificial Background', color='red', density=True)
        ax.set_xlabel('Entropy (Uncertainty)')
        ax.set_ylabel('Density')
        ax.set_title('Uncertainty Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. 最大概率分布对比
        ax = axes[1]
        ax.hist(real_max_prob, bins=50, alpha=0.7, label='Real Predictions', color='blue', density=True)
        ax.hist(artificial_max_prob, bins=50, alpha=0.7, label='Artificial Background', color='red', density=True)
        ax.set_xlabel('Maximum Probability')
        ax.set_ylabel('Density')
        ax.set_title('Confidence Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 概率分布形状对比（前几个类别）
        ax = axes[2]
        n_classes_show = min(5, self.n_classes)
        x = np.arange(n_classes_show)
        width = 0.35
        
        real_class_means = np.mean(real_softmax[:, :n_classes_show], axis=0)
        artificial_class_means = np.mean(artificial_softmax[:, :n_classes_show], axis=0)
        
        ax.bar(x - width/2, real_class_means, width, label='Real', alpha=0.7, color='blue')
        ax.bar(x + width/2, artificial_class_means, width, label='Artificial', alpha=0.7, color='red')
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Mean Probability')
        ax.set_title(f'Mean Class Probabilities (First {n_classes_show} Classes)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 统计摘要
        ax = axes[3]
        ax.axis('off')
        
        summary_text = f"""Region Comparison Summary:

🔍 Real Prediction Regions:
   Voxels: {len(real_softmax):,}
   Mean Entropy: {np.mean(real_entropy):.4f}
   Std Entropy: {np.std(real_entropy):.4f}
   Mean Max Prob: {np.mean(real_max_prob):.4f}
   
🤖 Artificial Background Regions:
   Voxels: {len(artificial_softmax):,}
   Mean Entropy: {np.mean(artificial_entropy):.4f}
   Std Entropy: {np.std(artificial_entropy):.4f}
   Mean Max Prob: {np.mean(artificial_max_prob):.4f}
   
📊 Statistical Tests:
   Entropy Difference: {np.mean(real_entropy) - np.mean(artificial_entropy):.4f}
   Confidence Difference: {np.mean(artificial_max_prob) - np.mean(real_max_prob):.4f}
   
💡 Key Insights:
   • Artificial regions have near-zero entropy
   • Real regions show natural uncertainty patterns
   • This hybrid structure affects global statistics
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Real vs Artificial Region Comparison\nBackground Excluded Mode', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Region comparison saved: {save_path}")
        
        plt.show()
        
        return {
            'real_entropy': real_entropy,
            'artificial_entropy': artificial_entropy,
            'real_max_prob': real_max_prob,
            'artificial_max_prob': artificial_max_prob
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid data analysis for background excluded mode")
    parser.add_argument('-s', '--softmax', required=True, help="Softmax file path")
    parser.add_argument('-i', '--info', required=True, help="Info JSON file path")
    parser.add_argument('-l', '--labels', required=True, help="Labels file path")
    parser.add_argument('-o', '--output', help="Output path for visualizations")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = HybridDataAnalyzer(args.softmax, args.info, args.labels)
    
    # 前景性能分析
    fg_output = args.output + "_foreground.png" if args.output else None
    df = analyzer.analyze_foreground_performance(fg_output)
    
    # 如果是背景排除模式，执行区域对比分析
    if not analyzer.include_background:
        region_output = args.output + "_regions.png" if args.output else None  
        analyzer.compare_real_vs_artificial_regions(region_output)
    
    # 保存CSV报告
    if args.output and df is not None:
        csv_path = args.output.replace('.png', '_performance.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Performance report saved: {csv_path}")
    
    print(f"\n🎉 Hybrid data analysis completed!")

if __name__ == "__main__":
    main()