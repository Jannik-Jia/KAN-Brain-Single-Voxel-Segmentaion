#!/usr/bin/env python3
"""
概率分布分析工具
分析每个类别的softmax概率分布特征
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from scipy import stats

class ProbabilityDistributionAnalyzer:
    """概率分布分析器"""
    
    def __init__(self, softmax_path: str, info_path: str):
        """初始化分析器"""
        print("📊 Loading probability data...")
        
        self.softmax_nii = nib.load(softmax_path)
        self.softmax = self.softmax_nii.get_fdata()
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.n_classes = self.info['n_classes']
        self.class_names = self.info.get('class_names', list(range(self.n_classes)))
        self.include_background = self.info['include_background']
        
        print(f"✅ Data loaded: {self.softmax.shape}, {self.n_classes} classes")
        print(f"📋 Include background: {self.include_background}")
    
    def analyze_class_probability_distributions(self, prob_threshold: float = 0.01, 
                                              save_path: Optional[str] = None):
        """
        分析每个类别的概率分布
        
        Args:
            prob_threshold: 概率阈值，只分析高于此值的概率
            save_path: 保存路径
        """
        print(f"📈 Analyzing probability distributions (threshold: {prob_threshold})...")
        
        # 收集统计信息
        class_stats = []
        
        for class_id in range(self.n_classes):
            class_probs = self.softmax[..., class_id].flatten()
            
            # 过滤低概率值
            high_probs = class_probs[class_probs > prob_threshold]
            
            if len(high_probs) > 0:
                stats_dict = {
                    'class_id': class_id,
                    'class_name': (self.class_names[class_id] 
                                  if class_id < len(self.class_names) 
                                  else f'Class_{class_id}'),
                    'n_high_prob_voxels': len(high_probs),
                    'coverage_ratio': len(high_probs) / len(class_probs),
                    'mean': high_probs.mean(),
                    'std': high_probs.std(),
                    'median': np.median(high_probs),
                    'min': high_probs.min(),
                    'max': high_probs.max(),
                    'q25': np.percentile(high_probs, 25),
                    'q75': np.percentile(high_probs, 75),
                    'skewness': stats.skew(high_probs),
                    'kurtosis': stats.kurtosis(high_probs),
                    'entropy': -np.sum(high_probs * np.log(high_probs + 1e-10)),
                    'confidence_score': (high_probs > 0.8).mean(),  # 高置信度比例
                    'uncertainty_score': ((high_probs > 0.1) & (high_probs < 0.9)).mean()  # 中等置信度比例
                }
                class_stats.append(stats_dict)
        
        # 转换为DataFrame
        df = pd.DataFrame(class_stats)
        df = df.sort_values('n_high_prob_voxels', ascending=False)
        
        # 可视化
        self._visualize_probability_distributions(df, prob_threshold, save_path)
        
        return df
    
    def _visualize_probability_distributions(self, df: pd.DataFrame, 
                                           prob_threshold: float,
                                           save_path: Optional[str]):
        """可视化概率分布"""
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        # 1. 每个类别的体素数量（Coverage）
        ax = axes[0]
        top_classes = df.head(15)
        bars = ax.bar(range(len(top_classes)), top_classes['n_high_prob_voxels'])
        ax.set_title(f'Voxel Count per Class (prob > {prob_threshold})')
        ax.set_xlabel('Class Rank')
        ax.set_ylabel('Number of Voxels')
        ax.set_yscale('log')
        
        # 着色
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.tab20(i))
        
        # 2. 平均概率分布
        ax = axes[1]
        ax.hist(df['mean'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(df['mean'].mean(), color='red', linestyle='--', 
                  label=f'Overall Mean: {df["mean"].mean():.3f}')
        ax.set_xlabel('Mean Probability')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Distribution of Class Mean Probabilities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 置信度 vs 覆盖率
        ax = axes[2]
        scatter = ax.scatter(df['coverage_ratio'], df['confidence_score'], 
                           s=np.log10(df['n_high_prob_voxels'] + 1) * 20,
                           c=df['mean'], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Coverage Ratio')
        ax.set_ylabel('Confidence Score (prob > 0.8)')
        ax.set_title('Confidence vs Coverage')
        plt.colorbar(scatter, ax=ax, label='Mean Probability')
        ax.grid(True, alpha=0.3)
        
        # 4. 概率标准差分布
        ax = axes[3]
        ax.hist(df['std'], bins=25, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(df['std'].median(), color='red', linestyle='--',
                  label=f'Median Std: {df["std"].median():.3f}')
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Probability Standard Deviation Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 偏度分析
        ax = axes[4]
        ax.hist(df['skewness'], bins=25, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='Normal Distribution')
        ax.set_xlabel('Skewness')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Probability Distribution Skewness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Top类别的详细概率分布
        ax = axes[5]
        top_5_classes = df.head(5)
        
        for i, (_, row) in enumerate(top_5_classes.iterrows()):
            class_id = int(row['class_id'])
            class_probs = self.softmax[..., class_id].flatten()
            high_probs = class_probs[class_probs > prob_threshold]
            
            if len(high_probs) > 0:
                ax.hist(high_probs, bins=50, alpha=0.6, 
                       label=f"Class {class_id}", density=True)
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Top 5 Classes Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. 不确定性分析
        ax = axes[6]
        ax.scatter(df['confidence_score'], df['uncertainty_score'], 
                  s=60, alpha=0.7, c=df['mean'], cmap='RdYlBu_r')
        ax.set_xlabel('Confidence Score (prob > 0.8)')
        ax.set_ylabel('Uncertainty Score (0.1 < prob < 0.9)')
        ax.set_title('Confidence vs Uncertainty')
        ax.grid(True, alpha=0.3)
        
        # 8. 概率分布形状分析（峰度）
        ax = axes[7]
        ax.scatter(df['skewness'], df['kurtosis'], 
                  s=np.log10(df['n_high_prob_voxels'] + 1) * 10,
                  alpha=0.7, c=df['mean'], cmap='plasma')
        ax.set_xlabel('Skewness')
        ax.set_ylabel('Kurtosis')
        ax.set_title('Distribution Shape Analysis')
        ax.grid(True, alpha=0.3)
        
        # 9. 统计摘要
        ax = axes[8]
        ax.axis('off')
        
        summary_text = f"""Probability Distribution Summary:
        
        📊 Total Classes: {len(df)}
        📊 Avg High-Prob Voxels per Class: {df['n_high_prob_voxels'].mean():,.0f}
        📊 Avg Coverage Ratio: {df['coverage_ratio'].mean():.3f}
        
        🎯 Mean Probability Statistics:
           Mean: {df['mean'].mean():.3f}
           Std: {df['mean'].std():.3f}
           Range: [{df['mean'].min():.3f}, {df['mean'].max():.3f}]
        
        🎲 Confidence Analysis:
           Avg Confidence: {df['confidence_score'].mean():.3f}
           Avg Uncertainty: {df['uncertainty_score'].mean():.3f}
           
        🏆 Top 3 Most Confident Classes:
           {top_classes.iloc[0]['class_name']}: {top_classes.iloc[0]['confidence_score']:.3f}
           {top_classes.iloc[1]['class_name']}: {top_classes.iloc[1]['confidence_score']:.3f}
           {top_classes.iloc[2]['class_name']}: {top_classes.iloc[2]['confidence_score']:.3f}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Class Probability Distribution Analysis\n'
                    f'Background {"Included" if self.include_background else "Excluded"} | '
                    f'Threshold: {prob_threshold}', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Probability analysis saved: {save_path}")
        
        plt.show()
    
    def generate_detailed_class_report(self, prob_threshold: float = 0.01,
                                     save_path: Optional[str] = None):
        """生成详细的类别报告"""
        
        print("📝 Generating detailed class report...")
        
        df = self.analyze_class_probability_distributions(prob_threshold)
        
        if save_path:
            csv_path = save_path.replace('.png', '.csv') if save_path.endswith('.png') else save_path + '.csv'
            df.to_csv(csv_path, index=False)
            print(f"💾 Class report saved: {csv_path}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("📊 PROBABILITY DISTRIBUTION SUMMARY")
        print("="*80)
        
        print(f"Background Mode: {'Included' if self.include_background else 'Excluded'}")
        print(f"Probability Threshold: {prob_threshold}")
        print(f"Total Classes Analyzed: {len(df)}")
        
        print(f"\n🏆 Top 10 Classes by Coverage:")
        top_10 = df.head(10)
        for _, row in top_10.iterrows():
            print(f"  {row['class_name']}: {row['n_high_prob_voxels']:,} voxels "
                  f"(coverage: {row['coverage_ratio']:.1%}, "
                  f"confidence: {row['confidence_score']:.3f})")
        
        print(f"\n🎲 Confidence Distribution:")
        high_conf = (df['confidence_score'] > 0.8).sum()
        med_conf = ((df['confidence_score'] > 0.5) & (df['confidence_score'] <= 0.8)).sum()
        low_conf = (df['confidence_score'] <= 0.5).sum()
        
        print(f"  High Confidence (>0.8): {high_conf} classes")
        print(f"  Medium Confidence (0.5-0.8): {med_conf} classes") 
        print(f"  Low Confidence (≤0.5): {low_conf} classes")
        
        return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze probability distributions")
    parser.add_argument('-s', '--softmax', required=True, help="Softmax file path")
    parser.add_argument('-i', '--info', required=True, help="Info JSON file path")
    parser.add_argument('-t', '--threshold', type=float, default=0.01, 
                       help="Probability threshold (default: 0.01)")
    parser.add_argument('-o', '--output', help="Output path for visualization")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ProbabilityDistributionAnalyzer(args.softmax, args.info)
    
    # 运行分析
    df = analyzer.generate_detailed_class_report(
        prob_threshold=args.threshold,
        save_path=args.output
    )
    
    print(f"\n🎉 Analysis completed! Generated {len(df)} class reports.")

if __name__ == "__main__":
    main()