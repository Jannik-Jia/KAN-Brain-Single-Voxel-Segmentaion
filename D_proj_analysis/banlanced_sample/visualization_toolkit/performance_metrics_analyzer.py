#!/usr/bin/env python3
"""
性能指标分析工具
针对每个解剖标签计算详细的性能指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetricsAnalyzer:
    """性能指标分析器"""
    
    def __init__(self, softmax_path: str, info_path: str, label_path: str):
        """初始化分析器"""
        print("📊 Loading performance analysis data...")
        
        # 加载数据
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
        
        # 验证形状
        if self.softmax.shape[:3] != self.labels.shape:
            raise ValueError(f"Shape mismatch! Softmax: {self.softmax.shape[:3]}, Labels: {self.labels.shape}")
        
        print(f"✅ Data loaded successfully")
        print(f"📐 Volume shape: {self.labels.shape}")
        print(f"📋 Classes: {self.n_classes}")
        print(f"🎯 Background: {'Included' if self.include_background else 'Excluded'}")
        
        # 识别有效区域
        self._identify_analysis_regions()
    
    def _identify_analysis_regions(self):
        """识别分析区域"""
        
        # 获取所有体素的展平版本
        self.labels_flat = self.labels.flatten()
        self.predictions_flat = self.predictions.flatten()
        self.softmax_flat = self.softmax.reshape(-1, self.n_classes)
        
        # 根据背景模式确定分析策略
        if self.include_background:
            # 包含背景：分析所有体素
            self.analysis_mask = np.ones_like(self.labels_flat, dtype=bool)
            self.analysis_mode = "Full Volume"
        else:
            # 排除背景：只分析前景体素（label≠0）
            self.analysis_mask = self.labels_flat != 0
            self.analysis_mode = "Foreground Only"
        
        # 应用掩码
        self.valid_labels = self.labels_flat[self.analysis_mask]
        self.valid_predictions = self.predictions_flat[self.analysis_mask]
        self.valid_softmax = self.softmax_flat[self.analysis_mask]
        
        print(f"🎯 Analysis Mode: {self.analysis_mode}")
        print(f"📊 Total voxels: {len(self.labels_flat):,}")
        print(f"📊 Analysis voxels: {len(self.valid_labels):,} ({len(self.valid_labels)/len(self.labels_flat)*100:.1f}%)")
        
        # 检查类别分布
        unique_labels = np.unique(self.valid_labels)
        print(f"📊 Unique labels in analysis: {len(unique_labels)} classes")
    
    def compute_detailed_metrics(self) -> pd.DataFrame:
        """计算详细的性能指标"""
        
        print("🔢 Computing detailed performance metrics...")
        
        # 计算全局指标
        overall_accuracy = accuracy_score(self.valid_labels, self.valid_predictions)
        
        # 计算类别级指标
        precision, recall, f1, support = precision_recall_fscore_support(
            self.valid_labels, self.valid_predictions,
            labels=range(self.n_classes), zero_division=0
        )
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(
            self.valid_labels, self.valid_predictions,
            labels=range(self.n_classes)
        )
        
        # 构建结果DataFrame
        results = []
        
        for class_id in range(self.n_classes):
            class_name = (self.class_names[class_id] 
                         if class_id < len(self.class_names) 
                         else f'Class_{class_id}')
            
            # 基本指标
            metrics = {
                'class_id': class_id,
                'class_name': class_name,
                'support': support[class_id],
                'precision': precision[class_id],
                'recall': recall[class_id],
                'f1_score': f1[class_id],
            }
            
            # 计算类别特定的准确率
            if support[class_id] > 0:
                class_mask = self.valid_labels == class_id
                class_accuracy = (self.valid_predictions[class_mask] == class_id).mean()
                metrics['accuracy'] = class_accuracy
            else:
                metrics['accuracy'] = np.nan
            
            # 计算IoU和Dice系数
            pred_mask = self.valid_predictions == class_id
            true_mask = self.valid_labels == class_id
            
            intersection = (pred_mask & true_mask).sum()
            union = (pred_mask | true_mask).sum()
            
            if union > 0:
                iou = intersection / union
                dice = 2 * intersection / (pred_mask.sum() + true_mask.sum())
            else:
                iou = np.nan
                dice = np.nan
            
            metrics['iou'] = iou
            metrics['dice'] = dice
            
            # 计算平均预测置信度
            class_probs = self.valid_softmax[:, class_id]
            if support[class_id] > 0:
                # 对于真实标签为此类的体素，计算平均置信度
                true_class_probs = class_probs[self.valid_labels == class_id]
                metrics['mean_confidence'] = true_class_probs.mean()
                metrics['confidence_std'] = true_class_probs.std()
            else:
                metrics['mean_confidence'] = np.nan
                metrics['confidence_std'] = np.nan
            
            # 计算错误分析
            if support[class_id] > 0:
                # 最常被误分类为哪个类别
                wrong_preds = self.valid_predictions[self.valid_labels == class_id]
                wrong_preds = wrong_preds[wrong_preds != class_id]
                
                if len(wrong_preds) > 0:
                    most_confused_class = np.bincount(wrong_preds).argmax()
                    confusion_rate = (wrong_preds == most_confused_class).sum() / support[class_id]
                    metrics['most_confused_with'] = most_confused_class
                    metrics['confusion_rate'] = confusion_rate
                else:
                    metrics['most_confused_with'] = -1  # 没有误分类
                    metrics['confusion_rate'] = 0.0
            else:
                metrics['most_confused_with'] = -1
                metrics['confusion_rate'] = np.nan
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # 添加排名
        for metric in ['precision', 'recall', 'f1_score', 'iou', 'dice']:
            df[f'{metric}_rank'] = df[metric].rank(ascending=False, method='min', na_option='bottom')
        
        # 按F1分数排序
        df = df.sort_values('f1_score', ascending=False, na_position='last')
        
        # 保存全局信息
        self.overall_accuracy = overall_accuracy
        self.conf_matrix = conf_matrix
        self.metrics_df = df
        
        return df
    
    def visualize_performance_metrics(self, save_path: Optional[str] = None):
        """可视化性能指标"""
        
        print("🎨 Creating performance visualizations...")
        
        df = self.metrics_df
        
        # 过滤有效类别（有支持的类别）
        valid_df = df[df['support'] > 0].copy()
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        # 1. 支持度分布
        ax = axes[0]
        top_classes = valid_df.head(15)
        bars = ax.bar(range(len(top_classes)), top_classes['support'])
        ax.set_title('Sample Support by Class (Top 15)')
        ax.set_xlabel('Class Rank')
        ax.set_ylabel('Number of Samples')
        ax.set_yscale('log')
        
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.tab20(i))
        
        # 2. F1分数分布
        ax = axes[1]
        valid_f1 = valid_df['f1_score'].dropna()
        ax.hist(valid_f1, bins=25, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(valid_f1.mean(), color='red', linestyle='--',
                  label=f'Mean: {valid_f1.mean():.3f}')
        ax.axvline(valid_f1.median(), color='orange', linestyle='--',
                  label=f'Median: {valid_f1.median():.3f}')
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Number of Classes')
        ax.set_title('F1 Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Precision vs Recall
        ax = axes[2]
        scatter = ax.scatter(valid_df['recall'], valid_df['precision'],
                           s=np.log10(valid_df['support'] + 1) * 20,
                           c=valid_df['f1_score'], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall (size=log(support))')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        plt.colorbar(scatter, ax=ax, label='F1 Score')
        ax.grid(True, alpha=0.3)
        
        # 4. IoU vs Dice相关性
        ax = axes[3]
        valid_iou_dice = valid_df.dropna(subset=['iou', 'dice'])
        if len(valid_iou_dice) > 0:
            ax.scatter(valid_iou_dice['iou'], valid_iou_dice['dice'], alpha=0.7)
            
            # 理论曲线
            iou_theory = np.linspace(0, 1, 100)
            dice_theory = 2 * iou_theory / (1 + iou_theory)
            ax.plot(iou_theory, dice_theory, 'r--', alpha=0.5, label='Theoretical')
            
            ax.set_xlabel('IoU Score')
            ax.set_ylabel('Dice Score')
            ax.set_title('IoU vs Dice Correlation')
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 支持度对性能的影响
        ax = axes[4]
        support_log = np.log10(valid_df['support'] + 1)
        ax.scatter(support_log, valid_df['f1_score'], alpha=0.6, color='green')
        
        # 趋势线
        valid_mask = ~np.isnan(valid_df['f1_score'])
        if valid_mask.sum() > 1:
            z = np.polyfit(support_log[valid_mask], valid_df['f1_score'][valid_mask], 1)
            p = np.poly1d(z)
            ax.plot(support_log, p(support_log), "r--", alpha=0.8,
                   label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()
        
        ax.set_xlabel('Log10(Support + 1)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Sample Size Effect on Performance')
        ax.grid(True, alpha=0.3)
        
        # 6. 置信度分析
        ax = axes[5]
        conf_df = valid_df.dropna(subset=['mean_confidence'])
        if len(conf_df) > 0:
            ax.scatter(conf_df['mean_confidence'], conf_df['f1_score'],
                      s=60, alpha=0.7, c=conf_df['support'], 
                      cmap='plasma', norm=plt.Normalize(vmin=conf_df['support'].min(),
                                                       vmax=conf_df['support'].max()))
            ax.set_xlabel('Mean Prediction Confidence')
            ax.set_ylabel('F1 Score')
            ax.set_title('Confidence vs Performance')
        ax.grid(True, alpha=0.3)
        
        # 7. 性能热图（Top类别）
        ax = axes[6]
        top_15 = valid_df.head(15)
        if len(top_15) > 0:
            metrics_matrix = top_15[['precision', 'recall', 'f1_score', 'iou', 'dice']].T
            
            im = ax.imshow(metrics_matrix.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_yticks(range(5))
            ax.set_yticklabels(['Precision', 'Recall', 'F1', 'IoU', 'Dice'])
            ax.set_xticks(range(len(top_15)))
            ax.set_xticklabels([f'C{int(x)}' for x in top_15['class_id']], rotation=45)
            ax.set_title('Performance Metrics Heatmap (Top 15 Classes)')
            plt.colorbar(im, ax=ax)
        
        # 8. 错误分析
        ax = axes[7]
        confusion_df = valid_df[valid_df['confusion_rate'] > 0].head(10)
        if len(confusion_df) > 0:
            bars = ax.bar(range(len(confusion_df)), confusion_df['confusion_rate'])
            ax.set_title('Top 10 Most Confused Classes')
            ax.set_xlabel('Class Rank')
            ax.set_ylabel('Confusion Rate')
            ax.set_xticklabels([f'C{int(x)}→C{int(y)}' for x, y in 
                               zip(confusion_df['class_id'], confusion_df['most_confused_with'])],
                              rotation=45)
            
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.Reds(confusion_df.iloc[i]['confusion_rate']))
        
        # 9. 综合统计摘要
        ax = axes[8]
        ax.axis('off')
        
        summary_text = f"""Performance Metrics Summary:
        
        📊 Analysis Mode: {self.analysis_mode}
        📊 Overall Accuracy: {self.overall_accuracy:.3f}
        📊 Classes Analyzed: {len(valid_df)}
        
        🎯 F1 Score Statistics:
           Mean: {valid_df['f1_score'].mean():.3f}
           Std: {valid_df['f1_score'].std():.3f}
           Median: {valid_df['f1_score'].median():.3f}
           Range: [{valid_df['f1_score'].min():.3f}, {valid_df['f1_score'].max():.3f}]
        
        🏆 Top 3 Performing Classes:
           {valid_df.iloc[0]['class_name']}: F1={valid_df.iloc[0]['f1_score']:.3f}
           {valid_df.iloc[1]['class_name']}: F1={valid_df.iloc[1]['f1_score']:.3f}
           {valid_df.iloc[2]['class_name']}: F1={valid_df.iloc[2]['f1_score']:.3f}
           
        📊 IoU/Dice Statistics:
           Mean IoU: {valid_df['iou'].mean():.3f}
           Mean Dice: {valid_df['dice'].mean():.3f}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'Performance Metrics Analysis\n'
                    f'Background {"Included" if self.include_background else "Excluded"} | '
                    f'Overall Accuracy: {self.overall_accuracy:.3f}',
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Performance visualization saved: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, save_path: Optional[str] = None):
        """生成性能报告"""
        
        print("📝 Generating performance report...")
        
        # 计算指标
        df = self.compute_detailed_metrics()
        
        # 可视化
        if save_path:
            viz_path = save_path.replace('.csv', '_visualization.png') if save_path.endswith('.csv') else save_path + '_visualization.png'
            self.visualize_performance_metrics(viz_path)
        else:
            self.visualize_performance_metrics()
        
        # 保存CSV报告
        if save_path:
            csv_path = save_path if save_path.endswith('.csv') else save_path + '.csv'
            df.to_csv(csv_path, index=False)
            print(f"💾 Performance report saved: {csv_path}")
        
        # 打印摘要
        self._print_performance_summary(df)
        
        return df
    
    def _print_performance_summary(self, df: pd.DataFrame):
        """打印性能摘要"""
        
        valid_df = df[df['support'] > 0]
        
        print("\n" + "="*80)
        print("📊 PERFORMANCE METRICS SUMMARY")
        print("="*80)
        
        print(f"Analysis Mode: {self.analysis_mode}")
        print(f"Overall Accuracy: {self.overall_accuracy:.4f}")
        print(f"Classes with Data: {len(valid_df)}/{len(df)}")
        
        print(f"\n🎯 Performance Statistics:")
        for metric in ['f1_score', 'precision', 'recall', 'iou', 'dice']:
            values = valid_df[metric].dropna()
            if len(values) > 0:
                print(f"  {metric.upper()}: μ={values.mean():.3f}, σ={values.std():.3f}, "
                      f"range=[{values.min():.3f}, {values.max():.3f}]")
        
        print(f"\n🏆 Top 10 Classes by F1 Score:")
        top_10 = valid_df.head(10)
        for _, row in top_10.iterrows():
            print(f"  {row['class_name']}: F1={row['f1_score']:.3f}, "
                  f"Prec={row['precision']:.3f}, Rec={row['recall']:.3f}, "
                  f"Support={int(row['support'])}")
        
        print(f"\n⚠️ Bottom 5 Classes by F1 Score (with support > 100):")
        bottom_classes = valid_df[valid_df['support'] > 100].tail(5)
        for _, row in bottom_classes.iterrows():
            print(f"  {row['class_name']}: F1={row['f1_score']:.3f}, "
                  f"Support={int(row['support'])}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze performance metrics")
    parser.add_argument('-s', '--softmax', required=True, help="Softmax file path")
    parser.add_argument('-i', '--info', required=True, help="Info JSON file path")
    parser.add_argument('-l', '--labels', required=True, help="Labels file path")
    parser.add_argument('-o', '--output', help="Output path for reports")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = PerformanceMetricsAnalyzer(args.softmax, args.info, args.labels)
    
    # 运行分析
    df = analyzer.generate_performance_report(save_path=args.output)
    
    print(f"\n🎉 Analysis completed! Generated report for {len(df)} classes.")

if __name__ == "__main__":
    main()