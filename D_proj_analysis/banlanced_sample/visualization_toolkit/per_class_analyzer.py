#!/usr/bin/env python3
"""
独立的Per-Class性能分析工具
可以直接对现有的softmax文件进行详细的每类别分析
"""

import numpy as np
import nibabel as nib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
import logging
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report,
    balanced_accuracy_score
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_softmax_and_labels(softmax_path, labels_path, info_path=None):
    """
    加载softmax预测文件和标签文件
    """
    logger.info(f"📂 加载数据文件...")
    
    # 加载softmax预测
    softmax_nii = nib.load(softmax_path)
    softmax = softmax_nii.get_fdata()
    logger.info(f"  Softmax形状: {softmax.shape}")
    
    # 加载标签
    labels_nii = nib.load(labels_path)
    labels = labels_nii.get_fdata()
    logger.info(f"  Labels形状: {labels.shape}")
    
    # 计算argmax预测
    predictions = np.argmax(softmax, axis=-1)
    logger.info(f"  Predictions形状: {predictions.shape}")
    
    # 展平为1D数组
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()
    
    # 加载info文件（如果提供）
    info = None
    if info_path and Path(info_path).exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        logger.info(f"  Info文件已加载")
    
    logger.info(f"✅ 数据加载完成: {len(labels_flat):,} 个体素")
    
    return {
        'softmax': softmax,
        'labels': labels,
        'predictions': predictions,
        'labels_flat': labels_flat,
        'predictions_flat': predictions_flat,
        'info': info
    }

def calculate_per_class_metrics_detailed(y_true, y_pred, class_names=None, volume_info=None, reverse_mapping=None):
    """
    计算详细的per-class指标，特别适用于医学图像分割
    """
    logger.info("🧮 计算per-class指标...")
    
    # 获取所有出现的类别
    unique_labels = np.unique(y_true)
    unique_preds = np.unique(y_pred)
    all_classes = np.unique(np.concatenate([unique_labels, unique_preds]))
    
    logger.info(f"  标签中的类别: {len(unique_labels)} 个")
    logger.info(f"  预测中的类别: {len(unique_preds)} 个") 
    logger.info(f"  总共分析: {len(all_classes)} 个类别")
    
    if class_names is None:
        class_names = [f"Class_{int(i)}" for i in all_classes]
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    
    # 为每个类别计算指标
    per_class_results = []
    
    for i, class_id in enumerate(all_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{int(class_id)}"
        
        # 二分类指标计算
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        # 基本计数
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        # 基本指标
        support = np.sum(y_true_binary)  # 真实样本数
        predicted_positive = np.sum(y_pred_binary)  # 预测为正样本数
        
        # 计算性能指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 医学图像分割常用指标
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0  # Dice系数
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # IoU/Jaccard
        
        # 准确率和平衡准确率
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2
        
        # 额外的诊断信息
        prevalence = support / len(y_true)  # 该类别在数据中的占比
        predicted_prevalence = predicted_positive / len(y_pred)  # 预测的占比
        
        # 检测能力指标
        positive_likelihood_ratio = recall / (1 - specificity) if specificity < 1 else float('inf')
        negative_likelihood_ratio = (1 - recall) / specificity if specificity > 0 else float('inf')
        
        # 确定显示的类别ID (如果有反向映射，显示原始FreeSurfer标签)
        display_class_id = int(class_id)
        if reverse_mapping is not None and class_id in reverse_mapping:
            display_class_id = reverse_mapping[class_id]
        
        # 汇总结果
        class_result = {
            'class_id': display_class_id,  # 显示原始FreeSurfer标签ID
            'internal_class_id': int(class_id),  # 内部连续索引
            'class_name': class_name,
            
            # 基本计数
            'support': int(support),
            'predicted_positive': int(predicted_positive),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            
            # 核心性能指标
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            
            # 医学图像分割指标
            'dice_coefficient': float(dice),
            'iou_jaccard': float(iou),
            
            # 准确率指标
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            
            # 统计信息
            'prevalence': float(prevalence),
            'predicted_prevalence': float(predicted_prevalence),
            
            # 诊断能力
            'positive_lr': float(positive_likelihood_ratio) if not np.isinf(positive_likelihood_ratio) else None,
            'negative_lr': float(negative_likelihood_ratio) if not np.isinf(negative_likelihood_ratio) else None,
        }
        
        per_class_results.append(class_result)
    
    # 计算总体指标
    overall_metrics = {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'overall_accuracy': float(np.mean(y_true == y_pred)),
        'total_samples': int(len(y_true)),
        'total_classes_in_labels': int(len(unique_labels)),
        'total_classes_predicted': int(len(unique_preds)),
        'total_classes_analyzed': int(len(all_classes)),
        'has_reverse_mapping': reverse_mapping is not None,
        'mapping_info': 'FreeSurfer labels mapped to continuous indices' if reverse_mapping else 'Direct class IDs'
    }
    
    return {
        'per_class_metrics': per_class_results,
        'overall_metrics': overall_metrics,
        'confusion_matrix': cm.tolist(),
        'class_ids': all_classes.tolist()
    }

def create_comprehensive_visualizations(metrics_dict, output_dir):
    """创建详细的per-class可视化对比报告"""
    
    logger.info("📊 生成详细的per-class可视化对比...")
    output_dir = Path(output_dir)
    
    # 转换为DataFrame便于操作
    df = pd.DataFrame(metrics_dict['per_class_metrics'])
    
    # 设置图形样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. 创建每个指标的详细对比图
    create_individual_metric_comparisons(df, output_dir)
    
    # 2. 创建综合概览图
    create_comprehensive_overview(df, output_dir)
    
    # 3. 创建性能排名分析
    create_performance_ranking_analysis(df, output_dir)
    
    # 4. 创建相关性分析
    create_correlation_analysis(df, output_dir)

def create_individual_metric_comparisons(df, output_dir):
    """为每个指标创建详细的class-by-class对比图"""
    
    logger.info("  📊 生成各指标的详细对比图...")
    
    # 核心指标列表
    core_metrics = [
        ('f1_score', 'F1 Score', 'F1分数'),
        ('dice_coefficient', 'Dice Coefficient', 'Dice系数'),
        ('precision', 'Precision', '精确率'),
        ('recall', 'Recall (Sensitivity)', '召回率/敏感性'),
        ('specificity', 'Specificity', '特异性'),
        ('iou_jaccard', 'IoU (Jaccard Index)', 'IoU交并比'),
        ('balanced_accuracy', 'Balanced Accuracy', '平衡准确率')
    ]
    
    n_classes = len(df)
    
    for metric_col, metric_title, metric_title_zh in core_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f'{metric_title} - Class-by-Class Comparison\n{metric_title_zh} 逐类别对比', 
                     fontsize=16, fontweight='bold')
        
        # 上图：所有类别的柱状图
        ax1 = axes[0]
        
        # 按指标值排序以便更好地可视化
        df_sorted = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
        
        # 创建颜色映射（基于性能水平）
        colors = []
        for value in df_sorted[metric_col]:
            if value >= 0.8:
                colors.append('green')      # 优秀
            elif value >= 0.6:
                colors.append('lightgreen') # 良好
            elif value >= 0.3:
                colors.append('yellow')     # 一般
            elif value > 0.05:
                colors.append('orange')     # 较差
            else:
                colors.append('red')        # 失败
        
        bars = ax1.bar(range(len(df_sorted)), df_sorted[metric_col], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax1.set_title(f'{metric_title} for All Classes (Sorted by Performance)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class Rank (Best to Worst)')
        ax1.set_ylabel(metric_title)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加平均线和标准差区间
        mean_val = df[metric_col].mean()
        std_val = df[metric_col].std()
        ax1.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        ax1.axhspan(mean_val-std_val, mean_val+std_val, alpha=0.1, color='red', 
                   label=f'±1 Std: {std_val:.3f}')
        
        # 添加性能阈值线
        if metric_col in ['f1_score', 'dice_coefficient', 'precision', 'recall']:
            ax1.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Excellent (0.8)')
            ax1.axhline(y=0.6, color='yellow', linestyle=':', alpha=0.7, label='Good (0.6)')
            ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='Fair (0.3)')
        
        ax1.legend(loc='upper right')
        
        # 为前20个添加类别ID标签
        for i in range(min(20, len(df_sorted))):
            bar = bars[i]
            height = bar.get_height()
            if height > 0.01:  # 只为非零值添加标签
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'ID:{int(df_sorted.iloc[i]["class_id"])}', 
                        ha='center', va='bottom', fontsize=8, rotation=45)
        
        # 下图：按原始类别ID顺序的柱状图
        ax2 = axes[1]
        
        # 按class_id排序
        df_by_id = df.sort_values('class_id').reset_index(drop=True)
        
        # 使用相同的颜色编码
        colors_by_id = []
        for value in df_by_id[metric_col]:
            if value >= 0.8:
                colors_by_id.append('green')
            elif value >= 0.6:
                colors_by_id.append('lightgreen')
            elif value >= 0.3:
                colors_by_id.append('yellow')
            elif value > 0.05:
                colors_by_id.append('orange')
            else:
                colors_by_id.append('red')
        
        bars2 = ax2.bar(range(len(df_by_id)), df_by_id[metric_col], 
                       color=colors_by_id, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.set_title(f'{metric_title} by Original Class ID Order', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class Index (by Class ID)')
        ax2.set_ylabel(metric_title)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加同样的统计线
        ax2.axhline(y=mean_val, color='red', linestyle='--', linewidth=2)
        ax2.axhspan(mean_val-std_val, mean_val+std_val, alpha=0.1, color='red')
        
        # 设置x轴标签为class_id（每10个显示一个）
        step = max(1, len(df_by_id) // 20)  # 最多显示20个标签
        tick_positions = range(0, len(df_by_id), step)
        tick_labels = [f'{int(df_by_id.iloc[i]["class_id"])}' for i in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        metric_viz_path = output_dir / f"{metric_col}_detailed_comparison.png"
        plt.savefig(metric_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    📊 {metric_title}对比图已保存: {metric_viz_path}")
    
    # 创建support分布的特殊可视化（使用对数刻度）
    create_support_distribution_viz(df, output_dir)

def create_support_distribution_viz(df, output_dir):
    """创建样本数量分布的特殊可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Support (Sample Count) Distribution Analysis\n样本数量分布分析', 
                 fontsize=16, fontweight='bold')
    
    # 按support排序
    df_by_support = df.sort_values('support', ascending=False).reset_index(drop=True)
    
    # 1. 对数刻度柱状图
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(df_by_support)), df_by_support['support'], 
                  color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax1.set_title('Support Distribution (Log Scale)', fontweight='bold')
    ax1.set_xlabel('Class Rank (Most to Least Samples)')
    ax1.set_ylabel('Sample Count (Log Scale)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_support = df['support'].mean()
    median_support = df['support'].median()
    ax1.axhline(y=mean_support, color='blue', linestyle='--', label=f'Mean: {mean_support:.0f}')
    ax1.axhline(y=median_support, color='green', linestyle='--', label=f'Median: {median_support:.0f}')
    ax1.legend()
    
    # 2. Support vs F1性能散点图
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['support'], df['f1_score'], 
                         alpha=0.6, s=60, c='purple')
    ax2.set_title('Support vs F1 Performance', fontweight='bold')
    ax2.set_xlabel('Support (Sample Count, Log Scale)')
    ax2.set_ylabel('F1 Score')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 添加趋势线
    from scipy import stats
    log_support = np.log10(df['support'] + 1)  # +1 to handle zeros
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_support, df['f1_score'])
    line_x = np.logspace(0, np.log10(df['support'].max()), 100)
    line_y = slope * np.log10(line_x + 1) + intercept
    ax2.plot(line_x, line_y, 'r--', alpha=0.8, label=f'Trend (R²={r_value**2:.3f})')
    ax2.legend()
    
    # 3. 类别不平衡分析
    ax3 = axes[1, 0]
    
    # 按support分组
    support_ranges = [
        ('Very High (>10k)', df['support'] > 10000),
        ('High (1k-10k)', (df['support'] >= 1000) & (df['support'] <= 10000)),
        ('Medium (100-1k)', (df['support'] >= 100) & (df['support'] < 1000)),
        ('Low (10-100)', (df['support'] >= 10) & (df['support'] < 100)),
        ('Very Low (<10)', df['support'] < 10)
    ]
    
    support_counts = []
    support_labels = []
    avg_f1_by_support = []
    
    for label, mask in support_ranges:
        count = mask.sum()
        if count > 0:
            support_counts.append(count)
            support_labels.append(label)
            avg_f1_by_support.append(df[mask]['f1_score'].mean())
    
    bars = ax3.bar(support_labels, support_counts, alpha=0.7, color='lightblue', edgecolor='navy')
    ax3.set_title('Class Count by Support Range', fontweight='bold')
    ax3.set_ylabel('Number of Classes')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加平均F1信息
    for i, (bar, avg_f1) in enumerate(zip(bars, avg_f1_by_support)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'Avg F1:\n{avg_f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 累积分布
    ax4 = axes[1, 1]
    
    sorted_supports = np.sort(df['support'].values)
    cumulative_pct = np.arange(1, len(sorted_supports) + 1) / len(sorted_supports) * 100
    
    ax4.plot(sorted_supports, cumulative_pct, 'b-', linewidth=2)
    ax4.set_title('Cumulative Support Distribution', fontweight='bold')
    ax4.set_xlabel('Sample Count (Log Scale)')
    ax4.set_ylabel('Cumulative Percentage of Classes')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 添加百分位数线
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(sorted_supports, p)
        ax4.axvline(x=value, color='red', linestyle='--', alpha=0.7)
        ax4.text(value, p, f'  {p}th: {value:.0f}', rotation=90, va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    support_viz_path = output_dir / "support_distribution_analysis.png"
    plt.savefig(support_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    📊 样本数量分布分析已保存: {support_viz_path}")

def create_comprehensive_overview(df, output_dir):
    """创建综合概览图（原有的功能）"""
    
    logger.info("  📊 生成综合概览图...")
    
    # 1. 主要性能指标概览 (4x2 layout)
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle('Per-Class Performance Analysis - Comprehensive Overview', fontsize=16, fontweight='bold')
    
    # F1 Score分布
    ax = axes[0, 0]
    bars = ax.bar(range(len(df)), df['f1_score'], color='skyblue', alpha=0.7, edgecolor='navy')
    ax.set_title('F1 Score per Class', fontweight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 添加平均线
    mean_f1 = df['f1_score'].mean()
    ax.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.7, label=f'Mean F1: {mean_f1:.3f}')
    ax.legend()
    
    # Precision vs Recall散点图
    ax = axes[0, 1]
    scatter = ax.scatter(df['recall'], df['precision'], 
                        s=np.sqrt(df['support'])*2, alpha=0.6, 
                        c=df['f1_score'], cmap='viridis')
    ax.set_title('Precision vs Recall\n(size ∝ √support, color = F1)', fontweight='bold')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # 对角线
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # Dice coefficient比较
    ax = axes[1, 0]
    ax.bar(range(len(df)), df['dice_coefficient'], color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax.set_title('Dice Coefficient per Class', fontweight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Dice Coefficient')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    mean_dice = df['dice_coefficient'].mean()
    ax.axhline(y=mean_dice, color='red', linestyle='--', alpha=0.7, label=f'Mean Dice: {mean_dice:.3f}')
    ax.legend()
    
    # Support分布 (样本数量)
    ax = axes[1, 1]
    bars = ax.bar(range(len(df)), df['support'], color='orange', alpha=0.7, edgecolor='darkorange')
    ax.set_title('Support (Sample Count) per Class', fontweight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Sample Count (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Sensitivity (Recall) vs Specificity
    ax = axes[2, 0]
    ax.scatter(df['recall'], df['specificity'], 
              s=np.sqrt(df['support'])*2, alpha=0.6, 
              c=df['balanced_accuracy'], cmap='plasma')
    ax.set_title('Sensitivity vs Specificity\n(size ∝ √support, color = Balanced Acc)', fontweight='bold')
    ax.set_xlabel('Sensitivity (Recall)')
    ax.set_ylabel('Specificity')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 性能分级分布
    ax = axes[2, 1]
    
    # 按F1 score分级
    excellent = sum(1 for x in df['f1_score'] if x >= 0.8)
    good = sum(1 for x in df['f1_score'] if 0.6 <= x < 0.8)
    fair = sum(1 for x in df['f1_score'] if 0.3 <= x < 0.6)
    poor = sum(1 for x in df['f1_score'] if 0.05 <= x < 0.3)
    zero = sum(1 for x in df['f1_score'] if x < 0.05)
    
    categories = ['Excellent\n(≥0.8)', 'Good\n(0.6-0.8)', 'Fair\n(0.3-0.6)', 'Poor\n(0.05-0.3)', 'Failed\n(<0.05)']
    counts = [excellent, good, fair, poor, zero]
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Performance Distribution\n(by F1 Score)', fontweight='bold')
    
    # Prevalence vs Performance
    ax = axes[3, 0]
    ax.scatter(df['prevalence'], df['f1_score'], 
              s=60, alpha=0.6, c='purple')
    ax.set_title('Class Prevalence vs F1 Performance', fontweight='bold')
    ax.set_xlabel('Class Prevalence (fraction of total)')
    ax.set_ylabel('F1 Score')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 预测vs真实分布对比
    ax = axes[3, 1]
    x_pos = np.arange(len(df))
    width = 0.35
    
    ax.bar(x_pos - width/2, df['prevalence'], width, label='True Prevalence', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, df['predicted_prevalence'], width, label='Predicted Prevalence', alpha=0.7, color='red')
    
    ax.set_title('True vs Predicted Class Prevalence', fontweight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Prevalence (fraction)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存主要性能图
    main_viz_path = output_dir / "comprehensive_per_class_analysis.png"
    plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 主要分析图已保存: {main_viz_path}")
    
    # 2. 详细的Top/Bottom performers分析
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Top and Bottom Performers Analysis', fontsize=16, fontweight='bold')
    
    # 按不同指标排序
    df_sorted_f1 = df.sort_values('f1_score', ascending=False)
    df_sorted_dice = df.sort_values('dice_coefficient', ascending=False)
    
    # Top 15 by F1
    ax = axes[0, 0]
    top_15_f1 = df_sorted_f1.head(15)
    bars = ax.barh(range(len(top_15_f1)), top_15_f1['f1_score'], color='green', alpha=0.7)
    ax.set_title('Top 15 Classes by F1 Score', fontweight='bold')
    ax.set_xlabel('F1 Score')
    ax.set_yticks(range(len(top_15_f1)))
    ax.set_yticklabels([f"ID:{int(x)}" for x in top_15_f1['class_id']])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Top 15 by Dice
    ax = axes[0, 1]
    top_15_dice = df_sorted_dice.head(15)
    bars = ax.barh(range(len(top_15_dice)), top_15_dice['dice_coefficient'], color='lightgreen', alpha=0.7)
    ax.set_title('Top 15 Classes by Dice Coefficient', fontweight='bold')
    ax.set_xlabel('Dice Coefficient')
    ax.set_yticks(range(len(top_15_dice)))
    ax.set_yticklabels([f"ID:{int(x)}" for x in top_15_dice['class_id']])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Bottom performers (F1 > 0)
    ax = axes[1, 0]
    bottom_candidates_f1 = df_sorted_f1[df_sorted_f1['f1_score'] > 0].tail(15)
    if len(bottom_candidates_f1) > 0:
        bars = ax.barh(range(len(bottom_candidates_f1)), bottom_candidates_f1['f1_score'], color='red', alpha=0.7)
        ax.set_title('Bottom 15 Classes by F1 Score (F1>0)', fontweight='bold')
        ax.set_xlabel('F1 Score')
        ax.set_yticks(range(len(bottom_candidates_f1)))
        ax.set_yticklabels([f"ID:{int(x)}" for x in bottom_candidates_f1['class_id']])
        ax.grid(True, alpha=0.3, axis='x')
    
    # Zero performers
    ax = axes[1, 1]
    zero_performers = df[df['f1_score'] == 0]
    if len(zero_performers) > 0:
        # 显示这些类别的support数量
        bars = ax.barh(range(len(zero_performers)), zero_performers['support'], color='darkred', alpha=0.7)
        ax.set_title(f'Classes with F1=0 ({len(zero_performers)} classes)', fontweight='bold')
        ax.set_xlabel('Support (Sample Count)')
        ax.set_xscale('log')
        ax.set_yticks(range(min(15, len(zero_performers))))
        ax.set_yticklabels([f"ID:{int(x)}" for x in zero_performers['class_id'].head(15)])
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No classes with F1=0\n✅ Excellent!', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Classes with F1=0', fontweight='bold')
    
    plt.tight_layout()
    
    performers_path = output_dir / "top_bottom_performers_detailed.png"
    plt.savefig(performers_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 性能排名分析已保存: {performers_path}")

def create_performance_ranking_analysis(df, output_dir):
    """创建详细的性能排名分析图表"""
    
    logger.info("  📊 生成性能排名分析图...")
    
    # 创建一个大的图形展示不同角度的性能排名
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Detailed Performance Ranking Analysis', fontsize=18, fontweight='bold')
    
    # 1. Top performers by different metrics (并排对比)
    ax = axes[0, 0]
    
    # 选择前10个F1最高的类别
    top_f1 = df.nlargest(10, 'f1_score')
    
    x_pos = np.arange(len(top_f1))
    width = 0.25
    
    # 绘制多个指标的并排柱状图
    bars1 = ax.bar(x_pos - width, top_f1['f1_score'], width, label='F1 Score', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x_pos, top_f1['dice_coefficient'], width, label='Dice Coefficient', color='lightgreen', alpha=0.8)  
    bars3 = ax.bar(x_pos + width, top_f1['balanced_accuracy'], width, label='Balanced Accuracy', color='salmon', alpha=0.8)
    
    ax.set_title('Top 10 Classes: Multiple Metrics Comparison', fontweight='bold', fontsize=14)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"ID:{int(x)}" for x in top_f1['class_id']], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # 在柱子顶部添加具体数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # 只显示非零值
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 性能层级分布
    ax = axes[0, 1]
    
    # 定义性能层级
    excellent = len(df[df['f1_score'] >= 0.8])
    good = len(df[(df['f1_score'] >= 0.6) & (df['f1_score'] < 0.8)])
    fair = len(df[(df['f1_score'] >= 0.3) & (df['f1_score'] < 0.6)])  
    poor = len(df[(df['f1_score'] >= 0.05) & (df['f1_score'] < 0.3)])
    zero = len(df[df['f1_score'] < 0.05])
    
    labels = ['Excellent\n(F1≥0.8)', 'Good\n(0.6≤F1<0.8)', 'Fair\n(0.3≤F1<0.6)', 'Poor\n(0.05≤F1<0.3)', 'Failed\n(F1<0.05)']
    sizes = [excellent, good, fair, poor, zero]
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    explode = (0.05, 0, 0, 0, 0.1)  # 突出显示excellent和failed
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     explode=explode, startangle=90, textprops={'fontsize': 10})
    ax.set_title('Performance Distribution by F1 Score Tiers', fontweight='bold', fontsize=14)
    
    # 3. Support vs Performance关系
    ax = axes[0, 2]
    
    # 使用对数尺度显示support，因为可能范围很大
    scatter = ax.scatter(df['support'], df['f1_score'], 
                        s=60, alpha=0.6, c=df['dice_coefficient'], cmap='viridis')
    ax.set_title('Sample Support vs F1 Performance', fontweight='bold', fontsize=14)
    ax.set_xlabel('Support (Sample Count)')
    ax.set_ylabel('F1 Score')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Dice Coefficient')
    
    # 添加趋势线
    if len(df[df['f1_score'] > 0]) > 1:
        valid_data = df[df['f1_score'] > 0]
        log_support = np.log10(valid_data['support'])
        try:
            z = np.polyfit(log_support, valid_data['f1_score'], 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(valid_data['support'].min()), 
                                 np.log10(valid_data['support'].max()), 100)
            ax.plot(x_trend, p(np.log10(x_trend)), "r--", alpha=0.7, linewidth=2, label='Trend')
            ax.legend()
        except:
            pass  # 如果拟合失败就跳过
    
    # 4. 类别密度：各指标的分布直方图
    ax = axes[1, 0]
    
    metrics_to_plot = ['f1_score', 'precision', 'recall']
    colors = ['blue', 'green', 'red']
    alpha = 0.6
    
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        valid_values = df[df[metric] > 0][metric]  # 排除零值
        if len(valid_values) > 0:
            ax.hist(valid_values, bins=20, alpha=alpha, label=metric.replace('_', ' ').title(), 
                   color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribution of Non-Zero Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_xlabel('Score Value')
    ax.set_ylabel('Number of Classes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 最差性能类别详细分析 (F1 > 0但表现差)
    ax = axes[1, 1]
    
    # 找出F1>0但表现最差的10个类别
    worst_performers = df[(df['f1_score'] > 0) & (df['f1_score'] < 0.5)].nsmallest(10, 'f1_score')
    
    if len(worst_performers) > 0:
        x_pos = np.arange(len(worst_performers))
        
        # 创建柱状图显示这些类别的各项指标
        bars1 = ax.bar(x_pos - 0.2, worst_performers['precision'], 0.2, label='Precision', color='orange', alpha=0.8)
        bars2 = ax.bar(x_pos, worst_performers['recall'], 0.2, label='Recall', color='red', alpha=0.8)
        bars3 = ax.bar(x_pos + 0.2, worst_performers['f1_score'], 0.2, label='F1 Score', color='darkred', alpha=0.8)
        
        ax.set_title('Worst Performers (F1>0): Detailed Breakdown', fontweight='bold', fontsize=14)
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"ID:{int(x)}" for x in worst_performers['class_id']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No poor performers found!\n✅ All classes with F1>0\nperform reasonably well', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax.set_title('Worst Performers Analysis', fontweight='bold', fontsize=14)
    
    # 6. 成功预测的类别：Specificity vs Sensitivity
    ax = axes[1, 2]
    
    successful_classes = df[df['f1_score'] > 0.1]  # 只考虑F1>0.1的类别
    
    if len(successful_classes) > 0:
        scatter = ax.scatter(successful_classes['recall'], successful_classes['specificity'],
                           s=successful_classes['support']/10, alpha=0.6,
                           c=successful_classes['f1_score'], cmap='RdYlGn')
        ax.set_title('Sensitivity vs Specificity\n(Successful Classes Only)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Sensitivity (Recall)')
        ax.set_ylabel('Specificity')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 添加理想区域标示
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='High Specificity')
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.3, label='High Sensitivity')
        ax.legend()
        
        plt.colorbar(scatter, ax=ax, label='F1 Score')
    else:
        ax.text(0.5, 0.5, 'No classes with F1>0.1\nto analyze', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Sensitivity vs Specificity', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    ranking_path = output_dir / "detailed_performance_ranking_analysis.png"
    plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 详细性能排名分析已保存: {ranking_path}")

def create_correlation_analysis(df, output_dir):
    """创建指标间相关性分析图表"""
    
    logger.info("  📊 生成指标相关性分析图...")
    
    # 创建一个专门的相关性分析图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Performance Metrics Correlation Analysis', fontsize=18, fontweight='bold')
    
    # 选择用于相关性分析的核心指标
    correlation_metrics = ['f1_score', 'dice_coefficient', 'precision', 'recall', 
                          'specificity', 'iou_jaccard', 'balanced_accuracy']
    
    # 创建相关性数据（只包含非零类别以获得有意义的相关性）
    df_nonzero = df[df['f1_score'] > 0]
    
    if len(df_nonzero) < 3:
        # 如果非零类别太少，使用全部数据
        df_corr = df[correlation_metrics]
    else:
        df_corr = df_nonzero[correlation_metrics]
    
    # 1. 相关性热力图
    ax = axes[0, 0]
    
    correlation_matrix = df_corr.corr()
    
    # 创建热力图
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # 添加数值标注
    for i in range(len(correlation_metrics)):
        for j in range(len(correlation_metrics)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black',
                          fontweight='bold')
    
    ax.set_title('Metrics Correlation Heatmap', fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(correlation_metrics)))
    ax.set_yticks(range(len(correlation_metrics)))
    ax.set_xticklabels([m.replace('_', '\n') for m in correlation_metrics], rotation=45, ha='right')
    ax.set_yticklabels([m.replace('_', ' ').title() for m in correlation_metrics])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # 2. F1 vs Dice 散点图 (最重要的关系)
    ax = axes[0, 1]
    
    if len(df_nonzero) > 0:
        scatter = ax.scatter(df_nonzero['f1_score'], df_nonzero['dice_coefficient'], 
                           s=np.sqrt(df_nonzero['support'])*3, alpha=0.6, 
                           c=df_nonzero['balanced_accuracy'], cmap='viridis')
        
        # 添加完美相关线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
        
        # 计算和显示相关系数
        corr_coef = df_nonzero['f1_score'].corr(df_nonzero['dice_coefficient'])
        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, label='Balanced Accuracy')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient non-zero\ndata for correlation', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_title('F1 Score vs Dice Coefficient\n(size ∝ √support)', fontweight='bold', fontsize=14)
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Dice Coefficient')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 3. Precision vs Recall关系的深入分析
    ax = axes[1, 0]
    
    if len(df_nonzero) > 0:
        # 根据F1分数分组
        high_f1 = df_nonzero[df_nonzero['f1_score'] >= 0.7]
        mid_f1 = df_nonzero[(df_nonzero['f1_score'] >= 0.3) & (df_nonzero['f1_score'] < 0.7)]
        low_f1 = df_nonzero[df_nonzero['f1_score'] < 0.3]
        
        # 不同F1水平的类别用不同颜色
        if len(high_f1) > 0:
            ax.scatter(high_f1['recall'], high_f1['precision'], 
                      s=60, alpha=0.8, color='green', label=f'High F1 (≥0.7): {len(high_f1)} classes')
        if len(mid_f1) > 0:
            ax.scatter(mid_f1['recall'], mid_f1['precision'], 
                      s=60, alpha=0.8, color='orange', label=f'Mid F1 (0.3-0.7): {len(mid_f1)} classes')
        if len(low_f1) > 0:
            ax.scatter(low_f1['recall'], low_f1['precision'], 
                      s=60, alpha=0.8, color='red', label=f'Low F1 (<0.3): {len(low_f1)} classes')
        
        # 添加F1等值线
        recall_range = np.linspace(0.01, 1, 100)
        for f1_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            precision_curve = f1_level * recall_range / (2 * recall_range - f1_level)
            # 只显示有效范围
            valid_idx = (precision_curve > 0) & (precision_curve <= 1)
            if np.any(valid_idx):
                ax.plot(recall_range[valid_idx], precision_curve[valid_idx], 
                       '--', alpha=0.4, color='gray', linewidth=1)
                # 在曲线上标注F1值
                if np.any(valid_idx):
                    mid_idx = len(recall_range[valid_idx]) // 2
                    ax.text(recall_range[valid_idx][mid_idx], precision_curve[valid_idx][mid_idx], 
                           f'F1={f1_level}', fontsize=8, alpha=0.7, 
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
        
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data available\nfor analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_title('Precision-Recall Space with F1 Isolines', fontweight='bold', fontsize=14)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. 支持样本数对性能的影响分析
    ax = axes[1, 1]
    
    if len(df_nonzero) > 0:
        # 按支持样本数分组
        df_sorted = df_nonzero.sort_values('support')
        
        # 创建支持度区间
        n_classes = len(df_sorted)
        group_size = max(1, n_classes // 5)  # 分成5组
        
        support_groups = []
        f1_means = []
        f1_stds = []
        group_labels = []
        
        for i in range(0, n_classes, group_size):
            group = df_sorted.iloc[i:i+group_size]
            support_groups.append(group['support'].mean())
            f1_means.append(group['f1_score'].mean())
            f1_stds.append(group['f1_score'].std())
            
            support_range = f"{int(group['support'].min())}-{int(group['support'].max())}"
            group_labels.append(f'{support_range}\n({len(group)} classes)')
        
        # 绘制柱状图
        x_pos = np.arange(len(f1_means))
        bars = ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, 
                     alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)
        
        ax.set_title('F1 Performance by Support Groups', fontweight='bold', fontsize=14)
        ax.set_xlabel('Support Range (Sample Count)')
        ax.set_ylabel('Mean F1 Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上标注均值
        for i, (bar, mean_val) in enumerate(zip(bars, f1_means)):
            if mean_val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor support analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('F1 Performance by Support Groups', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    correlation_path = output_dir / "detailed_correlation_analysis.png"
    plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 指标相关性分析已保存: {correlation_path}")

def save_detailed_results(metrics_dict, output_dir):
    """保存详细的分析结果"""
    
    output_dir = Path(output_dir)
    
    # 1. 保存CSV格式的详细数据
    df = pd.DataFrame(metrics_dict['per_class_metrics'])
    csv_path = output_dir / "per_class_detailed_metrics.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    logger.info(f"  📊 CSV数据已保存: {csv_path}")
    
    # 2. 保存JSON格式的完整数据
    json_path = output_dir / "per_class_complete_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"  📋 JSON数据已保存: {json_path}")
    
    # 3. 生成人类友好的总结报告
    report_path = output_dir / "per_class_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("详细的Per-Class性能分析报告\n")
        f.write("=" * 100 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体统计
        overall = metrics_dict['overall_metrics']
        f.write("📊 总体性能统计:\n")
        f.write("-" * 50 + "\n")
        f.write(f"总体素数量: {overall['total_samples']:,}\n")
        f.write(f"标签中的类别数: {overall['total_classes_in_labels']}\n")
        f.write(f"预测中的类别数: {overall['total_classes_predicted']}\n")
        f.write(f"分析的总类别数: {overall['total_classes_analyzed']}\n\n")
        
        f.write(f"总体准确率: {overall['overall_accuracy']:.6f}\n")
        f.write(f"平衡准确率: {overall['balanced_accuracy']:.6f}\n")
        f.write(f"宏平均F1: {overall['macro_f1']:.6f}\n")
        f.write(f"微平均F1: {overall['micro_f1']:.6f}\n")
        f.write(f"加权F1: {overall['weighted_f1']:.6f}\n")
        f.write(f"宏平均精确率: {overall['macro_precision']:.6f}\n")
        f.write(f"宏平均召回率: {overall['macro_recall']:.6f}\n\n")
        
        # 性能分级统计
        per_class_data = metrics_dict['per_class_metrics']
        
        excellent = sum(1 for x in per_class_data if x['f1_score'] >= 0.8)
        good = sum(1 for x in per_class_data if 0.6 <= x['f1_score'] < 0.8)
        fair = sum(1 for x in per_class_data if 0.3 <= x['f1_score'] < 0.6)
        poor = sum(1 for x in per_class_data if 0.05 <= x['f1_score'] < 0.3)
        zero = sum(1 for x in per_class_data if x['f1_score'] < 0.05)
        
        f.write("🎯 性能分级统计 (按F1分数):\n")
        f.write("-" * 50 + "\n")
        f.write(f"优秀 (F1 ≥ 0.8):    {excellent:3d} 个类别 ({excellent/len(per_class_data)*100:.1f}%)\n")
        f.write(f"良好 (0.6 ≤ F1 < 0.8): {good:3d} 个类别 ({good/len(per_class_data)*100:.1f}%)\n")
        f.write(f"一般 (0.3 ≤ F1 < 0.6): {fair:3d} 个类别 ({fair/len(per_class_data)*100:.1f}%)\n")
        f.write(f"较差 (0.05≤ F1 < 0.3): {poor:3d} 个类别 ({poor/len(per_class_data)*100:.1f}%)\n")
        f.write(f"失败 (F1 < 0.05):    {zero:3d} 个类别 ({zero/len(per_class_data)*100:.1f}%)\n\n")
        
        # Top performers
        sorted_by_f1 = sorted(per_class_data, key=lambda x: x['f1_score'], reverse=True)
        
        f.write("🏆 Top 10 表现最佳的类别:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'ClassID':<8} {'F1':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        for i, class_data in enumerate(sorted_by_f1[:10]):
            f.write(f"{i+1:<6} {class_data['class_id']:<8} {class_data['f1_score']:<8.4f} "
                   f"{class_data['dice_coefficient']:<8.4f} {class_data['precision']:<10.4f} "
                   f"{class_data['recall']:<8.4f} {class_data['support']:<10,}\n")
        
        # Bottom performers (non-zero)
        non_zero_classes = [x for x in sorted_by_f1 if x['f1_score'] > 0]
        f.write(f"\n⚠️  最需改进的类别 (F1>0, 显示最差的10个):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'ClassID':<8} {'F1':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        for i, class_data in enumerate(non_zero_classes[-10:]):
            rank = len(non_zero_classes) - len(non_zero_classes[-10:]) + i + 1
            f.write(f"{rank:<6} {class_data['class_id']:<8} {class_data['f1_score']:<8.4f} "
                   f"{class_data['dice_coefficient']:<8.4f} {class_data['precision']:<10.4f} "
                   f"{class_data['recall']:<8.4f} {class_data['support']:<10,}\n")
        
        # Zero performers
        zero_classes = [x for x in per_class_data if x['f1_score'] == 0]
        if zero_classes:
            f.write(f"\n❌ 无法预测的类别 (F1=0, {len(zero_classes)}个):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'ClassID':<8} {'Support':<10} {'Predicted':<10} {'Reason':<30}\n")
            f.write("-" * 60 + "\n")
            for class_data in zero_classes[:20]:  # 最多显示20个
                reason = ""
                if class_data['support'] == 0:
                    reason = "No ground truth samples"
                elif class_data['predicted_positive'] == 0:
                    reason = "Never predicted"
                else:
                    reason = "All predictions wrong"
                    
                f.write(f"{class_data['class_id']:<8} {class_data['support']:<10,} "
                       f"{class_data['predicted_positive']:<10,} {reason:<30}\n")
            
            if len(zero_classes) > 20:
                f.write(f"... 还有 {len(zero_classes)-20} 个类别未显示\n")
    
    logger.info(f"  📋 总结报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Per-Class性能分析工具')
    
    parser.add_argument('-s', '--softmax', required=True,
                       help='Softmax预测文件路径 (.nii.gz)')
    parser.add_argument('-l', '--labels', required=True,
                       help='标签文件路径 (.nii.gz)')
    parser.add_argument('-i', '--info', 
                       help='Info文件路径 (.json)')
    parser.add_argument('-o', '--output', default='per_class_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    logger.info(f"🚀 开始Per-Class性能分析...")
    logger.info(f"  Softmax文件: {args.softmax}")
    logger.info(f"  标签文件: {args.labels}")
    logger.info(f"  输出目录: {output_dir}")
    
    # 加载数据
    data = load_softmax_and_labels(args.softmax, args.labels, args.info)
    
    # 计算per-class指标
    metrics = calculate_per_class_metrics_detailed(
        data['labels_flat'], 
        data['predictions_flat'],
        volume_info=data['info']
    )
    
    # 生成可视化
    create_comprehensive_visualizations(metrics, output_dir)
    
    # 保存详细结果
    save_detailed_results(metrics, output_dir)
    
    # 打印摘要
    logger.info("🎉 Per-Class分析完成!")
    logger.info(f"📁 结果已保存到: {output_dir}")
    
    overall = metrics['overall_metrics']
    logger.info(f"📊 快速摘要:")
    logger.info(f"  总类别数: {overall['total_classes_analyzed']}")
    logger.info(f"  宏平均F1: {overall['macro_f1']:.4f}")
    logger.info(f"  加权F1: {overall['weighted_f1']:.4f}")
    logger.info(f"  平衡准确率: {overall['balanced_accuracy']:.4f}")

if __name__ == "__main__":
    main()