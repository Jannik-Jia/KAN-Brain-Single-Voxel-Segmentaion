#!/usr/bin/env python3
"""
Independent Per-Class Performance Analysis Tool
Can directly perform detailed per-class analysis on existing softmax files
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_softmax_and_labels(softmax_path, labels_path, info_path=None):
    """
    Load softmax prediction file and label file
    """
    logger.info(f"📂 Loading data files...")

    # Load softmax predictions
    softmax_nii = nib.load(softmax_path)
    softmax = softmax_nii.get_fdata()
    logger.info(f"  Softmax shape: {softmax.shape}")

    # Load labels
    labels_nii = nib.load(labels_path)
    labels = labels_nii.get_fdata()
    logger.info(f"  Labels shape: {labels.shape}")

    # Calculate argmax predictions
    predictions = np.argmax(softmax, axis=-1)
    logger.info(f"  Predictions shape: {predictions.shape}")

    # Flatten to 1D arrays
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()

    # Load info file (if provided)
    info = None
    if info_path and Path(info_path).exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        logger.info(f"  Info file loaded")

    logger.info(f"✅ Data loading complete: {len(labels_flat):,} voxels")
    
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
    Calculate detailed per-class metrics, especially suitable for medical image segmentation
    """
    logger.info("🧮 Calculating per-class metrics...")

    # Get all classes that appear
    unique_labels = np.unique(y_true)
    unique_preds = np.unique(y_pred)
    all_classes = np.unique(np.concatenate([unique_labels, unique_preds]))

    logger.info(f"  Classes in labels: {len(unique_labels)}")
    logger.info(f"  Classes in predictions: {len(unique_preds)}")
    logger.info(f"  Total classes to analyze: {len(all_classes)}")
    
    if class_names is None:
        class_names = [f"Class_{int(i)}" for i in all_classes]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    
    # Calculate metrics for each class
    per_class_results = []
    
    for i, class_id in enumerate(all_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{int(class_id)}"
        
        # Binary classification metrics calculation
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        # Basic counts
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        # Basic metrics
        support = np.sum(y_true_binary)  # Actual sample count
        predicted_positive = np.sum(y_pred_binary)  # Predicted positive count
        
        # Calculate performance metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Common metrics for medical image segmentation
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0  # Dice coefficient
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # IoU/Jaccard
        
        # Accuracy and balanced accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2
        
        # Additional diagnostic information
        prevalence = support / len(y_true)  # Prevalence of this class in data
        predicted_prevalence = predicted_positive / len(y_pred)  # Predicted prevalence
        
        # Detection capability metrics
        positive_likelihood_ratio = recall / (1 - specificity) if specificity < 1 else float('inf')
        negative_likelihood_ratio = (1 - recall) / specificity if specificity > 0 else float('inf')
        
        # Determine displayed class ID (show original FreeSurfer label if reverse mapping available)
        display_class_id = int(class_id)
        if reverse_mapping is not None and class_id in reverse_mapping:
            display_class_id = reverse_mapping[class_id]
        
        # Summary results
        class_result = {
            'class_id': display_class_id,  # Display original FreeSurfer label ID
            'internal_class_id': int(class_id),  # Internal continuous index
            'class_name': class_name,
            
            # Basic counts
            'support': int(support),
            'predicted_positive': int(predicted_positive),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            
            # Core performance metrics
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            
            # Medical image segmentation metrics
            'dice_coefficient': float(dice),
            'iou_jaccard': float(iou),
            
            # Accuracy metrics
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            
            # Statistical information
            'prevalence': float(prevalence),
            'predicted_prevalence': float(predicted_prevalence),
            
            # Diagnostic capability
            'positive_lr': float(positive_likelihood_ratio) if not np.isinf(positive_likelihood_ratio) else None,
            'negative_lr': float(negative_likelihood_ratio) if not np.isinf(negative_likelihood_ratio) else None,
        }
        
        per_class_results.append(class_result)
    
    # Calculate overall metrics
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
    """Create detailed per-class visualization comparison report"""
    
    logger.info("📊 Generating detailed per-class visualizations...")
    output_dir = Path(output_dir)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metrics_dict['per_class_metrics'])
    
    # Set figure style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Create detailed comparison charts for each metric
    create_individual_metric_comparisons(df, output_dir)
    
    # 2. Create comprehensive overview charts
    create_comprehensive_overview(df, output_dir)
    
    # 3. Create performance ranking analysis
    create_performance_ranking_analysis(df, output_dir)
    
    # 4. Create correlation analysis
    create_correlation_analysis(df, output_dir)

def create_individual_metric_comparisons(df, output_dir):
    """Create detailed class-by-class comparison charts for each metric"""
    
    logger.info("  📊 Generating detailed comparison charts for each metric...")

    # Core metrics list
    core_metrics = [
        ('f1_score', 'F1 Score', 'F1 Score'),
        ('dice_coefficient', 'Dice Coefficient', 'Dice Coefficient'),
        ('precision', 'Precision', 'Precision'),
        ('recall', 'Recall (Sensitivity)', 'Recall/Sensitivity'),
        ('specificity', 'Specificity', 'Specificity'),
        ('iou_jaccard', 'IoU (Jaccard Index)', 'IoU (Intersection over Union)'),
        ('balanced_accuracy', 'Balanced Accuracy', 'Balanced Accuracy')
    ]

    n_classes = len(df)

    for metric_col, metric_title, metric_title_en in core_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f'{metric_title} - Class-by-Class Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Upper plot: Bar chart for all classes
        ax1 = axes[0]
        
        # Sort by metric value for better visualization
        df_sorted = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
        
        # Create color mapping (based on performance level)
        colors = []
        for value in df_sorted[metric_col]:
            if value >= 0.8:
                colors.append('green')      # Excellent
            elif value >= 0.6:
                colors.append('lightgreen') # Good
            elif value >= 0.3:
                colors.append('yellow')     # Fair
            elif value > 0.05:
                colors.append('orange')     # Poor
            else:
                colors.append('red')        # Failed
        
        bars = ax1.bar(range(len(df_sorted)), df_sorted[metric_col], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax1.set_title(f'{metric_title} for All Classes (Sorted by Performance)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class Rank (Best to Worst)')
        ax1.set_ylabel(metric_title)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add mean line and standard deviation range
        mean_val = df[metric_col].mean()
        std_val = df[metric_col].std()
        ax1.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        ax1.axhspan(mean_val-std_val, mean_val+std_val, alpha=0.1, color='red', 
                   label=f'±1 Std: {std_val:.3f}')
        
        # Add performance threshold lines
        if metric_col in ['f1_score', 'dice_coefficient', 'precision', 'recall']:
            ax1.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Excellent (0.8)')
            ax1.axhline(y=0.6, color='yellow', linestyle=':', alpha=0.7, label='Good (0.6)')
            ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='Fair (0.3)')
        
        ax1.legend(loc='upper right')
        
        # Add class ID labels for top 20
        for i in range(min(20, len(df_sorted))):
            bar = bars[i]
            height = bar.get_height()
            if height > 0.01:  # Only add labels for non-zero values
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'ID:{int(df_sorted.iloc[i]["class_id"])}', 
                        ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Lower plot: Bar chart by original class ID order
        ax2 = axes[1]
        
        # Sort by class_id
        df_by_id = df.sort_values('class_id').reset_index(drop=True)
        
        # Use same color encoding
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
        
        # Add same statistical lines
        ax2.axhline(y=mean_val, color='red', linestyle='--', linewidth=2)
        ax2.axhspan(mean_val-std_val, mean_val+std_val, alpha=0.1, color='red')
        
        # Set x-axis labels to class_id (show every 10th)
        step = max(1, len(df_by_id) // 20)  # Show maximum 20 labels
        tick_positions = range(0, len(df_by_id), step)
        tick_labels = [f'{int(df_by_id.iloc[i]["class_id"])}' for i in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        metric_viz_path = output_dir / f"{metric_col}_detailed_comparison.png"
        plt.savefig(metric_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    📊 {metric_title}comparison chart saved: {metric_viz_path}")
    
    # Create special visualization for support distribution (using log scale)
    create_support_distribution_viz(df, output_dir)

def create_support_distribution_viz(df, output_dir):
    """Create special visualization for sample count distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Support (Sample Count) Distribution Analysis\nSample Count Distribution Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Sort by support
    df_by_support = df.sort_values('support', ascending=False).reset_index(drop=True)
    
    # 1. Log scale bar chart
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(df_by_support)), df_by_support['support'], 
                  color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax1.set_title('Support Distribution (Log Scale)', fontweight='bold')
    ax1.set_xlabel('Class Rank (Most to Least Samples)')
    ax1.set_ylabel('Sample Count (Log Scale)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistical information
    mean_support = df['support'].mean()
    median_support = df['support'].median()
    ax1.axhline(y=mean_support, color='blue', linestyle='--', label=f'Mean: {mean_support:.0f}')
    ax1.axhline(y=median_support, color='green', linestyle='--', label=f'Median: {median_support:.0f}')
    ax1.legend()
    
    # 2. Support vs F1Performance scatter plot
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['support'], df['f1_score'], 
                         alpha=0.6, s=60, c='purple')
    ax2.set_title('Support vs F1 Performance', fontweight='bold')
    ax2.set_xlabel('Support (Sample Count, Log Scale)')
    ax2.set_ylabel('F1 Score')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    from scipy import stats
    log_support = np.log10(df['support'] + 1)  # +1 to handle zeros
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_support, df['f1_score'])
    line_x = np.logspace(0, np.log10(df['support'].max()), 100)
    line_y = slope * np.log10(line_x + 1) + intercept
    ax2.plot(line_x, line_y, 'r--', alpha=0.8, label=f'Trend (R²={r_value**2:.3f})')
    ax2.legend()
    
    # 3. Class imbalance analysis
    ax3 = axes[1, 0]
    
    # Group by support
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
    
    # Add average F1 information on bars
    for i, (bar, avg_f1) in enumerate(zip(bars, avg_f1_by_support)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'Avg F1:\n{avg_f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    
    sorted_supports = np.sort(df['support'].values)
    cumulative_pct = np.arange(1, len(sorted_supports) + 1) / len(sorted_supports) * 100
    
    ax4.plot(sorted_supports, cumulative_pct, 'b-', linewidth=2)
    ax4.set_title('Cumulative Support Distribution', fontweight='bold')
    ax4.set_xlabel('Sample Count (Log Scale)')
    ax4.set_ylabel('Cumulative Percentage of Classes')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(sorted_supports, p)
        ax4.axvline(x=value, color='red', linestyle='--', alpha=0.7)
        ax4.text(value, p, f'  {p}th: {value:.0f}', rotation=90, va='bottom')
    
    plt.tight_layout()
    
    # Save chart
    support_viz_path = output_dir / "support_distribution_analysis.png"
    plt.savefig(support_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    📊 Sample Count Distribution Analysis saved: {support_viz_path}")

def create_comprehensive_overview(df, output_dir):
    """Create comprehensive overview charts（original functionality）"""
    
    logger.info("  📊 Generating comprehensive overview chart...")
    
    # 1. Overview of main performance metrics (4x2 layout)
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle('Per-Class Performance Analysis - Comprehensive Overview', fontsize=16, fontweight='bold')
    
    # F1 ScoreDistribution
    ax = axes[0, 0]
    bars = ax.bar(range(len(df)), df['f1_score'], color='skyblue', alpha=0.7, edgecolor='navy')
    ax.set_title('F1 Score per Class', fontweight='bold')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_f1 = df['f1_score'].mean()
    ax.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.7, label=f'Mean F1: {mean_f1:.3f}')
    ax.legend()
    
    # Precision vs RecallScatter plot
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
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal line
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # Dice coefficientComparison
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
    
    # SupportDistribution (Sample Count)
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
    
    # Performance grade distribution
    ax = axes[2, 1]
    
    # Grade by F1 score
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
    
    # Predicted vs ActualDistributionComparison
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
    
    # Save main performance chart
    main_viz_path = output_dir / "comprehensive_per_class_analysis.png"
    plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 Main analysis chart saved: {main_viz_path}")
    
    # 2. Detailed Top/Bottom performers analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Top and Bottom Performers Analysis', fontsize=16, fontweight='bold')
    
    # Sort by different metrics
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
        # Show support count for these classes
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
    
    logger.info(f"  📊 Performance ranking analysis saved: {performers_path}")

def create_performance_ranking_analysis(df, output_dir):
    """Create detailed performance ranking analysis charts"""
    
    logger.info("  📊 Generating performance ranking analysis...")
    
    # Create a large figure showing performance ranking from different angles
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Detailed Performance Ranking Analysis', fontsize=18, fontweight='bold')
    
    # 1. Top performers by different metrics (side-by-side comparison)
    ax = axes[0, 0]
    
    # Select top 10 classes with highest F1
    top_f1 = df.nlargest(10, 'f1_score')
    
    x_pos = np.arange(len(top_f1))
    width = 0.25
    
    # Plot side-by-side bar charts for multiple metrics
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
    
    # Add specific values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only show non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Performance levelDistribution
    ax = axes[0, 1]
    
    # Define performance levels
    excellent = len(df[df['f1_score'] >= 0.8])
    good = len(df[(df['f1_score'] >= 0.6) & (df['f1_score'] < 0.8)])
    fair = len(df[(df['f1_score'] >= 0.3) & (df['f1_score'] < 0.6)])  
    poor = len(df[(df['f1_score'] >= 0.05) & (df['f1_score'] < 0.3)])
    zero = len(df[df['f1_score'] < 0.05])
    
    labels = ['Excellent\n(F1≥0.8)', 'Good\n(0.6≤F1<0.8)', 'Fair\n(0.3≤F1<0.6)', 'Poor\n(0.05≤F1<0.3)', 'Failed\n(F1<0.05)']
    sizes = [excellent, good, fair, poor, zero]
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    explode = (0.05, 0, 0, 0, 0.1)  # Highlight excellent and failed
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     explode=explode, startangle=90, textprops={'fontsize': 10})
    ax.set_title('Performance Distribution by F1 Score Tiers', fontweight='bold', fontsize=14)
    
    # 3. Support vs Performancerelationship
    ax = axes[0, 2]
    
    # Use log scale for support as range may be large
    scatter = ax.scatter(df['support'], df['f1_score'], 
                        s=60, alpha=0.6, c=df['dice_coefficient'], cmap='viridis')
    ax.set_title('Sample Support vs F1 Performance', fontweight='bold', fontsize=14)
    ax.set_xlabel('Support (Sample Count)')
    ax.set_ylabel('F1 Score')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Dice Coefficient')
    
    # Add trend line
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
            pass  # Skip if fitting fails
    
    # 4. Class density: distribution histogram for each metric
    ax = axes[1, 0]
    
    metrics_to_plot = ['f1_score', 'precision', 'recall']
    colors = ['blue', 'green', 'red']
    alpha = 0.6
    
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        valid_values = df[df[metric] > 0][metric]  # Exclude zero values
        if len(valid_values) > 0:
            ax.hist(valid_values, bins=20, alpha=alpha, label=metric.replace('_', ' ').title(), 
                   color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribution of Non-Zero Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_xlabel('Score Value')
    ax.set_ylabel('Number of Classes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Detailed analysis of worst performing classes (F1 > 0 but poor performance)
    ax = axes[1, 1]
    
    # Find 10 worst performing classes with F1>0
    worst_performers = df[(df['f1_score'] > 0) & (df['f1_score'] < 0.5)].nsmallest(10, 'f1_score')
    
    if len(worst_performers) > 0:
        x_pos = np.arange(len(worst_performers))
        
        # Create bar chart showing metrics for these classes
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
    
    # 6. Successfully predicted classes：Specificity vs Sensitivity
    ax = axes[1, 2]
    
    successful_classes = df[df['f1_score'] > 0.1]  # Only consider classes with F1>0.1
    
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
        
        # Add ideal region markers
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
    
    logger.info(f"  📊 Detailed performance ranking analysis saved: {ranking_path}")

def create_correlation_analysis(df, output_dir):
    """Create metric correlation analysis charts"""
    
    logger.info("  📊 Generating metric correlation analysis...")
    
    # Create a dedicated correlation analysis chart
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Performance Metrics Correlation Analysis', fontsize=18, fontweight='bold')
    
    # Select core metrics for correlation analysis
    correlation_metrics = ['f1_score', 'dice_coefficient', 'precision', 'recall', 
                          'specificity', 'iou_jaccard', 'balanced_accuracy']
    
    # Create correlation data (only non-zero classes for meaningful correlation)
    df_nonzero = df[df['f1_score'] > 0]
    
    if len(df_nonzero) < 3:
        # If too few non-zero classes, use all data
        df_corr = df[correlation_metrics]
    else:
        df_corr = df_nonzero[correlation_metrics]
    
    # 1. Correlation heatmap
    ax = axes[0, 0]
    
    correlation_matrix = df_corr.corr()
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add value annotations
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
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # 2. F1 vs Dice scatter plot (most important relationship)
    ax = axes[0, 1]
    
    if len(df_nonzero) > 0:
        scatter = ax.scatter(df_nonzero['f1_score'], df_nonzero['dice_coefficient'], 
                           s=np.sqrt(df_nonzero['support'])*3, alpha=0.6, 
                           c=df_nonzero['balanced_accuracy'], cmap='viridis')
        
        # Add perfect correlation line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
        
        # Calculate and display correlation coefficient
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
    
    # 3. In-depth analysis of Precision vs Recall relationship
    ax = axes[1, 0]
    
    if len(df_nonzero) > 0:
        # Group by F1 score
        high_f1 = df_nonzero[df_nonzero['f1_score'] >= 0.7]
        mid_f1 = df_nonzero[(df_nonzero['f1_score'] >= 0.3) & (df_nonzero['f1_score'] < 0.7)]
        low_f1 = df_nonzero[df_nonzero['f1_score'] < 0.3]
        
        # Different colors for different F1 levels
        if len(high_f1) > 0:
            ax.scatter(high_f1['recall'], high_f1['precision'], 
                      s=60, alpha=0.8, color='green', label=f'High F1 (≥0.7): {len(high_f1)} classes')
        if len(mid_f1) > 0:
            ax.scatter(mid_f1['recall'], mid_f1['precision'], 
                      s=60, alpha=0.8, color='orange', label=f'Mid F1 (0.3-0.7): {len(mid_f1)} classes')
        if len(low_f1) > 0:
            ax.scatter(low_f1['recall'], low_f1['precision'], 
                      s=60, alpha=0.8, color='red', label=f'Low F1 (<0.3): {len(low_f1)} classes')
        
        # Add F1 contour lines
        recall_range = np.linspace(0.01, 1, 100)
        for f1_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            precision_curve = f1_level * recall_range / (2 * recall_range - f1_level)
            # Only show valid range
            valid_idx = (precision_curve > 0) & (precision_curve <= 1)
            if np.any(valid_idx):
                ax.plot(recall_range[valid_idx], precision_curve[valid_idx], 
                       '--', alpha=0.4, color='gray', linewidth=1)
                # Annotate F1 values on curves
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
    
    # 4. Analysis of impact of support sample count on performance
    ax = axes[1, 1]
    
    if len(df_nonzero) > 0:
        # Group by support sample count
        df_sorted = df_nonzero.sort_values('support')
        
        # Create support bins
        n_classes = len(df_sorted)
        group_size = max(1, n_classes // 5)  # Divide into 5 groups
        
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
        
        # Plot bar chart
        x_pos = np.arange(len(f1_means))
        bars = ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, 
                     alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)
        
        ax.set_title('F1 Performance by Support Groups', fontweight='bold', fontsize=14)
        ax.set_xlabel('Support Range (Sample Count)')
        ax.set_ylabel('Mean F1 Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annotate mean values on bars
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
    
    logger.info(f"  📊 Metric correlation analysis saved: {correlation_path}")

def save_detailed_results(metrics_dict, output_dir):
    """Save detailed analysis results"""
    
    output_dir = Path(output_dir)
    
    # 1. Save detailed data in CSV format
    df = pd.DataFrame(metrics_dict['per_class_metrics'])
    csv_path = output_dir / "per_class_detailed_metrics.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    logger.info(f"  📊 CSV data saved: {csv_path}")
    
    # 2. Save complete data in JSON format
    json_path = output_dir / "per_class_complete_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"  📋 JSON data saved: {json_path}")
    
    # 3. Generate human-friendly summary report
    report_path = output_dir / "per_class_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Detailed Per-Class Performance Analysis Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        overall = metrics_dict['overall_metrics']
        f.write("📊 Overall Performance Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total voxels: {overall['total_samples']:,}\n")
        f.write(f"Classes in labels: {overall['total_classes_in_labels']}\n")
        f.write(f"Classes in predictions: {overall['total_classes_predicted']}\n")
        f.write(f"Total classes analyzed: {overall['total_classes_analyzed']}\n\n")
        
        f.write(f"Overall accuracy: {overall['overall_accuracy']:.6f}\n")
        f.write(f"Balanced accuracy: {overall['balanced_accuracy']:.6f}\n")
        f.write(f"Macro-average F1: {overall['macro_f1']:.6f}\n")
        f.write(f"Micro-average F1: {overall['micro_f1']:.6f}\n")
        f.write(f"Weighted F1: {overall['weighted_f1']:.6f}\n")
        f.write(f"Macro-average precision: {overall['macro_precision']:.6f}\n")
        f.write(f"Macro-average recall: {overall['macro_recall']:.6f}\n\n")
        
        # Performance grade statistics
        per_class_data = metrics_dict['per_class_metrics']
        
        excellent = sum(1 for x in per_class_data if x['f1_score'] >= 0.8)
        good = sum(1 for x in per_class_data if 0.6 <= x['f1_score'] < 0.8)
        fair = sum(1 for x in per_class_data if 0.3 <= x['f1_score'] < 0.6)
        poor = sum(1 for x in per_class_data if 0.05 <= x['f1_score'] < 0.3)
        zero = sum(1 for x in per_class_data if x['f1_score'] < 0.05)
        
        f.write("🎯 Performance Grade Statistics (by F1 score):\n")
        f.write("-" * 50 + "\n")
        f.write(f"Excellent (F1 ≥ 0.8):    {excellent:3d} classes ({excellent/len(per_class_data)*100:.1f}%)\n")
        f.write(f"Good (0.6 ≤ F1 < 0.8): {good:3d} classes ({good/len(per_class_data)*100:.1f}%)\n")
        f.write(f"Fair (0.3 ≤ F1 < 0.6): {fair:3d} classes ({fair/len(per_class_data)*100:.1f}%)\n")
        f.write(f"Poor (0.05≤ F1 < 0.3): {poor:3d} classes ({poor/len(per_class_data)*100:.1f}%)\n")
        f.write(f"Failed (F1 < 0.05):    {zero:3d} classes ({zero/len(per_class_data)*100:.1f}%)\n\n")
        
        # Top performers
        sorted_by_f1 = sorted(per_class_data, key=lambda x: x['f1_score'], reverse=True)
        
        f.write("🏆 Top 10 Best Performing Classes:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'ClassID':<8} {'F1':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        for i, class_data in enumerate(sorted_by_f1[:10]):
            f.write(f"{i+1:<6} {class_data['class_id']:<8} {class_data['f1_score']:<8.4f} "
                   f"{class_data['dice_coefficient']:<8.4f} {class_data['precision']:<10.4f} "
                   f"{class_data['recall']:<8.4f} {class_data['support']:<10,}\n")
        
        # Bottom performers (non-zero)
        non_zero_classes = [x for x in sorted_by_f1 if x['f1_score'] > 0]
        f.write(f"\n⚠️  Classes Most in Need of Improvement (F1>0, Show 10 worst):\n")
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
            f.write(f"\n❌ Unpredictable Classes (F1=0, {len(zero_classes)}):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'ClassID':<8} {'Support':<10} {'Predicted':<10} {'Reason':<30}\n")
            f.write("-" * 60 + "\n")
            for class_data in zero_classes[:20]:  # Show maximum 20
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
                f.write(f"... ... and {len(zero_classes)-20} classes not shown\n")
    
    logger.info(f"  📋 Summary report saved: {report_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Per-ClassPerformance Analysis Tool')
    
    parser.add_argument('-s', '--softmax', required=True,
                       help='SoftmaxPrediction file path (.nii.gz)')
    parser.add_argument('-l', '--labels', required=True,
                       help='Label file path (.nii.gz)')
    parser.add_argument('-i', '--info', 
                       help='InfoFile path (.json)')
    parser.add_argument('-o', '--output', default='per_class_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    logger.info(f"🚀 Starting Per-Class performance analysis...")
    logger.info(f"  SoftmaxFile: {args.softmax}")
    logger.info(f"  Labels file: {args.labels}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Load data
    data = load_softmax_and_labels(args.softmax, args.labels, args.info)
    
    # Calculate per-class metrics
    metrics = calculate_per_class_metrics_detailed(
        data['labels_flat'], 
        data['predictions_flat'],
        volume_info=data['info']
    )
    
    # Generate visualizations
    create_comprehensive_visualizations(metrics, output_dir)
    
    # Save detailed results
    save_detailed_results(metrics, output_dir)
    
    # Print summary
    logger.info("🎉 Per-ClassAnalysis complete!")
    logger.info(f"📁 Results saved to: {output_dir}")
    
    overall = metrics['overall_metrics']
    logger.info(f"📊 Quick Summary:")
    logger.info(f"  Total classes: {overall['total_classes_analyzed']}")
    logger.info(f"  Macro-average F1: {overall['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {overall['weighted_f1']:.4f}")
    logger.info(f"  Balanced accuracy: {overall['balanced_accuracy']:.4f}")

if __name__ == "__main__":
    main()