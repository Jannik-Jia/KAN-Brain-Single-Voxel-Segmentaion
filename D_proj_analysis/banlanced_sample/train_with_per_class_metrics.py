#!/usr/bin/env python3
"""
扩展的训练脚本 - 包含详细的per-class性能指标计算
添加了训练完成后针对测试集每个label的详细分析
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 导入原始训练代码的功能
sys.path.append(str(Path(__file__).parent))
from train_with_3d_prediction_save import (
    RegModel, STANDARD_LABELS, create_label_mapping,
    load_and_process_subject_with_mask, 
    predictions_to_3d_volume,
    train_epoch, evaluate
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_per_class_metrics(y_true, y_pred, class_names=None, output_dir=None):
    """
    计算每个类别的详细性能指标
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like  
        预测标签
    class_names : list, optional
        类别名称列表
    output_dir : Path, optional
        输出目录，保存详细报告
    
    Returns:
    --------
    dict : 包含所有指标的字典
    """
    
    logger.info("🧮 计算每个类别的详细性能指标...")
    
    # 获取所有类别
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]
    
    logger.info(f"  分析 {n_classes} 个类别")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    # 初始化结果字典
    metrics = {
        'class_names': class_names,
        'unique_classes': unique_classes.tolist(),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {},
        'summary_metrics': {}
    }
    
    # 计算每个类别的指标
    per_class_data = []
    
    for i, class_id in enumerate(unique_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class_{class_id}"
        
        # 创建二分类mask
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        # 计算基本指标
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        support = np.sum(y_true_binary)  # 真实样本数量
        predicted_count = np.sum(y_pred_binary)  # 预测样本数量
        
        # 计算各种指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # IoU (Jaccard Index) / Dice coefficient
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Balanced accuracy for this class
        balanced_acc = (recall + specificity) / 2
        
        # 准确率 (对于多分类中的单个类别)
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        class_metrics = {
            'class_id': int(class_id),
            'class_name': class_name,
            'support': int(support),
            'predicted_count': int(predicted_count),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'iou': float(iou),
            'dice_coefficient': float(dice),
            'balanced_accuracy': float(balanced_acc),
            'accuracy': float(accuracy)
        }
        
        metrics['per_class_metrics'][class_id] = class_metrics
        per_class_data.append(class_metrics)
    
    # 计算总体指标
    overall_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    overall_f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    overall_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    overall_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    overall_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    overall_balanced_acc = balanced_accuracy_score(y_true, y_pred)
    overall_accuracy = np.mean(y_true == y_pred)
    
    metrics['summary_metrics'] = {
        'macro_f1': float(overall_f1_macro),
        'micro_f1': float(overall_f1_micro),
        'weighted_f1': float(overall_f1_weighted),
        'macro_precision': float(overall_precision_macro),
        'macro_recall': float(overall_recall_macro),
        'balanced_accuracy': float(overall_balanced_acc),
        'overall_accuracy': float(overall_accuracy),
        'total_samples': int(len(y_true))
    }
    
    # 创建DataFrame用于可视化和保存
    df_metrics = pd.DataFrame(per_class_data)
    
    # 如果提供了输出目录，保存详细结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV报告
        csv_path = output_dir / "per_class_metrics.csv"
        df_metrics.to_csv(csv_path, index=False)
        logger.info(f"  📊 详细指标已保存: {csv_path}")
        
        # 保存JSON报告  
        json_path = output_dir / "per_class_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  📋 JSON报告已保存: {json_path}")
        
        # 生成可视化图表
        create_per_class_visualizations(df_metrics, metrics, output_dir)
        
        # 生成详细的文本报告
        create_detailed_report(metrics, output_dir)
    
    return metrics, df_metrics

def create_per_class_visualizations(df_metrics, metrics, output_dir):
    """创建per-class性能可视化图表"""
    
    logger.info("📊 生成per-class性能可视化...")
    
    # 设置matplotlib参数
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    
    # 1. F1 Score柱状图
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # F1 Score
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df_metrics)), df_metrics['f1_score'], 
                    color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('F1 Score per Class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class Index')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 0.01:  # 只显示非零值
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Precision vs Recall散点图
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df_metrics['recall'], df_metrics['precision'], 
                         s=df_metrics['support']/100, alpha=0.6, 
                         c=df_metrics['f1_score'], cmap='viridis')
    ax2.set_title('Precision vs Recall (size=support, color=F1)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1 Score')
    
    # Support分布
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(df_metrics)), df_metrics['support'], 
                    color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax3.set_title('Support (Sample Count) per Class', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Class Index')
    ax3.set_ylabel('Sample Count')
    ax3.set_yscale('log')  # 使用对数刻度，因为类别不平衡
    ax3.grid(True, alpha=0.3)
    
    # IoU (Dice) 分布
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(df_metrics)), df_metrics['dice_coefficient'], 
                    color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax4.set_title('Dice Coefficient per Class', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Class Index')
    ax4.set_ylabel('Dice Coefficient')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    viz_path = output_dir / "per_class_performance.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 性能可视化已保存: {viz_path}")
    
    # 2. 混淆矩阵热力图
    if len(metrics['unique_classes']) <= 20:  # 只对类别数<=20的绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = np.array(metrics['confusion_matrix'])
        
        # 归一化混淆矩阵（按行归一化，显示召回率）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=metrics['unique_classes'][:20],
                   yticklabels=metrics['unique_classes'][:20])
        plt.title('Normalized Confusion Matrix (Recall)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  📊 混淆矩阵已保存: {cm_path}")
    
    # 3. Top/Bottom performers
    plt.figure(figsize=(16, 8))
    
    # 按F1排序
    df_sorted = df_metrics.sort_values('f1_score', ascending=False)
    
    # Top 10 performers
    plt.subplot(1, 2, 1)
    top_10 = df_sorted.head(10)
    bars = plt.bar(range(len(top_10)), top_10['f1_score'], 
                   color='green', alpha=0.7)
    plt.title('Top 10 Classes by F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Class Rank')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(top_10)), [f"ID:{int(x)}" for x in top_10['class_id']], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Bottom 10 performers (non-zero)
    plt.subplot(1, 2, 2)
    bottom_candidates = df_sorted[df_sorted['f1_score'] > 0].tail(10)
    if len(bottom_candidates) > 0:
        bars = plt.bar(range(len(bottom_candidates)), bottom_candidates['f1_score'], 
                       color='red', alpha=0.7)
        plt.title('Bottom 10 Classes by F1 Score (Non-zero)', fontsize=14, fontweight='bold')
        plt.xlabel('Class Rank')
        plt.ylabel('F1 Score')
        plt.xticks(range(len(bottom_candidates)), 
                   [f"ID:{int(x)}" for x in bottom_candidates['class_id']], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    performers_path = output_dir / "top_bottom_performers.png"
    plt.savefig(performers_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 性能排名已保存: {performers_path}")

def create_detailed_report(metrics, output_dir):
    """生成详细的文本报告"""
    
    report_path = output_dir / "detailed_performance_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("详细的Per-Class性能分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体指标
        f.write("📊 总体性能指标:\n")
        f.write("-" * 40 + "\n")
        summary = metrics['summary_metrics']
        f.write(f"总样本数: {summary['total_samples']:,}\n")
        f.write(f"总体准确率: {summary['overall_accuracy']:.4f}\n")
        f.write(f"平衡准确率: {summary['balanced_accuracy']:.4f}\n")
        f.write(f"宏平均F1: {summary['macro_f1']:.4f}\n")
        f.write(f"微平均F1: {summary['micro_f1']:.4f}\n")
        f.write(f"加权F1: {summary['weighted_f1']:.4f}\n")
        f.write(f"宏平均精确率: {summary['macro_precision']:.4f}\n")
        f.write(f"宏平均召回率: {summary['macro_recall']:.4f}\n\n")
        
        # 每个类别的详细指标
        f.write("📋 每个类别的详细指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<8} {'Support':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Dice':<10} {'IoU':<10}\n")
        f.write("-" * 80 + "\n")
        
        for class_id, class_metrics in metrics['per_class_metrics'].items():
            f.write(f"{class_id:<8} {class_metrics['support']:<10} "
                   f"{class_metrics['precision']:<10.4f} {class_metrics['recall']:<10.4f} "
                   f"{class_metrics['f1_score']:<10.4f} {class_metrics['dice_coefficient']:<10.4f} "
                   f"{class_metrics['iou']:<10.4f}\n")
        
        # 性能分析总结
        f.write("\n" + "=" * 80 + "\n")
        f.write("🎯 性能分析总结:\n")
        f.write("=" * 80 + "\n")
        
        # 统计各种性能水平的类别数量
        per_class_data = list(metrics['per_class_metrics'].values())
        excellent = sum(1 for x in per_class_data if x['f1_score'] >= 0.8)
        good = sum(1 for x in per_class_data if 0.6 <= x['f1_score'] < 0.8)
        fair = sum(1 for x in per_class_data if 0.3 <= x['f1_score'] < 0.6)
        poor = sum(1 for x in per_class_data if 0 < x['f1_score'] < 0.3)
        zero = sum(1 for x in per_class_data if x['f1_score'] == 0)
        
        f.write(f"性能优秀 (F1≥0.8): {excellent} 个类别\n")
        f.write(f"性能良好 (0.6≤F1<0.8): {good} 个类别\n")
        f.write(f"性能一般 (0.3≤F1<0.6): {fair} 个类别\n")
        f.write(f"性能较差 (0<F1<0.3): {poor} 个类别\n")
        f.write(f"无法预测 (F1=0): {zero} 个类别\n\n")
        
        # Top和Bottom performers
        sorted_by_f1 = sorted(per_class_data, key=lambda x: x['f1_score'], reverse=True)
        
        f.write("🏆 Top 10 表现最佳的类别:\n")
        f.write("-" * 50 + "\n")
        for i, class_data in enumerate(sorted_by_f1[:10]):
            f.write(f"{i+1:2d}. Class {class_data['class_id']:3d}: F1={class_data['f1_score']:.4f} "
                   f"(Support: {class_data['support']:,})\n")
        
        f.write("\n⚠️ 最需要改进的类别 (F1>0):\n")
        f.write("-" * 50 + "\n")
        non_zero_classes = [x for x in sorted_by_f1 if x['f1_score'] > 0]
        for i, class_data in enumerate(non_zero_classes[-10:]):
            f.write(f"{len(non_zero_classes)-i:2d}. Class {class_data['class_id']:3d}: F1={class_data['f1_score']:.4f} "
                   f"(Support: {class_data['support']:,})\n")
    
    logger.info(f"  📋 详细报告已保存: {report_path}")

def enhanced_train_model(config):
    """
    增强版训练函数，训练完成后计算详细的per-class指标
    """
    
    logger.info(f"🚀 开始增强版训练 (包含per-class指标分析)...")
    
    # 基本设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    include_background = config.get('include_background', True)
    output_dir = Path(config.get('output_dir', 'results'))
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据 - 使用原始的加载函数
    test_data_path = config['test_data_path']
    test_data_info = load_and_process_subject_with_mask(
        subject_dir=test_data_path,
        include_background=include_background
    )
    
    # ... 这里可以添加原始的训练循环代码，或者加载已训练的模型 ...
    # 为了示例，假设我们有训练好的模型
    
    logger.info(f"📊 开始计算详细的per-class性能指标...")
    
    # 加载测试数据
    X_test = test_data_info['features']
    y_test = test_data_info['labels']
    
    logger.info(f"测试集大小: {X_test.shape[0]:,} 样本, {X_test.shape[1]} 特征")
    
    # 如果有已训练的模型，进行预测
    if 'model_path' in config and Path(config['model_path']).exists():
        logger.info(f"加载预训练模型: {config['model_path']}")
        
        # 加载模型 (这里需要根据实际情况调整)
        checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
        
        # 重建模型
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        # 从模型权重推断维度
        first_layer_weight = model_state['fc1.weight']
        n_features = first_layer_weight.shape[1]
        n_classes = model_state['fc5.weight'].shape[0]
        
        logger.info(f"模型架构: {n_features} -> {n_classes}")
        
        # 调整特征维度
        if X_test.shape[1] != n_features:
            logger.info(f"调整特征维度: {X_test.shape[1]} -> {n_features}")
            if X_test.shape[1] > n_features:
                X_test = X_test[:, :n_features]
            else:
                padding = np.zeros((X_test.shape[0], n_features - X_test.shape[1]))
                X_test = np.concatenate([X_test, padding], axis=1)
        
        # 重建并加载模型
        model = RegModel(input_dim=n_features, num_classes=n_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        # 进行预测
        logger.info("🔮 进行测试集预测...")
        predictions = []
        batch_size = 8192
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), desc="预测中"):
                batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                batch_logits = model(batch_X)
                batch_pred = torch.argmax(batch_logits, dim=1)
                predictions.extend(batch_pred.cpu().numpy())
        
        y_pred = np.array(predictions)
        
        logger.info(f"预测完成: {len(y_pred):,} 个预测")
        
        # 创建输出目录
        metrics_output_dir = output_dir / f"per_class_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 计算详细的per-class指标
        metrics, df_metrics = calculate_per_class_metrics(
            y_true=y_test,
            y_pred=y_pred,
            class_names=[f"Label_{i}" for i in range(n_classes)],
            output_dir=metrics_output_dir
        )
        
        # 打印摘要结果
        logger.info("🎯 Per-Class分析完成!")
        logger.info(f"📁 详细结果保存在: {metrics_output_dir}")
        
        summary = metrics['summary_metrics']
        logger.info(f"📊 总体性能摘要:")
        logger.info(f"  宏平均F1: {summary['macro_f1']:.4f}")
        logger.info(f"  微平均F1: {summary['micro_f1']:.4f}")
        logger.info(f"  加权F1: {summary['weighted_f1']:.4f}")
        logger.info(f"  平衡准确率: {summary['balanced_accuracy']:.4f}")
        
        # 显示top表现的类别
        per_class_data = list(metrics['per_class_metrics'].values())
        top_performers = sorted(per_class_data, key=lambda x: x['f1_score'], reverse=True)[:5]
        
        logger.info(f"🏆 Top 5 表现最佳的类别:")
        for i, class_data in enumerate(top_performers):
            logger.info(f"  {i+1}. Class {class_data['class_id']}: F1={class_data['f1_score']:.4f}")
        
        return model, metrics, df_metrics
    
    else:
        logger.error("❌ 未提供有效的模型路径")
        return None, None, None

def main():
    """主函数 - 示例用法"""
    
    # 配置参数
    config = {
        'include_background': True,
        'test_data_path': '/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS/FOR_016_20250204_reproducibility',
        'model_path': '/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/balanced_3d_bg_incl_20250829_151356.pth',
        'output_dir': 'enhanced_training_results'
    }
    
    # 运行增强版训练
    model, metrics, df_metrics = enhanced_train_model(config)
    
    if metrics is not None:
        print("✅ 增强版训练和分析完成!")
        print(f"📊 分析了 {len(metrics['per_class_metrics'])} 个类别")
        print(f"📁 详细结果请查看输出目录")

if __name__ == "__main__":
    main()