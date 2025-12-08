#!/usr/bin/env python3
"""
扩展的对比分析脚本
支持所有新增的评估指标，提供详细的科学分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 定义所有指标的元数据
METRICS_METADATA = {
    'basic': {
        'best_test_f1': {'name': 'Best Test F1', 'description': 'Checkpoint Selection', 'better': 'higher'},
        'final_train_f1': {'name': 'Final Train F1', 'description': 'Training Performance', 'better': 'higher'},
        'final_test_f1': {'name': 'Final Test F1', 'description': 'Test Performance', 'better': 'higher'},
        'final_train_loss': {'name': 'Final Train Loss', 'description': 'Training Loss', 'better': 'lower'},
        'final_test_loss': {'name': 'Final Test Loss', 'description': 'Test Loss', 'better': 'lower'},
    },
    'accuracy': {
        'gross_accuracy': {'name': 'Gross Accuracy', 'description': 'Overall Accuracy', 'better': 'higher'},
        'top1_accuracy': {'name': 'Top-1 Accuracy', 'description': 'Top-1 Coverage', 'better': 'higher'},
        'top3_accuracy': {'name': 'Top-3 Accuracy', 'description': 'Top-3 Coverage', 'better': 'higher'},
        'top5_accuracy': {'name': 'Top-5 Accuracy', 'description': 'Top-5 Coverage', 'better': 'higher'},
    },
    'balance': {
        'balanced_accuracy': {'name': 'Balanced Accuracy', 'description': 'Class-Balanced Accuracy', 'better': 'higher'},
        'weighted_f1': {'name': 'Weighted F1', 'description': 'Sample-Weighted F1', 'better': 'higher'},
    },
    'consistency': {
        'cohen_kappa': {'name': "Cohen's Kappa", 'description': 'Agreement Metric', 'better': 'higher'},
    },
    'segmentation': {
        'macro_soft_dice': {'name': 'Macro Soft Dice', 'description': 'Soft Segmentation Quality', 'better': 'higher'},
    },
    'risk': {
        'risk_at_95_coverage': {'name': 'Risk@95% Cov', 'description': 'Error Rate at 95% Coverage', 'better': 'lower'},
    }
}


def load_experiment_results(results_dir: Path) -> List[Dict]:
    """
    加载所有实验的结果，包含完整的扩展指标

    Args:
        results_dir: 实验结果目录

    Returns:
        实验结果字典列表
    """
    results = []

    # 扫描所有实验目录
    exp_dirs = sorted(results_dir.glob('exp*'))

    if not exp_dirs:
        print(f"警告: 在 {results_dir} 中未找到任何实验目录")
        return results

    for exp_dir in exp_dirs:
        # 提取实验编号和排除的被试
        exp_name = exp_dir.name

        # 查找history文件
        history_files = list(exp_dir.glob('history_test*.json'))

        if not history_files:
            print(f"警告: {exp_dir} 中未找到history文件")
            continue

        # 读取训练历史
        try:
            with open(history_files[0], 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"错误: 无法读取 {history_files[0]}: {e}")
            continue

        # 提取基础指标
        result = {
            'experiment': exp_name,
            'best_test_f1': max(history['test_f1']) if history['test_f1'] else np.nan,
            'final_train_f1': history['train_f1'][-1] if history['train_f1'] else np.nan,
            'final_test_f1': history['test_f1'][-1] if history['test_f1'] else np.nan,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else np.nan,
            'final_test_loss': history['test_loss'][-1] if history['test_loss'] else np.nan,
            'history': history
        }

        # 提取扩展评估指标
        if 'best_test_metrics' in history and history['best_test_metrics']:
            metrics = history['best_test_metrics']
            result.update({
                'gross_accuracy': metrics.get('gross_accuracy', np.nan),
                'top1_accuracy': metrics.get('top1_accuracy', np.nan),
                'top3_accuracy': metrics.get('top3_accuracy', np.nan),
                'top5_accuracy': metrics.get('top5_accuracy', np.nan),
                'balanced_accuracy': metrics.get('balanced_accuracy', np.nan),
                'weighted_f1': metrics.get('weighted_f1', np.nan),
                'cohen_kappa': metrics.get('cohen_kappa', np.nan),
                'macro_soft_dice': metrics.get('macro_soft_dice', np.nan),
                'risk_at_95_coverage': metrics.get('risk_at_95_coverage', np.nan),
                'actual_coverage_95': metrics.get('actual_coverage_95', np.nan),
            })
        else:
            # 兼容旧版本
            result.update({
                'gross_accuracy': np.nan,
                'top1_accuracy': np.nan,
                'top3_accuracy': np.nan,
                'top5_accuracy': np.nan,
                'balanced_accuracy': np.nan,
                'weighted_f1': np.nan,
                'cohen_kappa': np.nan,
                'macro_soft_dice': np.nan,
                'risk_at_95_coverage': np.nan,
                'actual_coverage_95': np.nan,
            })

        # 解析实验类型
        if 'exclude_all' in exp_name:
            result['exclude_type'] = 'all'
            result['excluded_subject'] = 'ALL'
        else:
            # 提取被排除的被试名
            parts = exp_name.split('_exclude_')
            if len(parts) > 1:
                result['exclude_type'] = 'single'
                result['excluded_subject'] = parts[1]
            else:
                result['exclude_type'] = 'unknown'
                result['excluded_subject'] = 'UNKNOWN'

        results.append(result)

    print(f"成功加载 {len(results)} 个实验结果")
    return results


def compute_statistical_tests(single_results: List[Dict], all_result: Dict, metric_key: str) -> Dict:
    """
    计算统计显著性检验

    Args:
        single_results: 单个排除实验的结果列表
        all_result: 排除所有的实验结果
        metric_key: 指标键名

    Returns:
        统计检验结果字典
    """
    single_values = [r[metric_key] for r in single_results if not np.isnan(r.get(metric_key, np.nan))]
    all_value = all_result.get(metric_key, np.nan)

    if not single_values or np.isnan(all_value):
        return {'test': 'N/A', 'p_value': np.nan, 'significant': False}

    # 单样本t检验：all_value 是否显著不同于 single_values 的均值
    t_stat, p_value = stats.ttest_1samp(single_values, all_value)

    # Wilcoxon符号秩检验（非参数检验）
    try:
        # 计算差异
        differences = np.array(single_values) - all_value
        if np.all(differences == 0):
            w_stat, w_p_value = np.nan, 1.0
        else:
            w_stat, w_p_value = stats.wilcoxon(differences, alternative='two-sided')
    except:
        w_stat, w_p_value = np.nan, np.nan

    return {
        't_statistic': t_stat,
        't_p_value': p_value,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_p_value': w_p_value,
        'significant_t': p_value < 0.05 if not np.isnan(p_value) else False,
        'significant_w': w_p_value < 0.05 if not np.isnan(w_p_value) else False,
    }


def generate_comprehensive_table(results: List[Dict]) -> pd.DataFrame:
    """
    生成综合对比表格

    Args:
        results: 实验结果列表

    Returns:
        pandas DataFrame
    """
    df_data = []

    for r in results:
        row = {
            'Experiment': r['experiment'],
            'Excluded': r['excluded_subject'],
            'Type': r['exclude_type'],
            # 基础指标
            'Best F1': r['best_test_f1'],
            'Train F1': r['final_train_f1'],
            'Test F1': r['final_test_f1'],
            'Train Loss': r['final_train_loss'],
            'Test Loss': r['final_test_loss'],
        }

        # 扩展指标
        if not np.isnan(r.get('gross_accuracy', np.nan)):
            row.update({
                'Gross Acc': r['gross_accuracy'],
                'Top-1': r['top1_accuracy'],
                'Top-3': r['top3_accuracy'],
                'Top-5': r['top5_accuracy'],
                'Balanced Acc': r['balanced_accuracy'],
                'Weighted F1': r['weighted_f1'],
                'Kappa': r['cohen_kappa'],
                'Soft Dice': r['macro_soft_dice'],
                'Risk@95%': r['risk_at_95_coverage'],
            })

        df_data.append(row)

    return pd.DataFrame(df_data)


def print_comparison_tables(results: List[Dict]):
    """
    打印详细的对比表格

    Args:
        results: 实验结果列表
    """
    print("\n" + "="*120)
    print("实验结果对比表 - 基础指标")
    print("="*120)

    # 基础指标表格
    df_basic = []
    for r in results:
        df_basic.append({
            '实验': r['experiment'][:30],  # 截断过长的名称
            '排除': r['excluded_subject'][:15],
            '类型': r['exclude_type'],
            'Best F1': f"{r['best_test_f1']:.4f}",
            'Train F1': f"{r['final_train_f1']:.4f}",
            'Test F1': f"{r['final_test_f1']:.4f}",
            'Train Loss': f"{r['final_train_loss']:.4f}",
            'Test Loss': f"{r['final_test_loss']:.4f}",
        })

    df = pd.DataFrame(df_basic)
    print(df.to_string(index=False))

    # 扩展指标表格
    has_extended = any(not np.isnan(r.get('gross_accuracy', np.nan)) for r in results)

    if has_extended:
        print("\n" + "="*120)
        print("实验结果对比表 - 扩展评估指标")
        print("="*120)

        df_extended = []
        for r in results:
            if not np.isnan(r.get('gross_accuracy', np.nan)):
                df_extended.append({
                    '实验': r['experiment'][:30],
                    '排除': r['excluded_subject'][:15],
                    'Gross Acc': f"{r['gross_accuracy']:.4f}",
                    'Top-1': f"{r['top1_accuracy']:.4f}",
                    'Top-3': f"{r['top3_accuracy']:.4f}",
                    'Top-5': f"{r['top5_accuracy']:.4f}",
                    'Bal Acc': f"{r['balanced_accuracy']:.4f}",
                    'W-F1': f"{r['weighted_f1']:.4f}",
                    'Kappa': f"{r['cohen_kappa']:.4f}",
                    'Dice': f"{r['macro_soft_dice']:.4f}",
                    'Risk@95': f"{r['risk_at_95_coverage']:.4f}",
                })

        if df_extended:
            df_ext = pd.DataFrame(df_extended)
            print(df_ext.to_string(index=False))


def print_statistical_analysis(results: List[Dict]):
    """
    打印详细的统计分析

    Args:
        results: 实验结果列表
    """
    print("\n" + "="*120)
    print("统计分析")
    print("="*120)

    # 分组
    single_exps = [r for r in results if r['exclude_type'] == 'single']
    all_exps = [r for r in results if r['exclude_type'] == 'all']

    # 基础指标统计
    print("\n【基础指标统计】")
    print("-" * 120)

    for category, metrics in [('Basic Metrics', METRICS_METADATA['basic'])]:
        for key, meta in metrics.items():
            values = [r[key] for r in results if not np.isnan(r.get(key, np.nan))]
            if values:
                print(f"\n{meta['name']} ({meta['description']}):")
                print(f"  Mean ± Std: {np.mean(values):.4f} ± {np.std(values):.4f}")
                print(f"  Median [IQR]: {np.median(values):.4f} [{np.percentile(values, 25):.4f}, {np.percentile(values, 75):.4f}]")
                print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
                print(f"  CV: {np.std(values)/np.mean(values)*100:.2f}%")

    # 扩展指标统计
    has_extended = any(not np.isnan(r.get('gross_accuracy', np.nan)) for r in results)

    if has_extended:
        print("\n【扩展指标统计】")
        print("-" * 120)

        for category_name, category_metrics in [
            ('Accuracy Metrics', METRICS_METADATA['accuracy']),
            ('Balance Metrics', METRICS_METADATA['balance']),
            ('Consistency Metrics', METRICS_METADATA['consistency']),
            ('Segmentation Metrics', METRICS_METADATA['segmentation']),
            ('Risk Metrics', METRICS_METADATA['risk']),
        ]:
            print(f"\n{category_name}:")
            for key, meta in category_metrics.items():
                values = [r[key] for r in results if not np.isnan(r.get(key, np.nan))]
                if values:
                    print(f"  {meta['name']}:")
                    print(f"    Mean ± Std: {np.mean(values):.4f} ± {np.std(values):.4f}")
                    print(f"    Median [IQR]: {np.median(values):.4f} [{np.percentile(values, 25):.4f}, {np.percentile(values, 75):.4f}]")
                    print(f"    Range: [{np.min(values):.4f}, {np.max(values):.4f}]")

    # 排除策略对比
    if single_exps and all_exps:
        print("\n【排除策略对比分析】")
        print("-" * 120)

        all_exp = all_exps[0]

        # 基础指标对比
        print("\nBasic Metrics Comparison:")
        print(f"{'Metric':<25} {'Single (Mean±Std)':<25} {'All':<15} {'Diff':<15} {'p-value':<15}")
        print("-" * 100)

        for key in ['best_test_f1', 'final_train_loss', 'final_test_loss']:
            single_values = [r[key] for r in single_exps if not np.isnan(r.get(key, np.nan))]
            all_value = all_exp.get(key, np.nan)

            if single_values and not np.isnan(all_value):
                stats_result = compute_statistical_tests(single_exps, all_exp, key)
                mean_single = np.mean(single_values)
                std_single = np.std(single_values)
                diff = all_value - mean_single

                meta = METRICS_METADATA['basic'][key]
                sig_marker = "***" if stats_result.get('significant_t', False) else ""

                print(f"{meta['name']:<25} {mean_single:.4f}±{std_single:.4f} {'':<8} {all_value:<15.4f} {diff:+.4f} {'':<5} {stats_result.get('t_p_value', np.nan):<15.4f} {sig_marker}")

        # 扩展指标对比
        if has_extended and not np.isnan(all_exp.get('gross_accuracy', np.nan)):
            print("\nExtended Metrics Comparison:")
            print(f"{'Metric':<25} {'Single (Mean±Std)':<25} {'All':<15} {'Diff':<15} {'p-value':<15}")
            print("-" * 100)

            extended_keys = ['gross_accuracy', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy',
                           'balanced_accuracy', 'weighted_f1', 'cohen_kappa',
                           'macro_soft_dice', 'risk_at_95_coverage']

            for key in extended_keys:
                single_values = [r[key] for r in single_exps if not np.isnan(r.get(key, np.nan))]
                all_value = all_exp.get(key, np.nan)

                if single_values and not np.isnan(all_value):
                    stats_result = compute_statistical_tests(single_exps, all_exp, key)
                    mean_single = np.mean(single_values)
                    std_single = np.std(single_values)
                    diff = all_value - mean_single

                    # 查找指标元数据
                    meta_name = key
                    for cat_metrics in METRICS_METADATA.values():
                        if key in cat_metrics:
                            meta_name = cat_metrics[key]['name']
                            break

                    sig_marker = "***" if stats_result.get('significant_t', False) else ""

                    print(f"{meta_name:<25} {mean_single:.4f}±{std_single:.4f} {'':<8} {all_value:<15.4f} {diff:+.4f} {'':<5} {stats_result.get('t_p_value', np.nan):<15.4f} {sig_marker}")

        print("\n注: *** 表示 p < 0.05 (统计显著)")


def plot_extended_comparison(results: List[Dict], output_dir: Path):
    """
    绘制扩展的对比图表

    Args:
        results: 实验结果列表
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否有扩展指标
    has_extended = any(not np.isnan(r.get('gross_accuracy', np.nan)) for r in results)

    if not has_extended:
        print("警告: 没有扩展指标数据，跳过扩展图表绘制")
        return

    # 过滤有扩展指标的结果
    results_with_metrics = [r for r in results if not np.isnan(r.get('gross_accuracy', np.nan))]

    if not results_with_metrics:
        return

    # 图1: 综合指标对比 (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Comprehensive Metrics Comparison Across Experiments', fontsize=16, fontweight='bold')

    experiments = [r['excluded_subject'] for r in results_with_metrics]
    colors = ['red' if r['exclude_type'] == 'all' else 'steelblue' for r in results_with_metrics]

    # 定义要绘制的指标
    metrics_to_plot = [
        ('best_test_f1', 'Best Test F1 (Core Metric)'),
        ('gross_accuracy', 'Gross Accuracy'),
        ('top3_accuracy', 'Top-3 Accuracy'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('weighted_f1', 'Weighted F1'),
        ('cohen_kappa', "Cohen's Kappa"),
        ('macro_soft_dice', 'Macro Soft Dice'),
        ('risk_at_95_coverage', 'Risk @ 95% Coverage'),
        ('final_test_loss', 'Final Test Loss'),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        values = [r[metric_key] for r in results_with_metrics]

        bars = ax.bar(range(len(experiments)), values, color=colors, alpha=0.7, edgecolor='black')

        # 添加均值线
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')

        # 标注最大最小值
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        ax.text(max_idx, values[max_idx], f'{values[max_idx]:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkgreen')
        ax.text(min_idx, values[min_idx], f'{values[min_idx]:.3f}',
               ha='center', va='top', fontsize=8, fontweight='bold', color='darkred')

        ax.set_xlabel('Excluded Subject', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 综合指标对比图已保存: {output_dir / 'comprehensive_metrics_comparison.png'}")
    plt.close()

    # 图2: Top-K准确率对比
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(len(experiments))
    width = 0.25

    top1_values = [r['top1_accuracy'] for r in results_with_metrics]
    top3_values = [r['top3_accuracy'] for r in results_with_metrics]
    top5_values = [r['top5_accuracy'] for r in results_with_metrics]

    ax.bar(x - width, top1_values, width, label='Top-1', alpha=0.8, edgecolor='black')
    ax.bar(x, top3_values, width, label='Top-3', alpha=0.8, edgecolor='black')
    ax.bar(x + width, top5_values, width, label='Top-5', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Excluded Subject', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Top-K Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'topk_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Top-K准确率对比图已保存: {output_dir / 'topk_accuracy_comparison.png'}")
    plt.close()

    # 图3: 相关性热力图
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 构建相关性矩阵
    metrics_for_corr = [
        'best_test_f1', 'gross_accuracy', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy',
        'balanced_accuracy', 'weighted_f1', 'cohen_kappa', 'macro_soft_dice', 'risk_at_95_coverage'
    ]

    corr_data = []
    for metric in metrics_for_corr:
        corr_data.append([r[metric] for r in results_with_metrics])

    corr_matrix = np.corrcoef(corr_data)

    # 绘制热力图
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

    # 设置刻度
    metric_labels = ['Best F1', 'Gross Acc', 'Top-1', 'Top-3', 'Top-5',
                    'Bal Acc', 'W-F1', 'Kappa', 'Soft Dice', 'Risk@95']
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(metric_labels)

    # 添加相关系数文本
    for i in range(len(metrics_for_corr)):
        for j in range(len(metrics_for_corr)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ 指标相关性热力图已保存: {output_dir / 'metrics_correlation_heatmap.png'}")
    plt.close()

    # 图4: 箱线图对比（单个 vs 所有）
    single_exps = [r for r in results_with_metrics if r['exclude_type'] == 'single']
    all_exps = [r for r in results_with_metrics if r['exclude_type'] == 'all']

    if single_exps and all_exps:
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        fig.suptitle('Single Exclusion vs All Exclusion Comparison', fontsize=14, fontweight='bold')

        comparison_metrics = [
            ('best_test_f1', 'Best Test F1'),
            ('gross_accuracy', 'Gross Accuracy'),
            ('balanced_accuracy', 'Balanced Accuracy'),
            ('weighted_f1', 'Weighted F1'),
            ('cohen_kappa', "Cohen's Kappa"),
            ('macro_soft_dice', 'Macro Soft Dice'),
            ('risk_at_95_coverage', 'Risk @ 95%'),
            ('final_test_loss', 'Test Loss'),
        ]

        for idx, (metric_key, metric_name) in enumerate(comparison_metrics):
            ax = axes[idx // 4, idx % 4]

            single_values = [r[metric_key] for r in single_exps]
            all_values = [r[metric_key] for r in all_exps]

            bp = ax.boxplot([single_values, all_values],
                           labels=['Single', 'All'],
                           patch_artist=True,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

            # 设置箱体颜色
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')

            # 添加散点
            x_single = np.random.normal(1, 0.04, size=len(single_values))
            x_all = np.random.normal(2, 0.04, size=len(all_values))
            ax.scatter(x_single, single_values, alpha=0.4, s=30, color='blue')
            ax.scatter(x_all, all_values, alpha=0.4, s=30, color='red')

            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'single_vs_all_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"✓ 单个vs所有箱线图已保存: {output_dir / 'single_vs_all_boxplot.png'}")
        plt.close()


def save_detailed_report(results: List[Dict], output_file: Path):
    """
    保存详细的JSON报告

    Args:
        results: 实验结果列表
        output_file: 输出文件路径
    """
    # 计算统计信息
    single_exps = [r for r in results if r['exclude_type'] == 'single']
    all_exps = [r for r in results if r['exclude_type'] == 'all']

    report = {
        'summary': {
            'total_experiments': len(results),
            'single_exclusion_experiments': len(single_exps),
            'all_exclusion_experiments': len(all_exps),
        },
        'experiments': results,
    }

    # 添加统计分析
    statistics = {}

    # 所有指标的统计
    all_metrics_keys = list(METRICS_METADATA['basic'].keys())
    if any(not np.isnan(r.get('gross_accuracy', np.nan)) for r in results):
        for category_metrics in [METRICS_METADATA['accuracy'], METRICS_METADATA['balance'],
                                METRICS_METADATA['consistency'], METRICS_METADATA['segmentation'],
                                METRICS_METADATA['risk']]:
            all_metrics_keys.extend(category_metrics.keys())

    for key in all_metrics_keys:
        values = [r[key] for r in results if not np.isnan(r.get(key, np.nan))]
        if values:
            statistics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }

    report['statistics'] = statistics

    # 添加对比分析
    if single_exps and all_exps:
        comparison = {}
        all_exp = all_exps[0]

        for key in all_metrics_keys:
            single_values = [r[key] for r in single_exps if not np.isnan(r.get(key, np.nan))]
            all_value = all_exp.get(key, np.nan)

            if single_values and not np.isnan(all_value):
                stats_result = compute_statistical_tests(single_exps, all_exp, key)
                comparison[key] = {
                    'single_mean': float(np.mean(single_values)),
                    'single_std': float(np.std(single_values)),
                    'all_value': float(all_value),
                    'difference': float(all_value - np.mean(single_values)),
                    'relative_change': float((all_value - np.mean(single_values)) / np.mean(single_values) * 100),
                    'statistical_tests': {
                        't_test_p_value': float(stats_result.get('t_p_value', np.nan)),
                        'wilcoxon_p_value': float(stats_result.get('wilcoxon_p_value', np.nan)),
                        'significant': bool(stats_result.get('significant_t', False)),
                    }
                }

        report['comparison'] = comparison

    # 保存报告
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ 详细报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='扩展的排除实验结果对比分析')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='实验结果目录')
    parser.add_argument('--output_file', type=str, default='comparison_report_extended.json',
                       help='输出报告文件路径')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file)

    # 加载结果
    print(f"\n{'='*120}")
    print(f"加载实验结果: {results_dir}")
    print(f"{'='*120}")

    results = load_experiment_results(results_dir)

    if not results:
        print("错误: 未找到任何实验结果！")
        return

    # 打印对比表格
    print_comparison_tables(results)

    # 打印统计分析
    print_statistical_analysis(results)

    # 绘制扩展对比图
    print(f"\n{'='*120}")
    print("生成可视化图表...")
    print(f"{'='*120}\n")

    plot_extended_comparison(results, results_dir)

    # 保存详细报告
    print(f"\n{'='*120}")
    print("保存详细报告...")
    print(f"{'='*120}\n")

    save_detailed_report(results, output_file)

    print(f"\n{'='*120}")
    print("分析完成！")
    print(f"{'='*120}\n")
    print(f"结果目录: {results_dir}")
    print(f"详细报告: {output_file}")
    print(f"可视化图表:")
    print(f"  - comprehensive_metrics_comparison.png")
    print(f"  - topk_accuracy_comparison.png")
    print(f"  - metrics_correlation_heatmap.png")
    print(f"  - single_vs_all_boxplot.png")


if __name__ == '__main__':
    main()
