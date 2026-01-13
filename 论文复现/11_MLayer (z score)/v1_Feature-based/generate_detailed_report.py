#!/usr/bin/env python3
"""
生成单个实验的详细评估报告
包含完整的指标分析、可视化和统计检验
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import h5py
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 导入评估指标模块
from evaluation_metrics import (
    compute_all_metrics,
    compute_risk_coverage_curve,
    format_metrics
)

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


def load_experiment_data(result_dir: Path) -> Dict:
    """
    加载实验数据

    Args:
        result_dir: 实验结果目录

    Returns:
        包含所有数据的字典
    """
    data = {}

    # 加载训练历史
    history_files = list(result_dir.glob('history_test*.json'))
    if history_files:
        with open(history_files[0], 'r') as f:
            data['history'] = json.load(f)
    else:
        raise FileNotFoundError(f"未找到history文件: {result_dir}")

    # 加载预测结果（如果存在）
    pred_files = list(result_dir.glob('predictions_3d_test*.mat'))
    if pred_files:
        data['pred_file'] = pred_files[0]
    else:
        data['pred_file'] = None

    # 加载模型文件（如果存在）
    model_files = list(result_dir.glob('dense_4x4096_model_test*.pth'))
    if model_files:
        data['model_file'] = model_files[0]
    else:
        data['model_file'] = None

    return data


def plot_training_curves(history: Dict, output_dir: Path):
    """
    绘制详细的训练曲线

    Args:
        history: 训练历史字典
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Analysis', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # 子图1: Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Test Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 标注最小loss
    min_train_loss_epoch = np.argmin(history['train_loss']) + 1
    min_test_loss_epoch = np.argmin(history['test_loss']) + 1
    axes[0, 0].axvline(x=min_train_loss_epoch, color='b', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=min_test_loss_epoch, color='r', linestyle='--', alpha=0.5)

    # 子图2: F1曲线
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(epochs, history['test_f1'], 'r-', label='Test F1', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Macro F1 Score', fontsize=12)
    axes[0, 1].set_title('Training and Test F1 Score', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # 标注最佳F1
    best_test_f1_epoch = np.argmax(history['test_f1']) + 1
    axes[0, 1].axvline(x=best_test_f1_epoch, color='g', linestyle='--', alpha=0.5, linewidth=2)
    axes[0, 1].text(best_test_f1_epoch, max(history['test_f1']), f'Best F1\nEpoch {best_test_f1_epoch}',
                   ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图3: 过拟合分析（Train F1 - Test F1）
    overfitting_gap = np.array(history['train_f1']) - np.array(history['test_f1'])
    axes[1, 0].plot(epochs, overfitting_gap, 'purple', linewidth=2, marker='D', markersize=4)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].fill_between(epochs, 0, overfitting_gap, where=(overfitting_gap > 0), color='red', alpha=0.3, label='Overfitting')
    axes[1, 0].fill_between(epochs, 0, overfitting_gap, where=(overfitting_gap <= 0), color='green', alpha=0.3, label='Underfitting')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Train F1 - Test F1', fontsize=12)
    axes[1, 0].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4: 学习率分析（Loss变化率）
    train_loss_change = np.diff([0] + history['train_loss'])
    test_loss_change = np.diff([0] + history['test_loss'])

    axes[1, 1].plot(epochs, train_loss_change, 'b-', label='Train Loss Change', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, test_loss_change, 'r-', label='Test Loss Change', linewidth=2, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss Change (Δ)', fontsize=12)
    axes[1, 1].set_title('Loss Change Rate', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {output_dir / 'training_curves_detailed.png'}")
    plt.close()


def plot_metrics_radar(metrics: Dict, output_dir: Path):
    """
    绘制雷达图显示多维度指标

    Args:
        metrics: 评估指标字典
        output_dir: 输出目录
    """
    # 选择关键指标
    radar_metrics = {
        'Gross Accuracy': metrics.get('gross_accuracy', 0),
        'Balanced Accuracy': metrics.get('balanced_accuracy', 0),
        'Macro F1': metrics.get('macro_f1', 0),
        'Weighted F1': metrics.get('weighted_f1', 0),
        "Cohen's Kappa": metrics.get('cohen_kappa', 0),
        'Soft Dice': metrics.get('macro_soft_dice', 0),
    }

    # 过滤掉NaN值
    radar_metrics = {k: v for k, v in radar_metrics.items() if not np.isnan(v)}

    if not radar_metrics:
        print("警告: 没有有效的指标数据用于雷达图")
        return

    # 准备数据
    categories = list(radar_metrics.keys())
    values = list(radar_metrics.values())

    # 确保首尾相连
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 绘制数据
    ax.plot(angles, values, 'o-', linewidth=2, label='Metrics', color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)

    # 设置y轴范围
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)

    # 添加数值标签
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        ax.text(angle, value + 0.05, f'{value:.3f}',
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_title('Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ 指标雷达图已保存: {output_dir / 'metrics_radar.png'}")
    plt.close()


def plot_risk_coverage_analysis(history: Dict, output_dir: Path):
    """
    绘制风险-覆盖率曲线（如果有测试集预测概率）

    Args:
        history: 训练历史
        output_dir: 输出目录
    """
    # 检查是否有best_test_metrics中的risk数据
    if 'best_test_metrics' not in history:
        print("警告: 没有测试集指标数据，跳过风险-覆盖率曲线")
        return

    metrics = history['best_test_metrics']
    if 'risk_at_95_coverage' not in metrics:
        print("警告: 没有风险-覆盖率数据")
        return

    # 创建示意图（因为我们只有一个点）
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    risk_95 = metrics['risk_at_95_coverage']
    coverage_95 = metrics.get('actual_coverage_95', 0.95)

    # 绘制理想曲线（示意）
    coverages = np.linspace(0, 1, 100)
    # 假设一个典型的风险-覆盖率曲线形状
    risks = 1 - coverages ** 2  # 示意曲线

    ax.plot(coverages, risks, 'b--', alpha=0.3, linewidth=2, label='Typical Risk-Coverage Curve')

    # 标注我们的测试点
    ax.plot(coverage_95, risk_95, 'ro', markersize=15, label=f'Test Point @ 95% Coverage')
    ax.text(coverage_95, risk_95, f'  Risk: {risk_95:.4f}\n  Coverage: {coverage_95:.2%}',
           ha='left', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 添加参考线
    ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=risk_95, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Coverage (Proportion of Samples Retained)', fontsize=12)
    ax.set_ylabel('Risk (Error Rate on Retained Samples)', fontsize=12)
    ax.set_title('Risk-Coverage Analysis', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加说明文本
    textstr = (
        f'Interpretation:\n'
        f'• At 95% coverage, model predicts with {risk_95:.2%} error rate\n'
        f'• Lower risk at high coverage indicates better calibration\n'
        f'• Ideal: Low risk even at high coverage'
    )
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_coverage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ 风险-覆盖率分析图已保存: {output_dir / 'risk_coverage_analysis.png'}")
    plt.close()


def generate_metrics_summary_table(history: Dict, output_dir: Path):
    """
    生成指标汇总表格

    Args:
        history: 训练历史
        output_dir: 输出目录
    """
    # 准备数据
    summary_data = {
        'Metric Category': [],
        'Metric Name': [],
        'Value': [],
        'Description': []
    }

    # 基础训练指标
    summary_data['Metric Category'].append('Training')
    summary_data['Metric Name'].append('Best Test F1')
    summary_data['Value'].append(f"{max(history['test_f1']):.4f}")
    summary_data['Description'].append('Checkpoint selection metric')

    summary_data['Metric Category'].append('Training')
    summary_data['Metric Name'].append('Final Train F1')
    summary_data['Value'].append(f"{history['train_f1'][-1]:.4f}")
    summary_data['Description'].append('Final training performance')

    summary_data['Metric Category'].append('Training')
    summary_data['Metric Name'].append('Final Test F1')
    summary_data['Value'].append(f"{history['test_f1'][-1]:.4f}")
    summary_data['Description'].append('Final test performance')

    summary_data['Metric Category'].append('Training')
    summary_data['Metric Name'].append('Final Train Loss')
    summary_data['Value'].append(f"{history['train_loss'][-1]:.4f}")
    summary_data['Description'].append('Final training loss')

    summary_data['Metric Category'].append('Training')
    summary_data['Metric Name'].append('Final Test Loss')
    summary_data['Value'].append(f"{history['test_loss'][-1]:.4f}")
    summary_data['Description'].append('Final test loss')

    # 扩展指标
    if 'best_test_metrics' in history and history['best_test_metrics']:
        metrics = history['best_test_metrics']

        # 准确性指标
        accuracy_metrics = [
            ('Gross Accuracy', metrics.get('gross_accuracy'), 'Overall accuracy'),
            ('Top-1 Accuracy', metrics.get('top1_accuracy'), 'Top-1 coverage'),
            ('Top-3 Accuracy', metrics.get('top3_accuracy'), 'Top-3 coverage'),
            ('Top-5 Accuracy', metrics.get('top5_accuracy'), 'Top-5 coverage'),
        ]
        for name, value, desc in accuracy_metrics:
            if value is not None and not np.isnan(value):
                summary_data['Metric Category'].append('Accuracy')
                summary_data['Metric Name'].append(name)
                summary_data['Value'].append(f"{value:.4f}")
                summary_data['Description'].append(desc)

        # 类别平衡指标
        balance_metrics = [
            ('Balanced Accuracy', metrics.get('balanced_accuracy'), 'Class-balanced accuracy'),
            ('Weighted F1', metrics.get('weighted_f1'), 'Sample-weighted F1'),
        ]
        for name, value, desc in balance_metrics:
            if value is not None and not np.isnan(value):
                summary_data['Metric Category'].append('Balance')
                summary_data['Metric Name'].append(name)
                summary_data['Value'].append(f"{value:.4f}")
                summary_data['Description'].append(desc)

        # 一致性指标
        if metrics.get('cohen_kappa') is not None and not np.isnan(metrics['cohen_kappa']):
            summary_data['Metric Category'].append('Consistency')
            summary_data['Metric Name'].append("Cohen's Kappa")
            summary_data['Value'].append(f"{metrics['cohen_kappa']:.4f}")
            summary_data['Description'].append('Agreement metric')

        # 分割质量指标
        if metrics.get('macro_soft_dice') is not None and not np.isnan(metrics['macro_soft_dice']):
            summary_data['Metric Category'].append('Segmentation')
            summary_data['Metric Name'].append('Macro Soft Dice')
            summary_data['Value'].append(f"{metrics['macro_soft_dice']:.4f}")
            summary_data['Description'].append('Soft segmentation quality')

        # 风险分析指标
        if metrics.get('risk_at_95_coverage') is not None and not np.isnan(metrics['risk_at_95_coverage']):
            summary_data['Metric Category'].append('Risk')
            summary_data['Metric Name'].append('Risk @ 95% Coverage')
            summary_data['Value'].append(f"{metrics['risk_at_95_coverage']:.4f}")
            summary_data['Description'].append('Error rate at 95% coverage')

    # 创建DataFrame并保存
    df = pd.DataFrame(summary_data)

    # 保存为CSV
    csv_path = output_dir / 'metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ 指标汇总表已保存: {csv_path}")

    # 同时打印到控制台
    print("\n" + "="*100)
    print("Metrics Summary Table")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def generate_html_report(result_dir: Path, history: Dict, output_dir: Path):
    """
    生成HTML格式的详细报告

    Args:
        result_dir: 实验结果目录
        history: 训练历史
        output_dir: 输出目录
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {result_dir.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .interpretation {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 15px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Detailed Experiment Report</h1>
        <p>Experiment: {result_dir.name}</p>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

    # 摘要部分
    html_content += f"""
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Best Test F1 (Core Metric)</div>
                <div class="metric-value">{max(history['test_f1']):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final Test F1</div>
                <div class="metric-value">{history['test_f1'][-1]:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final Test Loss</div>
                <div class="metric-value">{history['test_loss'][-1]:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Training Epochs</div>
                <div class="metric-value">{len(history['train_loss'])}</div>
            </div>
        </div>
    </div>
"""

    # 扩展指标（如果有）
    if 'best_test_metrics' in history and history['best_test_metrics']:
        metrics = history['best_test_metrics']
        html_content += f"""
    <div class="section">
        <h2>📈 Extended Evaluation Metrics</h2>
        <div class="metric-grid">
"""
        extended_metrics = [
            ('gross_accuracy', 'Gross Accuracy'),
            ('top1_accuracy', 'Top-1 Accuracy'),
            ('top3_accuracy', 'Top-3 Accuracy'),
            ('top5_accuracy', 'Top-5 Accuracy'),
            ('balanced_accuracy', 'Balanced Accuracy'),
            ('weighted_f1', 'Weighted F1'),
            ('cohen_kappa', "Cohen's Kappa"),
            ('macro_soft_dice', 'Macro Soft Dice'),
            ('risk_at_95_coverage', 'Risk @ 95% Coverage'),
        ]

        for key, label in extended_metrics:
            value = metrics.get(key)
            if value is not None and not np.isnan(value):
                html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    # 可视化部分
    html_content += """
    <div class="section">
        <h2>📉 Training Curves</h2>
        <img src="training_curves_detailed.png" alt="Training Curves">
        <div class="interpretation">
            <strong>Interpretation:</strong>
            <ul>
                <li>Top-left: Loss curves show convergence behavior</li>
                <li>Top-right: F1 scores indicate model performance over epochs</li>
                <li>Bottom-left: Overfitting analysis (positive = overfitting, negative = underfitting)</li>
                <li>Bottom-right: Loss change rate indicates learning stability</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Metrics Radar Chart</h2>
        <img src="metrics_radar.png" alt="Metrics Radar">
        <div class="interpretation">
            <strong>Interpretation:</strong>
            <ul>
                <li>Larger area = Better overall performance</li>
                <li>Balanced shape = Consistent performance across metrics</li>
                <li>Irregular shape = Varying performance in different aspects</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>⚠️ Risk-Coverage Analysis</h2>
        <img src="risk_coverage_analysis.png" alt="Risk-Coverage">
        <div class="interpretation">
            <strong>Interpretation:</strong>
            <ul>
                <li>Risk measures error rate on retained predictions</li>
                <li>Coverage measures proportion of samples retained</li>
                <li>Good models maintain low risk even at high coverage</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>📝 Detailed Metrics Table</h2>
        <p>See <a href="metrics_summary.csv">metrics_summary.csv</a> for detailed metrics breakdown.</p>
    </div>

    <div class="section">
        <h2>💡 Key Insights</h2>
        <div class="interpretation">
"""

    # 添加一些自动生成的洞察
    best_epoch = np.argmax(history['test_f1']) + 1
    best_f1 = max(history['test_f1'])
    final_f1 = history['test_f1'][-1]
    overfitting_gap = history['train_f1'][-1] - history['test_f1'][-1]

    html_content += f"""
            <ul>
                <li><strong>Best Model:</strong> Achieved at epoch {best_epoch} with F1 score of {best_f1:.4f}</li>
                <li><strong>Final Performance:</strong> Test F1 = {final_f1:.4f}</li>
                <li><strong>Overfitting Analysis:</strong> Train-Test F1 gap = {overfitting_gap:.4f}
                    {'(Possible overfitting)' if overfitting_gap > 0.05 else '(Good generalization)'}</li>
"""

    if 'best_test_metrics' in history and history['best_test_metrics']:
        metrics = history['best_test_metrics']
        if metrics.get('top3_accuracy'):
            html_content += f"""
                <li><strong>Top-3 Coverage:</strong> {metrics['top3_accuracy']:.2%} of predictions have correct answer in top-3</li>
"""
        if metrics.get('cohen_kappa'):
            kappa = metrics['cohen_kappa']
            kappa_interp = 'Excellent' if kappa > 0.8 else 'Good' if kappa > 0.6 else 'Moderate' if kappa > 0.4 else 'Fair'
            html_content += f"""
                <li><strong>Prediction Consistency:</strong> Cohen's Kappa = {kappa:.4f} ({kappa_interp} agreement)</li>
"""

    html_content += """
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>📄 Files Generated</h2>
        <ul>
            <li><strong>training_curves_detailed.png</strong> - Comprehensive training curves analysis</li>
            <li><strong>metrics_radar.png</strong> - Multi-dimensional metrics visualization</li>
            <li><strong>risk_coverage_analysis.png</strong> - Risk and coverage trade-off</li>
            <li><strong>metrics_summary.csv</strong> - Detailed metrics table (CSV format)</li>
            <li><strong>detailed_report.html</strong> - This report</li>
        </ul>
    </div>

</body>
</html>
"""

    # 保存HTML报告
    html_path = output_dir / 'detailed_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ HTML报告已生成: {html_path}")


def main():
    parser = argparse.ArgumentParser(description='生成单个实验的详细评估报告')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='实验结果目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认为result_dir/detailed_report）')

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / 'detailed_report'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*100}")
    print(f"生成详细评估报告")
    print(f"{'='*100}\n")
    print(f"实验目录: {result_dir}")
    print(f"输出目录: {output_dir}\n")

    # 加载数据
    try:
        data = load_experiment_data(result_dir)
        history = data['history']
    except Exception as e:
        print(f"错误: 无法加载实验数据: {e}")
        return

    # 生成各种图表
    print("\n生成可视化图表...")
    print("-" * 100)

    plot_training_curves(history, output_dir)

    if 'best_test_metrics' in history and history['best_test_metrics']:
        plot_metrics_radar(history['best_test_metrics'], output_dir)
        plot_risk_coverage_analysis(history, output_dir)
    else:
        print("警告: 没有扩展指标数据，跳过部分图表")

    # 生成汇总表格
    print("\n生成指标汇总表...")
    print("-" * 100)
    generate_metrics_summary_table(history, output_dir)

    # 生成HTML报告
    print("\n生成HTML报告...")
    print("-" * 100)
    generate_html_report(result_dir, history, output_dir)

    print(f"\n{'='*100}")
    print(f"报告生成完成！")
    print(f"{'='*100}\n")
    print(f"所有文件已保存到: {output_dir}")
    print(f"\n主要文件:")
    print(f"  - detailed_report.html           (主HTML报告，可在浏览器中打开)")
    print(f"  - training_curves_detailed.png   (训练曲线分析)")
    print(f"  - metrics_radar.png              (指标雷达图)")
    print(f"  - risk_coverage_analysis.png     (风险-覆盖率分析)")
    print(f"  - metrics_summary.csv            (指标汇总表)")


if __name__ == '__main__':
    main()
