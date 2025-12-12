#!/usr/bin/env python3
"""
实验结果可视化工具
生成对比图表和性能分析
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys
import numpy as np

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_experiments(json_dir: Path, pattern: str = "*experiment*.json") -> List[Dict]:
    """加载实验 JSON 文件"""
    if not json_dir.exists() or not json_dir.is_dir():
        print(f"❌ 目录不存在: {json_dir}")
        return []

    json_files = list(json_dir.glob(pattern))
    experiments = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                experiments.append(json.load(f))
        except Exception as e:
            print(f"⚠️  警告: 无法加载 {json_file.name}: {e}")

    print(f"✅ 加载了 {len(experiments)} 个实验记录\n")
    return experiments


def get_nested_value(data: Dict, path: str, default=None):
    """从嵌套字典中安全获取值"""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def plot_metric_comparison(
    experiments: List[Dict],
    metrics: List[str],
    output_path: Path,
    figsize: tuple = (12, 6)
) -> None:
    """
    绘制多个指标的对比柱状图

    Args:
        experiments: 实验列表
        metrics: 要对比的指标列表
        output_path: 输出图片路径
        figsize: 图片大小
    """
    # 提取数据
    method_keys = []
    data = {metric: [] for metric in metrics}

    for exp in experiments:
        method_key = exp.get("method_key", "unknown")
        method_keys.append(method_key)

        for metric in metrics:
            value = get_nested_value(exp, f"results.global_metrics.{metric}")
            data[metric].append(value if value is not None else 0)

    # 创建图表
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = data[metric]

        # 绘制柱状图
        bars = ax.bar(range(len(method_keys)), values, alpha=0.7)

        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(method_keys)))
        ax.set_xticklabels(method_keys, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 指标对比图已保存到: {output_path}")
    plt.close()


def plot_training_curves(
    experiments: List[Dict],
    output_path: Path,
    figsize: tuple = (14, 10)
) -> None:
    """
    绘制训练曲线对比

    Args:
        experiments: 实验列表
        output_path: 输出图片路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    metrics = ['train_loss', 'test_loss', 'train_f1', 'test_f1']

    for exp in experiments:
        method_key = exp.get("method_key", "unknown")

        # 尝试加载训练历史
        history_path = get_nested_value(exp, "results.logs.train_curve_path")
        if not history_path:
            continue

        history_file = Path(history_path)
        if not history_file.exists():
            continue

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            continue

        # 绘制每个指标
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=method_key, marker='o', markersize=3)

    # 设置每个子图
    labels = ['Train Loss', 'Test Loss', 'Train F1', 'Test F1']
    for idx, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Over Epochs')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线对比图已保存到: {output_path}")
    plt.close()


def plot_performance_vs_params(
    experiments: List[Dict],
    metric: str,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> None:
    """
    绘制性能与参数量的关系散点图

    Args:
        experiments: 实验列表
        metric: 性能指标
        output_path: 输出图片路径
        figsize: 图片大小
    """
    method_keys = []
    param_counts = []
    metric_values = []

    for exp in experiments:
        method_key = exp.get("method_key", "unknown")
        param_count = get_nested_value(exp, "model.param_count_m")
        metric_value = get_nested_value(exp, f"results.global_metrics.{metric}")

        if param_count is not None and metric_value is not None:
            method_keys.append(method_key)
            param_counts.append(param_count)
            metric_values.append(metric_value)

    if not method_keys:
        print(f"⚠️  没有足够的数据绘制性能-参数量图")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点
    scatter = ax.scatter(param_counts, metric_values, s=100, alpha=0.6, c=range(len(method_keys)),
                        cmap='viridis')

    # 标注每个点
    for i, method_key in enumerate(method_keys):
        ax.annotate(method_key, (param_counts[i], metric_values[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Model Parameters (M)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.upper()} vs Model Parameters')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 性能-参数量图已保存到: {output_path}")
    plt.close()


def plot_heatmap_comparison(
    experiments: List[Dict],
    metrics: List[str],
    output_path: Path,
    figsize: tuple = (12, 8)
) -> None:
    """
    绘制多指标热力图

    Args:
        experiments: 实验列表
        metrics: 指标列表
        output_path: 输出图片路径
        figsize: 图片大小
    """
    method_keys = []
    data_matrix = []

    for exp in experiments:
        method_key = exp.get("method_key", "unknown")
        method_keys.append(method_key)

        row = []
        for metric in metrics:
            value = get_nested_value(exp, f"results.global_metrics.{metric}")
            row.append(value if value is not None else 0)
        data_matrix.append(row)

    if not data_matrix:
        print(f"⚠️  没有数据可以绘制热力图")
        return

    # 创建 DataFrame
    df = pd.DataFrame(data_matrix, index=method_keys, columns=metrics)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Metric Value'})

    ax.set_title('Multi-Metric Heatmap Comparison')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Methods')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 多指标热力图已保存到: {output_path}")
    plt.close()


def plot_radar_chart(
    experiments: List[Dict],
    metrics: List[str],
    output_path: Path,
    figsize: tuple = (10, 10)
) -> None:
    """
    绘制雷达图对比多个指标

    Args:
        experiments: 实验列表
        metrics: 指标列表
        output_path: 输出图片路径
        figsize: 图片大小
    """
    if len(metrics) < 3:
        print(f"⚠️  雷达图至少需要 3 个指标")
        return

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for exp in experiments:
        method_key = exp.get("method_key", "unknown")

        # 提取数据
        values = []
        for metric in metrics:
            value = get_nested_value(exp, f"results.global_metrics.{metric}")
            values.append(value if value is not None else 0)

        values += values[:1]  # 闭合

        # 绘制
        ax.plot(angles, values, 'o-', linewidth=2, label=method_key)
        ax.fill(angles, values, alpha=0.1)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Radar Chart', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 雷达图已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化实验结果对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成所有默认图表
  python visualize_experiments.py --dir ./results --output ./plots

  # 只生成指标对比图
  python visualize_experiments.py --dir ./results --output ./plots --plots metric_comparison

  # 自定义指标
  python visualize_experiments.py --dir ./results --output ./plots --metrics macro_f1 balanced_accuracy kappa

  # 生成雷达图
  python visualize_experiments.py --dir ./results --output ./plots --plots radar --metrics macro_f1 balanced_accuracy kappa ece
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        required=True,
        help='包含实验 JSON 文件的目录'
    )

    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*experiment*.json',
        help='文件名匹配模式'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./plots',
        help='输出目录'
    )

    parser.add_argument(
        '--plots',
        type=str,
        nargs='+',
        default=['metric_comparison', 'heatmap'],
        choices=['metric_comparison', 'training_curves', 'performance_vs_params',
                'heatmap', 'radar', 'all'],
        help='要生成的图表类型'
    )

    parser.add_argument(
        '--metrics', '-m',
        type=str,
        nargs='+',
        default=['macro_f1', 'balanced_accuracy', 'kappa'],
        help='要可视化的指标'
    )

    parser.add_argument(
        '--perf-metric',
        type=str,
        default='macro_f1',
        help='性能-参数量图使用的指标'
    )

    args = parser.parse_args()

    # 加载实验
    json_dir = Path(args.dir)
    experiments = load_experiments(json_dir, args.pattern)

    if not experiments:
        print("❌ 没有找到实验数据")
        return 1

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果选择 'all'，生成所有图表
    if 'all' in args.plots:
        plots = ['metric_comparison', 'training_curves', 'performance_vs_params',
                'heatmap', 'radar']
    else:
        plots = args.plots

    print(f"\n正在生成 {len(plots)} 种图表...\n")

    # 生成各种图表
    if 'metric_comparison' in plots:
        output_path = output_dir / 'metric_comparison.png'
        plot_metric_comparison(experiments, args.metrics, output_path)

    if 'training_curves' in plots:
        output_path = output_dir / 'training_curves.png'
        plot_training_curves(experiments, output_path)

    if 'performance_vs_params' in plots:
        output_path = output_dir / f'performance_vs_params_{args.perf_metric}.png'
        plot_performance_vs_params(experiments, args.perf_metric, output_path)

    if 'heatmap' in plots:
        output_path = output_dir / 'metric_heatmap.png'
        plot_heatmap_comparison(experiments, args.metrics, output_path)

    if 'radar' in plots:
        output_path = output_dir / 'radar_chart.png'
        plot_radar_chart(experiments, args.metrics, output_path)

    print(f"\n✅ 所有图表已生成到: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
