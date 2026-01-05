#!/usr/bin/env python3
"""
实验记录合并和表格生成工具
将多个实验 JSON 合并成对比表格
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys


def load_experiments(json_dir: Path, pattern: str = "*experiment*.json") -> List[Dict]:
    """
    加载目录下所有实验 JSON 文件

    Args:
        json_dir: 包含 JSON 文件的目录
        pattern: 文件名匹配模式

    Returns:
        实验记录列表
    """
    if not json_dir.exists() or not json_dir.is_dir():
        print(f"❌ 目录不存在或不是目录: {json_dir}")
        return []

    json_files = list(json_dir.glob(pattern))

    if not json_files:
        print(f"❌ 在 {json_dir} 中没有找到匹配 '{pattern}' 的文件")
        return []

    experiments = []
    failed_files = []

    print(f"找到 {len(json_files)} 个实验 JSON 文件")

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                experiment = json.load(f)
                experiments.append(experiment)
        except Exception as e:
            print(f"⚠️  警告: 无法加载 {json_file.name}: {e}")
            failed_files.append(json_file.name)

    if failed_files:
        print(f"⚠️  {len(failed_files)} 个文件加载失败")

    print(f"✅ 成功加载 {len(experiments)} 个实验记录\n")

    return experiments


def get_nested_value(data: Dict, path: str, default: Any = None) -> Any:
    """
    从嵌套字典中安全获取值

    Args:
        data: 数据字典
        path: 路径，如 "task.split.scheme"
        default: 默认值

    Returns:
        值或默认值
    """
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def extract_key_fields(experiment: Dict) -> Dict[str, Any]:
    """
    从实验记录中提取关键字段用于表格显示

    Args:
        experiment: 实验记录字典

    Returns:
        关键字段字典
    """
    # 提取所有需要对比的字段
    row = {
        # Meta 信息
        "experiment_id": experiment.get("experiment_id", "N/A"),
        "method_name": experiment.get("method_name", "N/A"),
        "method_key": experiment.get("method_key", "N/A"),
        "family": experiment.get("family", "N/A"),
        "subfamily": experiment.get("subfamily", "N/A"),
        "code_version": experiment.get("code_version", "N/A"),
        "seed": experiment.get("seed", None),

        # Task 信息
        "dataset": get_nested_value(experiment, "task.dataset", "N/A"),
        "label_space": get_nested_value(experiment, "task.label_space", "N/A"),
        "split_scheme": get_nested_value(experiment, "task.split.scheme", "N/A"),
        "split_id": get_nested_value(experiment, "task.split.split_id", "N/A"),
        "n_train_voxels": get_nested_value(experiment, "task.n_voxels.train", None),
        "n_test_voxels": get_nested_value(experiment, "task.n_voxels.test", None),

        # Preprocessing
        "normalisation_type": get_nested_value(experiment, "preprocessing.normalisation.type", "N/A"),
        "normalisation_fit_scope": get_nested_value(experiment, "preprocessing.normalisation.params.fit_scope", "N/A"),
        "class_weighting_scheme": get_nested_value(experiment, "preprocessing.class_weighting.scheme", "N/A"),

        # Model 信息
        "model_family": get_nested_value(experiment, "model.family", "N/A"),
        "param_count_m": get_nested_value(experiment, "model.param_count_m", None),
        "input_dim": get_nested_value(experiment, "model.details.input_dim", None),
        "output_dim": get_nested_value(experiment, "model.details.output_dim", None),
        "hidden_layers": str(get_nested_value(experiment, "model.details.hidden_layers", "N/A")),
        "activation": get_nested_value(experiment, "model.details.activation", "N/A"),
        "dropout": get_nested_value(experiment, "model.details.dropout", None),
        "residual": get_nested_value(experiment, "model.details.residual", None),
        "attention": get_nested_value(experiment, "model.details.attention", None),

        # Training 信息
        "loss_type": get_nested_value(experiment, "training.loss.type", "N/A"),
        "loss_class_weights": get_nested_value(experiment, "training.loss.class_weights", "N/A"),
        "optimizer_type": get_nested_value(experiment, "training.optimizer.type", "N/A"),
        "learning_rate": get_nested_value(experiment, "training.optimizer.lr", None),
        "weight_decay": get_nested_value(experiment, "training.optimizer.weight_decay", None),
        "scheduler_type": get_nested_value(experiment, "training.scheduler.type", "N/A"),
        "batch_size": get_nested_value(experiment, "training.batch_size", None),
        "n_epochs": get_nested_value(experiment, "training.n_epochs", None),
        "early_stopping_enabled": get_nested_value(experiment, "training.early_stopping.enabled", None),

        # Hardware 信息
        "gpu": get_nested_value(experiment, "hardware.gpu", "N/A"),
        "num_gpus": get_nested_value(experiment, "hardware.num_gpus", None),
        "train_time_hours": get_nested_value(experiment, "hardware.train_time_hours", None),

        # Results - 全局指标
        "evaluated_split": get_nested_value(experiment, "results.evaluated_split", "N/A"),

        # 基础准确率指标
        "gross_accuracy": get_nested_value(experiment, "results.global_metrics.gross_accuracy", None),
        "top1_accuracy": get_nested_value(experiment, "results.global_metrics.top1_accuracy", None),
        "top3_accuracy": get_nested_value(experiment, "results.global_metrics.top3_accuracy", None),
        "top5_accuracy": get_nested_value(experiment, "results.global_metrics.top5_accuracy", None),
        "balanced_accuracy": get_nested_value(experiment, "results.global_metrics.balanced_accuracy", None),

        # F1 和分割指标
        "macro_f1": get_nested_value(experiment, "results.global_metrics.macro_f1", None),
        "weighted_f1": get_nested_value(experiment, "results.global_metrics.weighted_f1", None),
        "macro_soft_dice": get_nested_value(experiment, "results.global_metrics.macro_soft_dice", None),

        # 一致性和校准指标
        "cohen_kappa": get_nested_value(experiment, "results.global_metrics.cohen_kappa", None),
        "kappa": get_nested_value(experiment, "results.global_metrics.kappa", None),
        "nll": get_nested_value(experiment, "results.global_metrics.nll", None),
        "ece": get_nested_value(experiment, "results.global_metrics.ece", None),
        "brier_score": get_nested_value(experiment, "results.global_metrics.brier_score", None),

        # 风险覆盖指标
        "risk_at_95_coverage": get_nested_value(experiment, "results.global_metrics.risk_at_95_coverage", None),
        "actual_coverage_95": get_nested_value(experiment, "results.global_metrics.actual_coverage_95", None),

        # 其他指标
        "gc": get_nested_value(experiment, "results.global_metrics.gc", None),

        # 向后兼容
        "top_3_accuracy": get_nested_value(experiment, "results.global_metrics.top_3_accuracy", None),
    }

    return row


def create_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    """
    从实验列表创建对比表格

    Args:
        experiments: 实验记录列表

    Returns:
        pandas DataFrame
    """
    if not experiments:
        print("❌ 没有实验数据可以处理")
        return pd.DataFrame()

    rows = [extract_key_fields(exp) for exp in experiments]
    df = pd.DataFrame(rows)

    # 按 method_key 排序
    if "method_key" in df.columns:
        df = df.sort_values("method_key")

    return df


def save_table_csv(df: pd.DataFrame, output_path: Path) -> None:
    """保存表格为 CSV 文件"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ CSV 表格已保存到: {output_path}")


def save_table_markdown(df: pd.DataFrame, output_path: Path) -> None:
    """保存表格为 Markdown 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 实验对比表格\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"✅ Markdown 表格已保存到: {output_path}")


def save_table_excel(df: pd.DataFrame, output_path: Path) -> None:
    """保存表格为 Excel 文件（如果安装了 openpyxl）"""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Experiments', index=False)
        print(f"✅ Excel 表格已保存到: {output_path}")
    except ImportError:
        print("⚠️  警告: 未安装 openpyxl，无法保存 Excel 文件")
        print("   安装方法: pip install openpyxl")


def print_summary(df: pd.DataFrame) -> None:
    """打印实验统计摘要"""
    print("\n" + "=" * 80)
    print("实验统计摘要")
    print("=" * 80)

    print(f"\n总实验数: {len(df)}")

    # 按 family 统计
    if "family" in df.columns:
        print("\n按模型 family 统计:")
        family_counts = df["family"].value_counts()
        for family, count in family_counts.items():
            print(f"  - {family}: {count} 个实验")

    # 按 normalisation_type 统计
    if "normalisation_type" in df.columns:
        print("\n按归一化方法统计:")
        norm_counts = df["normalisation_type"].value_counts()
        for norm, count in norm_counts.items():
            print(f"  - {norm}: {count} 个实验")

    # 结果指标统计
    metric_columns = [
        "gross_accuracy", "top1_accuracy", "top3_accuracy", "top5_accuracy",
        "balanced_accuracy", "macro_f1", "weighted_f1", "macro_soft_dice",
        "cohen_kappa", "kappa", "nll", "ece", "brier_score",
        "risk_at_95_coverage", "actual_coverage_95", "gc"
    ]
    available_metrics = [col for col in metric_columns if col in df.columns]

    if available_metrics:
        print("\n结果指标统计（非空值）:")
        for metric in available_metrics:
            non_null_count = df[metric].notna().sum()
            if non_null_count > 0:
                mean_val = df[metric].mean()
                max_val = df[metric].max()
                min_val = df[metric].min()
                print(f"  - {metric}:")
                print(f"      有效值数: {non_null_count}/{len(df)}")
                print(f"      平均值: {mean_val:.4f}")
                print(f"      最大值: {max_val:.4f}")
                print(f"      最小值: {min_val:.4f}")

    print("=" * 80)


def generate_performance_ranking(df: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """
    生成按指定指标排名的表格

    Args:
        df: 实验表格
        metric: 排名依据的指标

    Returns:
        排名后的表格
    """
    if metric not in df.columns:
        print(f"⚠️  警告: 指标 '{metric}' 不存在")
        return df

    # 过滤出有该指标值的实验
    ranked_df = df[df[metric].notna()].copy()

    if len(ranked_df) == 0:
        print(f"⚠️  警告: 没有实验有 '{metric}' 的值")
        return df

    # 按指标降序排序
    ranked_df = ranked_df.sort_values(metric, ascending=False)

    # 添加排名列
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))

    return ranked_df


def main():
    parser = argparse.ArgumentParser(
        description='合并多个实验 JSON 并生成对比表格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法：合并并生成 CSV
  python merge_experiments.py --dir ./results --output comparison.csv

  # 生成多种格式
  python merge_experiments.py --dir ./results --output comparison --formats csv markdown excel

  # 生成排名表格
  python merge_experiments.py --dir ./results --output comparison.csv --rank-by macro_f1

  # 只打印摘要，不保存文件
  python merge_experiments.py --dir ./results --summary-only
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
        help='文件名匹配模式 (默认: *experiment*.json)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出文件路径（不含扩展名，会根据 formats 自动添加）'
    )

    parser.add_argument(
        '--formats', '-f',
        type=str,
        nargs='+',
        default=['csv'],
        choices=['csv', 'markdown', 'excel'],
        help='输出格式 (默认: csv)'
    )

    parser.add_argument(
        '--rank-by', '-r',
        type=str,
        help='按指定指标生成排名表格 (如: macro_f1, gc, balanced_accuracy)'
    )

    parser.add_argument(
        '--summary-only', '-s',
        action='store_true',
        help='只打印统计摘要，不保存文件'
    )

    args = parser.parse_args()

    # 加载实验
    json_dir = Path(args.dir)
    experiments = load_experiments(json_dir, args.pattern)

    if not experiments:
        print("❌ 没有找到有效的实验记录")
        return 1

    # 创建对比表格
    print("正在创建对比表格...")
    df = create_comparison_table(experiments)

    if df.empty:
        print("❌ 无法创建表格")
        return 1

    print(f"✅ 成功创建表格，共 {len(df)} 行, {len(df.columns)} 列\n")

    # 打印摘要
    print_summary(df)

    # 如果只需要摘要，直接返回
    if args.summary_only:
        return 0

    # 生成排名表格
    if args.rank_by:
        print(f"\n正在按 '{args.rank_by}' 生成排名...")
        df = generate_performance_ranking(df, args.rank_by)

    # 保存文件
    if args.output:
        output_base = Path(args.output)

        if 'csv' in args.formats:
            output_path = output_base.with_suffix('.csv')
            save_table_csv(df, output_path)

        if 'markdown' in args.formats:
            output_path = output_base.with_suffix('.md')
            save_table_markdown(df, output_path)

        if 'excel' in args.formats:
            output_path = output_base.with_suffix('.xlsx')
            save_table_excel(df, output_path)

        print("\n✅ 所有表格生成完成！")
    else:
        print("\n⚠️  未指定输出文件，表格未保存")

    return 0


if __name__ == "__main__":
    sys.exit(main())
