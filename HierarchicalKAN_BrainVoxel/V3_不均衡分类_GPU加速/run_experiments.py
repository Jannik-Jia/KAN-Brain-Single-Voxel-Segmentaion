#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量运行实验的脚本，用于比较不同的参数设置
"""

import os
import sys
import subprocess
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import RESULTS_DIR, FIGURES_DIR, PROJECT_NAME
from utils.logging_utils import get_logger

# 获取日志记录器
logger = get_logger("experiments")

def run_experiment(data_subset='val', normalize='robust', pca=None, feature_selection=None,
                 min_clusters=2, max_clusters=10, output_prefix=None, skip_plots=True):
    """
    运行单个实验
    
    参数:
        data_subset: 数据子集
        normalize: 标准化方法
        pca: 是否使用PCA降维
        feature_selection: 是否使用特征选择
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
        output_prefix: 输出前缀
        skip_plots: 是否跳过绘图
        
    返回:
        returncode: 进程返回码
        output_prefix: 使用的输出前缀
    """
    # 构建命令
    cmd = ["python", "main.py", f"--data_subset={data_subset}", f"--normalize={normalize}"]
    
    # 添加PCA参数
    if pca is not None:
        if pca:
            cmd.append("--pca")
        else:
            cmd.append("--no_pca")
    
    # 添加特征选择参数
    if feature_selection is not None:
        if feature_selection:
            cmd.append("--feature_selection")
        else:
            cmd.append("--no_feature_selection")
    
    # 添加聚类参数
    cmd.extend([f"--min_clusters={min_clusters}", f"--max_clusters={max_clusters}"])
    
    # 如果未指定输出前缀，生成一个
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{PROJECT_NAME}_{'pca' if pca else 'nopca'}_{'fs' if feature_selection else 'nofs'}_{timestamp}"
    
    cmd.append(f"--output_prefix={output_prefix}")
    
    # 添加跳过绘图参数
    if skip_plots:
        cmd.append("--skip_plots")
    
    # 记录命令
    logger.info(f"运行实验: {' '.join(cmd)}")
    
    # 运行命令
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 处理输出
    if process.returncode == 0:
        logger.info(f"实验成功完成: {output_prefix}")
    else:
        logger.error(f"实验失败: {output_prefix}")
        logger.error(f"标准错误: {process.stderr}")
    
    return process.returncode, output_prefix

def run_parameter_grid(data_subset='val', normalize_methods=None, pca_options=None,
                      feature_selection_options=None, cluster_ranges=None, skip_plots=True):
    """
    运行参数网格实验
    
    参数:
        data_subset: 数据子集
        normalize_methods: 标准化方法列表
        pca_options: PCA选项列表
        feature_selection_options: 特征选择选项列表
        cluster_ranges: 聚类范围列表，格式为[(min1, max1), (min2, max2), ...]
        skip_plots: 是否跳过绘图
        
    返回:
        results: 实验结果列表
    """
    # 设置默认值
    if normalize_methods is None:
        normalize_methods = ['robust']
    if pca_options is None:
        pca_options = [True, False]
    if feature_selection_options is None:
        feature_selection_options = [True, False]
    if cluster_ranges is None:
        cluster_ranges = [(2, 10)]
    
    # 生成参数组合
    param_grid = list(itertools.product(
        normalize_methods, pca_options, feature_selection_options, cluster_ranges
    ))
    
    # 准备结果存储
    results = []
    
    # 运行每个参数组合
    for i, (normalize, pca, fs, cluster_range) in enumerate(param_grid):
        # 生成输出前缀
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{PROJECT_NAME}_exp{i}_{normalize}_{'pca' if pca else 'nopca'}_{'fs' if fs else 'nofs'}_{timestamp}"
        
        # 运行实验
        returncode, prefix = run_experiment(
            data_subset=data_subset,
            normalize=normalize,
            pca=pca,
            feature_selection=fs,
            min_clusters=cluster_range[0],
            max_clusters=cluster_range[1],
            output_prefix=output_prefix,
            skip_plots=skip_plots
        )
        
        # 添加结果
        results.append({
            'params': {
                'normalize': normalize,
                'pca': pca,
                'feature_selection': fs,
                'cluster_range': cluster_range
            },
            'output_prefix': prefix,
            'success': returncode == 0
        })
    
    return results

def compare_experiment_results(results):
    """
    比较实验结果
    
    参数:
        results: 实验结果列表
        
    返回:
        comparison: 比较结果
    """
    # 筛选成功的实验
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        logger.error("没有成功完成的实验！")
        return None
    
    # 准备比较数据
    comparison_data = []
    
    for result in successful_results:
        # 尝试加载实验报告
        report_path = os.path.join(RESULTS_DIR, f"{result['output_prefix']}_final_report.json")
        
        if not os.path.exists(report_path):
            logger.warning(f"找不到实验报告: {report_path}")
            continue
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # 提取关键指标
            entry = {
                'output_prefix': result['output_prefix'],
                'normalize': result['params']['normalize'],
                'pca': result['params']['pca'],
                'feature_selection': result['params']['feature_selection']
            }
            
            # 添加聚类一致性评分
            if 'mapping_evaluation' in report and 'best_mapping' in report['mapping_evaluation']:
                entry['consistency_score'] = report['mapping_evaluation']['best_mapping']['consistency_score']
            
            # 添加分类性能
            if 'classification_analysis' in report and 'best_performance' in report['classification_analysis']:
                entry['balanced_accuracy'] = report['classification_analysis']['best_performance']['balanced_accuracy']
                entry['f1_macro'] = report['classification_analysis']['best_performance']['f1_macro']
            
            comparison_data.append(entry)
        
        except Exception as e:
            logger.error(f"处理实验报告时出错: {e}")
    
    # 创建比较数据框
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # 排序（按平衡准确率）
        if 'balanced_accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('balanced_accuracy', ascending=False)
        
        # 输出比较结果
        logger.info("\n实验比较结果:")
        logger.info(str(comparison_df))
        
        # 可视化比较结果
        visualize_experiment_comparison(comparison_df)
        
        return comparison_df
    else:
        logger.warning("没有有效的比较数据！")
        return None

def visualize_experiment_comparison(comparison_df):
    """
    可视化实验比较结果
    
    参数:
        comparison_df: 比较数据框
    """
    if len(comparison_df) == 0:
        logger.warning("没有数据可视化！")
        return
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 8))
    
    # 对实验进行分组
    grouped = comparison_df.groupby(['pca', 'feature_selection'])
    
    # 设置条形图的位置
    bar_width = 0.35
    positions = np.arange(len(grouped))
    
    # 绘制平衡准确率
    if 'balanced_accuracy' in comparison_df.columns:
        plt.subplot(2, 1, 1)
        means = grouped['balanced_accuracy'].mean().values
        plt.bar(positions, means, bar_width, label='Balanced Accuracy')
        
        # 添加标签
        plt.xticks(positions, [f"PCA={g[0]}, FS={g[1]}" for g in grouped.groups.keys()])
        plt.ylabel('Balanced Accuracy')
        plt.title('Comparison of Balanced Accuracy Across Configurations')
        plt.grid(axis='y')
    
    # 绘制一致性得分
    if 'consistency_score' in comparison_df.columns:
        plt.subplot(2, 1, 2)
        means = grouped['consistency_score'].mean().values
        plt.bar(positions, means, bar_width, label='Consistency Score')
        
        # 添加标签
        plt.xticks(positions, [f"PCA={g[0]}, FS={g[1]}" for g in grouped.groups.keys()])
        plt.ylabel('Consistency Score')
        plt.title('Comparison of Clustering Consistency Across Configurations')
        plt.grid(axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, "experiment_comparison.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"实验比较图表已保存至: {save_path}")
    
    plt.close()
    
    # 如果有归一化方法的比较，绘制额外的图表
    if 'normalize' in comparison_df.columns and len(comparison_df['normalize'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        
        # 按归一化方法分组
        grouped_by_norm = comparison_df.groupby('normalize')
        
        # 绘制平衡准确率
        if 'balanced_accuracy' in comparison_df.columns:
            plt.subplot(2, 1, 1)
            means = grouped_by_norm['balanced_accuracy'].mean().values
            plt.bar(np.arange(len(grouped_by_norm)), means, bar_width)
            
            # 添加标签
            plt.xticks(np.arange(len(grouped_by_norm)), grouped_by_norm.groups.keys())
            plt.ylabel('Balanced Accuracy')
            plt.title('Comparison of Balanced Accuracy Across Normalization Methods')
            plt.grid(axis='y')
        
        # 绘制一致性得分
        if 'consistency_score' in comparison_df.columns:
            plt.subplot(2, 1, 2)
            means = grouped_by_norm['consistency_score'].mean().values
            plt.bar(np.arange(len(grouped_by_norm)), means, bar_width)
            
            # 添加标签
            plt.xticks(np.arange(len(grouped_by_norm)), grouped_by_norm.groups.keys())
            plt.ylabel('Consistency Score')
            plt.title('Comparison of Clustering Consistency Across Normalization Methods')
            plt.grid(axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(FIGURES_DIR, "normalization_comparison.png")
        plt.savefig(save_path, dpi=300)
        logger.info(f"归一化方法比较图表已保存至: {save_path}")
        
        plt.close()

def main():
    """主执行函数"""
    # 定义要比较的参数
    normalize_methods = ['robust', 'standard', 'none']
    pca_options = [True, False]
    feature_selection_options = [True, False]
    cluster_ranges = [(2, 7), (5, 10)]
    
    # 运行参数网格实验
    results = run_parameter_grid(
        data_subset='val',
        normalize_methods=normalize_methods,
        pca_options=pca_options,
        feature_selection_options=feature_selection_options,
        cluster_ranges=cluster_ranges,
        skip_plots=True
    )
    
    # 比较实验结果
    comparison = compare_experiment_results(results)
    
    # 保存比较结果
    if comparison is not None:
        output_path = os.path.join(RESULTS_DIR, "experiment_comparison.csv")
        comparison.to_csv(output_path, index=False)
        logger.info(f"实验比较结果已保存至: {output_path}")
    
    return comparison

if __name__ == "__main__":
    main()