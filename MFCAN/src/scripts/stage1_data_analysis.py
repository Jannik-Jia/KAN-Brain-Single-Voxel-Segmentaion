#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段一：数据分析脚本
包括基础统计分析、相关性分析和类别可分性分析
"""

import os
import argparse
import json
import sys
import time
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from analysis.basic_analyzer import BasicAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.separability_analyzer import ClassSeparabilityAnalyzer
from utils.logging_utils import Logger

def load_h5_data(file_path):
    """加载HDF5格式的数据"""
    import h5py
    data_dict = {}
    
    with h5py.File(file_path, 'r') as f:
        # 读取所有组和数据集
        def visit_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 将数据集加载到内存
                parts = name.split('/')
                current_dict = data_dict
                for i, part in enumerate(parts[:-1]):
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                current_dict[parts[-1]] = obj[()]
        
        f.visititems(visit_group)
    
    return data_dict

def main():
    """数据分析主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脑MRI数据分析')
    parser.add_argument('--config', type=str, default='configs/data_config.json', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/feature_groups.h5', help='特征组数据路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    args = parser.parse_args()
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("Stage1_DataAnalysis", log_dir="logs/stage1")
    logger = logger_manager.get_logger()
    
    logger.info("="*80)
    logger.info("阶段一：开始数据分析流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("成功加载配置文件")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置输出目录
    output_dir = args.output_dir or config.get('output_dir', 'results/analysis')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 记录实验开始
    logger_manager.log_experiment_start("阶段一：脑MRI数据分析", 
                                       f"数据文件: {args.data_path}, 时间戳: {timestamp}")
    
    start_time = time.time()
    
    # 步骤1：加载数据
    logger.info("步骤1: 加载特征组数据...")
    try:
        data_dict = load_h5_data(args.data_path)
        logger.info("数据加载成功")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据分析", 
                                        {"状态": "失败", "阶段": "数据加载", "错误": str(e)})
        return
    
    # 提取训练集特征和标签
    try:
        # 提取原始特征和标签
        if 'original' in data_dict and 'train' in data_dict['original']:
            train_features = data_dict['original']['train']['features']
            train_labels = data_dict['original']['train']['labels']
            logger.info(f"提取到原始训练集: 特征形状={train_features.shape}, 标签形状={train_labels.shape}")
        else:
            # 尝试其他可能的路径
            if 'train' in data_dict and 'features' in data_dict['train']:
                train_features = data_dict['train']['features']
                train_labels = data_dict['train']['labels']
                logger.info(f"提取到训练集: 特征形状={train_features.shape}, 标签形状={train_labels.shape}")
            else:
                raise KeyError("无法在数据字典中找到训练集特征和标签")
        
        # 提取分组特征
        group_features = {}
        if 'grouped' in data_dict and 'train' in data_dict['grouped']:
            for group_name, group_data in data_dict['grouped']['train'].items():
                if 'features' in group_data:
                    group_features[group_name] = group_data['features']
                    logger.info(f"提取到特征组 '{group_name}': 形状={group_features[group_name].shape}")
        else:
            # 尝试其他可能的数据结构
            for key, value in data_dict.items():
                if key.startswith('group_') and 'train' in value and 'features' in value['train']:
                    group_name = key.replace('group_', '')
                    group_features[group_name] = value['train']['features']
                    logger.info(f"提取到特征组 '{group_name}': 形状={group_features[group_name].shape}")
    
    except Exception as e:
        logger.error(f"提取特征和标签失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据分析", 
                                        {"状态": "失败", "阶段": "数据提取", "错误": str(e)})
        return
    
    # 步骤2：基础统计分析
    logger.info("步骤2: 执行基础统计分析...")
    try:
        basic_analyzer = BasicAnalyzer(config_path=args.config, output_dir=output_dir, logger=logger)
        
        # 分析全部特征
        logger.info("分析全部特征的基础统计量...")
        basic_stats = basic_analyzer.compute_basic_stats(train_features)
        basic_analyzer.generate_stats_report(output_dir=output_dir, prefix='all')
        
        # 分析各特征组
        for group_name, features in group_features.items():
            logger.info(f"分析特征组 '{group_name}' 的基础统计量...")
            group_stats = basic_analyzer.compute_basic_stats(features, feature_group=group_name)
            basic_analyzer.generate_stats_report(output_dir=output_dir, prefix=group_name)
        
        logger.info("基础统计分析完成")
    except Exception as e:
        logger.error(f"基础统计分析失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据分析", 
                                        {"状态": "部分完成", "阶段": "基础统计分析", "错误": str(e)})
    
    # 步骤3：相关性分析
    logger.info("步骤3: 执行特征相关性分析...")
    try:
        correlation_analyzer = CorrelationAnalyzer(config_path=args.config, output_dir=output_dir, logger=logger)
        
        # 分析全部特征内部相关性
        logger.info("分析全部特征的相关性...")
        corr_matrix = correlation_analyzer.analyze_intra_group_correlation(train_features)
        correlation_analyzer.visualize_correlation_matrix(corr_matrix, prefix='all')
        high_corr_pairs = correlation_analyzer.identify_correlation_clusters(corr_matrix, threshold=0.8)
        correlation_analyzer.save_high_correlations(high_corr_pairs, prefix='all')
        
        # 分析各特征组内部相关性
        for group_name, features in group_features.items():
            logger.info(f"分析特征组 '{group_name}' 的内部相关性...")
            group_corr = correlation_analyzer.analyze_intra_group_correlation(features)
            correlation_analyzer.visualize_correlation_matrix(group_corr, prefix=group_name)
            group_high_corr = correlation_analyzer.identify_correlation_clusters(group_corr, threshold=0.8)
            correlation_analyzer.save_high_correlations(group_high_corr, prefix=group_name)
        
        # 分析特征组间相关性
        if len(group_features) > 1:
            logger.info("分析特征组间相关性...")
            inter_group_corr = correlation_analyzer.analyze_inter_group_correlation(group_features)
            correlation_analyzer.visualize_inter_group_correlation(inter_group_corr)
        
        logger.info("相关性分析完成")
    except Exception as e:
        logger.error(f"相关性分析失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据分析", 
                                        {"状态": "部分完成", "阶段": "相关性分析", "错误": str(e)})
    
    # 步骤4：类别可分性分析
    logger.info("步骤4: 执行类别可分性分析...")
    try:
        separability_analyzer = ClassSeparabilityAnalyzer(config_path=args.config, 
                                                        output_dir=output_dir, logger=logger)
        
        # 分析全部特征的类别可分性
        logger.info("分析全部特征的类别可分性...")
        feature_significance = separability_analyzer.compute_feature_significance(train_features, train_labels)
        discriminative_features = separability_analyzer.identify_discriminative_features(
            train_features, train_labels, feature_significance)
        class_similarity = separability_analyzer.analyze_class_similarity(train_features, train_labels)
        
        # 可视化分析结果
        separability_analyzer.visualize_class_separability(feature_significance, prefix='all')
        separability_analyzer.visualize_class_similarity(class_similarity, prefix='all')
        
        # 保存分析结果
        separability_analyzer.save_analysis_results(
            feature_significance, discriminative_features, class_similarity, prefix='all')
        
        # 分析各特征组的类别可分性
        for group_name, features in group_features.items():
            logger.info(f"分析特征组 '{group_name}' 的类别可分性...")
            group_significance = separability_analyzer.compute_feature_significance(
                features, train_labels, feature_group=group_name)
            group_discriminative = separability_analyzer.identify_discriminative_features(
                features, train_labels, group_significance, feature_group=group_name)
            group_similarity = separability_analyzer.analyze_class_similarity(
                features, train_labels, feature_group=group_name)
            
            # 可视化特征组分析结果
            separability_analyzer.visualize_class_separability(
                group_significance, prefix=group_name)
            separability_analyzer.visualize_class_similarity(
                group_similarity, prefix=group_name)
            
            # 保存特征组分析结果
            separability_analyzer.save_analysis_results(
                group_significance, group_discriminative, group_similarity, prefix=group_name)
        
        logger.info("类别可分性分析完成")
    except Exception as e:
        logger.error(f"类别可分性分析失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据分析", 
                                        {"状态": "部分完成", "阶段": "类别可分性分析", "错误": str(e)})
    
    # 生成汇总报告
    logger.info("生成数据分析汇总报告...")
    try:
        # 创建汇总报告目录
        summary_dir = os.path.join(output_dir, 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        # 基础统计汇总
        basic_analyzer.generate_summary_report(output_path=os.path.join(summary_dir, 'basic_analysis_summary.md'))
        
        # 相关性分析汇总
        correlation_analyzer.generate_summary_report(output_path=os.path.join(summary_dir, 'correlation_analysis_summary.md'))
        
        # 类别可分性汇总
        separability_analyzer.generate_summary_report(output_path=os.path.join(summary_dir, 'separability_analysis_summary.md'))
        
        # 全局汇总报告
        all_summary_path = os.path.join(summary_dir, 'data_analysis_summary.md')
        with open(all_summary_path, 'w') as f:
            f.write(f"# 脑MRI数据分析汇总报告\n\n")
            f.write(f"## 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 数据概况\n\n")
            f.write(f"- 全部特征数量: {train_features.shape[1]}\n")
            f.write(f"- 样本数量: {train_features.shape[0]}\n")
            f.write(f"- 类别数量: {len(set(train_labels.flatten()))}\n")
            f.write(f"- 特征组数量: {len(group_features)}\n\n")
            
            f.write(f"## 特征组信息\n\n")
            for group_name, features in group_features.items():
                f.write(f"- {group_name}: {features.shape[1]} 特征\n")
            
            f.write(f"\n## 分析结果概要\n\n")
            f.write(f"详细分析结果请查看各子报告文件。\n\n")
            
            f.write(f"### 基础统计分析\n\n")
            f.write(f"- 见 [基础统计分析汇总](basic_analysis_summary.md)\n\n")
            
            f.write(f"### 相关性分析\n\n")
            f.write(f"- 见 [相关性分析汇总](correlation_analysis_summary.md)\n\n")
            
            f.write(f"### 类别可分性分析\n\n")
            f.write(f"- 见 [类别可分性分析汇总](separability_analysis_summary.md)\n\n")
        
        logger.info(f"数据分析汇总报告已保存至: {all_summary_path}")
    except Exception as e:
        logger.error(f"生成汇总报告失败: {e}")
    
    end_time = time.time()
    analysis_time = end_time - start_time
    
    logger.info(f"数据分析完成! 总耗时: {analysis_time:.2f} 秒")
    
    # 记录实验结束
    results = {
        "状态": "成功",
        "分析时间(秒)": analysis_time,
        "特征维度": train_features.shape[1],
        "样本数量": train_features.shape[0],
        "类别数量": len(set(train_labels.flatten())),
        "特征组数量": len(group_features),
        "输出目录": output_dir
    }
    logger_manager.log_experiment_end("阶段一：脑MRI数据分析", results)
    
    # 返回输出路径，方便后续脚本使用
    return {
        'analysis_output_dir': output_dir,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    main()