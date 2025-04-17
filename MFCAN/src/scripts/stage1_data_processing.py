#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段一：数据预处理脚本
包括数据加载、验证和预处理功能
"""

import os
import argparse
import json
import sys
import time
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.data_loader import DataLoader
from data.preprocessor import Preprocessor
from utils.logging_utils import Logger

def main():
    """数据预处理主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脑MRI数据预处理')
    parser.add_argument('--config', type=str, default='configs/data_config.json', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    args = parser.parse_args()
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("Stage1_DataProcessing", log_dir="logs/stage1")
    logger = logger_manager.get_logger()
    
    logger.info("="*80)
    logger.info("阶段一：开始数据预处理流程")
    logger.info(f"配置文件: {args.config}")
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("成功加载配置文件")
        logger_manager.log_config(config, "数据预处理配置")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置输出目录
    output_dir = args.output_dir or config.get('output_dir', 'data/processed')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 记录实验开始
    logger_manager.log_experiment_start("阶段一：脑MRI数据预处理", 
                                       f"配置文件: {args.config}, 时间戳: {timestamp}")
    
    start_time = time.time()
    
    # 步骤1：加载原始数据
    logger.info("步骤1: 加载原始数据...")
    try:
        data_loader = DataLoader(args.config)
        features_dict, labels_dict = data_loader.load_data('all')
        
        # 记录数据集大小
        dataset_sizes = {
            'train': len(features_dict.get('train', [])),
            'val': len(features_dict.get('val', [])),
            'test': len(features_dict.get('test', []))
        }
        logger.info(f"数据集大小 - 训练集: {dataset_sizes['train']}, "
                   f"验证集: {dataset_sizes['val']}, 测试集: {dataset_sizes['test']}")
        
    except Exception as e:
        logger.error(f"加载原始数据失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据预处理", 
                                        {"状态": "失败", "阶段": "数据加载", "错误": str(e)})
        return
    
    # 步骤2：验证数据
    logger.info("步骤2: 验证数据完整性...")
    try:
        validation_report = data_loader.validate_data(features_dict, labels_dict)
        
        # 保存验证报告
        validation_report_path = os.path.join(output_dir, f'data_validation_report_{timestamp}.json')
        with open(validation_report_path, 'w') as f:
            json.dump(validation_report, f, indent=4)
        logger.info(f"数据验证报告已保存至: {validation_report_path}")
        
        # 检查数据问题
        if validation_report['issues']:
            logger.warning("检测到以下数据问题:")
            for issue in validation_report['issues']:
                logger.warning(f"- {issue}")
        else:
            logger.info("未检测到数据问题")
            
    except Exception as e:
        logger.error(f"验证数据失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据预处理", 
                                        {"状态": "失败", "阶段": "数据验证", "错误": str(e)})
        return
    
    # 步骤3：特征分组
    logger.info("步骤3: 分离特征组...")
    try:
        grouped_features = data_loader.split_feature_groups(features_dict)
        
        # 记录特征组信息
        feature_groups = config.get('feature_groups', {})
        for group_name, indices in feature_groups.items():
            if isinstance(indices, list) and len(indices) == 2:
                start, end = indices
                feature_count = end - start + 1
                logger.info(f"特征组 '{group_name}': {feature_count} 个特征, 范围 [{start}, {end}]")
                
        # 保存原始数据和分组特征
        raw_data_path = os.path.join(output_dir, f'feature_groups_{timestamp}.h5')
        data_loader.save_processed_data(features_dict, labels_dict, grouped_features, raw_data_path)
        logger.info(f"原始数据和特征组已保存至: {raw_data_path}")
        
        # 保存一个不带时间戳的副本，方便后续脚本使用
        standard_data_path = os.path.join(output_dir, 'feature_groups.h5')
        data_loader.save_processed_data(features_dict, labels_dict, grouped_features, standard_data_path)
        logger.info(f"原始数据和特征组的标准副本已保存至: {standard_data_path}")
        
    except Exception as e:
        logger.error(f"特征分组失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据预处理", 
                                        {"状态": "失败", "阶段": "特征分组", "错误": str(e)})
        return
    
    # 步骤4：预处理数据
    logger.info("步骤4: 预处理数据...")
    try:
        preprocessor = Preprocessor(args.config)
        
        # 处理全部特征
        logger.info("处理全部特征...")
        transformed_data = preprocessor.fit_transform(features_dict)
        
        # 处理特征组
        logger.info("处理特征组...")
        for group_name, splits in grouped_features.items():
            logger.info(f"处理特征组: {group_name}")
            group_data = {
                'train': splits.get('train', None),
                'val': splits.get('val', None),
                'test': splits.get('test', None)
            }
            # 移除None值
            group_data = {k: v for k, v in group_data.items() if v is not None}
            
            if group_data:
                group_transformed = preprocessor.fit_transform(group_data)
                transformed_data[f"group_{group_name}"] = group_transformed
        
        # 保存预处理后的数据
        preprocessed_data_path = os.path.join(output_dir, f'preprocessed_data_{timestamp}.h5')
        preprocessor.save_transformed_data(transformed_data, labels_dict, preprocessed_data_path)
        logger.info(f"预处理后的数据已保存至: {preprocessed_data_path}")
        
        # 保存标准副本
        standard_preprocessed_path = os.path.join(output_dir, 'preprocessed_data.h5')
        preprocessor.save_transformed_data(transformed_data, labels_dict, standard_preprocessed_path)
        logger.info(f"预处理后的数据标准副本已保存至: {standard_preprocessed_path}")
        
        # 保存转换器信息
        transformers_info_path = os.path.join(output_dir, f'transformers_info_{timestamp}.json')
        preprocessor.save_transformers(transformers_info_path)
        logger.info(f"转换器信息已保存至: {transformers_info_path}")
        
        # 保存特征统计信息
        feature_stats_path = os.path.join(output_dir, f'feature_stats_{timestamp}.json')
        preprocessor.save_feature_stats(feature_stats_path)
        logger.info(f"特征统计信息已保存至: {feature_stats_path}")
        
        # 生成可视化
        viz_dir = os.path.join(output_dir, 'visualizations', timestamp)
        logger.info(f"生成特征可视化到: {viz_dir}")
        preprocessor.visualize_features(viz_dir)
        
    except Exception as e:
        logger.error(f"预处理数据失败: {e}")
        logger_manager.log_experiment_end("阶段一：脑MRI数据预处理", 
                                        {"状态": "失败", "阶段": "数据预处理", "错误": str(e)})
        return
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"数据预处理完成! 总耗时: {processing_time:.2f} 秒")
    
    # 记录实验结束
    results = {
        "状态": "成功",
        "处理时间(秒)": processing_time,
        "训练集大小": dataset_sizes['train'],
        "验证集大小": dataset_sizes['val'],
        "测试集大小": dataset_sizes['test'],
        "预处理方法": config.get('normalization', 'standard'),
        "应用PCA": config.get('apply_pca', False),
        "输出文件": {
            "特征组数据": raw_data_path,
            "预处理数据": preprocessed_data_path,
            "转换器信息": transformers_info_path,
            "特征统计": feature_stats_path
        }
    }
    logger_manager.log_experiment_end("阶段一：脑MRI数据预处理", results)
    
    # 返回输出路径，方便后续脚本使用
    return {
        'feature_groups_path': standard_data_path,
        'preprocessed_data_path': standard_preprocessed_path,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    main()