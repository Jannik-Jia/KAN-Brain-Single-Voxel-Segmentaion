#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序脚本 - 协调执行完整的分析流程
"""

import sys
import os
import random
import numpy as np
import time
from datetime import datetime

# 导入自定义模块
from config import *
from utils import setup_logger
from data_loader import load_multiclass_data_from_dirs, define_big_classes, map_to_big_classes
from feature_analysis import analyze_class_separability, compare_clustering_with_big_classes
from classification import enhanced_preprocess_feature_groups, explore_best_classification_strategy


def main():
    """
    主函数，执行完整的分析流程
    """
    # 设置日志记录器
    logger = setup_logger(SAVE_PATH)
    logger.info(f"开始分析，结果将保存到: {SAVE_PATH}")
    logger.info(f"使用随机种子: {RANDOM_SEED}")
    logger.info(f"特征总维度: {FEATURE_DIM}")
    logger.info(f"类别总数: {NUM_CLASS}")
    
    # 记录程序开始时间
    start_time = time.time()
    
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    try:
        # 1. 加载数据
        logger.info("1. 加载数据...")
        
        # 使用原始的数据加载方法
        dataset_dict = load_multiclass_data_from_dirs(
            DATA_DIRS, 
            apply_pca=False,  
            n_components=0, 
            norm=False,
            logger=logger
        )

        # 只提取验证集数据
        val_data = dataset_dict['val_samples']
        val_labels = dataset_dict['val_labels']

        logger.info(f"验证集数据: {val_data.shape}, 标签: {val_labels.shape}")

        # 2. 定义特征组索引
        feature_indices = {
            'diffusion': DIFF_FEATURES,
            'qti': QTI_FEATURES,
            'cest': CEST_FEATURES
        }

        # 3. 增强预处理 - 应用特征选择和标准化
        logger.info("3. 增强预处理 - 应用特征选择和标准化")
        processed_groups, selected_indices = enhanced_preprocess_feature_groups(
            val_data, feature_indices, val_labels, 
            normalize_method='robust', apply_feature_selection=True,
            n_features={'diffusion': 10, 'qti': 30, 'cest': 20},
            save_path=SAVE_PATH,
            logger=logger
        )

        # 4. 获取当前定义的大类标签
        logger.info("4. 获取当前定义的大类标签")
        fine_to_big, big_to_fine, big_class_names = define_big_classes(logger=logger)

        # 映射细分类标签为大类标签
        big_class_labels = map_to_big_classes(val_labels, fine_to_big, logger=logger)

        logger.info(f"大类标签范围: {np.min(big_class_labels)} 至 {np.max(big_class_labels)}")
        logger.info(f"大类标签唯一值: {np.unique(big_class_labels)}")

        # 5. 分析特征空间类别可分性
        logger.info("5. 分析特征空间类别可分性")
        separability_scores = analyze_class_separability(
            processed_groups, big_class_labels, save_path=SAVE_PATH,
            use_sampling=False,  # 使用全部验证集数据
            logger=logger
        )

        # 6. 探索最佳分类策略
        logger.info("6. 探索最佳分类策略")
        best_strategy = explore_best_classification_strategy(
            processed_groups, big_class_labels, save_path=SAVE_PATH,
            use_sampling=False,  # 使用全部验证集数据
            logger=logger
        )

        # 7. 比较聚类结果与现有大类划分的一致性
        logger.info("7. 比较聚类结果与现有大类划分的一致性")
        best_feature_group = max(separability_scores.items(), key=lambda x: x[1]['overall'])[0]
        best_cluster_method = max(
            best_strategy['clustering'][best_feature_group].items(), 
            key=lambda x: x[1]['silhouette']
        )[0]
        best_cluster_labels = best_strategy['clustering'][best_feature_group][best_cluster_method]['labels']

        # 比较一致性
        consistency_score = compare_clustering_with_big_classes(
            best_cluster_labels, big_class_labels, 
            f"{best_feature_group}_{best_cluster_method}",
            big_class_names=big_class_names,
            save_path=SAVE_PATH,
            logger=logger
        )

        # 8. 输出最终结论
        logger.info("\n最终分析结论:")
        logger.info(f"1. 最佳特征组合: {' + '.join(best_strategy['feature_combination'])}")
        logger.info(f"2. 最佳分类器: {best_strategy['classifier']} (准确率: {best_strategy['accuracy']:.4f})")
        logger.info(f"3. 各特征组最佳聚类方法:")
        for group, cluster_info in best_strategy['clustering'].items():
            best_method = max(cluster_info.items(), key=lambda x: x[1]['silhouette'])[0]
            logger.info(f"   - {group}: {best_method} ({cluster_info[best_method]['n_clusters']} 类)")
        logger.info(f"4. 聚类与预定义大类的一致性得分: {consistency_score:.4f}")

        # 根据分析结果给出建议
        logger.info("\n基于分析结果的建议:")

        # 根据一致性得分给出建议
        if consistency_score > 0.7:
            logger.info("- 当前预定义的大类划分与数据自然聚类高度一致，可以继续使用。")
        elif consistency_score > 0.5:
            logger.info("- 当前预定义的大类划分与数据自然聚类有一定一致性，但可以考虑调整部分类别的归属。")
        else:
            logger.info("- 当前预定义的大类划分与数据自然聚类差异较大，建议使用数据驱动的聚类结果重新定义大类。")

        # 根据特征组表现给出建议
        best_group = max(separability_scores.items(), key=lambda x: x[1]['overall'])[0]
        worst_group = min(separability_scores.items(), key=lambda x: x[1]['overall'])[0]
        logger.info(f"- {best_group} 特征组对类别区分最有效，{worst_group} 特征组效果最差。")

        # 根据最佳特征组合给出建议
        if len(best_strategy['feature_combination']) == len(processed_groups):
            logger.info("- 组合所有特征组能达到最佳分类效果，建议采用全特征模型。")
        else:
            logger.info(f"- 只使用 {' + '.join(best_strategy['feature_combination'])} 即可达到最佳分类效果，可以简化模型。")

        # 总结建议的下一步
        logger.info("\n建议的下一步:")
        if consistency_score <= 0.5:
            logger.info(f"1. 使用 {best_feature_group} 特征组的 {best_cluster_method} 聚类（{best_strategy['clustering'][best_feature_group][best_cluster_method]['n_clusters']} 类）的结果作为新的大类定义。")
            logger.info("2. 基于新的大类定义重新训练分层分类模型。")
        else:
            logger.info("1. 保留当前的大类定义，但考虑调整特征权重，提高最佳特征组的权重。")
            logger.info(f"2. 使用 {best_strategy['classifier']} 作为分类器，结合 {' + '.join(best_strategy['feature_combination'])} 特征组。")

        logger.info(f"3. 实现正则化和早停策略，以避免过拟合。")
        
        # 记录总运行时间
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        
        logger.info(f"\n分析完成！总计耗时: {time_str}")
        logger.info(f"所有结果已保存到: {SAVE_PATH}")
    
    except Exception as e:
        logger.error(f"错误: {str(e)}", exc_info=True)
        raise
    
    return 0


if __name__ == "__main__":
    # 执行主函数
    sys.exit(main())