#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段二：特征工程脚本
包括特征重要性分析和特征选择
"""

import os
import argparse
import json
import sys
import time
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from analysis.feature_importance import FeatureImportanceAnalyzer
from feature_engineering.feature_selector import FeatureSelector
from feature_engineering.dimension_reducer import DimensionReducer
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
    """特征工程主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脑MRI数据特征工程')
    parser.add_argument('--config', type=str, default='configs/data_config.json', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/feature_groups.h5', help='特征组数据路径')
    parser.add_argument('--analysis_dir', type=str, default=None, help='阶段一分析结果目录，若不指定则使用最新结果')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    parser.add_argument('--skip_pca', type=str, default='false', help='是否跳过PCA降维步骤')
    parser.add_argument('--feature_selection_method', type=str, default='combined', choices=['rf', 'mi', 'permutation', 'combined'], help='特征选择方法')
    parser.add_argument('--use_gpu', type=str, default='false', help='是否使用GPU加速计算')
    parser.add_argument('--skip_umap', type=str, default='false', help='是否跳过UMAP可视化步骤')
    
    args = parser.parse_args()
    
    # 解析布尔参数
    skip_pca = args.skip_pca.lower() == 'true'
    skip_umap = args.skip_umap.lower() == 'true'
    feature_selection_method = args.feature_selection_method
    use_gpu = args.use_gpu.lower() == 'true'



    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("Stage2_FeatureEngineering", log_dir="logs/stage2")
    logger = logger_manager.get_logger()
    
    logger.info("="*80)
    logger.info("阶段二：开始特征工程流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"特征选择方法: {feature_selection_method}")
    logger.info(f"跳过PCA: {skip_pca}")
    logger.info(f"使用GPU: {use_gpu}")
    logger.info(f"跳过UMAP: {skip_umap}")
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("成功加载配置文件")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置输出目录
    output_dir = args.output_dir or config.get('output_dir', 'results/feature_engineering')
    output_dir = os.path.join(output_dir, f"{feature_selection_method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 记录实验开始
    logger_manager.log_experiment_start("阶段二：脑MRI数据特征工程", 
                                       f"数据文件: {args.data_path}, 时间戳: {timestamp}, 特征选择方法: {feature_selection_method}")
    
    start_time = time.time()
    
    # 步骤1：加载数据
    logger.info("步骤1: 加载特征组数据...")
    try:
        data_dict = load_h5_data(args.data_path)
        logger.info("数据加载成功")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        logger_manager.log_experiment_end("阶段二：脑MRI数据特征工程", 
                                        {"状态": "失败", "阶段": "数据加载", "错误": str(e)})
        return
    
    # 提取训练集特征和标签
    try:
        # 提取原始特征和标签
        if 'original' in data_dict and 'train' in data_dict['original']:
            train_features = data_dict['original']['train']['features']
            train_labels = data_dict['original']['train']['labels']
            val_features = data_dict['original']['val']['features']
            val_labels = data_dict['original']['val']['labels']
            test_features = data_dict['original']['test']['features']
            test_labels = data_dict['original']['test']['labels']
            logger.info(f"提取到原始训练集: 特征形状={train_features.shape}, 标签形状={train_labels.shape}")
            logger.info(f"提取到原始验证集: 特征形状={val_features.shape}, 标签形状={val_labels.shape}")
            logger.info(f"提取到原始测试集: 特征形状={test_features.shape}, 标签形状={test_labels.shape}")
        else:
            # 尝试其他可能的路径
            if 'train' in data_dict and 'features' in data_dict['train']:
                train_features = data_dict['train']['features']
                train_labels = data_dict['train']['labels']
                val_features = data_dict['val']['features']
                val_labels = data_dict['val']['labels']
                test_features = data_dict['test']['features']
                test_labels = data_dict['test']['labels']
                logger.info(f"提取到训练集: 特征形状={train_features.shape}, 标签形状={train_labels.shape}")
                logger.info(f"提取到验证集: 特征形状={val_features.shape}, 标签形状={val_labels.shape}")
                logger.info(f"提取到测试集: 特征形状={test_features.shape}, 标签形状={test_labels.shape}")
            else:
                raise KeyError("无法在数据字典中找到训练集特征和标签")
        
        # 提取分组特征
        group_features = {}
        if 'grouped' in data_dict and 'train' in data_dict['grouped']:
            for group_name, group_data in data_dict['grouped']['train'].items():
                if 'features' in group_data:
                    group_features[group_name] = {
                        'train': group_data['features'],
                        'val': data_dict['grouped']['val'][group_name]['features'],
                        'test': data_dict['grouped']['test'][group_name]['features']
                    }
                    logger.info(f"提取到特征组 '{group_name}': 训练集形状={group_features[group_name]['train'].shape}")
        else:
            # 尝试其他可能的数据结构
            for key, value in data_dict.items():
                if key.startswith('group_') and 'train' in value and 'features' in value['train']:
                    group_name = key.replace('group_', '')
                    group_features[group_name] = {
                        'train': value['train']['features'],
                        'val': value['val']['features'],
                        'test': value['test']['features']
                    }
                    logger.info(f"提取到特征组 '{group_name}': 训练集形状={group_features[group_name]['train'].shape}")
    
    except Exception as e:
        logger.error(f"提取特征和标签失败: {e}")
        logger_manager.log_experiment_end("阶段二：脑MRI数据特征工程", 
                                        {"状态": "失败", "阶段": "数据提取", "错误": str(e)})
        return
    
    # 步骤2：特征重要性分析
    logger.info("步骤2: 执行特征重要性分析...")
    try:
        feature_importance_analyzer = FeatureImportanceAnalyzer(
            config_path=args.config, output_dir=output_dir, logger=logger, use_gpu=use_gpu)
        
        # 分析全部特征 - 使用指定的特征选择方法
        logger.info(f"分析全部特征的重要性 (使用 {feature_selection_method} 方法)...")
        if feature_selection_method != 'combined':
            importance_scores = feature_importance_analyzer.compute_importance_scores(
                train_features, train_labels, method=feature_selection_method)
        else:
            importance_scores = feature_importance_analyzer.compute_importance_scores(
                train_features, train_labels, method='all')
                
        ranked_features = feature_importance_analyzer.rank_features(importance_scores)
        thresholds = feature_importance_analyzer.generate_importance_thresholds(ranked_features)
        
        # 可视化特征重要性
        feature_importance_analyzer.visualize_importance(ranked_features, prefix='all')
        
        # 保存结果
        feature_importance_analyzer.save_results(
            ranked_features, thresholds, output_dir=os.path.join(output_dir, 'importance'), prefix='all')
        
        # 分析各特征组 - 使用指定的特征选择方法
        group_importance_results = {}
        for group_name, group_data in group_features.items():
            logger.info(f"分析特征组 '{group_name}' 的重要性 (使用 {feature_selection_method} 方法)...")
            if feature_selection_method != 'combined':
                group_importance = feature_importance_analyzer.compute_importance_scores(
                    group_data['train'], train_labels, method=feature_selection_method, feature_group=group_name)
            else:
                group_importance = feature_importance_analyzer.compute_importance_scores(
                    group_data['train'], train_labels, method='all', feature_group=group_name)
                    
            group_ranked = feature_importance_analyzer.rank_features(group_importance)
            group_thresholds = feature_importance_analyzer.generate_importance_thresholds(group_ranked)
            
            # 可视化
            feature_importance_analyzer.visualize_importance(group_ranked, prefix=group_name)
            
            # 保存结果
            feature_importance_analyzer.save_results(
                group_ranked, group_thresholds, 
                output_dir=os.path.join(output_dir, 'importance'), 
                prefix=group_name)
            
            # 存储结果以供特征选择使用
            group_importance_results[group_name] = {
                'importance': group_importance,
                'ranked': group_ranked,
                'thresholds': group_thresholds
            }
        
        # 生成特征重要性汇总报告
        feature_importance_analyzer.generate_summary_report(output_path=os.path.join(output_dir, 'importance_summary.md'))
        
        logger.info("特征重要性分析完成")
        
    except Exception as e:
        logger.error(f"特征重要性分析失败: {e}")
        logger_manager.log_experiment_end("阶段二：脑MRI数据特征工程", 
                                        {"状态": "部分完成", "阶段": "特征重要性分析", "错误": str(e)})
    
    # 步骤3：特征选择
    logger.info("步骤3: 执行特征选择...")
    try:
        feature_selector = FeatureSelector(
            config_path=args.config, importance_path=os.path.join(output_dir, 'importance'), 
            output_dir=output_dir, logger=logger)
        
        # 根据重要性选择全部特征子集
        logger.info("根据重要性选择全部特征子集...")
        selected_all = feature_selector.select_by_importance(
            train_features, ranked_features, coverage=0.98)
        
        # 移除冗余特征
        logger.info("移除冗余特征...")
        selected_all_nonredundant = feature_selector.remove_redundancy(
            train_features, selected_all, correlation_threshold=0.9)
        
        # 生成不同策略的特征子集
        logger.info("生成不同策略的特征子集...")
        feature_subsets_all = feature_selector.generate_feature_subsets(
            train_features, ranked_features, prefix='all')
        
        # 选择特征组子集
        group_selected_subsets = {}
        for group_name, importance_data in group_importance_results.items():
            logger.info(f"选择特征组 '{group_name}' 的子集...")
            
            group_features_train = group_features[group_name]['train']
            group_ranked = importance_data['ranked']
            
            # 根据重要性选择
            group_selected = feature_selector.select_by_importance(
                group_features_train, group_ranked, coverage=0.95)

            # 移除冗余
            group_selected_nonredundant = feature_selector.remove_redundancy(
                group_features_train, group_selected, correlation_threshold=0.9)
            
            # 生成子集
            group_subsets = feature_selector.generate_feature_subsets(
                group_features_train, group_ranked, prefix=group_name)
            
            group_selected_subsets[group_name] = {
                'importance': group_selected,
                'nonredundant': group_selected_nonredundant,
                'subsets': group_subsets
            }
        
        # 保存特征子集
        logger.info("保存特征子集...")
        all_selected_data = {
            'all': {
                'train': feature_selector.apply_selection(train_features, selected_all_nonredundant),
                'val': feature_selector.apply_selection(val_features, selected_all_nonredundant),
                'test': feature_selector.apply_selection(test_features, selected_all_nonredundant),
                'selected_indices': selected_all_nonredundant
            }
        }
        
        # 将特征组子集添加到数据中
        for group_name, selected_data in group_selected_subsets.items():
            nonredundant_indices = selected_data['nonredundant']
            all_selected_data[group_name] = {
                'train': feature_selector.apply_selection(group_features[group_name]['train'], nonredundant_indices),
                'val': feature_selector.apply_selection(group_features[group_name]['val'], nonredundant_indices),
                'test': feature_selector.apply_selection(group_features[group_name]['test'], nonredundant_indices),
                'selected_indices': nonredundant_indices
            }
        
        # 保存所选特征集
        output_filename = f"{feature_selection_method}_selected_features.h5"
        feature_selector.save_feature_subsets(
            all_selected_data, train_labels, val_labels, test_labels,
            output_path=os.path.join(output_dir, output_filename))
        
        # 生成特征选择报告
        feature_selector.generate_summary_report(output_path=os.path.join(output_dir, 'feature_selection_summary.md'))
        
        logger.info("特征选择完成")
        
    except Exception as e:
        logger.error(f"特征选择失败: {e}")
        logger_manager.log_experiment_end("阶段二：脑MRI数据特征工程", 
                                        {"状态": "部分完成", "阶段": "特征选择", "错误": str(e)})
    
    # 步骤4：降维与特征变换 (如果未跳过PCA)
    if not skip_pca:
        logger.info("步骤4: 执行降维与特征变换...")
        try:
            dimension_reducer = DimensionReducer(
                config_path=args.config, output_dir=output_dir, logger=logger)
            
            # 使用PCA降维 (如果未跳过PCA)
            if not skip_pca:
                logger.info("应用PCA降维到全部特征...")
                pca_features = dimension_reducer.apply_pca(
                    train_features, val_features, test_features, 
                    feature_group='all', variance=0.95)
            else:
                logger.info("根据参数设置跳过PCA降维步骤")
                pca_features = None
            
            # 使用UMAP降维 (可视化用) (如果未跳过UMAP)
            if not skip_umap:
                logger.info("应用UMAP降维到全部特征 (可视化用)...")
                umap_features = dimension_reducer.apply_umap(
                    train_features, train_labels, feature_group='all', n_components=2)
            else:
                logger.info("根据参数设置跳过UMAP可视化步骤")
                umap_features = None
            
            # 对特征组应用降维 (如果既不跳过PCA也不跳过UMAP)
            group_transformed_features = {}
            if not (skip_pca and skip_umap):
                for group_name, group_data in group_features.items():
                    if not skip_pca:
                        logger.info(f"应用PCA降维到特征组 '{group_name}'...")
                        group_pca = dimension_reducer.apply_pca(
                            group_data['train'], group_data['val'], group_data['test'],
                            feature_group=group_name, variance=0.95)
                    else:
                        group_pca = None
                        
                    if not skip_umap:
                        logger.info(f"应用UMAP降维到特征组 '{group_name}' (可视化用)...")
                        group_umap = dimension_reducer.apply_umap(
                            group_data['train'], train_labels, feature_group=group_name, n_components=2)
                    else:
                        group_umap = None
                        
                    if group_pca is not None or group_umap is not None:
                        group_transformed_features[group_name] = {}
                        if group_pca is not None:
                            group_transformed_features[group_name]['pca'] = group_pca
                        if group_umap is not None:
                            group_transformed_features[group_name]['umap'] = group_umap
            
            # 保存变换后的特征 (如果有)
            if (pca_features is not None or umap_features is not None) and group_transformed_features:
                logger.info("保存变换后的特征...")
                dimension_reducer.save_transformed_features(
                    pca_features, umap_features, group_transformed_features,
                    train_labels, val_labels, test_labels,
                    output_path=os.path.join(output_dir, 'transformed_features.h5'))
                
                # 保存变换器模型
                dimension_reducer.save_transformers(os.path.join(output_dir, 'transformers'))
                
                # 生成降维报告
                dimension_reducer.generate_summary_report(output_path=os.path.join(output_dir, 'dimensionality_reduction_summary.md'))
                
                logger.info("降维与特征变换完成")
            else:
                logger.info("跳过降维与特征变换结果保存")
                
        except Exception as e:
            logger.error(f"降维与特征变换失败: {e}")
            logger_manager.log_experiment_end("阶段二：脑MRI数据特征工程", 
                                            {"状态": "部分完成", "阶段": "降维与特征变换", "错误": str(e)})


                                         
    else:
        logger.info("根据参数设置跳过PCA降维步骤")
        # 如果需要，也可以添加一个方法 save_selected_features 到 dimension_reducer 类
        # 也可以直接使用 feature_selector 的结果
        pca_features = None
        group_transformed_features = None
    
    # 生成汇总报告
    logger.info("生成特征工程汇总报告...")
    try:
        # 创建汇总报告目录
        summary_dir = os.path.join(output_dir, 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        # 生成全局汇总报告
        summary_path = os.path.join(summary_dir, 'feature_engineering_summary.md')
        with open(summary_path, 'w') as f:
            f.write("# 脑MRI数据特征工程汇总报告\n\n")
            f.write(f"## 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 特征选择方法: {feature_selection_method}\n\n")
            
            f.write("## 处理概况\n\n")
            f.write(f"- 原始特征数量: {train_features.shape[1]}\n")
            f.write(f"- 特征组数量: {len(group_features)}\n")
            f.write(f"- 样本数量: {train_features.shape[0]}\n")
            f.write(f"- 类别数量: {len(set(train_labels.flatten()))}\n\n")
            
            f.write("## 特征重要性分析\n\n")
            f.write("详细分析结果请查看 [特征重要性汇总报告](../importance_summary.md)\n\n")
            
            # 添加顶级特征列表
            if 'ranked_features' in locals():
                top_features = ranked_features[:10]
                f.write("### 全局顶级特征\n\n")
                f.write("| 特征 | 重要性得分 |\n")
                f.write("|------|------------|\n")
                for feature in top_features:
                    f.write(f"| {feature['Feature']} | {feature['Importance']:.4f} |\n")
                f.write("\n")
            
            f.write("## 特征选择\n\n")
            f.write("详细分析结果请查看 [特征选择汇总报告](../feature_selection_summary.md)\n\n")
            
            # 添加特征选择统计
            if 'all_selected_data' in locals():
                f.write("### 特征选择统计\n\n")
                f.write("| 特征集 | 原始特征数 | 选择特征数 | 保留比例 |\n")
                f.write("|--------|------------|------------|----------|\n")
                
                for group_name, selected_data in all_selected_data.items():
                    if group_name == 'all':
                        original_count = train_features.shape[1]
                    else:
                        original_count = group_features[group_name]['train'].shape[1]
                        
                    selected_count = len(selected_data['selected_indices'])
                    retention_ratio = selected_count / original_count * 100
                    
                    f.write(f"| {group_name} | {original_count} | {selected_count} | {retention_ratio:.1f}% |\n")
                f.write("\n")
            
            # 添加降维统计 (如果未跳过PCA)
            if not skip_pca and 'pca_features' in locals() and pca_features is not None:
                f.write("## 降维与特征变换\n\n")
                f.write("详细分析结果请查看 [降维汇总报告](../dimensionality_reduction_summary.md)\n\n")
                
                f.write("### PCA降维统计\n\n")
                f.write("| 特征集 | 原始维度 | 降维后维度 | 保留方差 |\n")
                f.write("|--------|----------|------------|----------|\n")
                
                for group_name, pca_data in pca_features.items():
                    if group_name == 'train':
                        original_dims = train_features.shape[1]
                        reduced_dims = pca_data.shape[1]
                        variance = dimension_reducer.pca_variance.get('all', 95)
                        f.write(f"| all | {original_dims} | {reduced_dims} | {variance:.1f}% |\n")
                
                for group_name, transformed in group_transformed_features.items():
                    if 'pca' in transformed:
                        pca_data = transformed['pca']
                        if 'train' in pca_data:
                            original_dims = group_features[group_name]['train'].shape[1]
                            reduced_dims = pca_data['train'].shape[1]
                            variance = dimension_reducer.pca_variance.get(group_name, 95)
                            f.write(f"| {group_name} | {original_dims} | {reduced_dims} | {variance:.1f}% |\n")
                f.write("\n")
            
            f.write("## 总结与建议\n\n")
            
            # 根据分析结果给出具体建议
            f.write("根据特征工程分析结果，提出以下建议：\n\n")
            
            # 1. 特征选择建议
            if 'all_selected_data' in locals():
                all_retention = len(all_selected_data['all']['selected_indices']) / train_features.shape[1] * 100
                if all_retention < 50:
                    f.write(f"1. **特征选择**: 使用{feature_selection_method}方法选择特征可显著减少维度(保留约{all_retention:.1f}%的特征)而不损失性能。\n\n")
                else:
                    f.write(f"1. **特征选择**: 使用{feature_selection_method}方法选择特征后仍保留{all_retention:.1f}%的特征，可能只能适度提高性能，但有助于模型解释性。\n\n")
            
            # 2. 降维建议 (如果未跳过PCA)
            if not skip_pca and 'pca_features' in locals() and pca_features is not None:
                pca_ratio = pca_features['train'].shape[1] / train_features.shape[1] * 100
                if pca_ratio < 30:
                    f.write("2. **降维应用**: PCA可将特征维度显著降低至原维度的{:.1f}%，建议在训练中使用，特别是对于深度学习模型。\n\n".format(pca_ratio))
                else:
                    f.write("2. **降维应用**: PCA降维效果中等(保留约{:.1f}%的维度)，在计算资源受限情况下可考虑使用。\n\n".format(pca_ratio))
            elif skip_pca:
                f.write("2. **降维策略**: 本次分析跳过了PCA降维，直接使用{feature_selection_method}方法选择的特征，适合保留非线性关系。\n\n")
            
            # 3. 特征组优先级
            if 'group_importance_results' in locals() and len(group_importance_results) > 1:
                f.write("3. **特征组优先级**: 根据特征重要性分析，建议按以下优先级使用特征组：\n")
                
                # 计算每个组的平均重要性
                group_avg_importance = {}
                for group_name, importance_data in group_importance_results.items():
                    ranked = importance_data['ranked']
                    if ranked:
                        avg_importance = sum(feature['Importance'] for feature in ranked) / len(ranked)
                        group_avg_importance[group_name] = avg_importance
                
                # 按重要性排序
                sorted_groups = sorted(group_avg_importance.items(), key=lambda x: x[1], reverse=True)
                for i, (group_name, avg_imp) in enumerate(sorted_groups):
                    f.write(f"   - {i+1}. {group_name} (平均重要性: {avg_imp:.4f})\n")
                f.write("\n")
            
            # 4. 模型训练建议
            f.write("4. **模型训练建议**: 建议在模型训练中采用以下策略：\n")
            f.write(f"   - 使用{feature_selection_method}方法选择并移除冗余的特征子集训练基线模型\n")
            f.write("   - 尝试各特征组单独训练的模型集成\n")
            if not skip_pca:
                f.write("   - 对于深度学习模型，可以先使用PCA降维的特征进行快速测试\n")
            else:
                f.write("   - 对于非线性关系，直接使用特征选择结果而不进行PCA可能保留更多信息\n")
        
        logger.info(f"特征工程汇总报告已保存至: {summary_path}")
        
    except Exception as e:
        logger.error(f"生成汇总报告失败: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"特征工程完成! 总耗时: {processing_time:.2f} 秒")
    
    # 记录实验结束
    results = {
        "状态": "成功",
        "处理时间(秒)": processing_time,
        "原始特征数": train_features.shape[1],
        "特征组数": len(group_features),
        "特征选择方法": feature_selection_method,
        "跳过PCA": skip_pca,
        "输出目录": output_dir
    }
    
    # 添加特征选择结果
    if 'all_selected_data' in locals():
        results["selected_features_count"] = len(all_selected_data['all']['selected_indices'])
        results["feature_retention_ratio"] = results["selected_features_count"] / results["原始特征数"] * 100
    
    # 添加降维结果 (如果未跳过PCA)
    if not skip_pca and 'pca_features' in locals() and pca_features is not None:
        results["pca_dimensions"] = pca_features['train'].shape[1]
        results["pca_dimension_ratio"] = results["pca_dimensions"] / results["原始特征数"] * 100
    
    logger_manager.log_experiment_end("阶段二：脑MRI数据特征工程", results)
    
    # 返回输出路径，方便后续脚本使用
    return_dict = {
        'feature_engineering_output_dir': output_dir,
        'timestamp': timestamp
    }
    
    if 'all_selected_data' in locals():
        selected_features_filename = f"{feature_selection_method}_selected_features.h5"
        return_dict['selected_features_path'] = os.path.join(output_dir, selected_features_filename)
    
    if not skip_pca and 'pca_features' in locals() and pca_features is not None:
        return_dict['transformed_features_path'] = os.path.join(output_dir, 'transformed_features.h5')
    
    return return_dict
    
if __name__ == "__main__":
    main()