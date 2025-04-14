#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI数据特性分析主程序
用于分析脑部MRI数据集的特征和分布特性，为神经网络选择提供依据
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
import argparse
from datetime import datetime

# 导入自定义模块
from data_loader import load_multiclass_data, preprocess_data, define_big_classes, map_to_big_classes
from basic_analysis import analyze_basic_stats, analyze_feature_correlation, analyze_class_separability
from feature_analysis import analyze_feature_importance, evaluate_feature_selection_methods, analyze_feature_combinations
from dim_reduction import perform_comprehensive_reduction, extract_pca_components

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mri_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MRI数据特性分析工具")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--subset', type=str, default='val', choices=['train', 'val', 'test', 'all'],
                      help='使用的数据子集 (default: val)')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='数据采样比例 (0.0-1.0)')
    parser.add_argument('--load_by_label', action='store_true', help='使用按标签分类方式加载数据')

    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='结果输出目录')
    
    # 分析控制参数
    parser.add_argument('--normalize', type=str, default='robust', choices=['robust', 'standard', 'none'],
                      help='数据标准化方法 (default: robust)')
    parser.add_argument('--gpu', action='store_true', help='使用GPU加速')
    parser.add_argument('--skip_basic', action='store_true', help='跳过基本分析')
    parser.add_argument('--skip_feature', action='store_true', help='跳过特征分析')
    parser.add_argument('--skip_dim_reduction', action='store_true', help='跳过降维分析')
    
    # 降维参数
    parser.add_argument('--dim_methods', type=str, default='pca,tsne,umap,isomap',
                      help='要使用的降维方法，逗号分隔 (default: pca,tsne,umap,isomap)')
    
    # 特征选择参数
    parser.add_argument('--feature_methods', type=str, default='rf,mi',
                      help='要使用的特征选择方法，逗号分隔 (default: rf,mi)')
    
    return parser.parse_args()


def perform_analysis(args):
    """执行分析函数，支持直接从参数调用"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始MRI数据特性分析 - {start_time}")
    
    # 加载数据
    if hasattr(args, 'load_brain_test_data') and args.load_brain_test_data:
        # 使用固定路径的脑MRI数据加载函数
        logger.info("使用固定路径加载脑MRI测试数据...")
        from data_loader import load_brain_voxel_test_data
        dataset = load_brain_voxel_test_data(check_normalization=True)
        
        if dataset is None:
            logger.error("数据加载失败")
            return
            
        # 使用test数据进行分析
        data = dataset['test_samples']
        labels = dataset['test_labels']
        feature_groups = dataset['feature_groups']
    else:
        # 使用原有的加载函数
        logger.info(f"加载数据目录: {args.data_dir}")
        logger.info(f"子集: {args.subset}")
        logger.info(f"采样比例: {args.sample_ratio}")
        
        dataset = load_multiclass_data(args.data_dir, subset=args.subset, sample_ratio=args.sample_ratio)
        
        if f"{args.subset}_samples" not in dataset or f"{args.subset}_labels" not in dataset:
            logger.error(f"未能加载指定的子集: {args.subset}")
            return
        
        data = dataset[f"{args.subset}_samples"]
        labels = dataset[f"{args.subset}_labels"]
        feature_groups = dataset['feature_groups']
    
    logger.info(f"加载的数据: {data.shape}")
    logger.info(f"标签: {labels.shape}")
    logger.info(f"特征组: {list(feature_groups.keys())}")
    
    # 获取大类标签（如果可用）
    try:
        fine_to_big, big_to_fine, big_class_names = define_big_classes()
        big_labels = map_to_big_classes(labels, fine_to_big)
    except Exception as e:
        logger.warning(f"无法创建大类标签映射: {str(e)}")
        big_labels = labels  # 使用原始标签
        big_class_names = None
    
    # 第2步：数据预处理（只有在需要时）
    if args.normalize != 'none':
        logger.info("\n======== 第2步：数据预处理 ========")
        processed_data, preprocessing_info = preprocess_data(
            data, labels, feature_groups=feature_groups, 
            normalize_method=args.normalize, use_gpu=args.gpu
        )
        
        logger.info(f"预处理完成")
        for group, group_data in processed_data.items():
            logger.info(f"  {group}: {group_data.shape}")
    else:
        logger.info("\n======== 第2步：跳过数据预处理（数据已标准化）========")
        # 创建一个简单的处理后数据字典，仅包含原始数据
        processed_data = {
            'all_features': data
        }

    # 第3步：基本统计分析
    if not args.skip_basic:
        logger.info("\n======== 第3步：基本统计分析 ========")
        
        # 创建基本分析输出目录
        basic_dir = os.path.join(args.output_dir, "basic_analysis")
        os.makedirs(basic_dir, exist_ok=True)
        
        # 基本统计量分析
        logger.info("计算基本统计量...")
        basic_stats = analyze_basic_stats(
            data, feature_groups=feature_groups, 
            use_gpu=args.gpu, save_dir=basic_dir
        )
        
        # 相关性分析
        logger.info("分析特征相关性...")
        corr_results = analyze_feature_correlation(
            data, feature_groups=feature_groups, 
            use_gpu=args.gpu, save_dir=basic_dir
        )
        
        # 类别可分性分析
        logger.info("分析类别可分性...")
        separability_results = analyze_class_separability(
            data, big_labels, feature_groups=feature_groups, 
            use_gpu=args.gpu, save_dir=basic_dir
        )
        
        # 保存分析摘要
        summary_path = os.path.join(basic_dir, "basic_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MRI数据基本分析摘要\n")
            f.write("======================\n\n")
            
            f.write(f"数据维度: {data.shape}\n")
            f.write(f"类别数量: {len(np.unique(labels))}\n")
            f.write(f"大类数量: {len(np.unique(big_labels))}\n\n")
            
            f.write("特征组统计:\n")
            for group, indices in feature_groups.items():
                f.write(f"  {group}: {len(indices)}个特征\n")
            
            f.write("\n特征组相关性:\n")
            for group, result in corr_results.items():
                if group == 'between_groups':
                    continue
                f.write(f"  {group} 内部平均相关性: {result.get('mean_abs_corr', 'N/A'):.4f}\n")
            
            if 'between_groups' in corr_results:
                f.write("\n特征组间相关性:\n")
                group_names = corr_results['between_groups']['group_names']
                corr_matrix = corr_results['between_groups']['corr_matrix']
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        f.write(f"  {group_names[i]} - {group_names[j]}: {corr_matrix[i, j]:.4f}\n")
            
            f.write("\n类别可分性:\n")
            for group, result in separability_results.items():
                f.write(f"  {group} 显著特征比例: {result.get('significant_ratio', 'N/A'):.4f}\n")
                f.write(f"  {group} 平均F值: {result.get('mean_f', 'N/A'):.4f}\n")
        
        logger.info(f"基本分析摘要保存至: {summary_path}")
    
    # 第4步：特征分析
    if not args.skip_feature:
        logger.info("\n======== 第4步：特征重要性和特征选择分析 ========")
        
        # 创建特征分析输出目录
        feature_dir = os.path.join(args.output_dir, "feature_analysis")
        os.makedirs(feature_dir, exist_ok=True)
        
        # 特征重要性分析
        logger.info("分析特征重要性...")
        importance_results = analyze_feature_importance(
            data, big_labels, feature_groups=feature_groups,
            use_gpu=args.gpu, save_dir=feature_dir
        )
        
        # 特征选择方法评估
        if args.feature_methods:
            methods = args.feature_methods.split(',')
            logger.info(f"评估特征选择方法: {methods}...")
            selection_results = evaluate_feature_selection_methods(
                data, big_labels, feature_groups=feature_groups,
                methods=methods, use_gpu=args.gpu, save_dir=feature_dir
            )
        
        # 特征组合分析
        logger.info("分析特征组合效果...")
        combination_results = analyze_feature_combinations(
            data, big_labels, feature_groups=feature_groups,
            top_k=50, use_gpu=args.gpu, save_dir=feature_dir
        )
        
        # 保存特征分析摘要
        summary_path = os.path.join(feature_dir, "feature_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MRI数据特征分析摘要\n")
            f.write("======================\n\n")
            
            f.write("特征重要性分析:\n")
            for group, result in importance_results.items():
                f.write(f"  {group}:\n")
                f.write(f"    覆盖90%重要性需要的特征数: {result.get('n_features_90', 'N/A')} ({result.get('n_features_90', 0)/len(feature_groups[group])*100:.1f}%)\n")
                f.write(f"    覆盖95%重要性需要的特征数: {result.get('n_features_95', 'N/A')} ({result.get('n_features_95', 0)/len(feature_groups[group])*100:.1f}%)\n")
            
            if 'combination_results' in locals():
                f.write("\n最佳特征组合:\n")
                sorted_combos = sorted(combination_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                for i, (combo_name, result) in enumerate(sorted_combos[:3]):
                    f.write(f"  {i+1}. {combo_name}: 准确率 {result['accuracy']:.4f}, 特征数 {result['n_features']}\n")
        
        logger.info(f"特征分析摘要保存至: {summary_path}")
    
    # 第5步：降维分析
    if not args.skip_dim_reduction:
        logger.info("\n======== 第5步：降维分析 ========")
        
        # 创建降维分析输出目录
        dim_dir = os.path.join(args.output_dir, "dim_reduction")
        os.makedirs(dim_dir, exist_ok=True)
        
        # 使用的降维方法
        dim_methods = args.dim_methods.split(',')
        logger.info(f"使用的降维方法: {dim_methods}")
        
        # 执行降维分析
        logger.info("执行降维分析...")
        reduction_results = perform_comprehensive_reduction(
            data, big_labels, feature_groups=feature_groups,
            methods=dim_methods, use_gpu=args.gpu, 
            class_names=big_class_names,
            sample_ratio=min(0.5, args.sample_ratio),  # 限制样本数，避免计算过长
            save_dir=dim_dir
        )
        
        # 提取PCA主成分
        logger.info("提取PCA主成分...")
        pca_results = extract_pca_components(
            data, feature_groups=feature_groups,
            n_components=20, use_gpu=args.gpu, save_dir=dim_dir
        )
        
        # 保存降维分析摘要
        summary_path = os.path.join(dim_dir, "dim_reduction_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MRI数据降维分析摘要\n")
            f.write("======================\n\n")
            
            f.write("PCA分析:\n")
            for group, result in pca_results.items():
                f.write(f"  {group}:\n")
                f.write(f"    第一主成分解释方差: {result['explained_variance_ratio'][0]:.4f}\n")
                f.write(f"    前5个主成分累积解释方差: {np.sum(result['explained_variance_ratio'][:5]):.4f}\n")
                f.write(f"    前10个主成分累积解释方差: {np.sum(result['explained_variance_ratio'][:10]):.4f}\n")
                if len(result['explained_variance_ratio']) >= 20:
                    f.write(f"    前20个主成分累积解释方差: {np.sum(result['explained_variance_ratio'][:20]):.4f}\n")
        
        logger.info(f"降维分析摘要保存至: {summary_path}")
    
    # 第6步：综合分析和网络建议
    logger.info("\n======== 第6步：综合分析和网络建议 ========")
    
    # 创建综合分析输出目录
    summary_dir = os.path.join(args.output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 整合所有分析结果，提出网络架构建议
    summary_path = os.path.join(summary_dir, "network_recommendations.txt")
    with open(summary_path, 'w') as f:
        f.write("MRI数据分析综合报告和网络架构建议\n")
        f.write("==============================\n\n")
        
        f.write("数据概况:\n")
        f.write(f"  样本数量: {data.shape[0]}\n")
        f.write(f"  特征维度: {data.shape[1]}\n")
        f.write(f"  类别数量: {len(np.unique(labels))}\n")
        f.write(f"  大类数量: {len(np.unique(big_labels))}\n\n")
        
        # 类别分布特点
        unique_big, counts_big = np.unique(big_labels, return_counts=True)
        class_ratio = max(counts_big) / min(counts_big)
        f.write("类别分布特点:\n")
        f.write(f"  大类不平衡比例: {class_ratio:.2f}\n")
        f.write(f"  最小类样本数: {min(counts_big)}\n")
        f.write(f"  最大类样本数: {max(counts_big)}\n\n")
        
        # 特征相关性和冗余性
        if 'corr_results' in locals():
            mean_corrs = [result.get('mean_abs_corr', 0) for group, result in corr_results.items() 
                        if group != 'between_groups']
            avg_corr = np.mean(mean_corrs) if mean_corrs else 0
            f.write("特征相关性:\n")
            f.write(f"  平均特征相关性: {avg_corr:.4f}\n")
            f.write(f"  特征相关程度: {'高' if avg_corr > 0.5 else '中等' if avg_corr > 0.3 else '低'}\n\n")
        
        # 特征重要性分布
        if 'importance_results' in locals():
            avg_n90_ratio = np.mean([result.get('n_features_90', 0) / len(feature_groups[group]) 
                                  for group, result in importance_results.items()])
            f.write("特征重要性分布:\n")
            f.write(f"  平均覆盖90%重要性所需特征比例: {avg_n90_ratio:.2f}\n")
            f.write(f"  特征重要性集中程度: {'高' if avg_n90_ratio < 0.3 else '中等' if avg_n90_ratio < 0.6 else '低'}\n\n")
        
        # 数据降维特性
        if 'pca_results' in locals():
            avg_pc1 = np.mean([result['explained_variance_ratio'][0] for group, result in pca_results.items()])
            avg_pc10 = np.mean([np.sum(result['explained_variance_ratio'][:10]) for group, result in pca_results.items() 
                              if len(result['explained_variance_ratio']) >= 10])
            f.write("数据降维特性:\n")
            f.write(f"  平均第一主成分解释方差: {avg_pc1:.4f}\n")
            f.write(f"  平均前10个主成分累积解释方差: {avg_pc10:.4f}\n")
            f.write(f"  数据内在维度估计: {'低' if avg_pc10 > 0.95 else '中等' if avg_pc10 > 0.8 else '高'}\n\n")
        
        # 最佳特征组合
        if 'combination_results' in locals():
            best_combo = max(combination_results.items(), key=lambda x: x[1]['accuracy'])
            f.write("最佳特征组合:\n")
            f.write(f"  组合: {best_combo[0]}\n")
            f.write(f"  准确率: {best_combo[1]['accuracy']:.4f}\n")
            f.write(f"  特征数: {best_combo[1]['n_features']}\n\n")
        
        # 网络架构建议
        f.write("网络架构建议:\n")
        
        # 1. 判断是否需要特征选择/降维
        need_dim_reduction = False
        if 'pca_results' in locals():
            if avg_pc10 > 0.9:  # 如果10个主成分解释90%以上方差
                need_dim_reduction = True
        
        need_feature_selection = False
        if 'importance_results' in locals():
            if avg_n90_ratio < 0.5:  # 如果50%的特征就能解释90%的重要性
                need_feature_selection = True
        
        high_correlation = False
        if 'corr_results' in locals():
            if avg_corr > 0.4:  # 如果平均相关性较高
                high_correlation = True
        
        # 2. 推荐网络类型
        if need_dim_reduction or need_feature_selection or high_correlation:
            f.write("  1. 整体建议: 数据存在较高冗余，建议在网络中加入降维或特征选择机制\n")
            
            if need_dim_reduction and need_feature_selection:
                f.write("  2. 预处理: 建议使用PCA或自编码器进行降维，或使用特征选择减少输入维度\n")
            elif need_dim_reduction:
                f.write("  2. 预处理: 建议使用PCA降维，将特征维度降至原来的10%-20%\n")
            elif need_feature_selection:
                f.write("  2. 预处理: 建议使用基于特征重要性的选择，仅保留最重要的30%-50%特征\n")
            
            if high_correlation:
                f.write("  3. 正则化: 由于特征间高度相关，建议使用较强的L1或L2正则化\n")
            
            f.write("  4. 推荐网络架构:\n")
            f.write("     A. 自编码器 + 分类器: 先用自编码器学习低维表示，再接分类网络\n")
            f.write("     B. KAN (Kolmogorov-Arnold Network): 适合处理高维冗余数据\n")
            f.write("     C. 含注意力机制的MLP: 帮助模型关注重要特征\n")
            f.write("     D. 树集成 + 神经网络混合模型: 结合树模型的特征选择能力和神经网络的表达能力\n")
        else:
            f.write("  1. 整体建议: 数据维度适中，特征独立性较好，可使用常规神经网络架构\n")
            f.write("  2. 推荐网络架构:\n")
            f.write("     A. 多层感知机 (MLP): 简单有效，适合处理此类表格数据\n")
            f.write("     B. TabNet: 专为表格数据设计，具有特征选择能力\n")
            f.write("     C. ResNet风格的MLP: 添加残差连接以改善深层网络训练\n")
        
        # 3. 网络规模建议
        if data.shape[1] > 200:
            f.write("  5. 网络规模:\n")
            f.write("     - 输入层: 原始特征数 (可选PCA降维后)\n")
            f.write("     - 隐藏层: 建议4-6层，每层单元数从4096逐渐减少到1024\n")
            f.write("     - 添加Dropout (0.3-0.5)和BatchNorm以防止过拟合\n")
        else:
            f.write("  5. 网络规模:\n")
            f.write("     - 输入层: 原始特征数\n")
            f.write("     - 隐藏层: 建议3-4层，每层单元数从2048逐渐减少到512\n")
            f.write("     - 添加适度的Dropout (0.2-0.4)以防止过拟合\n")
        
        # 4. 类别不平衡处理
        if class_ratio > 10:
            f.write("  6. 类别不平衡处理:\n")
            f.write("     - 使用加权损失函数，权重与类别频率成反比\n")
            f.write("     - 考虑使用数据增强或过采样技术\n")
            f.write("     - 或采用分层训练策略，先在平衡子集上训练，再在全数据上微调\n")
        
        # 5. 特定于MRI数据的建议
        f.write("\n  7. 特定于MRI数据的建议:\n")
        if 'diffusion' in feature_groups and 'qti' in feature_groups and 'cest' in feature_groups:
            f.write("     - 考虑为不同的MRI模态（扩散、QTI、CEST）设计单独的编码器分支\n")
            f.write("     - 使用自注意力机制融合不同模态的特征\n")
            f.write("     - DTI特征与QTI特征可能有高度冗余，考虑使用特征选择分别处理\n")
    
    logger.info(f"综合分析和网络建议已保存至: {summary_path}")
    
    # 记录结束时间
    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"\n分析完成 - 总耗时: {elapsed.total_seconds()/60:.2f}分钟")



def main():
    """主执行函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 执行分析
    perform_analysis(args)
    
if __name__ == "__main__":
    main()