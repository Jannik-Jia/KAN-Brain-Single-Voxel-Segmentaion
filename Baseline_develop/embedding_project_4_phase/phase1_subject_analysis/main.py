#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: 受试者差异分析模块主程序
负责分析受试者间的特征差异和脑区特异性
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import Config
from common.data_io import DataIO
from common.visualization import Visualizer
from phase1_subject_analysis.global_analyzer import GlobalSubjectAnalyzer
from phase1_subject_analysis.region_analyzer import RegionSpecificityAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Phase 1: 受试者差异分析')
    
    parser.add_argument('--input_dir', type=str,
                       default=str(Config.PHASE0_OUTPUT_DIR),
                       help='Phase 0输出目录')
    
    parser.add_argument('--output_dir', type=str,
                       default=str(Config.PHASE1_OUTPUT_DIR),
                       help='输出目录')
    
    parser.add_argument('--skip_visualization', action='store_true',
                       help='跳过可视化生成')
    
    parser.add_argument('--min_samples_per_region', type=int,
                       default=Config.MIN_SAMPLES_FOR_SUBJECT_REGION,
                       help='每个受试者-脑区组合的最小样本数')
    
    parser.add_argument('--high_variation_percentile', type=int,
                       default=90,
                       help='高变异特征的百分位数阈值')
    
    parser.add_argument('--low_variation_percentile', type=int,
                       default=10,
                       help='低变异特征的百分位数阈值')
    
    return parser.parse_args()


def load_phase0_data(input_dir: Path) -> dict:
    """加载Phase 0的输出数据"""
    logger.info("加载Phase 0数据...")
    
    # 加载训练数据
    train_data = DataIO.load_numpy_data(input_dir / Config.TRAIN_DATA_FILE)
    val_data = DataIO.load_numpy_data(input_dir / Config.VAL_DATA_FILE)
    
    # 加载元数据
    data_stats = DataIO.load_json(input_dir / Config.DATA_STATS_FILE)
    label_mapping = DataIO.load_json(input_dir / Config.LABEL_MAPPING_FILE)
    
    # 加载Phase 0结果
    phase0_results = DataIO.load_phase_output(0, input_dir)
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'data_stats': data_stats,
        'label_mapping': label_mapping,
        'phase0_results': phase0_results
    }


def main():
    """主函数"""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("Phase 1: 受试者差异分析开始")
    logger.info("="*80)
    
    # 设置路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化目录
    viz_dir = output_dir / 'visualizations'
    if not args.skip_visualization:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载Phase 0数据
        phase0_data = load_phase0_data(input_dir)
        
        # 合并训练和验证数据用于分析
        X_all = np.vstack([phase0_data['train_data']['X'], 
                          phase0_data['val_data']['X']])
        y_all = np.vstack([phase0_data['train_data']['y'],
                          phase0_data['val_data']['y']])
        subjects_all = np.concatenate([phase0_data['train_data']['subjects'],
                                      phase0_data['val_data']['subjects']])
        
        logger.info(f"合并数据形状: {X_all.shape}")
        logger.info(f"受试者数量: {len(np.unique(subjects_all))}")
        
        # 步骤1: 全局受试者分析
        logger.info("\n步骤1: 执行全局受试者差异分析...")
        global_analyzer = GlobalSubjectAnalyzer(
            high_variation_percentile=args.high_variation_percentile,
            low_variation_percentile=args.low_variation_percentile
        )
        
        global_results = global_analyzer.analyze(X_all, subjects_all)
        
        # 保存全局分析结果
        DataIO.save_numpy_data({
            'subject_means': global_results['subject_stats']['means'],
            'subject_stds': global_results['subject_stats']['stds'],
            'subject_skews': global_results['subject_stats']['skews'],
            'subject_kurts': global_results['subject_stats']['kurts'],
            'sample_counts': global_results['subject_stats']['sample_counts'],
            'subject_ids': global_results['subject_stats']['subject_ids']
        }, output_dir / 'subject_statistics.npz')
        
        DataIO.save_numpy_data({
            'distance_matrix': global_results['similarity']['distance_matrix'],
            'correlation_matrix': global_results['similarity']['correlation_matrix'],
            'linkage_matrix': global_results['similarity']['linkage_matrix']
        }, output_dir / 'similarity_matrices.npz')
        
        DataIO.save_numpy_data({
            'f_stats': global_results['feature_variation']['f_stats'],
            'high_variation_features': global_results['feature_variation']['high_variation_features'],
            'low_variation_features': global_results['feature_variation']['low_variation_features'],
            'pca_result': global_results['feature_variation']['pca_result'],
            'pca_explained_variance': global_results['feature_variation']['pca_explained_variance']
        }, output_dir / 'feature_variation.npz')
        
        # 步骤2: 脑区特异性分析
        logger.info("\n步骤2: 执行脑区特异性分析...")
        
        # 处理标签
        if phase0_data['label_mapping']['is_one_hot']:
            y_regions = np.argmax(y_all, axis=1)
        else:
            y_regions = y_all.flatten()
        
        region_analyzer = RegionSpecificityAnalyzer(
            min_samples_per_combination=args.min_samples_per_region
        )
        
        region_results = region_analyzer.analyze(X_all, y_regions, subjects_all)
        
        # 保存脑区分析结果
        DataIO.save_json(region_results['region_specificity_scores'], 
                        output_dir / 'region_specificity.json')
        
        # 步骤3: 生成可视化
        if not args.skip_visualization:
            logger.info("\n步骤3: 生成可视化图表...")
            visualizer = Visualizer(save_dir=viz_dir)
            
            # 受试者聚类图
            if 'pca_result' in global_results['feature_variation']:
                visualizer.plot_scatter_2d(
                    global_results['feature_variation']['pca_result'][:, :2],
                    y=global_results['subject_stats']['subject_ids'],
                    title="Subject PCA Projection",      # 原: "受试者PCA投影"
                    xlabel="PC1", ylabel="PC2",
                    save_name="subject_clustering.png"
                )
            
            # 受试者相似性热图
            visualizer.plot_correlation_matrix(
                global_results['similarity']['correlation_matrix'],
                title="Inter-subject Correlation Matrix",   # 原: "受试者间相关性矩阵"
                labels=[f"S{int(sid)}" for sid in global_results['subject_stats']['subject_ids']],
                save_name="subject_similarity_heatmap.png"
            )


            
            # 特征变异分布
            visualizer.plot_distribution(
                global_results['feature_variation']['f_stats'],
                title="Feature Variation Distribution",     # 原: "特征变异分布"
                xlabel="Standardized F-statistic",          # 原: "标准化F统计量"
                ylabel="Number of Features",                # 原: "特征数量"
                save_name="variation_distribution.png"
            )
            
            # 脑区特异性热图
            if region_results['specificity_matrix'] is not None:
                # 可视化前20个脑区
                top_regions = sorted(region_results['region_specificity_scores'].keys(),
                                   key=lambda x: region_results['region_specificity_scores'][x]['specificity_score'],
                                   reverse=True)[:20]
                
                matrix_subset = []
                for region in top_regions:
                    if region in region_results['specificity_matrix']:
                        matrix_subset.append(region_results['specificity_matrix'][region])
                
                if matrix_subset:
                    visualizer.plot_correlation_matrix(
                        np.array(matrix_subset),
                        title="Region Specificity Score Heatmap (Top 20)",    # 原: "脑区特异性得分热图 (Top 20)"
                        labels=[f"R{r}" for r in top_regions],
                        save_name="region_specificity_heatmap.png"
                    )

        
        # 步骤4: 计算Phase 1决策得分
        logger.info("\n步骤4: 计算决策得分...")
        phase1_scores = compute_phase1_scores(global_results, region_results)
        
        DataIO.save_json(phase1_scores, output_dir / 'phase1_scores.json')
        
        # 保存Phase 1完整结果
        phase1_results = {
            'global_analysis': {
                'n_subjects': len(global_results['subject_stats']['subject_ids']),
                'mean_inter_subject_distance': global_results['similarity']['mean_distance'],
                'mean_inter_subject_correlation': global_results['similarity']['mean_correlation'],
                'feature_variation_summary': {
                    'n_high_variation': len(global_results['feature_variation']['high_variation_features']),
                    'n_low_variation': len(global_results['feature_variation']['low_variation_features']),
                    'pca_3pc_variance': float(np.sum(global_results['feature_variation']['pca_explained_variance'][:3]))
                }
            },
            'region_analysis': {
                'n_regions_analyzed': region_results['n_regions_analyzed'],
                'n_valid_combinations': region_results['n_valid_combinations'],
                'mean_specificity_score': region_results['mean_specificity_score'],
                'high_specificity_regions': region_results['high_specificity_regions'],
                'low_specificity_regions': region_results['low_specificity_regions']
            }
        }
        
        DataIO.save_phase_output(
            phase_number=1,
            output_dir=output_dir,
            results=phase1_results,
            scores=phase1_scores,
            metadata={
                'n_subjects': len(np.unique(subjects_all)),
                'n_samples': len(X_all),
                'n_features': X_all.shape[1],
                'execution_time': datetime.now().isoformat()
            }
        )
        
        logger.info("\n" + "="*80)
        logger.info("Phase 1: 受试者差异分析完成!")
        logger.info("="*80)
        logger.info(f"结果已保存到: {output_dir}")
        
        # 打印关键发现
        logger.info("\n关键发现:")
        logger.info(f"- 受试者间平均相关性: {global_results['similarity']['mean_correlation']:.3f}")
        logger.info(f"- 前3个主成分解释方差: {np.sum(global_results['feature_variation']['pca_explained_variance'][:3])*100:.1f}%")
        logger.info(f"- 高特异性脑区数量: {len(region_results['high_specificity_regions'])}")
        logger.info(f"- 低特异性脑区数量: {len(region_results['low_specificity_regions'])}")
        logger.info(f"- Subject Embedding需求评分: {phase1_scores['embedding_necessity_score']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Phase 1 执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


def compute_phase1_scores(global_results, region_results):
    """计算Phase 1的决策得分"""
    
    # 差异显著性得分
    pca_3pc_variance = np.sum(global_results['feature_variation']['pca_explained_variance'][:3])
    if pca_3pc_variance > 0.6:
        difference_significance = 1.0
    elif pca_3pc_variance > 0.4:
        difference_significance = 0.8
    else:
        difference_significance = 0.5
    
    # 模式线性度
    pattern_linearity = pca_3pc_variance
    
    # 受试者相似性
    subject_similarity = abs(global_results['similarity']['mean_correlation'])
    
    # 特征异质性
    n_features = len(global_results['feature_variation']['f_stats'])
    n_high_variation = len(global_results['feature_variation']['high_variation_features'])
    feature_heterogeneity = n_high_variation / n_features
    
    # 脑区特异性多样性
    if region_results['specificity_scores']:
        specificity_values = [s['specificity_score'] for s in region_results['region_specificity_scores'].values()]
        region_specificity_diversity = np.std(specificity_values) / (np.mean(specificity_values) + 1e-8)
    else:
        region_specificity_diversity = 0.0
    
    # 高特异性脑区比例
    n_high_specificity = len(region_results['high_specificity_regions'])
    n_total_regions = region_results['n_regions_analyzed']
    high_specificity_ratio = n_high_specificity / n_total_regions if n_total_regions > 0 else 0.0
    
    # 综合embedding需求评分
    embedding_necessity_score = (
        difference_significance * 0.2 +
        (1 - pattern_linearity) * 0.2 +  # 非线性程度
        (1 - subject_similarity) * 0.2 +  # 低相似性
        feature_heterogeneity * 0.1 +
        region_specificity_diversity * 0.15 +
        high_specificity_ratio * 0.15
    )
    
    return {
        'difference_significance': float(difference_significance),
        'pattern_linearity': float(pattern_linearity),
        'subject_similarity': float(subject_similarity),
        'feature_heterogeneity': float(feature_heterogeneity),
        'region_specificity_diversity': float(region_specificity_diversity),
        'high_specificity_ratio': float(high_specificity_ratio),
        'embedding_necessity_score': float(embedding_necessity_score),
        'analysis_coverage': 1.0  # Phase 1完成
    }


if __name__ == "__main__":
    sys.exit(main())