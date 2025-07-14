#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: 受试者差异分析模块主程序 (增强版)
包含多标签分析和条件UMAP可视化
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import Config
from common.data_io import DataIO
from common.visualization import Visualizer
from phase1_subject_analysis.global_analyzer import GlobalSubjectAnalyzer
from phase1_subject_analysis.region_analyzer import RegionSpecificityAnalyzer
from phase1_subject_analysis.multilabel_patient_analyzer import MultiLabelPatientAnalyzer
from phase1_subject_analysis.conditional_umap_visualizer import ConditionalUMAPVisualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略一些警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Phase 1: 受试者差异分析 (增强版)')
    
    parser.add_argument('--input_dir', type=str,
                       default=str(Config.PHASE0_OUTPUT_DIR),
                       help='Phase 0输出目录')
    
    parser.add_argument('--output_dir', type=str,
                       default=str(Config.PHASE1_OUTPUT_DIR),
                       help='输出目录')
    
    parser.add_argument('--skip_visualization', action='store_true',
                       help='跳过可视化生成')
    
    parser.add_argument('--skip_umap', action='store_true',
                       help='跳过UMAP可视化（节省时间）')
    
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='使用GPU加速')
    
    parser.add_argument('--min_samples_per_region', type=int,
                       default=Config.MIN_SAMPLES_FOR_SUBJECT_REGION,
                       help='每个受试者-脑区组合的最小样本数')
    
    parser.add_argument('--high_variation_percentile', type=int,
                       default=90,
                       help='高变异特征的百分位数阈值')
    
    parser.add_argument('--low_variation_percentile', type=int,
                       default=10,
                       help='低变异特征的百分位数阈值')
    
    parser.add_argument('--n_umap_neighbors', type=int, default=15,
                       help='UMAP的n_neighbors参数')
    
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                       help='UMAP的min_dist参数')
    
    return parser.parse_args()


def load_phase0_data(input_dir: Path) -> dict:
    """加载Phase 0的输出数据"""
    logger.info("加载Phase 0数据...")
    
    # 加载训练数据
    train_data = DataIO.load_numpy_data(input_dir / Config.TRAIN_DATA_FILE)
    val_data = DataIO.load_numpy_data(input_dir / Config.VAL_DATA_FILE)
    
    # 转换受试者ID为Python原生int类型
    train_data['subjects'] = train_data['subjects'].astype(int)
    val_data['subjects'] = val_data['subjects'].astype(int)
    
    # 如果标签也需要转换
    if 'y' in train_data and len(train_data['y'].shape) == 1:
        train_data['y'] = train_data['y'].astype(int)
    if 'y' in val_data and len(val_data['y'].shape) == 1:
        val_data['y'] = val_data['y'].astype(int)
    
    # 加载元数据
    data_stats = DataIO.load_json(input_dir / Config.DATA_STATS_FILE)
    label_mapping = DataIO.load_json(input_dir / Config.LABEL_MAPPING_FILE)
    
    # 加载Phase 0结果
    phase0_results = DataIO.load_phase_output(0, input_dir)
    
    # 加载增强的分析结果
    brain_region_analysis = DataIO.load_json(input_dir / 'brain_region_analysis.json')
    feature_group_analysis = DataIO.load_json(input_dir / 'feature_group_analysis.json')
    phase1_quick_index = DataIO.load_json(input_dir / 'phase1_quick_index.json')
    validation_report = DataIO.load_json(input_dir / 'validation_report.json')
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'data_stats': data_stats,
        'label_mapping': label_mapping,
        'phase0_results': phase0_results,
        'brain_region_analysis': brain_region_analysis,
        'feature_group_analysis': feature_group_analysis,
        'phase1_quick_index': phase1_quick_index,
        'validation_report': validation_report
    }


def main():
    """主函数"""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("Phase 1: 受试者差异分析开始 (增强版)")
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
        logger.info(f"脑区数量: {len(np.unique(np.argmax(y_all, axis=1) if len(y_all.shape) > 1 else y_all))}")
        
        # 处理标签
        if phase0_data['label_mapping']['is_one_hot']:
            y_regions = np.argmax(y_all, axis=1)
        else:
            y_regions = y_all.flatten()
        
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
        region_analyzer = RegionSpecificityAnalyzer(
            min_samples_per_combination=args.min_samples_per_region,
            brain_region_analysis=phase0_data.get('brain_region_analysis')
        )
        
        region_results = region_analyzer.analyze(X_all, y_regions, subjects_all)
        
        # 保存脑区分析结果
        DataIO.save_json(region_results['region_specificity_scores'], 
                        output_dir / 'region_specificity.json')
        
        # 步骤2.5: 多标签患者差异分析（新增）
        logger.info("\n步骤2.5: 执行多标签患者差异分析...")
        ml_analyzer = MultiLabelPatientAnalyzer(use_gpu=args.use_gpu)
        ml_results = ml_analyzer.analyze(X_all, y_regions, subjects_all)
        
        # 保存多标签分析结果
        DataIO.save_numpy_data({
            'variance_components': {
                'patient_variance_ratio': ml_results['variance_decomposition']['patient_variance_ratio'],
                'region_variance_ratio': ml_results['variance_decomposition']['region_variance_ratio'],
                'interaction_variance_ratio': ml_results['variance_decomposition']['interaction_variance_ratio'],
                'error_variance_ratio': ml_results['variance_decomposition']['error_variance_ratio']
            },
            'silhouette_scores': {
                'global': ml_results['silhouette_analysis']['global_silhouette'],
                'conditional': ml_results['silhouette_analysis']['conditional_silhouette']
            },
            'patient_distances': {
                'inter_mean': ml_results['distance_analysis']['inter_patient_distance_mean'],
                'intra_mean': ml_results['distance_analysis']['intra_patient_distance_mean'],
                'separability_index': ml_results['distance_analysis']['separability_index']
            }
        }, output_dir / 'multilabel_analysis.npz')
        
        # 保存完整的多标签分析结果
        DataIO.save_json(ml_results, output_dir / 'multilabel_analysis_full.json')
        
        # 步骤3: 生成基础可视化
        if not args.skip_visualization:
            logger.info("\n步骤3: 生成基础可视化图表...")
            visualizer = Visualizer(save_dir=viz_dir)
            
            # 受试者聚类图
            if 'pca_result' in global_results['feature_variation']:
                visualizer.plot_scatter_2d(
                    global_results['feature_variation']['pca_result'][:, :2],
                    y=global_results['subject_stats']['subject_ids'],
                    title="Subject PCA Projection",
                    xlabel="PC1", ylabel="PC2",
                    save_name="subject_clustering.png"
                )
            
            # 受试者相似性热图
            visualizer.plot_correlation_matrix(
                global_results['similarity']['correlation_matrix'],
                title="Inter-subject Correlation Matrix",
                labels=[f"S{int(sid)}" for sid in global_results['subject_stats']['subject_ids']],
                save_name="subject_similarity_heatmap.png"
            )
            
            # 特征变异分布
            visualizer.plot_distribution(
                global_results['feature_variation']['f_stats'],
                title="Feature Variation Distribution",
                xlabel="Standardized F-statistic",
                ylabel="Number of Features",
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
                        title="Region Specificity Score Heatmap (Top 20)",
                        labels=[f"R{r}" for r in top_regions],
                        save_name="region_specificity_heatmap.png"
                    )
        
        # 步骤3.5: 条件UMAP可视化（新增）
        if not args.skip_visualization and not args.skip_umap:
            logger.info("\n步骤3.5: 生成条件UMAP可视化...")
            umap_viz = ConditionalUMAPVisualizer(
                output_dir=viz_dir,
                use_gpu=args.use_gpu,
                n_neighbors=args.n_umap_neighbors,
                min_dist=args.umap_min_dist
            )
            
            # 生成全部可视化
            umap_results = umap_viz.generate_all_visualizations(
                X_all, subjects_all, y_regions, ml_results
            )
            
            # 保存UMAP结果摘要
            DataIO.save_json({
                'global_patient_clustering': umap_results['global_umap']['patient_clustering_score'],
                'mean_region_clustering': umap_results['region_umaps']['mean_clustering_score'],
                'high_variance_regions': umap_results['region_umaps']['high_variance_regions'],
                'low_variance_regions': umap_results['region_umaps']['low_variance_regions'],
                'conditional_improvement': umap_results['conditional_umap']['clustering_improvement']
            }, output_dir / 'umap_results_summary.json')
        
        # 步骤4: 计算增强的Phase 1决策得分
        logger.info("\n步骤4: 计算增强的决策得分...")
        phase1_scores = compute_enhanced_phase1_scores(
            global_results, region_results, ml_results,
            umap_results if 'umap_results' in locals() else None
        )
        
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
            },
            'multilabel_analysis': {
                'patient_variance_ratio': ml_results['variance_decomposition']['patient_variance_ratio'],
                'region_variance_ratio': ml_results['variance_decomposition']['region_variance_ratio'],
                'interaction_variance_ratio': ml_results['variance_decomposition']['interaction_variance_ratio'],
                'global_silhouette': ml_results['silhouette_analysis']['global_silhouette'],
                'conditional_silhouette': ml_results['silhouette_analysis']['conditional_silhouette'],
                'separability_index': ml_results['distance_analysis']['separability_index']
            }
        }
        
        if 'umap_results' in locals():
            phase1_results['umap_analysis'] = {
                'global_clustering': umap_results['global_umap']['patient_clustering_score'],
                'mean_region_clustering': umap_results['region_umaps']['mean_clustering_score'],
                'conditional_improvement': umap_results['conditional_umap']['clustering_improvement']
            }
        
        DataIO.save_phase_output(
            phase_number=1,
            output_dir=output_dir,
            results=phase1_results,
            scores=phase1_scores['scores'],
            metadata={
                'n_subjects': len(np.unique(subjects_all)),
                'n_samples': len(X_all),
                'n_features': X_all.shape[1],
                'n_regions': len(np.unique(y_regions)),
                'execution_time': datetime.now().isoformat(),
                'gpu_used': args.use_gpu,
                'umap_generated': not args.skip_umap
            }
        )
        
        logger.info("\n" + "="*80)
        logger.info("Phase 1: 受试者差异分析完成!")
        logger.info("="*80)
        logger.info(f"结果已保存到: {output_dir}")
        
        # 打印关键发现
        logger.info("\n关键发现:")
        logger.info(f"- 患者方差占比: {ml_results['variance_decomposition']['patient_variance_ratio']:.1%}")
        logger.info(f"- 脑区方差占比: {ml_results['variance_decomposition']['region_variance_ratio']:.1%}")
        logger.info(f"- 交互效应占比: {ml_results['variance_decomposition']['interaction_variance_ratio']:.1%}")
        logger.info(f"- 全局轮廓系数: {ml_results['silhouette_analysis']['global_silhouette']:.3f}")
        logger.info(f"- 条件轮廓系数: {ml_results['silhouette_analysis']['conditional_silhouette']:.3f}")
        logger.info(f"- 患者可分离性指数: {ml_results['distance_analysis']['separability_index']:.3f}")
        logger.info(f"- 高特异性脑区数量: {len(region_results['high_specificity_regions'])}")
        logger.info(f"\n📌 最终建议: {phase1_scores['recommendation']}")
        logger.info(f"📊 Subject Embedding需求评分: {phase1_scores['scores']['embedding_necessity_score']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Phase 1 执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


def compute_enhanced_phase1_scores(global_results, region_results, ml_results, 
                                  umap_results=None):
    """计算增强的Phase 1决策得分"""
    
    # 核心指标1：患者方差占比（最重要）
    patient_variance_ratio = ml_results['variance_decomposition']['patient_variance_ratio']
    
    # 核心指标2：全局轮廓系数
    global_silhouette = ml_results['silhouette_analysis']['global_silhouette']
    
    # 核心指标3：条件轮廓系数（移除脑区效应后）
    conditional_silhouette = ml_results['silhouette_analysis']['conditional_silhouette']
    
    # 核心指标4：患者可分离性指数
    separability_index = ml_results['distance_analysis']['separability_index']
    
    # 核心指标5：高患者效应脑区比例
    region_effects = ml_results['variance_decomposition']['region_specific_patient_effects']
    high_effect_regions = sum(1 for effect in region_effects.values() if effect > 0.1)
    high_effect_ratio = high_effect_regions / len(region_effects) if region_effects else 0
    
    # 如果有UMAP结果，加入聚类得分
    if umap_results:
        umap_clustering_score = umap_results['global_umap']['patient_clustering_score']
        conditional_improvement = umap_results['conditional_umap']['clustering_improvement']
    else:
        umap_clustering_score = 0.5  # 默认值
        conditional_improvement = 0.0
    
    # 原始指标（来自之前的分析）
    pca_3pc_variance = np.sum(global_results['feature_variation']['pca_explained_variance'][:3])
    subject_similarity = abs(global_results['similarity']['mean_correlation'])
    
    # 综合embedding需求评分
    embedding_necessity_score = (
        patient_variance_ratio * 0.35 +      # 最重要：患者效应强度
        global_silhouette * 0.15 +           # 轮廓系数
        conditional_silhouette * 0.10 +      # 条件轮廓系数
        min(separability_index / 5, 1) * 0.15 +  # 距离可分离性（归一化）
        high_effect_ratio * 0.10 +           # 脑区覆盖度
        (1 - pca_3pc_variance) * 0.10 +     # 非线性程度
        umap_clustering_score * 0.05         # UMAP聚类质量
    )
    
    # 生成建议
    if patient_variance_ratio > 0.15:
        recommendation = "CRITICAL: 强患者效应(>15%)，必须使用Subject Embedding"
    elif patient_variance_ratio > 0.10:
        recommendation = "HIGH: 明显患者效应(>10%)，强烈建议Subject Embedding"
    elif patient_variance_ratio > 0.05:
        recommendation = "MEDIUM: 中等患者效应(>5%)，建议使用Subject Embedding"
    else:
        recommendation = "LOW: 患者效应较弱(<5%)，Subject Embedding收益有限"
    
    # 补充建议
    if high_effect_ratio > 0.3:
        recommendation += f"\n   - 注意：{high_effect_ratio:.0%}的脑区显示高患者特异性"
    if conditional_improvement > 0.1:
        recommendation += f"\n   - 移除脑区效应后聚类改善{conditional_improvement:.1%}"
    
    return {
        'scores': {
            'patient_variance_ratio': float(patient_variance_ratio),
            'global_silhouette': float(global_silhouette),
            'conditional_silhouette': float(conditional_silhouette),
            'separability_index': float(separability_index),
            'high_effect_ratio': float(high_effect_ratio),
            'pattern_linearity': float(pca_3pc_variance),
            'subject_similarity': float(subject_similarity),
            'umap_clustering_score': float(umap_clustering_score),
            'embedding_necessity_score': float(embedding_necessity_score)
        },
        'recommendation': recommendation,
        'analysis_coverage': 1.0  # Phase 1完成
    }


if __name__ == "__main__":
    sys.exit(main())