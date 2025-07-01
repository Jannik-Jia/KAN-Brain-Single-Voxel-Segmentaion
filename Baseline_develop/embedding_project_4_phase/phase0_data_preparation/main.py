#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0: 数据准备模块主程序
负责数据加载、验证、分割和预处理
增强版本：包含脑区统计、特征组分析和完整的元数据
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import Config
from common.data_io import DataIO
from phase0_data_preparation.data_splitter import DataSplitter
from phase0_data_preparation.data_validator import DataValidator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Phase 0: 数据准备')
    
    parser.add_argument('--data_path', type=str, 
                       default=Config.DEFAULT_DATA_PATH,
                       help='输入数据文件路径')
    
    parser.add_argument('--train_start', type=int, 
                       default=Config.DEFAULT_TRAIN_RANGE[0],
                       help='训练集受试者起始ID')
    
    parser.add_argument('--train_end', type=int,
                       default=Config.DEFAULT_TRAIN_RANGE[1],
                       help='训练集受试者结束ID')
    
    parser.add_argument('--val_start', type=int,
                       default=Config.DEFAULT_VAL_RANGE[0],
                       help='验证集受试者起始ID')
    
    parser.add_argument('--val_end', type=int,
                       default=Config.DEFAULT_VAL_RANGE[1],
                       help='验证集受试者结束ID')
    
    parser.add_argument('--test_subject', type=int,
                       default=Config.DEFAULT_TEST_SUBJECT,
                       help='测试集受试者ID')
    
    parser.add_argument('--output_dir', type=str,
                       default=str(Config.PHASE0_OUTPUT_DIR),
                       help='输出目录')
    
    parser.add_argument('--random_state', type=int,
                       default=Config.DEFAULT_RANDOM_STATE,
                       help='随机种子')
    
    parser.add_argument('--validate_only', action='store_true',
                       help='仅执行数据验证，不进行分割')
    
    return parser.parse_args()


def generate_comprehensive_statistics(split_result, validation_report):
    """生成综合的数据统计信息"""
    # 基础统计
    basic_stats = {
        'train_samples': len(split_result['X_train']),
        'val_samples': len(split_result['X_val']),
        'test_samples': len(split_result['X_test']),
        'total_samples': len(split_result['X_train']) + len(split_result['X_val']) + len(split_result['X_test']),
        'feature_dim': split_result['feature_dim'],
        'n_classes': split_result['n_classes'],
        'split_strategy': split_result['split_strategy'],
        'split_config': split_result['split_config']
    }
    
    # 受试者统计
    subject_stats = {
        'train_subjects': [int(x) for x in split_result['train_subjects_ids']],
        'val_subjects': [int(x) for x in split_result['val_subjects_ids']],
        'test_subject': int(split_result['test_subject_id']),
        'n_train_subjects': len(split_result['train_subjects_ids']),
        'n_val_subjects': len(split_result['val_subjects_ids']),
        'n_test_subjects': 1,
        'subject_sample_distribution': validation_report['subject_analysis']['subject_sample_distribution']
    }
    
    # 特征组统计
    feature_groups = Config.get_feature_groups()
    feature_group_stats = {
        'feature_groups': feature_groups,
        'n_feature_groups': len(feature_groups),
        'feature_group_sizes': {name: len(indices) for name, indices in feature_groups.items()},
        'feature_group_details': validation_report['feature_analysis']['feature_group_stats']
    }
    
    # 脑区统计
    brain_region_stats = {
        'n_regions_total': validation_report['brain_region_analysis']['n_regions_total'],
        'n_regions_with_samples': validation_report['brain_region_analysis']['n_regions_with_samples'],
        'region_sample_counts': validation_report['brain_region_analysis']['region_counts'],
        'region_subject_coverage': {
            rid: info['subject_coverage'] 
            for rid, info in validation_report['brain_region_analysis']['region_subject_distribution'].items()
        },
        'region_size_categories': {
            'small_regions': validation_report['brain_region_analysis']['region_size_analysis']['small_regions'],
            'large_regions': validation_report['brain_region_analysis']['region_size_analysis']['large_regions'],
            'mean_region_size': validation_report['brain_region_analysis']['region_size_analysis']['mean_region_size']
        }
    }
    
    # 数据质量统计
    quality_stats = {
        'has_nan': validation_report['feature_analysis']['nan_count'] > 0,
        'has_inf': validation_report['feature_analysis']['inf_count'] > 0,
        'n_constant_features': validation_report['feature_analysis']['n_constant_features'],
        'constant_feature_indices': validation_report['feature_analysis']['constant_features'],
        'label_completeness': validation_report['overall_status']['completeness_score'],
        'subject_separation_ok': validation_report['overall_status']['subject_separation_score'] == 1.0,
        'data_balance_score': validation_report['overall_status']['balance_score']
    }
    
    # 合并所有统计信息
    comprehensive_stats = {
        'basic_statistics': basic_stats,
        'subject_statistics': subject_stats,
        'feature_statistics': feature_group_stats,
        'brain_region_statistics': brain_region_stats,
        'data_quality_statistics': quality_stats,
        'timestamp': datetime.now().isoformat(),
        'data_version': '1.0'
    }
    
    return comprehensive_stats


def main():
    """主函数"""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("Phase 0: 数据准备开始")
    logger.info("="*80)
    
    # 记录参数
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"训练集: 受试者 {args.train_start}-{args.train_end-1}")
    logger.info(f"验证集: 受试者 {args.val_start}-{args.val_end-1}")
    logger.info(f"测试集: 受试者 {args.test_subject}")
    logger.info(f"输出目录: {args.output_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 步骤1: 数据分割
        logger.info("\n步骤1: 执行数据分割...")
        splitter = DataSplitter(
            data_path=args.data_path,
            train_subjects_range=(args.train_start, args.train_end),
            val_subjects_range=(args.val_start, args.val_end),
            test_subject=args.test_subject,
            random_state=args.random_state
        )
        
        split_result = splitter.split_data()
        
        if split_result is None:
            logger.error("数据分割失败")
            return 1
        
        # 步骤2: 数据验证
        logger.info("\n步骤2: 执行增强数据验证...")
        validator = DataValidator()
        validation_report = validator.validate_split_data(split_result)
        
        # 保存验证报告
        DataIO.save_json(validation_report, output_dir / 'validation_report.json')
        
        if args.validate_only:
            logger.info("仅验证模式，跳过数据保存")
            return 0
        
        # 步骤3: 保存数据
        logger.info("\n步骤3: 保存处理后的数据...")
        
        # 保存训练集
        DataIO.save_numpy_data({
            'X': split_result['X_train'],
            'X_scaled': split_result['X_train_scaled'],
            'y': split_result['y_train'],
            'subjects': split_result['subjects_train']
        }, output_dir / Config.TRAIN_DATA_FILE)
        
        # 保存验证集
        DataIO.save_numpy_data({
            'X': split_result['X_val'],
            'X_scaled': split_result['X_val_scaled'],
            'y': split_result['y_val'],
            'subjects': split_result['subjects_val']
        }, output_dir / Config.VAL_DATA_FILE)
        
        # 保存测试集
        DataIO.save_numpy_data({
            'X': split_result['X_test'],
            'X_scaled': split_result['X_test_scaled'],
            'y': split_result['y_test'],
            'subjects': split_result['subjects_test']
        }, output_dir / Config.TEST_DATA_FILE)
        
        # 保存标准化器
        DataIO.save_pickle(split_result['scaler'], output_dir / Config.SCALER_FILE)
        
        # 步骤4: 生成和保存增强的元数据
        logger.info("\n步骤4: 生成增强元数据...")
        
        # 生成综合统计信息
        comprehensive_stats = generate_comprehensive_statistics(split_result, validation_report)
        DataIO.save_json(comprehensive_stats, output_dir / Config.DATA_STATS_FILE)
        
        # 增强的标签映射信息
        label_mapping = validator.extract_label_mapping(split_result)
        DataIO.save_json(label_mapping, output_dir / Config.LABEL_MAPPING_FILE)
        
        # 创建Phase 1快速索引
        phase1_index = validator.create_phase1_index(split_result)
        DataIO.save_json(phase1_index, output_dir / 'phase1_quick_index.json')
        
        # 保存脑区详细分析
        brain_region_detailed = {
            'region_analysis': validation_report['brain_region_analysis'],
            'region_balance_scores': validation_report['brain_region_analysis']['region_balance_scores'],
            'region_subject_distribution': validation_report['brain_region_analysis']['region_subject_distribution'],
            'recommendations': {
                'small_regions_warning': '小脑区可能需要特殊处理或数据增强',
                'poorly_balanced_regions': validation_report['brain_region_analysis']['poorly_balanced_regions'],
                'well_balanced_regions': validation_report['brain_region_analysis']['well_balanced_regions']
            }
        }
        DataIO.save_json(brain_region_detailed, output_dir / 'brain_region_analysis.json')
        
        # 保存特征组分析
        feature_group_analysis = {
            'feature_groups': Config.get_feature_groups(),
            'feature_group_stats': validation_report['feature_analysis']['feature_group_stats'],
            'feature_quality': {
                'constant_features_by_group': {},
                'quality_scores_by_group': {}
            }
        }
        
        # 分析每个特征组的常数特征
        constant_features = set(validation_report['feature_analysis']['constant_features'])
        for group_name, indices in Config.get_feature_groups().items():
            group_constant = [idx for idx in indices if idx in constant_features]
            feature_group_analysis['feature_quality']['constant_features_by_group'][group_name] = group_constant
            
            # 计算每个特征组的质量分数
            n_constant = len(group_constant)
            n_total = len(indices)
            quality_score = 1.0 - (n_constant / n_total) if n_total > 0 else 1.0
            feature_group_analysis['feature_quality']['quality_scores_by_group'][group_name] = quality_score
        
        DataIO.save_json(feature_group_analysis, output_dir / 'feature_group_analysis.json')
        
        # 保存Phase 0最终结果
        phase0_results = {
            'data_statistics': comprehensive_stats,
            'validation_report': validation_report,
            'label_mapping': label_mapping,
            'brain_region_analysis': brain_region_detailed,
            'feature_group_analysis': feature_group_analysis,
            'phase1_quick_index': phase1_index
        }
        
        phase0_scores = {
            'data_completeness': validation_report['overall_status']['completeness_score'],
            'label_coverage': validation_report['overall_status']['label_coverage'],
            'balance_score': validation_report['overall_status']['balance_score'],
            'region_coverage_score': validation_report['overall_status']['region_coverage_score'],
            'feature_quality_score': validation_report['overall_status']['feature_quality_score'],
            'overall_score': validation_report['overall_status']['overall_score']
        }
        
        DataIO.save_phase_output(
            phase_number=0,
            output_dir=output_dir,
            results=phase0_results,
            scores=phase0_scores,
            metadata={
                'data_path': args.data_path,
                'execution_time': datetime.now().isoformat(),
                'version': '2.0',  # 增强版本
                'enhancements': ['brain_region_analysis', 'feature_group_analysis', 'comprehensive_statistics']
            }
        )
        
        # 生成Phase 0摘要报告
        summary = {
            'status': validation_report['overall_status']['status'],
            'overall_score': validation_report['overall_status']['overall_score'],
            'key_metrics': {
                'total_samples': comprehensive_stats['basic_statistics']['total_samples'],
                'n_features': comprehensive_stats['basic_statistics']['feature_dim'],
                'n_brain_regions': brain_region_detailed['region_analysis']['n_regions_total'],
                'n_subjects': comprehensive_stats['subject_statistics']['n_train_subjects'] + 
                             comprehensive_stats['subject_statistics']['n_val_subjects'] + 
                             comprehensive_stats['subject_statistics']['n_test_subjects']
            },
            'warnings': [],
            'recommendations': []
        }
        
        # 添加警告和建议
        if validation_report['overall_status']['has_data_issues']:
            summary['warnings'].append('数据包含NaN或Inf值')
            summary['recommendations'].append('建议在后续处理中进行数据清洗')
        
        if len(validation_report['brain_region_analysis']['poorly_balanced_regions']) > 5:
            summary['warnings'].append(f"{len(validation_report['brain_region_analysis']['poorly_balanced_regions'])}个脑区样本分布不平衡")
            summary['recommendations'].append('考虑对小脑区使用数据增强或特殊处理策略')
        
        if validation_report['feature_analysis']['n_constant_features'] > 10:
            summary['warnings'].append(f"发现{validation_report['feature_analysis']['n_constant_features']}个常数特征")
            summary['recommendations'].append('建议在Phase 1中考虑特征选择或降维')
        
        DataIO.save_json(summary, output_dir / 'phase0_summary.json')
        
        logger.info("\n" + "="*80)
        logger.info("Phase 0: 数据准备完成!")
        logger.info("="*80)
        logger.info(f"所有文件已保存到: {output_dir}")
        logger.info(f"\n关键指标:")
        logger.info(f"  - 总样本数: {summary['key_metrics']['total_samples']:,}")
        logger.info(f"  - 特征维度: {summary['key_metrics']['n_features']}")
        logger.info(f"  - 脑区数量: {summary['key_metrics']['n_brain_regions']}")
        logger.info(f"  - 受试者数: {summary['key_metrics']['n_subjects']}")
        logger.info(f"  - 总体评分: {summary['overall_score']:.3f}")
        logger.info(f"  - 状态: {summary['status']}")
        
        if summary['warnings']:
            logger.warning("\n注意事项:")
            for warning in summary['warnings']:
                logger.warning(f"  - {warning}")
        
        if summary['recommendations']:
            logger.info("\n建议:")
            for rec in summary['recommendations']:
                logger.info(f"  - {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Phase 0 执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    sys.exit(main())