#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0: 数据准备模块主程序
负责数据加载、验证、分割和预处理
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
        logger.info("\n步骤2: 执行数据验证...")
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
        
        # 保存数据统计信息
        data_statistics = {
            'train_samples': len(split_result['X_train']),
            'val_samples': len(split_result['X_val']),
            'test_samples': len(split_result['X_test']),
            'feature_dim': split_result['feature_dim'],
            'n_classes': split_result['n_classes'],
            'train_subjects': split_result['train_subjects_ids'],
            'val_subjects': split_result['val_subjects_ids'],
            'test_subject': split_result['test_subject_id'],
            'split_strategy': split_result['split_strategy'],
            'split_config': split_result['split_config']
        }
        DataIO.save_json(data_statistics, output_dir / Config.DATA_STATS_FILE)
        
        # 保存标签映射信息
        label_mapping = validator.extract_label_mapping(split_result)
        DataIO.save_json(label_mapping, output_dir / Config.LABEL_MAPPING_FILE)
        
        # 保存Phase 0结果
        phase0_results = {
            'data_statistics': data_statistics,
            'validation_report': validation_report,
            'label_mapping': label_mapping
        }
        
        phase0_scores = {
            'data_completeness': validation_report['overall_status']['completeness_score'],
            'label_coverage': validation_report['overall_status']['label_coverage'],
            'balance_score': validation_report['overall_status']['balance_score']
        }
        
        DataIO.save_phase_output(
            phase_number=0,
            output_dir=output_dir,
            results=phase0_results,
            scores=phase0_scores,
            metadata={
                'data_path': args.data_path,
                'execution_time': datetime.now().isoformat()
            }
        )
        
        logger.info("\n" + "="*80)
        logger.info("Phase 0: 数据准备完成!")
        logger.info("="*80)
        logger.info(f"所有文件已保存到: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Phase 0 执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    sys.exit(main())