#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑区感知Subject Embedding可行性分析 - 主执行文件
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.data_loader import load_and_prepare_data_multi_subject_out
from src.analyzer import BrainAwareSubjectEmbeddingAnalyzer
from src.utils import setup_logging, save_config
from config.settings import Config


def setup_directories(base_path):
    """设置输出目录结构"""
    directories = [
        'logs',
        'outputs',
        'outputs/visualizations', 
        'outputs/reports',
        'outputs/analysis_results'
    ]
    
    for dir_name in directories:
        dir_path = Path(base_path) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return base_path


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑区感知Subject Embedding可行性分析')
    
    parser.add_argument('--data_path', type=str, 
                       default='/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat',
                       help='数据文件路径')
    
    parser.add_argument('--train_start', type=int, default=1,
                       help='训练集受试者起始ID')
    parser.add_argument('--train_end', type=int, default=31,
                       help='训练集受试者结束ID')
    
    parser.add_argument('--val_start', type=int, default=31,
                       help='验证集受试者起始ID')
    parser.add_argument('--val_end', type=int, default=38,
                       help='验证集受试者结束ID')
    
    parser.add_argument('--test_subject', type=int, default=38,
                       help='测试集受试者ID')
    
    parser.add_argument('--output_dir', type=str, 
                       default='./brain_aware_analysis_output',
                       help='输出目录路径')
    
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    parser.add_argument('--skip_visualization', action='store_true',
                       help='跳过可视化生成（节省时间）')
    
    return parser.parse_args()


def main():
    """主执行函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置输出目录
    output_dir = setup_directories(args.output_dir)
    
    # 设置日志
    log_file = Path(output_dir) / 'logs' / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file, args.log_level)
    
    logger.info("="*80)
    logger.info("🧠 开始脑区感知Subject Embedding可行性分析")
    logger.info("="*80)
    
    # 记录参数
    logger.info(f"📊 分析参数:")
    logger.info(f"  - 数据路径: {args.data_path}")
    logger.info(f"  - 训练集: 受试者{args.train_start}-{args.train_end-1}")
    logger.info(f"  - 验证集: 受试者{args.val_start}-{args.val_end-1}")
    logger.info(f"  - 测试集: 受试者{args.test_subject}")
    logger.info(f"  - 输出目录: {output_dir}")
    logger.info(f"  - 随机种子: {args.random_state}")
    
    try:
        start_time = time.time()
        
        # 保存配置
        config_data = {
            'data_path': args.data_path,
            'train_subjects_range': (args.train_start, args.train_end),
            'val_subjects_range': (args.val_start, args.val_end),
            'test_subject': args.test_subject,
            'random_state': args.random_state,
            'analysis_time': datetime.now().isoformat(),
            'output_directory': str(output_dir)
        }
        save_config(config_data, Path(output_dir) / 'outputs' / 'analysis_config.json')
        
        # Phase 0: 数据准备
        logger.info("\n" + "="*60)
        logger.info("📊 Phase 0: 数据加载与预处理")
        logger.info("="*60)
        
        data_dict = load_and_prepare_data_multi_subject_out(
            data_path=args.data_path,
            train_subjects_range=(args.train_start, args.train_end),
            val_subjects_range=(args.val_start, args.val_end),
            test_subject=args.test_subject,
            random_state=args.random_state
        )
        
        if not data_dict:
            logger.error("❌ 数据加载失败，分析终止")
            return 1
            
        logger.info("✅ 数据加载完成")
        
        # 初始化分析器
        analyzer_save_path = Path(output_dir) / 'outputs'
        analyzer = BrainAwareSubjectEmbeddingAnalyzer(save_path=str(analyzer_save_path))
        
        # 数据准备
        logger.info("\n📊 数据预处理...")
        data = analyzer.prepare_data_with_subjects_enhanced(data_dict)
        
        if data is None:
            logger.error("❌ 数据预处理失败，分析终止")
            return 1
            
        # Phase 1: 受试者差异分析
        logger.info("\n" + "="*60)
        logger.info("📊 Phase 1: 受试者间差异分析")
        logger.info("="*60)
        
        analyzer.phase1_subject_differences_analysis()
        logger.info("✅ Phase 1 完成")
        
        # Phase 2: 可分离性评估
        logger.info("\n" + "="*60)
        logger.info("📊 Phase 2: 受试者可分离性评估")
        logger.info("="*60)
        
        analyzer.phase2_subject_separability_analysis()
        logger.info("✅ Phase 2 完成")
        
        # Phase 3: Embedding适配性评估
        logger.info("\n" + "="*60)
        logger.info("📊 Phase 3: Embedding适配性评估")
        logger.info("="*60)
        
        analyzer.phase3_embedding_adaptability_analysis()
        logger.info("✅ Phase 3 完成")
        
        # 生成可视化
        if not args.skip_visualization:
            logger.info("\n" + "="*60)
            logger.info("📊 生成可视化图表")
            logger.info("="*60)
            
            try:
                analyzer.generate_visualizations()
                logger.info("✅ 可视化生成完成")
            except Exception as e:
                logger.warning(f"⚠️ 可视化生成部分失败: {e}")
        else:
            logger.info("⏭️ 跳过可视化生成")
        
        # Phase 4: 决策生成
        logger.info("\n" + "="*60)
        logger.info("📊 Phase 4: 决策建议生成")
        logger.info("="*60)
        
        decision_result = analyzer.phase4_decision_generation()
        logger.info("✅ Phase 4 完成")
        
        # 生成报告
        logger.info("\n📊 生成分析报告...")
        report_path = analyzer.generate_report()
        logger.info(f"✅ 报告已保存: {report_path}")
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 输出最终结果
        logger.info("\n" + "="*80)
        logger.info("🎉 脑区感知Subject Embedding分析完成!")
        logger.info("="*80)
        logger.info(f"⏱️ 总耗时: {total_time/60:.2f} 分钟")
        logger.info(f"📁 所有结果保存在: {output_dir}")
        
        if decision_result and 'comprehensive_recommendation' in decision_result:
            comp_rec = decision_result['comprehensive_recommendation']
            logger.info(f"🎯 最终建议: {comp_rec['final_recommendation']}")
            logger.info(f"📊 综合得分: {comp_rec['comprehensive_score']:.3f}")
            logger.info(f"🏗️ 推荐架构: {comp_rec['architecture_type'].upper()}")
            
            # 保存关键结果到单独文件
            key_results = {
                'final_recommendation': comp_rec['final_recommendation'],
                'comprehensive_score': comp_rec['comprehensive_score'],
                'architecture_type': comp_rec['architecture_type'],
                'implementation_priority': comp_rec['implementation_priority'],
                'analysis_time': datetime.now().isoformat(),
                'total_time_minutes': total_time/60
            }
            
            import json
            key_results_path = Path(output_dir) / 'outputs' / 'key_results.json'
            with open(key_results_path, 'w', encoding='utf-8') as f:
                json.dump(key_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 关键结果已保存: {key_results_path}")
        
        logger.info("\n🚀 分析成功完成！")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 分析过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        return 1
    
    finally:
        # 关闭日志处理器
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)