#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑区感知Subject Embedding可行性分析 - 主执行文件 (增强版)
🔥 新增功能：存档点系统、交互式恢复、异常处理增强
"""

import os
import sys
import time
import logging
import argparse
import signal
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
    """设置输出目录结构 - 增强版"""
    directories = [
        'logs',
        'outputs',
        'outputs/visualizations', 
        'outputs/reports',
        'outputs/analysis_results',
        'checkpoints',  # 🔥 新增：存档点目录
        'recovery_logs'  # 🔥 新增：恢复日志目录
    ]
    
    for dir_name in directories:
        dir_path = Path(base_path) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return base_path


def parse_arguments():
    """解析命令行参数 - 增强版"""
    parser = argparse.ArgumentParser(description='脑区感知Subject Embedding可行性分析 (增强版)')
    
    # 原有参数
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
    
    # 🔥 新增：存档点相关参数
    parser.add_argument('--disable_checkpoints', action='store_true',
                       help='禁用存档点系统')
    
    parser.add_argument('--resume_latest', action='store_true',
                       help='自动恢复最新存档点')
    
    parser.add_argument('--resume_from', type=str, metavar='CHECKPOINT',
                       help='从指定存档点恢复 (文件名或阶段名)')
    
    parser.add_argument('--list_checkpoints', action='store_true',
                       help='列出所有可用存档点并退出')
    
    parser.add_argument('--checkpoint_and_exit', type=str, metavar='NAME',
                       help='创建指定名称的存档点后退出')
    
    parser.add_argument('--force_restart', action='store_true',
                       help='强制重新开始，忽略所有存档点')
    
    parser.add_argument('--interactive_recovery', action='store_true',
                       help='启用交互式恢复选择')
    
    return parser.parse_args()


def signal_handler(signum, frame):
    """信号处理器 - 优雅退出"""
    logger = logging.getLogger(__name__)
    logger.info(f"\n🛑 接收到信号 {signum}，准备优雅退出...")
    
    # 这里可以添加清理逻辑
    # 比如保存当前状态到紧急存档点
    
    logger.info("🔄 清理完成，程序退出")
    sys.exit(0)


def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号


def handle_checkpoint_operations(args, analyzer=None):
    """处理存档点操作"""
    
    if args.list_checkpoints:
        if analyzer and analyzer.checkpoint_manager:
            checkpoints = analyzer.list_checkpoints()
            
            if not checkpoints:
                print("📂 未找到任何存档点")
                return True
            
            print("\n" + "="*80)
            print("📂 可用存档点列表:")
            print("="*80)
            
            for i, ckpt in enumerate(checkpoints):
                print(f"[{i}] {ckpt['name']}")
                print(f"    创建时间: {ckpt['creation_time']}")
                print(f"    阶段进度: Phase {ckpt['phase_completed']} ({ckpt['progress_percentage']:.1f}%)")
                print(f"    文件大小: {ckpt['file_size_mb']:.1f} MB")
                print(f"    描述: {ckpt['description']}")
                print()
            
            print(f"总计 {len(checkpoints)} 个存档点")
        else:
            print("❌ 存档点系统未初始化")
        
        return True
    
    return False


def create_analyzer_with_recovery(args, output_dir):
    """创建分析器并处理恢复逻辑"""
    
    # 获取logger实例
    logger = logging.getLogger(__name__)
    
    # 创建分析器
    enable_checkpoints = not args.disable_checkpoints
    analyzer = BrainAwareSubjectEmbeddingAnalyzer(
        save_path=output_dir,
        enable_checkpoints=enable_checkpoints
    )
    
    # 如果禁用了存档点，直接返回
    if args.disable_checkpoints:
        logger.info("📝 存档点系统已禁用，将重新开始分析")
        return analyzer, None
    
    # 强制重新开始
    if args.force_restart:
        logger.info("🔄 强制重新开始分析，忽略所有存档点")
        return analyzer, None
    
    # 从指定存档点恢复
    if args.resume_from:
        logger.info(f"🔄 尝试从存档点恢复: {args.resume_from}")
        if analyzer.load_checkpoint(args.resume_from):
            logger.info("✅ 存档点恢复成功")
            return analyzer, args.resume_from
        else:
            logger.error("❌ 存档点恢复失败，将重新开始分析")
            return analyzer, None
    
    # 自动恢复最新存档点
    if args.resume_latest:
        checkpoints = analyzer.list_checkpoints()
        if checkpoints:
            latest_checkpoint = checkpoints[0]['filename']
            logger.info(f"🔄 自动恢复最新存档点: {latest_checkpoint}")
            if analyzer.load_checkpoint(latest_checkpoint):
                logger.info("✅ 最新存档点恢复成功")
                return analyzer, latest_checkpoint
            else:
                logger.error("❌ 最新存档点恢复失败，将重新开始分析")
                return analyzer, None
        else:
            logger.info("📂 未找到存档点，将重新开始分析")
            return analyzer, None
    
    # 交互式恢复选择
    if args.interactive_recovery:
        if analyzer.interactive_recovery():
            logger.info("✅ 交互式恢复成功")
            return analyzer, "interactive"
        else:
            logger.info("🔄 用户选择重新开始分析")
            return analyzer, None
    
    # 默认：检查是否有存档点，如果有则询问
    checkpoints = analyzer.list_checkpoints()
    if checkpoints and not args.force_restart:
        try:
            # 非交互式环境或自动化脚本中，默认重新开始
            logger.info("📂 检测到存档点，但未指定恢复选项，将重新开始分析")
            logger.info("💡 提示：使用 --interactive_recovery 启用交互式选择")
            logger.info("💡 提示：使用 --resume_latest 自动恢复最新存档点")
        except:
            pass
    
    return analyzer, None


def determine_skip_phases(recovered_checkpoint_info, analyzer):
    """根据恢复的存档点确定要跳过的阶段"""
    logger = logging.getLogger(__name__)  # 获取logger实例
    
    if not recovered_checkpoint_info or not analyzer.checkpoint_manager:
        return []
    
    # 获取当前分析器的状态
    skip_phases = []
    
    # 检查各阶段的完成状态
    if hasattr(analyzer, 'analysis_results'):
        results = analyzer.analysis_results
        
        # Phase 1: 受试者差异分析
        if ('subject_stats' in results or 'global_feature_variation' in results):
            skip_phases.append(1)
            logger.info("📋 检测到Phase 1已完成，将跳过")
        
        # Phase 2: 可分离性评估
        if ('neural_network_baseline_analysis' in results or 'region_wise_separability' in results):
            skip_phases.append(2)
            logger.info("📋 检测到Phase 2已完成，将跳过")
        
        # Phase 3: Embedding设计
        if ('enhanced_dimensionality' in results or 'brain_aware_embedding_design' in results):
            skip_phases.append(3)
            logger.info("📋 检测到Phase 3已完成，将跳过")
        
        # Phase 4: 最终决策
        if 'final_comprehensive_decision' in results:
            skip_phases.append(4)
            logger.info("📋 检测到Phase 4已完成，将跳过")
    
    if skip_phases:
        logger.info(f"⏭️ 将跳过阶段: {skip_phases}")
    
    return skip_phases


def main():
    """主执行函数 - 增强版"""
    
    # 设置信号处理器
    setup_signal_handlers()
    
    # 解析参数
    args = parse_arguments()
    
    # 设置输出目录
    output_dir = setup_directories(args.output_dir)
    
    # 设置日志
    log_file = Path(output_dir) / 'logs' / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file, args.log_level)
    
    logger.info("="*80)
    logger.info("🧠 开始脑区感知Subject Embedding可行性分析 (增强版)")
    logger.info("="*80)
    
    # 记录参数
    logger.info(f"📊 分析参数:")
    logger.info(f"  - 数据路径: {args.data_path}")
    logger.info(f"  - 训练集: 受试者{args.train_start}-{args.train_end-1}")
    logger.info(f"  - 验证集: 受试者{args.val_start}-{args.val_end-1}")
    logger.info(f"  - 测试集: 受试者{args.test_subject}")
    logger.info(f"  - 输出目录: {output_dir}")
    logger.info(f"  - 随机种子: {args.random_state}")
    logger.info(f"  🔥 存档点系统: {'启用' if not args.disable_checkpoints else '禁用'}")
    
    try:
        start_time = time.time()
        
        # 创建分析器并处理恢复
        analyzer, recovered_checkpoint = create_analyzer_with_recovery(args, output_dir)
        
        # 处理存档点操作
        if handle_checkpoint_operations(args, analyzer):
            return 0
        
        # 如果指定了checkpoint_and_exit
        if args.checkpoint_and_exit:
            logger.info(f"🔄 创建存档点: {args.checkpoint_and_exit}")
            checkpoint_path = analyzer.save_checkpoint(
                checkpoint_name=args.checkpoint_and_exit,
                description="用户手动创建的存档点"
            )
            logger.info(f"✅ 存档点创建完成: {checkpoint_path}")
            return 0
        
        # 保存配置
        config_data = {
            'data_path': args.data_path,
            'train_subjects_range': (args.train_start, args.train_end),
            'val_subjects_range': (args.val_start, args.val_end),
            'test_subject': args.test_subject,
            'random_state': args.random_state,
            'analysis_time': datetime.now().isoformat(),
            'output_directory': str(output_dir),
            'enable_checkpoints': not args.disable_checkpoints,
            'recovered_from_checkpoint': recovered_checkpoint is not None
        }
        save_config(config_data, Path(output_dir) / 'outputs' / 'analysis_config.json')
        
        # Phase 0: 数据准备 (如果没有从存档点恢复，或者存档点不包含数据)
        if not recovered_checkpoint or not hasattr(analyzer, 'data') or not analyzer.data:
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
            
            # 数据准备
            logger.info("\n📊 数据预处理...")
            analyzer.prepare_data_with_subjects_enhanced(data_dict)
            
            if analyzer.data is None:
                logger.error("❌ 数据预处理失败，分析终止")
                return 1
        else:
            logger.info("✅ 从存档点恢复了数据状态，跳过数据加载")
        
        # 确定要跳过的阶段
        skip_phases = determine_skip_phases(recovered_checkpoint, analyzer)
        
        # 执行完整分析
        logger.info("\n🚀 开始执行分析流程...")
        final_decision = analyzer.run_complete_analysis(skip_phase=skip_phases)
        
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
        
        # 生成报告
        logger.info("\n📊 生成分析报告...")
        report_path = analyzer.generate_report()
        logger.info(f"✅ 报告已保存: {report_path}")
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 输出最终结果
        logger.info("\n" + "="*80)
        logger.info("🎉 脑区感知Subject Embedding分析完成! (增强版)")
        logger.info("="*80)
        logger.info(f"⏱️ 总耗时: {total_time/60:.2f} 分钟")
        logger.info(f"📁 所有结果保存在: {output_dir}")
        
        # 🔥 存档点统计
        if analyzer.checkpoint_manager:
            checkpoints = analyzer.list_checkpoints()
            logger.info(f"🔄 创建存档点数量: {len(checkpoints)}")
            if checkpoints:
                total_checkpoint_size = sum(ckpt['file_size_mb'] for ckpt in checkpoints)
                logger.info(f"💾 存档点总大小: {total_checkpoint_size:.1f} MB")
        
        if final_decision and 'comprehensive_recommendation' in final_decision:
            comp_rec = final_decision['comprehensive_recommendation']
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
                'total_time_minutes': total_time/60,
                'recovered_from_checkpoint': recovered_checkpoint is not None,
                'checkpoint_used': recovered_checkpoint if recovered_checkpoint else None,
                'phases_skipped': skip_phases
            }
            
            import json
            key_results_path = Path(output_dir) / 'outputs' / 'key_results.json'
            with open(key_results_path, 'w', encoding='utf-8') as f:
                json.dump(key_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 关键结果已保存: {key_results_path}")
        
        logger.info("\n🚀 增强版分析成功完成！")
        
        # 🔥 最终提示
        if analyzer.checkpoint_manager:
            logger.info("\n💡 存档点使用提示:")
            logger.info("  - 使用 --list_checkpoints 查看所有存档点")
            logger.info("  - 使用 --resume_from <checkpoint> 从指定存档点恢复")
            logger.info("  - 使用 --interactive_recovery 启用交互式恢复")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n🛑 用户中断分析")
        
        # 🔥 中断时创建紧急存档点
        if 'analyzer' in locals() and analyzer.checkpoint_manager:
            try:
                emergency_checkpoint = analyzer.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=analyzer,
                    error_info="用户中断分析",
                    stack_trace="KeyboardInterrupt"
                )
                logger.info(f"🔄 已创建紧急存档点: {emergency_checkpoint}")
            except Exception as e:
                logger.warning(f"紧急存档点创建失败: {e}")
        
        return 130  # 标准的KeyboardInterrupt退出码
        
    except Exception as e:
        logger.error(f"❌ 分析过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        
        # 🔥 异常时创建紧急存档点
        if 'analyzer' in locals() and analyzer.checkpoint_manager:
            try:
                emergency_checkpoint = analyzer.checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=analyzer,
                    error_info=str(e),
                    stack_trace=str(e)
                )
                logger.info(f"🔄 已创建紧急存档点: {emergency_checkpoint}")
            except Exception as checkpoint_error:
                logger.warning(f"紧急存档点创建失败: {checkpoint_error}")
        
        return 1
    
    finally:
        # 关闭日志处理器
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)