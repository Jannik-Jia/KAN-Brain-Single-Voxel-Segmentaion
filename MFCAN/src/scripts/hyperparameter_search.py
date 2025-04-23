#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFCAN 超参数优化脚本
"""

import os
import argparse
import sys
import time
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.hyperparameter_optimizer import HyperparameterOptimizer
from training.mfcan_trainer import MFCANTrainer
from utils.logging_utils import Logger

def main():
    """MFCAN超参数优化主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MFCAN超参数优化')
    parser.add_argument('--config', type=str, default='configs/mfcan_config.json', help='基础配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/reorganized_encoder_data.h5', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则自动生成')
    parser.add_argument('--search_method', type=str, default='grid', choices=['grid', 'random', 'bayesian'], 
                        help='搜索方法: grid-网格搜索, random-随机搜索, bayesian-贝叶斯优化')
    parser.add_argument('--n_iter', type=int, default=10, help='迭代次数（对于随机搜索和贝叶斯优化）')
    parser.add_argument('--max_combinations', type=int, default=None, help='最大组合数（对于网格搜索）')
    
    args = parser.parse_args()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"results/hyperopt/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger_manager = Logger("MFCAN_HyperOpt", log_dir="logs/hyperopt")
    logger = logger_manager.get_logger()
    
    logger.info("=" * 80)
    logger.info(f"MFCAN超参数优化 - 方法: {args.search_method}")
    logger.info(f"基础配置: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 记录实验开始
    logger_manager.log_experiment_start(f"MFCAN超参数优化 ({args.search_method})", 
                                       f"时间戳: {timestamp}, 方法: {args.search_method}")
    
    start_time = time.time()
    
    try:
        # 创建超参数优化器
        optimizer = HyperparameterOptimizer(args.config, output_dir, logger)
        
        # 定义参数空间
        if args.search_method == 'grid':
            # 网格搜索参数
            param_grid = {
                'training.initial_lr': [1e-3, 1e-4, 5e-5],
                'training.weight_decay': [1e-4, 1e-5],
                'training.aux_loss_weight': [0.2, 0.3, 0.5],
                'loss.attn_reg_weight': [0.01, 0.05, 0.1],
                'loss.class_balance_method': ['effective_samples', 'inverse', 'none']
            }
            configs = optimizer.grid_search(param_grid, args.max_combinations)
        
        elif args.search_method == 'random':
            # 随机搜索参数
            param_distributions = {
                'training.initial_lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                'training.weight_decay': [1e-3, 5e-4, 1e-4, 5e-5],
                'training.aux_loss_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
                'loss.attn_reg_weight': [0.005, 0.01, 0.02, 0.05, 0.1],
                'loss.class_balance_method': ['effective_samples', 'inverse', 'sqrt_inverse', 'none'],
                'training.dropout': [0.3, 0.4, 0.5, 0.6]
            }
            configs = optimizer.random_search(param_distributions, args.n_iter)
        
        elif args.search_method == 'bayesian':
            # 贝叶斯优化参数空间
            param_space = {
                'training.initial_lr': (1e-5, 1e-3, 'log-uniform'),
                'training.weight_decay': (1e-6, 1e-3, 'log-uniform'),
                'training.aux_loss_weight': (0.1, 0.5),
                'loss.attn_reg_weight': (0.005, 0.1),
                'loss.class_balance_method': ['effective_samples', 'inverse', 'none'],
                'training.dropout': (0.2, 0.6)
            }
            configs = optimizer.bayesian_optimization(param_space, args.n_iter)
        
        # 训练每个配置
        for config in configs:
            logger.info(f"Training with config {config['id']}")
            
            # 创建训练器
            trainer = MFCANTrainer(
                config_path=config['path'],
                data_path=args.data_path,
                output_dir=os.path.join(output_dir, config['id']),
                logger=logger
            )
            
            # 训练模型（使用较短的轮次进行快速评估）
            # 修改轮次数以加速超参数搜索
            trainer.pretrain_epochs = min(10, trainer.pretrain_epochs)
            trainer.finetune_epochs = min(20, trainer.finetune_epochs)
            
            # 执行训练
            trainer.train_full_pipeline()
            
            # 评估性能
            metrics = trainer.evaluate()
            
            # 记录结果
            optimizer.record_result(config['id'], metrics)
        
        # 可视化结果
        optimizer.visualize_results()
        
        # 获取最佳配置
        best_config = optimizer.get_best_config()
        
        # 保存最佳配置
        best_config_path = os.path.join(output_dir, "best_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"超参数优化完成! 总耗时: {total_time:.2f} 秒")
        logger.info(f"最佳配置已保存到: {best_config_path}")
        
        # 记录实验结束
        results = {
            "状态": "成功",
            "运行时间(秒)": total_time,
            "优化方法": args.search_method,
            "配置数量": len(configs),
            "最佳配置": best_config_path
        }
        logger_manager.log_experiment_end(f"MFCAN超参数优化 ({args.search_method})", results)
        
    except Exception as e:
        logger.error(f"超参数优化过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 记录实验失败
        logger_manager.log_experiment_end(f"MFCAN超参数优化 ({args.search_method})", 
                                        {"状态": "失败", "错误": str(e)})

if __name__ == "__main__":
    main()