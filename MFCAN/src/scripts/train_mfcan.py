#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFCAN 训练脚本，用于训练多模态特征融合与类别适应网络
"""

import os
import argparse
import sys
import time
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from training.mfcan_trainer import MFCANTrainer
from utils.logging_utils import Logger
from loss.mfcan_loss import MFCANLoss
def main():
    """MFCAN训练主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MFCAN训练脚本')
    parser.add_argument('--config', type=str, default='configs/mfcan_config.json', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/reorganized_encoder_data.h5', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'encoders', 'fusion', 'finetune'], 
                        help='训练模式：full-完整训练流程，encoders-只训练编码器，fusion-只训练融合机制，finetune-只微调完整模型')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径，用于继续训练')
    parser.add_argument('--balanced_sampling', action='store_true', help='是否使用平衡采样')
    parser.add_argument('--samples_per_class', type=int, default=1000, help='每个类别采样数量')
    parser.add_argument('--analyze_groups', action='store_true', help='是否分析特征组贡献')
    parser.add_argument('--max_group_size', type=int, default=3, help='特征组分析的最大组合大小')

    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        return
        
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件 {args.data_path} 不存在")
        return
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("MFCAN_Training", log_dir="logs/mfcan")
    logger = logger_manager.get_logger()
    
    logger.info("=" * 80)
    logger.info("开始MFCAN训练流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"训练模式: {args.mode}")
    
    # 设置输出目录
    output_dir = args.output_dir or f"results/mfcan/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 记录平衡采样设置
    if args.balanced_sampling:
        logger.info(f"启用平衡采样，每个类别采样 {args.samples_per_class} 个样本")
    else:
        logger.info("使用原始数据分布（未启用平衡采样）")
    
    # 记录实验开始
    logger_manager.log_experiment_start("MFCAN训练", f"模式: {args.mode}, 时间戳: {timestamp}")
    
    start_time = time.time()
    
    try:
        # 初始化训练器
        trainer = MFCANTrainer(
            config_path=args.config,
            data_path=None,  # 暂不加载数据，下面会手动加载
            output_dir=output_dir,
            logger=logger
        )
        
        # 加载数据 - 根据是否启用平衡采样选择加载方法
        if args.data_path:
            if args.balanced_sampling:
                trainer.load_data_with_balanced_sampling(
                    args.data_path, 
                    samples_per_class=args.samples_per_class,
                    use_balanced_sampler=True
                )
                logger.info(f"使用平衡采样加载数据，每个类别采样 {args.samples_per_class} 个样本")
            else:
                trainer.load_data(args.data_path)
                logger.info("使用原始数据分布加载数据")
        
        # 如果提供了预训练模型路径，则加载模型
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"加载预训练模型: {args.model_path}")
            trainer.load_model(args.model_path)
        
        # 根据训练模式执行不同的训练步骤
        if args.mode == 'full':
            logger.info("执行完整训练流程")
            history = trainer.train_full_pipeline()
        elif args.mode == 'encoders':
            logger.info("仅训练编码器")
            history = trainer.pretrain_encoders()
        elif args.mode == 'fusion':
            logger.info("仅训练融合机制")
            history = trainer.train_fusion_classifier()
        elif args.mode == 'finetune':
            logger.info("仅微调完整模型")
            history = trainer.finetune_full_model()
        
        # 保存最终模型
        model_path = trainer.save_model(os.path.join(output_dir, f"mfcan_{args.mode}_final.pth"))
        logger.info(f"最终模型已保存到: {model_path}")
        
        # 评估模型性能
        logger.info("在测试集上评估模型性能")
        metrics = trainer.evaluate()
        
        # 记录完成时间和指标
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"训练完成! 总耗时: {training_time:.2f} 秒")
        logger.info(f"性能指标: Accuracy={metrics['accuracy']:.4f}, F1-macro={metrics['f1_macro']:.4f}")
        
        # 记录实验结束
        results = {
            "状态": "成功",
            "训练时间(秒)": training_time,
            "准确率": metrics['accuracy'],
            "宏平均F1": metrics['f1_macro'],
            "加权F1": metrics['f1_weighted'],
            "训练模式": args.mode,
            "模型路径": model_path,
            "平衡采样": args.balanced_sampling,
            "每类样本数": args.samples_per_class if args.balanced_sampling else "原始分布"
        }
        logger_manager.log_experiment_end("MFCAN训练", results)
        
        if args.analyze_groups:
            logger.info("开始分析特征组贡献...")
            group_analysis_dir = os.path.join(output_dir, "feature_groups")
            trainer.analyze_feature_group_contributions(
                data_path=args.data_path,
                max_combination_size=args.max_group_size,
                output_dir=group_analysis_dir
            )
            logger.info(f"特征组分析完成，结果保存在 {group_analysis_dir}")

        return {
            'output_dir': output_dir,
            'model_path': model_path,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 记录实验失败
        logger_manager.log_experiment_end("MFCAN训练", {"状态": "失败", "错误": str(e)})
        
        return {
            'status': 'failed',
            'error': str(e)
        }

if __name__ == "__main__":
    main()