#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFCAN 继续训练脚本，从上次停止的地方继续
"""

import os
import sys
import time
import argparse
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from training.mfcan_trainer import MFCANTrainer
from utils.logging_utils import Logger
from models.mfcan import MFCAN

def main():
    """从融合阶段继续MFCAN训练"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='继续MFCAN训练')
    parser.add_argument('--config', type=str, default='configs/mfcan_config.json', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/reorganized_encoder_data.h5', help='数据文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("MFCAN_Continue", log_dir="logs/mfcan")
    logger = logger_manager.get_logger()
    
    logger.info("=" * 80)
    logger.info("继续MFCAN训练流程（从融合阶段开始）")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"预训练模型: {args.model_path}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 记录实验开始
    logger_manager.log_experiment_start("MFCAN继续训练", f"从融合阶段开始, 时间戳: {timestamp}")
    
    start_time = time.time()
    
    try:
        # 修复MFCAN的freeze_encoders和unfreeze_encoders方法
        def fixed_freeze_encoders(self):
            """修复后的冻结编码器方法"""
            logger = getattr(self, 'logger', None)
            if logger is None:
                from utils.logging_utils import Logger
                log_manager = Logger("MFCAN", log_dir="logs/model")
                logger = log_manager.get_logger()
            
            for param in self.diffusion_encoder.parameters():
                param.requires_grad = False
            for param in self.qti_encoder.parameters():
                param.requires_grad = False
            for param in self.cest_encoder.parameters():
                param.requires_grad = False
            logger.info("已冻结所有编码器参数")
        
        def fixed_unfreeze_encoders(self):
            """修复后的解冻编码器方法"""
            logger = getattr(self, 'logger', None)
            if logger is None:
                from utils.logging_utils import Logger
                log_manager = Logger("MFCAN", log_dir="logs/model")
                logger = log_manager.get_logger()
            
            for param in self.diffusion_encoder.parameters():
                param.requires_grad = True
            for param in self.qti_encoder.parameters():
                param.requires_grad = True
            for param in self.cest_encoder.parameters():
                param.requires_grad = True
            logger.info("已解冻所有编码器参数")
        
        # 替换方法
        MFCAN.freeze_encoders = fixed_freeze_encoders
        MFCAN.unfreeze_encoders = fixed_unfreeze_encoders
        
        # 初始化训练器
        trainer = MFCANTrainer(
            config_path=args.config,
            data_path=args.data_path,
            output_dir=args.output_dir,
            logger=logger
        )
        
        # 加载预训练模型
        logger.info(f"加载预训练模型: {args.model_path}")
        trainer.load_model(args.model_path)
        
        # 跳过编码器阶段，直接训练融合机制
        logger.info("开始融合机制训练阶段")
        trainer.train_fusion_classifier()
        
        # 微调完整模型
        logger.info("开始微调完整模型阶段")
        trainer.finetune_full_model()
        
        # 保存最终模型
        model_path = trainer.save_model(os.path.join(args.output_dir, "mfcan_final.pth"))
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
            "模型路径": model_path
        }
        logger_manager.log_experiment_end("MFCAN继续训练", results)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 记录实验失败
        logger_manager.log_experiment_end("MFCAN继续训练", {"状态": "失败", "错误": str(e)})

if __name__ == "__main__":
    main()