#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段三：编码器预训练脚本
预训练各特征组的编码器
"""

import os
import argparse
import json
import sys
import time
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from training.encoder_trainer import EncoderTrainer
from utils.logging_utils import Logger

def main():
    """编码器预训练主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脑MRI数据编码器预训练')
    parser.add_argument('--config', type=str, default='configs/encoders_config.json', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/feature_groups.h5', help='特征组数据路径')
    parser.add_argument('--modality', type=str, required=True, choices=['diffusion', 'qti', 'cest', 'all'], 
                       help='要训练的特征模态，"all"表示全部模态')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    args = parser.parse_args()
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("Stage3_EncoderPretraining", log_dir="logs/stage3")
    logger = logger_manager.get_logger()
    
    logger.info("=" * 80)
    logger.info("阶段三：开始编码器预训练流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"模态: {args.modality}")
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("成功加载配置文件")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置输出目录
    base_output_dir = args.output_dir or config.get('output_dir', 'models/encoders')
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 更新配置的保存路径
    config['save_dir'] = output_dir
    
    # 保存更新后的配置
    config_path = os.path.join(output_dir, 'encoder_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"保存运行配置到 {config_path}")
    
    # 获取要训练的模态列表
    modalities = ['diffusion', 'qti', 'cest'] if args.modality == 'all' else [args.modality]
    
    # 记录实验开始
    experiment_name = f"阶段三：编码器预训练 ({', '.join(modalities)})"
    logger_manager.log_experiment_start(experiment_name, f"时间戳: {timestamp}")
    
    start_time = time.time()
    
    # 对每个选定的模态进行预训练
    results = {}
    
    for modality in modalities:
        logger.info(f"开始预训练 {modality} 编码器")
        
        # 创建编码器训练器
        encoder_trainer = EncoderTrainer(
            modality=modality,
            config_path=config_path,
            logger=logger
        )
        
        try:
            # 加载数据
            data_loaders = encoder_trainer.load_data(args.data_path)
            
            # 训练编码器
            model, history = encoder_trainer.train(data_loaders)
            
            # 记录训练结果
            results[modality] = {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'best_val_acc': max(history['val_acc']),
                'epochs_trained': len(history['train_loss'])
            }
            
            logger.info(f"{modality} 编码器预训练完成 - 验证准确率: {results[modality]['best_val_acc']:.4f}")
            
        except Exception as e:
            logger.error(f"{modality} 编码器预训练失败: {e}")
            results[modality] = {'status': 'failed', 'error': str(e)}
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 生成预训练汇总报告
    report_path = os.path.join(output_dir, 'pretraining_summary.md')
    with open(report_path, 'w') as f:
        f.write("# 编码器预训练汇总报告\n\n")
        f.write(f"预训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {training_time:.2f} 秒\n\n")
        
        f.write("## 预训练结果\n\n")
        f.write("| 模态 | 训练轮次 | 最佳验证准确率 | 最终训练损失 | 最终验证损失 |\n")
        f.write("|------|---------|--------------|------------|------------|\n")
        
        for modality, result in results.items():
            if 'status' in result and result['status'] == 'failed':
                f.write(f"| {modality} | 失败 | - | - | - |\n")
            else:
                f.write(f"| {modality} | {result['epochs_trained']} | {result['best_val_acc']:.4f} | "
                       f"{result['final_train_loss']:.4f} | {result['final_val_loss']:.4f} |\n")
        
        f.write("\n## 编码器结构\n\n")
        f.write("每个编码器被预训练用于从特定模态特征中提取有意义的表示，具有以下结构:\n\n")
        
        f.write("### 扩散特征编码器 (DiffusionEncoder)\n\n")
        f.write("- 输入维度: 15\n")
        f.write("- 隐藏维度: 64\n")
        f.write("- 输出维度: 32\n")
        f.write("- 主要结构: 两层MLP带批归一化和残差连接\n\n")
        
        f.write("### QTI特征编码器 (QTIEncoder)\n\n")
        f.write("- 输入维度: 210\n")
        f.write("- 隐藏维度: [512, 256]\n")
        f.write("- 输出维度: 128\n")
        f.write("- 主要结构: 三层MLP带批归一化\n\n")
        
        f.write("### CEST特征编码器 (CESTEncoder)\n\n")
        f.write("- 输入维度: 116\n")
        f.write("- 隐藏维度: 256 → 128\n")
        f.write("- 输出维度: 128\n")
        f.write("- 主要结构: 三层MLP带批归一化和较高dropout\n\n")
        
        f.write("## 后续步骤\n\n")
        f.write("1. **融合机制训练**: 下一步将使用这些预训练编码器构建多模态融合模型\n")
        f.write("2. **特征表示分析**: 分析每个编码器学习的特征表示\n")
        f.write("3. **端到端微调**: 最后进行整体模型的端到端微调\n")
    
    logger.info(f"预训练汇总报告已保存至 {report_path}")
    
    # 记录实验结束
    results_summary = {
        "状态": "成功" if all('status' not in r for r in results.values()) else "部分完成",
        "训练时间(秒)": training_time,
    }
    
    # 添加每个模态的结果
    for modality, result in results.items():
        if 'status' not in result:
            results_summary[f"{modality}_准确率"] = result['best_val_acc']
    
    logger_manager.log_experiment_end(experiment_name, results_summary)
    
    logger.info(f"编码器预训练完成! 总耗时: {training_time:.2f} 秒")
    
    return {
        'output_dir': output_dir,
        'results': results,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    main()