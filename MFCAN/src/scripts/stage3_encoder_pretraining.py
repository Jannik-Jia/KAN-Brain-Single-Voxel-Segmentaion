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
import traceback
import h5py

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from training.encoder_trainer import EncoderTrainer
from utils.logging_utils import Logger

def main():
    """编码器预训练主函数 - 增强版"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脑MRI数据编码器预训练')
    parser.add_argument('--config', type=str, default='configs/encoders_config.json', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='data/processed/feature_groups.h5', help='特征组数据路径')
    parser.add_argument('--modality', type=str, required=True, choices=['diffusion', 'qti', 'cest', 'all'], 
                       help='要训练的特征模态，"all"表示全部模态')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    parser.add_argument('--continue_training', action='store_true', help='是否在训练失败后继续训练其他模态')
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
    logger_manager = Logger("Stage3_EncoderPretraining", log_dir="logs/stage3")
    logger = logger_manager.get_logger()
    
    logger.info("=" * 80)
    logger.info("阶段三：开始编码器预训练流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"模态: {args.modality}")
    
    # 分析数据文件，获取可用的模态
    try:
        available_modalities = analyze_data_file(args.data_path, logger)
        logger.info(f"在数据文件中找到的可用模态: {available_modalities}")
    except Exception as e:
        logger.error(f"分析数据文件失败: {e}")
        available_modalities = ['diffusion', 'qti', 'cest']  # 默认值
    
    # 获取要训练的模态列表
    if args.modality == 'all':
        modalities = available_modalities
    else:
        if args.modality not in available_modalities:
            logger.warning(f"模态 {args.modality} 在数据文件中不可用，使用默认模态")
            modalities = [args.modality]  # 仍然使用请求的模态
        else:
            modalities = [args.modality]
    
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
    
    # 记录实验开始
    experiment_name = f"阶段三：编码器预训练 ({', '.join(modalities)})"
    logger_manager.log_experiment_start(experiment_name, f"时间戳: {timestamp}")
    
    start_time = time.time()
    
    # 对每个选定的模态进行预训练
    results = {}
    
    for modality in modalities:
        logger.info(f"开始预训练 {modality} 编码器")
        
        # 创建模态特定的输出目录
        modality_output_dir = os.path.join(output_dir, modality)
        os.makedirs(modality_output_dir, exist_ok=True)
        
        # 更新模态特定的配置
        modality_config = config.copy()
        modality_config['save_dir'] = modality_output_dir
        
        # 保存模态特定的配置
        modality_config_path = os.path.join(modality_output_dir, f'{modality}_config.json')
        with open(modality_config_path, 'w') as f:
            json.dump(modality_config, f, indent=4)
        
        try:
            # 创建编码器训练器
            encoder_trainer = EncoderTrainer(
                modality=modality,
                config_path=modality_config_path,
                logger=logger
            )
            
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
            logger.error(traceback.format_exc())
            results[modality] = {'status': 'failed', 'error': str(e)}
            
            # 如果不继续训练，则退出循环
            if not args.continue_training:
                logger.error("由于训练失败且未启用continue_training，停止预训练其他模态")
                break
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 生成预训练汇总报告
    generate_summary_report(output_dir, results, modalities, training_time, logger)
    
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

def analyze_data_file(data_path, logger):
    """分析数据文件，获取可用的模态"""
    available_modalities = []
    
    try:
        with h5py.File(data_path, 'r') as f:
            # 递归搜索所有组
            def visit_group(name, obj):
                if isinstance(obj, h5py.Group):
                    if 'diffusion' in name and 'diffusion' not in available_modalities:
                        available_modalities.append('diffusion')
                    if 'qti' in name and 'qti' not in available_modalities:
                        available_modalities.append('qti')
                    if 'cest' in name and 'cest' not in available_modalities:
                        available_modalities.append('cest')
            
            f.visititems(visit_group)
            
            # 如果找不到模态，尝试其他结构
            if not available_modalities:
                if 'grouped' in f:
                    for modality in ['diffusion', 'qti', 'cest']:
                        if f'grouped/train/{modality}' in f:
                            available_modalities.append(modality)
                            
            # 如果仍然找不到，尝试更多结构
            if not available_modalities:
                for modality in ['diffusion', 'qti', 'cest']:
                    if f'group_{modality}' in f or modality in f:
                        available_modalities.append(modality)
    
    except Exception as e:
        logger.error(f"分析数据文件时出错: {e}")
    
    # 如果没有找到任何模态，使用默认值
    if not available_modalities:
        available_modalities = ['diffusion', 'qti', 'cest']
        logger.warning(f"在数据文件中未找到任何模态，使用默认模态: {available_modalities}")
    
    return available_modalities

def generate_summary_report(output_dir, results, modalities, training_time, logger):
    """生成预训练汇总报告"""
    report_path = os.path.join(output_dir, 'pretraining_summary.md')
    
    try:
        with open(report_path, 'w') as f:
            f.write("# 编码器预训练汇总报告\n\n")
            f.write(f"预训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {training_time:.2f} 秒\n\n")
            
            # 添加训练成功摘要
            success_count = sum(1 for r in results.values() if 'status' not in r)
            f.write(f"## 训练摘要\n\n")
            f.write(f"* 成功训练的模态: {success_count}/{len(modalities)}\n")
            if success_count < len(modalities):
                failed_modalities = [m for m, r in results.items() if 'status' in r]
                f.write(f"* 训练失败的模态: {', '.join(failed_modalities)}\n")
            f.write("\n")
            
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
            
            # 添加各个编码器的详细结构
            write_encoder_structure(f, 'diffusion')
            write_encoder_structure(f, 'qti')
            write_encoder_structure(f, 'cest')
            
            f.write("## 后续步骤\n\n")
            f.write("1. **融合机制训练**: 下一步将使用这些预训练编码器构建多模态融合模型\n")
            f.write("2. **特征表示分析**: 分析每个编码器学习的特征表示\n")
            f.write("3. **端到端微调**: 最后进行整体模型的端到端微调\n")
        
        logger.info(f"预训练汇总报告已保存至 {report_path}")
    except Exception as e:
        logger.error(f"生成预训练汇总报告失败: {e}")

def write_encoder_structure(file, modality):
    """写入特定编码器的结构信息"""
    if modality == 'diffusion':
        file.write("### 扩散特征编码器 (DiffusionEncoder)\n\n")
        file.write("- 输入维度: 15\n")
        file.write("- 隐藏维度: 64\n")
        file.write("- 输出维度: 32\n")
        file.write("- 主要结构: 两层MLP带批归一化和残差连接\n\n")
    elif modality == 'qti':
        file.write("### QTI特征编码器 (QTIEncoder)\n\n")
        file.write("- 输入维度: 210\n")
        file.write("- 隐藏维度: [512, 256]\n")
        file.write("- 输出维度: 128\n")
        file.write("- 主要结构: 三层MLP带批归一化\n\n")
    elif modality == 'cest':
        file.write("### CEST特征编码器 (CESTEncoder)\n\n")
        file.write("- 输入维度: 116\n")
        file.write("- 隐藏维度: 256 → 128\n")
        file.write("- 输出维度: 128\n")
        file.write("- 主要结构: 三层MLP带批归一化和较高dropout\n\n")

if __name__ == "__main__":
    main()