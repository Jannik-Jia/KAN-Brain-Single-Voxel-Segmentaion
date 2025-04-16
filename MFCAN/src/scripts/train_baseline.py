import os
import argparse
import json
import sys
import torch
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.baseline import BaselineMLP, FeatureGroupMLP, DeepMLP
from src.training.baseline_trainer import BaselineTrainer
from src.utils.logging_utils import Logger

def main():
    """训练基线模型主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练脑MRI数据基线模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 设置日志
    logger_manager = Logger("TrainBaseline", log_dir="logs/training")
    logger = logger_manager.get_logger()
    
    logger.info("="*80)
    logger.info("开始训练基线模型")
    logger.info(f"配置文件: {args.config}")
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("成功加载配置文件")
        logger_manager.log_config(config, "模型配置")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置设备
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model_type = config.get('model_type', 'mlp')
    input_dim = config.get('input_dim', 341)
    hidden_dims = config.get('hidden_dims', [1024, 512, 256])
    num_classes = config.get('num_classes', 102)
    
    try:
        if model_type == 'mlp':
            model = BaselineMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=config.get('dropout_rate', 0.3)
            )
            logger.info(f"创建基础MLP模型，输入维度: {input_dim}, 隐藏层: {hidden_dims}, 输出类别: {num_classes}")
        
        elif model_type == 'group_mlp':
            group_dims = config.get('group_dims', {'diffusion': 15, 'qti': 210, 'cest': 116})
            group_hidden_dims = config.get('group_hidden_dims', {
                'diffusion': [128, 64],
                'qti': [256, 128],
                'cest': [192, 96]
            })
            fusion_method = config.get('fusion_method', 'concat')
            
            model = FeatureGroupMLP(
                group_dims=group_dims,
                hidden_dims=group_hidden_dims,
                num_classes=num_classes,
                fusion_method=fusion_method,
                dropout_rate=config.get('dropout_rate', 0.3)
            )
            logger.info(f"创建分组MLP模型，特征组: {group_dims}, 融合方法: {fusion_method}")
        
        elif model_type == 'deep_mlp':
            model = DeepMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                use_residual=config.get('use_residual', True),
                use_self_attention=config.get('use_self_attention', True),
                use_feature_interaction=config.get('use_feature_interaction', True),
                dropout_rates=config.get('dropout_rates', None),
                num_attn_heads=config.get('num_attn_heads', 8),
                attn_layers=config.get('attn_layers', None)
            )
            logger.info(f"创建增强版DeepMLP模型，输入维度: {input_dim}, 隐藏层: {hidden_dims}")
        
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 记录模型参数数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数量: {total_params:,}")
        
    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        return
    
    # 创建训练器
    try:
        trainer = BaselineTrainer(model, args.config, device)
        logger.info("成功创建训练器")
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        return
    
    # 准备数据
    data_path = config.get('preprocessed_data_path')
    logger.info(f"从 {data_path} 加载预处理数据...")
    
    try:
        data_loaders = trainer.prepare_data(data_path)
        train_size = len(data_loaders['train'].dataset)
        val_size = len(data_loaders['val'].dataset)
        test_size = len(data_loaders['test'].dataset)
        logger.info(f"数据加载成功 - 训练集: {train_size} 样本, 验证集: {val_size} 样本, 测试集: {test_size} 样本")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return
    
    # 训练模型
    logger_manager.log_experiment_start(f"训练{model_type}模型", f"特征维度: {input_dim}, 类别数: {num_classes}")
    logger.info("开始训练模型...")
    
    try:
        history = trainer.train(data_loaders)
        logger.info("模型训练完成")
        
        # 记录训练历史
        train_metrics = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc']),
            'best_epoch': history['val_acc'].index(max(history['val_acc'])) + 1
        }
        logger_manager.log_metrics(train_metrics, prefix="训练")
        
    except Exception as e:
        logger.error(f"训练模型失败: {e}")
        logger_manager.log_experiment_end(f"训练{model_type}模型", {"状态": "失败", "错误": str(e)})
        return
    
    # 评估模型
    logger.info("在测试集上评估模型...")
    
    try:
        metrics = trainer.evaluate(data_loaders['test'])
        
        logger.info(f"测试集准确率: {metrics['accuracy']:.4f}")
        logger.info(f"测试集宏平均F1分数: {metrics['f1_macro']:.4f}")
        logger.info(f"测试集加权F1分数: {metrics['f1_weighted']:.4f}")
        logger.info(f"评估结果已保存到 {config.get('save_dir', 'results/baseline')}")
        
        # 记录测试结果
        test_results = {
            "准确率": metrics['accuracy'],
            "宏平均F1": metrics['f1_macro'],
            "加权F1": metrics['f1_weighted']
        }
        logger_manager.log_experiment_end(f"训练{model_type}模型", test_results)
        
    except Exception as e:
        logger.error(f"评估模型失败: {e}")
        logger_manager.log_experiment_end(f"训练{model_type}模型", {"状态": "部分完成", "错误": str(e)})

if __name__ == "__main__":
    main()