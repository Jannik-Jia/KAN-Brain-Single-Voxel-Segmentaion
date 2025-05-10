#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本，从已训练的模型继续评估流程
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

# 导入自定义模块
from config import load_config
from models import get_model
from data import BrainVoxelDataset, load_multiclass_data
from utils.metrics import evaluate_model
from utils.visualization import visualize_dataset_distribution

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BrainVoxel MLP Evaluation')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--experiment_dir', type=str, required=True, help='实验结果目录')
    parser.add_argument('--model_path', type=str, required=True, help='已训练模型的路径')
    parser.add_argument('--device', type=int, default=0, help='使用的设备（-1表示CPU）')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default=None, help='训练数据目录')
    parser.add_argument('--test_dir', type=str, default=None, help='测试数据目录')
    parser.add_argument('--val_dir', type=str, default=None, help='验证数据目录')
    
    # 其他参数
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    
    return parser.parse_args()

def setup_environment(config):
    """设置环境，包括随机种子和设备"""
    import random
    import torch
    
    # 设置随机种子
    random.seed(config.get('random_seed', 666))
    torch.manual_seed(config.get('random_seed', 666))
    torch.cuda.manual_seed(config.get('random_seed', 666))
    torch.cuda.manual_seed_all(config.get('random_seed', 666))
    np.random.seed(config.get('random_seed', 666))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置设备
    device = torch.device(f"cuda:{config.get('device', 0)}" if config.get('device', 0) >= 0 and torch.cuda.is_available() else "cpu")
    
    print(f"随机种子设置为: {config.get('random_seed', 666)}")
    print(f"使用设备: {device}")
    
    return device

def load_datasets(config):
    """加载数据集"""
    print("开始加载数据集...")
    
    # 构建数据目录字典
    data_dirs = {
        'train_dir': config.get('train_dir'),
        'test_dir': config.get('test_dir'),
        'val_dir': config.get('val_dir')
    }
    
    # 加载数据
    dataset_dict = load_multiclass_data(
        data_dirs,
        apply_pca_flag=config.get('apply_pca', False),
        n_components=config.get('n_pca', 0),
        norm=config.get('norm', True)
    )
    
    # 创建数据集
    train_dataset = BrainVoxelDataset(dataset_dict['train_samples'], dataset_dict['train_labels'])
    test_dataset = BrainVoxelDataset(dataset_dict['test_samples'], dataset_dict['test_labels'])
    val_dataset = BrainVoxelDataset(dataset_dict['val_samples'], dataset_dict['val_labels'])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
    
    print(f"数据加载完成! 共载入 {len(train_dataset)} 个训练样本，{len(val_dataset)} 个验证样本，{len(test_dataset)} 个测试样本")
    print(f"特征维度: {dataset_dict['feature_dim']}")
    
    return dataset_dict, train_loader, val_loader, test_loader

def load_model_from_checkpoint(model_path, model_type, input_dim, hidden_dims, num_classes, dropout_rate, activation, device):
    """从检查点加载模型"""
    print(f"从检查点加载模型: {model_path}")
    
    # 创建模型
    model = get_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        activation=activation
    )
    
    # 加载模型状态
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # 打印模型信息
    print(f"模型加载成功: {model_type}, 层大小: {hidden_dims}")
    
    return model, checkpoint

def evaluate_loaded_model(model, train_loader, val_loader, test_loader, device, result_path):
    """评估已加载的模型"""
    print("\n开始评估模型...")
    
    # 在训练集上评估
    print("\n在训练集上评估...")
    train_results = evaluate_model(
        model=model,
        data_loader=train_loader,
        device=device,
        result_path=result_path,
        dataset_name="train",
        detailed=True,
        plot=True
    )

    # 在验证集上评估
    print("\n在验证集上评估...")
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        result_path=result_path,
        dataset_name="val",
        detailed=True,
        plot=True
    )

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=result_path,
        dataset_name="test",
        detailed=True,
        plot=True
    )
    
    # 保存评估结果摘要
    summary_path = os.path.join(result_path, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("评估结果摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write("训练集结果:\n")
        f.write(f"  准确率: {train_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {train_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {train_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {train_results['kappa']:.4f}\n\n")
        
        f.write("验证集结果:\n")
        f.write(f"  准确率: {val_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {val_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {val_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {val_results['kappa']:.4f}\n\n")
        
        f.write("测试集结果:\n")
        f.write(f"  准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {test_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {test_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {test_results['kappa']:.4f}\n\n")
        
    print(f"评估结果摘要已保存至: {summary_path}")
    
    return train_results, val_results, test_results

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 校验路径参数
    if not os.path.exists(args.experiment_dir):
        print(f"错误: 实验目录 {args.experiment_dir} 不存在")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        sys.exit(1)
    
    # 加载配置
    config_path = args.config
    if config_path is None:
        # 尝试从实验目录加载配置
        possible_configs = [
            os.path.join(args.experiment_dir, "optimized_config.json"),
            os.path.join(args.experiment_dir, "config.json")
        ]
        for path in possible_configs:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        print("错误: 无法找到有效的配置文件")
        sys.exit(1)
    
    # 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 用命令行参数覆盖配置
    if args.train_dir:
        config['data_dirs'] = {
            'train_dir': args.train_dir,
            'test_dir': args.test_dir,
            'val_dir': args.val_dir
        }
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.device is not None:
        config['device'] = args.device
    
    # 设置环境
    device = setup_environment(config)
    
    # 加载数据集
    dataset_dict, train_loader, val_loader, test_loader = load_datasets(config)
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    try:
        # 从配置中获取模型参数
        model_type = config.get('model_type', 'base_mlp')
        hidden_dims = config.get('hidden_units', [4096, 4096, 4096, 4096])
        dropout_rate = config.get('dropout_rate', 0.5)
        activation = config.get('activation', 'relu')
        
        # 加载模型
        model, checkpoint = load_model_from_checkpoint(
            model_path=args.model_path,
            model_type=model_type,
            input_dim=dataset_dict['feature_dim'],
            hidden_dims=hidden_dims,
            num_classes=config.get('num_class', 102),
            dropout_rate=dropout_rate,
            activation=activation,
            device=device
        )
        
        # 评估模型
        train_results, val_results, test_results = evaluate_loaded_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            result_path=args.experiment_dir
        )
        
        # 输出最终结果
        print("\n评估完成!")
        print(f"测试集宏平均F1: {test_results['f1_macro']:.4f}")
        print(f"测试集准确率: {test_results['accuracy']:.4f}")
        print(f"测试集Kappa: {test_results['kappa']:.4f}")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()