#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主脚本文件，处理命令行参数，执行实验流程（支持两种数据格式）
"""

import os
import sys
import json
import time
import random
import argparse
import torch
import torch.nn as nn
import numpy as np

# 设置PyTorch序列化安全变量
try:
    safe_globals = [
        np.dtype,
        np.core.multiarray.scalar,
        np.ndarray,
        np.generic,
        np.float64,
        np.float32,
        np.int64,
        np.int32
    ]
    torch.serialization.add_safe_globals(safe_globals)
    print("已添加numpy类型到PyTorch安全全局变量列表")
except Exception as e:
    print(f"添加安全全局变量时出错 (可忽略): {e}")
    
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 导入自定义模块
from config import load_config, save_config, is_mat_format
from models import get_model
from data import load_data  # 使用统一的数据加载接口
from train import train_brain_voxel_mlp_multiclass
from utils.metrics import evaluate_model, calculate_class_weights, get_best_model, compare_class_performance
from utils.visualization import visualize_dataset_distribution, visualize_training_curves
from utils.optimization import run_bayesian_optimization
from utils.model_io import safe_load_model, load_model_with_architecture

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BrainVoxel MLP Training')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称，默认使用时间戳')
    parser.add_argument('--device', type=int, default=None, help='使用的设备（-1表示CPU）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    # 数据参数 - 支持两种格式
    parser.add_argument('--train_dir', type=str, default=None, help='训练数据目录（原版格式）')
    parser.add_argument('--test_dir', type=str, default=None, help='测试数据目录（原版格式）')
    parser.add_argument('--val_dir', type=str, default=None, help='验证数据目录（原版格式）')
    parser.add_argument('--mat_file_path', type=str, default=None, help='TRAIN38.mat文件路径（MAT格式）')
    parser.add_argument('--demo_mat_path', type=str, default=None, help='用于评估的DEMO38.mat文件路径')
    parser.add_argument('--test_size', type=float, default=None, help='测试集比例（MAT格式）')
    
    parser.add_argument('--apply_pca', action='store_true', help='是否应用PCA降维')
    parser.add_argument('--n_pca', type=int, default=None, help='PCA保留的主成分数量')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default=None, 
                        choices=['base_mlp', 'deep_mlp', 'residual_mlp'], 
                        help='模型类型')
    parser.add_argument('--hidden_units', type=str, default=None, 
                        help='隐藏层大小，逗号分隔的整数，例如"4096,4096,4096,4096"')
    parser.add_argument('--dropout_rate', type=float, default=None, help='Dropout比率')
    parser.add_argument('--activation', type=str, default=None, 
                        choices=['relu', 'gelu', 'swish'], help='激活函数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--val_epochs', type=int, default=None, help='验证频率')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default=None, 
                        choices=['adam', 'adamw'], help='优化器')
    
    # 学习率调度参数
    parser.add_argument('--use_lr_scheduler', action='store_true', help='是否使用学习率调度')
    parser.add_argument('--lr_scheduler_type', type=str, default=None, 
                        choices=['multistep', 'cosine', 'plateau'], help='学习率调度器类型')
    
    # 优化参数
    parser.add_argument('--run_bayesian_opt', action='store_true', help='是否运行贝叶斯优化')
    parser.add_argument('--n_trials', type=int, default=None, help='贝叶斯优化的试验次数')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    parser.add_argument('--old_serialization', action='store_true', help='使用旧的PyTorch序列化格式')
    
    return parser.parse_args()

def setup_environment(config):
    """设置环境，包括随机种子和设备"""
    # 设置随机种子
    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置设备
    device = torch.device(f"cuda:{config['device']}" if config['device'] >= 0 and torch.cuda.is_available() else "cpu")
    
    print(f"随机种子设置为: {config['random_seed']}")
    print(f"使用设备: {device}")
    
    return device




def train_and_evaluate(config, model, dataset_dict, train_loader, val_loader, test_loader, device):
    """训练和评估模型"""
   
    # 计算类别权重（包括背景类别，处理不平衡问题）
    # class_weights = calculate_class_weights(
    #     dataset_dict['train_labels'], 
    #     config['num_class']  # 这里应该是102
    # ).to(device)
    
    # 创建损失函数 - 移除ignore_index参数
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss() 
    
    # 创建优化器
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # 创建学习率调度器 - 这部分保持不变
    lr_scheduler = None
    if config['use_lr_scheduler']:
        if config['lr_scheduler_type'] == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        elif config['lr_scheduler_type'] == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=config['lr_milestones'], gamma=config['lr_gamma']
            )
        elif config['lr_scheduler_type'] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=config['lr_gamma'], patience=5, verbose=True
            )

    # 计算标准化参数
    normalization_params = None
    if config['norm']:
        # 从数据集中获取标准化参数
        if 'train_samples' in dataset_dict:
            train_samples = dataset_dict['train_samples']
            mean = np.mean(train_samples, axis=0)
            std = np.std(train_samples, axis=0)
            # 避免除零
            std[std == 0] = 1e-10
            
            # 创建标准化参数字典
            normalization_params = {
                'mean': mean.tolist(),  # 转为列表以确保可JSON序列化
                'std': std.tolist()
            }
            
            print("已计算标准化参数")
        else:
            print("警告: 无法计算标准化参数，因为没有找到训练样本")

    # 开始训练
    print("\n开始训练模型...")
    print(f"总轮数: {config['epochs']}, 批大小: {config['batch_size']}, 学习率: {config['lr']}")
    print(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
    print(f"类别数量: {config['num_class']} (包括背景)")  # 修改这个提示信息

    # 训练模型
    training_results = train_brain_voxel_mlp_multiclass(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['epochs'],
        val_epoch=config['val_epochs'],
        save_path=config['save_dir'],
        lr_scheduler=lr_scheduler,
        use_old_zipfile_serialization=config.get('use_old_zipfile_serialization', True),
        experiment_name=config.get('experiment_name', time.strftime("%Y%m%d_%H%M%S")),
        config=config,
        normalization_params=normalization_params
        # ignore_index=ignore_index  # 传递背景标签索引
    )
    
    # 可视化训练过程
    try:
        visualize_training_curves(
            training_results,
            save_path=os.path.join(config['save_dir'], "training_curves.png")
        )
    except Exception as e:
        print(f"可视化训练曲线时出错: {e}")
        
    # 获取最佳模型
    try:
        best_model_path = get_best_model(
            training_results['val_f1_macro_list'],
            training_results['val_epoch_list'],
            config['save_dir'],
            metric='f1'
        )
        
        # 使用安全加载工具
        checkpoint = safe_load_model(best_model_path, device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"成功加载最佳模型: {os.path.basename(best_model_path)}")
        
    except Exception as e:
        print(f"加载最佳模型时出错: {e}")
        print("将使用当前模型继续评估")

    # 评估模型
    print("\n使用最佳模型进行评估...")

    # 在训练集上评估
    print("\n在训练集上评估...")
    train_results = evaluate_model(
        model=model,
        data_loader=train_loader,
        device=device,
        result_path=config['save_dir'],
        dataset_name="train",
        detailed=True,
        plot=True,
        disable_progress=True,
        show_class_metrics=True
        # ignore_index=ignore_index
    )

    # 在验证集上评估
    print("\n在验证集上评估...")
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        result_path=config['save_dir'],
        dataset_name="val",
        detailed=True,
        plot=True,
        disable_progress=True,
        show_class_metrics=True
        # ignore_index=ignore_index
    )

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=config['save_dir'],
        dataset_name="test",
        detailed=True,
        plot=True,
        disable_progress=True,
        show_class_metrics=True
        # ignore_index=ignore_index
    )
    
    # 保存评估结果摘要
    summary_path = os.path.join(config['save_dir'], "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("评估结果摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}\n")
        # f.write(f"背景标签索引: {ignore_index}\n\n")
        
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
        
        f.write(f"模型: {config['model_type']}\n")
        f.write(f"隐藏层: {config['hidden_units']}\n")
        f.write(f"激活函数: {config['activation']}\n")
        f.write(f"Dropout率: {config['dropout_rate']}\n")
        f.write(f"优化器: {config['optimizer']}\n")
        f.write(f"学习率: {config['lr']}\n")
        f.write(f"权重衰减: {config['weight_decay']}\n")
    
    # 比较训练集、验证集和测试集中的类别性能
    compare_results = compare_class_performance(
        results_list=[train_results, val_results, test_results],
        dataset_names=["Train", "Validation", "Test"],
        result_path=config['save_dir']
    )
    
    print(f"评估完成，结果摘要已保存至{summary_path}")
    
    return train_results, val_results, test_results, best_model_path


# def train_and_evaluate(config, model, dataset_dict, train_loader, val_loader, test_loader, device):
#     """训练和评估模型"""
#     # 根据数据格式确定背景标签索引
#     ignore_index = config.get('ignore_index', -1 if not is_mat_format(config) else 0)
    
#     # 计算类别权重（处理不平衡问题）
#     class_weights = calculate_class_weights(dataset_dict['train_labels'], config['num_class']).to(device)
    
#     # 创建损失函数
#     criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    
#     # 创建优化器
#     if config['optimizer'] == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
#     else:  # adamw
#         optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
#     # 创建学习率调度器
#     lr_scheduler = None
#     if config['use_lr_scheduler']:
#         if config['lr_scheduler_type'] == 'cosine':
#             lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
#         elif config['lr_scheduler_type'] == 'multistep':
#             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#                 optimizer, milestones=config['lr_milestones'], gamma=config['lr_gamma']
#             )
#         elif config['lr_scheduler_type'] == 'plateau':
#             lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='max', factor=config['lr_gamma'], patience=5, verbose=True
#             )

#     # 计算标准化参数
#     normalization_params = None
#     if config['norm']:
#         # 从数据集中获取标准化参数
#         if 'train_samples' in dataset_dict:
#             train_samples = dataset_dict['train_samples']
#             mean = np.mean(train_samples, axis=0)
#             std = np.std(train_samples, axis=0)
#             # 避免除零
#             std[std == 0] = 1e-10
            
#             # 创建标准化参数字典
#             normalization_params = {
#                 'mean': mean.tolist(),  # 转为列表以确保可JSON序列化
#                 'std': std.tolist()
#             }
            
#             print("已计算标准化参数")
#         else:
#             print("警告: 无法计算标准化参数，因为没有找到训练样本")

#     # 开始训练
#     print("\n开始训练模型...")
#     print(f"总轮数: {config['epochs']}, 批大小: {config['batch_size']}, 学习率: {config['lr']}")
#     print(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
#     print(f"背景标签索引: {ignore_index}")

#     # 训练模型
#     training_results = train_brain_voxel_mlp_multiclass(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         device=device,
#         num_epochs=config['epochs'],
#         val_epoch=config['val_epochs'],
#         save_path=config['save_dir'],
#         lr_scheduler=lr_scheduler,
#         use_old_zipfile_serialization=config.get('use_old_zipfile_serialization', True),
#         experiment_name=config.get('experiment_name', time.strftime("%Y%m%d_%H%M%S")),
#         config=config,
#         normalization_params=normalization_params,
#         ignore_index=ignore_index  # 传递背景标签索引
#     )
    
#     # 可视化训练过程
#     try:
#         visualize_training_curves(
#             training_results,
#             save_path=os.path.join(config['save_dir'], "training_curves.png")
#         )
#     except Exception as e:
#         print(f"可视化训练曲线时出错: {e}")
        
#     # 获取最佳模型
#     try:
#         best_model_path = get_best_model(
#             training_results['val_f1_macro_list'],
#             training_results['val_epoch_list'],
#             config['save_dir'],
#             metric='f1'
#         )
        
#         # 使用安全加载工具
#         checkpoint = safe_load_model(best_model_path, device)
#         model.load_state_dict(checkpoint['state_dict'])
#         print(f"成功加载最佳模型: {os.path.basename(best_model_path)}")
        
#     except Exception as e:
#         print(f"加载最佳模型时出错: {e}")
#         print("将使用当前模型继续评估")

#     # 评估模型
#     print("\n使用最佳模型进行评估...")

#     # 在训练集上评估
#     print("\n在训练集上评估...")
#     train_results = evaluate_model(
#         model=model,
#         data_loader=train_loader,
#         device=device,
#         result_path=config['save_dir'],
#         dataset_name="train",
#         detailed=True,
#         plot=True,
#         disable_progress=True,
#         show_class_metrics=True,
#         ignore_index=ignore_index
#     )

#     # 在验证集上评估
#     print("\n在验证集上评估...")
#     val_results = evaluate_model(
#         model=model,
#         data_loader=val_loader,
#         device=device,
#         result_path=config['save_dir'],
#         dataset_name="val",
#         detailed=True,
#         plot=True,
#         disable_progress=True,
#         show_class_metrics=True,
#         ignore_index=ignore_index
#     )

#     # 在测试集上评估
#     print("\n在测试集上评估...")
#     test_results = evaluate_model(
#         model=model,
#         data_loader=test_loader,
#         device=device,
#         result_path=config['save_dir'],
#         dataset_name="test",
#         detailed=True,
#         plot=True,
#         disable_progress=True,
#         show_class_metrics=True,
#         ignore_index=ignore_index
#     )
    
#     # 保存评估结果摘要
#     summary_path = os.path.join(config['save_dir'], "evaluation_summary.txt")
#     with open(summary_path, 'w') as f:
#         f.write("评估结果摘要\n")
#         f.write("="*50 + "\n\n")
        
#         f.write(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}\n")
#         f.write(f"背景标签索引: {ignore_index}\n\n")
        
#         f.write("训练集结果:\n")
#         f.write(f"  准确率: {train_results['accuracy']:.4f}\n")
#         f.write(f"  宏平均F1: {train_results['f1_macro']:.4f}\n")
#         f.write(f"  平衡准确率: {train_results['balanced_accuracy']:.4f}\n")
#         f.write(f"  Kappa系数: {train_results['kappa']:.4f}\n\n")
        
#         f.write("验证集结果:\n")
#         f.write(f"  准确率: {val_results['accuracy']:.4f}\n")
#         f.write(f"  宏平均F1: {val_results['f1_macro']:.4f}\n")
#         f.write(f"  平衡准确率: {val_results['balanced_accuracy']:.4f}\n")
#         f.write(f"  Kappa系数: {val_results['kappa']:.4f}\n\n")
        
#         f.write("测试集结果:\n")
#         f.write(f"  准确率: {test_results['accuracy']:.4f}\n")
#         f.write(f"  宏平均F1: {test_results['f1_macro']:.4f}\n")
#         f.write(f"  平衡准确率: {test_results['balanced_accuracy']:.4f}\n")
#         f.write(f"  Kappa系数: {test_results['kappa']:.4f}\n\n")
        
#         f.write(f"模型: {config['model_type']}\n")
#         f.write(f"隐藏层: {config['hidden_units']}\n")
#         f.write(f"激活函数: {config['activation']}\n")
#         f.write(f"Dropout率: {config['dropout_rate']}\n")
#         f.write(f"优化器: {config['optimizer']}\n")
#         f.write(f"学习率: {config['lr']}\n")
#         f.write(f"权重衰减: {config['weight_decay']}\n")
    
#     # 比较训练集、验证集和测试集中的类别性能
#     compare_results = compare_class_performance(
#         results_list=[train_results, val_results, test_results],
#         dataset_names=["Train", "Validation", "Test"],
#         result_path=config['save_dir']
#     )
    
#     print(f"评估完成，结果摘要已保存至{summary_path}")
    
#     return train_results, val_results, test_results, best_model_path

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 用命令行参数覆盖配置
    for key, value in vars(args).items():
        if value is not None:
            if key == 'hidden_units' and isinstance(value, str):
                config[key] = [int(x) for x in value.split(',')]
            elif key in config:  # 只更新配置中存在的键
                config[key] = value
            elif key in ['train_dir', 'test_dir', 'val_dir']:
                # 处理数据目录
                if 'data_dirs' not in config:
                    config['data_dirs'] = {}
                config['data_dirs'][key] = value
    
    # 设置实验名称
    if not config.get('experiment_name'):
        data_format = 'MAT' if is_mat_format(config) else 'Original'
        config['experiment_name'] = f"{config['model_name']}_{data_format}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 创建保存目录
    save_dir = os.path.join(config['save_dir'], config['experiment_name'])
    config['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(config['log_dir'], config['experiment_name'])
    config['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(save_dir, "config.json"))
    
    # 设置环境
    device = setup_environment(config)
    
    # 加载数据集 - 使用统一接口
    print(f"使用数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
    dataset_dict, train_loader, val_loader, test_loader = load_data(config, mode='train')
    
    # 更新配置中的特征维度
    config['feature_dim'] = dataset_dict['feature_dim']
    
    # 创建或优化模型
    if config['run_bayesian_opt']:
        print("开始贝叶斯优化...")
        
        # 定义参数空间 - 保持与原版一致
        param_space = {
            'learning_rate': (1e-6, 1e-3, 'log'),
            'batch_size': [64, 128, 256, 512],
            'weight_decay': (1e-6, 1e-3, 'log'),
            'dropout_rate': (0.1, 0.7),
            'activation': ['relu', 'gelu', 'swish'],
            'optimizer': ['adam', 'adamw'],
            'lr_scheduler': ['cosine', 'step', 'plateau', 'none'],
            'model_type': ['base_mlp', 'deep_mlp', 'residual_mlp'],
        }
        
        # 准备数据加载器
        data_loaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # 运行贝叶斯优化
        study, best_params = run_bayesian_optimization(
            data_loaders=data_loaders,
            input_dim=dataset_dict['feature_dim'],
            num_classes=config['num_class'],
            device=device,
            param_space=param_space,
            n_trials=config['n_trials'],
            study_name=f"{config['model_name']}_bayesian_opt",
            save_path=config['save_dir'],
            config=config
        )
        
        # 更新配置
        config.update({
            'model_type': best_params.get('model_type', config['model_type']),
            'dropout_rate': best_params.get('dropout_rate', config['dropout_rate']),
            'activation': best_params.get('activation', config['activation']),
            'optimizer': best_params.get('optimizer', config['optimizer']),
            'lr': best_params.get('learning_rate', config['lr']),
            'weight_decay': best_params.get('weight_decay', config['weight_decay']),
            'lr_scheduler_type': best_params.get('lr_scheduler', config['lr_scheduler_type']),
        })
        
        # 根据最佳模型类型设置隐藏层配置
        model_type = best_params.get('model_type')
        if model_type == 'deep_mlp':
            if 'depth' in best_params and 'width_factor' in best_params:
                depth = best_params.get('depth', 6)
                width = best_params.get('width_factor', 2048)
                config['hidden_units'] = [width] * depth
            else:
                config['hidden_units'] = [2048] * 6
                
        elif model_type in ['base_mlp', 'residual_mlp']:
            if 'layer_sizes_idx' in best_params:
                layer_sizes_idx = best_params.get('layer_sizes_idx', 0)
                if model_type == 'base_mlp':
                    layer_sizes_options = [
                        [4096, 4096, 4096, 4096],
                        [3072, 3072, 3072, 3072],
                        [2048, 2048, 2048, 2048]
                    ]
                else:  # residual_mlp
                    layer_sizes_options = [
                        [4096, 4096, 4096, 4096],
                        [2048, 2048, 2048, 2048],
                        [1024, 2048, 2048, 1024],
                        [4096, 2048, 2048, 4096]
                    ]
                
                if layer_sizes_idx < len(layer_sizes_options):
                    config['hidden_units'] = layer_sizes_options[layer_sizes_idx]
        
        # 保存更新后的配置
        save_config(config, os.path.join(config['save_dir'], "optimized_config.json"))
        
        print("贝叶斯优化完成，已更新最优参数")
    
    # 创建模型
    model = get_model(
        model_type=config['model_type'],
        input_dim=dataset_dict['feature_dim'],
        hidden_dims=config['hidden_units'],
        num_classes=config['num_class'],
        dropout_rate=config['dropout_rate'],
        activation=config['activation']
    )
    
    print(f"创建{config['model_type']}模型，隐藏层: {config['hidden_units']}, 激活函数: {config['activation']}")
    model = model.to(device)
    
    # 训练和评估模型
    train_results, val_results, test_results, best_model_path = train_and_evaluate(
        config, model, dataset_dict, train_loader, val_loader, test_loader, device
    )
    
    # 打印最终结果
    print("\n实验完成!")
    print(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
    print(f"最佳模型: {os.path.basename(best_model_path)}")
    print(f"测试集准确率: {test_results['accuracy']:.4f}")
    print(f"测试集宏平均F1: {test_results['f1_macro']:.4f}")
    print(f"测试集Kappa系数: {test_results['kappa']:.4f}")
    print(f"所有结果已保存至: {save_dir}")
    
    # 创建最终报告
    final_report_path = os.path.join(save_dir, "final_report.txt")
    with open(final_report_path, 'w') as f:
        f.write(f"脑体素分类实验最终报告\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"实验名称: {config['experiment_name']}\n")
        f.write(f"日期时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}\n\n")
        
        f.write(f"模型架构: {config['model_type']}\n")
        f.write(f"隐藏层配置: {config['hidden_units']}\n")
        f.write(f"激活函数: {config['activation']}\n")
        f.write(f"Dropout率: {config['dropout_rate']}\n")
        f.write(f"优化器: {config['optimizer']}\n")
        f.write(f"学习率: {config['lr']}\n")
        f.write(f"权重衰减: {config['weight_decay']}\n\n")
        
        f.write(f"训练集性能:\n")
        f.write(f"  准确率: {train_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {train_results['f1_macro']:.4f}\n")
        f.write(f"  Kappa系数: {train_results['kappa']:.4f}\n\n")
        
        f.write(f"验证集性能:\n")
        f.write(f"  准确率: {val_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {val_results['f1_macro']:.4f}\n")
        f.write(f"  Kappa系数: {val_results['kappa']:.4f}\n\n")
        
        f.write(f"测试集性能:\n")
        f.write(f"  准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {test_results['f1_macro']:.4f}\n")
        f.write(f"  Kappa系数: {test_results['kappa']:.4f}\n\n")
        
        f.write(f"最佳模型保存路径: {best_model_path}\n")
        f.write(f"所有结果保存目录: {save_dir}\n")
    
    print(f"最终报告已保存至: {final_report_path}")

if __name__ == "__main__":
    main()