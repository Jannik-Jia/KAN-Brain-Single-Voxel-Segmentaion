#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主脚本文件，处理命令行参数，执行实验流程（支持两种数据格式和Patientwise标准化）
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
from utils.label_processing import create_criterion_with_background_config, get_label_info_string
from config import print_background_config_info, validate_config
from experiment_json_logger import ExperimentJSONLogger, create_experiment_logger_from_config

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
from config import load_config, save_config, is_mat_format, print_data_split_info
from models import get_model
from data import load_data, BrainVoxelMatDataset, create_patient_aware_dataloader
from train import train_brain_voxel_mlp_multiclass
from utils.metrics import evaluate_model, calculate_class_weights, get_best_model, compare_class_performance
from utils.visualization import visualize_dataset_distribution, visualize_training_curves
from utils.optimization import run_bayesian_optimization
from utils.model_io import safe_load_model, load_model_with_architecture
from data.mat_loader import validate_filtered_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BrainVoxel MLP Training with Patientwise Standardization')
    
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
    
    # 🔧 新增：Patientwise标准化相关参数
    parser.add_argument('--standardization_method', type=str, default=None,
                        choices=['global', 'patientwise'], help='标准化方法')
    parser.add_argument('--val_prob_idx', type=int, nargs='+', default=None,
                        help='验证集患者ID列表，如 --val_prob_idx 20')
    parser.add_argument('--test_prob_idx', type=int, nargs='+', default=None,
                        help='测试集患者ID列表，如 --test_prob_idx 38')
    parser.add_argument('--use_patient_aware_sampler', action='store_true',
                        help='是否使用patient-aware批次采样器')
    parser.add_argument('--min_patients_per_batch', type=int, default=3,
                        help='每个批次最少包含的患者数（仅patient-aware采样器）')
    
    parser.add_argument('--apply_pca', action='store_true', help='是否应用PCA降维')
    parser.add_argument('--n_pca', type=int, default=None, help='PCA保留的主成分数量')
    
    # 🔧 修改：背景处理参数 - 默认不过滤背景
    parser.add_argument('--filter_background', type=lambda x: x.lower() == 'true', default=None, 
                        help='是否过滤背景像素 (True/False)，默认False')
    parser.add_argument('--include_background_in_classes', type=lambda x: x.lower() == 'true', default=None,
                        help='是否将背景作为分类类别 (True/False)，默认True')
    parser.add_argument('--background_label_target', type=int, default=None,
                        help='训练时的背景标签值 (默认0，作为一个类别)')
    
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
    parser.add_argument('--num_class', type=int, default=None, help='类别数量')
    
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



def detect_actual_num_classes(dataset_dict):
    """
    检测实际的类别数量（支持101和103类数据集）
    """
    all_labels = []
    for key in ['train_labels', 'val_labels', 'test_labels']:
        if key in dataset_dict and dataset_dict[key] is not None:
            labels = dataset_dict[key]
            if len(labels) > 0:
                all_labels.extend(labels.flatten())
    
    unique_labels = np.unique(all_labels)
    actual_num_classes = len(unique_labels)
    
    print(f"\n🔍 检测实际类别数量:")
    print(f"  唯一标签: {unique_labels[:10]}...{unique_labels[-10:] if len(unique_labels) > 20 else ''}")
    print(f"  标签范围: {unique_labels.min()} - {unique_labels.max()}")
    print(f"  实际类别数: {actual_num_classes}")
    
    # 检查是否有缺失的标签
    expected_labels = np.arange(unique_labels.min(), unique_labels.max() + 1)
    missing_labels = set(expected_labels) - set(unique_labels)
    if missing_labels:
        print(f"  ⚠️ 缺失的标签: {sorted(missing_labels)}")
    
    # 识别数据集类型
    if actual_num_classes == 101 and unique_labels.max() == 100:
        print("  ✅ 检测到101类数据集（已移除标签43）")
    elif actual_num_classes == 103 and unique_labels.max() == 102:
        print("  ✅ 检测到103类原始数据集")
    else:
        print(f"  ⚠️ 非标准数据集，请检查标签")
    
    return actual_num_classes, unique_labels

def train_and_evaluate(config, model, dataset_dict, train_loader, val_loader, test_loader, device):
    """训练和评估模型"""
    
    # 🔧 打印背景处理配置信息
    print_background_config_info(config)
    
    # 🔧 检测实际类别数量
    actual_num_classes, unique_labels = detect_actual_num_classes(dataset_dict)
    
    # 如果检测到的类别数与配置不同，更新配置
    if actual_num_classes != config['num_class']:
        print(f"\n⚠️ 警告: 检测到的类别数({actual_num_classes})与配置({config['num_class']})不同")
        print(f"   更新配置中的类别数为: {actual_num_classes}")
        config['num_class'] = actual_num_classes
        
        # 如果模型输出维度不匹配，需要重建模型
        if model.output_layer.out_features != actual_num_classes:
            print(f"   重建模型以匹配实际类别数...")
            model = get_model(
                model_type=config['model_type'],
                input_dim=config['feature_dim'],
                hidden_dims=config['hidden_units'],
                num_classes=actual_num_classes,
                dropout_rate=config['dropout_rate'],
                activation=config['activation']
            ).to(device)
    
    # 计算类别权重
    class_weights = calculate_class_weights(
        dataset_dict['train_labels'], 
        config['num_class'],
        config  
    ).to(device)
    
    # 创建损失函数
    criterion = create_criterion_with_background_config(config, class_weights)
    
    # 打印损失函数信息
    ignore_idx = getattr(criterion, 'ignore_index', None)
    print(f"\n📊 损失函数配置:")
    print(f"  类型: CrossEntropyLoss")
    print(f"  类别权重: {'是' if class_weights is not None else '否'}")
    print(f"  ignore_index: {ignore_idx if ignore_idx is not None else '无'}")
    print(f"  标签处理: {get_label_info_string(config)}")
    print(f"  实际类别数: {actual_num_classes}")
    
    # 创建优化器
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # 创建学习率调度器
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

    # 计算标准化参数（仅用于记录，patientwise已经在数据加载时处理）
    normalization_params = None
    if config.get('standardization_method') == 'global' and config['norm']:
        if 'train_samples' in dataset_dict:
            train_samples = dataset_dict['train_samples']
            mean = np.mean(train_samples, axis=0)
            std = np.std(train_samples, axis=0)
            std[std == 0] = 1e-10
            
            normalization_params = {
                'mean': mean.tolist(),
                'std': std.tolist()
            }
            print("已记录全局标准化参数")
    elif config.get('standardization_method') == 'patientwise':
        print("使用Patientwise标准化，每个患者独立计算统计量")

    # 开始训练
    print("\n开始训练模型...")
    print(f"总轮数: {config['epochs']}, 批大小: {config['batch_size']}, 学习率: {config['lr']}")
    print(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
    print(f"标准化方法: {config.get('standardization_method', 'global')}")
    print(f"类别数量: {config['num_class']} ({get_label_info_string(config)})")
    
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
        show_class_metrics=True,
        config=config
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
        show_class_metrics=True,
        config=config
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
        show_class_metrics=True,
        config=config
    )
    
    # 保存评估结果摘要
    summary_path = os.path.join(config['save_dir'], "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("评估结果摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}\n")
        f.write(f"标准化方法: {config.get('standardization_method', 'global')}\n")
        f.write(f"背景处理: {get_label_info_string(config)}\n")
        f.write(f"实际类别数: {actual_num_classes}\n\n")
        
        f.write("训练集结果:\n")
        f.write(f"  样本数: {len(dataset_dict['train_samples'])}\n")
        f.write(f"  患者数: {len(dataset_dict.get('train_prob_idx', []))}\n")
        f.write(f"  准确率: {train_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {train_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {train_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {train_results['kappa']:.4f}\n\n")
        
        f.write("验证集结果:\n")
        f.write(f"  样本数: {len(dataset_dict['val_samples'])}\n")
        f.write(f"  患者ID: {dataset_dict.get('val_prob_idx_actual', [])}\n")
        f.write(f"  准确率: {val_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {val_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {val_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {val_results['kappa']:.4f}\n\n")
        
        f.write("测试集结果:\n")
        f.write(f"  样本数: {len(dataset_dict['test_samples'])}\n")
        f.write(f"  患者ID: {dataset_dict.get('test_prob_idx_actual', [])}\n")
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
        f.write(f"背景处理模式: {'过滤背景' if config.get('filter_background', False) else '保留背景'}\n")
        if not config.get('filter_background', False):
            f.write(f"背景分类: {'是' if config.get('include_background_in_classes', True) else '否'}\n")
    
    # 比较训练集、验证集和测试集中的类别性能
    compare_results = compare_class_performance(
        results_list=[train_results, val_results, test_results],
        dataset_names=["Train", "Validation", "Test"],
        result_path=config['save_dir']
    )
    
    print(f"评估完成，结果摘要已保存至{summary_path}")
    
    return train_results, val_results, test_results, best_model_path


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 🔧 设置默认值：不过滤背景，背景作为一个类别
    if 'filter_background' not in config:
        config['filter_background'] = False
    if 'include_background_in_classes' not in config:
        config['include_background_in_classes'] = True
    if 'background_label_target' not in config:
        config['background_label_target'] = 0
    
    # 🔧 设置默认的标准化方法和患者划分
    if 'standardization_method' not in config:
        config['standardization_method'] = 'patientwise'
    if 'val_prob_idx' not in config:
        config['val_prob_idx'] = [20]
    if 'test_prob_idx' not in config:
        config['test_prob_idx'] = [38]
    
    # 用命令行参数覆盖配置
    for key, value in vars(args).items():
        if value is not None:
            if key == 'hidden_units' and isinstance(value, str):
                config[key] = [int(x) for x in value.split(',')]
            elif key in ['val_prob_idx', 'test_prob_idx']:
                config[key] = value
            elif key in ['standardization_method', 'use_patient_aware_sampler', 'min_patients_per_batch']:
                config[key] = value
            elif key in config or key in ['filter_background', 'include_background_in_classes', 'background_label_target']:
                config[key] = value
            elif key in ['train_dir', 'test_dir', 'val_dir']:
                if 'data_dirs' not in config:
                    config['data_dirs'] = {}
                config['data_dirs'][key] = value
    
    # 验证配置
    if not validate_config(config):
        print("❌ 配置验证失败，请检查配置")
        return
    
    # 根据背景处理设置调整类别数
    if not config.get('filter_background', False) and config.get('include_background_in_classes', True):
        cur_num_class = config.get('num_class', 101)
        if cur_num_class == 101:
            print("✅ 使用101类数据集，背景作为第0类")
            config['num_class'] = 101  # 100有效类别 + 1背景
        else:
            print(f"⚠️ 非常规类别设置：当前num_class={cur_num_class}，使用配置值")


    
    # 设置实验名称
    if not config.get('experiment_name'):
        data_format = 'MAT' if is_mat_format(config) else 'Original'
        standardization = config.get('standardization_method', 'global')
        bg_suffix = ""
        if not config.get('filter_background', False):
            if config.get('include_background_in_classes', True):
                bg_suffix = "_BGClass"
            else:
                bg_suffix = "_BGIgnore"
        else:
            bg_suffix = "_BGFilter"
        config['experiment_name'] = f"{config['model_name']}_{data_format}_{standardization}{bg_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
    
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

    # 加载数据集
    print(f"使用数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
    print(f"使用标准化方法: {config.get('standardization_method', 'global')}")
    dataset_dict, train_loader, val_loader, test_loader = load_data(config, mode='train')
    
    # 打印数据信息
    print_data_split_info(config)
    
    # 验证数据
    if config.get('filter_background', False):
        validate_filtered_data(dataset_dict['train_samples'], dataset_dict['train_labels'], "最终训练数据")
        validate_filtered_data(dataset_dict['val_samples'], dataset_dict['val_labels'], "最终验证数据")
        validate_filtered_data(dataset_dict['test_samples'], dataset_dict['test_labels'], "最终测试数据")
    
    # 🔧 如果使用patient-aware采样器，重新创建训练数据加载器
    if config.get('use_patient_aware_sampler', False) and is_mat_format(config):
        print("\n使用Patient-aware批次采样器...")
        
        # 获取训练集的患者索引
        train_prob_idx = dataset_dict.get('train_prob_idx_actual', [])
        if len(train_prob_idx) > 0:
            # 重新创建训练数据集
            train_dataset = BrainVoxelMatDataset(
                dataset_dict['train_samples'], 
                dataset_dict['train_labels']
            )
            
            # 创建patient-aware数据加载器
            train_loader = create_patient_aware_dataloader(
                train_dataset,
                train_prob_idx,
                batch_size=config.get('batch_size', 128),
                sampler_type='aware',
                min_patients_per_batch=config.get('min_patients_per_batch', 3),
                drop_last=False,
                num_workers=0
            )
            print(f"已创建Patient-aware训练数据加载器")
    
    # 更新配置中的特征维度
    config['feature_dim'] = dataset_dict['feature_dim']
    
    # 创建或优化模型
    if config['run_bayesian_opt']:
        print("开始贝叶斯优化...")
        
        # 定义参数空间
        param_space = {
            'learning_rate': (1e-6, 1e-3, 'log'),
            'batch_size': [64, 128, 256],
            'weight_decay': (1e-6, 1e-3, 'log'),
            'dropout_rate': (0.1, 0.7),
            'activation': ['relu', 'gelu', 'swish'],
            'optimizer': ['adam', 'adamw'],
            'lr_scheduler': ['cosine', 'plateau', 'none'],
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
    print(f"标准化方法: {config.get('standardization_method', 'global')}")
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
        f.write(f"数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}\n")
        f.write(f"标准化方法: {config.get('standardization_method', 'global')}\n\n")
        
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

    # ========== 生成 Experiment JSON ==========
    try:
        print("\n" + "="*50)
        print("生成 Experiment JSON 记录...")

        # 创建 ExperimentJSONLogger
        exp_logger = create_experiment_logger_from_config(config)

        # 设置模型参数量
        exp_logger.set_model_param_count(model)

        # 设置体素数量
        exp_logger.set_voxel_counts(
            train_count=len(dataset_dict['train_samples']),
            val_count=len(dataset_dict['val_samples']),
            test_count=len(dataset_dict['test_samples'])
        )

        # 设置训练集被试
        train_subjects = dataset_dict.get('train_prob_idx_actual', [])
        if train_subjects:
            exp_logger.set_train_subjects(train_subjects)

        # 从评估结果记录指标
        exp_logger.log_results_from_eval_dict(test_results, split="test")

        # 设置结果文件路径
        exp_logger.set_result_paths(save_dir, config['experiment_name'])

        # 如果是贝叶斯优化选出的配置，添加备注
        if config.get('run_bayesian_opt', False):
            exp_logger.add_bayesopt_info(
                study_name=f"{config['model_name']}_bayesian_opt",
                best_trial_number=0  # TODO: 从优化结果中获取
            )

        # 保存 JSON
        json_path = exp_logger.save(save_dir, validate=True)
        print(f"Experiment JSON 已保存至: {json_path}")

    except Exception as e:
        print(f"生成 Experiment JSON 时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()