#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepMLP 6x2048 训练主程序
简化版：固定配置，无 BayesOpt，无验证集
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
from torch.utils.data import DataLoader

# 导入自定义模块
from config_deepmlp import load_config, save_config, print_config
from models import get_model
from data.single_mat_loader import create_train_val_test_datasets
from train import train_brain_voxel_mlp_multiclass
from utils.metrics import evaluate_model, calculate_class_weights


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepMLP 6x2048 Training')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--test_subject', type=str, default=None, help='测试被试名称')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--device', type=int, default=None, help='GPU 设备 ID')
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录')
    return parser.parse_args()


def setup_environment(config):
    """设置环境"""
    # 设置随机种子
    seed = config['random_seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置设备
    if config['device'] >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']}")
        print(f"使用 GPU: {torch.cuda.get_device_name(config['device'])}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    return device


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.test_subject:
        config['test_subject'] = args.test_subject
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.device is not None:
        config['device'] = args.device
    if args.save_dir:
        config['save_dir'] = args.save_dir

    # 打印配置
    print_config(config)

    # 设置实验名称
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"DeepMLP_6x2048_{config['test_subject']}_{timestamp}"
    config['experiment_name'] = experiment_name

    # 创建保存目录
    save_dir = os.path.join(config['save_dir'], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir

    # 保存配置
    save_config(config, os.path.join(save_dir, "config.json"))

    # 设置环境
    device = setup_environment(config)
    if device.type != "cuda":
        print("检测到未使用GPU，终止训练以避免CPU运行。请检查CUDA/驱动/设备ID后重试。")
        return

    # ==================== 加载数据 ====================
    print("\n加载数据...")
    (train_dataset, val_dataset, test_dataset,
     train_subjects, val_subjects, test_subject) = create_train_val_test_datasets(
        data_dir_1d=config['data_dir_1d'],
        data_dir_3d=config['data_dir_3d'],
        test_subject_name=config['test_subject'],
        val_ratio=0.1,  # 10% 作为验证集
        feature_dim=config['feature_dim'],
        use_zscore=config['use_zscore'],
        random_seed=config['random_seed']
    )

    # 检测标签范围，确保覆盖所有可能标签（即便单个被试缺类）
    train_min, train_max = train_dataset.all_labels.min(), train_dataset.all_labels.max()
    val_min, val_max = val_dataset.all_labels.min(), val_dataset.all_labels.max()
    test_min, test_max = test_dataset.labels.min(), test_dataset.labels.max()

    overall_min = min(train_min, val_min, test_min)
    overall_max = max(train_max, val_max, test_max)

    # 固定输出至少覆盖 102 类，避免标签稀疏导致 num_class 过小
    target_num_classes = max(102, int(overall_max) + 1)

    if overall_min < 0:
        raise ValueError(f"标签存在负值，最小值为 {overall_min}")
    if overall_max >= target_num_classes:
        raise ValueError(
            f"标签最大值 {overall_max} 超过 num_class 上限 {target_num_classes - 1}"
        )

    print(f"\n标签范围 (train/val/test): {overall_min} - {overall_max}")
    print(f"设置 num_class = {target_num_classes}（覆盖全类以防缺失类别）")
    config['num_class'] = target_num_classes

    # 配置早停参数（仅用于监控记录，不提前终止训练）
    n_epochs = config.get('epochs', 30)
    if n_epochs >= 80:
        early_stop_cfg = {
            "enabled": True,
            "monitor": "val_macro_f1",
            "mode": "max",
            "min_epochs": 40,
            "patience": 10,
            "delta": 0.0005
        }
    else:
        early_stop_cfg = {
            "enabled": True,
            "monitor": "val_macro_f1",
            "mode": "max",
            "min_epochs": 15,
            "patience": 5,
            "delta": 0.001
        }
    config['early_stopping'] = early_stop_cfg
    print(f"早停配置: {early_stop_cfg}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset):,} 个体素 ({len(train_subjects)} 个被试)")
    print(f"  验证集: {len(val_dataset):,} 个体素 ({len(val_subjects)} 个被试)")
    print(f"  测试集: {len(test_dataset):,} 个体素 (1 个被试)")

    # ==================== 创建模型 ====================
    print("\n创建模型...")
    model = get_model(
        model_type=config['model_type'],
        input_dim=config['feature_dim'],
        hidden_dims=config['hidden_units'],
        num_classes=config['num_class'],
        dropout_rate=config['dropout_rate'],
        activation=config['activation']
    )
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,}")

    # ==================== 计算类别权重 ====================
    print("\n计算类别权重...")
    class_weights = calculate_class_weights(
        train_dataset.all_labels,
        config['num_class'],
        config
    ).to(device)

    # ==================== 创建损失函数和优化器 ====================
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    lr_scheduler = None
    if config['use_lr_scheduler']:
        if config['lr_scheduler_type'] == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs']
            )

    # ==================== 训练 ====================
    print("\n开始训练...")
    train_start = time.time()

    training_results = train_brain_voxel_mlp_multiclass(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['epochs'],
        val_epoch=1,
        save_path=save_dir,
        lr_scheduler=lr_scheduler,
        experiment_name=experiment_name,
        config=config
    )

    train_time = time.time() - train_start
    print(f"\n训练完成，用时: {train_time / 60:.2f} 分钟")

    # ==================== 加载最佳模型 ====================
    print("\n加载最佳模型...")
    from utils.metrics import get_best_model
    from utils.model_io import safe_load_model

    try:
        best_model_path = get_best_model(
            training_results['val_f1_macro_list'],
            training_results['val_epoch_list'],
            save_dir,
            metric='f1'
        )
        checkpoint = safe_load_model(best_model_path, device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"已加载最佳模型: {os.path.basename(best_model_path)}")
    except Exception as e:
        print(f"加载最佳模型失败: {e}，使用当前模型")

    # ==================== 评估 ====================
    print("\n在测试集上评估...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=save_dir,
        dataset_name="test",
        detailed=True,
        plot=True,
        show_class_metrics=True,
        config=config
    )

    # ==================== 打印结果 ====================
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"准确率 (Accuracy): {test_results['accuracy']:.4f}")
    print(f"平衡准确率 (Balanced Acc): {test_results['balanced_accuracy']:.4f}")
    print(f"宏平均 F1 (Macro F1): {test_results['f1_macro']:.4f}")
    print(f"加权 F1 (Weighted F1): {test_results['f1_weighted']:.4f}")
    print(f"Kappa: {test_results['kappa']:.4f}")
    print("=" * 60)

    # ==================== 生成 Experiment JSON ====================
    print("\n生成 Experiment JSON...")
    try:
        from experiment_json_logger import ExperimentJSONLogger

        logger = ExperimentJSONLogger(
            experiment_id=config['experiment_id'],
            method_name=config['method_name'],
            method_key=config['method_key'],
            family=config['family'],
            subfamily=config['subfamily'],
            seed=config['random_seed']
        )

        # 设置配置
        logger.set_from_config(config)

        # 设置体素数量
        logger.set_voxel_counts(
            train_count=len(train_dataset),
            val_count=len(val_dataset),
            test_count=len(test_dataset)
        )

        # 设置被试信息
        logger.set_train_subjects(train_subjects)
        logger.data["task"]["split"]["val_subjects"] = val_subjects
        logger.data["task"]["split"]["test_subjects"] = [test_subject]

        # 设置模型参数量
        logger.set_model_param_count(model)

        # 设置训练时间
        logger.set_training_time(train_time)

        # 记录结果
        logger.log_results_from_eval_dict(test_results, split="test")

        # 添加说明：early_stopping 是计划配置，实际训练固定 N 个 epoch
        logger.data["results"]["notes"] = (
            f"训练固定 {config['epochs']} 个 epoch，early_stopping 为计划配置未实际执行。"
            f"最佳模型基于验证集 macro_f1 选择。"
        )

        # 设置结果路径
        logger.set_result_paths(save_dir, experiment_name)

        # 保存
        json_path = logger.save(save_dir, validate=True)
        print(f"Experiment JSON 已保存: {json_path}")

    except Exception as e:
        print(f"生成 Experiment JSON 失败: {e}")
        import traceback
        traceback.print_exc()

    # ==================== 保存最终报告 ====================
    report_path = os.path.join(save_dir, "final_report.txt")
    with open(report_path, 'w') as f:
        f.write("DeepMLP 6x2048 训练报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"实验名称: {experiment_name}\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"模型: DeepMLP 6x2048\n")
        f.write(f"激活函数: {config['activation']}\n")
        f.write(f"Dropout: {config['dropout_rate']}\n\n")
        f.write(f"数据划分:\n")
        f.write(f"  训练集: {len(train_subjects)} 个被试, {len(train_dataset):,} 个体素\n")
        f.write(f"  验证集: {len(val_subjects)} 个被试, {len(val_dataset):,} 个体素\n")
        f.write(f"  测试集: {test_subject}, {len(test_dataset):,} 个体素\n\n")
        f.write(f"测试结果:\n")
        f.write(f"  准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"  平衡准确率: {test_results['balanced_accuracy']:.4f}\n")
        f.write(f"  宏平均 F1: {test_results['f1_macro']:.4f}\n")
        f.write(f"  Kappa: {test_results['kappa']:.4f}\n\n")
        f.write(f"训练时间: {train_time / 60:.2f} 分钟\n")
        f.write(f"结果保存目录: {save_dir}\n")

    print(f"\n最终报告已保存: {report_path}")
    print(f"所有结果已保存至: {save_dir}")


if __name__ == "__main__":
    main()
