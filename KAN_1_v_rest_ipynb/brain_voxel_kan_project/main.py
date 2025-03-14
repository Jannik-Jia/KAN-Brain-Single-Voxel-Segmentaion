#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脑体素KAN训练和评估主脚本
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# 导入自定义模块
from config import *
from datasets import BrainVoxelDataset, BrainVoxelDataManager
from models import BrainVoxelKAN
from train import train_brain_voxel_kan, evaluate_model, plot_training_progress, visualize_model_performance
from utils import set_random_seed, visualize_dataset_distribution, get_best_model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和评估脑体素KAN模型')
    
    parser.add_argument('--label_id', type=int, default=LABEL_ID,
                      help='目标标签ID')
    parser.add_argument('--pca', action='store_true', default=APPLY_PCA,
                      help='是否应用PCA降维')
    parser.add_argument('--n_pca', type=int, default=N_PCA,
                      help='PCA降维的目标维度，0表示自动选择')
    parser.add_argument('--norm', action='store_true', default=NORM,
                      help='是否对数据进行归一化')
    parser.add_argument('--sampling', type=str, default=SAMPLING_STRATEGY,
                      choices=['balanced', 'stratified', 'modified_stratified', 'hard_negative'],
                      help='采样策略')
    parser.add_argument('--epochs', type=int, default=EPOCH,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='批处理大小')
    parser.add_argument('--val_epoch', type=int, default=VAL_EPOCH,
                      help='验证频率')
    parser.add_argument('--lr', type=float, default=LR,
                      help='学习率')
    parser.add_argument('--device', type=int, default=DEVICE,
                      help='计算设备编号，-1表示使用CPU')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                      help='结果保存路径')
    parser.add_argument('--eval_only', action='store_true',
                      help='仅评估模型，不进行训练')
    parser.add_argument('--checkpoint', type=str, default=CHECK_POINT,
                      help='加载预训练模型的路径')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(RANDOM_SEED)
    
    # 设置计算设备
    device = torch.device(f"cuda:{args.device}" if args.device>=0 and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存路径
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # 创建数据管理器
    data_manager = BrainVoxelDataManager(
        DATA_DIRS['train_dir'],
        DATA_DIRS['test_dir'],
        DATA_DIRS['val_dir']
    )
    
    # 获取数据集
    print("加载数据集...")
    dataset_dict = data_manager.get_datasets(
        args.label_id,
        sampling_strategy=args.sampling,
        apply_pca_flag=args.pca,
        n_components=args.n_pca,
        norm=args.norm,
        neg_pos_ratio=NEG_POS_RATIO,
        neg_label_count=NEG_LABEL_COUNT
    )
    
    # 保存PCA模型引用
    pca_model = dataset_dict.get('pca_model')
    
    # 可视化数据集分布
    visualize_dataset_distribution(
        dataset_dict, 
        args.label_id,
        os.path.join(args.save_path, f'dataset_distribution_label_{args.label_id}.png')
    )
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataset = BrainVoxelDataset(dataset_dict['train_samples'], dataset_dict['train_labels'])
    test_dataset = BrainVoxelDataset(dataset_dict['test_samples'], dataset_dict['test_labels'])
    val_dataset = BrainVoxelDataset(dataset_dict['val_samples'], dataset_dict['val_labels'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 获取特征维度
    feature_dim = dataset_dict['feature_dim']
    print(f"特征维度: {feature_dim}")
    
    # 创建模型
    model = BrainVoxelKAN(feature_dim, 64, NUM_CLASS, FIXED_GRID).to(device)
    print(f"模型创建完成，网格大小: {FIXED_GRID}")
    
    # 加载预训练模型(如果指定)
    if args.checkpoint:
        print(f"加载预训练模型: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    # 训练或评估
    if not args.eval_only:
        # 训练模型
        print(f"开始训练模型 (epoch: {args.epochs}, batch_size: {args.batch_size})...")
        training_results = train_brain_voxel_kan(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            val_epoch=args.val_epoch,
            save_path=args.save_path
        )
        
        # 绘制训练进度
        plot_training_progress(
            training_results,
            os.path.join(args.save_path, f'training_progress_label_{args.label_id}.png')
        )
        
        # 获取最佳模型路径
        best_model_path = get_best_model(
            training_results['val_auc_pr_list'],
            training_results['val_epoch_list'],
            args.save_path,
            metric='auc_pr',
            del_others=False
        )
        
        # 加载最佳模型
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    # 评估最佳模型
    print("评估模型性能...")
    
    # 评估训练集
    print("在训练集上评估...")
    train_metrics = evaluate_model(model, train_loader, device)
    print(f"训练集评估结果:")
    print(f"  准确率: {train_metrics['accuracy']:.4f}")
    print(f"  精确率: {train_metrics['precision']:.4f}")
    print(f"  召回率: {train_metrics['recall']:.4f}")
    print(f"  F1分数: {train_metrics['f1']:.4f}")
    print(f"  AUC-PR: {train_metrics['auc_pr']:.4f}")
    
    # 可视化训练集性能
    visualize_model_performance(
        train_metrics, 
        title=f"训练集性能 - 标签 {args.label_id}",
        save_path=os.path.join(args.save_path, f'train_performance_label_{args.label_id}.png')
    )
    
    # 评估测试集
    print("在测试集上评估...")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"测试集评估结果:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1']:.4f}")
    print(f"  AUC-PR: {test_metrics['auc_pr']:.4f}")
    
    # 可视化测试集性能
    visualize_model_performance(
        test_metrics, 
        title=f"测试集性能 - 标签 {args.label_id}",
        save_path=os.path.join(args.save_path, f'test_performance_label_{args.label_id}.png')
    )
    
    # 评估验证集
    print("在验证集上评估...")
    val_metrics = evaluate_model(model, val_loader, device)
    print(f"验证集评估结果:")
    print(f"  准确率: {val_metrics['accuracy']:.4f}")
    print(f"  精确率: {val_metrics['precision']:.4f}")
    print(f"  召回率: {val_metrics['recall']:.4f}")
    print(f"  F1分数: {val_metrics['f1']:.4f}")
    print(f"  AUC-PR: {val_metrics['auc_pr']:.4f}")
    
    # 可视化验证集性能
    visualize_model_performance(
        val_metrics, 
        title=f"验证集性能 - 标签 {args.label_id}",
        save_path=os.path.join(args.save_path, f'val_performance_label_{args.label_id}.png')
    )
    
    # 评估merged数据集
    print("在merged数据集上评估...")
    from train import evaluate_merged_dataset
    
    merged_metrics = evaluate_merged_dataset(
        model=model,
        merged_dir=DATA_DIRS['merged_dir'],
        label_id=args.label_id,
        apply_pca_flag=args.pca,
        pca_model=pca_model,
        norm=args.norm,
        device=device,
        batch_size=args.batch_size
    )
    
    print(f"Merged数据集评估结果:")
    print(f"  准确率: {merged_metrics['accuracy']:.4f}")
    print(f"  精确率: {merged_metrics['precision']:.4f}")
    print(f"  召回率: {merged_metrics['recall']:.4f}")
    print(f"  F1分数: {merged_metrics['f1']:.4f}")
    print(f"  AUC-PR: {merged_metrics['auc_pr']:.4f}")
    
    # 可视化merged数据集性能
    visualize_model_performance(
        merged_metrics, 
        title=f"Merged数据集性能 - 标签 {args.label_id}",
        save_path=os.path.join(args.save_path, f'merged_performance_label_{args.label_id}.png')
    )
    
    # 绘制四个数据集的性能对比
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"不同数据集性能对比 - 标签 {args.label_id}", fontsize=16)
    
    # 准确率对比
    plt.subplot(2, 2, 1)
    datasets = ['Training', 'Testing', 'Validation', 'Merged']
    accuracies = [
        train_metrics['accuracy'], 
        test_metrics['accuracy'], 
        val_metrics['accuracy'], 
        merged_metrics['accuracy']
    ]
    plt.bar(datasets, accuracies, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.grid(axis='y')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # 召回率对比
    plt.subplot(2, 2, 2)
    recalls = [
        train_metrics['recall'], 
        test_metrics['recall'], 
        val_metrics['recall'],
        merged_metrics['recall']
    ]
    plt.bar(datasets, recalls, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Recall')
    plt.title('Recall Comparison')
    plt.grid(axis='y')
    for i, v in enumerate(recalls):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # AUC-PR对比
    plt.subplot(2, 2, 3)
    auc_prs = [
        train_metrics['auc_pr'], 
        test_metrics['auc_pr'], 
        val_metrics['auc_pr'],
        merged_metrics['auc_pr']
    ]
    plt.bar(datasets, auc_prs, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('AUC-PR')
    plt.title('AUC-PR Comparison')
    plt.grid(axis='y')
    for i, v in enumerate(auc_prs):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # F1-score对比
    plt.subplot(2, 2, 4)
    f1_scores = [
        train_metrics['f1'],
        test_metrics['f1'],
        val_metrics['f1'],
        merged_metrics['f1']
    ]
    plt.bar(datasets, f1_scores, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')
    plt.grid(axis='y')
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    comparison_path = os.path.join(args.save_path, f'dataset_comparison_label_{args.label_id}.png')
    plt.savefig(comparison_path)
    print(f"数据集性能对比图已保存至: {comparison_path}")
    
    print("评估完成!")

if __name__ == "__main__":
    main()