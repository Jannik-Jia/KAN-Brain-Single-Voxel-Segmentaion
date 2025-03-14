#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多epoch评估脚本，评估每个epoch的模型表现并保存结果
"""
import os
import sys
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
import json
from datetime import datetime

# 导入自定义模块
from config import *
from datasets import BrainVoxelDataset, BrainVoxelSampler, BrainVoxelDataManager
from models import BrainVoxelKAN
from train import evaluate_model, evaluate_merged_dataset, get_performance_metrics_for_epoch, evaluate_thresholds
from utils import set_random_seed, apply_pca, load_model_for_epoch, save_plot_comparison

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估多个epoch的模型表现')
    
    parser.add_argument('--label_id', type=int, default=LABEL_ID,
                      help='目标标签ID')
    parser.add_argument('--start_epoch', type=int, default=START_EPOCH,
                      help='起始epoch')
    parser.add_argument('--end_epoch', type=int, default=END_EPOCH,
                      help='结束epoch')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                      help='模型和结果保存路径')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='批处理大小')
    parser.add_argument('--device', type=int, default=DEVICE,
                      help='计算设备编号，-1表示使用CPU')
    parser.add_argument('--save_results', action='store_true', default=SAVE_RESULTS,
                      help='是否保存评估结果')
    parser.add_argument('--plot_results', action='store_true', default=PLOT_RESULTS,
                      help='是否绘制评估结果图表')
    parser.add_argument('--metrics', type=str, nargs='+', default=EVALUATION_METRICS,
                      help='要评估的指标列表')
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLD_VALUES,
                      help='用于评估的阈值列表')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(RANDOM_SEED)
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if args.device>=0 and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存路径
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # 创建评估结果保存路径
    eval_save_path = os.path.join(args.save_path, f"eval_results_label_{args.label_id}")
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    
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
        sampling_strategy=SAMPLING_STRATEGY,
        apply_pca_flag=APPLY_PCA,
        n_components=N_PCA,
        norm=NORM,
        neg_pos_ratio=NEG_POS_RATIO,
        neg_label_count=NEG_LABEL_COUNT
    )
    
    # 保存PCA模型引用
    pca_model = dataset_dict.get('pca_model')
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataset = BrainVoxelDataset(dataset_dict['train_samples'], dataset_dict['train_labels'])
    test_dataset = BrainVoxelDataset(dataset_dict['test_samples'], dataset_dict['test_labels'])
    val_dataset = BrainVoxelDataset(dataset_dict['val_samples'], dataset_dict['val_labels'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 获取特征维度
    feature_dim = dataset_dict['feature_dim']
    
    # 初始化结果字典
    results = {
        'epochs': [],
        'train': {metric: [] for metric in args.metrics},
        'test': {metric: [] for metric in args.metrics},
        'val': {metric: [] for metric in args.metrics},
        'merged': {metric: [] for metric in args.metrics},
        'threshold_results': {}
    }
    
    # 评估每个epoch的模型
    print(f"开始评估从epoch {args.start_epoch}到{args.end_epoch}的模型...")
    
    # 保存评估开始时间
    start_time = datetime.now()
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch + 1), desc="评估进度"):
        try:
            # 加载模型
            model, model_path = load_model_for_epoch(
                epoch, 
                args.save_path, 
                feature_dim, 
                NUM_CLASS, 
                FIXED_GRID, 
                device
            )
            
            # 获取性能指标
            metrics = get_performance_metrics_for_epoch(
                model,
                train_loader,
                test_loader,
                val_loader,
                DATA_DIRS['merged_dir'],
                args.label_id,
                APPLY_PCA,
                pca_model,
                NORM,
                device,
                args.batch_size
            )
            
            # 存储epoch
            results['epochs'].append(epoch)
            
            # 存储各指标
            for metric_name in args.metrics:
                results['train'][metric_name].append(metrics['train'][metric_name])
                results['test'][metric_name].append(metrics['test'][metric_name])
                results['val'][metric_name].append(metrics['val'][metric_name])
                results['merged'][metric_name].append(metrics['merged'][metric_name])
            
            # 每10个epoch评估一次不同阈值的性能
            if epoch % 10 == 0 or epoch == args.end_epoch:
                print(f"评估epoch {epoch}在不同阈值下的性能...")
                threshold_results = evaluate_thresholds(metrics, args.thresholds)
                results['threshold_results'][epoch] = threshold_results
            
            # 打印主要评估指标
            print(f"\nEpoch {epoch} 评估结果:")
            for dataset_name in ['train', 'test', 'val', 'merged']:
                print(f"  {dataset_name.capitalize()} Set - ", end="")
                metric_strings = []
                for metric_name in args.metrics:
                    metric_value = metrics[dataset_name][metric_name]
                    metric_strings.append(f"{metric_name}: {metric_value:.4f}")
                print(", ".join(metric_strings))
            
        except FileNotFoundError:
            print(f"警告: 未找到epoch {epoch}的模型文件，跳过评估")
            continue
        except Exception as e:
            print(f"评估epoch {epoch}时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 计算评估总时间
    end_time = datetime.now()
    evaluation_time = (end_time - start_time).total_seconds()
    print(f"评估完成，总用时: {evaluation_time:.2f}秒")
    
    # 保存结果
    if args.save_results:
        result_file = os.path.join(eval_save_path, f"multi_epoch_evaluation_label_{args.label_id}.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"结果已保存至: {result_file}")
        
        # 导出为CSV格式
        csv_data = {'epoch': results['epochs']}
        for dataset_name in ['train', 'test', 'val', 'merged']:
            for metric_name in args.metrics:
                csv_data[f"{dataset_name}_{metric_name}"] = results[dataset_name][metric_name]
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(eval_save_path, f"multi_epoch_evaluation_label_{args.label_id}.csv")
        df.to_csv(csv_file, index=False)
        print(f"CSV结果已保存至: {csv_file}")
        
        # 保存执行参数
        params = {
            'label_id': args.label_id,
            'start_epoch': args.start_epoch,
            'end_epoch': args.end_epoch,
            'metrics': args.metrics,
            'thresholds': args.thresholds,
            'evaluation_time': evaluation_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'apply_pca': APPLY_PCA,
            'n_pca': N_PCA,
            'sampling_strategy': SAMPLING_STRATEGY,
            'neg_pos_ratio': NEG_POS_RATIO
        }
        
        params_file = os.path.join(eval_save_path, f"evaluation_params_label_{args.label_id}.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"评估参数已保存至: {params_file}")
    
    # 绘制结果
    if args.plot_results:
        epochs = results['epochs']
        
        # 为每个指标创建单独的图表
        for metric_name in args.metrics:
            plot_path = os.path.join(eval_save_path, f"{metric_name}_comparison_label_{args.label_id}.png")
            save_plot_comparison(
                results['train'][metric_name],
                results['test'][metric_name],
                results['val'][metric_name],
                results['merged'][metric_name],
                epochs,
                metric_name,
                plot_path
            )
            print(f"图表已保存至: {plot_path}")
        
        # 创建所有指标合并图表
        plt.figure(figsize=(15, 10))
        for i, metric_name in enumerate(args.metrics):
            if i >= 4:  # 最多显示4个指标
                break
                
            plt.subplot(2, 2, i+1)
            plt.plot(epochs, results['train'][metric_name], 'b-', label='Training')
            plt.plot(epochs, results['test'][metric_name], 'g-', label='Testing')
            plt.plot(epochs, results['val'][metric_name], 'r-', label='Validation')
            plt.plot(epochs, results['merged'][metric_name], 'm-', label='Merged')
            
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} across Epochs')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        all_metrics_path = os.path.join(eval_save_path, f"all_metrics_comparison_label_{args.label_id}.png")
        plt.savefig(all_metrics_path)
        print(f"所有指标对比图已保存至: {all_metrics_path}")
        
        # 绘制阈值性能图表
        if results['threshold_results']:
            # 找到最佳epoch（基于验证集AUC-PR）
            if 'auc_pr' in args.metrics:
                best_epoch_idx = np.argmax(results['val']['auc_pr'])
                best_epoch = results['epochs'][best_epoch_idx]
            else:
                best_epoch = max(results['threshold_results'].keys())
            
            if best_epoch in results['threshold_results']:
                plt.figure(figsize=(15, 10))
                
                for i, dataset_name in enumerate(['train', 'test', 'val', 'merged']):
                    plt.subplot(2, 2, i+1)
                    
                    thresholds = [result['threshold'] for result in results['threshold_results'][best_epoch][dataset_name]]
                    precision = [result['precision'] for result in results['threshold_results'][best_epoch][dataset_name]]
                    recall = [result['recall'] for result in results['threshold_results'][best_epoch][dataset_name]]
                    f1 = [result['f1'] for result in results['threshold_results'][best_epoch][dataset_name]]
                    
                    plt.plot(thresholds, precision, 'b-', label='Precision')
                    plt.plot(thresholds, recall, 'g-', label='Recall')
                    plt.plot(thresholds, f1, 'r-', label='F1 Score')
                    
                    plt.xlabel('Threshold')
                    plt.ylabel('Score')
                    plt.title(f'{dataset_name.capitalize()} Set - Threshold Analysis (Epoch {best_epoch})')
                    plt.grid(True)
                    plt.legend()
                
                plt.tight_layout()
                threshold_plot_path = os.path.join(eval_save_path, f"threshold_analysis_label_{args.label_id}.png")
                plt.savefig(threshold_plot_path)
                print(f"阈值分析图表已保存至: {threshold_plot_path}")
    
    # 找出各指标的最佳epoch
    print("\n各指标的最佳epoch:")
    for metric_name in args.metrics:
        best_train_idx = np.argmax(results['train'][metric_name])
        best_test_idx = np.argmax(results['test'][metric_name])
        best_val_idx = np.argmax(results['val'][metric_name])
        best_merged_idx = np.argmax(results['merged'][metric_name])
        
        best_train_epoch = results['epochs'][best_train_idx]
        best_test_epoch = results['epochs'][best_test_idx]
        best_val_epoch = results['epochs'][best_val_idx]
        best_merged_epoch = results['epochs'][best_merged_idx]
        
        best_train_value = results['train'][metric_name][best_train_idx]
        best_test_value = results['test'][metric_name][best_test_idx]
        best_val_value = results['val'][metric_name][best_val_idx]
        best_merged_value = results['merged'][metric_name][best_merged_idx]
        
        print(f"  {metric_name}:")
        print(f"    训练集 - Epoch {best_train_epoch}: {best_train_value:.4f}")
        print(f"    测试集 - Epoch {best_test_epoch}: {best_test_value:.4f}")
        print(f"    验证集 - Epoch {best_val_epoch}: {best_val_value:.4f}")
        print(f"    合并集 - Epoch {best_merged_epoch}: {best_merged_value:.4f}")
    
    # 创建最佳epoch摘要文件
    if args.save_results:
        best_epochs = {}
        for metric_name in args.metrics:
            best_merged_idx = np.argmax(results['merged'][metric_name])
            best_merged_epoch = results['epochs'][best_merged_idx]
            best_merged_value = results['merged'][metric_name][best_merged_idx]
            
            best_epochs[metric_name] = {
                'epoch': int(best_merged_epoch),
                'value': float(best_merged_value)
            }
        
        best_epochs_file = os.path.join(eval_save_path, f"best_epochs_label_{args.label_id}.json")
        with open(best_epochs_file, 'w') as f:
            json.dump(best_epochs, f, indent=4)
        print(f"最佳epoch摘要已保存至: {best_epochs_file}")
    
    return results

if __name__ == "__main__":
    main()