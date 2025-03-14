#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练和评估函数模块
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, 
    precision_score, f1_score, precision_recall_curve, 
    auc, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_brain_voxel_kan(model, train_loader, test_loader, criterion, optimizer, device, 
                         num_epochs=100, val_epoch=1, save_path="./Results"):
    """
    训练脑体素KAN模型，与1D KAN训练流程保持一致，并增加更全面的指标评估
    
    参数:
        model: KAN模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        val_epoch: 验证频率
        save_path: 模型保存路径
    
    返回:
        训练结果统计信息
    """
    # 初始化统计变量
    loss_list = []
    acc_list = []
    val_acc_list = []
    val_epoch_list = []
    val_f1_list = []
    val_auc_pr_list = []
    val_recall_list = []
    
    # 保存起始时间
    train_st = time.time()
    
    # 计算批次数量和样本数量
    batch_num = len(train_loader)
    train_num = len(train_loader.dataset)
    test_num = len(test_loader.dataset)
    
    try:
        # 训练循环
        for e in tqdm(range(num_epochs), desc="Training:"):
            # 设置模型为训练模式
            model.train()
            avg_loss = 0.0
            train_acc = 0
            
            # 批次循环
            for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=batch_num, leave=False):
                # 将数据移动到指定设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 累计损失和准确率
                avg_loss += loss.item()
                _, pred = torch.max(out, dim=1)
                train_acc += (pred == target).sum().item()
            
            # 计算本轮平均损失和准确率
            loss_list.append(avg_loss / train_num)
            acc_list.append(train_acc / train_num)
            print(f"epoch {e}/{num_epochs} loss:{loss_list[-1]:.6f}  acc:{acc_list[-1]:.4f}")
            
            # 验证阶段
            if (e+1) % val_epoch == 0 or (e+1) == num_epochs:
                val_acc = 0
                model.eval()
                
                # 收集验证数据的预测结果
                all_preds = []
                all_probs = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
                        data, target = data.to(device), target.to(device)
                        out = model(data)
                        probs = torch.softmax(out, dim=1)
                        _, pred = torch.max(out, dim=1)
                        
                        all_preds.extend(pred.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())  # 保存正类概率
                        all_targets.extend(target.cpu().numpy())
                        val_acc += (pred == target).sum().item()
                
                # 计算全面的评估指标
                val_accuracy = val_acc / test_num
                val_recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
                val_precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
                val_f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
                
                # 计算AUC-PR
                precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
                val_auc_pr = auc(recall_curve, precision_curve)
                
                # 保存验证结果
                val_acc_list.append(val_accuracy)
                val_epoch_list.append(e)
                val_f1_list.append(val_f1)
                val_auc_pr_list.append(val_auc_pr)
                val_recall_list.append(val_recall)
                
                # 显示全面的评估指标
                print(f"epoch {e}/{num_epochs}  val_acc:{val_accuracy:.4f}  val_f1:{val_f1:.4f}  val_recall:{val_recall:.4f}  val_auc_pr:{val_auc_pr:.4f}")
                
                # 打印混淆矩阵
                conf_matrix = confusion_matrix(all_targets, all_preds)
                print(f"Confusion Matrix:\n{conf_matrix}")
                print(f"True Positives: {conf_matrix[1][1]}, False Positives: {conf_matrix[0][1]}")
                print(f"True Negatives: {conf_matrix[0][0]}, False Negatives: {conf_matrix[1][0]}")
                
                # 保存当前模型
                save_name = os.path.join(save_path, f"epoch_{e}_acc_{val_accuracy:.4f}_f1_{val_f1:.4f}_aucpr_{val_auc_pr:.4f}.pth")
                save_dict = {
                    'state_dict': model.state_dict(), 
                    'epoch': e+1, 
                    'optimizer': optimizer.state_dict(),
                    'loss_list': loss_list, 
                    'acc_list': acc_list, 
                    'val_acc_list': val_acc_list, 
                    'val_epoch_list': val_epoch_list,
                    'val_f1_list': val_f1_list,
                    'val_auc_pr_list': val_auc_pr_list,
                    'val_recall_list': val_recall_list
                }
                torch.save(save_dict, save_name)
                
    except Exception as exc:
        print(exc)
        import traceback
        traceback.print_exc()
        
    finally:
        print(f'训练停止于epoch {e if "e" in locals() else "unknown"}')
    
    # 计算总训练时间
    train_time = time.time() - train_st
    print(f"训练时间: {train_time:.2f}秒")
    
    # 返回训练结果
    return {
        'loss_list': loss_list,
        'acc_list': acc_list,
        'val_acc_list': val_acc_list,
        'val_epoch_list': val_epoch_list,
        'val_f1_list': val_f1_list,
        'val_auc_pr_list': val_auc_pr_list,
        'val_recall_list': val_recall_list,
        'train_time': train_time
    }

def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
    
    返回:
        评估结果
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="评估中"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    
    # 计算AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
    auc_pr = auc(recall_curve, precision_curve)
    
    # 计算ROC-AUC
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 生成分类报告
    report = classification_report(all_targets, all_preds, target_names=['Negative', 'Positive'])
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc_pr': auc_pr,
        'roc_auc': roc_auc,
        'report': report,
        'conf_matrix': conf_matrix,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets
    }

def evaluate_merged_dataset(model, merged_dir, label_id, apply_pca_flag=True, pca_model=None, norm=True, device=None, batch_size=64):
    """
    在merged数据集上评估模型性能
    
    参数:
        model: 训练好的模型
        merged_dir: merged数据集目录
        label_id: 目标标签ID
        apply_pca_flag: 是否应用PCA
        pca_model: PCA模型
        norm: 是否标准化数据
        device: 计算设备
        batch_size: 批处理大小
        
    返回:
        评估结果
    """
    from datasets import BrainVoxelSampler, BrainVoxelDataset
    from torch.utils.data import DataLoader
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建采样器
    merged_sampler = BrainVoxelSampler(merged_dir)
    
    # 获取目标标签的正样本
    pos_file = merged_sampler.get_file_path(label_id)
    if not pos_file:
        raise ValueError(f"在merged数据集中找不到标签 {label_id} 的文件")
    
    # 加载正样本
    pos_samples = np.load(pos_file)
    pos_labels = np.ones(len(pos_samples))
    
    # 获取所有负样本
    other_labels = [l for l in merged_sampler.valid_labels if l != label_id]
    all_neg_samples = []
    all_neg_labels = []
    
    for other_label in other_labels:
        neg_file = merged_sampler.get_file_path(other_label)
        if neg_file:
            neg_samples = np.load(neg_file)
            all_neg_samples.append(neg_samples)
            all_neg_labels.append(np.zeros(len(neg_samples)))
    
    # 合并所有样本
    all_neg_samples = np.vstack(all_neg_samples) if all_neg_samples else np.array([]).reshape(0, pos_samples.shape[1])
    all_neg_labels = np.concatenate(all_neg_labels) if all_neg_labels else np.array([])
    
    all_samples = np.vstack([pos_samples, all_neg_samples])
    all_labels = np.concatenate([pos_labels, all_neg_labels])
    
    # 应用PCA（如果需要）
    if apply_pca_flag and pca_model is not None:
        # 使用已训练的PCA模型
        print(f"使用训练好的PCA模型转换merged数据...")
        all_samples = pca_model.transform(all_samples)
        if norm:
            # 应用标准化
            all_samples = (all_samples - np.min(all_samples, axis=0)) / (np.max(all_samples, axis=0) - np.min(all_samples, axis=0) + 1e-10)
    
    # 创建数据集和加载器
    merged_dataset = BrainVoxelDataset(all_samples, all_labels)
    merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=False)
    
    # 评估模型
    model.eval()
    return evaluate_model(model, merged_loader, device)

def get_performance_metrics_for_epoch(model, train_loader, test_loader, val_loader, merged_dir, label_id, 
                                     apply_pca_flag, pca_model, norm, device, batch_size=64):
    """
    获取指定模型在所有数据集上的性能指标
    
    参数:
        model: 已加载的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        val_loader: 验证数据加载器
        merged_dir: merged数据集目录
        label_id: 目标标签ID
        apply_pca_flag: 是否应用PCA
        pca_model: PCA模型
        norm: 是否标准化数据
        device: 计算设备
        batch_size: 批处理大小
        
    返回:
        包含所有数据集评估结果的字典
    """
    # 在各数据集上评估
    print("评估训练集性能...")
    train_metrics = evaluate_model(model, train_loader, device)
    
    print("评估测试集性能...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("评估验证集性能...")
    val_metrics = evaluate_model(model, val_loader, device)
    
    print("评估merged数据集性能...")
    merged_metrics = evaluate_merged_dataset(
        model=model,
        merged_dir=merged_dir,
        label_id=label_id,
        apply_pca_flag=apply_pca_flag,
        pca_model=pca_model,
        norm=norm,
        device=device,
        batch_size=batch_size
    )
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'val': val_metrics,
        'merged': merged_metrics
    }

def evaluate_thresholds(predictions_dict, thresholds):
    """
    评估不同阈值下的模型性能
    
    参数:
        predictions_dict: 包含预测结果的字典
        thresholds: 阈值列表
        
    返回:
        包含不同阈值下性能指标的字典
    """
    results = {}
    
    for dataset_name, metrics in predictions_dict.items():
        targets = metrics['targets']
        probabilities = metrics['probabilities']
        
        threshold_results = []
        
        for threshold in thresholds:
            # 应用阈值
            threshold_preds = [1 if prob >= threshold else 0 for prob in probabilities]
            
            # 计算指标
            accuracy = accuracy_score(targets, threshold_preds)
            precision = precision_score(targets, threshold_preds, zero_division=0)
            recall = recall_score(targets, threshold_preds, zero_division=0)
            f1 = f1_score(targets, threshold_preds, zero_division=0)
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        results[dataset_name] = threshold_results
    
    return results

def plot_training_progress(training_results, save_path=None):
    """
    绘制训练进度图表
    
    参数:
        training_results: 训练结果字典
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(range(len(training_results['loss_list'])), training_results['loss_list'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(range(len(training_results['acc_list'])), training_results['acc_list'], label='Train Acc')
    plt.plot(training_results['val_epoch_list'], training_results['val_acc_list'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制AUC-PR曲线
    plt.subplot(2, 2, 3)
    plt.plot(training_results['val_epoch_list'], training_results['val_auc_pr_list'], 'g-', label='AUC-PR')
    plt.plot(training_results['val_epoch_list'], training_results['val_f1_list'], 'r--', label='F1 Score')
    plt.title('AUC-PR & F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # 绘制召回率曲线
    plt.subplot(2, 2, 4)
    plt.plot(training_results['val_epoch_list'], training_results['val_recall_list'], 'm-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return plt

def visualize_model_performance(metrics, title="Model Performance", save_path=None):
    """
    可视化模型性能
    
    参数:
        metrics: 评估指标字典
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 提取混淆矩阵
    conf_matrix = metrics['conf_matrix']
    
    # 绘制混淆矩阵
    plt.subplot(2, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # 绘制PR曲线
    plt.subplot(2, 2, 2)
    precision, recall_points, _ = precision_recall_curve(metrics['targets'], metrics['probabilities'])
    plt.plot(recall_points, precision, 'b-', label=f'PR Curve (AUC = {metrics["auc_pr"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    
    # 绘制ROC曲线
    plt.subplot(2, 2, 3)
    fpr, tpr, _ = roc_curve(metrics['targets'], metrics['probabilities'])
    plt.plot(fpr, tpr, 'g-', label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    
    # 绘制直方图
    plt.subplot(2, 2, 4)
    plt.hist([metrics['probabilities'][i] for i in range(len(metrics['targets'])) if metrics['targets'][i] == 1], 
             bins=20, alpha=0.5, label='Positive Samples')
    plt.hist([metrics['probabilities'][i] for i in range(len(metrics['targets'])) if metrics['targets'][i] == 0], 
             bins=20, alpha=0.5, label='Negative Samples')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return plt