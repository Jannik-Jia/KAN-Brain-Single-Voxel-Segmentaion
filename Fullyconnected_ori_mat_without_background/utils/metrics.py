#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标计算和模型评价相关函数（兼容两种数据格式）
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, 
    recall_score, cohen_kappa_score, confusion_matrix, classification_report
)





def calculate_class_weights(train_labels, num_classes=102):
    """
    计算类别权重，基于过滤后的数据（标签范围1-102）
    """
    # 处理one-hot编码的标签
    if len(train_labels.shape) > 1 and train_labels.shape[1] > 1:
        labels_indices = np.argmax(train_labels, axis=1)  # 得到1-102
    else:
        labels_indices = train_labels.flatten()
    
    print(f"权重计算 - 标签范围: {np.min(labels_indices)} - {np.max(labels_indices)}")
    
    # 统计标签1-102的分布，创建长度为103的数组，然后取[1:]切片
    class_counts = np.bincount(labels_indices.astype(int), minlength=103)[1:]  # 跳过索引0
    total_samples = len(labels_indices)
    weights = np.zeros(num_classes, dtype=np.float32)
    
    # 为每个类别计算权重
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            weights[i] = 0.0  # 没有样本的类别权重为0
    
    print(f"类别权重计算完成:")
    print(f"  权重范围: {np.min(weights[weights > 0]):.4f} - {np.max(weights):.4f}")
    print(f"  零权重类别数: {np.sum(weights == 0)}")
    
    return torch.FloatTensor(weights)


def evaluate_model(model, data_loader, device, result_path=None, dataset_name="", 
                   class_names=None, detailed=True, plot=True, disable_progress=True, 
                   show_class_metrics=True):
    """
    评估模型性能并可选生成详细报告，兼容两种数据格式
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        result_path: 结果保存路径，None表示不保存
        dataset_name: 数据集名称 ("train", "val", "test")
        class_names: 类别名称列表
        detailed: 是否生成详细评估报告
        plot: 是否生成可视化图表
        disable_progress: 是否禁用进度条
        show_class_metrics: 是否显示每个类别的指标
    
    返回:
        评估结果字典
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []  # 存储预测概率
    
    desc = f"评估{dataset_name}集" if dataset_name else "评估中"
    
    with torch.no_grad():
        loader_iterator = data_loader if disable_progress else tqdm(data_loader, desc=desc)
        for data, target in loader_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    if len(all_preds) == 0 or len(all_targets) == 0:
        print(f"警告: {dataset_name}集中没有样本")  # 修改警告信息
        return {
            'accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'kappa': 0.0,
            'class_precision': np.array([]),
            'class_recall': np.array([]),
            'class_f1': np.array([]),
            'class_samples': np.array([]),
            'unique_classes': np.array([]),
            'confusion_matrix': np.array([]),
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    # 获取唯一类别
    unique_classes = np.unique(all_targets)
    class_count = len(unique_classes)
    
    # 计算主要评估指标
    accuracy = accuracy_score(all_targets, all_preds)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    # 计算每个类的精确率、召回率和F1分数
    class_precision = precision_score(all_targets, all_preds, average=None, zero_division=0, labels=unique_classes)
    class_recall = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=unique_classes)
    class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=unique_classes)
    
    # 计算每个类别的样本数
    class_samples = np.bincount(all_targets.astype(np.int64), minlength=int(np.max(unique_classes))+1)
    class_samples = class_samples[unique_classes]
    
    # 打印基本评估结果
    print(f"\n{'-'*50}")
    print(f"{dataset_name}集评估结果:")
    print(f"样本数量: {len(all_targets)}")
    print(f"类别数量: {class_count}")
    print(f"准确率: {accuracy:.4f}")
    print(f"平衡准确率: {balanced_acc:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"加权F1分数: {f1_weighted:.4f}")
    print(f"Kappa系数: {kappa:.4f}")
    
    # 统计类别表现
    if len(class_f1) > 0:
        best_class_idx = np.argmax(class_f1)
        worst_class_idx = np.argmin(class_f1)
        best_class = unique_classes[best_class_idx]
        worst_class = unique_classes[worst_class_idx]
        
        print(f"\n类别表现摘要:")
        print(f"表现最好的类别: 类别{best_class} (F1={class_f1[best_class_idx]:.4f}, 样本数={class_samples[best_class_idx]})")
        print(f"表现最差的类别: 类别{worst_class} (F1={class_f1[worst_class_idx]:.4f}, 样本数={class_samples[worst_class_idx]})")
    
    # 如果需要显示每个标签的指标
    if show_class_metrics and len(unique_classes) > 0:
        print(f"\n{'-'*50}")
        print(f"{dataset_name}集每个标签的F1分数:")
        print(f"{'标签ID':<8}{'样本数':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}")
        
        # 创建类别指标字典
        class_metrics = {}
        for idx, cls in enumerate(unique_classes):
            class_metrics[cls] = {
                'samples': class_samples[idx],
                'precision': class_precision[idx],
                'recall': class_recall[idx],
                'f1': class_f1[idx]
            }
        
        # 确定显示范围
        max_label = np.max(unique_classes)
        min_label = np.min(unique_classes)
        
        # 显示所有出现的类别
        for label in range(int(min_label), int(max_label) + 1):
            if label in class_metrics:
                # 数据集中存在此标签
                metrics = class_metrics[label]
                print(f"{label:<8}{metrics['samples']:<10}{metrics['precision']:.4f}{'':6}{metrics['recall']:.4f}{'':6}{metrics['f1']:.4f}")

        
        # 添加标签分布信息
        non_zero_classes = np.sum(class_samples > 0)
        total_possible_classes = int(max_label) - int(min_label) + 1

        zero_classes = total_possible_classes - non_zero_classes
        
        print(f"\n标签分布统计:")
        print(f"可能的标签数量: {total_possible_classes}")
        print(f"有样本的标签数量: {non_zero_classes}")
        print(f"无样本的标签数量: {zero_classes}")
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=unique_classes)
    
    # 如果需要生成详细报告
    if detailed and result_path:
        # 确保结果路径存在
        os.makedirs(result_path, exist_ok=True)
        
        # 生成分类报告
        target_names = class_names if class_names else [f"Class {i}" for i in unique_classes]
        report = classification_report(all_targets, all_preds, labels=unique_classes, target_names=target_names)
        
        # 保存详细评估报告
        report_file = os.path.join(result_path, f"{dataset_name}_evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write(f"模型在{dataset_name}集上的评估结果\n")
            f.write("="*50 + "\n\n")
            f.write(f"样本数量: {len(all_targets)}\n")
            f.write(f"类别数量: {class_count}\n\n")
            
            f.write("主要评估指标:\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"平衡准确率: {balanced_acc:.4f}\n")
            f.write(f"宏平均F1分数: {f1_macro:.4f}\n")
            f.write(f"加权F1分数: {f1_weighted:.4f}\n")
            f.write(f"Kappa系数: {kappa:.4f}\n\n")
            
            f.write("分类报告:\n")
            f.write(report)
            
            f.write("\n每个类别的详细指标:\n")
            f.write(f"{'类别ID':<10} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}\n")
            for i, cls in enumerate(unique_classes):
                f.write(f"{cls:<10} {class_samples[i]:<10} {class_precision[i]:.4f}:<10 {class_recall[i]:.4f}:<10 {class_f1[i]:.4f}:<10\n")
        
        # 保存CSV格式的类别性能
        csv_file = os.path.join(result_path, f"{dataset_name}_class_metrics.csv")
        with open(csv_file, 'w') as f:
            f.write("Class,SampleCount,Precision,Recall,F1Score\n")
            for i, cls in enumerate(unique_classes):
                f.write(f"{cls},{class_samples[i]},{class_precision[i]:.6f},{class_recall[i]:.6f},{class_f1[i]:.6f}\n")
        
        # 保存预测概率和真实标签
        np.savez(
            os.path.join(result_path, f"{dataset_name}_predictions.npz"),
            predictions=all_preds,
            targets=all_targets,
            probabilities=all_probs,
            classes=unique_classes
        )
        
        # 保存主要评估指标为CSV
        metrics_file = os.path.join(result_path, f"{dataset_name}_metrics.csv")
        with open(metrics_file, 'w') as f:
            f.write("Metric,Value\n")
            f.write(f"accuracy,{accuracy:.6f}\n")
            f.write(f"balanced_accuracy,{balanced_acc:.6f}\n")
            f.write(f"f1_macro,{f1_macro:.6f}\n")
            f.write(f"f1_weighted,{f1_weighted:.6f}\n")
            f.write(f"kappa,{kappa:.6f}\n")
            f.write(f"num_samples,{len(all_targets)}\n")
            f.write(f"num_classes,{class_count}\n")
        
        # 如果需要生成可视化
        if plot and len(conf_matrix) > 0:
            try:
                # 混淆矩阵可视化
                plt.figure(figsize=(12, 10))
                # 使用对数缩放
                conf_mat_log = np.log1p(conf_matrix)  # log(1+x)以处理零值
                mask = conf_matrix == 0
                # 绘制混淆矩阵热图
                sns.heatmap(conf_mat_log, annot=False, fmt='d', cmap='Blues', mask=mask)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix - {dataset_name} set (log scale)')
                plt.savefig(os.path.join(result_path, f"{dataset_name}_confusion_matrix.png"))
                plt.close()
                
                # 类别F1分数可视化
                if len(class_f1) > 0:
                    plt.figure(figsize=(15, 6))
                    
                    # 按F1分数排序
                    sorted_indices = np.argsort(class_f1)
                    # 选择最好和最差的20个类别（如果可用）
                    num_to_show = min(20, len(sorted_indices))
                    worst_indices = sorted_indices[:num_to_show]
                    best_indices = sorted_indices[-num_to_show:]
                    
                    # 绘制最差类别
                    plt.subplot(1, 2, 1)
                    bars = plt.barh(range(len(worst_indices)), class_f1[worst_indices])
                    plt.yticks(range(len(worst_indices)), [f"Class {unique_classes[i]}" for i in worst_indices])
                    plt.xlabel('F1 Score')
                    plt.title('Worst Performing Classes')
                    plt.grid(True, axis='x')
                    
                    # 为每个柱状图添加样本数量标注
                    for i, bar in enumerate(bars):
                        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                 f"n={class_samples[worst_indices[i]]}", va='center')
                    
                    # 绘制最好类别
                    plt.subplot(1, 2, 2)
                    bars = plt.barh(range(len(best_indices)), class_f1[best_indices])
                    plt.yticks(range(len(best_indices)), [f"Class {unique_classes[i]}" for i in best_indices])
                    plt.xlabel('F1 Score')
                    plt.title('Best Performing Classes')
                    plt.grid(True, axis='x')
                    
                    # 为每个柱状图添加样本数量标注
                    for i, bar in enumerate(bars):
                        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                 f"n={class_samples[best_indices[i]]}", va='center')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_path, f"{dataset_name}_class_performance.png"))
                    plt.close()
            except Exception as e:
                print(f"生成可视化图表时出错: {e}")
    
    # 返回评估结果字典
    result = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'class_samples': class_samples,
        'unique_classes': unique_classes,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
    
    return result

# 保留原版函数以确保向后兼容
def compare_class_performance(results_list, dataset_names, result_path=None):
    """
    比较不同数据集或模型间的类别性能
    """
    if len(results_list) != len(dataset_names):
        raise ValueError("结果列表和数据集名称列表长度必须相同")
    
    # 找出所有结果中的共有类别
    common_classes = set(results_list[0]['unique_classes'])
    for result in results_list[1:]:
        common_classes &= set(result['unique_classes'])
    
    common_classes = sorted(list(common_classes))
    
    if not common_classes:
        print("没有找到共有类别，无法进行比较")
        return None
    
    # 为每个共有类别，比较不同数据集的性能
    comparison = {cls: {} for cls in common_classes}
    
    for cls in common_classes:
        for i, result in enumerate(results_list):
            dataset = dataset_names[i]
            # 找到类别在结果中的索引
            cls_idx = np.where(result['unique_classes'] == cls)[0][0]
            
            comparison[cls][dataset] = {
                'f1': result['class_f1'][cls_idx],
                'precision': result['class_precision'][cls_idx],
                'recall': result['class_recall'][cls_idx],
                'samples': result['class_samples'][cls_idx]
            }
    
    # 生成比较报告
    if result_path:
        os.makedirs(result_path, exist_ok=True)
        
        # 保存比较报告为CSV
        csv_file = os.path.join(result_path, "class_performance_comparison.csv")
        with open(csv_file, 'w') as f:
            # 写入表头
            header = "Class"
            for dataset in dataset_names:
                header += f",{dataset}_F1,{dataset}_Precision,{dataset}_Recall,{dataset}_Samples"
            f.write(header + "\n")
            
            # 写入每个类别的比较数据
            for cls in common_classes:
                row = f"{cls}"
                for dataset in dataset_names:
                    row += f",{comparison[cls][dataset]['f1']:.6f}"
                    row += f",{comparison[cls][dataset]['precision']:.6f}"
                    row += f",{comparison[cls][dataset]['recall']:.6f}"
                    row += f",{comparison[cls][dataset]['samples']}"
                f.write(row + "\n")
        
        # 生成性能比较可视化
        plt.figure(figsize=(15, 8))
        
        # 提取所有数据集的F1分数
        f1_scores = []
        for dataset in dataset_names:
            f1_scores.append([comparison[cls][dataset]['f1'] for cls in common_classes])
        
        # 绘制箱线图比较
        plt.subplot(1, 2, 1)
        plt.boxplot(f1_scores, labels=dataset_names)
        plt.title('F1 Score Distribution Comparison')
        plt.ylabel('F1 Score')
        plt.grid(True, axis='y')
        
        # 绘制每个类别在不同数据集的性能对比
        plt.subplot(1, 2, 2)
        
        # 选择一部分类别进行可视化（太多会很拥挤）
        if len(common_classes) > 20:
            # 选择性能差异最大的类别
            diff_max = []
            for cls in common_classes:
                f1_values = [comparison[cls][dataset]['f1'] for dataset in dataset_names]
                diff_max.append(max(f1_values) - min(f1_values))
            
            # 获取差异前20大的类别
            selected_idx = np.argsort(diff_max)[-20:]
            selected_classes = [common_classes[i] for i in selected_idx]
        else:
            selected_classes = common_classes
        
        # 生成分组柱状图
        bar_width = 0.8 / len(dataset_names)
        x = np.arange(len(selected_classes))
        
        for i, dataset in enumerate(dataset_names):
            # 提取当前数据集的F1分数
            dataset_f1 = [comparison[cls][dataset]['f1'] for cls in selected_classes]
            plt.bar(x + i*bar_width, dataset_f1, width=bar_width, label=dataset)
        
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison by Class')
        plt.xticks(x + bar_width * (len(dataset_names) - 1) / 2, selected_classes, rotation=90)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, "class_performance_comparison.png"))
        plt.close()
    
    return comparison

def get_best_model(metrics_list, epoch_list, save_path, metric='f1', del_others=False):
    """
    通过指定评估指标找到最佳模型，适应不同的文件名前缀
    """
    metrics_list = np.array(metrics_list)
    epoch_list = np.array(epoch_list)
    best_index = np.argwhere(metrics_list == np.max(metrics_list))[-1].item()
    best_epoch = epoch_list[best_index]
    best_metric = metrics_list[best_index]
    
    # 获取目录中所有的.pth文件
    all_model_files = glob.glob(os.path.join(save_path, "*.pth"))
    
    if not all_model_files:
        raise FileNotFoundError(f"在目录 {save_path} 中没有找到任何.pth模型文件")
    
    # 查找包含正确轮次和指标的模型文件
    matching_files = []
    for file_path in all_model_files:
        file_name = os.path.basename(file_path)
        # 检查文件名中是否包含正确的轮次和指标值
        epoch_pattern = f"epoch_{best_epoch}_"
        metric_pattern = f"f1_{best_metric:.4f}"
        
        if epoch_pattern in file_name and metric_pattern in file_name:
            matching_files.append(file_path)
    
    # 如果没有找到精确匹配，尝试只匹配轮次
    if not matching_files:
        print(f"没有找到精确匹配的模型文件，尝试只匹配轮次 {best_epoch}")
        for file_path in all_model_files:
            file_name = os.path.basename(file_path)
            epoch_pattern = f"epoch_{best_epoch}_"
            if epoch_pattern in file_name:
                matching_files.append(file_path)
    
    # 如果仍然没有匹配，使用最后修改的文件
    if not matching_files:
        print(f"警告: 无法找到epoch {best_epoch}对应的模型文件，将使用最近修改的模型文件")
        matching_files = [max(all_model_files, key=os.path.getmtime)]
    
    best_model_path = matching_files[0]
    print(f"选择的最佳模型: {os.path.basename(best_model_path)}")
    
    # 删除其他模型(如果需要)
    if del_others:
        for f in all_model_files:
            if f != best_model_path:
                os.remove(f)
    
    return best_model_path