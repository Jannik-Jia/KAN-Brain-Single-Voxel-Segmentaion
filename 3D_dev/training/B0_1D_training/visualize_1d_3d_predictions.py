#!/usr/bin/env python3
"""
可视化1D训练后映射回3D的预测结果
包括：概率图、不确定性图、类别分布等
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import h5py
from matplotlib.colors import ListedColormap
import seaborn as sns

def load_predictions(pred_file: Path):
    """加载3D预测概率（支持HDF5/MAT v7.3格式）"""
    try:
        # 首先尝试用h5py读取（新格式）
        with h5py.File(pred_file, 'r') as f:
            prob_volume = f['softmax_probabilities'][()]
            print(f"成功加载HDF5格式预测文件")
            print(f"  测试被试: {f.attrs.get('test_subject', 'unknown')}")
            print(f"  形状: {prob_volume.shape}")
            return prob_volume  # (384, 336, 256, 102)
    except:
        # 如果失败，尝试用scipy读取（旧格式）
        try:
            data = scipy.io.loadmat(pred_file)
            print(f"成功加载MAT v5格式预测文件")
            return data['softmax_probabilities']
        except:
            raise ValueError(f"无法加载预测文件: {pred_file}")

def load_ground_truth(mat_file: Path):
    """加载真实标签"""
    with h5py.File(mat_file, 'r') as f:
        region_labels = f['region_labels'][()]
        region_mask = f['region_mask'][()]
        
        # 处理转置
        if region_labels.shape != (384, 336, 256):
            region_labels = region_labels.T
        if region_mask.shape != (384, 336, 256):
            region_mask = region_mask.T
    
    return region_labels, region_mask

def compute_prediction_metrics(prob_volume):
    """计算预测指标"""
    # 预测类别（最大概率）
    pred_labels = np.argmax(prob_volume, axis=-1)  # (384, 336, 256)
    
    # 最大概率（置信度）
    max_prob = np.max(prob_volume, axis=-1)  # (384, 336, 256)
    
    # 熵（不确定性）- 只计算非背景体素
    epsilon = 1e-10
    entropy = -np.sum(prob_volume * np.log(prob_volume + epsilon), axis=-1)
    
    # Top-2概率差（另一种不确定性度量）
    sorted_probs = np.sort(prob_volume, axis=-1)
    prob_diff = sorted_probs[..., -1] - sorted_probs[..., -2]
    
    return {
        'pred_labels': pred_labels,
        'max_prob': max_prob,
        'entropy': entropy,
        'prob_diff': prob_diff
    }

def visualize_slices(prob_volume, gt_labels, gt_mask, slice_idx=128, save_path=None):
    """可视化某个切片的预测结果"""
    
    # 计算预测指标
    metrics = compute_prediction_metrics(prob_volume)
    
    # 选择切片
    prob_slice = prob_volume[:, :, slice_idx, :]
    gt_slice = gt_labels[:, :, slice_idx]
    mask_slice = gt_mask[:, :, slice_idx]
    pred_slice = metrics['pred_labels'][:, :, slice_idx]
    conf_slice = metrics['max_prob'][:, :, slice_idx]
    entropy_slice = metrics['entropy'][:, :, slice_idx]
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 真实标签
    im1 = axes[0, 0].imshow(gt_slice, cmap='tab20', vmin=0, vmax=101)
    axes[0, 0].set_title(f'Ground Truth Labels (slice {slice_idx})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # 2. 预测标签
    im2 = axes[0, 1].imshow(pred_slice, cmap='tab20', vmin=0, vmax=101)
    axes[0, 1].set_title(f'Predicted Labels (slice {slice_idx})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # 3. 预测错误（只在有效区域显示）
    error_map = np.zeros_like(pred_slice, dtype=float)
    valid_mask = (mask_slice > 0) & (gt_slice > 0)
    error_map[valid_mask] = (pred_slice[valid_mask] != gt_slice[valid_mask] - 1).astype(float)
    im3 = axes[0, 2].imshow(error_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[0, 2].set_title('Prediction Errors')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # 4. 置信度图
    conf_masked = np.ma.masked_where(mask_slice == 0, conf_slice)
    im4 = axes[1, 0].imshow(conf_masked, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Confidence (Max Probability)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # 5. 熵（不确定性）
    entropy_masked = np.ma.masked_where(mask_slice == 0, entropy_slice)
    im5 = axes[1, 1].imshow(entropy_masked, cmap='hot')
    axes[1, 1].set_title('Uncertainty (Entropy)')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # 6. 概率差（Top1 - Top2）
    prob_diff_slice = metrics['prob_diff'][:, :, slice_idx]
    prob_diff_masked = np.ma.masked_where(mask_slice == 0, prob_diff_slice)
    im6 = axes[1, 2].imshow(prob_diff_masked, cmap='coolwarm', vmin=0, vmax=1)
    axes[1, 2].set_title('Probability Margin (Top1 - Top2)')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_class_performance(prob_volume, gt_labels, gt_mask):
    """分析每个类别的性能"""
    metrics = compute_prediction_metrics(prob_volume)
    pred_labels = metrics['pred_labels']
    
    # 只分析有效体素
    valid_mask = (gt_mask > 0) & (gt_labels > 0)
    
    # 调整真实标签（从1-102到0-101）
    gt_adjusted = gt_labels - 1
    
    # 计算每个类别的准确率
    class_performance = {}
    
    for class_id in range(102):
        class_mask = valid_mask & (gt_adjusted == class_id)
        if np.sum(class_mask) > 0:
            correct = np.sum(pred_labels[class_mask] == class_id)
            total = np.sum(class_mask)
            accuracy = correct / total
            
            # 平均置信度
            avg_confidence = np.mean(metrics['max_prob'][class_mask])
            
            # 平均熵
            avg_entropy = np.mean(metrics['entropy'][class_mask])
            
            class_performance[class_id] = {
                'accuracy': accuracy,
                'total_voxels': total,
                'correct_voxels': correct,
                'avg_confidence': avg_confidence,
                'avg_entropy': avg_entropy
            }
    
    return class_performance

def plot_class_performance(class_performance, save_path=None):
    """绘制类别性能图"""
    classes = sorted(class_performance.keys())
    accuracies = [class_performance[c]['accuracy'] for c in classes]
    confidences = [class_performance[c]['avg_confidence'] for c in classes]
    voxel_counts = [class_performance[c]['total_voxels'] for c in classes]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 准确率条形图
    axes[0, 0].bar(classes, accuracies)
    axes[0, 0].set_xlabel('Class ID')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Per-Class Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率vs置信度散点图
    axes[0, 1].scatter(confidences, accuracies, s=[v/100 for v in voxel_counts], alpha=0.6)
    axes[0, 1].set_xlabel('Average Confidence')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy vs Confidence (size = voxel count)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 体素数量分布
    axes[1, 0].bar(classes, voxel_counts)
    axes[1, 0].set_xlabel('Class ID')
    axes[1, 0].set_ylabel('Number of Voxels')
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 准确率直方图
    axes[1, 1].hist(accuracies, bins=20, edgecolor='black')
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('Number of Classes')
    axes[1, 1].set_title('Accuracy Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计
    print("\n===== 类别性能统计 =====")
    print(f"平均准确率: {np.mean(accuracies):.4f}")
    print(f"中位数准确率: {np.median(accuracies):.4f}")
    print(f"最高准确率: {np.max(accuracies):.4f} (Class {classes[np.argmax(accuracies)]})")
    print(f"最低准确率: {np.min(accuracies):.4f} (Class {classes[np.argmin(accuracies)]})")
    
    # 打印Top-5最好和最差的类别
    sorted_indices = np.argsort(accuracies)
    print("\nTop-5 最差类别:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        class_id = classes[idx]
        perf = class_performance[class_id]
        print(f"  Class {class_id}: Acc={perf['accuracy']:.4f}, Voxels={perf['total_voxels']}")
    
    print("\nTop-5 最好类别:")
    for i in range(max(0, len(sorted_indices)-5), len(sorted_indices)):
        idx = sorted_indices[i]
        class_id = classes[idx]
        perf = class_performance[class_id]
        print(f"  Class {class_id}: Acc={perf['accuracy']:.4f}, Voxels={perf['total_voxels']}")

def compute_overall_metrics(prob_volume, gt_labels, gt_mask):
    """计算整体指标"""
    metrics = compute_prediction_metrics(prob_volume)
    pred_labels = metrics['pred_labels']
    
    # 只分析有效体素
    valid_mask = (gt_mask > 0) & (gt_labels > 0)
    
    # 调整真实标签
    gt_adjusted = gt_labels - 1
    
    # 计算准确率
    correct = np.sum(pred_labels[valid_mask] == gt_adjusted[valid_mask])
    total = np.sum(valid_mask)
    accuracy = correct / total
    
    # 平均置信度
    avg_confidence = np.mean(metrics['max_prob'][valid_mask])
    
    # 平均熵
    avg_entropy = np.mean(metrics['entropy'][valid_mask])
    
    print("\n===== 整体性能指标 =====")
    print(f"总体素数: {total:,}")
    print(f"正确预测: {correct:,}")
    print(f"准确率: {accuracy:.4f}")
    print(f"平均置信度: {avg_confidence:.4f}")
    print(f"平均熵: {avg_entropy:.4f}")
    
    return {
        'accuracy': accuracy,
        'total_voxels': total,
        'correct_voxels': correct,
        'avg_confidence': avg_confidence,
        'avg_entropy': avg_entropy
    }

def main():
    parser = argparse.ArgumentParser(description='可视化1D训练的3D预测结果')
    parser.add_argument('--pred_file', type=str, required=True,
                       help='预测概率文件路径（.mat）')
    parser.add_argument('--gt_file', type=str, required=True,
                       help='真实标签文件路径（3D MAT）')
    parser.add_argument('--slice_idx', type=int, default=128,
                       help='要可视化的切片索引')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='保存图像的目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载预测概率...")
    prob_volume = load_predictions(Path(args.pred_file))
    print(f"概率体积形状: {prob_volume.shape}")
    
    print("加载真实标签...")
    gt_labels, gt_mask = load_ground_truth(Path(args.gt_file))
    print(f"标签形状: {gt_labels.shape}")
    print(f"掩膜形状: {gt_mask.shape}")
    
    # 计算整体指标
    overall_metrics = compute_overall_metrics(prob_volume, gt_labels, gt_mask)
    
    # 分析类别性能
    print("\n分析类别性能...")
    class_performance = analyze_class_performance(prob_volume, gt_labels, gt_mask)
    
    # 可视化切片
    print(f"\n可视化切片 {args.slice_idx}...")
    visualize_slices(
        prob_volume, gt_labels, gt_mask,
        slice_idx=args.slice_idx,
        save_path=output_dir / f'slice_{args.slice_idx}_visualization.png'
    )
    
    # 绘制类别性能
    plot_class_performance(
        class_performance,
        save_path=output_dir / 'class_performance.png'
    )
    
    print(f"\n所有图像已保存至: {output_dir}")

if __name__ == '__main__':
    main()