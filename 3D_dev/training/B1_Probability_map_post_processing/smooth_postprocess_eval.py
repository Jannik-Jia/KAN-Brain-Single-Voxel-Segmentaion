#!/usr/bin/env python3
"""
体素级softmax概率图的轻量平滑后处理和评估
- 2D平均平滑 (3x3, 7x7核)
- 使用valid mask进行有效像素加权
- 评估Macro F1、AUPRC、准确率、κ系数
- 生成混淆矩阵对比分析
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import h5py
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    confusion_matrix, average_precision_score, roc_auc_score
)
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

def load_predictions_and_gt(pred_file: Path, gt_file: Path):
    """加载预测概率和真实标签"""
    # 加载预测（只使用HDF5格式，训练脚本统一输出HDF5）
    with h5py.File(pred_file, 'r') as f:
        softmax_vol = f['softmax_probabilities'][()]  # (384, 336, 256, 102)
        test_subject = f.attrs.get('test_subject', 'unknown')
        print(f"成功加载HDF5格式预测文件，测试被试: {test_subject}")
        print(f"  预测概率体积形状: {softmax_vol.shape}")
    
    # 加载真实标签
    with h5py.File(gt_file, 'r') as f:
        region_labels = f['region_labels'][()]
        region_mask = f['region_mask'][()]
        
        # 严格形状验证：不允许隐式轴转换，必须显式匹配预期形状
        assert region_labels.shape == (384, 336, 256), \
            f"region_labels 形状不符合预期 (384, 336, 256)，实际为 {region_labels.shape}，文件: {gt_file}"
        
        assert region_mask.shape == (384, 336, 256), \
            f"region_mask 形状不符合预期 (384, 336, 256)，实际为 {region_mask.shape}，文件: {gt_file}"
    
    return softmax_vol, region_labels, region_mask

def smooth_2d_with_mask(softmax_vol, mask_valid, kernel_size=3):
    """
    对softmax概率进行2D平滑，使用mask进行有效像素加权
    
    Args:
        softmax_vol: (D, H, W, 102) softmax概率
        mask_valid: (D, H, W) 有效像素掩膜
        kernel_size: 卷积核大小 (3 或 7)
    
    Returns:
        smoothed_vol: (D, H, W, 102) 平滑后概率
    """
    print(f"执行2D平滑 (kernel_size={kernel_size})...")
    
    D, H, W, C = softmax_vol.shape
    smoothed_vol = np.zeros_like(softmax_vol)
    
    # 创建卷积核
    pad = kernel_size // 2
    
    # 对每个深度切片进行处理
    for d in range(D):
        if d % 50 == 0:
            print(f"  处理切片 {d}/{D}")
            
        prob_slice = softmax_vol[d, :, :, :]  # (H, W, 102)
        mask_slice = mask_valid[d, :, :]      # (H, W)
        
        # 对每个类别单独处理
        for c in range(C):
            prob_2d = prob_slice[:, :, c]  # (H, W)
            
            # 使用mask加权的平滑
            smoothed_2d = np.zeros_like(prob_2d)
            
            # 对每个像素进行邻域平滑
            for i in range(H):
                for j in range(W):
                    if mask_slice[i, j] == 0:
                        continue  # 跳过无效像素
                    
                    # 定义邻域范围
                    i_start = max(0, i - pad)
                    i_end = min(H, i + pad + 1)
                    j_start = max(0, j - pad)
                    j_end = min(W, j + pad + 1)
                    
                    # 提取邻域
                    prob_patch = prob_2d[i_start:i_end, j_start:j_end]
                    mask_patch = mask_slice[i_start:i_end, j_start:j_end]
                    
                    # 只考虑有效像素
                    valid_indices = mask_patch > 0
                    if np.sum(valid_indices) > 0:
                        # 加权平均（有效像素的平均）
                        smoothed_2d[i, j] = np.mean(prob_patch[valid_indices])
                    else:
                        # 如果邻域内没有有效像素，保持原值
                        smoothed_2d[i, j] = prob_2d[i, j]
            
            smoothed_vol[d, :, :, c] = smoothed_2d
    
    print(f"2D平滑完成 (kernel_size={kernel_size})")
    return smoothed_vol

def smooth_2d_with_mask_fast(softmax_vol, mask_valid, kernel_size=3):
    """
    快速版本的2D平滑，使用ndimage
    """
    print(f"执行快速2D平滑 (kernel_size={kernel_size})...")
    
    D, H, W, C = softmax_vol.shape
    smoothed_vol = np.zeros_like(softmax_vol)
    
    # 创建卷积核权重
    kernel = np.ones((kernel_size, kernel_size))
    
    # 对每个深度切片和每个类别进行处理
    for d in range(D):
        if d % 50 == 0:
            print(f"  处理切片 {d}/{D}")
            
        mask_slice = mask_valid[d, :, :]
        
        for c in range(C):
            prob_2d = softmax_vol[d, :, :, c]
            
            # 将无效区域设为0
            prob_masked = prob_2d * mask_slice
            
            # 使用ndimage进行卷积
            numerator = ndimage.convolve(prob_masked, kernel, mode='constant', cval=0.0)
            denominator = ndimage.convolve(mask_slice.astype(float), kernel, mode='constant', cval=0.0)
            
            # 避免除零
            valid_denom = denominator > 0
            smoothed_2d = np.zeros_like(prob_2d)
            smoothed_2d[valid_denom] = numerator[valid_denom] / denominator[valid_denom]
            
            # 对于无有效邻域的像素，保持原值
            invalid_denom = (denominator == 0) & (mask_slice > 0)
            smoothed_2d[invalid_denom] = prob_2d[invalid_denom]
            
            smoothed_vol[d, :, :, c] = smoothed_2d
    
    print(f"快速2D平滑完成 (kernel_size={kernel_size})")
    return smoothed_vol

def get_predictions_from_softmax(softmax_vol):
    """从softmax概率获取预测标签"""
    return np.argmax(softmax_vol, axis=-1)

def compute_multiclass_auprc(y_true_onehot, y_scores, average='macro'):
    """
    计算多类别AUPRC
    
    Args:
        y_true_onehot: (n_samples, n_classes) one-hot编码的真实标签
        y_scores: (n_samples, n_classes) 预测概率
        average: 'macro' 或 'micro'
    
    Returns:
        auprc: AUPRC值
    """
    try:
        if average == 'macro':
            # 对每个类别计算AUPRC，然后平均
            auprcs = []
            for i in range(y_true_onehot.shape[1]):
                if np.sum(y_true_onehot[:, i]) > 0:  # 该类别存在正样本
                    auprc = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
                    auprcs.append(auprc)
            return np.mean(auprcs) if auprcs else 0.0
        else:
            # micro平均：所有类别一起计算
            return average_precision_score(y_true_onehot, y_scores, average='micro')
    except:
        return 0.0

def evaluate_predictions(y_true, y_pred, y_scores=None, class_names=None):
    """
    全面评估预测结果
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_scores: 预测概率 (可选，用于AUPRC计算)
        class_names: 类别名称 (可选)
    
    Returns:
        metrics: 评估指标字典
    """
    metrics = {}
    
    # 基本分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # 如果提供了概率分数，计算AUPRC
    if y_scores is not None:
        # 转换为one-hot编码
        n_classes = y_scores.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        metrics['macro_auprc'] = compute_multiclass_auprc(y_true_onehot, y_scores, 'macro')
        metrics['micro_auprc'] = compute_multiclass_auprc(y_true_onehot, y_scores, 'micro')
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_confusion_matrices(metrics_dict, save_path=None, max_classes=20):
    """绘制多个方法的混淆矩阵对比"""
    n_methods = len(metrics_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, metrics) in enumerate(metrics_dict.items()):
        cm = metrics['confusion_matrix']
        
        # 如果类别太多，只显示前N个
        if cm.shape[0] > max_classes:
            # 选择频率最高的类别
            class_counts = np.sum(cm, axis=1)
            top_classes = np.argsort(class_counts)[-max_classes:]
            cm_subset = cm[np.ix_(top_classes, top_classes)]
        else:
            cm_subset = cm
            top_classes = None
        
        # 归一化到[0,1]用于显示
        cm_norm = cm_subset.astype('float') / (cm_subset.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        im = axes[idx].imshow(cm_norm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(f'{method_name}\nAcc: {metrics["accuracy"]:.3f}, F1: {metrics["macro_f1"]:.3f}')
        
        # 添加colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
        
        if top_classes is not None:
            axes[idx].set_xlabel(f'Predicted Label (Top {max_classes} classes)')
            axes[idx].set_ylabel(f'True Label (Top {max_classes} classes)')
        else:
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_report(metrics_dict):
    """打印对比报告"""
    print("\n" + "="*80)
    print("后处理平滑效果对比报告")
    print("="*80)
    
    # 准备数据
    methods = list(metrics_dict.keys())
    metrics_names = ['accuracy', 'macro_f1', 'kappa', 'macro_auprc', 'micro_auprc']
    
    # 打印表头
    header = f"{'Method':<15}"
    for metric in metrics_names:
        if metric in metrics_dict[methods[0]]:
            header += f"{metric.upper():<12}"
    print(header)
    print("-" * len(header))
    
    # 打印每种方法的结果
    results_dict = {}
    for method in methods:
        row = f"{method:<15}"
        results_dict[method] = {}
        for metric in metrics_names:
            if metric in metrics_dict[method]:
                value = metrics_dict[method][metric]
                results_dict[method][metric] = value
                row += f"{value:<12.4f}"
        print(row)
    
    # 计算改进
    if 'original' in results_dict:
        print("\n" + "="*50)
        print("改进效果 (与原始预测对比)")
        print("="*50)
        
        baseline = results_dict['original']
        for method in methods:
            if method == 'original':
                continue
            
            print(f"\n{method} vs original:")
            for metric in metrics_names:
                if metric in baseline and metric in results_dict[method]:
                    original_val = baseline[metric]
                    new_val = results_dict[method][metric]
                    delta = new_val - original_val
                    print(f"  Δ{metric}: {delta:+.4f} ({original_val:.4f} → {new_val:.4f})")

def main():
    parser = argparse.ArgumentParser(description='体素级softmax概率图平滑后处理和评估')
    parser.add_argument('--pred_file', type=str, required=True,
                       help='预测概率文件路径（.mat）')
    parser.add_argument('--gt_file', type=str, required=True,
                       help='真实标签文件路径（3D MAT）')
    parser.add_argument('--output_dir', type=str, default='./smooth_eval_results',
                       help='结果保存目录')
    parser.add_argument('--fast_smooth', action='store_true',
                       help='使用快速平滑算法')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("加载数据...")
    softmax_vol, gt_labels, gt_mask = load_predictions_and_gt(
        Path(args.pred_file), Path(args.gt_file)
    )
    
    print(f"数据形状:")
    print(f"  Softmax概率: {softmax_vol.shape}")
    print(f"  真实标签: {gt_labels.shape}")
    print(f"  有效掩膜: {gt_mask.shape}")
    
    # 准备掩膜：区分平滑用和评估用
    region_mask_binary = (gt_mask > 0)           # 脑组织区域，用于平滑
    valid_mask = (gt_mask > 0) & (gt_labels > 0)  # 有标签区域，用于评估
    
    print(f"掩膜统计:")
    print(f"  脑组织体素数: {np.sum(region_mask_binary):,}")
    print(f"  有标签体素数: {np.sum(valid_mask):,}")
    print(f"  标签覆盖率: {np.sum(valid_mask)/np.sum(region_mask_binary)*100:.2f}%")
    
    # 提取有效像素用于评估
    valid_indices = np.where(valid_mask)
    y_true = gt_labels[valid_indices].astype(int)  # 保持0-101，与102通道对齐
    
    # 原始预测
    print("\n处理原始预测...")
    pred_original = get_predictions_from_softmax(softmax_vol)
    y_pred_original = pred_original[valid_indices]  # 0-101预测标签
    y_scores_original = softmax_vol[valid_indices]   # 102通道概率，含背景通道0
    
    # 3x3平滑（使用完整脑组织掩膜）
    print("\n执行3x3平滑...")
    if args.fast_smooth:
        softmax_k3 = smooth_2d_with_mask_fast(softmax_vol, region_mask_binary, kernel_size=3)
    else:
        softmax_k3 = smooth_2d_with_mask(softmax_vol, region_mask_binary, kernel_size=3)
    
    pred_k3 = get_predictions_from_softmax(softmax_k3)
    y_pred_k3 = pred_k3[valid_indices]
    y_scores_k3 = softmax_k3[valid_indices]
    
    # 7x7平滑（使用完整脑组织掩膜）
    print("\n执行7x7平滑...")
    if args.fast_smooth:
        softmax_k7 = smooth_2d_with_mask_fast(softmax_vol, region_mask_binary, kernel_size=7)
    else:
        softmax_k7 = smooth_2d_with_mask(softmax_vol, region_mask_binary, kernel_size=7)
    
    pred_k7 = get_predictions_from_softmax(softmax_k7)
    y_pred_k7 = pred_k7[valid_indices]
    y_scores_k7 = softmax_k7[valid_indices]
    
    # 评估所有方法
    print("\n评估预测结果...")
    
    metrics_original = evaluate_predictions(y_true, y_pred_original, y_scores_original)
    metrics_k3 = evaluate_predictions(y_true, y_pred_k3, y_scores_k3)
    metrics_k7 = evaluate_predictions(y_true, y_pred_k7, y_scores_k7)
    
    # 整合结果
    all_metrics = {
        'original': metrics_original,
        'smooth_k3': metrics_k3,
        'smooth_k7': metrics_k7
    }
    
    # 打印对比报告
    print_comparison_report(all_metrics)
    
    # 绘制混淆矩阵
    print("\n生成可视化...")
    plot_confusion_matrices(
        all_metrics, 
        save_path=output_dir / 'confusion_matrices_comparison.png'
    )
    
    # 保存详细结果
    import json
    results_summary = {}
    for method, metrics in all_metrics.items():
        results_summary[method] = {
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'kappa': float(metrics['kappa']),
        }
        if 'macro_auprc' in metrics:
            results_summary[method]['macro_auprc'] = float(metrics['macro_auprc'])
            results_summary[method]['micro_auprc'] = float(metrics['micro_auprc'])
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # 保存平滑后的预测结果
    print("保存平滑后预测结果...")
    
    # 从HDF5文件提取测试被试信息
    with h5py.File(args.pred_file, 'r') as f:
        test_subject = f.attrs.get('test_subject', 0)
        test_file_1d = f.attrs.get('test_file_1d', '')
        test_file_3d = f.attrs.get('test_file_3d', '')
    
    # 保存3x3平滑结果（HDF5格式，支持大体积）
    k3_path = output_dir / f'predictions_smooth_k3_test{test_subject}.mat'
    print(f"保存3x3平滑结果: {k3_path}")
    print(f"  体积大小: ~{softmax_k3.nbytes / (1024**3):.1f}GB")
    
    with h5py.File(str(k3_path), 'w') as f:
        # 保存概率数据，使用压缩减小文件大小
        f.create_dataset('softmax_probabilities', 
                       data=softmax_k3.astype(np.float32),
                       compression='gzip', 
                       compression_opts=4)
        f.create_dataset('predicted_labels',
                       data=pred_k3.astype(np.uint8),
                       compression='gzip',
                       compression_opts=4)
        
        # 保存元数据
        f.attrs['test_subject'] = int(test_subject)
        f.attrs['test_file_1d'] = str(test_file_1d)
        f.attrs['test_file_3d'] = str(test_file_3d)
        f.attrs['smooth_kernel_size'] = 3
        f.attrs['shape'] = softmax_k3.shape
        f.attrs['prob_range'] = [float(softmax_k3.min()), float(softmax_k3.max())]
    
    # 保存7x7平滑结果（HDF5格式，支持大体积）
    k7_path = output_dir / f'predictions_smooth_k7_test{test_subject}.mat'
    print(f"保存7x7平滑结果: {k7_path}")
    print(f"  体积大小: ~{softmax_k7.nbytes / (1024**3):.1f}GB")
    
    with h5py.File(str(k7_path), 'w') as f:
        # 保存概率数据，使用压缩减小文件大小
        f.create_dataset('softmax_probabilities', 
                       data=softmax_k7.astype(np.float32),
                       compression='gzip', 
                       compression_opts=4)
        f.create_dataset('predicted_labels',
                       data=pred_k7.astype(np.uint8),
                       compression='gzip',
                       compression_opts=4)
        
        # 保存元数据
        f.attrs['test_subject'] = int(test_subject)
        f.attrs['test_file_1d'] = str(test_file_1d)
        f.attrs['test_file_3d'] = str(test_file_3d)
        f.attrs['smooth_kernel_size'] = 7
        f.attrs['shape'] = softmax_k7.shape
        f.attrs['prob_range'] = [float(softmax_k7.min()), float(softmax_k7.max())]
    
    print(f"\n所有结果已保存至: {output_dir}")
    print("包含文件:")
    print("  - evaluation_results.json: 评估指标汇总")
    print("  - confusion_matrices_comparison.png: 混淆矩阵对比")
    print("  - predictions_smooth_k3_test*.mat: 3x3平滑预测结果")
    print("  - predictions_smooth_k7_test*.mat: 7x7平滑预测结果")

if __name__ == '__main__':
    main()