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

def smooth_2d_with_mask(softmax_vol, mask_valid, kernel_size=3, slice_axis=2):
    """
    对softmax概率进行2D平滑，使用mask进行有效像素加权
    
    Args:
        softmax_vol: (D, H, W, 102) softmax概率
        mask_valid: (D, H, W) 有效像素掩膜
        kernel_size: 卷积核大小 (3 或 7)
        slice_axis: 切片轴向 (0=sagittal, 1=coronal, 2=axial)
    
    Returns:
        smoothed_vol: (D, H, W, 102) 平滑后概率
    """
    print(f"执行2D平滑 (kernel_size={kernel_size}, slice_axis={slice_axis})...")
    
    # 保存原始形状用于最后还原
    original_shape = softmax_vol.shape
    original_mask_shape = mask_valid.shape
    
    # 根据不同轴向转置数据
    if slice_axis == 0:  # sagittal (沿D轴切片)
        softmax_vol = softmax_vol.transpose(1, 2, 0, 3)  # (H, W, D, C)
        mask_valid = mask_valid.transpose(1, 2, 0)       # (H, W, D)
    elif slice_axis == 1:  # coronal (沿H轴切片)
        softmax_vol = softmax_vol.transpose(0, 2, 1, 3)  # (D, W, H, C)
        mask_valid = mask_valid.transpose(0, 2, 1)       # (D, W, H)
    # slice_axis == 2 (axial) 保持原始形状
    
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
    
    # 还原到原始轴向
    if slice_axis == 0:  # sagittal -> 还原
        smoothed_vol = smoothed_vol.transpose(2, 0, 1, 3)  # (D, H, W, C)
    elif slice_axis == 1:  # coronal -> 还原
        smoothed_vol = smoothed_vol.transpose(0, 2, 1, 3)  # (D, H, W, C)
    # slice_axis == 2 (axial) 无需转换
    
    print(f"2D平滑完成 (kernel_size={kernel_size})")
    return smoothed_vol

def smooth_2d_with_mask_fast(softmax_vol, mask_valid, kernel_size=3, slice_axis=2):
    """
    快速版本的2D平滑，使用ndimage
    
    Args:
        slice_axis: 切片轴向 (0=sagittal, 1=coronal, 2=axial)
                   - 0: 沿第0维切片，在(H,W)平面平滑
                   - 1: 沿第1维切片，在(D,W)平面平滑  
                   - 2: 沿第2维切片，在(D,H)平面平滑
    """
    axis_names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    print(f"执行快速2D平滑 (kernel_size={kernel_size}, {axis_names[slice_axis]}方向)...")
    
    # 根据切片轴向调整数据布局
    if slice_axis == 0:
        # 沿第0维切片：(384,336,256,102) -> 在(336,256)平面平滑
        vol = softmax_vol  # (D, H, W, C)
        mask = mask_valid  # (D, H, W)
        n_slices, plane_h, plane_w = vol.shape[0], vol.shape[1], vol.shape[2]
    elif slice_axis == 1:
        # 沿第1维切片：需要转置 -> 在(384,256)平面平滑
        vol = np.transpose(softmax_vol, (1, 0, 2, 3))  # (H, D, W, C)
        mask = np.transpose(mask_valid, (1, 0, 2))     # (H, D, W)
        n_slices, plane_h, plane_w = vol.shape[0], vol.shape[1], vol.shape[2]
    elif slice_axis == 2:
        # 沿第2维切片：需要转置 -> 在(384,336)平面平滑
        vol = np.transpose(softmax_vol, (2, 0, 1, 3))  # (W, D, H, C)
        mask = np.transpose(mask_valid, (2, 0, 1))     # (W, D, H)
        n_slices, plane_h, plane_w = vol.shape[0], vol.shape[1], vol.shape[2]
    
    D, H, W, C = vol.shape
    smoothed_vol = np.zeros_like(vol)
    
    # 创建卷积核权重
    kernel = np.ones((kernel_size, kernel_size))
    
    # 对每个切片和每个类别进行处理
    for d in range(D):
        if d % 50 == 0:
            print(f"  处理{axis_names[slice_axis]}切片 {d}/{D}")
            
        mask_slice = mask[d, :, :]
        
        for c in range(C):
            prob_2d = vol[d, :, :, c]
            
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
    
    # 转置回原始轴序
    if slice_axis == 0:
        result = smoothed_vol  # 不需要转置
    elif slice_axis == 1:
        result = np.transpose(smoothed_vol, (1, 0, 2, 3))  # (H, D, W, C) -> (D, H, W, C)
    elif slice_axis == 2:
        result = np.transpose(smoothed_vol, (1, 2, 0, 3))  # (W, D, H, C) -> (D, H, W, C)
    
    print(f"快速2D平滑完成 (kernel_size={kernel_size}, {axis_names[slice_axis]}方向)")
    return result

def get_predictions_from_softmax(softmax_vol):
    """从softmax概率获取预测标签"""
    return np.argmax(softmax_vol, axis=-1)

def renorm_softmax(vol):
    """按体素重新归一化softmax概率，确保每体素∑p=1"""
    s = vol.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return vol / s

def compute_uncertainty_metrics(softmax_vol):
    """计算不确定性指标：归一化熵和概率间隔"""
    # 归一化熵 H_norm = -∑ p_i log p_i / log(102)
    epsilon = 1e-10
    log_probs = np.log(softmax_vol + epsilon)
    entropy = -np.sum(softmax_vol * log_probs, axis=-1)
    max_entropy = np.log(softmax_vol.shape[-1])  # log(102)
    normalized_entropy = entropy / max_entropy
    
    # 概率间隔 margin = p_top1 - p_top2
    sorted_probs = np.sort(softmax_vol, axis=-1)
    margin = sorted_probs[..., -1] - sorted_probs[..., -2]  # top1 - top2
    
    return normalized_entropy, margin

def smooth_2d_class_consistent(softmax_vol, mask_valid, kernel_size=3, slice_axis=2):
    """
    同类门控平滑：只在邻域内同类预测像素上做平均
    边界不过度平滑，保持类别一致性
    """
    axis_names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    print(f"执行同类门控平滑 (kernel_size={kernel_size}, {axis_names[slice_axis]}方向)...")
    
    # 根据切片轴向调整数据布局
    if slice_axis == 0:
        vol = softmax_vol  # (D, H, W, C)
        mask = mask_valid  # (D, H, W)
    elif slice_axis == 1:
        vol = np.transpose(softmax_vol, (1, 0, 2, 3))  # (H, D, W, C)
        mask = np.transpose(mask_valid, (1, 0, 2))     # (H, D, W)
    elif slice_axis == 2:
        vol = np.transpose(softmax_vol, (2, 0, 1, 3))  # (W, D, H, C)
        mask = np.transpose(mask_valid, (2, 0, 1))     # (W, D, H)
    
    D, H, W, C = vol.shape
    smoothed_vol = np.zeros_like(vol)
    
    # 创建卷积核权重
    kernel = np.ones((kernel_size, kernel_size))
    
    # 对每个切片进行处理
    for d in range(D):
        if d % 50 == 0:
            print(f"  处理{axis_names[slice_axis]}切片 {d}/{D}")
            
        prob_slice = vol[d, :, :, :]  # (H, W, C)
        mask_slice = mask[d, :, :]    # (H, W)
        
        # 获取预测标签
        pred_slice = np.argmax(prob_slice, axis=-1)  # (H, W)
        
        # 对每个类别单独处理
        for c in range(C):
            prob_2d_c = prob_slice[:, :, c]  # (H, W)
            
            # 构造同类掩膜：pred_slice == c 且在有效区域内
            mask_c = ((pred_slice == c) & (mask_slice > 0)).astype(float)
            
            if np.sum(mask_c) > 0:  # 该类别存在
                # 同类门控平滑：只在同类邻域内平均
                numerator = ndimage.convolve(prob_2d_c * mask_c, kernel, mode='constant', cval=0.0)
                denominator = ndimage.convolve(mask_c, kernel, mode='constant', cval=0.0) + 1e-8
                
                smoothed_2d_c = numerator / denominator
                
                # 对于没有同类邻域的像素，保持原值
                no_neighbors = (denominator <= 1e-8) & (mask_slice > 0)
                smoothed_2d_c[no_neighbors] = prob_2d_c[no_neighbors]
                
            else:
                # 该类别不存在，保持原值
                smoothed_2d_c = prob_2d_c.copy()
            
            smoothed_vol[d, :, :, c] = smoothed_2d_c
    
    # 转置回原始轴序
    if slice_axis == 0:
        result = smoothed_vol
    elif slice_axis == 1:
        result = np.transpose(smoothed_vol, (1, 0, 2, 3))
    elif slice_axis == 2:
        result = np.transpose(smoothed_vol, (1, 2, 0, 3))
    
    print(f"同类门控平滑完成 (kernel_size={kernel_size}, {axis_names[slice_axis]}方向)")
    return result

def smooth_2d_uncertainty_gated(softmax_vol, mask_valid, kernel_size=3, slice_axis=2, 
                               gate_type='entropy', tau=0.5, kappa=0.1):
    """
    不确定性门控平滑：只在不确定的地方适度平滑
    
    Args:
        gate_type: 'entropy' 或 'margin'
        tau: 阈值参数 (entropy: ~0.5, margin: ~0.3)
        kappa: 平滑参数 (~0.1)
    """
    axis_names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    print(f"执行不确定性门控平滑 ({gate_type}, kernel_size={kernel_size}, {axis_names[slice_axis]}方向)...")
    print(f"  门控参数: tau={tau}, kappa={kappa}")
    
    # 计算不确定性指标
    normalized_entropy, margin = compute_uncertainty_metrics(softmax_vol)
    
    # 计算门控权重
    if gate_type == 'entropy':
        # 高熵 -> 高权重 (更多平滑)
        gate_input = (normalized_entropy - tau) / kappa
        uncertainty_weight = 1 / (1 + np.exp(-gate_input))  # sigmoid
    elif gate_type == 'margin':
        # 低间隔 -> 高权重 (更多平滑)
        gate_input = (tau - margin) / kappa
        uncertainty_weight = 1 / (1 + np.exp(-gate_input))  # sigmoid
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")
    
    # 先做常规平滑
    smoothed_vol = smooth_2d_with_mask_fast(softmax_vol, mask_valid, kernel_size, slice_axis)
    
    # 不确定性门控融合：p_out = (1-w) * p_in + w * p_smooth
    # 只在有效区域应用门控，使用向量化广播
    valid_mask = mask_valid > 0
    w = uncertainty_weight
    
    # 向量化融合：一次广播完成所有通道
    gated_vol = np.where(
        valid_mask[..., None],  # 广播到所有通道
        (1 - w)[..., None] * softmax_vol + w[..., None] * smoothed_vol,
        softmax_vol  # 无效区域保持原值
    )
    
    # 统计门控效果
    mean_weight = np.mean(uncertainty_weight[valid_mask])
    high_uncertainty_ratio = np.mean((uncertainty_weight[valid_mask] > 0.5))
    
    print(f"  门控统计: 平均权重={mean_weight:.3f}, 高不确定性比例={high_uncertainty_ratio:.2%}")
    print(f"不确定性门控平滑完成")
    return gated_vol

def verify_probability_normalization(softmax_vol, mask_valid, method_name=""):
    """验证概率归一化是否正确"""
    print(f"验证{method_name}概率归一化...")
    
    # 只检查有效区域
    valid_voxels = mask_valid > 0
    if not np.any(valid_voxels):
        print("  警告: 没有有效体素用于验证")
        return
    
    # 计算有效体素的概率和
    prob_sums = np.sum(softmax_vol[valid_voxels], axis=1)  # (n_valid_voxels,)
    
    # 检查概率和接近1的体素数量
    tolerance = 1e-6
    perfect_count = np.sum(np.abs(prob_sums - 1.0) < tolerance)
    total_count = len(prob_sums)
    
    # 统计偏差
    max_deviation = np.max(np.abs(prob_sums - 1.0))
    mean_deviation = np.mean(np.abs(prob_sums - 1.0))
    
    print(f"  完美归一化体素: {perfect_count:,}/{total_count:,} ({perfect_count/total_count*100:.2f}%)")
    print(f"  最大偏差: {max_deviation:.8f}")
    print(f"  平均偏差: {mean_deviation:.8f}")
    
    if perfect_count == total_count:
        print(f"  ✅ 所有体素概率和严格为1")
    else:
        print(f"  ⚠️ {total_count-perfect_count}个体素概率和偏离1")
    
    return perfect_count == total_count

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
    parser.add_argument('--slice_axis', type=int, default=None,
                       help='切片轴向 (0=sagittal, 1=coronal, 2=axial)。None=自动评估所有轴向')
    parser.add_argument('--compare_all_axes', action='store_true',
                       help='比较所有轴向的平滑效果，确定最佳轴向')
    parser.add_argument('--use_class_gating', action='store_true',
                       help='启用同类门控平滑（保护类别边界）')
    parser.add_argument('--use_uncertainty_gating', action='store_true',
                       help='启用不确定性门控平滑（基于熵或置信度）')
    parser.add_argument('--uncertainty_type', type=str, default='entropy', choices=['entropy', 'margin'],
                       help='不确定性度量类型: entropy（熵）或 margin（置信边界）')
    parser.add_argument('--uncertainty_tau', type=float, default=0.5,
                       help='不确定性阈值参数τ (0-1)')
    parser.add_argument('--uncertainty_kappa', type=float, default=0.1,
                       help='不确定性融合强度κ (0-1)')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 7],
                       help='平滑核大小列表（默认: 3 7）')
    
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
    verify_probability_normalization(softmax_vol, region_mask_binary, "原始")
    pred_original = get_predictions_from_softmax(softmax_vol)
    y_pred_original = pred_original[valid_indices]  # 0-101预测标签
    y_scores_original = softmax_vol[valid_indices]   # 102通道概率，含背景通道0
    
    # 多轴向评估或单一轴向平滑
    if args.compare_all_axes:
        print("\n=== 多轴向平滑效果对比 ===")
        axis_results = {}
        axis_names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
        
        for axis in [0, 1, 2]:
            print(f"\n--- 评估{axis_names[axis]}轴向 (axis={axis}) ---")
            
            # 3x3平滑
            if args.fast_smooth:
                softmax_k3_axis = smooth_2d_with_mask_fast(softmax_vol, region_mask_binary, kernel_size=3, slice_axis=axis)
            else:
                softmax_k3_axis = smooth_2d_with_mask(softmax_vol, region_mask_binary, kernel_size=3, slice_axis=axis)
            
            # 归一化
            softmax_k3_axis = renorm_softmax(softmax_k3_axis)
            verify_probability_normalization(softmax_k3_axis, region_mask_binary, f"3x3平滑({axis_names[axis]})")
            
            # 评估
            pred_k3_axis = get_predictions_from_softmax(softmax_k3_axis)
            y_pred_k3_axis = pred_k3_axis[valid_indices]
            y_scores_k3_axis = softmax_k3_axis[valid_indices]
            
            metrics_k3_axis = evaluate_predictions(y_true, y_pred_k3_axis, y_scores_k3_axis)
            
            # 保存结果
            axis_results[f'{axis_names[axis]}_axis{axis}'] = metrics_k3_axis
            print(f"  {axis_names[axis]}轴向指标: Acc={metrics_k3_axis['accuracy']:.4f}, F1={metrics_k3_axis['macro_f1']:.4f}, AUPRC={metrics_k3_axis.get('macro_auprc', 0):.4f}")
        
        # 比较结果，找出最佳轴向
        print("\n=== 轴向对比结果 ===")
        best_axis = None
        best_f1 = -1
        
        for axis in [0, 1, 2]:
            key = f'{axis_names[axis]}_axis{axis}'
            f1 = axis_results[key]['macro_f1']
            auprc = axis_results[key].get('macro_auprc', 0)
            acc = axis_results[key]['accuracy']
            print(f"{axis_names[axis].ljust(9)} (axis={axis}): Acc={acc:.4f}, Macro F1={f1:.4f}, AUPRC={auprc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_axis = axis
        
        print(f"\n🏆 最佳轴向: {axis_names[best_axis]} (axis={best_axis}), Macro F1={best_f1:.4f}")
        print(f"建议使用: --slice_axis {best_axis}")
        
        # 保存轴向对比结果
        import json
        axis_comparison = {}
        for axis in [0, 1, 2]:
            key = f'{axis_names[axis]}_axis{axis}'
            axis_comparison[f'axis_{axis}_{axis_names[axis]}'] = {
                'accuracy': float(axis_results[key]['accuracy']),
                'macro_f1': float(axis_results[key]['macro_f1']),
                'kappa': float(axis_results[key]['kappa']),
            }
            if 'macro_auprc' in axis_results[key]:
                axis_comparison[f'axis_{axis}_{axis_names[axis]}']['macro_auprc'] = float(axis_results[key]['macro_auprc'])
        
        with open(output_dir / 'axis_comparison.json', 'w') as f:
            json.dump(axis_comparison, f, indent=2)
        
        return  # 结束多轴向对比模式
    
    # 单一轴向模式
    slice_axis = args.slice_axis if args.slice_axis is not None else 2  # 默认axial
    axis_names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    print(f"\n使用{axis_names[slice_axis]}轴向进行平滑 (axis={slice_axis})")
    
    # 存储所有平滑结果
    all_smoothed = {}
    
    # 标准平滑方法
    for kernel_size in args.kernel_sizes:
        print(f"\n执行{kernel_size}x{kernel_size}标准平滑...")
        if args.fast_smooth:
            smoothed = smooth_2d_with_mask_fast(softmax_vol, region_mask_binary, kernel_size=kernel_size, slice_axis=slice_axis)
        else:
            smoothed = smooth_2d_with_mask(softmax_vol, region_mask_binary, kernel_size=kernel_size, slice_axis=slice_axis)
        
        # 平滑后按体素做softmax归一化（必要！）
        print(f"重新归一化{kernel_size}x{kernel_size}平滑结果...")
        smoothed = renorm_softmax(smoothed)
        verify_probability_normalization(smoothed, region_mask_binary, f"{kernel_size}x{kernel_size}标准平滑")
        
        all_smoothed[f'smooth_k{kernel_size}'] = smoothed
    
    # 门控平滑方法（仅使用最小核，通常是3x3）
    base_kernel = min(args.kernel_sizes)
    
    if args.use_class_gating:
        print(f"\n执行{base_kernel}x{base_kernel}同类门控平滑...")
        smoothed_class = smooth_2d_class_consistent(softmax_vol, region_mask_binary, kernel_size=base_kernel, slice_axis=slice_axis)
        smoothed_class = renorm_softmax(smoothed_class)
        verify_probability_normalization(smoothed_class, region_mask_binary, f"{base_kernel}x{base_kernel}同类门控平滑")
        all_smoothed[f'class_gated_k{base_kernel}'] = smoothed_class
        
    if args.use_uncertainty_gating:
        print(f"\n执行{base_kernel}x{base_kernel}不确定性门控平滑 ({args.uncertainty_type})...")
        smoothed_uncertain = smooth_2d_uncertainty_gated(
            softmax_vol, region_mask_binary, 
            kernel_size=base_kernel, slice_axis=slice_axis,
            gate_type=args.uncertainty_type,
            tau=args.uncertainty_tau,
            kappa=args.uncertainty_kappa
        )
        smoothed_uncertain = renorm_softmax(smoothed_uncertain)
        verify_probability_normalization(smoothed_uncertain, region_mask_binary, f"{base_kernel}x{base_kernel}不确定性门控平滑")
        all_smoothed[f'uncertainty_{args.uncertainty_type}_k{base_kernel}'] = smoothed_uncertain
    
    # 评估所有方法
    print("\n评估预测结果...")
    
    # 原始预测
    metrics_original = evaluate_predictions(y_true, y_pred_original, y_scores_original)
    all_metrics = {'original': metrics_original}
    
    # 评估所有平滑方法
    for method_name, smoothed_vol in all_smoothed.items():
        pred_smoothed = get_predictions_from_softmax(smoothed_vol)
        y_pred_smoothed = pred_smoothed[valid_indices]
        y_scores_smoothed = smoothed_vol[valid_indices]
        
        metrics_smoothed = evaluate_predictions(y_true, y_pred_smoothed, y_scores_smoothed)
        all_metrics[method_name] = metrics_smoothed
    
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
    if all_smoothed:
        print("保存平滑后预测结果...")
        
        # 从HDF5文件提取测试被试信息
        with h5py.File(args.pred_file, 'r') as f:
            test_subject = f.attrs.get('test_subject', 0)
            test_file_1d = f.attrs.get('test_file_1d', '')
            test_file_3d = f.attrs.get('test_file_3d', '')
        
        # 保存每种平滑方法的结果
        saved_files = []
        for method_name, smoothed_vol in all_smoothed.items():
            # 生成文件名
            file_path = output_dir / f'predictions_{method_name}_test{test_subject}.mat'
            print(f"保存{method_name}结果: {file_path}")
            print(f"  体积大小: ~{smoothed_vol.nbytes / (1024**3):.1f}GB")
            
            pred_smoothed = get_predictions_from_softmax(smoothed_vol)
            
            with h5py.File(str(file_path), 'w') as f:
                # 保存概率数据，使用压缩减小文件大小
                f.create_dataset('softmax_probabilities', 
                               data=smoothed_vol.astype(np.float32),
                               compression='gzip', 
                               compression_opts=4)
                f.create_dataset('predicted_labels',
                               data=pred_smoothed.astype(np.uint8),
                               compression='gzip',
                               compression_opts=4)
                
                # 保存元数据
                f.attrs['test_subject'] = int(test_subject)
                f.attrs['test_file_1d'] = str(test_file_1d)
                f.attrs['test_file_3d'] = str(test_file_3d)
                f.attrs['smooth_method'] = method_name
                f.attrs['slice_axis'] = slice_axis
                f.attrs['shape'] = smoothed_vol.shape
                f.attrs['prob_range'] = [float(smoothed_vol.min()), float(smoothed_vol.max())]
                
                # 根据方法类型保存特定参数
                if 'uncertainty' in method_name:
                    f.attrs['uncertainty_type'] = args.uncertainty_type
                    f.attrs['uncertainty_tau'] = args.uncertainty_tau  
                    f.attrs['uncertainty_kappa'] = args.uncertainty_kappa
            
            saved_files.append(f"  - predictions_{method_name}_test{test_subject}.mat")
    
        print(f"\n所有结果已保存至: {output_dir}")
        print("包含文件:")
        print("  - evaluation_results.json: 评估指标汇总")
        print("  - confusion_matrices_comparison.png: 混淆矩阵对比")
        for saved_file in saved_files:
            print(saved_file)

if __name__ == '__main__':
    main()