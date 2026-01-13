#!/usr/bin/env python3
"""
完整的评估指标模块
包含所有用于脑部体素分割任务的评估指标
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score
)
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ==================== 准确性指标 ====================

def compute_gross_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Gross Accuracy (GC) - 总体准确率

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    Returns:
        准确率 [0, 1]
    """
    return accuracy_score(y_true, y_pred)

def compute_top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 1) -> float:
    """
    Top-K 准确率

    Args:
        y_true: 真实标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)
        k: Top-K 中的 K

    Returns:
        Top-K 准确率 [0, 1]
    """
    # 获取每个样本的 Top-K 预测类别
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]  # (n_samples, k)

    # 检查真实标签是否在 Top-K 中
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

    return np.mean(correct)

def compute_top_k_accuracies(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """
    计算 Top-1, Top-3, Top-5 准确率

    Args:
        y_true: 真实标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)

    Returns:
        字典包含 top1, top3, top5 准确率
    """
    return {
        'top1_accuracy': compute_top_k_accuracy(y_true, y_probs, k=1),
        'top3_accuracy': compute_top_k_accuracy(y_true, y_probs, k=3),
        'top5_accuracy': compute_top_k_accuracy(y_true, y_probs, k=5),
    }

# ==================== 类别平衡指标 ====================

def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-F1 分数
    核心指标，用于模型 Checkpoint 选择

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    Returns:
        Macro-F1 分数 [0, 1]
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def compute_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted F1 分数

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    Returns:
        Weighted F1 分数 [0, 1]
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced Accuracy - 平衡准确率
    考虑类别不平衡的准确率

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    Returns:
        平衡准确率 [0, 1]
    """
    return balanced_accuracy_score(y_true, y_pred)

# ==================== 一致性指标 ====================

def compute_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cohen's Kappa 系数
    衡量预测与真实标签的一致性

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)

    Returns:
        Kappa 系数 [-1, 1]，1表示完全一致，0表示随机一致
    """
    return cohen_kappa_score(y_true, y_pred)

# ==================== 分割质量指标 ====================

def compute_soft_dice_per_class(y_true_onehot: np.ndarray, y_probs: np.ndarray,
                                smooth: float = 1e-5) -> np.ndarray:
    """
    计算每个类别的 Soft Dice 系数

    Args:
        y_true_onehot: One-hot 真实标签 (n_samples, n_classes)
        y_probs: 预测概率 (n_samples, n_classes)
        smooth: 平滑项，避免除零

    Returns:
        每个类别的 Dice 系数 (n_classes,)
    """
    n_classes = y_probs.shape[1]
    dice_scores = np.zeros(n_classes)

    for c in range(n_classes):
        # 计算交集和并集
        intersection = np.sum(y_true_onehot[:, c] * y_probs[:, c])
        union = np.sum(y_true_onehot[:, c]) + np.sum(y_probs[:, c])

        # Dice 系数
        dice_scores[c] = (2.0 * intersection + smooth) / (union + smooth)

    return dice_scores

def compute_macro_soft_dice(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Macro 3D Soft Dice 系数
    所有类别的平均 Soft Dice

    Args:
        y_true: 真实标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)

    Returns:
        Macro Soft Dice 系数 [0, 1]
    """
    n_classes = y_probs.shape[1]

    # 转换为 one-hot
    y_true_onehot = np.eye(n_classes)[y_true]

    # 计算每个类别的 Dice
    dice_per_class = compute_soft_dice_per_class(y_true_onehot, y_probs)

    # 返回宏平均（只计算出现过的类别）
    present_classes = np.unique(y_true)
    dice_present = dice_per_class[present_classes]

    return np.mean(dice_present)

def compute_3d_soft_dice(prob_volume: np.ndarray,
                         gt_labels: np.ndarray,
                         mask: np.ndarray) -> float:
    """
    计算 3D 体积的 Macro Soft Dice

    Args:
        prob_volume: 预测概率体积 (H, W, D, n_classes)
        gt_labels: 真实标签体积 (H, W, D)，值为 1-102
        mask: 有效区域掩码 (H, W, D)

    Returns:
        Macro 3D Soft Dice 系数
    """
    # 提取有效体素
    valid_mask = (mask > 0) & (gt_labels > 0)

    y_true = gt_labels[valid_mask] - 1  # 转换为 0-101
    y_probs = prob_volume[valid_mask]   # (n_voxels, n_classes)

    return compute_macro_soft_dice(y_true, y_probs)

# ==================== 风险分析 ====================

def compute_risk_coverage_curve(y_true: np.ndarray,
                                y_probs: np.ndarray,
                                n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算风险-覆盖率曲线

    风险定义：被覆盖样本中的错误率
    覆盖率定义：保留的样本比例

    通过调整置信度阈值，可以权衡风险和覆盖率

    Args:
        y_true: 真实标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)
        n_points: 曲线上的点数

    Returns:
        thresholds: 阈值数组
        risks: 对应的风险（错误率）
        coverages: 对应的覆盖率
    """
    # 获取预测结果和最大概率
    y_pred = np.argmax(y_probs, axis=1)
    max_probs = np.max(y_probs, axis=1)

    # 生成阈值序列
    thresholds = np.linspace(0, 1, n_points)

    risks = []
    coverages = []

    for threshold in thresholds:
        # 选择置信度高于阈值的样本
        mask = max_probs >= threshold

        if np.sum(mask) == 0:
            # 没有样本被覆盖
            risks.append(np.nan)
            coverages.append(0.0)
        else:
            # 计算覆盖率
            coverage = np.mean(mask)
            coverages.append(coverage)

            # 计算风险（被覆盖样本中的错误率）
            covered_true = y_true[mask]
            covered_pred = y_pred[mask]
            risk = 1.0 - accuracy_score(covered_true, covered_pred)
            risks.append(risk)

    return np.array(thresholds), np.array(risks), np.array(coverages)

def compute_risk_at_coverage(y_true: np.ndarray,
                             y_probs: np.ndarray,
                             target_coverage: float = 0.95) -> Dict[str, float]:
    """
    计算指定覆盖率下的风险

    Args:
        y_true: 真实标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)
        target_coverage: 目标覆盖率 (如 0.95 表示覆盖 95% 样本)

    Returns:
        包含 risk, coverage, threshold 的字典
    """
    thresholds, risks, coverages = compute_risk_coverage_curve(y_true, y_probs)

    # 找到最接近目标覆盖率的点
    valid_idx = ~np.isnan(risks)
    if np.sum(valid_idx) == 0:
        return {'risk': np.nan, 'coverage': 0.0, 'threshold': 1.0}

    valid_coverages = coverages[valid_idx]
    valid_risks = risks[valid_idx]
    valid_thresholds = thresholds[valid_idx]

    # 找到覆盖率 >= target_coverage 的最小风险点
    above_target = valid_coverages >= target_coverage
    if np.sum(above_target) == 0:
        # 无法达到目标覆盖率，返回最大覆盖率的情况
        idx = np.argmax(valid_coverages)
    else:
        # 在满足覆盖率的点中选择风险最小的
        idx = np.where(above_target)[0][np.argmin(valid_risks[above_target])]

    return {
        'risk': valid_risks[idx],
        'coverage': valid_coverages[idx],
        'threshold': valid_thresholds[idx]
    }

# ==================== 综合评估函数 ====================

def compute_all_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_probs: np.ndarray) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # 1. 准确性指标
    metrics['gross_accuracy'] = compute_gross_accuracy(y_true, y_pred)
    top_k = compute_top_k_accuracies(y_true, y_probs)
    metrics.update(top_k)

    # 2. 类别平衡指标
    metrics['macro_f1'] = compute_macro_f1(y_true, y_pred)
    metrics['weighted_f1'] = compute_weighted_f1(y_true, y_pred)
    metrics['balanced_accuracy'] = compute_balanced_accuracy(y_true, y_pred)

    # 3. 一致性指标
    metrics['cohen_kappa'] = compute_cohen_kappa(y_true, y_pred)

    # 4. 分割质量指标
    metrics['macro_soft_dice'] = compute_macro_soft_dice(y_true, y_probs)

    # 5. 风险分析
    risk_95 = compute_risk_at_coverage(y_true, y_probs, target_coverage=0.95)
    metrics['risk_at_95_coverage'] = risk_95['risk']
    metrics['actual_coverage_95'] = risk_95['coverage']

    return metrics

def compute_all_metrics_from_loader(model,
                                    data_loader,
                                    device: str = 'cuda') -> Dict[str, float]:
    """
    从 DataLoader 计算所有指标

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备

    Returns:
        包含所有指标的字典
    """
    model.eval()

    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)

            # Softmax 概率
            probs = torch.softmax(output, dim=1)

            # 预测标签
            pred = torch.argmax(output, dim=1)

            all_true.extend(target.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为 numpy 数组
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_probs = np.array(all_probs)

    return compute_all_metrics(y_true, y_pred, y_probs)

# ==================== 格式化输出 ====================

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    格式化输出指标

    Args:
        metrics: 指标字典
        prefix: 前缀（如 "train_" 或 "test_"）

    Returns:
        格式化的字符串
    """
    lines = []

    # 按类别组织输出
    if f'{prefix}gross_accuracy' in metrics:
        lines.append(f"=== 准确性指标 ===")
        lines.append(f"  Gross Accuracy: {metrics[f'{prefix}gross_accuracy']:.4f}")
        if f'{prefix}top1_accuracy' in metrics:
            lines.append(f"  Top-1 Accuracy: {metrics[f'{prefix}top1_accuracy']:.4f}")
            lines.append(f"  Top-3 Accuracy: {metrics[f'{prefix}top3_accuracy']:.4f}")
            lines.append(f"  Top-5 Accuracy: {metrics[f'{prefix}top5_accuracy']:.4f}")

    if f'{prefix}macro_f1' in metrics:
        lines.append(f"\n=== 类别平衡指标 ===")
        lines.append(f"  Macro-F1: {metrics[f'{prefix}macro_f1']:.4f}")
        lines.append(f"  Weighted F1: {metrics[f'{prefix}weighted_f1']:.4f}")
        lines.append(f"  Balanced Accuracy: {metrics[f'{prefix}balanced_accuracy']:.4f}")

    if f'{prefix}cohen_kappa' in metrics:
        lines.append(f"\n=== 一致性指标 ===")
        lines.append(f"  Cohen's Kappa: {metrics[f'{prefix}cohen_kappa']:.4f}")

    if f'{prefix}macro_soft_dice' in metrics:
        lines.append(f"\n=== 分割质量指标 ===")
        lines.append(f"  Macro 3D Soft Dice: {metrics[f'{prefix}macro_soft_dice']:.4f}")

    if f'{prefix}risk_at_95_coverage' in metrics:
        lines.append(f"\n=== 风险分析 ===")
        lines.append(f"  Risk at 95% Coverage: {metrics[f'{prefix}risk_at_95_coverage']:.4f}")
        lines.append(f"  Actual Coverage: {metrics[f'{prefix}actual_coverage_95']:.4f}")

    return "\n".join(lines)

if __name__ == '__main__':
    # 测试代码
    print("评估指标模块测试")

    # 模拟数据
    n_samples = 1000
    n_classes = 102

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # 归一化

    # 计算所有指标
    metrics = compute_all_metrics(y_true, y_pred, y_probs)

    # 打印结果
    print("\n" + format_metrics(metrics))
    print("\n测试完成！")
