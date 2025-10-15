#!/usr/bin/env python3
"""
单轮训练脚本 (Single-Split Training Runner)
按被试分层的36/1/1划分，支持软标签训练与3D还原

版本: v1.0
日期: 2025-01-11
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import sys

# 添加1d-3d-convert模块路径
# 当前文件: training/downsampling/train_runner.py
# 目标路径: 3D_dev/dataset_create/1d-3d-convert/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'dataset_create' / '1d-3d-convert'))

from data_3d_1d_mapper import Data3D1DMapper


# ============================================================================
# 数据集类
# ============================================================================

class BrainVoxelDataset(Dataset):
    """脑体素数据集（支持软标签）"""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 subject_ids: Optional[List[str]] = None):
        """
        Args:
            features: (n_voxels, 351) 特征数据
            labels: (n_voxels, 102) 软标签（概率分布）
            subject_ids: 每个体素对应的被试ID（可选，用于调试）
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# 模型定义
# ============================================================================

class Dense4x4096Model(nn.Module):
    """4×4096全连接网络"""

    def __init__(self, input_dim=351, num_classes=102, dropout=0.5):
        super(Dense4x4096Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # logits
        return x


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int = 42):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_subjects(all_subject_ids: List[str],
                   test_id: Optional[str] = None,
                   val_id: Optional[str] = None,
                   seed: int = 42) -> Dict[str, Any]:
    """
    被试划分：36 train + 1 val + 1 test

    Args:
        all_subject_ids: 所有被试ID列表
        test_id: 指定test被试（可选）
        val_id: 指定val被试（可选）
        seed: 随机种子

    Returns:
        划分结果字典
    """
    np.random.seed(seed)
    sorted_ids = sorted(all_subject_ids)

    # 如果未指定，默认取最后两个
    if test_id is None:
        test_id = sorted_ids[-1]
    if val_id is None:
        val_id = sorted_ids[-2]

    # 验证ID有效性
    assert test_id in sorted_ids, f"test_id {test_id} 不在被试列表中"
    assert val_id in sorted_ids, f"val_id {val_id} 不在被试列表中"
    assert test_id != val_id, "test_id和val_id不能相同"

    # 构建train列表
    train_ids = [sid for sid in sorted_ids if sid not in [test_id, val_id]]

    split_info = {
        'train_ids': train_ids,
        'val_id': val_id,
        'test_id': test_id,
        'n_train': len(train_ids),
        'n_val': 1,
        'n_test': 1,
        'seed': seed
    }

    return split_info


def load_subject_data(data_root: Path, subject_id: str) -> Dict[str, np.ndarray]:
    """
    加载单个被试的1D和3D数据

    Args:
        data_root: 数据根目录
        subject_id: 被试ID

    Returns:
        数据字典
    """
    file_1d = data_root / '1d' / f'{subject_id}_1d.npz'
    file_3d = data_root / '3d' / f'{subject_id}_3d.npz'

    if not file_1d.exists():
        raise FileNotFoundError(f"1D文件不存在: {file_1d}")
    if not file_3d.exists():
        raise FileNotFoundError(f"3D文件不存在: {file_3d}")

    data_1d = np.load(file_1d)
    data_3d = np.load(file_3d)

    return {
        'multidim_data': data_1d['multidim_data'],       # (n_vox, 351)
        'seg_one_hot': data_1d['seg_one_hot'],           # (102, n_vox)
        'region_seg': data_1d['region_seg'],             # (n_vox,)
        'region': data_1d['region'],                     # (Z', X', Y')
        'n_voxels': int(data_1d['n_voxels']),
        'proba_labels': data_3d['proba_labels'],         # (Z', X', Y', 102)
        'region_mask_lr': data_3d['region_mask_lr']      # (Z', X', Y')
    }


def zscore_per_subject(features: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    逐被试z-score标准化（351维特征）

    Args:
        features: (n_vox, 351) 原始特征

    Returns:
        normalized_features: (n_vox, 351) 标准化后的特征
        stats: {'mean': (351,), 'std': (351,)}
    """
    mean = features.mean(axis=0)  # (351,)
    std = features.std(axis=0)    # (351,)
    std = np.maximum(std, 1e-8)   # 防止除零

    normalized = (features - mean) / std

    stats = {
        'mean': mean,
        'std': std
    }

    return normalized, stats


def compute_class_weights(train_labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    计算类别权重（用于处理类别不平衡）

    Args:
        train_labels: (n_vox, 102) 训练集软标签
        alpha: 平衡系数（默认0.5）

    Returns:
        class_weights: (102,) 类别权重
    """
    # 计算每个类别的总概率（软标签加权）
    class_freq = train_labels.sum(axis=0)  # (102,)
    class_freq = np.maximum(class_freq, 1e-8)

    # 计算权重: w_c = (1 / f_c)^alpha
    weights = np.power(1.0 / class_freq, alpha)

    # 归一化
    weights = weights / weights.sum() * len(weights)

    return weights.astype(np.float32)


def kernel_l2_regularization(model: nn.Module, weight_decay: float = 1e-5) -> torch.Tensor:
    """
    L2正则化（仅对权重矩阵，不包括偏置）
    模拟TensorFlow的kernel_regularizer行为

    Args:
        model: 模型
        weight_decay: 权重衰减系数

    Returns:
        l2_reg: L2正则化项
    """
    l2_reg = torch.tensor(0., dtype=torch.float32, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        # 只对权重矩阵应用L2正则化，跳过偏置项
        if 'weight' in name and param.requires_grad:
            l2_reg += torch.norm(param, p=2) ** 2
    return weight_decay * l2_reg


def soft_cross_entropy_loss(logits: torch.Tensor,
                            soft_targets: torch.Tensor,
                            class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    软标签交叉熵损失

    Args:
        logits: (batch, 102) 模型输出
        soft_targets: (batch, 102) 软标签（概率分布）
        class_weights: (102,) 类别权重（可选）

    Returns:
        loss: 标量
    """
    log_probs = F.log_softmax(logits, dim=1)  # (batch, 102)

    if class_weights is not None:
        # 样本权重 = sum(p_i * w_i)
        sample_weights = (soft_targets * class_weights.unsqueeze(0)).sum(dim=1)  # (batch,)
        loss = -(soft_targets * log_probs).sum(dim=1) * sample_weights  # (batch,)
        loss = loss.mean()
    else:
        loss = -(soft_targets * log_probs).sum(dim=1).mean()

    return loss


# ============================================================================
# 评估指标
# ============================================================================

def compute_soft_ece(pred_probs: np.ndarray,
                     true_probs: np.ndarray,
                     n_bins: int = 15) -> Dict[str, Any]:
    """
    计算Soft Expected Calibration Error (总体, max-prob口径)

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签
        n_bins: bin数量

    Returns:
        dict: ECE, bin统计, 可靠性曲线数据
    """
    # 最大概率和预测类别
    q_max = pred_probs.max(axis=1)  # (n_vox,)
    pred_labels = pred_probs.argmax(axis=1)  # (n_vox,)

    # 真实软标签在预测类上的质量（"对的概率"）
    y_soft = true_probs[np.arange(len(pred_labels)), pred_labels]  # (n_vox,)

    # 分bin
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (q_max > bin_lower) & (q_max <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_confidence = q_max[in_bin].mean()
            bin_accuracy = y_soft[in_bin].mean()
            bin_diff = abs(bin_confidence - bin_accuracy)

            ece += (bin_size / len(q_max)) * bin_diff

            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'bin_confidence': float(bin_confidence),
                'bin_accuracy': float(bin_accuracy),
                'bin_size': int(bin_size)
            })

    return {
        'ece': float(ece),
        'bin_stats': bin_stats
    }


def compute_classwise_ece(pred_probs: np.ndarray,
                          true_probs: np.ndarray,
                          n_bins: int = 15) -> Dict[str, float]:
    """
    计算Classwise ECE (SCE - Static Calibration Error)
    对每个类k单独计算one-vs-rest的ECE

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签
        n_bins: bin数量

    Returns:
        dict: 平均SCE和每类的ECE
    """
    n_classes = pred_probs.shape[1]
    class_eces = []

    for k in range(n_classes):
        q_k = pred_probs[:, k]  # 类k的预测概率
        p_k = true_probs[:, k]  # 类k的真实软标签

        # 分bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece_k = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (q_k > bin_lower) & (q_k <= bin_upper)
            bin_size = in_bin.sum()

            if bin_size > 0:
                bin_confidence = q_k[in_bin].mean()
                bin_accuracy = p_k[in_bin].mean()
                ece_k += (bin_size / len(q_k)) * abs(bin_confidence - bin_accuracy)

        class_eces.append(ece_k)

    return {
        'classwise_ece_mean': float(np.mean(class_eces)),
        'classwise_ece_std': float(np.std(class_eces)),
        'per_class_ece': [float(x) for x in class_eces]
    }


def compute_brier_decomposition(pred_probs: np.ndarray,
                                 true_probs: np.ndarray) -> Dict[str, float]:
    """
    Brier分解 (Murphy decomposition): Brier = Reliability - Resolution + Uncertainty
    多类别版本：对每个类的二元问题分解后取平均

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签

    Returns:
        dict: reliability, resolution, uncertainty, brier_score
    """
    n_classes = pred_probs.shape[1]

    reliabilities = []
    resolutions = []
    uncertainties = []

    for k in range(n_classes):
        q_k = pred_probs[:, k]
        p_k = true_probs[:, k]

        # Uncertainty: 真实标签的方差
        p_mean = p_k.mean()
        uncertainty = p_mean * (1 - p_mean)

        # 使用10个bin进行分解
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            in_bin = (q_k > bin_boundaries[i]) & (q_k <= bin_boundaries[i+1])
            bin_size = in_bin.sum()

            if bin_size > 0:
                q_bin_mean = q_k[in_bin].mean()
                p_bin_mean = p_k[in_bin].mean()

                # Reliability: 预测与真实的偏差
                reliability += (bin_size / len(q_k)) * (q_bin_mean - p_bin_mean) ** 2

                # Resolution: bin内真实标签与总体均值的差异
                resolution += (bin_size / len(q_k)) * (p_bin_mean - p_mean) ** 2

        reliabilities.append(reliability)
        resolutions.append(resolution)
        uncertainties.append(uncertainty)

    return {
        'brier_reliability': float(np.mean(reliabilities)),
        'brier_resolution': float(np.mean(resolutions)),
        'brier_uncertainty': float(np.mean(uncertainties)),
        'brier_decomposed': float(np.mean(reliabilities) - np.mean(resolutions) + np.mean(uncertainties))
    }


def compute_class_mass_error(pred_probs: np.ndarray,
                             true_probs: np.ndarray,
                             top_k: int = 10) -> Dict[str, Any]:
    """
    体积一致性误差 (Class-mass error)
    评估每类概率总和的偏差

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签
        top_k: 返回偏差最大的前k个类

    Returns:
        dict: L1相对误差, 绝对误差, 前k个偏差最大的类
    """
    # 每类的总概率质量
    T_k = true_probs.sum(axis=0)  # (102,)
    P_k = pred_probs.sum(axis=0)  # (102,)

    # 绝对误差
    abs_errors = np.abs(P_k - T_k)

    # L1相对误差
    l1_rel_error = abs_errors.sum() / T_k.sum()

    # 找到偏差最大的类
    top_error_indices = np.argsort(abs_errors)[::-1][:top_k]
    top_errors = []

    for idx in top_error_indices:
        top_errors.append({
            'class_id': int(idx),
            'true_mass': float(T_k[idx]),
            'pred_mass': float(P_k[idx]),
            'abs_error': float(abs_errors[idx]),
            'rel_error': float(abs_errors[idx] / T_k[idx]) if T_k[idx] > 0 else 0.0
        })

    return {
        'class_mass_l1_rel_error': float(l1_rel_error),
        'class_mass_l1_abs_error': float(abs_errors.sum()),
        'top_error_classes': top_errors
    }


def compute_soft_confusion_matrix(pred_probs: np.ndarray,
                                   true_probs: np.ndarray) -> np.ndarray:
    """
    软混淆矩阵: M[i,j] = sum_voxel (true_prob[i] * pred_prob[j])
    表示"类i的概率质量有多少被预测为类j"

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签

    Returns:
        soft_cm: (102, 102) 行归一化的软混淆矩阵
    """
    # M[i,j] = Σ_voxel true[voxel, i] * pred[voxel, j]
    soft_cm = true_probs.T @ pred_probs  # (102, 102)

    # 行归一化
    row_sums = soft_cm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)  # 防止除零
    soft_cm_normalized = soft_cm / row_sums

    return soft_cm_normalized


def compute_aurc(pred_probs: np.ndarray,
                 true_probs: np.ndarray,
                 n_thresholds: int = 100) -> Dict[str, Any]:
    """
    Area Under Risk-Coverage curve (AURC)
    选择性预测评估：在不同置信阈值下的准确度-覆盖率权衡

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签
        n_thresholds: 阈值数量

    Returns:
        dict: AURC, 曲线数据
    """
    # 最大概率和预测类别
    q_max = pred_probs.max(axis=1)
    pred_labels = pred_probs.argmax(axis=1)

    # 真实软标签在预测类上的质量（"准确度"）
    y_soft = true_probs[np.arange(len(pred_labels)), pred_labels]

    # 按置信度排序
    sorted_indices = np.argsort(q_max)[::-1]  # 从高到低
    q_max_sorted = q_max[sorted_indices]
    y_soft_sorted = y_soft[sorted_indices]

    # 计算不同覆盖率下的准确度
    thresholds = np.linspace(0, 1, n_thresholds)
    coverages = []
    accuracies = []
    risks = []

    for threshold in thresholds:
        covered = q_max >= threshold
        coverage = covered.sum() / len(q_max)

        if covered.sum() > 0:
            accuracy = y_soft[covered].mean()
            risk = 1 - accuracy
        else:
            accuracy = 0.0
            risk = 1.0

        coverages.append(float(coverage))
        accuracies.append(float(accuracy))
        risks.append(float(risk))

    # 计算AURC (使用梯形法则)
    aurc = np.trapz(risks, coverages)

    return {
        'aurc': float(aurc),
        'thresholds': [float(t) for t in thresholds],
        'coverages': coverages,
        'accuracies': accuracies,
        'risks': risks
    }


def compute_entropy_stats(pred_probs: np.ndarray) -> Dict[str, float]:
    """
    计算预测熵的统计量

    Args:
        pred_probs: (n_vox, 102) 预测概率

    Returns:
        dict: 熵的均值、中位数、分位数等
    """
    eps = 1e-8
    entropy = -(pred_probs * np.log(pred_probs + eps)).sum(axis=1)  # (n_vox,)

    return {
        'entropy_mean': float(entropy.mean()),
        'entropy_median': float(np.median(entropy)),
        'entropy_std': float(entropy.std()),
        'entropy_q25': float(np.percentile(entropy, 25)),
        'entropy_q75': float(np.percentile(entropy, 75)),
        'entropy_q95': float(np.percentile(entropy, 95)),
        'entropy_max': float(entropy.max())
    }


def compute_metrics(pred_probs: np.ndarray,
                    true_probs: np.ndarray,
                    pred_hard: Optional[np.ndarray] = None,
                    true_hard: Optional[np.ndarray] = None,
                    compute_advanced: bool = True) -> Dict[str, Any]:
    """
    计算评估指标（包含高级软标签指标）

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实概率（软标签）
        pred_hard: (n_vox,) 预测硬标签（可选，否则自动argmax）
        true_hard: (n_vox,) 真实硬标签（可选，否则自动argmax）
        compute_advanced: 是否计算高级指标（ECE, Brier分解等）

    Returns:
        metrics: 指标字典
    """
    if pred_hard is None:
        pred_hard = pred_probs.argmax(axis=1)
    if true_hard is None:
        true_hard = true_probs.argmax(axis=1)

    # ========== 基础指标 ==========
    # Gross accuracy
    gross_acc = accuracy_score(true_hard, pred_hard)

    # NLL (Negative Log-Likelihood)
    eps = 1e-8
    nll = -np.log(np.clip(pred_probs, eps, 1.0))
    nll = (nll * true_probs).sum(axis=1).mean()

    # Brier score
    brier = np.mean((pred_probs - true_probs) ** 2)

    # Macro/Micro F1
    macro_f1 = f1_score(true_hard, pred_hard, average='macro', zero_division=0)
    micro_f1 = f1_score(true_hard, pred_hard, average='micro', zero_division=0)

    # Top-k accuracy
    top3_acc = compute_topk_accuracy(pred_probs, true_hard, k=3)
    top5_acc = compute_topk_accuracy(pred_probs, true_hard, k=5)

    metrics = {
        'gross_accuracy': float(gross_acc),
        'nll': float(nll),
        'brier_score': float(brier),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc)
    }

    # ========== 高级软标签指标 ==========
    if compute_advanced:
        # Soft ECE (总体)
        ece_results = compute_soft_ece(pred_probs, true_probs)
        metrics['soft_ece'] = ece_results['ece']
        metrics['soft_ece_bins'] = ece_results['bin_stats']

        # Classwise ECE
        classwise_ece = compute_classwise_ece(pred_probs, true_probs)
        metrics['classwise_ece_mean'] = classwise_ece['classwise_ece_mean']
        metrics['classwise_ece_std'] = classwise_ece['classwise_ece_std']
        # per_class_ece 太大，不存在主metrics中

        # Brier 分解
        brier_decomp = compute_brier_decomposition(pred_probs, true_probs)
        metrics.update(brier_decomp)

        # 体积一致性误差
        class_mass = compute_class_mass_error(pred_probs, true_probs)
        metrics['class_mass_l1_rel_error'] = class_mass['class_mass_l1_rel_error']
        metrics['class_mass_l1_abs_error'] = class_mass['class_mass_l1_abs_error']
        metrics['class_mass_top_errors'] = class_mass['top_error_classes']

        # AURC
        aurc_results = compute_aurc(pred_probs, true_probs)
        metrics['aurc'] = aurc_results['aurc']
        metrics['aurc_curve'] = {
            'thresholds': aurc_results['thresholds'],
            'coverages': aurc_results['coverages'],
            'accuracies': aurc_results['accuracies'],
            'risks': aurc_results['risks']
        }

        # 熵统计
        entropy_stats = compute_entropy_stats(pred_probs)
        metrics.update(entropy_stats)

    return metrics


def compute_topk_accuracy(pred_probs: np.ndarray, true_labels: np.ndarray, k: int = 3) -> float:
    """计算Top-k准确率"""
    topk_preds = np.argsort(pred_probs, axis=1)[:, -k:]  # (n_vox, k)
    correct = np.any(topk_preds == true_labels[:, None], axis=1)
    return correct.mean()


def compute_confusion_matrix(pred_labels: np.ndarray,
                             true_labels: np.ndarray,
                             normalize: str = 'true') -> np.ndarray:
    """
    计算混淆矩阵

    Args:
        pred_labels: (n_vox,) 预测标签
        true_labels: (n_vox,) 真实标签
        normalize: 归一化方式 ('true', 'pred', 'all', None)

    Returns:
        cm: (102, 102) 混淆矩阵
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(102), normalize=normalize)
    return cm


# ============================================================================
# 3D还原
# ============================================================================

def restore_predictions_to_3d(pred_probs_1d: np.ndarray,
                              region_mask: np.ndarray,
                              mapper: Data3D1DMapper) -> Tuple[np.ndarray, np.ndarray]:
    """
    将1D预测还原到3D空间

    Args:
        pred_probs_1d: (n_vox, 102) 1D预测概率
        region_mask: (Z', X', Y') ROI掩码
        mapper: 3D-1D映射器

    Returns:
        pred_probs_3d: (Z', X', Y', 102) 3D预测概率
        pred_argmax_3d: (Z', X', Y') 3D预测硬标签
    """
    # 使用mapper还原到3D
    pred_probs_3d = mapper.map_1d_predictions_to_3d(
        pred_probs_1d,
        region_mask=region_mask
    )

    # 生成硬标签
    pred_argmax_3d = pred_probs_3d.argmax(axis=-1).astype(np.uint8)

    return pred_probs_3d, pred_argmax_3d


# ============================================================================
# 3D高级指标
# ============================================================================

def compute_3d_soft_dice(pred_probs_3d: np.ndarray,
                         true_probs_3d: np.ndarray,
                         region_mask: np.ndarray) -> Dict[str, Any]:
    """
    计算3D Soft Dice系数（每类）

    Args:
        pred_probs_3d: (Z, X, Y, 102) 预测概率
        true_probs_3d: (Z, X, Y, 102) 真实软标签
        region_mask: (Z, X, Y) ROI掩码

    Returns:
        dict: 宏/微平均Dice, 每类Dice
    """
    mask = region_mask > 0
    n_classes = pred_probs_3d.shape[-1]

    dice_scores = []

    for k in range(n_classes):
        P_k = pred_probs_3d[..., k][mask]  # (n_vox,)
        T_k = true_probs_3d[..., k][mask]  # (n_vox,)

        # Soft Dice: 2 * sum(P·T) / (sum(P²) + sum(T²))
        numerator = 2 * (P_k * T_k).sum()
        denominator = (P_k ** 2).sum() + (T_k ** 2).sum()

        if denominator > 0:
            dice_k = numerator / denominator
        else:
            dice_k = 0.0

        dice_scores.append(float(dice_k))

    return {
        'soft_dice_macro': float(np.mean(dice_scores)),
        'soft_dice_micro': float(np.mean(dice_scores)),  # 对于Dice, macro=micro
        'per_class_soft_dice': dice_scores
    }


def compute_3d_prob_metrics(pred_probs_3d: np.ndarray,
                            true_probs_3d: np.ndarray,
                            region_mask: np.ndarray) -> Dict[str, float]:
    """
    计算3D概率指标（仅ROI内）

    Args:
        pred_probs_3d: (Z, X, Y, 102) 预测概率
        true_probs_3d: (Z, X, Y, 102) 真实软标签
        region_mask: (Z, X, Y) ROI掩码

    Returns:
        dict: 3D NLL, 3D Brier
    """
    mask = region_mask > 0

    # 转为1D
    pred_1d = pred_probs_3d[mask]  # (n_vox, 102)
    true_1d = true_probs_3d[mask]  # (n_vox, 102)

    # 3D NLL
    eps = 1e-8
    nll_3d = -np.log(np.clip(pred_1d, eps, 1.0))
    nll_3d = (nll_3d * true_1d).sum(axis=1).mean()

    # 3D Brier
    brier_3d = np.mean((pred_1d - true_1d) ** 2)

    return {
        '3d_nll': float(nll_3d),
        '3d_brier': float(brier_3d)
    }


# ============================================================================
# 可视化
# ============================================================================

def save_confusion_matrix(cm: np.ndarray, save_path: Path):
    """保存混淆矩阵为CSV"""
    np.savetxt(save_path, cm, delimiter=',', fmt='%.6f')


def plot_reliability_diagram(bin_stats: List[Dict], save_path: Path):
    """
    绘制可靠性曲线 (Reliability Diagram)

    Args:
        bin_stats: Soft ECE的bin统计
        save_path: 保存路径
    """
    confidences = [b['bin_confidence'] for b in bin_stats]
    accuracies = [b['bin_accuracy'] for b in bin_stats]
    sizes = [b['bin_size'] for b in bin_stats]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制对角线 (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # 绘制实际的可靠性曲线
    ax.scatter(confidences, accuracies, s=[s/10 for s in sizes], alpha=0.7, label='Model Calibration')
    ax.plot(confidences, accuracies, 'b-', alpha=0.5)

    ax.set_xlabel('Confidence (Max Probability)', fontsize=12)
    ax.set_ylabel('Accuracy (Soft Label Quality)', fontsize=12)
    ax.set_title('Reliability Diagram (Soft ECE)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_risk_coverage_curve(aurc_data: Dict, save_path: Path):
    """
    绘制Risk-Coverage曲线

    Args:
        aurc_data: AURC结果字典
        save_path: 保存路径
    """
    coverages = aurc_data['coverages']
    risks = aurc_data['risks']
    accuracies = aurc_data['accuracies']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Risk-Coverage curve
    ax1.plot(coverages, risks, 'b-', linewidth=2)
    ax1.fill_between(coverages, risks, alpha=0.3)
    ax1.set_xlabel('Coverage', fontsize=12)
    ax1.set_ylabel('Risk (1 - Accuracy)', fontsize=12)
    ax1.set_title(f'Risk-Coverage Curve (AURC={aurc_data["aurc"]:.4f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Accuracy-Coverage curve
    ax2.plot(coverages, accuracies, 'g-', linewidth=2)
    ax2.fill_between(coverages, accuracies, alpha=0.3)
    ax2.set_xlabel('Coverage', fontsize=12)
    ax2.set_ylabel('Accuracy (Soft Label Quality)', fontsize=12)
    ax2.set_title('Accuracy-Coverage Curve', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_entropy_histogram(pred_probs: np.ndarray, save_path: Path):
    """
    绘制预测熵的直方图

    Args:
        pred_probs: (n_vox, 102) 预测概率
        save_path: 保存路径
    """
    eps = 1e-8
    entropy = -(pred_probs * np.log(pred_probs + eps)).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(entropy, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(entropy.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {entropy.mean():.3f}')
    ax.axvline(np.median(entropy), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(entropy):.3f}')

    ax.set_xlabel('Entropy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Entropy Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_slices(pred_3d: np.ndarray,
                   true_3d: np.ndarray,
                   region_mask: np.ndarray,
                   save_dir: Path,
                   subject_id: str,
                   slice_indices: Optional[List[int]] = None):
    """
    绘制3D切片对比图

    Args:
        pred_3d: (Z', X', Y') 预测标签
        true_3d: (Z', X', Y') 真实标签
        region_mask: (Z', X', Y') ROI掩码
        save_dir: 保存目录
        subject_id: 被试ID
        slice_indices: 切片索引（默认中间2个）
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    Z, X, Y = pred_3d.shape

    if slice_indices is None:
        # 默认选择中间2个切片
        slice_indices = [Z // 3, 2 * Z // 3]

    for z_idx in slice_indices:
        if z_idx >= Z:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 预测
        axes[0].imshow(pred_3d[z_idx], cmap='tab20', vmin=0, vmax=101)
        axes[0].set_title(f'Prediction (z={z_idx})')
        axes[0].axis('off')

        # 真实
        axes[1].imshow(true_3d[z_idx], cmap='tab20', vmin=0, vmax=101)
        axes[1].set_title(f'Ground Truth (z={z_idx})')
        axes[1].axis('off')

        # 差异
        diff = (pred_3d[z_idx] != true_3d[z_idx]) & (region_mask[z_idx] > 0)
        axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Difference (z={z_idx})')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = save_dir / f'{subject_id}_slice_z{z_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# 温度缩放 (Temperature Scaling)
# ============================================================================

class TemperatureScaler:
    """温度缩放校准器"""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, true_probs: np.ndarray,
            max_iter: int = 100, lr: float = 0.01) -> float:
        """
        在验证集上拟合最优温度

        Args:
            logits: (n_vox, 102) 模型logits
            true_probs: (n_vox, 102) 真实软标签
            max_iter: 最大迭代次数
            lr: 学习率

        Returns:
            optimal_temperature: 最优温度值
        """
        import torch
        import torch.nn.functional as F_torch
        from torch.optim import LBFGS

        logits_torch = torch.FloatTensor(logits).requires_grad_(False)
        true_probs_torch = torch.FloatTensor(true_probs).requires_grad_(False)

        # 初始化温度参数
        temperature = torch.nn.Parameter(torch.ones(1))

        # 使用LBFGS优化
        optimizer = LBFGS([temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            # 应用温度缩放
            scaled_logits = logits_torch / temperature
            # 计算软标签NLL损失
            log_probs = F_torch.log_softmax(scaled_logits, dim=1)
            loss = -(true_probs_torch * log_probs).sum(dim=1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)

        # 保存最优温度
        self.temperature = float(temperature.item())

        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        应用温度缩放

        Args:
            logits: (n_vox, 102) 模型logits

        Returns:
            calibrated_probs: (n_vox, 102) 校准后的概率
        """
        import torch
        import torch.nn.functional as F_torch

        logits_torch = torch.FloatTensor(logits)
        scaled_logits = logits_torch / self.temperature
        calibrated_probs = F_torch.softmax(scaled_logits, dim=1).numpy()

        return calibrated_probs


# ============================================================================
# 主训练流程
# ============================================================================

def run_single_split(
    data_root: str,
    val_id: Optional[str] = None,
    test_id: Optional[str] = None,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    use_class_weights: bool = False,
    class_weight_alpha: float = 0.5,
    max_vox_per_subject: Optional[int] = None,
    grad_clip_norm: Optional[float] = 1.0,
    save_dir: str = "runs/quickstart",
    fold_name: Optional[str] = None
):
    """
    单轮训练主函数

    Args:
        data_root: 数据根目录（包含1d/和3d/子目录）
        val_id: 验证集被试ID（可选）
        test_id: 测试集被试ID（可选）
        seed: 随机种子
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        weight_decay: 权重衰减
        use_class_weights: 是否使用类别权重
        class_weight_alpha: 类别权重平衡系数
        max_vox_per_subject: 每个被试最大体素数（用于控制显存）
        grad_clip_norm: 梯度裁剪范数（默认1.0，设为None禁用）
        save_dir: 保存目录（如果指定fold_name，将创建save_dir/fold_name子目录）
        fold_name: Fold名称（可选，例如"fold_1"或"test_sub100307"），用于多fold训练时组织结果
    """
    # 设置随机种子
    set_seed(seed)

    # 路径设置
    data_root = Path(data_root)
    base_save_dir = Path(save_dir)

    # 如果指定了fold_name，创建子目录
    if fold_name is not None:
        save_dir = base_save_dir / fold_name
        logger_prefix = f"[{fold_name}] "
    else:
        save_dir = base_save_dir
        logger_prefix = ""

    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = save_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    metrics_dir = save_dir
    pred_dir = save_dir / 'pred_3d'
    pred_dir.mkdir(exist_ok=True)

    figs_dir = save_dir / 'figs'
    figs_dir.mkdir(exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("开始单轮训练 (36/1/1 Split)")
    logger.info("=" * 80)

    # ========================================================================
    # 步骤1: 枚举被试并划分
    # ========================================================================
    logger.info("\n步骤1: 枚举被试并划分...")

    dir_1d = data_root / '1d'
    if not dir_1d.exists():
        raise FileNotFoundError(f"1D数据目录不存在: {dir_1d}")

    # 枚举所有被试
    all_npz_files = sorted(dir_1d.glob('*_1d.npz'))
    all_subject_ids = [f.stem.replace('_1d', '') for f in all_npz_files]

    logger.info(f"找到 {len(all_subject_ids)} 个被试")

    if len(all_subject_ids) < 3:
        raise ValueError(f"被试数不足3个，无法划分36/1/1")

    # 划分被试
    split_info = split_subjects(all_subject_ids, test_id=test_id, val_id=val_id, seed=seed)

    logger.info(f"训练集: {split_info['n_train']} 个被试")
    logger.info(f"验证集: {split_info['val_id']}")
    logger.info(f"测试集: {split_info['test_id']}")

    # 保存划分信息
    split_summary_path = save_dir / 'split_summary.json'
    with open(split_summary_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"划分信息已保存: {split_summary_path}")

    # ========================================================================
    # 步骤2: 加载数据并逐被试z-score
    # ========================================================================
    logger.info("\n步骤2: 加载数据并逐被试z-score标准化...")

    train_features_list = []
    train_labels_list = []

    val_features = None
    val_labels = None
    val_region_mask = None

    test_features = None
    test_labels = None
    test_region_mask = None

    norm_stats = {}

    # 加载训练集
    logger.info("加载训练集...")
    for subject_id in split_info['train_ids']:
        data = load_subject_data(data_root, subject_id)

        features = data['multidim_data']  # (n_vox, 351)
        labels = data['seg_one_hot'].T    # (102, n_vox) -> (n_vox, 102)

        # 逐被试z-score
        features_norm, stats = zscore_per_subject(features)
        norm_stats[subject_id] = {
            'mean': stats['mean'].tolist(),
            'std': stats['std'].tolist()
        }

        # 可选：限制体素数
        if max_vox_per_subject is not None and len(features_norm) > max_vox_per_subject:
            indices = np.random.choice(len(features_norm), max_vox_per_subject, replace=False)
            features_norm = features_norm[indices]
            labels = labels[indices]

        train_features_list.append(features_norm)
        train_labels_list.append(labels)

        logger.info(f"  {subject_id}: {features_norm.shape[0]} 体素")

    # 合并训练集
    train_features = np.concatenate(train_features_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    logger.info(f"训练集总体素数: {train_features.shape[0]}")

    # 加载验证集
    logger.info("加载验证集...")
    val_data = load_subject_data(data_root, split_info['val_id'])
    val_features, val_stats = zscore_per_subject(val_data['multidim_data'])
    val_labels = val_data['seg_one_hot'].T
    val_region_mask = val_data['region']
    norm_stats[split_info['val_id']] = {
        'mean': val_stats['mean'].tolist(),
        'std': val_stats['std'].tolist()
    }
    logger.info(f"  {split_info['val_id']}: {val_features.shape[0]} 体素")

    # 加载测试集
    logger.info("加载测试集...")
    test_data = load_subject_data(data_root, split_info['test_id'])
    test_features, test_stats = zscore_per_subject(test_data['multidim_data'])
    test_labels = test_data['seg_one_hot'].T
    test_region_mask = test_data['region']
    norm_stats[split_info['test_id']] = {
        'mean': test_stats['mean'].tolist(),
        'std': test_stats['std'].tolist()
    }
    logger.info(f"  {split_info['test_id']}: {test_features.shape[0]} 体素")

    # 保存归一化统计量
    norm_stats_path = save_dir / 'norm_stats.json'
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"归一化统计量已保存: {norm_stats_path}")

    # ========================================================================
    # 步骤3: 构建DataLoader
    # ========================================================================
    logger.info("\n步骤3: 构建DataLoader...")

    train_dataset = BrainVoxelDataset(train_features, train_labels)
    val_dataset = BrainVoxelDataset(val_features, val_labels)
    test_dataset = BrainVoxelDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    logger.info(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    logger.info(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")

    # 计算类别权重（可选）
    class_weights = None
    if use_class_weights:
        logger.info("计算类别权重...")
        class_weights = compute_class_weights(train_labels, alpha=class_weight_alpha)
        logger.info(f"类别权重范围: [{class_weights.min():.4f}, {class_weights.max():.4f}]")

        # 保存类别权重
        weights_path = save_dir / 'class_weights.npy'
        np.save(weights_path, class_weights)
        logger.info(f"类别权重已保存: {weights_path}")

    # ========================================================================
    # 步骤4: 构建模型并训练
    # ========================================================================
    logger.info("\n步骤4: 构建模型并训练...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    model = Dense4x4096Model(input_dim=351, num_classes=102, dropout=0.5).to(device)
    # 注意：不使用optimizer的weight_decay参数，而是手动添加kernel-only L2正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        class_weights_tensor = None

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_nll': [],
        'val_gross_acc': []
    }

    best_val_nll = float('inf')
    best_epoch = 0

    # 训练循环
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info("-" * 80)

        # 训练阶段
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (features_batch, labels_batch) in enumerate(train_loader):
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            logits = model(features_batch)
            base_loss = soft_cross_entropy_loss(logits, labels_batch, class_weights_tensor)

            # 添加kernel-only L2正则化（只对权重矩阵，不包括偏置）
            l2_reg = kernel_l2_regularization(model, weight_decay=weight_decay)
            loss = base_loss + l2_reg

            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            train_loss_accum += loss.item()

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss_accum / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss_accum = 0.0
        val_probs_list = []
        val_true_list = []

        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)

                logits = model(features_batch)
                base_loss = soft_cross_entropy_loss(logits, labels_batch, class_weights_tensor)

                # 添加kernel-only L2正则化
                l2_reg = kernel_l2_regularization(model, weight_decay=weight_decay)
                loss = base_loss + l2_reg

                val_loss_accum += loss.item()

                probs = F.softmax(logits, dim=1)
                val_probs_list.append(probs.cpu().numpy())
                val_true_list.append(labels_batch.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # 计算验证指标
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_true = np.concatenate(val_true_list, axis=0)
        val_metrics = compute_metrics(val_probs, val_true)

        history['val_nll'].append(val_metrics['nll'])
        history['val_gross_acc'].append(val_metrics['gross_accuracy'])

        logger.info(f"Epoch {epoch+1} 结果:")
        logger.info(f"  训练损失: {avg_train_loss:.4f}")
        logger.info(f"  验证损失: {avg_val_loss:.4f}")
        logger.info(f"  验证NLL: {val_metrics['nll']:.4f}")
        logger.info(f"  验证Gross Acc: {val_metrics['gross_accuracy']:.4f}")
        logger.info(f"  验证Macro-F1: {val_metrics['macro_f1']:.4f}")

        # 保存最优模型
        if val_metrics['nll'] < best_val_nll:
            best_val_nll = val_metrics['nll']
            best_epoch = epoch + 1
            best_model_path = checkpoint_dir / 'best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nll': best_val_nll,
                'val_gross_acc': val_metrics['gross_accuracy']
            }, best_model_path)
            logger.info(f"  → 保存最优模型 (val NLL: {best_val_nll:.4f})")

    logger.info(f"\n训练完成！最优模型: Epoch {best_epoch}, Val NLL: {best_val_nll:.4f}")

    # ========================================================================
    # 步骤5: 加载最优模型并在Val/Test上评估（含高级指标）
    # ========================================================================
    logger.info("\n步骤5: 加载最优模型并评估...")

    # 加载最优模型
    checkpoint = torch.load(checkpoint_dir / 'best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 初始化mapper
    mapper = Data3D1DMapper(log_level='WARNING')

    # 评估验证集（收集logits和probs）
    logger.info("\n评估验证集...")
    val_logits_list = []
    val_probs_list = []
    with torch.no_grad():
        for features_batch, _ in val_loader:
            features_batch = features_batch.to(device)
            logits = model(features_batch)
            probs = F.softmax(logits, dim=1)
            val_logits_list.append(logits.cpu().numpy())
            val_probs_list.append(probs.cpu().numpy())

    val_logits = np.concatenate(val_logits_list, axis=0)  # (n_vox, 102)
    val_probs = np.concatenate(val_probs_list, axis=0)  # (n_vox, 102)
    val_metrics = compute_metrics(val_probs, val_labels, compute_advanced=True)

    logger.info("验证集指标:")
    for key, value in val_metrics.items():
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                logger.info(f"  {key}: <empty ndarray>")
            elif value.size == 1:
                logger.info(f"  {key}: {float(value):.4f}")
            else:
                logger.info(f"  {key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}, shape={value.shape}")
        # Handle scalar values
        elif isinstance(value, (int, float, np.integer, np.floating)):
            logger.info(f"  {key}: {value:.4f}")
        # Handle lists
        elif isinstance(value, list):
            if len(value) == 0:
                logger.info(f"  {key}: <empty list>")
            elif isinstance(value[0], (int, float, np.integer, np.floating)):
                # For numeric lists, show statistics
                logger.info(f"  {key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}, min={np.min(value):.4f}, max={np.max(value):.4f}")
            else:
                # For non-numeric lists
                logger.info(f"  {key}: <list with {len(value)} items>")
        # Handle dicts
        elif isinstance(value, dict):
            if len(value) == 0:
                logger.info(f"  {key}: <empty dict>")
            else:
                # Skip non-empty nested dicts (like aurc_curve) in console logging
                continue
        else:
            # For other types, show the type info
            logger.info(f"  {key}: <{type(value).__name__}>")

    # 保存验证集指标
    val_metrics_path = metrics_dir / 'metrics_val.json'
    with open(val_metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    logger.info(f"验证集指标已保存: {val_metrics_path}")

    # 验证集混淆矩阵
    val_pred_hard = val_probs.argmax(axis=1)
    val_true_hard = val_labels.argmax(axis=1)
    val_cm = compute_confusion_matrix(val_pred_hard, val_true_hard, normalize='true')
    val_cm_path = metrics_dir / 'confusion_val.csv'
    save_confusion_matrix(val_cm, val_cm_path)
    logger.info(f"验证集混淆矩阵已保存: {val_cm_path}")

    # 保存验证集软混淆矩阵
    logger.info("计算并保存软混淆矩阵...")
    val_soft_cm = compute_soft_confusion_matrix(val_probs, val_labels)
    val_soft_cm_path = metrics_dir / 'soft_confusion_val.csv'
    save_confusion_matrix(val_soft_cm, val_soft_cm_path)
    logger.info(f"验证集软混淆矩阵已保存: {val_soft_cm_path}")

    # 生成验证集可视化图表
    logger.info("生成高级指标可视化...")

    # 可靠性曲线
    if 'soft_ece_bins' in val_metrics and len(val_metrics['soft_ece_bins']) > 0:
        reliability_path = figs_dir / 'val_reliability_diagram.png'
        plot_reliability_diagram(val_metrics['soft_ece_bins'], reliability_path)
        logger.info(f"  可靠性曲线已保存: {reliability_path}")

    # Risk-Coverage曲线
    if 'aurc_curve' in val_metrics:
        risk_coverage_path = figs_dir / 'val_risk_coverage.png'
        # Add the aurc value to the curve data for plotting
        aurc_data_with_value = {**val_metrics['aurc_curve'], 'aurc': val_metrics['aurc']}
        plot_risk_coverage_curve(aurc_data_with_value, risk_coverage_path)
        logger.info(f"  Risk-Coverage曲线已保存: {risk_coverage_path}")

    # 熵分布直方图
    entropy_hist_path = figs_dir / 'val_entropy_histogram.png'
    plot_entropy_histogram(val_probs, entropy_hist_path)
    logger.info(f"  熵分布直方图已保存: {entropy_hist_path}")

    # 验证集3D还原
    logger.info("还原验证集预测到3D...")
    val_probs_3d, val_argmax_3d = restore_predictions_to_3d(val_probs, val_region_mask, mapper)

    # 保存验证集3D预测
    val_pred_3d_path = pred_dir / f"val_{split_info['val_id']}_pred_softmax_3d.npz"
    np.savez_compressed(val_pred_3d_path, pred_softmax_3d=val_probs_3d)
    logger.info(f"验证集3D预测已保存: {val_pred_3d_path}")

    val_argmax_3d_path = pred_dir / f"val_{split_info['val_id']}_argmax_3d.npz"
    np.savez_compressed(val_argmax_3d_path, pred_argmax_3d=val_argmax_3d)
    logger.info(f"验证集3D硬标签已保存: {val_argmax_3d_path}")

    # 计算验证集3D准确率
    val_true_3d = val_data['proba_labels'].argmax(axis=-1)
    val_3d_acc = (val_argmax_3d[val_region_mask > 0] == val_true_3d[val_region_mask > 0]).mean()
    logger.info(f"验证集3D Gross Accuracy: {val_3d_acc:.4f}")

    # 绘制验证集切片
    logger.info("绘制验证集切片...")
    plot_3d_slices(val_argmax_3d, val_true_3d, val_region_mask,
                   figs_dir, split_info['val_id'])

    # 计算验证集3D高级指标
    logger.info("计算3D高级指标...")
    val_true_probs_3d = val_data['proba_labels']  # (Z', X', Y', 102)

    # 3D Soft Dice
    val_3d_dice_results = compute_3d_soft_dice(val_probs_3d, val_true_probs_3d, val_region_mask)
    logger.info(f"  3D Soft Dice (macro): {val_3d_dice_results['soft_dice_macro']:.4f}")

    # 3D概率指标
    val_3d_prob_metrics = compute_3d_prob_metrics(val_probs_3d, val_true_probs_3d, val_region_mask)
    logger.info(f"  3D NLL: {val_3d_prob_metrics['3d_nll']:.4f}")
    logger.info(f"  3D Brier: {val_3d_prob_metrics['3d_brier']:.4f}")

    # 保存3D高级指标
    val_3d_adv_metrics_path = metrics_dir / 'metrics_val_3d_advanced.json'
    with open(val_3d_adv_metrics_path, 'w') as f:
        json.dump({
            'soft_dice': val_3d_dice_results,
            'prob_metrics': val_3d_prob_metrics
        }, f, indent=2)
    logger.info(f"验证集3D高级指标已保存: {val_3d_adv_metrics_path}")

    # 温度缩放校准
    logger.info("\n温度缩放校准...")
    temp_scaler = TemperatureScaler()
    optimal_temp = temp_scaler.fit(val_logits, val_labels, max_iter=100)
    logger.info(f"最优温度: {optimal_temp:.4f}")

    # 应用温度缩放到验证集
    val_probs_calibrated = temp_scaler.transform(val_logits)
    val_metrics_calibrated = compute_metrics(val_probs_calibrated, val_labels, compute_advanced=True)

    logger.info("验证集校准后指标:")
    logger.info(f"  Gross Acc: {val_metrics_calibrated['gross_accuracy']:.4f} (原始: {val_metrics['gross_accuracy']:.4f})")
    logger.info(f"  NLL: {val_metrics_calibrated['nll']:.4f} (原始: {val_metrics['nll']:.4f})")
    logger.info(f"  Soft ECE: {val_metrics_calibrated['soft_ece']:.4f} (原始: {val_metrics['soft_ece']:.4f})")

    # 保存温度缩放结果
    temp_scaling_results = {
        'optimal_temperature': float(optimal_temp),
        'val_metrics_before': {
            'gross_accuracy': val_metrics['gross_accuracy'],
            'nll': val_metrics['nll'],
            'soft_ece': val_metrics['soft_ece'],
            'brier_score': val_metrics['brier_score']
        },
        'val_metrics_after': {
            'gross_accuracy': val_metrics_calibrated['gross_accuracy'],
            'nll': val_metrics_calibrated['nll'],
            'soft_ece': val_metrics_calibrated['soft_ece'],
            'brier_score': val_metrics_calibrated['brier_score']
        }
    }
    temp_scaling_path = metrics_dir / 'temperature_scaling.json'
    with open(temp_scaling_path, 'w') as f:
        json.dump(temp_scaling_results, f, indent=2)
    logger.info(f"温度缩放结果已保存: {temp_scaling_path}")

    # 评估测试集（收集logits和probs）
    logger.info("\n评估测试集...")
    test_logits_list = []
    test_probs_list = []
    with torch.no_grad():
        for features_batch, _ in test_loader:
            features_batch = features_batch.to(device)
            logits = model(features_batch)
            probs = F.softmax(logits, dim=1)
            test_logits_list.append(logits.cpu().numpy())
            test_probs_list.append(probs.cpu().numpy())

    test_logits = np.concatenate(test_logits_list, axis=0)  # (n_vox, 102)
    test_probs = np.concatenate(test_probs_list, axis=0)  # (n_vox, 102)
    test_metrics = compute_metrics(test_probs, test_labels, compute_advanced=True)

    logger.info("测试集指标:")
    for key, value in test_metrics.items():
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                logger.info(f"  {key}: <empty ndarray>")
            elif value.size == 1:
                logger.info(f"  {key}: {float(value):.4f}")
            else:
                logger.info(f"  {key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}, shape={value.shape}")
        # Handle scalar values
        elif isinstance(value, (int, float, np.integer, np.floating)):
            logger.info(f"  {key}: {value:.4f}")
        # Handle lists
        elif isinstance(value, list):
            if len(value) == 0:
                logger.info(f"  {key}: <empty list>")
            elif isinstance(value[0], (int, float, np.integer, np.floating)):
                # For numeric lists, show statistics
                logger.info(f"  {key}: mean={np.mean(value):.4f}, std={np.std(value):.4f}, min={np.min(value):.4f}, max={np.max(value):.4f}")
            else:
                # For non-numeric lists
                logger.info(f"  {key}: <list with {len(value)} items>")
        # Handle dicts
        elif isinstance(value, dict):
            if len(value) == 0:
                logger.info(f"  {key}: <empty dict>")
            else:
                # Skip non-empty nested dicts (like aurc_curve) in console logging
                continue
        else:
            # For other types, show the type info
            logger.info(f"  {key}: <{type(value).__name__}>")

    # 保存测试集指标
    test_metrics_path = metrics_dir / 'metrics_test.json'
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"测试集指标已保存: {test_metrics_path}")

    # 测试集混淆矩阵
    test_pred_hard = test_probs.argmax(axis=1)
    test_true_hard = test_labels.argmax(axis=1)
    test_cm = compute_confusion_matrix(test_pred_hard, test_true_hard, normalize='true')
    test_cm_path = metrics_dir / 'confusion_test.csv'
    save_confusion_matrix(test_cm, test_cm_path)
    logger.info(f"测试集混淆矩阵已保存: {test_cm_path}")

    # 保存测试集软混淆矩阵
    logger.info("计算并保存软混淆矩阵...")
    test_soft_cm = compute_soft_confusion_matrix(test_probs, test_labels)
    test_soft_cm_path = metrics_dir / 'soft_confusion_test.csv'
    save_confusion_matrix(test_soft_cm, test_soft_cm_path)
    logger.info(f"测试集软混淆矩阵已保存: {test_soft_cm_path}")

    # 生成测试集可视化图表
    logger.info("生成高级指标可视化...")

    # 可靠性曲线
    if 'soft_ece_bins' in test_metrics and len(test_metrics['soft_ece_bins']) > 0:
        reliability_path = figs_dir / 'test_reliability_diagram.png'
        plot_reliability_diagram(test_metrics['soft_ece_bins'], reliability_path)
        logger.info(f"  可靠性曲线已保存: {reliability_path}")

    # Risk-Coverage曲线
    if 'aurc_curve' in test_metrics:
        risk_coverage_path = figs_dir / 'test_risk_coverage.png'
        # Add the aurc value to the curve data for plotting
        aurc_data_with_value = {**test_metrics['aurc_curve'], 'aurc': test_metrics['aurc']}
        plot_risk_coverage_curve(aurc_data_with_value, risk_coverage_path)
        logger.info(f"  Risk-Coverage曲线已保存: {risk_coverage_path}")

    # 熵分布直方图
    entropy_hist_path = figs_dir / 'test_entropy_histogram.png'
    plot_entropy_histogram(test_probs, entropy_hist_path)
    logger.info(f"  熵分布直方图已保存: {entropy_hist_path}")

    # 测试集3D还原
    logger.info("还原测试集预测到3D...")
    test_probs_3d, test_argmax_3d = restore_predictions_to_3d(test_probs, test_region_mask, mapper)

    # 保存测试集3D预测
    test_pred_3d_path = pred_dir / f"test_{split_info['test_id']}_pred_softmax_3d.npz"
    np.savez_compressed(test_pred_3d_path, pred_softmax_3d=test_probs_3d)
    logger.info(f"测试集3D预测已保存: {test_pred_3d_path}")

    test_argmax_3d_path = pred_dir / f"test_{split_info['test_id']}_argmax_3d.npz"
    np.savez_compressed(test_argmax_3d_path, pred_argmax_3d=test_argmax_3d)
    logger.info(f"测试集3D硬标签已保存: {test_argmax_3d_path}")

    # 计算测试集3D准确率
    test_true_3d = test_data['proba_labels'].argmax(axis=-1)
    test_3d_acc = (test_argmax_3d[test_region_mask > 0] == test_true_3d[test_region_mask > 0]).mean()
    logger.info(f"测试集3D Gross Accuracy: {test_3d_acc:.4f}")

    # 绘制测试集切片
    logger.info("绘制测试集切片...")
    plot_3d_slices(test_argmax_3d, test_true_3d, test_region_mask,
                   figs_dir, split_info['test_id'])

    # 计算测试集3D高级指标
    logger.info("计算3D高级指标...")
    test_true_probs_3d = test_data['proba_labels']  # (Z', X', Y', 102)

    # 3D Soft Dice
    test_3d_dice_results = compute_3d_soft_dice(test_probs_3d, test_true_probs_3d, test_region_mask)
    logger.info(f"  3D Soft Dice (macro): {test_3d_dice_results['soft_dice_macro']:.4f}")

    # 3D概率指标
    test_3d_prob_metrics = compute_3d_prob_metrics(test_probs_3d, test_true_probs_3d, test_region_mask)
    logger.info(f"  3D NLL: {test_3d_prob_metrics['3d_nll']:.4f}")
    logger.info(f"  3D Brier: {test_3d_prob_metrics['3d_brier']:.4f}")

    # 保存3D高级指标
    test_3d_adv_metrics_path = metrics_dir / 'metrics_test_3d_advanced.json'
    with open(test_3d_adv_metrics_path, 'w') as f:
        json.dump({
            'soft_dice': test_3d_dice_results,
            'prob_metrics': test_3d_prob_metrics
        }, f, indent=2)
    logger.info(f"测试集3D高级指标已保存: {test_3d_adv_metrics_path}")

    # 应用温度缩放到测试集
    logger.info("\n应用温度缩放到测试集...")
    test_probs_calibrated = temp_scaler.transform(test_logits)
    test_metrics_calibrated = compute_metrics(test_probs_calibrated, test_labels, compute_advanced=True)

    logger.info("测试集校准后指标:")
    logger.info(f"  Gross Acc: {test_metrics_calibrated['gross_accuracy']:.4f} (原始: {test_metrics['gross_accuracy']:.4f})")
    logger.info(f"  NLL: {test_metrics_calibrated['nll']:.4f} (原始: {test_metrics['nll']:.4f})")
    logger.info(f"  Soft ECE: {test_metrics_calibrated['soft_ece']:.4f} (原始: {test_metrics['soft_ece']:.4f})")
    logger.info(f"  Brier: {test_metrics_calibrated['brier_score']:.4f} (原始: {test_metrics['brier_score']:.4f})")

    # 更新温度缩放结果（添加测试集）
    temp_scaling_results['test_metrics_before'] = {
        'gross_accuracy': test_metrics['gross_accuracy'],
        'nll': test_metrics['nll'],
        'soft_ece': test_metrics['soft_ece'],
        'brier_score': test_metrics['brier_score']
    }
    temp_scaling_results['test_metrics_after'] = {
        'gross_accuracy': test_metrics_calibrated['gross_accuracy'],
        'nll': test_metrics_calibrated['nll'],
        'soft_ece': test_metrics_calibrated['soft_ece'],
        'brier_score': test_metrics_calibrated['brier_score']
    }

    # 重新保存温度缩放结果（包含测试集）
    with open(temp_scaling_path, 'w') as f:
        json.dump(temp_scaling_results, f, indent=2)
    logger.info(f"温度缩放结果已更新（包含测试集）: {temp_scaling_path}")

    # ========================================================================
    # 步骤6: 生成运行摘要
    # ========================================================================
    logger.info("\n步骤6: 生成运行摘要...")

    run_summary = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'grad_clip_norm': grad_clip_norm,
        'use_class_weights': use_class_weights,
        'class_weight_alpha': class_weight_alpha if use_class_weights else None,
        'max_vox_per_subject': max_vox_per_subject,
        'data_root': str(data_root),
        'n_train_subjects': split_info['n_train'],
        'n_train_voxels': int(train_features.shape[0]),
        'n_val_voxels': int(val_features.shape[0]),
        'n_test_voxels': int(test_features.shape[0]),
        'best_epoch': best_epoch,
        'best_val_nll': float(best_val_nll),
        # Basic metrics
        'val_metrics': val_metrics,
        'val_3d_gross_acc': float(val_3d_acc),
        'test_metrics': test_metrics,
        'test_3d_gross_acc': float(test_3d_acc),
        # Advanced 3D metrics
        'val_3d_soft_dice_macro': val_3d_dice_results['soft_dice_macro'],
        'val_3d_nll': val_3d_prob_metrics['3d_nll'],
        'val_3d_brier': val_3d_prob_metrics['3d_brier'],
        'test_3d_soft_dice_macro': test_3d_dice_results['soft_dice_macro'],
        'test_3d_nll': test_3d_prob_metrics['3d_nll'],
        'test_3d_brier': test_3d_prob_metrics['3d_brier'],
        # Temperature scaling
        'optimal_temperature': temp_scaling_results['optimal_temperature'],
        'val_nll_before_temp_scaling': temp_scaling_results['val_metrics_before']['nll'],
        'val_nll_after_temp_scaling': temp_scaling_results['val_metrics_after']['nll'],
        'val_soft_ece_before_temp_scaling': temp_scaling_results['val_metrics_before']['soft_ece'],
        'val_soft_ece_after_temp_scaling': temp_scaling_results['val_metrics_after']['soft_ece'],
        'test_nll_before_temp_scaling': temp_scaling_results['test_metrics_before']['nll'],
        'test_nll_after_temp_scaling': temp_scaling_results['test_metrics_after']['nll'],
        'test_soft_ece_before_temp_scaling': temp_scaling_results['test_metrics_before']['soft_ece'],
        'test_soft_ece_after_temp_scaling': temp_scaling_results['test_metrics_after']['soft_ece'],
        # File references
        'files': {
            'temperature_scaling': str(temp_scaling_path.relative_to(save_dir)),
            'val_3d_advanced_metrics': str(val_3d_adv_metrics_path.relative_to(save_dir)),
            'test_3d_advanced_metrics': str(test_3d_adv_metrics_path.relative_to(save_dir)),
            'val_soft_confusion': str(val_soft_cm_path.relative_to(save_dir)),
            'test_soft_confusion': str(test_soft_cm_path.relative_to(save_dir))
        }
    }

    summary_path = save_dir / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    logger.info(f"运行摘要已保存: {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("训练流程全部完成！")
    logger.info("=" * 80)
    logger.info(f"结果保存在: {save_dir}")
    logger.info(f"  - 最优模型: {checkpoint_dir / 'best.pth'}")
    logger.info(f"  - 验证集指标: {val_metrics_path}")
    logger.info(f"  - 测试集指标: {test_metrics_path}")
    logger.info(f"  - 3D预测: {pred_dir}")
    logger.info(f"  - 切片图像: {figs_dir}")
    logger.info("=" * 80)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='单轮训练脚本 (36/1/1 Split)')
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据根目录（包含1d/和3d/子目录）')
    parser.add_argument('--val-id', type=str, default=None,
                       help='验证集被试ID（可选）')
    parser.add_argument('--test-id', type=str, default=None,
                       help='测试集被试ID（可选）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数（默认3）')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='批大小（默认256）')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='学习率（默认1e-5）')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='权重衰减（默认1e-5）')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='是否使用类别权重')
    parser.add_argument('--class-weight-alpha', type=float, default=0.5,
                       help='类别权重平衡系数（默认0.5）')
    parser.add_argument('--max-vox-per-subject', type=int, default=None,
                       help='每个被试最大体素数（可选，用于控制显存）')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='梯度裁剪范数（默认1.0，设为0禁用）')
    parser.add_argument('--save-dir', type=str, default='runs/quickstart',
                       help='保存目录（默认runs/quickstart）')
    parser.add_argument('--fold-name', type=str, default=None,
                       help='Fold名称（可选），用于多fold训练时创建子目录，例如"fold_1"')

    args = parser.parse_args()

    run_single_split(
        data_root=args.data_root,
        val_id=args.val_id,
        test_id=args.test_id,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_class_weights=args.use_class_weights,
        class_weight_alpha=args.class_weight_alpha,
        max_vox_per_subject=args.max_vox_per_subject,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        save_dir=args.save_dir,
        fold_name=args.fold_name
    )
