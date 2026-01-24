#!/usr/bin/env python3
"""
单轮训练脚本 (Single-Split Training Runner)
按被试分层的36/1/1划分，支持软标签/硬标签训练与3D还原

支持两种训练模式:
  - soft: 使用软标签 q_i 作为训练目标 (Soft+CEST)
  - hard: 使用硬标签 one_hot(argmax(q_i)) 作为训练目标 (Hard+CEST)

评估时统一使用软标签 q_i 计算所有指标，保证两种训练模式可比

版本: v2.0 (支持 Hard+CEST 对照实验)
日期: 2025-01-23
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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import sys

# 添加1d-3d-convert模块路径（相对train_runner.py，兼容不同仓库布局，避免硬编码绝对路径）
_HERE = Path(__file__).resolve().parent
_MAPPER_REL_PATHS = [
    Path('..') / '..' / 'dataset_create' / '1d-3d-convert',
    Path('..') / '..' / '3D_dev' / 'dataset_create' / '1d-3d-convert',
]

for _rel_path in _MAPPER_REL_PATHS:
    _candidate = (_HERE / _rel_path).resolve()
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise ImportError(
        "找不到 data_3d_1d_mapper.py。请确认存在 dataset_create/1d-3d-convert "
        "或 3D_dev/dataset_create/1d-3d-convert 目录。"
    )

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
            labels: (n_voxels, 102) 软标签（概率分布）或 one-hot 标签
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


def detect_labels_source(data_1d: Dict, data_3d_proba: np.ndarray,
                         data_3d_mask: np.ndarray, n_features: int) -> str:
    """
    自动检测标签来源

    Args:
        data_1d: 1D数据字典 (包含 seg_one_hot)
        data_3d_proba: (Z, X, Y, 102) 3D概率标签
        data_3d_mask: (Z, X, Y) 区域掩码
        n_features: 特征数量

    Returns:
        'seg_one_hot_1d' 或 'proba_labels_3d'
    """
    seg_one_hot = data_1d['seg_one_hot']  # (102, n_vox)
    labels_1d = seg_one_hot.T  # (n_vox, 102)

    # 检查是否为软标签：行和约为1，且存在非0/1的值
    row_sums = labels_1d.sum(axis=1)
    is_sum_one = np.allclose(row_sums, 1.0, atol=1e-3)
    has_soft_values = np.any((labels_1d > 0) & (labels_1d < 1))

    if is_sum_one and has_soft_values:
        return 'seg_one_hot_1d'

    # 尝试使用3D proba_labels
    q_from_3d = data_3d_proba[data_3d_mask > 0]  # (n_vox, 102)
    if q_from_3d.shape[0] == n_features:
        return 'proba_labels_3d'

    # 如果seg_one_hot行和为1（即使是one-hot），仍然可用
    if is_sum_one:
        return 'seg_one_hot_1d'

    raise ValueError(
        f"无法自动检测标签来源:\n"
        f"  seg_one_hot: shape={seg_one_hot.shape}, sum≈1: {is_sum_one}, has_soft: {has_soft_values}\n"
        f"  proba_labels_3d[mask]: shape={q_from_3d.shape}, expected n_vox={n_features}\n"
        "请明确指定 --labels-source"
    )


def get_soft_labels(data: Dict, labels_source: str) -> np.ndarray:
    """
    获取软标签 q_1d (用于评估和soft训练)

    Args:
        data: 被试数据字典
        labels_source: 标签来源 ('seg_one_hot_1d' 或 'proba_labels_3d')

    Returns:
        q_1d: (n_vox, 102) 软标签
    """
    if labels_source == 'seg_one_hot_1d':
        return data['seg_one_hot'].T.astype(np.float32)  # (n_vox, 102)
    elif labels_source == 'proba_labels_3d':
        mask = data['region_mask_lr']
        q_1d = data['proba_labels'][mask > 0].astype(np.float32)  # (n_vox, 102)
        return q_1d
    else:
        raise ValueError(f"未知的 labels_source: {labels_source}")


def soft_to_hard_onehot(soft_labels: np.ndarray) -> np.ndarray:
    """
    将软标签转换为 one-hot 硬标签

    Args:
        soft_labels: (n_vox, 102) 软标签

    Returns:
        hard_onehot: (n_vox, 102) one-hot 编码的硬标签
    """
    n_vox, n_classes = soft_labels.shape
    hard_labels = soft_labels.argmax(axis=1)  # (n_vox,)
    hard_onehot = np.zeros_like(soft_labels)
    hard_onehot[np.arange(n_vox), hard_labels] = 1.0
    return hard_onehot


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
    软标签交叉熵损失（也适用于 one-hot 硬标签）

    Args:
        logits: (batch, 102) 模型输出
        soft_targets: (batch, 102) 软标签（概率分布）或 one-hot
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

    ECE^soft: 使用 true_probs[argmax(pred)] 作为"正确率"

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


def compute_topk_accuracy(pred_probs: np.ndarray, true_labels: np.ndarray, k: int = 3) -> float:
    """计算Top-k准确率 (硬标签定义)"""
    topk_preds = np.argsort(pred_probs, axis=1)[:, -k:]  # (n_vox, k)
    correct = np.any(topk_preds == true_labels[:, None], axis=1)
    return correct.mean()


def compute_selective_risk_metrics(pred_probs: np.ndarray,
                                    true_probs: np.ndarray,
                                    true_hard: np.ndarray) -> Dict[str, float]:
    """
    计算选择性预测风险指标 (Chapter 6 核心指标)

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实软标签
        true_hard: (n_vox,) 真实硬标签 = argmax(q)

    Returns:
        dict: risk_soft_at_95cov, risk_hard_at_95cov, coverage_at_5pct_risk_soft
    """
    n = len(pred_probs)
    pred_hard = pred_probs.argmax(axis=1)

    # 置信度 = max(p_i)
    conf = pred_probs.max(axis=1)  # (n_vox,)

    # Soft correctness: q[i, argmax(p_i)]
    correctness_soft = true_probs[np.arange(n), pred_hard]  # (n_vox,)

    # Hard correctness: 1[argmax(p) == argmax(q)]
    correctness_hard = (pred_hard == true_hard).astype(np.float32)  # (n_vox,)

    # 按置信度降序排序
    sorted_idx = np.argsort(conf)[::-1]
    conf_sorted = conf[sorted_idx]
    correctness_soft_sorted = correctness_soft[sorted_idx]
    correctness_hard_sorted = correctness_hard[sorted_idx]

    # ========== risk_soft_at_95cov ==========
    # 保留 95% 的样本（置信度最高的 95%）
    n_retain_95 = int(np.ceil(0.95 * n))
    risk_soft_at_95cov = 1.0 - correctness_soft_sorted[:n_retain_95].mean()

    # ========== risk_hard_at_95cov ==========
    risk_hard_at_95cov = 1.0 - correctness_hard_sorted[:n_retain_95].mean()

    # ========== coverage_at_5pct_risk_soft ==========
    # 找到使 risk_soft <= 0.05 的最大 coverage
    # 从高置信度开始累积，找到 risk 首次超过 0.05 的点
    coverage_at_5pct_risk_soft = 0.0
    cumsum_correct = np.cumsum(correctness_soft_sorted)
    for k in range(1, n + 1):
        avg_correct = cumsum_correct[k - 1] / k
        risk_k = 1.0 - avg_correct
        coverage_k = k / n
        if risk_k <= 0.05:
            coverage_at_5pct_risk_soft = coverage_k
        else:
            break  # risk 超过 5%，停止

    return {
        'risk_soft_at_95cov': float(risk_soft_at_95cov),
        'risk_hard_at_95cov': float(risk_hard_at_95cov),
        'coverage_at_5pct_risk_soft': float(coverage_at_5pct_risk_soft)
    }


def compute_metrics(pred_probs: np.ndarray,
                    true_probs: np.ndarray,
                    pred_hard: Optional[np.ndarray] = None,
                    true_hard: Optional[np.ndarray] = None,
                    compute_advanced: bool = True,
                    ece_n_bins: int = 15) -> Dict[str, Any]:
    """
    计算评估指标（包含高级软标签指标）

    注意：所有指标都基于 true_probs (软标签 q) 计算，保证 soft/hard 训练可比

    Args:
        pred_probs: (n_vox, 102) 预测概率
        true_probs: (n_vox, 102) 真实概率（软标签 q）
        pred_hard: (n_vox,) 预测硬标签（可选，否则自动argmax）
        true_hard: (n_vox,) 真实硬标签（可选，否则自动argmax(q)）
        compute_advanced: 是否计算高级指标（ECE, Brier分解等）
        ece_n_bins: ECE 的 bin 数量

    Returns:
        metrics: 指标字典
    """
    if pred_hard is None:
        pred_hard = pred_probs.argmax(axis=1)
    if true_hard is None:
        true_hard = true_probs.argmax(axis=1)

    # ========== 基础指标 ==========
    # Gross accuracy (Top-1 hard): argmax(p) == argmax(q)
    gross_acc = accuracy_score(true_hard, pred_hard)

    # NLL: E[-log p(q)] = -Σ q * log p
    eps = 1e-8
    nll = -np.log(np.clip(pred_probs, eps, 1.0))
    nll = (nll * true_probs).sum(axis=1).mean()

    # Brier score: E[(p - q)^2]
    brier = np.mean((pred_probs - true_probs) ** 2)

    n_classes = pred_probs.shape[1]
    cm_counts = confusion_matrix(true_hard, pred_hard, labels=np.arange(n_classes))
    support = cm_counts.sum(axis=1)  # 每类真实样本数
    with np.errstate(divide='ignore', invalid='ignore'):
        recall_per_class = np.diag(cm_counts) / np.maximum(support, 1)  # support=0 时 recall=0
    present = support > 0
    balanced_acc = float(np.mean(recall_per_class[present])) if np.any(present) else 0.0

    # Macro/Micro F1 (基于硬标签；宏平均仅在真实出现的类上平均，避免分母随fold变化)
    f1_per_class = f1_score(
        true_hard, pred_hard,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0
    )
    macro_f1 = float(f1_per_class[present].mean()) if np.any(present) else 0.0
    micro_f1 = f1_score(true_hard, pred_hard, average='micro', zero_division=0)
    weighted_f1 = f1_score(true_hard, pred_hard, average='weighted', zero_division=0)

    # Cohen's Kappa
    kappa = cohen_kappa_score(true_hard, pred_hard, labels=np.arange(n_classes))

    # Top-k accuracy (硬标签定义: argmax(q) in top-k of p)
    top3_acc = compute_topk_accuracy(pred_probs, true_hard, k=3)
    top5_acc = compute_topk_accuracy(pred_probs, true_hard, k=5)

    metrics = {
        'gross_accuracy': float(gross_acc),  # = top1_hard
        'top1_accuracy': float(gross_acc),   # 显式命名
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'nll': float(nll),
        'brier_score': float(brier),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'weighted_f1': float(weighted_f1),
        'balanced_accuracy': float(balanced_acc),
        'kappa': float(kappa)
    }

    # ========== 高级软标签指标 ==========
    if compute_advanced:
        # Soft ECE (总体)
        ece_results = compute_soft_ece(pred_probs, true_probs, n_bins=ece_n_bins)
        metrics['soft_ece'] = ece_results['ece']
        metrics['soft_ece_bins'] = ece_results['bin_stats']

        # Classwise ECE
        classwise_ece = compute_classwise_ece(pred_probs, true_probs, n_bins=ece_n_bins)
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

        # 选择性预测风险指标 (Chapter 6)
        selective_risk = compute_selective_risk_metrics(pred_probs, true_probs, true_hard)
        metrics.update(selective_risk)

    return metrics


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
    """
    温度缩放校准器 (log_T 参数化版本)

    使用 log_T 参数化保证 T > 0:
        T = exp(log_T) + eps
    这确保 argmax 不变性可审计。
    """

    def __init__(self, eps: float = 1e-6):
        self.temperature = 1.0
        self.log_T = 0.0  # log(1.0) = 0
        self.eps = eps
        self.eps_triggered = False
        self.fit_info = {}

    def fit(self, logits: np.ndarray, true_probs: np.ndarray,
            max_iter: int = 100, lr: float = 0.01) -> float:
        """
        在验证集上拟合最优温度 (log_T 参数化)

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

        # 初始化 log_T 参数 (log(1.0) = 0)
        log_T = torch.nn.Parameter(torch.zeros(1))

        # 使用LBFGS优化
        optimizer = LBFGS([log_T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            # T = exp(log_T) + eps，保证 T > 0
            T = torch.exp(log_T) + self.eps
            # 应用温度缩放
            scaled_logits = logits_torch / T
            # 计算软标签NLL损失
            log_probs = F_torch.log_softmax(scaled_logits, dim=1)
            loss = -(true_probs_torch * log_probs).sum(dim=1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)

        # 计算最终温度
        final_log_T = float(log_T.item())
        T_from_exp = float(np.exp(final_log_T))
        final_T = T_from_exp + self.eps

        # 检查 eps 是否对结果有显著影响
        self.eps_triggered = (self.eps / final_T) > 0.01  # eps 贡献超过 1%

        # 保存结果
        self.log_T = final_log_T
        self.temperature = final_T

        # 保存拟合信息
        self.fit_info = {
            'max_iter': max_iter,
            'lr': lr,
            'optimizer': 'LBFGS',
            'parameterization': 'log_T',
            'log_T': final_log_T,
            'T_from_exp': T_from_exp,
            'eps': self.eps,
            'eps_triggered': self.eps_triggered
        }

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
    exclude_file: Optional[str] = None,
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
    fold_name: Optional[str] = None,
    # ========== 新增参数 ==========
    supervision: str = 'soft',
    labels_source: str = 'auto',
    save_prepost_preds: bool = True,
    ece_n_bins: int = 15
):
    """
    单轮训练主函数

    Args:
        data_root: 数据根目录（包含1d/和3d/子目录）
        exclude_file: 可选，包含需要排除的被试关键字的文件（每行一个，支持部分匹配，忽略空行/注释）
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
        supervision: 训练监督模式 ('soft' 或 'hard')
        labels_source: 标签来源 ('auto', 'seg_one_hot_1d', 'proba_labels_3d')
        save_prepost_preds: 是否保存校准前/后的预测
        ece_n_bins: ECE 计算的 bin 数量
    """
    # 验证参数
    assert supervision in ['soft', 'hard'], f"supervision 必须是 'soft' 或 'hard'，当前: {supervision}"
    assert labels_source in ['auto', 'seg_one_hot_1d', 'proba_labels_3d'], \
        f"labels_source 必须是 'auto', 'seg_one_hot_1d', 'proba_labels_3d'，当前: {labels_source}"

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

    pred_1d_dir = save_dir / 'pred_1d'
    pred_1d_dir.mkdir(exist_ok=True)

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
    logger.info(f"开始单轮训练 (36/1/1 Split) - Supervision: {supervision.upper()}")
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

    logger.info(f"初始找到 {len(all_subject_ids)} 个被试")

    # 应用排除列表（可选）
    if exclude_file is not None:
        exclude_path = Path(exclude_file)
        if not exclude_path.exists():
            raise FileNotFoundError(f"排除列表不存在: {exclude_path}")

        with open(exclude_path, 'r') as f:
            exclude_keywords = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')
            ]

        if exclude_keywords:
            filtered_ids = []
            excluded_ids = []
            for sid in all_subject_ids:
                if any(keyword in sid for keyword in exclude_keywords):
                    excluded_ids.append(sid)
                else:
                    filtered_ids.append(sid)

            logger.info(f"应用排除列表 {exclude_path}，过滤掉 {len(excluded_ids)} 个被试: {excluded_ids}")
            all_subject_ids = filtered_ids
        else:
            logger.warning(f"排除列表 {exclude_path} 为空，未过滤任何被试")

    logger.info(f"过滤后保留 {len(all_subject_ids)} 个被试: {all_subject_ids}")

    if len(all_subject_ids) < 3:
        raise ValueError(f"被试数不足3个，无法划分36/1/1")

    # 确认指定的测试/验证被试未被排除
    if test_id is not None and test_id not in all_subject_ids:
        raise ValueError(f"指定的测试被试 {test_id} 不在过滤后的被试列表中")
    if val_id is not None and val_id not in all_subject_ids:
        raise ValueError(f"指定的验证被试 {val_id} 不在过滤后的被试列表中")

    # 划分被试
    split_info = split_subjects(all_subject_ids, test_id=test_id, val_id=val_id, seed=seed)

    logger.info(f"训练集: {split_info['n_train']} 个被试")
    logger.info(f"验证集: {split_info['val_id']}")
    logger.info(f"测试集: {split_info['test_id']}")

    # 保存划分信息（包含 supervision 和 labels_source）
    split_info['supervision'] = supervision
    split_info['labels_source'] = labels_source

    split_summary_path = save_dir / 'split_summary.json'
    with open(split_summary_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"划分信息已保存: {split_summary_path}")

    # ========================================================================
    # 步骤2: 加载数据并逐被试z-score
    # ========================================================================
    logger.info("\n步骤2: 加载数据并逐被试z-score标准化...")
    logger.info(f"  训练监督模式: {supervision}")
    logger.info(f"  标签来源: {labels_source}")

    train_features_list = []
    train_labels_list = []       # 训练用标签 (soft 或 one-hot hard)
    train_soft_labels_list = []  # 软标签 q (用于评估和 class weights)

    val_features = None
    val_labels = None            # 训练用标签
    val_soft_labels = None       # 软标签 q (用于评估)
    val_region_mask = None

    test_features = None
    test_labels = None
    test_soft_labels = None
    test_region_mask = None

    norm_stats = {}
    detected_labels_source = labels_source

    # 加载训练集
    logger.info("加载训练集...")
    for subject_id in split_info['train_ids']:
        data = load_subject_data(data_root, subject_id)

        features = data['multidim_data']  # (n_vox, 351)

        # 检测或使用指定的标签来源
        if labels_source == 'auto':
            detected_labels_source = detect_labels_source(
                {'seg_one_hot': data['seg_one_hot']},
                data['proba_labels'],
                data['region_mask_lr'],
                features.shape[0]
            )
            logger.info(f"  {subject_id}: 自动检测标签来源 -> {detected_labels_source}")

        # 获取软标签 q_1d
        q_1d = get_soft_labels(data, detected_labels_source)  # (n_vox, 102)

        # 根据 supervision 模式构建训练标签
        if supervision == 'soft':
            train_label = q_1d
        else:  # hard
            train_label = soft_to_hard_onehot(q_1d)

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
            train_label = train_label[indices]
            q_1d = q_1d[indices]

        train_features_list.append(features_norm)
        train_labels_list.append(train_label)
        train_soft_labels_list.append(q_1d)

        logger.info(f"  {subject_id}: {features_norm.shape[0]} 体素")

    # 合并训练集
    train_features = np.concatenate(train_features_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    train_soft_labels = np.concatenate(train_soft_labels_list, axis=0)

    logger.info(f"训练集总体素数: {train_features.shape[0]}")

    # 加载验证集
    logger.info("加载验证集...")
    val_data = load_subject_data(data_root, split_info['val_id'])
    val_features, val_stats = zscore_per_subject(val_data['multidim_data'])

    # 获取验证集软标签
    val_soft_labels = get_soft_labels(val_data, detected_labels_source)
    if supervision == 'soft':
        val_labels = val_soft_labels
    else:
        val_labels = soft_to_hard_onehot(val_soft_labels)

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

    # 获取测试集软标签
    test_soft_labels = get_soft_labels(test_data, detected_labels_source)
    if supervision == 'soft':
        test_labels = test_soft_labels
    else:
        test_labels = soft_to_hard_onehot(test_soft_labels)

    test_region_mask = test_data['region']
    norm_stats[split_info['test_id']] = {
        'mean': test_stats['mean'].tolist(),
        'std': test_stats['std'].tolist()
    }
    logger.info(f"  {split_info['test_id']}: {test_features.shape[0]} 体素")

    # 更新划分信息中的实际标签来源
    split_info['labels_source_detected'] = detected_labels_source

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

    # 计算类别权重（基于软标签，保证 soft/hard 一致）
    class_weights = None
    if use_class_weights:
        logger.info("计算类别权重（基于软标签）...")
        class_weights = compute_class_weights(train_soft_labels, alpha=class_weight_alpha)
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
        'val_gross_acc': [],
        'val_macro_f1': [],
        'val_balanced_acc': [],
        'val_top3_acc': [],
        'val_top5_acc': [],
        'lr': []
    }

    best_val_nll = float('inf')
    best_epoch = 0

    # 训练循环
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0].get('lr', lr)
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
        history['lr'].append(current_lr)

        # 验证阶段（使用软标签评估）
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

        avg_val_loss = val_loss_accum / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # 计算验证指标（使用软标签 q 评估）
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_metrics = compute_metrics(val_probs, val_soft_labels, ece_n_bins=ece_n_bins)

        history['val_nll'].append(val_metrics['nll'])
        history['val_gross_acc'].append(val_metrics['gross_accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
        history['val_top3_acc'].append(val_metrics['top3_accuracy'])
        history['val_top5_acc'].append(val_metrics['top5_accuracy'])

        logger.info(f"Epoch {epoch+1} 结果:")
        logger.info(f"  LR: {current_lr:.6g}")
        logger.info(f"  训练损失: {avg_train_loss:.4f}")
        logger.info(f"  验证损失: {avg_val_loss:.4f}")
        logger.info(f"  验证NLL: {val_metrics['nll']:.4f}")
        logger.info(f"  验证Gross Acc: {val_metrics['gross_accuracy']:.4f}")
        logger.info(f"  验证Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  验证Macro-F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"  验证Top-3: {val_metrics['top3_accuracy']:.4f}")
        logger.info(f"  验证Top-5: {val_metrics['top5_accuracy']:.4f}")

        # 保存最优模型（基于 NLL on soft labels）
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
    val_probs = np.concatenate(val_probs_list, axis=0)    # (n_vox, 102)

    # 计算验证集未校准指标（使用软标签 q 评估）
    val_metrics_uncalibrated = compute_metrics(val_probs, val_soft_labels,
                                                compute_advanced=True, ece_n_bins=ece_n_bins)

    logger.info("验证集未校准指标:")
    logger.info(f"  Gross Acc (Top-1 Hard): {val_metrics_uncalibrated['gross_accuracy']:.4f}")
    logger.info(f"  Top-3 Acc: {val_metrics_uncalibrated['top3_accuracy']:.4f}")
    logger.info(f"  Top-5 Acc: {val_metrics_uncalibrated['top5_accuracy']:.4f}")
    logger.info(f"  NLL: {val_metrics_uncalibrated['nll']:.4f}")
    logger.info(f"  Soft ECE: {val_metrics_uncalibrated['soft_ece']:.4f}")
    logger.info(f"  Macro-F1: {val_metrics_uncalibrated['macro_f1']:.4f}")
    logger.info(f"  Brier: {val_metrics_uncalibrated['brier_score']:.4f}")

    # 保存验证集未校准指标
    val_metrics_uncal_path = metrics_dir / 'metrics_val_uncalibrated.json'
    with open(val_metrics_uncal_path, 'w') as f:
        json.dump(val_metrics_uncalibrated, f, indent=2)
    logger.info(f"验证集未校准指标已保存: {val_metrics_uncal_path}")

    # 温度缩放校准（在验证集上拟合）
    logger.info("\n温度缩放校准...")
    temp_scaler = TemperatureScaler()
    optimal_temp = temp_scaler.fit(val_logits, val_soft_labels, max_iter=100)
    logger.info(f"最优温度: {optimal_temp:.4f}")

    # 应用温度缩放到验证集
    val_probs_calibrated = temp_scaler.transform(val_logits)
    val_metrics_calibrated = compute_metrics(val_probs_calibrated, val_soft_labels,
                                              compute_advanced=True, ece_n_bins=ece_n_bins)

    logger.info("验证集校准后指标:")
    logger.info(f"  Gross Acc: {val_metrics_calibrated['gross_accuracy']:.4f} (原始: {val_metrics_uncalibrated['gross_accuracy']:.4f})")
    logger.info(f"  NLL: {val_metrics_calibrated['nll']:.4f} (原始: {val_metrics_uncalibrated['nll']:.4f})")
    logger.info(f"  Soft ECE: {val_metrics_calibrated['soft_ece']:.4f} (原始: {val_metrics_uncalibrated['soft_ece']:.4f})")

    # 保存验证集校准后指标
    val_metrics_cal_path = metrics_dir / 'metrics_val_calibrated.json'
    with open(val_metrics_cal_path, 'w') as f:
        json.dump(val_metrics_calibrated, f, indent=2)
    logger.info(f"验证集校准后指标已保存: {val_metrics_cal_path}")

    # 验证集混淆矩阵
    val_pred_hard = val_probs.argmax(axis=1)
    val_true_hard = val_soft_labels.argmax(axis=1)
    val_cm = compute_confusion_matrix(val_pred_hard, val_true_hard, normalize='true')
    val_cm_path = metrics_dir / 'confusion_val.csv'
    save_confusion_matrix(val_cm, val_cm_path)
    logger.info(f"验证集混淆矩阵已保存: {val_cm_path}")

    # 保存验证集软混淆矩阵
    val_soft_cm = compute_soft_confusion_matrix(val_probs, val_soft_labels)
    val_soft_cm_path = metrics_dir / 'soft_confusion_val.csv'
    save_confusion_matrix(val_soft_cm, val_soft_cm_path)

    # 生成验证集可视化图表
    logger.info("生成高级指标可视化...")

    # 可靠性曲线
    if 'soft_ece_bins' in val_metrics_uncalibrated and len(val_metrics_uncalibrated['soft_ece_bins']) > 0:
        reliability_path = figs_dir / 'val_reliability_diagram.png'
        plot_reliability_diagram(val_metrics_uncalibrated['soft_ece_bins'], reliability_path)

    # Risk-Coverage曲线
    if 'aurc_curve' in val_metrics_uncalibrated:
        risk_coverage_path = figs_dir / 'val_risk_coverage.png'
        aurc_data_with_value = {**val_metrics_uncalibrated['aurc_curve'], 'aurc': val_metrics_uncalibrated['aurc']}
        plot_risk_coverage_curve(aurc_data_with_value, risk_coverage_path)

    # 熵分布直方图
    entropy_hist_path = figs_dir / 'val_entropy_histogram.png'
    plot_entropy_histogram(val_probs, entropy_hist_path)

    # 验证集3D还原（未校准和校准后）
    logger.info("还原验证集预测到3D...")
    val_probs_3d_preT, val_argmax_3d_preT = restore_predictions_to_3d(val_probs, val_region_mask, mapper)
    val_probs_3d_postT, val_argmax_3d_postT = restore_predictions_to_3d(val_probs_calibrated, val_region_mask, mapper)

    # 计算验证集3D准确率
    val_true_3d = val_data['proba_labels'].argmax(axis=-1)
    val_3d_acc_preT = (val_argmax_3d_preT[val_region_mask > 0] == val_true_3d[val_region_mask > 0]).mean()
    val_3d_acc_postT = (val_argmax_3d_postT[val_region_mask > 0] == val_true_3d[val_region_mask > 0]).mean()
    logger.info(f"验证集3D Gross Accuracy: preT={val_3d_acc_preT:.4f}, postT={val_3d_acc_postT:.4f}")

    # 计算验证集3D高级指标
    val_true_probs_3d = val_data['proba_labels']

    val_3d_dice_preT = compute_3d_soft_dice(val_probs_3d_preT, val_true_probs_3d, val_region_mask)
    val_3d_dice_postT = compute_3d_soft_dice(val_probs_3d_postT, val_true_probs_3d, val_region_mask)
    logger.info(f"验证集3D Soft Dice (macro): preT={val_3d_dice_preT['soft_dice_macro']:.4f}, postT={val_3d_dice_postT['soft_dice_macro']:.4f}")

    val_3d_prob_preT = compute_3d_prob_metrics(val_probs_3d_preT, val_true_probs_3d, val_region_mask)
    val_3d_prob_postT = compute_3d_prob_metrics(val_probs_3d_postT, val_true_probs_3d, val_region_mask)

    # 保存3D高级指标
    val_3d_metrics = {
        'uncalibrated': {
            'gross_accuracy': float(val_3d_acc_preT),
            'soft_dice_macro': val_3d_dice_preT['soft_dice_macro'],
            '3d_nll': val_3d_prob_preT['3d_nll'],
            '3d_brier': val_3d_prob_preT['3d_brier']
        },
        'calibrated': {
            'gross_accuracy': float(val_3d_acc_postT),
            'soft_dice_macro': val_3d_dice_postT['soft_dice_macro'],
            '3d_nll': val_3d_prob_postT['3d_nll'],
            '3d_brier': val_3d_prob_postT['3d_brier']
        }
    }
    val_3d_metrics_path = metrics_dir / 'metrics_val_3d.json'
    with open(val_3d_metrics_path, 'w') as f:
        json.dump(val_3d_metrics, f, indent=2)
    logger.info(f"验证集3D指标已保存: {val_3d_metrics_path}")

    # 保存验证集3D预测
    if save_prepost_preds:
        logger.info("保存验证集预测...")
        # 1D logits and probs
        np.save(pred_1d_dir / 'val_logits.npy', val_logits)
        np.save(pred_1d_dir / 'val_probs_preT.npy', val_probs)
        np.save(pred_1d_dir / 'val_probs_postT.npy', val_probs_calibrated)

        # 3D predictions
        np.savez_compressed(pred_dir / f"val_{split_info['val_id']}_pred_softmax_3d_preT.npz",
                            pred_softmax_3d=val_probs_3d_preT)
        np.savez_compressed(pred_dir / f"val_{split_info['val_id']}_pred_softmax_3d_postT.npz",
                            pred_softmax_3d=val_probs_3d_postT)

    # 绘制验证集切片
    plot_3d_slices(val_argmax_3d_preT, val_true_3d, val_region_mask,
                   figs_dir, split_info['val_id'])

    # ========================================================================
    # 评估测试集
    # ========================================================================
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

    test_logits = np.concatenate(test_logits_list, axis=0)
    test_probs = np.concatenate(test_probs_list, axis=0)

    # 计算测试集未校准指标
    test_metrics_uncalibrated = compute_metrics(test_probs, test_soft_labels,
                                                 compute_advanced=True, ece_n_bins=ece_n_bins)

    logger.info("测试集未校准指标:")
    logger.info(f"  Gross Acc (Top-1 Hard): {test_metrics_uncalibrated['gross_accuracy']:.4f}")
    logger.info(f"  Top-3 Acc: {test_metrics_uncalibrated['top3_accuracy']:.4f}")
    logger.info(f"  Top-5 Acc: {test_metrics_uncalibrated['top5_accuracy']:.4f}")
    logger.info(f"  NLL: {test_metrics_uncalibrated['nll']:.4f}")
    logger.info(f"  Soft ECE: {test_metrics_uncalibrated['soft_ece']:.4f}")
    logger.info(f"  Macro-F1: {test_metrics_uncalibrated['macro_f1']:.4f}")
    logger.info(f"  Brier: {test_metrics_uncalibrated['brier_score']:.4f}")

    # 保存测试集未校准指标
    test_metrics_uncal_path = metrics_dir / 'metrics_test_uncalibrated.json'
    with open(test_metrics_uncal_path, 'w') as f:
        json.dump(test_metrics_uncalibrated, f, indent=2)
    logger.info(f"测试集未校准指标已保存: {test_metrics_uncal_path}")

    # 应用温度缩放到测试集
    logger.info("\n应用温度缩放到测试集...")
    test_probs_calibrated = temp_scaler.transform(test_logits)
    test_metrics_calibrated = compute_metrics(test_probs_calibrated, test_soft_labels,
                                               compute_advanced=True, ece_n_bins=ece_n_bins)

    logger.info("测试集校准后指标:")
    logger.info(f"  Gross Acc: {test_metrics_calibrated['gross_accuracy']:.4f} (原始: {test_metrics_uncalibrated['gross_accuracy']:.4f})")
    logger.info(f"  NLL: {test_metrics_calibrated['nll']:.4f} (原始: {test_metrics_uncalibrated['nll']:.4f})")
    logger.info(f"  Soft ECE: {test_metrics_calibrated['soft_ece']:.4f} (原始: {test_metrics_uncalibrated['soft_ece']:.4f})")
    logger.info(f"  Brier: {test_metrics_calibrated['brier_score']:.4f} (原始: {test_metrics_uncalibrated['brier_score']:.4f})")

    # 保存测试集校准后指标
    test_metrics_cal_path = metrics_dir / 'metrics_test_calibrated.json'
    with open(test_metrics_cal_path, 'w') as f:
        json.dump(test_metrics_calibrated, f, indent=2)
    logger.info(f"测试集校准后指标已保存: {test_metrics_cal_path}")

    # 测试集混淆矩阵
    test_pred_hard = test_probs.argmax(axis=1)
    test_true_hard = test_soft_labels.argmax(axis=1)
    test_cm = compute_confusion_matrix(test_pred_hard, test_true_hard, normalize='true')
    test_cm_path = metrics_dir / 'confusion_test.csv'
    save_confusion_matrix(test_cm, test_cm_path)

    # 保存测试集软混淆矩阵
    test_soft_cm = compute_soft_confusion_matrix(test_probs, test_soft_labels)
    test_soft_cm_path = metrics_dir / 'soft_confusion_test.csv'
    save_confusion_matrix(test_soft_cm, test_soft_cm_path)

    # 生成测试集可视化图表
    if 'soft_ece_bins' in test_metrics_uncalibrated and len(test_metrics_uncalibrated['soft_ece_bins']) > 0:
        reliability_path = figs_dir / 'test_reliability_diagram.png'
        plot_reliability_diagram(test_metrics_uncalibrated['soft_ece_bins'], reliability_path)

    if 'aurc_curve' in test_metrics_uncalibrated:
        risk_coverage_path = figs_dir / 'test_risk_coverage.png'
        aurc_data_with_value = {**test_metrics_uncalibrated['aurc_curve'], 'aurc': test_metrics_uncalibrated['aurc']}
        plot_risk_coverage_curve(aurc_data_with_value, risk_coverage_path)

    entropy_hist_path = figs_dir / 'test_entropy_histogram.png'
    plot_entropy_histogram(test_probs, entropy_hist_path)

    # 测试集3D还原
    logger.info("还原测试集预测到3D...")
    test_probs_3d_preT, test_argmax_3d_preT = restore_predictions_to_3d(test_probs, test_region_mask, mapper)
    test_probs_3d_postT, test_argmax_3d_postT = restore_predictions_to_3d(test_probs_calibrated, test_region_mask, mapper)

    # 计算测试集3D准确率
    test_true_3d = test_data['proba_labels'].argmax(axis=-1)
    test_3d_acc_preT = (test_argmax_3d_preT[test_region_mask > 0] == test_true_3d[test_region_mask > 0]).mean()
    test_3d_acc_postT = (test_argmax_3d_postT[test_region_mask > 0] == test_true_3d[test_region_mask > 0]).mean()
    logger.info(f"测试集3D Gross Accuracy: preT={test_3d_acc_preT:.4f}, postT={test_3d_acc_postT:.4f}")

    # 计算测试集3D高级指标
    test_true_probs_3d = test_data['proba_labels']

    test_3d_dice_preT = compute_3d_soft_dice(test_probs_3d_preT, test_true_probs_3d, test_region_mask)
    test_3d_dice_postT = compute_3d_soft_dice(test_probs_3d_postT, test_true_probs_3d, test_region_mask)
    logger.info(f"测试集3D Soft Dice (macro): preT={test_3d_dice_preT['soft_dice_macro']:.4f}, postT={test_3d_dice_postT['soft_dice_macro']:.4f}")

    test_3d_prob_preT = compute_3d_prob_metrics(test_probs_3d_preT, test_true_probs_3d, test_region_mask)
    test_3d_prob_postT = compute_3d_prob_metrics(test_probs_3d_postT, test_true_probs_3d, test_region_mask)

    # 保存3D高级指标
    test_3d_metrics = {
        'uncalibrated': {
            'gross_accuracy': float(test_3d_acc_preT),
            'soft_dice_macro': test_3d_dice_preT['soft_dice_macro'],
            '3d_nll': test_3d_prob_preT['3d_nll'],
            '3d_brier': test_3d_prob_preT['3d_brier']
        },
        'calibrated': {
            'gross_accuracy': float(test_3d_acc_postT),
            'soft_dice_macro': test_3d_dice_postT['soft_dice_macro'],
            '3d_nll': test_3d_prob_postT['3d_nll'],
            '3d_brier': test_3d_prob_postT['3d_brier']
        }
    }
    test_3d_metrics_path = metrics_dir / 'metrics_test_3d.json'
    with open(test_3d_metrics_path, 'w') as f:
        json.dump(test_3d_metrics, f, indent=2)
    logger.info(f"测试集3D指标已保存: {test_3d_metrics_path}")

    # 保存测试集预测
    if save_prepost_preds:
        logger.info("保存测试集预测...")
        # 1D logits and probs
        np.save(pred_1d_dir / 'test_logits.npy', test_logits)
        np.save(pred_1d_dir / 'test_probs_preT.npy', test_probs)
        np.save(pred_1d_dir / 'test_probs_postT.npy', test_probs_calibrated)

        # 3D predictions
        np.savez_compressed(pred_dir / f"test_{split_info['test_id']}_pred_softmax_3d_preT.npz",
                            pred_softmax_3d=test_probs_3d_preT)
        np.savez_compressed(pred_dir / f"test_{split_info['test_id']}_pred_softmax_3d_postT.npz",
                            pred_softmax_3d=test_probs_3d_postT)

    # 绘制测试集切片
    plot_3d_slices(test_argmax_3d_preT, test_true_3d, test_region_mask,
                   figs_dir, split_info['test_id'])

    # ========================================================================
    # 保存温度缩放详细结果
    # ========================================================================
    # 提取论文主表指标子集 (避免保存过大的 bin_stats 等)
    def extract_temp_scaling_metrics(metrics: Dict) -> Dict:
        """提取温度缩放前后对比所需的关键指标"""
        return {
            'gross_accuracy': metrics.get('gross_accuracy'),
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'weighted_f1': metrics.get('weighted_f1'),
            'kappa': metrics.get('kappa'),
            'top3_accuracy': metrics.get('top3_accuracy'),
            'top5_accuracy': metrics.get('top5_accuracy'),
            'nll': metrics.get('nll'),
            'soft_ece': metrics.get('soft_ece'),
            'brier_score': metrics.get('brier_score'),
            'risk_soft_at_95cov': metrics.get('risk_soft_at_95cov'),
            'risk_hard_at_95cov': metrics.get('risk_hard_at_95cov'),
            'coverage_at_5pct_risk_soft': metrics.get('coverage_at_5pct_risk_soft'),
        }

    temp_scaling_results = {
        'optimal_temperature': float(optimal_temp),
        'log_T': temp_scaler.log_T,
        'eps': temp_scaler.eps,
        'eps_triggered': temp_scaler.eps_triggered,
        'val_id': split_info['val_id'],
        'test_id': split_info['test_id'],
        'optimizer_settings': temp_scaler.fit_info,
        # 完整指标集 (便于论文审计)
        'val_metrics_before_full': extract_temp_scaling_metrics(val_metrics_uncalibrated),
        'val_metrics_after_full': extract_temp_scaling_metrics(val_metrics_calibrated),
        'test_metrics_before_full': extract_temp_scaling_metrics(test_metrics_uncalibrated),
        'test_metrics_after_full': extract_temp_scaling_metrics(test_metrics_calibrated),
    }
    temp_scaling_path = metrics_dir / 'temperature_scaling.json'
    with open(temp_scaling_path, 'w') as f:
        json.dump(temp_scaling_results, f, indent=2)
    logger.info(f"温度缩放详细结果已保存: {temp_scaling_path}")

    # ========================================================================
    # 步骤6: 生成运行摘要
    # ========================================================================
    logger.info("\n步骤6: 生成运行摘要...")

    # 提取论文主表所需的关键指标
    def extract_paper_metrics(metrics: Dict) -> Dict:
        """提取论文主表需要的关键指标 (10+ 项)"""
        return {
            # Hard-comparable (基于 y=argmax(q))
            'gross_accuracy': metrics['gross_accuracy'],
            'top1_accuracy': metrics.get('top1_accuracy', metrics['gross_accuracy']),
            'top3_accuracy': metrics['top3_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'balanced_accuracy': metrics.get('balanced_accuracy', None),
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'weighted_f1': metrics.get('weighted_f1', None),
            'kappa': metrics.get('kappa', None),
            # Probabilistic (基于 soft target q)
            'nll': metrics['nll'],
            'soft_ece': metrics['soft_ece'],
            'brier_score': metrics['brier_score'],
            'aurc': metrics.get('aurc', None),
            # Chapter 6 选择性预测指标
            'risk_soft_at_95cov': metrics.get('risk_soft_at_95cov', None),
            'risk_hard_at_95cov': metrics.get('risk_hard_at_95cov', None),
            'coverage_at_5pct_risk_soft': metrics.get('coverage_at_5pct_risk_soft', None)
        }

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

        # 新增字段
        'supervision': supervision,
        'labels_source': labels_source,
        'labels_source_detected': detected_labels_source,
        'ece_n_bins': ece_n_bins,

        # Split info
        'train_ids': split_info['train_ids'],
        'val_id': split_info['val_id'],
        'test_id': split_info['test_id'],
        'n_train_subjects': split_info['n_train'],
        'n_train_voxels': int(train_features.shape[0]),
        'n_val_voxels': int(val_features.shape[0]),
        'n_test_voxels': int(test_features.shape[0]),

        # Training results
        'best_epoch': best_epoch,
        'best_val_nll': float(best_val_nll),

        # Temperature scaling
        'optimal_temperature': temp_scaling_results['optimal_temperature'],

        # Validation metrics (uncalibrated and calibrated)
        'val_metrics_uncalibrated': extract_paper_metrics(val_metrics_uncalibrated),
        'val_metrics_calibrated': extract_paper_metrics(val_metrics_calibrated),

        # Test metrics (uncalibrated and calibrated)
        'test_metrics_uncalibrated': extract_paper_metrics(test_metrics_uncalibrated),
        'test_metrics_calibrated': extract_paper_metrics(test_metrics_calibrated),

        # 3D metrics
        'val_3d_gross_acc_preT': float(val_3d_acc_preT),
        'val_3d_gross_acc_postT': float(val_3d_acc_postT),
        'val_3d_soft_dice_macro_preT': val_3d_dice_preT['soft_dice_macro'],
        'val_3d_soft_dice_macro_postT': val_3d_dice_postT['soft_dice_macro'],
        'test_3d_gross_acc_preT': float(test_3d_acc_preT),
        'test_3d_gross_acc_postT': float(test_3d_acc_postT),
        'test_3d_soft_dice_macro_preT': test_3d_dice_preT['soft_dice_macro'],
        'test_3d_soft_dice_macro_postT': test_3d_dice_postT['soft_dice_macro'],

        # File references (relative paths)
        'files': {
            'split_summary': 'split_summary.json',
            'temperature_scaling': 'temperature_scaling.json',
            'metrics_val_uncalibrated': 'metrics_val_uncalibrated.json',
            'metrics_val_calibrated': 'metrics_val_calibrated.json',
            'metrics_test_uncalibrated': 'metrics_test_uncalibrated.json',
            'metrics_test_calibrated': 'metrics_test_calibrated.json',
            'metrics_val_3d': 'metrics_val_3d.json',
            'metrics_test_3d': 'metrics_test_3d.json',
            'best_model': 'checkpoints/best.pth'
        }
    }

    summary_path = save_dir / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    logger.info(f"运行摘要已保存: {summary_path}")

    # 保存训练历史
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"训练历史已保存: {history_path}")

    logger.info("\n" + "=" * 80)
    logger.info(f"训练流程全部完成！(Supervision: {supervision.upper()})")
    logger.info("=" * 80)
    logger.info(f"结果保存在: {save_dir}")
    logger.info(f"  - 最优模型: {checkpoint_dir / 'best.pth'}")
    logger.info(f"  - 验证集指标 (未校准): {val_metrics_uncal_path}")
    logger.info(f"  - 验证集指标 (校准后): {val_metrics_cal_path}")
    logger.info(f"  - 测试集指标 (未校准): {test_metrics_uncal_path}")
    logger.info(f"  - 测试集指标 (校准后): {test_metrics_cal_path}")
    logger.info(f"  - 温度缩放详情: {temp_scaling_path}")
    logger.info(f"  - 运行摘要: {summary_path}")
    logger.info("=" * 80)

    # 打印论文主表格式的摘要
    logger.info("\n论文主表摘要 (Test Set - Calibrated):")
    logger.info("-" * 60)
    logger.info(f"  Supervision: {supervision.upper()}")
    logger.info(f"  Top-1 (Gross Acc): {test_metrics_calibrated['gross_accuracy']:.4f}")
    logger.info(f"  Top-3: {test_metrics_calibrated['top3_accuracy']:.4f}")
    logger.info(f"  Top-5: {test_metrics_calibrated['top5_accuracy']:.4f}")
    logger.info(f"  NLL (q): {test_metrics_calibrated['nll']:.4f}")
    logger.info(f"  ECE^soft: {test_metrics_calibrated['soft_ece']:.4f}")
    logger.info(f"  Macro-F1: {test_metrics_calibrated['macro_f1']:.4f}")
    logger.info(f"  3D Soft Dice (macro): {test_3d_dice_postT['soft_dice_macro']:.4f}")
    logger.info("-" * 60)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='单轮训练脚本 (36/1/1 Split) - 支持 Soft/Hard CEST')
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据根目录（包含1d/和3d/子目录）')
    parser.add_argument('--exclude-file', type=str, default=None,
                       help='可选，包含需排除的被试关键字的文件路径（每行一个，忽略空行和#注释）')
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

    # ========== 新增参数 ==========
    parser.add_argument('--supervision', type=str, default='soft', choices=['soft', 'hard'],
                       help='训练监督模式: soft (软标签) 或 hard (硬标签)，默认 soft')
    parser.add_argument('--labels-source', type=str, default='auto',
                       choices=['auto', 'seg_one_hot_1d', 'proba_labels_3d'],
                       help='标签来源: auto (自动检测), seg_one_hot_1d, proba_labels_3d，默认 auto')
    parser.add_argument('--save-prepost-preds', action='store_true', default=True,
                       help='是否保存校准前/后的预测（默认 True）')
    parser.add_argument('--no-save-prepost-preds', action='store_false', dest='save_prepost_preds',
                       help='不保存校准前/后的预测')
    parser.add_argument('--ece-n-bins', type=int, default=15,
                       help='ECE 计算的 bin 数量（默认 15）')

    args = parser.parse_args()

    run_single_split(
        data_root=args.data_root,
        exclude_file=args.exclude_file,
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
        fold_name=args.fold_name,
        supervision=args.supervision,
        labels_source=args.labels_source,
        save_prepost_preds=args.save_prepost_preds,
        ece_n_bins=args.ece_n_bins
    )
