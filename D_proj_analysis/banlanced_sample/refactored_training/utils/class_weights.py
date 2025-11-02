#!/usr/bin/env python
# coding: utf-8

"""
类权重计算工具
Class weights computation utilities
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def compute_class_weights(labels, num_classes, method='inverse_freq'):
    """
    计算类权重以处理类别不平衡

    Parameters:
    -----------
    labels : np.ndarray
        训练集标签
    num_classes : int
        类别总数
    method : str
        计算方法
        - 'inverse_freq': 频次的倒数
        - 'effective_num': Effective Number of Samples方法

    Returns:
    --------
    class_weights : torch.Tensor
        类权重张量，形状 (num_classes,)
    """
    # 统计每个类别的样本数
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = np.zeros(num_classes, dtype=np.float32)

    for cls, count in zip(unique, counts):
        if 0 <= cls < num_classes:
            class_counts[cls] = count

    logger.info(f"\n类别分布统计:")
    logger.info(f"{'类别ID':<10} {'样本数':<15} {'占比(%)':<10}")
    logger.info("-" * 40)

    total_samples = labels.shape[0]
    for cls_id in range(num_classes):
        count = class_counts[cls_id]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        logger.info(f"{cls_id:<10} {int(count):<15,} {percentage:<10.2f}")

    if method == 'inverse_freq':
        # 避免除以零
        class_counts = np.maximum(class_counts, 1.0)

        # 计算权重：样本数的倒数
        weights = 1.0 / class_counts

        # 归一化使平均权重为1
        weights = weights / weights.mean()

    elif method == 'effective_num':
        # Effective Number of Samples
        # 参考: https://arxiv.org/abs/1901.05555
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        effective_num = np.maximum(effective_num, 1e-7)
        weights = (1.0 - beta) / effective_num

        # 归一化
        weights = weights / weights.mean()

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"\n类权重统计 (method={method}):")
    logger.info(f"{'类别ID':<10} {'权重':<15} {'样本数':<15}")
    logger.info("-" * 45)

    for cls_id in range(num_classes):
        logger.info(f"{cls_id:<10} {weights[cls_id]:<15.4f} {int(class_counts[cls_id]):<15,}")

    logger.info(f"\n权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
    logger.info(f"权重均值: {weights.mean():.4f}")
    logger.info(f"权重标准差: {weights.std():.4f}")

    # 转换为torch.Tensor
    class_weights = torch.FloatTensor(weights)

    return class_weights


def get_balanced_sampler_weights(labels, num_classes):
    """
    为DataLoader的WeightedRandomSampler计算样本权重

    Parameters:
    -----------
    labels : np.ndarray
        训练集标签
    num_classes : int
        类别总数

    Returns:
    --------
    sample_weights : np.ndarray
        每个样本的权重
    """
    # 计算类权重
    class_weights = compute_class_weights(labels, num_classes, method='inverse_freq')
    class_weights_np = class_weights.numpy()

    # 为每个样本分配权重
    sample_weights = np.zeros(len(labels), dtype=np.float32)

    for i, label in enumerate(labels):
        if 0 <= label < num_classes:
            sample_weights[i] = class_weights_np[label]

    return sample_weights
