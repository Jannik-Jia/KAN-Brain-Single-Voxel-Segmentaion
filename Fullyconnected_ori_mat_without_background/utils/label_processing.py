#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标签处理工具函数 - 支持101个标签的数据集
"""

import numpy as np

def process_labels_for_training(labels, config):
    """
    根据配置处理标签用于训练
    
    参数:
        labels: 原始标签 (one-hot或索引格式)
        config: 配置字典
    
    返回:
        processed_labels: 处理后的标签
    """
    # 处理one-hot编码
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # one-hot格式，获取标签索引
        labels = np.argmax(labels, axis=1)
    else:
        # 已经是索引格式
        labels = labels.flatten()
    
    labels = np.array(labels, dtype=int)
    
    if config.get('filter_background', False):
        # 模式1: 过滤背景（假设背景已经在数据加载时过滤）
        # 标签应该是1-100，映射到0-99
        processed = labels - 1
        # 确保没有负值（防止意外的背景标签）
        processed = np.maximum(processed, 0)
        return processed
    else:
        # 模式2: 保留背景
        if config.get('include_background_in_classes', True):
            # 子模式2a: 背景作为分类类别
            # 0-100 → 0-100 (不变)
            return labels
        else:
            # 子模式2b: 背景被忽略
            # 0 → -1, 1-100 → 0-99
            processed = labels.copy()
            processed[labels == 0] = -1
            processed[labels > 0] = labels[labels > 0] - 1
            return processed

def should_filter_background_samples(config):
    """
    判断是否需要在数据加载时过滤背景样本
    
    返回:
        bool: True表示需要过滤背景样本
    """
    return config.get('filter_background', False)

def get_ignore_index(config):
    """
    获取损失函数的ignore_index参数
    
    返回:
        int or None: ignore_index值，None表示不忽略任何标签
    """
    if config.get('filter_background', False):
        return None  # 过滤模式不需要ignore
    else:
        if config.get('include_background_in_classes', True):
            return None  # 背景作为分类类别，不忽略
        else:
            return -1  # 背景映射为-1并忽略

def create_criterion_with_background_config(config, class_weights=None):
    """
    根据背景配置创建损失函数
    
    参数:
        config: 配置字典
        class_weights: 类别权重
    
    返回:
        criterion: 损失函数
    """
    import torch.nn as nn
    
    ignore_index = get_ignore_index(config)
    
    if ignore_index is not None:
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)

def get_label_info_string(config):
    """
    获取标签处理信息的描述字符串
    
    返回:
        str: 描述字符串
    """
    if config.get('filter_background', False):
        return "背景已过滤，标签范围0-99"
    else:
        if config.get('include_background_in_classes', True):
            return "包含背景分类，标签范围0-100"
        else:
            return "背景被忽略，标签范围0-99，背景=-1"

def get_effective_num_classes(config):
    """
    获取实际的类别数量（101标签数据集）
    
    返回:
        int: 实际类别数量
    """
    if config.get('filter_background', False):
        return 100  # 过滤背景：1-100 → 0-99
    else:
        if config.get('include_background_in_classes', True):
            return 101  # 包含背景：0-100 → 0-100
        else:
            return 100  # 忽略背景：0→-1, 1-100 → 0-99