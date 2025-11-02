#!/usr/bin/env python
# coding: utf-8

"""
训练和评估工具
Training and evaluation utilities
"""

import torch
import numpy as np
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, loss):
    """
    计算分类指标

    Parameters:
    -----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测结果（logits或类别）
    loss : float
        损失值

    Returns:
    --------
    dict : 包含loss, accuracy, macro_f1
    """
    if y_pred.ndim == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    accuracy = np.mean(y_pred_classes == y_true)
    macro_f1 = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch

    Parameters:
    -----------
    model : nn.Module
        模型
    train_loader : DataLoader
        训练数据加载器
    optimizer : torch.optim.Optimizer
        优化器
    criterion : nn.Module
        损失函数
    device : torch.device
        设备

    Returns:
    --------
    dict : 训练指标
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    n_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        all_predictions.append(output.detach().cpu().numpy())
        all_labels.append(target.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_labels = np.hstack(all_labels)
    avg_loss = total_loss / n_batches

    return calculate_metrics(all_labels, all_predictions, avg_loss)


def evaluate(model, X_data, y_data, criterion, device, batch_size=8192):
    """
    评估模型

    Parameters:
    -----------
    model : nn.Module
        模型
    X_data : np.ndarray
        特征数据
    y_data : np.ndarray
        标签数据
    criterion : nn.Module
        损失函数
    device : torch.device
        设备
    batch_size : int
        批大小

    Returns:
    --------
    dict : 评估指标
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    n_batches = 0

    n_samples = len(X_data)

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_X = torch.FloatTensor(X_data[start_idx:end_idx]).to(device)
            batch_y = torch.LongTensor(y_data[start_idx:end_idx]).to(device)

            output = model(batch_X)
            loss = criterion(output, batch_y)

            total_loss += loss.item()
            n_batches += 1
            all_predictions.append(output.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    avg_loss = total_loss / n_batches

    return calculate_metrics(y_data, all_predictions, avg_loss)
