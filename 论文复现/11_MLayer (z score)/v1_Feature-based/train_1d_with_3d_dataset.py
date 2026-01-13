#!/usr/bin/env python3
"""
使用1D训练集进行训练，然后映射回3D
基于Alex的架构，但使用351维输入（而不是341维）
训练后将softmax概率映射回3D体积
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import h5py
import scipy.io
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import gc
import copy

# 导入评估指标模块
from evaluation_metrics import compute_all_metrics

# ==================== 模型定义（基于Alex的架构）====================

class RegModel(nn.Module):
    """
    Alex的1D全连接网络
    原始是341维输入，现在改为351维
    """
    def __init__(self, input_dim=351, num_classes=102):
        super(RegModel, self).__init__()
        # 4层全连接网络 + 输出层
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, num_classes)  # 输出层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # 不应用softmax，让CrossEntropyLoss处理
        return x


# ==================== M-Layer 模型定义 ====================

class MLayerBlock(nn.Module):
    """
    Matrix Exponentiation Layer Block

    核心思想：用矩阵指数 expm(·) 作为非线性变换
    1. 将输入向量映射成方阵 M
    2. 计算矩阵指数 E = expm(M)
    3. 将 E 展平后映射回向量空间

    数值稳定性：
    - scale: 缩放输入矩阵，防止 exp 爆炸 (默认 0.01)
    - clip: 可选的数值裁剪，进一步约束输入范围

    参数量设计：
    - to_matrix: Linear(4096 → 4096) = 16,781,312 参数
    - from_matrix: Linear(4096 → 4096) = 16,781,312 参数
    - 总计 33,562,624 参数，与 fc2+fc3 完全匹配
    """
    def __init__(self, hidden_dim=4096, matrix_size=64, scale=0.01, clip=10.0):
        super(MLayerBlock, self).__init__()

        # 严格验证：matrix_size^2 必须等于 hidden_dim
        assert matrix_size * matrix_size == hidden_dim, \
            f"matrix_size^2 ({matrix_size}^2={matrix_size*matrix_size}) 必须等于 hidden_dim ({hidden_dim})"

        self.hidden_dim = hidden_dim
        self.matrix_size = matrix_size
        self.scale = scale
        self.clip = clip  # None 表示不做裁剪

        # 线性映射层
        self.to_matrix = nn.Linear(hidden_dim, hidden_dim)  # 4096 → 4096
        self.from_matrix = nn.Linear(hidden_dim, hidden_dim)  # 4096 → 4096

    def forward(self, h):
        """
        Args:
            h: 输入张量，形状 [B, hidden_dim] = [B, 4096]
        Returns:
            out: 输出张量，形状 [B, hidden_dim] = [B, 4096]
        """
        B = h.shape[0]

        # Shape check: 输入必须是 [B, hidden_dim]
        assert h.shape == (B, self.hidden_dim), \
            f"输入形状错误: 期望 [{B}, {self.hidden_dim}], 得到 {h.shape}"

        # Step 1: 线性映射并 reshape 成方阵
        M_flat = self.to_matrix(h)  # [B, 4096]
        M = M_flat.view(B, self.matrix_size, self.matrix_size)  # [B, 64, 64]

        # Shape check: 确保矩阵形状正确
        assert M.shape == (B, self.matrix_size, self.matrix_size), \
            f"矩阵形状错误: 期望 [{B}, {self.matrix_size}, {self.matrix_size}], 得到 {M.shape}"

        # Step 2: 数值稳定性处理
        # 缩放：防止矩阵元素过大导致 exp 爆炸
        M = self.scale * M

        # 可选裁剪：进一步约束数值范围
        if self.clip is not None:
            M = torch.clamp(M, -self.clip, self.clip)

        # Step 3: 矩阵指数 (核心非线性)
        # torch.matrix_exp 计算 exp(M) = I + M + M^2/2! + M^3/3! + ...
        E = torch.matrix_exp(M)  # [B, 64, 64]

        # Step 4: 展平并映射回向量空间
        vecE = E.reshape(B, self.hidden_dim)  # [B, 4096]
        out = self.from_matrix(vecE)  # [B, 4096]

        return out


class RegModel_MLayer(nn.Module):
    """
    M-Layer 变体模型

    结构对比：
    Baseline:  fc1 → fc2 → fc3 → fc4 → fc5
    M-Layer:   fc1 → MLayerBlock → fc4 → fc5

    参数量严格匹配 baseline (52,203,622)：
    - fc1: (351×4096) + 4096 = 1,441,792
    - MLayerBlock: 33,562,624 (等于 fc2 + fc3)
    - fc4: (4096×4096) + 4096 = 16,781,312
    - fc5: (4096×102) + 102 = 417,894
    - 总计: 52,203,622
    """
    def __init__(self, input_dim=351, num_classes=102,
                 matrix_size=64, scale=0.01, clip=10.0):
        super(RegModel_MLayer, self).__init__()

        hidden_dim = 4096

        # 验证 matrix_size 设置正确
        assert matrix_size * matrix_size == hidden_dim, \
            f"matrix_size^2 ({matrix_size}^2={matrix_size*matrix_size}) 必须等于 {hidden_dim}"

        # fc1: 输入层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 351 → 4096

        # MLayerBlock: 替换 fc2 + fc3
        self.mblock = MLayerBlock(
            hidden_dim=hidden_dim,
            matrix_size=matrix_size,
            scale=scale,
            clip=clip
        )

        # fc4: 隐藏层
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  # 4096 → 4096

        # fc5: 输出层
        self.fc5 = nn.Linear(hidden_dim, num_classes)  # 4096 → 102

        self.dropout = nn.Dropout(0.5)

        # 保存配置（用于 checkpoint）
        self.m_layer_config = {
            'matrix_size': matrix_size,
            'scale': scale,
            'clip': clip
        }

    def forward(self, x):
        B = x.shape[0]

        # fc1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))  # [B, 4096]

        # MLayerBlock + ReLU + Dropout
        x = self.mblock(x)  # [B, 4096]
        x = self.dropout(F.relu(x))  # 保持与 baseline 类似的激活+正则节奏

        # fc4 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc4(x)))  # [B, 4096]

        # fc5: 输出 logits
        x = self.fc5(x)  # [B, 102]

        # Shape check: 确保输出正确
        assert x.shape == (B, 102), \
            f"输出形状错误: 期望 [{B}, 102], 得到 {x.shape}"

        return x


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def print_model_comparison(input_dim=351, num_classes=102,
                           matrix_size=64, scale=0.01, clip=10.0):
    """打印 baseline 和 m_layer 模型的参数量对比"""
    # 创建临时模型计算参数量
    baseline = RegModel(input_dim=input_dim, num_classes=num_classes)
    mlayer = RegModel_MLayer(
        input_dim=input_dim,
        num_classes=num_classes,
        matrix_size=matrix_size,
        scale=scale,
        clip=clip
    )

    baseline_params = count_parameters(baseline)
    mlayer_params = count_parameters(mlayer)
    diff = abs(mlayer_params - baseline_params)
    diff_ratio = diff / baseline_params * 100

    print(f"\n{'='*60}")
    print("模型参数量对比")
    print(f"{'='*60}")
    print(f"Baseline (RegModel):      {baseline_params:,} 参数")
    print(f"M-Layer (RegModel_MLayer): {mlayer_params:,} 参数")
    print(f"差值: {diff:,} ({diff_ratio:.4f}%)")

    if diff == 0:
        print("✅ 参数量完全匹配！")
    elif diff_ratio < 0.01:
        print("✅ 参数量基本匹配（差值 < 0.01%）")
    else:
        print(f"⚠️ 警告：参数量差异较大 ({diff_ratio:.2f}%)")

    print(f"{'='*60}\n")

    # 清理
    del baseline, mlayer

    return baseline_params, mlayer_params

# ==================== L2正则化 ====================

def kernel_l2_regularization(model, weight_decay=0.00001):
    """
    只对权重矩阵应用L2正则化，不包括偏置
    完全模拟TensorFlow的kernel_regularizer
    """
    l2_reg = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            l2_reg += torch.norm(param, p=2) ** 2
    return weight_decay * l2_reg

# ==================== 1D数据集类 ====================

class Brain1D_Dataset(Dataset):
    """
    从1D MAT文件中加载数据进行训练
    """
    
    def __init__(self, mat_files: List[Path], is_train: bool = True,
                 samples_per_subject: Optional[int] = None):
        """
        Args:
            mat_files: 1D MAT文件路径列表
            is_train: 是否为训练模式
            samples_per_subject: 每个被试采样的体素数（None表示全部）
        """
        self.mat_files = mat_files
        self.is_train = is_train
        self.samples_per_subject = samples_per_subject

        
        # 加载所有数据
        self.all_data = []
        self.all_labels = []
        
        print(f"加载{len(mat_files)}个被试的1D数据...")
        for mat_file in tqdm(mat_files):
            self._load_subject(mat_file)
        
        # 转换为numpy数组
        self.all_data = np.vstack(self.all_data).astype(np.float32)
        self.all_labels = np.concatenate(self.all_labels).astype(np.int64)
        
        print(f"总样本数: {len(self.all_data)}")
        print(f"特征维度: {self.all_data.shape[1]}")
        print(f"类别数: {len(np.unique(self.all_labels))}")
        
    
    def _load_subject(self, mat_file: Path):
        """加载单个被试的1D数据并进行独立标准化"""
        with h5py.File(mat_file, 'r') as f:
            # 加载1D数据
            multidim_data = f['multidim_data'][()]  # (351, n_voxels)或是转置前
            seg_one_hot = f['seg_one_hot'][()]      # (102, n_voxels)
            
            # 转置以适应 (n_voxels, 351)
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T
            
            # 从one-hot转换为类别标签
            labels = np.argmax(seg_one_hot, axis=1)
            
            # ✅ 新增：Patient-wise Z-score 标准化
            # 逻辑：(X - mean) / (std + epsilon)
            # axis=0 表示沿着体素方向计算，保留351个特征的均值/方差
            epsilon = 1e-6
            patient_mean = np.mean(multidim_data, axis=0)
            patient_std = np.std(multidim_data, axis=0)
            
            # 原地修改以节省内存，转换为float32
            multidim_data = (multidim_data - patient_mean) / (patient_std + epsilon)
            multidim_data = multidim_data.astype(np.float32)

            # 采样（如果指定了samples_per_subject）
            if self.samples_per_subject is not None and len(multidim_data) > self.samples_per_subject:
                indices = np.random.choice(len(multidim_data), self.samples_per_subject, replace=False)
                multidim_data = multidim_data[indices]
                labels = labels[indices]
            
            # 添加到总数据中
            self.all_data.append(multidim_data)
            self.all_labels.append(labels)
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

# ==================== 测试集数据类（用于预测和映射）====================

class TestDataset(Dataset):
    """
    用于测试集预测的数据类
    保存3D位置信息用于映射回3D
    """
    
    def __init__(self, mat_file_1d: Path, mat_file_3d: Path):
        
        self.mat_file_1d = mat_file_1d
        self.mat_file_3d = mat_file_3d
        
        # 加载1D数据
        with h5py.File(mat_file_1d, 'r') as f:
            multidim_data = f['multidim_data'][()]
            seg_one_hot = f['seg_one_hot'][()]
            
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T
            
            self.features = multidim_data.astype(np.float32)
            self.labels = np.argmax(seg_one_hot, axis=1)
        
        # ✅ 新增：Patient-wise Z-score 标准化 (针对测试被试自己)
        print(f"正在对测试被试进行标准化: {mat_file_1d.name}")
        epsilon = 1e-6
        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0)
        self.features = (self.features - mean) / (std + epsilon)
        
        # 加载3D mask（用于映射）
        with h5py.File(mat_file_3d, 'r') as f:
            region_mask = f['region_mask'][()]
            region_labels = f['region_labels'][()]
            self.region_mask = region_mask
            self.region_labels = region_labels
        
        print(f"测试数据: {len(self.features)} 个体素")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================== 训练器类 ====================

class Trainer:
    def __init__(self, model, device='cuda', learning_rate=0.00001,
                 min_delta_acc=1e-4, lr_factor=0.5, lr_patience=2, min_lr=1e-7):
        self.model = model.to(device)
        self.device = device
        self.min_delta_acc = min_delta_acc

        # Alex的配置：Adam优化器，学习率1e-5
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器：ReduceLROnPlateau，监控test_acc (mode='max')
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_factor,
            patience=lr_patience,
            threshold=min_delta_acc,
            threshold_mode='abs',
            min_lr=min_lr,
            verbose=False  # 我们手动打印更详细的信息
        )

        # 训练历史 - 扩展为包含所有指标
        self.history = {
            'train_loss': [], 'test_loss': [],
            'train_f1': [], 'test_f1': [],
            'train_acc': [], 'test_acc': [],  # 每个epoch记录gross accuracy
            'lr': [],  # 新增：记录每个epoch的学习率
            # 新增指标
            'train_metrics': [], 'test_metrics': []
        }
    
    def train_epoch(self, train_loader, compute_full_metrics=False):
        """训练一个epoch"""
        self.model.train()
        epoch_train_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            # 计算基础损失
            base_loss = self.criterion(output, target)

            # 添加L2正则化（只对权重）
            l2_reg = kernel_l2_regularization(self.model, weight_decay=0.00001)
            total_loss = base_loss + l2_reg

            total_loss.backward()
            self.optimizer.step()

            epoch_train_loss += total_loss.item()

            # 记录预测、标签和概率
            probs = torch.softmax(output.data, dim=1)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            if compute_full_metrics:
                all_probs.extend(probs.cpu().numpy())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 计算gross accuracy（每个epoch都计算）
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(all_labels, all_preds)

        # 计算完整指标（可选，避免每个epoch都计算）
        train_metrics = None
        if compute_full_metrics and len(all_probs) > 0:
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_probs = np.array(all_probs)
            train_metrics = compute_all_metrics(y_true, y_pred, y_probs)

        return avg_train_loss, train_f1, train_acc, train_metrics
    
    def evaluate(self, val_loader, compute_full_metrics=True):
        """评估模型"""
        self.model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                # 计算损失
                base_loss = self.criterion(output, target)
                l2_reg = kernel_l2_regularization(self.model, weight_decay=0.00001)
                total_loss = base_loss + l2_reg

                total_val_loss += total_loss.item()

                # 记录预测和标签
                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                # 只在需要计算完整指标时才收集概率（节省内存）
                if compute_full_metrics:
                    probs = torch.softmax(output, dim=1)
                    all_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 计算gross accuracy（每个epoch都计算）
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(all_labels, all_preds)

        # 计算完整指标
        val_metrics = None
        if compute_full_metrics and len(all_probs) > 0:
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_probs = np.array(all_probs)
            val_metrics = compute_all_metrics(y_true, y_pred, y_probs)

        return avg_val_loss, val_f1, val_acc, val_metrics
    
    def train(self, train_loader, test_loader, epochs=25, patience=5):
        """
        完整训练流程（对应Alex的25个epochs）

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 最大训练轮数
            patience: early stopping耐心值（连续多少个epoch没改善就停止）
        """
        best_test_acc = 0  # 改用gross accuracy作为最佳模型标准
        best_epoch = 0  # 记录最佳模型出现的epoch
        patience_counter = 0  # early stopping计数器
        early_stopped = False  # 是否提前停止

        # 初始化best_model_state（防止从未更新的情况）
        self.best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n===== Epoch {epoch+1}/{epochs} (lr={current_lr:.2e}) =====")

            # 是否计算完整指标（最后一个epoch或即将early stop）
            # 注意：我们不能提前知道是否会early stop，所以先按正常逻辑
            compute_full = (epoch == epochs - 1)

            # 训练
            train_loss, train_f1, train_acc, train_metrics = self.train_epoch(train_loader, compute_full_metrics=compute_full)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['train_acc'].append(train_acc)
            if train_metrics:
                self.history['train_metrics'].append(train_metrics)

            # 测试（验证）
            test_loss, test_f1, test_acc, test_metrics = self.evaluate(test_loader, compute_full_metrics=compute_full)
            self.history['test_loss'].append(test_loss)
            self.history['test_f1'].append(test_f1)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)  # 记录当前epoch的学习率
            if test_metrics:
                self.history['test_metrics'].append(test_metrics)

            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}")

            # 学习率调度器更新
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(test_acc)
            new_lr = self.optimizer.param_groups[0]['lr']

            # 检测学习率是否下降（用于后续 early-stopping 逻辑）
            lr_reduced = (new_lr < prev_lr)
            if lr_reduced:
                print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

            # 在最后一个epoch显示完整指标
            if compute_full and test_metrics:
                print("\n=== 最终评估指标 ===")
                print(f"Gross Accuracy: {test_metrics['gross_accuracy']:.4f}")
                print(f"Top-1/3/5 Accuracy: {test_metrics['top1_accuracy']:.4f} / {test_metrics['top3_accuracy']:.4f} / {test_metrics['top5_accuracy']:.4f}")
                print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
                print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
                print(f"Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
                print(f"Macro Soft Dice: {test_metrics['macro_soft_dice']:.4f}")
                print(f"Risk@95% Coverage: {test_metrics['risk_at_95_coverage']:.4f}")

            # 保存最佳模型（基于测试gross accuracy）+ Early Stopping逻辑
            # 第一个epoch或有改善时保存（改善需超过min_delta阈值）
            # 判断是否有显著改善
            significant_improvement = (epoch == 0) or (test_acc > best_test_acc + self.min_delta_acc)

            if epoch == 0:
                # 第一个epoch：初始化best
                best_test_acc = test_acc
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"初始测试Gross Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
            elif significant_improvement:
                # 有显著改善（超过min_delta阈值）
                best_test_acc = test_acc
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0  # 重置patience计数器
                print(f"新的最佳测试Gross Accuracy: {best_test_acc:.4f} (Epoch {best_epoch}, 提升>{self.min_delta_acc})")
            else:
                # 未显著改善
                if lr_reduced:
                    # LR刚降低，重置patience_counter，给新学习率更多机会
                    patience_counter = 0
                    print(f"LR刚降低：重置early-stopping patience_counter=0")
                else:
                    patience_counter += 1
                    delta = test_acc - best_test_acc
                    print(f"测试Gross Accuracy未显著改善 (delta={delta:.6f}, 需要>{self.min_delta_acc}, patience: {patience_counter}/{patience})")

                # 检查是否需要early stopping
                if patience_counter >= patience:
                    print(f"\n早停触发！连续{patience}个epoch测试Gross Accuracy未改善")
                    print(f"最佳模型出现在Epoch {best_epoch}，Gross Accuracy: {best_test_acc:.4f}")
                    early_stopped = True
                    break  # 跳出训练循环

        # 训练结束（可能是正常结束或early stopping）
        if early_stopped:
            print(f"\n训练因early stopping在第{epoch+1}个epoch结束（共运行{epoch+1}/{epochs}个epoch）")
        else:
            print(f"\n训练正常完成，共运行{epochs}个epoch")

        # 恢复最佳模型
        print(f"恢复最佳模型（来自Epoch {best_epoch}）")
        self.model.load_state_dict(self.best_model_state)

        # 在最佳模型上重新评估完整指标（确保即使提前停止也有完整指标）
        print("\n===== 最佳模型的完整评估 =====")
        _, best_f1, best_acc, best_metrics = self.evaluate(test_loader, compute_full_metrics=True)
        self.history['best_test_metrics'] = best_metrics

        # 保存early stopping相关信息
        self.history['best_epoch'] = best_epoch
        self.history['best_test_acc'] = best_test_acc  # 按min_delta规则的最佳acc
        self.history['early_stopped'] = early_stopped
        self.history['total_epochs'] = epoch + 1  # 实际运行的epoch数

        if best_metrics:
            print(f"Best Test Gross Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
            print(f"Best Test F1: {best_f1:.4f}")
            print(f"Top-1/3/5 Accuracy: {best_metrics['top1_accuracy']:.4f} / {best_metrics['top3_accuracy']:.4f} / {best_metrics['top5_accuracy']:.4f}")
            print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1: {best_metrics['macro_f1']:.4f}")
            print(f"Weighted F1: {best_metrics['weighted_f1']:.4f}")
            print(f"Cohen's Kappa: {best_metrics['cohen_kappa']:.4f}")
            print(f"Macro Soft Dice: {best_metrics['macro_soft_dice']:.4f}")
            print(f"Risk@95% Coverage: {best_metrics['risk_at_95_coverage']:.4f}")

        return self.history

# ==================== 预测并映射回3D ====================

def predict_and_map_to_3d(model, test_dataset, device='cuda', batch_size=512):
    """
    预测测试集的softmax概率并映射回3D体积
    
    Args:
        model: 训练好的模型
        test_dataset: TestDataset实例
        
    Returns:
        predictions_3d: 3D概率体积 (384, 336, 256, 102)
    """
    model.eval()
    
    # 创建数据加载器
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 预测所有样本
    all_predictions = []
    
    print("预测所有体素...")
    with torch.no_grad():
        for data, _ in tqdm(loader):
            data = data.to(device)
            output = model(data)
            # 应用softmax获取概率
            probs = F.softmax(output, dim=1)
            all_predictions.append(probs.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    
    print(f"预测完成，共{len(all_predictions)}个体素")
    
    # 映射回3D体积 - 使用与dataset_create相同的直接布尔索引方法
    print("映射回3D体积...")
    
    # 获取region掩码（布尔类型）
    region_mask = test_dataset.region_mask
    region = region_mask.astype(bool)  # 转换为布尔掩码，与dataset_create保持一致
    n_voxels = np.sum(region)
    
    # 检查体素数量是否匹配
    if n_voxels != len(all_predictions):
        print(f"警告：体素数量不匹配！")
        print(f"  3D region中的体素数: {n_voxels}")
        print(f"  预测的体素数: {len(all_predictions)}")
        print(f"  差异: {abs(n_voxels - len(all_predictions))}")
        raise ValueError("体素数量必须完全匹配，请检查数据集创建和加载逻辑")
    
    # 创建4D概率体积：(384, 336, 256, 102)
    predictions_3d = np.zeros((384, 336, 256, 102), dtype=np.float32)
    
    # 使用直接布尔索引映射 - 与dataset_create的方法完全一致
    # 参考：data_4d[region] = features (batch_convert_validated_with_logging.py第186行)
    predictions_3d[region, :] = all_predictions  # 直接布尔索引赋值
    
    # 验证映射结果的正确性
    print(f"映射完成，3D体积形状: {predictions_3d.shape}")
    print(f"  有效体素数: {n_voxels}")
    print(f"  概率范围: [{predictions_3d[region].min():.4f}, {predictions_3d[region].max():.4f}]")
    
    # ===== 关键验证1：映射前后逐体素值一致性自检（修正前）=====
    print("验证映射往返一致性（修正前）...")
    recovered_before_fix = predictions_3d[region].copy()  # 从3D体积中取回有效体素的概率
    
    # 检查形状是否匹配
    if recovered_before_fix.shape != all_predictions.shape:
        raise ValueError(f"往返形状不匹配: 原始{all_predictions.shape} vs 恢复{recovered_before_fix.shape}")
    
    # 逐体素精确对比（使用严格的 atol=0）- 验证映射逻辑正确性
    is_consistent_before = np.allclose(recovered_before_fix, all_predictions, atol=0)
    
    if is_consistent_before:
        print(f"  ✅ roundtrip_before_fix: 映射逻辑完全正确，{n_voxels}个体素完全匹配")
    else:
        # 计算不一致的详细信息
        diff = np.abs(recovered_before_fix - all_predictions)
        max_diff = np.max(diff)
        n_diff_voxels = np.sum((diff > 0).any(axis=1))  # 按体素聚合：任一通道有差异的体素数
        print(f"  ❌ roundtrip_before_fix: 映射逻辑错误！")
        print(f"     最大差异: {max_diff}")
        print(f"     不一致体素数: {n_diff_voxels}/{n_voxels}")
        
        # 显示前几个不一致的位置
        diff_indices = np.where((diff > 0).any(axis=1))[0][:5]
        for idx in diff_indices:
            print(f"     体素{idx}: 原始{all_predictions[idx][:3]}... vs 恢复{recovered_before_fix[idx][:3]}...")
        
        raise AssertionError("预测概率映射往返不一致！映射逻辑存在错误")
    
    # ===== 严格的概率约束验证与修正 =====
    print("验证和修正softmax概率约束...")
    
    # 1. 检查有效区域的概率和
    region_probs = predictions_3d[region]  # (n_voxels, 102)
    prob_sums = np.sum(region_probs, axis=1)
    
    # 严格验证概率和是否为1
    perfect_probs = np.sum(np.abs(prob_sums - 1.0) < 1e-6)  # 严格接近1的体素数
    imperfect_probs = n_voxels - perfect_probs
    
    if imperfect_probs > 0:
        max_deviation = np.max(np.abs(prob_sums - 1.0))
        print(f"  ⚠️ 发现{imperfect_probs}/{n_voxels}个体素概率和偏离1，最大偏差: {max_deviation:.8f}")
        
        # 重新归一化修正概率
        print("  🔧 重新归一化修正概率...")
        region_probs_corrected = region_probs / prob_sums.reshape(-1, 1)  # 按行归一化
        predictions_3d[region] = region_probs_corrected
        
        # 验证修正后的概率和
        corrected_sums = np.sum(predictions_3d[region], axis=1)
        perfect_after = np.sum(np.abs(corrected_sums - 1.0) < 1e-10)
        print(f"  ✅ 修正完成，{perfect_after}/{n_voxels}个体素概率和严格为1")
    else:
        print(f"  ✅ 所有{n_voxels}个体素概率和严格为1")
    
    # 2. 检查概率值范围 [0, 1]
    region_probs = predictions_3d[region]
    min_prob = np.min(region_probs)
    max_prob = np.max(region_probs)
    
    if min_prob < 0 or max_prob > 1:
        print(f"  ⚠️ 发现概率值超出[0,1]范围: [{min_prob:.8f}, {max_prob:.8f}]")
        
        # 阈值裁剪
        print("  🔧 进行阈值裁剪...")
        region_probs_clipped = np.clip(region_probs, 0.0, 1.0)
        
        # 重新归一化（裁剪后可能破坏概率和=1的约束）
        clip_sums = np.sum(region_probs_clipped, axis=1)
        region_probs_renorm = region_probs_clipped / clip_sums.reshape(-1, 1)
        predictions_3d[region] = region_probs_renorm
        
        # 验证最终结果
        final_min = np.min(predictions_3d[region])
        final_max = np.max(predictions_3d[region])
        final_sums = np.sum(predictions_3d[region], axis=1)
        perfect_final = np.sum(np.abs(final_sums - 1.0) < 1e-10)
        
        print(f"  ✅ 裁剪完成，概率范围: [{final_min:.8f}, {final_max:.8f}]")
        print(f"  ✅ 重新归一化完成，{perfect_final}/{n_voxels}个体素概率和严格为1")
    else:
        print(f"  ✅ 所有概率值在[0,1]范围内: [{min_prob:.8f}, {max_prob:.8f}]")
    
    # 3. 严格验证背景区域（必须全为0）
    background_sum = np.sum(predictions_3d[~region])
    if background_sum > 0:
        print(f"  ❌ 背景区域概率和非零: {background_sum:.10f}")
        raise ValueError("背景区域必须全为0！映射逻辑存在错误")
    else:
        print(f"  ✅ 背景区域概率和严格为0")
    
    # 4. 预测类别分布统计
    predicted_labels = np.argmax(predictions_3d[region], axis=1)
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    print(f"  📊 预测了 {len(unique_labels)} 个不同类别")
    print(f"  📊 最频繁的类别: {unique_labels[np.argmax(counts)]} (出现{np.max(counts)}次)")
    
    print("概率约束验证与修正完成！")
    
    # ===== 关键验证2：修正后的往返一致性检查（验证修正没破坏掩膜对应关系）=====
    print("验证映射往返一致性（修正后）...")
    recovered_after_fix = predictions_3d[region]  # 修正后从3D体积中取回有效体素的概率
    
    # 检查形状是否匹配
    if recovered_after_fix.shape != all_predictions.shape:
        raise ValueError(f"修正后往返形状不匹配: 原始{all_predictions.shape} vs 恢复{recovered_after_fix.shape}")
    
    # 验证修正没有破坏shape与掩膜对应关系（允许数值变化，但结构必须一致）
    shape_consistent = (recovered_after_fix.shape == all_predictions.shape)
    mask_consistent = np.sum(predictions_3d[~region]) == 0  # 背景依然为0
    
    if shape_consistent and mask_consistent:
        # 计算修正引起的数值变化
        if not np.allclose(recovered_after_fix, all_predictions, atol=0):
            diff_after = np.abs(recovered_after_fix - all_predictions)
            max_diff_after = np.max(diff_after)
            changed_voxels = np.sum((diff_after > 1e-10).any(axis=1))  # 按体素聚合：任一通道变化的体素数
            print(f"  ✅ roundtrip_after_fix: 结构完整，修正影响了{changed_voxels}/{n_voxels}个体素")
            print(f"     最大修正差异: {max_diff_after:.8f}")
        else:
            print(f"  ✅ roundtrip_after_fix: 结构完整，无需修正")
    else:
        raise AssertionError("修正破坏了掩膜对应关系！修正逻辑存在错误")
    
    return predictions_3d

# ==================== 辅助函数：JSON序列化 ====================

def convert_to_json_serializable(obj):
    """
    递归转换对象为JSON可序列化的格式
    处理numpy类型、字典、列表等
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ==================== 辅助函数：读取排除列表 ====================

def load_exclude_list(exclude_file: Path) -> set:
    """
    从文件中读取要排除的被试名列表

    Args:
        exclude_file: 排除列表文件路径

    Returns:
        被试名集合
    """
    exclude_set = set()
    if exclude_file and exclude_file.exists():
        with open(exclude_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if line and not line.startswith('#'):
                    exclude_set.add(line)
    return exclude_set

def should_exclude_subject(subject_name: str, exclude_set: set) -> bool:
    """
    判断被试是否应该被排除

    Args:
        subject_name: 被试名（如 "ODP_01_YHC04"）
        exclude_set: 要排除的被试名集合（如 {"YHC04", "YHC2"}）

    Returns:
        True 如果应该排除，False 否则
    """
    # 检查被试名中是否包含任何排除列表中的字符串
    for exclude_name in exclude_set:
        if exclude_name in subject_name:
            return True
    return False

# ==================== 单次训练函数（用于交叉验证）====================

def train_single_fold(
    test_subject_name: str,
    train_subject_names: List[str],
    idx_1d: Dict,
    idx_3d: Dict,
    fold_output_dir: Path,
    args,
    device,
    logger
) -> Dict:
    """
    训练单个fold

    Args:
        test_subject_name: 测试被试名
        train_subject_names: 训练被试名列表
        idx_1d: 1D文件索引字典
        idx_3d: 3D文件索引字典
        fold_output_dir: 该fold的输出目录
        args: 命令行参数
        device: 计算设备
        logger: 日志器

    Returns:
        fold_result: 包含该fold训练结果的字典
    """
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # 获取文件路径
    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]
    train_files_1d = [idx_1d[name] for name in train_subject_names]

    logger.info(f'训练集: {len(train_files_1d)} 个被试')
    logger.info(f'测试被试名: {test_subject_name}')

    # 创建训练数据集
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        is_train=True,
        samples_per_subject=args.samples_per_subject
    )

    # 创建测试数据集
    test_dataset = TestDataset(test_file_1d, test_file_3d)

    # 验证1D与3D标签一致性
    mask = test_dataset.region_mask.astype(bool)
    labels_3d = test_dataset.region_labels[mask]
    labels_1d = test_dataset.labels

    if len(labels_1d) != len(labels_3d):
        raise ValueError(f"标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")

    if not np.array_equal(labels_1d, labels_3d):
        raise AssertionError("测试集标签在一维与三维不一致！")

    logger.info(f'训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型（根据 model_type 选择）
    if args.model_type == 'baseline':
        model = RegModel(input_dim=351, num_classes=102)
        logger.info('使用 Baseline 模型 (4×4096 MLP)')
    elif args.model_type == 'm_layer':
        model = RegModel_MLayer(
            input_dim=351,
            num_classes=102,
            matrix_size=args.m_matrix_size,
            scale=args.m_scale,
            clip=args.m_clip
        )
        logger.info(f'使用 M-Layer 模型 (matrix_size={args.m_matrix_size}, '
                   f'scale={args.m_scale}, clip={args.m_clip})')
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    logger.info(f'模型参数量: {count_parameters(model):,}')

    # 如果指定了 --load_model，先加载权重作为初始化
    if args.load_model:
        logger.info(f'加载预训练模型作为初始化: {args.load_model}')
        checkpoint = torch.load(args.load_model, map_location='cpu')

        # 检查模型类型兼容性
        saved_model_type = checkpoint.get('model_type', 'baseline')
        if saved_model_type != args.model_type:
            raise ValueError(
                f"模型类型不匹配！Checkpoint 中保存的是 '{saved_model_type}'，"
                f"但当前指定的是 '{args.model_type}'。"
                f"请确保使用相同的 --model_type 参数。"
            )

        model.load_state_dict(checkpoint['model_state_dict'])

    # 创建训练器
    trainer = Trainer(
        model,
        device=device,
        learning_rate=0.00001,
        min_delta_acc=args.min_delta_acc,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr
    )

    # 训练模型
    history = trainer.train(train_loader, test_loader, epochs=args.epochs, patience=args.patience)

    # 保存模型
    model_path = fold_output_dir / 'model.pth'
    save_dict = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'test_subject': test_subject_name,
        'args': vars(args),
        'model_type': args.model_type,  # 保存模型类型
    }
    # 如果是 M-Layer 模型，保存其配置
    if args.model_type == 'm_layer':
        save_dict['m_layer_config'] = {
            'matrix_size': args.m_matrix_size,
            'scale': args.m_scale,
            'clip': args.m_clip
        }
    torch.save(save_dict, model_path)

    # 保存训练历史
    history_path = fold_output_dir / 'history.json'
    history_serializable = convert_to_json_serializable(history)
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    # 构建fold结果
    fold_result = {
        'test_subject': test_subject_name,
        'best_test_acc': history['best_test_acc'],
        'best_epoch': history['best_epoch'],
        'early_stopped': history['early_stopped'],
        'total_epochs': history['total_epochs'],
        'final_test_acc': history['test_acc'][-1],
        'final_test_f1': history['test_f1'][-1],
    }

    # 如果有best_test_metrics，添加更多指标
    if 'best_test_metrics' in history and history['best_test_metrics']:
        metrics = history['best_test_metrics']
        fold_result.update({
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'weighted_f1': metrics.get('weighted_f1'),
            'cohen_kappa': metrics.get('cohen_kappa'),
            'macro_soft_dice': metrics.get('macro_soft_dice'),
        })

    # 清理GPU内存
    del model, trainer, train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='使用1D训练集进行训练，映射回3D')
    parser.add_argument('--data_dir_1d', type=str, required=True,
                       help='1D MAT文件目录（去除_3d后缀的文件）')
    parser.add_argument('--data_dir_3d', type=str, required=True,
                       help='3D MAT文件目录（用于获取3D mask）')
    parser.add_argument('--output_dir', type=str, default='./results_1d',
                       help='输出目录')
    parser.add_argument('--test_subject', type=int, default=38,
                       help='测试被试编号(1-38)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小（Alex使用128）')
    parser.add_argument('--epochs', type=int, default=25,
                       help='训练轮数（Alex使用25）')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping耐心值（连续多少个epoch test gross acc没改善就停止训练，默认5）')
    parser.add_argument('--min_delta_acc', type=float, default=1e-4,
                       help='test_acc改善的最小阈值，低于此值不算改善（默认1e-4）')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='ReduceLROnPlateau的衰减因子（默认0.5）')
    parser.add_argument('--lr_patience', type=int, default=2,
                       help='ReduceLROnPlateau的耐心值（默认2）')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='学习率下限（默认1e-7）')
    parser.add_argument('--samples_per_subject', type=int, default=None,
                       help='每个被试采样的体素数（None表示全部）')
    parser.add_argument('--save_predictions', action='store_true',
                       help='保存3D预测概率')
    parser.add_argument('--load_model', type=str, default=None,
                       help='加载预训练模型路径')
    parser.add_argument('--predict_only', action='store_true',
                       help='仅进行预测，不训练模型')
    parser.add_argument('--exclude_subjects', type=str, default=None,
                       help='要排除的被试名列表文件（txt格式，每行一个被试名）')
    parser.add_argument('--exclude_single', type=str, default=None,
                       help='单次排除的被试名（用于每次排除一个的实验）')
    parser.add_argument('--fixed_test_subject', type=str, default=None,
                       help='固定测试被试名（用于对比实验）')
    parser.add_argument('--cross_validation', action='store_true',
                       help='启用交叉验证模式：每个可用被试轮流作为测试集')

    # ===== M-Layer 模型相关参数 =====
    parser.add_argument('--model_type', type=str, default='m_layer',
                       choices=['baseline', 'm_layer'],
                       help='模型类型: baseline (4x4096 MLP) 或 m_layer (矩阵指数层，默认)')
    parser.add_argument('--m_matrix_size', type=int, default=64,
                       help='M-Layer 矩阵大小 (默认64，必须满足 matrix_size^2=4096)')
    parser.add_argument('--m_scale', type=float, default=0.01,
                       help='M-Layer 输入缩放因子 (默认0.01，用于数值稳定性)')
    parser.add_argument('--m_clip', type=float, default=10.0,
                       help='M-Layer 数值裁剪范围 (默认10.0，设为-1表示不裁剪)')

    args = parser.parse_args()

    # 处理 m_clip: -1 表示不裁剪
    if args.m_clip < 0:
        args.m_clip = None
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # ===== 模型类型配置验证 =====
    logger.info(f'模型类型: {args.model_type}')
    if args.model_type == 'm_layer':
        # 验证 matrix_size^2 == 4096
        if args.m_matrix_size * args.m_matrix_size != 4096:
            raise ValueError(
                f"m_matrix_size^2 ({args.m_matrix_size}^2={args.m_matrix_size**2}) "
                f"必须等于 4096。请使用 --m_matrix_size 64"
            )
        logger.info(f'M-Layer 配置: matrix_size={args.m_matrix_size}, '
                   f'scale={args.m_scale}, clip={args.m_clip}')

    # 打印参数量对比（仅在程序启动时打印一次）
    print_model_comparison(
        input_dim=351,
        num_classes=102,
        matrix_size=args.m_matrix_size,
        scale=args.m_scale,
        clip=args.m_clip
    )

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子（和Alex一样）
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 获取1D和3D MAT文件
    data_dir_1d = Path(args.data_dir_1d)
    data_dir_3d = Path(args.data_dir_3d)
    
    # 查找文件
    mat_files_1d = list(data_dir_1d.glob('*.mat'))
    mat_files_3d = list(data_dir_3d.glob('*_3d_validated.mat'))
    
    logger.info(f'找到1D文件: {len(mat_files_1d)}个')
    logger.info(f'找到3D文件: {len(mat_files_3d)}个')
    
    # 被试名提取函数
    def subject_key_1d(p):
        """提取1D文件的被试名 e.g. "ODP_01_qhlazec" """
        return Path(p).stem
    
    def subject_key_3d(p):
        """提取3D文件的被试名 e.g. "ODP_01_qhlazec_3d_validated" -> "ODP_01_qhlazec" """
        return Path(p).stem.replace('_3d_validated', '')
    
    # 按被试名构建索引
    idx_1d = {subject_key_1d(p): p for p in mat_files_1d}
    idx_3d = {subject_key_3d(p): p for p in mat_files_3d}
    
    # 验证被试集合一致性
    subjects_1d = set(idx_1d.keys())
    subjects_3d = set(idx_3d.keys())
    
    logger.info(f'1D被试: {len(subjects_1d)}个')
    logger.info(f'3D被试: {len(subjects_3d)}个')
    
    assert subjects_1d == subjects_3d, f"一维和三维被试集合不一致！差异: {subjects_1d ^ subjects_3d}"
    
    # 按被试名排序，确保一致性
    all_subject_names = sorted(idx_1d.keys())

    # ===== 排除指定被试的逻辑 =====
    exclude_set = set()

    # 1. 从文件读取排除列表
    if args.exclude_subjects:
        exclude_file = Path(args.exclude_subjects)
        exclude_set = load_exclude_list(exclude_file)
        logger.info(f'从文件加载排除列表: {exclude_file}')
        logger.info(f'要排除的被试标识: {exclude_set}')

    # 2. 单次排除（用于逐个排除实验）
    if args.exclude_single:
        exclude_set.add(args.exclude_single)
        logger.info(f'单次排除被试: {args.exclude_single}')

    # 3. 应用排除过滤
    subject_names = [name for name in all_subject_names
                     if not should_exclude_subject(name, exclude_set)]

    excluded_count = len(all_subject_names) - len(subject_names)
    if excluded_count > 0:
        excluded_names = [name for name in all_subject_names
                         if should_exclude_subject(name, exclude_set)]
        logger.info(f'排除了 {excluded_count} 个被试')
        logger.info(f'被排除的被试: {excluded_names}')

    logger.info(f'剩余可用被试: {len(subject_names)}个')

    # ===== 交叉验证模式 vs 单次训练模式 =====
    if args.cross_validation:
        # ===== 交叉验证模式 =====
        logger.info(f'\n{"="*60}')
        logger.info(f'交叉验证模式: {len(subject_names)}-fold cross validation')
        logger.info(f'{"="*60}\n')

        fold_results = []
        total_folds = len(subject_names)

        for fold_idx, test_subject_name in enumerate(subject_names):
            fold_num = fold_idx + 1
            logger.info(f'\n{"="*60}')
            logger.info(f'Fold {fold_num}/{total_folds}: 测试被试 = {test_subject_name}')
            logger.info(f'{"="*60}')

            # 设置该fold的随机种子（保证可复现）
            fold_seed = 42 + fold_idx
            torch.manual_seed(fold_seed)
            torch.cuda.manual_seed(fold_seed)
            np.random.seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)

            # 训练集：除测试被试外的所有被试
            train_subject_names_fold = [name for name in subject_names if name != test_subject_name]

            # 该fold的输出目录
            fold_output_dir = output_dir / f'fold_{fold_num:02d}_{test_subject_name}'

            # 训练该fold
            try:
                fold_result = train_single_fold(
                    test_subject_name=test_subject_name,
                    train_subject_names=train_subject_names_fold,
                    idx_1d=idx_1d,
                    idx_3d=idx_3d,
                    fold_output_dir=fold_output_dir,
                    args=args,
                    device=device,
                    logger=logger
                )
                fold_result['fold'] = fold_num
                fold_result['status'] = 'success'
                fold_results.append(fold_result)

                logger.info(f'Fold {fold_num} 完成: best_test_acc={fold_result["best_test_acc"]:.4f}')

            except Exception as e:
                logger.error(f'Fold {fold_num} 失败: {e}')
                fold_results.append({
                    'fold': fold_num,
                    'test_subject': test_subject_name,
                    'status': 'failed',
                    'error': str(e)
                })

        # ===== 生成交叉验证汇总报告 =====
        logger.info(f'\n{"="*60}')
        logger.info('交叉验证完成，生成汇总报告...')
        logger.info(f'{"="*60}\n')

        # 筛选成功的fold
        successful_folds = [r for r in fold_results if r.get('status') == 'success']
        failed_folds = [r for r in fold_results if r.get('status') == 'failed']

        if successful_folds:
            # 计算统计量
            test_accs = [r['best_test_acc'] for r in successful_folds]
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)

            best_fold = max(successful_folds, key=lambda x: x['best_test_acc'])
            worst_fold = min(successful_folds, key=lambda x: x['best_test_acc'])

            # 构建汇总报告
            cv_summary = {
                'config': {
                    'total_folds': total_folds,
                    'successful_folds': len(successful_folds),
                    'failed_folds': len(failed_folds),
                    'excluded_subjects': list(exclude_set) if exclude_set else [],
                    'epochs': args.epochs,
                    'patience': args.patience,
                    'batch_size': args.batch_size,
                    'min_delta_acc': args.min_delta_acc,
                    'lr_factor': args.lr_factor,
                    'lr_patience': args.lr_patience,
                    'min_lr': args.min_lr,
                },
                'fold_results': fold_results,
                'summary': {
                    'mean_test_acc': float(mean_acc),
                    'std_test_acc': float(std_acc),
                    'best_fold': {
                        'fold': best_fold['fold'],
                        'test_subject': best_fold['test_subject'],
                        'test_acc': best_fold['best_test_acc']
                    },
                    'worst_fold': {
                        'fold': worst_fold['fold'],
                        'test_subject': worst_fold['test_subject'],
                        'test_acc': worst_fold['best_test_acc']
                    }
                }
            }

            # 如果有更多指标，也计算它们的统计量
            if 'balanced_accuracy' in successful_folds[0] and successful_folds[0]['balanced_accuracy'] is not None:
                ba_values = [r['balanced_accuracy'] for r in successful_folds if r.get('balanced_accuracy') is not None]
                if ba_values:
                    cv_summary['summary']['mean_balanced_accuracy'] = float(np.mean(ba_values))
                    cv_summary['summary']['std_balanced_accuracy'] = float(np.std(ba_values))

            if 'macro_f1' in successful_folds[0] and successful_folds[0]['macro_f1'] is not None:
                f1_values = [r['macro_f1'] for r in successful_folds if r.get('macro_f1') is not None]
                if f1_values:
                    cv_summary['summary']['mean_macro_f1'] = float(np.mean(f1_values))
                    cv_summary['summary']['std_macro_f1'] = float(np.std(f1_values))

            if 'cohen_kappa' in successful_folds[0] and successful_folds[0]['cohen_kappa'] is not None:
                kappa_values = [r['cohen_kappa'] for r in successful_folds if r.get('cohen_kappa') is not None]
                if kappa_values:
                    cv_summary['summary']['mean_cohen_kappa'] = float(np.mean(kappa_values))
                    cv_summary['summary']['std_cohen_kappa'] = float(np.std(kappa_values))

            # 保存汇总报告
            summary_path = output_dir / 'cv_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(cv_summary, f, indent=2)

            # 打印汇总
            logger.info(f'===== 交叉验证汇总 =====')
            logger.info(f'成功完成: {len(successful_folds)}/{total_folds} folds')
            if failed_folds:
                logger.info(f'失败: {len(failed_folds)} folds')
            logger.info(f'平均测试Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
            logger.info(f'最佳Fold: {best_fold["fold"]} ({best_fold["test_subject"]}), acc={best_fold["best_test_acc"]:.4f}')
            logger.info(f'最差Fold: {worst_fold["fold"]} ({worst_fold["test_subject"]}), acc={worst_fold["best_test_acc"]:.4f}')
            logger.info(f'汇总报告已保存至: {summary_path}')

        else:
            logger.error('所有fold都失败了！')

        return  # 交叉验证模式结束

    # ===== 单次训练模式（原有逻辑）=====
    # 选择测试被试
    if args.fixed_test_subject:
        # 固定测试被试名（用于对比实验）
        test_subject_name = None
        for name in subject_names:
            if args.fixed_test_subject in name:
                test_subject_name = name
                break

        if test_subject_name is None:
            raise ValueError(f"未找到匹配的固定测试被试: {args.fixed_test_subject}")

        logger.info(f'使用固定测试被试: {test_subject_name}')
    else:
        # 按索引选择测试被试
        test_idx = args.test_subject - 1
        if test_idx >= len(subject_names):
            raise ValueError(f"测试被试编号{args.test_subject}超出范围[1, {len(subject_names)}]")

        test_subject_name = subject_names[test_idx]

    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]

    # 训练集：除测试被试外的所有被试
    train_subject_names = [name for name in subject_names if name != test_subject_name]
    train_files_1d = [idx_1d[name] for name in train_subject_names]

    logger.info(f'训练集: {len(train_files_1d)} 个被试')
    logger.info(f'测试被试名: {test_subject_name}')
    logger.info(f'测试集1D: {test_file_1d.name}')
    logger.info(f'测试集3D: {test_file_3d.name}')
    
    # 验证文件匹配正确性
    logger.info(f'验证: 1D被试名 = {subject_key_1d(test_file_1d)}')
    logger.info(f'验证: 3D被试名 = {subject_key_3d(test_file_3d)}')
    assert subject_key_1d(test_file_1d) == subject_key_3d(test_file_3d), "测试文件被试名不匹配！"
    
    # 创建训练数据集
    logger.info('加载训练数据...')
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        is_train=True,
        samples_per_subject=args.samples_per_subject
    )
    
    # 创建测试数据集（需要1D和3D文件）
    logger.info('加载测试数据...')
    test_dataset = TestDataset(test_file_1d, test_file_3d)
    
    # ===== 关键验证：1D与3D标签一致性自检 =====
    logger.info('验证1D与3D标签一致性...')
    mask = test_dataset.region_mask.astype(bool)
    labels_3d = test_dataset.region_labels[mask]  # 从3D掩膜位置提取标签
    labels_1d = test_dataset.labels               # 1D数据集的标签
    
    if len(labels_1d) != len(labels_3d):
        logger.error(f"标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")
        raise ValueError("1D和3D标签数量不一致")
    
    labels_match = np.array_equal(labels_1d, labels_3d)
    if labels_match:
        logger.info(f'  ✅ 标签一致性验证通过：{len(labels_1d)}个体素标签完全匹配')
    else:
        # 计算不匹配的详细信息
        n_mismatch = np.sum(labels_1d != labels_3d)
        mismatch_rate = n_mismatch / len(labels_1d) * 100
        
        logger.error(f'  ❌ 标签一致性验证失败！')
        logger.error(f'     不匹配体素数: {n_mismatch}/{len(labels_1d)} ({mismatch_rate:.2f}%)')
        
        # 显示前几个不匹配的位置
        mismatch_indices = np.where(labels_1d != labels_3d)[0][:10]
        for idx in mismatch_indices:
            logger.error(f'     体素{idx}: 1D标签={labels_1d[idx]}, 3D标签={labels_3d[idx]}')
        
        raise AssertionError("测试集标签在一维与三维不一致！可能是文件错配")
    
    logger.info(f'训练样本数: {len(train_dataset)}')
    logger.info(f'测试样本数: {len(test_dataset)}')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型（根据 model_type 选择）
    logger.info('创建模型...')
    if args.model_type == 'baseline':
        model = RegModel(input_dim=351, num_classes=102)
        logger.info('使用 Baseline 模型 (4×4096 MLP)')
    elif args.model_type == 'm_layer':
        model = RegModel_MLayer(
            input_dim=351,
            num_classes=102,
            matrix_size=args.m_matrix_size,
            scale=args.m_scale,
            clip=args.m_clip
        )
        logger.info(f'使用 M-Layer 模型 (matrix_size={args.m_matrix_size}, '
                   f'scale={args.m_scale}, clip={args.m_clip})')
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    logger.info(f'模型参数量: {count_parameters(model):,}')

    # 检查是否为预测模式（仅由 --predict_only 决定）
    if args.predict_only:
        # 预测模式必须提供 --load_model
        if not args.load_model:
            raise ValueError("预测模式需要指定模型路径 --load_model")

        logger.info(f'加载预训练模型: {args.load_model}')
        checkpoint = torch.load(args.load_model, map_location='cpu')

        # 检查模型类型兼容性
        saved_model_type = checkpoint.get('model_type', 'baseline')
        if saved_model_type != args.model_type:
            raise ValueError(
                f"模型类型不匹配！Checkpoint 中保存的是 '{saved_model_type}'，"
                f"但当前指定的是 '{args.model_type}'。"
                f"请确保使用相同的 --model_type 参数。"
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # 重新创建测试数据集
        test_dataset = TestDataset(test_file_1d, test_file_3d)

        # ===== 关键验证：1D与3D标签一致性自检（预测模式）=====
        logger.info('验证1D与3D标签一致性（预测模式）...')
        mask = test_dataset.region_mask.astype(bool)
        labels_3d = test_dataset.region_labels[mask]
        labels_1d = test_dataset.labels

        if len(labels_1d) != len(labels_3d):
            logger.error(f"标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")
            raise ValueError("1D和3D标签数量不一致")

        labels_match = np.array_equal(labels_1d, labels_3d)
        if labels_match:
            logger.info(f'  ✅ 标签一致性验证通过：{len(labels_1d)}个体素标签完全匹配')
        else:
            n_mismatch = np.sum(labels_1d != labels_3d)
            mismatch_rate = n_mismatch / len(labels_1d) * 100
            logger.error(f'  ❌ 标签一致性验证失败！')
            logger.error(f'     不匹配体素数: {n_mismatch}/{len(labels_1d)} ({mismatch_rate:.2f}%)')
            mismatch_indices = np.where(labels_1d != labels_3d)[0][:10]
            for idx in mismatch_indices:
                logger.error(f'     体素{idx}: 1D标签={labels_1d[idx]}, 3D标签={labels_3d[idx]}')
            raise AssertionError("测试集标签在一维与三维不一致！可能是文件错配")

        logger.info(f'测试样本数: {len(test_dataset)}')
        history = None  # 预测模式不需要训练历史

    else:
        # 训练模式
        # 如果指定了 --load_model，先加载权重作为初始化
        if args.load_model:
            logger.info(f'加载预训练模型作为初始化: {args.load_model}')
            checkpoint = torch.load(args.load_model, map_location='cpu')

            # 检查模型类型兼容性
            saved_model_type = checkpoint.get('model_type', 'baseline')
            if saved_model_type != args.model_type:
                raise ValueError(
                    f"模型类型不匹配！Checkpoint 中保存的是 '{saved_model_type}'，"
                    f"但当前指定的是 '{args.model_type}'。"
                    f"请确保使用相同的 --model_type 参数。"
                )

            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

        # 创建训练器（传入学习率调度相关参数）
        trainer = Trainer(
            model,
            device=device,
            learning_rate=0.00001,
            min_delta_acc=args.min_delta_acc,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr
        )

        # 训练模型
        logger.info('开始训练...')
        logger.info(f'Early stopping patience: {args.patience}')
        logger.info(f'Min delta for improvement: {args.min_delta_acc}')
        logger.info(f'LR scheduler: factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.min_lr}')
        history = trainer.train(train_loader, test_loader, epochs=args.epochs, patience=args.patience)
    
    # 保存模型（仅训练模式）
    if not args.predict_only:
        # 根据模型类型决定文件名
        if args.model_type == 'baseline':
            model_filename = f'dense_4x4096_model_test{args.test_subject}.pth'
        else:
            model_filename = f'm_layer_model_test{args.test_subject}.pth'

        model_path = output_dir / model_filename
        save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler': None,
            'history': history,
            'args': vars(args),
            'model_type': args.model_type,  # 保存模型类型
        }
        # 如果是 M-Layer 模型，保存其配置
        if args.model_type == 'm_layer':
            save_dict['m_layer_config'] = {
                'matrix_size': args.m_matrix_size,
                'scale': args.m_scale,
                'clip': args.m_clip
            }
        torch.save(save_dict, model_path)
        logger.info(f'模型保存至: {model_path}')
        
        # 保存训练历史（转换numpy类型为JSON可序列化格式）
        history_path = output_dir / f'history_test{args.test_subject}.json'
        history_serializable = convert_to_json_serializable(history)
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    # 如果需要，保存3D预测概率
    if args.save_predictions:
        logger.info('生成3D预测概率体积...')
        predictions_3d = predict_and_map_to_3d(model, test_dataset, device, args.batch_size)
        
        # 保存预测 - 使用HDF5格式处理大文件（13GB超过MAT v5的2GB限制）
        pred_path = output_dir / f'predictions_3d_test{args.test_subject}.mat'
        
        logger.info(f'保存预测到: {pred_path}')
        logger.info(f'  形状: {predictions_3d.shape}')
        logger.info(f'  概率范围: [{predictions_3d.min():.4f}, {predictions_3d.max():.4f}]')
        logger.info(f'  预计文件大小: ~{predictions_3d.nbytes / (1024**3):.1f}GB')
        
        # 使用h5py保存（MATLAB v7.3格式）
        import h5py
        with h5py.File(str(pred_path), 'w') as f:
            # 保存概率数据，使用压缩减小文件大小
            f.create_dataset('softmax_probabilities', 
                           data=predictions_3d.astype(np.float32),
                           compression='gzip', 
                           compression_opts=4)  # 中等压缩级别
            
            # 保存元数据
            f.attrs['test_subject'] = args.test_subject
            f.attrs['test_file_1d'] = str(test_file_1d)
            f.attrs['test_file_3d'] = str(test_file_3d)
            f.attrs['shape'] = predictions_3d.shape
            f.attrs['prob_range'] = [float(predictions_3d.min()), float(predictions_3d.max())]
        
        logger.info(f'预测已保存（HDF5/MAT v7.3格式，带压缩）')
    
    # 最终报告
    if args.predict_only:
        logger.info('\n===== 预测完成 =====')
        logger.info(f'测试被试: {args.test_subject}')
        if args.save_predictions:
            logger.info('3D softmax概率已保存')
    else:
        logger.info('\n===== 训练完成 =====')

        # Early stopping信息
        if history.get('early_stopped', False):
            logger.info(f'Early Stopping: 是（在第{history["total_epochs"]}个epoch停止）')
            logger.info(f'最佳模型来自: Epoch {history["best_epoch"]}')
        else:
            logger.info(f'Early Stopping: 否（完成全部{history["total_epochs"]}个epoch）')
            logger.info(f'最佳模型来自: Epoch {history["best_epoch"]}')

        # 最佳指标（按min_delta规则确定的best，与保存的模型一致）
        logger.info(f'最佳测试Gross Accuracy: {history["best_test_acc"]:.4f} (Epoch {history["best_epoch"]})')

        # 最终epoch的指标
        logger.info(f'最终训练Loss: {history["train_loss"][-1]:.4f}')
        logger.info(f'最终训练F1: {history["train_f1"][-1]:.4f}')
        logger.info(f'最终训练Acc: {history["train_acc"][-1]:.4f}')
        logger.info(f'最终测试Loss: {history["test_loss"][-1]:.4f}')
        logger.info(f'最终测试F1: {history["test_f1"][-1]:.4f}')
        logger.info(f'最终测试Acc: {history["test_acc"][-1]:.4f}')

if __name__ == '__main__':
    main()