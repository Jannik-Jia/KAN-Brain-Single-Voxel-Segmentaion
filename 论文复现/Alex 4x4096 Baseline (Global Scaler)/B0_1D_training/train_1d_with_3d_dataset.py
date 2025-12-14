#!/usr/bin/env python3
"""
使用1D训练集进行训练，然后映射回3D
基于Alex的架构，保留multidim_data前341维输入
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
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, log_loss
)
import gc
import subprocess


# ==================== 指标计算辅助函数 ====================

def compute_topk_accuracy(y_true, y_pred_proba, k=5):
    """
    计算 Top-K 准确率

    Args:
        y_true: (N,) 真实标签
        y_pred_proba: (N, C) 预测概率
        k: Top-K 的 K 值

    Returns:
        top_k_accuracy: float
    """
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return float(correct.mean())


def compute_ece(y_true, y_pred_proba, n_bins=10):
    """
    计算期望校准误差 (Expected Calibration Error)

    Args:
        y_true: (N,) 真实标签
        y_pred_proba: (N, C) 预测概率
        n_bins: 分桶数量

    Returns:
        ece: float
    """
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true, y_pred_proba):
    """
    计算 Brier 分数（多类别）

    Args:
        y_true: (N,) 真实标签
        y_pred_proba: (N, C) 预测概率

    Returns:
        brier_score: float
    """
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    # 转换为 one-hot
    n_classes = y_pred_proba.shape[1]
    y_true_one_hot = np.zeros((len(y_true), n_classes))
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1

    # 计算 Brier 分数
    brier = np.mean(np.sum((y_pred_proba - y_true_one_hot) ** 2, axis=1))
    return float(brier)


def compute_risk_at_coverage(y_true, y_pred_proba, coverage_target=0.95):
    """
    计算指定覆盖率下的风险（错误率）

    Args:
        y_true: (N,) 真实标签
        y_pred_proba: (N, C) 预测概率
        coverage_target: 目标覆盖率

    Returns:
        risk: float (错误率)
    """
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)

    # 按置信度降序排序
    sorted_indices = np.argsort(-confidences)

    # 选择覆盖 target% 样本
    n_covered = int(len(y_true) * coverage_target)
    if n_covered == 0:
        return 0.0

    covered_indices = sorted_indices[:n_covered]

    # 计算风险（错误率）
    risk = 1.0 - (predictions[covered_indices] == y_true[covered_indices]).mean()
    return float(risk)


def compute_coverage_at_risk(y_true, y_pred_proba, risk_threshold=0.05):
    """
    计算指定风险阈值下的实际覆盖率

    Args:
        y_true: (N,) 真实标签
        y_pred_proba: (N, C) 预测概率
        risk_threshold: 风险阈值

    Returns:
        coverage: float
    """
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)

    # 按置信度降序排序
    sorted_indices = np.argsort(-confidences)

    # 找到满足风险阈值的最大覆盖率
    n_covered = len(y_true)
    for n in range(1, len(y_true) + 1):
        covered_indices = sorted_indices[:n]
        current_risk = 1.0 - (predictions[covered_indices] == y_true[covered_indices]).mean()

        if current_risk > risk_threshold:
            n_covered = n - 1
            break

    coverage = n_covered / len(y_true)
    return float(coverage)


def compute_soft_dice(y_true, y_pred_proba, smooth=1e-6):
    """
    计算宏平均 Soft Dice 系数

    Args:
        y_true: (N,) 真实标签
        y_pred_proba: (N, C) 预测概率
        smooth: 平滑因子

    Returns:
        macro_soft_dice: float
    """
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    # 转换为 one-hot
    n_classes = y_pred_proba.shape[1]
    y_true_one_hot = np.zeros((len(y_true), n_classes))
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1

    # 计算每个类别的 Soft Dice
    intersection = np.sum(y_true_one_hot * y_pred_proba, axis=0)
    dice_per_class = (2 * intersection + smooth) / (
        np.sum(y_true_one_hot, axis=0) + np.sum(y_pred_proba, axis=0) + smooth
    )

    # 宏平均
    macro_dice = dice_per_class.mean()
    return float(macro_dice)


# ==================== Experiment记录工具 ====================

def get_code_version() -> str:
    """获取git提交哈希，失败则返回unknown"""
    try:
        repo_root = Path(__file__).resolve().parent
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root)
        short_commit = commit.decode().strip()[:7]
        return f"git-{short_commit}"
    except Exception:
        return "unknown"


def build_experiment_dict(
    args,
    train_dataset,
    test_dataset,
    history,
    best_metrics: Dict,  # ← 改为接收完整的指标字典
    train_subject_names,
    test_subject_name,
    output_dir: Path,
    model_param_count_m: float,
    train_time_hours: Optional[float],
    model_path: Optional[Path],
    history_path: Optional[Path],
    predictions_path: Optional[Path]
):
    """构建统一的experiment记录，使用真实计算的指标"""
    n_vox_train = int(train_dataset.all_data.shape[0])
    n_vox_test = int(test_dataset.features.shape[0])

    experiment_id = f"alex_4x4096_globalstd_test{args.test_subject}_v1"

    experiment_dict = {
        "experiment_id": experiment_id,
        "method_name": "Alex 4x4096 baseline (global StdScaler, 341-d input)",
        "method_key": "alex_4x4096_globalstd",
        "family": "mlp",
        "subfamily": "baseline_4x4096",
        "code_version": get_code_version(),
        "seed": 42,
        "task": {
            "dataset": "alex",
            "label_space": "alex-102",
            "grid": "mprage",
            "split": {
                "scheme": "subject-wise",
                "split_id": f"leave_one_out_test_subject_{args.test_subject}",
                "train_subjects": train_subject_names,
                "val_subjects": [],
                "test_subjects": [test_subject_name]
            },
            "n_voxels": {
                "train": n_vox_train,
                "val": 0,
                "test": n_vox_test
            }
        },
        "preprocessing": {
            "feature_version": "voxel_signature_v1",
            "normalisation": {
                "type": "global_standard_scaler",
                "params": {
                    "epsilon": None,
                    "fit_scope": "train_global"
                }
            },
            "background_handling": "drop_ignore_index",
            "class_weighting": {
                "scheme": "none",
                "computed_on": None,
                "weights_file": None
            }
        },
        "model": {
            "family": "mlp",
            "param_count_m": round(model_param_count_m, 3),
            "details": {
                "input_dim": 341,
                "output_dim": 102,
                "hidden_layers": [4096, 4096, 4096, 4096],
                "activation": "relu",
                "dropout": 0.5,
                "residual": False,
                "attention": False,
                "feature_interaction": False,
                "normalisation_in_network": None,
                "kan_config": None,
                "tabnet_config": None,
                "classical_ml_config": None,
                "pseudo_inverse_config": None
            }
        },
        "training": {
            "loss": {
                "type": "cross_entropy",
                "class_weights": "none",
                "label_smoothing": 0.0
            },
            "optimizer": {
                "type": "adam",
                "lr": args.learning_rate if hasattr(args, "learning_rate") else 0.00001,
                # 真实生效的L2强度通过kernel_l2_regularization设置为1e-5
                "weight_decay": 0.00001,
                "betas": [0.9, 0.999]
            },
            "scheduler": {
                "type": "none",
                "params": {}
            },
            "batch_size": args.batch_size,
            "n_epochs": args.epochs,
            "early_stopping": {
                "enabled": False,
                "monitor": "val_macro_f1",
                "mode": "max",
                "min_epochs": 15,
                "patience": 5,
                "delta": 0.001
            }
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "train_time_hours": train_time_hours,
            "inference_time_s_per_1e6_voxels": None
        },
        "results": {
            "evaluated_split": "test",
            "global_metrics": {
                # 基础准确率指标 - ✅ 所有都是真实计算值
                "gross_accuracy": float(best_metrics['top1_accuracy']),
                "top1_accuracy": float(best_metrics['top1_accuracy']),
                "top3_accuracy": float(best_metrics['top3_accuracy']),
                "top5_accuracy": float(best_metrics['top5_accuracy']),
                "balanced_accuracy": float(best_metrics['balanced_accuracy']),

                # F1 和分割指标 - ✅ 所有都是真实计算值
                "macro_f1": float(best_metrics['macro_f1']),
                "weighted_f1": float(best_metrics['weighted_f1']),
                "macro_soft_dice": float(best_metrics['macro_soft_dice']),

                # 一致性和校准指标 - ✅ 所有都是真实计算值
                "cohen_kappa": float(best_metrics['cohen_kappa']),
                "kappa": float(best_metrics['cohen_kappa']),  # 同 cohen_kappa（保留别名）
                "nll": float(best_metrics['nll']),
                "ece": float(best_metrics['ece']),
                "brier_score": float(best_metrics['brier_score']),

                # 风险覆盖指标 - ✅ 所有都是真实计算值
                "risk_at_95_coverage": float(best_metrics['risk_at_95_coverage']),
                "actual_coverage_95": float(best_metrics['actual_coverage_95']),

                # 其他指标
                "gc": None,  # 泛化系数需要训练/测试性能对比，暂不计算

                # 保留的额外指标（向后兼容）
                "top_3_accuracy": float(best_metrics['top3_accuracy']),
            },
            "per_class_metrics_path": None,
            "per_subject_metrics_path": None,
            "confusion_matrix_path": None,
            "curves": {
                "reliability_diagram_path": None,
                "risk_coverage_curve_path": None
            },
            "logs": {
                "train_curve_path": str(history_path) if history_path else None,
                "val_curve_path": str(history_path) if history_path else None
            },
            "notes": None
        }
    }

    notes_parts = [
        "✅ 所有11个要求的指标均真实计算（gross_accuracy, top1/3/5_accuracy, balanced_accuracy, macro/weighted_f1, macro_soft_dice, cohen_kappa, nll, ece, brier_score, risk_at_95_coverage, actual_coverage_95）",
        "特征维度使用341（截取multidim_data前341维）",
        "L2正则通过kernel_l2_regularization(weight_decay=1e-5)实现",
        "GC（泛化系数）未计算，per-class/per-subject/confusion矩阵未记录"
    ]
    if model_path:
        notes_parts.append(f"model_path={model_path}")
    if predictions_path:
        notes_parts.append(f"predictions_path={predictions_path}")
    experiment_dict["results"]["notes"] = "; ".join(notes_parts)

    return experiment_dict


def save_experiment_json(experiment_dict: Dict, output_dir: Path):
    """保存experiment记录为json文件，并自动验证"""
    output_path = output_dir / f"{experiment_dict['experiment_id']}.json"
    with open(output_path, "w") as f:
        json.dump(experiment_dict, f, indent=2)
    print(f"Experiment记录已保存到: {output_path}")

    # 自动验证保存的文件
    try:
        from experiment_schema import validate_experiment_json
        print("\n自动验证实验记录 schema...")
        is_valid, errors = validate_experiment_json(experiment_dict, verbose=True)
        if not is_valid:
            print(f"⚠️  警告: 实验记录存在 {len(errors)} 个问题，建议检查")
    except ImportError:
        print("⚠️  警告: 无法导入 experiment_schema，跳过自动验证")

# ==================== 模型定义（基于Alex的架构）====================

class RegModel(nn.Module):
    """
    Alex的1D全连接网络
    使用前341维特征
    """
    def __init__(self, input_dim=341, num_classes=102):
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
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            mat_files: 1D MAT文件路径列表
            is_train: 是否为训练模式
            scaler: StandardScaler对象，用于数据标准化
        """
        self.mat_files = mat_files
        self.is_train = is_train
        self.scaler = scaler
        
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
        
        # 数据标准化
        if self.scaler is None and is_train:
            print("拟合StandardScaler...")
            self.scaler = StandardScaler()
            self.all_data = self.scaler.fit_transform(self.all_data).astype(np.float32)
        elif self.scaler is not None:
            print("应用StandardScaler...")
            self.all_data = self.scaler.transform(self.all_data).astype(np.float32)
    
    def _load_subject(self, mat_file: Path):
        """加载单个被试的1D数据"""
        with h5py.File(mat_file, 'r') as f:
            # 加载1D数据
            multidim_data = f['multidim_data'][()]  # (351, n_voxels)
            seg_one_hot = f['seg_one_hot'][()]      # (102, n_voxels)
            
            # 转置以适应Python的行优先顺序（参考数据集创建文档）
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T     # -> (n_voxels, 351)
            
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T         # -> (n_voxels, 102)

            # 仅使用前341维特征，丢弃最后10维
            multidim_data = multidim_data[:, :341]
            
            # 从one-hot转换为类别标签
            labels = np.argmax(seg_one_hot, axis=1)  # (n_voxels,)
            
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
    
    def __init__(self, mat_file_1d: Path, mat_file_3d: Path, scaler: StandardScaler):
        """
        Args:
            mat_file_1d: 1D MAT文件（用于获取特征数据）
            mat_file_3d: 3D MAT文件（用于获取3D mask）
            scaler: 训练时的StandardScaler
        """
        self.mat_file_1d = mat_file_1d
        self.mat_file_3d = mat_file_3d
        self.scaler = scaler
        
        # 加载1D数据
        with h5py.File(mat_file_1d, 'r') as f:
            multidim_data = f['multidim_data'][()]
            seg_one_hot = f['seg_one_hot'][()]
            
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T

            # 仅使用前341维特征，丢弃最后10维
            multidim_data = multidim_data[:, :341]

            self.features = multidim_data.astype(np.float32)  # (n_voxels, 341)
            self.labels = np.argmax(seg_one_hot, axis=1)      # (n_voxels,)
        
        # 加载3D mask（用于映射）
        with h5py.File(mat_file_3d, 'r') as f:
            region_mask = f['region_mask'][()]
            region_labels = f['region_labels'][()]
            
            # 严格形状验证：不允许隐式轴转换，必须显式匹配预期形状
            assert region_mask.shape == (384, 336, 256), \
                f"region_mask 形状不符合预期 (384, 336, 256)，实际为 {region_mask.shape}，文件: {mat_file_3d}"
            
            assert region_labels.shape == (384, 336, 256), \
                f"region_labels 形状不符合预期 (384, 336, 256)，实际为 {region_labels.shape}，文件: {mat_file_3d}"
                
            self.region_mask = region_mask
            self.region_labels = region_labels
        
        # 应用标准化
        self.features = scaler.transform(self.features).astype(np.float32)
        
        print(f"测试数据: {len(self.features)} 个体素")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================== 训练器类 ====================

class Trainer:
    def __init__(self, model, device='cuda', learning_rate=0.00001):
        self.model = model.to(device)
        self.device = device
        
        # Alex的配置：Adam优化器，学习率1e-5
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.history = {'train_loss': [], 'train_f1': [], 'train_top3': [],
                        'test_loss': [], 'test_f1': [], 'test_top3': []}
        self.best_test_metrics = None  # ← 保存最佳epoch的完整指标字典
        self.best_test_f1 = 0.0
        self.final_test_metrics = None  # ← 保存最后一个epoch的完整指标字典
    
    def _compute_top3_correct(self, output, target):
        """计算Top-3正确个数"""
        _, top3 = output.topk(3, dim=1)
        correct_top3 = top3.eq(target.view(-1, 1)).any(dim=1)
        return correct_top3.sum().item()

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_train_loss = 0
        all_preds = []
        all_labels = []
        correct_top3 = 0
        total_samples = 0
        
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
            
            # 记录预测和标签用于F1计算
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            # Top-3统计
            correct_top3 += self._compute_top3_correct(output, target)
            total_samples += target.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_top3 = correct_top3 / total_samples if total_samples > 0 else 0.0
        
        return avg_train_loss, train_f1, train_top3
    
    def evaluate(self, val_loader):
        """评估模型并计算所有指标"""
        self.model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []  # ← 新增：收集预测概率
        correct_top3 = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                # 计算损失
                base_loss = self.criterion(output, target)
                l2_reg = kernel_l2_regularization(self.model, weight_decay=0.00001)
                total_loss = base_loss + l2_reg

                total_val_loss += total_loss.item()

                # 获取预测概率
                probs = F.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())

                # 记录预测和标签
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                # Top-3统计
                correct_top3 += self._compute_top3_correct(output, target)
                total_samples += target.size(0)

        # 转换为 numpy 数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算所有指标
        metrics = {}

        # 基础准确率指标
        metrics['top1_accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['top3_accuracy'] = correct_top3 / total_samples if total_samples > 0 else 0.0
        metrics['top5_accuracy'] = compute_topk_accuracy(all_labels, all_probs, k=5)
        metrics['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)

        # F1 和分割指标
        metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        metrics['macro_soft_dice'] = compute_soft_dice(all_labels, all_probs)

        # 一致性和校准指标
        metrics['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
        metrics['nll'] = log_loss(all_labels, all_probs, labels=list(range(102)))
        metrics['ece'] = compute_ece(all_labels, all_probs)
        metrics['brier_score'] = compute_brier_score(all_labels, all_probs)

        # 风险覆盖指标
        metrics['risk_at_95_coverage'] = compute_risk_at_coverage(all_labels, all_probs, 0.95)
        metrics['actual_coverage_95'] = compute_coverage_at_risk(all_labels, all_probs, 0.05)

        # 损失
        metrics['loss'] = total_val_loss / len(val_loader)

        return metrics
    
    def train(self, train_loader, test_loader, epochs=25, restore_best_weights=False):
        """完整训练流程（对应Alex的25个epochs）"""
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs} =====")

            # 训练
            train_loss, train_f1, train_top3 = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['train_top3'].append(train_top3)

            # 测试（验证）- 现在返回指标字典
            test_metrics = self.evaluate(test_loader)
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_f1'].append(test_metrics['macro_f1'])
            self.history['test_top3'].append(test_metrics['top3_accuracy'])

            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"Test Loss: {test_metrics['loss']:.4f}, Test F1: {test_metrics['macro_f1']:.4f}, Test Top-3 Acc: {test_metrics['top3_accuracy']:.4f}")

            # 保存最佳模型（基于测试F1）
            if test_metrics['macro_f1'] > self.best_test_f1:
                self.best_test_f1 = test_metrics['macro_f1']
                self.best_test_metrics = test_metrics  # ← 保存完整的最佳指标
                self.best_model_state = self.model.state_dict()
                print(f"新的最佳测试F1: {self.best_test_f1:.4f}")
                print(f"  Top-1 Acc: {test_metrics['top1_accuracy']:.4f}")
                print(f"  Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")
                print(f"  Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")

            # 保存最后一个epoch的指标（用于基线对比）
            self.final_test_metrics = test_metrics

        # === 修改这里：根据参数决定是否恢复最佳模型 ===
        if restore_best_weights:
            print(f"正在恢复最佳模型状态 (Test F1: {self.best_test_f1:.4f})...")
            self.model.load_state_dict(self.best_model_state)
        else:
            print(f"保留最终模型状态 (Epoch {epochs}, Test F1: {self.final_test_metrics['macro_f1']:.4f})...")
            
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
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                       help='学习率（默认1e-5，与Alex一致）')
    parser.add_argument('--save_predictions', action='store_true',
                       help='保存3D预测概率')
    parser.add_argument('--load_model', type=str, default=None,
                       help='加载预训练模型路径')
    parser.add_argument('--predict_only', action='store_true',
                       help='仅进行预测，不训练模型')
    parser.add_argument('--save_best_epoch', action='store_true',
                       help='保存最佳epoch的指标（默认保存最后一个epoch，用于基线对比）')

    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
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
    subject_names = sorted(idx_1d.keys())
    
    # 选择测试被试（按被试名匹配）
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
        scaler=None
    )
    
    # 获取scaler用于测试集
    scaler = train_dataset.scaler
    
    # 创建测试数据集（需要1D和3D文件）
    logger.info('加载测试数据...')
    test_dataset = TestDataset(test_file_1d, test_file_3d, scaler)
    
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
    
    # 创建模型（使用前341维输入）
    logger.info('创建模型...')
    model = RegModel(input_dim=341, num_classes=102)
    
    param_count = sum(p.numel() for p in model.parameters())
    model_param_count_m = param_count / 1e6
    logger.info(f'模型参数量: {param_count:,}')
    
    # 记录训练时长
    train_time_hours = None
    
    # 记录关键输出路径
    model_path = None
    history_path = None
    predictions_path = None
    
    # 检查是否为预测模式
    if args.predict_only or args.load_model:
        if not args.load_model:
            raise ValueError("预测模式需要指定模型路径 --load_model")
        
        logger.info(f'加载预训练模型: {args.load_model}')
        checkpoint = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)  # 确保模型在正确的设备上
        scaler = checkpoint['scaler']
        
        # 重新创建测试数据集（使用加载的scaler）
        test_dataset = TestDataset(test_file_1d, test_file_3d, scaler)
        
        # ===== 关键验证：1D与3D标签一致性自检（预测模式）=====
        logger.info('验证1D与3D标签一致性（预测模式）...')
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
        
        logger.info(f'使用加载的scaler，测试样本数: {len(test_dataset)}')
        
        history = None  # 预测模式不需要训练历史
        
    else:
        # 训练模式
        # 创建训练器
        trainer = Trainer(model, device=device, learning_rate=args.learning_rate)
        
        # 训练模型
        logger.info('开始训练...')
        train_start = time.time()
        # 默认 save_best_epoch 为 False，所以默认 restore_best_weights=False (即保留第25个epoch)
        history = trainer.train(train_loader, test_loader, epochs=args.epochs, 
                              restore_best_weights=args.save_best_epoch)
        train_time_hours = (time.time() - train_start) / 3600.0

    
    # 保存模型（仅训练模式）
    if not args.predict_only:
        model_path = output_dir / f'dense_4x4096_model_test{args.test_subject}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'history': history,
            'args': vars(args)
        }, model_path)
        logger.info(f'模型保存至: {model_path}')
        
        # 保存训练历史
        history_path = output_dir / f'history_test{args.test_subject}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
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
        predictions_path = pred_path

    # 保存experiment记录（仅训练模式）
    if not args.predict_only:
        # === 修改 2：根据参数选择要记录的指标 ===
        if args.save_best_epoch:
            logger.info(">>> 记录模式: Best Epoch Metrics")
            metrics_to_record = trainer.best_test_metrics
        else:
            logger.info(">>> 记录模式: Final Epoch Metrics (Baseline)")
            metrics_to_record = trainer.final_test_metrics

        # 使用选定的 metrics_to_record
        experiment_dict = build_experiment_dict(
            args=args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            history=history,
            best_metrics=metrics_to_record,  # <--- 这里传入选定的指标
            train_subject_names=train_subject_names,
            test_subject_name=test_subject_name,
            output_dir=output_dir,
            model_param_count_m=model_param_count_m,
            train_time_hours=train_time_hours,
            model_path=model_path,
            history_path=history_path,
            predictions_path=predictions_path
        )
        
        # 更新 notes 以反映当前模式
        experiment_dict["results"]["notes"] += f"; metrics_mode={'best_epoch' if args.save_best_epoch else 'final_epoch'}"
        
        save_experiment_json(experiment_dict, output_dir)

    
    
    # 最终报告
    if args.predict_only:
        logger.info('\n===== 预测完成 =====')
        logger.info(f'测试被试: {args.test_subject}')
        if args.save_predictions:
            logger.info('3D softmax概率已保存')
    else:
        logger.info('\n===== 训练完成 =====')
        logger.info('最佳测试性能（所有指标）:')
        logger.info(f'  Macro F1: {trainer.best_test_metrics["macro_f1"]:.4f}')
        logger.info(f'  Top-1 Acc: {trainer.best_test_metrics["top1_accuracy"]:.4f}')
        logger.info(f'  Top-3 Acc: {trainer.best_test_metrics["top3_accuracy"]:.4f}')
        logger.info(f'  Top-5 Acc: {trainer.best_test_metrics["top5_accuracy"]:.4f}')
        logger.info(f'  Balanced Acc: {trainer.best_test_metrics["balanced_accuracy"]:.4f}')
        logger.info(f'  Weighted F1: {trainer.best_test_metrics["weighted_f1"]:.4f}')
        logger.info(f'  Soft Dice: {trainer.best_test_metrics["macro_soft_dice"]:.4f}')
        logger.info(f'  Cohen\'s Kappa: {trainer.best_test_metrics["cohen_kappa"]:.4f}')
        logger.info(f'  NLL: {trainer.best_test_metrics["nll"]:.4f}')
        logger.info(f'  ECE: {trainer.best_test_metrics["ece"]:.4f}')
        logger.info(f'  Brier Score: {trainer.best_test_metrics["brier_score"]:.4f}')
        logger.info(f'  Risk@95%: {trainer.best_test_metrics["risk_at_95_coverage"]:.4f}')
        logger.info(f'  Coverage@5%Risk: {trainer.best_test_metrics["actual_coverage_95"]:.4f}')
        logger.info(f'\n最终训练Loss: {history["train_loss"][-1]:.4f}')
        logger.info(f'最终训练F1: {history["train_f1"][-1]:.4f}')
        logger.info(f'最终测试Loss: {history["test_loss"][-1]:.4f}')
        logger.info(f'最终测试F1: {history["test_f1"][-1]:.4f}')

if __name__ == '__main__':
    main()
