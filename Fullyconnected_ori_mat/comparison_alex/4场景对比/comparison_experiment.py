#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版本：PyTorch脑体素分类比较脚本
解决验证集中背景类导致的NaN损失问题
"""

import os
import gc
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
MAT_FILE_PATH = "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
OUTPUT_DIR = "./comparison_results_fixed"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
NUM_EPOCHS_DEMO = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
L2_REG_FIXEDMLP = 1e-5

NUM_ACTUAL_CLASSES = 101
NUM_TOTAL_CLASSES_WITH_BG = 102

# 确保结果可重现
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"--- 修复版脚本配置 ---")
print(f"Device: {DEVICE}")
print(f"MAT File Path: {MAT_FILE_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Epochs: {NUM_EPOCHS_DEMO}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"---------------------------")

# --- 数据加载工具 ---
def load_mat_data_from_h5(mat_file_path):
    print(f"Loading MAT data from: {mat_file_path}")
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MAT file not found at {mat_file_path}")
    arrays = {}
    with h5py.File(mat_file_path, 'r') as f:
        for k, v in f.items():
            arrays[k] = np.array(v)
    print("MAT data loaded.")
    return arrays

def check_and_transpose_data_fn(arrays):
    print("Checking and transposing data...")
    data_key, region_key, prob_idx_key = 'data', 'region', 'prob_idx'
    expected_feature_dim = 341

    # 检查键是否存在
    for key_orig, key_var_name in [('data', 'data_key'), ('region', 'region_key'), ('prob_idx', 'prob_idx_key')]:
        if eval(key_var_name) not in arrays:
            alt_key = None
            if key_orig == 'data' and 'TRAIN_DATA' in arrays: 
                alt_key = 'TRAIN_DATA'
            elif key_orig == 'region' and 'TRAIN_REGION' in arrays: 
                alt_key = 'TRAIN_REGION'
            if alt_key:
                print(f"Using alternative key '{alt_key}' for '{eval(key_var_name)}'.")
                globals()[key_var_name] = alt_key
            else:
                raise KeyError(f"Key '{eval(key_var_name)}' not found. Available: {list(arrays.keys())}")
    
    # 转置数据
    data_arr = arrays[data_key]
    if data_arr.shape[0] == expected_feature_dim and data_arr.ndim == 2:
        data_transposed = data_arr.T
    elif data_arr.shape[1] == expected_feature_dim and data_arr.ndim == 2:
        data_transposed = data_arr
    else:
        raise ValueError(f"Unexpected shape for '{data_key}': {data_arr.shape}")

    region_arr = arrays[region_key]
    if region_arr.shape[0] == NUM_TOTAL_CLASSES_WITH_BG and region_arr.ndim == 2:
        region_transposed = region_arr.T
    elif region_arr.shape[1] == NUM_TOTAL_CLASSES_WITH_BG and region_arr.ndim == 2:
        region_transposed = region_arr
    else:
        raise ValueError(f"Unexpected shape for '{region_key}': {region_arr.shape}")
    
    prob_idx_transposed = arrays[prob_idx_key].flatten()

    # 验证数据一致性
    assert data_transposed.shape[0] == region_transposed.shape[0] == prob_idx_transposed.shape[0], "Sample count mismatch"
    assert data_transposed.shape[1] == expected_feature_dim, f"Feature count mismatch"
    assert region_transposed.shape[1] == NUM_TOTAL_CLASSES_WITH_BG, f"Class count mismatch"
    
    print(f"Data shapes: data {data_transposed.shape}, region {region_transposed.shape}, prob_idx {prob_idx_transposed.shape}")
    return data_transposed, region_transposed, prob_idx_transposed

def analyze_class_distribution(region_data, prob_idx, split_patient_id=38):
    """分析类别分布，特别是背景类"""
    print("--- 类别分布分析 ---")
    
    # 获取原始标签
    raw_labels = np.argmax(region_data, axis=1)
    
    # 训练集和验证集分布
    train_mask = prob_idx != split_patient_id
    val_mask = prob_idx == split_patient_id
    
    train_labels = raw_labels[train_mask]
    val_labels = raw_labels[val_mask]
    
    # 统计背景类比例
    train_bg_ratio = np.sum(train_labels == 0) / len(train_labels)
    val_bg_ratio = np.sum(val_labels == 0) / len(val_labels)
    
    print(f"训练集背景类比例: {train_bg_ratio:.4f} ({np.sum(train_labels == 0)}/{len(train_labels)})")
    print(f"验证集背景类比例: {val_bg_ratio:.4f} ({np.sum(val_labels == 0)}/{len(val_labels)})")
    
    # 统计非背景类
    train_non_bg = np.sum(train_labels > 0)
    val_non_bg = np.sum(val_labels > 0)
    
    print(f"训练集非背景样本: {train_non_bg}")
    print(f"验证集非背景样本: {val_non_bg}")
    
    return {
        'train_bg_ratio': train_bg_ratio,
        'val_bg_ratio': val_bg_ratio,
        'train_non_bg': train_non_bg,
        'val_non_bg': val_non_bg
    }

def split_data_fn(data_len, prob_idx, random_state=RANDOM_STATE):
    print("Splitting data by patient ID...")
    all_indices = np.arange(data_len)
    val_mask = (prob_idx == 38)
    val_indices = all_indices[val_mask]
    train_test_pool_indices = all_indices[~val_mask]

    if len(train_test_pool_indices) > 0:
        train_indices, test_indices = train_test_split(
            train_test_pool_indices, test_size=0.01, random_state=random_state, shuffle=True
        )
    else:
        train_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)

    print(f"Total: {data_len}, Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_indices, val_indices, test_indices

def create_scaler_fn(all_data_np, train_indices_np):
    print("Creating StandardScaler...")
    scaler = StandardScaler()
    if len(train_indices_np) == 0:
        print("Warning: Empty training set for scaler")
        scaler.mean_ = np.zeros(all_data_np.shape[1])
        scaler.scale_ = np.ones(all_data_np.shape[1])
        scaler.var_ = np.ones(all_data_np.shape[1])
        scaler.n_samples_seen_ = 0
        return scaler
    
    print(f"Fitting scaler on {len(train_indices_np)} training samples")
    scaler.fit(all_data_np[train_indices_np])
    print("StandardScaler fitted")
    return scaler

# --- 修复的PyTorch数据集 ---
class BrainVoxelPyTorchDataset(Dataset):
    def __init__(self, all_features_np, all_one_hot_labels_np, indices_np, scaler_obj, 
                 process_background=True, filter_background_in_val=False):
        """
        filter_background_in_val: 如果为True，在验证集中过滤掉背景类样本
        这可以避免验证时出现全是ignore_index的批次
        """
        self.process_background = process_background
        self.filter_background_in_val = filter_background_in_val
        
        if len(indices_np) == 0:
            print(f"Dataset Warning: Empty indices")
            self.features_scaled_tensor = torch.empty(0, all_features_np.shape[1], dtype=torch.float32)
            self.processed_labels_tensor = torch.empty(0, dtype=torch.long)
            self.valid_indices = np.array([], dtype=int)
            return

        # 获取子集数据
        features_subset_np = all_features_np[indices_np]
        one_hot_labels_subset_np = all_one_hot_labels_np[indices_np]
        
        # 标准化特征
        if hasattr(scaler_obj, 'mean_') and scaler_obj.mean_ is not None:
            features_scaled_np = scaler_obj.transform(features_subset_np)
        else:
            print("Dataset Warning: Scaler not fitted, using raw features")
            features_scaled_np = features_subset_np

        # 处理标签
        indexed_raw_labels_np = np.argmax(one_hot_labels_subset_np, axis=1)
        
        if process_background:
            processed_labels_np = indexed_raw_labels_np - 1
            processed_labels_np[indexed_raw_labels_np == 0] = -1
            
            # 如果要过滤背景类（主要用于验证集）
            if filter_background_in_val:
                non_bg_mask = processed_labels_np != -1
                if np.sum(non_bg_mask) > 0:
                    features_scaled_np = features_scaled_np[non_bg_mask]
                    processed_labels_np = processed_labels_np[non_bg_mask]
                    self.valid_indices = indices_np[non_bg_mask]
                    print(f"Dataset: Filtered out {np.sum(~non_bg_mask)} background samples, kept {len(processed_labels_np)}")
                else:
                    print("Dataset Warning: All samples are background after filtering")
                    self.features_scaled_tensor = torch.empty(0, features_scaled_np.shape[1], dtype=torch.float32)
                    self.processed_labels_tensor = torch.empty(0, dtype=torch.long)
                    self.valid_indices = np.array([], dtype=int)
                    return
            else:
                self.valid_indices = indices_np
        else:
            processed_labels_np = indexed_raw_labels_np
            self.valid_indices = indices_np

        self.features_scaled_tensor = torch.FloatTensor(features_scaled_np)
        self.processed_labels_tensor = torch.LongTensor(processed_labels_np)

    def __len__(self):
        return len(self.processed_labels_tensor)

    def __getitem__(self, idx):
        return self.features_scaled_tensor[idx], self.processed_labels_tensor[idx]

# --- 模型定义 ---
class FixedMLP(nn.Module):
    def __init__(self, input_dim=341, num_output_classes=101, dropout_rate=0.5, l2_reg=L2_REG_FIXEDMLP):
        super(FixedMLP, self).__init__()
        self.l2_reg = l2_reg
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, num_output_classes)
        )
        # Xavier初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_l2_loss(self):
        l2_loss = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, 2)**2
        return self.l2_reg * l2_loss

# --- 类权重计算 ---
def calculate_weights_fn_robust(labels_for_weights_tensor, num_effective_classes, 
                               process_background_flag, smoothing_epsilon=1e-9):
    print(f"Calculating class weights for {num_effective_classes} classes (BG processed: {process_background_flag})")
    
    if labels_for_weights_tensor.numel() == 0:
        print("Warning: No labels for weight calculation")
        return torch.ones(num_effective_classes, dtype=torch.float32).to(DEVICE)

    labels_np = labels_for_weights_tensor.cpu().numpy()
    
    if process_background_flag:
        # 只考虑非背景类
        actual_class_labels_np = labels_np[labels_np >= 0]
        if len(actual_class_labels_np) == 0:
            print("Warning: No valid non-background labels")
            return torch.ones(num_effective_classes, dtype=torch.float32).to(DEVICE)
        counts_np = np.bincount(actual_class_labels_np, minlength=num_effective_classes)
    else:
        counts_np = np.bincount(labels_np, minlength=num_effective_classes)
    
    # 计算权重：inverse frequency
    weights_np = 1.0 / (counts_np + smoothing_epsilon)
    weights_np[counts_np == 0] = 1.0  # 未见过的类默认权重为1
    
    print(f"  Sample class counts (first 10): {counts_np[:min(10, len(counts_np))]}")
    print(f"  Sample weights (first 10): {weights_np[:min(10, len(weights_np))]:.4f}")
    
    return torch.FloatTensor(weights_np).to(DEVICE)

# --- 训练和评估函数 ---
def train_epoch_fn(model, dataloader, criterion, optimizer, device, add_l2_loss_flag=True):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_preds_list, all_true_list = [], []
    
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0

    for features, labels in dataloader:
        if features.numel() == 0:
            continue
            
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        
        classification_loss = criterion(outputs, labels)
        final_loss = classification_loss
        
        if add_l2_loss_flag and hasattr(model, 'get_l2_loss'):
            final_loss += model.get_l2_loss()
        
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"NaN/Inf loss in training! Skipping batch. Loss: {classification_loss.item()}")
            continue
            
        final_loss.backward()
        optimizer.step()
        
        running_loss += final_loss.item() * features.size(0)
        total_samples += features.size(0)
        
        # 计算指标
        active_mask = (labels != criterion.ignore_index) if criterion.ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
        if active_mask.sum() > 0:
            all_preds_list.extend(torch.argmax(outputs[active_mask], dim=1).cpu().numpy())
            all_true_list.extend(labels[active_mask].cpu().numpy())

    if total_samples == 0:
        return 0.0, 0.0, 0.0
        
    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_true_list, all_preds_list) if len(all_true_list) > 0 else 0.0
    
    # 计算F1分数
    if len(all_true_list) > 0:
        unique_labels = np.unique(all_true_list + all_preds_list)
        if criterion.ignore_index is not None:
            unique_labels = unique_labels[unique_labels != criterion.ignore_index]
        epoch_f1 = f1_score(all_true_list, all_preds_list, labels=unique_labels if len(unique_labels) > 0 else None, 
                           average='macro', zero_division=0)
    else:
        epoch_f1 = 0.0
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate_model_fn(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds_list, all_true_list = [], []
    
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            if features.numel() == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss in validation batch {batch_idx}! Loss: {loss.item()}")
                # 检查是否所有标签都是ignore_index
                if criterion.ignore_index is not None and torch.all(labels == criterion.ignore_index):
                    print("  All labels in this batch are ignore_index, skipping...")
                    continue
                else:
                    running_loss = float('nan')
                    break
            
            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            
            # 计算指标
            active_mask = (labels != criterion.ignore_index) if criterion.ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
            if active_mask.sum() > 0:
                all_preds_list.extend(torch.argmax(outputs[active_mask], dim=1).cpu().numpy())
                all_true_list.extend(labels[active_mask].cpu().numpy())

    if total_samples == 0 or np.isnan(running_loss):
        return float('nan') if np.isnan(running_loss) else 0.0, 0.0, 0.0
        
    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_true_list, all_preds_list) if len(all_true_list) > 0 else 0.0
    
    # 计算F1分数
    if len(all_true_list) > 0:
        unique_labels = np.unique(all_true_list + all_preds_list)
        if criterion.ignore_index is not None:
            unique_labels = unique_labels[unique_labels != criterion.ignore_index]
        epoch_f1 = f1_score(all_true_list, all_preds_list, labels=unique_labels if len(unique_labels) > 0 else None,
                           average='macro', zero_division=0)
    else:
        epoch_f1 = 0.0
    
    return epoch_loss, epoch_acc, epoch_f1

# --- 场景运行函数 ---
def run_scenario(scenario_name, process_bg, use_weights,
                 all_features_np, all_labels_one_hot_np,
                 train_indices_np, val_indices_np, scaler_obj,
                 num_epochs, batch_size, device, learning_rate,
                 filter_val_bg=True):  # 新参数：是否在验证集中过滤背景类
    
    print(f"\n--- Running {scenario_name} ---")
    print(f"Process Background: {process_bg}, Use Weights: {use_weights}, Filter Val BG: {filter_val_bg}")

    num_model_output_classes = NUM_ACTUAL_CLASSES if process_bg else NUM_TOTAL_CLASSES_WITH_BG
    
    # 创建数据集
    train_dataset = BrainVoxelPyTorchDataset(
        all_features_np, all_labels_one_hot_np, train_indices_np, scaler_obj, 
        process_background=process_bg, filter_background_in_val=False
    )
    
    # 验证集：如果处理背景且要过滤，则过滤掉背景类
    val_dataset = BrainVoxelPyTorchDataset(
        all_features_np, all_labels_one_hot_np, val_indices_np, scaler_obj, 
        process_background=process_bg, filter_background_in_val=(process_bg and filter_val_bg)
    )

    if len(train_dataset) == 0:
        print(f"SKIPPING {scenario_name}: Empty training dataset")
        empty_hist = {k: [] for k in ['train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']}
        return {'val_acc': 0.0, 'val_f1': 0.0, 'history': empty_hist}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型
    model = FixedMLP(num_output_classes=num_model_output_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 类权重
    class_weights_tensor = None
    if use_weights:
        class_weights_tensor = calculate_weights_fn_robust(
            train_dataset.processed_labels_tensor,
            num_model_output_classes,
            process_background_flag=process_bg
        )
    
    # 损失函数
    if process_bg:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=-1).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

    print(f"  Model output classes: {num_model_output_classes}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Criterion ignore_index: {criterion.ignore_index}")

    # 训练历史
    history = {'train_loss':[], 'train_acc':[], 'train_f1':[], 'val_loss':[], 'val_acc':[], 'val_f1':[]}
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_f1 = train_epoch_fn(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_model_fn(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - start_time
        print(f"{scenario_name} - Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) -> "
              f"Train L: {train_loss:.4f}, A: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Val L: {val_loss:.4f}, A: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if np.isnan(val_loss):
            print(f"Stopping {scenario_name} due to NaN validation loss")
            break

    final_val_acc = history['val_acc'][-1] if history['val_acc'] and not np.isnan(history['val_acc'][-1]) else 0.0
    final_val_f1 = history['val_f1'][-1] if history['val_f1'] and not np.isnan(history['val_f1'][-1]) else 0.0
    
    # 清理内存
    del model, train_dataset, val_dataset, train_loader, val_loader, criterion, optimizer
    if class_weights_tensor is not None:
        del class_weights_tensor
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {'val_acc': final_val_acc, 'val_f1': final_val_f1, 'history': history}

# --- 主实验函数 ---
def main_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- 修复版主实验开始 ---")
    
    try:
        # 加载数据
        mat_arrays = load_mat_data_from_h5(MAT_FILE_PATH)
        all_features_raw_np, all_labels_one_hot_raw_np, prob_idx_raw_np = check_and_transpose_data_fn(mat_arrays)
        
        # 分析类别分布
        class_dist = analyze_class_distribution(all_labels_one_hot_raw_np, prob_idx_raw_np)
        
        # 数据划分
        train_idx_np, val_idx_np, test_idx_np = split_data_fn(len(all_features_raw_np), prob_idx_raw_np)
        
        # 创建标准化器
        scaler_obj = create_scaler_fn(all_features_raw_np, train_idx_np)
        print(f"Scaler mean (first 5): {scaler_obj.mean_[:5] if hasattr(scaler_obj, 'mean_') and scaler_obj.mean_ is not None else 'Not fitted'}")
        
    except Exception as e:
        print(f"ERROR during data loading: {e}")
        return

    results_summary = {}
    
    # 定义四个场景
    scenarios_to_run = [
        {"name": "S1 (WithBG, WithW)", "process_bg": True, "use_weights": True},
        {"name": "S2 (NoBG, WithW)", "process_bg": False, "use_weights": True},
        {"name": "S3 (WithBG, NoW)", "process_bg": True, "use_weights": False},
        {"name": "S4 (NoBG, NoW)", "process_bg": False, "use_weights": False},
    ]

    # 运行每个场景
    for sc_config in scenarios_to_run:
        if len(train_idx_np) == 0:
            print(f"SKIPPING {sc_config['name']} due to empty training set")
            empty_hist = {k: [] for k in ['train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']}
            results_summary[sc_config['name']] = {'val_acc': 0.0, 'val_f1': 0.0, 'history': empty_hist}
            continue

        sc_results = run_scenario(
            sc_config["name"], sc_config["process_bg"], sc_config["use_weights"],
            all_features_raw_np, all_labels_one_hot_raw_np,
            train_idx_np, val_idx_np, scaler_obj,
            NUM_EPOCHS_DEMO, BATCH_SIZE, DEVICE, LEARNING_RATE,
            filter_val_bg=True  # 对于处理背景的场景，过滤验证集中的背景类
        )
        results_summary[sc_config["name"]] = sc_results

    # 保存结果摘要
    summary_file_path = os.path.join(OUTPUT_DIR, "final_results_summary_fixed.txt")
    print(f"\n--- 最终结果摘要 (保存至 {summary_file_path}) ---")
    with open(summary_file_path, "w") as f:
        header = f"{'场景':<40} | {'最终验证准确率':<20} | {'最终验证宏F1':<20}\n"
        separator = "-" * 85 + "\n"
        print(header.strip())
        f.write(header)
        print(separator.strip())
        f.write(separator)
        
        for scenario, metrics in results_summary.items():
            val_acc_display = f"{metrics['val_acc']:.4f}" if isinstance(metrics['val_acc'], float) else str(metrics['val_acc'])
            val_f1_display = f"{metrics['val_f1']:.4f}" if isinstance(metrics['val_f1'], float) else str(metrics['val_f1'])
            line = f"{scenario:<40} | {val_acc_display:<20} | {val_f1_display:<20}\n"
            print(line.strip())
            f.write(line)

    # 绘制训练曲线对比图
    if results_summary:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        for i, metric_key in enumerate(['val_f1', 'val_acc']):
            ax = axes[i]
            metric_name = "宏F1分数" if metric_key == 'val_f1' else "准确率"
            
            for scenario_name, result_data in results_summary.items():
                history = result_data.get('history', {})
                if history and history.get(metric_key):
                    metric_values = np.array(history[metric_key])
                    epochs_run = np.arange(1, len(metric_values) + 1)
                    
                    # 只绘制非NaN值
                    valid_indices = ~np.isnan(metric_values)
                    if np.any(valid_indices):
                        marker = 'o' if i == 0 else 's'
                        ax.plot(epochs_run[valid_indices], metric_values[valid_indices], 
                               marker=marker, linestyle='-', label=f"{scenario_name}", markersize=4)
                        
            ax.set_title(f'验证{metric_name}对比 (总轮数: {NUM_EPOCHS_DEMO})')
            ax.set_xlabel('训练轮数')
            ax.set_ylabel(metric_name)
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.arange(1, NUM_EPOCHS_DEMO + 1, max(1, NUM_EPOCHS_DEMO // 10)))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_file_path = os.path.join(OUTPUT_DIR, "validation_curves_comparison_fixed.png")
        plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"验证曲线对比图保存至: {plot_file_path}")
        
        # 绘制训练损失对比
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for scenario_name, result_data in results_summary.items():
            history = result_data.get('history', {})
            if history and history.get('train_loss'):
                loss_values = np.array(history['train_loss'])
                epochs_run = np.arange(1, len(loss_values) + 1)
                valid_indices = ~np.isnan(loss_values)
                if np.any(valid_indices):
                    ax.plot(epochs_run[valid_indices], loss_values[valid_indices], 
                           marker='x', linestyle='-', label=f"{scenario_name}", markersize=3)
        
        ax.set_title(f'训练损失对比 (总轮数: {NUM_EPOCHS_DEMO})')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('训练损失')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 使用对数刻度更好地显示损失
        
        plt.tight_layout()
        loss_plot_path = os.path.join(OUTPUT_DIR, "training_loss_comparison_fixed.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"训练损失对比图保存至: {loss_plot_path}")
        
    else:
        print("没有结果可以绘制")

    print("\n--- 实验分析建议 ---")
    print("1. 如果S1和S3仍然出现NaN，可能需要进一步调整验证策略")
    print("2. 背景类占验证集的比例过高，可能需要重新考虑验证策略")
    print("3. 可以尝试使用加权采样或分层验证来处理类别不平衡")
    print("4. 如果S2和S4表现良好，说明不处理背景类可能是更好的选择")

if __name__ == "__main__":
    main_experiment()
    print("修复版脑体素分类对比实验完成。")