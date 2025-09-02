#!/usr/bin/env python
# coding: utf-8

"""
Balanced MRI数据训练脚本 - 带3D softmax预测保存功能
- 直接从NIfTI文件加载
- 保存测试集的3D softmax volume用于可视化
- 正确处理去除背景后的空间映射
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import nibabel as nib
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, confusion_matrix, 
    classification_report, balanced_accuracy_score
)

# 检查是否能导入可视化工具包
try:
    from visualization_toolkit.per_class_analyzer import (
        calculate_per_class_metrics_detailed,
        create_comprehensive_visualizations,
        save_detailed_results
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Warning: visualization_toolkit not found. Per-class analysis will be skipped.")

# ... (前面的代码保持不变，这里省略) ...
# Setup logging, set_seed, device, STANDARD_LABELS, create_label_mapping, RegModel 等函数

def setup_logging(log_file='balanced_training.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STANDARD_LABELS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 103
]

def create_label_mapping():
    forward_mapping = {original: continuous for continuous, original in enumerate(STANDARD_LABELS)}
    reverse_mapping = {continuous: original for continuous, original in enumerate(STANDARD_LABELS)}
    return forward_mapping, reverse_mapping

class RegModel(nn.Module):
    """Deep fully connected network for MRI voxel classification (Alex identical structure)"""
    
    def __init__(self, input_dim=42, num_classes=52, hidden_dim=4096, 
                 num_hidden_layers=4, dropout_rate=0.5):
        super(RegModel, self).__init__()
        
        # 完全对应Alex的结构：无BatchNorm，使用单独的层定义
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 严格按照Alex的前向传播结构
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # 输出层无激活函数
        return x

def load_and_process_subject_with_mask(subject_dir, include_background=True, exclude_features=None, 
                                      forward_mapping=None):
    """
    加载并处理单个受试者数据，同时保存空间位置掩码
    
    Returns:
    --------
    dict : 包含以下关键字段
        - 'features': 处理后的特征
        - 'labels': 处理后的标签  
        - 'original_shape_3d': 原始3D形状
        - 'spatial_mask': 3D布尔掩码，标记哪些体素被包含在训练中
        - 'flat_indices': 展平后被保留的体素在原始展平数组中的索引
    """
    subject_dir = Path(subject_dir)
    balanced_dir = subject_dir / "balanced_output"
    
    data_4d_path = balanced_dir / "balanced_data_4d10000.nii.gz"
    label_3d_path = balanced_dir / "balanced_labels_3d10000.nii.gz"
    
    if not data_4d_path.exists() or not label_3d_path.exists():
        raise FileNotFoundError(f"找不到balanced数据文件：{subject_dir.name}")
    
    # 加载NIfTI数据
    img_4d = nib.load(data_4d_path)
    label_3d = nib.load(label_3d_path)
    
    data_4d = img_4d.get_fdata().astype(np.float32)
    labels_3d = label_3d.get_fdata().astype(np.int32)
    
    original_shape_3d = labels_3d.shape
    logger.info(f"  {subject_dir.name}: 4D{data_4d.shape}, 3D{labels_3d.shape}")
    
    # 展平数据（使用C顺序）
    n_voxels = np.prod(labels_3d.shape)
    n_modalities = data_4d.shape[3]
    
    features = data_4d.reshape(n_voxels, n_modalities)  # 使用默认C order
    labels_flat = labels_3d.flatten()  # 使用默认C order
    
    # 验证数据对应
    assert features.shape[0] == labels_flat.shape[0], f"特征和标签数量不匹配"
    
    # 标签映射
    if forward_mapping is not None:
        logger.info(f"  映射标签...")
        unique_orig = np.unique(labels_flat)
        labels_mapped = np.zeros_like(labels_flat)
        
        for orig_label in unique_orig:
            if orig_label in forward_mapping:
                mask = labels_flat == orig_label
                labels_mapped[mask] = forward_mapping[orig_label]
                count = np.sum(mask)
                if orig_label != 0:  # 不显示背景映射信息
                    logger.info(f"    {orig_label:3d} → {forward_mapping[orig_label]:2d} ({count:,} 体素)")
        
        labels_flat = labels_mapped
    
    # 🔑 关键：创建空间掩码
    if not include_background:
        # 排除背景体素（标签=0）
        spatial_mask_flat = labels_flat != 0  # 1D掩码
        spatial_mask_3d = spatial_mask_flat.reshape(original_shape_3d)  # 3D掩码 - 使用默认C order
        
        # 保留的体素索引
        flat_indices = np.where(spatial_mask_flat)[0]  # 在原始展平数组中的索引
        
        # 应用掩码
        features = features[spatial_mask_flat]
        labels_flat = labels_flat[spatial_mask_flat]
        
        logger.info(f"  排除背景后: {len(features):,} / {n_voxels:,} 体素 ({len(features)/n_voxels*100:.2f}%)")
    else:
        # 包含所有体素
        spatial_mask_3d = np.ones(original_shape_3d, dtype=bool)
        flat_indices = np.arange(n_voxels)  # 所有索引
        logger.info(f"  包含背景: {len(features):,} 体素")
    
    # 排除指定特征
    if exclude_features:
        keep_mask = np.ones(features.shape[1], dtype=bool)
        keep_mask[exclude_features] = False
        features = features[:, keep_mask]
        logger.info(f"  排除特征{exclude_features}后: {features.shape[1]} 个特征")
    
    return {
        'features': features,
        'labels': labels_flat,
        'subject_id': subject_dir.name,
        'n_voxels': len(features),
        'original_shape_3d': original_shape_3d,  # 🔑 原始3D形状
        'spatial_mask_3d': spatial_mask_3d,      # 🔑 3D空间掩码
        'flat_indices': flat_indices,            # 🔑 展平索引
        'affine': img_4d.affine,                 # 🔑 仿射矩阵（用于保存NIfTI）
        'header': img_4d.header                  # 🔑 NIfTI头信息
    }

def predictions_to_3d_volume(predictions, spatial_info, include_background=True, 
                            background_class=0):
    """
    将预测结果还原为3D softmax volume
    
    Parameters:
    -----------
    predictions : np.ndarray
        模型预测的softmax概率，形状 (n_valid_voxels, n_classes)
    spatial_info : dict
        包含空间信息的字典
    include_background : bool
        训练时是否包含了背景
    background_class : int
        背景类别索引
        
    Returns:
    --------
    volume_3d : np.ndarray
        3D softmax volume，形状 (X, Y, Z, n_classes)
    """
    original_shape_3d = spatial_info['original_shape_3d']
    spatial_mask_3d = spatial_info['spatial_mask_3d']
    flat_indices = spatial_info['flat_indices']
    
    n_classes = predictions.shape[1]
    
    # 创建完整的3D softmax volume
    volume_3d = np.zeros((*original_shape_3d, n_classes), dtype=np.float32)
    
    if include_background:
        # 如果训练时包含了背景，直接映射回去
        volume_flat = volume_3d.reshape(-1, n_classes)  # 展平到 (total_voxels, n_classes) - 使用默认C order
        volume_flat[flat_indices] = predictions  # 直接赋值
        
    else:
        # 如果训练时排除了背景，需要特殊处理
        # 1. 非背景区域：使用预测结果
        volume_flat = volume_3d.reshape(-1, n_classes)  # 使用默认C order
        volume_flat[flat_indices] = predictions
        
        # 2. 背景区域：设置为背景类概率=1，其他类概率=0
        background_indices = np.where(~spatial_mask_3d.flatten())[0]  # 同样使用默认C order
        volume_flat[background_indices, background_class] = 1.0  # 背景类概率=1
        # 其他类概率已经是0（初始化时）
    
    logger.info(f"  3D softmax volume形状: {volume_3d.shape}")
    logger.info(f"  概率总和检查: min={np.sum(volume_3d, axis=-1).min():.6f}, "
                f"max={np.sum(volume_3d, axis=-1).max():.6f}")
    
    return volume_3d

def predict_and_save_3d_softmax(model, test_data_info, device, output_dir, 
                                include_background=True, batch_size=8192):
    """
    预测测试集并保存3D softmax volume
    
    Parameters:
    -----------
    test_data_info : dict
        测试集数据信息（来自load_and_process_subject_with_mask）
    """
    logger.info(f"\n🔮 预测测试集并保存3D softmax volume...")
    
    test_features = test_data_info['features']
    test_labels = test_data_info['labels']
    subject_id = test_data_info['subject_id']
    
    model.eval()
    all_predictions = []
    
    # 批量预测
    n_samples = len(test_features)
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, batch_size), desc="预测中"):
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_X = torch.FloatTensor(test_features[start_idx:end_idx]).to(device)
            
            # 前向传播得到logits
            logits = model(batch_X)
            
            # 转换为softmax概率
            probabilities = F.softmax(logits, dim=1)
            
            all_predictions.append(probabilities.cpu().numpy())
    
    # 合并所有预测结果
    predictions = np.vstack(all_predictions)  # (n_valid_voxels, n_classes)
    
    logger.info(f"  预测完成: {predictions.shape}")
    logger.info(f"  概率范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
    
    # 还原为3D volume
    volume_3d = predictions_to_3d_volume(
        predictions, test_data_info, 
        include_background=include_background
    )
    
    # 保存为NIfTI文件
    bg_str = 'incl' if include_background else 'excl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 使用原始的仿射矩阵和头信息
    affine = test_data_info['affine']
    header = test_data_info['header'].copy()
    
    # 更新头信息（4D数据）
    header.set_data_shape(volume_3d.shape)
    header.set_data_dtype(np.float32)
    
    # 创建NIfTI图像
    softmax_img = nib.Nifti1Image(volume_3d, affine, header)
    
    # 保存文件
    softmax_path = output_dir / f"test_softmax_3d_{subject_id}_bg_{bg_str}_{timestamp}.nii.gz"
    nib.save(softmax_img, softmax_path)
    
    logger.info(f"💾 3D softmax已保存: {softmax_path}")
    logger.info(f"  文件大小: {softmax_path.stat().st_size / (1024**2):.2f} MB")
    
    # 额外保存一些有用信息
    info_dict = {
        'subject_id': subject_id,
        'original_shape_3d': test_data_info['original_shape_3d'],
        'softmax_shape': volume_3d.shape,
        'include_background': include_background,
        'n_valid_voxels': len(test_features),
        'n_classes': predictions.shape[1],
        'timestamp': timestamp,
        'class_names': STANDARD_LABELS,
        'prediction_stats': {
            'min_prob': float(predictions.min()),
            'max_prob': float(predictions.max()),
            'mean_prob': float(predictions.mean())
        }
    }
    
    info_path = output_dir / f"test_softmax_info_{subject_id}_bg_{bg_str}_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=2)
    
    logger.info(f"💾 预测信息已保存: {info_path}")
    
    return volume_3d, softmax_path

def patient_wise_standardization(all_data):
    """Patient-wise标准化"""
    scalers_info = {}
    
    for data in all_data:
        subject_id = data['subject_id']
        features = data['features']
        
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1.0, std)
        
        features_std = (features - mean) / std
        data['features'] = features_std.astype(np.float32)
        
        scalers_info[subject_id] = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        
        logger.info(f"  {subject_id}: 标准化完成")
    
    return all_data, scalers_info

def load_balanced_dataset_with_spatial_info(root_dir, include_background=True, exclude_features=None):
    """加载数据并保存空间信息"""
    root_dir = Path(root_dir)
    forward_mapping, reverse_mapping = create_label_mapping()
    
    logger.info(f"标签映射: {len(STANDARD_LABELS)}个类别 → 0-51连续值")
    
    # 查找受试者
    subject_dirs = sorted([d for d in root_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("FOR_")])
    
    valid_subjects = []
    for subject_dir in subject_dirs:
        balanced_dir = subject_dir / "balanced_output"
        data_file = balanced_dir / "balanced_data_4d10000.nii.gz"
        label_file = balanced_dir / "balanced_labels_3d10000.nii.gz"
        if data_file.exists() and label_file.exists():
            valid_subjects.append(subject_dir)
    
    logger.info(f"找到 {len(valid_subjects)} 个有balanced数据的受试者")
    logger.info(f"包含背景: {'是' if include_background else '否'}")
    
    # 加载所有数据（保存空间信息）
    all_data = []
    
    for subject_dir in tqdm(valid_subjects, desc="加载数据"):
        try:
            data = load_and_process_subject_with_mask(
                subject_dir, 
                include_background=include_background,
                exclude_features=exclude_features,
                forward_mapping=forward_mapping
            )
            all_data.append(data)
            
        except Exception as e:
            logger.error(f"加载 {subject_dir.name} 失败: {e}")
            continue
    
    # Patient-wise标准化
    logger.info(f"\n🔄 Patient-wise标准化...")
    all_data, scalers_info = patient_wise_standardization(all_data)
    
    return all_data, scalers_info

# 训练相关函数（calculate_metrics, train_epoch, evaluate等保持不变）
def calculate_metrics(y_true, y_pred, loss):
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

def plot_training_history(history, save_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy  
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # F1 Score
    ax3.plot(epochs, history['train_f1'], 'b-', label='Train F1')
    ax3.plot(epochs, history['test_f1'], 'r-', label='Test F1')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro F1')
    ax3.set_title('Macro F1 Score')
    ax3.legend()
    ax3.grid(True)
    
    # Test metrics
    ax4.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, history['test_f1'], 'orange', label='Test F1')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Loss', color='g')
    ax4_twin.set_ylabel('Test F1', color='orange')
    ax4.set_title('Test Performance')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model(config):
    """主训练函数 - 集成3D softmax保存功能"""
    logger.info("=" * 80)
    logger.info("Balanced MRI数据训练 - 带3D softmax保存")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据（保存空间信息）
    logger.info("\n📂 加载balanced数据...")
    all_data, scalers_info = load_balanced_dataset_with_spatial_info(
        config['root_dir'], 
        include_background=config['include_background'],
        exclude_features=config.get('exclude_features')
    )
    
    # train/test分割
    test_data_info = all_data[0]  # 第一个做测试
    train_data_list = all_data[1:]  # 其他做训练
    
    logger.info(f"\n数据分割:")
    logger.info(f"  测试集: {test_data_info['subject_id']} ({test_data_info['n_voxels']:,} 体素)")
    logger.info(f"  训练集: {len(train_data_list)} 个受试者")
    
    # 合并训练数据
    train_features = np.vstack([data['features'] for data in train_data_list])
    train_labels = np.hstack([data['labels'] for data in train_data_list])
    
    test_features = test_data_info['features']
    test_labels = test_data_info['labels']
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # 创建模型
    actual_input_dim = train_features.shape[1]
    model = RegModel(
        input_dim=actual_input_dim,
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    logger.info(f"\n🧠 模型架构 (Alex identical): 输入{actual_input_dim} → 隐藏{config['hidden_dim']} × 4层 → 输出{config['num_classes']}")
    logger.info(f"  网络结构: FC1({actual_input_dim}→4096) → FC2-FC4(4096→4096) → FC5(4096→{config['num_classes']})")
    logger.info(f"  激活函数: ReLU + Dropout({config['dropout_rate']})，无BatchNorm")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': []
    }
    
    best_test_f1 = 0
    best_epoch = 0
    best_model_state = None
    
    logger.info(f"\n🚀 开始训练 ({config['num_epochs']} epochs)...")
    start_time = time.time()
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 评估
        test_metrics = evaluate(model, test_features, test_labels, criterion, device, 
                               batch_size=config.get('test_batch_size', 8192))
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['macro_f1'])
        
        # 保存最佳模型
        if test_metrics['macro_f1'] > best_test_f1:
            best_test_f1 = test_metrics['macro_f1']
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        # 打印进度
        logger.info(f"\nEpoch [{epoch+1}/{config['num_epochs']}] "
                   f"Time: {time.time()-epoch_start:.2f}s")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['macro_f1']:.4f}")
        logger.info(f"  Test  - Loss: {test_metrics['loss']:.4f}, "
                   f"Acc: {test_metrics['accuracy']:.4f}, "
                   f"F1: {test_metrics['macro_f1']:.4f}")
        
        torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    logger.info(f"\n✅ 训练完成! 总耗时: {training_time:.2f}秒")
    logger.info(f"最佳测试F1: {best_test_f1:.4f} (Epoch {best_epoch+1})")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 保存结果
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    # 🔑 关键：预测并保存3D softmax
    volume_3d, softmax_path = predict_and_save_3d_softmax(
        model, test_data_info, device, output_dir,
        include_background=config['include_background'],
        batch_size=config.get('test_batch_size', 8192)
    )
    
    # 保存模型和其他结果
    bg_str = 'incl' if config['include_background'] else 'excl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = output_dir / f"balanced_3d_bg_{bg_str}_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'best_epoch': best_epoch,
        'scalers_info': scalers_info,
        'test_subject_id': test_data_info['subject_id'],
        'softmax_path': str(softmax_path)  # 保存softmax文件路径
    }, model_path)
    
    # 绘制训练历史
    plot_path = output_dir / f"training_history_3d_bg_{bg_str}_{timestamp}.png"
    plot_training_history(history, plot_path)
    
    logger.info(f"💾 模型已保存: {model_path}")
    logger.info(f"💾 训练图表已保存: {plot_path}")
    
    # 立即进行Per-Class性能分析 (在训练完成后，模型仍在内存中)
    if VISUALIZATION_AVAILABLE:
        logger.info("="*80)
        logger.info("🎯 开始Per-Class性能量化分析...")
        logger.info("="*80)
        
        try:
            # 使用现有的测试数据进行分析，无需重新加载模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            
            # 准备测试数据
            X_test = test_data_info['features']  
            y_test = test_data_info['labels']
            
            # 确保数据格式正确
            if not isinstance(X_test, np.ndarray):
                X_test = np.array(X_test)
            if not isinstance(y_test, np.ndarray):
                y_test = np.array(y_test)
            
            logger.info(f"📊 对 {X_test.shape[0]:,} 个测试样本进行预测...")
            logger.info(f"  特征维度: {X_test.shape}")
            logger.info(f"  标签唯一值数量: {len(np.unique(y_test))}")
            
            # 批量预测 - 优化内存使用
            batch_size = min(8192, len(X_test))  # 动态调整batch size
            all_predictions = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(X_test), batch_size), desc="预测进度"):
                    batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                    batch_pred = model(batch_X)
                    batch_pred_labels = torch.argmax(batch_pred, dim=1).cpu().numpy()
                    all_predictions.extend(batch_pred_labels)
                    
                    # 清理GPU内存
                    del batch_X, batch_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            predictions_array = np.array(all_predictions)
            
            # 验证预测结果
            logger.info(f"✅ 预测完成!")
            logger.info(f"  预测标签范围: {predictions_array.min()} ~ {predictions_array.max()}")
            logger.info(f"  预测中的唯一标签数: {len(np.unique(predictions_array))}")
            logger.info(f"  真实标签范围: {y_test.min()} ~ {y_test.max()}")
            logger.info(f"  真实标签中的唯一值数: {len(np.unique(y_test))}")
            
            logger.info(f"🧮 开始计算per-class指标...")
            
            # 计算per-class指标
            per_class_analysis_dir = output_dir / f"per_class_analysis_bg_{bg_str}_{timestamp}"
            per_class_analysis_dir.mkdir(exist_ok=True)
            
            # 使用现有的数据信息
            volume_info = {
                'subject_id': test_data_info.get('subject_id', 'unknown'),
                'bg_mode': 'included' if config['include_background'] else 'excluded',
                'timestamp': timestamp,
                'total_voxels': X_test.shape[0],
                'n_classes': len(np.unique(y_test))
            }
            
            # 计算详细的per-class指标
            # 注意：需要传递标签映射信息以正确显示FreeSurfer标签ID
            forward_mapping, reverse_mapping = create_label_mapping()
            
            metrics_dict = calculate_per_class_metrics_detailed(
                y_test, predictions_array, 
                volume_info=volume_info,
                reverse_mapping=reverse_mapping  # 传递反向映射以显示原始标签
            )
            
            # 生成可视化
            create_comprehensive_visualizations(metrics_dict, per_class_analysis_dir)
            
            # 保存详细结果
            save_detailed_results(metrics_dict, per_class_analysis_dir)
            
            # 输出关键统计信息到日志
            logger.info("="*60)
            logger.info("📊 Per-Class分析完成！关键统计:")
            logger.info("="*60)
            
            overall = metrics_dict['overall_metrics']
            logger.info(f"总体素数量: {overall['total_samples']:,}")
            logger.info(f"分析的类别数: {overall['total_classes_analyzed']}")
            
            # 显示前10个表现最好的类别
            per_class_data = metrics_dict['per_class_metrics']
            sorted_classes = sorted(per_class_data, key=lambda x: x['f1_score'], reverse=True)
            
            logger.info("\n🏆 F1 Score Top 10:")
            for i, class_info in enumerate(sorted_classes[:10]):
                logger.info(f"  {i+1:2d}. FreeSurfer Label {class_info['class_id']:3d}: F1={class_info['f1_score']:.4f}, "
                          f"Precision={class_info['precision']:.4f}, Recall={class_info['recall']:.4f}")
            
            # 统计性能分布
            excellent = len([x for x in per_class_data if x['f1_score'] >= 0.8])
            good = len([x for x in per_class_data if 0.6 <= x['f1_score'] < 0.8])
            fair = len([x for x in per_class_data if 0.3 <= x['f1_score'] < 0.6])
            poor = len([x for x in per_class_data if 0.05 <= x['f1_score'] < 0.3])
            failed = len([x for x in per_class_data if x['f1_score'] < 0.05])
            
            logger.info(f"\n📈 性能分布:")
            logger.info(f"  优秀 (F1≥0.8): {excellent} 类别")
            logger.info(f"  良好 (0.6≤F1<0.8): {good} 类别") 
            logger.info(f"  一般 (0.3≤F1<0.6): {fair} 类别")
            logger.info(f"  较差 (0.05≤F1<0.3): {poor} 类别")
            logger.info(f"  失败 (F1<0.05): {failed} 类别")
            
            logger.info(f"\n💾 Per-class分析结果已保存到: {per_class_analysis_dir}")
            logger.info(f"🎯 重点查看:")
            logger.info(f"  • comprehensive_per_class_analysis.png - 完整的逐class性能对比")
            logger.info(f"  • per_class_detailed_metrics.csv - 详细量化数据")
            logger.info(f"  • per_class_summary_report.txt - 文字分析报告")
            
        except Exception as e:
            logger.error(f"❌ Per-class分析失败: {str(e)}")
            logger.error("训练结果已保存，但per-class分析未完成")
    else:
        logger.info("⚠️ 跳过per-class分析 (visualization_toolkit未找到)")
    
    return {
        'model': model,
        'history': history,
        'config': config,
        'softmax_path': softmax_path
    }

def main():
    parser = argparse.ArgumentParser(description='训练Balanced数据并保存3D softmax')
    parser.add_argument('--include_background', action='store_true', 
                       help='包含背景体素（默认排除）')
    parser.add_argument('--exclude_features', type=int, nargs='*', default=[14],
                       help='要排除的特征索引（默认[14]）')
    parser.add_argument('--epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8192, help='批大小')
    parser.add_argument('--lr', type=float, default=0.00001, help='学习率')
    
    args = parser.parse_args()
    
    # 设置日志
    bg_str = 'incl' if args.include_background else 'excl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"balanced_3d_bg_{bg_str}_{timestamp}.log"
    global logger
    logger = setup_logging(log_file)
    
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 配置
    config = {
        'root_dir': '/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS',
        'include_background': args.include_background,
        'exclude_features': args.exclude_features if args.exclude_features else None,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'test_batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 0.00001,
        'num_classes': 52,
        'hidden_dim': 4096,
        'num_hidden_layers': 4,
        'dropout_rate': 0.5
    }
    
    # 训练模型
    result = train_model(config)
    
    logger.info("\n" + "=" * 80)
    logger.info("训练完成!")
    logger.info("=" * 80)
    
    logger.info(f"\n💡 3D softmax已保存到: {result['softmax_path']}")
    logger.info("现在你可以用它进行可视化了!")
    
    logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()