#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存友好的batch加载训练脚本
完全模仿原始brain_voxel.ipynb的方式，避免内存溢出
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, balanced_accuracy_score
import time
import sys
import gc

# 导入你现有的MAT加载函数
sys.path.append('.')
from data.mat_loader import load_mat_data

class MemoryEfficientDataLoader:
    """
    内存高效的数据加载器，模仿原始notebook的方式
    """
    def __init__(self, data_indices, labels, scaler, mat_arrays, batch_size=128, shuffle=True):
        """
        参数:
            data_indices: 数据索引数组
            labels: 对应的标签
            scaler: 标准化器
            mat_arrays: 原始MAT数据字典
            batch_size: 批大小
            shuffle: 是否打乱
        """
        self.data_indices = data_indices
        self.labels = labels
        self.scaler = scaler
        self.mat_arrays = mat_arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 获取原始数据引用（不复制，节省内存）
        self.raw_data = mat_arrays['data'].transpose() if 'data' in mat_arrays else mat_arrays['data'].T
        
        print(f"DataLoader创建:")
        print(f"  数据索引数量: {len(data_indices)}")
        print(f"  标签形状: {labels.shape}")
        print(f"  批大小: {batch_size}")
        print(f"  原始数据形状: {self.raw_data.shape}")
    
    def __len__(self):
        return (len(self.data_indices) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(len(self.data_indices))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            
            # 获取当前批次的数据索引
            current_data_indices = self.data_indices[batch_indices]
            current_labels = self.labels[batch_indices]
            
            # 从原始数据中提取当前批次（这里才加载到内存）
            batch_data = self.raw_data[current_data_indices, :]
            
            # 应用标准化
            batch_data_scaled = self.scaler.transform(batch_data)
            
            # 转换为torch张量
            batch_data_tensor = torch.FloatTensor(batch_data_scaled)
            batch_labels_tensor = torch.FloatTensor(current_labels) if len(current_labels.shape) > 1 else torch.LongTensor(current_labels)
            
            yield batch_data_tensor, batch_labels_tensor

class MemoryEfficientIndexDataLoader:
    """
    用于索引标签的内存高效数据加载器
    """
    def __init__(self, data_indices, labels, scaler, mat_arrays, batch_size=128, shuffle=True):
        self.data_indices = data_indices
        self.scaler = scaler
        self.mat_arrays = mat_arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 获取原始数据引用
        self.raw_data = mat_arrays['data'].transpose() if 'data' in mat_arrays else mat_arrays['data'].T
        
        # 处理标签：one-hot -> 索引
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            index_labels = np.argmax(labels, axis=1)
        else:
            index_labels = labels.astype(int)
        
        # 背景标签处理：0->-1, 1-102->0-101
        background_mask = index_labels == 0
        index_labels = index_labels - 1
        index_labels[background_mask] = -1
        
        self.labels = index_labels
        
        print(f"索引标签DataLoader创建:")
        print(f"  数据索引数量: {len(data_indices)}")
        print(f"  标签范围: {np.min(self.labels)} - {np.max(self.labels)}")
        print(f"  背景标签(-1)数量: {np.sum(self.labels == -1)}")
        print(f"  有效标签数量: {np.sum(self.labels >= 0)}")
    
    def __len__(self):
        return (len(self.data_indices) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(len(self.data_indices))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            
            # 获取当前批次的数据索引和标签
            current_data_indices = self.data_indices[batch_indices]
            current_labels = self.labels[batch_indices]
            
            # 从原始数据中提取当前批次
            batch_data = self.raw_data[current_data_indices, :]
            
            # 应用标准化
            batch_data_scaled = self.scaler.transform(batch_data)
            
            # 转换为torch张量
            batch_data_tensor = torch.FloatTensor(batch_data_scaled)
            batch_labels_tensor = torch.LongTensor(current_labels)
            
            yield batch_data_tensor, batch_labels_tensor

def load_data_memory_efficient(mat_file_path, random_state=42):
    """
    内存高效的数据加载，只保存索引和标准化器
    """
    print("=== 内存高效数据加载 ===")
    
    # 加载MAT文件（只读一次）
    print(f"加载MAT文件: {mat_file_path}")
    arrays = load_mat_data(mat_file_path)
    
    # 获取基本信息（不复制大数组）
    data_shape = arrays['data'].transpose().shape if 'data' in arrays else arrays['data'].T.shape
    region_shape = arrays['region'].transpose().shape if 'region' in arrays else arrays['region'].T.shape
    prob_idx = arrays['prob_idx'].transpose().flatten() if 'prob_idx' in arrays else arrays['prob_idx'].flatten()
    
    print(f"数据形状:")
    print(f"  data: {data_shape}")
    print(f"  region: {region_shape}")
    print(f"  prob_idx: {prob_idx.shape}")
    
    # 按患者分割（只保存索引）
    train_patient_indices = np.where(prob_idx != 38)[0]  # 患者1-37
    val_patient_indices = np.where(prob_idx == 38)[0]    # 患者38
    
    print(f"患者分割:")
    print(f"  患者1-37样本数: {len(train_patient_indices)}")
    print(f"  患者38样本数: {len(val_patient_indices)}")
    
    # 从患者1-37中分出1%作为测试集
    train_indices, test_indices = train_test_split(
        train_patient_indices, test_size=0.01, random_state=42
    )
    
    print(f"最终分割:")
    print(f"  训练集索引数: {len(train_indices)}")
    print(f"  测试集索引数: {len(test_indices)}")
    print(f"  验证集索引数: {len(val_patient_indices)}")
    
    # 创建标准化器（使用全部训练集拟合）
    print("创建标准化器（使用全部训练集）...")
    print(f"训练集占比: {len(train_indices) / data_shape[0] * 100:.1f}% 的总数据")
    print(f"训练集样本数: {len(train_indices):,}")
    
    scaler = StandardScaler()
    
    # 使用全部训练集拟合scaler，但分批处理避免内存溢出
    raw_data = arrays['data'].transpose() if 'data' in arrays else arrays['data'].T
    
    # 分批拟合标准化器
    batch_size_for_fitting = 10000  # 每次处理1万个样本
    print(f"分批拟合StandardScaler，每批 {batch_size_for_fitting:,} 个样本...")
    
    # 第一步：计算均值
    running_sum = None
    total_samples = 0
    
    for i in range(0, len(train_indices), batch_size_for_fitting):
        end_idx = min(i + batch_size_for_fitting, len(train_indices))
        batch_indices = train_indices[i:end_idx]
        batch_data = raw_data[batch_indices, :]
        
        if running_sum is None:
            running_sum = np.sum(batch_data, axis=0)
        else:
            running_sum += np.sum(batch_data, axis=0)
        
        total_samples += len(batch_data)
        
        if (i // batch_size_for_fitting + 1) % 10 == 0:
            print(f"  已处理 {i + len(batch_indices):,}/{len(train_indices):,} 个训练样本")
        
        # 清理内存
        del batch_data
        gc.collect()
    
    mean = running_sum / total_samples
    print(f"均值计算完成，处理了 {total_samples:,} 个样本")
    
    # 第二步：计算方差
    running_sum_sq_diff = None
    total_samples_check = 0
    
    print("计算标准差...")
    for i in range(0, len(train_indices), batch_size_for_fitting):
        end_idx = min(i + batch_size_for_fitting, len(train_indices))
        batch_indices = train_indices[i:end_idx]
        batch_data = raw_data[batch_indices, :]
        
        # 计算每个样本与均值的差的平方
        diff_sq = (batch_data - mean) ** 2
        
        if running_sum_sq_diff is None:
            running_sum_sq_diff = np.sum(diff_sq, axis=0)
        else:
            running_sum_sq_diff += np.sum(diff_sq, axis=0)
        
        total_samples_check += len(batch_data)
        
        if (i // batch_size_for_fitting + 1) % 10 == 0:
            print(f"  已处理 {i + len(batch_indices):,}/{len(train_indices):,} 个训练样本")
        
        # 清理内存
        del batch_data, diff_sq
        gc.collect()
    
    # 计算标准差
    variance = running_sum_sq_diff / (total_samples_check - 1)  # 使用样本方差
    std = np.sqrt(variance)
    
    print(f"标准差计算完成，验证总样本数: {total_samples_check:,}")
    
    # 手动设置StandardScaler的参数
    scaler.mean_ = mean
    scaler.var_ = variance
    scaler.scale_ = std
    scaler.n_samples_seen_ = total_samples
    
    # 验证scaler是否正确设置
    print(f"StandardScaler设置完成:")
    print(f"  特征均值范围: [{np.min(mean):.6f}, {np.max(mean):.6f}]")
    print(f"  特征标准差范围: [{np.min(std):.6f}, {np.max(std):.6f}]")
    print(f"  总处理样本数: {total_samples:,}")
    
    # 清理临时变量
    del running_sum, running_sum_sq_diff, mean, std, variance
    gc.collect()
    
    print("标准化器创建完成")
    
    # 提取标签数据（相对较小，可以保存在内存中）
    region_data = arrays['region'].transpose() if 'region' in arrays else arrays['region'].T
    train_labels = region_data[train_indices, :]
    test_labels = region_data[test_indices, :]
    val_labels = region_data[val_patient_indices, :]
    
    print(f"标签数据形状:")
    print(f"  训练标签: {train_labels.shape}")
    print(f"  测试标签: {test_labels.shape}")
    print(f"  验证标签: {val_labels.shape}")
    
    return {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'val_indices': val_patient_indices,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'val_labels': val_labels,
        'scaler': scaler,
        'mat_arrays': arrays,  # 保存原始数据引用，按需加载
        'feature_dim': data_shape[1],
        'num_classes': region_shape[1] if len(region_shape) > 1 else int(np.max(region_data)) + 1
    }

class MemoryEfficientMLP(nn.Module):
    """
    内存高效的MLP模型
    """
    def __init__(self, input_dim=341, num_classes=102, dropout_rate=0.5, l2_reg=1e-5, use_softmax=True):
        super(MemoryEfficientMLP, self).__init__()
        
        self.use_softmax = use_softmax
        
        layers = [
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, num_classes)
        ]
        
        if use_softmax:
            layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)
        
        # L2正则化
        self.l2_reg = l2_reg
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_l2_loss(self):
        l2_loss = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                l2_loss += torch.sum(layer.weight ** 2)
        return self.l2_reg * l2_loss

def calculate_class_weights_from_labels(labels, num_classes=102):
    """
    从标签计算类别权重
    """
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # one-hot标签
        class_counts = np.sum(labels, axis=0)
    else:
        # 索引标签
        valid_labels = labels[labels >= 0]  # 排除背景标签-1
        class_counts = np.bincount(valid_labels, minlength=num_classes)
    
    total_samples = np.sum(class_counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total_samples / (num_classes * class_counts[i])
    
    print(f"类别权重计算:")
    print(f"  权重范围: {np.min(weights[weights > 0]):.4f} - {np.max(weights):.4f}")
    print(f"  零权重类别数: {np.sum(weights == 0)}")
    
    return torch.FloatTensor(weights)

def train_memory_efficient_model(data_dict, device='cuda:0', save_path='./memory_efficient_results', 
                                use_index_labels=True):
    """
    内存高效的模型训练
    """
    print(f"\n=== 开始内存高效训练 ===")
    print(f"标签格式: {'索引标签' if use_index_labels else 'One-hot标签'}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # 训练配置
    batch_size = 128
    num_epochs = 25
    learning_rate = 1e-5
    num_classes = 102
    
    print(f"训练配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  device: {device}")
    
    # 创建数据加载器
    if use_index_labels:
        train_loader = MemoryEfficientIndexDataLoader(
            data_dict['train_indices'], data_dict['train_labels'], 
            data_dict['scaler'], data_dict['mat_arrays'], batch_size, shuffle=True
        )
        val_loader = MemoryEfficientIndexDataLoader(
            data_dict['val_indices'], data_dict['val_labels'], 
            data_dict['scaler'], data_dict['mat_arrays'], batch_size, shuffle=False
        )
        
        # 计算类别权重
        class_weights = calculate_class_weights_from_labels(
            train_loader.labels, num_classes
        ).to(device)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        
    else:
        train_loader = MemoryEfficientDataLoader(
            data_dict['train_indices'], data_dict['train_labels'], 
            data_dict['scaler'], data_dict['mat_arrays'], batch_size, shuffle=True
        )
        val_loader = MemoryEfficientDataLoader(
            data_dict['val_indices'], data_dict['val_labels'], 
            data_dict['scaler'], data_dict['mat_arrays'], batch_size, shuffle=False
        )
        
        # 损失函数（one-hot标签）
        criterion = nn.CrossEntropyLoss()
    
    print(f"数据加载器创建完成:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 创建模型
    model = MemoryEfficientMLP(
        input_dim=data_dict['feature_dim'],
        num_classes=num_classes,
        dropout_rate=0.5,
        l2_reg=1e-5,
        use_softmax=not use_index_labels  # 索引标签时不用softmax
    )
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练记录
    train_losses = []
    train_accuracies = []
    train_f1_scores = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_kappa_scores = []
    
    print(f"\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        all_train_preds = []
        all_train_true = []
        num_train_batches = 0
        
        print(f"Epoch {epoch+1}/{num_epochs} - 训练阶段...")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 处理不同标签格式的损失计算
            if use_index_labels:
                classification_loss = criterion(output, target)
                # 预测和真实标签提取（用于指标计算）
                pred = torch.argmax(output, dim=1)
                valid_mask = target != -1
                if valid_mask.sum() > 0:
                    all_train_preds.extend(pred[valid_mask].cpu().numpy())
                    all_train_true.extend(target[valid_mask].cpu().numpy())
            else:
                classification_loss = criterion(output, target)
                # one-hot标签的处理
                pred = torch.argmax(output, dim=1)
                true = torch.argmax(target, dim=1)
                all_train_preds.extend(pred.cpu().numpy())
                all_train_true.extend(true.cpu().numpy())
            
            # 添加L2正则化
            l2_loss = model.get_l2_loss()
            total_loss = classification_loss + l2_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            num_train_batches += 1
            
            # 内存清理
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # 计算训练指标
        avg_train_loss = epoch_train_loss / num_train_batches
        if len(all_train_true) > 0:
            train_acc = accuracy_score(all_train_true, all_train_preds)
            train_f1 = f1_score(all_train_true, all_train_preds, average='macro')
        else:
            train_acc = train_f1 = 0.0
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)
        
        # 验证阶段
        print(f"Epoch {epoch+1}/{num_epochs} - 验证阶段...")
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_true = []
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 损失计算
                if use_index_labels:
                    classification_loss = criterion(output, target)
                    pred = torch.argmax(output, dim=1)
                    valid_mask = target != -1
                    if valid_mask.sum() > 0:
                        all_val_preds.extend(pred[valid_mask].cpu().numpy())
                        all_val_true.extend(target[valid_mask].cpu().numpy())
                else:
                    classification_loss = criterion(output, target)
                    pred = torch.argmax(output, dim=1)
                    true = torch.argmax(target, dim=1)
                    all_val_preds.extend(pred.cpu().numpy())
                    all_val_true.extend(true.cpu().numpy())
                
                l2_loss = model.get_l2_loss()
                total_loss = classification_loss + l2_loss
                epoch_val_loss += total_loss.item()
                num_val_batches += 1
                
                # 内存清理
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
        
        # 计算验证指标
        avg_val_loss = epoch_val_loss / num_val_batches
        if len(all_val_true) > 0:
            val_acc = accuracy_score(all_val_true, all_val_preds)
            val_f1 = f1_score(all_val_true, all_val_preds, average='macro')
            val_kappa = cohen_kappa_score(all_val_true, all_val_preds)
        else:
            val_acc = val_f1 = val_kappa = 0.0
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_kappa_scores.append(val_kappa)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Kappa: {val_kappa:.4f}")
        
        # 保存最佳模型
        if epoch == 0 or val_f1 > max(val_f1_scores[:-1]):
            model_name = 'best_index_model.pth' if use_index_labels else 'best_onehot_model.pth'
            best_model_path = os.path.join(save_path, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_kappa': val_kappa,
                'scaler': data_dict['scaler'],
                'use_index_labels': use_index_labels
            }, best_model_path)
            print(f"  -> 保存最佳模型 (F1: {val_f1:.4f})")
        
        # 内存清理
        gc.collect()
        torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
    
    # 最终结果
    best_f1 = max(val_f1_scores)
    best_epoch = val_f1_scores.index(best_f1) + 1
    best_acc = val_accuracies[val_f1_scores.index(best_f1)]
    best_kappa = val_kappa_scores[val_f1_scores.index(best_f1)]
    
    print(f"\n=== 最终结果 ===")
    print(f"最佳验证F1分数: {best_f1:.4f} (第{best_epoch}轮)")
    print(f"对应验证准确率: {best_acc:.4f}")
    print(f"对应Kappa系数: {best_kappa:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'train_f1_scores': train_f1_scores,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'val_kappa_scores': val_kappa_scores,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_acc': best_acc,
        'best_kappa': best_kappa,
        'training_time': training_time
    }

def main():
    """主函数"""
    print("=== 内存高效的训练脚本 ===")
    
    mat_file_path = "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    save_path = './memory_efficient_results'
    
    print(f"配置:")
    print(f"  MAT文件: {mat_file_path}")
    print(f"  设备: {device}")
    print(f"  保存路径: {save_path}")
    
    if not os.path.exists(mat_file_path):
        print(f"错误: MAT文件不存在: {mat_file_path}")
        return
    
    try:
        # 1. 内存高效数据加载
        print("\n" + "="*50)
        print("步骤1: 内存高效数据加载")
        print("="*50)
        data_dict = load_data_memory_efficient(mat_file_path)
        
        # 2. 训练索引标签版本
        print("\n" + "="*50)
        print("步骤2: 训练索引标签版本")
        print("="*50)
        index_results = train_memory_efficient_model(
            data_dict, device, save_path + '_index', use_index_labels=True
        )
        
        # 3. 训练one-hot标签版本
        print("\n" + "="*50)
        print("步骤3: 训练One-hot标签版本")
        print("="*50)
        onehot_results = train_memory_efficient_model(
            data_dict, device, save_path + '_onehot', use_index_labels=False
        )
        
        # 4. 对比结果
        print("\n" + "="*50)
        print("步骤4: 对比分析")
        print("="*50)
        print(f"索引标签版本:")
        print(f"  最佳F1: {index_results['best_f1']:.4f}")
        print(f"  最佳准确率: {index_results['best_acc']:.4f}")
        print(f"  最佳Kappa: {index_results['best_kappa']:.4f}")
        print(f"  训练时间: {index_results['training_time']:.2f}秒")
        
        print(f"\nOne-hot标签版本:")
        print(f"  最佳F1: {onehot_results['best_f1']:.4f}")
        print(f"  最佳准确率: {onehot_results['best_acc']:.4f}")
        print(f"  最佳Kappa: {onehot_results['best_kappa']:.4f}")
        print(f"  训练时间: {onehot_results['training_time']:.2f}秒")
        
        # 计算改进
        f1_improvement = index_results['best_f1'] - onehot_results['best_f1']
        print(f"\n性能对比:")
        print(f"  F1分数差异: {f1_improvement:+.4f}")
        
        if f1_improvement > 0.02:
            print(f"  -> 索引标签版本显著更好!")
        elif f1_improvement > 0.005:
            print(f"  -> 索引标签版本略好")
        elif abs(f1_improvement) <= 0.005:
            print(f"  -> 两个版本性能相当")
        else:
            print(f"  -> One-hot版本更好")
        
        print(f"\n内存高效训练完成!")
        print(f"内存使用优化: 按batch加载数据，避免一次性加载全部数据到内存")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()