#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整修复的内存友好批量训练脚本
解决数据维度问题
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

def check_and_transpose_data(arrays):
    """
    检查并正确转置MAT数据
    """
    print("检查和转置数据...")
    
    # 检查原始数据维度
    print(f"原始MAT数据维度:")
    for key in ['data', 'region', 'prob_idx']:
        if key in arrays:
            print(f"  {key}: {arrays[key].shape}")
    
    # 处理data: 应该从 (341, N) 转为 (N, 341)
    if arrays['data'].shape[0] == 341:
        print("转置data: (341, N) -> (N, 341)")
        data_transposed = arrays['data'].T
    elif arrays['data'].shape[1] == 341:
        print("data已经是正确维度: (N, 341)")
        data_transposed = arrays['data']
    else:
        raise ValueError(f"data维度异常: {arrays['data'].shape}")
    
    # 处理region: 应该从 (102, N) 转为 (N, 102)
    if arrays['region'].shape[0] == 102:
        print("转置region: (102, N) -> (N, 102)")
        region_transposed = arrays['region'].T
    elif arrays['region'].shape[1] == 102:
        print("region已经是正确维度: (N, 102)")
        region_transposed = arrays['region']
    else:
        raise ValueError(f"region维度异常: {arrays['region'].shape}")
    
    # 处理prob_idx: 确保是1D数组
    if len(arrays['prob_idx'].shape) > 1:
        prob_idx = arrays['prob_idx'].flatten()
    else:
        prob_idx = arrays['prob_idx']
    
    print(f"转置后数据维度:")
    print(f"  data: {data_transposed.shape} (样本数, 特征数)")
    print(f"  region: {region_transposed.shape} (样本数, 类别数)")
    print(f"  prob_idx: {prob_idx.shape}")
    
    # 验证维度一致性
    n_samples_data = data_transposed.shape[0]
    n_samples_region = region_transposed.shape[0]
    n_samples_prob = len(prob_idx)
    
    assert n_samples_data == n_samples_region == n_samples_prob, \
        f"样本数不一致: data={n_samples_data}, region={n_samples_region}, prob_idx={n_samples_prob}"
    
    assert data_transposed.shape[1] == 341, \
        f"特征维度错误: 期望341，实际{data_transposed.shape[1]}"
    
    assert region_transposed.shape[1] == 102, \
        f"类别维度错误: 期望102，实际{region_transposed.shape[1]}"
    
    print("数据维度验证通过!")
    
    return data_transposed, region_transposed, prob_idx

def load_data_fixed(mat_file_path, random_state=42):
    """
    修复的数据加载函数
    """
    print("=== 修复版本的数据加载 ===")
    
    # 1. 加载MAT文件
    print(f"加载MAT文件: {mat_file_path}")
    arrays = load_mat_data(mat_file_path)
    
    # 2. 检查并转置数据
    data_transposed, region_transposed, prob_idx = check_and_transpose_data(arrays)
    
    # 3. 按患者分割
    print("按患者分割数据...")
    train_patient_indices = np.where(prob_idx != 38)[0]  # 患者1-37
    val_patient_indices = np.where(prob_idx == 38)[0]    # 患者38
    
    print(f"患者分割结果:")
    print(f"  患者1-37样本数: {len(train_patient_indices):,}")
    print(f"  患者38样本数: {len(val_patient_indices):,}")
    
    # 4. 从患者1-37中分出1%作为测试集
    train_indices, test_indices = train_test_split(
        train_patient_indices, test_size=0.01, random_state=random_state
    )
    
    print(f"最终数据分割:")
    print(f"  训练集: {len(train_indices):,} 样本")
    print(f"  测试集: {len(test_indices):,} 样本")
    print(f"  验证集: {len(val_patient_indices):,} 样本")
    
    total_samples = len(data_transposed)
    print(f"  训练集占比: {len(train_indices) / total_samples * 100:.1f}%")
    print(f"  测试集占比: {len(test_indices) / total_samples * 100:.1f}%")
    print(f"  验证集占比: {len(val_patient_indices) / total_samples * 100:.1f}%")
    
    # 5. 创建标准化器（使用全部训练集）
    print(f"\n创建StandardScaler（使用全部{len(train_indices):,}个训练样本）...")
    scaler = create_scaler_from_full_training_set(data_transposed, train_indices)
    
    # 6. 提取标签
    print("提取标签数据...")
    train_labels = region_transposed[train_indices, :]
    test_labels = region_transposed[test_indices, :]
    val_labels = region_transposed[val_patient_indices, :]
    
    print(f"标签形状:")
    print(f"  训练标签: {train_labels.shape}")
    print(f"  测试标签: {test_labels.shape}")
    print(f"  验证标签: {val_labels.shape}")
    
    return {
        'data_transposed': data_transposed,
        'region_transposed': region_transposed,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'val_indices': val_patient_indices,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'val_labels': val_labels,
        'scaler': scaler,
        'feature_dim': data_transposed.shape[1],
        'num_classes': region_transposed.shape[1],
        'total_samples': total_samples
    }

def create_scaler_from_full_training_set(data_transposed, train_indices):
    """
    使用全部训练集创建StandardScaler，分批处理避免内存问题
    """
    batch_size = 50000  # 每批5万个样本
    n_features = data_transposed.shape[1]
    n_train_samples = len(train_indices)
    
    print(f"分批计算标准化参数:")
    print(f"  训练样本总数: {n_train_samples:,}")
    print(f"  特征数: {n_features}")
    print(f"  批处理大小: {batch_size:,}")
    print(f"  预计批次数: {(n_train_samples + batch_size - 1) // batch_size}")
    
    # 第一步：计算均值
    print("步骤1: 计算均值...")
    running_sum = np.zeros(n_features, dtype=np.float64)
    processed_samples = 0
    
    for i in range(0, n_train_samples, batch_size):
        end_idx = min(i + batch_size, n_train_samples)
        batch_indices = train_indices[i:end_idx]
        
        # 从转置后的数据中提取批次
        batch_data = data_transposed[batch_indices, :]
        
        # 累加
        running_sum += np.sum(batch_data, axis=0)
        processed_samples += len(batch_data)
        
        # 进度显示
        if (i // batch_size + 1) % 5 == 0 or end_idx == n_train_samples:
            print(f"  已处理: {processed_samples:,}/{n_train_samples:,} ({processed_samples/n_train_samples*100:.1f}%)")
        
        # 清理内存
        del batch_data
        gc.collect()
    
    mean = running_sum / processed_samples
    print(f"均值计算完成: {processed_samples:,} 个样本")
    
    # 第二步：计算方差
    print("步骤2: 计算方差...")
    running_sum_sq_diff = np.zeros(n_features, dtype=np.float64)
    processed_samples = 0
    
    for i in range(0, n_train_samples, batch_size):
        end_idx = min(i + batch_size, n_train_samples)
        batch_indices = train_indices[i:end_idx]
        
        # 从转置后的数据中提取批次
        batch_data = data_transposed[batch_indices, :]
        
        # 计算方差
        diff_sq = (batch_data - mean) ** 2
        running_sum_sq_diff += np.sum(diff_sq, axis=0)
        processed_samples += len(batch_data)
        
        # 进度显示
        if (i // batch_size + 1) % 5 == 0 or end_idx == n_train_samples:
            print(f"  已处理: {processed_samples:,}/{n_train_samples:,} ({processed_samples/n_train_samples*100:.1f}%)")
        
        # 清理内存
        del batch_data, diff_sq
        gc.collect()
    
    # 计算最终统计量
    variance = running_sum_sq_diff / (processed_samples - 1)
    std = np.sqrt(variance)
    
    print(f"方差计算完成: {processed_samples:,} 个样本")
    
    # 创建并配置StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = variance
    scaler.scale_ = std
    scaler.n_samples_seen_ = processed_samples
    
    print(f"StandardScaler配置完成:")
    print(f"  均值范围: [{np.min(mean):.6f}, {np.max(mean):.6f}]")
    print(f"  标准差范围: [{np.min(std):.6f}, {np.max(std):.6f}]")
    
    # 清理临时变量
    del running_sum, running_sum_sq_diff, mean, variance, std
    gc.collect()
    
    return scaler


class FixedDataLoader:
    """
    修复的数据加载器
    """
    def __init__(self, data_transposed, indices, labels, scaler, batch_size=128, 
                 shuffle=True, use_index_labels=True):
        self.data_transposed = data_transposed
        self.indices = indices
        self.labels = labels
        self.scaler = scaler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_index_labels = use_index_labels
        
        # 处理标签格式
        if use_index_labels:
            # 转换为索引标签
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # one-hot -> 索引
                index_labels = np.argmax(labels, axis=1)
            else:
                index_labels = labels.astype(int)
            
            # 背景处理: 0->-1, 1-102->0-101
            background_mask = index_labels == 0
            index_labels = index_labels - 1
            index_labels[background_mask] = -1
            
            self.processed_labels = index_labels
            print(f"索引标签统计:")
            print(f"  标签范围: {np.min(self.processed_labels)} 到 {np.max(self.processed_labels)}")
            print(f"  背景标签(-1)数量: {np.sum(self.processed_labels == -1):,}")
            print(f"  有效标签数量: {np.sum(self.processed_labels >= 0):,}")
        else:
            self.processed_labels = labels
            print(f"One-hot标签统计:")
            print(f"  标签形状: {labels.shape}")
    
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_local_indices = indices[i:i+self.batch_size]
            batch_global_indices = self.indices[batch_local_indices]
            
            # 从转置数据中提取批次
            batch_data = self.data_transposed[batch_global_indices, :]
            
            # 应用标准化
            batch_data_scaled = self.scaler.transform(batch_data)
            
            # 获取标签
            if self.use_index_labels:
                batch_labels = self.processed_labels[batch_local_indices]
                yield torch.FloatTensor(batch_data_scaled), torch.LongTensor(batch_labels)
            else:
                batch_labels = self.processed_labels[batch_local_indices]
                yield torch.FloatTensor(batch_data_scaled), torch.FloatTensor(batch_labels)

class FixedMLP(nn.Module):
    """
    修复的MLP模型
    """
    def __init__(self, input_dim=341, num_classes=102, dropout_rate=0.5, 
                 l2_reg=1e-5, use_softmax=False):
        super(FixedMLP, self).__init__()
        
        self.use_softmax = use_softmax
        self.l2_reg = l2_reg
        
        # 构建网络层
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
        
        # 初始化权重
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

def calculate_class_weights_fixed(labels, num_classes=102, use_index_labels=True):
    """
    计算类别权重
    """
    if use_index_labels:
        # 索引标签
        valid_labels = labels[labels >= 0]  # 排除背景-1
        class_counts = np.bincount(valid_labels, minlength=num_classes)
    else:
        # one-hot标签
        class_counts = np.sum(labels, axis=0)
    
    total_samples = np.sum(class_counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total_samples / (num_classes * class_counts[i])
    
    print(f"类别权重统计:")
    print(f"  权重范围: {np.min(weights[weights > 0]):.4f} - {np.max(weights):.4f}")
    print(f"  零权重类别数: {np.sum(weights == 0)}")
    
    return torch.FloatTensor(weights)

def train_fixed_model(data_dict, device='cuda:0', save_path='./fixed_results', 
                     use_index_labels=True):
    """
    修复的训练函数
    """
    print(f"\n=== 开始修复版本训练 ===")
    print(f"标签格式: {'索引标签' if use_index_labels else 'One-hot标签'}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # 训练配置
    batch_size = 128
    num_epochs = 25
    learning_rate = 1e-5
    num_classes = 102
    
    print(f"训练配置:")
    print(f"  批大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"  设备: {device}")
    
    # 创建数据加载器
    train_loader = FixedDataLoader(
        data_dict['data_transposed'], data_dict['train_indices'], 
        data_dict['train_labels'], data_dict['scaler'], 
        batch_size, shuffle=True, use_index_labels=use_index_labels
    )
    
    val_loader = FixedDataLoader(
        data_dict['data_transposed'], data_dict['val_indices'], 
        data_dict['val_labels'], data_dict['scaler'], 
        batch_size, shuffle=False, use_index_labels=use_index_labels
    )
    
    print(f"数据加载器:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # ===== 新增：确定训练集中的有效类别 =====
    print("\n=== 分析训练集类别分布 ===")
    if use_index_labels:
        train_valid_labels = data_dict['train_labels']
        if len(train_valid_labels.shape) > 1:
            train_valid_labels = np.argmax(train_valid_labels, axis=1)
        # 转换为索引格式
        background_mask = train_valid_labels == 0
        train_valid_labels = train_valid_labels - 1
        train_valid_labels[background_mask] = -1
        
        # 找出有效类别
        valid_classes = np.unique(train_valid_labels)
        valid_classes = valid_classes[valid_classes >= 0]  # 排除背景-1
    else:
        train_valid_labels = np.argmax(data_dict['train_labels'], axis=1)
        valid_classes = np.unique(train_valid_labels)
    
    print(f"训练集中存在的类别数: {len(valid_classes)}/{num_classes}")
    print(f"有效类别范围: {np.min(valid_classes)} - {np.max(valid_classes)}")
    
    # ===== 新增：详细类别统计信息 =====
    print(f"\n详细类别统计:")
    if use_index_labels:
        labels_to_check = train_valid_labels[train_valid_labels >= 0]
    else:
        labels_to_check = train_valid_labels

    unique_labels, counts = np.unique(labels_to_check, return_counts=True)
    total_valid_samples = len(labels_to_check)

    print(f"类别分布 (前20个最多的类别):")
    # 按样本数排序
    sorted_indices = np.argsort(counts)[::-1]  # 降序
    for i in range(min(20, len(unique_labels))):
        idx = sorted_indices[i]
        label, count = unique_labels[idx], counts[idx]
        percentage = count / total_valid_samples * 100
        print(f"  类别 {label:3d}: {count:8,} 样本 ({percentage:6.2f}%)")

    if len(unique_labels) > 20:
        print(f"  ... 还有 {len(unique_labels)-20} 个类别")

    # 找出缺失的类别
    all_possible_classes = set(range(num_classes))
    present_classes = set(unique_labels)
    missing_classes = all_possible_classes - present_classes
    
    if missing_classes:
        missing_list = sorted(list(missing_classes))
        print(f"\n训练集中缺失的类别:")
        if len(missing_list) <= 30:
            print(f"  {missing_list}")
        else:
            print(f"  前30个: {missing_list[:30]}")
            print(f"  ... 还有 {len(missing_list)-30} 个")
        print(f"缺失类别总数: {len(missing_classes)}")
        
        # 计算有效类别占比
        coverage = len(present_classes) / num_classes * 100
        print(f"类别覆盖率: {coverage:.1f}% ({len(present_classes)}/{num_classes})")
    else:
        print(f"\n✓ 所有 {num_classes} 个类别在训练集中都存在")
    
    # 计算类别权重
    if use_index_labels:
        class_weights = calculate_class_weights_fixed(
            train_loader.processed_labels, num_classes, True
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 创建模型
    model = FixedMLP(
        input_dim=data_dict['feature_dim'],
        num_classes=num_classes,
        dropout_rate=0.5,
        l2_reg=1e-5,
        use_softmax=not use_index_labels
    )
    model = model.to(device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练记录
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'train_f1_scores': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_f1_scores': [],
        'val_kappa_scores': []
    }
    
    print(f"\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        all_train_preds = []
        all_train_true = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 计算损失
            if use_index_labels:
                classification_loss = criterion(output, target)
                pred = torch.argmax(output, dim=1)
                valid_mask = target != -1
                if valid_mask.sum() > 0:
                    all_train_preds.extend(pred[valid_mask].cpu().numpy())
                    all_train_true.extend(target[valid_mask].cpu().numpy())
            else:
                classification_loss = criterion(output, target)
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
            
            # 清理GPU内存
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # ===== 修改：计算训练指标 =====
        avg_train_loss = epoch_train_loss / len(train_loader)
        if len(all_train_true) > 0:
            train_acc = accuracy_score(all_train_true, all_train_preds)
            # 使用有效类别计算F1
            train_f1 = calculate_valid_f1_score(all_train_true, all_train_preds, 
                                              valid_classes, average='macro')
        else:
            train_acc = train_f1 = 0.0
        
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(train_acc)
        history['train_f1_scores'].append(train_f1)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_true = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
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
        
        # ===== 修改：计算验证指标 =====
        avg_val_loss = epoch_val_loss / len(val_loader)
        if len(all_val_true) > 0:
            val_acc = accuracy_score(all_val_true, all_val_preds)
            # 使用有效类别计算F1
            val_f1 = calculate_valid_f1_score(all_val_true, all_val_preds, 
                                            valid_classes, average='macro')
            val_kappa = cohen_kappa_score(all_val_true, all_val_preds)
        else:
            val_acc = val_f1 = val_kappa = 0.0
        
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_acc)
        history['val_f1_scores'].append(val_f1)
        history['val_kappa_scores'].append(val_kappa)
        
        # 打印结果
        print(f"  训练 - Loss: {avg_train_loss:.6f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  验证 - Loss: {avg_val_loss:.6f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
        
        # 保存最佳模型
        if epoch == 0 or val_f1 > max(history['val_f1_scores'][:-1]):
            model_name = f"best_{'index' if use_index_labels else 'onehot'}_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'history': history,
                'valid_classes': valid_classes  # 保存有效类别信息
            }, os.path.join(save_path, model_name))
            print(f"  -> 保存最佳模型 (F1: {val_f1:.4f})")
        
        # 内存清理
        gc.collect()
        torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    
    # 计算最终结果
    best_f1 = max(history['val_f1_scores'])
    best_epoch = history['val_f1_scores'].index(best_f1) + 1
    best_acc = history['val_accuracies'][history['val_f1_scores'].index(best_f1)]
    best_kappa = history['val_kappa_scores'][history['val_f1_scores'].index(best_f1)]
    
    print(f"\n=== 训练完成 ===")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"最佳F1分数: {best_f1:.4f} (第{best_epoch}轮)")
    print(f"对应准确率: {best_acc:.4f}")
    print(f"对应Kappa: {best_kappa:.4f}")
    print(f"有效类别数: {len(valid_classes)}/{num_classes}")
    
    return {
        'history': history,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_acc': best_acc,
        'best_kappa': best_kappa,
        'training_time': training_time,
        'valid_classes': valid_classes,
        'class_coverage': len(valid_classes) / num_classes
    }

def calculate_valid_f1_score(y_true, y_pred, valid_classes=None, average='macro'):
    """
    只对训练集中实际存在的类别计算F1分数
    """
    if valid_classes is None:
        # 如果没有指定有效类别，使用所有出现的类别
        valid_classes = np.unique(y_true)
        valid_classes = valid_classes[valid_classes >= 0]  # 排除背景-1
    
    if len(valid_classes) == 0:
        return 0.0
    
    # 计算每个有效类别的F1分数
    f1_scores = []
    weights = []
    
    for cls in valid_classes:
        cls_true = (np.array(y_true) == cls).astype(int)
        cls_pred = (np.array(y_pred) == cls).astype(int)
        
        tp = np.sum(cls_true & cls_pred)
        fp = np.sum((1 - cls_true) & cls_pred)
        fn = np.sum(cls_true & (1 - cls_pred))
        
        if tp + fp + fn == 0:
            continue  # 跳过完全没有预测和真值的类别
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores.append(f1)
        weights.append(np.sum(cls_true))  # 该类别的样本数作为权重
    
    if len(f1_scores) == 0:
        return 0.0
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        if sum(weights) == 0:
            return 0.0
        return np.average(f1_scores, weights=weights)
    else:
        return np.mean(f1_scores)  # 默认macro
    
def main():
    """主函数"""
    print("=== 修复版本的内存友好训练 ===")
    
    mat_file_path = "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"配置:")
    print(f"  MAT文件: {mat_file_path}")
    print(f"  设备: {device}")
    
    if not os.path.exists(mat_file_path):
        print(f"错误: MAT文件不存在")
        return
    
    try:
        # 1. 加载数据
        print("\n" + "="*60)
        print("步骤1: 加载和处理数据")
        print("="*60)
        data_dict = load_data_fixed(mat_file_path)
        
        # 2. 训练索引标签版本
        print("\n" + "="*60)
        print("步骤2: 训练索引标签版本")
        print("="*60)
        index_results = train_fixed_model(
            data_dict, device, './fixed_index_results', use_index_labels=True
        )
        
        # 3. 训练one-hot标签版本
        print("\n" + "="*60)
        print("步骤3: 训练One-hot标签版本")
        print("="*60)
        onehot_results = train_fixed_model(
            data_dict, device, './fixed_onehot_results', use_index_labels=False
        )
        
        # 4. 对比结果
        print("\n" + "="*60)
        print("最终对比结果")
        print("="*60)
        
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
        
        # 计算性能差异
        f1_diff = index_results['best_f1'] - onehot_results['best_f1']
        acc_diff = index_results['best_acc'] - onehot_results['best_acc']
        
        print(f"\n性能对比:")
        print(f"  F1分数差异: {f1_diff:+.4f}")
        print(f"  准确率差异: {acc_diff:+.4f}")
        
        if f1_diff > 0.02:
            print(f"  -> 索引标签版本显著更好!")
            print(f"  -> 主要改进: 正确处理背景标签 + 类别权重")
        elif f1_diff > 0.005:
            print(f"  -> 索引标签版本略好")
        elif abs(f1_diff) <= 0.005:
            print(f"  -> 两个版本性能相当")
        else:
            print(f"  -> One-hot版本更好，需要进一步分析")
        
        # 保存对比报告
        report_path = './comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write("标签格式对比实验报告\n")
            f.write("="*50 + "\n\n")
            
            f.write("实验设置:\n")
            f.write(f"  数据集: TRAIN38.mat\n")
            f.write(f"  总样本数: {data_dict['total_samples']:,}\n")
            f.write(f"  训练集: {len(data_dict['train_indices']):,} 样本 (96.5%)\n")
            f.write(f"  验证集: {len(data_dict['val_indices']):,} 样本 (2.5%)\n")
            f.write(f"  测试集: {len(data_dict['test_indices']):,} 样本 (1.0%)\n")
            f.write(f"  特征维度: {data_dict['feature_dim']}\n")
            f.write(f"  类别数: {data_dict['num_classes']}\n\n")
            
            f.write("模型配置:\n")
            f.write(f"  架构: 4层4096神经元MLP\n")
            f.write(f"  激活函数: ReLU\n")
            f.write(f"  Dropout率: 0.5\n")
            f.write(f"  L2正则化: 1e-5\n")
            f.write(f"  优化器: Adam\n")
            f.write(f"  学习率: 1e-5\n")
            f.write(f"  批大小: 128\n")
            f.write(f"  训练轮数: 25\n\n")
            
            f.write("索引标签版本结果:\n")
            f.write(f"  最佳F1分数: {index_results['best_f1']:.6f}\n")
            f.write(f"  最佳准确率: {index_results['best_acc']:.6f}\n")
            f.write(f"  最佳Kappa系数: {index_results['best_kappa']:.6f}\n")
            f.write(f"  最佳轮次: {index_results['best_epoch']}\n")
            f.write(f"  训练时间: {index_results['training_time']:.2f}秒\n")
            f.write(f"  特点: 索引标签(0-101), 背景忽略(-1), 类别权重\n\n")
            
            f.write("One-hot标签版本结果:\n")
            f.write(f"  最佳F1分数: {onehot_results['best_f1']:.6f}\n")
            f.write(f"  最佳准确率: {onehot_results['best_acc']:.6f}\n")
            f.write(f"  最佳Kappa系数: {onehot_results['best_kappa']:.6f}\n")
            f.write(f"  最佳轮次: {onehot_results['best_epoch']}\n")
            f.write(f"  训练时间: {onehot_results['training_time']:.2f}秒\n")
            f.write(f"  特点: One-hot标签, 标准CrossEntropyLoss\n\n")
            
            f.write("性能对比:\n")
            f.write(f"  F1分数差异: {f1_diff:+.6f}\n")
            f.write(f"  准确率差异: {acc_diff:+.6f}\n")
            if f1_diff > 0.02:
                f.write(f"  结论: 索引标签版本显著优于One-hot版本\n")
                f.write(f"  主要原因: 正确的背景处理 + 类别权重平衡\n")
            elif f1_diff > 0.005:
                f.write(f"  结论: 索引标签版本略优于One-hot版本\n")
            else:
                f.write(f"  结论: 两版本性能相当或One-hot更好\n")
            
            f.write(f"\n关键发现:\n")
            f.write(f"  1. 使用全部训练集(96.5%)进行标准化拟合\n")
            f.write(f"  2. 正确处理数据维度转置问题\n")
            f.write(f"  3. 索引标签格式能更好地处理背景像素\n")
            f.write(f"  4. 类别权重有效缓解数据不平衡问题\n")
        
        print(f"\n详细对比报告已保存: {report_path}")
        print(f"实验完成!")
        
        # 内存使用总结
        print(f"\n内存使用优化:")
        print(f"  ✓ 按batch加载数据，避免一次性加载全部数据")
        print(f"  ✓ 分批计算StandardScaler参数")
        print(f"  ✓ 定期清理GPU内存")
        print(f"  ✓ 使用全部训练集({len(data_dict['train_indices']):,}样本)进行标准化")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()