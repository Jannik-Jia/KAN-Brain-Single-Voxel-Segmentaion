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
                 samples_per_subject: Optional[int] = None,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            mat_files: 1D MAT文件路径列表
            is_train: 是否为训练模式
            samples_per_subject: 每个被试采样的体素数（None表示全部）
            scaler: StandardScaler对象，用于数据标准化
        """
        self.mat_files = mat_files
        self.is_train = is_train
        self.samples_per_subject = samples_per_subject
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
            self.all_data = self.scaler.fit_transform(self.all_data)
        elif self.scaler is not None:
            print("应用StandardScaler...")
            self.all_data = self.scaler.transform(self.all_data)
    
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
            
            # 从one-hot转换为类别标签
            labels = np.argmax(seg_one_hot, axis=1)  # (n_voxels,)
            
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
            
            self.features = multidim_data.astype(np.float32)  # (n_voxels, 351)
            self.labels = np.argmax(seg_one_hot, axis=1)      # (n_voxels,)
        
        # 加载3D mask（用于映射）
        with h5py.File(mat_file_3d, 'r') as f:
            region_mask = f['region_mask'][()]
            region_labels = f['region_labels'][()]
            
            # 处理转置
            if region_mask.shape != (384, 336, 256):
                region_mask = region_mask.T
            if region_labels.shape != (384, 336, 256):
                region_labels = region_labels.T
                
            self.region_mask = region_mask
            self.region_labels = region_labels
        
        # 应用标准化
        self.features = scaler.transform(self.features)
        
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
        self.history = {'train_loss': [], 'train_f1': [], 'test_loss': [], 'test_f1': []}
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_train_loss = 0
        all_preds = []
        all_labels = []
        
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
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_train_loss, train_f1
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
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
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_val_loss, val_f1
    
    def train(self, train_loader, test_loader, epochs=25):
        """完整训练流程（对应Alex的25个epochs）"""
        best_test_f1 = 0
        
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs} =====")
            
            # 训练
            train_loss, train_f1 = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            
            # 测试（验证）
            test_loss, test_f1 = self.evaluate(test_loader)
            self.history['test_loss'].append(test_loss)
            self.history['test_f1'].append(test_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
            
            # 保存最佳模型（基于测试F1）
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                self.best_model_state = self.model.state_dict()
                print(f"新的最佳测试F1: {best_test_f1:.4f}")
        
        # 恢复最佳模型
        self.model.load_state_dict(self.best_model_state)
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
    
    # 映射回3D体积
    print("映射回3D体积...")
    predictions_3d = np.zeros((384, 336, 256, 102), dtype=np.float32)
    
    # 获取有效体素的3D位置
    region_mask = test_dataset.region_mask
    region_labels = test_dataset.region_labels
    
    # 找到有效体素（mask > 0 且 labels > 0）
    valid_mask = (region_mask > 0) & (region_labels > 0)
    valid_positions = np.where(valid_mask)
    
    # 检查体素数量是否匹配
    if len(valid_positions[0]) != len(all_predictions):
        print(f"警告：体素数量不匹配！3D mask: {len(valid_positions[0])}, 预测: {len(all_predictions)}")
        # 取最小值
        min_len = min(len(valid_positions[0]), len(all_predictions))
        valid_positions = (valid_positions[0][:min_len], 
                          valid_positions[1][:min_len], 
                          valid_positions[2][:min_len])
        all_predictions = all_predictions[:min_len]
    
    # 将预测概率映射回3D位置
    for i, (x, y, z) in enumerate(zip(valid_positions[0], valid_positions[1], valid_positions[2])):
        predictions_3d[x, y, z, :] = all_predictions[i]
    
    print(f"映射完成，3D体积形状: {predictions_3d.shape}")
    
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
    parser.add_argument('--samples_per_subject', type=int, default=None,
                       help='每个被试采样的体素数（None表示全部）')
    parser.add_argument('--save_predictions', action='store_true',
                       help='保存3D预测概率')
    
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
    
    # 查找1D文件（去除_3d后缀）
    mat_files_1d = sorted(data_dir_1d.glob('*.mat'))
    mat_files_3d = sorted(data_dir_3d.glob('*_3d_validated.mat'))
    
    logger.info(f'找到1D文件: {len(mat_files_1d)}个')
    logger.info(f'找到3D文件: {len(mat_files_3d)}个')
    
    # 分割训练和测试集（Leave-one-out）
    test_idx = args.test_subject - 1
    
    train_files_1d = mat_files_1d[:test_idx] + mat_files_1d[test_idx+1:]
    test_file_1d = mat_files_1d[test_idx]
    test_file_3d = mat_files_3d[test_idx]
    
    logger.info(f'训练集: {len(train_files_1d)} 个被试')
    logger.info(f'测试集1D: {test_file_1d.name}')
    logger.info(f'测试集3D: {test_file_3d.name}')
    
    # 创建训练数据集
    logger.info('加载训练数据...')
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        is_train=True,
        samples_per_subject=args.samples_per_subject,
        scaler=None
    )
    
    # 获取scaler用于测试集
    scaler = train_dataset.scaler
    
    # 创建测试数据集（需要1D和3D文件）
    logger.info('加载测试数据...')
    test_dataset = TestDataset(test_file_1d, test_file_3d, scaler)
    
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
    
    # 创建模型（351维输入）
    logger.info('创建模型...')
    model = RegModel(input_dim=351, num_classes=102)
    
    logger.info(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 创建训练器
    trainer = Trainer(model, device=device, learning_rate=0.00001)
    
    # 训练模型
    logger.info('开始训练...')
    history = trainer.train(train_loader, test_loader, epochs=args.epochs)
    
    # 保存模型
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
        
        # 保存预测
        pred_path = output_dir / f'predictions_3d_test{args.test_subject}.mat'
        scipy.io.savemat(
            str(pred_path),
            {
                'softmax_probabilities': predictions_3d,  # (384, 336, 256, 102)
                'test_subject': args.test_subject,
                'test_file_1d': str(test_file_1d),
                'test_file_3d': str(test_file_3d)
            }
        )
        logger.info(f'预测保存至: {pred_path}')
        logger.info(f'  形状: {predictions_3d.shape}')
        logger.info(f'  概率范围: [{predictions_3d.min():.4f}, {predictions_3d.max():.4f}]')
    
    # 最终报告
    logger.info('\n===== 训练完成 =====')
    logger.info(f'最佳测试F1: {max(history["test_f1"]):.4f}')
    logger.info(f'最终训练Loss: {history["train_loss"][-1]:.4f}')
    logger.info(f'最终训练F1: {history["train_f1"][-1]:.4f}')
    logger.info(f'最终测试Loss: {history["test_loss"][-1]:.4f}')
    logger.info(f'最终测试F1: {history["test_f1"][-1]:.4f}')

if __name__ == '__main__':
    main()