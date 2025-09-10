#!/usr/bin/env python3
"""
3×3和7×7基线模型训练脚本
Leave-one-out训练策略：留1个被试做测试集，其余37个做训练集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score
import gc
import random

# ==================== 模型定义 ====================

class SE2D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, max(1, ch//r), 1, bias=True)
        self.fc2 = nn.Conv2d(max(1, ch//r), ch, 1, bias=True)
    
    def forward(self, x):
        w = self.avg(x)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ImprovedConv2D_Baseline(nn.Module):
    def __init__(self, input_channels=351, num_classes=102, mid=128,
                 use_refine=False, use_se=True, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 通道混合（保持空间尺寸）
        self.mix = nn.Conv2d(input_channels, mid, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, mid)

        # 空间聚合（根据kernel_size调整）
        # 3×3 -> 1×1 或 7×7 -> 1×1
        self.agg = nn.Conv2d(mid, mid, kernel_size=kernel_size, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, mid)

        # 可选：1×1 残差精炼
        self.use_refine = use_refine
        if use_refine:
            self.refine = nn.Conv2d(mid, mid, kernel_size=1, bias=False)
            self.gn3 = nn.GroupNorm(8, mid)

        # 可选：SE 通道注意力
        self.se = SE2D(mid) if use_se else nn.Identity()

        # 分类头
        self.head = nn.Linear(mid, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) - H,W是patch大小
        x = F.silu(self.gn1(self.mix(x)))
        x = F.silu(self.gn2(self.agg(x)))

        if self.use_refine:
            y = x
            x = F.silu(self.gn3(self.refine(x)))
            x = x + y

        x = self.se(x)
        x = x.flatten(1)
        return self.head(x)


class ImprovedConv2D_ParamMatched(nn.Module):
    """带可配置MLP头部的Conv2D模型，用于参数量对齐"""
    
    def __init__(
        self,
        input_channels: int = 351,
        num_classes: int = 102,
        mid: int = 128,
        use_refine: bool = False,
        use_se: bool = True,
        kernel_size: int = 3,
        head_dims: Optional[Tuple[int, ...]] = None,
        mlp_hidden: int = 512,
        mlp_layers: int = 1,
        p_drop: float = 0.20,
        activation: str = 'silu'
    ):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 通道混合（保持空间尺寸）
        self.mix = nn.Conv2d(input_channels, mid, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, mid)

        # 空间聚合（根据kernel_size调整）
        # 3×3 -> 1×1 或 7×7 -> 1×1
        self.agg = nn.Conv2d(mid, mid, kernel_size=kernel_size, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, mid)

        # 可选：1×1 残差精炼
        self.use_refine = use_refine
        if use_refine:
            self.refine = nn.Conv2d(mid, mid, kernel_size=1, bias=False)
            self.gn3 = nn.GroupNorm(8, mid)

        # 可选：SE 通道注意力
        self.se = SE2D(mid) if use_se else nn.Identity()

        # 构建MLP头部
        self._build_mlp_head(mid, num_classes, head_dims, mlp_hidden, mlp_layers, p_drop, activation)

    def _build_mlp_head(
        self,
        input_dim: int,
        output_dim: int,
        head_dims: Optional[Tuple[int, ...]],
        mlp_hidden: int,
        mlp_layers: int,
        p_drop: float,
        activation: str
    ):
        """构建MLP头部"""
        layers = []
        
        # 选择激活函数
        act_fn = F.silu if activation == 'silu' else F.gelu
        
        if head_dims is not None:
            # 使用精确指定的隐藏层维度
            dims = [input_dim] + list(head_dims) + [output_dim]
        else:
            # 使用均匀宽度的隐藏层
            if mlp_layers == 1:
                # 兼容旧版本：单层直接连接
                dims = [input_dim, output_dim]
            else:
                # 多层MLP
                dims = [input_dim] + [mlp_hidden] * mlp_layers + [output_dim]
        
        # 构建层
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # 除了最后一层，都加激活和dropout
            if i < len(dims) - 2:
                # 使用模块形式的激活函数
                if activation == 'silu':
                    layers.append(nn.SiLU())
                else:
                    layers.append(nn.GELU())
                
                if p_drop > 0:
                    layers.append(nn.Dropout(p_drop))
        
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, H, W) - H,W是patch大小
        x = F.silu(self.gn1(self.mix(x)))
        x = F.silu(self.gn2(self.agg(x)))

        if self.use_refine:
            y = x
            x = F.silu(self.gn3(self.refine(x)))
            x = x + y

        x = self.se(x)
        x = x.flatten(1)  # (B, mid)
        return self.head(x)  # (B, num_classes)

# ==================== 数据集定义 ====================

class Brain3DPatchDataset(Dataset):
    """3D脑体素patch数据集"""
    
    def __init__(self, mat_files: List[Path], patch_size: int = 3, 
                 samples_per_subject: int = 10000, is_train: bool = True,
                 cache_data: bool = True):
        """
        Args:
            mat_files: 3D MAT文件路径列表
            patch_size: patch尺寸 (3或7)
            samples_per_subject: 每个被试采样的体素数
            is_train: 是否为训练模式
            cache_data: 是否缓存数据到内存
        """
        self.mat_files = mat_files
        self.patch_size = patch_size
        self.samples_per_subject = samples_per_subject
        self.is_train = is_train
        self.cache_data = cache_data
        
        # 预加载所有数据到内存（如果cache_data=True）
        self.cached_data = {}
        if cache_data:
            print(f"预加载{len(mat_files)}个被试数据到内存...")
            for mat_file in tqdm(mat_files):
                self.cached_data[str(mat_file)] = self.load_subject_data(mat_file)
        
        # 计算总样本数
        self.total_samples = len(mat_files) * samples_per_subject
        
        # 生成所有采样索引
        self.sample_indices = self._generate_sample_indices()
    
    def load_subject_data(self, mat_file: Path) -> Dict[str, np.ndarray]:
        """加载单个被试的3D数据"""
        with h5py.File(mat_file, 'r') as f:
            # 加载必要数据
            data = f['data'][()]
            region_labels = f['region_labels'][()]
            region_mask = f['region_mask'][()]
            
            print(f"原始数据形状: data={data.shape}, labels={region_labels.shape}, mask={region_mask.shape}")
            
            # 根据实际的数据格式进行正确的转置
            if data.shape[0] == 351:
                # 如果第一个维度是351，说明是 (351, 384, 336, 256) 格式
                data = np.transpose(data, (1, 2, 3, 0))  # -> (384, 336, 256, 351)
            elif data.shape[-1] == 351:
                # 如果最后一个维度是351，已经是正确格式
                pass
            else:
                raise ValueError(f"无法识别数据格式，shape: {data.shape}")
            
            # labels和mask的处理
            if region_labels.shape != (384, 336, 256):
                region_labels = region_labels.T
            if region_mask.shape != (384, 336, 256):
                region_mask = region_mask.T
                
            print(f"转置后数据形状: data={data.shape}, labels={region_labels.shape}, mask={region_mask.shape}")
            
            # 对每个patient的351个channel进行z-score标准化
            # data shape: (384, 336, 256, 351)
            data = data.astype(np.float32)
            
            # 计算每个channel的均值和标准差
            for ch in range(data.shape[3]):  # 351个channels
                channel_data = data[:, :, :, ch]
                
                # 计算当前channel的均值和标准差
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                
                # 避免除以0
                if std_val > 1e-8:
                    data[:, :, :, ch] = (channel_data - mean_val) / std_val
                else:
                    # 如果标准差为0，则将该channel设为0
                    data[:, :, :, ch] = 0
            
            print(f"已完成351维channel的z-score标准化")
            
        return {
            'data': data,
            'labels': region_labels.astype(np.int64),
            'mask': region_mask.astype(bool)
        }
    
    def _generate_sample_indices(self) -> List[Tuple[int, int, int, int]]:
        """生成所有采样位置索引"""
        indices = []
        
        for file_idx, mat_file in enumerate(self.mat_files):
            if self.cache_data:
                subject_data = self.cached_data[str(mat_file)]
            else:
                subject_data = self.load_subject_data(mat_file)
            
            mask = subject_data['mask']
            labels = subject_data['labels']
            
            # 获取所有有效体素位置（标签>0且在mask内）
            valid_positions = np.where((mask > 0) & (labels > 0))
            valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))
            
            # 采样
            if len(valid_coords) > self.samples_per_subject:
                if self.is_train:
                    # 训练时随机采样
                    sampled_coords = random.sample(valid_coords, self.samples_per_subject)
                else:
                    # 测试时均匀采样
                    step = len(valid_coords) // self.samples_per_subject
                    sampled_coords = valid_coords[::step][:self.samples_per_subject]
            else:
                # 如果有效体素少于目标数，全部使用
                sampled_coords = valid_coords
            
            # 添加到索引列表（文件索引，x，y，z）
            for x, y, z in sampled_coords:
                indices.append((file_idx, x, y, z))
        
        # 训练时打乱
        if self.is_train:
            random.shuffle(indices)
        
        return indices
    
    def extract_patch_2d(self, data: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
        """提取指定位置的2D patch（在xy平面上）"""
        p = self.patch_size // 2
        
        # 在x-y平面上提取patch，z固定
        x_min = max(0, x - p)
        x_max = min(data.shape[0], x + p + 1)
        y_min = max(0, y - p)
        y_max = min(data.shape[1], y + p + 1)
        
        # 提取2D patch (在z切片上)
        patch = data[x_min:x_max, y_min:y_max, z, :]
        
        # 如果patch不足目标大小，进行padding
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size, data.shape[3]), 
                             dtype=data.dtype)
            
            # 计算padding偏移
            x_off = p - (x - x_min)
            y_off = p - (y - y_min)
            
            # 复制数据到padded数组
            padded[x_off:x_off+patch.shape[0], 
                   y_off:y_off+patch.shape[1], :] = patch
            patch = padded
        
        return patch  # (patch_size, patch_size, 351)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        file_idx, x, y, z = self.sample_indices[idx]
        
        # 获取被试数据
        if self.cache_data:
            mat_file = str(self.mat_files[file_idx])
            subject_data = self.cached_data[mat_file]
        else:
            subject_data = self.load_subject_data(self.mat_files[file_idx])
        
        # 提取2D patch (在xy平面上，z位置固定)
        patch_2d = self.extract_patch_2d(subject_data['data'], x, y, z)
        
        # 获取中心体素的标签（注意：标签是1-102，需要转为0-101）
        label = subject_data['labels'][x, y, z] - 1
        
        # 转换为PyTorch格式 (C, H, W)
        # patch_2d shape: (patch_size, patch_size, 351)
        patch_tensor = torch.from_numpy(patch_2d.transpose(2, 0, 1).astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor

# ==================== 训练函数 ====================

class Trainer:
    def __init__(self, model, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (patches, labels) in enumerate(pbar):
            patches, labels = patches.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = self.model(patches)
                loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, macro_f1
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for patches, labels in tqdm(test_loader, desc='Evaluating'):
                patches, labels = patches.to(self.device), labels.to(self.device)
                
                outputs = self.model(patches)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, macro_f1

# ==================== 工厂函数与辅助函数 ====================

def count_params(model: nn.Module) -> int:
    """计算模型的总参数量"""
    return sum(p.numel() for p in model.parameters())


def make_param_matched_model(
    target: str = "35M",
    input_channels: int = 351,
    num_classes: int = 102,
    kernel_size: int = 3,
    custom_head_dims: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> ImprovedConv2D_ParamMatched:
    """工厂函数：创建参数量对齐的模型
    
    Args:
        target: 目标参数量档位 ("35M", "52M", "69M")
        input_channels: 输入通道数
        num_classes: 输出类别数
        kernel_size: 卷积核大小
        custom_head_dims: 自定义头部维度（覆盖预设）
        **kwargs: 其他传递给模型的参数
    
    Returns:
        ImprovedConv2D_ParamMatched: 配置好的模型实例
    """
    
    # 预设配置
    configs = {
        "35M": {
            "mid": 128,
            "head_dims": (4139, 4139, 4139)  # 约35M参数
        },
        "52M": {
            "mid": 768,
            "head_dims": (4608, 4608, 4608)  # 约52M参数
        },
        "69M": {
            "mid": 896,
            "head_dims": (4352, 4352, 4352, 4352)  # 约69M参数
        }
    }
    
    if target not in configs:
        raise ValueError(f"Unknown target: {target}. Choose from {list(configs.keys())}")
    
    config = configs[target]
    
    # 如果提供了自定义head_dims，使用它
    if custom_head_dims is not None:
        config["head_dims"] = custom_head_dims
    
    # 合并配置与额外参数
    model_kwargs = {
        "input_channels": input_channels,
        "num_classes": num_classes,
        "kernel_size": kernel_size,
        **config,
        **kwargs
    }
    
    return ImprovedConv2D_ParamMatched(**model_kwargs)

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='训练3×3和7×7基线模型')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='3D MAT文件目录')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--patch_size', type=int, choices=[3, 7], default=3,
                       help='Patch大小')
    parser.add_argument('--test_subject', type=int, default=1,
                       help='测试被试编号(1-38)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--samples_per_subject', type=int, default=10000,
                       help='每个被试采样的体素数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--use_se', action='store_true', default=True,
                       help='使用SE模块')
    parser.add_argument('--use_refine', action='store_true', default=False,
                       help='使用refine层')
    parser.add_argument('--model_size', type=str, default=None,
                       choices=['baseline', '35M', '52M', '69M'],
                       help='模型大小配置 (None表示使用baseline)')
    
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 获取所有MAT文件
    data_dir = Path(args.data_dir)
    mat_files = sorted(data_dir.glob('subject*_3d_validated.mat'))
    
    if len(mat_files) == 0:
        # 尝试其他命名模式
        mat_files = sorted(data_dir.glob('*.mat'))
    
    logger.info(f'找到 {len(mat_files)} 个MAT文件')
    
    if len(mat_files) < 2:
        logger.error('至少需要2个被试的数据')
        return
    
    # 分割训练和测试集
    test_idx = args.test_subject - 1
    if test_idx >= len(mat_files):
        logger.error(f'测试被试编号 {args.test_subject} 超出范围')
        return
    
    test_files = [mat_files[test_idx]]
    train_files = mat_files[:test_idx] + mat_files[test_idx+1:]
    
    logger.info(f'训练集: {len(train_files)} 个被试')
    logger.info(f'测试集: {test_files[0].name}')
    
    # 创建数据集
    logger.info('创建数据集...')
    train_dataset = Brain3DPatchDataset(
        train_files, 
        patch_size=args.patch_size,
        samples_per_subject=args.samples_per_subject,
        is_train=True,
        cache_data=True
    )
    
    test_dataset = Brain3DPatchDataset(
        test_files,
        patch_size=args.patch_size,
        samples_per_subject=args.samples_per_subject * 2,  # 测试时多采样一些
        is_train=False,
        cache_data=True
    )
    
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
    
    # 创建模型
    logger.info(f'创建 {args.patch_size}×{args.patch_size} 模型...')
    
    if args.model_size is None or args.model_size == 'baseline':
        # 使用原始基线模型
        model = ImprovedConv2D_Baseline(
            input_channels=351,
            num_classes=102,
            mid=128,
            use_refine=args.use_refine,
            use_se=args.use_se,
            kernel_size=args.patch_size
        )
        logger.info('使用基线模型（~58K参数）')
    else:
        # 使用参数匹配模型
        model = make_param_matched_model(
            target=args.model_size,
            input_channels=351,
            num_classes=102,
            kernel_size=args.patch_size,
            use_refine=args.use_refine,
            use_se=args.use_se
        )
        logger.info(f'使用参数匹配模型（目标: {args.model_size}）')
    
    logger.info(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 创建训练器
    trainer = Trainer(model, device=device, learning_rate=args.lr)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_f1': [],
        'test_loss': [],
        'test_f1': []
    }
    
    best_test_f1 = 0
    best_epoch = 0
    
    # 训练循环
    logger.info('开始训练...')
    for epoch in range(args.epochs):
        logger.info(f'\n===== Epoch {epoch+1}/{args.epochs} =====')
        
        # 训练
        train_loss, train_f1 = trainer.train_epoch(train_loader)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        
        # 评估
        test_loss, test_f1 = trainer.evaluate(test_loader)
        history['test_loss'].append(test_loss)
        history['test_f1'].append(test_f1)
        
        logger.info(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        logger.info(f'Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}')
        
        # 保存最佳模型
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'test_f1': test_f1,
                'args': vars(args)
            }
            
            checkpoint_path = output_dir / f'best_model_patch{args.patch_size}_test{args.test_subject}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'保存最佳模型 (F1: {best_test_f1:.4f})')
        
        # 学习率调度
        trainer.scheduler.step()
        
        # 定期垃圾回收
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 保存训练历史
    history_file = output_dir / f'history_patch{args.patch_size}_test{args.test_subject}.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 最终报告
    logger.info('\n===== 训练完成 =====')
    logger.info(f'最佳测试 F1: {best_test_f1:.4f} (Epoch {best_epoch})')
    logger.info(f'结果保存在: {output_dir}')

if __name__ == '__main__':
    import sys
    
    # 检查是否运行自检
    if len(sys.argv) > 1 and sys.argv[1] == '--self-check':
        print("===== 模型自检 =====")
        print()
        
        # 测试设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试数据
        B = 2
        C = 351
        H = W = 3  # 3×3 patch
        x = torch.randn(B, C, H, W).to(device)
        
        print("输入形状:", x.shape)
        print()
        
        # 测试原始模型（作为基准）
        print("1. 原始模型（基准）")
        model_baseline = ImprovedConv2D_Baseline(
            input_channels=C,
            num_classes=102,
            mid=128
        ).to(device)
        
        with torch.no_grad():
            out = model_baseline(x)
        
        print(f"   参数量: {count_params(model_baseline):,}")
        print(f"   输出形状: {out.shape}")
        print()
        
        # 测试35M参数模型
        print("2. 35M参数模型")
        model_35m = make_param_matched_model(
            target="35M",
            input_channels=C,
            num_classes=102,
            kernel_size=3
        ).to(device)
        
        with torch.no_grad():
            out = model_35m(x)
        
        params_35m = count_params(model_35m)
        print(f"   参数量: {params_35m:,}")
        print(f"   目标: ~35,000,000 (误差: {abs(params_35m - 35_000_000) / 35_000_000 * 100:.2f}%)")
        print(f"   输出形状: {out.shape}")
        print()
        
        # 测试52M参数模型
        print("3. 52M参数模型")
        model_52m = make_param_matched_model(
            target="52M",
            input_channels=C,
            num_classes=102,
            kernel_size=3
        ).to(device)
        
        with torch.no_grad():
            out = model_52m(x)
        
        params_52m = count_params(model_52m)
        print(f"   参数量: {params_52m:,}")
        print(f"   目标: ~52,000,000 (误差: {abs(params_52m - 52_000_000) / 52_000_000 * 100:.2f}%)")
        print(f"   输出形状: {out.shape}")
        print()
        
        # 测试69M参数模型
        print("4. 69M参数模型")
        model_69m = make_param_matched_model(
            target="69M",
            input_channels=C,
            num_classes=102,
            kernel_size=3
        ).to(device)
        
        with torch.no_grad():
            out = model_69m(x)
        
        params_69m = count_params(model_69m)
        print(f"   参数量: {params_69m:,}")
        print(f"   目标: ~69,000,000 (误差: {abs(params_69m - 69_000_000) / 69_000_000 * 100:.2f}%)")
        print(f"   输出形状: {out.shape}")
        print()
        
        # 测试兼容性：不传head_dims时的默认行为
        print("5. 兼容性测试（默认单层头）")
        model_compat = ImprovedConv2D_ParamMatched(
            input_channels=C,
            num_classes=102,
            mid=128,
            mlp_layers=1  # 默认单层，应该与基准模型参数量相同
        ).to(device)
        
        with torch.no_grad():
            out = model_compat(x)
        
        print(f"   参数量: {count_params(model_compat):,}")
        print(f"   基准参数量: {count_params(model_baseline):,}")
        print(f"   输出形状: {out.shape}")
        print()
        
        print("===== 自检完成 =====")
    else:
        main()