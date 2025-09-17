#!/usr/bin/env python3
"""
MRI ResNet 批量训练脚本

支持分批加载MAT文件的ResNet训练脚本，避免内存溢出问题。
每次只加载指定数量的MAT文件到内存，训练完成后释放内存并加载下一批。

用法:
    python train_mri_resnet_batch.py --data_dir /path/to/mat/files --test_subject 1 --batch_files 3
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加模型目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'models'))

from resnet import mri_resnet50, count_parameters, get_model_info
from batch_dataset_improved import create_improved_batch_data_loaders, ImprovedBatchMRIBrain2DPatchDataset
from losses import create_loss_function, mixup_data, MixupLoss

warnings.filterwarnings('ignore')


def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """设置日志配置"""
    log_level = logging.INFO if verbose else logging.WARNING

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 设置日志器
    logger = logging.getLogger('MRIResNetBatchTrainer')
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class MRIResNetBatchTrainer:
    """支持批量数据加载的MRI ResNet训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: ImprovedBatchMRIBrain2DPatchDataset,
        test_loader: DataLoader,
        device: str = 'cuda',
        loss_type: str = 'cb_focal',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        gamma: float = 1.5,
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
        tau: float = 1.0,
        batch_size: int = 256,
        num_workers: int = 4,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model: ResNet模型
            train_dataset: 批量训练数据集
            test_loader: 测试数据加载器
            device: 训练设备
            loss_type: 损失函数类型
            learning_rate: 学习率
            weight_decay: 权重衰减
            use_mixup: 启用mixup增强
            mixup_alpha: Mixup alpha参数
            use_ema: 使用指数移动平均
            ema_decay: EMA衰减率
            gamma: Focal loss gamma参数
            beta: Class-balanced loss beta参数
            label_smoothing: 标签平滑因子
            tau: Logit adjustment温度参数
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            logger: 日志器实例
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_loader = test_loader
        self.device = device
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_ema = use_ema
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # 获取类别数量统计
        class_counts = torch.tensor([
            train_dataset.class_counts.get(i, 1) for i in range(1, 103)
        ], dtype=torch.float32).to(device)

        # 设置损失函数
        self.base_criterion = create_loss_function(
            loss_type=loss_type,
            class_counts=class_counts,
            reduction='mean',
            gamma=gamma,
            beta=beta,
            label_smoothing=label_smoothing,
            tau=tau
        ).to(device)

        self.criterion = self.base_criterion

        # 创建mixup损失函数
        if use_mixup:
            self.mixup_criterion = MixupLoss(self.base_criterion)
        else:
            self.mixup_criterion = None

        # 设置优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()

        # EMA模型
        if use_ema:
            self.ema_model = self._create_ema_model(ema_decay)
        else:
            self.ema_model = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'test_loss': [],
            'test_f1': [],
            'learning_rates': []
        }

        self.logger.info(f"批量训练器初始化完成，使用{loss_type}损失函数")
        self.logger.info(f"模型参数: {count_parameters(model):,}")
        self.logger.info(f"类别分布: {len(train_dataset.class_counts)}个类别")

    def _create_ema_model(self, decay: float):
        """创建指数移动平均模型"""
        from torch.optim.swa_utils import AveragedModel

        def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return decay * averaged_model_parameter + (1 - decay) * model_parameter

        return AveragedModel(self.model, avg_fn=ema_avg)

    def train_batch_epoch(self, batch_info: Dict) -> Tuple[float, float]:
        """训练当前批次的一个epoch"""
        self.model.train()

        # 创建当前批次的数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        total_loss = 0.0
        all_preds = []
        all_labels = []

        batch_desc = f"训练批次 {batch_info['current_batch']}/{batch_info['total_batches']}"
        pbar = tqdm(train_loader, desc=batch_desc, leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # 应用mixup增强
            if self.use_mixup and np.random.random() < 0.5:
                mixed_images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.mixup_alpha
                )

                self.optimizer.zero_grad()

                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    outputs = self.model(mixed_images)
                    loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 使用原始标签计算指标
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            else:
                # 标准训练
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.base_criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 收集预测结果
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()

            # 更新EMA
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'Batch': f'{batch_info["current_batch"]}/{batch_info["total_batches"]}'
            })

        # 计算指标
        avg_loss = total_loss / len(train_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, macro_f1

    def train_full_epoch(self) -> Tuple[float, float]:
        """训练完整的一个epoch（所有批次）"""
        epoch_losses = []
        epoch_f1s = []

        # 重置到第一批
        self.train_dataset.reset_to_first_batch()

        # 训练所有批次
        while True:
            batch_info = self.train_dataset.get_current_batch_info()

            # 训练当前批次
            batch_loss, batch_f1 = self.train_batch_epoch(batch_info)
            epoch_losses.append(batch_loss)
            epoch_f1s.append(batch_f1)

            # 切换到下一批
            has_next = self.train_dataset.next_batch()
            if not has_next:  # 已完成所有批次
                break

        # 计算epoch平均指标
        avg_loss = np.mean(epoch_losses)
        avg_f1 = np.mean(epoch_f1s)

        # 为下一个epoch准备新的样本分配
        self.train_dataset.start_new_epoch()

        return avg_loss, avg_f1

    def evaluate(self, use_ema: bool = False) -> Tuple[float, float, Dict]:
        """评估模型"""
        model = self.ema_model if (use_ema and self.ema_model) else self.model
        model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='评估中', leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算综合指标
        avg_loss = total_loss / len(self.test_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 类别指标
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        metrics = {
            'macro_f1': macro_f1,
            'macro_precision': np.mean(precision),
            'macro_recall': np.mean(recall),
            'per_class_f1': f1.tolist(),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'support': support.tolist()
        }

        return avg_loss, macro_f1, metrics

    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
        save_best: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        完整训练循环（批量版本）

        Args:
            epochs: 训练轮数
            patience: 早停耐心值
            save_best: 保存最佳模型
            output_dir: 输出目录

        Returns:
            训练历史
        """
        best_f1 = 0.0
        patience_counter = 0
        best_epoch = 0

        self.logger.info(f"开始批量训练，共{epochs}轮")
        self.logger.info(f"早停耐心值: {patience}")

        train_dataset_info = self.train_dataset.get_current_batch_info()
        self.logger.info(f"训练数据分为{train_dataset_info['total_batches']}批，"
                        f"每批最多{self.train_dataset.batch_files}个文件")

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练（所有批次）
            train_loss, train_f1 = self.train_full_epoch()

            # 评估
            test_loss, test_f1, detailed_metrics = self.evaluate(use_ema=self.use_ema)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['test_loss'].append(test_loss)
            self.history['test_f1'].append(test_f1)
            self.history['learning_rates'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # 日志记录
            self.logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # 早停和模型保存
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                patience_counter = 0

                if save_best and output_dir:
                    self.save_checkpoint(
                        output_dir / 'best_model.pth',
                        epoch=epoch,
                        metrics=detailed_metrics,
                        is_best=True
                    )
                    self.logger.info(f"新的最佳F1: {best_f1:.4f} at epoch {epoch+1}")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"早停于epoch {epoch+1}")
                    self.logger.info(f"最佳F1: {best_f1:.4f} at epoch {best_epoch+1}")
                    break

            # 每10轮保存检查点
            if output_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    output_dir / f'checkpoint_epoch_{epoch+1}.pth',
                    epoch=epoch,
                    metrics=detailed_metrics
                )

        # 最终评估
        if save_best and output_dir and (output_dir / 'best_model.pth').exists():
            self.load_checkpoint(output_dir / 'best_model.pth')
            final_loss, final_f1, final_metrics = self.evaluate(use_ema=self.use_ema)

            self.logger.info(f"最终评估 - Loss: {final_loss:.4f}, F1: {final_f1:.4f}")

            # 保存详细结果
            results = {
                'best_epoch': best_epoch + 1,
                'best_f1': best_f1,
                'final_f1': final_f1,
                'history': self.history,
                'final_metrics': final_metrics,
                'batch_training_info': {
                    'total_batches': train_dataset_info['total_batches'],
                    'batch_files': self.train_dataset.batch_files
                }
            }

            if output_dir:
                with open(output_dir / 'training_results.json', 'w') as f:
                    json.dump(results, f, indent=2)

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Optional[Dict] = None,
        is_best: bool = False
    ):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'metrics': metrics,
            'is_best': is_best
        }

        if self.ema_model:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.ema_model and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        self.history = checkpoint.get('history', self.history)

        return checkpoint.get('metrics', {})


def find_mat_files(data_dir: Path) -> List[Path]:
    """查找所有MAT文件"""
    mat_files = sorted(data_dir.glob('subject*_3d_validated.mat'))

    if len(mat_files) == 0:
        mat_files = sorted(data_dir.glob('*.mat'))

    if not mat_files:
        raise FileNotFoundError(f"在{data_dir}中未找到MAT文件")

    return mat_files


def create_leave_one_out_split(
    mat_files: List[Path],
    test_subject: int
) -> Tuple[List[Path], List[Path]]:
    """创建Leave-One-Out训练/测试分割"""
    if test_subject < 1 or test_subject > len(mat_files):
        raise ValueError(f"test_subject必须在1到{len(mat_files)}之间")

    test_files = [mat_files[test_subject - 1]]
    train_files = mat_files[:test_subject - 1] + mat_files[test_subject:]

    return train_files, test_files


def plot_training_history(history: Dict, output_dir: Path):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失图
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['test_loss'], label='Test Loss', color='red')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # F1分数图
    ax2.plot(epochs, history['train_f1'], label='Train F1', color='blue')
    ax2.plot(epochs, history['test_f1'], label='Test F1', color='red')
    ax2.set_title('Training and Test F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)

    # 学习率
    ax3.plot(epochs, history['learning_rates'], color='green')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)

    # F1差异（过拟合指示器）
    f1_diff = np.array(history['train_f1']) - np.array(history['test_f1'])
    ax4.plot(epochs, f1_diff, color='purple')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Overfitting Indicator (Train F1 - Test F1)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Difference')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='训练MRI ResNet进行脑区分类（批量加载版本）')

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='包含3D MAT文件的目录')
    parser.add_argument('--test_subject', type=int, required=True,
                       help='Leave-One-Out的测试被试编号 (1-38)')
    parser.add_argument('--output_dir', type=str, default='./results_batch',
                       help='结果输出目录')

    # 批量加载参数
    parser.add_argument('--batch_files', type=int, default=3,
                       help='每批次加载的文件数量（默认3个）')

    # 模型参数
    parser.add_argument('--base_width', type=int, default=104,
                       help='ResNet基础宽度（默认104，约50M参数）')
    parser.add_argument('--input_channels', type=int, default=351,
                       help='输入通道数')
    parser.add_argument('--num_classes', type=int, default=102,
                       help='输出类别数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')

    # 数据参数
    parser.add_argument('--patch_size', type=int, default=7,
                       help='图像块大小')
    parser.add_argument('--samples_per_subject', type=int, default=10000,
                       help='每个被试的样本数（匹配3D CNN基线）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--balance_classes', action='store_true',
                       help='启用数据集类别平衡')
    parser.add_argument('--augmentation', action='store_true',
                       help='启用数据增强')

    # 损失和优化
    parser.add_argument('--loss_type', type=str, default='cb_focal',
                       choices=['ce', 'weighted_ce', 'focal', 'cb_focal', 'logit_adj', 'balanced_softmax'],
                       help='损失函数类型')
    parser.add_argument('--use_mixup', action='store_true',
                       help='启用mixup增强')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup alpha参数')
    parser.add_argument('--use_ema', action='store_true',
                       help='使用指数移动平均')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='Focal loss gamma参数')
    parser.add_argument('--beta', type=float, default=0.9999,
                       help='Class-balanced loss beta参数')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='标签平滑因子')
    parser.add_argument('--tau', type=float, default=1.0,
                       help='Logit adjustment温度参数')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                       help='详细日志输出')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir) / f'resnet_batch_test_subject_{args.test_subject}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir, args.verbose)

    logger.info("=== MRI ResNet 批量训练 ===")
    logger.info(f"参数: {vars(args)}")

    # 查找MAT文件
    data_dir = Path(args.data_dir)
    mat_files = find_mat_files(data_dir)
    logger.info(f"找到{len(mat_files)}个MAT文件")

    # 创建Leave-One-Out分割
    train_files, test_files = create_leave_one_out_split(mat_files, args.test_subject)
    logger.info(f"训练被试: {len(train_files)}")
    logger.info(f"测试被试: {args.test_subject}")
    logger.info(f"每批次加载: {args.batch_files}个文件")

    # 创建改进的批量数据加载器
    train_loader, test_loader, train_dataset = create_improved_batch_data_loaders(
        train_files=train_files,
        test_files=test_files,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_subject=args.samples_per_subject,
        balance_classes=args.balance_classes,
        augmentation=args.augmentation,
        batch_files=args.batch_files
    )

    # 记录数据信息
    batch_info = train_dataset.get_current_batch_info()
    logger.info(f"训练数据分为{batch_info['total_batches']}批")
    logger.info(f"当前批次样本数: {len(train_dataset)}")
    logger.info(f"测试样本数: {len(test_loader.dataset)}")

    # 创建模型
    model = mri_resnet50(
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        base_width=args.base_width
    )

    # 记录模型信息
    model_info = get_model_info(model, input_size=(1, args.input_channels, args.patch_size, args.patch_size))
    logger.info(f"模型参数: {model_info['total_parameters']:,}")
    logger.info(f"模型大小: {model_info['parameter_size_mb']:.2f} MB")

    # 创建批量训练器
    trainer = MRIResNetBatchTrainer(
        model=model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        device=args.device,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_ema=args.use_ema,
        gamma=args.gamma,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        tau=args.tau,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logger=logger
    )

    # 训练模型
    history = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        save_best=True,
        output_dir=output_dir
    )

    # 绘制训练历史
    plot_training_history(history, output_dir)

    logger.info("批量训练完成！")
    logger.info(f"结果保存至: {output_dir}")


if __name__ == "__main__":
    main()