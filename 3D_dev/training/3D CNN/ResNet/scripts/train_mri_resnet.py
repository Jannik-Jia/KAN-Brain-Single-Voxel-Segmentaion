#!/usr/bin/env python3
"""
MRI ResNet Training Script

Training script for ResNet-50 on MRI brain voxel classification.
Supports Leave-One-Out cross-validation and various loss functions
for handling class imbalance.

Usage:
    python train_mri_resnet.py --data_dir /path/to/mat/files --test_subject 1
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

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

from resnet import mri_resnet50, count_parameters, get_model_info
from dataset import MRIBrain2DPatchDataset, create_data_loaders
# Import simplified version for testing
import sys
sys.path.append('../models')
from dataset_simple import MRIBrain2DPatchDatasetSimple
from losses import create_loss_function, mixup_data, MixupLoss

warnings.filterwarnings('ignore')


def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    log_level = logging.INFO if verbose else logging.WARNING
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('MRIResNetTrainer')
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class MRIResNetTrainer:
    """MRI ResNet Trainer with advanced loss functions and optimization"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
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
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model: ResNet model
            train_loader: Training data loader  
            test_loader: Testing data loader
            device: Device for training
            loss_type: Type of loss function
            learning_rate: Learning rate
            weight_decay: Weight decay
            use_mixup: Enable mixup augmentation
            mixup_alpha: Mixup alpha parameter
            use_ema: Use exponential moving average
            ema_decay: EMA decay rate
            gamma: Focal loss gamma parameter
            beta: Class-balanced loss beta parameter
            label_smoothing: Label smoothing factor
            tau: Logit adjustment temperature parameter
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_ema = use_ema
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Get class counts from training dataset
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'class_counts') and train_dataset.class_counts:
            class_counts = torch.tensor([
                train_dataset.class_counts.get(i, 1) for i in range(1, 103)
            ], dtype=torch.float32).to(device)
        else:
            # For simplified dataset, use uniform weights
            class_counts = torch.ones(102, dtype=torch.float32).to(device)
            self.logger.info("Using uniform class weights (simplified dataset mode)")
        
        # Setup loss function
        self.base_criterion = create_loss_function(
            loss_type=loss_type,
            class_counts=class_counts,
            reduction='mean',
            gamma=gamma,
            beta=beta,
            label_smoothing=label_smoothing,
            tau=tau
        ).to(device)
        
        # Keep reference to base criterion for non-mixup cases
        self.criterion = self.base_criterion
        
        # Create mixup criterion if enabled
        if use_mixup:
            self.mixup_criterion = MixupLoss(self.base_criterion)
        else:
            self.mixup_criterion = None
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # EMA model
        if use_ema:
            self.ema_model = self._create_ema_model(ema_decay)
        else:
            self.ema_model = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'test_loss': [],
            'test_f1': [],
            'learning_rates': []
        }
        
        self.logger.info(f"Trainer initialized with {loss_type} loss")
        self.logger.info(f"Model parameters: {count_parameters(model):,}")
        self.logger.info(f"Class distribution: {len(train_dataset.class_counts)} classes")
    
    def _create_ema_model(self, decay: float):
        """Create exponential moving average model"""
        from torch.optim.swa_utils import AveragedModel
        
        def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return decay * averaged_model_parameter + (1 - decay) * model_parameter
        
        return AveragedModel(self.model, avg_fn=ema_avg)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Apply mixup if enabled
            if self.use_mixup and np.random.random() < 0.5:
                mixed_images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.mixup_alpha
                )
                
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(mixed_images)
                    # Use mixup_criterion for mixup loss
                    loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Use original labels for metrics
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            else:
                # Standard training
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    # Use base_criterion for standard loss
                    loss = self.base_criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Collect predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update EMA
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, macro_f1
    
    def evaluate(self, use_ema: bool = False) -> Tuple[float, float, Dict]:
        """Evaluate model"""
        model = self.ema_model if (use_ema and self.ema_model) else self.model
        model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating', leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    # Use standard CE for evaluation
                    loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute comprehensive metrics
        avg_loss = total_loss / len(self.test_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Per-class metrics
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
        Complete training loop
        
        Args:
            epochs: Number of training epochs
            patience: Early stopping patience
            save_best: Save best model
            output_dir: Output directory for saving
        
        Returns:
            Training history
        """
        best_f1 = 0.0
        patience_counter = 0
        best_epoch = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Early stopping patience: {patience}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Set epoch for memory-efficient mode (reshuffles data)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
                
            # Training
            train_loss, train_f1 = self.train_epoch()
            
            # Evaluation
            test_loss, test_f1, detailed_metrics = self.evaluate(use_ema=self.use_ema)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['test_loss'].append(test_loss)
            self.history['test_f1'].append(test_f1)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )
            
            # Early stopping and model saving
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
                    self.logger.info(f"New best F1: {best_f1:.4f} at epoch {epoch+1}")
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    self.logger.info(f"Best F1: {best_f1:.4f} at epoch {best_epoch+1}")
                    break
            
            # Save checkpoint every 10 epochs
            if output_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    output_dir / f'checkpoint_epoch_{epoch+1}.pth',
                    epoch=epoch,
                    metrics=detailed_metrics
                )
        
        # Final evaluation with best model
        if save_best and output_dir and (output_dir / 'best_model.pth').exists():
            self.load_checkpoint(output_dir / 'best_model.pth')
            final_loss, final_f1, final_metrics = self.evaluate(use_ema=self.use_ema)
            
            self.logger.info(f"Final evaluation - Loss: {final_loss:.4f}, F1: {final_f1:.4f}")
            
            # Save detailed results
            results = {
                'best_epoch': best_epoch + 1,
                'best_f1': best_f1,
                'final_f1': final_f1,
                'history': self.history,
                'final_metrics': final_metrics
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
        """Save model checkpoint"""
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
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.ema_model and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint.get('metrics', {})


def load_and_merge_config(args):
    """Load default config and merge with command line arguments"""
    # Path to default config
    script_dir = Path(__file__).parent
    default_config_path = script_dir.parent / 'configs' / 'default_config.json'
    
    # Start with empty config
    config = {}
    
    # Load default config if it exists
    if default_config_path.exists():
        with open(default_config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded default config from: {default_config_path}")
    
    # Load custom config if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            # Recursively merge custom config
            config = merge_configs(config, custom_config)
            print(f"Loaded custom config from: {config_path}")
        else:
            print(f"Warning: Config file not found: {config_path}")
    
    # Apply config values to args, but only if not explicitly set via command line
    # This allows command line args to override config file
    apply_config_to_args(args, config)
    
    return args


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Recursively merge two configuration dictionaries"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def apply_config_to_args(args, config: dict):
    """Apply config values to args object"""
    # Mapping of config sections to argument names
    config_mapping = {
        # Model configuration
        ('model', 'base_width'): 'base_width',
        ('model', 'input_channels'): 'input_channels',
        ('model', 'num_classes'): 'num_classes',
        
        # Data configuration
        ('data', 'patch_size'): 'patch_size',
        ('data', 'samples_per_subject'): 'samples_per_subject',
        ('data', 'balance_classes'): 'balance_classes',
        ('data', 'augmentation'): 'augmentation',
        ('data', 'weighted_sampling'): 'weighted_sampling',
        
        # Training configuration
        ('training', 'epochs'): 'epochs',
        ('training', 'batch_size'): 'batch_size',
        ('training', 'learning_rate'): 'learning_rate',
        ('training', 'weight_decay'): 'weight_decay',
        ('training', 'patience'): 'patience',
        ('training', 'num_workers'): 'num_workers',
        ('training', 'seed'): 'seed',
        
        # Loss configuration
        ('loss', 'type'): 'loss_type',
        ('loss', 'gamma'): 'gamma',
        ('loss', 'beta'): 'beta',
        ('loss', 'label_smoothing'): 'label_smoothing',
        ('loss', 'tau'): 'tau',
        
        # Optimization configuration
        ('optimization', 'gradient_clip'): 'gradient_clip',
        ('optimization', 'mixed_precision'): 'mixed_precision',
        
        # Augmentation configuration
        ('augmentation', 'use_mixup'): 'use_mixup',
        ('augmentation', 'mixup_alpha'): 'mixup_alpha',
        ('augmentation', 'use_ema'): 'use_ema',
        ('augmentation', 'ema_decay'): 'ema_decay',
    }
    
    # Apply config values
    for (section, key), arg_name in config_mapping.items():
        if section in config and key in config[section]:
            config_value = config[section][key]
            
            if hasattr(args, arg_name):
                # Always apply config values - command line args can still override
                setattr(args, arg_name, config_value)
                print(f"Applied config: {arg_name} = {config_value}")


def find_mat_files(data_dir: Path) -> List[Path]:
    """Find all data files in directory - supports both .mat and .h5 formats"""
    # Try zscore normalized .h5 files first (from zscore_dataset_converter.py)
    data_files = sorted(data_dir.glob('*.h5'))

    if len(data_files) == 0:
        # Fallback to original .mat files
        data_files = sorted(data_dir.glob('subject*_3d_validated.mat'))

    if len(data_files) == 0:
        # Try other naming patterns if the first one fails
        data_files = sorted(data_dir.glob('*.mat'))

    if not data_files:
        raise FileNotFoundError(f"No data files (.h5 or .mat) found in {data_dir}")

    return data_files


def create_leave_one_out_split(
    mat_files: List[Path],
    test_subject: int
) -> Tuple[List[Path], List[Path]]:
    """Create Leave-One-Out train/test split"""
    if test_subject < 1 or test_subject > len(mat_files):
        raise ValueError(f"test_subject must be between 1 and {len(mat_files)}")
    
    test_files = [mat_files[test_subject - 1]]
    train_files = mat_files[:test_subject - 1] + mat_files[test_subject:]
    
    return train_files, test_files


def plot_training_history(history: Dict, output_dir: Path):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plots
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['test_loss'], label='Test Loss', color='red')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # F1 Score plots
    ax2.plot(epochs, history['train_f1'], label='Train F1', color='blue')
    ax2.plot(epochs, history['test_f1'], label='Test F1', color='red')
    ax2.set_title('Training and Test F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    ax3.plot(epochs, history['learning_rates'], color='green')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # F1 difference (overfitting indicator)
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
    parser = argparse.ArgumentParser(description='Train MRI ResNet for brain region classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing 3D MAT files')
    parser.add_argument('--test_subject', type=int, required=True,
                       help='Subject number for Leave-One-Out (1-38)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--base_width', type=int, default=104,
                       help='ResNet base width (default: 104 for ~50M params)')
    parser.add_argument('--input_channels', type=int, default=351,
                       help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=102,
                       help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Data arguments
    parser.add_argument('--patch_size', type=int, default=7,
                       help='Patch size')
    parser.add_argument('--samples_per_subject', type=int, default=10000,
                       help='Samples per subject (match 3D CNN baseline)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--balance_classes', action='store_true',
                       help='Enable class balancing in dataset')
    parser.add_argument('--augmentation', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--weighted_sampling', action='store_true',
                       help='Enable weighted random sampling')
    parser.add_argument('--memory_efficient', action='store_true',
                       help='Enable memory-efficient mode (loads all 71M patches)')
    parser.add_argument('--use_simple_dataset', action='store_true',
                       help='Use simplified dataset implementation for performance testing')
    
    # Loss and optimization
    parser.add_argument('--loss_type', type=str, default='cb_focal',
                       choices=['ce', 'weighted_ce', 'focal', 'cb_focal', 'logit_adj', 'balanced_softmax'],
                       help='Loss function type')
    parser.add_argument('--use_mixup', action='store_true',
                       help='Enable mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup alpha parameter')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use exponential moving average')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='Focal loss gamma parameter')
    parser.add_argument('--beta', type=float, default=0.9999,
                       help='Class-balanced loss beta parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    parser.add_argument('--tau', type=float, default=1.0,
                       help='Logit adjustment temperature parameter')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file (overrides default values)')
    
    args = parser.parse_args()
    
    # Load default config and merge with command line args
    args = load_and_merge_config(args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f'resnet_test_subject_{args.test_subject}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.verbose)
    
    logger.info("=== MRI ResNet Training ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Find MAT files
    data_dir = Path(args.data_dir)
    mat_files = find_mat_files(data_dir)
    file_format = "H5 (zscore normalized)" if mat_files[0].suffix == ".h5" else "MAT (original)"
    logger.info(f"Found {len(mat_files)} {file_format} files")
    
    # Create Leave-One-Out split
    train_files, test_files = create_leave_one_out_split(mat_files, args.test_subject)
    logger.info(f"Training subjects: {len(train_files)}")
    logger.info(f"Test subject: {args.test_subject}")
    
    # Create data loaders - with optional simple dataset for performance testing
    if getattr(args, 'use_simple_dataset', False):
        logger.info("Using simplified dataset implementation for performance testing")

        # Use simplified dataset
        train_dataset = MRIBrain2DPatchDatasetSimple(
            mat_files=train_files,
            patch_size=args.patch_size,
            samples_per_subject=args.samples_per_subject,
            is_train=True,
            cache_data=False,
            balance_classes=getattr(args, 'balance_classes', False),
            augmentation=getattr(args, 'augmentation', False),
            memory_efficient=True
        )

        test_dataset = MRIBrain2DPatchDatasetSimple(
            mat_files=test_files,
            patch_size=args.patch_size,
            samples_per_subject=None,
            is_train=False,
            cache_data=True,
            balance_classes=False,
            augmentation=False,
            memory_efficient=False
        )

        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Dataset handles its own shuffling
            num_workers=0,  # Start with single-threaded
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    else:
        # Use original data loaders
        train_loader, test_loader = create_data_loaders(
            train_files=train_files,
            test_files=test_files,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            samples_per_subject=args.samples_per_subject,
            balance_classes=getattr(args, 'balance_classes', False),
            augmentation=getattr(args, 'augmentation', False),
            weighted_sampling=getattr(args, 'weighted_sampling', False),
            memory_efficient=getattr(args, 'memory_efficient', False)
        )
    
    # Log memory mode
    if getattr(args, 'memory_efficient', False):
        logger.info("Using MEMORY-EFFICIENT mode: ~1.2GB memory, 71M patches per epoch")
        logger.info(f"Training dataset size: {len(train_loader.dataset):,} patches")
    else:
        logger.info("Using ORIGINAL mode: cached data, limited patches per subject")
        logger.info(f"Training dataset size: {len(train_loader.dataset):,} patches")
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = mri_resnet50(
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        base_width=args.base_width
    )
    
    # Log model info
    model_info = get_model_info(model, input_size=(1, args.input_channels, args.patch_size, args.patch_size))
    logger.info(f"Model parameters: {model_info['total_parameters']:,}")
    logger.info(f"Model size: {model_info['parameter_size_mb']:.2f} MB")
    
    # Create trainer
    trainer = MRIResNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=args.device,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_ema=args.use_ema,
        gamma=getattr(args, 'gamma', 1.5),
        beta=getattr(args, 'beta', 0.9999),
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
        tau=getattr(args, 'tau', 1.0),
        logger=logger
    )
    
    # Train model
    history = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        save_best=True,
        output_dir=output_dir
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    logger.info("Training completed!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()