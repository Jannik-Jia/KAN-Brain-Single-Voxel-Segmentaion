#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brain Voxel Classification Comparison Script
============================================

This script compares two different approaches for brain voxel classification:
1. Replica approach: Exactly replicates the original brain_voxel.ipynb with one-hot labels
2. Index label approach: Uses index labels with class weights and proper background handling

The script includes comprehensive logging and saves detailed reports for analysis.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, balanced_accuracy_score

# Import existing MAT loader (adjust path as needed)
sys.path.append('.')
try:
    from data.mat_loader import load_mat_data
except ImportError:
    print("Warning: Could not import mat_loader. Please ensure the module is available.")
    def load_mat_data(filepath):
        """Placeholder function - replace with actual implementation"""
        raise NotImplementedError("Please implement or import the actual mat_loader function")


def setup_logging(log_dir="./logs", experiment_name="brain_voxel_comparison"):
    """Setup comprehensive logging system"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Brain Voxel Classification Comparison Started ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    return logger


class ExactReplicaDataset(Dataset):
    """Dataset class that exactly replicates the original notebook logic with one-hot labels"""
    
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)  # Keep one-hot format
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class IndexLabelDataset(Dataset):
    """Dataset class using index labels instead of one-hot"""
    
    def __init__(self, data, labels, ignore_background=True, logger=None):
        self.data = torch.FloatTensor(data)
        self.logger = logger or logging.getLogger(__name__)
        
        # Convert one-hot labels to index labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Input is one-hot format, convert to indices
            self.labels = torch.LongTensor(np.argmax(labels, axis=1))
        else:
            # Already in index format
            self.labels = torch.LongTensor(labels.astype(int))
        
        # Handle background labels if requested
        if ignore_background:
            # Convert label 0 (background) to -1 (ignore index)
            # Convert labels 1-102 to 0-101
            background_mask = self.labels == 0
            self.labels = self.labels - 1  # 1-102 -> 0-101
            self.labels[background_mask] = -1  # background -> -1
        
        self.logger.info(f"Label conversion completed:")
        self.logger.info(f"  Label shape: {self.labels.shape}")
        self.logger.info(f"  Label range: {torch.min(self.labels)} - {torch.max(self.labels)}")
        self.logger.info(f"  Background labels (-1): {torch.sum(self.labels == -1)}")
        self.logger.info(f"  Valid labels: {torch.sum(self.labels >= 0)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ReplicaMLPModel(nn.Module):
    """Exact replica of the original notebook's MLP model architecture"""
    
    def __init__(self, input_dim=341, num_classes=102, dropout_rate=0.5, l2_reg=1e-5):
        super(ReplicaMLPModel, self).__init__()
        
        self.layers = nn.Sequential(
            # Layer 1: 341 -> 4096
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: 4096 -> 4096  
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 4096 -> 4096
            nn.Linear(4096, 4096), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer: 4096 -> 102
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)  # Keep Softmax for replica version
        )
        
        self.l2_reg = l2_reg
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                l2_loss += torch.sum(layer.weight ** 2)
        return self.l2_reg * l2_loss


class IndexLabelMLPModel(nn.Module):
    """MLP model optimized for index labels (no output Softmax)"""
    
    def __init__(self, input_dim=341, num_classes=102, dropout_rate=0.5, l2_reg=1e-5):
        super(IndexLabelMLPModel, self).__init__()
        
        self.layers = nn.Sequential(
            # Layer 1: 341 -> 4096
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: 4096 -> 4096  
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 4096 -> 4096
            nn.Linear(4096, 4096), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer: 4096 -> 102 (no Softmax - handled by CrossEntropyLoss)
            nn.Linear(4096, num_classes)
        )
        
        self.l2_reg = l2_reg
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                l2_loss += torch.sum(layer.weight ** 2)
        return self.l2_reg * l2_loss


def load_and_process_data(mat_file_path, random_state=42, logger=None):
    """Load and process MAT data with exact replica of original notebook logic"""
    logger = logger or logging.getLogger(__name__)
    logger.info("=== Loading and Processing Data (Exact Replica) ===")
    
    # Load MAT file
    logger.info(f"Loading MAT file: {mat_file_path}")
    arrays = load_mat_data(mat_file_path)
    
    # Transpose data (consistent with original notebook)
    train_data = arrays['data'].transpose() if 'data' in arrays else arrays['data'].T
    train_region = arrays['region'].transpose() if 'region' in arrays else arrays['region'].T
    prob_idx = arrays['prob_idx'].transpose().flatten() if 'prob_idx' in arrays else arrays['prob_idx'].flatten()
    
    logger.info(f"Original data shapes:")
    logger.info(f"  train_data: {train_data.shape}")
    logger.info(f"  train_region: {train_region.shape}")
    logger.info(f"  prob_idx: {prob_idx.shape}")
    logger.info(f"  Patient ID range: {np.min(prob_idx)} - {np.max(prob_idx)}")
    
    # Step 1: Split by patient (exact replica of notebook logic)
    curr_set = np.where(prob_idx != 38)[0]  # Patients 1-37
    set_data = train_data[curr_set, :]
    set_region = train_region[curr_set, :]
    logger.info(f"Patients 1-37 data shape: {set_data.shape}")
    
    curr_val = np.where(prob_idx == 38)[0]   # Patient 38
    val_data = train_data[curr_val, :]
    val_label = train_region[curr_val, :]
    logger.info(f"Patient 38 data shape: {val_data.shape}")
    
    # Step 2: Split 1% from patients 1-37 as test set (exact replica)
    X_train1, X_train2, y_train1, y_train2 = train_test_split(
        set_data, set_region, 
        test_size=0.01,  # Exact same as original notebook
        random_state=42  # Exact same as original notebook
    )
    
    logger.info(f"Data split results:")
    logger.info(f"  Training set: {X_train1.shape} (99% of patients 1-37)")
    logger.info(f"  Test set: {X_train2.shape} (1% of patients 1-37)")
    logger.info(f"  Validation set: {val_data.shape} (100% of patient 38)")
    
    # Step 3: Standardization (fit only on training set, exact replica)
    logger.info("Applying standardization...")
    scaler = StandardScaler()
    scaler.fit(X_train1)  # Fit only on training set
    
    X_train1_scaled = scaler.transform(X_train1)
    X_train2_scaled = scaler.transform(X_train2)  # Test set
    val_data_scaled = scaler.transform(val_data)   # Validation set
    
    logger.info("Standardization completed")
    
    # Check label format
    logger.info(f"Label format check:")
    logger.info(f"  Training label shape: {y_train1.shape}")
    logger.info(f"  Label sum (should be 1 for one-hot): {np.sum(y_train1[0])}")
    logger.info(f"  Label range: {np.min(y_train1)} - {np.max(y_train1)}")
    
    return {
        'train_data': X_train1_scaled,
        'train_labels': y_train1,
        'test_data': X_train2_scaled, 
        'test_labels': y_train2,
        'val_data': val_data_scaled,
        'val_labels': val_label,
        'scaler': scaler,
        'feature_dim': X_train1_scaled.shape[1],
        'num_classes': y_train1.shape[1]
    }


def calculate_class_weights(labels, num_classes=102, ignore_index=-1, logger=None):
    """Calculate class weights for handling imbalanced classes"""
    logger = logger or logging.getLogger(__name__)
    
    # Count samples for each class
    class_counts = np.bincount(labels[labels >= 0], minlength=num_classes)
    total_samples = len(labels[labels >= 0])  # Exclude background labels
    
    # Calculate weights
    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            weights[i] = 0.0
    
    logger.info(f"Class weights calculated:")
    logger.info(f"  Weight range: {np.min(weights[weights > 0]):.4f} - {np.max(weights):.4f}")
    logger.info(f"  Zero weight classes: {np.sum(weights == 0)}")
    
    return torch.FloatTensor(weights)


def train_model(data_dict, model_type='replica', device='cuda:0', save_path='./results', logger=None):
    """Train model with comprehensive logging and monitoring"""
    logger = logger or logging.getLogger(__name__)
    logger.info(f"\n=== Starting Training ({model_type.upper()} version) ===")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Training configuration
    config = {
        'batch_size': 128,
        'num_epochs': 25,
        'learning_rate': 1e-5,
        'num_classes': 102,
        'dropout_rate': 0.5,
        'l2_reg': 1e-5
    }
    
    logger.info(f"Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  device: {device}")
    
    # Create datasets
    if model_type == 'replica':
        train_dataset = ExactReplicaDataset(data_dict['train_data'], data_dict['train_labels'])
        val_dataset = ExactReplicaDataset(data_dict['val_data'], data_dict['val_labels'])
        model = ReplicaMLPModel(
            input_dim=data_dict['feature_dim'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            l2_reg=config['l2_reg']
        )
        criterion = nn.CrossEntropyLoss()
        class_weights = None
    else:  # index_label
        train_dataset = IndexLabelDataset(data_dict['train_data'], data_dict['train_labels'], 
                                        ignore_background=True, logger=logger)
        val_dataset = IndexLabelDataset(data_dict['val_data'], data_dict['val_labels'], 
                                      ignore_background=True, logger=logger)
        model = IndexLabelMLPModel(
            input_dim=data_dict['feature_dim'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            l2_reg=config['l2_reg']
        )
        # Calculate class weights for imbalanced classes
        train_labels_index = train_dataset.labels.numpy()
        class_weights = calculate_class_weights(train_labels_index, config['num_classes'], logger=logger)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=-1)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    
    # Initialize model and optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training history
    history = {
        'train_losses': [], 'train_accuracies': [], 'train_f1_scores': [],
        'val_losses': [], 'val_accuracies': [], 'val_f1_scores': [],
        'val_kappa_scores': [], 'val_balanced_acc_scores': []
    }
    
    logger.info(f"\nStarting training...")
    start_time = time.time()
    
    best_f1 = 0.0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        all_train_preds = []
        all_train_true = []
        num_train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate loss
            if model_type == 'replica':
                classification_loss = criterion(output, target)
            else:
                classification_loss = criterion(output, target)
            
            l2_loss = model.get_l2_loss()
            total_loss = classification_loss + l2_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_train_loss += total_loss.item()
            
            if model_type == 'replica':
                pred = torch.argmax(output, dim=1)
                true = torch.argmax(target, dim=1)
                all_train_preds.extend(pred.cpu().numpy())
                all_train_true.extend(true.cpu().numpy())
            else:
                pred = torch.argmax(output, dim=1)
                # Only evaluate non-background pixels
                valid_mask = target != -1
                if valid_mask.sum() > 0:
                    all_train_preds.extend(pred[valid_mask].cpu().numpy())
                    all_train_true.extend(target[valid_mask].cpu().numpy())
            
            num_train_batches += 1
        
        # Calculate training metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        if len(all_train_true) > 0:
            train_acc = accuracy_score(all_train_true, all_train_preds)
            train_f1 = f1_score(all_train_true, all_train_preds, average='macro')
        else:
            train_acc = train_f1 = 0.0
        
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(train_acc)
        history['train_f1_scores'].append(train_f1)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_true = []
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if model_type == 'replica':
                    classification_loss = criterion(output, target)
                else:
                    classification_loss = criterion(output, target)
                
                l2_loss = model.get_l2_loss()
                total_loss = classification_loss + l2_loss
                
                epoch_val_loss += total_loss.item()
                
                pred = torch.argmax(output, dim=1)
                
                if model_type == 'replica':
                    true = torch.argmax(target, dim=1)
                    all_val_preds.extend(pred.cpu().numpy())
                    all_val_true.extend(true.cpu().numpy())
                else:
                    # Only evaluate non-background pixels
                    valid_mask = target != -1
                    if valid_mask.sum() > 0:
                        all_val_preds.extend(pred[valid_mask].cpu().numpy())
                        all_val_true.extend(target[valid_mask].cpu().numpy())
                
                num_val_batches += 1
        
        # Calculate validation metrics
        avg_val_loss = epoch_val_loss / num_val_batches
        if len(all_val_true) > 0:
            val_acc = accuracy_score(all_val_true, all_val_preds)
            val_f1 = f1_score(all_val_true, all_val_preds, average='macro')
            val_kappa = cohen_kappa_score(all_val_true, all_val_preds)
            val_balanced_acc = balanced_accuracy_score(all_val_true, all_val_preds)
        else:
            val_acc = val_f1 = val_kappa = val_balanced_acc = 0.0
        
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_acc)
        history['val_f1_scores'].append(val_f1)
        history['val_kappa_scores'].append(val_kappa)
        history['val_balanced_acc_scores'].append(val_balanced_acc)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")
        logger.info(f"  Train - Loss: {avg_train_loss:.6f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"  Val   - Loss: {avg_val_loss:.6f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        if model_type == 'index_label':
            logger.info(f"  Val   - Kappa: {val_kappa:.4f}, Balanced Acc: {val_balanced_acc:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = os.path.join(save_path, f'best_{model_type}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config,
                'scaler': data_dict['scaler'],
                'class_weights': class_weights.cpu() if class_weights is not None else None
            }, best_model_path)
            logger.info(f"  -> Saved best model (F1: {val_f1:.4f})")
    
    training_time = time.time() - start_time
    logger.info(f"\nTraining completed! Total time: {training_time:.2f} seconds")
    
    # Final results
    best_f1 = max(history['val_f1_scores'])
    best_epoch = history['val_f1_scores'].index(best_f1) + 1
    best_acc = history['val_accuracies'][history['val_f1_scores'].index(best_f1)]
    
    logger.info(f"\n=== Final Results ({model_type.upper()}) ===")
    logger.info(f"Best validation F1: {best_f1:.4f} (epoch {best_epoch})")
    logger.info(f"Corresponding accuracy: {best_acc:.4f}")
    
    if model_type == 'index_label':
        best_kappa = history['val_kappa_scores'][history['val_f1_scores'].index(best_f1)]
        best_balanced_acc = history['val_balanced_acc_scores'][history['val_f1_scores'].index(best_f1)]
        logger.info(f"Corresponding Kappa: {best_kappa:.4f}")
        logger.info(f"Corresponding Balanced Acc: {best_balanced_acc:.4f}")
        history['best_kappa'] = best_kappa
        history['best_balanced_acc'] = best_balanced_acc
    
    # Update history with final results
    history.update({
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_acc': best_acc,
        'training_time': training_time,
        'config': config
    })
    
    # Save training history
    history_path = os.path.join(save_path, f'{model_type}_training_history.npy')
    np.save(history_path, history)
    
    return history, model


def plot_training_curves(history, model_type, save_path):
    """Plot comprehensive training curves"""
    plt.figure(figsize=(20, 8))
    
    # Loss curves
    plt.subplot(2, 4, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(2, 4, 2)
    plt.plot(history['train_accuracies'], label='Train Acc')
    plt.plot(history['val_accuracies'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1 score curves
    plt.subplot(2, 4, 3)
    plt.plot(history['train_f1_scores'], label='Train F1')
    plt.plot(history['val_f1_scores'], label='Val F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    if model_type == 'index_label':
        # Kappa score
        plt.subplot(2, 4, 4)
        plt.plot(history['val_kappa_scores'], label='Val Kappa', color='purple')
        plt.title('Validation Kappa Score')
        plt.xlabel('Epoch')
        plt.ylabel('Kappa Score')
        plt.legend()
        plt.grid(True)
        
        # Balanced accuracy
        plt.subplot(2, 4, 5)
        plt.plot(history['val_balanced_acc_scores'], label='Val Balanced Acc', color='orange')
        plt.title('Validation Balanced Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Balanced Accuracy')
        plt.legend()
        plt.grid(True)
    
    # Training vs Validation F1 difference
    plt.subplot(2, 4, 6)
    train_val_diff = [t - v for t, v in zip(history['train_f1_scores'], history['val_f1_scores'])]
    plt.plot(train_val_diff, label='Train F1 - Val F1', color='red')
    plt.title('Training vs Validation F1 Difference')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Difference')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_type}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_report(history, model_type, data_dict, save_path, logger):
    """Save comprehensive training report"""
    report_path = os.path.join(save_path, f'{model_type}_detailed_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Training set: {data_dict['train_data'].shape}\n")
        f.write(f"  Validation set: {data_dict['val_data'].shape}\n")
        f.write(f"  Test set: {data_dict['test_data'].shape}\n")
        f.write(f"  Feature dimension: {data_dict['feature_dim']}\n")
        f.write(f"  Number of classes: {data_dict['num_classes']}\n\n")
        
        f.write("Model Configuration:\n")
        for key, value in history['config'].items():
            f.write(f"  {key}: {value}\n")
        f.write(f"  Model type: {model_type}\n")
        f.write("  Architecture: 4x4096 MLP\n\n")
        
        f.write("Training Results:\n")
        f.write(f"  Best validation F1: {history['best_f1']:.6f}\n")
        f.write(f"  Best validation accuracy: {history['best_acc']:.6f}\n")
        f.write(f"  Best epoch: {history['best_epoch']}\n")
        f.write(f"  Training time: {history['training_time']:.2f} seconds\n")
        
        if model_type == 'index_label':
            f.write(f"  Best Kappa score: {history['best_kappa']:.6f}\n")
            f.write(f"  Best balanced accuracy: {history['best_balanced_acc']:.6f}\n")
        
        f.write("\nKey Features:\n")
        if model_type == 'replica':
            f.write("  - Exact replica of original notebook\n")
            f.write("  - One-hot label format\n")
            f.write("  - Softmax output layer\n")
            f.write("  - Standard CrossEntropyLoss\n")
        else:
            f.write("  - Index label format (0-101, background=-1)\n")
            f.write("  - Class weights for imbalanced data\n")
            f.write("  - CrossEntropyLoss with ignore_index=-1\n")
            f.write("  - No output Softmax (handled by loss function)\n")
        
        f.write(f"\nFinal Training Loss: {history['train_losses'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {history['val_losses'][-1]:.6f}\n")
        f.write(f"Final Training F1: {history['train_f1_scores'][-1]:.6f}\n")
        f.write(f"Final Validation F1: {history['val_f1_scores'][-1]:.6f}\n")
    
    logger.info(f"Detailed report saved to: {report_path}")


def compare_results(replica_history, index_history, save_path, logger):
    """Compare results between replica and index label approaches"""
    logger.info(f"\n=== COMPARISON ANALYSIS ===")
    
    logger.info(f"Replica (One-hot) Results:")
    logger.info(f"  Best F1: {replica_history['best_f1']:.4f}")
    logger.info(f"  Best Accuracy: {replica_history['best_acc']:.4f}")
    logger.info(f"  Best Epoch: {replica_history['best_epoch']}")
    logger.info(f"  Training Time: {replica_history['training_time']:.2f}s")
    
    logger.info(f"\nIndex Label Results:")
    logger.info(f"  Best F1: {index_history['best_f1']:.4f}")
    logger.info(f"  Best Accuracy: {index_history['best_acc']:.4f}")
    logger.info(f"  Best Kappa: {index_history['best_kappa']:.4f}")
    logger.info(f"  Best Balanced Acc: {index_history['best_balanced_acc']:.4f}")
    logger.info(f"  Best Epoch: {index_history['best_epoch']}")
    logger.info(f"  Training Time: {index_history['training_time']:.2f}s")
    
    # Calculate differences
    f1_diff = index_history['best_f1'] - replica_history['best_f1']
    acc_diff = index_history['best_acc'] - replica_history['best_acc']
    time_diff = index_history['training_time'] - replica_history['training_time']
    
    logger.info(f"\nPerformance Differences (Index - Replica):")
    logger.info(f"  F1 Score: {f1_diff:+.4f}")
    logger.info(f"  Accuracy: {acc_diff:+.4f}")
    logger.info(f"  Time: {time_diff:+.2f}s")
    
    # Analysis
    logger.info(f"\nAnalysis:")
    if abs(f1_diff) < 0.01:
        logger.info("  -> Difference is minimal, label format has little impact")
    elif f1_diff > 0:
        logger.info("  -> Index label approach performs better")
    else:
        logger.info("  -> One-hot approach performs better")
    
    # Save comparison report
    comparison_path = os.path.join(save_path, 'comparison_report.txt')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("Brain Voxel Classification Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("REPLICA (ONE-HOT) APPROACH:\n")
        f.write(f"  Best F1 Score: {replica_history['best_f1']:.6f}\n")
        f.write(f"  Best Accuracy: {replica_history['best_acc']:.6f}\n")
        f.write(f"  Best Epoch: {replica_history['best_epoch']}\n")
        f.write(f"  Training Time: {replica_history['training_time']:.2f}s\n\n")
        
        f.write("INDEX LABEL APPROACH:\n")
        f.write(f"  Best F1 Score: {index_history['best_f1']:.6f}\n")
        f.write(f"  Best Accuracy: {index_history['best_acc']:.6f}\n")
        f.write(f"  Best Kappa Score: {index_history['best_kappa']:.6f}\n")
        f.write(f"  Best Balanced Accuracy: {index_history['best_balanced_acc']:.6f}\n")
        f.write(f"  Best Epoch: {index_history['best_epoch']}\n")
        f.write(f"  Training Time: {index_history['training_time']:.2f}s\n\n")
        
        f.write("PERFORMANCE DIFFERENCES (Index - Replica):\n")
        f.write(f"  F1 Score Difference: {f1_diff:+.6f}\n")
        f.write(f"  Accuracy Difference: {acc_diff:+.6f}\n")
        f.write(f"  Time Difference: {time_diff:+.2f}s\n\n")
        
        f.write("KEY DIFFERENCES:\n")
        f.write("  Replica Approach:\n")
        f.write("    - One-hot encoded labels\n")
        f.write("    - Softmax output layer\n")
        f.write("    - Standard CrossEntropyLoss\n")
        f.write("    - May include background in training\n\n")
        f.write("  Index Label Approach:\n")
        f.write("    - Index labels (0-101, background=-1)\n")
        f.write("    - Class weights for imbalanced data\n")
        f.write("    - CrossEntropyLoss with ignore_index=-1\n")
        f.write("    - Explicit background handling\n")
        f.write("    - No output layer Softmax\n\n")
        
        if abs(f1_diff) < 0.01:
            f.write("CONCLUSION: Label format has minimal impact on performance.\n")
        elif f1_diff > 0:
            f.write("CONCLUSION: Index label approach shows better performance.\n")
        else:
            f.write("CONCLUSION: One-hot approach shows better performance.\n")
    
    logger.info(f"Comparison report saved to: {comparison_path}")
    
    # Create comparison visualization
    create_comparison_plot(replica_history, index_history, save_path)


def create_comparison_plot(replica_history, index_history, save_path):
    """Create comparison visualization"""
    plt.figure(figsize=(16, 10))
    
    # F1 Score comparison
    plt.subplot(2, 3, 1)
    plt.plot(replica_history['val_f1_scores'], label='Replica (One-hot)', linewidth=2)
    plt.plot(index_history['val_f1_scores'], label='Index Label', linewidth=2)
    plt.title('Validation F1 Score Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Accuracy comparison
    plt.subplot(2, 3, 2)
    plt.plot(replica_history['val_accuracies'], label='Replica (One-hot)', linewidth=2)
    plt.plot(index_history['val_accuracies'], label='Index Label', linewidth=2)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss comparison
    plt.subplot(2, 3, 3)
    plt.plot(replica_history['val_losses'], label='Replica (One-hot)', linewidth=2)
    plt.plot(index_history['val_losses'], label='Index Label', linewidth=2)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Best metrics bar chart
    plt.subplot(2, 3, 4)
    methods = ['Replica\n(One-hot)', 'Index\nLabel']
    f1_scores = [replica_history['best_f1'], index_history['best_f1']]
    accuracies = [replica_history['best_acc'], index_history['best_acc']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    plt.bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8)
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Best Performance Comparison')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (f1, acc) in enumerate(zip(f1_scores, accuracies)):
        plt.text(i - width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    plt.subplot(2, 3, 5)
    times = [replica_history['training_time'], index_history['training_time']]
    colors = ['skyblue', 'lightcoral']
    bars = plt.bar(methods, times, color=colors, alpha=0.8)
    plt.xlabel('Method')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time:.1f}s', ha='center', va='bottom')
    
    # Performance difference over epochs
    plt.subplot(2, 3, 6)
    f1_diff = [idx - rep for idx, rep in zip(index_history['val_f1_scores'], 
                                           replica_history['val_f1_scores'])]
    plt.plot(f1_diff, label='Index - Replica F1', color='green', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score Difference')
    plt.title('Performance Difference Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    # Setup
    logger = setup_logging()
    
    # Configuration
    config = {
        'mat_file_path': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat",
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'base_save_path': './brain_voxel_results',
        'run_replica': True,
        'run_index_label': True,
        'run_comparison': True
    }
    
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Check if MAT file exists
    if not os.path.exists(config['mat_file_path']):
        logger.error(f"MAT file not found: {config['mat_file_path']}")
        return
    
    try:
        # Load and process data
        logger.info("\n" + "="*60)
        logger.info("LOADING AND PROCESSING DATA")
        logger.info("="*60)
        
        data_dict = load_and_process_data(config['mat_file_path'], logger=logger)
        
        results = {}
        
        # Run replica training
        if config['run_replica']:
            logger.info("\n" + "="*60)
            logger.info("REPLICA TRAINING (ONE-HOT LABELS)")
            logger.info("="*60)
            
            replica_save_path = os.path.join(config['base_save_path'], 'replica_results')
            replica_history, replica_model = train_model(
                data_dict, 
                model_type='replica', 
                device=config['device'], 
                save_path=replica_save_path, 
                logger=logger
            )
            
            plot_training_curves(replica_history, 'replica', replica_save_path)
            save_detailed_report(replica_history, 'replica', data_dict, replica_save_path, logger)
            results['replica'] = replica_history
        
        # Run index label training
        if config['run_index_label']:
            logger.info("\n" + "="*60)
            logger.info("INDEX LABEL TRAINING")
            logger.info("="*60)
            
            index_save_path = os.path.join(config['base_save_path'], 'index_label_results')
            index_history, index_model = train_model(
                data_dict, 
                model_type='index_label', 
                device=config['device'], 
                save_path=index_save_path, 
                logger=logger
            )
            
            plot_training_curves(index_history, 'index_label', index_save_path)
            save_detailed_report(index_history, 'index_label', data_dict, index_save_path, logger)
            results['index_label'] = index_history
        
        # Run comparison
        if config['run_comparison'] and len(results) == 2:
            logger.info("\n" + "="*60)
            logger.info("COMPARISON ANALYSIS")
            logger.info("="*60)
            
            compare_results(
                results['replica'], 
                results['index_label'], 
                config['base_save_path'], 
                logger
            )
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        logger.info("Results Summary:")
        for approach, history in results.items():
            logger.info(f"  {approach.upper()}:")
            logger.info(f"    Best F1: {history['best_f1']:.4f}")
            logger.info(f"    Best Accuracy: {history['best_acc']:.4f}")
            logger.info(f"    Training Time: {history['training_time']:.2f}s")
        
        logger.info(f"\nAll results saved to: {config['base_save_path']}")
        logger.info("Check the generated reports and plots for detailed analysis!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()