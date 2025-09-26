#!/usr/bin/env python3
"""
38-Fold Cross Validation Training Script
Based on alex's torch implementation with enhanced metrics monitoring
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (classification_report, f1_score,
                           precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Setup logging
def setup_logging(log_dir, fold=None):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fold is not None:
        log_file = os.path.join(log_dir, f"train_fold{fold}_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file

# Model Definition
class RegModel(nn.Module):
    """Dense neural network model (4x4096 architecture)"""
    def __init__(self, input_dim=351, num_classes=102):
        super(RegModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

def kernel_l2_regularization(model, weight_decay=0.00001):
    """L2 regularization for weights only (not biases)"""
    l2_reg = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            l2_reg += torch.norm(param, p=2) ** 2
    return weight_decay * l2_reg

# Data Loading Functions
def load_subject_data(filepath):
    """Load single subject's 1D data"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for k in f.keys():
            if not k.startswith('#'):
                v = f[k][()]
                if k == 'multidim_data' and v.shape[0] == 351:
                    v = v.T  # (351, n_voxels) -> (n_voxels, 351)
                elif k == 'seg_one_hot' and v.shape[0] == 102:
                    v = v.T  # (102, n_voxels) -> (n_voxels, 102)
                elif k == 'region_seg':
                    v = v.flatten()
                data[k] = v
    return data

def load_all_subjects(dataset_index_path, data_dir=None):
    """Load all subjects' data"""
    with open(dataset_index_path, 'r') as f:
        dataset_index = json.load(f)

    all_subjects = {}

    logging.info(f"Loading {len(dataset_index)} subjects...")
    for idx in tqdm(sorted(dataset_index.keys(), key=lambda x: int(x)), desc="Loading subjects"):
        subject = dataset_index[idx]
        filepath = subject['filepath']

        if data_dir:
            filename = subject['filename']
            filepath = os.path.join(data_dir, filename)

        try:
            subject_data = load_subject_data(filepath)
            all_subjects[int(idx)] = {
                'data': subject_data.get('multidim_data'),
                'labels': subject_data.get('seg_one_hot'),
                'subject_id': subject['subject_id'],
                'prob_idx': subject['prob_idx']
            }
            logging.debug(f"Loaded subject {idx}: {subject['subject_id']}, shape: {subject_data.get('multidim_data').shape}")
        except Exception as e:
            logging.error(f"Failed to load subject {idx}: {subject['subject_id']}, error: {e}")

    logging.info(f"Successfully loaded {len(all_subjects)} subjects")
    return all_subjects

def prepare_fold_data(all_subjects_data, test_subject_id, val_subject_id=None, train_ratio=0.99,
                     patient_wise_zscore=True):
    """
    Prepare data for one fold

    Args:
        all_subjects_data: All subjects data dictionary
        test_subject_id: Subject ID for test set (1-38)
        val_subject_id: Optional validation subject ID. If None, split from training data
        train_ratio: Ratio for train/val split when val_subject_id is None
        patient_wise_zscore: If True, apply z-score normalization per patient
    """
    # Collect training data with patient-wise z-score
    train_data_list = []
    train_labels_list = []

    for subject_id, subject_data in all_subjects_data.items():
        if subject_id != test_subject_id and subject_id != val_subject_id:
            if subject_data['data'] is not None and subject_data['labels'] is not None:
                subject_features = subject_data['data'].copy()

                # Apply patient-wise z-score normalization if requested
                if patient_wise_zscore:
                    scaler_patient = StandardScaler()
                    subject_features = scaler_patient.fit_transform(subject_features)
                    logging.debug(f"Applied patient-wise z-score for subject {subject_id}")

                train_data_list.append(subject_features)
                train_labels_list.append(subject_data['labels'])

    # Merge training data
    X_train_all = np.vstack(train_data_list)
    y_train_all = np.vstack(train_labels_list)

    # Handle validation set
    if val_subject_id is None:
        # Split from training data (for backward compatibility with original notebook)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all,
            train_size=train_ratio,
            random_state=42
        )
    else:
        # Use specific subject as validation
        X_train = X_train_all
        y_train = y_train_all
        X_val = all_subjects_data[val_subject_id]['data'].copy()
        y_val = all_subjects_data[val_subject_id]['labels']

        if patient_wise_zscore:
            scaler_val = StandardScaler()
            X_val = scaler_val.fit_transform(X_val)

    # Get test data
    X_test = all_subjects_data[test_subject_id]['data'].copy()
    y_test = all_subjects_data[test_subject_id]['labels']

    # Apply patient-wise z-score to test data
    if patient_wise_zscore:
        scaler_test = StandardScaler()
        X_test = scaler_test.fit_transform(X_test)
        logging.debug(f"Applied patient-wise z-score for test subject {test_subject_id}")

    # If not using patient-wise z-score, apply global standardization (original behavior)
    if not patient_wise_zscore:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        logging.info("Applied global standardization across all training data")
    else:
        scaler = None  # No global scaler when using patient-wise z-score
        logging.info("Using patient-wise z-score normalization")

    logging.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logging.info(f"Training with {len(train_data_list)} subjects, testing on subject {test_subject_id}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# Metrics Calculation
def calculate_detailed_metrics(y_true, y_pred, num_classes=102):
    """Calculate detailed classification metrics"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(num_classes)),
        zero_division=0
    )

    metrics = {
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'macro_avg': {
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1)
        },
        'micro_avg': {},
        'weighted_avg': {}
    }

    # Micro average
    micro_metrics = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['micro_avg'] = {
        'precision': micro_metrics[0],
        'recall': micro_metrics[1],
        'f1': micro_metrics[2]
    }

    # Weighted average
    weighted_metrics = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_avg'] = {
        'precision': weighted_metrics[0],
        'recall': weighted_metrics[1],
        'f1': weighted_metrics[2]
    }

    # Gross accuracy
    metrics['gross_accuracy'] = np.mean(y_true == y_pred)

    return metrics

def log_detailed_metrics(metrics, epoch=None, dataset_name='Test'):
    """Log detailed classification metrics"""
    if epoch is not None:
        logging.info(f"=== Epoch {epoch} - {dataset_name} Set Detailed Metrics ===")
    else:
        logging.info(f"=== {dataset_name} Set Detailed Metrics ===")

    logging.info(f"Gross Accuracy (Overall): {metrics['gross_accuracy']:.4f}")
    logging.info(f"Macro Average - P: {metrics['macro_avg']['precision']:.4f}, "
                f"R: {metrics['macro_avg']['recall']:.4f}, F1: {metrics['macro_avg']['f1']:.4f}")
    logging.info(f"Micro Average - P: {metrics['micro_avg']['precision']:.4f}, "
                f"R: {metrics['micro_avg']['recall']:.4f}, F1: {metrics['micro_avg']['f1']:.4f}")
    logging.info(f"Weighted Average - P: {metrics['weighted_avg']['precision']:.4f}, "
                f"R: {metrics['weighted_avg']['recall']:.4f}, F1: {metrics['weighted_avg']['f1']:.4f}")

    zero_support_classes = [i for i, s in enumerate(metrics['per_class']['support']) if s == 0]
    if zero_support_classes:
        logging.warning(f"Classes with no samples (support=0): {zero_support_classes}")

# Training Function
def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                device, batch_size=128, no_epochs=25, learning_rate=0.00001,
                log_interval=5, use_amp=False, num_workers=4):
    """
    Train model with detailed metrics monitoring

    Args:
        use_amp: Use automatic mixed precision for faster training
        num_workers: Number of workers for data loading
    """

    # Check if model is DataParallel
    is_data_parallel = isinstance(model, DataParallel)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device if not is_data_parallel else 'cuda')
    y_train_tensor = torch.FloatTensor(y_train).to(device if not is_data_parallel else 'cuda')
    X_val_tensor = torch.FloatTensor(X_val).to(device if not is_data_parallel else 'cuda')
    y_val_tensor = torch.FloatTensor(y_val).to(device if not is_data_parallel else 'cuda')
    X_test_tensor = torch.FloatTensor(X_test).to(device if not is_data_parallel else 'cuda')
    y_test_tensor = torch.FloatTensor(y_test).to(device if not is_data_parallel else 'cuda')

    # Create data loader with pin_memory for faster GPU transfer
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if not is_data_parallel else 0,  # DataParallel doesn't work well with multiple workers
        pin_memory=True if device.type == 'cuda' else False
    )

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Setup mixed precision training if requested
    scaler = GradScaler() if use_amp else None

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': [],
        'test_detailed_metrics': []
    }

    # Training loop
    for epoch in range(no_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            target_indices = torch.argmax(target, dim=1)

            if use_amp:
                # Mixed precision training
                with autocast():
                    output = model(data)
                    base_loss = criterion(output, target_indices)
                    l2_reg = kernel_l2_regularization(model, weight_decay=0.00001)
                    total_loss = base_loss + l2_reg

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                output = model(data)
                base_loss = criterion(output, target_indices)
                l2_reg = kernel_l2_regularization(model, weight_decay=0.00001)
                total_loss = base_loss + l2_reg

                total_loss.backward()
                optimizer.step()

            epoch_train_loss += total_loss.item()

            _, predicted = torch.max(output.data, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(target_indices.cpu().numpy())

        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_labels))
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            # Validation set
            val_output = model(X_val_tensor)
            val_target_indices = torch.argmax(y_val_tensor, dim=1)
            val_base_loss = criterion(val_output, val_target_indices)
            val_l2_reg = kernel_l2_regularization(model, weight_decay=0.00001)
            val_total_loss = val_base_loss + val_l2_reg

            _, val_predicted = torch.max(val_output.data, 1)
            val_acc = (val_predicted == val_target_indices).float().mean().item()
            val_f1 = f1_score(val_target_indices.cpu().numpy(), val_predicted.cpu().numpy(),
                             average='macro', zero_division=0)

            # Test set
            test_output = model(X_test_tensor)
            test_target_indices = torch.argmax(y_test_tensor, dim=1)
            test_base_loss = criterion(test_output, test_target_indices)
            test_l2_reg = kernel_l2_regularization(model, weight_decay=0.00001)
            test_total_loss = test_base_loss + test_l2_reg

            _, test_predicted = torch.max(test_output.data, 1)
            test_labels_np = test_target_indices.cpu().numpy()
            test_preds_np = test_predicted.cpu().numpy()

            # Calculate detailed test metrics
            test_detailed = calculate_detailed_metrics(test_labels_np, test_preds_np)
            test_acc = test_detailed['gross_accuracy']
            test_f1 = test_detailed['macro_avg']['f1']

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_total_loss.item())
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['test_loss'].append(test_total_loss.item())
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        history['test_detailed_metrics'].append(test_detailed)

        # Log progress
        if epoch % 1 == 0 or epoch == no_epochs - 1:
            logging.info(f"Epoch {epoch+1}/{no_epochs}:")
            logging.info(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Macro-F1: {train_f1:.4f}")
            logging.info(f"  Val   - Loss: {val_total_loss.item():.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
            logging.info(f"  Test  - Loss: {test_total_loss.item():.4f}, Gross Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")

            # Log detailed metrics at intervals
            if epoch == 0 or epoch == no_epochs - 1 or epoch % log_interval == 0:
                log_detailed_metrics(test_detailed, epoch+1, 'Test')

    return history

# ONNX Export
def export_to_onnx(model, input_dim, save_path, device):
    """Export model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    logging.info(f"Model exported to ONNX: {save_path}")

# Main training functions
def train_single_fold(args, all_subjects_data):
    """Train a single fold"""
    # Setup device and multi-GPU if available
    if args.cpu:
        device = torch.device('cpu')
        use_multi_gpu = False
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Check for multiple GPUs
        n_gpus = torch.cuda.device_count()
        use_multi_gpu = n_gpus > 1 and not args.no_multi_gpu

    logging.info(f"Using device: {device}")
    if use_multi_gpu:
        logging.info(f"Using {n_gpus} GPUs with DataParallel")
        # A6000 has 48GB memory each, we can use larger batch size
        if args.batch_size == 128:  # If using default, scale up
            args.batch_size = 256 * n_gpus
            logging.info(f"Auto-scaled batch size to {args.batch_size} for {n_gpus} GPUs")

    # Prepare data
    logging.info(f"Preparing data for fold {args.fold} (test subject: {args.fold})")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_fold_data(
        all_subjects_data,
        test_subject_id=args.fold,
        val_subject_id=None,
        train_ratio=0.99,
        patient_wise_zscore=True  # Use patient-wise z-score normalization
    )

    # Create model
    model = RegModel(input_dim=351)

    if use_multi_gpu:
        # Wrap model with DataParallel for multi-GPU training
        model = model.cuda()
        model = DataParallel(model)
        logging.info(f"Model wrapped with DataParallel for {n_gpus} GPUs")
    else:
        model = model.to(device)

    logging.info("Model created")

    # Train model
    logging.info("Starting training...")
    history = train_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        device=device,
        batch_size=args.batch_size,
        no_epochs=args.epochs,
        learning_rate=args.lr,
        log_interval=args.log_interval,
        use_amp=args.use_amp,  # Use mixed precision if specified
        num_workers=args.num_workers
    )

    # Save model
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save PyTorch model
    model_path = os.path.join(model_dir, f'model_fold{args.fold}.pth')
    # Handle DataParallel wrapper when saving
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved: {model_path}")

    # Export to ONNX
    onnx_path = os.path.join(model_dir, f'model_fold{args.fold}.onnx')
    # Use the underlying model for ONNX export if using DataParallel
    export_model = model.module if isinstance(model, DataParallel) else model
    export_to_onnx(export_model, 351, onnx_path, device)

    # Save training history
    history_dir = os.path.join(args.output_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    history_to_save = {k: v for k, v in history.items() if k != 'test_detailed_metrics'}
    final_metrics = history['test_detailed_metrics'][-1]
    history_to_save['final_metrics'] = {
        'gross_accuracy': final_metrics['gross_accuracy'],
        'macro_f1': final_metrics['macro_avg']['f1'],
        'macro_precision': final_metrics['macro_avg']['precision'],
        'macro_recall': final_metrics['macro_avg']['recall'],
        'micro_f1': final_metrics['micro_avg']['f1'],
        'weighted_f1': final_metrics['weighted_avg']['f1']
    }

    history_path = os.path.join(history_dir, f'history_fold{args.fold}.json')
    with open(history_path, 'w') as f:
        json.dump(history_to_save, f, indent=2)
    logging.info(f"Training history saved: {history_path}")

    # Log final results
    logging.info("=" * 50)
    logging.info("Training completed!")
    logging.info(f"Final Test Results for Fold {args.fold}:")
    logging.info(f"  Gross Accuracy: {final_metrics['gross_accuracy']:.4f}")
    logging.info(f"  Macro F1: {final_metrics['macro_avg']['f1']:.4f}")
    logging.info(f"  Loss: {history['test_loss'][-1]:.4f}")

    return history

def train_all_folds(args, all_subjects_data):
    """Train all 38 folds"""
    results = []

    for fold_id in range(1, 39):
        logging.info("=" * 70)
        logging.info(f"Starting Fold {fold_id}/38")
        logging.info("=" * 70)

        try:
            # Update fold in args
            args.fold = fold_id

            # Train single fold
            history = train_single_fold(args, all_subjects_data)

            # Save results
            final_metrics = history['test_detailed_metrics'][-1]
            fold_result = {
                'fold': fold_id,
                'subject_id': all_subjects_data[fold_id]['subject_id'],
                'test_accuracy': final_metrics['gross_accuracy'],
                'test_f1': final_metrics['macro_avg']['f1'],
                'test_loss': history['test_loss'][-1],
                'val_accuracy': history['val_acc'][-1],
                'val_f1': history['val_f1'][-1],
                'val_loss': history['val_loss'][-1]
            }
            results.append(fold_result)

        except Exception as e:
            logging.error(f"Failed to train fold {fold_id}: {e}")
            results.append({
                'fold': fold_id,
                'error': str(e)
            })

    # Calculate and log summary statistics
    valid_results = [r for r in results if 'test_accuracy' in r]
    if valid_results:
        avg_accuracy = np.mean([r['test_accuracy'] for r in valid_results])
        std_accuracy = np.std([r['test_accuracy'] for r in valid_results])
        avg_f1 = np.mean([r['test_f1'] for r in valid_results])
        std_f1 = np.std([r['test_f1'] for r in valid_results])

        logging.info("=" * 70)
        logging.info("38-Fold Cross Validation Complete!")
        logging.info(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        logging.info(f"Average Macro-F1: {avg_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"Successful folds: {len(valid_results)}/{len(results)}")

    # Save all results
    results_path = os.path.join(args.output_dir, 'cross_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Cross-validation results saved: {results_path}")

    return results

def main():
    parser = argparse.ArgumentParser(description='38-Fold Cross Validation Training')

    # Config file argument
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration JSON file')

    # Data arguments
    parser.add_argument('--dataset-index', type=str, default=None,
                       help='Path to dataset index JSON file (overrides config)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Optional local data directory to override paths in JSON')

    # Training arguments
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train (1-38). If not specified, train all folds.')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.00001,
                       help='Learning rate')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for models and logs')
    parser.add_argument('--log-interval', type=int, default=5,
                       help='Epoch interval for detailed metrics logging')

    # System arguments
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU even if GPU is available')
    parser.add_argument('--no-multi-gpu', action='store_true',
                       help='Disable multi-GPU training even if multiple GPUs available')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision (FP16) for faster training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load config file if exists
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {args.config}")

    # Apply config defaults (command line args override config file)
    if args.dataset_index is None:
        args.dataset_index = config.get('paths', {}).get('dataset_index', '/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D/dataset_index_validated.json')
    if args.batch_size == 128:  # Using default
        args.batch_size = config.get('training', {}).get('batch_size', 128)
    if args.epochs == 25:  # Using default
        args.epochs = config.get('training', {}).get('epochs', 25)
    if args.lr == 0.00001:  # Using default
        args.lr = config.get('training', {}).get('learning_rate', 0.00001)
    if args.output_dir == './output':  # Using default
        args.output_dir = config.get('paths', {}).get('output_base', './output')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_dir = os.path.join(args.output_dir, 'logs')
    log_file = setup_logging(log_dir, args.fold)

    # Log configuration
    logging.info("Training Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    logging.info("Loading all subjects data...")
    all_subjects_data = load_all_subjects(args.dataset_index, args.data_dir)

    # Train
    if args.fold is not None:
        if args.fold < 1 or args.fold > 38:
            logging.error(f"Invalid fold number: {args.fold}. Must be between 1 and 38.")
            sys.exit(1)

        logging.info(f"Training single fold: {args.fold}")
        train_single_fold(args, all_subjects_data)
    else:
        logging.info("Training all 38 folds...")
        train_all_folds(args, all_subjects_data)

    logging.info(f"All tasks completed. Logs saved to: {log_file}")

if __name__ == '__main__':
    main()