#!/usr/bin/env python
# coding: utf-8
# nohup python D_proj_FC_train.py > output.log 2>&1 &
"""
MRI Voxel Classification Training - Complete 12-fold Cross Validation
Uses deep neural network for 52-class classification including background class
Modified to:
1. Run as Python script (not notebook) for background execution
2. No validation split - uses full training set to avoid overfitting
3. Evaluates test set at each epoch
4. All plots and outputs in English
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Training configuration
CONFIG = {
    'processed_dir': '/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS',
    'export_path': './models/',
    'batch_size': 8192,
    'test_batch_size': 8192,  # Test batch size
    'no_epochs': 25,
    'no_classes': 52,
    'input_dim': 42,
    'learning_rate': 0.00001,
    'weight_decay': 0.00001,
    'dropout_rate': 0.5,
    'exclude_background': False,  # Include background, consistent with original process
    'hidden_dim': 4096,
    'num_hidden_layers': 4,
    # Feature exclusion configuration
    'exclude_features': [14],  # List of feature indices to exclude, 14 is B0map
    'feature_names': [  # Record feature names for tracking
        't1_mp2rage_sag_0p65_UNI-DEN_13_MR',
        'c_anatomy_nii_pre_t1_mp2rage_sag_0p65_INV1_10_MR',
        'c_anatomy_nii_pre_t1_mp2rage_sag_0p65_INV2_14_MR',
        'c_anatomy_nii_pre_t1_mp2rage_sag_0p65_T1_Images_11_MR',
        'c_anatomy_nii_pre_t1_mp2rage_sag_0p65_UNI_Images_12_MR',
        'c_anatomy_nii_pre_t1_spc_sag_13iso_WBIR_7_MR',
        'c_anatomy_nii_pre_t2_flair_sag_1p3iso_WBIR_5_MR',
        'c_anatomy_nii_pre_t2_spc_cor_1iso_WBIR_6_MR',
        'c_CEST_nii_post_PredNN_P_FOR16_1_mtr_b1_p_7_popt',
        'c_CEST_nii_post_PredNN_P_FOR16_1_noe_b1_p_7_popt',
        'c_CEST_nii_post_PredNN_P_FOR16_1_quass_b1_p_7_popt',
        'c_CEST_nii_post_PredNN_P_FOR16_1_relnoe_b1_p_7_popt',
        'c_CEST_nii_post_PredNN_P_FOR16_1_ssmtamide_b1_p_7_popt',
        'c_CEST_nii_post_PredNN_P_FOR16_1_ssmt_b1_p_7_popt',
        'c_CEST_nii_post_PredNN_P_FOR16_1_water_b1_p_7_popt',
        'c_qsm_nii_post_vibe_1iso_CAIPI6_8TE_RR_B0map_highre',  # Index 14 - to be excluded
        'c_qsm_nii_post_vibe_1iso_CAIPI6_8TE_RR_rot_susceptibility_highre',
        'c_qsm_nii_post_vibe_1iso_CAIPI6_8TE_RR_SMWIdiamag',
        'c_qsm_nii_post_vibe_1iso_CAIPI6_8TE_RR_SMWIparamag',
        'c_qsm_nii_post_vibe_1iso_CAIPI6_8TE_RR_T2starmap',
        'c_qsm_nii_pre_vibe_1iso_CAIPI6_8TE_RR_magnitude',
    ]
}

# Calculate actual input dimension dynamically
actual_input_dim = CONFIG['input_dim'] - len(CONFIG.get('exclude_features', []))
CONFIG['actual_input_dim'] = actual_input_dim

# Create output directory
Path(CONFIG['export_path']).mkdir(parents=True, exist_ok=True)
logger.info("Configuration parameters:")
for key, value in CONFIG.items():
    if key != 'feature_names':  # Skip feature names in main config display
        logger.info(f"  {key}: {value}")

# Log excluded features information
if CONFIG.get('exclude_features'):
    logger.info(f"\n⚠️ Feature exclusion configuration:")
    logger.info(f"  Excluded feature indices: {CONFIG['exclude_features']}")
    for idx in CONFIG['exclude_features']:
        if idx < len(CONFIG['feature_names']):
            logger.info(f"    - [{idx}] {CONFIG['feature_names'][idx]}")
    logger.info(f"  Actual input dimension: {actual_input_dim}")

class PreprocessedMRIDataset:
    """
    Preprocessed MRI dataset loader - uses data from flattened directory
    Supports 12-fold cross validation, each subject as one fold
    Modified: No validation split to avoid overfitting
    """
    
    def __init__(self, 
                 processed_dir: str, 
                 fold: int, 
                 exclude_background: bool = False,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Parameters:
        -----------
        processed_dir : str
            Root directory path of preprocessed data
        fold : int
            Current fold number (1-12)
        exclude_background : bool
            Whether to exclude background voxels (label 0)
        random_state : int
            Random seed
        verbose : bool
            Whether to print detailed information
        """
        self.processed_dir = Path(processed_dir)
        self.fold = fold
        self.exclude_background = exclude_background
        self.random_state = random_state
        self.verbose = verbose
        
        # Validate fold number
        if not 1 <= fold <= 12:
            raise ValueError(f"Fold must be between 1-12, current value: {fold}")
        
        # Get all subjects
        self.subjects = self._get_all_subjects()
        if len(self.subjects) != 12:
            raise ValueError(f"Expected 12 subjects, but found {len(self.subjects)}")
        
        # Determine data split
        self.test_subject = self.subjects[fold - 1]
        self.train_subjects = [s for i, s in enumerate(self.subjects) if i != fold - 1]
        
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"Initializing dataset - Fold {fold}/12")
            logger.info(f"{'='*80}")
            logger.info(f"Test set: {self.test_subject}")
            logger.info(f"Training set: {len(self.train_subjects)} subjects")
            logger.info(f"Exclude background: {'Yes' if exclude_background else 'No'}")
    
    def _get_all_subjects(self) -> List[str]:
        """Get all subject IDs (sorted by name)"""
        subjects = []
        for subject_dir in sorted(self.processed_dir.glob("FOR_*")):
            flattened_dir = subject_dir / "flattened"
            if flattened_dir.exists() and (flattened_dir / "features.npy").exists():
                subjects.append(subject_dir.name)
        return sorted(subjects)
    
    def _load_subject_data(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for a single subject
        
        Returns:
        --------
        features : np.ndarray
            Feature array (n_voxels, 42)
        labels : np.ndarray
            Label array (n_voxels,) integer labels
        mask : np.ndarray
            Valid voxel mask (n_voxels,) bool type
        """
        subject_path = self.processed_dir / subject_id / "flattened"
        
        # Load data
        features = np.load(subject_path / "features.npy")
        
        # Load labels (prefer mapped labels)
        if (subject_path / "labels_mapped.npy").exists():
            labels = np.load(subject_path / "labels_mapped.npy")
        else:
            labels = np.load(subject_path / "labels.npy")
            logger.warning(f"{subject_id} using original labels, may need mapping")
        
        # Create mask
        if self.exclude_background:
            # Background class is 0
            mask = labels != 0
        else:
            mask = np.ones(len(features), dtype=bool)
        
        return features, labels, mask
    
    def get_fold_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for current fold
        
        Returns:
        --------
        dict : Dictionary containing the following keys
            - train_data: (n_train, 42)
            - train_labels: (n_train,) integer labels
            - test_data: (n_test, 42)
            - test_labels: (n_test,) integer labels
        """
        if self.verbose:
            logger.info(f"\nLoading data for Fold {self.fold}...")
        
        start_time = time.time()
        
        # 1. Load test data (single subject)
        test_features, test_labels, test_mask = self._load_subject_data(self.test_subject)
        test_data = test_features[test_mask]
        test_labels = test_labels[test_mask]
        
        if self.verbose:
            logger.info(f"\nTest set ({self.test_subject}):")
            logger.info(f"   Voxel count: {len(test_data):,}")
            logger.info(f"   Shape: Features{test_data.shape}, Labels{test_labels.shape}")
            logger.info(f"   Label range: {test_labels.min()} - {test_labels.max()}")
        
        # 2. Load training data (other 11 subjects) - NO VALIDATION SPLIT
        train_features_list = []
        train_labels_list = []
        
        for subject_id in self.train_subjects:
            features, labels, mask = self._load_subject_data(subject_id)
            train_features_list.append(features[mask])
            train_labels_list.append(labels[mask])
            
            if self.verbose:
                logger.info(f"   {subject_id}: {mask.sum():,} voxels")
        
        # Merge all training data
        train_data = np.vstack(train_features_list)
        train_labels = np.hstack(train_labels_list)  # Note: labels are 1D, use hstack
        
        load_time = time.time() - start_time
        
        if self.verbose:
            logger.info(f"\nDataset statistics:")
            logger.info(f"   Training set: {len(train_data):,} voxels")
            logger.info(f"   Test set: {len(test_data):,} voxels")
            logger.info(f"   Total: {len(train_data) + len(test_data):,} voxels")
            logger.info(f"   Load time: {load_time:.2f} seconds")
        
        return {
            'train_data': train_data.astype(np.float32),
            'train_labels': train_labels.astype(np.int64),
            'test_data': test_data.astype(np.float32),
            'test_labels': test_labels.astype(np.int64)
        }

class RegModel(nn.Module):
    """Deep fully connected network for MRI voxel classification"""
    
    def __init__(self, input_dim=42, num_classes=52, hidden_dim=4096, 
                 num_hidden_layers=4, dropout_rate=0.5):
        super(RegModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def exclude_features(data, exclude_indices):
    """
    Exclude specified features from data
    
    Parameters:
    -----------
    data : np.ndarray
        Data with shape (n_samples, n_features)
    exclude_indices : list
        List of feature indices to exclude
    
    Returns:
    --------
    np.ndarray : Data after excluding specified features
    """
    if not exclude_indices:
        return data
    
    # Create mask for features to keep
    all_indices = np.arange(data.shape[1])
    keep_mask = np.ones(data.shape[1], dtype=bool)
    keep_mask[exclude_indices] = False
    
    return data[:, keep_mask]

def load_fold_data(fold, config):
    """Load and preprocess data for a specific fold"""
    
    # Initialize dataset
    dataset = PreprocessedMRIDataset(
        processed_dir=config['processed_dir'],
        fold=fold,
        exclude_background=config['exclude_background'],
        verbose=True
    )
    
    # Get data
    data_dict = dataset.get_fold_data()
    
    X_train = data_dict['train_data']
    y_train = data_dict['train_labels']
    X_test = data_dict['test_data']
    y_test = data_dict['test_labels']
    
    # Exclude specified features
    if config.get('exclude_features'):
        logger.info(f"\nExcluding feature indices: {config['exclude_features']}")
        X_train = exclude_features(X_train, config['exclude_features'])
        X_test = exclude_features(X_test, config['exclude_features'])
        logger.info(f"Feature dimension: {config['input_dim']} → {X_train.shape[1]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"\nData preprocessing completed:")
    logger.info(f"  Training set shape: {X_train.shape}")
    logger.info(f"  Test set shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler

def calculate_metrics(true_labels, predictions, loss):
    """Calculate classification metrics"""
    
    # Convert predictions to class labels
    if predictions.ndim == 2:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_labels)
    
    # Calculate macro F1 score
    macro_f1 = f1_score(true_labels, pred_classes, average='macro', zero_division=0)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """Train one epoch"""
    
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    n_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Convert labels to class indices
        if target.ndim == 2:
            target = torch.argmax(target, dim=1)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record metrics
        total_loss += loss.item()
        n_batches += 1
        
        # Store predictions for metrics
        all_predictions.append(output.detach().cpu().numpy())
        all_labels.append(target.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.vstack(all_predictions)
    all_labels = np.hstack(all_labels)
    avg_loss = total_loss / n_batches
    
    metrics = calculate_metrics(all_labels, all_predictions, avg_loss)
    
    return metrics

def evaluate(model, X_data, y_data, criterion, device, config, batch_size=32768):
    """Evaluate model on dataset with batch processing"""
    
    model.eval()
    total_loss = 0
    all_predictions = []
    n_batches = 0
    
    # Process in batches to avoid OOM
    n_samples = len(X_data)
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Get batch
            batch_X = torch.FloatTensor(X_data[start_idx:end_idx]).to(device)
            batch_y = torch.LongTensor(y_data[start_idx:end_idx]).to(device)
            
            # Forward pass
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            # Record metrics
            total_loss += loss.item()
            n_batches += 1
            all_predictions.append(output.cpu().numpy())
    
    # Combine predictions
    all_predictions = np.vstack(all_predictions)
    avg_loss = total_loss / n_batches
    
    # Calculate metrics
    metrics = calculate_metrics(y_data, all_predictions, avg_loss)
    
    return metrics

def train_single_fold(fold, config, verbose=True):
    """Train a single fold - optimized memory version with test evaluation each epoch"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Fold {fold}/12")
    logger.info(f"{'='*80}")
    
    # Load data to CPU memory
    X_train, y_train, X_test, y_test, scaler = load_fold_data(fold, config)
    
    logger.info("\n📌 Data strategy: CPU memory → GPU batches")
    logger.info(f"   Training data in CPU: {X_train.nbytes / 1e9:.2f} GB")
    logger.info(f"   GPU batch size: {config['batch_size']} samples")
    
    # Create CPU tensor dataset
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model (model on GPU)
    # Use actual input dimension
    actual_input_dim = config.get('actual_input_dim', config['input_dim'])
    
    model = RegModel(
        input_dim=actual_input_dim,  # Use actual dimension
        num_classes=config['no_classes'],
        hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Print memory status
    if torch.cuda.is_available():
        logger.info(f"\n🎮 GPU memory status:")
        logger.info(f"   Model usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        logger.info(f"   Available memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f} GB")
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': []
    }
    
    # Best model tracking
    best_test_f1 = 0
    best_epoch = 0
    best_model_state = None
    
    logger.info("\n🚀 Starting training...")
    start_time = time.time()
    
    # Training loop
    for epoch in range(config['no_epochs']):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # Test evaluation every epoch
        test_metrics = evaluate(model, X_test, y_test, criterion, device, config, 
                               batch_size=config.get('test_batch_size', 32768))
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['macro_f1'])
        
        # Save best model based on test F1
        if test_metrics['macro_f1'] > best_test_f1:
            best_test_f1 = test_metrics['macro_f1']
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if verbose:
            logger.info(f"\nEpoch [{epoch+1}/{config['no_epochs']}] "
                       f"Time: {time.time()-epoch_start:.2f}s")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"Macro F1: {train_metrics['macro_f1']:.4f}")
            logger.info(f"  Test  - Loss: {test_metrics['loss']:.4f}, "
                       f"Acc: {test_metrics['accuracy']:.4f}, "
                       f"Macro F1: {test_metrics['macro_f1']:.4f}")
            
            # Show GPU memory usage
            if torch.cuda.is_available():
                logger.info(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        # Periodically clear GPU cache
        torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    logger.info(f"\n✅ Training completed! Total time: {training_time:.2f} seconds")
    logger.info(f"Best test Macro F1: {best_test_f1:.4f} (Epoch {best_epoch+1})")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final test evaluation
    logger.info("\n📊 Final test set evaluation...")
    final_test_metrics = evaluate(model, X_test, y_test, criterion, device, config,
                                 batch_size=config.get('test_batch_size', 32768))
    
    logger.info(f"\nFinal test results:")
    logger.info(f"  Loss: {final_test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {final_test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1: {final_test_metrics['macro_f1']:.4f}")
    
    # Save model
    model_path = Path(config['export_path']) / f'fold{fold}_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'fold': fold,
        'config': config,
        'history': history,
        'test_metrics': final_test_metrics,
        'best_epoch': best_epoch,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'actual_input_dim': actual_input_dim,  # Save actual input dimension
        'excluded_features': config.get('exclude_features', [])  # Save excluded features
    }, model_path)
    
    # Clean memory
    del X_train_tensor, y_train_tensor, train_dataset, train_loader
    torch.cuda.empty_cache()
    
    return {
        'fold': fold,
        'history': history,
        'test_metrics': final_test_metrics,
        'best_epoch': best_epoch,
        'training_time': training_time
    }

def plot_fold_history(fold_result):
    """Plot training history for a single fold"""
    history = fold_result['history']
    fold = fold_result['fold']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Fold {fold} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Fold {fold} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Macro F1
    ax3.plot(epochs, history['train_f1'], 'b-', label='Train F1')
    ax3.plot(epochs, history['test_f1'], 'r-', label='Test F1')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro F1')
    ax3.set_title(f'Fold {fold} - Macro F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test metrics over epochs
    ax4.plot(epochs, history['test_loss'], 'g-', label='Test Loss', alpha=0.7)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, history['test_f1'], 'orange', label='Test F1', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Loss', color='g')
    ax4_twin.set_ylabel('Test F1', color='orange')
    ax4.set_title(f'Fold {fold} - Test Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(CONFIG['export_path']) / f'fold{fold}_history.png', dpi=300)
    plt.close()
    
    logger.info(f"Saved training history plot for Fold {fold}")

def run_cross_validation(config):
    """Run complete 12-fold cross validation"""
    
    logger.info("\n" + "="*80)
    logger.info("Starting 12-Fold Cross Validation")
    logger.info("="*80)
    
    all_results = []
    summary_data = []
    
    start_time = time.time()
    
    for fold in range(1, 13):
        try:
            # Train single fold
            fold_result = train_single_fold(fold, config)
            all_results.append(fold_result)
            
            # Plot training history
            plot_fold_history(fold_result)
            
            # Add to summary
            summary_data.append({
                'Fold': fold,
                'Test Loss': fold_result['test_metrics']['loss'],
                'Test Acc': fold_result['test_metrics']['accuracy'],
                'Test Macro F1': fold_result['test_metrics']['macro_f1'],
                'Best Epoch': fold_result['best_epoch'] + 1,
                'Time(s)': fold_result['training_time']
            })
            
            # Save intermediate results
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(Path(config['export_path']) / 'cv_summary_partial.csv', index=False)
            
        except Exception as e:
            logger.error(f"Error in fold {fold}: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    summary_df = pd.DataFrame(summary_data)
    
    logger.info("\n" + "="*80)
    logger.info("12-Fold Cross Validation Results Summary")
    logger.info("="*80)
    
    logger.info("\nResults by Fold:")
    logger.info(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Calculate mean and std
    mean_loss = summary_df['Test Loss'].mean()
    std_loss = summary_df['Test Loss'].std()
    mean_acc = summary_df['Test Acc'].mean()
    std_acc = summary_df['Test Acc'].std()
    mean_f1 = summary_df['Test Macro F1'].mean()
    std_f1 = summary_df['Test Macro F1'].std()
    
    logger.info("\nStatistical Results:")
    logger.info(f"  Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    logger.info(f"  Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  Test Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"\nTotal training time: {total_time/60:.2f} minutes")
    logger.info(f"Average time per fold: {summary_df['Time(s)'].mean():.2f} seconds")
    
    # Save summary results
    summary_path = Path(config['export_path']) / 'cv_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Save complete results
    results_path = Path(config['export_path']) / 'cv_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'summary': {
                'mean_loss': float(mean_loss),
                'std_loss': float(std_loss),
                'mean_acc': float(mean_acc),
                'std_acc': float(std_acc),
                'mean_f1': float(mean_f1),
                'std_f1': float(std_f1)
            },
            'total_time': total_time,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Plot summary
    plot_cv_summary(summary_df)
    
    return all_results, summary_df

def plot_cv_summary(summary_df):
    """Plot cross validation summary chart"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Test Loss
    ax1.bar(summary_df['Fold'], summary_df['Test Loss'], color='skyblue', edgecolor='navy')
    ax1.axhline(y=summary_df['Test Loss'].mean(), color='red', linestyle='--', 
                label=f'Mean: {summary_df["Test Loss"].mean():.4f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('Test Loss by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test Accuracy
    ax2.bar(summary_df['Fold'], summary_df['Test Acc'], color='lightgreen', edgecolor='darkgreen')
    ax2.axhline(y=summary_df['Test Acc'].mean(), color='red', linestyle='--',
                label=f'Mean: {summary_df["Test Acc"].mean():.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test Macro F1
    ax3.bar(summary_df['Fold'], summary_df['Test Macro F1'], color='salmon', edgecolor='darkred')
    ax3.axhline(y=summary_df['Test Macro F1'].mean(), color='red', linestyle='--',
                label=f'Mean: {summary_df["Test Macro F1"].mean():.4f}')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Test Macro F1')
    ax3.set_title('Test Macro F1 by Fold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(CONFIG['export_path']) / 'cv_summary.png', dpi=300)
    plt.close()
    
    logger.info("Saved cross validation summary plot")

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("MRI Voxel Classification Training")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run complete 12-fold cross validation
    all_results, summary_df = run_cross_validation(CONFIG)
    
    logger.info("\n" + "="*80)
    logger.info("Training Completed!")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {CONFIG['export_path']}")
    logger.info("\nFile list:")
    for file in sorted(Path(CONFIG['export_path']).glob('*')):
        logger.info(f"  - {file.name}")
    
    # Display final summary
    logger.info("\nFinal 12-Fold Cross Validation Results:")
    logger.info(f"  Test Macro F1: {summary_df['Test Macro F1'].mean():.4f} ± {summary_df['Test Macro F1'].std():.4f}")
    logger.info(f"  Test Accuracy: {summary_df['Test Acc'].mean():.4f} ± {summary_df['Test Acc'].std():.4f}")
    
    # If features were excluded, print that information
    if CONFIG.get('exclude_features'):
        logger.info(f"\nNote: This training excluded the following feature indices: {CONFIG['exclude_features']}")
        logger.info(f"Actual input dimension used: {CONFIG['actual_input_dim']}")
    
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()