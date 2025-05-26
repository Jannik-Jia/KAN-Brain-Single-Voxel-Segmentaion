#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Brain Voxel Classification Comparison Script
Compares four scenarios of background handling and class weighting.
"""

import os
import gc
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt

# --- Configuration ---
# !!! IMPORTANT: Update MAT_FILE_PATH before running !!!
MAT_FILE_PATH = "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat"
OUTPUT_DIR = "./comparison_results_batch" # Directory to save plots and summary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
NUM_EPOCHS_DEMO = 25 # Increased for a more meaningful run, adjust as needed
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
L2_REG_FIXEDMLP = 1e-5 # L2 regularization for FixedMLP

# Class definitions based on typical brain voxel data structure
NUM_ACTUAL_CLASSES = 101        # When BG is processed (ignored), actual classes (0-100)
NUM_TOTAL_CLASSES_WITH_BG = 102 # When BG is NOT processed, total classes (0-101, where 0 is BG)

# --- Ensure deterministic behavior ---
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"--- Script Configuration ---")
print(f"Device: {DEVICE}")
print(f"MAT File Path: {MAT_FILE_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Epochs for Demo: {NUM_EPOCHS_DEMO}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Random State: {RANDOM_STATE}")
print(f"---------------------------")


# --- 1. MAT Data Loading Utilities ---
def load_mat_data_from_h5(mat_file_path):
    print(f"Loading MAT data from: {mat_file_path}")
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MAT file not found at {mat_file_path}")
    arrays = {}
    with h5py.File(mat_file_path, 'r') as f:
        for k, v in f.items():
            arrays[k] = np.array(v)
    print("MAT data loaded.")
    return arrays

def check_and_transpose_data_fn(arrays):
    print("Checking and transposing data...")
    data_key, region_key, prob_idx_key = 'data', 'region', 'prob_idx'
    expected_feature_dim = 341

    for key_orig, key_var_name in [('data', 'data_key'), ('region', 'region_key'), ('prob_idx', 'prob_idx_key')]:
        if eval(key_var_name) not in arrays:
            alt_key = None
            if key_orig == 'data' and 'TRAIN_DATA' in arrays: alt_key = 'TRAIN_DATA'
            elif key_orig == 'region' and 'TRAIN_REGION' in arrays: alt_key = 'TRAIN_REGION'
            # Add more alternative key checks if needed
            if alt_key:
                print(f"Original key '{eval(key_var_name)}' not found, using alternative '{alt_key}'.")
                globals()[key_var_name] = alt_key # Update the key variable
            else:
                raise KeyError(f"Essential key '{eval(key_var_name)}' (or its common alternatives) not found. Available keys: {list(arrays.keys())}")
    
    data_arr = arrays[data_key]
    if data_arr.shape[0] == expected_feature_dim and data_arr.ndim == 2 : data_transposed = data_arr.T
    elif data_arr.shape[1] == expected_feature_dim and data_arr.ndim == 2: data_transposed = data_arr
    else: raise ValueError(f"Unexpected shape for '{data_key}': {data_arr.shape}. Expected one dim {expected_feature_dim}, 2D array.")

    region_arr = arrays[region_key]
    if region_arr.shape[0] == NUM_TOTAL_CLASSES_WITH_BG and region_arr.ndim == 2: region_transposed = region_arr.T
    elif region_arr.shape[1] == NUM_TOTAL_CLASSES_WITH_BG and region_arr.ndim == 2: region_transposed = region_arr
    else: raise ValueError(f"Unexpected shape for '{region_key}': {region_arr.shape}. Expected one dim {NUM_TOTAL_CLASSES_WITH_BG}, 2D array.")
    
    prob_idx_transposed = arrays[prob_idx_key].flatten()

    assert data_transposed.shape[0] == region_transposed.shape[0] == prob_idx_transposed.shape[0], "Sample count mismatch"
    assert data_transposed.shape[1] == expected_feature_dim, f"Feature count mismatch, expected {expected_feature_dim}"
    assert region_transposed.shape[1] == NUM_TOTAL_CLASSES_WITH_BG, f"Class count mismatch, expected {NUM_TOTAL_CLASSES_WITH_BG}"
    print(f"Data shapes post-transpose: data {data_transposed.shape}, region {region_transposed.shape}, prob_idx {prob_idx_transposed.shape}")
    return data_transposed, region_transposed, prob_idx_transposed

def split_data_fn(data_len, prob_idx, random_state=RANDOM_STATE):
    print("Splitting data by patient ID...")
    all_indices = np.arange(data_len)
    val_mask = (prob_idx == 38) # Patient 38 for validation
    val_indices = all_indices[val_mask]
    train_test_pool_indices = all_indices[~val_mask]

    if len(train_test_pool_indices) > 0:
        train_indices, test_indices = train_test_split(
            train_test_pool_indices, test_size=0.01, random_state=random_state, shuffle=True
        )
    else:
        train_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
        print("Warning: train_test_pool_indices is empty. Train and test sets will be empty.")

    print(f"Total samples: {data_len}. Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    if len(train_indices) == 0 and NUM_EPOCHS_DEMO > 0 :
        print("CRITICAL WARNING: Training set is empty. Expect issues.")
    return train_indices, val_indices, test_indices

# --- 2. Scaler Creation ---
def create_scaler_fn(all_data_np, train_indices_np):
    print("Creating and fitting StandardScaler on training data...")
    scaler = StandardScaler()
    if len(train_indices_np) == 0:
        print("Warning: Training set is empty, scaler cannot be fitted. Using default (identity) scaler.")
        # Create a dummy fitted scaler to avoid errors downstream if transform is called
        scaler.mean_ = np.zeros(all_data_np.shape[1])
        scaler.scale_ = np.ones(all_data_np.shape[1])
        scaler.var_ = np.ones(all_data_np.shape[1])
        scaler.n_samples_seen_ = 0
        return scaler
    
    print(f"Fitting scaler on {len(train_indices_np)} training samples.")
    scaler.fit(all_data_np[train_indices_np])
    print("StandardScaler fitted.")
    return scaler

# --- 3. Custom PyTorch Dataset ---
class BrainVoxelPyTorchDataset(Dataset):
    def __init__(self, all_features_np, all_one_hot_labels_np, indices_np, scaler_obj, process_background=True):
        self.process_background = process_background
        if len(indices_np) == 0:
            print(f"Dataset Warning ({'BG processed' if process_background else 'BG as class'}): Initializing with zero samples.")
            self.features_scaled_tensor = torch.empty(0, all_features_np.shape[1] if all_features_np.ndim > 1 and all_features_np.shape[0] > 0 else 0, dtype=torch.float32)
            self.processed_labels_tensor = torch.empty(0, dtype=torch.long)
            return

        features_subset_np = all_features_np[indices_np]
        one_hot_labels_subset_np = all_one_hot_labels_np[indices_np]
        
        if hasattr(scaler_obj, 'mean_') and scaler_obj.mean_ is not None:
            features_scaled_np = scaler_obj.transform(features_subset_np)
        else:
            print("Dataset Warning: Scaler not fitted. Using raw features for this dataset partition.")
            features_scaled_np = features_subset_np
        self.features_scaled_tensor = torch.FloatTensor(features_scaled_np)

        indexed_raw_labels_np = np.argmax(one_hot_labels_subset_np, axis=1)
        if process_background:
            processed_labels_np = indexed_raw_labels_np - 1
            processed_labels_np[indexed_raw_labels_np == 0] = -1
        else:
            processed_labels_np = indexed_raw_labels_np
        self.processed_labels_tensor = torch.LongTensor(processed_labels_np)
        
        # print(f"Dataset created. BG processed: {self.process_background}. Samples: {len(self.processed_labels_tensor)}. Label range: [{self.processed_labels_tensor.min().item() if len(self) > 0 else 'N/A'} to {self.processed_labels_tensor.max().item() if len(self) > 0 else 'N/A'}]")


    def __len__(self):
        return len(self.processed_labels_tensor)

    def __getitem__(self, idx):
        return self.features_scaled_tensor[idx], self.processed_labels_tensor[idx]

# --- 4. Model Definition (FixedMLP) ---
class FixedMLP(nn.Module):
    def __init__(self, input_dim=341, num_output_classes=101, dropout_rate=0.5, l2_reg=L2_REG_FIXEDMLP):
        super(FixedMLP, self).__init__()
        self.l2_reg = l2_reg
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(4096, num_output_classes)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
    def forward(self, x): return self.layers(x)
    def get_l2_loss(self):
        l2_loss = torch.tensor(0., device=next(self.parameters()).device) # Ensure tensor on same device as model
        for param in self.parameters():
            if param.requires_grad: l2_loss += torch.norm(param, 2)**2
        return self.l2_reg * l2_loss

# --- 5. Robust Class Weight Calculation ---
def calculate_weights_fn_robust(labels_for_weights_tensor, num_effective_classes, process_background_flag, smoothing_epsilon=1e-9):
    print(f"Calculating class weights for {num_effective_classes} effective classes. BG processed: {process_background_flag}")
    if labels_for_weights_tensor.numel() == 0:
        print("Warning: No labels for weight calc. Returning equal weights.")
        return torch.ones(num_effective_classes, dtype=torch.float32).to(DEVICE)

    labels_np = labels_for_weights_tensor.cpu().numpy()
    if process_background_flag:
        actual_class_labels_np = labels_np[labels_np >= 0]
        if len(actual_class_labels_np) == 0:
            print("Warning: No valid (non-BG) labels for weight calc. Equal weights.")
            return torch.ones(num_effective_classes, dtype=torch.float32).to(DEVICE)
        counts_np = np.bincount(actual_class_labels_np, minlength=num_effective_classes)
    else:
        counts_np = np.bincount(labels_np, minlength=num_effective_classes)
    
    weights_np = 1.0 / (counts_np + smoothing_epsilon)
    weights_np[counts_np == 0] = 1.0 # Default weight for unseen classes
    # print(f"  Class counts (sample): {counts_np[:min(10, len(counts_np))]}")
    # print(f"  Weights (sample): {weights_np[:min(10, len(weights_np))]}")
    return torch.FloatTensor(weights_np).to(DEVICE)

# --- 6. Training and Evaluation Loops ---
def train_epoch_fn(model, dataloader, criterion, optimizer, device, add_l2_loss_flag=True):
    model.train()
    running_loss = 0.0; total_samples = 0
    all_preds_list, all_true_list = [], []
    if len(dataloader) == 0: return 0.0,0.0,0.0 # Handle empty dataloader

    for features, labels in dataloader:
        if features.nelement() == 0: continue
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        classification_loss = criterion(outputs, labels)
        final_loss = classification_loss
        if add_l2_loss_flag and hasattr(model, 'get_l2_loss'):
            final_loss += model.get_l2_loss()
        
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"NaN/Inf loss in training batch! Skipper. Cls_loss: {classification_loss.item()}")
            continue
        final_loss.backward()
        optimizer.step()
        running_loss += final_loss.item() * features.size(0)
        total_samples += features.size(0)
        
        active_mask = (labels != criterion.ignore_index) if criterion.ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
        if active_mask.sum() > 0:
            all_preds_list.extend(torch.argmax(outputs[active_mask], dim=1).cpu().numpy())
            all_true_list.extend(labels[active_mask].cpu().numpy())
            
    if total_samples == 0: return 0.0, 0.0, 0.0
    epoch_loss = running_loss / total_samples
    unique_lbls = np.unique(all_true_list + all_preds_list) if len(all_true_list) > 0 else np.array([])
    if criterion.ignore_index is not None: unique_lbls = unique_lbls[unique_lbls != criterion.ignore_index]

    epoch_acc = accuracy_score(all_true_list, all_preds_list) if len(all_true_list) > 0 else 0.0
    epoch_f1 = f1_score(all_true_list, all_preds_list, labels=unique_lbls if len(unique_lbls)>0 else None, average='macro', zero_division=0) if len(all_true_list) > 0 else 0.0
    return epoch_loss, epoch_acc, epoch_f1

def evaluate_model_fn(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0; total_samples = 0
    all_preds_list, all_true_list = [], []
    if len(dataloader) == 0: return 0.0,0.0,0.0 # Handle empty dataloader

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            if features.nelement() == 0: continue
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss in validation batch {batch_idx}! Cls_loss: {loss.item()}")
                # To debug, uncomment these lines:
                # print(f"  Problematic Features (norm): {torch.norm(features).item()}")
                # print(f"  Problematic Labels: {labels.cpu().numpy()}")
                # print(f"  Unique Labels in problematic batch: {torch.unique(labels).cpu().numpy()}")
                # print(f"  Model Outputs (sample of first 5, first 10 logits): \n{outputs[:5, :10].cpu().numpy()}")
                # print(f"  Criterion ignore_index: {criterion.ignore_index}")
                # print(f"  Criterion weights (sum): { (criterion.weight.sum().item()) if criterion.weight is not None else 'No weights'}")
                # if criterion.ignore_index is not None and torch.all(labels == criterion.ignore_index):
                #     print("  VAL BATCH WARNING: All labels in this batch are ignore_index.")
                running_loss = float('nan') # Propagate NaN
                break # Exit loop on NaN validation loss for this epoch
            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            active_mask = (labels != criterion.ignore_index) if criterion.ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
            if active_mask.sum() > 0:
                all_preds_list.extend(torch.argmax(outputs[active_mask], dim=1).cpu().numpy())
                all_true_list.extend(labels[active_mask].cpu().numpy())
    
    if total_samples == 0 or np.isnan(running_loss): return float('nan') if np.isnan(running_loss) else 0.0, 0.0, 0.0
    epoch_loss = running_loss / total_samples
    unique_lbls = np.unique(all_true_list + all_preds_list) if len(all_true_list) > 0 else np.array([])
    if criterion.ignore_index is not None: unique_lbls = unique_lbls[unique_lbls != criterion.ignore_index]
    epoch_acc = accuracy_score(all_true_list, all_preds_list) if len(all_true_list) > 0 else 0.0
    epoch_f1 = f1_score(all_true_list, all_preds_list, labels=unique_lbls if len(unique_lbls)>0 else None, average='macro', zero_division=0) if len(all_true_list) > 0 else 0.0
    return epoch_loss, epoch_acc, epoch_f1

# --- Helper function to run a scenario ---
def run_scenario(scenario_name, process_bg, use_weights,
                 all_features_np, all_labels_one_hot_np,
                 train_indices_np, val_indices_np, scaler_obj,
                 num_epochs, batch_size, device, learning_rate):
    print(f"\n\n--- Running {scenario_name} ---")
    print(f"Process Background: {process_bg}, Use Class Weights: {use_weights}")

    num_model_output_classes = NUM_ACTUAL_CLASSES if process_bg else NUM_TOTAL_CLASSES_WITH_BG
    
    train_dataset = BrainVoxelPyTorchDataset(all_features_np, all_labels_one_hot_np, train_indices_np, scaler_obj, process_background=process_bg)
    val_dataset = BrainVoxelPyTorchDataset(all_features_np, all_labels_one_hot_np, val_indices_np, scaler_obj, process_background=process_bg)

    if len(train_dataset) == 0:
        print(f"SKIPPING {scenario_name}: Training dataset is empty.")
        empty_hist = {k: [] for k in ['train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']}
        return {'val_acc': 0.0, 'val_f1': 0.0, 'history': empty_hist}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory= (device.type == 'cuda') )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory= (device.type == 'cuda') )

    model = FixedMLP(num_output_classes=num_model_output_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    class_weights_tensor = None
    if use_weights:
        class_weights_tensor = calculate_weights_fn_robust(
            train_dataset.processed_labels_tensor, # Use tensor directly
            num_model_output_classes,
            process_background_flag=process_bg
        )
    
    if process_bg:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=-1).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device) # No ignore_index

    print(f"  Model output classes: {num_model_output_classes}")
    print(f"  Criterion ignore_index: {criterion.ignore_index if process_bg else 'None (BG is class 0)'}")
    # print(f"  Using class weights: {use_weights} " + (f"(Sample weights: {class_weights_tensor[:3].cpu().numpy() if class_weights_tensor is not None and class_weights_tensor.numel() > 0 else 'N/A'})" ))


    history = {'train_loss':[], 'train_acc':[], 'train_f1':[], 'val_loss':[], 'val_acc':[], 'val_f1':[]}
    
    for epoch in range(num_epochs):
        start_e_time = time.time()
        train_loss, train_acc, train_f1 = train_epoch_fn(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_model_fn(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc); history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_f1'].append(val_f1)
        
        print(f"{scenario_name} - Epoch {epoch+1}/{num_epochs} ({(time.time()-start_e_time):.2f}s) -> "
              f"Train L: {train_loss:.4f}, A: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Val L: {val_loss:.4f}, A: {val_acc:.4f}, F1: {val_f1:.4f}")
        if np.isnan(val_loss): break # Stop if validation loss becomes NaN
              
    final_val_acc = history['val_acc'][-1] if history['val_acc'] and not np.isnan(history['val_acc'][-1]) else 0.0
    final_val_f1 = history['val_f1'][-1] if history['val_f1'] and not np.isnan(history['val_f1'][-1]) else 0.0
    
    del model, train_dataset, val_dataset, train_loader, val_loader, criterion, optimizer
    if class_weights_tensor is not None: del class_weights_tensor
    gc.collect(); torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return {'val_acc': final_val_acc, 'val_f1': final_val_f1, 'history': history}

def main_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Main Experiment Starting ---")
    
    all_features_raw_np, all_labels_one_hot_raw_np, prob_idx_raw_np, scaler_obj = None, None, None, None
    train_idx_np, val_idx_np, test_idx_np = np.array([]), np.array([]), np.array([])

    try:
        mat_arrays = load_mat_data_from_h5(MAT_FILE_PATH)
        all_features_raw_np, all_labels_one_hot_raw_np, prob_idx_raw_np = check_and_transpose_data_fn(mat_arrays)
        train_idx_np, val_idx_np, test_idx_np = split_data_fn(len(all_features_raw_np), prob_idx_raw_np)
        scaler_obj = create_scaler_fn(all_features_raw_np, train_idx_np)
        print(f"Scaler mean (first 5): {scaler_obj.mean_[:5] if hasattr(scaler_obj, 'mean_') and scaler_obj.mean_ is not None else 'Scaler not fitted or empty train set'}")
    except Exception as e:
        print(f"ERROR during data loading/preprocessing: {e}")
        print("Aborting experiment.")
        return

    results_summary = {}
    
    scenarios_to_run = [
        {"name": "S1 (WithBG, WithW)", "process_bg": True, "use_weights": True},
        {"name": "S2 (NoBG, WithW)", "process_bg": False, "use_weights": True},
        {"name": "S3 (WithBG, NoW)", "process_bg": True, "use_weights": False},
        {"name": "S4 (NoBG, NoW)", "process_bg": False, "use_weights": False},
    ]

    for sc_config in scenarios_to_run:
        if len(train_idx_np) == 0: # Critical check
             print(f"SKIPPING Scenario {sc_config['name']} due to empty training set from initial split.")
             empty_hist = {k: [] for k in ['train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']}
             results_summary[sc_config['name']] = {'val_acc': 0.0, 'val_f1': 0.0, 'history': empty_hist}
             continue

        sc_results = run_scenario(
            sc_config["name"], sc_config["process_bg"], sc_config["use_weights"],
            all_features_raw_np, all_labels_one_hot_raw_np,
            train_idx_np, val_idx_np, scaler_obj,
            NUM_EPOCHS_DEMO, BATCH_SIZE, DEVICE, LEARNING_RATE
        )
        results_summary[sc_config["name"]] = sc_results

    summary_file_path = os.path.join(OUTPUT_DIR, "final_results_summary.txt")
    print(f"\n\n--- Final Results Summary (saved to {summary_file_path}) ---")
    with open(summary_file_path, "w") as f:
        header = f"{'Scenario':<40} | {'Final Val Accuracy':<20} | {'Final Val Macro F1':<20}\n"
        separator = "-" * 85 + "\n"
        print(header.strip())
        f.write(header)
        print(separator.strip())
        f.write(separator)
        for scenario, metrics in results_summary.items():
            val_acc_display = metrics['val_acc']
            val_f1_display = metrics['val_f1']
            if isinstance(val_acc_display, float): val_acc_display = f"{val_acc_display:.4f}"
            if isinstance(val_f1_display, float): val_f1_display = f"{val_f1_display:.4f}"
            line = f"{scenario:<40} | {val_acc_display:<20} | {val_f1_display:<20}\n"
            print(line.strip())
            f.write(line)

    if results_summary:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12)) # Increased width for legend
        for i, metric_key in enumerate(['val_f1', 'val_acc']):
            ax = axes[i]
            metric_name = "Macro F1-Score" if metric_key == 'val_f1' else "Accuracy"
            for scenario_name, result_data in results_summary.items():
                history = result_data.get('history', {})
                if history and history.get(metric_key):
                    metric_values = np.array(history[metric_key])
                    # Plot only non-NaN values if any
                    epochs_run = np.arange(1, len(metric_values) + 1)
                    valid_indices = ~np.isnan(metric_values)
                    ax.plot(epochs_run[valid_indices], metric_values[valid_indices], marker='o' if i==0 else 'x', linestyle='-', label=f"{scenario_name}")
            ax.set_title(f'Validation {metric_name} Comparison (Epochs: {NUM_EPOCHS_DEMO})')
            ax.set_xlabel('Epoch'); ax.set_ylabel(metric_name)
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
            ax.grid(True); ax.set_xticks(np.arange(1, NUM_EPOCHS_DEMO + 1))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend
        plot_file_path = os.path.join(OUTPUT_DIR, "validation_curves_comparison.png")
        plt.savefig(plot_file_path) # bbox_inches='tight' removed for rect
        plt.close(fig)
        print(f"Validation curves plot saved to {plot_file_path}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main_experiment()
    print("Comparison experiment Python script finished.")