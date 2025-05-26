#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Brain Voxel Classification Comparison Script
Compares four scenarios of background handling and class weighting.
MODIFIED TO DEBUG NaN LOSS.
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
OUTPUT_DIR = "./comparison_results_batch_debug" # Directory to save plots and summary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
NUM_EPOCHS_DEMO = 5 # Keep low for debugging, increase for full run (was 25 in user log)
BATCH_SIZE = 128 # User log showed 128
LEARNING_RATE = 1e-5
L2_REG_FIXEDMLP = 1e-5

NUM_ACTUAL_CLASSES = 101
NUM_TOTAL_CLASSES_WITH_BG = 102

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

def load_mat_data_from_h5(mat_file_path):
    print(f"Loading MAT data from: {mat_file_path}")
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MAT file not found at {mat_file_path}")
    arrays = {}
    with h5py.File(mat_file_path, 'r') as f:
        for k, v in f.items(): arrays[k] = np.array(v)
    print("MAT data loaded.")
    return arrays

def check_and_transpose_data_fn(arrays):
    print("Checking and transposing data...")
    data_key, region_key, prob_idx_key = 'data', 'region', 'prob_idx'
    expected_feature_dim = 341
    keys_to_update = {'data_key': 'data', 'region_key': 'region', 'prob_idx_key': 'prob_idx'}
    for var_name, orig_key_val in keys_to_update.items():
        current_key_val = eval(var_name) # Gets value of data_key, region_key, etc.
        if current_key_val not in arrays:
            alt_key = None
            if orig_key_val == 'data' and 'TRAIN_DATA' in arrays: alt_key = 'TRAIN_DATA'
            elif orig_key_val == 'region' and 'TRAIN_REGION' in arrays: alt_key = 'TRAIN_REGION'
            if alt_key:
                print(f"Original key '{current_key_val}' not found for {orig_key_val}, using alternative '{alt_key}'.")
                globals()[var_name] = alt_key
            else:
                raise KeyError(f"Essential key '{current_key_val}' for {orig_key_val} not found. Available: {list(arrays.keys())}")
    
    data_arr = arrays[data_key]; region_arr = arrays[region_key]
    if data_arr.shape[0] == expected_feature_dim and data_arr.ndim == 2 : data_transposed = data_arr.T
    elif data_arr.shape[1] == expected_feature_dim and data_arr.ndim == 2: data_transposed = data_arr
    else: raise ValueError(f"Shape error for '{data_key}': {data_arr.shape}. Expected one dim {expected_feature_dim}, 2D.")

    if region_arr.shape[0] == NUM_TOTAL_CLASSES_WITH_BG and region_arr.ndim == 2: region_transposed = region_arr.T
    elif region_arr.shape[1] == NUM_TOTAL_CLASSES_WITH_BG and region_arr.ndim == 2: region_transposed = region_arr
    else: raise ValueError(f"Shape error for '{region_key}': {region_arr.shape}. Expected one dim {NUM_TOTAL_CLASSES_WITH_BG}, 2D.")
    
    prob_idx_transposed = arrays[prob_idx_key].flatten()
    assert data_transposed.shape[0]==region_transposed.shape[0]==prob_idx_transposed.shape[0], "Sample count mismatch"
    assert data_transposed.shape[1]==expected_feature_dim, f"Feature error, expected {expected_feature_dim}"
    assert region_transposed.shape[1]==NUM_TOTAL_CLASSES_WITH_BG, f"Class error, expected {NUM_TOTAL_CLASSES_WITH_BG}"
    print(f"Data shapes: data {data_transposed.shape}, region {region_transposed.shape}, prob_idx {prob_idx_transposed.shape}")
    return data_transposed, region_transposed, prob_idx_transposed

def split_data_fn(data_len, prob_idx, random_state=RANDOM_STATE):
    print("Splitting data by patient ID...")
    all_indices = np.arange(data_len)
    val_mask = (prob_idx == 38); val_indices = all_indices[val_mask]
    train_test_pool_indices = all_indices[~val_mask]
    if len(train_test_pool_indices) > 0:
        train_indices, test_indices = train_test_split(
            train_test_pool_indices, test_size=0.01, random_state=random_state, shuffle=True)
    else:
        train_indices, test_indices = np.array([], dtype=int), np.array([], dtype=int)
        print("Warning: train_test_pool_indices empty. Train/test sets will be empty.")
    print(f"Total: {data_len}. Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    if len(train_indices) == 0 and NUM_EPOCHS_DEMO > 0 : print("CRITICAL WARNING: Training set empty.")
    return train_indices, val_indices, test_indices

def create_scaler_fn(all_data_np, train_indices_np):
    print("Creating/fitting StandardScaler...")
    scaler = StandardScaler()
    if len(train_indices_np) == 0:
        print("Warning: Training set empty, scaler not fitted. Using identity scaler.")
        scaler.mean_ = np.zeros(all_data_np.shape[1]); scaler.scale_ = np.ones(all_data_np.shape[1])
        return scaler
    print(f"Fitting scaler on {len(train_indices_np)} train samples.")
    scaler.fit(all_data_np[train_indices_np])
    print("StandardScaler fitted.")
    return scaler

class BrainVoxelPyTorchDataset(Dataset):
    def __init__(self, all_features_np, all_one_hot_labels_np, indices_np, scaler_obj, process_background=True):
        self.process_background = process_background
        if len(indices_np) == 0:
            self.features_scaled_tensor = torch.empty(0, all_features_np.shape[1] if all_features_np.ndim > 1 else 0, dtype=torch.float32)
            self.processed_labels_tensor = torch.empty(0, dtype=torch.long)
            return

        features_subset_np = all_features_np[indices_np]
        one_hot_labels_subset_np = all_one_hot_labels_np[indices_np]
        
        if hasattr(scaler_obj, 'mean_') and scaler_obj.mean_ is not None:
            features_scaled_np = scaler_obj.transform(features_subset_np)
            # --- ADD NaN/Inf CHECK FOR SCALED FEATURES ---
            if np.any(np.isnan(features_scaled_np)) or np.any(np.isinf(features_scaled_np)):
                print(f"WARNING: NaN or Inf found in SCALED features for dataset (BG processed: {process_background}).")
                # Option: replace NaNs with 0 or mean, Infs with large numbers, or raise error
                features_scaled_np = np.nan_to_num(features_scaled_np, nan=0.0, posinf=1e6, neginf=-1e6) # Example fix
        else:
            print("Dataset Warning: Scaler not fitted. Using raw features.")
            features_scaled_np = features_subset_np
        self.features_scaled_tensor = torch.FloatTensor(features_scaled_np)

        indexed_raw_labels_np = np.argmax(one_hot_labels_subset_np, axis=1)
        if process_background:
            processed_labels_np = indexed_raw_labels_np - 1
            processed_labels_np[indexed_raw_labels_np == 0] = -1
        else:
            processed_labels_np = indexed_raw_labels_np
        self.processed_labels_tensor = torch.LongTensor(processed_labels_np)

    def __len__(self): return len(self.processed_labels_tensor)
    def __getitem__(self, idx): return self.features_scaled_tensor[idx], self.processed_labels_tensor[idx]

class FixedMLP(nn.Module): # Definition as before
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
        for ly in self.layers:
            if isinstance(ly, nn.Linear): nn.init.xavier_uniform_(ly.weight); nn.init.zeros_(ly.bias) if ly.bias is not None else None
    def forward(self, x): return self.layers(x)
    def get_l2_loss(self):
        l2 = torch.tensor(0., device=next(self.parameters()).device)
        for p in self.parameters(): l2 += torch.norm(p, 2)**2
        return self.l2_reg * l2

def calculate_weights_fn_robust(labels_tensor, num_eff_classes, process_bg_flag, smooth_eps=1e-9):
    print(f"Calculating weights for {num_eff_classes} classes. BG processed: {process_bg_flag}")
    if labels_tensor.numel() == 0: return torch.ones(num_eff_classes, dtype=torch.float32).to(DEVICE)
    labels_np = labels_tensor.cpu().numpy()
    if process_bg_flag:
        actual_cls_lbls_np = labels_np[labels_np >= 0]
        if len(actual_cls_lbls_np) == 0: return torch.ones(num_eff_classes, dtype=torch.float32).to(DEVICE)
        counts_np = np.bincount(actual_cls_lbls_np, minlength=num_eff_classes)
    else:
        counts_np = np.bincount(labels_np, minlength=num_eff_classes)
    
    weights_np = 1.0 / (counts_np + smooth_eps)
    # Ensure classes not present in 'labels_tensor' but part of 'num_eff_classes' get a defined weight
    weights_np[counts_np == 0] = 1.0 # Assign weight 1.0 to classes with zero count
    
    # Sanity check for NaN/Inf in weights (should not happen with epsilon and above logic)
    if np.any(np.isnan(weights_np)) or np.any(np.isinf(weights_np)):
        print("ERROR: NaN/Inf detected in calculated weights! Replacing with 1.0")
        weights_np = np.nan_to_num(weights_np, nan=1.0, posinf=1.0, neginf=1.0) # Failsafe
        weights_np[weights_np <= 0] = 1.0 # Ensure positive

    return torch.FloatTensor(weights_np).to(DEVICE)

def train_epoch_fn(model, dataloader, criterion, optimizer, device, add_l2_flag=True):
    model.train(); running_loss=0.0; total_samples=0; all_preds,all_true=[],[]
    if len(dataloader) == 0: return 0.,0.,0.
    for feats, lbls in dataloader:
        if feats.nelement()==0: continue
        feats, lbls = feats.to(device), lbls.to(device)
        optimizer.zero_grad(); outputs = model(feats)
        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
            print(f"NaN/Inf DETECTED IN MODEL OUTPUTS (TRAIN). Skipping batch. Feats sum: {feats.sum().item()}")
            continue
        cls_loss = criterion(outputs, lbls)
        loss = cls_loss + model.get_l2_loss() if add_l2_flag and hasattr(model,'get_l2_loss') else cls_loss
        if torch.isnan(loss) or torch.isinf(loss): print(f"NaN/Inf TRAIN loss! Cls: {cls_loss.item()}. Skipping."); continue
        loss.backward(); optimizer.step()
        running_loss += loss.item()*feats.size(0); total_samples += feats.size(0)
        mask = (lbls != criterion.ignore_index) if criterion.ignore_index is not None else torch.ones_like(lbls,dtype=torch.bool)
        if mask.sum()>0: all_preds.extend(torch.argmax(outputs[mask],dim=1).cpu().numpy()); all_true.extend(lbls[mask].cpu().numpy())
    if total_samples==0: return 0.,0.,0.
    loss_val = running_loss/total_samples
    acc = accuracy_score(all_true,all_preds) if len(all_true)>0 else 0.
    ulbls = np.unique(all_true+all_preds) if len(all_true)>0 else np.array([])
    if criterion.ignore_index is not None: ulbls=ulbls[ulbls!=criterion.ignore_index]
    f1 = f1_score(all_true,all_preds,labels=ulbls if len(ulbls)>0 else None,average='macro',zero_division=0) if len(all_true)>0 else 0.
    return loss_val, acc, f1

def evaluate_model_fn(model, dataloader, criterion, device, scenario_name_for_debug="Eval"):
    model.eval(); running_loss=0.0; total_samples=0; all_preds,all_true=[],[]
    if len(dataloader) == 0: return 0.,0.,0.
    with torch.no_grad():
        for batch_idx, (feats, lbls) in enumerate(dataloader):
            if feats.nelement()==0: continue
            feats, lbls = feats.to(device), lbls.to(device)
            outputs = model(feats)
            # --- ADDED PRE-CRITERION OUTPUT CHECK ---
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                print(f"ERROR ({scenario_name_for_debug}): NaN/Inf DETECTED IN MODEL OUTPUTS (VALIDATION batch {batch_idx}).")
                print(f"  Features sum: {feats.sum().item()}, Features norm: {torch.norm(feats).item()}")
                # For more detail, print a sample: print(f"  Feature sample: {feats[0, :10]}")
                # Skip this batch for loss calculation if outputs are bad
                running_loss = float('nan') # Mark epoch loss as NaN
                break # Exit batch loop for this epoch
            
            loss = criterion(outputs, lbls)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR ({scenario_name_for_debug}): NaN/Inf VAL loss in batch {batch_idx}! Cls_loss: {loss.item()}") # loss.item() might be nan
                # --- UNCOMMENT FOR FULL DEBUG ON NAN LOSS BATCH ---
                # print(f"  Val Batch {batch_idx} - Labels: {lbls.cpu().numpy()}")
                # print(f"  Val Batch {batch_idx} - Unique Labels: {torch.unique(lbls).cpu().numpy()}")
                # if criterion.ignore_index is not None and torch.all(lbls == criterion.ignore_index):
                #     print(f"  VAL BATCH WARNING ({scenario_name_for_debug}): All labels in this batch are ignore_index.")
                # print(f"  Model Outputs (norm): {torch.norm(outputs).item()}")
                # print(f"  Model Outputs (sample of first 5, first 10 logits): \n{outputs[:min(5, outputs.size(0)), :10].cpu().numpy()}")
                # print(f"  Criterion ignore_index: {criterion.ignore_index}")
                # print(f"  Criterion weights sum: {(criterion.weight.sum().item()) if criterion.weight is not None else 'No weights'}")
                # try:
                #     torch.save({'f':feats.cpu(),'l':lbls.cpu(),'o':outputs.cpu()}, f'nan_val_batch_{scenario_name_for_debug}_{batch_idx}.pt')
                #     print(f"  Saved problematic batch to nan_val_batch_{scenario_name_for_debug}_{batch_idx}.pt")
                # except Exception as e_save: print(f"Could not save batch: {e_save}")
                # --- END DEBUG ---
                running_loss = float('nan') # Propagate NaN for epoch loss
                break # Exit batch loop for this epoch

            running_loss += loss.item()*feats.size(0); total_samples += feats.size(0)
            mask = (lbls != criterion.ignore_index) if criterion.ignore_index is not None else torch.ones_like(lbls,dtype=torch.bool)
            if mask.sum()>0: all_preds.extend(torch.argmax(outputs[mask],dim=1).cpu().numpy()); all_true.extend(lbls[mask].cpu().numpy())
    
    if total_samples==0 or np.isnan(running_loss): return float('nan') if np.isnan(running_loss) else 0.,0.,0.
    loss_val = running_loss/total_samples
    acc = accuracy_score(all_true,all_preds) if len(all_true)>0 else 0.
    ulbls = np.unique(all_true+all_preds) if len(all_true)>0 else np.array([])
    if criterion.ignore_index is not None: ulbls=ulbls[ulbls!=criterion.ignore_index]
    f1 = f1_score(all_true,all_preds,labels=ulbls if len(ulbls)>0 else None,average='macro',zero_division=0) if len(all_true)>0 else 0.
    return loss_val, acc, f1

def run_scenario(scenario_name, process_bg, use_weights, *args): # Unpack args
    all_features_np, all_labels_one_hot_np, train_indices_np, val_indices_np, scaler_obj, \
    num_epochs, batch_size, device, learning_rate = args

    print(f"\n\n--- Running {scenario_name} ---")
    print(f"Process BG: {process_bg}, Use Weights: {use_weights}")
    num_model_out_classes = NUM_ACTUAL_CLASSES if process_bg else NUM_TOTAL_CLASSES_WITH_BG
    train_ds = BrainVoxelPyTorchDataset(all_features_np,all_labels_one_hot_np,train_indices_np,scaler_obj,process_bg)
    val_ds = BrainVoxelPyTorchDataset(all_features_np,all_labels_one_hot_np,val_indices_np,scaler_obj,process_bg)
    if len(train_ds)==0:
        print(f"SKIPPING {scenario_name}: Train dataset empty.")
        empty_h = {k:[] for k in ['train_loss','train_acc','train_f1','val_loss','val_acc','val_f1']}
        return {'val_acc':0.,'val_f1':0.,'history':empty_h}
    train_loader = DataLoader(train_ds,batch_size,shuffle=True,num_workers=0,pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_ds,batch_size,shuffle=False,num_workers=0,pin_memory=(device.type=='cuda'))
    model = FixedMLP(num_output_classes=num_model_out_classes).to(device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    weights_tensor = calculate_weights_fn_robust(train_ds.processed_labels_tensor,num_model_out_classes,process_bg) if use_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights_tensor,ignore_index=-1 if process_bg else -999).to(device) # -999 effectively means no ignore
    print(f"  Model classes: {num_model_out_classes}, Criterion ignore: {criterion.ignore_index if process_bg else 'None'}")
    history={k:[] for k in ['train_loss','train_acc','train_f1','val_loss','val_acc','val_f1']}
    for ep in range(num_epochs):
        st_time = time.time()
        tr_l,tr_a,tr_f1 = train_epoch_fn(model,train_loader,criterion,optimizer,device)
        val_l,val_a,val_f1 = evaluate_model_fn(model,val_loader,criterion,device, scenario_name)
        for k,v in zip(history.keys(),[tr_l,tr_a,tr_f1,val_l,val_a,val_f1]): history[k].append(v)
        print(f"{scenario_name} - Ep {ep+1}/{num_epochs} ({(time.time()-st_time):.1f}s) -> Tr L:{tr_l:.4f} A:{tr_a:.3f} F1:{tr_f1:.3f} | Val L:{val_l:.4f} A:{val_a:.3f} F1:{val_f1:.3f}")
        if np.isnan(val_l): print(f"VAL LOSS IS NAN FOR {scenario_name}, EPOCH {ep+1}. STOPPING SCENARIO."); break
    fin_val_a = history['val_acc'][-1] if history['val_acc'] and not np.isnan(history['val_acc'][-1]) else 0.
    fin_val_f1 = history['val_f1'][-1] if history['val_f1'] and not np.isnan(history['val_f1'][-1]) else 0.
    del model,train_ds,val_ds,train_loader,val_loader,criterion,optimizer; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    return {'val_acc':fin_val_a, 'val_f1':fin_val_f1, 'history':history}

def main_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Main Experiment Starting ---")
    all_f_np, all_l_np, p_idx_np, scaler = None,None,None,None
    tr_idx, vl_idx, ts_idx = np.array([]),np.array([]),np.array([])
    try:
        mat_arrays = load_mat_data_from_h5(MAT_FILE_PATH)
        all_f_np,all_l_np,p_idx_np = check_and_transpose_data_fn(mat_arrays)
        tr_idx,vl_idx,ts_idx = split_data_fn(len(all_f_np),p_idx_np)
        scaler = create_scaler_fn(all_f_np,tr_idx)
    except Exception as e: print(f"Data loading/prep ERROR: {e}\n{traceback.format_exc()}"); return
    
    results_summary={}
    scenarios_to_run = [
        {"name": "S1(WithBG,WithW)", "pb": True, "uw": True}, {"name": "S2(NoBG,WithW)", "pb": False, "uw": True},
        {"name": "S3(WithBG,NoW)", "pb": True, "uw": False}, {"name": "S4(NoBG,NoW)", "pb": False, "uw": False}]
    
    common_args = (all_f_np, all_l_np, tr_idx, vl_idx, scaler, NUM_EPOCHS_DEMO, BATCH_SIZE, DEVICE, LEARNING_RATE)

    for sc_conf in scenarios_to_run:
        if len(tr_idx) == 0: # Critical check
             print(f"SKIPPING {sc_conf['name']} due to empty training set.")
             results_summary[sc_conf['name']] = {'val_acc':0.,'val_f1':0.,'history':{k:[] for k in ['train_loss','train_acc','train_f1','val_loss','val_acc','val_f1']}}
             continue
        sc_res = run_scenario(sc_conf["name"],sc_conf["pb"],sc_conf["uw"], *common_args)
        results_summary[sc_conf["name"]] = sc_res

    summary_fp = os.path.join(OUTPUT_DIR, "final_summary.txt")
    print(f"\n--- Final Summary (saved to {summary_fp}) ---")
    with open(summary_fp, "w") as f:
        hdr = f"{'Scenario':<30} | {'Val Acc':<10} | {'Val F1':<10}\n"+("-" * 55)+"\n"
        print(hdr.strip())
        f.write(hdr)
        for sc, mets in results_summary.items():
            line = f"{sc:<30} | {mets['val_acc']:.4f}{' ':<5} | {mets['val_f1']:.4f}\n"
            print(line.strip()); f.write(line)
    
    if results_summary:
        fig,axes = plt.subplots(2,1,figsize=(15,12))
        for i,mk in enumerate(['val_f1','val_acc']):
            ax=axes[i]; mn="Macro F1" if mk=='val_f1' else "Accuracy"
            for sc_n,res_d in results_summary.items():
                h=res_d.get('history',{}); mv=np.array(h.get(mk,[]))
                if len(mv)==0: continue
                eps=np.arange(1,len(mv)+1); idxs=~np.isnan(mv)
                ax.plot(eps[idxs],mv[idxs],marker='o' if i==0 else 'x',label=sc_n)
            ax.set_title(f'Validation {mn} (Epochs:{NUM_EPOCHS_DEMO})'); ax.set_xlabel('Epoch'); ax.set_ylabel(mn)
            ax.legend(loc='center left',bbox_to_anchor=(1.01,0.5)); ax.grid(True); ax.set_xticks(np.arange(1,NUM_EPOCHS_DEMO+1))
        plt.tight_layout(rect=[0,0,0.85,1]); plot_fp = os.path.join(OUTPUT_DIR,"validation_curves.png")
        plt.savefig(plot_fp); plt.close(fig); print(f"Plots saved to {plot_fp}")
    else: print("No results to plot.")

if __name__ == "__main__":
    import traceback # For more detailed error messages if main_experiment fails
    try:
        main_experiment()
    except Exception as e_main:
        print(f"An error occurred in main_experiment: {e_main}")
        print(traceback.format_exc())
    print("Comparison script finished.")