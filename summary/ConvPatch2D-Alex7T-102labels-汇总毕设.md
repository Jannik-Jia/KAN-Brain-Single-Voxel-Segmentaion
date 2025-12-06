# ConvPatch2D on Alex7T-102labels

## 1. Role in the thesis
Patch-based CNN baseline that injects minimal spatial context (3×3 or 7×7 in the xy-plane) while remaining lightweight. Serves as a bridge between the 1D MLP and deeper ResNet experiments.

## 2. Code files and entry points
- `training/3D CNN/train_baseline_3x3_7x7.py`: main training script with model definitions (ImprovedConv2D_Baseline and parameter-matched variants).
- `training/3D CNN/README.md`: usage guide and hyperparameter descriptions for 3×3/7×7 patches.
- `training/3D CNN/run_leave_one_out.sh`: batch leave-one-out runner; `test_data_loading.py` for data checks.
- `training/3D CNN/analyze_results.py`: aggregates leave-one-out results and plots statistics.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T 3D MAT files (384×336×256×351) with `region_labels` and `region_mask`.
- Inputs: 2D patches (3×3 or 7×7) cut in the xy-plane at fixed z; 351 channels per voxel, sampled from in-mask voxels (default 10k per train subject, 2× for test).
- Labels: 102 hard classes (labels shifted to 0–101 inside the loader); background excluded.
- Splits: leave-one-out by subject (37 train / 1 test) selected via CLI.

## 4. Preprocessing pipeline
- Per-channel z-score normalisation per subject over the full 3D volume.
- Patch extraction with padding at borders to maintain fixed patch size; caches subject data in memory when enabled.
- Valid voxel sampling restricted to mask & label > 0; optional shuffle for training.

## 5. Model architecture
- Stem: 1×1 conv mixes channels (351→mid, default mid=128) + GroupNorm + SiLU.
- Aggregation: single conv with kernel size equal to patch size (3 or 7) to collapse spatial dims + GroupNorm + SiLU.
- Optional refine 1×1 residual block; optional SE channel attention.
- Head: linear classifier or MLP head (parameter-matched variants targeting ~35M/52M/69M params with large hidden dims and dropout).

## 6. Training configuration
- Loss: Cross-Entropy; optimizer AdamW (lr=1e-4, weight_decay=1e-4); CosineAnnealingLR (T_max=50, eta_min=1e-6).
- Batch size 256; epochs=50; mixed precision via GradScaler; no explicit augmentation beyond patch sampling.
- Best checkpoint chosen by test macro-F1 each epoch; history saved to JSON.

## 7. Evaluation metrics and outputs
- Metrics: macro-F1 and average loss for train/test per epoch; parameter counts printed.
- Outputs: `best_model_patch{3|7}_test{N}.pth`, `history_patch{...}.json`, logs from leave-one-out runs.

## 8. Thesis-ready interpretation
Evaluates how small spatial context (3×3 vs 7×7) and parameter scaling affect voxel-wise accuracy relative to the 1D MLP. Provides an ablation stepping stone toward deeper ResNet architectures.

## 9. Limitations and open questions
- Operates on 2D slices only (no full 3D convolutions); limited augmentation.
- Samples a fixed number of voxels per subject, so full-data training impact is unknown.
- No explicit handling of class imbalance or calibration.
