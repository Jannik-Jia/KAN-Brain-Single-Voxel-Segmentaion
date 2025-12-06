# ResNet50Patch on Alex7T-102labels

## 1. Role in the thesis
Deeper spatial model (~50M parameters) to test whether richer spatial processing and imbalance-aware losses outperform the lightweight ConvPatch2D and MLP baselines on 7×7 patches.

## 2. Code files and entry points
- `training/3D CNN/ResNet/scripts/train_mri_resnet.py`: main trainer with loss/augmentation options, early stopping, and logging.
- `training/3D CNN/ResNet/models/resnet.py`: MRI-optimised ResNet-50 definition (expand-first stem, base_width=104).
- `training/3D CNN/ResNet/models/dataset.py`: patch loader with balancing/augmentation; `models/losses.py`: focal/class-balanced/logit-adjusted losses and mixup helpers.
- `training/3D CNN/ResNet/configs/default_config.json`: default hyperparameters; `scripts/run_leave_one_out.sh` for batch CV; `scripts/analyze_resnet_results.py` for result aggregation.

## 3. Dataset and labels
- Dataset: same Alex 7T MAT files (384×336×256×351), using 7×7 patches by default.
- Inputs: 351 channels per patch; optional class-balanced sampling or weighted sampler.
- Labels: 102 hard classes (label shifted to 0–101 inside loader); mask removes background.
- Splits: leave-one-out by subject; optional sampling per subject (default 10k) or described “memory-efficient” full-data mode.

## 4. Preprocessing pipeline
- Per-channel z-score per subject; cached in memory if enabled.
- Optional augmentation: random flips/rotations and light Gaussian noise (when `--augmentation`).
- Weighted sampling and class weight computation available for imbalance mitigation.

## 5. Model architecture
- ResNet-50 (3-4-6-3 bottleneck blocks) with expand-first 3×3 stem (351→512), no max pooling, base_width=104 (~50M params).
- Spatial flow: 7→4→2→1; channel flow: 512→256→512→1024→2048→102 classifier.
- Optional EMA model tracking, mixup support, gradient clipping.

## 6. Training configuration
- Loss options: CE, weighted CE, focal, class-balanced focal (default gamma=1.5, beta=0.9999), logit-adjusted CE (tau), balanced softmax; optional label smoothing.
- Optimiser AdamW (lr=1e-4, weight_decay=1e-4); CosineAnnealingLR (T_max=100, eta_min=1e-6); gradient clip=1.0.
- Batch size 256; epochs=100; patience=15 for early stopping; optional mixup (alpha=0.2) and EMA (decay=0.999).

## 7. Evaluation metrics and outputs
- Metrics: macro-F1 for train/test, per-class precision/recall/F1/support; learning-rate and overfitting diagnostics.
- Outputs: `best_model.pth`, checkpoints every 10 epochs, `training_results.json`, `training_history.png`, `training.log` under `resnet_test_subject_*` folders.

## 8. Thesis-ready interpretation
Tests whether deeper spatial modeling plus imbalance-aware objectives yield gains over simpler patch CNNs and MLPs. Results inform the ablation narrative on spatial context, class imbalance handling, and advanced optimisation tricks (mixup/EMA).

## 9. Limitations and open questions
- “Memory-efficient” full-data mode is described but default loader still samples; impact of full 71M-voxel training is unverified here.
- No explicit calibration metrics; tuning of loss hyperparameters (gamma/beta/tau) may affect outcomes.
- Patch size fixed to 7 unless retrained; data paths must be set manually.
