# FC4x4096 on Alex7T-102labels

## 1. Role in the thesis
Replicates the original Alex fully-connected voxel classifier (4×4096) on the Alex ultra-multimodal 7T dataset, establishing the hard-label baseline. Produces per-voxel probability volumes that feed later post-processing and comparisons to spatial or calibration-focused models.

## 2. Code files and entry points
- `training/B0_1D_training/train_1d_with_3d_dataset.py`: main training/prediction on 1D MAT files with reconstruction to 3D volumes.
- `training/B0_1D_training/run_1d_training.sh`, `training/B0_1D_training/run_1d_leave_one_out.sh`: wrappers for single run or full 38-fold leave-one-out.
- `training/B0_1D_training/visualize_1d_3d_predictions.py`: slice visualisation and quick accuracy checks from saved probability maps.
- `training/B0_1D_training/README_1D_TRAINING.md`: usage notes and architecture recap.
- `erosion/train_38fold.py`, `erosion/run_training.sh`, `erosion/analyze_results.py`: extended cross-validation with AMP/DataParallel, ONNX export, and fold-wise reporting.
- `erosion/adjacency_matrix/compute_adjacency_matrices.py`: optional adjacency computation of the 102 regions for error/confusion analysis.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; 38 subjects with volumes (384, 336, 256).
- Inputs: 351-modal voxel features (`multidim_data`, transposed to (n_voxels, 351)) aligned to region masks.
- Labels: 102-class hard labels derived from FreeSurfer parcellations (`seg_one_hot` → argmax); background voxels excluded.
- Splits: leave-one-out by subject (37 train / 1 test), with optional per-subject sampling; erosion scripts also support internal val splits.

## 4. Preprocessing pipeline
- Per-subject z-score normalisation across the 351 channels (StandardScaler).
- Optional voxel subsampling per subject for speed; otherwise full ROI coverage.
- Region masks enforce valid voxels; predictions are mapped back to 3D (384×336×256×102) via Boolean indexing.
- Shape validation and transpose handling to keep channel/voxel ordering consistent.

## 5. Model architecture
- Fully connected network: 351 → 4096 → 4096 → 4096 → 4096 → 102 with ReLU + Dropout(0.5) after each hidden layer.
- No batch/weight norm; manual L2 penalty on weight matrices only (1e-5).
- ONNX export supported in the erosion pipeline for deployment.

## 6. Training configuration
- Optimiser Adam (lr=1e-5) with manual L2 penalty; batch size 128; epochs=25.
- Loss: Cross-Entropy on hard labels; no label smoothing or class weighting by default.
- Leave-one-out subject selection via CLI; erosion variant enforces GPU use, supports AMP and multi-GPU DataParallel.
- Logs train/test loss and macro-F1 per epoch; best model cached by test F1.

## 7. Evaluation metrics and outputs
- Metrics: macro-F1, gross accuracy; erosion pipeline adds per-class precision/recall/F1, macro/micro/weighted averages, and confusion reports.
- Outputs: `.pth` checkpoints (per test subject), optional `.onnx`, training history JSON/logs, 3D probability volumes (`predictions_3d_test*.mat`), slice/uncertainty visualisations.

## 8. Thesis-ready interpretation
Baseline voxel-wise classifier mirroring the published Alex setup, establishing how far a non-spatial MLP can go on 351-channel inputs. The pipeline also demonstrates reliable reconstruction of probability maps back into 3D space, underpinning later smoothing and spatial ablations. Results from the 38-fold runs anchor comparisons against soft-label, convolutional, and calibration-focused experiments.

## 9. Limitations and open questions
- Fixed hyperparameters with no automated tuning; no explicit calibration or class-imbalance handling.
- Relies on consistent 1D MAT formatting; potential orientation risk if preprocessing changes.
- Sampling strategy and lack of augmentation may limit generalisation; only hard labels are supported.
