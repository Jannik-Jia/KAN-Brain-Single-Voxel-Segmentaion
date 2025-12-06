# FC4x4096 on Alex7T-softlabels

## 1. Role in the thesis
Soft-label variant of the 4×4096 network trained on downsampled CEST-resolution data with probabilistic labels, aimed at exploiting partial-volume information and assessing calibration/uncertainty handling.

## 2. Code files and entry points
- `training/downsampling/train_runner.py`: main single-split trainer (36 train / 1 val / 1 test) with soft-label metrics and 3D reconstruction.
- `training/downsampling/notebook_quickstart.ipynb`, `training/downsampling/notebook_loso_37fold.ipynb`, `training/downsampling/NOTEBOOK_VERIFICATION.md`: notebook-driven runs and verification steps.
- `dataset_create/downsampling/mri_downsampling_pipeline.py`, `dataset_create/downsampling/DOWNSAMPLED_DATA_FORMAT.md`: generate CEST-resolution NPZ files with soft labels using modality-specific PSF/spacing.
- `dataset_create/1d-3d-convert/data_3d_1d_mapper.py`: shared converter for 3D↔1D mapping, maintaining C-order voxel alignment.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T data downsampled to CEST resolution (≈1.8×1.8×3.0 mm³), stored as paired 1D/3D NPZ files.
- Inputs: 351-channel features per voxel (`multidim_data`, shape (n_vox, 351)).
- Labels: 102-class probabilistic labels (`seg_one_hot` / `proba_labels`) preserving partial-volume information; background masked by `region`.
- Splits: deterministic 36/1/1 subject split (train/val/test); optional subject overrides via CLI.

## 4. Preprocessing pipeline
- Per-subject z-score over 351 channels; optional cap on voxels per subject for memory control.
- Downsampling pipeline applies family-specific PSF/resampling and recomputation for parameter maps; masks keep ROI alignment.
- Data3D1DMapper restores predictions to 3D softmax volumes while preserving voxel order; normalisation stats logged per subject.

## 5. Model architecture
- Dense 4×4096 MLP with Dropout(0.5) and ReLU; 102-way output logits.
- Manual L2 regularisation on weights (1e-5); no batch norm or attention.

## 6. Training configuration
- Loss: soft Cross-Entropy against probability labels, optional class weights (alpha=0.5), L2 regularisation, gradient clipping (default 1.0).
- Optimiser Adam (lr=1e-5, weight_decay via manual L2); default batch size 256; default epochs 3 (quick runs) with best checkpoint by val NLL.
- Temperature scaling applied post-hoc for calibration; random seed 42 by default.

## 7. Evaluation metrics and outputs
- Metrics: NLL, gross accuracy, macro/micro/weighted F1, top-k accuracy, Brier score and Murphy decomposition, class-mass error, soft-ECE (per-class and mean), AURC (risk–coverage), entropy histograms.
- Calibration: reliability diagrams, temperature-scaling summaries, soft confusion matrices.
- 3D metrics: soft Dice (macro), 3D NLL/Brier within ROI, slice plots after restoration.
- Outputs: best model (`best.pth`), `metrics_{val,test}.json`, `temperature_scaling.json`, `run_summary.json`, confusion CSVs, reliability/risk-coverage/entropy plots, 3D predictions (`val_*/test_*_pred_softmax_3d.npz`).

## 8. Thesis-ready interpretation
Tests whether probabilistic labels and calibration improve voxel-wise prediction and uncertainty estimates relative to the hard-label baseline. The pipeline highlights calibration-focused metrics (soft-ECE, AURC, Brier) and enables reporting of 3D probability quality (soft Dice), informing sections on soft labels and calibration.

## 9. Limitations and open questions
- Default runs are short (epochs=3); longer training may be needed for peak performance.
- Assumes availability of downsampled NPZ data and consistent CEST-resolution masks.
- No full leave-one-out; results represent a single 36/1/1 split unless expanded manually.
