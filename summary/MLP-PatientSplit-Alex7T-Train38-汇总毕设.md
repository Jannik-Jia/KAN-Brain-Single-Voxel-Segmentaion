# MLP-PatientSplit on Alex7T-Train38

## 1. Role in the thesis
- Support pipeline for patient-wise splits and deployment on `.mat` datasets (e.g., TRAIN38/DEMO38), enabling evaluation of trained MLPs on new subjects and reconstruction of 3D label volumes.
- Complements the main FC/DeepMLP training by providing data conversion, scaling, and prediction tooling for external cases.

## 2. Code files and entry points
- `data/mat_loader.py`: loads `.mat` files (keys `data`/`multidim_data`, `region`, `prob_idx`), splits by patient IDs, fits/applies StandardScaler, builds PyTorch dataloaders via `load_and_process_data`.
- `predict.py`, `predict.sh`: load saved MLP checkpoints (with architecture metadata), apply scaler or embedded normalisation params, predict voxel classes, threshold low-confidence voxels, and map predictions back to 3D volumes; save `.mat` outputs and optional 3D visualisations.
- `predict_standardized.py`: simplified demo38 pipeline that standardises on the fly and writes prediction/probability volumes.
- `predict_config.json`: default paths/keys for inference; `utils/model_io.py` used for safe checkpoint loading.

## 3. Dataset and labels
- Input: MATLAB `.mat` volumes; features typically 341 channels (matching Alex 7T voxel signatures), labels in `region` arrays; patient IDs in `prob_idx` for splitting.
- Default split logic: if no IDs are provided, use `prob_idx != 38` for training and `prob_idx == 38` for validation, then carve out a small test subset; configurable patient lists in `config['dataset_split']`.
- Label space likely the same 102-class atlas; background handling follows upstream models; hard labels only.
- Splits are patient-wise when IDs are supplied; otherwise rely on the default heuristic.

## 4. Preprocessing pipeline
- StandardScaler fitted on training samples inside `process_train38_data`; saved as `scaler.joblib` or loaded from checkpoints for eval/inference.
- No PCA in this path; region masks are used to reshape flat predictions back to 3D volumes.

## 5. Model architecture
- Reuses saved MLP architectures via checkpoint metadata (base/deep/residual); inference scripts construct the appropriate model with recorded hyperparameters.
- Designed for dense tabular voxel features; activation/dropout/skip/bottleneck options depend on the loaded checkpoint.

## 6. Training configuration
- This module prepares dataloaders and scaling; training itself would mirror the main loops (cross-entropy with optional class weights), but no dedicated patient-split training script is present here.
- Prediction scripts focus on inference using existing weights; optional probability thresholding marks uncertain voxels as unknown (-1).

## 7. Evaluation metrics and outputs
- When plugged into the main evaluation utilities, metrics match the baseline set (accuracy, balanced accuracy, macro/weighted F1, kappa, per-class reports).
- Prediction outputs: `predictions.mat`, `probabilities.mat`, `volume_3d.mat`, optional `probability_volume.mat`, `normalization_info.txt`, and 3D visualisation PNGs; logs stored alongside results.

## 8. Thesis-ready interpretation
- Provides the deployment pathway to test MLP classifiers on new 7T acquisitions and to reconstruct volumetric segmentations from voxel-level predictions.
- Useful for demonstrating generalisation to held-out subjects (e.g., patient 38) and for producing qualitative figures of predicted tissue maps.

## 9. Limitations and open questions
- Number of classes and label semantics in external `.mat` files must match the trained model; otherwise, evaluation is undefined.
- No dedicated end-to-end training script for the patient-split data; assumes existing checkpoints trained on the restructured dataset.
- Scaling strategy depends on the availability of the saved scaler or embedded normalisation params; mismatch could harm performance.
