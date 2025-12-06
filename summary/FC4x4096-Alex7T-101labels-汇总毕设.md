# FC4x4096 on Alex7T-101labels

## 1. Role in the thesis
Fully connected baseline for voxel-wise classification on the Alex ultra-multimodal 7T dataset from German et al. 2021. Establishes a reproducible reference (patientwise standardisation, 101-class with background) against which deeper/residual variants and later architectures can be compared.

## 2. Code files and entry points
- `main.py`: CLI orchestrating config load, dataset prep, model build, training, evaluation, reporting.
- `config.py`: default hyperparameters, patientwise/global standardisation presets, background-handling presets, fixed prob_idx splits.
- `models/base_mlp.py`: 4×4096 fully connected MLP definition with configurable activation/dropout.
- `train.py`: training loop with background-aware accuracy/F1, CSV + text logging, checkpoint saving.
- `data/mat_loader_patientwise.py`, `data/mat_loader.py`: MAT loaders with patientwise or global scaling, fixed or random splits, background filtering, scaler persistence.
- `data/samplers.py`: patient-aware batch sampler and balanced patient sampler.
- `utils/metrics.py`, `utils/visualization.py`: accuracy/F1/balanced-accuracy/kappa, per-class stats, confusion matrix plots, training curves.
- `utils/label_processing.py`: maps labels for keep/ignore/filter background modes and builds criterion accordingly.
- `utils/model_io.py`: saves checkpoints with architecture metadata and normalization parameters; safe loading.
- Runners: `run.sh` (interactive training on 101-label MAT), `evaluate.py` (offline evaluation), `predict.py` + `predict.sh` (MAT inference + 3D remapping), notebooks `alex版本优化*.ipynb`, `train_with_best_hyperparams.ipynb` for manual tuning.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021, provided as `TRAIN38_no_label43.mat`.
- Input: 341-channel voxel signatures (multi-contrast features; exact modality list not encoded in code).
- Labels: 101 classes by default (0 background + 1–100 tissues; label 43 removed). Optional 100-class variant if background is filtered or ignored.
- Splits: fixed prob_idx split—validation `[20]`, test `[38]`, remaining patients for training. Presets for alternatives (e.g., test `[13,23,38]`, larger test sets) or random split with `test_size=0.01` when fixed IDs are absent.
- Sampling: optional patient-aware batches enforce ≥3 patients per batch; class weights computed from training labels for imbalance mitigation.

## 4. Preprocessing pipeline
- Patientwise standardisation (default): per-patient mean/std with epsilon 1e-10; alternative global `StandardScaler` on training data.
- Scaler saved as `scaler.joblib` and reloaded for eval/predict; normalization params also stored in checkpoints.
- PCA disabled by default (`apply_pca=False`, `n_pca=0`); hook exists but not used in configs.
- Background handling configurable: keep as class 0, ignore in loss (mapped to -1), or drop background samples entirely; labels remapped accordingly.
- Data integrity checks report label ranges and prob_idx coverage to avoid split leakage.

## 5. Model architecture
- Sequential MLP: four hidden layers of 4096 units (configurable list), activation selectable (`relu` default; `gelu`/`swish` available), dropout 0.5 after each layer, linear classifier to `num_class` outputs.
- Input dimension taken from data (341 features); no skip/residual connections.
- Checkpoints record architecture, hidden sizes, activation, dropout, normalisation parameters.

## 6. Training configuration
- Loss: cross-entropy with class weights; `ignore_index` used when background is ignored.
- Optimiser: AdamW default (Adam optional); lr=1e-5, weight_decay=1e-5; batch size 128; epochs 30; validation every 3 epochs; seed 666.
- LR schedulers: cosine annealing default; multistep and ReduceLROnPlateau available.
- Checkpoints saved every validation step with filenames containing epoch/acc/F1; logs in text and CSV under `results/<experiment_name>/` and `logs/<experiment_name>/`.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen’s kappa; per-class precision/recall/F1 and sample counts.
- Visuals: confusion matrix (log-scaled), best/worst class barplots, training curves, dataset distribution plots.
- Outputs: `<dataset>_evaluation_report.txt`, `<dataset>_class_metrics.csv`, `<dataset>_metrics.csv`, `<dataset>_predictions.npz`, `_confusion_matrix.png`, `training_curves.png`, `evaluation_summary.txt`, checkpoints `.pth` plus `_architecture.json` and scaler/joblib.
- Best model chosen by highest validation macro-F1 (`utils.metrics.get_best_model`).
- Inference (`predict.py`): loads saved scaler/normalisation, outputs `predictions.mat`, `probabilities.mat`, `volume_3d.mat`, optional 3D visualisation, and `normalization_info.txt`.

## 8. Thesis-ready interpretation
- Establishes the baseline feasibility of voxel-wise FC classification on 341-channel signatures with patientwise normalisation and explicit background treatment.
- Supplies reference confusion matrices and per-class behaviour for later comparisons (deep MLP, residual MLP, hyperparameter search, or alternative architectures like KAN/TabNet).
- Demonstrates end-to-end pipeline from MAT ingestion to 3D prediction export, supporting thesis sections on baseline models and data processing.

## 9. Limitations and open questions
- Hard-coded absolute data paths (`/home/jovyan/...`) require relocation; voxel counts per split are not logged in code.
- Validation/test each rely on one prob_idx by default (20/38), so subject-level generalisation is uncertain.
- No calibration metrics (ECE/MCE) or uncertainty estimates; no spatial context or augmentation beyond per-voxel features.
- PCA/dimensionality reduction disabled; class-imbalance mitigation limited to static weights; background-filter variant not paired with dedicated evaluation scripts.
