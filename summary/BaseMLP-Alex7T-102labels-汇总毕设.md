
# BaseMLP on Alex7T-102labels

## 1. Role in the thesis
- Serves as the primary fully connected baseline (4x4096 units) for voxel-wise tissue classification on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Establishes reference performance for later ablations (deep MLP, residual MLP, Bayesian hyperparameter search) and for calibration/inference pipelines.

## 2. Code files and entry points
- `config.py`: default hyperparameters (4x4096 ReLU MLP, dropout 0.5, AdamW 1e-5, batch 128, 30 epochs, cosine LR); patient ID splits defined here.
- `main.py`: CLI training driver; loads patient-based data, optionally runs Optuna search, trains, evaluates, and writes reports/checkpoints.
- `train.py`: training loop with class-weighted cross-entropy, validation every `val_epochs`, checkpoint naming with epoch/acc/F1.
- `eval.py` and `evaluate.py`: wrappers to reload saved checkpoints and rerun evaluation.
- Data pipeline: `utils/patient_data_adapter.py` (patient-based loaders, StandardScaler saving), `BrainVoxel38PatientLoader.py` (per-patient loaders, one-hot labels), `data/dataset.py` and `data/samplers.py` (legacy label-index loading, optional PCA), notebooks (`fullyconnected_brainvoxel_完全分离的数据集创建方法_102分类_L1.ipynb`, `数据库再次修改接入38折验证.ipynb`, etc.) documenting data restructuring.
- Models: `models/base_mlp.py`; model factory in `models/__init__.py`.
- Metrics/visuals: `utils/metrics.py`, `utils/visualization.py`; logging and checkpoint I/O in `utils/model_io.py`.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021, reorganized into 38 patient IDs with per-voxel features.
- Input features: 341 channels (CEST/QTI/SMWI/MPRAGE-derived voxel signatures); default `feature_dim=341`; PCA disabled by default but supported.
- Labels: 102-region cortical/subcortical/parcellation classes; raw label 0 is treated as background and mapped to ignore index -1 during training.
- Splits: patient-based; default train IDs 1–20, val 21–28, test 29–38 (configurable); patient-based loaders ensure no voxel leakage across subjects.
- Sampling: no explicit class balancing; class weights computed as inverse frequency via `utils/metrics.calculate_class_weights`.

## 4. Preprocessing pipeline
- Standardization with `sklearn.StandardScaler` fitted on training features; scaler is saved to `save_dir/scalers/...pkl` and reused for val/test and inference.
- Background voxels mapped to -1 and ignored in loss/metrics; one-hot labels from patient loader collapsed to indices.
- Optional PCA in `data/dataset.apply_pca`/`load_multiclass_data` (not used in default patient-based flow).
- No dimensionality reduction or feature selection beyond optional PCA; no explicit missing-value handling beyond dataset construction.

## 5. Model architecture
- Base MLP: sequential fully connected stack with hidden sizes `[4096, 4096, 4096, 4096]`, ReLU/GELU/Swish activation selectable, dropout 0.5 after each hidden layer, linear output to 102 logits.
- No skip/residual connections; purely dense feedforward; parameter counts logged via `get_model_info` when saved.
- Config tags: `model_type='base_mlp'`, `model_name='BrainVoxel_102Class_MLP'`.

## 6. Training configuration
- Loss: class-weighted `CrossEntropyLoss` with ignore index -1 for background.
- Optimizer: AdamW (default) or Adam; lr 1e-5, weight_decay 1e-5.
- Scheduler: optional (default cosine over 30 epochs); alternatives multistep/plateau supported via CLI.
- Batch size 128; epochs 30; validation every 3 epochs; no data augmentation.
- Bayesian search (enabled by default in `config.py`) can sweep lr, weight decay, dropout, activation, scheduler, and alternative layer sizes, but baseline can be run without search by disabling `--run_bayesian_opt`.
- Class weights computed per run; gradient clipping not used.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen’s kappa; per-class precision/recall/F1 and sample counts.
- Calibration-style artifacts: softmax probabilities saved to `.npz`; confusion matrix (log-scaled heatmap) and per-class bar charts saved via `utils/visualization`.
- Outputs: training logs `*_training_log.txt`, metrics CSV `*_metrics.csv`, per-split `*_class_metrics.csv`, predictions `.npz`, confusion matrix `.png`, training curves `training_curves.png`, `evaluation_summary.txt`, `final_report.txt`, and checkpoints named with epoch/acc/F1 under `results/<experiment_name>/`.
- Best model selection via `utils/metrics.get_best_model` based on validation macro F1; summary printed in console and saved.

## 8. Thesis-ready interpretation
- Tests how far a plain 4x4096 fully connected classifier can go on voxel signatures from the Alex ultra-multimodal 7T dataset; forms the main performance and calibration reference.
- Demonstrates that patient-level splits and standardization are sufficient for a strong baseline before exploring architectural changes or calibration tricks.
- Results (accuracy/F1/kappa, confusion matrices) can populate a “Baseline voxel-wise classifier” subsection and support later comparisons.

## 9. Limitations and open questions
- Absolute voxel counts and class balance summaries depend on external `label_index.txt`/reorganized data and are not embedded in the repo (numbers must be pulled from runtime logs).
- Paths to data (`/home/jovyan/.../reorganized_fold_data`) are hard-coded and need adjustment on new machines.
- No explicit early stopping in the baseline training loop; overfitting control relies on dropout and weight decay.
- Does not include explicit calibration metrics (ECE/MCE) beyond accuracy/F1/kappa.
