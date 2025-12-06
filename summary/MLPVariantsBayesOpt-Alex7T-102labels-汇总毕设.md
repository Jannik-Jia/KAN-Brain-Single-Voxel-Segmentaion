
# MLPVariantsBayesOpt on Alex7T-102labels

## 1. Role in the thesis
- Core voxel-wise baseline for the Alex ultra-multimodal 7T dataset (German et al. 2021), using fully-connected MLPs and BayesOpt to tune depth/width/activation.
- Establishes reference performance for 102-class tissue/parcellation labels (background included) under both MAT and directory-style data layouts.
- Provides calibrated training/evaluation pipeline (train/val/test splits, metrics, checkpointing) used to benchmark later architectures.

## 2. Code files and entry points
- main.py: CLI entry; parses args, loads config, sets up data loaders (MAT or original), builds model (base/deep/residual MLP), optionally runs Bayesian optimisation, trains and evaluates.
- config.py: default hyperparameters, dataset paths, label format handling, PCA/normalisation flags, patient split hints; auto-detects MAT vs directory data.
- train.py: training loop for multiclass MLP; logs metrics, saves checkpoints/CSV logs, selects best model by val F1.
- data/mat_loader.py & data/dataset.py: MAT loader with background filtering (label 0), patient split (prob_idx==38 as test; remaining split 75/25 train/val), StandardScaler; optional PCA/normalisation; BrainVoxelDataset for directory data.
- models/base_mlp.py, models/deep_mlp.py, models/residual_mlp.py, models/__init__.py: 4×4096 baseline plus deep and residual variants with activation/dropout options.
- utils/metrics.py, utils/visualization.py, utils/model_io.py, utils/optimization.py: metrics (accuracy/F1/balanced acc/kappa, confusion matrices), curve plotting, safe save/load with architecture metadata, Optuna BayesOpt search space.
- evaluate.py: reload trained checkpoints and run full evaluation.
- predict.py & predict_config.json: inference/mapping back to 3D grids (supports DEMO38.mat), scaler handling, optional .mat export.
- run.sh, predict.sh: runnable presets for training (MAT/original; optional BayesOpt) and inference with scaler lookup.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; TRAIN38.mat/DEMO38.mat or restructured .npy directories.
- Input dimensionality: 341 scalar features per voxel (modalities aggregated to per-voxel vectors; PCA optional but off by default).
- Label space: 102 classes including background; MAT loader filters out background voxels then remaps 1–102 → 0–101; directory loader treats 0 as background and shifts to 0–101 indices.
- Splits: MAT path uses prob_idx==38 as held-out test; remaining subjects randomly split 75/25 into train/val. Directory path relies on pre-separated train/test/val folders; config also lists patient IDs for a subject-wise split template. Test_size CLI can override MAT split (default 1%).
- Sampling/balancing: class-weight computation utilities exist but default training uses unweighted CE; BayesOpt uses smaller epoch budgets for speed.

## 4. Preprocessing pipeline
- Background voxels removed before scaling; labels converted to contiguous class indices.
- StandardScaler fitted on training voxels (MAT) or per-feature z-scoring from training samples; scaler saved alongside checkpoints (scaler.joblib).
- Optional PCA (n_components configurable; helper to analyse explained variance) but default n_pca=0.
- Optional normalisation parameters (mean/std) computed for training_curves plots and stored with checkpoints.

## 5. Model architecture
- base_mlp: sequential Linear/Activation/Dropout stacks (default hidden_units=[4096,4096,4096,4096], activation relu/gelu/swish, dropout 0.5) → Linear to 102 logits.
- deep_mlp: configurable depth (BayesOpt 5–8 layers) with constant width (1024/2048/3072) and optional skip connections.
- residual_mlp: per-block residuals with optional bottleneck (factor 0.5) over layer size options (e.g., 4×4096, 4×2048, 1024-2048-2048-1024, 4096-2048-2048-4096).
- BayesOpt space also searches activation, dropout, optimiser, scheduler, and chooses model_type.

## 6. Training configuration
- Loss: CrossEntropyLoss (no ignore_index; background kept as class 0). Class-weight computation available but commented out in main loop.
- Optimiser: AdamW (default) or Adam; lr default 1e-5, weight_decay 1e-5; batch_size 128; epochs 30; val every 3 epochs.
- LR schedulers: cosine annealing (default), multistep, or ReduceLROnPlateau; scheduler choice tunable via BayesOpt.
- BayesOpt: Optuna n_trials default 30 with reduced epochs (5–15) during search; explores model depth/width, dropout, activation, optimiser, scheduler, lr/weight_decay.
- Checkpointing: saves per-val-epoch models with metric tags; selects best by val F1; JSON architecture dump saved alongside .pth.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen’s kappa; per-class precision/recall/F1; confusion matrices; train/val curves.
- Higher-is-better for accuracy/F1/kappa/balanced acc. Confusion matrices visualise class confusions.
- Outputs: results/<experiment_name>/ contains *_training_log.txt, *_metrics.csv, training_curves.png, evaluation_summary.txt, final_report.txt, best checkpoints (.pth + *_architecture.json), optional scaler.joblib, plots from utils.visualization.

## 8. Thesis-ready interpretation
- Establishes a strong FC baseline and hyperparameter-tuned variants for voxel-wise classification on the Alex ultra-multimodal dataset. Demonstrates impact of depth/width/activation/dropout choices and learning-rate schedules on 102-class performance.
- Results support later comparisons (e.g., TabNet/KAN/other multimodal models) by providing reproducible train/val/test metrics and calibrated reporting.

## 9. Limitations and open questions
- Data paths are hard-coded to external locations (/home/jovyan/…); dataset not bundled in repo. Ensure paths updated for local runs.
- Background treated as a learnable class (no ignore_index); may affect calibration vs. foreground-only training.
- No explicit calibration metrics (ECE/NLL) or uncertainty estimation implemented; only accuracy/F1/kappa reported.
- BayesOpt runtime can be high; search uses shortened epoch budgets, so final training should be rerun with full epochs.
