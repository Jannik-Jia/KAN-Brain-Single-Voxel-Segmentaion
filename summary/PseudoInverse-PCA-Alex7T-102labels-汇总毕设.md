# PseudoInverse-PCA on Alex7T-102labels

## 1. Role in the thesis
- Closed-form pseudo-inverse linear baseline using PCA to compress ultra-multimodal voxel signatures; establishes a fast analytic reference against deeper networks and probes how classical dimensionality reduction and regularization influence voxel-wise performance.

## 2. Code files and entry points
- `proj/src/main.py`: CLI entrypoint toggling PCA vs feature-selection runs, logging, GPU initialization, and baseline/full sweep orchestration.
- `proj/src/brain_voxel_dataloader.py`: loads per-label `.npy` voxel arrays from `train/`/`test/`/`val/` directories using `label_index.txt`; handles normalization, PCA, class balancing, and caching of transforms.
- `proj/src/utils.py`: defines PCA-centric parameter grids and sampling helpers.
- `proj/src/experiment_manager.py`: runs experiments, caches preprocessing, trains the pseudo-inverse model, logs metrics, and emits summary HTML/CSV reports.
- `proj/src/pseudoinverse_model.py`: multi-class linear classifier solved by pseudo-inverse with optional L2 (Tikhonov) or truncated-SVD regularization; computes feature importance from weights.
- `proj/src/model_evaluator.py`: evaluates accuracy/F1/Kappa and draws confusion matrices and class-level plots.
- `proj/src/visualization_utils.py`: feature-importance and weight-distribution visualizations.
- `proj/src/gpu_utils.py`: CPU/GPU abstraction with CuPy/cuML fallbacks.
- `proj/scripts/run_baseline.sh`: example PCA baseline launcher (`EXPERIMENT_TYPE=pca`).
- `proj/scripts/run_full_experiments.sh`: full sweep launcher; can be switched to PCA mode.
- `PseudoInverse_FeatureSelection_BrainVoxel.ipynb`: earlier prototype mirroring the same pipeline.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021 reorganized as voxel-level signature arrays under train/test/val folders with per-label `label_index.txt` metadata.
- Input dimensionality: raw feature length not explicit (likely hundreds of modalities including MPRAGE/QTI/CEST/QSM); PCA explores 50–300 components or auto-selects components explaining 95% variance.
- Label space: 102 classes (pseudo-inverse default), hard labels from the label index; assumed FreeSurfer-based parcellations mapped to the CEST grid; invalid labels skipped via `valid_labels`.
- Splits: directory-based train/val/test; optional `max_samples_per_label` sub-sampling and class balancing to `target_samples` per class via over/undersampling; subject-level scheme unspecified (UNKNOWN).

## 4. Preprocessing pipeline
- Optional z-score or min-max normalization before or after PCA (`scaling_before_pca`).
- PCA dimensionality reduction with fixed `n_components` or automatic component count to hit `auto_pca_variance` (default 0.95); transforms cached via `_get_config_key` and `precompute_transformations`.
- Class balancing oversamples/undersamples each class to `target_samples`; data kept on GPU when available.
- No explicit missing-data handling beyond loading stored arrays; feature importance computed later from learned weights.

## 5. Model architecture
- Linear multi-class model solved analytically: adds a bias column and computes pseudo-inverse of the design matrix times one-hot targets.
- Regularization options: none, L2/Tikhonov with strength `alpha`, or truncated SVD with singular values below `alpha * max(s)` zeroed.
- Outputs 102-way logits; feature importance from mean absolute weights across classes.

## 6. Training configuration
- Single closed-form solve per experiment (no epochs); GPU-accelerated matrix ops when CuPy is available, CPU fallback otherwise.
- Parameter grid via `create_feature_selection_param_grid` + sampling: `n_components` {50,80,100,150,200,250,300}, normalization {standard,minmax,None}, regularization {none,l2,truncated}, `alpha` in {0.001,0.01,0.05,0.1,0.5,1.0}, class balancing on/off, `max_experiments` cap; `max_samples_per_label` limits per-class voxel count.
- Baseline runs compare PCA vs PCA+L2 vs PCA+class balancing; `auto_pca_variance` can override `n_components` by explained variance target.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa (higher is better); classification report and per-class F1 vectors retained.
- Visuals: `performance_comparison.png`, `confusion_matrix_{train|test|val}.png` (log-scaled for large matrices), `class_f1_sorted.png`, `sample_count_vs_f1.png`, `top_bottom_classes.png`, `feature_importance.png`, `weight_distribution.png`.
- Logs and artifacts: `experiment_log_with_fs.csv`; per-experiment `params.json`, `status.json`, `model.pkl`; optional `summary_report_with_fs` directory with sorted CSV and HTML report under the experiment base dir.

## 8. Thesis-ready interpretation
This group tests whether PCA-compressed voxel signatures from the Alex ultra-multimodal 7T dataset remain linearly separable across 102 tissue classes. Ablations on component count, regularization, and class balancing reveal how much dimensionality reduction and classic priors affect accuracy/F1.

Position it as the analytic baseline subsection before deep networks; the generated confusion matrices and feature-importance plots provide quick sanity checks on separability and modality influence after PCA.

## 9. Limitations and open questions
- Raw feature count and modality mapping are not encoded in the repo; assumes pre-extracted voxel signatures from the Alex dataset.
- Train/val/test split policy (per-voxel vs per-subject) is unspecified; external metadata needed.
- No calibration or uncertainty metrics; evaluation centers on accuracy/F1 only.
- Scripts expect external `label_index.txt` and `.npy` files at configured paths; reruns require those assets.
