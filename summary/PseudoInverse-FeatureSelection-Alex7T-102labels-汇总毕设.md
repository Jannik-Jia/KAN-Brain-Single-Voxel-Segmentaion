# PseudoInverse-FeatureSelection on Alex7T-102labels

## 1. Role in the thesis
- Direct feature-selection variant (LASSO/elastic-net) of the pseudo-inverse classifier, often skipping PCA to select sparse modality subsets and test whether sparsity improves voxel-wise accuracy and interpretability.

## 2. Code files and entry points
- `proj/src/main.py`: enables feature-selection mode via `--focus_on_fs`/`--skip_pca`, controls experiment sampling and logging.
- `proj/src/feature_selector.py`: implements LASSO and torch-based elastic-net selectors, stability analysis (Jaccard), and saving/loading of selectors.
- `proj/src/utils.py`: `create_feature_selection_without_pca_param_grid` and sampling strategies tailored to feature-selection sweeps.
- `proj/src/brain_voxel_dataloader.py`: `preprocess_data_with_feature_selection` applies scaling, optional PCA, and invokes the selector; supports caching.
- `proj/src/experiment_manager.py`: orchestrates per-experiment or global feature selection, saves selectors, visualizes selection results, and runs the pseudo-inverse classifier on selected features.
- `proj/src/pseudoinverse_model.py`, `proj/src/model_evaluator.py`, `proj/src/visualization_utils.py`, `proj/src/gpu_utils.py`: same roles as in the PCA pipeline but downstream of selected features.
- `proj/scripts/run_baseline.sh`: feature-selection baseline when `EXPERIMENT_TYPE=fs`.
- `proj/scripts/run_full_experiments.sh`: defaulted to fs mode with `SKIP_PCA=true`, launching large sweeps (e.g., 300 combos).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021 prepared as per-label voxel-signature `.npy` files with `label_index.txt` metadata in train/test/val folders.
- Input dimensionality: raw feature length unspecified (likely >300 modalities); feature selectors operate directly on original features when `skip_pca` is set, selecting between 50 and 300 features.
- Label space: 102 hard classes from the label index; assumed FreeSurfer-derived tissue/ROI labels on the CEST grid.
- Splits and sampling: same directory-based train/val/test scheme; optional `max_samples_per_label` limit (scripts use up to 2000) and class balancing to `target_samples`; subject-level partition UNKNOWN.

## 4. Preprocessing pipeline
- Optional standard or min-max scaling; class balancing as in the PCA pipeline.
- Feature selection: `FeatureSelector` supports `lasso` (L1 logistic regression, OVR) and `elastic_net_torch` (PyTorch regression with L1/L2 penalties); `selection_mode` fixed top-k or threshold fallback; `max_features` 50–300; `l1_ratio` tunes sparsity.
- Stability analysis via K-fold Jaccard similarity and selection frequency; caching of transformed datasets and selectors; `skip_pca` enforces selection on the original feature space.

## 5. Model architecture
- Same pseudo-inverse linear classifier (102 outputs) applied after feature selection; regularization options none/L2/truncated SVD as in the PCA baseline.
- Feature importance available from both the selector (pre-model) and the pseudo-inverse weights (post-selection).

## 6. Training configuration
- Closed-form model fit; feature-selection stage can be iterative (max_iter 1000/2000, tol 1e-4/1e-5) for LASSO/elastic net.
- Parameter grid from `create_feature_selection_without_pca_param_grid`: `apply_pca=False`, normalization {standard,minmax,None}, class balancing toggle, pseudo-inverse regularization {none,l2,truncated} with `alpha` in {0.0001,0.001,0.005,0.01,0.05,0.1,0.5,5.0}, feature_selection {lasso, elastic_net_torch}, `selection_mode` fixed, `max_features` {50,100,150,200,250,300}, `l1_ratio` {0.1,0.3,0.5,0.7,0.9,1.0}, `max_iter` {1000,2000}, `tol` {1e-4,1e-5}; `max_experiments` caps sampling (scripts use 300).
- Baseline fs run compares LASSO with and without additional L2 regularization; GPU optional via CuPy/cuML.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa (higher is better) on train/test/val; per-class F1 and confusion matrices saved.
- Additional feature-selection visuals: `feature_selection_visualization.png`, `feature_stability_visualization.png`, `feature_selection_impact.png`, `selection_ratio_impact.png`, `l1_ratio_impact.png`, `feature_selection_comparison.png`, plus best-experiment copies in `summary_report_with_fs`.
- Artifacts: `feature_selector.pkl`, `model.pkl`, params/status JSON, experiment log `experiment_log_with_fs.csv`, sorted CSV/HTML reports in `summary_report_with_fs`, optional `global_feature_selection/` outputs when using global mode.

## 8. Thesis-ready interpretation
This group studies whether direct sparsity on the full modality set can match or exceed PCA-based compression for voxel-wise tissue classification. The selected-feature counts, stability (Jaccard/frequency), and performance deltas illustrate how many modalities are truly needed and which selection settings are most reliable.

Use it as a feature-selection ablation section: contrast against the PCA baseline to argue for or against explicit sparsity and to highlight modality importance patterns.

## 9. Limitations and open questions
- Actual selected feature indices and performance numbers are not included in the repo; results depend on external experiment outputs.
- Subject-level splitting and leakage controls are not encoded; current setup is per-voxel sampling with optional balancing.
- Calibration/uncertainty metrics absent; focus remains on accuracy/F1.
- Assumes availability of pre-extracted voxel signatures and `label_index.txt` files at configured paths.
