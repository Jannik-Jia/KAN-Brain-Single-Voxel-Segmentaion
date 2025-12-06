# KAN-Binary on Alex7T-1vRest

## 1. Role in the thesis
- Establishes a one-vs-rest voxel-wise baseline using Kolmogorov-Arnold Networks (KAN) on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Explores whether KAN handles high-dimensional multimodal voxel signatures better than simple MLPs, with sampling strategies and threshold sweeps for calibration-like analysis.
- Serves as the entry point for per-label detectors before attempting full 102-way classification.

## 2. Code files and entry points
- `brain_voxel_kan_project/main.py`: CLI training/evaluation for a binary FastKAN model; handles PCA, sampling strategy selection, checkpointing, and dataset visualisation.
- `brain_voxel_kan_project/train.py`: Core training loop with accuracy/F1/recall/AUC-PR tracking, checkpoint naming (`epoch_*_acc_*_f1_*_aucpr_*.pth`), and evaluation helpers (ROC/PR curves).
- `brain_voxel_kan_project/datasets.py`: Loads per-label `.npy` feature blocks via `label_index.txt`, supports balanced/stratified/modified stratified sampling and optional PCA plus min-max scaling.
- `brain_voxel_kan_project/models.py`: Defines `BrainVoxelKAN` (FastKAN with layers [input_dim, hidden_dim, num_classes]).
- `brain_voxel_kan_project/evaluate_epochs.py`: Sweeps saved checkpoints across epochs, recomputes metrics on train/test/val/merged sets, exports `.pkl/.csv/.png` summaries and threshold analyses.
- `brain_voxel_kan_project/config.py`: Default hyperparameters (341-dim input, label_id=1, binary classes, PCA on, neg:pos=5, grid=10, lr=1e-3, batch=500, epochs=100).
- `brain_voxel_kan_project/run_full_experiment.sh`: Automates 200-epoch training plus full epoch evaluation (batch 640).
- Notebooks (`1DKAN_brainvoxel.ipynb`, `brain_voxel_fast_kan_step_same_data_adam_*`, `brain_voxel_fast_kan_step_same_data_adam_focalloss*`, `brain_voxel_fast_kan_step_same_data_adam_创建数据集*`): interactive iterations for data restructuring, focal-loss/early-stopping variants, and per-label dataset creation.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021, flattened into per-voxel feature vectors.
- Input dimensionality: 341 modalities/features per voxel (diffusion/QTI/CEST/MPRAGE/QSM/SMWI families); PCA can reduce to an automatically chosen number of components (about 95 percent variance) or a user-specified `N_PCA`.
- Label space: 102 anatomical classes available; each run picks a `label_id` as positive (default 1) and treats all other labels as negative (binary one-vs-rest).
- Labels are hard integer IDs read from per-label `.npy` files indexed in `label_index.txt`; no soft labels or partial-volume handling in this pipeline.
- Splits: pre-generated `restructured/{train,test,val,merged}` directories; samplers draw balanced/stratified/modified-stratified batches with configurable neg:pos ratio (default 5:1) and optional restriction to a subset of negative labels.

## 4. Preprocessing pipeline
- Optional PCA (sklearn) fitted on concatenated train/test/val samples; when `N_PCA=0`, `analyze_pca_variance` picks the minimal components explaining roughly 95 percent variance.
- Post-PCA min-max scaling across all sets; if PCA is off, raw features can be left untouched or z-scored (notebooks demonstrate both).
- Data balancing occurs at sampling time rather than via loss weighting; background voxels are excluded by construction.

## 5. Model architecture
- FastKAN with layers `[input_dim, 64, 2]` and fixed spline grid size 10 (configurable); no dropout or batch norm.
- FastKAN provides learnable activation splines; feature importance can be extracted from input spline weights (`utils.analyze_kan_model`).

## 6. Training configuration
- Loss: `nn.CrossEntropyLoss` over 2 classes.
- Optimiser: Adam (`lr=1e-3`, `weight_decay=1e-6`); batch size 500 (shell script uses 640); epochs 100-200; validation every epoch.
- Checkpoints: saved each eval epoch; `get_best_model` selects best epoch by AUC-PR/F1/accuracy; optional checkpoint reload via `CHECK_POINT`.
- Variants in notebooks test focal loss, early stopping, and alternative negative sampling ratios.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, precision, recall, F1, AUC-PR (primary), ROC-AUC; confusion matrices and threshold sweeps for different decision cut-offs.
- Outputs: `.pth` checkpoints, `.pkl/.csv` multi-epoch summaries, `.png` plots for metric trajectories and threshold analyses saved under `Results/BrainVoxel_1DKAN/BrainVoxel` (and `eval_results_label_<id>/`).
- Interpretation: higher is better for accuracy/F1/recall/AUC-PR/ROC-AUC; confusion matrices expose false-positive/false-negative trade-offs for given thresholds.

## 8. Thesis-ready interpretation
- Demonstrates a lightweight KAN baseline for per-structure voxel detection using the full 341-channel signature, probing whether spline-based activations improve discrimination over classic fully-connected nets.
- The modified-stratified sampling plus PCA pipeline shows how to control class imbalance while preserving multimodal variance; threshold sweeps enable downstream calibration plots.
- Suitable for a Methods subsection on "Voxel-wise one-vs-rest baselines" and for reporting per-label sensitivity/specificity or precision-recall curves.

## 9. Limitations and open questions
- Data paths are absolute to a Jupyter environment and not versioned; actual voxel counts and trained weights are absent from the repo.
- Performance numbers are not logged here; reliability of focal-loss and early-stopping variants is UNKNOWN because results are not retained.
- Depends on external `fastkan` library and pre-generated `restructured` datasets; reproducibility requires those assets.
