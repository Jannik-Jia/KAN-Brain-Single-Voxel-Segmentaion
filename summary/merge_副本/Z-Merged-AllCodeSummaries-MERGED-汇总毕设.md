# Merged thesis code summaries

This file is an automatic, lossless merge of 36 Markdown summary files from:

`SUMMARY_DIR = /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/summary`

All original `*-汇总毕设.md` files remain unchanged.
The sections below are grouped by source file.

## Index of source files

- 0-Overall-AllModels-AllDatasets-汇总毕设.md
- AttnResMLP-Alex7T-102labels-汇总毕设.md
- AttnResMLP-Alex7T-30labels-汇总毕设.md
- BaseMLP-Alex7T-102labels-汇总毕设.md
- BayesOpt-MLP-Alex7T-101labels-汇总毕设.md
- ClassWeights-Alex7T-TRAIN38-汇总毕设.md
- ClassicalML-Alex7T-102labels-汇总毕设.md
- ClassicalML-GPU-Alex7T-102labels-汇总毕设.md
- ClassicalML-Imbalanced-Alex7T-102labels-汇总毕设.md
- ConvPatch2D-Alex7T-102labels-汇总毕设.md
- DataProfiling-Alex7T-102labels-汇总毕设.md
- Deep4x4096-Alex7T-RegionCLS-汇总毕设.md
- DeepMLP-Advanced-Alex7T-102labels-汇总毕设.md
- DeepMLP-Alex7T-101labels-汇总毕设.md
- DeepMLP-Alex7T-102labels-汇总毕设.md
- DeepMLP6x2048-Alex7T-102labels-汇总毕设.md
- FC4x4096-Alex7T-101labels-汇总毕设.md
- FC4x4096-Alex7T-102labels-patientwiseStd-汇总毕设.md
- FC4x4096-Alex7T-102labels-汇总毕设.md
- FC4x4096-Alex7T-softlabels-汇总毕设.md
- FixedMLP-BackgroundWeighting-Alex7T-102labels-汇总毕设.md
- KAN-Alex7T-102labels-汇总毕设.md
- KAN-Binary-Alex7T-1vRest-汇总毕设.md
- KAN-Multiclass-Alex7T-102labels-汇总毕设.md
- KAN-RFFeatures-Alex7T-102labels-汇总毕设.md
- MLP-PatientSplit-Alex7T-Train38-汇总毕设.md
- MLPVariantsBayesOpt-Alex7T-102labels-汇总毕设.md
- OriginalMLP-CouplingLR-Alex7T-102labels-汇总毕设.md
- ProbSmoothing-Alex7T-102labels-汇总毕设.md
- PseudoInverse-FeatureSelection-Alex7T-102labels-汇总毕设.md
- PseudoInverse-PCA-Alex7T-102labels-汇总毕设.md
- QC-MultimodalRegistration-Alex7T-351modes-汇总毕设.md
- ResNet50Patch-Alex7T-102labels-汇总毕设.md
- ResidualMLP-Alex7T-101labels-汇总毕设.md
- ResidualMLP-Alex7T-102labels-汇总毕设.md
- SubjectEmbedding-Alex7T-EmbeddingFeasibility-汇总毕设.md

---
## Source file: 0-Overall-AllModels-AllDatasets-汇总毕设.md
Path: 0-Overall-AllModels-AllDatasets-汇总毕设.md
---

# Overall Summary

This codebase covers the full pipeline for voxel-wise classification on the Alex ultra-multimodal 7T dataset: hard-label MLP baselines, soft-label/calibration experiments, spatial patch CNNs (lightweight and ResNet), probability-map post-processing, and multimodal QC/downsampling utilities. Soft-label runs emphasise calibration and partial-volume effects, while spatial models test the benefit of neighbourhood context. QC tooling documents registration quality before training.

| ModelShort | DatasetShort | Purpose in thesis | Key scripts | Key metrics | Output files |
| --- | --- | --- | --- | --- | --- |
| FC4x4096 | Alex7T-102labels | Hard-label baseline and probability-map generator | `training/B0_1D_training/train_1d_with_3d_dataset.py`; `erosion/train_38fold.py` | Macro-F1, gross accuracy | `.pth`, predictions_3d `.mat`, history JSON/ONNX |
| FC4x4096 | Alex7T-softlabels | Soft-label & calibration experiment on downsampled data | `training/downsampling/train_runner.py`; `dataset_create/downsampling/mri_downsampling_pipeline.py` | NLL, soft-ECE, Brier, AURC, soft Dice | `best.pth`, metrics JSON, reliability/risk-coverage plots, 3D NPZ preds |
| ConvPatch2D | Alex7T-102labels | Lightweight spatial baseline (3×3/7×7 patches) | `training/3D CNN/train_baseline_3x3_7x7.py` | Macro-F1 | `best_model_patch*.pth`, history JSON |
| ResNet50Patch | Alex7T-102labels | Deep spatial model with imbalance-aware losses | `training/3D CNN/ResNet/scripts/train_mri_resnet.py` | Macro-F1, per-class P/R/F1 | `best_model.pth`, training_results.json, training_history.png |
| ProbSmoothing | Alex7T-102labels | Post-processing of FC probability maps | `training/B1_Probability_map_post_processing/smooth_postprocess_eval.py` | Accuracy, macro-F1, AUPRC, κ | Smoothed HDF5 preds, CSV metrics, plots |
| QC-MultimodalRegistration | Alex7T-351modes | Registration/QC analysis of multimodal inputs | `dataset_create/multimodal_mri_qc_analysis/multimodal_mri_qc_analysis.ipynb` | LNCC, NGF, MIND-SSD, ASSD/HD95, edge IoU | PNG figures, `modality_qc.csv`, `qc_analysis_report.json` |

Suggested thesis mapping: use FC4x4096 (hard) as the primary baseline section; FC4x4096 (soft) for soft-label/calibration analysis; ConvPatch2D vs ResNet50Patch for spatial context ablations; ProbSmoothing for post-processing discussion; QC-MultimodalRegistration for data quality/appendix. Note: `HSIConvKAN-main` contains reference KAN implementations but is not wired into the MRI data pipelines (usage unknown).


---
## Source file: AttnResMLP-Alex7T-102labels-汇总毕设.md
Path: AttnResMLP-Alex7T-102labels-汇总毕设.md
---

# AttnResMLP on Alex7T-102labels

## 1. Role in the thesis
- Attention-enhanced residual MLP baseline for voxel-wise classification on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Tests transformer-like upgrades (Stage1 basic residual, Stage2 adds self-attention, Stage3 adds pre-LN + FFN) against simpler FC/KAN ideas while keeping per-voxel inputs.
- Focuses on handling 102 hard labels with class imbalance mitigation and evaluates across train/val/test and merged sets.

## 2. Code files and entry points
- : end-to-end notebook defining data samplers, loaders, optional PCA, model variants (Stages 1–3), training loop, evaluation, and visualizations.
- Key components: , , , , model classes , , , , , training/eval helpers (, , plotting).
- Outputs stored under  relative to the notebook.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; voxel features under .
- Input dimensionality: default 341 channels (diffusion/QTI, CEST offsets, Amide/Amine/NOE/MT, etc.), optional PCA to  (default 0 = no PCA).
- Labels: 102 tissue/region classes (ids 1–102) plus background 0; background remapped to  and ignored by the loss; class mapping saved to .
- Splits: restructured data from  with approx 60/20/20 train/val/test; additional “merged” set loads all labels for a sanity check.
- Sampling: optional balancing (, ) with simple copy augmentation capped by ; can also use all samples ().

## 4. Preprocessing pipeline
- Optional PCA on concatenated train/test/val followed by per-component min-max normalization;  by default (raw 341-D features).
- No explicit normalization when PCA is off; assumes upstream feature scaling.
- Label handling: background -> ; class weights computed from training label counts.
- Data restructuring helper merges legacy train/val and splits 0.6/0.2/0.2 (run once upstream).
- No missing-value handling or modality-specific scaling in code.

## 5. Model architecture
- Stage1 : input linear + GELU + BatchNorm -> stacked s over  (default [256,256,256]) -> linear classifier.
- Stage2 adds multi-head  (num_heads=4, dropout=0.1) after each residual block with LayerNorm and residual fusion.
- Stage3 Transformer-style: input embedding (Linear -> LayerNorm -> SiLU -> Dropout repeated), blocks of pre-LN multi-head attention (num_heads=8, dropout=0.2) + FFN (4x expansion with SiLU) + projection residuals, final LayerNorm + classifier.
- Regularization: dropout 0.1/0.2, BatchNorm in early stages; L1/entropy regularization flags defined (, , ) but not applied in the loss.

## 6. Training configuration
- Loss:  with inverse-frequency class weights,  for background.
- Optimizer: AdamW (, ); seed 666.
- LR schedules: optional; default  milestones [15,35,50,75], gamma 0.6; cosine and plateau alternatives present.
- Batch size 256; epochs 100; validation every 3 epochs.
- Data balance via resampling/augmentation; no mixup or cutmix; no explicit gradient clipping.
- Checkpoints saved each validation step to ; config saved to .

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa; per-class precision/recall/F1; confusion matrix (log-scaled heatmaps).
- Visualizations: class distribution bars, per-class accuracy bars, confusion matrices, multi-dataset metric comparisons.
- Outputs:  plots (e.g., , , ), text reports (, ), checkpoints , , .

## 8. Thesis-ready interpretation
- Provides a strong FC/MLP-style baseline enriched with attention and residual connections for the full 102-class label set on the Alex dataset.
- Illustrates how transformer-style components affect voxel-wise classification and class-imbalance behavior, informing comparisons to KAN/TabNet/linear baselines.
- Fits a “Baseline and attention-enhanced MLPs” section or an ablation on attention depth.

## 9. Limitations and open questions
- Exact metric values are not embedded; rely on saved reports for numbers.
- Hard-coded data paths to  and reliance on pre-built  reduce portability.
- No calibration metrics (ECE/NLL) or uncertainty estimates; normalization when PCA is off is unclear.
- L1/entropy regularization flags are unused; Stage1/2/3 comparisons are not reported in the notebook.


---
## Source file: AttnResMLP-Alex7T-30labels-汇总毕设.md
Path: AttnResMLP-Alex7T-30labels-汇总毕设.md
---

# AttnResMLP on Alex7T-30labels

## 1. Role in the thesis
- Subset study to validate the attention-enhanced residual MLP on a 30-class slice of the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Reduces class space to speed iteration and test sampling/balancing strategies before full 102-class runs.

## 2. Code files and entry points
- : data subset selection, loaders, model variants (Stages 1–3), training and evaluation identical to the full-classes notebook.
- Key helpers reused:  (chooses top classes by sample count), , , , model classes, , evaluation/plotting.
- Outputs under  (relative to notebook).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; voxel features from .
- Input dimensionality: 341 channels; optional PCA (default off).
- Labels: 30-class subset selected from the 102-class pool; selection requires >=1000 train samples and at least ~100 in test/val; labels remapped to 0–29 with background ignored as .
- Splits: same restructured train/test/val (approx 60/20/20); subset selection performed after reading label indices.
- Sampling: balancing to  per class with simple copy augmentation; class mapping saved.

## 4. Preprocessing pipeline
- Same as full setup: optional PCA + min-max normalization; otherwise raw features; class-weight calculation; background ignored.
- No additional harmonization or missing-value handling.

## 5. Model architecture
- Same Stage1/2/3 attention-enhanced residual MLP options as the full experiment (hidden dims [256,256,256], attention heads 4 or 8, dropout 0.1–0.2).
- Residual blocks with GELU and BatchNorm; Stage3 uses pre-LN transformer-like blocks.

## 6. Training configuration
- Loss:  with class weights, .
- Optimizer AdamW (, ); epochs 100; batch 256; validation every 3 epochs.
- LR scheduler options identical (default MultiStep [15,35,50,75] gamma 0.6).
- Balancing and augmentation enabled; checkpointing and config export to .

## 7. Evaluation metrics and outputs
- Same metric set: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa, per-class metrics, confusion matrices.
- Visual outputs and text reports parallel the 102-class notebook, with paths adjusted to the subset save directory.

## 8. Thesis-ready interpretation
- Fast-turnaround experiment to probe whether the attention-residual MLP scales down gracefully and whether balancing/augmentation pipelines behave as expected on fewer classes.
- Useful as a methodological appendix or pilot study motivating the choice of architecture and sampling strategy before full-scale training.

## 9. Limitations and open questions
- Actual selected class identities depend on data availability and are only logged in .
- Same hard-coded data paths and lack of calibration metrics as the full experiment.
- Performance numbers are not in the notebook text; must read saved reports.


---
## Source file: BaseMLP-Alex7T-102labels-汇总毕设.md
Path: BaseMLP-Alex7T-102labels-汇总毕设.md
---


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


---
## Source file: BayesOpt-MLP-Alex7T-101labels-汇总毕设.md
Path: BayesOpt-MLP-Alex7T-101labels-汇总毕设.md
---

# BayesOpt-MLP on Alex7T-101labels

## 1. Role in the thesis
Automated hyperparameter and architecture search across base/deep/residual MLPs on the Alex ultra-multimodal 7T dataset. Provides a data-driven way to select the strongest FC-style model before comparing against alternative architectures.

## 2. Code files and entry points
- `utils/optimization.py`: Optuna objective/search space, including learning rate/weight decay/dropout/activation/optimiser/scheduler/model_type/depth/width/skip/bottleneck options; visualisation and reporting utilities.
- Integration hooks in `main.py` (`--run_bayesian_opt`, `--n_trials`) and interactive launcher `run.sh` (auto-tests all three MLP types with user-chosen standardisation/background mode).
- Uses shared loaders/trainers (`train.py`, `data/mat_loader_patientwise.py`, `data/samplers.py`, `utils/metrics.py`, `utils/model_io.py`).

## 3. Dataset and labels
Same Alex ultra-multimodal 7T setup with 341 features and 101 labels (background configurable). Patientwise standardisation is assumed for most searches; fixed prob_idx split (val 20, test 38) unless presets override. Class weights computed from training labels.

## 4. Preprocessing pipeline
Identical to other groups: patientwise/global standardisation, optional background filtering/ignoring, scaler persistence, optional PCA hook (off by default), patient-aware batching if enabled.

## 5. Model architecture
- Search spans `model_type` ∈ {base_mlp, deep_mlp, residual_mlp}.
- Hidden widths cover 1024–8192 variants and depth 4–12 with width strategies (constant/decreasing/increasing/hourglass/bell); residual bottlenecks optional; skip connections toggleable.
- Activation (`relu`/`gelu`/`swish`), dropout up to 0.8.

## 6. Training configuration
- Trials train for 5–20 epochs (bounded by `epochs`) with cross-entropy + class weights; optimiser choices Adam/AdamW/SGD/RMSprop; schedulers cosine/step/plateau/none; optional gradient clipping.
- Validation macro-F1 is the optimisation target; MedianPruner used for early stopping. Best params saved to JSON and `optimized_config.json`; subsequent full training reuses best settings.

## 7. Evaluation metrics and outputs
- Trial metrics logged via Optuna; aggregated results exported to `<study_name>_results.json` and analysis figures (`architecture_performance_boxplot.png`, `architecture_trial_counts.png`, `performance_evolution.png`, parameter importance/history plots when available) under the study directory.
- The final model still evaluated with the standard metric suite (accuracy, balanced accuracy, macro/weighted F1, kappa) and saved checkpoints/plots as in other groups.

## 8. Thesis-ready interpretation
- Captures the effect of systematic hyperparameter search on FC-style models, showing whether performance gains arise from architecture depth/width choices or optimisation settings rather than new model families.
- Can underpin a subsection on automated model selection and provide the chosen configuration for downstream comparisons.

## 9. Limitations and open questions
- Search uses shortened training epochs; the winning config should be retrained fully to confirm gains.
- Optuna adds computational overhead; results depend on the fixed val patient (20) and may overfit that subject.
- Requires Optuna dependency and may need GPU resources for large widths; not all trials save intermediate checkpoints.


---
## Source file: ClassWeights-Alex7T-TRAIN38-汇总毕设.md
Path: ClassWeights-Alex7T-TRAIN38-汇总毕设.md
---

# ClassWeights on Alex7T-TRAIN38

## 1. Role in the thesis
Class-imbalance analysis and class-weight generation for voxel-wise classifiers on the Alex ultra-multimodal 7T dataset from German et al. 2021 (TRAIN38.mat subset). Derives balanced, inverse-frequency, sqrt/log, Class-Balanced Effective Number, and prob_idx-aware weights to stabilize cross-entropy or focal losses for rare tissues. Functions as a preprocessing/calibration utility feeding downstream FC/MLP/KAN or TabNet models rather than a standalone classifier.

## 2. Code files and entry points
- data_analysis/train38_weighting.ipynb: end-to-end pipeline to load TRAIN38.mat, z-score features, analyze imbalance, compute multiple weight strategies, generate reports/plots, and export PyTorch-ready weights via analyze_mat_file.
- data_analysis/Effective Number权重计算.ipynb: lightweight Effective Number helper (EffectiveNumberWeights, compute_class_weights_from_labels, create_pytorch_weighted_loss) demonstrating standalone weight computation and focal-loss wrapper.

## 3. Dataset and labels
- Dataset: TRAIN38.mat (path in code: /home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat), assumed drawn from the Alex ultra-multimodal 7T dataset from German et al. 2021; contains data (features), region (labels), and prob_idx (grouping index).
- Input dimensionality: features are transposed to [n_samples, n_features] with expected ~341 channels (code prints “341个特征维度” after normalization); exact sample count depends on the MAT file (not stored in repo).
- Label space: hard class IDs from region; number of classes inferred from label range; missing classes tracked; no soft labels.
- Splits: analysis runs on the full MAT file; train/val/test splitting is not encoded—user must align weights with the intended split to avoid leakage.

## 4. Preprocessing pipeline
- Loads HDF5/MAT keys data, region, prob_idx, transposes to row-major samples.
- Applies per-feature z-score normalization (keeps a copy of the original for comparison); warns on zero-variance features and preserves their raw values.
- Quality checks for NaN/Inf, zero rows, data range, and prob_idx coverage; optional external weight config loaded from YAML/JSON/NPY/NPZ/CSV.
- No dimensionality reduction; feature correlation (>0.9) and outlier counts are reported for awareness.

## 5. Model architecture
- Not a network; provides weight strategies to plug into downstream classifiers.
- Strategies: sklearn balanced, inverse frequency, sqrt-balanced, log-balanced, Class-Balanced Effective Number (Cui et al. 2019) with auto-β selection plus β variants, prob_idx-aware averaging, and external weights merging. Mean-normalization applied to custom strategies; Effective Number kept as raw 1/E_n.
- Sample loss wrappers: PyTorch CrossEntropyLoss or a small FocalLoss (γ=2.0) with injected weights.

## 6. Training configuration
- No optimizer/epoch loop here; focuses on preparing CLASS_WEIGHTS_TENSOR for later training.
- analyze_mat_file orchestrates load → stats → weight computation → report/plots → exports (recommended_class_weights.py, optional weights_<strategy>.json/npy/npz/csv).
- Effective Number β auto-tuned from class counts (heuristics differ for fine/coarse label spaces); manual β list [0.9, 0.99, 0.999, 0.9999] also computed.

## 7. Evaluation metrics and outputs
- Dataset stats: total samples, features, dtype, memory, label format/range, prob_idx counts; better balance corresponds to imbalance ratio near 1.
- Class distribution: min/mean/median/max samples per class, percentiles, missing-class list, imbalance level (light → extreme); prob_idx-wise class coverage.
- Feature stats: zero/low-variance counts, high-correlation pairs (>0.9), outliers per feature, normalization effect diagnostics.
- Weight diagnostics: range/mean per strategy, recommended Effective Number β.
- Outputs: text report comprehensive_analysis_report.txt, visualizations (class_distribution.png, weight_strategies.png, feature_analysis.png, prob_idx_analysis.png, optional normalization_comparison.png), exported weights (recommended_class_weights.py, weights_<strategy>.json/.npy/.npz/.csv).

## 8. Thesis-ready interpretation
These notebooks document the imbalance characteristics of the Alex ultra-multimodal 7T voxel dataset and provide principled class weights to stabilize voxel-wise classifiers. They support sections on handling class imbalance and loss calibration, enabling comparisons between unweighted baselines and Effective Number/prob_idx-aware weighting when training FC/MLP/KAN/TabNet models. The generated reports and plots supply evidence for how severe imbalance is and why specific β values are chosen.

## 9. Limitations and open questions
- MAT data not bundled; exact sample counts, class counts, and prob_idx semantics remain unknown.
- Weights are computed on the full dataset; users must recompute on the training split to avoid leakage.
- Data path is hard-coded; adjust for local storage and verify that the 341-feature assumption matches current preprocessing.
- No downstream training or calibration metrics are included here; impact on accuracy/calibration needs separate experiments.


---
## Source file: ClassicalML-Alex7T-102labels-汇总毕设.md
Path: ClassicalML-Alex7T-102labels-汇总毕设.md
---

# ClassicalML on Alex7T-102labels

## 1. Role in the thesis
Classical feature-group analysis and hierarchical clustering/classification on the Alex ultra-multimodal 7T dataset (102 labels, 341 features). Serves as a baseline to gauge class separability, test big-class mappings, and identify promising feature subsets before heavier KAN models. Provides recommendations on feature combinations and classifiers for coarse tissue grouping.

## 2. Code files and entry points
- main.py: end-to-end pipeline on the validation set; loads data, preprocesses feature groups, runs separability analysis, clustering, classification, and reports consistency with predefined big classes.
- config.py: global hyperparameters (feature indices, PCA toggles, sampling flags, training LR/EPOCH placeholders), data paths, save directory construction.
- data_loader.py: loads train/test/val *.npy voxel files, builds label arrays, optional PCA+min-max scaling, defines 7-category big-class mapping.
- feature_analysis.py: SelectKBest (F-stat) feature selection, PCA/UMAP/TSNE/MDS visualisations, clustering search (kmeans/spectral/agglomerative) with silhouette/Calinski-Harabasz/Davies-Bouldin, cluster-vs-label heatmaps, consistency scoring vs big classes.
- classification.py: robust/standard scaling, per-group feature selection, feature-combination search, cross-validated classifier comparison (KNN/SVM/RF/MLP), classifier comparison plots.
- utils.py: logger setup, PCA explained-variance estimation; run_analysis.sh to launch nohup jobs.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; voxel-wise features restructured into train/test/val folders.
- Input: 341 modalities per voxel (0–14 diffusion, 15–224 QTI/b-tensor, 225–340 CEST). Volume of voxels not specified; validation split used in main.py.
- Labels: 102 anatomical classes (FreeSurfer-like) with hard integer labels; mapping to 7 big classes (ventricular, white matter, cortical GM, deep nuclei, limbic, brain stem, other).
- Splits: train/test/val directories; main pipeline analyses val only; class counts logged per split in data_loader.

## 4. Preprocessing pipeline
- Optional PCA (config APPLY_PCA True but main.py loads raw features with apply_pca=False) and optional min–max normalisation after PCA.
- Enhanced preprocessing: per-group robust scaling (RobustScaler), SelectKBest F-stat feature selection (10 diffusion, 30 QTI, 20 CEST) with saved importance plots.
- No explicit handling for missing values; assumes valid voxel rows. Sampling flags exist but main.py disables sampling.

## 5. Model architecture
- Classical baselines: KNN (k=5), RBF SVM (C=1, probability=True), RandomForest (50–100 trees), small MLP (50 hidden units) used for separability and classifier comparison.
- Clustering: kmeans, spectral, agglomerative across 2–7 clusters per feature group; best selected by silhouette.
- No deep KAN here; architecture exploration is purely feature+classical model based.

## 6. Training configuration
- StratifiedShuffleSplit (3 folds, 70/30) or 5-fold cross-validation depending on function; metric = accuracy.
- No epochs/optimiser; classical sklearn training. Learning rate/weight decay in config unused in main run.
- Best feature combination chosen by cross-val accuracy; classifier ranking reported with mean±std.

## 7. Evaluation metrics and outputs
- Classification: cross-val accuracy (higher better) with std; random-guess baseline shown. Confusion matrices not saved in this version.
- Clustering: silhouette (higher), Calinski-Harabasz (higher), Davies-Bouldin (lower); cluster size distributions.
- Consistency: Hungarian-matched cluster vs big-class alignment score (higher better).
- Visualisations: feature importance bars, PCA/UMAP plots, clustering metric curves, cluster-vs-label heatmaps; logs and figures saved under Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<timestamp>/ and nohup_output/*.out when run via run_analysis.sh.

## 8. Thesis-ready interpretation
- Establishes a non-neural baseline for voxel-wise classification and big-class validation on Alex 7T data. Shows which modality groups are most separable, which feature combinations help, and whether predefined 7 big classes align with data-driven clusters.
- Provides evidence for using all features vs selected subsets and whether classical models already achieve reasonable accuracy, guiding whether KAN/TabNet complexity is justified.

## 9. Limitations and open questions
- Operates only on the validation split in main.py; no end-to-end train/val/test evaluation.
- Hard-coded data paths to /home/jovyan/... and assumes pre-shuffled npy files.
- Does not handle severe class imbalance or soft labels; no calibration metrics.
- KAN not integrated; serves mainly as exploratory analysis.


---
## Source file: ClassicalML-GPU-Alex7T-102labels-汇总毕设.md
Path: ClassicalML-GPU-Alex7T-102labels-汇总毕设.md
---

# ClassicalML-GPU on Alex7T-102labels

## 1. Role in the thesis
GPU-acceleration and step-skipping variant (V3) aimed at rerunning expensive analyses selectively while keeping the imbalance-handling toolkit. Supports faster iterations for clustering and classifier comparisons on the Alex 7T voxel data.

## 2. Code files and entry points
- V3_不均衡分类_GPU加速/main.py: variant of the V2 pipeline with caching stubs (check_step_completed), reordered group processing, and hooks to skip finished steps; imports pickle for saving intermediate results.
- V3_不均衡分类_GPU加速/config/config.py plus data/preprocessing.py, analysis/*, utils/*: mirror the V2 structure for preprocessing, clustering, classification, evaluation.
- run_brain_voxel.sh, run_experiments.sh, experiments/feature_selection_exp.py, clustering_exp.py: batch execution helpers.
- output/ directories for figures, logs, results, models.

## 3. Dataset and labels
- Same Alex ultra-multimodal 7T voxel dataset with 341-channel inputs and 102 hard labels; big-class mapping to 7 categories retained.
- Train/test/val npy shards loaded; sampling flags available but default to full data.

## 4. Preprocessing pipeline
- Robust/standard normalisation options, PCA per group (10/30/20/60), SelectKBest feature selection; optional sampling for speed.
- Ordering prefers all_features/qti/cest before diffusion; diffusion clustering can be skipped if prior results exist.

## 5. Model architecture
- Classical sklearn classifiers (KNN, SVM, RF, MLP) for big-class prediction; clustering with kmeans/spectral/agglomerative.
- No neural/KAN implementation in this branch.

## 6. Training configuration
- Cross-validation similar to V2 with accuracy/balanced accuracy/F1/kappa; clustering over 2–10 clusters.
- Designed to skip repeated heavy steps rather than adjust hyperparameters; no epoch-based training.

## 7. Evaluation metrics and outputs
- Same metrics as V2 (accuracy family; silhouette/Calinski-Harabasz/Davies-Bouldin).
- Visuals and logs saved under V3_不均衡分类_GPU加速/output/; caching placeholders present but not fully implemented.

## 8. Thesis-ready interpretation
- Demonstrates engineering effort to make the classical hierarchy analysis scalable; useful for discussing runtime considerations and reproducibility in resource-limited settings.

## 9. Limitations and open questions
- Many pipeline sections remain commented or rely on external cached artefacts; GPU acceleration is implied but not explicit (no cupy usage).
- Skipping logic and cache loading are incomplete, so end-to-end runs may require manual edits.


---
## Source file: ClassicalML-Imbalanced-Alex7T-102labels-汇总毕设.md
Path: ClassicalML-Imbalanced-Alex7T-102labels-汇总毕设.md
---

# ClassicalML-Imbalanced on Alex7T-102labels

## 1. Role in the thesis
Imbalance-aware classical analysis pipeline (V2) to evaluate feature preprocessing, clustering, and classifiers under skewed voxel counts. Provides reproducible CLI + batch scripts for sweeps, informing how resampling and feature choices affect baseline performance before neural models.

## 2. Code files and entry points
- V2_不均衡分类/main.py: CLI pipeline covering data loading, big-class mapping, preprocessing (normalisation, PCA, feature selection), clustering, classifier comparison, confusion matrices, mapping evaluation; supports subset selection and plot skipping.
- V2_不均衡分类/config/config.py: configuration for feature groups, PCA dims, feature selection method, clustering range, evaluation metrics, paths, random seeds, device.
- data/data_loader.py & data/preprocessing.py: load train/test/val npy files, report class counts, apply scaling/PCA/feature selection per group, optional sampling flags.
- analysis/* (feature_analysis.py, dimensionality.py, clustering.py, classification.py): feature importance, separability, visualisation (PCA/UMAP), clustering search, classifier evaluation.
- utils/logging_utils.py, utils/evaluation.py, utils/model_utils.py: logging, metric computation (accuracy, balanced accuracy, f1_weighted, kappa; clustering metrics), saving results, mapping generation/visualisation.
- run_brain_voxel.sh, run_experiments.sh, experiments/*.py: shell and batch experiment runners (feature selection, clustering, combined experiments).
- output/figures, output/logs, output/results placeholders for generated artefacts.

## 3. Dataset and labels
- Alex ultra-multimodal 7T dataset; voxel features (341 dims split into diffusion/QTI/CEST) with 102 hard labels.
- Big-class mapping to 7 anatomical categories identical to the main branch; integer labels.
- Data splits through train/test/val directories; optional analysis on individual subset or all merged; class distributions logged.
- Sampling controls: USE_SAMPLING flag (default False) and SAMPLE_RATIO=0.1 for quick tests; imblearn SMOTE/RandomUnderSampler imported for imbalance mitigation.

## 4. Preprocessing pipeline
- Robust/standard/none normalisation options; PCA per group (10/30/20/60 components) configurable; SelectKBest feature selection with f_classif, mutual_info, or chi2.
- Feature grouping retained to compare modality contributions; optional down-sampling for speed.
- No explicit missing-data handling beyond numpy loading; assumes valid voxel rows.

## 5. Model architecture
- Classical classifiers in sklearn: KNN, RBF SVM, RandomForest, MLP; evaluated on big-class labels.
- Clustering: kmeans, spectral, agglomerative over 2–10 clusters with silhouette/Calinski-Harabasz/Davies-Bouldin tracking.
- Mapping evaluation to align clusters with predefined big classes.

## 6. Training configuration
- Cross-validation (5-fold or stratified splits) with metrics accuracy, balanced accuracy, weighted F1, Cohen’s kappa.
- No epoch-based training; hyperparameters largely defaults except tree counts or kernel choices; ability to skip expensive plots for batch runs.

## 7. Evaluation metrics and outputs
- Classification metrics above (higher is better for accuracies/F1/kappa).
- Clustering metrics: silhouette and Calinski-Harabasz (higher better), Davies-Bouldin (lower better); stability checks in utils/evaluation.
- Visuals: feature importance, dimensionality reduction scatters, cluster-vs-label heatmaps, confusion matrices; outputs stored under V2_不均衡分类/output/ subfolders with timestamps from RUN_ID in config.
- Logs saved to output/logs via logging_utils.

## 8. Thesis-ready interpretation
- Quantifies how classical models behave under class imbalance and different preprocessing choices, yielding baselines for big-class granularity and feature usefulness.
- Supports writing sections on imbalance handling and feature-selection ablations before presenting KAN or other neural models.

## 9. Limitations and open questions
- Data paths hard-coded to /home/jovyan/...; assumes precomputed npy shards.
- Imbalance mitigation (SMOTE/undersampling) is wired but not clearly exercised in the default CLI flow; effect sizes unknown.
- Still focuses on big-class labels rather than full 102-way classification; no calibration metrics.


---
## Source file: ConvPatch2D-Alex7T-102labels-汇总毕设.md
Path: ConvPatch2D-Alex7T-102labels-汇总毕设.md
---

# ConvPatch2D on Alex7T-102labels

## 1. Role in the thesis
Patch-based CNN baseline that injects minimal spatial context (3×3 or 7×7 in the xy-plane) while remaining lightweight. Serves as a bridge between the 1D MLP and deeper ResNet experiments.

## 2. Code files and entry points
- `training/3D CNN/train_baseline_3x3_7x7.py`: main training script with model definitions (ImprovedConv2D_Baseline and parameter-matched variants).
- `training/3D CNN/README.md`: usage guide and hyperparameter descriptions for 3×3/7×7 patches.
- `training/3D CNN/run_leave_one_out.sh`: batch leave-one-out runner; `test_data_loading.py` for data checks.
- `training/3D CNN/analyze_results.py`: aggregates leave-one-out results and plots statistics.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T 3D MAT files (384×336×256×351) with `region_labels` and `region_mask`.
- Inputs: 2D patches (3×3 or 7×7) cut in the xy-plane at fixed z; 351 channels per voxel, sampled from in-mask voxels (default 10k per train subject, 2× for test).
- Labels: 102 hard classes (labels shifted to 0–101 inside the loader); background excluded.
- Splits: leave-one-out by subject (37 train / 1 test) selected via CLI.

## 4. Preprocessing pipeline
- Per-channel z-score normalisation per subject over the full 3D volume.
- Patch extraction with padding at borders to maintain fixed patch size; caches subject data in memory when enabled.
- Valid voxel sampling restricted to mask & label > 0; optional shuffle for training.

## 5. Model architecture
- Stem: 1×1 conv mixes channels (351→mid, default mid=128) + GroupNorm + SiLU.
- Aggregation: single conv with kernel size equal to patch size (3 or 7) to collapse spatial dims + GroupNorm + SiLU.
- Optional refine 1×1 residual block; optional SE channel attention.
- Head: linear classifier or MLP head (parameter-matched variants targeting ~35M/52M/69M params with large hidden dims and dropout).

## 6. Training configuration
- Loss: Cross-Entropy; optimizer AdamW (lr=1e-4, weight_decay=1e-4); CosineAnnealingLR (T_max=50, eta_min=1e-6).
- Batch size 256; epochs=50; mixed precision via GradScaler; no explicit augmentation beyond patch sampling.
- Best checkpoint chosen by test macro-F1 each epoch; history saved to JSON.

## 7. Evaluation metrics and outputs
- Metrics: macro-F1 and average loss for train/test per epoch; parameter counts printed.
- Outputs: `best_model_patch{3|7}_test{N}.pth`, `history_patch{...}.json`, logs from leave-one-out runs.

## 8. Thesis-ready interpretation
Evaluates how small spatial context (3×3 vs 7×7) and parameter scaling affect voxel-wise accuracy relative to the 1D MLP. Provides an ablation stepping stone toward deeper ResNet architectures.

## 9. Limitations and open questions
- Operates on 2D slices only (no full 3D convolutions); limited augmentation.
- Samples a fixed number of voxels per subject, so full-data training impact is unknown.
- No explicit handling of class imbalance or calibration.


---
## Source file: DataProfiling-Alex7T-102labels-汇总毕设.md
Path: DataProfiling-Alex7T-102labels-汇总毕设.md
---

# DataProfiling on Alex7T-102labels

## 1. Role in the thesis
MRI data profiling toolkit (V5) to quantify feature distributions, correlations, separability, and feature importance before model training. Informs modality usefulness and guides downstream architecture choices.

## 2. Code files and entry points
- V5_data_analysis/main.py: CLI entry performing preprocessing, basic stats, feature analysis, and dimensionality reduction; supports GPU flag.
- V5_data_analysis/data_loader.py: loads brain voxel test data (npy) with one-hot or integer labels, reports normalization status, defines feature groups; attempts GPU libs (cupy/cuml) if available.
- basic_analysis.py, feature_analysis.py, dim_reduction.py: compute summary stats, correlations, class separability, feature importance (RF, mutual information), evaluate feature-selection methods, and run PCA/TSNE/UMAP/Isomap reductions.
- run_analysis.sh: helper to execute the pipeline; output directories created under analysis_results.

## 3. Dataset and labels
- Alex ultra-multimodal 7T dataset, primarily the test split loaded from /home/jovyan/.../restructured/test.
- 341 features grouped into diffusion/QTI/CEST/all_features; 102 labels (integer and one-hot forms available).
- Focuses on test subset for profiling; class counts reported from label index files.

## 4. Preprocessing pipeline
- Optional robust/standard normalization in preprocess_data; checks whether data are already standardized.
- GPU acceleration attempted via cupy/cuml; falls back to numpy/sklearn if unavailable.
- No PCA at load time; dimensionality reduction handled in analysis stage.

## 5. Model architecture
- Not a training pipeline; uses random forest feature importance and mutual information for scoring features. No neural or classical classifier training beyond separability scoring.

## 6. Training configuration
- N/A; analyses operate on the provided dataset without epoch-based training.

## 7. Evaluation metrics and outputs
- Basic stats (mean/std ranges per feature), feature correlations within/between groups, class separability ratios (F-scores), feature importance scores (RF/MI).
- Dimensionality reduction plots (PCA/TSNE/UMAP/Isomap) to visualise class spread.
- Outputs saved under analysis_results subfolders (basic_analysis, feature_analysis, dim_reduction) with text summaries and figures.

## 8. Thesis-ready interpretation
- Provides descriptive evidence on which modality groups carry discriminative signal and how correlated channels are, supporting methodological choices (normalisation, feature selection) in later classifiers.

## 9. Limitations and open questions
- Operates mainly on the test subset; conclusions may not generalise without train/val checks.
- Hard-coded data paths; assumes presence of label index files and npy shards.
- No direct link to downstream model performance; serves as exploratory analysis only.


---
## Source file: Deep4x4096-Alex7T-RegionCLS-汇总毕设.md
Path: Deep4x4096-Alex7T-RegionCLS-汇总毕设.md
---

# Deep4x4096 on Alex7T-RegionCLS

## 1. Role in the thesis
- Serves as the main voxel-wise baseline classifier and subject-embedding need probe on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Measures whether a simple dense net can separate ≈101–102 brain regions across subjects and how far it generalises (random split vs LOSO) before moving to more advanced embedding strategies.

## 2. Code files and entry points
- embedding_project_4_phase/orchestrator.py: runs Phase 0–4 end-to-end or per-phase with logging.
- embedding_project_4_phase/phase0_data_preparation/main.py plus data_splitter.py and data_validator.py: load TRAIN38_no_label43.mat, multi-subject-out split, scaling, label mapping, feature/region QC, save npz/json stats.
- embedding_project_4_phase/phase1_subject_analysis/main.py with global_analyzer.py, region_analyzer.py, multilabel_patient_analyzer.py, conditional_umap_visualizer.py: analyse subject variance, region specificity, UMAP clustering.
- embedding_project_4_phase/phase2_separability/main.py plus baseline_tester.py, loso_evaluator.py, embedding_assessor.py: train logistic regression and Deep4x4096 baselines, LOSO evaluation, region-level embedding need scoring; optional model saving.
- embedding_project_4_phase/common/config.py, common/deep_network.py, common/data_io.py, common/visualization.py: hyperparameters, 4×4096 model definition, IO helpers, plotting; run.sh to launch with checkpoint/log handling.
- Outputs land under embedding_project_4_phase/data_exchange/phase{0..3}_output and subfolders (visualizations, trained_models, *results.json/npz).

## 3. Dataset and labels
- Alex ultra-multimodal 7T dataset from German et al. 2021; default input file TRAIN38_no_label43.mat.
- Input features: 341 channels (15 QTI params, 210 raw b-tensor samples, 4 CEST params, 112 Z-spectrum points).
- Labels: one-hot brain region IDs (≈101–102 classes; label 43 removed), derived from FreeSurfer-style parcellations; hard labels.
- Split: subject-wise Multi-Subject-Out (train IDs 1–30, val 31–37, test 38); LOSO uses all train+val subjects.
- Sampling: per-voxel training; region-wise analysis aggregates per subject/region mean when ≥50 voxels.

## 4. Preprocessing pipeline
- StandardScaler fitted on train voxels, applied to val/test; scaler saved for reuse.
- Data integrity checks: NaN/Inf detection, constant-feature detection, region/sample coverage reports, class balance summaries; feature groups tracked.
- Region-aware dataset builder averages voxels per (subject, region) pair when ≥50 samples.
- Outputs metadata (data_statistics.json, label_mapping.json, feature_group_analysis.json) for downstream phases.

## 5. Model architecture
- Deep4x4096: fully connected stack 341 → 4096×4 → n_classes, ReLU + dropout 0.5, final linear logits; mirrors prior alex TensorFlow model.
- LogisticRegression baseline: C=0.1, max_iter=1000; optional region-level model.
- Region-wise deep variant optionally trained with smaller batch (64) and 15 epochs.
- Hyperparameters centralised in embedding_project_4_phase/common/config.py (ALEX_HYPERPARAMS).

## 6. Training configuration
- Loss: cross-entropy with explicit L2/kernel regularisation on weights (weight_decay 1e-5).
- Optimiser: Adam; lr 1e-5, batch_size 128, epochs 25 (global), dropout 0.5; torch seed fixed for reproducibility.
- Validation: accuracy/F1 tracked each epoch; region-wise runs use 80/20 split; LOSO evaluates held-out subjects; option to save trained models.
- No data augmentation; GPU/CPU auto-selected.

## 7. Evaluation metrics and outputs
- Classification: accuracy, macro/weighted F1 on validation; region-wise accuracy/F1; LOSO mean accuracy and generalisation gap (baseline – LOSO).
- Embedding need probes: region embedding necessity scores, counts of critical/high-priority regions, deep network authority score (performance, convergence, generalisation).
- Subject-variance metrics from Phase 1: PCA explained variance, distance/correlation matrices, silhouette scores, patient/region variance ratios, UMAP clustering scores.
- Visualisations: performance comparison bars, generalisation gap plot, embedding necessity heatmap, subject similarity heatmaps, PCA/UMAP scatter plots; saved as .png in phase output dirs.
- Logs/metadata: baseline_results.json, loso_results.json, deep_network_analysis.json, region_classification_performance.json, region_embedding_needs.json, phase*_scores.json, validation_report.json, npz feature summaries.

## 8. Thesis-ready interpretation
- Tests whether a classic dense 4×4096 network and simple baselines can classify brain regions voxel-wise on the Alex ultra-multimodal 7T dataset and how much performance drops when holding out subjects.
- Provides quantitative evidence for (i) baseline accuracy ceilings, (ii) cross-subject generalisation gaps, and (iii) which brain regions most demand subject-specific embeddings.
- Results can support sections on baseline performance, generalisation limits, and motivation for subject embedding/advanced architectures, with figures for modality separability and embedding-need heatmaps.

## 9. Limitations and open questions
- Numeric results are not stored in code; must inspect saved JSON/NPZ from actual runs.
- Assumes TRAIN38_no_label43.mat layout and hard-coded subject IDs; other datasets or label schemes would need config changes.
- GPU availability optional; long runs may be costly without acceleration.
- TabNet/KAN etc. are not integrated here despite folder names; only logistic and Deep4x4096 are operational.


---
## Source file: DeepMLP-Advanced-Alex7T-102labels-汇总毕设.md
Path: DeepMLP-Advanced-Alex7T-102labels-汇总毕设.md
---

# DeepMLP-Advanced on Alex7T-102labels

## 1. Role in the thesis
- Exploratory variant of the deep MLP adding stronger regularization (shake-shake, stochastic depth), mixed activations, and trilinear feature pooling for the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Intended as an ablation to test whether richer feature interactions and regularization improve 102-class voxel classification.

## 2. Code files and entry points
- : defines advanced modules and runs training/evaluation; also contains leftover KAN evaluation hooks.
- Key components:  with optional trilinear pooling and predefined feature groups (anatomical/functional/diffusion),  with positional embeddings,  supporting shake-shake and stochastic depth, , ,  class.
- Training/eval helpers similar to the base deep MLP (, , plotting, ); outputs under .

## 3. Dataset and labels
- Same Alex ultra-multimodal 7T dataset and 341-channel voxel features from the restructured train/test/val folders.
- Labels: 102 classes plus background ignored as ; uses class weights; hard labels only.
- Splits: train/test/val (~60/20/20) with optional balancing to 10k per class.

## 4. Preprocessing pipeline
- Optional PCA + min-max normalization (off by default); balancing via resampling/augmentation; background removal; config export to .
- No additional harmonization beyond semantic feature grouping inside the model.

## 5. Model architecture
- Hidden dims ; attention layers at indices [1,3,5] with 16 heads; dropout rates  (note potential length mismatch vs hidden dims).
- FeatureInteractionLayer supports trilinear pooling and feature-group-specific projections for anatomical (0–120), functional (120–240), diffusion (240–341) channels.
- Residual blocks can use shake-shake regularization (alpha/beta random mixing), stochastic depth (rate 0.2), mixed GELU/SiLU activations; self-attention includes learned positional embedding.
- Classifier is a linear head; weight init uses Kaiming variants.

## 6. Training configuration
- Loss:  with class weights and ; background ignored.
- Optimizer AdamW (, ); batch size 512 with ; epochs 150.
- Scheduler: OneCycleLR (, ); mixed precision; gradient clipping 10; validation every 3 epochs.
- Checkpointing via  and per-epoch saves; includes plotting of training curves and confusion matrices.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa; per-class metrics; confusion matrices; class F1 distribution plots.
- Outputs:  for training curves and confusion matrices, text test reports (), checkpoints , .

## 8. Thesis-ready interpretation
- Tests aggressive regularization and feature pooling ideas to see if deeper MLPs can better exploit multimodal structure than the base model.
- Suitable for an ablation subsection comparing feature-interaction/regularization strategies within dense architectures.

## 9. Limitations and open questions
- Dropout list length vs hidden layers may need alignment; code mixes DeepMLP and KAN training snippets, indicating work-in-progress.
- Hard-coded data paths and missing calibration metrics remain.
- Actual performance numbers not present in the notebook body.


---
## Source file: DeepMLP-Alex7T-101labels-汇总毕设.md
Path: DeepMLP-Alex7T-101labels-汇总毕设.md
---

# DeepMLP on Alex7T-101labels

## 1. Role in the thesis
Deeper fully connected variant testing whether extra depth/width and optional skip connections improve voxel-wise classification beyond the 4×4096 baseline on the Alex ultra-multimodal 7T dataset. Serves as an ablation on network capacity and gradient flow.

## 2. Code files and entry points
- `models/deep_mlp.py`: deep MLP with configurable depth/width and optional skip connections between non-adjacent layers.
- Shared training stack: `main.py`, `config.py`, `train.py`, `data/mat_loader_patientwise.py`, `data/mat_loader.py`, `data/samplers.py`, `utils/metrics.py`, `utils/label_processing.py`, `utils/model_io.py`, `utils/visualization.py`.
- Hyperparameter search support via `utils/optimization.py` (depth/width strategies, activations, schedulers) and `run.sh` for interactive runs.
- Evaluation/inference identical to baseline via `evaluate.py`, `predict.py`, notebooks for tuning (`alex版本优化*.ipynb`).

## 3. Dataset and labels
Same as baseline: 341-feature voxels from the Alex ultra-multimodal 7T dataset; 101-class label space (background optional). Fixed prob_idx split (val 20, test 38) with presets for alternate test IDs or random split. Class weights derived from training labels; optional patient-aware batching.

## 4. Preprocessing pipeline
Identical to baseline: patientwise or global standardisation, optional PCA (off), configurable background handling, scaler persistence, split integrity checks.

## 5. Model architecture
- Depth and width configurable: hidden_dims list can extend beyond four layers; BayesOpt explores depths 4–12 with width strategies (constant, decreasing, increasing, hourglass, bell).
- Optional `use_skip_connections` to add additive skips between non-adjacent layers to aid gradient flow.
- Activation choices (`relu`/`gelu`/`swish`), dropout configurable; linear output sized to `num_class`.

## 6. Training configuration
- Same loss/optimiser/scheduler setup as baseline; defaults lr=1e-5, weight_decay=1e-5, batch size 128, epochs 30, cosine LR schedule.
- When `--run_bayesian_opt` is enabled, Optuna trials (5–20 epochs) search learning rate, dropout, activation, optimiser (Adam/AdamW/SGD/RMSprop), scheduler type, depth/width, and skip usage; best params saved to `optimized_config.json`.
- Checkpointing/logging identical to baseline.

## 7. Evaluation metrics and outputs
- Metrics and artefacts match baseline (accuracy, balanced accuracy, macro/weighted F1, kappa, per-class stats, confusion matrices, training curves, predictions npz/mat files).
- Best model still selected via validation macro-F1; outputs stored under `results/<experiment_name>/`.

## 8. Thesis-ready interpretation
- Explores whether deeper MLPs with skip connections better capture nonlinear interactions among the 341 modalities than the shallow baseline.
- Intended for an ablation subsection on network depth/capacity; helps justify whether further complexity (e.g., TabNet/KAN) is needed after MLP scaling.

## 9. Limitations and open questions
- Concrete best-performing depth/width not hard-coded; must inspect saved configs/logs to know selected architecture.
- Optuna trials use shortened training, so full-epoch performance may differ; computationally heavy on large search spaces.
- Same subject-level split limitations and lack of calibration/augmentation as the baseline.


---
## Source file: DeepMLP-Alex7T-102labels-汇总毕设.md
Path: DeepMLP-Alex7T-102labels-汇总毕设.md
---


# DeepMLP on Alex7T-102labels

## 1. Role in the thesis
- Deeper MLP variant to test whether increased depth/width and optional skip connections improve voxel-wise classification beyond the 4×4096 baseline on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Primarily explored through the shared Optuna search; not the default config but available for ablations on capacity versus overfitting.

## 2. Code files and entry points
- `models/deep_mlp.py`: Builds variable-depth MLPs (width 1024–3072, depth 5–8) with optional skip adapters between non-adjacent layers.
- Shared pipeline files: `main.py`, `train.py`, `models/config.py`, `data/mat_loader.py`, `utils/optimization.py`, `utils/metrics.py`, `utils/model_io.py`, `utils/visualization.py` (as in the FC baseline).
- Selection: activate via `--model_type deep_mlp` in `main.py` or allow Optuna (`model_type` search space) to choose it.

## 3. Dataset and labels
- Same TRAIN38/DEMO38 Alex ultra-multimodal 7T dataset; feature_dim=341, num_class=102 with background=0 ignored.
- Patient-wise splits identical to the baseline unless overridden; class weights applied from training labels.

## 4. Preprocessing pipeline
- Identical StandardScaler normalization, scaler persistence, and optional normalization_params storage as in the FC baseline; PCA flags unused.

## 5. Model architecture
- Input layer → stack of linear layers with chosen activation (ReLU/GELU/Swish) and dropout; widths are constant per config but configurable via hidden_dims list.
- Optional skip connections: adapters align dimensions from earlier to later layers before addition in forward pass.
- Final linear maps to 102 classes; no batch norm.

## 6. Training configuration
- Same loss/optimizer/scheduler setup as the FC baseline; epochs/validation cadence identical unless overridden.
- Optuna search covers depth (5–8), width_factor in {1024, 2048, 3072}, use_skip_connections flag, plus shared hyper-parameters (lr, weight_decay, dropout, activation, optimizer, scheduler, batch size).

## 7. Evaluation metrics and outputs
- Uses the same evaluation stack (accuracy, balanced acc, macro/weighted F1, kappa, per-class metrics, confusion plots) and the same artifact naming/location under `results/<experiment>`.

## 8. Thesis-ready interpretation
- Provides an ablation on depth/skip connections for dense voxel classifiers, indicating whether added capacity/skip paths materially change class-wise F1 or generalization.
- If Optuna selects DeepMLP, its performance can highlight the limits of purely dense feature mixers before considering modality-aware or attention-based models.

## 9. Limitations and open questions
- DeepMLP is not the default path; empirical results depend on Optuna selecting it, so direct dedicated runs may be needed.
- get_model_info fields (`input_dim`, `activation_name`, etc.) are not explicitly stored in the class, which may lead to incomplete metadata in saved checkpoints.
- As with the baseline, label source and voxel counts remain UNKNOWN, and hard-coded data paths in scripts must be updated.


---
## Source file: DeepMLP6x2048-Alex7T-102labels-汇总毕设.md
Path: DeepMLP6x2048-Alex7T-102labels-汇总毕设.md
---

# DeepMLP6x2048 on Alex7T-102labels

## 1. Role in the thesis
- Best-performing MLP variant derived from Bayesian search: deeper network with StandardScaler normalisation to push baseline performance on the Alex 7T voxel-signature task.
- Intended as the strong FC reference model for comparisons with more exotic architectures and for downstream inference (e.g., demo38 deployment).

## 2. Code files and entry points
- `train_best_model.py`: end-to-end script that fits a StandardScaler, trains the deep MLP with fixed best hyperparameters, saves scaler + checkpoints + reports.
- `data/dataset.py`: loads restructured train/test/val voxel features; used here with `norm=False` so that external StandardScaler can be applied consistently.
- `utils/metrics.py`, `utils/visualization.py`: evaluation and plotting for the final model.
- `utils/model_io.py`: checkpoint saving with architecture metadata, JSON sidecars, and safe loading.
- `predict.py` / `predict.sh`: inference pipeline consuming the saved model and scaler for external `.mat` volumes.

## 3. Dataset and labels
- Same Alex ultra-multimodal 7T dataset (341-channel voxel signatures, 102 classes, background ignored) with fixed train/val/test folders.
- Label space and hard labels identical to the FC baseline; per-voxel sampling with no additional balancing beyond class weights in the loss.

## 4. Preprocessing pipeline
- StandardScaler fitted on training samples only, persisted as `<experiment_name>_scaler.pkl`; the same scaler is applied to val/test and saved for inference.
- Normalisation statistics (mean/std ranges) written to a text report; no PCA or dimensionality reduction.

## 5. Model architecture
- `DeepMLP` (`models/deep_mlp.py`) with 6 hidden layers of 2048 units, GELU activation, dropout ≈0.2567 after each layer, optional skip connections enabled.
- Linear head to 102 logits; no batch norm; implemented for dense tabular-style voxel features.

## 6. Training configuration
- Loss: weighted cross-entropy with background ignore index.
- Optimiser: Adam, lr ≈1.697e-4, weight_decay ≈1.33e-5; batch size 128; 30 epochs; validation every epoch.
- LR schedule: StepLR with step_size=2, gamma≈0.2216.
- Class weights computed from training labels; checkpoints saved each validation with architecture metadata and normalisation params recorded.

## 7. Evaluation metrics and outputs
- Metrics identical to the baseline: accuracy, balanced accuracy, macro/weighted F1, kappa, per-class precision/recall/F1, confusion matrices.
- Outputs stored under `./best_model_results/<experiment_name>` plus logs in `./best_model_logs`: scaler `.pkl`, normalisation stats text, training CSV/logs, per-split metrics/plots, final `test_final` evaluation files, and model `.pth` + `_architecture.json`.

## 8. Thesis-ready interpretation
- Represents the tuned FC/MLP alternative that benefits from deeper capacity and consistent z-scoring; should yield stronger macro-F1 and more stable class-wise performance.
- Suitable as the primary baseline against which KAN/TabNet/other approaches are compared; saved scaler enables reproducible inference on new subjects.

## 9. Limitations and open questions
- Hyperparameters are fixed from one search; no cross-validation or robustness analysis to different splits.
- Still relies on voxel-level splits; potential subject-level leakage is untested.
- No calibration or uncertainty assessment; only accuracy-style metrics are logged.
- Assumes availability of the saved scaler; mismatch would degrade inference.


---
## Source file: FC4x4096-Alex7T-101labels-汇总毕设.md
Path: FC4x4096-Alex7T-101labels-汇总毕设.md
---

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


---
## Source file: FC4x4096-Alex7T-102labels-patientwiseStd-汇总毕设.md
Path: FC4x4096-Alex7T-102labels-patientwiseStd-汇总毕设.md
---


# FC4x4096 on Alex7T-102labels-patientwiseStd

## 1. Role in the thesis
A PyTorch reimplementation of the 4×4096 fully connected baseline for voxel-wise classification on the Alex ultra-multimodal 7T dataset from German et al. 2021. This notebook tests patient-wise standardization (per-subject z-scoring) to better match per-patient spectral distributions compared to a global normalization. It serves as a sanity check and reproducibility step against the original TensorFlow baseline while evaluating how patient-specific scaling impacts validation on one held-out subject.

## 2. Code files and entry points
- `update_standarlize/alex的torch版本_新标准化措施.ipynb`: End-to-end notebook with four cells: (1) data loading from `.mat` files and patientwise standardization utilities, (2) definition and training loop for the 4×4096 dense network, (3) inference on `DEMO38.mat` with patientwise stats applied, (4) gradient-based saliency visualization for selected validation voxels.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021, loaded from `TRAIN38.mat` (`data`, `region`, `prob_idx`) and `DEMO38.mat` (`multidim_data`, optional `prob_idx`).
- Input dimensionality: 341-channel voxel signatures (per-voxel spectral/modal features), millions of voxels aggregated across subjects (exact count not specified). 
- Label space: 102-class one-hot labels in `region`; likely derived from FreeSurfer-style parcellations aligned to the CEST/native grid. Labels are hard one-hot vectors.
- Splits: Training uses all patients except patient 38 (`prob_idx != 38`); validation uses only patient 38 (`prob_idx == 38`). An additional `train_test_split` holds out 1% of the non-38 data, but only `X_train1`/`y_train1` are used for training (the small holdout is unused). Testing/inference cell expects `DEMO38.mat` and applies train stats where possible.

## 4. Preprocessing pipeline
- Patientwise standardization: compute mean/std per patient and z-score each voxel within its patient; zero std replaced with 1.0. Returns and stores `train_patient_stats` and `val_patient_stats`.
- Validation standardization uses validation-patient stats (not training stats); test-time standardization uses training stats when the patient id is known, otherwise falls back to per-patient stats computed on the test data. 
- No additional feature scaling, clipping, or imputation beyond the per-patient z-scoring; assumes input features are dense.

## 5. Model architecture
- Fully connected network matching the TensorFlow baseline: layers of sizes 341 → 4096 → 4096 → 4096 → 4096 → 102.
- Activation: ReLU after each hidden layer; Dropout(p=0.5) applied after each hidden activation.
- Output: linear logits (softmax applied externally during inference); no batch normalization or residual connections.

## 6. Training configuration
- Optimizer: Adam (lr=1e-5); manual L2 regularization on weight matrices only (weight_decay term 1e-5 added to loss).
- Loss: CrossEntropyLoss on argmax of one-hot labels.
- Batch size: 128; Epochs: 25; deterministic seeds set for torch and NumPy; uses GPU if available.
- Metrics tracked per epoch: training loss/accuracy/F1 (macro) and validation loss/accuracy/F1 on patient 38.
- Model checkpoint: saved to `outputs/dense_4x4096_model.pth`.

## 7. Evaluation metrics and outputs
- Metrics: macro F1 (class-balanced) and overall accuracy for train and validation; printed per epoch; best validation accuracy/F1 reported at the end. Higher is better for both accuracy and F1.
- Outputs: `outputs/train_patient_stats.npy`, `outputs/val_patient_stats.npy` (per-patient mean/std); `outputs/dense_4x4096_model.pth` (checkpoint); prediction export to `dense_4x4096_model_prediction.mat` via `scipy.io.savemat` with user-specified `predictpath`; gradient-based saliency plots for selected validation voxels (displayed, not saved by default).

## 8. Thesis-ready interpretation
This experiment checks the reproducibility of the core FC baseline while introducing patient-wise normalization to mitigate inter-subject distribution shifts in voxel signatures. It demonstrates that the 4×4096 dense architecture can be trained in PyTorch with similar regularization to the TensorFlow version and uses a single held-out subject (patient 38) as validation. The saliency visualization offers a preliminary interpretability view of which spectral channels drive predictions.

## 9. Limitations and open questions
- Validation relies on a single subject (patient 38); generalization to other subjects is untested. 
- The small extra split from `train_test_split` is unused, and no dedicated test set beyond patient 38 is evaluated.
- `predictpath` is a placeholder; prediction saving requires a user-provided path. 
- Class definitions and modality breakdown for the 341 channels are not encoded in the notebook; downstream calibration or broader metrics (ECE, confusion matrices) are absent. 
- Validation standardization uses its own stats rather than train stats, which may not reflect deployment-time behavior.


---
## Source file: FC4x4096-Alex7T-102labels-汇总毕设.md
Path: FC4x4096-Alex7T-102labels-汇总毕设.md
---

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


---
## Source file: FC4x4096-Alex7T-softlabels-汇总毕设.md
Path: FC4x4096-Alex7T-softlabels-汇总毕设.md
---

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


---
## Source file: FixedMLP-BackgroundWeighting-Alex7T-102labels-汇总毕设.md
Path: FixedMLP-BackgroundWeighting-Alex7T-102labels-汇总毕设.md
---


# FixedMLP-BackgroundWeighting on Alex7T-102labels

## 1. Role in the thesis
- Ablation to diagnose background-handling and class-imbalance strategies for voxel-wise classification on the Alex ultra-multimodal dataset.
- Compares four scenarios: with/without background training and with/without inverse-frequency class weights to resolve NaN losses and performance drops.

## 2. Code files and entry points
- comparison_alex/4场景对比/comparison_experiment.py: end-to-end experiment driver defining four scenarios, data loading, training, evaluation, and plotting.
- comparison_alex/4场景对比/run_4_senario.sh: nohup wrapper to launch the experiment and log output.
- comparison_alex/4场景对比/4场景对比alex代码数据.ipynb: exploratory notebook version of the same comparison (not executed here).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021 (TRAIN38.mat).
- Input: 341 features per voxel (transposed to samples × features).
- Labels: one-hot 102 classes including background; scenarios either keep background as a class or remap it to ignore_index (-1) while training on 101 foreground classes.
- Splits: patient-wise—prob_idx==38 held out as validation; remaining patients split with 1% as test and the rest as training.

## 4. Preprocessing pipeline
- StandardScaler fitted on training voxels; applied to val/test; scaler stats printed for QC.
- Optional background filtering in dataset class depending on scenario; when filtering, validation background can be removed to avoid all-ignore batches.
- No PCA or additional feature selection.

## 5. Model architecture
- FixedMLP: 4×4096 ReLU + Dropout(0.5) fully-connected stack with Xavier init; optional L2 regularisation (1e-5) added to loss.
- Output dimension switches between 101 (when background ignored) and 102 (when background kept).

## 6. Training configuration
- Optimiser: Adam, lr=1e-5; batch_size=128; epochs=25 (NUM_EPOCHS_DEMO).
- Loss: CrossEntropyLoss; ignore_index=-1 when background filtered. Class weights optionally applied via inverse frequency per scenario.
- Metrics tracked each epoch: validation accuracy and macro F1; training loss/acc/F1 also logged.

## 7. Evaluation metrics and outputs
- Metrics: validation accuracy and macro F1 (higher is better); NaN checks to prevent invalid losses.
- Outputs: comparison_results_fixed/ with final_results_summary_fixed.txt (per-scenario val metrics), validation_curves_comparison_fixed.png, training_loss_comparison_fixed.png, and per-scenario histories in memory.

## 8. Thesis-ready interpretation
- Tests whether treating background as a learnable class or ignoring it, and whether applying class weights, stabilises training and improves validation performance. Intended to justify background handling choices in the main pipeline.

## 9. Limitations and open questions
- Paths hard-coded to TRAIN38.mat under /home/jovyan/...; adjust before reuse.
- Only validation metrics reported; no held-out test evaluation or calibration analysis.
- Uses a single fixed architecture and learning rate; scenarios isolate background/weighting effects but not architecture-dependent behaviour.


---
## Source file: KAN-Alex7T-102labels-汇总毕设.md
Path: KAN-Alex7T-102labels-汇总毕设.md
---

# KAN on Alex7T-102labels

## 1. Role in the thesis
Notebook-based attempts to train hierarchical Kolmogorov–Arnold Networks (FastKAN) for voxel-wise tissue classification on the Alex 7T dataset. Provides the first neural baselines beyond classical models, testing expert KAN blocks per modality group and exploring feasibility/accuracy.

## 2. Code files and entry points
- 102LABEL_1DKAN_brainvoxel_分层分类.ipynb: core prototype defining ExpertKAN (FastKAN-based), hyperparameters, data loading, feature selection, and training loops.
- V2_102LABEL_1DKAN_brainvoxel_分层分类_多方法尝试.ipynb, V3_102LABEL_1DKAN_brainvoxel_分层分类_多方法尝试_加速.ipynb, V4_102LABEL_数据结构分层分类_多方法尝试_加速.ipynb: iterative variants with acceleration, additional methods, and data-structure tweaks.
- v2_102LABEL_1DKAN_brainvoxel_完全分离的数据集创建方法.ipynb: dataset restructuring/clean split preparation.
- Shared constants match config.py (MODEL_NAME, feature indices, SAVE_PATH pattern).

## 3. Dataset and labels
- Alex ultra-multimodal 7T dataset; 341-channel voxel signatures (diffusion 0–14, QTI 15–224, CEST 225–340).
- 102 hard labels mapped to 7 big classes for coarse classification; sampling ratio typically 0.1 for speed; train/test/val npy shards.
- Labels treated as integers and occasionally one-hot; no soft-label handling noted.

## 4. Preprocessing pipeline
- Optional PCA (APPLY_PCA True), normalisation flag NORM, and SelectKBest F-stat feature selection per modality group (10/30/20 features typical).
- Robust scaling in enhanced preprocessing; dataset creation notebook focuses on fully separated train/val/test splits.
- No explicit missing-value handling; assumes valid voxels.

## 5. Model architecture
- ExpertKAN class wraps FastKAN with layers_hidden=[input_dim, hidden_dim, num_classes] and grid_size=10.
- Hidden dims per group: diffusion 64, QTI 128, CEST 64; intended as modality experts that can be fused (fusion/gating strategy not fully specified in notebooks).
- Training hyperparameters: EPOCH=50, BATCH_SIZE=640, LR=1e-3, WEIGHT_DECAY=1e-6, RANDOM_SEED=666.

## 6. Training configuration
- Loss: CrossEntropyLoss; optimiser: Adam with above LR/weight decay.
- DataLoader usage inferred; sampling flag USE_SAMPLING=True in some cells to downsample to 10% for faster runs.
- No explicit scheduler/early stopping; checkpointing path via SAVE_PATH in config-like cells.

## 7. Evaluation metrics and outputs
- Classification accuracy the primary metric; confusion matrices plotted via sklearn; ROC/AUC imports present for per-class curves (higher is better).
- Feature selection/cluster analysis code reused from classical pipeline for comparison; figures/logs saved under Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<timestamp>/ when SAVE_PATH is honoured.

## 8. Thesis-ready interpretation
- Establishes feasibility of FastKAN-based voxel classifiers and compares modality-specific experts to classical baselines. Supports thesis sections on neural architectures for ultra-multimodal voxel signatures and motivates or contrasts with MLP/TabNet experiments.

## 9. Limitations and open questions
- Fusion of expert KANs into a full hierarchy is not clearly implemented; reliance on notebook execution makes reproducibility harder.
- Data paths are hard-coded; runs may not cover full train/val/test due to sampling and runtime constraints.
- Calibration metrics and soft-label handling absent; comparative results vs classical baselines need consolidation.


---
## Source file: KAN-Binary-Alex7T-1vRest-汇总毕设.md
Path: KAN-Binary-Alex7T-1vRest-汇总毕设.md
---

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


---
## Source file: KAN-Multiclass-Alex7T-102labels-汇总毕设.md
Path: KAN-Multiclass-Alex7T-102labels-汇总毕设.md
---

# KAN-Multiclass on Alex7T-102labels

## 1. Role in the thesis
- Extends the KAN approach to full 102-class voxel-wise classification on the Alex ultra-multimodal 7T dataset.
- Tests whether spline-based activations with PCA/standardisation and class balancing can scale beyond one-vs-rest detectors.
- Provides a comparative baseline for later multimodal architectures and for reporting multiclass confusion trends.

## 2. Code files and entry points
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类.ipynb`: Main multiclass FastKAN pipeline with data restructuring, optional PCA, balanced sampling, two hidden KAN layers `[256,128]`, grid size 8, and multi-step learning-rate scheduler.
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类_L1.ipynb`: Variant without PCA (raw 341 dims), hidden 128, stronger weight decay (1e-3) and tooling to monitor KAN weight magnitudes/entropies (regularisation-oriented).
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类_多重尝试/…`: Additional trial copy of the main notebook.
- Supporting cells implement dataset merging/splitting (`merge_and_shuffle_datasets`, `split_merged_dataset`), class-weight computation, and feature-importance plotting.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021.
- Input: 341-channel voxel signatures; PCA optional (default on) with automatic component count, or raw features if PCA is disabled.
- Labels: 102 anatomical classes; background set to 0 and remapped to -1 to be ignored by `CrossEntropyLoss`.
- Splits: original train/val merged then re-split 60/20/20 into `restructured/{train,test,val}`; balancing augments minority classes up to `TARGET_SAMPLES` (10000) via simple duplication when enabled.

## 4. Preprocessing pipeline
- Optional PCA followed by normalisation; if PCA is off, z-score is applied to raw features.
- Dataset restructuring ensures per-class `.npy` blocks and a `label_index.txt` for loader access; background voxels dropped.
- Class balancing via oversampling (simple copy) with a cap on augmentation multiplier; optional class weights for the loss (commented in notebooks).

## 5. Model architecture
- FastKAN with layers `[feature_dim, 256, 128, 102]` and spline grid size 8-10 (configurable); no dropout.
- Alternative simpler variant with single hidden size 128 (L1 notebook).
- Feature-importance plots derived from input spline weights; background class ignored in loss.

## 6. Training configuration
- Loss: `nn.CrossEntropyLoss` with optional class weights and `ignore_index=-1` for background.
- Optimiser: Adam (`lr=2e-5`, `weight_decay=5e-3` or `1e-3`); batch size 128; epochs 100; validation every 3 epochs.
- Learning-rate scheduler: MultiStep milestones [25,50,75] with gamma 0.5 (toggleable cosine/plateau options in code).
- Checkpoint naming mirrors the binary pipeline; best-epoch selection via max accuracy/F1 (utility reused from binary code).

## 7. Evaluation metrics and outputs
- Metrics: overall accuracy, balanced accuracy, macro/weighted F1, Cohen kappa; per-class precision/recall/F1; confusion matrices.
- Visuals: dataset distribution plots, PR/ROC curves, and feature-importance bar charts for top PCs/features.
- Outputs intended under `Results/BrainVoxel_102Class/BrainVoxel` (checkpoints/plots); actual numeric results are not stored in the repo.

## 8. Thesis-ready interpretation
- Evaluates feasibility of end-to-end 102-way voxel classification with KAN using all 341 modalities, highlighting class imbalance handling and PCA-driven compression.
- Provides evidence for how performance changes when moving from one-vs-rest to full multiclass, useful for a Methods or Experiments subsection on full parcellation with KAN.
- Feature-importance inspection can motivate modality-level discussions in the appendix.

## 9. Limitations and open questions
- Absolute data paths and external `.npy` blocks are required; reproducibility without the dataset is not possible from the repo alone.
- No logged accuracies/F1/kappa values; scheduler and balancing effects remain UNKNOWN empirically.
- Regularisation settings (L1/entropy monitors) are exploratory and not fully reported.


---
## Source file: KAN-RFFeatures-Alex7T-102labels-汇总毕设.md
Path: KAN-RFFeatures-Alex7T-102labels-汇总毕设.md
---

# KAN-RFFeatures on Alex7T-102labels

## 1. Role in the thesis
- Ablation studying whether random-forest feature selection (77 dims) plus KAN regularisation improves 102-class voxel classification on the Alex ultra-multimodal 7T dataset.
- Tests dimensionality reduction via classical feature engineering before KAN, contrasting with PCA-based or full-feature models.

## 2. Code files and entry points
- `1DKAN_随机森林特征选择_完全分离的数据集创建方法_102分类_L1.ipynb`: Loads RF-selected features from `rf_selected_features.h5`, z-score normalises them, optionally balances training data, and trains a FastKAN `[77,128,102]` model with scheduler support.
- Contains helper functions to inspect modality-group indices (diffusion/QTI/CEST), prepare balanced datasets, and monitor KAN weight/entropy statistics.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021.
- Input: 77 features chosen by a prior random-forest selector (indices grouped by diffusion/QTI/CEST); data provided in HDF5 splits (`all/train`, `all/val`, `all/test`).
- Labels: 102-class parcellation; labels are integers, no soft targets; background handling not explicitly described (assumed absent in the RF export).
- Splits: precomputed train/val/test inside the HDF5; optional balancing to `TARGET_SAMPLES` (3000) per class via down-sampling (no oversampling, `MAX_MULTIPLIER=1`).

## 4. Preprocessing pipeline
- Z-score normalisation using training-set mean/std applied to val/test; PCA disabled because feature selection already reduced dimensionality.
- Optional class balancing via down-sampling; augmentation hooks exist but default to no augmentation.

## 5. Model architecture
- FastKAN with layers `[77, 128, 102]`, spline grid size 8; no dropout or batch norm.
- Hyperparameters include L1 and entropy regularisation coefficients (`LAMBDA_L1=0.005`, `LAMBDA_ENTROPY=2.0`), but their integration into the loss is not clearly implemented (effect UNKNOWN).

## 6. Training configuration
- Loss: `nn.CrossEntropyLoss(ignore_index=-1)` (background-ignore placeholder); class weights not used by default.
- Optimiser: Adam (`lr=2e-5`, `weight_decay=1e-3`); batch size 128; epochs 100; validation every 3 epochs.
- Learning-rate scheduler options mirror the multiclass baseline (MultiStep [25,50,75], gamma 0.5; cosine/plateau alternatives available).

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen kappa; per-class precision/recall/F1 and confusion matrices for train/test/val.
- Outputs expected under `Results/BrainVoxel_102Class_RF/BrainVoxel` (checkpoints/plots), but trained weights/CSVs are not versioned in the repo.

## 8. Thesis-ready interpretation
- Serves as a feature-selection ablation: comparing RF-selected 77-dim inputs against full 341-dim inputs to see if classical feature filtering plus KAN regularisation stabilises multiclass performance or reduces compute.
- Useful for a Methods/Experiments subsection on classical feature selection versus PCA for voxel-signature models.

## 9. Limitations and open questions
- RF-selected feature file path is absolute and external; data are missing here, so results are not reproducible from the repo alone.
- Regularisation terms are defined but not clearly applied in the loss; their empirical impact is UNKNOWN.
- No logged metrics; effectiveness of the 77-dim subset relative to PCA/full features remains to be demonstrated.


---
## Source file: MLP-PatientSplit-Alex7T-Train38-汇总毕设.md
Path: MLP-PatientSplit-Alex7T-Train38-汇总毕设.md
---

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


---
## Source file: MLPVariantsBayesOpt-Alex7T-102labels-汇总毕设.md
Path: MLPVariantsBayesOpt-Alex7T-102labels-汇总毕设.md
---


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


---
## Source file: OriginalMLP-CouplingLR-Alex7T-102labels-汇总毕设.md
Path: OriginalMLP-CouplingLR-Alex7T-102labels-汇总毕设.md
---


# OriginalMLP-CouplingLR on Alex7T-102labels

## 1. Role in the thesis
- Reproduction of the original notebook-style 4×4096 MLP training to study coupling between architecture width/depth and learning rate on the Alex ultra-multimodal dataset.
- Provides quick sweeps to select stable hyperparameters before full-scale training.

## 2. Code files and entry points
- comparison_alex/coupling_test.py: defines OriginalStyleMLP and runs grid over architectures × learning rates; logs results and produces analysis report.
- comparison_alex/run_coupling_test.sh: convenience launcher (nohup) referencing the Python script.
- comparison_alex/对比.ipynb: related exploratory notebook (not executed here).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021 (TRAIN38.mat).
- Input: 341-channel voxel vectors.
- Labels: one-hot 102 classes including background; no background removal in this reproduction.
- Splits: prob_idx!=38 used for train/test (1% test fraction), prob_idx==38 used as validation.

## 4. Preprocessing pipeline
- StandardScaler fitted on training subset; applied to validation and the 1% test split.
- No PCA or feature selection; data transposed to (samples × 341) and labels to (samples × 102).

## 5. Model architecture
- OriginalStyleMLP baseline: 4×4096 ReLU with Dropout(0.5) and L2 regularisation (1e-5) on weights.
- Variants tested: wide_shallow (2×8192) and narrow_deep (6×2048) using the same activation/dropout/L2 scheme.

## 6. Training configuration
- Optimiser: Adam; learning rates swept over {5e-6, 1e-5, 2e-5, 5e-5}.
- Batch size 128; epochs 12 for fast sweeps (original notebook used 25). Loss: CrossEntropyLoss + L2 penalty.
- Metrics per run: validation accuracy, macro F1, Cohen’s kappa; training history recorded.

## 7. Evaluation metrics and outputs
- Metrics: validation accuracy/F1/kappa (higher is better); tracks convergence per experiment.
- Outputs: coupling_test_results/<timestamp>/ storing JSON/analysis reports (results.json, analysis_report.txt) summarising best lr per architecture and recommending configs.

## 8. Thesis-ready interpretation
- Rapid grid shows how learning rate interacts with architecture depth/width for the 102-class voxel task. Helps justify chosen lr/architecture for the main MLP baseline and highlights stability ranges.

## 9. Limitations and open questions
- Uses reduced epochs; numbers are indicative, not final benchmarks.
- Background class included; no experiments with ignore_index or soft labels.
- No external test evaluation or calibration metrics; focus is validation coupling only.


---
## Source file: ProbSmoothing-Alex7T-102labels-汇总毕设.md
Path: ProbSmoothing-Alex7T-102labels-汇总毕设.md
---

# ProbSmoothing on Alex7T-102labels

## 1. Role in the thesis
Post-processing study that smooths voxel-wise probability maps from the FC baseline to test lightweight spatial regularisation without retraining the classifier.

## 2. Code files and entry points
- `training/B1_Probability_map_post_processing/smooth_postprocess_eval.py`: core smoothing/evaluation script (standard and gated versions).
- `training/B1_Probability_map_post_processing/run_smooth_evaluation.sh`, `training/B1_Probability_map_post_processing/test_gated_smooth.sh`: runnable presets.
- `training/B1_Probability_map_post_processing/README_SMOOTH_POSTPROCESSING.md`: rationale, metrics, and usage notes.

## 3. Dataset and labels
- Inputs: 3D probability volumes from the FC baseline (`softmax_vol` shape ~384×336×256×102) and corresponding ground-truth labels/masks from the Alex dataset.
- Labels: 102 hard classes; masking restricts evaluation to brain voxels.

## 4. Preprocessing pipeline
- Applies 2D average smoothing per slice (3×3 or 7×7 kernels) with mask-aware normalisation; supports sagittal/coronal/axial axes.
- Optional gating: class-gated smoothing (only within predicted class) or uncertainty-gated smoothing (entropy/margin-based sigmoid blend) to preserve edges/high-confidence regions.

## 5. Model architecture
- No learned model; deterministic smoothing operators applied to existing probability maps.

## 6. Training configuration
- Configuration via CLI flags: kernel size, axis, fast convolution path, gating options, uncertainty parameters; no optimisation loop.

## 7. Evaluation metrics and outputs
- Metrics: gross accuracy, macro-F1, Cohen’s κ, macro/micro AUPRC, delta improvements vs raw probabilities; confusion matrices.
- Outputs: smoothed probability volumes (HDF5), CSV metrics, plots (confusion comparisons), and logs of gating behaviour.

## 8. Thesis-ready interpretation
Demonstrates how mild spatial smoothing can refine noisy voxel predictions, offering a low-cost alternative to retraining. Useful for discussing post-processing effects on imbalanced tissue classes.

## 9. Limitations and open questions
- Effectiveness depends on baseline probability quality; optimal kernel/gating parameters may vary across subjects/classes.
- Purely 2D smoothing may miss through-plane consistency; no recalibration of probabilities beyond smoothing.


---
## Source file: PseudoInverse-FeatureSelection-Alex7T-102labels-汇总毕设.md
Path: PseudoInverse-FeatureSelection-Alex7T-102labels-汇总毕设.md
---

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


---
## Source file: PseudoInverse-PCA-Alex7T-102labels-汇总毕设.md
Path: PseudoInverse-PCA-Alex7T-102labels-汇总毕设.md
---

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


---
## Source file: QC-MultimodalRegistration-Alex7T-351modes-汇总毕设.md
Path: QC-MultimodalRegistration-Alex7T-351modes-汇总毕设.md
---

# QC-MultimodalRegistration on Alex7T-351modes

## 1. Role in the thesis
Quality-control suite assessing multimodal registration/alignment before training, ensuring the 351-channel inputs are spatially consistent and free of gross artefacts.

## 2. Code files and entry points
- `dataset_create/multimodal_mri_qc_analysis/multimodal_mri_qc_analysis.ipynb`: main notebook for similarity analysis (task 1).
- `dataset_create/multimodal_mri_qc_analysis/multimodal_qc_tasks_extra.py`, `multimodal_mri_qc_analysis1.py`: scripted tasks 2–5 for edge/ROI/QC scoring.
- `dataset_create/multimodal_mri_qc_analysis/MULTIMODAL_QC_ANALYSIS_README.md`, `QUICKSTART_QC_ANALYSIS.md`: workflow, thresholds, and data requirements.
- `dataset_create/multimodal_mri_qc_analysis/CHANGELOG_QC_v1.1.0.md`, `CHANGELOG_QC_v1.2.0.md`: algorithm and performance updates.

## 3. Dataset and labels
- Dataset: 3D “minimal” Alex 7T volumes (384×336×256×351) at MPRAGE spacing ≈0.65 mm; ROI masks/labels shared with training data.
- Labels: uses FreeSurfer-derived regions for ROI-based checks; no model training labels are produced here.

## 4. Preprocessing pipeline
- Loads 351 channels, applies modality-family groupings, spacing-aware metrics (0.65 mm), and optional CEST slab localisation.
- Supports anisotropic PSF considerations and masking to focus on brain voxels.

## 5. Model architecture
- Not applicable (analysis/QC only).

## 6. Training configuration
- Notebook/script parameters for thresholds: LNCC/NGF minima, MIND-SSD, ASSD/HD95 maxima (mm), edge IoU, MAD-based outlier detection.
- Optional ROI lists and UMAP/PCA settings for dimensionality reduction.

## 7. Evaluation metrics and outputs
- Similarity matrices: LNCC, NGF, MIND-SSD across modality families (higher is better for LNCC/NGF; lower for MIND-SSD).
- Edge consistency: ASSD, HD95 (lower is better), edge IoU (higher is better) with adaptive Canny thresholds.
- ROI signal consistency heatmaps; PCA/UMAP scatter plots; QC PASS/WARN/FAIL scoring with CSV/JSON summaries.
- Outputs: PNG figures (similarity matrices, edge metrics, ROI heatmaps, PCA/UMAP), `modality_qc.csv`, `qc_analysis_report.json` under `qc_analysis_results/`.

## 8. Thesis-ready interpretation
Provides objective evidence of multimodal alignment quality and identifies problematic channels before model training. Supports thesis sections on data curation and registration reliability, especially when justifying exclusion of poor-quality cases.

## 9. Limitations and open questions
- Requires correct data paths and parameter tuning per dataset; runtime can be high for all 351 channels.
- QC thresholds are heuristic; decisions on PASS/WARN/FAIL may need human verification.


---
## Source file: ResNet50Patch-Alex7T-102labels-汇总毕设.md
Path: ResNet50Patch-Alex7T-102labels-汇总毕设.md
---

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


---
## Source file: ResidualMLP-Alex7T-101labels-汇总毕设.md
Path: ResidualMLP-Alex7T-101labels-汇总毕设.md
---

# ResidualMLP on Alex7T-101labels

## 1. Role in the thesis
Residual MLP variant assessing whether skip-connected blocks (with optional bottlenecks) stabilise training and improve accuracy on the Alex ultra-multimodal 7T voxel signatures relative to plain/deep MLPs.

## 2. Code files and entry points
- `models/residual_mlp.py`: residual blocks with optional bottleneck compression/expansion inside each block.
- Training/eval/predict stack shared with other MLPs (`main.py`, `train.py`, `config.py`, data loaders, samplers, metrics, model_io, visualisation).
- Hyperparameter search for bottleneck usage/width via `utils/optimization.py`; runnable through `run.sh` with `--model_type residual_mlp` or BayesOpt.

## 3. Dataset and labels
Same Alex ultra-multimodal 7T setup (341 features, 101 classes with background configurable). Fixed prob_idx split (val 20, test 38) unless overridden; supports background filtering/ignoring and patient-aware batching/class weighting.

## 4. Preprocessing pipeline
Shared with other groups: patientwise/global standardisation, optional PCA (off), configurable background handling, scaler persistence, split validation.

## 5. Model architecture
- Input layer followed by a series of residual blocks; each block applies two linear layers with activation/dropout and adds a shortcut (linear projection if dimensions differ).
- Optional bottleneck when hidden width >1000 to compress by `bottleneck_factor` (default 0.5) before expansion.
- Activation selectable (`relu`/`gelu`/`swish`), dropout configurable; final linear head to `num_class`.

## 6. Training configuration
- Same as baseline: cross-entropy with class weights, AdamW/Adam optimisers, cosine/multistep/plateau schedulers, batch size 128, epochs 30, validation every 3 epochs.
- BayesOpt trials can toggle bottleneck use and block widths; configs saved alongside checkpoints.

## 7. Evaluation metrics and outputs
- Identical metric set and artefacts (accuracy, balanced accuracy, macro/weighted F1, kappa, per-class stats, confusion matrices, training curves, predictions exports). Best checkpoints picked by validation macro-F1.

## 8. Thesis-ready interpretation
- Tests whether residual connections alleviate optimisation difficulties and class imbalance sensitivity in high-dimensional voxel signatures.
- Suitable for an ablation subsection comparing architectural tweaks (plain vs deep vs residual) before moving to more novel models.

## 9. Limitations and open questions
- Residual implementation stores limited metadata (bottleneck flags) in checkpoints; must inspect saved configs for exact block layouts.
- No spatial priors or calibration metrics; same reliance on single validation/test patients.
- Performance gains, if any, depend on BayesOpt-selected widths; defaults may mirror baseline capacity.


---
## Source file: ResidualMLP-Alex7T-102labels-汇总毕设.md
Path: ResidualMLP-Alex7T-102labels-汇总毕设.md
---


# ResidualMLP on Alex7T-102labels

## 1. Role in the thesis
- Residual MLP variant intended to mitigate optimization difficulty in very wide layers via residual shortcuts and optional bottlenecks, still targeting the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Explored through the shared Optuna search to test whether residual connections/bottlenecks yield more stable voxel-wise performance than the plain FC baseline.

## 2. Code files and entry points
- `models/residual_mlp.py`: Defines residual blocks with optional bottleneck compression/expansion and linear shortcuts when layer widths change.
- Uses the same training/evaluation stack as other MLPs: `main.py`, `train.py`, `models/config.py`, `data/mat_loader.py`, `utils/optimization.py`, `utils/metrics.py`, `utils/model_io.py`, `utils/visualization.py`.
- Select via `--model_type residual_mlp` or allow Optuna to pick it.

## 3. Dataset and labels
- Same TRAIN38/DEMO38 Alex ultra-multimodal 7T dataset; 341-channel inputs, 102 classes with background=0 ignored; patient-wise splits and class weighting match the baseline.

## 4. Preprocessing pipeline
- Same StandardScaler normalization and scaler persistence; PCA flags unused; background ignored in loss/metrics.

## 5. Model architecture
- Input linear layer → sequence of residual blocks. Each block: main path with two linear layers plus activation/dropout (or bottleneck compress-expand when width>1000 and use_bottleneck=True), shortcut as identity or linear to match dimensions, activation after addition.
- Final linear layer outputs 102 logits. Hidden_dims options include [4096,4096,4096,4096], [2048,...], [1024,2048,2048,1024], [4096,2048,2048,4096].

## 6. Training configuration
- Same loss, optimizer choices, schedulers, epochs, batch sizes, and validation cadence as other MLPs.
- Optuna explores use_bottleneck flag and bottleneck_factor (0.25–0.5) alongside shared hyper-parameters (lr, weight_decay, dropout, activation, optimizer, scheduler, batch size).

## 7. Evaluation metrics and outputs
- Identical metric set and artifact locations as other MLP variants (`results/<experiment>` logs, CSVs, plots, checkpoints, summaries).

## 8. Thesis-ready interpretation
- Tests whether residual/bottleneck structure improves optimization stability and per-class F1 on dense voxel signatures relative to the plain FC baseline.
- Useful as an ablation to argue whether architectural tweaks within dense nets suffice or if more structured models are necessary.

## 9. Limitations and open questions
- ResidualMLP is only reached via manual flag or Optuna; no dedicated config ensures it was actually run.
- Metadata fields such as dropout_rate/use_bottleneck are not stored as attributes, so get_model_info may be incomplete; review checkpoint metadata before reuse.
- Shared uncertainties remain (atlas provenance, voxel counts, hard-coded data paths).


---
## Source file: SubjectEmbedding-Alex7T-EmbeddingFeasibility-汇总毕设.md
Path: SubjectEmbedding-Alex7T-EmbeddingFeasibility-汇总毕设.md
---

# SubjectEmbedding on Alex7T-EmbeddingFeasibility

## 1. Role in the thesis
- End-to-end brain-aware subject-embedding feasibility analysis on the Alex ultra-multimodal 7T dataset from German et al. 2021, with checkpoints and decision logic.
- Evaluates subject differences, separability, embedding design, and generates implementation recommendations while retaining the alex 4×4096 dense network as the neural baseline.

## 2. Code files and entry points
- embedding_project/main.py: CLI entrypoint with checkpoint/resume options; orchestrates data loading, analysis phases, visualisation, and report generation.
- embedding_project/src/analyzer.py: BrainAwareSubjectEmbeddingAnalyzer implementing phases (data prep, subject difference, separability, embedding design, decision), deep network wrappers, region-specific nets, visualisation, final report.
- embedding_project/src/data_loader.py: loads TRAIN38_no_label43.mat, Multi-Subject-Out split, scaling.
- embedding_project/config/settings.py plus src/settings.py: feature groups, thresholds, matplotlib setup, default paths/seeds.
- embedding_project/checkpoint_manager.py and checkpoint_tools.py: checkpoint save/restore, metadata, cleaning and export utilities.
- embedding_project/src/utils.py: logging, config persistence, progress tracking; run.sh to launch with environment checks.

## 3. Dataset and labels
- Alex ultra-multimodal 7T dataset from German et al. 2021; default path TRAIN38_no_label43.mat.
- Features: 341 channels (15 QTI, 210 b-tensor, 4 CEST params, 112 Z-spectrum).
- Labels: one-hot brain region IDs (≈101–102 classes; label 43 missing), hard labels.
- Split: train subjects 1–30, val 31–37, test 38; subject IDs stored as prob_idx; masks built for each split.
- Sampling: voxel-level; builds subject×region tensor using mean features when ≥50 voxels per combination; tracks missing regions.

## 4. Preprocessing pipeline
- StandardScaler fitted on train voxels, applied to val/test; both raw and scaled stored.
- Validates required keys, reports subject counts and missing regions; logs missing classes and label coverage.
- Builds subject-region tensors, computes coverage stats, and stores label_info (one-hot dim, missing classes, min/max label).
- Matplotlib forced to Agg backend for headless plotting; environment checks recorded.

## 5. Model architecture
- Deep 4×4096 dense net (ReLU + dropout 0.5) with linear logits; output dims inferred from labels (≥101 classes) and matches alex baseline.
- Region-specific smaller net: 1024 → 512 → 256 → output with dropout 0.3 for per-region subject recognition experiments.
- DeepNetworkWrapper exposes sklearn-like fit/predict/predict_proba with stored training history; includes kernel L2 regularisation.
- Uses GPU if available; seeds fixed for deterministic behaviour.

## 6. Training configuration
- Loss: cross-entropy plus explicit L2 regularisation on weights.
- Optimiser: Adam with lr 1e-5, weight_decay 1e-5; epochs 25; batch_size 128 (region nets: bs 64, epochs 10 in quick tests); dropout 0.5.
- Data preparation: torch TensorDataset, DataLoader shuffle in training; softmax only applied during predict_proba.
- Checkpoints saved after each phase; emergency checkpoints created on failure or interrupt.

## 7. Evaluation metrics and outputs
- Classification: training accuracy tracked per epoch; optional region-specific deep accuracy and specificity strength; baseline analyses compare deep vs simpler models (logistic/RF placeholders).
- Subject-difference metrics: PCA variance, inter-subject distance/correlation matrices, ANOVA/Kruskal tests, silhouette scores, KMeans/AGG/DBSCAN clustering.
- Embedding necessity: patient variance ratios, region-wise specificity scores, separability indices, UMAP clustering quality, comprehensive embedding necessity scores and tiered region priorities.
- Visualisations: correlations, PCA/TSNE/UMAP plots, dimensionality comparison, region-specificity heatmaps; saved under output_dir/visualizations.
- Reports and logs: brain_aware_analysis_output/outputs/*.json (analysis_config, key_results, phase results), report txt, checkpoints/*.ckpt, recovery logs, plotted .png, trained model states inside checkpoints.

## 8. Thesis-ready interpretation
- Provides a structured diagnosis of whether subject embeddings are required for voxel-wise brain-region classification on Alex 7T data, combining statistical variation analysis and neural baselines.
- Generates actionable recommendations (uniform vs selective vs hierarchical embedding architectures, priority regions, implementation timeline) that can feed a Methods/Discussion section on embedding strategy selection.
- Useful for motivating personalised or region-adaptive models by showing where baseline separability fails and where subject-specific signals dominate.

## 9. Limitations and open questions
- Actual performance numbers depend on executed runs; none are hardcoded.
- Assumes specific .mat schema and subject ID ranges; alternative datasets would need adaptation.
- Region-level thresholds (≥50 voxels) and missing-class handling may bias small structures; consider sensitivity analysis.
- TabNet/KAN or calibration methods are not implemented in this pipeline despite being explored elsewhere.

