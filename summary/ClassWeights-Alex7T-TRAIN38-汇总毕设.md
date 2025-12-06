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
