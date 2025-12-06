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
