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
