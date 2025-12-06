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
