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
