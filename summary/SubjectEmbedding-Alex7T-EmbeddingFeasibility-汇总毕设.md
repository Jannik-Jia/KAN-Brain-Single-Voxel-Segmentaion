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
