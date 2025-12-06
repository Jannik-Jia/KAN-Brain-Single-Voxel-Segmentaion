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
