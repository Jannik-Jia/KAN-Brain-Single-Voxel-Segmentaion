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
