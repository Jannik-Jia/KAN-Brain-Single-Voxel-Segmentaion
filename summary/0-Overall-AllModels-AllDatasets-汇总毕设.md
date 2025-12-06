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
