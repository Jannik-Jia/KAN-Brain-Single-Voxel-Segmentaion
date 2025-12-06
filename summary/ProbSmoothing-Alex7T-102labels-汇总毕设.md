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
