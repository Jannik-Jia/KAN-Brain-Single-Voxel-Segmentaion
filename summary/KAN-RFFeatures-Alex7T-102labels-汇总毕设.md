# KAN-RFFeatures on Alex7T-102labels

## 1. Role in the thesis
- Ablation studying whether random-forest feature selection (77 dims) plus KAN regularisation improves 102-class voxel classification on the Alex ultra-multimodal 7T dataset.
- Tests dimensionality reduction via classical feature engineering before KAN, contrasting with PCA-based or full-feature models.

## 2. Code files and entry points
- `1DKAN_随机森林特征选择_完全分离的数据集创建方法_102分类_L1.ipynb`: Loads RF-selected features from `rf_selected_features.h5`, z-score normalises them, optionally balances training data, and trains a FastKAN `[77,128,102]` model with scheduler support.
- Contains helper functions to inspect modality-group indices (diffusion/QTI/CEST), prepare balanced datasets, and monitor KAN weight/entropy statistics.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021.
- Input: 77 features chosen by a prior random-forest selector (indices grouped by diffusion/QTI/CEST); data provided in HDF5 splits (`all/train`, `all/val`, `all/test`).
- Labels: 102-class parcellation; labels are integers, no soft targets; background handling not explicitly described (assumed absent in the RF export).
- Splits: precomputed train/val/test inside the HDF5; optional balancing to `TARGET_SAMPLES` (3000) per class via down-sampling (no oversampling, `MAX_MULTIPLIER=1`).

## 4. Preprocessing pipeline
- Z-score normalisation using training-set mean/std applied to val/test; PCA disabled because feature selection already reduced dimensionality.
- Optional class balancing via down-sampling; augmentation hooks exist but default to no augmentation.

## 5. Model architecture
- FastKAN with layers `[77, 128, 102]`, spline grid size 8; no dropout or batch norm.
- Hyperparameters include L1 and entropy regularisation coefficients (`LAMBDA_L1=0.005`, `LAMBDA_ENTROPY=2.0`), but their integration into the loss is not clearly implemented (effect UNKNOWN).

## 6. Training configuration
- Loss: `nn.CrossEntropyLoss(ignore_index=-1)` (background-ignore placeholder); class weights not used by default.
- Optimiser: Adam (`lr=2e-5`, `weight_decay=1e-3`); batch size 128; epochs 100; validation every 3 epochs.
- Learning-rate scheduler options mirror the multiclass baseline (MultiStep [25,50,75], gamma 0.5; cosine/plateau alternatives available).

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen kappa; per-class precision/recall/F1 and confusion matrices for train/test/val.
- Outputs expected under `Results/BrainVoxel_102Class_RF/BrainVoxel` (checkpoints/plots), but trained weights/CSVs are not versioned in the repo.

## 8. Thesis-ready interpretation
- Serves as a feature-selection ablation: comparing RF-selected 77-dim inputs against full 341-dim inputs to see if classical feature filtering plus KAN regularisation stabilises multiclass performance or reduces compute.
- Useful for a Methods/Experiments subsection on classical feature selection versus PCA for voxel-signature models.

## 9. Limitations and open questions
- RF-selected feature file path is absolute and external; data are missing here, so results are not reproducible from the repo alone.
- Regularisation terms are defined but not clearly applied in the loss; their empirical impact is UNKNOWN.
- No logged metrics; effectiveness of the 77-dim subset relative to PCA/full features remains to be demonstrated.
