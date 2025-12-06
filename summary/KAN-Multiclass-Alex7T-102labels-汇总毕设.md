# KAN-Multiclass on Alex7T-102labels

## 1. Role in the thesis
- Extends the KAN approach to full 102-class voxel-wise classification on the Alex ultra-multimodal 7T dataset.
- Tests whether spline-based activations with PCA/standardisation and class balancing can scale beyond one-vs-rest detectors.
- Provides a comparative baseline for later multimodal architectures and for reporting multiclass confusion trends.

## 2. Code files and entry points
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类.ipynb`: Main multiclass FastKAN pipeline with data restructuring, optional PCA, balanced sampling, two hidden KAN layers `[256,128]`, grid size 8, and multi-step learning-rate scheduler.
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类_L1.ipynb`: Variant without PCA (raw 341 dims), hidden 128, stronger weight decay (1e-3) and tooling to monitor KAN weight magnitudes/entropies (regularisation-oriented).
- `1DKAN_brainvoxel_完全分离的数据集创建方法_102分类_多重尝试/…`: Additional trial copy of the main notebook.
- Supporting cells implement dataset merging/splitting (`merge_and_shuffle_datasets`, `split_merged_dataset`), class-weight computation, and feature-importance plotting.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021.
- Input: 341-channel voxel signatures; PCA optional (default on) with automatic component count, or raw features if PCA is disabled.
- Labels: 102 anatomical classes; background set to 0 and remapped to -1 to be ignored by `CrossEntropyLoss`.
- Splits: original train/val merged then re-split 60/20/20 into `restructured/{train,test,val}`; balancing augments minority classes up to `TARGET_SAMPLES` (10000) via simple duplication when enabled.

## 4. Preprocessing pipeline
- Optional PCA followed by normalisation; if PCA is off, z-score is applied to raw features.
- Dataset restructuring ensures per-class `.npy` blocks and a `label_index.txt` for loader access; background voxels dropped.
- Class balancing via oversampling (simple copy) with a cap on augmentation multiplier; optional class weights for the loss (commented in notebooks).

## 5. Model architecture
- FastKAN with layers `[feature_dim, 256, 128, 102]` and spline grid size 8-10 (configurable); no dropout.
- Alternative simpler variant with single hidden size 128 (L1 notebook).
- Feature-importance plots derived from input spline weights; background class ignored in loss.

## 6. Training configuration
- Loss: `nn.CrossEntropyLoss` with optional class weights and `ignore_index=-1` for background.
- Optimiser: Adam (`lr=2e-5`, `weight_decay=5e-3` or `1e-3`); batch size 128; epochs 100; validation every 3 epochs.
- Learning-rate scheduler: MultiStep milestones [25,50,75] with gamma 0.5 (toggleable cosine/plateau options in code).
- Checkpoint naming mirrors the binary pipeline; best-epoch selection via max accuracy/F1 (utility reused from binary code).

## 7. Evaluation metrics and outputs
- Metrics: overall accuracy, balanced accuracy, macro/weighted F1, Cohen kappa; per-class precision/recall/F1; confusion matrices.
- Visuals: dataset distribution plots, PR/ROC curves, and feature-importance bar charts for top PCs/features.
- Outputs intended under `Results/BrainVoxel_102Class/BrainVoxel` (checkpoints/plots); actual numeric results are not stored in the repo.

## 8. Thesis-ready interpretation
- Evaluates feasibility of end-to-end 102-way voxel classification with KAN using all 341 modalities, highlighting class imbalance handling and PCA-driven compression.
- Provides evidence for how performance changes when moving from one-vs-rest to full multiclass, useful for a Methods or Experiments subsection on full parcellation with KAN.
- Feature-importance inspection can motivate modality-level discussions in the appendix.

## 9. Limitations and open questions
- Absolute data paths and external `.npy` blocks are required; reproducibility without the dataset is not possible from the repo alone.
- No logged accuracies/F1/kappa values; scheduler and balancing effects remain UNKNOWN empirically.
- Regularisation settings (L1/entropy monitors) are exploratory and not fully reported.
