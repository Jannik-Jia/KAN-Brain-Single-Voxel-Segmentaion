# AttnResMLP on Alex7T-102labels

## 1. Role in the thesis
- Attention-enhanced residual MLP baseline for voxel-wise classification on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Tests transformer-like upgrades (Stage1 basic residual, Stage2 adds self-attention, Stage3 adds pre-LN + FFN) against simpler FC/KAN ideas while keeping per-voxel inputs.
- Focuses on handling 102 hard labels with class imbalance mitigation and evaluates across train/val/test and merged sets.

## 2. Code files and entry points
- : end-to-end notebook defining data samplers, loaders, optional PCA, model variants (Stages 1–3), training loop, evaluation, and visualizations.
- Key components: , , , , model classes , , , , , training/eval helpers (, , plotting).
- Outputs stored under  relative to the notebook.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; voxel features under .
- Input dimensionality: default 341 channels (diffusion/QTI, CEST offsets, Amide/Amine/NOE/MT, etc.), optional PCA to  (default 0 = no PCA).
- Labels: 102 tissue/region classes (ids 1–102) plus background 0; background remapped to  and ignored by the loss; class mapping saved to .
- Splits: restructured data from  with approx 60/20/20 train/val/test; additional “merged” set loads all labels for a sanity check.
- Sampling: optional balancing (, ) with simple copy augmentation capped by ; can also use all samples ().

## 4. Preprocessing pipeline
- Optional PCA on concatenated train/test/val followed by per-component min-max normalization;  by default (raw 341-D features).
- No explicit normalization when PCA is off; assumes upstream feature scaling.
- Label handling: background -> ; class weights computed from training label counts.
- Data restructuring helper merges legacy train/val and splits 0.6/0.2/0.2 (run once upstream).
- No missing-value handling or modality-specific scaling in code.

## 5. Model architecture
- Stage1 : input linear + GELU + BatchNorm -> stacked s over  (default [256,256,256]) -> linear classifier.
- Stage2 adds multi-head  (num_heads=4, dropout=0.1) after each residual block with LayerNorm and residual fusion.
- Stage3 Transformer-style: input embedding (Linear -> LayerNorm -> SiLU -> Dropout repeated), blocks of pre-LN multi-head attention (num_heads=8, dropout=0.2) + FFN (4x expansion with SiLU) + projection residuals, final LayerNorm + classifier.
- Regularization: dropout 0.1/0.2, BatchNorm in early stages; L1/entropy regularization flags defined (, , ) but not applied in the loss.

## 6. Training configuration
- Loss:  with inverse-frequency class weights,  for background.
- Optimizer: AdamW (, ); seed 666.
- LR schedules: optional; default  milestones [15,35,50,75], gamma 0.6; cosine and plateau alternatives present.
- Batch size 256; epochs 100; validation every 3 epochs.
- Data balance via resampling/augmentation; no mixup or cutmix; no explicit gradient clipping.
- Checkpoints saved each validation step to ; config saved to .

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa; per-class precision/recall/F1; confusion matrix (log-scaled heatmaps).
- Visualizations: class distribution bars, per-class accuracy bars, confusion matrices, multi-dataset metric comparisons.
- Outputs:  plots (e.g., , , ), text reports (, ), checkpoints , , .

## 8. Thesis-ready interpretation
- Provides a strong FC/MLP-style baseline enriched with attention and residual connections for the full 102-class label set on the Alex dataset.
- Illustrates how transformer-style components affect voxel-wise classification and class-imbalance behavior, informing comparisons to KAN/TabNet/linear baselines.
- Fits a “Baseline and attention-enhanced MLPs” section or an ablation on attention depth.

## 9. Limitations and open questions
- Exact metric values are not embedded; rely on saved reports for numbers.
- Hard-coded data paths to  and reliance on pre-built  reduce portability.
- No calibration metrics (ECE/NLL) or uncertainty estimates; normalization when PCA is off is unclear.
- L1/entropy regularization flags are unused; Stage1/2/3 comparisons are not reported in the notebook.
