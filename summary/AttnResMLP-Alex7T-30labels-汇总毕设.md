# AttnResMLP on Alex7T-30labels

## 1. Role in the thesis
- Subset study to validate the attention-enhanced residual MLP on a 30-class slice of the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Reduces class space to speed iteration and test sampling/balancing strategies before full 102-class runs.

## 2. Code files and entry points
- : data subset selection, loaders, model variants (Stages 1–3), training and evaluation identical to the full-classes notebook.
- Key helpers reused:  (chooses top classes by sample count), , , , model classes, , evaluation/plotting.
- Outputs under  (relative to notebook).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021; voxel features from .
- Input dimensionality: 341 channels; optional PCA (default off).
- Labels: 30-class subset selected from the 102-class pool; selection requires >=1000 train samples and at least ~100 in test/val; labels remapped to 0–29 with background ignored as .
- Splits: same restructured train/test/val (approx 60/20/20); subset selection performed after reading label indices.
- Sampling: balancing to  per class with simple copy augmentation; class mapping saved.

## 4. Preprocessing pipeline
- Same as full setup: optional PCA + min-max normalization; otherwise raw features; class-weight calculation; background ignored.
- No additional harmonization or missing-value handling.

## 5. Model architecture
- Same Stage1/2/3 attention-enhanced residual MLP options as the full experiment (hidden dims [256,256,256], attention heads 4 or 8, dropout 0.1–0.2).
- Residual blocks with GELU and BatchNorm; Stage3 uses pre-LN transformer-like blocks.

## 6. Training configuration
- Loss:  with class weights, .
- Optimizer AdamW (, ); epochs 100; batch 256; validation every 3 epochs.
- LR scheduler options identical (default MultiStep [15,35,50,75] gamma 0.6).
- Balancing and augmentation enabled; checkpointing and config export to .

## 7. Evaluation metrics and outputs
- Same metric set: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa, per-class metrics, confusion matrices.
- Visual outputs and text reports parallel the 102-class notebook, with paths adjusted to the subset save directory.

## 8. Thesis-ready interpretation
- Fast-turnaround experiment to probe whether the attention-residual MLP scales down gracefully and whether balancing/augmentation pipelines behave as expected on fewer classes.
- Useful as a methodological appendix or pilot study motivating the choice of architecture and sampling strategy before full-scale training.

## 9. Limitations and open questions
- Actual selected class identities depend on data availability and are only logged in .
- Same hard-coded data paths and lack of calibration metrics as the full experiment.
- Performance numbers are not in the notebook text; must read saved reports.
