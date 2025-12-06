# DeepMLP-Advanced on Alex7T-102labels

## 1. Role in the thesis
- Exploratory variant of the deep MLP adding stronger regularization (shake-shake, stochastic depth), mixed activations, and trilinear feature pooling for the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Intended as an ablation to test whether richer feature interactions and regularization improve 102-class voxel classification.

## 2. Code files and entry points
- : defines advanced modules and runs training/evaluation; also contains leftover KAN evaluation hooks.
- Key components:  with optional trilinear pooling and predefined feature groups (anatomical/functional/diffusion),  with positional embeddings,  supporting shake-shake and stochastic depth, , ,  class.
- Training/eval helpers similar to the base deep MLP (, , plotting, ); outputs under .

## 3. Dataset and labels
- Same Alex ultra-multimodal 7T dataset and 341-channel voxel features from the restructured train/test/val folders.
- Labels: 102 classes plus background ignored as ; uses class weights; hard labels only.
- Splits: train/test/val (~60/20/20) with optional balancing to 10k per class.

## 4. Preprocessing pipeline
- Optional PCA + min-max normalization (off by default); balancing via resampling/augmentation; background removal; config export to .
- No additional harmonization beyond semantic feature grouping inside the model.

## 5. Model architecture
- Hidden dims ; attention layers at indices [1,3,5] with 16 heads; dropout rates  (note potential length mismatch vs hidden dims).
- FeatureInteractionLayer supports trilinear pooling and feature-group-specific projections for anatomical (0–120), functional (120–240), diffusion (240–341) channels.
- Residual blocks can use shake-shake regularization (alpha/beta random mixing), stochastic depth (rate 0.2), mixed GELU/SiLU activations; self-attention includes learned positional embedding.
- Classifier is a linear head; weight init uses Kaiming variants.

## 6. Training configuration
- Loss:  with class weights and ; background ignored.
- Optimizer AdamW (, ); batch size 512 with ; epochs 150.
- Scheduler: OneCycleLR (, ); mixed precision; gradient clipping 10; validation every 3 epochs.
- Checkpointing via  and per-epoch saves; includes plotting of training curves and confusion matrices.

## 7. Evaluation metrics and outputs
- Metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa; per-class metrics; confusion matrices; class F1 distribution plots.
- Outputs:  for training curves and confusion matrices, text test reports (), checkpoints , .

## 8. Thesis-ready interpretation
- Tests aggressive regularization and feature pooling ideas to see if deeper MLPs can better exploit multimodal structure than the base model.
- Suitable for an ablation subsection comparing feature-interaction/regularization strategies within dense architectures.

## 9. Limitations and open questions
- Dropout list length vs hidden layers may need alignment; code mixes DeepMLP and KAN training snippets, indicating work-in-progress.
- Hard-coded data paths and missing calibration metrics remain.
- Actual performance numbers not present in the notebook body.
