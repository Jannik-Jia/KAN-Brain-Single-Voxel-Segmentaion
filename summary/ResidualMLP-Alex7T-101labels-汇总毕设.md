# ResidualMLP on Alex7T-101labels

## 1. Role in the thesis
Residual MLP variant assessing whether skip-connected blocks (with optional bottlenecks) stabilise training and improve accuracy on the Alex ultra-multimodal 7T voxel signatures relative to plain/deep MLPs.

## 2. Code files and entry points
- `models/residual_mlp.py`: residual blocks with optional bottleneck compression/expansion inside each block.
- Training/eval/predict stack shared with other MLPs (`main.py`, `train.py`, `config.py`, data loaders, samplers, metrics, model_io, visualisation).
- Hyperparameter search for bottleneck usage/width via `utils/optimization.py`; runnable through `run.sh` with `--model_type residual_mlp` or BayesOpt.

## 3. Dataset and labels
Same Alex ultra-multimodal 7T setup (341 features, 101 classes with background configurable). Fixed prob_idx split (val 20, test 38) unless overridden; supports background filtering/ignoring and patient-aware batching/class weighting.

## 4. Preprocessing pipeline
Shared with other groups: patientwise/global standardisation, optional PCA (off), configurable background handling, scaler persistence, split validation.

## 5. Model architecture
- Input layer followed by a series of residual blocks; each block applies two linear layers with activation/dropout and adds a shortcut (linear projection if dimensions differ).
- Optional bottleneck when hidden width >1000 to compress by `bottleneck_factor` (default 0.5) before expansion.
- Activation selectable (`relu`/`gelu`/`swish`), dropout configurable; final linear head to `num_class`.

## 6. Training configuration
- Same as baseline: cross-entropy with class weights, AdamW/Adam optimisers, cosine/multistep/plateau schedulers, batch size 128, epochs 30, validation every 3 epochs.
- BayesOpt trials can toggle bottleneck use and block widths; configs saved alongside checkpoints.

## 7. Evaluation metrics and outputs
- Identical metric set and artefacts (accuracy, balanced accuracy, macro/weighted F1, kappa, per-class stats, confusion matrices, training curves, predictions exports). Best checkpoints picked by validation macro-F1.

## 8. Thesis-ready interpretation
- Tests whether residual connections alleviate optimisation difficulties and class imbalance sensitivity in high-dimensional voxel signatures.
- Suitable for an ablation subsection comparing architectural tweaks (plain vs deep vs residual) before moving to more novel models.

## 9. Limitations and open questions
- Residual implementation stores limited metadata (bottleneck flags) in checkpoints; must inspect saved configs for exact block layouts.
- No spatial priors or calibration metrics; same reliance on single validation/test patients.
- Performance gains, if any, depend on BayesOpt-selected widths; defaults may mirror baseline capacity.
