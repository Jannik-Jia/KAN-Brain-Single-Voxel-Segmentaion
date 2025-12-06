
# ResidualMLP on Alex7T-102labels

## 1. Role in the thesis
- Residual MLP variant intended to mitigate optimization difficulty in very wide layers via residual shortcuts and optional bottlenecks, still targeting the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Explored through the shared Optuna search to test whether residual connections/bottlenecks yield more stable voxel-wise performance than the plain FC baseline.

## 2. Code files and entry points
- `models/residual_mlp.py`: Defines residual blocks with optional bottleneck compression/expansion and linear shortcuts when layer widths change.
- Uses the same training/evaluation stack as other MLPs: `main.py`, `train.py`, `models/config.py`, `data/mat_loader.py`, `utils/optimization.py`, `utils/metrics.py`, `utils/model_io.py`, `utils/visualization.py`.
- Select via `--model_type residual_mlp` or allow Optuna to pick it.

## 3. Dataset and labels
- Same TRAIN38/DEMO38 Alex ultra-multimodal 7T dataset; 341-channel inputs, 102 classes with background=0 ignored; patient-wise splits and class weighting match the baseline.

## 4. Preprocessing pipeline
- Same StandardScaler normalization and scaler persistence; PCA flags unused; background ignored in loss/metrics.

## 5. Model architecture
- Input linear layer → sequence of residual blocks. Each block: main path with two linear layers plus activation/dropout (or bottleneck compress-expand when width>1000 and use_bottleneck=True), shortcut as identity or linear to match dimensions, activation after addition.
- Final linear layer outputs 102 logits. Hidden_dims options include [4096,4096,4096,4096], [2048,...], [1024,2048,2048,1024], [4096,2048,2048,4096].

## 6. Training configuration
- Same loss, optimizer choices, schedulers, epochs, batch sizes, and validation cadence as other MLPs.
- Optuna explores use_bottleneck flag and bottleneck_factor (0.25–0.5) alongside shared hyper-parameters (lr, weight_decay, dropout, activation, optimizer, scheduler, batch size).

## 7. Evaluation metrics and outputs
- Identical metric set and artifact locations as other MLP variants (`results/<experiment>` logs, CSVs, plots, checkpoints, summaries).

## 8. Thesis-ready interpretation
- Tests whether residual/bottleneck structure improves optimization stability and per-class F1 on dense voxel signatures relative to the plain FC baseline.
- Useful as an ablation to argue whether architectural tweaks within dense nets suffice or if more structured models are necessary.

## 9. Limitations and open questions
- ResidualMLP is only reached via manual flag or Optuna; no dedicated config ensures it was actually run.
- Metadata fields such as dropout_rate/use_bottleneck are not stored as attributes, so get_model_info may be incomplete; review checkpoint metadata before reuse.
- Shared uncertainties remain (atlas provenance, voxel counts, hard-coded data paths).
