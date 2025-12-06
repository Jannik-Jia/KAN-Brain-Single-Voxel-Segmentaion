
# DeepMLP on Alex7T-102labels

## 1. Role in the thesis
- Deeper MLP variant to test whether increased depth/width and optional skip connections improve voxel-wise classification beyond the 4×4096 baseline on the Alex ultra-multimodal 7T dataset from German et al. 2021.
- Primarily explored through the shared Optuna search; not the default config but available for ablations on capacity versus overfitting.

## 2. Code files and entry points
- `models/deep_mlp.py`: Builds variable-depth MLPs (width 1024–3072, depth 5–8) with optional skip adapters between non-adjacent layers.
- Shared pipeline files: `main.py`, `train.py`, `models/config.py`, `data/mat_loader.py`, `utils/optimization.py`, `utils/metrics.py`, `utils/model_io.py`, `utils/visualization.py` (as in the FC baseline).
- Selection: activate via `--model_type deep_mlp` in `main.py` or allow Optuna (`model_type` search space) to choose it.

## 3. Dataset and labels
- Same TRAIN38/DEMO38 Alex ultra-multimodal 7T dataset; feature_dim=341, num_class=102 with background=0 ignored.
- Patient-wise splits identical to the baseline unless overridden; class weights applied from training labels.

## 4. Preprocessing pipeline
- Identical StandardScaler normalization, scaler persistence, and optional normalization_params storage as in the FC baseline; PCA flags unused.

## 5. Model architecture
- Input layer → stack of linear layers with chosen activation (ReLU/GELU/Swish) and dropout; widths are constant per config but configurable via hidden_dims list.
- Optional skip connections: adapters align dimensions from earlier to later layers before addition in forward pass.
- Final linear maps to 102 classes; no batch norm.

## 6. Training configuration
- Same loss/optimizer/scheduler setup as the FC baseline; epochs/validation cadence identical unless overridden.
- Optuna search covers depth (5–8), width_factor in {1024, 2048, 3072}, use_skip_connections flag, plus shared hyper-parameters (lr, weight_decay, dropout, activation, optimizer, scheduler, batch size).

## 7. Evaluation metrics and outputs
- Uses the same evaluation stack (accuracy, balanced acc, macro/weighted F1, kappa, per-class metrics, confusion plots) and the same artifact naming/location under `results/<experiment>`.

## 8. Thesis-ready interpretation
- Provides an ablation on depth/skip connections for dense voxel classifiers, indicating whether added capacity/skip paths materially change class-wise F1 or generalization.
- If Optuna selects DeepMLP, its performance can highlight the limits of purely dense feature mixers before considering modality-aware or attention-based models.

## 9. Limitations and open questions
- DeepMLP is not the default path; empirical results depend on Optuna selecting it, so direct dedicated runs may be needed.
- get_model_info fields (`input_dim`, `activation_name`, etc.) are not explicitly stored in the class, which may lead to incomplete metadata in saved checkpoints.
- As with the baseline, label source and voxel counts remain UNKNOWN, and hard-coded data paths in scripts must be updated.
