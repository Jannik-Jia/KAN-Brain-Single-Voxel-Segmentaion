# DeepMLP on Alex7T-101labels

## 1. Role in the thesis
Deeper fully connected variant testing whether extra depth/width and optional skip connections improve voxel-wise classification beyond the 4×4096 baseline on the Alex ultra-multimodal 7T dataset. Serves as an ablation on network capacity and gradient flow.

## 2. Code files and entry points
- `models/deep_mlp.py`: deep MLP with configurable depth/width and optional skip connections between non-adjacent layers.
- Shared training stack: `main.py`, `config.py`, `train.py`, `data/mat_loader_patientwise.py`, `data/mat_loader.py`, `data/samplers.py`, `utils/metrics.py`, `utils/label_processing.py`, `utils/model_io.py`, `utils/visualization.py`.
- Hyperparameter search support via `utils/optimization.py` (depth/width strategies, activations, schedulers) and `run.sh` for interactive runs.
- Evaluation/inference identical to baseline via `evaluate.py`, `predict.py`, notebooks for tuning (`alex版本优化*.ipynb`).

## 3. Dataset and labels
Same as baseline: 341-feature voxels from the Alex ultra-multimodal 7T dataset; 101-class label space (background optional). Fixed prob_idx split (val 20, test 38) with presets for alternate test IDs or random split. Class weights derived from training labels; optional patient-aware batching.

## 4. Preprocessing pipeline
Identical to baseline: patientwise or global standardisation, optional PCA (off), configurable background handling, scaler persistence, split integrity checks.

## 5. Model architecture
- Depth and width configurable: hidden_dims list can extend beyond four layers; BayesOpt explores depths 4–12 with width strategies (constant, decreasing, increasing, hourglass, bell).
- Optional `use_skip_connections` to add additive skips between non-adjacent layers to aid gradient flow.
- Activation choices (`relu`/`gelu`/`swish`), dropout configurable; linear output sized to `num_class`.

## 6. Training configuration
- Same loss/optimiser/scheduler setup as baseline; defaults lr=1e-5, weight_decay=1e-5, batch size 128, epochs 30, cosine LR schedule.
- When `--run_bayesian_opt` is enabled, Optuna trials (5–20 epochs) search learning rate, dropout, activation, optimiser (Adam/AdamW/SGD/RMSprop), scheduler type, depth/width, and skip usage; best params saved to `optimized_config.json`.
- Checkpointing/logging identical to baseline.

## 7. Evaluation metrics and outputs
- Metrics and artefacts match baseline (accuracy, balanced accuracy, macro/weighted F1, kappa, per-class stats, confusion matrices, training curves, predictions npz/mat files).
- Best model still selected via validation macro-F1; outputs stored under `results/<experiment_name>/`.

## 8. Thesis-ready interpretation
- Explores whether deeper MLPs with skip connections better capture nonlinear interactions among the 341 modalities than the shallow baseline.
- Intended for an ablation subsection on network depth/capacity; helps justify whether further complexity (e.g., TabNet/KAN) is needed after MLP scaling.

## 9. Limitations and open questions
- Concrete best-performing depth/width not hard-coded; must inspect saved configs/logs to know selected architecture.
- Optuna trials use shortened training, so full-epoch performance may differ; computationally heavy on large search spaces.
- Same subject-level split limitations and lack of calibration/augmentation as the baseline.
