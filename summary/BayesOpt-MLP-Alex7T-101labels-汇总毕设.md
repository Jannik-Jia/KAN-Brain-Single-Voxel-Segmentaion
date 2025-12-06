# BayesOpt-MLP on Alex7T-101labels

## 1. Role in the thesis
Automated hyperparameter and architecture search across base/deep/residual MLPs on the Alex ultra-multimodal 7T dataset. Provides a data-driven way to select the strongest FC-style model before comparing against alternative architectures.

## 2. Code files and entry points
- `utils/optimization.py`: Optuna objective/search space, including learning rate/weight decay/dropout/activation/optimiser/scheduler/model_type/depth/width/skip/bottleneck options; visualisation and reporting utilities.
- Integration hooks in `main.py` (`--run_bayesian_opt`, `--n_trials`) and interactive launcher `run.sh` (auto-tests all three MLP types with user-chosen standardisation/background mode).
- Uses shared loaders/trainers (`train.py`, `data/mat_loader_patientwise.py`, `data/samplers.py`, `utils/metrics.py`, `utils/model_io.py`).

## 3. Dataset and labels
Same Alex ultra-multimodal 7T setup with 341 features and 101 labels (background configurable). Patientwise standardisation is assumed for most searches; fixed prob_idx split (val 20, test 38) unless presets override. Class weights computed from training labels.

## 4. Preprocessing pipeline
Identical to other groups: patientwise/global standardisation, optional background filtering/ignoring, scaler persistence, optional PCA hook (off by default), patient-aware batching if enabled.

## 5. Model architecture
- Search spans `model_type` ∈ {base_mlp, deep_mlp, residual_mlp}.
- Hidden widths cover 1024–8192 variants and depth 4–12 with width strategies (constant/decreasing/increasing/hourglass/bell); residual bottlenecks optional; skip connections toggleable.
- Activation (`relu`/`gelu`/`swish`), dropout up to 0.8.

## 6. Training configuration
- Trials train for 5–20 epochs (bounded by `epochs`) with cross-entropy + class weights; optimiser choices Adam/AdamW/SGD/RMSprop; schedulers cosine/step/plateau/none; optional gradient clipping.
- Validation macro-F1 is the optimisation target; MedianPruner used for early stopping. Best params saved to JSON and `optimized_config.json`; subsequent full training reuses best settings.

## 7. Evaluation metrics and outputs
- Trial metrics logged via Optuna; aggregated results exported to `<study_name>_results.json` and analysis figures (`architecture_performance_boxplot.png`, `architecture_trial_counts.png`, `performance_evolution.png`, parameter importance/history plots when available) under the study directory.
- The final model still evaluated with the standard metric suite (accuracy, balanced accuracy, macro/weighted F1, kappa) and saved checkpoints/plots as in other groups.

## 8. Thesis-ready interpretation
- Captures the effect of systematic hyperparameter search on FC-style models, showing whether performance gains arise from architecture depth/width choices or optimisation settings rather than new model families.
- Can underpin a subsection on automated model selection and provide the chosen configuration for downstream comparisons.

## 9. Limitations and open questions
- Search uses shortened training epochs; the winning config should be retrained fully to confirm gains.
- Optuna adds computational overhead; results depend on the fixed val patient (20) and may overfit that subject.
- Requires Optuna dependency and may need GPU resources for large widths; not all trials save intermediate checkpoints.
