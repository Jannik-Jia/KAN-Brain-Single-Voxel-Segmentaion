# DeepMLP6x2048 on Alex7T-102labels

## 1. Role in the thesis
- Best-performing MLP variant derived from Bayesian search: deeper network with StandardScaler normalisation to push baseline performance on the Alex 7T voxel-signature task.
- Intended as the strong FC reference model for comparisons with more exotic architectures and for downstream inference (e.g., demo38 deployment).

## 2. Code files and entry points
- `train_best_model.py`: end-to-end script that fits a StandardScaler, trains the deep MLP with fixed best hyperparameters, saves scaler + checkpoints + reports.
- `data/dataset.py`: loads restructured train/test/val voxel features; used here with `norm=False` so that external StandardScaler can be applied consistently.
- `utils/metrics.py`, `utils/visualization.py`: evaluation and plotting for the final model.
- `utils/model_io.py`: checkpoint saving with architecture metadata, JSON sidecars, and safe loading.
- `predict.py` / `predict.sh`: inference pipeline consuming the saved model and scaler for external `.mat` volumes.

## 3. Dataset and labels
- Same Alex ultra-multimodal 7T dataset (341-channel voxel signatures, 102 classes, background ignored) with fixed train/val/test folders.
- Label space and hard labels identical to the FC baseline; per-voxel sampling with no additional balancing beyond class weights in the loss.

## 4. Preprocessing pipeline
- StandardScaler fitted on training samples only, persisted as `<experiment_name>_scaler.pkl`; the same scaler is applied to val/test and saved for inference.
- Normalisation statistics (mean/std ranges) written to a text report; no PCA or dimensionality reduction.

## 5. Model architecture
- `DeepMLP` (`models/deep_mlp.py`) with 6 hidden layers of 2048 units, GELU activation, dropout ≈0.2567 after each layer, optional skip connections enabled.
- Linear head to 102 logits; no batch norm; implemented for dense tabular-style voxel features.

## 6. Training configuration
- Loss: weighted cross-entropy with background ignore index.
- Optimiser: Adam, lr ≈1.697e-4, weight_decay ≈1.33e-5; batch size 128; 30 epochs; validation every epoch.
- LR schedule: StepLR with step_size=2, gamma≈0.2216.
- Class weights computed from training labels; checkpoints saved each validation with architecture metadata and normalisation params recorded.

## 7. Evaluation metrics and outputs
- Metrics identical to the baseline: accuracy, balanced accuracy, macro/weighted F1, kappa, per-class precision/recall/F1, confusion matrices.
- Outputs stored under `./best_model_results/<experiment_name>` plus logs in `./best_model_logs`: scaler `.pkl`, normalisation stats text, training CSV/logs, per-split metrics/plots, final `test_final` evaluation files, and model `.pth` + `_architecture.json`.

## 8. Thesis-ready interpretation
- Represents the tuned FC/MLP alternative that benefits from deeper capacity and consistent z-scoring; should yield stronger macro-F1 and more stable class-wise performance.
- Suitable as the primary baseline against which KAN/TabNet/other approaches are compared; saved scaler enables reproducible inference on new subjects.

## 9. Limitations and open questions
- Hyperparameters are fixed from one search; no cross-validation or robustness analysis to different splits.
- Still relies on voxel-level splits; potential subject-level leakage is untested.
- No calibration or uncertainty assessment; only accuracy-style metrics are logged.
- Assumes availability of the saved scaler; mismatch would degrade inference.
