
# FC4x4096 on Alex7T-102labels-patientwiseStd

## 1. Role in the thesis
A PyTorch reimplementation of the 4×4096 fully connected baseline for voxel-wise classification on the Alex ultra-multimodal 7T dataset from German et al. 2021. This notebook tests patient-wise standardization (per-subject z-scoring) to better match per-patient spectral distributions compared to a global normalization. It serves as a sanity check and reproducibility step against the original TensorFlow baseline while evaluating how patient-specific scaling impacts validation on one held-out subject.

## 2. Code files and entry points
- `update_standarlize/alex的torch版本_新标准化措施.ipynb`: End-to-end notebook with four cells: (1) data loading from `.mat` files and patientwise standardization utilities, (2) definition and training loop for the 4×4096 dense network, (3) inference on `DEMO38.mat` with patientwise stats applied, (4) gradient-based saliency visualization for selected validation voxels.

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021, loaded from `TRAIN38.mat` (`data`, `region`, `prob_idx`) and `DEMO38.mat` (`multidim_data`, optional `prob_idx`).
- Input dimensionality: 341-channel voxel signatures (per-voxel spectral/modal features), millions of voxels aggregated across subjects (exact count not specified). 
- Label space: 102-class one-hot labels in `region`; likely derived from FreeSurfer-style parcellations aligned to the CEST/native grid. Labels are hard one-hot vectors.
- Splits: Training uses all patients except patient 38 (`prob_idx != 38`); validation uses only patient 38 (`prob_idx == 38`). An additional `train_test_split` holds out 1% of the non-38 data, but only `X_train1`/`y_train1` are used for training (the small holdout is unused). Testing/inference cell expects `DEMO38.mat` and applies train stats where possible.

## 4. Preprocessing pipeline
- Patientwise standardization: compute mean/std per patient and z-score each voxel within its patient; zero std replaced with 1.0. Returns and stores `train_patient_stats` and `val_patient_stats`.
- Validation standardization uses validation-patient stats (not training stats); test-time standardization uses training stats when the patient id is known, otherwise falls back to per-patient stats computed on the test data. 
- No additional feature scaling, clipping, or imputation beyond the per-patient z-scoring; assumes input features are dense.

## 5. Model architecture
- Fully connected network matching the TensorFlow baseline: layers of sizes 341 → 4096 → 4096 → 4096 → 4096 → 102.
- Activation: ReLU after each hidden layer; Dropout(p=0.5) applied after each hidden activation.
- Output: linear logits (softmax applied externally during inference); no batch normalization or residual connections.

## 6. Training configuration
- Optimizer: Adam (lr=1e-5); manual L2 regularization on weight matrices only (weight_decay term 1e-5 added to loss).
- Loss: CrossEntropyLoss on argmax of one-hot labels.
- Batch size: 128; Epochs: 25; deterministic seeds set for torch and NumPy; uses GPU if available.
- Metrics tracked per epoch: training loss/accuracy/F1 (macro) and validation loss/accuracy/F1 on patient 38.
- Model checkpoint: saved to `outputs/dense_4x4096_model.pth`.

## 7. Evaluation metrics and outputs
- Metrics: macro F1 (class-balanced) and overall accuracy for train and validation; printed per epoch; best validation accuracy/F1 reported at the end. Higher is better for both accuracy and F1.
- Outputs: `outputs/train_patient_stats.npy`, `outputs/val_patient_stats.npy` (per-patient mean/std); `outputs/dense_4x4096_model.pth` (checkpoint); prediction export to `dense_4x4096_model_prediction.mat` via `scipy.io.savemat` with user-specified `predictpath`; gradient-based saliency plots for selected validation voxels (displayed, not saved by default).

## 8. Thesis-ready interpretation
This experiment checks the reproducibility of the core FC baseline while introducing patient-wise normalization to mitigate inter-subject distribution shifts in voxel signatures. It demonstrates that the 4×4096 dense architecture can be trained in PyTorch with similar regularization to the TensorFlow version and uses a single held-out subject (patient 38) as validation. The saliency visualization offers a preliminary interpretability view of which spectral channels drive predictions.

## 9. Limitations and open questions
- Validation relies on a single subject (patient 38); generalization to other subjects is untested. 
- The small extra split from `train_test_split` is unused, and no dedicated test set beyond patient 38 is evaluated.
- `predictpath` is a placeholder; prediction saving requires a user-provided path. 
- Class definitions and modality breakdown for the 341 channels are not encoded in the notebook; downstream calibration or broader metrics (ECE, confusion matrices) are absent. 
- Validation standardization uses its own stats rather than train stats, which may not reflect deployment-time behavior.
