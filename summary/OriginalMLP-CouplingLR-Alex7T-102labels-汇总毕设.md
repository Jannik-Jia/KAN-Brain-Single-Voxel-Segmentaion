
# OriginalMLP-CouplingLR on Alex7T-102labels

## 1. Role in the thesis
- Reproduction of the original notebook-style 4×4096 MLP training to study coupling between architecture width/depth and learning rate on the Alex ultra-multimodal dataset.
- Provides quick sweeps to select stable hyperparameters before full-scale training.

## 2. Code files and entry points
- comparison_alex/coupling_test.py: defines OriginalStyleMLP and runs grid over architectures × learning rates; logs results and produces analysis report.
- comparison_alex/run_coupling_test.sh: convenience launcher (nohup) referencing the Python script.
- comparison_alex/对比.ipynb: related exploratory notebook (not executed here).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021 (TRAIN38.mat).
- Input: 341-channel voxel vectors.
- Labels: one-hot 102 classes including background; no background removal in this reproduction.
- Splits: prob_idx!=38 used for train/test (1% test fraction), prob_idx==38 used as validation.

## 4. Preprocessing pipeline
- StandardScaler fitted on training subset; applied to validation and the 1% test split.
- No PCA or feature selection; data transposed to (samples × 341) and labels to (samples × 102).

## 5. Model architecture
- OriginalStyleMLP baseline: 4×4096 ReLU with Dropout(0.5) and L2 regularisation (1e-5) on weights.
- Variants tested: wide_shallow (2×8192) and narrow_deep (6×2048) using the same activation/dropout/L2 scheme.

## 6. Training configuration
- Optimiser: Adam; learning rates swept over {5e-6, 1e-5, 2e-5, 5e-5}.
- Batch size 128; epochs 12 for fast sweeps (original notebook used 25). Loss: CrossEntropyLoss + L2 penalty.
- Metrics per run: validation accuracy, macro F1, Cohen’s kappa; training history recorded.

## 7. Evaluation metrics and outputs
- Metrics: validation accuracy/F1/kappa (higher is better); tracks convergence per experiment.
- Outputs: coupling_test_results/<timestamp>/ storing JSON/analysis reports (results.json, analysis_report.txt) summarising best lr per architecture and recommending configs.

## 8. Thesis-ready interpretation
- Rapid grid shows how learning rate interacts with architecture depth/width for the 102-class voxel task. Helps justify chosen lr/architecture for the main MLP baseline and highlights stability ranges.

## 9. Limitations and open questions
- Uses reduced epochs; numbers are indicative, not final benchmarks.
- Background class included; no experiments with ignore_index or soft labels.
- No external test evaluation or calibration metrics; focus is validation coupling only.
