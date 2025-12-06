
# FixedMLP-BackgroundWeighting on Alex7T-102labels

## 1. Role in the thesis
- Ablation to diagnose background-handling and class-imbalance strategies for voxel-wise classification on the Alex ultra-multimodal dataset.
- Compares four scenarios: with/without background training and with/without inverse-frequency class weights to resolve NaN losses and performance drops.

## 2. Code files and entry points
- comparison_alex/4场景对比/comparison_experiment.py: end-to-end experiment driver defining four scenarios, data loading, training, evaluation, and plotting.
- comparison_alex/4场景对比/run_4_senario.sh: nohup wrapper to launch the experiment and log output.
- comparison_alex/4场景对比/4场景对比alex代码数据.ipynb: exploratory notebook version of the same comparison (not executed here).

## 3. Dataset and labels
- Dataset: Alex ultra-multimodal 7T dataset from German et al. 2021 (TRAIN38.mat).
- Input: 341 features per voxel (transposed to samples × features).
- Labels: one-hot 102 classes including background; scenarios either keep background as a class or remap it to ignore_index (-1) while training on 101 foreground classes.
- Splits: patient-wise—prob_idx==38 held out as validation; remaining patients split with 1% as test and the rest as training.

## 4. Preprocessing pipeline
- StandardScaler fitted on training voxels; applied to val/test; scaler stats printed for QC.
- Optional background filtering in dataset class depending on scenario; when filtering, validation background can be removed to avoid all-ignore batches.
- No PCA or additional feature selection.

## 5. Model architecture
- FixedMLP: 4×4096 ReLU + Dropout(0.5) fully-connected stack with Xavier init; optional L2 regularisation (1e-5) added to loss.
- Output dimension switches between 101 (when background ignored) and 102 (when background kept).

## 6. Training configuration
- Optimiser: Adam, lr=1e-5; batch_size=128; epochs=25 (NUM_EPOCHS_DEMO).
- Loss: CrossEntropyLoss; ignore_index=-1 when background filtered. Class weights optionally applied via inverse frequency per scenario.
- Metrics tracked each epoch: validation accuracy and macro F1; training loss/acc/F1 also logged.

## 7. Evaluation metrics and outputs
- Metrics: validation accuracy and macro F1 (higher is better); NaN checks to prevent invalid losses.
- Outputs: comparison_results_fixed/ with final_results_summary_fixed.txt (per-scenario val metrics), validation_curves_comparison_fixed.png, training_loss_comparison_fixed.png, and per-scenario histories in memory.

## 8. Thesis-ready interpretation
- Tests whether treating background as a learnable class or ignoring it, and whether applying class weights, stabilises training and improves validation performance. Intended to justify background handling choices in the main pipeline.

## 9. Limitations and open questions
- Paths hard-coded to TRAIN38.mat under /home/jovyan/...; adjust before reuse.
- Only validation metrics reported; no held-out test evaluation or calibration analysis.
- Uses a single fixed architecture and learning rate; scenarios isolate background/weighting effects but not architecture-dependent behaviour.
