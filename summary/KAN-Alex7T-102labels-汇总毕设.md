# KAN on Alex7T-102labels

## 1. Role in the thesis
Notebook-based attempts to train hierarchical Kolmogorov–Arnold Networks (FastKAN) for voxel-wise tissue classification on the Alex 7T dataset. Provides the first neural baselines beyond classical models, testing expert KAN blocks per modality group and exploring feasibility/accuracy.

## 2. Code files and entry points
- 102LABEL_1DKAN_brainvoxel_分层分类.ipynb: core prototype defining ExpertKAN (FastKAN-based), hyperparameters, data loading, feature selection, and training loops.
- V2_102LABEL_1DKAN_brainvoxel_分层分类_多方法尝试.ipynb, V3_102LABEL_1DKAN_brainvoxel_分层分类_多方法尝试_加速.ipynb, V4_102LABEL_数据结构分层分类_多方法尝试_加速.ipynb: iterative variants with acceleration, additional methods, and data-structure tweaks.
- v2_102LABEL_1DKAN_brainvoxel_完全分离的数据集创建方法.ipynb: dataset restructuring/clean split preparation.
- Shared constants match config.py (MODEL_NAME, feature indices, SAVE_PATH pattern).

## 3. Dataset and labels
- Alex ultra-multimodal 7T dataset; 341-channel voxel signatures (diffusion 0–14, QTI 15–224, CEST 225–340).
- 102 hard labels mapped to 7 big classes for coarse classification; sampling ratio typically 0.1 for speed; train/test/val npy shards.
- Labels treated as integers and occasionally one-hot; no soft-label handling noted.

## 4. Preprocessing pipeline
- Optional PCA (APPLY_PCA True), normalisation flag NORM, and SelectKBest F-stat feature selection per modality group (10/30/20 features typical).
- Robust scaling in enhanced preprocessing; dataset creation notebook focuses on fully separated train/val/test splits.
- No explicit missing-value handling; assumes valid voxels.

## 5. Model architecture
- ExpertKAN class wraps FastKAN with layers_hidden=[input_dim, hidden_dim, num_classes] and grid_size=10.
- Hidden dims per group: diffusion 64, QTI 128, CEST 64; intended as modality experts that can be fused (fusion/gating strategy not fully specified in notebooks).
- Training hyperparameters: EPOCH=50, BATCH_SIZE=640, LR=1e-3, WEIGHT_DECAY=1e-6, RANDOM_SEED=666.

## 6. Training configuration
- Loss: CrossEntropyLoss; optimiser: Adam with above LR/weight decay.
- DataLoader usage inferred; sampling flag USE_SAMPLING=True in some cells to downsample to 10% for faster runs.
- No explicit scheduler/early stopping; checkpointing path via SAVE_PATH in config-like cells.

## 7. Evaluation metrics and outputs
- Classification accuracy the primary metric; confusion matrices plotted via sklearn; ROC/AUC imports present for per-class curves (higher is better).
- Feature selection/cluster analysis code reused from classical pipeline for comparison; figures/logs saved under Results/HierarchicalKAN_BrainVoxel/BrainVoxel/<timestamp>/ when SAVE_PATH is honoured.

## 8. Thesis-ready interpretation
- Establishes feasibility of FastKAN-based voxel classifiers and compares modality-specific experts to classical baselines. Supports thesis sections on neural architectures for ultra-multimodal voxel signatures and motivates or contrasts with MLP/TabNet experiments.

## 9. Limitations and open questions
- Fusion of expert KANs into a full hierarchy is not clearly implemented; reliance on notebook execution makes reproducibility harder.
- Data paths are hard-coded; runs may not cover full train/val/test due to sampling and runtime constraints.
- Calibration metrics and soft-label handling absent; comparative results vs classical baselines need consolidation.
