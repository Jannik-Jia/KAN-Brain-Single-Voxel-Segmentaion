# Notebook Verification Report

**Date**: 2025-10-11
**Notebook**: `notebook_quickstart.ipynb`
**Training Script**: `train_runner.py`

---

## ✅ Verification Summary

All features have been successfully integrated and verified. The notebook is ready to run.

---

## 🔧 Issues Fixed

### 1. Missing Parameter: `class_weight_alpha`
**Issue**: The `class_weight_alpha` parameter (default 0.5) was not included in the notebook configuration.

**Fix**: Added to cell-2:
```python
CLASS_WEIGHT_ALPHA = 0.5  # 类别权重平衡系数（0=均匀权重，1=完全反比于频率）
```

And passed to `run_single_split()` in cell-7:
```python
class_weight_alpha=CLASS_WEIGHT_ALPHA,
```

---

### 2. Incorrect File Paths: Soft Confusion Matrices
**Issue**: Notebook was looking for soft confusion matrices in `metrics/` subdirectory, but they are saved in the root `save_dir`.

**Original (Incorrect)**:
```python
val_soft_cm_path = save_dir_path / 'metrics' / 'soft_confusion_val.csv'
```

**Fixed**:
```python
val_soft_cm_path = save_dir_path / 'soft_confusion_val.csv'
```

---

### 3. Incorrect File Names: Advanced Metrics Visualizations
**Issue**: Notebook was looking for files named `reliability_diagram_val.png`, but `train_runner.py` saves them as `val_reliability_diagram.png`.

**Original (Incorrect)**:
```python
viz_files = [
    'reliability_diagram_val.png',
    'reliability_diagram_test.png',
    ...
]
```

**Fixed**:
```python
viz_files = [
    'val_reliability_diagram.png',
    'val_risk_coverage.png',
    'val_entropy_histogram.png',
    'test_reliability_diagram.png',
    'test_risk_coverage.png',
    'test_entropy_histogram.png'
]
```

---

### 4. Redundant Image Display
**Issue**: Cell-19 was displaying ALL .png files in figs/, including both 3D slices and advanced metrics visualizations (which were already shown in cell-15).

**Fix**: Updated cell-19 to only show 3D slice images:
```python
all_figs = sorted(figs_dir.glob('*slice*.png'))  # Only slice images
```

---

## ✅ Verified Features

### Data Loading
- ✅ Correct data structure expected: `data_root/1d/*_1d.npz` and `data_root/3d/*_3d.npz`
- ✅ Notebook cell-5 properly validates data directory structure
- ✅ All subject IDs extracted correctly from filenames

### Parameters
All parameters are correctly configured and passed to `run_single_split()`:
- ✅ `data_root` - Data directory path
- ✅ `val_id` - Validation subject ID (optional)
- ✅ `test_id` - Test subject ID (optional)
- ✅ `seed` - Random seed for reproducibility
- ✅ `epochs` - Number of training epochs
- ✅ `batch_size` - Batch size for training
- ✅ `lr` - Learning rate
- ✅ `weight_decay` - L2 regularization (kernel-only)
- ✅ `grad_clip_norm` - Gradient clipping norm
- ✅ `use_class_weights` - Enable class weighting
- ✅ `class_weight_alpha` - Class weight balance coefficient
- ✅ `max_vox_per_subject` - Memory control (optional)
- ✅ `save_dir` - Output directory

### Advanced Metrics Implementation
All advanced soft-label metrics are implemented and accessible:

#### Must-Have Metrics (必加)
1. ✅ **Soft ECE** (Overall + Per-class) - Expected Calibration Error for soft labels
2. ✅ **Brier Decomposition** (Murphy) - Reliability, Resolution, Uncertainty
3. ✅ **Class Mass Error** - Probability conservation per class
4. ✅ **Soft Confusion Matrix** - Probability mass transfer M[i,j] = sum(true_prob[i] * pred_prob[j])
5. ✅ **AURC** - Area Under Risk-Coverage curve for selective prediction

#### Strongly Recommended Metrics (强烈推荐)
6. ✅ **3D Probability Metrics** - 3D Soft Dice, 3D NLL, 3D Brier
7. ✅ **Entropy Statistics** - Mean, Median, Std of prediction entropy
8. ✅ **Temperature Scaling** - Fitted on validation, applied to test

### Visualizations
All visualizations use **pure English** text (no Chinese in images):
- ✅ Regular confusion matrices (Blues colormap)
- ✅ Soft confusion matrices (Viridis colormap)
- ✅ Reliability diagrams (ECE visualization)
- ✅ Risk-coverage curves (AURC visualization)
- ✅ Entropy histograms (Uncertainty distribution)
- ✅ 3D slice comparisons (True vs Predicted labels)

### Output Files
The notebook correctly displays all generated files:

**Metrics** (JSON):
- `metrics_val.json` - Validation metrics with all advanced metrics
- `metrics_test.json` - Test metrics with all advanced metrics
- `metrics_val_3d_advanced.json` - 3D-specific validation metrics
- `metrics_test_3d_advanced.json` - 3D-specific test metrics
- `temperature_scaling.json` - Temperature calibration results

**Confusion Matrices** (CSV):
- `confusion_val.csv` - Regular validation confusion matrix
- `confusion_test.csv` - Regular test confusion matrix
- `soft_confusion_val.csv` - Soft validation confusion matrix
- `soft_confusion_test.csv` - Soft test confusion matrix

**Visualizations** (PNG):
- `val_reliability_diagram.png` - Validation calibration curve
- `val_risk_coverage.png` - Validation AURC curve
- `val_entropy_histogram.png` - Validation entropy distribution
- `test_reliability_diagram.png` - Test calibration curve
- `test_risk_coverage.png` - Test AURC curve
- `test_entropy_histogram.png` - Test entropy distribution
- `{subject_id}_slice_z{idx}.png` - 3D slice visualizations

**3D Predictions** (NPZ):
- `val_{val_id}_pred_softmax_3d.npz` - Validation probability volumes
- `val_{val_id}_argmax_3d.npz` - Validation argmax labels
- `test_{test_id}_pred_softmax_3d.npz` - Test probability volumes
- `test_{test_id}_argmax_3d.npz` - Test argmax labels

**Summary Files** (JSON):
- `run_summary.json` - Complete run summary with all metrics
- `split_summary.json` - Train/Val/Test split information
- `norm_stats.json` - Normalization statistics per subject

**Model Checkpoint** (PTH):
- `checkpoints/best.pth` - Best model weights (lowest validation NLL)

---

## 📋 Notebook Structure

The notebook is organized as follows:

1. **Configuration & Import** (cells 1-3)
   - Set all hyperparameters
   - Import dependencies

2. **Data Validation** (cells 4-5)
   - Verify data directory structure
   - List available subjects

3. **Run Training** (cells 6-7)
   - Execute complete training pipeline
   - Includes all advanced metrics computation

4. **View Results** (cells 8-11)
   - Basic run summary
   - Advanced soft-label metrics display

5. **Visualize Confusion Matrices** (cells 12-17)
   - Regular confusion matrices (hard labels)
   - Soft confusion matrices (probability mass)
   - Advanced calibration metrics (reliability, AURC, entropy)

6. **3D Slice Visualization** (cells 18-19)
   - Display 3D prediction slices

7. **File Structure Overview** (cells 20-21)
   - Directory tree of all outputs

8. **Next Steps** (cell 22)
   - Suggestions for further experimentation

---

## 🚀 Usage Instructions

1. **Set data path** in cell-2:
   ```python
   DATA_ROOT = "/path/to/your/data"  # Contains 1d/ and 3d/ subdirectories
   ```

2. **Adjust hyperparameters** (optional):
   ```python
   EPOCHS = 3              # Quick test mode
   BATCH_SIZE = 256        # Adjust for your GPU
   USE_CLASS_WEIGHTS = False  # Enable for class imbalance
   ```

3. **Run all cells** sequentially

4. **View results** in `runs/quickstart/`:
   - Metrics JSON files
   - Confusion matrices
   - Visualizations
   - 3D predictions
   - Complete run summary

---

## ✅ Final Status

**All features are correctly implemented and integrated.**

The notebook:
- ✅ Includes all required parameters
- ✅ Uses correct file paths
- ✅ Displays all advanced metrics
- ✅ Shows all visualizations (pure English)
- ✅ Properly validates data structure
- ✅ Provides comprehensive results summary

**The notebook is ready for production use.**

---

## 🔬 Advanced Metrics Reference

### Soft ECE (Expected Calibration Error)
Measures calibration quality by comparing confidence vs. accuracy in bins.
- Lower is better (0 = perfect calibration)
- Computed for max-probability predictions

### Classwise ECE (Static Calibration)
Per-class one-vs-rest ECE computation.
- Evaluates calibration for each of 102 brain regions
- Averaged to get overall classwise ECE

### Brier Decomposition (Murphy)
Decomposes Brier score into interpretable components:
- **Reliability**: Calibration error (lower is better)
- **Resolution**: Sharpness of predictions (higher is better)
- **Uncertainty**: Inherent data uncertainty (constant)

### Class Mass Error
Evaluates probability conservation per class:
- Measures deviation from expected frequency
- sum_i P(class_i) should match empirical frequency

### Soft Confusion Matrix
Shows probability mass transfer between classes:
- M[i,j] = sum of (true_prob[i] * pred_prob[j])
- More informative than hard-label confusion for soft labels

### AURC (Area Under Risk-Coverage Curve)
Evaluates selective prediction quality:
- Risk = Error rate at given coverage
- Coverage = Fraction of samples retained
- Useful for uncertainty-aware applications

### 3D Soft Dice
Dice coefficient computed on probability distributions:
- Measures overlap in 3D space
- More appropriate than argmax Dice for soft labels

### Temperature Scaling
Post-hoc calibration method:
- Finds optimal temperature T on validation set
- Applies scaling: probs = softmax(logits / T)
- Improves calibration without retraining

---

**End of Report**
