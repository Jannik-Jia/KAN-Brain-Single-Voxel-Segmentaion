# MRI ResNet Usage Guide

Quick reference guide for training and using the MRI ResNet system.

## 📋 Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU with 16GB+ memory
- 64GB+ system RAM (recommended)
- SSD storage for fast data access

## 🏃 Quick Start (5 Minutes)

### 1. Test Model Creation

```bash
cd ResNet/scripts
python -c "
from models.resnet import mri_resnet50, get_model_info
model = mri_resnet50()
info = get_model_info(model)
print(f'Model created successfully!')
print(f'Parameters: {info[\"total_parameters\"]:,}')
print(f'Model size: {info[\"parameter_size_mb\"]:.2f} MB')
"
```

Expected output:
```
Model created successfully!
Parameters: 49,913,606
Model size: 190.38 MB
```

### 2. Test Data Loading

```bash
# Using the pre-configured data path
DATA_DIR="/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"

python -c "
import glob
from pathlib import Path
files = [Path(f) for f in glob.glob('$DATA_DIR/*.mat')]
print(f'Found {len(files)} MAT files')
if files:
    print(f'Example file: {files[0]}')
else:
    print('No MAT files found - check your data path')
"
```

### 3. Quick Single Training

```bash
# 5-epoch test run
python train_mri_resnet.py \
    --data_dir /path/to/your/mat/files \
    --test_subject 1 \
    --output_dir ./quick_test \
    --epochs 5 \
    --batch_size 128 \
    --samples_per_subject 1000 \
    --verbose
```

## 🎯 Standard Workflows

### Workflow 1: Single Subject Training

For testing or debugging:

```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 38 \
    --output_dir ./single_test \
    --epochs 50 \
    --batch_size 256 \
    --loss_type cb_focal \
    --use_mixup \
    --use_ema
```

### Workflow 2: Full Leave-One-Out

For complete evaluation:

```bash
# 1. Edit configuration
vim run_leave_one_out.sh
# Set: DATA_DIR="/path/to/your/mat/files"

# 2. Run training (takes several hours)
bash run_leave_one_out.sh

# 3. Analyze results
python analyze_resnet_results.py \
    --results_dir ./results_leave_one_out_resnet
```

### Workflow 3: Parameter Exploration

Test different configurations:

```bash
# Test different loss functions
for loss in cb_focal logit_adj focal weighted_ce; do
    python train_mri_resnet.py \
        --data_dir /path/to/data \
        --test_subject 1 \
        --output_dir ./loss_test_$loss \
        --loss_type $loss \
        --epochs 20
done

# Test different model sizes
for width in 96 104 112; do
    python train_mri_resnet.py \
        --data_dir /path/to/data \
        --test_subject 1 \
        --output_dir ./width_test_$width \
        --base_width $width \
        --epochs 20
done
```

## 🔧 Common Configurations

### High Performance Setup

For maximum accuracy (requires powerful hardware):

```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --base_width 112 \
    --batch_size 512 \
    --epochs 150 \
    --learning_rate 5e-5 \
    --loss_type cb_focal \
    --use_mixup \
    --use_ema \
    --num_workers 8
```

### Memory-Efficient Setup

For limited GPU memory:

```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --base_width 96 \
    --batch_size 128 \
    --samples_per_subject 10000 \
    --num_workers 2
```

### Fast Debugging Setup

For quick testing:

```bash
python train_mri_resnet.py \
    --data_dir /path/to/data \
    --test_subject 1 \
    --output_dir ./debug \
    --epochs 3 \
    --batch_size 64 \
    --samples_per_subject 500 \
    --patience 2
```

## 📊 Understanding Results

### Key Metrics

| Metric | Good | Excellent | Description |
|--------|------|-----------|-------------|
| Macro F1 | >0.70 | >0.80 | Main evaluation metric |
| Training Time | <3h | <2h | Per subject (depends on hardware) |
| Convergence Epoch | <60 | <40 | Early convergence is better |

### Result Files

After training, check these files:

```bash
results/
├── resnet_test_subject_1/
│   ├── training_results.json    # Key metrics
│   ├── best_model.pth          # Best model weights
│   ├── training_history.png    # Training curves
│   └── training.log            # Detailed logs
```

Important values in `training_results.json`:
```json
{
  "best_f1": 0.7845,        # Best macro F1 score
  "final_f1": 0.7823,       # Final F1 score  
  "best_epoch": 42          # Epoch of best performance
}
```

## ⚠️ Troubleshooting

### Issue: "CUDA out of memory"

```bash
# Solution 1: Reduce batch size
--batch_size 128

# Solution 2: Reduce samples
--samples_per_subject 8000

# Solution 3: Use smaller model
--base_width 96
```

### Issue: "No MAT files found"

```bash
# Check file pattern
ls /path/to/data/*.mat

# Try different patterns
--data_dir /path/to/data  # Should contain *.mat files
```

### Issue: Training very slow

```bash
# Check GPU usage
nvidia-smi

# Increase workers if CPU allows
--num_workers 6

# Check data location (should be on SSD)
```

### Issue: Poor F1 scores

```bash
# Try different loss function
--loss_type logit_adj

# Reduce learning rate
--learning_rate 5e-5

# Increase training time
--epochs 120
```

## 📈 Monitoring Training

### Real-time Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f results/resnet_test_subject_1/training.log

# Check intermediate results
cat results/resnet_test_subject_1/training_results.json
```

### Early Stopping Indicators

Good training should show:
- Decreasing loss in first 10 epochs
- F1 score improving steadily
- Test F1 not far behind training F1

Stop training if:
- Loss stops decreasing after 20 epochs
- F1 score plateaus early and doesn't improve
- Large gap between train and test F1 (overfitting)

## 🎛️ Advanced Options

### Custom Data Augmentation

Edit `models/dataset.py`:

```python
def apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
    # Add your custom augmentations here
    if random.random() < 0.3:
        # Custom noise
        noise_std = 0.02 * np.std(patch)
        patch = patch + np.random.normal(0, noise_std, patch.shape)
    
    return patch
```

### Custom Loss Function

```python
from models.losses import create_loss_function

# Create loss with custom parameters
loss_fn = create_loss_function(
    'cb_focal',
    class_counts=your_class_counts,
    beta=0.99999,    # Higher beta = more reweighting
    gamma=2.0,       # Higher gamma = more focus on hard examples
    label_smoothing=0.1
)
```

### Hyperparameter Optimization

Use tools like Optuna for automated hyperparameter search:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    base_width = trial.suggest_int('base_width', 96, 120, step=8)
    
    # Run training with these parameters
    # Return validation F1 score
    return f1_score
```

## 📋 Checklist for Production Use

Before running full Leave-One-Out:

- [ ] Test single subject training works
- [ ] Verify data loading with your files  
- [ ] Check GPU memory usage with your batch size
- [ ] Confirm output directory has sufficient space (~500GB)
- [ ] Test analysis script works
- [ ] Set up monitoring/logging
- [ ] Plan for ~2-3 days total training time

## 💡 Tips for Best Results

1. **Start Small**: Test with 1-2 subjects first
2. **Monitor Closely**: Watch first few subjects for issues
3. **Save Frequently**: Training can take days
4. **Use SSD Storage**: Much faster data loading
5. **Balance Classes**: Enable class balancing for better minority class performance
6. **Ensemble Results**: Consider averaging predictions from multiple runs

---

For detailed technical information, see the main [README.md](../README.md).
For implementation details, check the source code comments.