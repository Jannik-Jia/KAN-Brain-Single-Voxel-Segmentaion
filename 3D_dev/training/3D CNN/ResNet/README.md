# MRI ResNet for Brain Region Classification

A PyTorch implementation of ResNet-50 optimized for MRI brain voxel classification with severe class imbalance. This system achieves ~50M parameters and is specifically designed for 7×7 patch classification of 102 brain regions from 351-channel multimodal MRI data.

## 🎯 Key Features

- **Optimized ResNet-50**: Custom architecture with ~50M parameters for brain MRI analysis
- **Class Imbalance Handling**: Advanced loss functions (Class-Balanced Focal, Logit-Adjusted CE)
- **Leave-One-Out Cross-Validation**: Complete automated training and evaluation pipeline
- **Data Augmentation**: Mixup, spatial transformations, and class-balanced sampling
- **Advanced Optimization**: Mixed precision training, EMA, gradient clipping
- **Comprehensive Analysis**: Detailed result analysis and visualization tools

## 🏗️ Architecture Overview

### ResNet-50 Specifications

```
Input: (Batch, 351, 7, 7) - 7×7 patches with 351 MRI channels
Architecture: 3-4-6-3 Bottleneck blocks with base_width=104
Spatial Flow: 7 → 7 → 4 → 2 → 1
Channel Flow: 512 → 256 → 512 → 1024 → 2048 → 102
Parameters: ~49.9M
```

### Key Design Choices

1. **Expand-First Stem**: 3×3/s1/p1 conv (351→512) preserves weak signals
2. **Wide Bottlenecks**: base_width=104 provides strong channel representation
3. **ResNet-B Downsampling**: Stride handled by 3×3 conv in bottlenecks
4. **No Max Pooling**: Preserves spatial information in small patches

## 📁 Directory Structure

```
ResNet/
├── models/                    # Core model implementations
│   ├── resnet.py             # ResNet architecture
│   ├── dataset.py            # Data loading and preprocessing
│   └── losses.py             # Imbalanced loss functions
├── scripts/                   # Training and analysis scripts
│   ├── train_mri_resnet.py   # Main training script
│   ├── run_leave_one_out.sh  # Batch training script
│   └── analyze_resnet_results.py  # Results analysis
├── configs/                   # Configuration files
│   └── default_config.json   # Default training config
├── results/                   # Training results (auto-generated)
├── docs/                      # Additional documentation
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n mri_resnet python=3.8
conda activate mri_resnet

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional requirements
pip install h5py scikit-learn matplotlib seaborn tqdm pandas
```

### 2. Data Preparation

Ensure your 3D MAT files are in the correct format:
- **Data shape**: (384, 336, 256, 351) or (351, 384, 336, 256)
- **Labels shape**: (384, 336, 256) with values 1-102
- **Mask shape**: (384, 336, 256) with boolean values
- **File naming**: `*_3d_validated.mat` or similar pattern

### 3. Single Training Run

```bash
cd ResNet/scripts

# Train ResNet for subject 1 as test set
python train_mri_resnet.py \
    --data_dir /path/to/your/mat/files \
    --test_subject 1 \
    --output_dir ./results_test \
    --epochs 50 \
    --batch_size 256 \
    --loss_type cb_focal \
    --use_mixup \
    --use_ema
```

### 4. Leave-One-Out Cross-Validation

```bash
# Edit the script to set your data directory
vim run_leave_one_out.sh
# Modify: DATA_DIR="/path/to/your/mat/files"

# Run complete Leave-One-Out training (38 subjects)
bash run_leave_one_out.sh
```

### 5. Analyze Results

```bash
# Analyze Leave-One-Out results
python analyze_resnet_results.py \
    --results_dir ./results_leave_one_out_resnet
```

## ⚙️ Configuration Options

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_width` | 104 | Width multiplier for bottleneck blocks |
| `input_channels` | 351 | Number of input feature channels |
| `num_classes` | 102 | Number of output classes |
| `patch_size` | 7 | Spatial size of input patches |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Maximum training epochs |
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `weight_decay` | 1e-4 | L2 regularization weight |
| `patience` | 15 | Early stopping patience |

### Loss Functions

| Type | Description | Best For |
|------|-------------|----------|
| `cb_focal` | Class-Balanced Focal Loss | Severe imbalance (recommended) |
| `logit_adj` | Logit-Adjusted Cross-Entropy | Long-tail distributions |
| `focal` | Standard Focal Loss | Moderate imbalance |
| `weighted_ce` | Weighted Cross-Entropy | Simple reweighting |
| `ce` | Standard Cross-Entropy | Balanced data |

### Data Augmentation

- **Mixup**: Interpolates images and labels for regularization
- **Spatial**: Random flips, rotations (90° multiples)
- **Noise**: Small amount of Gaussian noise (3% probability)
- **Class Balancing**: Balanced sampling or weighted sampling

## 🔬 Advanced Usage

### Custom Loss Function

```python
from models.losses import create_loss_function

# Create custom loss with specific parameters
loss_fn = create_loss_function(
    loss_type='cb_focal',
    class_counts=your_class_counts,
    beta=0.9999,
    gamma=1.5,
    label_smoothing=0.0
)
```

### Model Customization

```python
from models.resnet import mri_resnet50

# Create model with custom parameters
model = mri_resnet50(
    input_channels=351,
    num_classes=102,
    base_width=104,  # Adjust for different parameter counts
    zero_init_residual=False
)

# Check model information
from models.resnet import get_model_info
info = get_model_info(model)
print(f"Parameters: {info['total_parameters']:,}")
```

### Data Loading Customization

```python
from models.dataset import MRIBrain2DPatchDataset

# Create custom dataset
dataset = MRIBrain2DPatchDataset(
    mat_files=your_mat_files,
    patch_size=7,
    samples_per_subject=20000,
    balance_classes=True,
    augmentation=True,
    min_samples_per_class=100
)
```

## 📊 Expected Performance

Based on the architecture design optimized for imbalanced brain region classification:

### Hardware Requirements

- **GPU Memory**: 16GB+ recommended (A6000, V100, RTX 3090)
- **System Memory**: 64GB+ recommended
- **Storage**: 500GB+ SSD for fast data access
- **Training Time**: ~2-4 hours per subject (depending on hardware)

### Performance Targets

- **Macro F1 Score**: Target 0.75+ (varies by dataset difficulty)
- **Convergence**: Typically 30-60 epochs
- **Stability**: Low variance across subjects with proper configuration

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Reduce batch size
--batch_size 128

# Reduce samples per subject
--samples_per_subject 10000

# Disable data caching (slower but less memory)
# Modify dataset.py: cache_data=False
```

#### Slow Training

```bash
# Increase number of workers
--num_workers 8

# Use mixed precision (enabled by default)
# Ensure CUDA and cuDNN are properly installed

# Use SSD storage for data
```

#### Poor Convergence

```bash
# Try different loss functions
--loss_type logit_adj

# Adjust learning rate
--learning_rate 5e-5

# Enable/disable mixup
--use_mixup  # or remove flag to disable

# Increase training epochs
--epochs 150
```

#### Class Imbalance Issues

```bash
# Use class-balanced focal loss (recommended)
--loss_type cb_focal

# Enable balanced sampling in dataset
# Modify dataset.py: balance_classes=True

# Adjust minimum samples per class
# Modify dataset.py: min_samples_per_class=100
```

### Debugging Tips

1. **Check Data Loading**:
   ```bash
   python -c "
   from models.dataset import MRIBrain2DPatchDataset
   from pathlib import Path
   import glob
   
   files = [Path(f) for f in glob.glob('/your/path/*.mat')]
   dataset = MRIBrain2DPatchDataset(files[:1], patch_size=7)
   print(f'Dataset size: {len(dataset)}')
   print(f'Sample shape: {dataset[0][0].shape}')
   "
   ```

2. **Model Forward Pass**:
   ```bash
   python -c "
   from models.resnet import mri_resnet50
   import torch
   
   model = mri_resnet50()
   x = torch.randn(4, 351, 7, 7)
   y = model(x)
   print(f'Input: {x.shape}, Output: {y.shape}')
   "
   ```

3. **Check GPU Usage**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 📈 Results Analysis

The analysis script generates comprehensive reports including:

### Visualizations

- **F1 Score Distribution**: Histogram and box plots
- **Per-Subject Performance**: Line plots showing variance
- **Training Convergence**: Learning curves for sample subjects  
- **Per-Class Analysis**: Best and worst performing classes
- **Overfitting Analysis**: Training vs. test performance

### Output Files

- `resnet_results_summary.png`: Main visualization dashboard
- `per_class_performance.png`: Detailed class-wise analysis
- `detailed_results.csv`: Raw results for all subjects
- `summary_statistics.json`: Machine-readable summary
- `summary_report.txt`: Human-readable report

## 🔬 Architecture Comparison

### Parameter Scaling Options

| Configuration | Base Width | Parameters | Use Case |
|---------------|------------|------------|----------|
| Lightweight | 96 | ~44M | Resource constrained |
| **Recommended** | **104** | **~50M** | **Balanced performance** |
| High Capacity | 112 | ~56M | Maximum accuracy |

### vs. Other Approaches

| Method | Parameters | Input | Spatial Context |
|--------|------------|-------|-----------------|
| 1D FC (Baseline) | ~67M | (351,) | None |
| **ResNet-50** | **~50M** | **(351,7,7)** | **2D neighborhood** |
| 3D CNN | ~6M | (351,3,3,3) | Full 3D |

## 🤝 Contributing

This is a research implementation. For improvements:

1. **Model Architecture**: Experiment with different widths or depths
2. **Loss Functions**: Implement additional imbalanced learning methods
3. **Data Augmentation**: Add domain-specific augmentations
4. **Optimization**: Experiment with learning rate schedules

## 📚 References

1. **ResNet**: "Deep Residual Learning for Image Recognition" - He et al. (2016)
2. **Focal Loss**: "Focal Loss for Dense Object Detection" - Lin et al. (2017)  
3. **Class-Balanced Loss**: "Class-Balanced Loss Based on Effective Number of Samples" - Cui et al. (2019)
4. **Logit Adjustment**: "Long-tail Learning via Logit Adjustment" - Menon et al. (2020)
5. **Mixup**: "mixup: Beyond Empirical Risk Minimization" - Zhang et al. (2017)

## 📄 License

This implementation is provided for research purposes. Please cite appropriately if used in publications.

## 🔗 Related Work

- **3D CNN Baseline**: See `../train_baseline_3x3_7x7.py` for comparison
- **1D Training**: See `../B0_1D_training/` for 1D baseline implementation

---

**Note**: This README provides comprehensive usage instructions. For implementation details, see the individual Python files which contain extensive documentation and comments.