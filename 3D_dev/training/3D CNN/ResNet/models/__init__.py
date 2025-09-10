"""
MRI ResNet Models Package

This package contains the core components for MRI brain region classification:
- ResNet-50 architecture optimized for 7×7 patches
- Advanced loss functions for imbalanced classification
- Data loaders with class balancing and augmentation
"""

from .resnet import (
    mri_resnet50,
    MRIResNet, 
    Bottleneck,
    count_parameters,
    get_model_info
)

from .losses import (
    FocalLoss,
    ClassBalancedFocalLoss, 
    LogitAdjustedCrossEntropy,
    BalancedSoftmaxLoss,
    MixupLoss,
    create_loss_function,
    mixup_data
)

from .dataset import (
    MRIBrain2DPatchDataset,
    create_data_loaders
)

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"

__all__ = [
    # ResNet models
    'mri_resnet50',
    'MRIResNet',
    'Bottleneck', 
    'count_parameters',
    'get_model_info',
    
    # Loss functions
    'FocalLoss',
    'ClassBalancedFocalLoss',
    'LogitAdjustedCrossEntropy', 
    'BalancedSoftmaxLoss',
    'MixupLoss',
    'create_loss_function',
    'mixup_data',
    
    # Dataset
    'MRIBrain2DPatchDataset',
    'create_data_loaders'
]