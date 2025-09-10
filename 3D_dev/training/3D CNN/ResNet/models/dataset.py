"""
Data loader for MRI ResNet training

Based on the 3D CNN data loading logic but optimized for 7×7 patches.
Supports imbalanced class sampling and data augmentation for ResNet training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import h5py
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import Counter
import logging


class MRIBrain2DPatchDataset(Dataset):
    """
    MRI Brain 2D Patch Dataset for ResNet training
    
    Extracts 7×7 patches from 3D MRI volumes with 351 channels.
    Optimized for handling class imbalance in brain region classification.
    """
    
    def __init__(
        self, 
        mat_files: List[Path], 
        patch_size: int = 7, 
        samples_per_subject: Optional[int] = None,
        is_train: bool = True,
        cache_data: bool = True,
        balance_classes: bool = False,
        min_samples_per_class: int = 50,
        augmentation: bool = False
    ):
        """
        Args:
            mat_files: List of 3D MAT file paths
            patch_size: Patch size (typically 7 for ResNet)
            samples_per_subject: Number of samples per subject (None = all valid)
            is_train: Training mode flag
            cache_data: Cache data to memory for faster access
            balance_classes: Whether to balance class distribution
            min_samples_per_class: Minimum samples per class when balancing
            augmentation: Enable data augmentation
        """
        self.mat_files = mat_files
        self.patch_size = patch_size
        self.samples_per_subject = samples_per_subject
        self.is_train = is_train
        self.cache_data = cache_data
        self.balance_classes = balance_classes
        self.min_samples_per_class = min_samples_per_class
        self.augmentation = augmentation and is_train
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for loaded data
        self.cached_data = {}
        if cache_data:
            self.logger.info(f"Pre-loading {len(mat_files)} subjects to memory...")
            for mat_file in tqdm(mat_files, desc="Loading data"):
                self.cached_data[str(mat_file)] = self.load_subject_data(mat_file)
        
        # Generate sample indices
        self.sample_indices = self._generate_sample_indices()
        
        # Class distribution info
        self.class_counts = self._analyze_class_distribution()
        self.class_weights = self._compute_class_weights()
        
        self.logger.info(f"Dataset initialized with {len(self.sample_indices)} samples")
        if balance_classes:
            self.logger.info(f"Class balancing enabled with min {min_samples_per_class} samples per class")
    
    def load_subject_data(self, mat_file: Path) -> Dict[str, np.ndarray]:
        """Load single subject 3D data - matches 3D CNN baseline exactly"""
        with h5py.File(mat_file, 'r') as f:
            # Load necessary data
            data = f['data'][()]
            region_labels = f['region_labels'][()]
            region_mask = f['region_mask'][()]
            
            # Debug print exactly like 3D CNN baseline (always print during loading)
            print(f"原始数据形状: data={data.shape}, labels={region_labels.shape}, mask={region_mask.shape}")
            
            # Handle data format exactly like 3D CNN baseline
            if data.shape[0] == 351:
                # If first dimension is 351: (351, 384, 336, 256) -> (384, 336, 256, 351)
                data = np.transpose(data, (1, 2, 3, 0))
            elif data.shape[-1] == 351:
                # If last dimension is 351, already in correct format
                pass
            else:
                raise ValueError(f"无法识别数据格式，shape: {data.shape}")
            
            # Handle labels and mask exactly like 3D CNN baseline
            if region_labels.shape != (384, 336, 256):
                region_labels = region_labels.T
            if region_mask.shape != (384, 336, 256):
                region_mask = region_mask.T
                
            # Debug print exactly like 3D CNN baseline (always print during loading)
            print(f"转置后数据形状: data={data.shape}, labels={region_labels.shape}, mask={region_mask.shape}")
            
            # 对每个patient的351个channel进行z-score标准化 - 与3D CNN baseline完全一致
            # data shape: (384, 336, 256, 351)
            data = data.astype(np.float32)
            
            # 计算每个channel的均值和标准差
            for ch in range(data.shape[3]):  # 351个channels
                channel_data = data[:, :, :, ch]
                
                # 计算当前channel的均值和标准差
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                
                # 避免除以0
                if std_val > 1e-8:
                    data[:, :, :, ch] = (channel_data - mean_val) / std_val
                else:
                    # 如果标准差为0，则将该channel设为0
                    data[:, :, :, ch] = 0
            
            print(f"已完成351维channel的z-score标准化")
            
        return {
            'data': data,
            'labels': region_labels.astype(np.int64),
            'mask': region_mask.astype(bool)
        }
    
    def _generate_sample_indices(self) -> List[Tuple[int, int, int, int]]:
        """Generate sampling indices - exactly like 3D CNN baseline"""
        indices = []
        
        for file_idx, mat_file in enumerate(self.mat_files):
            if self.cache_data:
                subject_data = self.cached_data[str(mat_file)]
            else:
                subject_data = self.load_subject_data(mat_file)
            
            mask = subject_data['mask']
            labels = subject_data['labels']
            
            # Get all valid voxel positions (label > 0 and within mask) - exactly like 3D CNN
            valid_positions = np.where((mask > 0) & (labels > 0))
            valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))
            
            # Sampling exactly like 3D CNN baseline
            if self.samples_per_subject and len(valid_coords) > self.samples_per_subject:
                if self.is_train:
                    # Training: random sampling
                    sampled_coords = random.sample(valid_coords, self.samples_per_subject)
                else:
                    # Testing: uniform sampling
                    step = len(valid_coords) // self.samples_per_subject
                    sampled_coords = valid_coords[::step][:self.samples_per_subject]
            else:
                # If valid voxels less than target, use all
                sampled_coords = valid_coords
            
            # Add to index list (file_idx, x, y, z)
            for x, y, z in sampled_coords:
                indices.append((file_idx, x, y, z))
        
        # Training: shuffle indices
        if self.is_train:
            random.shuffle(indices)
        
        return indices
    
    def _balanced_sampling(self, class_indices: Dict[int, List[Tuple]]) -> List[Tuple]:
        """Apply balanced sampling strategy"""
        balanced_indices = []
        
        # Count samples per class
        class_counts = {k: len(v) for k, v in class_indices.items() if len(v) > 0}
        
        # Determine target samples per class
        if len(class_counts) > 0:
            median_count = np.median(list(class_counts.values()))
            target_samples = max(self.min_samples_per_class, int(median_count))
            
            for class_id, coords in class_indices.items():
                if len(coords) == 0:
                    continue
                
                if len(coords) >= target_samples:
                    # Downsample majority classes
                    sampled = random.sample(coords, target_samples)
                else:
                    # Upsample minority classes (with replacement)
                    sampled = random.choices(coords, k=target_samples)
                
                balanced_indices.extend(sampled)
        
        return balanced_indices
    
    def _analyze_class_distribution(self) -> Dict[int, int]:
        """Analyze class distribution in the dataset"""
        class_counts = Counter()
        
        for file_idx, x, y, z in self.sample_indices:
            if self.cache_data:
                mat_file = str(self.mat_files[file_idx])
                labels = self.cached_data[mat_file]['labels']
            else:
                subject_data = self.load_subject_data(self.mat_files[file_idx])
                labels = subject_data['labels']
            
            label = labels[x, y, z]
            class_counts[label] += 1
        
        return dict(class_counts)
    
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for loss function"""
        if not self.class_counts:
            return torch.ones(102)
        
        # Compute inverse frequency weights
        total_samples = sum(self.class_counts.values())
        weights = []
        
        for class_id in range(1, 103):  # Classes 1-102
            count = self.class_counts.get(class_id, 1)
            weight = total_samples / (len(self.class_counts) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def extract_patch_2d(self, data: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
        """
        Extract 2D patch in xy-plane at fixed z coordinate
        
        Args:
            data: 4D array (384, 336, 256, 351)
            x, y, z: Center coordinates
        
        Returns:
            2D patch (patch_size, patch_size, 351)
        """
        p = self.patch_size // 2
        
        # Extract patch in xy-plane
        x_min = max(0, x - p)
        x_max = min(data.shape[0], x + p + 1)
        y_min = max(0, y - p)
        y_max = min(data.shape[1], y + p + 1)
        
        # Extract 2D patch (fixed z slice)
        patch = data[x_min:x_max, y_min:y_max, z, :]
        
        # Handle boundary cases with padding
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size, data.shape[3]), 
                             dtype=data.dtype)
            
            # Calculate padding offsets
            x_off = p - (x - x_min)
            y_off = p - (y - y_min)
            
            # Copy data to padded array
            padded[x_off:x_off+patch.shape[0], 
                   y_off:y_off+patch.shape[1], :] = patch
            patch = padded
        
        return patch  # (patch_size, patch_size, 351)
    
    def apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to patch
        
        Args:
            patch: 2D patch (patch_size, patch_size, 351)
        
        Returns:
            Augmented patch
        """
        if not self.augmentation:
            return patch
        
        # Random horizontal flip
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
        
        # Random vertical flip
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0).copy()
        
        # Random 90-degree rotations
        if random.random() < 0.5:
            k = random.randint(1, 3)
            patch = np.rot90(patch, k=k, axes=(0, 1)).copy()
        
        # Gaussian noise (small amount)
        if random.random() < 0.3:
            noise_std = 0.01 * np.std(patch)
            noise = np.random.normal(0, noise_std, patch.shape)
            patch = patch + noise
        
        return patch
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        file_idx, x, y, z = self.sample_indices[idx]
        
        # Get subject data - exactly like 3D CNN baseline
        if self.cache_data:
            mat_file = str(self.mat_files[file_idx])
            subject_data = self.cached_data[mat_file]
        else:
            subject_data = self.load_subject_data(self.mat_files[file_idx])
        
        # Extract 2D patch (in xy-plane, z fixed) - exactly like 3D CNN baseline
        patch_2d = self.extract_patch_2d(subject_data['data'], x, y, z)
        
        # Apply augmentation if enabled (disabled by default)
        if self.augmentation:
            patch_2d = self.apply_augmentation(patch_2d)
        
        # Get center voxel label (convert from 1-102 to 0-101) - exactly like 3D CNN baseline
        label = subject_data['labels'][x, y, z] - 1
        
        # Convert to PyTorch format (C, H, W) - exactly like 3D CNN baseline
        # patch_2d shape: (patch_size, patch_size, 351)
        patch_tensor = torch.from_numpy(patch_2d.transpose(2, 0, 1).astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for weighted loss"""
        return self.class_weights
    
    def create_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler for balanced training"""
        if not self.is_train:
            raise ValueError("Weighted sampler only available for training data")
        
        # Compute sample weights
        sample_weights = []
        for file_idx, x, y, z in self.sample_indices:
            if self.cache_data:
                mat_file = str(self.mat_files[file_idx])
                labels = self.cached_data[mat_file]['labels']
            else:
                subject_data = self.load_subject_data(self.mat_files[file_idx])
                labels = subject_data['labels']
            
            label = labels[x, y, z] - 1  # Convert to 0-indexed
            weight = self.class_weights[label].item()
            sample_weights.append(weight)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def create_data_loaders(
    train_files: List[Path],
    test_files: List[Path],
    patch_size: int = 7,
    batch_size: int = 256,
    num_workers: int = 4,
    samples_per_subject: Optional[int] = 10000,  # Match 3D CNN default
    balance_classes: bool = False,  # Start with standard sampling like 3D CNN
    augmentation: bool = False,     # Start without augmentation like 3D CNN
    weighted_sampling: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders
    
    Args:
        train_files: Training MAT files
        test_files: Testing MAT files
        patch_size: Patch size (typically 7)
        batch_size: Batch size
        num_workers: Number of data loading workers
        samples_per_subject: Samples per subject
        balance_classes: Enable class balancing
        augmentation: Enable data augmentation
        weighted_sampling: Use weighted random sampling
    
    Returns:
        (train_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = MRIBrain2DPatchDataset(
        mat_files=train_files,
        patch_size=patch_size,
        samples_per_subject=samples_per_subject,
        is_train=True,
        cache_data=True,
        balance_classes=balance_classes,
        augmentation=augmentation
    )
    
    test_dataset = MRIBrain2DPatchDataset(
        mat_files=test_files,
        patch_size=patch_size,
        samples_per_subject=samples_per_subject * 2 if samples_per_subject else None,  # More samples for testing like 3D CNN
        is_train=False,
        cache_data=True,
        balance_classes=False,  # No balancing for test
        augmentation=False      # No augmentation for test
    )
    
    # Create samplers
    train_sampler = None
    if weighted_sampling and balance_classes:
        train_sampler = train_dataset.create_weighted_sampler()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensures consistent batch size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import glob
    
    # Find MAT files (adjust path as needed)
    mat_files = [Path(f) for f in glob.glob("/path/to/mat/files/*.mat")]
    
    if len(mat_files) > 0:
        print(f"Found {len(mat_files)} MAT files")
        
        # Test dataset
        dataset = MRIBrain2DPatchDataset(
            mat_files=mat_files[:2],  # Test with first 2 files
            patch_size=7,
            samples_per_subject=1000,
            is_train=True,
            cache_data=True,
            balance_classes=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {len(dataset.class_counts)} classes")
        
        # Test data loading
        patch, label = dataset[0]
        print(f"Patch shape: {patch.shape}")  # Should be (351, 7, 7)
        print(f"Label: {label.item()}")       # Should be 0-101
        
        # Test data loader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_patches, batch_labels = next(iter(loader))
        print(f"Batch patches shape: {batch_patches.shape}")  # (4, 351, 7, 7)
        print(f"Batch labels shape: {batch_labels.shape}")     # (4,)
        
    else:
        print("No MAT files found. Please adjust the path.")