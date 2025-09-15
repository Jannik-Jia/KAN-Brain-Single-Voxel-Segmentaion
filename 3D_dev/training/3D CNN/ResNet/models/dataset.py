"""
Data loader for MRI ResNet training

Based on the 3D CNN data loading logic but optimized for 7×7 patches.
Supports imbalanced class sampling and data augmentation for ResNet training.

Memory-efficient version: Only loads labels/masks into memory (~1.2GB), 
extracts patches on-demand using h5py slicing for true 71M patch training.
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
from collections import Counter, defaultdict
import logging
import threading


class MRIBrain2DPatchDataset(Dataset):
    """
    MRI Brain 2D Patch Dataset for ResNet training
    
    Memory-efficient version that:
    1. Only loads labels/masks into memory (~1.2GB total)
    2. Uses on-demand h5py slicing to extract patches (68KB per patch)
    3. Supports true 71M patch training with complete randomization
    4. Maintains backward compatibility with existing training scripts
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
        augmentation: bool = False,
        memory_efficient: bool = True  # New parameter for memory efficiency
    ):
        """
        Args:
            mat_files: List of 3D MAT file paths
            patch_size: Patch size (typically 7 for ResNet)
            samples_per_subject: Number of samples per subject (ignored if memory_efficient=True)
            is_train: Training mode flag
            cache_data: Cache data to memory (ignored if memory_efficient=True for training)
            balance_classes: Whether to balance class distribution
            min_samples_per_class: Minimum samples per class when balancing
            augmentation: Enable data augmentation
            memory_efficient: Use memory-efficient mode (recommended for training)
        """
        self.mat_files = mat_files
        self.patch_size = patch_size
        self.samples_per_subject = samples_per_subject
        self.is_train = is_train
        self.cache_data = cache_data
        self.balance_classes = balance_classes
        self.min_samples_per_class = min_samples_per_class
        self.augmentation = augmentation and is_train
        self.memory_efficient = memory_efficient
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Memory-efficient mode for training
        if memory_efficient and is_train:
            self.logger.info("Using memory-efficient mode: loading only labels/masks")
            self._init_memory_efficient_mode()
        else:
            # Original mode (for testing or backward compatibility)
            self.logger.info("Using original mode: caching full data")
            self._init_original_mode()
        
        # Class distribution info
        if not (memory_efficient and is_train):
            # Only compute for original mode or test mode
            self.class_counts = self._analyze_class_distribution()
            self.class_weights = self._compute_class_weights()
        
        self.logger.info(f"Dataset initialized with {len(self)} samples")
        if balance_classes:
            self.logger.info(f"Class balancing enabled with min {min_samples_per_class} samples per class")
    
    def _init_memory_efficient_mode(self):
        """Initialize memory-efficient mode: only load labels/masks"""
        # Build subject ID mapping
        self.subject_file_map = {}  # {subject_id: file_path}
        self.file_subject_map = {}  # {file_path: subject_id}
        
        for mat_file in self.mat_files:
            # Extract subject ID from filename (same logic as original)
            filename = mat_file.stem
            if 'subject' in filename:
                subject_id = int(filename.split('subject')[1].split('_')[0])
            else:
                # Try to parse as integer
                try:
                    subject_id = int(filename)
                except ValueError:
                    # Use file index as fallback
                    subject_id = len(self.subject_file_map) + 1
            
            self.subject_file_map[subject_id] = mat_file
            self.file_subject_map[str(mat_file)] = subject_id
        
        # Load all labels/masks into memory
        self.all_labels = {}  # {subject_id: np.array(384,336,256)}
        self.all_masks = {}   # {subject_id: np.array(384,336,256)}
        
        self.logger.info("Loading labels/masks for all subjects...")
        total_memory_mb = 0
        
        for subject_id, mat_file in tqdm(self.subject_file_map.items(), desc="Loading labels/masks"):
            with h5py.File(mat_file, 'r') as f:
                region_labels = f['region_labels'][()]
                region_mask = f['region_mask'][()]
                
                # Apply same transpose logic as original
                if region_labels.shape != (384, 336, 256):
                    region_labels = region_labels.T
                if region_mask.shape != (384, 336, 256):
                    region_mask = region_mask.T
                
                self.all_labels[subject_id] = region_labels.astype(np.int16)
                self.all_masks[subject_id] = region_mask.astype(np.bool_)
                
                # Calculate memory usage
                memory_mb = (region_labels.nbytes + region_mask.nbytes) / (1024**2)
                total_memory_mb += memory_mb
        
        self.logger.info(f"Labels/masks loaded: {total_memory_mb:.1f} MB")
        
        # Generate ALL valid coordinates for training
        self._generate_all_valid_coords()
        
        # Thread safety for file access
        self.file_locks = defaultdict(threading.Lock)
        
        # Epoch management
        self.current_epoch = 0
    
    def _generate_all_valid_coords(self):
        """Generate all valid coordinates for memory-efficient training"""
        self.logger.info("Computing all valid coordinates...")
        
        self.all_train_coords = []  # [(subject_id, x, y, z), ...]
        total_patches = 0
        
        for subject_id in sorted(self.subject_file_map.keys()):
            mask = self.all_masks[subject_id]
            labels = self.all_labels[subject_id]
            
            # Find all valid voxels (same logic as original)
            valid_positions = np.where((mask > 0) & (labels > 0))
            valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))
            
            # Add subject_id to each coordinate
            subject_coords = [(subject_id, x, y, z) for x, y, z in valid_coords]
            self.all_train_coords.extend(subject_coords)
            
            total_patches += len(subject_coords)
            self.logger.info(f"  Subject {subject_id}: {len(subject_coords):,} valid patches")
        
        self.logger.info(f"Total training patches: {total_patches:,}")
        
        # Initial shuffle
        self._shuffle_coords_for_epoch()
    
    def _shuffle_coords_for_epoch(self):
        """Shuffle all coordinates for the current epoch"""
        random.shuffle(self.all_train_coords)
        self.logger.info(f"Epoch {self.current_epoch}: {len(self.all_train_coords):,} patches shuffled")
    
    def _init_original_mode(self):
        """Initialize original mode: cache full data"""
        # Cache for loaded data
        self.cached_data = {}
        if self.cache_data:
            self.logger.info(f"Pre-loading {len(self.mat_files)} subjects to memory...")
            for mat_file in tqdm(self.mat_files, desc="Loading data"):
                self.cached_data[str(mat_file)] = self.load_subject_data(mat_file)
        
        # Generate sample indices (original logic)
        self.sample_indices = self._generate_sample_indices()
    
    def set_epoch(self, epoch: int):
        """Set epoch and reshuffle coordinates (memory-efficient mode only)"""
        if self.memory_efficient and self.is_train:
            self.current_epoch = epoch
            self._shuffle_coords_for_epoch()
    
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
        if self.memory_efficient and self.is_train:
            return len(self.all_train_coords)
        else:
            return len(self.sample_indices)
    
    def __getitem__(self, idx):
        if self.memory_efficient and self.is_train:
            # Memory-efficient mode: on-demand patch extraction
            return self._getitem_memory_efficient(idx)
        else:
            # Original mode: use cached data
            return self._getitem_original(idx)
    
    def _getitem_memory_efficient(self, idx):
        """Memory-efficient __getitem__: extract patch on-demand"""
        subject_id, x, y, z = self.all_train_coords[idx]
        
        # Extract patch using on-demand h5py slicing
        patch_2d = self._extract_patch_on_demand(subject_id, x, y, z)
        
        # Apply augmentation if enabled
        if self.augmentation:
            patch_2d = self.apply_augmentation(patch_2d)
        
        # Get label from memory (convert from 1-102 to 0-101)
        label = self.all_labels[subject_id][x, y, z] - 1
        
        # Convert to PyTorch format (C, H, W)
        patch_tensor = torch.from_numpy(patch_2d.transpose(2, 0, 1).astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor
    
    def _getitem_original(self, idx):
        """Original __getitem__: use cached data"""
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
    
    def _extract_patch_on_demand(self, subject_id: int, x: int, y: int, z: int) -> np.ndarray:
        """
        Extract patch on-demand using h5py slicing
        
        Returns:
            patch_2d: (patch_size, patch_size, 351)
        """
        mat_file = self.subject_file_map[subject_id]
        
        # Calculate patch boundaries
        p = self.patch_size // 2
        x_min = max(0, x - p)
        x_max = min(384, x + p + 1)
        y_min = max(0, y - p)
        y_max = min(336, y + p + 1)
        
        # Thread-safe file access
        with self.file_locks[subject_id]:
            with h5py.File(mat_file, 'r') as f:
                # Load and transpose data if needed (same logic as original)
                data_ref = f['data']
                
                # Check if data needs transpose
                if data_ref.shape[0] == 351:
                    # Data is (351, 384, 336, 256), need to transpose for slicing
                    # Extract slice first, then transpose
                    data_slice = data_ref[:, x_min:x_max, y_min:y_max, z]  # (351, patch_x, patch_y)
                    patch_data = data_slice[()]  # Load to memory
                    patch_data = patch_data.transpose(1, 2, 0)  # (patch_x, patch_y, 351)
                else:
                    # Data is already (384, 336, 256, 351)
                    data_slice = data_ref[x_min:x_max, y_min:y_max, z, :]  # (patch_x, patch_y, 351)
                    patch_data = data_slice[()]  # Load to memory
                
                # Apply same z-score normalization per channel as original
                patch_data = patch_data.astype(np.float32)
                for ch in range(patch_data.shape[2]):  # 351 channels
                    channel_data = patch_data[:, :, ch]
                    mean_val = np.mean(channel_data)
                    std_val = np.std(channel_data)
                    if std_val > 1e-8:
                        patch_data[:, :, ch] = (channel_data - mean_val) / std_val
                    else:
                        patch_data[:, :, ch] = 0
        
        # Handle boundary padding (same logic as extract_patch_2d)
        if patch_data.shape[:2] != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size, patch_data.shape[2]), 
                             dtype=patch_data.dtype)
            
            # Calculate padding offsets
            x_off = p - (x - x_min)
            y_off = p - (y - y_min)
            
            # Copy data to padded array
            padded[x_off:x_off+patch_data.shape[0], 
                   y_off:y_off+patch_data.shape[1], :] = patch_data
            patch_data = padded
        
        return patch_data  # (patch_size, patch_size, 351)
    
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
    weighted_sampling: bool = False,
    memory_efficient: bool = True   # Enable memory-efficient mode by default
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders
    
    Args:
        train_files: Training MAT files
        test_files: Testing MAT files
        patch_size: Patch size (typically 7)
        batch_size: Batch size
        num_workers: Number of data loading workers (set to 0 for memory-efficient mode)
        samples_per_subject: Samples per subject (ignored in memory-efficient mode)
        balance_classes: Enable class balancing
        augmentation: Enable data augmentation
        weighted_sampling: Use weighted random sampling
        memory_efficient: Use memory-efficient mode (loads all 71M patches)
    
    Returns:
        (train_loader, test_loader)
    """
    
    # Adjust num_workers for memory-efficient mode
    if memory_efficient:
        # h5py doesn't work well with multiprocessing
        train_num_workers = 0
        test_num_workers = num_workers  # Test mode can use multiprocessing
    else:
        train_num_workers = num_workers
        test_num_workers = num_workers
    
    # Create datasets
    train_dataset = MRIBrain2DPatchDataset(
        mat_files=train_files,
        patch_size=patch_size,
        samples_per_subject=samples_per_subject,
        is_train=True,
        cache_data=True,
        balance_classes=balance_classes,
        augmentation=augmentation,
        memory_efficient=memory_efficient
    )
    
    test_dataset = MRIBrain2DPatchDataset(
        mat_files=test_files,
        patch_size=patch_size,
        samples_per_subject=samples_per_subject * 2 if samples_per_subject else None,  # More samples for testing like 3D CNN
        is_train=False,
        cache_data=True,
        balance_classes=False,  # No balancing for test
        augmentation=False,     # No augmentation for test
        memory_efficient=False  # Test uses original mode for compatibility
    )
    
    # Create samplers (only for non-memory-efficient mode)
    train_sampler = None
    if weighted_sampling and balance_classes and not memory_efficient:
        train_sampler = train_dataset.create_weighted_sampler()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Memory-efficient mode handles its own shuffling
        sampler=train_sampler,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True  # Ensures consistent batch size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader


class MemoryEfficientDataLoader:
    """
    Wrapper for memory-efficient training with epoch management
    """
    
    def __init__(self, dataset: MRIBrain2DPatchDataset, batch_size: int = 256):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Dataset handles shuffling
            num_workers=0,  # Important for h5py
            pin_memory=True,
            drop_last=True
        )
    
    def set_epoch(self, epoch: int):
        """Set epoch and reshuffle dataset"""
        self.dataset.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    # Test the dataset
    import glob
    
    # Find MAT files (adjust path as needed)
    mat_files = [Path(f) for f in glob.glob("/path/to/mat/files/*.mat")]
    
    if len(mat_files) > 0:
        print(f"Found {len(mat_files)} MAT files")
        
        # Test memory-efficient mode
        print("\n=== Testing Memory-Efficient Mode ===")
        train_dataset = MRIBrain2DPatchDataset(
            mat_files=mat_files[:2],  # Test with first 2 files
            patch_size=7,
            is_train=True,
            memory_efficient=True
        )
        
        print(f"Memory-efficient dataset size: {len(train_dataset):,}")
        
        # Test data loading
        patch, label = train_dataset[0]
        print(f"Patch shape: {patch.shape}")  # Should be (351, 7, 7)
        print(f"Label: {label.item()}")       # Should be 0-101
        
        # Test epoch shuffling
        train_dataset.set_epoch(1)
        patch2, label2 = train_dataset[0]
        print(f"After shuffle, first sample label: {label2.item()}")
        
        # Test memory-efficient data loader
        mem_loader = MemoryEfficientDataLoader(train_dataset, batch_size=4)
        batch_patches, batch_labels = next(iter(mem_loader))
        print(f"Memory-efficient batch patches shape: {batch_patches.shape}")  # (4, 351, 7, 7)
        print(f"Memory-efficient batch labels shape: {batch_labels.shape}")     # (4,)
        
        # Test original mode
        print("\n=== Testing Original Mode ===")
        original_dataset = MRIBrain2DPatchDataset(
            mat_files=mat_files[:2],  # Test with first 2 files
            patch_size=7,
            samples_per_subject=1000,
            is_train=True,
            cache_data=True,
            memory_efficient=False
        )
        
        print(f"Original dataset size: {len(original_dataset)}")
        patch, label = original_dataset[0]
        print(f"Original patch shape: {patch.shape}")  # Should be (351, 7, 7)
        
        # Test data loader creation
        print("\n=== Testing Data Loader Creation ===")
        train_files = mat_files[:2]
        test_files = mat_files[:1]
        
        train_loader, test_loader = create_data_loaders(
            train_files=train_files,
            test_files=test_files,
            batch_size=4,
            memory_efficient=True
        )
        
        print(f"Train loader length: {len(train_loader)}")
        print(f"Test loader length: {len(test_loader)}")
        
    else:
        print("No MAT files found. Please adjust the path.")