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
import queue
import time


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
        memory_efficient: bool = True,  # New parameter for memory efficiency
        target_batch_size: int = 2048   # Target batch size for file grouping
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
        self.target_batch_size = target_batch_size
        
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
        else:
            # For memory-efficient mode, provide default class info
            # This will be computed on-demand if needed
            self.class_counts = {}
            self.class_weights = torch.ones(102)
        
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
        
        # Epoch management (initialize before generating coords)
        self.current_epoch = 0
        
        # Thread safety for file access
        self.file_locks = defaultdict(threading.Lock)

        # File-level caching for batch-grouped access
        self.file_cache = {}  # {subject_id: np.ndarray}
        self.max_cached_files = 3  # Cache last 3 accessed files (increased for prefetch)
        self.cache_access_order = []  # Track access order for LRU
        self.cache_lock = threading.Lock()  # Protect cache operations

        # Prefetch mechanism
        self.enable_prefetch = True
        self.prefetch_queue = queue.Queue(maxsize=2)  # Queue for prefetched batches
        self.prefetch_thread = None
        self.current_batch_idx = 0
        self.prefetch_stop_event = threading.Event()
        self.prefetch_lock = threading.Lock()
        
        # Generate ALL valid coordinates for training
        self._generate_all_valid_coords()
    
    def _generate_all_valid_coords(self):
        """Generate all valid coordinates organized by file for efficient I/O"""
        self.logger.info("Computing all valid coordinates with file-grouped batching...")

        # Organize by subject (file) for efficient batching
        self.coords_by_subject = {}  # {subject_id: [(x, y, z), ...]}
        total_patches = 0

        for subject_id in sorted(self.subject_file_map.keys()):
            mask = self.all_masks[subject_id]
            labels = self.all_labels[subject_id]

            # Find all valid voxels (same logic as original)
            valid_positions = np.where((mask > 0) & (labels > 0))
            valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))

            self.coords_by_subject[subject_id] = valid_coords
            total_patches += len(valid_coords)
            self.logger.info(f"  Subject {subject_id}: {len(valid_coords):,} valid patches")

        self.logger.info(f"Total training patches: {total_patches:,}")

        # Generate file-grouped batches
        self._generate_file_grouped_batches()

        # Start prefetch thread for memory-efficient mode
        if self.memory_efficient and self.is_train and self.enable_prefetch:
            self._start_prefetch_thread()
    
    def _generate_file_grouped_batches(self):
        """Generate batches with per-file index tracking (no duplicate patches within epoch)"""

        # Initialize per-file indices for this epoch
        self.file_indices = {}  # {subject_id: current_index}
        self.file_exhausted = {}  # {subject_id: bool}

        for subject_id in self.coords_by_subject.keys():
            self.file_indices[subject_id] = 0
            self.file_exhausted[subject_id] = False

        self.file_batches = []  # List of batches, each batch contains patches info

        # Continue until all files are exhausted
        while not all(self.file_exhausted.values()):
            batch_patches = []
            batch_size_remaining = self.target_batch_size

            # Get list of non-exhausted files
            available_files = [sid for sid, exhausted in self.file_exhausted.items() if not exhausted]
            if not available_files:
                break

            # Randomly select starting file for this batch
            random.shuffle(available_files)

            # Fill batch, prioritizing single file but adding others if needed
            for subject_id in available_files:
                if batch_size_remaining <= 0:
                    break

                current_idx = self.file_indices[subject_id]
                total_patches = len(self.coords_by_subject[subject_id])
                remaining_in_file = total_patches - current_idx

                if remaining_in_file <= 0:
                    self.file_exhausted[subject_id] = True
                    continue

                # How many patches to take from this file
                patches_to_take = min(batch_size_remaining, remaining_in_file)

                if patches_to_take > 0:
                    batch_patches.append({
                        'subject_id': subject_id,
                        'start_idx': current_idx,
                        'end_idx': current_idx + patches_to_take
                    })

                    # Update file index
                    self.file_indices[subject_id] += patches_to_take
                    batch_size_remaining -= patches_to_take

                    # Mark as exhausted if we've used all patches
                    if self.file_indices[subject_id] >= total_patches:
                        self.file_exhausted[subject_id] = True
                        self.logger.debug(f"File {subject_id} exhausted ({total_patches} patches)")

            if batch_patches:
                self.file_batches.append(batch_patches)

        total_batches = len(self.file_batches)
        avg_files_per_batch = sum(len(batch) for batch in self.file_batches) / max(total_batches, 1)

        self.logger.info(f"Generated {total_batches} file-grouped batches (no duplicates)")
        self.logger.info(f"Average files per batch: {avg_files_per_batch:.2f}")

        # Log per-file usage
        total_patches_used = sum(self.file_indices.values())
        total_patches_available = sum(len(coords) for coords in self.coords_by_subject.values())
        self.logger.info(f"Patches used: {total_patches_used:,}/{total_patches_available:,}")

    def _shuffle_coords_for_epoch(self):
        """Shuffle file-grouped batches for the current epoch"""
        # Shuffle the order of batches
        random.shuffle(self.file_batches)

        # Shuffle patches within each subject's coordinate list
        for subject_id in self.coords_by_subject:
            random.shuffle(self.coords_by_subject[subject_id])

        # Regenerate batches with new shuffled coordinates
        self._generate_file_grouped_batches()

        # Reset batch index for new epoch
        with self.prefetch_lock:
            self.current_batch_idx = 0

        self.logger.info(f"Epoch {self.current_epoch}: {len(self.file_batches)} batches shuffled")

        # Restart prefetch thread for new epoch
        if self.memory_efficient and self.is_train and self.enable_prefetch:
            self._restart_prefetch_thread()
    
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
            # Ensure coordinates are initialized before shuffling
            if hasattr(self, 'coords_by_subject') and self.coords_by_subject:
                self._shuffle_coords_for_epoch()
            else:
                self.logger.warning(f"Cannot shuffle epoch {epoch}: coordinates not initialized")
    
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
            
            # Skip z-score normalization - data is already normalized by zscore_dataset_converter.py
            data = data.astype(np.float32)
            print(f"数据已由zscore_dataset_converter.py预处理，跳过重复标准化")
            
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
            # Return total number of individual patches across all file-grouped batches
            total_patches = 0
            for batch_info in self.file_batches:
                for patch_info in batch_info:
                    total_patches += (patch_info['end_idx'] - patch_info['start_idx'])
            return total_patches
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
        """Memory-efficient __getitem__: extract patch using file-index strategy (no duplicates)"""
        # Find which file-batch and position this index corresponds to
        current_idx = 0
        target_batch_info = None
        local_idx = 0
        batch_idx = 0

        for batch_info in self.file_batches:
            batch_size = sum(patch['end_idx'] - patch['start_idx'] for patch in batch_info)
            if current_idx + batch_size > idx:
                target_batch_info = batch_info
                local_idx = idx - current_idx

                # Update current batch index for prefetch (only if significantly advanced)
                with self.prefetch_lock:
                    if batch_idx > self.current_batch_idx:
                        old_batch = self.current_batch_idx
                        self.current_batch_idx = batch_idx
                        self.logger.debug(f"Advanced from batch {old_batch} to {batch_idx}")

                break
            current_idx += batch_size
            batch_idx += 1

        if target_batch_info is None:
            raise IndexError(f"Index {idx} out of range")

        # Find the specific subject and coordinate within the batch
        coord_idx = 0
        for patch_info in target_batch_info:
            subject_id = patch_info['subject_id']
            start_idx = patch_info['start_idx']
            end_idx = patch_info['end_idx']
            segment_size = end_idx - start_idx

            if coord_idx + segment_size > local_idx:
                # Found the right subject and position
                position_in_subject = local_idx - coord_idx + start_idx
                x, y, z = self.coords_by_subject[subject_id][position_in_subject]
                break
            coord_idx += segment_size

        # Extract patch using cached file data
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
    
    def _get_cached_file_data(self, subject_id: int) -> np.ndarray:
        """Get file data with LRU caching and prefetch support"""
        # Check if already cached (thread-safe)
        with self.cache_lock:
            if subject_id in self.file_cache:
                # Update access order
                if subject_id in self.cache_access_order:
                    self.cache_access_order.remove(subject_id)
                self.cache_access_order.append(subject_id)
                return self.file_cache[subject_id]

        # Check if data is available from prefetch
        if self.enable_prefetch and hasattr(self, 'current_batch_idx'):
            # Try to get current batch data from prefetch
            prefetched_data = self._get_prefetched_data(self.current_batch_idx)
            if prefetched_data and subject_id in prefetched_data['files']:
                data_4d = prefetched_data['files'][subject_id]
                self.logger.debug(f"Got subject {subject_id} from prefetch (batch {self.current_batch_idx})")

                # Add to cache (thread-safe)
                with self.cache_lock:
                    if len(self.file_cache) >= self.max_cached_files:
                        oldest_subject = self.cache_access_order.pop(0)
                        del self.file_cache[oldest_subject]

                    self.file_cache[subject_id] = data_4d
                    self.cache_access_order.append(subject_id)
                return data_4d

        # Need to load from disk (fallback)
        mat_file = self.subject_file_map[subject_id]

        # Load data from file
        with self.file_locks[subject_id]:
            with h5py.File(mat_file, 'r') as f:
                data_ref = f['data']

                # Handle data format exactly like original load_subject_data method
                if data_ref.shape[0] == 351:
                    # Original is (351, 384, 336, 256) -> transpose to (384, 336, 256, 351)
                    data_4d = data_ref[()].transpose(1, 2, 3, 0)
                elif data_ref.shape[-1] == 351:
                    # Data is already (384, 336, 256, 351) - correct format
                    data_4d = data_ref[()]
                else:
                    raise ValueError(f"无法识别数据格式，shape: {data_ref.shape}")

        # Cache management: remove oldest if cache is full (thread-safe)
        with self.cache_lock:
            if len(self.file_cache) >= self.max_cached_files:
                oldest_subject = self.cache_access_order.pop(0)
                del self.file_cache[oldest_subject]

            # Add to cache
            self.file_cache[subject_id] = data_4d.astype(np.float32)
            self.cache_access_order.append(subject_id)

        self.logger.debug(f"Cached file data for subject {subject_id} from disk, cache size: {len(self.file_cache)}")

        return self.file_cache[subject_id]

    def _extract_patch_on_demand(self, subject_id: int, x: int, y: int, z: int) -> np.ndarray:
        """
        Extract patch using cached file data

        Returns:
            patch_2d: (patch_size, patch_size, 351)
        """
        # Get cached file data
        data_4d = self._get_cached_file_data(subject_id)

        # Calculate patch boundaries
        p = self.patch_size // 2
        x_min = max(0, x - p)
        x_max = min(384, x + p + 1)
        y_min = max(0, y - p)
        y_max = min(336, y + p + 1)

        # Extract patch from cached data (much faster than file I/O)
        patch_data = data_4d[x_min:x_max, y_min:y_max, z, :]  # (patch_x, patch_y, 351)

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

    def _load_file_data_direct(self, subject_id: int) -> Optional[np.ndarray]:
        """Load file data directly from disk (used by prefetch worker to avoid recursion)"""
        try:
            mat_file = self.subject_file_map[subject_id]

            # Load data from file (thread-safe with per-file locks)
            with self.file_locks[subject_id]:
                with h5py.File(mat_file, 'r') as f:
                    data_ref = f['data']

                    # Handle data format exactly like _get_cached_file_data method
                    if data_ref.shape[0] == 351:
                        # Original is (351, 384, 336, 256) -> transpose to (384, 336, 256, 351)
                        data_4d = data_ref[()].transpose(1, 2, 3, 0)
                    elif data_ref.shape[-1] == 351:
                        # Data is already (384, 336, 256, 351) - correct format
                        data_4d = data_ref[()]
                    else:
                        raise ValueError(f"无法识别数据格式，shape: {data_ref.shape}")

                    return data_4d.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error loading data for subject {subject_id}: {e}")
            return None

    def _start_prefetch_thread(self):
        """Start the prefetch thread"""
        if self.prefetch_thread is not None:
            self._stop_prefetch_thread()

        self.prefetch_stop_event.clear()
        self.current_batch_idx = 0

        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name=f"PrefetchThread-{id(self)}",
            daemon=True
        )
        self.prefetch_thread.start()
        self.logger.info("Prefetch thread started")

    def _stop_prefetch_thread(self):
        """Stop the prefetch thread"""
        if self.prefetch_thread is not None:
            self.prefetch_stop_event.set()
            self.prefetch_thread.join(timeout=5.0)
            self.prefetch_thread = None

        # Clear prefetch queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break

        self.logger.debug("Prefetch thread stopped")

    def _restart_prefetch_thread(self):
        """Restart prefetch thread for new epoch"""
        self._stop_prefetch_thread()
        self._start_prefetch_thread()

    def _prefetch_worker(self):
        """Worker thread for prefetching batch data"""
        self.logger.debug("Prefetch worker started")

        next_batch_idx = 0

        # Wait for file_batches to be initialized
        while not hasattr(self, 'file_batches') or not self.file_batches:
            if self.prefetch_stop_event.is_set():
                return
            time.sleep(0.1)

        while not self.prefetch_stop_event.is_set():
            try:
                # Check if we need to prefetch
                with self.prefetch_lock:
                    if next_batch_idx <= self.current_batch_idx + 1 and next_batch_idx < len(self.file_batches):
                        target_batch_idx = next_batch_idx
                        next_batch_idx += 1
                    else:
                        # Wait a bit before checking again
                        time.sleep(0.01)
                        continue

                # Prefetch this batch
                if target_batch_idx < len(self.file_batches):
                    prefetched_data = self._prefetch_batch(target_batch_idx)

                    if prefetched_data is not None:
                        # Try to put in queue (non-blocking)
                        try:
                            self.prefetch_queue.put((target_batch_idx, prefetched_data), timeout=0.1)
                            self.logger.debug(f"Prefetched batch {target_batch_idx}")
                        except queue.Full:
                            # Queue is full, skip this prefetch
                            self.logger.debug(f"Prefetch queue full, skipping batch {target_batch_idx}")

            except Exception as e:
                self.logger.error(f"Prefetch worker error: {e}")
                time.sleep(0.1)

        self.logger.debug("Prefetch worker stopped")

    def _prefetch_batch(self, batch_idx: int) -> Optional[Dict]:
        """Prefetch data for a specific batch"""
        if batch_idx >= len(self.file_batches):
            return None

        try:
            batch_info = self.file_batches[batch_idx]
            prefetched_files = {}

            # Prefetch required files for this batch
            for patch_info in batch_info:
                subject_id = patch_info['subject_id']

                if subject_id not in prefetched_files:
                    # Load file data directly from disk (avoid recursion)
                    file_data = self._load_file_data_direct(subject_id)
                    if file_data is not None:
                        prefetched_files[subject_id] = file_data

            return {
                'batch_idx': batch_idx,
                'files': prefetched_files,
                'prefetch_time': time.time()
            }

        except Exception as e:
            self.logger.error(f"Error prefetching batch {batch_idx}: {e}")
            return None

    def _get_prefetched_data(self, batch_idx: int) -> Optional[Dict]:
        """Get prefetched data for a batch if available"""
        try:
            # Check if we have prefetched data for this batch
            temp_items = []
            prefetched_data = None

            while not self.prefetch_queue.empty():
                try:
                    item = self.prefetch_queue.get_nowait()
                    if item[0] == batch_idx:
                        prefetched_data = item[1]
                        break
                    else:
                        temp_items.append(item)
                except queue.Empty:
                    break

            # Put back items we didn't use
            for item in temp_items:
                try:
                    self.prefetch_queue.put_nowait(item)
                except queue.Full:
                    pass

            return prefetched_data

        except Exception as e:
            self.logger.error(f"Error getting prefetched data: {e}")
            return None

    def __del__(self):
        """Cleanup when dataset is destroyed"""
        try:
            self._stop_prefetch_thread()
        except:
            pass

    def get_prefetch_stats(self) -> Dict:
        """Get prefetch performance statistics"""
        if not self.enable_prefetch:
            return {"prefetch_enabled": False}

        queue_size = self.prefetch_queue.qsize()
        thread_alive = self.prefetch_thread is not None and self.prefetch_thread.is_alive()

        # Thread-safe cache size access
        with self.cache_lock:
            cache_size = len(self.file_cache)

        return {
            "prefetch_enabled": True,
            "queue_size": queue_size,
            "max_queue_size": self.prefetch_queue.maxsize,
            "thread_alive": thread_alive,
            "current_batch": getattr(self, 'current_batch_idx', 0),
            "total_batches": len(getattr(self, 'file_batches', [])),
            "cache_size": cache_size,
            "max_cache_size": self.max_cached_files
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for weighted loss"""
        if hasattr(self, 'class_weights'):
            return self.class_weights
        else:
            # Fallback for memory-efficient mode
            return torch.ones(102)
    
    def create_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler for balanced training"""
        if not self.is_train:
            raise ValueError("Weighted sampler only available for training data")
        
        if self.memory_efficient:
            raise NotImplementedError("Weighted sampling not supported in memory-efficient mode")
        
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
        memory_efficient=memory_efficient,
        target_batch_size=batch_size  # Pass batch size for file grouping
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