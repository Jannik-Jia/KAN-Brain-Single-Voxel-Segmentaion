#!/usr/bin/env python3
"""
Dataset Conversion Tool with Patient-wise Z-score Normalization and Patch-friendly Format

This tool creates a new dataset directory with:
1. Patient-wise z-score normalization for all 351 dimensions
2. Patch-friendly HDF5 compression optimized for 7x7 patch extraction
3. Optimized data layout for efficient patch reading

Author: Claude Code
"""

import numpy as np
import h5py
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
import logging
import sys
import traceback
import psutil
import os
from datetime import datetime
import gc

class ZScoreDatasetConverter:
    """
    Dataset converter with patient-wise z-score normalization and patch-friendly format
    """

    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 log_level: str = 'INFO',
                 chunk_size: Tuple[int, ...] = (32, 32, 1, 351)):
        self.input_dir = Path(input_dir)

        # Create output directory name based on input directory
        input_name = self.input_dir.name
        if not output_dir:
            # If no output_dir specified, create zscore version in parent directory
            self.output_dir = self.input_dir.parent / f"{input_name}_zscore_normalized"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging FIRST - before any self.logger usage
        self.setup_logging(log_level)

        # HDF5 chunking strategy optimized for 7×7×1 patch extraction
        self.chunk_size = chunk_size

        self.logger.info(f"HDF5 chunk size optimized for 7×7×1 patch extraction: {self.chunk_size}")

        # Results tracking
        self.conversion_results = []
        self.failed_subjects = []

        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        self.logger.info(f"Z-Score Dataset Converter initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Chunk size for HDF5: {self.chunk_size}")
        self.logger.info(f"Initial memory usage: {self.initial_memory:.1f} MB")

    def setup_logging(self, log_level: str):
        """Setup logging system"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"zscore_conversion_{timestamp}.log"

        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Logging system started, log file: {log_file}")

    def log_memory_usage(self, context: str = ""):
        """Log memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.initial_memory
        self.logger.debug(f"Memory usage {context}: {current_memory:.1f} MB (delta: {memory_delta:+.1f} MB)")

        if current_memory > 8000:  # 8GB warning
            self.logger.warning(f"High memory usage: {current_memory:.1f} MB")

    def load_original_data(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """Load original MAT file data - supports both 1D and 3D formats"""
        self.logger.debug(f"Loading original data: {mat_path}")
        start_time = time.time()

        try:
            data = {}
            with h5py.File(mat_path, "r") as f:
                available_keys = list(f.keys())
                self.logger.debug(f"File keys: {available_keys}")

                # Determine data format based on available keys
                has_multidim_data = 'multidim_data' in available_keys  # 1D format
                has_3d_data = 'data' in available_keys                # 3D format

                if has_3d_data and not has_multidim_data:
                    # This is 3D validated format - convert back to "1D-like" format for processing
                    self.logger.info("Detected 3D validated format - converting to processable format")

                    # Load 3D data
                    data_4d = f['data'][()]        # (384, 336, 256, 351) or transposed
                    region_mask = f['region_mask'][()]  # (384, 336, 256)
                    region_labels = f['region_labels'][()]  # (384, 336, 256)

                    # Handle transposition if needed
                    if data_4d.shape[0] == 351:
                        data_4d = data_4d.transpose(1, 2, 3, 0)  # → (384, 336, 256, 351)
                        self.logger.debug(f"Transposed data from {f['data'].shape} to {data_4d.shape}")

                    # Extract valid brain tissue voxels
                    region_bool = region_mask.astype(bool)
                    n_voxels = np.sum(region_bool)

                    # Convert to 1D-like format for z-score processing
                    data['region'] = region_mask.astype(np.uint8)
                    data['multidim_data'] = data_4d[region_bool]  # (n_voxels, 351)

                    # Create seg_one_hot from region_labels (convert 1-102 to one-hot)
                    label_values = region_labels[region_bool]  # (n_voxels,)
                    seg_one_hot = np.zeros((102, n_voxels), dtype=np.uint8)

                    # Handle label conversion: 1-102 → one-hot encoding
                    valid_labels = (label_values >= 1) & (label_values <= 102)
                    valid_indices = label_values[valid_labels] - 1  # Convert to 0-101
                    valid_voxel_indices = np.where(valid_labels)[0]
                    seg_one_hot[valid_indices, valid_voxel_indices] = 1

                    data['seg_one_hot'] = seg_one_hot

                    # Copy other data
                    data['big_seg'] = f.get('big_seg', region_mask)  # Use region_mask if big_seg not available

                    # Create region_seg from region_labels
                    data['region_seg'] = label_values

                    self.logger.info(f"Converted 3D format: {n_voxels} brain voxels, shape {data_4d.shape}")

                elif has_multidim_data:
                    # Original 1D format
                    self.logger.info("Detected original 1D format")

                    for k in f.keys():
                        if not k.startswith("#"):
                            v = f[k][()]

                            # Apply same transformations as current dataset
                            if k == 'multidim_data' and v.shape[0] == 351:
                                v = v.T  # Transpose feature matrix
                            elif k == 'region_seg':
                                v = v.flatten()  # Flatten

                            data[k] = v
                            self.logger.debug(f"Loaded {k}: {v.shape}")
                else:
                    raise ValueError(f"Unrecognized data format. Available keys: {available_keys}")

            load_time = time.time() - start_time
            self.logger.info(f"Data loaded successfully: {mat_path.name} (time: {load_time:.2f}s)")

            return data

        except Exception as e:
            self.logger.error(f"Failed to load {mat_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def convert_to_3d_with_zscore(self, data: Dict[str, np.ndarray], subject_id: str) -> Dict[str, np.ndarray]:
        """
        Convert 1D data to 3D format with patient-wise z-score normalization

        Key improvements:
        1. Patient-wise z-score normalization for all 351 dimensions
        2. Optimized data layout for patch extraction
        """
        self.logger.debug(f"Converting {subject_id} to 3D with z-score normalization")
        start_time = time.time()

        try:
            region = data['region'].astype(bool)
            n_voxels = np.sum(region)
            self.logger.debug(f"ROI voxels: {n_voxels}")

            output_data = {}

            # 1. 3D volume data (unchanged)
            output_data['big_seg'] = data['big_seg'].copy()
            output_data['region_mask'] = data['region'].astype(np.uint8)
            self.logger.debug(f"3D volume shapes: big_seg={data['big_seg'].shape}, region={data['region'].shape}")

            # 2. Multi-modal feature data with PATIENT-WISE Z-SCORE NORMALIZATION
            features = data['multidim_data']  # (n_voxels, 351)
            n_voxels_check, n_features = features.shape

            if n_voxels != n_voxels_check:
                self.logger.warning(f"Voxel count mismatch: region={n_voxels}, features={n_voxels_check}")

            # === PATIENT-WISE Z-SCORE NORMALIZATION (ONLY FOR BRAIN TISSUE VOXELS) ===
            self.logger.info(f"Applying patient-wise z-score normalization for {n_features} dimensions")
            self.logger.info(f"Normalization will be computed ONLY on {n_voxels} brain tissue voxels (excluding background)")

            # 🚀 VECTORIZED Z-SCORE NORMALIZATION (更高效且安全)
            # features shape: (n_voxels, 351) - 所有都是脑组织体素

            # 计算每个维度的均值和标准差 (向量化操作)
            mu = features.mean(axis=0, keepdims=True)      # shape: (1, 351)
            sigma = features.std(axis=0, keepdims=True)    # shape: (1, 351)

            # 安全性检查：避免除零错误
            epsilon = 1e-8
            safe_sigma = np.where(sigma > epsilon, sigma, 1.0)  # 将接近0的std替换为1.0
            zero_std_mask = sigma <= epsilon                     # 记录哪些维度std接近0

            # 向量化z-score标准化
            features_normalized = ((features - mu) / safe_sigma).astype(np.float32, copy=False)

            # 对于std接近0的维度，设置为0（保持原有逻辑）
            if np.any(zero_std_mask):
                zero_dims = np.where(zero_std_mask.squeeze())[0]
                features_normalized[:, zero_dims] = 0.0
                self.logger.debug(f"Dimensions with std≈0 (set to zero): {zero_dims[:10]}...")  # 只显示前10个

            # 日志记录前几个维度的统计信息
            for ch in range(min(5, n_features)):
                mean_val = mu[0, ch]
                std_val = sigma[0, ch]
                self.logger.debug(f"Dim {ch}: mean={mean_val:.6f}, std={std_val:.6f} (brain tissue only)")

            # 验证向量化结果的正确性（可选的完整性检查）
            if self.logger.level <= 10:  # DEBUG级别时才做验证
                sample_ch = 0
                expected = (features[:, sample_ch] - mu[0, sample_ch]) / safe_sigma[0, sample_ch]
                actual = features_normalized[:, sample_ch]
                if zero_std_mask[0, sample_ch]:
                    expected[:] = 0.0  # 如果std≈0，期望值应该是0
                assert np.allclose(expected, actual, atol=1e-6), "向量化实现验证失败"
                self.logger.debug("✅ 向量化z-score实现验证通过")

            self.logger.info(f"✅ Z-score normalization completed for all {n_features} dimensions (vectorized)")
            self.logger.info(f"✅ Normalization computed on brain tissue voxels only (background excluded)")
            self.logger.info(f"✅ Dimensions with constant values (std≈0): {np.sum(zero_std_mask)}/{n_features}")

            # Convert to 4D: (n_voxels, 351) → (384, 336, 256, 351)
            self.logger.debug(f"Converting features to 4D: {features_normalized.shape} → 4D")
            data_4d = np.zeros((*region.shape, n_features), dtype=np.float32)
            data_4d[region] = features_normalized
            output_data['data'] = data_4d
            self.logger.debug(f"Features converted to 4D: {data_4d.shape}")

            # 3. One-Hot labels conversion: (102, n_voxels) → (384, 336, 256)
            seg_one_hot = data['seg_one_hot']
            self.logger.debug(f"One-hot labels shape: {seg_one_hot.shape}")

            # 🔧 CRITICAL FIX: Convert from 0-101 to 1-102 to match original dataset format
            # argmax gives 0-101, but original dataset uses 1-102
            labels_1d = np.argmax(seg_one_hot, axis=0).astype(np.uint8) + 1  # +1 to get 1-102

            labels_3d = np.zeros(region.shape, dtype=np.uint8)
            labels_3d[region] = labels_1d
            output_data['region_labels'] = labels_3d

            self.logger.debug(f"Label range after conversion: {np.min(labels_1d)}-{np.max(labels_1d)} (should be 1-102)")

            # 4. Original labels reconstruction: (n_voxels,) → (384, 336, 256)
            region_seg = data['region_seg']
            region_seg_3d = np.zeros(region.shape, dtype=region_seg.dtype)
            region_seg_3d[region] = region_seg
            output_data['region_seg_3d'] = region_seg_3d

            # 5. Subject index volume
            output_data['subject_metadata'] = {
                'subject_id': subject_id,
                'n_voxels': n_voxels,
                'n_features': n_features,
                'normalization': 'patient_wise_zscore',
                'conversion_timestamp': datetime.now().isoformat()
            }

            convert_time = time.time() - start_time
            self.logger.info(f"Subject {subject_id} converted with z-score normalization (time: {convert_time:.2f}s)")
            self.log_memory_usage(f"after converting {subject_id}")

            return output_data

        except Exception as e:
            self.logger.error(f"Failed to convert subject {subject_id}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def save_patch_friendly_format(self, data_3d: Dict[str, np.ndarray], output_path: Path):
        """
        Save data in patch-friendly HDF5 format

        Optimizations for 7×7×1 (2D) patch extraction:
        1. Data chunking: (32, 32, 1, 351) - optimal for 2D patch access
        2. Labels/masks chunking: (64, 64, 16) - prevents frequent decompression
        3. Z-dimension chunking = 1: perfect for single-slice 2D patch reading
        4. Channel chunking = 351: keeps all features together for each patch
        5. LZF compression: fast decompression for all data types
        6. No fletcher32: eliminates checksum overhead for 10-20% speed boost
        """
        self.logger.debug(f"Saving patch-friendly format: {output_path}")
        start_time = time.time()

        try:
            with h5py.File(output_path, 'w') as f:
                for key, value in data_3d.items():
                    if key == 'subject_metadata':
                        # Save metadata as attributes
                        for meta_key, meta_value in value.items():
                            f.attrs[f'meta_{meta_key}'] = str(meta_value)
                        continue

                    original_shape = value.shape

                    if key == 'data' and value.ndim == 4:
                        # === 2D PATCH-FRIENDLY DATA STORAGE (7×7×1) ===
                        # Keep data in (384, 336, 256, 351) format for direct patch extraction
                        save_value = value.astype(np.float32)

                        # Use chunking optimized for 7×7×1 (2D) patch extraction
                        # Chunk size: (32, 32, 1, 351) designed for efficient 2D patch reading
                        chunk_shape = self.chunk_size

                        # Create dataset with optimized chunking and compression
                        f.create_dataset(
                            key,
                            data=save_value,
                            chunks=chunk_shape,
                            compression='lzf',   # Fast decompression for frequent patch access
                            shuffle=False        # LZF doesn't benefit from shuffle
                            # fletcher32=False (default) - skip checksum for training speed
                        )

                        self.logger.debug(f"Saved {key} with patch-friendly chunking: {original_shape} → chunks={chunk_shape}")

                    else:
                        # === LABELS/MASKS OPTIMIZATION FOR PATCH ACCESS ===
                        if value.dtype == bool:
                            save_value = value.astype(np.uint8)
                        else:
                            save_value = value

                        # Use contiguous array for better I/O
                        save_value = np.ascontiguousarray(save_value)

                        # 🚀 CRITICAL: Explicit chunking for labels/masks patch access
                        if key in ['region_labels', 'region_mask'] and save_value.ndim == 3:
                            # Small chunks for efficient patch-wise label/mask reading
                            label_chunk_shape = (64, 64, 16)  # Optimized for random patch access

                            f.create_dataset(
                                key,
                                data=save_value,
                                chunks=label_chunk_shape,
                                compression='lzf',      # Fast decompression for frequent reads
                                # fletcher32=False (default) - skip checksum for speed
                            )
                            self.logger.debug(f"Saved {key} with patch-optimized chunking: {original_shape} → chunks={label_chunk_shape}")
                        else:
                            # Other data (like big_seg, region_seg_3d) with default chunking
                            f.create_dataset(
                                key,
                                data=save_value,
                                compression='lzf',      # Also use LZF for consistency
                                # fletcher32=False (default) - skip checksum for speed
                            )
                            self.logger.debug(f"Saved {key} with LZF compression: {original_shape}")

                        self.logger.debug(f"Saved {key}: {original_shape}")

                # Add global attributes for easy identification
                f.attrs['format_version'] = '1.0'
                f.attrs['patch_optimized'] = True
                f.attrs['normalization'] = 'patient_wise_zscore'
                f.attrs['recommended_patch_size'] = 7
                f.attrs['chunk_size'] = str(self.chunk_size)
                f.attrs['creation_time'] = datetime.now().isoformat()

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            save_time = time.time() - start_time
            self.logger.info(f"Patch-friendly file saved: {output_path.name} ({file_size_mb:.1f} MB, time: {save_time:.2f}s)")

            return file_size_mb

        except Exception as e:
            self.logger.error(f"Failed to save {output_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def validate_conversion(self, original_data: Dict[str, np.ndarray],
                          converted_path: Path) -> Dict[str, bool]:
        """Validate the conversion by checking data integrity"""
        self.logger.debug("Validating conversion...")

        try:
            results = {}
            region = original_data['region'].astype(bool)

            # Read back converted data
            with h5py.File(converted_path, 'r') as f:
                # Check if z-score normalization was applied correctly
                converted_data = f['data'][()]
                recovered_features = converted_data[region]  # (n_voxels, 351)

                # Validate that each dimension has approximately mean=0, std=1
                feature_validation = True
                for ch in range(recovered_features.shape[1]):
                    channel_data = recovered_features[:, ch]
                    mean_val = np.mean(channel_data)
                    std_val = np.std(channel_data)

                    # Check if normalization was applied (mean≈0, std≈1, or std≈0)
                    if not (abs(mean_val) < 1e-6 and (abs(std_val - 1.0) < 1e-6 or abs(std_val) < 1e-8)):
                        if ch < 5:  # Only log first few failures to avoid spam
                            self.logger.warning(f"Dimension {ch}: mean={mean_val:.6f}, std={std_val:.6f}")
                        feature_validation = False

                results['zscore_normalization'] = feature_validation

                # Validate other data integrity
                results['labels'] = np.array_equal(
                    original_data['region_seg'],
                    f['region_seg_3d'][()][region]
                )

                results['masks'] = np.array_equal(
                    original_data['region'],
                    f['region_mask'][()].astype(bool)
                )

                results['big_seg'] = np.array_equal(
                    original_data['big_seg'],
                    f['big_seg'][()]
                )

                # 🔧 NEW: Validate label range (should be 1-102)
                converted_labels = f['region_labels'][()]
                label_voxels = converted_labels[region]
                min_label = np.min(label_voxels)
                max_label = np.max(label_voxels)

                # Check if labels are in expected range 1-102
                labels_in_range = (min_label >= 1) and (max_label <= 102)
                results['label_range_1_102'] = labels_in_range

                self.logger.debug(f"Label range validation: {min_label}-{max_label} (expected: 1-102)")
                if not labels_in_range:
                    self.logger.warning(f"Label range issue: found {min_label}-{max_label}, expected 1-102")

            all_valid = all(results.values())
            if all_valid:
                self.logger.info("Validation passed ✅")
            else:
                self.logger.warning(f"Validation results: {results}")

            return results

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {'error': False}

    def process_single_subject(self, mat_path: Path) -> Dict[str, Any]:
        """Process a single subject with full conversion pipeline"""
        start_time = time.time()
        subject_id = mat_path.stem

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Processing subject: {subject_id}")
        self.logger.info(f"File: {mat_path.name}")

        result = {
            'subject_id': subject_id,
            'filename': mat_path.name,
            'start_time': start_time
        }

        try:
            # 1. Load original data
            self.logger.debug("Step 1: Loading original data")
            original_data = self.load_original_data(mat_path)

            # 2. Convert to 3D with z-score normalization
            self.logger.debug("Step 2: Converting to 3D with z-score normalization")
            converted_3d = self.convert_to_3d_with_zscore(original_data, subject_id)

            # 3. Save in patch-friendly format
            self.logger.debug("Step 3: Saving in patch-friendly format")
            # Keep original filename format but with .h5 extension for better compatibility
            output_filename = f"{subject_id}.h5"  # Same name as original but .h5 format
            output_path = self.output_dir / output_filename
            file_size_mb = self.save_patch_friendly_format(converted_3d, output_path)

            # 4. Validate conversion
            self.logger.debug("Step 4: Validating conversion")
            validation_results = self.validate_conversion(original_data, output_path)
            all_valid = all(validation_results.values())

            # 5. Record results
            processing_time = time.time() - start_time

            result.update({
                'status': 'success',
                'output_path': str(output_path),
                'file_size_mb': file_size_mb,
                'processing_time': processing_time,
                'validation_passed': all_valid,
                'validation_details': validation_results,
                'n_voxels': np.sum(original_data['region'].astype(bool)),
                'features_shape': original_data['multidim_data'].shape,
                'normalization': 'patient_wise_zscore'
            })

            if all_valid:
                self.logger.info(f"✅ Subject {subject_id} processed successfully ({processing_time:.1f}s, {file_size_mb:.1f}MB)")
            else:
                self.logger.warning(f"⚠️ Subject {subject_id} processed but validation incomplete")

        except Exception as e:
            self.logger.error(f"❌ Subject {subject_id} processing failed")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")

            result.update({
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'processing_time': time.time() - start_time
            })
            self.failed_subjects.append(subject_id)

        finally:
            # Memory cleanup
            gc.collect()

        return result

    def process_all_subjects(self, pattern: str = "*.mat") -> Tuple[List[str], List[str]]:
        """Process all subjects in the input directory"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting batch processing with z-score normalization")
        self.logger.info(f"Input pattern: {pattern}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"{'='*80}")

        # Find all MAT files
        mat_files = sorted(list(self.input_dir.glob(pattern)))

        if len(mat_files) == 0:
            self.logger.error(f"No MAT files found with pattern '{pattern}' in {self.input_dir}")
            return [], []

        self.logger.info(f"Found {len(mat_files)} MAT files to process")

        total_start_time = time.time()
        successful_subjects = []

        # Process each subject
        for mat_path in tqdm(mat_files, desc="Processing subjects"):
            result = self.process_single_subject(mat_path)
            self.conversion_results.append(result)

            if result['status'] == 'success':
                successful_subjects.append(result['subject_id'])

            # Memory monitoring
            if len(self.conversion_results) % 5 == 0:
                self.log_memory_usage(f"after {len(self.conversion_results)} subjects")

        total_time = time.time() - total_start_time

        # Generate final report
        self._generate_final_report(total_time)

        return successful_subjects, self.failed_subjects

    def _generate_final_report(self, total_time: float):
        """Generate comprehensive final report"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("FINAL PROCESSING REPORT")
        self.logger.info(f"{'='*80}")

        successful = [r for r in self.conversion_results if r['status'] == 'success']
        failed = [r for r in self.conversion_results if r['status'] == 'failed']
        fully_validated = [r for r in successful if r.get('validation_passed', False)]

        self.logger.info(f"Processing Statistics:")
        self.logger.info(f"  Total processed: {len(self.conversion_results)}")
        self.logger.info(f"  Successfully converted: {len(successful)}")
        self.logger.info(f"  Fully validated: {len(fully_validated)}")
        self.logger.info(f"  Failed: {len(failed)}")
        self.logger.info(f"  Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"  Final memory: {self.process.memory_info().rss / 1024 / 1024:.1f} MB")

        if successful:
            avg_time = np.mean([r['processing_time'] for r in successful])
            avg_size = np.mean([r['file_size_mb'] for r in successful])
            total_size = sum([r['file_size_mb'] for r in successful])
            self.logger.info(f"  Average processing time: {avg_time:.1f}s")
            self.logger.info(f"  Average file size: {avg_size:.1f}MB")
            self.logger.info(f"  Total output size: {total_size:.1f}MB")

        # Save detailed JSON report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"zscore_conversion_report_{timestamp}.json"

        report_data = {
            'conversion_summary': {
                'total_processed': len(self.conversion_results),
                'successful': len(successful),
                'failed': len(failed),
                'fully_validated': len(fully_validated),
                'total_time_minutes': total_time/60,
                'normalization_method': 'patient_wise_zscore'
            },
            'successful_subjects': [r['subject_id'] for r in successful],
            'failed_subjects': self.failed_subjects,
            'detailed_results': self.conversion_results,
            'processing_parameters': {
                'chunk_size': self.chunk_size,
                'compression': 'gzip',
                'patch_optimized': True,
                'format_version': '1.0'
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(f"Detailed report saved: {report_path}")

        # Create dataset info file
        info_path = self.output_dir / "dataset_info.json"
        dataset_info = {
            'dataset_name': 'MRI Brain Data - Z-Score Normalized',
            'format_version': '1.0',
            'normalization': 'patient_wise_zscore',
            'patch_optimized': True,
            'recommended_patch_size': 7,
            'data_layout': 'XYZC (384, 336, 256, 351)',
            'chunk_size': self.chunk_size,
            'compression': 'gzip level 3',
            'total_subjects': len(successful),
            'creation_date': datetime.now().isoformat(),
            'usage_notes': [
                'Data is normalized per-patient across all 351 dimensions',
                'HDF5 files are chunked for efficient 7x7 patch extraction',
                'Use h5py for optimal performance when reading patches',
                'Each file contains: data, region_labels, region_mask, big_seg, region_seg_3d'
            ]
        }

        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        self.logger.info(f"Dataset info saved: {info_path}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Convert MRI dataset with patient-wise z-score normalization')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing MAT files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for converted files')
    parser.add_argument('--pattern', type=str, default='*.mat',
                       help='File pattern to match (default: *.mat)')
    parser.add_argument('--chunk-size', type=str, default='32,32,1,351',
                       help='HDF5 chunk size as comma-separated values (default: 32,32,1,351 - optimized for 7×7×1 patch)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Parse chunk size
    try:
        chunk_size = tuple(map(int, args.chunk_size.split(',')))
        if len(chunk_size) != 4:
            raise ValueError("Chunk size must have 4 dimensions")
    except ValueError as e:
        print(f"Error parsing chunk size: {e}")
        return

    # Create converter
    converter = ZScoreDatasetConverter(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        log_level=args.log_level,
        chunk_size=chunk_size
    )

    try:
        successful, failed = converter.process_all_subjects(pattern=args.pattern)

        if len(failed) == 0:
            converter.logger.info(f"\n🎉 All subjects processed successfully!")
        else:
            converter.logger.warning(f"\n⚠️ {len(failed)} subjects failed processing")

    except Exception as e:
        converter.logger.error(f"Program failed: {e}")
        converter.logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()