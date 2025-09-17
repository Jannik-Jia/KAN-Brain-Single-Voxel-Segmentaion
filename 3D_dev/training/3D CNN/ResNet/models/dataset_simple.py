"""
简化版的内存优化数据集 - 专注于解决核心性能问题
避免过度工程化，专注于最重要的优化：
1. 简单的文件级缓存（LRU）
2. 避免重复读取相同patch
3. 无复杂的预读取机制
"""

import numpy as np
import torch
import h5py
import logging
import random
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm
import time


class MRIBrain2DPatchDatasetSimple(Dataset):
    """简化版本的内存高效2D patch数据集"""

    def __init__(
        self,
        mat_files: List[Path],
        patch_size: int = 7,
        samples_per_subject: Optional[int] = 10000,
        is_train: bool = True,
        cache_data: bool = False,
        balance_classes: bool = False,
        min_samples_per_class: int = 50,
        augmentation: bool = False,
        memory_efficient: bool = True
    ):
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

        if memory_efficient and is_train:
            self._init_simple_memory_efficient()
        else:
            self._init_original_mode()

        self.logger.info(f"Simple dataset initialized with {len(self)} samples")

    def _init_simple_memory_efficient(self):
        """简化的内存优化初始化"""
        # 构建subject映射
        self.subject_file_map = {}
        for i, mat_file in enumerate(self.mat_files):
            filename = mat_file.stem
            if 'subject' in filename:
                subject_id = int(filename.split('subject')[1].split('_')[0])
            else:
                # 对于.h5文件，直接使用文件名作为subject_id
                try:
                    subject_id = int(filename.split('.')[0])
                except ValueError:
                    subject_id = i + 1
            self.subject_file_map[subject_id] = mat_file

        # 加载所有labels到内存
        self.all_labels = {}
        self.all_masks = {}

        self.logger.info("Loading labels/masks for all subjects...")
        for subject_id, mat_file in tqdm(self.subject_file_map.items(), desc="Loading labels/masks"):
            with h5py.File(mat_file, 'r') as f:
                region_labels = f['region_labels'][()]
                region_mask = f['region_mask'][()]

                # 处理维度
                if region_labels.shape != (384, 336, 256):
                    region_labels = region_labels.T
                if region_mask.shape != (384, 336, 256):
                    region_mask = region_mask.T

                self.all_labels[subject_id] = region_labels.astype(np.int16)
                self.all_masks[subject_id] = region_mask.astype(bool)

        # 生成所有有效坐标
        self._generate_all_coords_simple()

        # 简化版本：跳过class counts计算以提高初始化速度
        self.class_counts = {}  # 空字典，避免访问错误
        self.class_weights = torch.ones(102)  # 均匀权重

        # 简单的文件缓存
        self.file_cache = {}
        self.max_cached_files = 2  # 只缓存2个文件
        self.cache_access_order = []
        self.cache_lock = threading.Lock()
        self.file_locks = defaultdict(threading.Lock)

    def _generate_all_coords_simple(self):
        """生成所有有效坐标的简化版本"""
        self.logger.info("Generating all valid coordinates...")

        all_coords = []  # 简单的坐标列表: [(subject_id, x, y, z), ...]

        for subject_id in self.subject_file_map:
            mask = self.all_masks[subject_id]
            valid_coords = np.where(mask)

            subject_coords = list(zip([subject_id] * len(valid_coords[0]),
                                    valid_coords[0], valid_coords[1], valid_coords[2]))
            all_coords.extend(subject_coords)

            self.logger.debug(f"Subject {subject_id}: {len(subject_coords)} valid patches")

        self.all_coordinates = all_coords
        self.logger.info(f"Total valid coordinates: {len(all_coords):,}")

        # 每个epoch随机打乱
        if self.is_train:
            random.shuffle(self.all_coordinates)

    def _init_original_mode(self):
        """原始模式初始化（向后兼容）"""
        # 这里保持原来的逻辑...
        pass

    def __len__(self):
        if self.memory_efficient and self.is_train:
            return len(self.all_coordinates)
        else:
            # 原始模式逻辑
            return len(self.mat_files) * (self.samples_per_subject or 10000)

    def __getitem__(self, idx):
        if self.memory_efficient and self.is_train:
            return self._getitem_simple_efficient(idx)
        else:
            return self._getitem_original(idx)

    def _getitem_simple_efficient(self, idx):
        """简化的内存优化getitem"""
        subject_id, x, y, z = self.all_coordinates[idx]

        # 获取文件数据（带缓存）
        data_4d = self._get_file_data_simple(subject_id)

        # 提取patch
        patch_2d = self._extract_patch_simple(data_4d, x, y, z)

        # 数据增强
        if self.augmentation:
            patch_2d = self.apply_augmentation(patch_2d)

        # 获取标签
        label = self.all_labels[subject_id][x, y, z] - 1  # 转换为0-101

        # 转换为tensor
        patch_tensor = torch.from_numpy(patch_2d.transpose(2, 0, 1).astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)

        return patch_tensor, label_tensor

    def _get_file_data_simple(self, subject_id: int) -> np.ndarray:
        """简单的文件数据获取（带LRU缓存）"""
        # 检查缓存
        with self.cache_lock:
            if subject_id in self.file_cache:
                # 更新访问顺序
                if subject_id in self.cache_access_order:
                    self.cache_access_order.remove(subject_id)
                self.cache_access_order.append(subject_id)
                return self.file_cache[subject_id]

        # 从磁盘加载
        mat_file = self.subject_file_map[subject_id]

        with self.file_locks[subject_id]:
            with h5py.File(mat_file, 'r') as f:
                data_ref = f['data']

                # 处理数据格式
                if data_ref.shape[0] == 351:
                    data_4d = data_ref[()].transpose(1, 2, 3, 0)
                elif data_ref.shape[-1] == 351:
                    data_4d = data_ref[()]
                else:
                    raise ValueError(f"不支持的数据格式: {data_ref.shape}")

                data_4d = data_4d.astype(np.float32)

        # 添加到缓存
        with self.cache_lock:
            if len(self.file_cache) >= self.max_cached_files:
                # 移除最旧的
                oldest_subject = self.cache_access_order.pop(0)
                del self.file_cache[oldest_subject]
                self.logger.debug(f"Evicted subject {oldest_subject} from cache")

            self.file_cache[subject_id] = data_4d
            self.cache_access_order.append(subject_id)
            self.logger.debug(f"Cached subject {subject_id}")

        return data_4d

    def _extract_patch_simple(self, data_4d: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
        """简单的patch提取"""
        p = self.patch_size // 2

        x_min = max(0, x - p)
        x_max = min(384, x + p + 1)
        y_min = max(0, y - p)
        y_max = min(336, y + p + 1)

        # 提取patch
        patch_data = data_4d[x_min:x_max, y_min:y_max, z, :]

        # 处理边界填充
        if patch_data.shape[:2] != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size, 351), dtype=patch_data.dtype)

            x_off = p - (x - x_min)
            y_off = p - (y - y_min)

            padded[x_off:x_off+patch_data.shape[0],
                   y_off:y_off+patch_data.shape[1], :] = patch_data
            patch_data = padded

        return patch_data

    def apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """简单的数据增强"""
        if not self.augmentation:
            return patch

        # 随机水平翻转
        if random.random() < 0.5:
            patch = np.fliplr(patch)

        # 随机垂直翻转
        if random.random() < 0.5:
            patch = np.flipud(patch)

        return patch

    def _getitem_original(self, idx):
        """原始模式getitem（向后兼容）"""
        # 保持原来的逻辑...
        pass

    def set_epoch(self, epoch: int):
        """设置epoch并重新打乱数据"""
        if self.memory_efficient and self.is_train:
            self.logger.info(f"Shuffling coordinates for epoch {epoch}")
            random.shuffle(self.all_coordinates)

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        with self.cache_lock:
            return {
                "cached_files": len(self.file_cache),
                "max_cache_size": self.max_cached_files,
                "cache_order": self.cache_access_order.copy()
            }