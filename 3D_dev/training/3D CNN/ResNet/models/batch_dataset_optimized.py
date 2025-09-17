"""
优化的批量加载数据集模块，减少文件读取次数

优化策略：
1. 合并预扫描和类别分析为一次遍历
2. 延迟加载：只在真正需要时加载完整数据
3. 智能缓存：避免重复读取
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import h5py
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from collections import Counter
import logging
import gc


class OptimizedBatchMRIBrain2DPatchDataset(Dataset):
    """
    优化的批量加载MRI数据集类

    优化重点：
    - 减少文件读取次数：合并预扫描和类别分析
    - 延迟加载：只在需要时加载完整数据
    - 避免重复训练相同的patch
    """

    def __init__(
        self,
        mat_files: List[Path],
        patch_size: int = 7,
        samples_per_subject: Optional[int] = None,
        is_train: bool = True,
        balance_classes: bool = False,
        min_samples_per_class: int = 50,
        augmentation: bool = False,
        batch_files: int = 3  # 每次加载的文件数量
    ):
        self.mat_files = mat_files
        self.patch_size = patch_size
        self.samples_per_subject = samples_per_subject
        self.is_train = is_train
        self.balance_classes = balance_classes
        self.min_samples_per_class = min_samples_per_class
        self.augmentation = augmentation and is_train
        self.batch_files = batch_files

        # 初始化日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 批次管理
        self.current_batch_idx = 0
        self.file_batches = self._create_file_batches()
        self.current_cached_data = {}

        # === 优化的索引管理 ===
        self.all_voxel_indices = {}  # {file_path: [(x,y,z), ...]}
        self.class_counts = {}  # 全局类别统计
        self.epoch_sample_indices = []  # 当前epoch的所有样本索引
        self.current_batch_indices = []  # 当前批次的样本索引
        self.used_voxel_indices = set()  # 当前epoch已使用的体素索引
        self.current_epoch = 0

        # === 关键优化：一次性预扫描 ===
        self._single_pass_precompute()

        # 计算类别权重
        self.class_weights = self._compute_class_weights()

        # 初始化第一个epoch的样本分配
        self._allocate_epoch_samples()

        # 延迟加载：不在初始化时加载数据
        self._data_loaded = False

        self.logger.info(f"优化批量数据集初始化完成，共{len(self.mat_files)}个文件")
        self.logger.info(f"分为{len(self.file_batches)}批，当前epoch分配了{len(self.epoch_sample_indices)}个唯一样本")
        self.logger.info(f"发现{len(self.class_counts)}个类别，延迟加载模式启用")

    def _single_pass_precompute(self):
        """一次性预扫描：同时获取体素位置和类别分布"""
        self.logger.info("一次性预扫描所有文件（获取体素位置和类别分布）...")

        class_counts = Counter()

        for mat_file in tqdm(self.mat_files, desc="预扫描文件"):
            with h5py.File(mat_file, 'r') as f:
                region_labels = f['region_labels'][()]
                region_mask = f['region_mask'][()]

                # 处理数据格式
                if region_labels.shape != (384, 336, 256):
                    region_labels = region_labels.T
                if region_mask.shape != (384, 336, 256):
                    region_mask = region_mask.T

                # 获取所有有效体素位置
                valid_positions = np.where((region_mask > 0) & (region_labels > 0))
                valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))

                # 存储体素位置
                self.all_voxel_indices[str(mat_file)] = valid_coords

                # 同时进行类别统计（采样以避免内存溢出）
                if len(valid_coords) > 10000:  # 如果体素太多，采样统计
                    sample_size = 10000
                    sampled_coords = random.sample(valid_coords, sample_size)
                else:
                    sampled_coords = valid_coords

                # 统计类别数量
                for x, y, z in sampled_coords:
                    label = region_labels[x, y, z]
                    class_counts[label] += 1

        # 保存类别统计
        self.class_counts = dict(class_counts)

        total_voxels = sum(len(coords) for coords in self.all_voxel_indices.values())
        self.logger.info(f"预扫描完成：{total_voxels}个有效体素，{len(self.class_counts)}个类别")

    def _create_file_batches(self) -> List[List[Path]]:
        """将文件分批，优化最后一批的大小"""
        batches = []
        total_files = len(self.mat_files)

        # 如果能被batch_files整除，正常分批
        if total_files % self.batch_files == 0:
            for i in range(0, total_files, self.batch_files):
                batch = self.mat_files[i:i + self.batch_files]
                batches.append(batch)
        else:
            # 如果有余数，将最后的文件合并到倒数第二批
            remainder = total_files % self.batch_files

            if remainder == 1:
                # 余数为1：将最后1个文件合并到倒数第二批
                # 例：37个文件，batch_files=3 → [0:3], [3:6], ..., [30:33], [33:37] (4个文件)
                for i in range(0, total_files - self.batch_files - 1, self.batch_files):
                    batch = self.mat_files[i:i + self.batch_files]
                    batches.append(batch)
                # 最后一批包含 batch_files + 1 个文件
                last_batch = self.mat_files[total_files - self.batch_files - 1:]
                batches.append(last_batch)

                self.logger.info(f"优化分批：最后一批包含{len(last_batch)}个文件 (合并避免单独1个文件)")
            else:
                # 余数为2或更多：正常分批
                for i in range(0, total_files, self.batch_files):
                    batch = self.mat_files[i:i + self.batch_files]
                    batches.append(batch)

        return batches

    def _allocate_epoch_samples(self):
        """为当前epoch分配样本，确保无重复"""
        self.logger.info(f"为epoch {self.current_epoch + 1}分配样本...")

        # 清空之前的分配
        self.epoch_sample_indices.clear()
        self.used_voxel_indices.clear()

        # 为每个文件分配样本
        for mat_file in self.mat_files:
            file_path = str(mat_file)
            available_coords = self.all_voxel_indices[file_path]

            # 确定采样数量
            if self.samples_per_subject and self.samples_per_subject > 0 and len(available_coords) > self.samples_per_subject:
                if self.is_train:
                    # 训练时随机采样（每个epoch不同）
                    random.seed(self.current_epoch * 42 + hash(file_path) % 1000)
                    sampled_coords = random.sample(available_coords, self.samples_per_subject)
                else:
                    # 测试时均匀采样
                    step = len(available_coords) // self.samples_per_subject
                    sampled_coords = available_coords[::step][:self.samples_per_subject]
            else:
                # samples_per_subject=0或None时，使用所有有效体素
                sampled_coords = available_coords.copy()
                self.logger.info(f"文件 {Path(file_path).name}: 使用所有 {len(sampled_coords)} 个有效体素")

            # 添加到epoch样本列表
            for x, y, z in sampled_coords:
                voxel_id = f"{file_path}:{x}:{y}:{z}"
                if voxel_id not in self.used_voxel_indices:
                    self.epoch_sample_indices.append((file_path, x, y, z))
                    self.used_voxel_indices.add(voxel_id)

        # 训练时打乱epoch内的样本顺序
        if self.is_train:
            random.seed(self.current_epoch * 123)
            random.shuffle(self.epoch_sample_indices)

        self.logger.info(f"Epoch {self.current_epoch + 1}分配了{len(self.epoch_sample_indices)}个唯一样本")

    def _compute_class_weights(self) -> torch.Tensor:
        """计算类别权重"""
        if not self.class_counts:
            return torch.ones(102)

        total_samples = sum(self.class_counts.values())
        weights = []

        for class_id in range(1, 103):  # 类别1-102
            count = self.class_counts.get(class_id, 1)
            weight = total_samples / (len(self.class_counts) * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def _ensure_current_batch_loaded(self):
        """确保当前批次已加载（延迟加载）"""
        if not self._data_loaded or not self.current_cached_data:
            self._load_current_batch()

    def _load_current_batch(self):
        """加载当前批次的数据"""
        # 清理之前的数据
        self.current_cached_data.clear()
        gc.collect()

        # 获取当前批次的文件
        current_files = self.file_batches[self.current_batch_idx]

        self.logger.info(f"加载批次 {self.current_batch_idx + 1}/{len(self.file_batches)}，"
                        f"包含{len(current_files)}个文件")

        # 加载当前批次的数据
        for mat_file in tqdm(current_files, desc=f"加载批次{self.current_batch_idx + 1}"):
            self.current_cached_data[str(mat_file)] = self.load_subject_data(mat_file)

        # 获取当前批次的样本索引
        self.current_batch_indices = self._get_current_batch_samples()

        # 标记数据已加载
        self._data_loaded = True

        self.logger.info(f"批次加载完成，当前批次包含{len(self.current_batch_indices)}个样本")

    def _get_current_batch_samples(self) -> List[Tuple[str, int, int, int]]:
        """获取当前批次应该包含的样本（从epoch样本中筛选）"""
        current_files = set(str(f) for f in self.file_batches[self.current_batch_idx])

        # 筛选属于当前批次文件的样本
        batch_samples = [
            (file_path, x, y, z) for file_path, x, y, z in self.epoch_sample_indices
            if file_path in current_files
        ]

        return batch_samples

    def load_subject_data(self, mat_file: Path) -> Dict[str, np.ndarray]:
        """加载单个被试数据 - 与原始dataset.py完全一致"""
        with h5py.File(mat_file, 'r') as f:
            # 加载必要数据
            data = f['data'][()]
            region_labels = f['region_labels'][()]
            region_mask = f['region_mask'][()]

            # 调试输出（与原始版本一致）
            print(f"原始数据形状: data={data.shape}, labels={region_labels.shape}, mask={region_mask.shape}")

            # 处理数据格式（与原始版本一致）
            if data.shape[0] == 351:
                data = np.transpose(data, (1, 2, 3, 0))
            elif data.shape[-1] == 351:
                pass
            else:
                raise ValueError(f"无法识别数据格式，shape: {data.shape}")

            # 处理标签和掩码（与原始版本一致）
            if region_labels.shape != (384, 336, 256):
                region_labels = region_labels.T
            if region_mask.shape != (384, 336, 256):
                region_mask = region_mask.T

            print(f"转置后数据形状: data={data.shape}, labels={region_labels.shape}, mask={region_mask.shape}")

            # Z-score标准化（与原始版本一致）
            data = data.astype(np.float32)
            for ch in range(data.shape[3]):  # 351个channels
                channel_data = data[:, :, :, ch]
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)

                if std_val > 1e-8:
                    data[:, :, :, ch] = (channel_data - mean_val) / std_val
                else:
                    data[:, :, :, ch] = 0

            print(f"已完成351维channel的z-score标准化")

        return {
            'data': data,
            'labels': region_labels.astype(np.int64),
            'mask': region_mask.astype(bool)
        }

    def next_batch(self) -> bool:
        """切换到下一批数据"""
        if self.current_batch_idx + 1 < len(self.file_batches):
            self.current_batch_idx += 1
            self._data_loaded = False  # 标记需要重新加载
            return True
        else:
            # 已经是最后一批，准备下一个epoch
            return False

    def start_new_epoch(self):
        """开始新的epoch"""
        self.current_epoch += 1
        self.current_batch_idx = 0
        self._data_loaded = False  # 标记需要重新加载

        # 重新分配样本（避免重复，但每个epoch不同）
        self._allocate_epoch_samples()

        self.logger.info(f"开始Epoch {self.current_epoch + 1}，"
                        f"分配了{len(self.epoch_sample_indices)}个唯一样本")

    def reset_to_first_batch(self):
        """重置到第一批数据（当前epoch内）"""
        self.current_batch_idx = 0
        self._data_loaded = False  # 标记需要重新加载

    def get_current_batch_info(self) -> Dict:
        """获取当前批次信息"""
        return {
            'current_batch': self.current_batch_idx + 1,
            'total_batches': len(self.file_batches),
            'current_files': len(self.file_batches[self.current_batch_idx]),
            'current_samples': len(self.current_batch_indices) if hasattr(self, 'current_batch_indices') else 0,
            'epoch_total_samples': len(self.epoch_sample_indices),
            'current_epoch': self.current_epoch + 1,
            'data_loaded': self._data_loaded
        }

    def get_epoch_coverage_stats(self) -> Dict:
        """获取epoch覆盖统计"""
        total_possible = sum(len(coords) for coords in self.all_voxel_indices.values())
        current_samples = len(self.epoch_sample_indices)

        return {
            'total_possible_voxels': total_possible,
            'current_epoch_samples': current_samples,
            'coverage_ratio': current_samples / total_possible if total_possible > 0 else 0,
            'unique_samples_guaranteed': True
        }

    def extract_patch_2d(self, data: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
        """提取2D图像块 - 与原始版本一致"""
        p = self.patch_size // 2

        x_min = max(0, x - p)
        x_max = min(data.shape[0], x + p + 1)
        y_min = max(0, y - p)
        y_max = min(data.shape[1], y + p + 1)

        patch = data[x_min:x_max, y_min:y_max, z, :]

        if patch.shape[:2] != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size, data.shape[3]), dtype=data.dtype)
            x_off = p - (x - x_min)
            y_off = p - (y - y_min)
            padded[x_off:x_off+patch.shape[0], y_off:y_off+patch.shape[1], :] = patch
            patch = padded

        return patch

    def apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """应用数据增强 - 与原始版本一致"""
        if not self.augmentation:
            return patch

        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()

        if random.random() < 0.5:
            patch = np.flip(patch, axis=0).copy()

        if random.random() < 0.5:
            k = random.randint(1, 3)
            patch = np.rot90(patch, k=k, axes=(0, 1)).copy()

        if random.random() < 0.3:
            noise_std = 0.01 * np.std(patch)
            noise = np.random.normal(0, noise_std, patch.shape)
            patch = patch + noise

        return patch

    def __len__(self):
        # 确保数据已加载
        self._ensure_current_batch_loaded()
        return len(self.current_batch_indices)

    def __getitem__(self, idx):
        # 确保数据已加载
        self._ensure_current_batch_loaded()

        if idx >= len(self.current_batch_indices):
            raise IndexError(f"Index {idx} out of range for current batch size {len(self.current_batch_indices)}")

        mat_file_path, x, y, z = self.current_batch_indices[idx]

        # 获取被试数据
        subject_data = self.current_cached_data[mat_file_path]

        # 提取2D图像块
        patch_2d = self.extract_patch_2d(subject_data['data'], x, y, z)

        # 应用数据增强
        if self.augmentation:
            patch_2d = self.apply_augmentation(patch_2d)

        # 获取中心体素标签（从1-102转换为0-101）
        label = subject_data['labels'][x, y, z] - 1

        # 转换为PyTorch格式 (C, H, W)
        patch_tensor = torch.from_numpy(patch_2d.transpose(2, 0, 1).astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)

        return patch_tensor, label_tensor

    def get_class_weights(self) -> torch.Tensor:
        """获取类别权重"""
        return self.class_weights


def create_optimized_batch_data_loaders(
    train_files: List[Path],
    test_files: List[Path],
    patch_size: int = 7,
    batch_size: int = 256,
    num_workers: int = 4,
    samples_per_subject: Optional[int] = 10000,
    balance_classes: bool = False,
    augmentation: bool = False,
    batch_files: int = 3  # 每次加载的文件数量
) -> Tuple[DataLoader, DataLoader, 'OptimizedBatchMRIBrain2DPatchDataset']:
    """
    创建优化的批量加载数据加载器

    Returns:
        (train_loader, test_loader, train_dataset)
        返回train_dataset用于批次管理
    """

    # 创建优化的批量数据集
    train_dataset = OptimizedBatchMRIBrain2DPatchDataset(
        mat_files=train_files,
        patch_size=patch_size,
        samples_per_subject=samples_per_subject,
        is_train=True,
        balance_classes=balance_classes,
        augmentation=augmentation,
        batch_files=batch_files
    )

    # 测试集仍使用原始方式（因为通常只有一个文件）
    from dataset import MRIBrain2DPatchDataset
    test_dataset = MRIBrain2DPatchDataset(
        mat_files=test_files,
        patch_size=patch_size,
        samples_per_subject=samples_per_subject * 2 if samples_per_subject else None,
        is_train=False,
        cache_data=True,
        balance_classes=False,
        augmentation=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 优化版本内部已经处理了shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader, train_dataset


if __name__ == "__main__":
    # 测试优化的批量数据集
    import glob

    mat_files = [Path(f) for f in glob.glob("/path/to/mat/files/*.mat")]

    if len(mat_files) >= 6:
        print(f"找到 {len(mat_files)} 个MAT文件")

        # 测试优化的批量数据集
        dataset = OptimizedBatchMRIBrain2DPatchDataset(
            mat_files=mat_files[:6],  # 测试6个文件
            patch_size=7,
            samples_per_subject=1000,
            is_train=True,
            batch_files=3  # 每次加载3个文件
        )

        print(f"当前批次信息: {dataset.get_current_batch_info()}")
        print(f"Epoch覆盖统计: {dataset.get_epoch_coverage_stats()}")

        # 测试延迟加载
        print("\\n=== 测试延迟加载 ===")
        print(f"初始化后数据是否已加载: {dataset._data_loaded}")

        # 第一次访问数据时才真正加载
        print(f"数据集大小: {len(dataset)}")
        print(f"访问后数据是否已加载: {dataset._data_loaded}")

        # 测试数据加载
        patch, label = dataset[0]
        print(f"图像块形状: {patch.shape}")
        print(f"标签: {label.item()}")

    else:
        print("需要至少6个MAT文件进行测试")