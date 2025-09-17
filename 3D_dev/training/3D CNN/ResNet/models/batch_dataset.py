"""
批量加载数据集模块，支持分批加载mat文件避免内存溢出

基于原始dataset.py，添加批量加载功能：
- 每次只加载指定数量的mat文件
- 训练完一批后释放内存，加载下一批
- 支持Leave-One-Out交叉验证的批量训练
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
import gc


class BatchMRIBrain2DPatchDataset(Dataset):
    """
    批量加载的MRI数据集类

    每次只加载指定数量的mat文件到内存，训练完成后释放内存加载下一批
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
        """
        Args:
            mat_files: 所有mat文件路径列表
            patch_size: 图像块大小
            samples_per_subject: 每个被试的样本数
            is_train: 是否为训练模式
            balance_classes: 是否平衡类别
            min_samples_per_class: 每个类别的最小样本数
            augmentation: 是否数据增强
            batch_files: 每次批量加载的文件数量
        """
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
        self.current_sample_indices = []

        # 类别统计信息（需要预扫描所有文件获取）
        self.class_counts = self._analyze_global_class_distribution()
        self.class_weights = self._compute_class_weights()

        # 加载第一批数据
        self._load_current_batch()

        self.logger.info(f"批量数据集初始化完成，共{len(self.mat_files)}个文件，分为{len(self.file_batches)}批")
        self.logger.info(f"每批加载{self.batch_files}个文件，当前批次: {self.current_batch_idx + 1}")

    def _create_file_batches(self) -> List[List[Path]]:
        """将文件分批"""
        batches = []
        for i in range(0, len(self.mat_files), self.batch_files):
            batch = self.mat_files[i:i + self.batch_files]
            batches.append(batch)
        return batches

    def _analyze_global_class_distribution(self) -> Dict[int, int]:
        """预扫描所有文件分析类别分布（不加载完整数据）"""
        self.logger.info("预扫描所有文件分析类别分布...")
        class_counts = Counter()

        for mat_file in tqdm(self.mat_files, desc="扫描类别分布"):
            with h5py.File(mat_file, 'r') as f:
                region_labels = f['region_labels'][()]
                region_mask = f['region_mask'][()]

                # 处理数据格式
                if region_labels.shape != (384, 336, 256):
                    region_labels = region_labels.T
                if region_mask.shape != (384, 336, 256):
                    region_mask = region_mask.T

                # 统计有效体素的类别
                valid_positions = np.where((region_mask > 0) & (region_labels > 0))

                # 采样统计（避免内存溢出）
                if len(valid_positions[0]) > 50000:  # 如果体素太多，采样统计
                    sample_size = 50000
                    indices = np.random.choice(len(valid_positions[0]), sample_size, replace=False)
                    sampled_labels = region_labels[valid_positions[0][indices],
                                                 valid_positions[1][indices],
                                                 valid_positions[2][indices]]
                else:
                    sampled_labels = region_labels[valid_positions]

                # 统计类别数量
                unique_labels, counts = np.unique(sampled_labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    class_counts[label] += count

        self.logger.info(f"扫描完成，发现{len(class_counts)}个类别")
        return dict(class_counts)

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

        # 生成当前批次的样本索引
        self.current_sample_indices = self._generate_batch_sample_indices(current_files)

        self.logger.info(f"批次加载完成，当前批次包含{len(self.current_sample_indices)}个样本")

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

    def _generate_batch_sample_indices(self, batch_files: List[Path]) -> List[Tuple[str, int, int, int]]:
        """为当前批次生成样本索引"""
        indices = []

        for mat_file in batch_files:
            subject_data = self.current_cached_data[str(mat_file)]
            mask = subject_data['mask']
            labels = subject_data['labels']

            # 获取所有有效体素位置
            valid_positions = np.where((mask > 0) & (labels > 0))
            valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))

            # 采样（与原始版本一致）
            if self.samples_per_subject and len(valid_coords) > self.samples_per_subject:
                if self.is_train:
                    sampled_coords = random.sample(valid_coords, self.samples_per_subject)
                else:
                    step = len(valid_coords) // self.samples_per_subject
                    sampled_coords = valid_coords[::step][:self.samples_per_subject]
            else:
                sampled_coords = valid_coords

            # 添加到索引列表（使用文件路径而不是索引）
            for x, y, z in sampled_coords:
                indices.append((str(mat_file), x, y, z))

        # 训练时打乱索引
        if self.is_train:
            random.shuffle(indices)

        return indices

    def next_batch(self) -> bool:
        """切换到下一批数据"""
        if self.current_batch_idx + 1 < len(self.file_batches):
            self.current_batch_idx += 1
            self._load_current_batch()
            return True
        else:
            # 已经是最后一批，重置到第一批
            self.current_batch_idx = 0
            self._load_current_batch()
            return False  # 表示已经完成一轮

    def reset_to_first_batch(self):
        """重置到第一批数据"""
        self.current_batch_idx = 0
        self._load_current_batch()

    def get_current_batch_info(self) -> Dict:
        """获取当前批次信息"""
        return {
            'current_batch': self.current_batch_idx + 1,
            'total_batches': len(self.file_batches),
            'current_files': len(self.file_batches[self.current_batch_idx]),
            'current_samples': len(self.current_sample_indices)
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
        return len(self.current_sample_indices)

    def __getitem__(self, idx):
        mat_file_path, x, y, z = self.current_sample_indices[idx]

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


def create_batch_data_loaders(
    train_files: List[Path],
    test_files: List[Path],
    patch_size: int = 7,
    batch_size: int = 256,
    num_workers: int = 4,
    samples_per_subject: Optional[int] = 10000,
    balance_classes: bool = False,
    augmentation: bool = False,
    batch_files: int = 3  # 每次加载的文件数量
) -> Tuple[DataLoader, DataLoader, 'BatchMRIBrain2DPatchDataset']:
    """
    创建批量加载的数据加载器

    Returns:
        (train_loader, test_loader, train_dataset)
        返回train_dataset用于批次管理
    """

    # 创建批量数据集
    train_dataset = BatchMRIBrain2DPatchDataset(
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
        shuffle=True,  # 每个批次内部打乱
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
    # 测试批量数据集
    import glob

    mat_files = [Path(f) for f in glob.glob("/path/to/mat/files/*.mat")]

    if len(mat_files) >= 6:
        print(f"找到 {len(mat_files)} 个MAT文件")

        # 测试批量数据集
        dataset = BatchMRIBrain2DPatchDataset(
            mat_files=mat_files[:6],  # 测试6个文件
            patch_size=7,
            samples_per_subject=1000,
            is_train=True,
            batch_files=3  # 每次加载3个文件
        )

        print(f"当前批次信息: {dataset.get_current_batch_info()}")
        print(f"数据集大小: {len(dataset)}")

        # 测试数据加载
        patch, label = dataset[0]
        print(f"图像块形状: {patch.shape}")
        print(f"标签: {label.item()}")

        # 测试批次切换
        print("\\n切换到下一批...")
        has_next = dataset.next_batch()
        print(f"当前批次信息: {dataset.get_current_batch_info()}")
        print(f"数据集大小: {len(dataset)}")

    else:
        print("需要至少6个MAT文件进行测试")