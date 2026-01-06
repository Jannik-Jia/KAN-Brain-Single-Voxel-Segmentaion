#!/usr/bin/env python3
"""
单个 MAT 文件数据加载器
用于 Alex7T 数据集的体素分类任务

支持 Per-Patient Z-Score 标准化
每个被试一个独立的 .mat 文件
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm


def per_patient_zscore(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    对单个患者的数据进行逐通道 z-score 标准化

    Args:
        data: (n_voxels, n_channels) 单个患者的特征数据
        epsilon: 防止除零的小常数

    Returns:
        标准化后的数据 (n_voxels, n_channels)
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std = np.where(std < epsilon, epsilon, std)
    normalized = (data - mean) / std
    return normalized.astype(np.float32)


class Brain1D_Dataset(Dataset):
    """
    脑部 1D 体素分类数据集 (训练/验证用)

    Args:
        mat_files: 1D MAT 文件路径列表
        feature_dim: 使用的特征维度数 (默认 341)
        use_zscore: 是否使用 per-patient z-score 标准化
    """

    def __init__(
        self,
        mat_files: List[Path],
        feature_dim: int = 341,
        use_zscore: bool = True
    ):
        self.mat_files = mat_files
        self.feature_dim = feature_dim
        self.use_zscore = use_zscore

        self.all_data = []
        self.all_labels = []
        self.subject_names = []

        print(f"加载 {len(mat_files)} 个被试的 1D 数据...")

        for mat_file in tqdm(mat_files, desc="Loading subjects"):
            self._load_subject(mat_file)

        # 转换为 numpy 数组
        self.all_data = np.vstack(self.all_data).astype(np.float32)
        self.all_labels = np.concatenate(self.all_labels).astype(np.int64)

        # 打印统计信息
        print(f"总样本数: {len(self.all_data):,}")
        print(f"特征维度: {self.all_data.shape[1]}")
        print(f"类别数: {len(np.unique(self.all_labels))}")
        print(f"标准化: {'Per-Patient Z-Score' if use_zscore else '无'}")

    def _load_subject(self, mat_file: Path):
        """加载单个被试的 1D 数据"""
        with h5py.File(mat_file, 'r') as f:
            multidim_data = f['multidim_data'][()]
            seg_one_hot = f['seg_one_hot'][()]

            # 转置以适应 Python 的行优先顺序
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T  # -> (n_voxels, 351)

            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T  # -> (n_voxels, 102)

            # 截取指定维度的特征
            multidim_data = multidim_data[:, :self.feature_dim]

            # Per-Patient Z-Score 标准化
            if self.use_zscore:
                multidim_data = per_patient_zscore(multidim_data)

            # 从 one-hot 转换为类别标签
            labels = np.argmax(seg_one_hot, axis=1)

            self.all_data.append(multidim_data)
            self.all_labels.append(labels)
            self.subject_names.append(mat_file.stem)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]


class TestDataset(Dataset):
    """
    测试数据集 (单个被试)

    保存 3D 位置信息用于映射回 3D 体积

    Args:
        mat_file_1d: 1D MAT 文件路径
        mat_file_3d: 3D MAT 文件路径 (包含 region_mask 和 region_labels)
        feature_dim: 使用的特征维度数
        use_zscore: 是否使用 per-patient z-score 标准化
    """

    def __init__(
        self,
        mat_file_1d: Path,
        mat_file_3d: Path,
        feature_dim: int = 341,
        use_zscore: bool = True
    ):
        self.mat_file_1d = mat_file_1d
        self.mat_file_3d = mat_file_3d
        self.feature_dim = feature_dim

        # 加载 1D 数据
        with h5py.File(mat_file_1d, 'r') as f:
            multidim_data = f['multidim_data'][()]
            seg_one_hot = f['seg_one_hot'][()]

            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T

            multidim_data = multidim_data[:, :feature_dim]

            if use_zscore:
                self.features = per_patient_zscore(multidim_data)
            else:
                self.features = multidim_data.astype(np.float32)

            self.labels = np.argmax(seg_one_hot, axis=1)

        # 加载 3D mask
        with h5py.File(mat_file_3d, 'r') as f:
            self.region_mask = f['region_mask'][()]
            self.region_labels = f['region_labels'][()]

        print(f"测试数据: {len(self.features):,} 个体素")
        print(f"3D mask 形状: {self.region_mask.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_subject_key_1d(path: Path) -> str:
    """提取 1D 文件的被试名"""
    return path.stem


def get_subject_key_3d(path: Path) -> str:
    """提取 3D 文件的被试名"""
    return path.stem.replace('_3d_validated', '')


def create_train_val_test_datasets(
    data_dir_1d: str,
    data_dir_3d: str,
    test_subject_name: str,
    val_ratio: float = 0.1,
    feature_dim: int = 341,
    use_zscore: bool = True,
    random_seed: int = 666
) -> Tuple[Brain1D_Dataset, Brain1D_Dataset, TestDataset, List[str], List[str], str]:
    """
    创建训练、验证和测试数据集

    从非测试被试中划分出一部分作为验证集

    Args:
        data_dir_1d: 1D 数据目录
        data_dir_3d: 3D 数据目录
        test_subject_name: 测试被试名称 (如 'YHC_10_lnvguay')
        val_ratio: 验证集占非测试被试的比例 (默认 0.1 = 10%)
        feature_dim: 特征维度
        use_zscore: 是否使用 z-score 标准化
        random_seed: 随机种子

    Returns:
        (train_dataset, val_dataset, test_dataset,
         train_subject_names, val_subject_names, test_subject_name)
    """
    import random
    random.seed(random_seed)

    data_dir_1d = Path(data_dir_1d)
    data_dir_3d = Path(data_dir_3d)

    # 获取所有 1D 和 3D 文件
    files_1d = sorted(data_dir_1d.glob('*.mat'))
    files_3d = sorted(data_dir_3d.glob('*_3d_validated.mat'))

    print(f"找到 1D 文件: {len(files_1d)} 个")
    print(f"找到 3D 文件: {len(files_3d)} 个")

    # 构建索引
    idx_1d = {get_subject_key_1d(p): p for p in files_1d}
    idx_3d = {get_subject_key_3d(p): p for p in files_3d}

    # 验证测试被试存在
    if test_subject_name not in idx_1d:
        available = list(idx_1d.keys())[:5]
        raise ValueError(f"测试被试 '{test_subject_name}' 不存在。可用被试示例: {available}...")

    # 分离测试集
    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]

    # 除测试被试外的所有被试
    non_test_subjects = [name for name in idx_1d.keys() if name != test_subject_name]

    # 随机划分训练集和验证集
    random.shuffle(non_test_subjects)
    n_val = max(1, int(len(non_test_subjects) * val_ratio))

    val_subject_names = non_test_subjects[:n_val]
    train_subject_names = non_test_subjects[n_val:]

    train_files_1d = [idx_1d[name] for name in train_subject_names]
    val_files_1d = [idx_1d[name] for name in val_subject_names]

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_subject_names)} 个被试")
    print(f"  验证集: {len(val_subject_names)} 个被试 ({val_subject_names})")
    print(f"  测试集: 1 个被试 ({test_subject_name})")

    # 创建数据集
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        feature_dim=feature_dim,
        use_zscore=use_zscore
    )

    val_dataset = Brain1D_Dataset(
        val_files_1d,
        feature_dim=feature_dim,
        use_zscore=use_zscore
    )

    test_dataset = TestDataset(
        test_file_1d,
        test_file_3d,
        feature_dim=feature_dim,
        use_zscore=use_zscore
    )

    return (train_dataset, val_dataset, test_dataset,
            train_subject_names, val_subject_names, test_subject_name)


if __name__ == "__main__":
    print("Single MAT Loader 模块测试")
    print("=" * 50)

    # 测试 per_patient_zscore
    test_data = np.random.randn(1000, 341).astype(np.float32) * 10 + 5
    normalized = per_patient_zscore(test_data)
    print(f"标准化后均值: {normalized.mean(axis=0)[:3]}... (应接近 0)")
    print(f"标准化后标准差: {normalized.std(axis=0)[:3]}... (应接近 1)")
