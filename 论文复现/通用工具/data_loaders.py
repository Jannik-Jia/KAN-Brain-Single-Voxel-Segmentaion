#!/usr/bin/env python3
"""
通用数据加载器模块
支持多种标准化方式：Per-Patient Z-Score / Global StandardScaler

使用方法：
    from data_loaders import Brain1D_Dataset, TestDataset, NormMode

    # Per-Patient Z-Score 模式（推荐）
    train_dataset = Brain1D_Dataset(train_files, norm_mode=NormMode.PER_PATIENT_ZSCORE)
    test_dataset = TestDataset(test_1d, test_3d, norm_mode=NormMode.PER_PATIENT_ZSCORE)

    # Global StandardScaler 模式
    train_dataset = Brain1D_Dataset(train_files, norm_mode=NormMode.GLOBAL_SCALER)
    test_dataset = TestDataset(test_1d, test_3d, norm_mode=NormMode.GLOBAL_SCALER,
                               scaler=train_dataset.scaler)
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Optional, Union
from enum import Enum
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ==================== 标准化模式枚举 ====================

class NormMode(Enum):
    """标准化模式"""
    PER_PATIENT_ZSCORE = "per_patient_zscore"      # 每个患者独立逐通道z-score
    GLOBAL_SCALER = "global_standard_scaler"       # 全局StandardScaler
    NONE = "none"                                   # 不进行标准化


# ==================== 标准化函数 ====================

def per_patient_zscore(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    对单个患者的数据进行逐通道z-score标准化

    每个通道独立计算均值和标准差，不跨患者共享统计量。
    适用于患者间特征分布差异较大的场景。

    Args:
        data: (n_voxels, n_channels) 单个患者的特征数据
        epsilon: 防止除零的小常数

    Returns:
        标准化后的数据 (n_voxels, n_channels)

    Example:
        >>> patient_data = np.random.randn(10000, 341)
        >>> normalized = per_patient_zscore(patient_data)
        >>> print(normalized.mean(axis=0))  # 接近 0
        >>> print(normalized.std(axis=0))   # 接近 1
    """
    # 计算每个通道的均值和标准差
    mean = np.mean(data, axis=0, keepdims=True)  # (1, n_channels)
    std = np.std(data, axis=0, keepdims=True)    # (1, n_channels)

    # 防止除零
    std = np.where(std < epsilon, epsilon, std)

    # z-score标准化
    normalized = (data - mean) / std

    return normalized.astype(np.float32)


# ==================== 训练数据集类 ====================

class Brain1D_Dataset(Dataset):
    """
    脑部1D体素分类数据集

    支持两种标准化模式：
    - PER_PATIENT_ZSCORE: 每个患者独立进行逐通道z-score标准化
    - GLOBAL_SCALER: 所有训练数据拟合一个全局StandardScaler

    Args:
        mat_files: 1D MAT文件路径列表
        norm_mode: 标准化模式 (NormMode枚举)
        feature_dim: 使用的特征维度数 (默认341，截取multidim_data前341维)
        scaler: 外部传入的StandardScaler (仅GLOBAL_SCALER模式有效)

    Attributes:
        all_data: (N, feature_dim) 所有体素的特征数据
        all_labels: (N,) 所有体素的标签
        scaler: StandardScaler对象 (仅GLOBAL_SCALER模式)
        norm_mode: 当前使用的标准化模式

    Example:
        >>> from data_loaders import Brain1D_Dataset, NormMode
        >>> train_files = list(Path("./data").glob("*.mat"))
        >>> dataset = Brain1D_Dataset(train_files, norm_mode=NormMode.PER_PATIENT_ZSCORE)
        >>> print(f"样本数: {len(dataset)}, 特征维度: {dataset.all_data.shape[1]}")
    """

    def __init__(
        self,
        mat_files: List[Path],
        norm_mode: NormMode = NormMode.PER_PATIENT_ZSCORE,
        feature_dim: int = 341,
        scaler: Optional[StandardScaler] = None
    ):
        self.mat_files = mat_files
        self.norm_mode = norm_mode
        self.feature_dim = feature_dim
        self.scaler = scaler

        # 加载所有数据
        self.all_data = []
        self.all_labels = []

        norm_desc = {
            NormMode.PER_PATIENT_ZSCORE: "Per-Patient Z-Score",
            NormMode.GLOBAL_SCALER: "Global StandardScaler",
            NormMode.NONE: "无标准化"
        }
        print(f"加载{len(mat_files)}个被试的1D数据（{norm_desc[norm_mode]}）...")

        for mat_file in tqdm(mat_files, desc="Loading subjects"):
            self._load_subject(mat_file)

        # 转换为numpy数组
        self.all_data = np.vstack(self.all_data).astype(np.float32)
        self.all_labels = np.concatenate(self.all_labels).astype(np.int64)

        # Global Scaler 模式：在合并后统一标准化
        if norm_mode == NormMode.GLOBAL_SCALER:
            if self.scaler is None:
                print("拟合全局 StandardScaler...")
                self.scaler = StandardScaler()
                self.all_data = self.scaler.fit_transform(self.all_data).astype(np.float32)
            else:
                print("应用外部传入的 StandardScaler...")
                self.all_data = self.scaler.transform(self.all_data).astype(np.float32)

        # 打印统计信息
        print(f"总样本数: {len(self.all_data):,}")
        print(f"特征维度: {self.all_data.shape[1]}")
        print(f"类别数: {len(np.unique(self.all_labels))}")
        print(f"标准化模式: {norm_mode.value}")

    def _load_subject(self, mat_file: Path):
        """加载单个被试的1D数据"""
        with h5py.File(mat_file, 'r') as f:
            # 加载1D数据
            multidim_data = f['multidim_data'][()]  # (351, n_voxels) or (n_voxels, 351)
            seg_one_hot = f['seg_one_hot'][()]      # (102, n_voxels) or (n_voxels, 102)

            # 转置以适应Python的行优先顺序
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T     # -> (n_voxels, 351)

            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T         # -> (n_voxels, 102)

            # 截取指定维度的特征
            multidim_data = multidim_data[:, :self.feature_dim]

            # Per-Patient Z-Score：在加载时就对每个患者独立标准化
            if self.norm_mode == NormMode.PER_PATIENT_ZSCORE:
                multidim_data = per_patient_zscore(multidim_data)

            # 从one-hot转换为类别标签
            labels = np.argmax(seg_one_hot, axis=1)  # (n_voxels,)

            # 添加到总数据中
            self.all_data.append(multidim_data)
            self.all_labels.append(labels)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

    def get_norm_config(self) -> dict:
        """返回标准化配置信息，用于实验记录"""
        config = {
            "type": self.norm_mode.value,
            "params": {}
        }

        if self.norm_mode == NormMode.PER_PATIENT_ZSCORE:
            config["params"] = {
                "epsilon": 1e-8,
                "fit_scope": "per_patient"
            }
        elif self.norm_mode == NormMode.GLOBAL_SCALER:
            config["params"] = {
                "epsilon": None,
                "fit_scope": "train_global"
            }

        return config


# ==================== 测试数据集类 ====================

class TestDataset(Dataset):
    """
    脑部1D体素分类测试数据集

    支持两种标准化模式，并保存3D位置信息用于映射回3D体积。

    Args:
        mat_file_1d: 1D MAT文件路径（用于获取特征数据）
        mat_file_3d: 3D MAT文件路径（用于获取3D mask）
        norm_mode: 标准化模式 (NormMode枚举)
        feature_dim: 使用的特征维度数 (默认341)
        scaler: StandardScaler对象 (仅GLOBAL_SCALER模式需要)

    Attributes:
        features: (N, feature_dim) 体素特征
        labels: (N,) 体素标签
        region_mask: (384, 336, 256) 3D脑区掩膜
        region_labels: (384, 336, 256) 3D脑区标签

    Example:
        >>> test_dataset = TestDataset(
        ...     test_1d_file, test_3d_file,
        ...     norm_mode=NormMode.PER_PATIENT_ZSCORE
        ... )
        >>> print(f"测试体素数: {len(test_dataset)}")
    """

    def __init__(
        self,
        mat_file_1d: Path,
        mat_file_3d: Path,
        norm_mode: NormMode = NormMode.PER_PATIENT_ZSCORE,
        feature_dim: int = 341,
        scaler: Optional[StandardScaler] = None
    ):
        self.mat_file_1d = mat_file_1d
        self.mat_file_3d = mat_file_3d
        self.norm_mode = norm_mode
        self.feature_dim = feature_dim
        self.scaler = scaler

        # 加载1D数据
        with h5py.File(mat_file_1d, 'r') as f:
            multidim_data = f['multidim_data'][()]
            seg_one_hot = f['seg_one_hot'][()]

            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T

            # 截取指定维度的特征
            multidim_data = multidim_data[:, :feature_dim]

            # 根据标准化模式处理
            if norm_mode == NormMode.PER_PATIENT_ZSCORE:
                self.features = per_patient_zscore(multidim_data)
            elif norm_mode == NormMode.GLOBAL_SCALER:
                if scaler is None:
                    raise ValueError("GLOBAL_SCALER模式需要传入训练时的scaler")
                self.features = scaler.transform(multidim_data).astype(np.float32)
            else:
                self.features = multidim_data.astype(np.float32)

            self.labels = np.argmax(seg_one_hot, axis=1)

        # 加载3D mask（用于映射）
        with h5py.File(mat_file_3d, 'r') as f:
            region_mask = f['region_mask'][()]
            region_labels = f['region_labels'][()]

            # 严格形状验证
            assert region_mask.shape == (384, 336, 256), \
                f"region_mask 形状不符合预期 (384, 336, 256)，实际为 {region_mask.shape}"

            assert region_labels.shape == (384, 336, 256), \
                f"region_labels 形状不符合预期 (384, 336, 256)，实际为 {region_labels.shape}"

            self.region_mask = region_mask
            self.region_labels = region_labels

        norm_desc = {
            NormMode.PER_PATIENT_ZSCORE: "Per-Patient Z-Score",
            NormMode.GLOBAL_SCALER: "Global StandardScaler",
            NormMode.NONE: "无标准化"
        }
        print(f"测试数据: {len(self.features):,} 个体素（{norm_desc[norm_mode]}）")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def verify_label_consistency(self) -> bool:
        """
        验证1D与3D标签一致性

        Returns:
            bool: 标签是否一致
        """
        mask = self.region_mask.astype(bool)
        labels_3d = self.region_labels[mask]
        labels_1d = self.labels

        if len(labels_1d) != len(labels_3d):
            print(f"❌ 标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")
            return False

        if np.array_equal(labels_1d, labels_3d):
            print(f"✅ 标签一致性验证通过：{len(labels_1d):,}个体素完全匹配")
            return True
        else:
            n_mismatch = np.sum(labels_1d != labels_3d)
            print(f"❌ 标签不匹配: {n_mismatch}/{len(labels_1d)} ({100*n_mismatch/len(labels_1d):.2f}%)")
            return False


# ==================== 辅助函数 ====================

def get_subject_key_1d(path: Path) -> str:
    """提取1D文件的被试名 e.g. 'ODP_01_qhlazec.mat' -> 'ODP_01_qhlazec'"""
    return path.stem


def get_subject_key_3d(path: Path) -> str:
    """提取3D文件的被试名 e.g. 'ODP_01_qhlazec_3d_validated.mat' -> 'ODP_01_qhlazec'"""
    return path.stem.replace('_3d_validated', '')


def build_subject_index(
    files_1d: List[Path],
    files_3d: List[Path]
) -> tuple:
    """
    构建被试名到文件路径的索引，并验证一致性

    Args:
        files_1d: 1D文件路径列表
        files_3d: 3D文件路径列表

    Returns:
        (idx_1d, idx_3d, subject_names): 两个索引字典和排序后的被试名列表

    Raises:
        AssertionError: 如果被试集合不一致

    Example:
        >>> idx_1d, idx_3d, names = build_subject_index(files_1d, files_3d)
        >>> for name in names:
        ...     train_file = idx_1d[name]
        ...     test_file = idx_3d[name]
    """
    idx_1d = {get_subject_key_1d(p): p for p in files_1d}
    idx_3d = {get_subject_key_3d(p): p for p in files_3d}

    subjects_1d = set(idx_1d.keys())
    subjects_3d = set(idx_3d.keys())

    assert subjects_1d == subjects_3d, \
        f"1D和3D被试集合不一致！差异: {subjects_1d ^ subjects_3d}"

    subject_names = sorted(idx_1d.keys())

    print(f"被试索引构建完成: {len(subject_names)} 个被试")

    return idx_1d, idx_3d, subject_names


# ==================== 便捷工厂函数 ====================

def create_datasets(
    data_dir_1d: Union[str, Path],
    data_dir_3d: Union[str, Path],
    test_subject_idx: int,
    norm_mode: NormMode = NormMode.PER_PATIENT_ZSCORE,
    feature_dim: int = 341
) -> tuple:
    """
    便捷函数：创建训练和测试数据集

    Args:
        data_dir_1d: 1D数据目录
        data_dir_3d: 3D数据目录
        test_subject_idx: 测试被试索引 (1-based)
        norm_mode: 标准化模式
        feature_dim: 特征维度

    Returns:
        (train_dataset, test_dataset, train_subject_names, test_subject_name)

    Example:
        >>> train_ds, test_ds, train_names, test_name = create_datasets(
        ...     "./data/1D", "./data/3D", test_subject_idx=38,
        ...     norm_mode=NormMode.PER_PATIENT_ZSCORE
        ... )
    """
    data_dir_1d = Path(data_dir_1d)
    data_dir_3d = Path(data_dir_3d)

    # 获取文件列表
    files_1d = list(data_dir_1d.glob('*.mat'))
    files_3d = list(data_dir_3d.glob('*_3d_validated.mat'))

    print(f"找到 1D 文件: {len(files_1d)} 个")
    print(f"找到 3D 文件: {len(files_3d)} 个")

    # 构建索引
    idx_1d, idx_3d, subject_names = build_subject_index(files_1d, files_3d)

    # 选择测试被试
    test_idx = test_subject_idx - 1
    if test_idx >= len(subject_names) or test_idx < 0:
        raise ValueError(f"测试被试索引 {test_subject_idx} 超出范围 [1, {len(subject_names)}]")

    test_subject_name = subject_names[test_idx]
    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]

    # 训练集：除测试被试外的所有被试
    train_subject_names = [name for name in subject_names if name != test_subject_name]
    train_files_1d = [idx_1d[name] for name in train_subject_names]

    print(f"训练集: {len(train_files_1d)} 个被试")
    print(f"测试被试: {test_subject_name}")

    # 创建训练数据集
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        norm_mode=norm_mode,
        feature_dim=feature_dim
    )

    # 创建测试数据集
    scaler = train_dataset.scaler if norm_mode == NormMode.GLOBAL_SCALER else None
    test_dataset = TestDataset(
        test_file_1d,
        test_file_3d,
        norm_mode=norm_mode,
        feature_dim=feature_dim,
        scaler=scaler
    )

    return train_dataset, test_dataset, train_subject_names, test_subject_name


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("数据加载器模块测试")
    print("=" * 60)

    # 测试 per_patient_zscore 函数
    print("\n测试 per_patient_zscore 函数:")
    test_data = np.random.randn(1000, 341).astype(np.float32) * 10 + 5
    normalized = per_patient_zscore(test_data)
    print(f"  原始数据均值: {test_data.mean(axis=0)[:3]}...")
    print(f"  标准化后均值: {normalized.mean(axis=0)[:3]}...")
    print(f"  标准化后标准差: {normalized.std(axis=0)[:3]}...")

    # 打印可用的标准化模式
    print("\n可用的标准化模式:")
    for mode in NormMode:
        print(f"  - {mode.name}: {mode.value}")

    print("\n" + "=" * 60)
    print("模块加载成功！")
    print("=" * 60)
