#!/usr/bin/env python
# coding: utf-8

"""
数据加载和预处理工具
Data loading and preprocessing utilities
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# FreeSurfer标准标签
STANDARD_LABELS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 103
]


def create_label_mapping():
    """
    创建FreeSurfer标签到连续索引的映射

    Returns:
        forward_mapping: dict, FreeSurfer标签 -> 连续索引 (0-51)
        reverse_mapping: dict, 连续索引 -> FreeSurfer标签
    """
    forward_mapping = {original: continuous for continuous, original in enumerate(STANDARD_LABELS)}
    reverse_mapping = {continuous: original for continuous, original in enumerate(STANDARD_LABELS)}
    return forward_mapping, reverse_mapping


def load_and_process_subject_with_mask(subject_dir, include_background=True, exclude_features=None,
                                      forward_mapping=None):
    """
    加载并处理单个受试者数据，同时保存空间位置掩码

    Parameters:
    -----------
    subject_dir : str or Path
        受试者目录路径
    include_background : bool
        是否包含背景体素
    exclude_features : list of int
        要排除的特征索引列表
    forward_mapping : dict
        标签映射字典

    Returns:
    --------
    dict : 包含以下关键字段
        - 'features': 处理后的特征
        - 'labels': 处理后的标签
        - 'original_shape_3d': 原始3D形状
        - 'spatial_mask_3d': 3D布尔掩码，标记哪些体素被包含在训练中
        - 'flat_indices': 展平后被保留的体素在原始展平数组中的索引
        - 'affine': 仿射矩阵
        - 'header': NIfTI头信息
    """
    subject_dir = Path(subject_dir)
    balanced_dir = subject_dir / "balanced_output"

    data_4d_path = balanced_dir / "balanced_data_4d10000.nii.gz"
    label_3d_path = balanced_dir / "balanced_labels_3d10000.nii.gz"

    if not data_4d_path.exists() or not label_3d_path.exists():
        raise FileNotFoundError(f"找不到balanced数据文件：{subject_dir.name}")

    # 加载NIfTI数据
    img_4d = nib.load(data_4d_path)
    label_3d = nib.load(label_3d_path)

    data_4d = img_4d.get_fdata().astype(np.float32)
    labels_3d = label_3d.get_fdata().astype(np.int32)

    original_shape_3d = labels_3d.shape
    logger.info(f"  {subject_dir.name}: 4D{data_4d.shape}, 3D{labels_3d.shape}")

    # 展平数据（使用C顺序）
    n_voxels = np.prod(labels_3d.shape)
    n_modalities = data_4d.shape[3]

    features = data_4d.reshape(n_voxels, n_modalities)  # 使用默认C order
    labels_flat = labels_3d.flatten()  # 使用默认C order

    # 验证数据对应
    assert features.shape[0] == labels_flat.shape[0], f"特征和标签数量不匹配"

    # 标签映射
    if forward_mapping is not None:
        logger.info(f"  映射标签...")
        unique_orig = np.unique(labels_flat)
        labels_mapped = np.zeros_like(labels_flat)

        for orig_label in unique_orig:
            if orig_label in forward_mapping:
                mask = labels_flat == orig_label
                labels_mapped[mask] = forward_mapping[orig_label]
                count = np.sum(mask)
                if orig_label != 0:  # 不显示背景映射信息
                    logger.info(f"    {orig_label:3d} → {forward_mapping[orig_label]:2d} ({count:,} 体素)")

        labels_flat = labels_mapped

    # 创建空间掩码
    if not include_background:
        # 排除背景体素（标签=0）
        spatial_mask_flat = labels_flat != 0  # 1D掩码
        spatial_mask_3d = spatial_mask_flat.reshape(original_shape_3d)  # 3D掩码

        # 保留的体素索引
        flat_indices = np.where(spatial_mask_flat)[0]  # 在原始展平数组中的索引

        # 应用掩码
        features = features[spatial_mask_flat]
        labels_flat = labels_flat[spatial_mask_flat]

        logger.info(f"  排除背景后: {len(features):,} / {n_voxels:,} 体素 ({len(features)/n_voxels*100:.2f}%)")
    else:
        # 包含所有体素
        spatial_mask_3d = np.ones(original_shape_3d, dtype=bool)
        flat_indices = np.arange(n_voxels)  # 所有索引
        logger.info(f"  包含背景: {len(features):,} 体素")

    # 排除指定特征
    if exclude_features:
        keep_mask = np.ones(features.shape[1], dtype=bool)
        keep_mask[exclude_features] = False
        features = features[:, keep_mask]
        logger.info(f"  排除特征{exclude_features}后: {features.shape[1]} 个特征")

    return {
        'features': features,
        'labels': labels_flat,
        'subject_id': subject_dir.name,
        'n_voxels': len(features),
        'original_shape_3d': original_shape_3d,
        'spatial_mask_3d': spatial_mask_3d,
        'flat_indices': flat_indices,
        'affine': img_4d.affine,
        'header': img_4d.header
    }


def patient_wise_standardization(all_data):
    """
    Patient-wise标准化

    Parameters:
    -----------
    all_data : list of dict
        所有受试者的数据列表

    Returns:
    --------
    all_data : list of dict
        标准化后的数据
    scalers_info : dict
        每个受试者的标准化参数
    """
    scalers_info = {}

    for data in all_data:
        subject_id = data['subject_id']
        features = data['features']

        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1.0, std)

        features_std = (features - mean) / std
        data['features'] = features_std.astype(np.float32)

        scalers_info[subject_id] = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }

        logger.info(f"  {subject_id}: 标准化完成")

    return all_data, scalers_info


def load_balanced_dataset_with_spatial_info(root_dir, include_background=True, exclude_features=None):
    """
    加载balanced数据集并保存空间信息

    Parameters:
    -----------
    root_dir : str or Path
        数据集根目录
    include_background : bool
        是否包含背景体素
    exclude_features : list of int
        要排除的特征索引

    Returns:
    --------
    all_data : list of dict
        所有受试者的数据
    scalers_info : dict
        标准化参数
    """
    root_dir = Path(root_dir)
    forward_mapping, reverse_mapping = create_label_mapping()

    logger.info(f"标签映射: {len(STANDARD_LABELS)}个类别 → 0-{len(STANDARD_LABELS)-1}连续值")

    # 查找受试者
    subject_dirs = sorted([d for d in root_dir.iterdir()
                          if d.is_dir() and d.name.startswith("FOR_")])

    valid_subjects = []
    for subject_dir in subject_dirs:
        balanced_dir = subject_dir / "balanced_output"
        data_file = balanced_dir / "balanced_data_4d10000.nii.gz"
        label_file = balanced_dir / "balanced_labels_3d10000.nii.gz"
        if data_file.exists() and label_file.exists():
            valid_subjects.append(subject_dir)

    logger.info(f"找到 {len(valid_subjects)} 个有balanced数据的受试者")
    logger.info(f"包含背景: {'是' if include_background else '否'}")

    # 加载所有数据（保存空间信息）
    all_data = []

    for subject_dir in tqdm(valid_subjects, desc="加载数据"):
        try:
            data = load_and_process_subject_with_mask(
                subject_dir,
                include_background=include_background,
                exclude_features=exclude_features,
                forward_mapping=forward_mapping
            )
            all_data.append(data)

        except Exception as e:
            logger.error(f"加载 {subject_dir.name} 失败: {e}")
            continue

    # Patient-wise标准化
    logger.info(f"\n🔄 Patient-wise标准化...")
    all_data, scalers_info = patient_wise_standardization(all_data)

    return all_data, scalers_info
