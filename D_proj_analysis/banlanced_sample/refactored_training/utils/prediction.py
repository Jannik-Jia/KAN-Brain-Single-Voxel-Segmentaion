#!/usr/bin/env python
# coding: utf-8

"""
预测和3D重建工具
Prediction and 3D reconstruction utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import logging

logger = logging.getLogger(__name__)


def predictions_to_3d_volume(predictions, spatial_info, include_background=True,
                            background_class=0):
    """
    将预测结果还原为3D softmax volume

    Parameters:
    -----------
    predictions : np.ndarray
        模型预测的softmax概率，形状 (n_valid_voxels, n_classes)
    spatial_info : dict
        包含空间信息的字典
    include_background : bool
        训练时是否包含了背景
    background_class : int
        背景类别索引

    Returns:
    --------
    volume_3d : np.ndarray
        3D softmax volume，形状 (X, Y, Z, n_classes)
    """
    original_shape_3d = spatial_info['original_shape_3d']
    spatial_mask_3d = spatial_info['spatial_mask_3d']
    flat_indices = spatial_info['flat_indices']

    n_classes = predictions.shape[1]

    # 创建完整的3D softmax volume
    volume_3d = np.zeros((*original_shape_3d, n_classes), dtype=np.float32)

    if include_background:
        # 如果训练时包含了背景，直接映射回去
        volume_flat = volume_3d.reshape(-1, n_classes)  # 展平到 (total_voxels, n_classes)
        volume_flat[flat_indices] = predictions  # 直接赋值

    else:
        # 如果训练时排除了背景，需要特殊处理
        # 1. 非背景区域：使用预测结果
        volume_flat = volume_3d.reshape(-1, n_classes)
        volume_flat[flat_indices] = predictions

        # 2. 背景区域：设置为背景类概率=1，其他类概率=0
        background_indices = np.where(~spatial_mask_3d.flatten())[0]
        volume_flat[background_indices, background_class] = 1.0  # 背景类概率=1

    logger.info(f"  3D softmax volume形状: {volume_3d.shape}")
    logger.info(f"  概率总和检查: min={np.sum(volume_3d, axis=-1).min():.6f}, "
                f"max={np.sum(volume_3d, axis=-1).max():.6f}")

    return volume_3d


def predict_and_save_3d_softmax(model, test_data_info, device, output_dir,
                                include_background=True, batch_size=8192,
                                class_labels=None):
    """
    预测测试集并保存3D softmax volume

    Parameters:
    -----------
    model : nn.Module
        训练好的模型
    test_data_info : dict
        测试集数据信息（来自load_and_process_subject_with_mask）
    device : torch.device
        设备
    output_dir : Path
        输出目录
    include_background : bool
        是否包含背景
    batch_size : int
        批大小
    class_labels : list
        类别标签列表

    Returns:
    --------
    volume_3d : np.ndarray
        3D softmax volume
    softmax_path : Path
        保存的文件路径
    """
    logger.info(f"\n🔮 预测测试集并保存3D softmax volume...")

    test_features = test_data_info['features']
    test_labels = test_data_info['labels']
    subject_id = test_data_info['subject_id']

    model.eval()
    all_predictions = []

    # 批量预测
    n_samples = len(test_features)

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, batch_size), desc="预测中"):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_X = torch.FloatTensor(test_features[start_idx:end_idx]).to(device)

            # 前向传播得到logits
            logits = model(batch_X)

            # 转换为softmax概率
            probabilities = F.softmax(logits, dim=1)

            all_predictions.append(probabilities.cpu().numpy())

    # 合并所有预测结果
    predictions = np.vstack(all_predictions)  # (n_valid_voxels, n_classes)

    logger.info(f"  预测完成: {predictions.shape}")
    logger.info(f"  概率范围: [{predictions.min():.6f}, {predictions.max():.6f}]")

    # 还原为3D volume
    volume_3d = predictions_to_3d_volume(
        predictions, test_data_info,
        include_background=include_background
    )

    # 保存为NIfTI文件
    bg_str = 'incl' if include_background else 'excl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 使用原始的仿射矩阵和头信息
    affine = test_data_info['affine']
    header = test_data_info['header'].copy()

    # 更新头信息（4D数据）
    header.set_data_shape(volume_3d.shape)
    header.set_data_dtype(np.float32)

    # 创建NIfTI图像
    softmax_img = nib.Nifti1Image(volume_3d, affine, header)

    # 保存文件
    softmax_path = output_dir / f"test_softmax_3d_{subject_id}_bg_{bg_str}_{timestamp}.nii.gz"
    nib.save(softmax_img, softmax_path)

    logger.info(f"💾 3D softmax已保存: {softmax_path}")
    logger.info(f"  文件大小: {softmax_path.stat().st_size / (1024**2):.2f} MB")

    # 额外保存一些有用信息
    info_dict = {
        'subject_id': subject_id,
        'original_shape_3d': test_data_info['original_shape_3d'],
        'softmax_shape': volume_3d.shape,
        'include_background': include_background,
        'n_valid_voxels': len(test_features),
        'n_classes': predictions.shape[1],
        'timestamp': timestamp,
        'class_names': class_labels if class_labels else list(range(predictions.shape[1])),
        'prediction_stats': {
            'min_prob': float(predictions.min()),
            'max_prob': float(predictions.max()),
            'mean_prob': float(predictions.mean())
        }
    }

    info_path = output_dir / f"test_softmax_info_{subject_id}_bg_{bg_str}_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=2)

    logger.info(f"💾 预测信息已保存: {info_path}")

    return volume_3d, softmax_path
