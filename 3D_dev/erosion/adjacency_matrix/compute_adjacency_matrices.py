#!/usr/bin/env python3
"""
3D脑区邻接矩阵计算脚本

计算每个被试的102个脑区之间的3D空间邻接矩阵，用于与混淆矩阵进行Hadamard乘积分析。
邻接定义：两个脑区在3D空间中是否有接触的体素（6连通或26连通）。

用法:
    python compute_adjacency_matrices.py --data_dir /path/to/3d_validated --output_dir ./adjacency_matrices
"""

import os
import sys
import argparse
import logging
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import warnings

import numpy as np
import h5py
from tqdm import tqdm
from scipy import ndimage
from collections import Counter

warnings.filterwarnings('ignore')


def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """设置日志配置"""
    log_level = logging.INFO if verbose else logging.WARNING

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f'adjacency_computation_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 设置记录器
    logger = logging.getLogger('AdjacencyComputer')
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class AdjacencyMatrixComputer:
    """3D脑区邻接矩阵计算器"""

    def __init__(self, connectivity: int = 6, standard_matrix_size: int = 102, logger: Optional[logging.Logger] = None):
        """
        Args:
            connectivity: 连通性定义 (6, 18, 或 26)
            standard_matrix_size: 标准化矩阵大小 (默认102，对应标签0-101)
            logger: 日志记录器
        """
        self.connectivity = connectivity
        self.standard_matrix_size = standard_matrix_size
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # 定义邻接结构元素
        if connectivity == 6:
            # 6连通：面相邻
            self.structure = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            # 18连通：面和边相邻
            self.structure = ndimage.generate_binary_structure(3, 2)
        elif connectivity == 26:
            # 26连通：面、边和顶点相邻
            self.structure = ndimage.generate_binary_structure(3, 3)
        else:
            raise ValueError(f"不支持的连通性: {connectivity}，支持6、18或26")

        self.logger.info(f"使用{connectivity}连通性定义邻接关系")

    def load_subject_data(self, mat_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """加载被试数据 - 与ResNet数据加载方式一致，包含详细验证"""
        validation_info = {
            'file_path': str(mat_file),
            'file_size_mb': mat_file.stat().st_size / (1024**2),
            'loading_errors': [],
            'warnings': []
        }

        with h5py.File(mat_file, 'r') as f:
            # 记录文件内容
            validation_info['file_keys'] = list(f.keys())

            # 加载region_labels和region_mask
            region_labels = f['region_labels'][()]
            region_mask = f['region_mask'][()]

            # 记录原始数据信息
            validation_info['original_shapes'] = {
                'region_labels': region_labels.shape,
                'region_mask': region_mask.shape
            }
            validation_info['original_dtypes'] = {
                'region_labels': str(region_labels.dtype),
                'region_mask': str(region_mask.dtype)
            }

            self.logger.info(f"原始数据: labels={region_labels.shape}, mask={region_mask.shape}")

            # 处理标签格式（与ResNet一致）
            labels_transposed = False
            mask_transposed = False

            if region_labels.shape != (384, 336, 256):
                region_labels = region_labels.T
                labels_transposed = True
                validation_info['warnings'].append("region_labels已转置")

            if region_mask.shape != (384, 336, 256):
                region_mask = region_mask.T
                mask_transposed = True
                validation_info['warnings'].append("region_mask已转置")

            validation_info['transposed'] = {
                'region_labels': labels_transposed,
                'region_mask': mask_transposed
            }

            # 最终形状验证
            expected_shape = (384, 336, 256)
            if region_labels.shape != expected_shape:
                error_msg = f"region_labels形状错误: {region_labels.shape}, 期望: {expected_shape}"
                validation_info['loading_errors'].append(error_msg)
                raise ValueError(error_msg)

            if region_mask.shape != expected_shape:
                error_msg = f"region_mask形状错误: {region_mask.shape}, 期望: {expected_shape}"
                validation_info['loading_errors'].append(error_msg)
                raise ValueError(error_msg)

            # 转换为正确的数据类型
            region_labels = region_labels.astype(np.int16)
            region_mask = region_mask.astype(bool)

            validation_info['final_shapes'] = {
                'region_labels': region_labels.shape,
                'region_mask': region_mask.shape
            }
            validation_info['final_dtypes'] = {
                'region_labels': str(region_labels.dtype),
                'region_mask': str(region_mask.dtype)
            }

            return region_labels, region_mask, validation_info

    def validate_labels(self, region_labels: np.ndarray, region_mask: np.ndarray) -> Dict:
        """详细验证标签数据"""
        validation_info = {
            'mask_statistics': {},
            'label_statistics': {},
            'label_range_analysis': {},
            'data_quality_checks': {}
        }

        # 掩膜统计
        total_voxels = np.prod(region_mask.shape)
        valid_voxels = np.sum(region_mask)
        validation_info['mask_statistics'] = {
            'total_voxels': int(total_voxels),
            'valid_voxels': int(valid_voxels),
            'valid_percentage': float(valid_voxels / total_voxels * 100),
            'invalid_voxels': int(total_voxels - valid_voxels)
        }

        # 所有标签统计（包括无效区域）
        all_labels = region_labels.flatten()
        validation_info['label_statistics']['all_labels'] = {
            'min': int(np.min(all_labels)),
            'max': int(np.max(all_labels)),
            'unique_count': len(np.unique(all_labels)),
            'unique_values': np.unique(all_labels).tolist()
        }

        # 有效区域标签统计
        valid_labels = region_labels[region_mask]
        validation_info['label_statistics']['valid_labels'] = {
            'min': int(np.min(valid_labels)),
            'max': int(np.max(valid_labels)),
            'unique_count': len(np.unique(valid_labels)),
            'unique_values': sorted(np.unique(valid_labels).tolist())
        }

        # 分析标签范围
        unique_valid = np.unique(valid_labels)
        validation_info['label_range_analysis'] = {
            'min_label': int(np.min(unique_valid)),
            'max_label': int(np.max(unique_valid)),
            'label_span': int(np.max(unique_valid) - np.min(unique_valid) + 1),
            'actual_unique_count': len(unique_valid),
            'missing_labels_in_range': []
        }

        # 检查标签范围内的缺失标签
        min_label = int(np.min(unique_valid))
        max_label = int(np.max(unique_valid))
        expected_range = set(range(min_label, max_label + 1))
        actual_labels = set(unique_valid.tolist())
        missing_labels = sorted(expected_range - actual_labels)
        validation_info['label_range_analysis']['missing_labels_in_range'] = missing_labels

        # 判断标签格式
        if min_label == 0 and max_label <= 101:
            label_format = "0-101 (102类)"
        elif min_label == 1 and max_label <= 102:
            label_format = "1-102 (102类)"
        elif min_label == 0 and max_label <= 102:
            if 102 in unique_valid:
                label_format = "0-102 (103类)"
            else:
                label_format = "0-101范围但最大值≤102"
        else:
            label_format = f"自定义范围 {min_label}-{max_label}"

        validation_info['label_range_analysis']['detected_format'] = label_format

        # 每个标签的体素计数
        label_counts = {}
        for label in unique_valid:
            count = np.sum(valid_labels == label)
            label_counts[int(label)] = int(count)

        validation_info['label_statistics']['label_voxel_counts'] = label_counts

        # 数据质量检查
        quality_checks = {}

        # 检查是否有孤立的体素
        isolated_voxel_count = 0
        for label in unique_valid:
            label_mask = (region_labels == label) & region_mask
            if np.sum(label_mask) == 1:  # 只有一个体素
                isolated_voxel_count += 1

        quality_checks['isolated_single_voxel_labels'] = isolated_voxel_count

        # 检查标签连续性
        if missing_labels:
            quality_checks['has_missing_labels_in_range'] = True
            quality_checks['missing_label_count'] = len(missing_labels)
        else:
            quality_checks['has_missing_labels_in_range'] = False
            quality_checks['missing_label_count'] = 0

        # 检查异常大小的区域
        median_size = np.median(list(label_counts.values()))
        large_regions = {k: v for k, v in label_counts.items() if v > median_size * 10}
        small_regions = {k: v for k, v in label_counts.items() if v < median_size / 10}

        quality_checks['unusually_large_regions'] = large_regions
        quality_checks['unusually_small_regions'] = small_regions
        quality_checks['median_region_size'] = float(median_size)

        validation_info['data_quality_checks'] = quality_checks

        # 记录详细信息到日志
        self.logger.info(f"标签验证完成:")
        self.logger.info(f"  - 检测到标签格式: {label_format}")
        self.logger.info(f"  - 有效体素: {valid_voxels:,} ({valid_voxels/total_voxels*100:.1f}%)")
        self.logger.info(f"  - 唯一标签数: {len(unique_valid)}")
        self.logger.info(f"  - 标签范围: {min_label}-{max_label}")
        if missing_labels:
            self.logger.warning(f"  - 范围内缺失标签: {missing_labels}")

        return validation_info

    def compute_adjacency_matrix_optimized(self, region_labels: np.ndarray,
                                         region_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        优化的邻接矩阵计算

        返回:
            adjacency_matrix: (N, N) 邻接矩阵，对称 (N根据实际标签数量动态确定)
            stats: 统计信息字典
        """
        # 只考虑有效的脑组织区域
        masked_labels = region_labels.copy()
        masked_labels[~region_mask] = -1  # 无效区域标记为-1

        # 获取所有存在的标签
        unique_labels = np.unique(masked_labels)
        unique_labels = unique_labels[unique_labels >= 0]  # 排除-1

        self.logger.info(f"发现{len(unique_labels)}个有效脑区标签: {sorted(unique_labels)}")

        # 使用标准化矩阵大小
        matrix_size = self.standard_matrix_size
        max_label = int(np.max(unique_labels))

        # 验证标签是否超出标准范围
        if max_label >= matrix_size:
            self.logger.warning(f"最大标签{max_label}超出标准矩阵大小{matrix_size}，将自动扩展矩阵")
            matrix_size = max_label + 1

        self.logger.info(f"创建标准化{matrix_size}×{matrix_size}邻接矩阵 (标准大小: {self.standard_matrix_size})")
        self.logger.info(f"当前数据标签范围: 0-{max_label}, 缺失的标签对应行/列将为零")

        # 初始化标准大小的邻接矩阵
        adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=np.uint8)
        contact_counts = np.zeros((matrix_size, matrix_size), dtype=np.uint32)  # 接触体素数量

        # 为每个标签创建二值掩膜并检测邻接
        total_pairs = len(unique_labels) * (len(unique_labels) - 1) // 2

        with tqdm(total=total_pairs, desc="计算邻接关系") as pbar:
            for i, label1 in enumerate(unique_labels):
                # 创建当前标签的掩膜
                mask1 = (masked_labels == label1)

                # 膨胀操作找到边界邻接区域
                dilated_mask1 = ndimage.binary_dilation(mask1, structure=self.structure)

                for j, label2 in enumerate(unique_labels[i+1:], i+1):
                    # 创建第二个标签的掩膜
                    mask2 = (masked_labels == label2)

                    # 检查是否有重叠（邻接）
                    overlap = dilated_mask1 & mask2

                    if np.any(overlap):
                        # 计算接触体素数量
                        contact_count = np.sum(overlap)

                        # 设置邻接关系（对称）
                        adjacency_matrix[label1, label2] = 1
                        adjacency_matrix[label2, label1] = 1

                        # 记录接触体素数量
                        contact_counts[label1, label2] = contact_count
                        contact_counts[label2, label1] = contact_count

                        self.logger.debug(f"区域{label1}和{label2}邻接，接触体素数: {contact_count}")

                    pbar.update(1)

        # 统计信息
        total_adjacencies = np.sum(adjacency_matrix) // 2  # 除以2因为矩阵对称
        avg_contacts = np.mean(contact_counts[contact_counts > 0]) if total_adjacencies > 0 else 0

        stats = {
            'matrix_size': matrix_size,
            'total_regions': len(unique_labels),
            'total_adjacencies': int(total_adjacencies),
            'adjacency_density': float(total_adjacencies) / (len(unique_labels) * (len(unique_labels) - 1) / 2) if len(unique_labels) > 1 else 0,
            'avg_contact_voxels': float(avg_contacts),
            'max_contact_voxels': int(np.max(contact_counts)),
            'min_label': int(np.min(unique_labels)),
            'max_label': int(np.max(unique_labels)),
            'unique_labels': unique_labels.tolist(),
            'connectivity': self.connectivity,
            'standard_matrix_size': self.standard_matrix_size,
            'is_standardized': matrix_size == self.standard_matrix_size
        }

        self.logger.info(f"邻接关系统计: {total_adjacencies}个邻接对, "
                        f"密度: {stats['adjacency_density']:.3f}, "
                        f"平均接触体素: {avg_contacts:.1f}")

        return adjacency_matrix, contact_counts, stats

    def save_adjacency_data(self, adjacency_matrix: np.ndarray,
                           contact_counts: np.ndarray,
                           stats: Dict,
                           validation_info: Dict,
                           label_validation: Dict,
                           output_file: Path):
        """
        保存邻接矩阵数据到HDF5文件，包含完整的验证信息

        优化存储格式，支持高效的小块读取
        """
        with h5py.File(output_file, 'w') as f:
            # 创建组
            adj_group = f.create_group('adjacency')
            contact_group = f.create_group('contact_counts')
            meta_group = f.create_group('metadata')
            validation_group = f.create_group('validation')

            # 动态确定块大小
            matrix_size = adjacency_matrix.shape[0]
            chunk_size = min(51, matrix_size // 2 + 1)

            # 保存邻接矩阵（压缩存储）
            adj_group.create_dataset(
                'matrix',
                data=adjacency_matrix,
                compression='gzip',
                compression_opts=9,
                chunks=(chunk_size, chunk_size),
                dtype=np.uint8
            )

            # 保存接触计数矩阵
            contact_group.create_dataset(
                'counts',
                data=contact_counts,
                compression='gzip',
                compression_opts=9,
                chunks=(chunk_size, chunk_size),
                dtype=np.uint32
            )

            # 保存上三角形式（节省空间）
            adj_upper = np.triu(adjacency_matrix)
            contact_upper = np.triu(contact_counts)

            adj_group.create_dataset(
                'upper_triangle',
                data=adj_upper,
                compression='gzip',
                compression_opts=9
            )

            contact_group.create_dataset(
                'upper_triangle',
                data=contact_upper,
                compression='gzip',
                compression_opts=9
            )

            # 保存稀疏表示（便于分析）
            adj_pairs = np.where(adj_upper)
            adj_values = adj_upper[adj_pairs]
            contact_values = contact_upper[adj_pairs]

            sparse_group = f.create_group('sparse')
            sparse_group.create_dataset('row_indices', data=adj_pairs[0])
            sparse_group.create_dataset('col_indices', data=adj_pairs[1])
            sparse_group.create_dataset('adjacency_values', data=adj_values)
            sparse_group.create_dataset('contact_values', data=contact_values)

            # 保存计算统计信息
            for key, value in stats.items():
                if isinstance(value, (list, np.ndarray)):
                    meta_group.create_dataset(key, data=value)
                else:
                    meta_group.attrs[key] = value

            # 保存验证信息
            self._save_dict_to_group(validation_group, 'data_loading', validation_info)
            self._save_dict_to_group(validation_group, 'label_validation', label_validation)

            # 添加时间戳和版本信息
            meta_group.attrs['created_time'] = datetime.now().isoformat()
            meta_group.attrs['format_version'] = '2.0'
            meta_group.attrs['script_version'] = 'compute_adjacency_matrices_v2'

        self.logger.info(f"邻接矩阵数据已保存到: {output_file}")

    def _save_dict_to_group(self, group, name: str, data_dict: Dict):
        """递归保存字典到HDF5组"""
        subgroup = group.create_group(name)

        for key, value in data_dict.items():
            try:
                # 确保key是字符串（HDF5要求）
                str_key = str(key)

                if isinstance(value, dict):
                    self._save_dict_to_group(subgroup, str_key, value)
                elif isinstance(value, (list, np.ndarray)):
                    subgroup.create_dataset(str_key, data=value)
                elif isinstance(value, (int, float, str, bool)):
                    subgroup.attrs[str_key] = value
                else:
                    # 尝试转换为字符串
                    subgroup.attrs[str_key] = str(value)
            except Exception as e:
                self.logger.warning(f"无法保存 {key}: {e}")
                subgroup.attrs[f"{str(key)}_error"] = str(e)

    def save_detailed_report(self, subject_info: Dict, output_dir: Path):
        """保存每个被试的详细说明文件"""
        # 保留原始文件名，确保一一对应
        original_name = Path(subject_info['file_path']).stem  # 完整的原始文件名

        # JSON详细报告
        json_file = output_dir / f"{original_name}_adjacency_detailed_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(subject_info, f, indent=2, ensure_ascii=False)

        # Markdown易读报告
        md_file = output_dir / f"{original_name}_adjacency_report.md"
        self._generate_markdown_report(subject_info, md_file)

        self.logger.info(f"详细报告已保存: {json_file}")
        self.logger.info(f"易读报告已保存: {md_file}")

    def _generate_markdown_report(self, subject_info: Dict, output_file: Path):
        """生成Markdown格式的易读报告"""
        data_loading = subject_info.get('data_loading_validation', {})
        label_validation = subject_info.get('label_validation', {})
        adjacency_stats = subject_info.get('adjacency_statistics', {})

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 脑区邻接矩阵计算报告\n\n")

            # 基本信息
            f.write(f"## 📋 基本信息\n\n")
            f.write(f"- **文件路径**: `{data_loading.get('file_path', 'N/A')}`\n")
            f.write(f"- **文件大小**: {data_loading.get('file_size_mb', 0):.1f} MB\n")
            f.write(f"- **处理时间**: {subject_info.get('processing_time', 0):.1f} 秒\n")
            f.write(f"- **连通性**: {adjacency_stats.get('connectivity', 'N/A')}\n")
            f.write(f"- **生成时间**: {subject_info.get('timestamp', 'N/A')}\n\n")

            # 标签验证结果
            if label_validation:
                f.write(f"## 🏷️ 标签验证结果\n\n")

                label_stats = label_validation.get('label_statistics', {})
                if 'valid_labels' in label_stats:
                    valid_stats = label_stats['valid_labels']
                    f.write(f"### 标签范围信息\n")
                    f.write(f"- **标签范围**: {valid_stats.get('min', 'N/A')} - {valid_stats.get('max', 'N/A')}\n")
                    f.write(f"- **唯一标签数**: {valid_stats.get('unique_count', 'N/A')}\n")

                range_analysis = label_validation.get('label_range_analysis', {})
                if range_analysis:
                    f.write(f"- **检测到的格式**: {range_analysis.get('detected_format', 'N/A')}\n")
                    missing = range_analysis.get('missing_labels_in_range', [])
                    if missing:
                        f.write(f"- **范围内缺失标签**: {missing}\n")
                    f.write(f"\n")

                # 数据质量
                mask_stats = label_validation.get('mask_statistics', {})
                if mask_stats:
                    f.write(f"### 数据质量\n")
                    f.write(f"- **总体素数**: {mask_stats.get('total_voxels', 0):,}\n")
                    f.write(f"- **有效体素数**: {mask_stats.get('valid_voxels', 0):,}\n")
                    f.write(f"- **有效比例**: {mask_stats.get('valid_percentage', 0):.1f}%\n")

                quality_checks = label_validation.get('data_quality_checks', {})
                if quality_checks:
                    f.write(f"- **中位区域大小**: {quality_checks.get('median_region_size', 0):.0f} 体素\n")
                    if quality_checks.get('isolated_single_voxel_labels', 0) > 0:
                        f.write(f"- **⚠️ 孤立单体素区域**: {quality_checks.get('isolated_single_voxel_labels', 0)} 个\n")
                    f.write(f"\n")

            # 邻接矩阵统计
            if adjacency_stats:
                f.write(f"## 🔗 邻接矩阵统计\n\n")
                f.write(f"- **矩阵大小**: {adjacency_stats.get('matrix_size', 'N/A')} × {adjacency_stats.get('matrix_size', 'N/A')}\n")
                f.write(f"- **参与区域数**: {adjacency_stats.get('total_regions', 'N/A')}\n")
                f.write(f"- **邻接对数**: {adjacency_stats.get('total_adjacencies', 'N/A')}\n")
                f.write(f"- **邻接密度**: {adjacency_stats.get('adjacency_density', 0):.3f}\n")
                f.write(f"- **平均接触体素数**: {adjacency_stats.get('avg_contact_voxels', 0):.1f}\n")
                f.write(f"- **最大接触体素数**: {adjacency_stats.get('max_contact_voxels', 'N/A')}\n\n")

            # 存在的标签列表
            unique_labels = adjacency_stats.get('unique_labels', [])
            if unique_labels:
                f.write(f"## 📊 存在的脑区标签\n\n")
                f.write(f"总共 {len(unique_labels)} 个区域:\n\n")

                # 按10个一行显示
                for i in range(0, len(unique_labels), 10):
                    row_labels = unique_labels[i:i+10]
                    f.write(f"`{', '.join(map(str, row_labels))}`\n\n")

            # 警告和错误
            warnings = data_loading.get('warnings', [])
            errors = data_loading.get('loading_errors', [])

            if warnings or errors:
                f.write(f"## ⚠️ 注意事项\n\n")

                if warnings:
                    f.write(f"### 警告\n")
                    for warning in warnings:
                        f.write(f"- {warning}\n")
                    f.write(f"\n")

                if errors:
                    f.write(f"### 错误\n")
                    for error in errors:
                        f.write(f"- ❌ {error}\n")
                    f.write(f"\n")

            # 文件信息
            hdf5_file = subject_info.get('output_file', '')
            if hdf5_file:
                f.write(f"## 📁 输出文件\n\n")
                f.write(f"- **HDF5文件**: `{Path(hdf5_file).name}`\n")
                f.write(f"- **完整路径**: `{hdf5_file}`\n\n")

            f.write(f"---\n\n")
            f.write(f"*此报告由邻接矩阵计算系统自动生成*\n")


def find_mat_files(data_dir: Path) -> List[Path]:
    """查找MAT文件 - 与ResNet训练脚本一致"""
    # 首先尝试查找 3D validated 文件
    mat_files = sorted(data_dir.glob('subject*_3d_validated.mat'))

    if not mat_files:
        # 回退到通用MAT文件
        mat_files = sorted(data_dir.glob('*.mat'))

    if not mat_files:
        raise FileNotFoundError(f"在{data_dir}中未找到MAT文件")

    return mat_files


def main():
    parser = argparse.ArgumentParser(description='计算3D脑区邻接矩阵')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='3D验证数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='输出目录')
    parser.add_argument('--connectivity', type=int, default=6, choices=[6, 18, 26],
                      help='连通性定义 (6, 18, 或 26)')
    parser.add_argument('--standard_matrix_size', type=int, default=102,
                      help='标准化矩阵大小 (默认102，对应标签0-101)')
    parser.add_argument('--start_subject', type=int, default=1,
                      help='起始被试编号')
    parser.add_argument('--end_subject', type=int, default=38,
                      help='结束被试编号')
    parser.add_argument('--test_only', action='store_true',
                      help='仅测试单个被试')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='详细日志输出')

    args = parser.parse_args()

    # 设置路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir, args.verbose)
    logger.info(f"开始计算邻接矩阵")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"连通性: {args.connectivity}")
    logger.info(f"标准矩阵大小: {args.standard_matrix_size}")

    # 查找所有MAT文件
    try:
        mat_files = find_mat_files(data_dir)
        logger.info(f"发现{len(mat_files)}个MAT文件")
    except FileNotFoundError as e:
        logger.error(e)
        return 1

    # 初始化计算器
    computer = AdjacencyMatrixComputer(
        connectivity=args.connectivity,
        standard_matrix_size=args.standard_matrix_size,
        logger=logger
    )

    # 处理文件范围
    if args.test_only:
        # 测试模式：只处理第一个文件
        files_to_process = mat_files[:1]
        logger.info("测试模式：只处理第一个被试")
    else:
        # 根据被试编号筛选文件
        files_to_process = []
        for mat_file in mat_files:
            # 从文件名提取被试编号
            try:
                if 'subject' in mat_file.name:
                    subject_num = int(mat_file.name.split('subject')[1].split('_')[0])
                    if args.start_subject <= subject_num <= args.end_subject:
                        files_to_process.append(mat_file)
                else:
                    # 如果没有subject前缀，使用文件顺序
                    files_to_process.append(mat_file)
            except (ValueError, IndexError):
                logger.warning(f"无法解析文件名中的被试编号: {mat_file.name}")
                files_to_process.append(mat_file)

    logger.info(f"将处理{len(files_to_process)}个文件")

    # 批处理统计
    processing_stats = {
        'total_files': len(files_to_process),
        'successful': 0,
        'failed': 0,
        'processing_times': [],
        'start_time': datetime.now().isoformat()
    }

    # 处理每个文件
    for mat_file in tqdm(files_to_process, desc="处理被试"):
        start_time = time.time()

        try:
            logger.info(f"处理: {mat_file.name}")

            # 加载数据并获取验证信息
            region_labels, region_mask, data_loading_validation = computer.load_subject_data(mat_file)

            # 详细验证标签数据
            label_validation = computer.validate_labels(region_labels, region_mask)

            # 计算邻接矩阵
            adjacency_matrix, contact_counts, adjacency_stats = computer.compute_adjacency_matrix_optimized(
                region_labels, region_mask
            )

            # 生成输出文件名 - 基于原文件名
            original_name = mat_file.stem  # 保留原始文件名（不含扩展名）
            output_file = output_dir / f"{original_name}_adjacency_conn{args.connectivity}.h5"

            # 保存邻接矩阵数据（包含验证信息）
            computer.save_adjacency_data(
                adjacency_matrix, contact_counts, adjacency_stats,
                data_loading_validation, label_validation, output_file
            )

            # 记录处理时间
            process_time = time.time() - start_time

            # 汇总所有信息生成详细报告
            subject_info = {
                'file_path': str(mat_file),
                'output_file': str(output_file),
                'processing_time': process_time,
                'timestamp': datetime.now().isoformat(),
                'data_loading_validation': data_loading_validation,
                'label_validation': label_validation,
                'adjacency_statistics': adjacency_stats
            }

            # 保存每个被试的详细说明文件
            computer.save_detailed_report(subject_info, output_dir)

            processing_stats['processing_times'].append(process_time)
            processing_stats['successful'] += 1

            logger.info(f"✅ {mat_file.name} 处理成功 ({process_time:.1f}秒)")

            # 内存清理
            del region_labels, region_mask, adjacency_matrix, contact_counts
            gc.collect()

        except Exception as e:
            logger.error(f"❌ 处理{mat_file.name}时出错: {str(e)}")
            processing_stats['failed'] += 1
            continue

    # 保存批处理报告
    processing_stats['end_time'] = datetime.now().isoformat()
    processing_stats['avg_processing_time'] = np.mean(processing_stats['processing_times']) if processing_stats['processing_times'] else 0
    processing_stats['total_processing_time'] = sum(processing_stats['processing_times'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f'batch_adjacency_report_{timestamp}.json'

    with open(report_file, 'w') as f:
        json.dump(processing_stats, f, indent=2)

    # 最终报告
    logger.info("=" * 50)
    logger.info("邻接矩阵计算完成!")
    logger.info(f"成功处理: {processing_stats['successful']}/{processing_stats['total_files']}")
    logger.info(f"平均处理时间: {processing_stats['avg_processing_time']:.1f}秒")
    logger.info(f"总处理时间: {processing_stats['total_processing_time']:.1f}秒")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"批处理报告: {report_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())