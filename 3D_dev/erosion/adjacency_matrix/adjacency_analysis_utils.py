#!/usr/bin/env python3
"""
邻接矩阵分析工具

提供高效的邻接矩阵读取、分析和与混淆矩阵计算Hadamard乘积的功能。
支持小块读取，优化IO效率。
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
import pandas as pd


class AdjacencyMatrixReader:
    """邻接矩阵高效读取器"""

    def __init__(self, adjacency_file: Path):
        """
        Args:
            adjacency_file: 邻接矩阵HDF5文件路径
        """
        self.adjacency_file = adjacency_file
        self.logger = logging.getLogger(self.__class__.__name__)

        # 验证文件存在
        if not adjacency_file.exists():
            raise FileNotFoundError(f"邻接矩阵文件不存在: {adjacency_file}")

        # 缓存元数据
        self._metadata = None
        self._load_metadata()

    def _load_metadata(self):
        """加载元数据"""
        with h5py.File(self.adjacency_file, 'r') as f:
            meta_group = f['metadata']
            self._metadata = {}

            # 读取属性
            for key in meta_group.attrs:
                self._metadata[key] = meta_group.attrs[key]

            # 读取数据集
            for key in meta_group.keys():
                self._metadata[key] = meta_group[key][()]

    @property
    def metadata(self) -> Dict:
        """获取元数据"""
        return self._metadata.copy()

    def load_full_adjacency_matrix(self) -> np.ndarray:
        """加载完整的邻接矩阵"""
        with h5py.File(self.adjacency_file, 'r') as f:
            return f['adjacency/matrix'][()]

    def load_adjacency_submatrix(self, row_indices: List[int],
                                col_indices: List[int]) -> np.ndarray:
        """
        加载邻接矩阵的子矩阵（高效小块读取）

        Args:
            row_indices: 行索引列表
            col_indices: 列索引列表

        Returns:
            子矩阵 (len(row_indices), len(col_indices))
        """
        with h5py.File(self.adjacency_file, 'r') as f:
            adjacency_matrix = f['adjacency/matrix']

            # 使用fancy indexing进行高效读取
            submatrix = adjacency_matrix[np.ix_(row_indices, col_indices)]
            return submatrix

    def load_contact_counts(self) -> np.ndarray:
        """加载接触计数矩阵"""
        with h5py.File(self.adjacency_file, 'r') as f:
            return f['contact_counts/counts'][()]

    def load_sparse_adjacency(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载稀疏格式的邻接矩阵

        Returns:
            row_indices, col_indices, adjacency_values, contact_values
        """
        with h5py.File(self.adjacency_file, 'r') as f:
            sparse_group = f['sparse']
            return (
                sparse_group['row_indices'][()],
                sparse_group['col_indices'][()],
                sparse_group['adjacency_values'][()],
                sparse_group['contact_values'][()]
            )

    def get_adjacency_pairs(self) -> List[Tuple[int, int]]:
        """获取所有邻接的区域对"""
        row_indices, col_indices, _, _ = self.load_sparse_adjacency()
        return list(zip(row_indices, col_indices))

    def get_region_neighbors(self, region_id: int) -> List[int]:
        """获取指定区域的所有邻接区域"""
        adjacency_matrix = self.load_full_adjacency_matrix()
        neighbors = np.where(adjacency_matrix[region_id] == 1)[0]
        return neighbors.tolist()


class ConfusionAdjacencyAnalyzer:
    """混淆矩阵与邻接矩阵分析器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compute_hadamard_product(self,
                                confusion_matrix: np.ndarray,
                                adjacency_matrix: np.ndarray,
                                normalize: bool = True) -> np.ndarray:
        """
        计算混淆矩阵与邻接矩阵的Hadamard乘积

        Args:
            confusion_matrix: 混淆矩阵 (102, 102)
            adjacency_matrix: 邻接矩阵 (102, 102)
            normalize: 是否对混淆矩阵进行归一化

        Returns:
            Hadamard乘积结果
        """
        if confusion_matrix.shape != adjacency_matrix.shape:
            raise ValueError(f"矩阵维度不匹配: {confusion_matrix.shape} vs {adjacency_matrix.shape}")

        # 归一化混淆矩阵（按行）
        if normalize:
            row_sums = np.sum(confusion_matrix, axis=1, keepdims=True)
            # 避免除零
            row_sums[row_sums == 0] = 1
            normalized_confusion = confusion_matrix / row_sums
        else:
            normalized_confusion = confusion_matrix

        # Hadamard乘积
        hadamard_product = normalized_confusion * adjacency_matrix.astype(float)

        return hadamard_product

    def find_high_confusion_adjacency_pairs(self,
                                          confusion_matrix: np.ndarray,
                                          adjacency_matrix: np.ndarray,
                                          threshold: float = 0.1) -> List[Tuple[int, int, float]]:
        """
        找到高混淆×高邻接的区域对

        Args:
            confusion_matrix: 混淆矩阵
            adjacency_matrix: 邻接矩阵
            threshold: 混淆值阈值

        Returns:
            [(region1, region2, confusion_value), ...] 按混淆值降序排列
        """
        hadamard_product = self.compute_hadamard_product(confusion_matrix, adjacency_matrix)

        # 找到非零且超过阈值的位置
        high_confusion_pairs = []

        for i in range(hadamard_product.shape[0]):
            for j in range(hadamard_product.shape[1]):
                if i != j and hadamard_product[i, j] > threshold:
                    high_confusion_pairs.append((i, j, hadamard_product[i, j]))

        # 按混淆值降序排列
        high_confusion_pairs.sort(key=lambda x: x[2], reverse=True)

        return high_confusion_pairs

    def compute_rank_correlation(self,
                               confusion_matrix: np.ndarray,
                               adjacency_matrix: np.ndarray) -> Tuple[float, float]:
        """
        计算混淆矩阵与邻接矩阵的rank相关性

        Returns:
            spearman_correlation, p_value
        """
        # 获取上三角部分（排除对角线）
        triu_indices = np.triu_indices_from(confusion_matrix, k=1)

        confusion_values = confusion_matrix[triu_indices]
        adjacency_values = adjacency_matrix[triu_indices]

        # 计算Spearman相关性
        correlation, p_value = stats.spearmanr(confusion_values, adjacency_values)

        return correlation, p_value

    def analyze_adjacency_confusion_relationship(self,
                                               confusion_matrix: np.ndarray,
                                               adjacency_matrix: np.ndarray,
                                               region_names: Optional[List[str]] = None) -> Dict:
        """
        全面分析邻接性与混淆性的关系

        Returns:
            分析结果字典
        """
        # 基本统计
        hadamard_product = self.compute_hadamard_product(confusion_matrix, adjacency_matrix)

        # Rank相关性
        spearman_corr, p_value = self.compute_rank_correlation(confusion_matrix, adjacency_matrix)

        # 高混淆邻接对
        high_pairs = self.find_high_confusion_adjacency_pairs(
            confusion_matrix, adjacency_matrix, threshold=0.05
        )

        # 邻接对的混淆统计
        adjacent_pairs = np.where(adjacency_matrix == 1)
        adjacent_confusions = confusion_matrix[adjacent_pairs]
        non_adjacent_mask = adjacency_matrix == 0
        np.fill_diagonal(non_adjacent_mask, False)  # 排除对角线
        non_adjacent_pairs = np.where(non_adjacent_mask)
        non_adjacent_confusions = confusion_matrix[non_adjacent_pairs]

        analysis_results = {
            'rank_correlation': {
                'spearman_correlation': float(spearman_corr),
                'p_value': float(p_value),
                'is_significant': p_value < 0.05
            },
            'confusion_stats': {
                'adjacent_pairs': {
                    'mean_confusion': float(np.mean(adjacent_confusions)),
                    'std_confusion': float(np.std(adjacent_confusions)),
                    'max_confusion': float(np.max(adjacent_confusions)),
                    'count': len(adjacent_confusions)
                },
                'non_adjacent_pairs': {
                    'mean_confusion': float(np.mean(non_adjacent_confusions)),
                    'std_confusion': float(np.std(non_adjacent_confusions)),
                    'max_confusion': float(np.max(non_adjacent_confusions)),
                    'count': len(non_adjacent_confusions)
                }
            },
            'high_confusion_adjacency_pairs': [
                {
                    'region1': int(pair[0]),
                    'region2': int(pair[1]),
                    'confusion_value': float(pair[2]),
                    'region1_name': region_names[pair[0]] if region_names else f"Region_{pair[0]}",
                    'region2_name': region_names[pair[1]] if region_names else f"Region_{pair[1]}"
                }
                for pair in high_pairs[:20]  # 取前20个
            ],
            'hadamard_stats': {
                'mean': float(np.mean(hadamard_product)),
                'std': float(np.std(hadamard_product)),
                'max': float(np.max(hadamard_product)),
                'sum': float(np.sum(hadamard_product))
            }
        }

        return analysis_results


def load_multiple_adjacency_matrices(adjacency_dir: Path,
                                   file_pattern: str = "*_adjacency_*.h5") -> Dict[str, AdjacencyMatrixReader]:
    """
    批量加载多个邻接矩阵文件

    Args:
        adjacency_dir: 邻接矩阵文件目录
        file_pattern: 文件名模式

    Returns:
        {subject_id: AdjacencyMatrixReader} 字典
    """
    adjacency_files = list(adjacency_dir.glob(file_pattern))

    if not adjacency_files:
        raise FileNotFoundError(f"在{adjacency_dir}中未找到符合模式{file_pattern}的文件")

    readers = {}
    for adj_file in sorted(adjacency_files):
        # 从文件名提取被试ID
        subject_id = adj_file.stem.split('_adjacency_')[0]
        readers[subject_id] = AdjacencyMatrixReader(adj_file)

    return readers


def create_summary_report(adjacency_readers: Dict[str, AdjacencyMatrixReader],
                         output_file: Path):
    """
    创建邻接矩阵汇总报告

    Args:
        adjacency_readers: 邻接矩阵读取器字典
        output_file: 输出CSV文件路径
    """
    summary_data = []

    for subject_id, reader in adjacency_readers.items():
        metadata = reader.metadata

        summary_data.append({
            'subject_id': subject_id,
            'total_regions': metadata.get('total_regions', 0),
            'total_adjacencies': metadata.get('total_adjacencies', 0),
            'adjacency_density': metadata.get('adjacency_density', 0.0),
            'avg_contact_voxels': metadata.get('avg_contact_voxels', 0.0),
            'max_contact_voxels': metadata.get('max_contact_voxels', 0),
            'connectivity': metadata.get('connectivity', 0),
            'file_path': str(reader.adjacency_file)
        })

    # 创建DataFrame并保存
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)

    return df