"""
通用工具包
提供数据加载、指标计算、实验记录等功能

使用方法:
    # 方式1：直接导入模块
    from 通用工具 import data_loaders
    from 通用工具.data_loaders import Brain1D_Dataset, NormMode

    # 方式2：添加到 sys.path
    import sys
    sys.path.append("/path/to/论文复现/通用工具")
    from data_loaders import Brain1D_Dataset, TestDataset, NormMode
    from metrics import MetricCalculator
"""

from .data_loaders import (
    NormMode,
    per_patient_zscore,
    Brain1D_Dataset,
    TestDataset,
    create_datasets,
    build_subject_index,
    get_subject_key_1d,
    get_subject_key_3d
)

from .metrics import MetricCalculator

__all__ = [
    # 数据加载
    'NormMode',
    'per_patient_zscore',
    'Brain1D_Dataset',
    'TestDataset',
    'create_datasets',
    'build_subject_index',
    'get_subject_key_1d',
    'get_subject_key_3d',
    # 指标计算
    'MetricCalculator',
]
