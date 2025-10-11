#!/usr/bin/env python3
"""
MRI Multi-modal Downsampling Pipeline - Batch Processing
=========================================================

自动化批处理脚本，用于将3D_validated数据集下采样到CEST分辨率

特性：
- 自动遍历所有3D验证数据
- 详细日志记录和进度跟踪
- 断点续传支持
- 内存监控和错误处理
- 生成QA报告和处理摘要

Author: KAN-Brain Project
Date: 2025-01-11
Version: 1.2.0 (with BUGFIX v1.2 - anisotropic Gaussian and probability labels fixed)
"""

import numpy as np
import scipy.io as sio
import h5py
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
import pandas as pd
import logging
import sys
import traceback
import psutil
import os
from datetime import datetime
import signal
import pickle
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 导入downsampling pipeline
from mri_downsampling_pipeline import MRIDownsamplingPipeline

# 导入3D-1D转换工具用于验证
sys.path.insert(0, str(Path(__file__).parent.parent / '1d-3d-convert'))
try:
    from data_3d_1d_mapper import Data3D1DMapper
    MAPPER_AVAILABLE = True
except ImportError:
    MAPPER_AVAILABLE = False
    print("Warning: data_3d_1d_mapper not available, data verification will be skipped")


class BatchDownsamplingProcessor:
    """批量downsampling处理器，带完整日志和错误处理"""

    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 log_level: str = 'INFO',
                 target_spacing: Tuple[float, float, float] = (1.8, 1.8, 3.0),
                 output_3d_dir: Optional[Path] = None,
                 output_1d_dir: Optional[Path] = None):
        """
        初始化批量处理器

        Args:
            input_dir: 输入目录（3D_validated数据）
            output_dir: 输出目录（用于日志、报告等）
            log_level: 日志级别
            target_spacing: 目标分辨率 (X, Y, Z) in mm
            output_3d_dir: 3D数据输出目录（可选，默认为output_dir/3d）
            output_1d_dir: 1D数据输出目录（可选，默认为output_dir/1d）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3D和1D数据分开保存
        self.output_3d_dir = Path(output_3d_dir) if output_3d_dir else self.output_dir / '3d'
        self.output_1d_dir = Path(output_1d_dir) if output_1d_dir else self.output_dir / '1d'
        self.output_3d_dir.mkdir(parents=True, exist_ok=True)
        self.output_1d_dir.mkdir(parents=True, exist_ok=True)

        self.target_spacing = target_spacing

        # 设置日志系统
        self.setup_logging(log_level)

        # 结果跟踪
        self.processing_results = []
        self.failed_subjects = []

        # 进程监控
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # 断点续传
        self.checkpoint_file = self.output_dir / "downsampling_checkpoint.pkl"
        self.completed_subjects = self.load_checkpoint()

        # 信号处理（优雅退出）
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.logger.info(f"批量Downsampling处理器初始化完成")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"3D数据输出: {self.output_3d_dir}")
        self.logger.info(f"1D数据输出: {self.output_1d_dir}")
        self.logger.info(f"目标分辨率: {self.target_spacing} mm")
        self.logger.info(f"初始内存使用: {self.initial_memory:.1f} MB")

    def setup_logging(self, log_level: str):
        """设置日志系统"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"batch_downsampling_{timestamp}.log"

        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"日志系统已启动，日志文件: {log_file}")

    def signal_handler(self, signum, frame):
        """处理中断信号"""
        self.logger.warning(f"收到信号 {signum}，正在保存检查点并退出...")
        self.save_checkpoint()
        self.generate_emergency_report()
        sys.exit(0)

    def load_checkpoint(self) -> set:
        """加载检查点"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.logger.info(f"加载检查点: 已完成 {len(checkpoint)} 个被试")
                    return checkpoint
            except Exception as e:
                self.logger.error(f"加载检查点失败: {e}")
        return set()

    def save_checkpoint(self):
        """保存检查点"""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.completed_subjects, f)
            self.logger.debug(f"检查点已保存: {len(self.completed_subjects)} 个被试完成")
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")

    def log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.initial_memory
        self.logger.debug(f"内存使用 {context}: {current_memory:.1f} MB (变化: {memory_delta:+.1f} MB)")

        if current_memory > 16000:  # 16GB
            self.logger.warning(f"内存使用过高: {current_memory:.1f} MB")

    def load_3d_validated_data(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """
        加载3D验证数据

        Args:
            mat_path: MAT文件路径

        Returns:
            包含data, region_mask, region_labels的字典
        """
        self.logger.debug(f"加载3D验证数据: {mat_path.name}")
        start_time = time.time()

        try:
            # 尝试使用scipy.io.loadmat加载
            try:
                mat_data = sio.loadmat(mat_path)
                self.logger.debug(f"使用scipy.io.loadmat加载成功")
            except (NotImplementedError, ValueError) as e:
                # 如果是HDF5格式或版本不兼容，使用h5py
                self.logger.info(f"scipy.io.loadmat失败 ({str(e)}), 尝试使用h5py加载HDF5格式")
                mat_data = {}
                with h5py.File(mat_path, 'r') as f:
                    # 读取关键数据
                    for key in ['data', 'region_mask', 'region_labels']:
                        if key in f:
                            mat_data[key] = f[key][()]
                            self.logger.debug(f"读取 {key}: {mat_data[key].shape}")

                            # MATLAB保存的HDF5可能需要转置
                            # MATLAB是列优先（Fortran order），Python是行优先（C order）
                            if mat_data[key].ndim >= 2:
                                # 对于2D以上的数组，可能需要转置轴
                                # data应该是(C, Z, Y, X) -> (Z, X, Y, C)或(X, Y, Z, C)
                                # 我们需要检查并调整
                                shape = mat_data[key].shape
                                self.logger.debug(f"  原始shape: {shape}")

                                # 如果是4D数据且第一维是351，需要转置
                                if key == 'data' and len(shape) == 4 and shape[0] == 351:
                                    # MATLAB: (351, Z, Y, X) -> Python: (Z, X, Y, 351)
                                    mat_data[key] = np.transpose(mat_data[key], (3, 2, 1, 0))
                                    self.logger.debug(f"  转置后shape: {mat_data[key].shape}")
                                # 如果是3D数据，可能也需要转置
                                elif key in ['region_mask', 'region_labels'] and len(shape) == 3:
                                    # 检查是否需要转置（MATLAB可能是 (Z, Y, X)）
                                    if shape != (384, 336, 256):
                                        # 尝试转置
                                        mat_data[key] = np.transpose(mat_data[key], (2, 1, 0))
                                        self.logger.debug(f"  转置后shape: {mat_data[key].shape}")

                self.logger.info(f"使用h5py加载HDF5格式成功")

            # 验证必需的keys
            required_keys = ['data', 'region_mask', 'region_labels']
            for key in required_keys:
                if key not in mat_data:
                    raise ValueError(f"缺少必需的key: {key}")

            # 验证和修正数据shape
            data = mat_data['data']
            region_mask = mat_data['region_mask']
            region_labels = mat_data['region_labels']

            # 验证data shape
            if data.ndim != 4:
                raise ValueError(f"data应该是4D数组，实际是{data.ndim}D: {data.shape}")

            # 如果data的最后一维不是351，可能需要调整
            if data.shape[-1] != 351:
                self.logger.warning(f"data shape异常: {data.shape}, 尝试调整...")
                # 如果第一维是351，转置
                if data.shape[0] == 351:
                    data = np.transpose(data, (1, 2, 3, 0))
                    self.logger.info(f"data转置后shape: {data.shape}")
                else:
                    raise ValueError(f"无法识别的data shape: {data.shape}, 期望最后一维是351")

            # 验证3D shape（期望是384, 336, 256）
            expected_spatial = (384, 336, 256)
            data_spatial = data.shape[:3]

            # 如果空间维度不匹配，尝试转置
            if data_spatial != expected_spatial:
                self.logger.warning(f"空间维度不匹配: data={data_spatial}, 期望={expected_spatial}")

                # 尝试所有可能的转置组合，找到匹配的
                from itertools import permutations
                found = False
                for perm in permutations([0, 1, 2]):
                    test_shape = tuple(data_spatial[i] for i in perm)
                    if test_shape == expected_spatial:
                        self.logger.info(f"找到匹配的转置: {perm}")
                        # 应用转置（包括通道维度）
                        full_perm = tuple(list(perm) + [3])
                        data = np.transpose(data, full_perm)
                        # 对mask和labels应用同样的转置（只前3维）
                        if region_mask.shape != expected_spatial:
                            region_mask = np.transpose(region_mask, perm)
                        if region_labels.shape != expected_spatial:
                            region_labels = np.transpose(region_labels, perm)
                        found = True
                        break

                if not found:
                    raise ValueError(f"无法将data shape {data_spatial}转换为期望的{expected_spatial}")

            # 最终验证
            if data.shape[:3] != expected_spatial or data.shape[3] != 351:
                raise ValueError(f"最终data shape验证失败: {data.shape}, 期望: {expected_spatial + (351,)}")

            if region_mask.shape != expected_spatial:
                raise ValueError(f"region_mask shape验证失败: {region_mask.shape}, 期望: {expected_spatial}")

            if region_labels.shape != expected_spatial:
                raise ValueError(f"region_labels shape验证失败: {region_labels.shape}, 期望: {expected_spatial}")

            load_time = time.time() - start_time
            self.logger.info(f"数据加载成功: {mat_path.name} (耗时: {load_time:.2f}秒)")
            self.logger.info(f"  data shape: {data.shape}")
            self.logger.info(f"  region_mask shape: {region_mask.shape}")
            self.logger.info(f"  region_labels shape: {region_labels.shape}")
            self.log_memory_usage(f"加载 {mat_path.name} 后")

            return {
                'data': data,
                'region_mask': region_mask,
                'region_labels': region_labels
            }

        except Exception as e:
            self.logger.error(f"加载文件失败 {mat_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _generate_data_info_file(self,
                                 info_path: Path,
                                 data_dict: Dict[str, Any],
                                 data_type: str,
                                 subject_id: str):
        """
        生成数据描述文件

        Args:
            info_path: 描述文件路径
            data_dict: 数据字典
            data_type: 数据类型（"3D"或"1D"）
            subject_id: 被试ID
        """
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{data_type}数据文件说明\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"被试ID: {subject_id}\n")
            f.write(f"文件名: {subject_id}_{data_type.lower()}.npz\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据格式版本: v1.4.0\n\n")

            f.write("-" * 80 + "\n")
            f.write("数据内容说明\n")
            f.write("-" * 80 + "\n\n")

            if data_type == "3D":
                # 3D数据说明
                f.write("包含的Key及其维度:\n\n")

                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        f.write(f"[{key}]\n")
                        f.write(f"  维度: {value.shape}\n")
                        f.write(f"  数据类型: {value.dtype}\n")
                        f.write(f"  坐标系: (Z, X, Y")
                        if value.ndim == 4:
                            f.write(", C)\n")
                        else:
                            f.write(")\n")

                        # 添加说明
                        if key == 'data_lr':
                            f.write(f"  说明: 下采样后的多模态特征数据\n")
                            f.write(f"        - 351个通道包含所有成像模态\n")
                            f.write(f"        - 通道包括: QTI(15), DWI(210), CEST(4+54+54), MPRAGE, GRE, QSM等\n")
                        elif key == 'proba_labels':
                            f.write(f"  说明: 概率标签（软标签）\n")
                            f.write(f"        - 102个脑区类别的概率分布\n")
                            f.write(f"        - 每个体素的102个概率之和≈1.0\n")
                            f.write(f"        - 注意: 不是严格one-hot，而是概率分布\n")
                            f.write(f"        - 标签值0-101都是有效脑区（0不是背景！）\n")
                        elif key == 'region_mask_lr':
                            f.write(f"  说明: ROI掩码\n")
                            f.write(f"        - 值为0表示背景（脑外）\n")
                            f.write(f"        - 值为1表示ROI内（脑内体素）\n")
                        f.write("\n")
                    else:
                        f.write(f"[{key}]\n")
                        f.write(f"  值: {value}\n")
                        f.write(f"  类型: {type(value).__name__}\n\n")

                f.write("-" * 80 + "\n")
                f.write("加载示例:\n")
                f.write("-" * 80 + "\n\n")
                f.write("```python\n")
                f.write("import numpy as np\n\n")
                f.write(f"# 加载3D数据\n")
                f.write(f"data = np.load('{subject_id}_3d.npz')\n\n")
                f.write("# 提取数据\n")
                f.write("data_lr = data['data_lr']              # 多模态特征\n")
                f.write("proba_labels = data['proba_labels']    # 概率标签\n")
                f.write("region_mask_lr = data['region_mask_lr']  # ROI掩码\n\n")
                f.write("# 提取特定模态（例如MPRAGE）\n")
                f.write("mprage = data_lr[..., 341]  # MPRAGE在通道341\n")
                f.write("```\n\n")

            elif data_type == "1D":
                # 1D数据说明
                f.write("包含的Key及其维度:\n\n")

                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        f.write(f"[{key}]\n")
                        f.write(f"  维度: {value.shape}\n")
                        f.write(f"  数据类型: {value.dtype}\n")

                        # 添加说明
                        if key == 'multidim_data':
                            f.write(f"  说明: 1D特征矩阵\n")
                            f.write(f"        - 每行对应一个ROI内的体素\n")
                            f.write(f"        - 351个通道与3D数据的data_lr相同\n")
                            f.write(f"        - 按C-order（行优先）排列\n")
                        elif key == 'seg_one_hot':
                            f.write(f"  说明: 1D概率标签（软标签）\n")
                            f.write(f"        - 每列对应一个体素的102个区域概率\n")
                            f.write(f"        - 每列之和≈1.0\n")
                            f.write(f"        - 注意: 实际是概率分布，不是严格one-hot\n")
                            f.write(f"        - 标签值0-101都是有效脑区（0不是背景！）\n")
                        elif key == 'region_seg':
                            f.write(f"  说明: 1D区域标签（硬标签）\n")
                            f.write(f"        - 每个体素的最可能区域（argmax）\n")
                            f.write(f"        - 值范围: 0-101\n")
                        elif key == 'region':
                            f.write(f"  说明: ROI掩码（用于1D→3D重建）\n")
                            f.write(f"        - 与3D文件中的region_mask_lr相同\n")
                            f.write(f"        - 值为0表示背景，值为1表示ROI内\n")
                        f.write("\n")
                    else:
                        f.write(f"[{key}]\n")
                        f.write(f"  值: {value}\n")
                        f.write(f"  类型: {type(value).__name__}\n")
                        if key == 'n_voxels':
                            f.write(f"  说明: ROI内的总体素数\n")
                        f.write("\n")

                f.write("-" * 80 + "\n")
                f.write("加载示例:\n")
                f.write("-" * 80 + "\n\n")
                f.write("```python\n")
                f.write("import numpy as np\n\n")
                f.write(f"# 加载1D数据\n")
                f.write(f"data = np.load('{subject_id}_1d.npz')\n\n")
                f.write("# 提取数据\n")
                f.write("X = data['multidim_data']  # 特征矩阵 (n_voxels, 351)\n")
                f.write("y_soft = data['seg_one_hot']  # 软标签 (102, n_voxels)\n")
                f.write("y_hard = data['region_seg']   # 硬标签 (n_voxels,)\n")
                f.write("n_voxels = int(data['n_voxels'])  # 体素数\n\n")
                f.write("# 用于机器学习\n")
                f.write("# X: 训练特征\n")
                f.write("# y_hard: 分类标签（如果需要硬标签）\n")
                f.write("# y_soft: 概率标签（如果需要软标签）\n")
                f.write("```\n\n")

            f.write("-" * 80 + "\n")
            f.write("重要提示:\n")
            f.write("-" * 80 + "\n\n")
            f.write("1. 区域标签0不是背景！\n")
            f.write("   - 标签值0-101都是有效的脑区标签\n")
            f.write("   - 背景由region_mask_lr/region定义（值为0的位置）\n\n")
            f.write("2. 概率标签不是严格one-hot编码\n")
            f.write("   - proba_labels/seg_one_hot是概率分布\n")
            f.write("   - 每个体素可能属于多个区域（软标签）\n")
            f.write("   - 反映下采样过程中的部分容积效应\n\n")
            f.write("3. 坐标系统\n")
            f.write("   - 3D数据: (Z, X, Y, C) 坐标系\n")
            f.write("   - 1D数据: 按C-order（行优先）展平\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"详细文档: DOWNSAMPLED_DATA_FORMAT.md\n")
            f.write(f"生成工具: batch_downsampling_pipeline.py v1.2.0\n")
            f.write("=" * 80 + "\n")

    def _smart_verify(self,
                      results: Dict[str, Any],
                      subject_id: str,
                      subject_idx: int,
                      total_subjects: int,
                      verify_mode: str) -> Dict[str, Any]:
        """
        智能验证策略

        Args:
            results: Pipeline输出结果
            subject_id: 被试ID
            subject_idx: 当前被试索引
            total_subjects: 总被试数
            verify_mode: 验证模式

        Returns:
            验证结果字典
        """
        # 根据验证模式决定验证策略
        if verify_mode == 'none':
            return {
                'verified': True,
                'reason': 'Verification skipped',
                'checks': {},
                'verification_type': 'none'
            }

        elif verify_mode == 'full':
            # 完整验证：所有被试都进行保存后验证
            self.logger.info(f"  → 完整验证（重新加载文件）")
            return self.verify_saved_data(subject_id)

        elif verify_mode == 'lightweight':
            # 轻量级验证：所有被试都进行保存前验证
            self.logger.info(f"  → 轻量级验证（内存数据）")
            return self.verify_before_save(results, subject_id)

        elif verify_mode == 'smart':
            # 智能验证策略
            is_first = (subject_idx == 0)
            is_last = (subject_idx == total_subjects - 1)

            if is_first:
                # 第一个被试：完整验证，确保pipeline正常启动
                self.logger.info(f"  → 首个被试：完整验证（重新加载文件）")
                return self.verify_saved_data(subject_id)

            elif is_last and total_subjects > 1:
                # 最后一个被试：完整验证，确保pipeline一直正常
                self.logger.info(f"  → 最后被试：完整验证（重新加载文件）")
                return self.verify_saved_data(subject_id)

            else:
                # 中间被试：轻量级验证，快速高效
                self.logger.info(f"  → 轻量级验证（内存数据，快速）")
                return self.verify_before_save(results, subject_id)

        else:
            self.logger.warning(f"未知的验证模式: {verify_mode}，使用轻量级验证")
            return self.verify_before_save(results, subject_id)

    def verify_before_save(self, results: Dict[str, Any], subject_id: str) -> Dict[str, Any]:
        """
        保存前轻量级验证（内存中的数据，零I/O开销）

        Args:
            results: Pipeline输出结果
            subject_id: 被试ID

        Returns:
            验证结果字典
        """
        try:
            self.logger.debug(f"轻量级验证（保存前）: {subject_id}")

            checks = {}

            # 检查1: 必需的key存在性
            required_3d_keys = ['data_lr', 'proba_labels', 'region_mask_lr']
            has_3d = all(k in results for k in required_3d_keys)
            checks['keys_3d'] = has_3d

            if not has_3d:
                return {
                    'verified': False,
                    'reason': 'Missing required 3D keys',
                    'checks': checks,
                    'verification_type': 'lightweight'
                }

            # 检查2: 概率和验证（3D）
            prob_sum_3d = results['proba_labels'][results['region_mask_lr'] > 0].sum(axis=-1).mean()
            checks['prob_sum_3d'] = float(prob_sum_3d)
            checks['prob_sum_valid_3d'] = (0.99 <= prob_sum_3d <= 1.01)

            # 检查3: 如果有1D数据，验证对应关系
            if 'multidim_data' in results:
                checks['keys_1d'] = True
                n_voxels = results['n_voxels']
                checks['n_voxels'] = n_voxels

                # 体素数一致性
                n_voxels_from_mask = np.sum(results['region_mask_lr'] > 0)
                checks['n_voxels_match'] = (n_voxels_from_mask == n_voxels)

                # 形状一致性
                checks['multidim_data_shape'] = (results['multidim_data'].shape[0] == n_voxels)
                checks['seg_one_hot_shape'] = (results['seg_one_hot'].shape[1] == n_voxels)

                # 概率和验证（1D）
                prob_sum_1d = results['seg_one_hot'].sum(axis=0).mean()
                checks['prob_sum_1d'] = float(prob_sum_1d)
                checks['prob_sum_valid_1d'] = (0.99 <= prob_sum_1d <= 1.01)

                # 3D-1D对应关系（抽样检查，快速）
                features_3d = results['data_lr'][results['region_mask_lr'] > 0]
                # 只抽样100个体素快速检查
                sample_size = min(100, n_voxels)
                sample_indices = np.random.choice(n_voxels, size=sample_size, replace=False)

                max_feat_diff = np.abs(features_3d[sample_indices] - results['multidim_data'][sample_indices]).max()
                checks['max_feat_diff_sampled'] = float(max_feat_diff)
                checks['correspondence_valid'] = (max_feat_diff < 1e-5)
            else:
                checks['keys_1d'] = False

            # 判断总体验证结果
            all_passed = checks['keys_3d'] and checks['prob_sum_valid_3d']
            if checks.get('keys_1d', False):
                all_passed = all_passed and all([
                    checks['n_voxels_match'],
                    checks['multidim_data_shape'],
                    checks['seg_one_hot_shape'],
                    checks['prob_sum_valid_1d'],
                    checks['correspondence_valid']
                ])

            return {
                'verified': all_passed,
                'reason': 'Lightweight checks passed' if all_passed else 'Some lightweight checks failed',
                'checks': checks,
                'verification_type': 'lightweight'
            }

        except Exception as e:
            self.logger.error(f"轻量级验证出错: {e}")
            return {
                'verified': False,
                'reason': f'Verification error: {str(e)}',
                'checks': {},
                'verification_type': 'lightweight'
            }

    def verify_saved_data(self, subject_id: str) -> Dict[str, Any]:
        """
        验证已保存的3D和1D数据

        Args:
            subject_id: 被试ID

        Returns:
            验证结果字典
        """
        if not MAPPER_AVAILABLE:
            return {
                'verified': False,
                'reason': 'Mapper not available',
                'checks': {}
            }

        try:
            self.logger.info(f"验证数据: {subject_id}")

            # 加载3D和1D数据
            file_3d = self.output_3d_dir / f'{subject_id}_3d.npz'
            file_1d = self.output_1d_dir / f'{subject_id}_1d.npz'

            if not file_3d.exists():
                return {
                    'verified': False,
                    'reason': f'3D file not found: {file_3d}',
                    'checks': {}
                }

            if not file_1d.exists():
                return {
                    'verified': False,
                    'reason': f'1D file not found: {file_1d}',
                    'checks': {}
                }

            data_3d = np.load(file_3d)
            data_1d = np.load(file_1d)

            checks = {}

            # 检查1: 必需的key存在性
            required_3d_keys = ['data_lr', 'proba_labels', 'region_mask_lr']
            required_1d_keys = ['multidim_data', 'seg_one_hot', 'region_seg', 'region', 'n_voxels']

            has_3d = all(k in data_3d for k in required_3d_keys)
            has_1d = all(k in data_1d for k in required_1d_keys)

            checks['keys_3d'] = has_3d
            checks['keys_1d'] = has_1d

            if not has_3d or not has_1d:
                return {
                    'verified': False,
                    'reason': 'Missing required keys',
                    'checks': checks
                }

            # 检查2: 体素数一致性
            n_voxels_from_mask = np.sum(data_3d['region_mask_lr'] > 0)
            n_voxels_recorded = int(data_1d['n_voxels'])

            checks['n_voxels_match'] = (n_voxels_from_mask == n_voxels_recorded)
            checks['n_voxels'] = n_voxels_recorded

            # 检查3: 数据形状一致性
            checks['multidim_data_shape'] = (data_1d['multidim_data'].shape[0] == n_voxels_recorded)
            checks['seg_one_hot_shape'] = (data_1d['seg_one_hot'].shape[1] == n_voxels_recorded)

            # 检查4: 概率和验证
            prob_sum_3d = data_3d['proba_labels'][data_3d['region_mask_lr'] > 0].sum(axis=-1).mean()
            prob_sum_1d = data_1d['seg_one_hot'].sum(axis=0).mean()

            checks['prob_sum_3d'] = float(prob_sum_3d)
            checks['prob_sum_1d'] = float(prob_sum_1d)
            checks['prob_sum_valid_3d'] = (0.99 <= prob_sum_3d <= 1.01)
            checks['prob_sum_valid_1d'] = (0.99 <= prob_sum_1d <= 1.01)

            # 检查5: 3D-1D对应关系（抽样检查）
            features_from_3d = data_3d['data_lr'][data_3d['region_mask_lr'] > 0]
            max_feat_diff = np.abs(features_from_3d - data_1d['multidim_data']).max()

            labels_from_3d = data_3d['proba_labels'][data_3d['region_mask_lr'] > 0]
            max_label_diff = np.abs(labels_from_3d.T - data_1d['seg_one_hot']).max()

            checks['max_feat_diff'] = float(max_feat_diff)
            checks['max_label_diff'] = float(max_label_diff)
            checks['correspondence_valid'] = (max_feat_diff < 1e-5 and max_label_diff < 1e-5)

            # 判断总体验证结果
            all_passed = all([
                checks['keys_3d'],
                checks['keys_1d'],
                checks['n_voxels_match'],
                checks['multidim_data_shape'],
                checks['seg_one_hot_shape'],
                checks['prob_sum_valid_3d'],
                checks['prob_sum_valid_1d'],
                checks['correspondence_valid']
            ])

            if all_passed:
                self.logger.info(f"✓ 数据验证通过: {subject_id}")
            else:
                self.logger.warning(f"⚠ 数据验证发现问题: {subject_id}")
                for check, result in checks.items():
                    if isinstance(result, bool) and not result:
                        self.logger.warning(f"  - {check}: FAILED")

            return {
                'verified': all_passed,
                'reason': 'Full checks passed' if all_passed else 'Some checks failed',
                'checks': checks,
                'verification_type': 'full'
            }

        except Exception as e:
            self.logger.error(f"验证过程出错: {e}")
            return {
                'verified': False,
                'reason': f'Verification error: {str(e)}',
                'checks': {},
                'verification_type': 'full'
            }

    def save_downsampled_data(self,
                             results: Dict[str, Any],
                             subject_id: str) -> Dict[str, float]:
        """
        保存downsampled数据（3D和1D分开保存）

        Args:
            results: Pipeline输出结果
            subject_id: 被试ID

        Returns:
            文件大小字典 {'3d_mb': float, '1d_mb': float}
        """
        start_time = time.time()
        file_sizes = {}

        try:
            # 1. 保存3D数据
            output_3d_path = self.output_3d_dir / f"{subject_id}_3d.npz"
            self.logger.info(f"保存3D数据到: {output_3d_path}")

            save_3d_dict = {
                'data_lr': results['data_lr'],
                'proba_labels': results['proba_labels'],
                'region_mask_lr': results['region_mask_lr']
            }

            np.savez_compressed(output_3d_path, **save_3d_dict)
            file_sizes['3d_mb'] = output_3d_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"3D数据保存成功: {output_3d_path.name} ({file_sizes['3d_mb']:.1f} MB)")

            # 1.1 生成3D数据描述文件
            info_3d_path = self.output_3d_dir / f"{subject_id}_3d_README.txt"
            self._generate_data_info_file(info_3d_path, save_3d_dict, "3D", subject_id)
            self.logger.debug(f"3D数据描述文件: {info_3d_path.name}")

            # 2. 保存1D数据（如果有）
            if 'multidim_data' in results:
                output_1d_path = self.output_1d_dir / f"{subject_id}_1d.npz"
                self.logger.info(f"保存1D数据到: {output_1d_path}")

                save_1d_dict = {
                    'multidim_data': results['multidim_data'],
                    'seg_one_hot': results['seg_one_hot'],
                    'region_seg': results['region_seg'],
                    'region': results['region_mask_lr'],  # 保存mask用于重建
                    'n_voxels': results['n_voxels']
                }

                np.savez_compressed(output_1d_path, **save_1d_dict)
                file_sizes['1d_mb'] = output_1d_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"1D数据保存成功: {output_1d_path.name} ({file_sizes['1d_mb']:.1f} MB)")
                self.logger.info(f"  multidim_data: {results['multidim_data'].shape}")
                self.logger.info(f"  seg_one_hot: {results['seg_one_hot'].shape}")
                self.logger.info(f"  n_voxels: {results['n_voxels']}")

                # 2.1 生成1D数据描述文件
                info_1d_path = self.output_1d_dir / f"{subject_id}_1d_README.txt"
                self._generate_data_info_file(info_1d_path, save_1d_dict, "1D", subject_id)
                self.logger.debug(f"1D数据描述文件: {info_1d_path.name}")
            else:
                file_sizes['1d_mb'] = 0.0
                self.logger.warning("未生成1D数据（可能使用了save_axis_order='proc'）")

            # 3. 保存metadata到输出根目录
            metadata_path = self.output_dir / f"{subject_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(results['metadata'], f, indent=2, default=str)
            self.logger.debug(f"Metadata保存: {metadata_path}")

            # 4. 保存QA metrics到输出根目录
            qa_path = self.output_dir / f"{subject_id}_qa_metrics.json"
            with open(qa_path, 'w') as f:
                json.dump(results['qa_metrics'], f, indent=2, default=str)
            self.logger.debug(f"QA metrics保存: {qa_path}")

            save_time = time.time() - start_time
            total_size = file_sizes['3d_mb'] + file_sizes['1d_mb']
            self.logger.info(f"所有数据保存完成: 总大小 {total_size:.1f} MB, 耗时: {save_time:.2f}秒")

            return file_sizes

        except Exception as e:
            self.logger.error(f"保存文件失败: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def process_single_subject(self,
                               mat_path: Path,
                               subject_id: str,
                               subject_idx: int = 0,
                               total_subjects: int = 1,
                               verify_mode: str = 'smart') -> Dict[str, Any]:
        """
        处理单个被试

        Args:
            mat_path: 输入MAT文件路径
            subject_id: 被试ID
            subject_idx: 当前被试索引（从0开始）
            total_subjects: 总被试数
            verify_mode: 验证模式 ('smart', 'full', 'lightweight', 'none')

        Returns:
            处理结果字典
        """
        start_time = time.time()

        self.logger.info(f"{'='*80}")
        self.logger.info(f"开始处理被试: {subject_id}")
        self.logger.info(f"输入文件: {mat_path.name}")

        result = {
            'subject_id': subject_id,
            'filename': mat_path.name,
            'start_time': datetime.now().isoformat()
        }

        try:
            # 步骤1: 加载3D验证数据
            self.logger.info("步骤1: 加载3D验证数据")
            input_data = self.load_3d_validated_data(mat_path)

            # 验证输入shape
            data_shape = input_data['data'].shape
            self.logger.info(f"输入数据shape: {data_shape}")
            if data_shape[:3] != (384, 336, 256) or data_shape[3] != 351:
                self.logger.warning(f"输入数据shape不符合预期: {data_shape}")

            # 步骤2: 初始化downsampling pipeline
            self.logger.info("步骤2: 初始化Downsampling Pipeline")
            pipeline_output_dir = self.output_dir / subject_id
            pipeline_output_dir.mkdir(exist_ok=True)

            pipeline = MRIDownsamplingPipeline(
                output_dir=pipeline_output_dir,
                log_level='INFO',
                random_seed=42
            )

            # 步骤3: 运行downsampling
            self.logger.info("步骤3: 运行Downsampling Pipeline")
            pipeline_results = pipeline.run(
                data=input_data['data'],
                region_mask=input_data['region_mask'],
                region_labels=input_data['region_labels'],
                align_to_128x104x18=False,
                save_axis_order='orig'  # 生成1D格式数据
            )

            # 步骤4: 保存结果
            self.logger.info("步骤4: 保存Downsampled数据")
            file_sizes = self.save_downsampled_data(pipeline_results, subject_id)

            # 步骤5: 智能验证
            self.logger.info("步骤5: 数据验证")
            verification_result = self._smart_verify(
                pipeline_results, subject_id, subject_idx, total_subjects, verify_mode
            )

            # 记录结果
            processing_time = time.time() - start_time

            result.update({
                'status': 'success',
                'output_3d_path': str(self.output_3d_dir / f"{subject_id}_3d.npz"),
                'output_1d_path': str(self.output_1d_dir / f"{subject_id}_1d.npz") if file_sizes.get('1d_mb', 0) > 0 else None,
                'file_size_3d_mb': file_sizes['3d_mb'],
                'file_size_1d_mb': file_sizes.get('1d_mb', 0),
                'file_size_total_mb': file_sizes['3d_mb'] + file_sizes.get('1d_mb', 0),
                'processing_time': processing_time,
                'input_shape': data_shape,
                'output_shape': pipeline_results['data_lr'].shape,
                'proba_labels_shape': pipeline_results['proba_labels'].shape,
                'slab_thickness_mm': pipeline_results['metadata']['slab_localization']['slab_thickness_mm'],
                'coverage_fallback': pipeline_results['metadata']['slab_localization']['coverage_fallback'],
                'verification': verification_result
            })

            # 标记为已完成
            self.completed_subjects.add(subject_id)

            # 根据验证结果决定成功标识
            if verification_result['verified']:
                self.logger.info(f"✅ 被试 {subject_id} 处理成功（已验证）")
            else:
                self.logger.warning(f"⚠️ 被试 {subject_id} 处理完成，但验证失败: {verification_result['reason']}")

            self.logger.info(f"   处理时间: {processing_time:.1f}秒")
            self.logger.info(f"   3D数据: {file_sizes['3d_mb']:.1f}MB → {self.output_3d_dir}/{subject_id}_3d.npz")
            self.logger.info(f"   3D shape: {pipeline_results['data_lr'].shape}")
            if 'multidim_data' in pipeline_results:
                self.logger.info(f"   1D数据: {file_sizes['1d_mb']:.1f}MB → {self.output_1d_dir}/{subject_id}_1d.npz")
                self.logger.info(f"   1D shape: {pipeline_results['multidim_data'].shape}")
                self.logger.info(f"   ROI体素数: {pipeline_results['n_voxels']}")
            self.logger.info(f"   总大小: {file_sizes['3d_mb'] + file_sizes.get('1d_mb', 0):.1f}MB")
            verify_type = verification_result.get('verification_type', 'unknown')
            verify_status = '✓ 通过' if verification_result['verified'] else '✗ 失败'
            self.logger.info(f"   验证状态: {verify_status} ({verify_type})")

        except Exception as e:
            self.logger.error(f"❌ 被试 {subject_id} 处理失败")
            self.logger.error(f"错误类型: {type(e).__name__}")
            self.logger.error(f"错误信息: {str(e)}")
            self.logger.error(f"详细堆栈:\n{traceback.format_exc()}")

            result.update({
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'processing_time': time.time() - start_time
            })
            self.failed_subjects.append(subject_id)

        finally:
            # 定期保存检查点
            if len(self.completed_subjects) % 3 == 0:
                self.save_checkpoint()

            # 清理内存
            import gc
            gc.collect()

        return result

    def create_dataset_index(self) -> Dict[str, Dict[str, str]]:
        """创建数据集索引"""
        self.logger.info("创建数据集索引...")

        # 查找所有3d_validated.mat文件
        mat_files = sorted(list(self.input_dir.glob("*_3d_validated.mat")))

        if len(mat_files) == 0:
            self.logger.error(f"未找到3D验证文件在: {self.input_dir}")
            raise ValueError(f"未找到3D验证文件在: {self.input_dir}")

        self.logger.info(f"找到 {len(mat_files)} 个3D验证文件")

        index_mapping = {}
        for filepath in mat_files:
            # 提取subject_id（去掉_3d_validated后缀）
            subject_id = filepath.stem.replace('_3d_validated', '')

            index_mapping[subject_id] = {
                "subject_id": subject_id,
                "filename": filepath.name,
                "filepath": str(filepath),
                "file_size_mb": filepath.stat().st_size / (1024 * 1024)
            }
            self.logger.debug(f"索引: {subject_id} - {filepath.name} ({index_mapping[subject_id]['file_size_mb']:.1f} MB)")

        # 保存索引
        json_path = self.output_dir / "downsampling_dataset_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, indent=2, ensure_ascii=False)

        self.logger.info(f"索引已保存到: {json_path}")
        return index_mapping

    def process_all_subjects(self,
                             skip_existing: bool = True,
                             verify_mode: str = 'smart') -> Tuple[List, List]:
        """
        批量处理所有被试

        Args:
            skip_existing: 是否跳过已存在的文件
            verify_mode: 验证模式 ('smart', 'full', 'lightweight', 'none')

        Returns:
            (成功列表, 失败列表)
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"批量Downsampling处理开始")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"跳过已存在: {skip_existing}")
        self.logger.info(f"验证模式: {verify_mode}")
        if verify_mode == 'smart':
            self.logger.info(f"  → 智能验证：首尾完整验证，中间轻量级验证")
        elif verify_mode == 'full':
            self.logger.info(f"  → 完整验证：所有被试重新加载文件验证（慢）")
        elif verify_mode == 'lightweight':
            self.logger.info(f"  → 轻量级验证：所有被试内存验证（快）")
        elif verify_mode == 'none':
            self.logger.info(f"  → 不验证：跳过所有验证（最快）")
        self.logger.info(f"已完成被试数: {len(self.completed_subjects)}")
        self.logger.info(f"{'='*80}")

        # 创建索引
        index_mapping = self.create_dataset_index()

        total_start_time = time.time()
        successful_subjects = []

        # 处理每个被试
        total_subjects = len(index_mapping)
        for subject_idx, (subject_id, subject_info) in enumerate(tqdm(index_mapping.items(), desc="Downsampling进度")):
            # 检查是否已经处理过（断点续传）
            if subject_id in self.completed_subjects:
                self.logger.info(f"跳过已完成的被试: {subject_id}")
                successful_subjects.append(subject_id)
                continue

            # 检查是否跳过已存在的文件
            if skip_existing:
                output_3d_path = self.output_3d_dir / f"{subject_id}_3d.npz"
                output_1d_path = self.output_1d_dir / f"{subject_id}_1d.npz"
                if output_3d_path.exists() and output_1d_path.exists():
                    self.logger.info(f"跳过已存在文件: {subject_id}")
                    self.completed_subjects.add(subject_id)
                    successful_subjects.append(subject_id)
                    continue

            # 处理被试（传递索引和总数用于智能验证）
            mat_path = Path(subject_info['filepath'])
            result = self.process_single_subject(
                mat_path, subject_id,
                subject_idx=subject_idx,
                total_subjects=total_subjects,
                verify_mode=verify_mode
            )
            self.processing_results.append(result)

            if result['status'] == 'success':
                successful_subjects.append(subject_id)

            # 更新进度信息
            tqdm.write(f"内存: {self.process.memory_info().rss / 1024 / 1024:.1f} MB | "
                      f"成功: {len(successful_subjects)} | "
                      f"失败: {len(self.failed_subjects)}")

        total_time = time.time() - total_start_time

        # 保存最终检查点
        self.save_checkpoint()

        # 生成报告
        self._generate_batch_report(total_time)

        return successful_subjects, self.failed_subjects

    def generate_emergency_report(self):
        """生成紧急报告（程序中断时）"""
        self.logger.info("生成紧急报告...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        emergency_report = {
            'timestamp': timestamp,
            'completed_subjects': list(self.completed_subjects),
            'failed_subjects': self.failed_subjects,
            'last_results': self.processing_results[-10:] if self.processing_results else [],
            'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024
        }

        emergency_file = self.output_dir / f"emergency_report_{timestamp}.json"
        with open(emergency_file, 'w') as f:
            json.dump(emergency_report, f, indent=2, default=str)

        self.logger.info(f"紧急报告已保存: {emergency_file}")

    def _generate_batch_report(self, total_time: float):
        """生成批量处理报告"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("生成批量Downsampling报告")
        self.logger.info(f"{'='*80}")

        successful = [r for r in self.processing_results if r['status'] == 'success']
        failed = [r for r in self.processing_results if r['status'] == 'failed']

        # 统计验证结果
        verified_count = sum(1 for r in successful if r.get('verification', {}).get('verified', False))
        verification_failed = len(successful) - verified_count

        self.logger.info(f"处理统计:")
        self.logger.info(f"  总处理数: {len(self.processing_results)}")
        self.logger.info(f"  成功处理: {len(successful)}")
        self.logger.info(f"  处理失败: {len(failed)}")
        self.logger.info(f"  验证通过: {verified_count}")
        self.logger.info(f"  验证失败: {verification_failed}")
        self.logger.info(f"  总时间: {total_time/60:.1f}分钟")
        self.logger.info(f"  最终内存: {self.process.memory_info().rss / 1024 / 1024:.1f} MB")

        if successful:
            avg_time = np.mean([r['processing_time'] for r in successful])
            avg_size_3d = np.mean([r.get('file_size_3d_mb', 0) for r in successful])
            avg_size_1d = np.mean([r.get('file_size_1d_mb', 0) for r in successful])
            avg_size_total = np.mean([r.get('file_size_total_mb', 0) for r in successful])
            total_size = sum([r.get('file_size_total_mb', 0) for r in successful])
            self.logger.info(f"  平均处理时间: {avg_time:.1f}秒")
            self.logger.info(f"  平均文件大小: 3D={avg_size_3d:.1f}MB, 1D={avg_size_1d:.1f}MB, 总计={avg_size_total:.1f}MB")
            self.logger.info(f"  总输出大小: {total_size:.1f}MB")

        # 失败的被试
        if failed:
            self.logger.error(f"\n处理失败的被试:")
            for r in failed:
                self.logger.error(f"  - {r['subject_id']}")
                self.logger.error(f"    错误类型: {r.get('error_type', 'Unknown')}")
                self.logger.error(f"    错误信息: {r.get('error', 'Unknown')}")

        # 验证失败的被试
        if verification_failed > 0:
            self.logger.warning(f"\n验证失败的被试:")
            for r in successful:
                verification = r.get('verification', {})
                if not verification.get('verified', False):
                    self.logger.warning(f"  - {r['subject_id']}")
                    self.logger.warning(f"    原因: {verification.get('reason', 'Unknown')}")
                    # 显示具体的检查失败项
                    checks = verification.get('checks', {})
                    for check, value in checks.items():
                        if isinstance(value, bool) and not value:
                            self.logger.warning(f"      × {check}")

        # 保存详细报告
        self._save_detailed_reports()

    def _save_detailed_reports(self):
        """保存详细报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON报告
        json_report_path = self.output_dir / f"downsampling_report_{timestamp}.json"
        report_data = {
            'timestamp': timestamp,
            'pipeline_version': '1.2.0',
            'target_spacing_mm': self.target_spacing,
            'completed_subjects': list(self.completed_subjects),
            'failed_subjects': self.failed_subjects,
            'processing_results': self.processing_results,
            'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024
        }

        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        self.logger.info(f"JSON报告已保存: {json_report_path}")

        # CSV摘要
        if self.processing_results:
            summary_data = []
            for r in self.processing_results:
                # 提取验证信息
                verification = r.get('verification', {})
                verified = verification.get('verified', False)
                verification_reason = verification.get('reason', '')
                checks = verification.get('checks', {})

                summary_data.append({
                    'subject_id': r['subject_id'],
                    'status': r['status'],
                    'verified': verified,
                    'verification_reason': verification_reason,
                    'n_voxels': checks.get('n_voxels', 0),
                    'prob_sum_3d': checks.get('prob_sum_3d', 0),
                    'prob_sum_1d': checks.get('prob_sum_1d', 0),
                    'max_feat_diff': checks.get('max_feat_diff', 0),
                    'max_label_diff': checks.get('max_label_diff', 0),
                    'file_size_3d_mb': r.get('file_size_3d_mb', 0),
                    'file_size_1d_mb': r.get('file_size_1d_mb', 0),
                    'file_size_total_mb': r.get('file_size_total_mb', 0),
                    'processing_time': r['processing_time'],
                    'input_shape': str(r.get('input_shape', '')),
                    'output_shape': str(r.get('output_shape', '')),
                    'slab_thickness_mm': r.get('slab_thickness_mm', 0),
                    'coverage_fallback': r.get('coverage_fallback', False),
                    'output_3d_path': r.get('output_3d_path', ''),
                    'output_1d_path': r.get('output_1d_path', ''),
                    'error': r.get('error', ''),
                    'error_type': r.get('error_type', '')
                })

            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / f"downsampling_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV摘要已保存: {csv_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量Downsampling MRI数据到CEST分辨率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理所有被试
  python batch_downsampling_pipeline.py

  # 指定输入输出目录
  python batch_downsampling_pipeline.py --input-dir /path/to/3D_validated --output-dir /path/to/output

  # 测试模式（处理前3个）
  python batch_downsampling_pipeline.py --test-only

  # 强制重新处理（不跳过已存在文件）
  python batch_downsampling_pipeline.py --no-skip-existing
        """
    )

    parser.add_argument('--input-dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated',
                       help='输入目录（3D验证数据）')
    parser.add_argument('--output-dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling',
                       help='输出根目录（日志、报告等）')
    parser.add_argument('--output-3d-dir', type=str,
                       default=None,
                       help='3D数据输出目录（默认为output-dir/3d）')
    parser.add_argument('--output-1d-dir', type=str,
                       default=None,
                       help='1D数据输出目录（默认为output-dir/1d）')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='不跳过已存在的文件（强制重新处理）')
    parser.add_argument('--test-only', action='store_true',
                       help='测试模式（只处理前3个被试）')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--target-spacing', type=str, default='1.8,1.8,3.0',
                       help='目标分辨率 (X,Y,Z) in mm，逗号分隔')
    parser.add_argument('--verify-mode', type=str, default='smart',
                       choices=['smart', 'full', 'lightweight', 'none'],
                       help='验证模式：\n'
                            '  smart: 智能验证（首尾完整验证，中间轻量级，推荐）\n'
                            '  full: 完整验证所有被试（慢）\n'
                            '  lightweight: 轻量级验证所有被试（快）\n'
                            '  none: 跳过验证（最快）')

    args = parser.parse_args()

    # 解析target_spacing
    target_spacing = tuple(map(float, args.target_spacing.split(',')))

    # 创建批量处理器
    processor = BatchDownsamplingProcessor(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        log_level=args.log_level,
        target_spacing=target_spacing,
        output_3d_dir=Path(args.output_3d_dir) if args.output_3d_dir else None,
        output_1d_dir=Path(args.output_1d_dir) if args.output_1d_dir else None
    )

    try:
        if args.test_only:
            print("测试模式：处理前3个被试")
            # 手动处理前3个
            index_mapping = processor.create_dataset_index()
            test_subjects = list(index_mapping.keys())[:3]
            for subject_id in test_subjects:
                mat_path = Path(index_mapping[subject_id]['filepath'])
                result = processor.process_single_subject(mat_path, subject_id)
                processor.processing_results.append(result)
            processor._generate_batch_report(0)
        else:
            successful, failed = processor.process_all_subjects(
                skip_existing=not args.no_skip_existing,
                verify_mode=args.verify_mode
            )

        # 最终结果
        if len(processor.failed_subjects) == 0:
            processor.logger.info(f"\n🎉 所有被试Downsampling成功！")
        else:
            processor.logger.warning(f"\n⚠️ 有 {len(processor.failed_subjects)} 个被试处理失败，请检查日志")

    except KeyboardInterrupt:
        processor.logger.warning("用户中断程序")
        processor.save_checkpoint()
        processor.generate_emergency_report()
    except Exception as e:
        processor.logger.error(f"程序异常退出: {e}")
        processor.logger.error(traceback.format_exc())
        processor.save_checkpoint()
        processor.generate_emergency_report()
        raise


if __name__ == "__main__":
    # 如果直接运行，提供交互式界面
    import sys

    if len(sys.argv) == 1:
        print("="*80)
        print("MRI Multi-modal Downsampling Pipeline - 批量处理工具")
        print("Version 1.2.0 (BUGFIX v1.2 applied)")
        print("="*80)
        print("\n特性：")
        print("1. 自动遍历所有3D验证数据")
        print("2. 详细日志记录和进度跟踪")
        print("3. 断点续传支持")
        print("4. 内存监控和错误处理")
        print("5. 生成QA报告和处理摘要")
        print("="*80)

        processor = BatchDownsamplingProcessor(
            input_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated"),
            output_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling/3d"),
            log_level='INFO'
        )

        print(f"\n输入目录: {processor.input_dir}")
        print(f"输出目录: {processor.output_dir}")
        print(f"已完成被试: {len(processor.completed_subjects)}")

        response = input("\n选择模式:\n1. 测试（只处理前3个被试）\n2. 处理所有被试\n3. 断点续传（继续之前的任务）\n请输入 (1/2/3): ")

        try:
            if response == '1':
                print("\n测试模式启动...")
                index_mapping = processor.create_dataset_index()
                test_subjects = list(index_mapping.keys())[:3]
                for idx, subject_id in enumerate(test_subjects):
                    mat_path = Path(index_mapping[subject_id]['filepath'])
                    result = processor.process_single_subject(
                        mat_path, subject_id,
                        subject_idx=idx,
                        total_subjects=len(test_subjects),
                        verify_mode='smart'
                    )
                    processor.processing_results.append(result)
                processor._generate_batch_report(0)
            elif response == '2':
                print("\n全量处理模式启动...")
                successful, failed = processor.process_all_subjects(
                    skip_existing=True,
                    verify_mode='smart'
                )
            elif response == '3':
                print(f"\n断点续传模式启动...")
                print(f"已完成被试: {sorted(processor.completed_subjects)}")
                successful, failed = processor.process_all_subjects(
                    skip_existing=True,
                    verify_mode='smart'
                )
            else:
                print("无效选择")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            processor.save_checkpoint()
            processor.generate_emergency_report()
    else:
        main()
