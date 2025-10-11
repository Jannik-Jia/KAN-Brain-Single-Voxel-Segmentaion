#!/usr/bin/env python3
"""
3D和1D数据互相映射工具
支持任意维度的3D数据降采样后的1D转换和映射
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging
from scipy.ndimage import zoom


class Data3D1DMapper:
    """3D和1D数据互相映射的工具类"""

    def __init__(self, log_level: str = 'INFO', expected_features: int = 351):
        """
        初始化映射器

        Args:
            log_level: 日志级别
            expected_features: 期望的特征数量（默认351），用于判断是否需要转置
        """
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.expected_features = expected_features
        self.logger.info(f"初始化Data3D1DMapper，期望特征数: {expected_features}")

    def downsample_3d_volume(self, volume: np.ndarray,
                            target_shape: Optional[Tuple[int, ...]] = None,
                            scale_factors: Optional[Tuple[float, ...]] = None) -> np.ndarray:
        """
        降采样3D体积

        Args:
            volume: 输入3D或4D体积
            target_shape: 目标形状 (x, y, z) 或 (x, y, z, channels)
            scale_factors: 缩放因子 (sx, sy, sz) 或 (sx, sy, sz, 1)

        Returns:
            降采样后的体积
        """
        if target_shape is not None and scale_factors is not None:
            raise ValueError("不能同时指定target_shape和scale_factors")

        if target_shape is not None:
            # 计算缩放因子
            if volume.ndim == 4:
                # 4D数据，保持最后一维不变
                scale_factors = tuple(t/s for t, s in zip(target_shape[:3], volume.shape[:3]))
                scale_factors = scale_factors + (1,)  # 特征维度不缩放
            else:
                scale_factors = tuple(t/s for t, s in zip(target_shape, volume.shape))

        if scale_factors is None:
            raise ValueError("必须指定target_shape或scale_factors之一")

        # 对于掩码使用最近邻插值，对于连续数据使用线性插值
        if volume.dtype == bool or np.issubdtype(volume.dtype, np.integer):
            order = 0  # 最近邻
        else:
            order = 1  # 线性

        downsampled = zoom(volume, scale_factors, order=order)

        # 对于布尔掩码，确保结果是布尔类型
        if volume.dtype == bool:
            downsampled = downsampled > 0.5

        self.logger.info(f"降采样: {volume.shape} -> {downsampled.shape}")
        return downsampled

    def convert_3d_to_1d(self, data_3d: Dict[str, np.ndarray],
                        downsample_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, np.ndarray]:
        """
        将3D数据转换为1D格式

        Args:
            data_3d: 包含3D数据的字典
                - 'data': (x, y, z, features) 4D特征数据
                - 'region_mask': (x, y, z) 3D ROI掩码
                - 'region_labels': (x, y, z) 3D标签（从seg_one_hot转换的）
                - 'region_seg_3d': (x, y, z) FreeSurfer原始分割标签【不需要，可忽略】
                - 'big_seg': (x, y, z) 完整大脑分割
            downsample_shape: 可选的降采样目标形状 (x, y, z)

        Returns:
            包含1D数据的字典（与原始1D格式一致）
        """
        self.logger.info("开始3D到1D转换...")

        # 获取原始形状（支持两种命名）
        if 'region_mask' in data_3d:
            region_mask = data_3d['region_mask'].astype(bool)
        elif 'region' in data_3d:
            region_mask = data_3d['region'].astype(bool)
        else:
            raise ValueError("需要region_mask或region作为ROI掩码")

        original_shape = region_mask.shape
        self.logger.info(f"原始3D形状: {original_shape}")

        # 降采样处理
        if downsample_shape is not None:
            self.logger.info(f"降采样到: {downsample_shape}")

            # 降采样region_mask
            region_mask = self.downsample_3d_volume(
                region_mask,
                target_shape=downsample_shape
            ).astype(bool)

            # 降采样其他数据
            if 'data' in data_3d:
                data_3d['data'] = self.downsample_3d_volume(
                    data_3d['data'],
                    target_shape=downsample_shape + (data_3d['data'].shape[-1],)
                )

            if 'region_labels' in data_3d:
                data_3d['region_labels'] = self.downsample_3d_volume(
                    data_3d['region_labels'],
                    target_shape=downsample_shape
                ).astype(np.uint8)

            if 'region_seg_3d' in data_3d:
                data_3d['region_seg_3d'] = self.downsample_3d_volume(
                    data_3d['region_seg_3d'],
                    target_shape=downsample_shape
                ).astype(data_3d['region_seg_3d'].dtype)

            if 'big_seg' in data_3d:
                data_3d['big_seg'] = self.downsample_3d_volume(
                    data_3d['big_seg'],
                    target_shape=downsample_shape
                ).astype(data_3d['big_seg'].dtype)

        # 获取ROI内的体素数
        n_voxels = np.sum(region_mask)
        self.logger.info(f"ROI内体素数: {n_voxels}")

        # 创建1D数据字典（与原始格式一致）
        data_1d = {
            'region': region_mask,  # 保持3D的ROI掩码
            'original_shape': original_shape,  # 保存原始形状信息
            'downsampled_shape': region_mask.shape if downsample_shape is not None else original_shape,
            'n_voxels': n_voxels
        }

        # 转换4D特征数据到2D
        if 'data' in data_3d:
            features_4d = data_3d['data']
            n_features = features_4d.shape[-1]
            # 提取ROI内的特征
            features_1d = features_4d[region_mask]  # (n_voxels, n_features)
            data_1d['multidim_data'] = features_1d
            self.logger.info(f"特征数据: {features_4d.shape} -> {features_1d.shape}")

        # 转换3D标签到1D
        if 'region_labels' in data_3d:
            labels_3d = data_3d['region_labels']
            labels_1d = labels_3d[region_mask]  # (n_voxels,)

            # 创建one-hot编码 (固定102个类别，0-101)
            n_classes = 102  # 标签固定为102个类别(0-101)
            seg_one_hot = np.zeros((n_classes, n_voxels), dtype=np.float32)
            for i in range(n_voxels):
                label_idx = int(labels_1d[i])
                if label_idx < n_classes:
                    seg_one_hot[label_idx, i] = 1
                else:
                    self.logger.warning(f"标签值 {label_idx} 超出范围[0-101]")
            data_1d['seg_one_hot'] = seg_one_hot
            self.logger.info(f"标签数据: {labels_3d.shape} -> {labels_1d.shape}")
            self.logger.info(f"One-hot编码: {seg_one_hot.shape} (类别数: {n_classes})")

        # 转换3D FreeSurfer标签到1D【不需要，仅为兼容性保留】
        if 'region_seg_3d' in data_3d:
            region_seg_3d = data_3d['region_seg_3d']
            region_seg_1d = region_seg_3d[region_mask]  # (n_voxels,)
            data_1d['region_seg'] = region_seg_1d
            self.logger.info(f"FreeSurfer标签【不需要】: {region_seg_3d.shape} -> {region_seg_1d.shape}")

        # 保持3D的完整分割（如果存在）
        if 'big_seg' in data_3d:
            data_1d['big_seg'] = data_3d['big_seg']

        return data_1d

    def convert_1d_to_3d(self, data_1d: Dict[str, np.ndarray],
                        target_shape: Optional[Tuple[int, int, int]] = None,
                        prob_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        将1D数据转换回3D格式

        Args:
            data_1d: 包含1D数据的字典
                - 'multidim_data': (n_voxels, features) 2D特征数据
                - 'region': (x, y, z) 3D ROI掩码
                - 'seg_one_hot': (n_classes, n_voxels) one-hot标签
                - 其他数据...
            target_shape: 可选的目标3D形状 (x, y, z)，用于上采样
            prob_idx: 可选的被试索引

        Returns:
            包含3D数据的字典
        """
        self.logger.info("开始1D到3D转换...")

        # 获取ROI掩码
        region_mask = data_1d['region'].astype(bool)
        current_shape = region_mask.shape
        n_voxels = np.sum(region_mask)

        self.logger.info(f"当前3D形状: {current_shape}")
        self.logger.info(f"ROI内体素数: {n_voxels}")

        # 创建3D数据字典
        data_3d = {
            'region_mask': region_mask.astype(np.uint8)
        }

        # 转换2D特征到4D
        if 'multidim_data' in data_1d:
            features_2d = data_1d['multidim_data']
            n_features = features_2d.shape[1]

            # 创建4D数组
            features_4d = np.zeros((*current_shape, n_features), dtype=features_2d.dtype)
            features_4d[region_mask] = features_2d
            data_3d['data'] = features_4d
            self.logger.info(f"特征数据: {features_2d.shape} -> {features_4d.shape}")

        # 从one-hot转换标签为region_labels
        if 'seg_one_hot' in data_1d:
            seg_one_hot = data_1d['seg_one_hot']
            labels_1d = np.argmax(seg_one_hot, axis=0).astype(np.uint8)
            labels_3d = np.zeros(current_shape, dtype=np.uint8)
            labels_3d[region_mask] = labels_1d
            data_3d['region_labels'] = labels_3d
            self.logger.info(f"从one-hot恢复标签: {seg_one_hot.shape} -> {labels_3d.shape}")

        # 转换1D FreeSurfer标签到3D【不需要，仅为兼容性保留】
        if 'region_seg' in data_1d:
            region_seg_1d = data_1d['region_seg']
            region_seg_3d = np.zeros(current_shape, dtype=region_seg_1d.dtype)
            region_seg_3d[region_mask] = region_seg_1d
            data_3d['region_seg_3d'] = region_seg_3d
            self.logger.info(f"FreeSurfer标签【不需要】: {region_seg_1d.shape} -> {region_seg_3d.shape}")

        # 保持3D的完整分割（直接复制）
        if 'big_seg' in data_1d:
            data_3d['big_seg'] = data_1d['big_seg'].copy()

        # 添加被试索引（如果提供）
        if prob_idx is not None:
            prob_idx_3d = np.zeros(current_shape, dtype=np.uint8)
            prob_idx_3d[region_mask] = prob_idx
            data_3d['prob_idx'] = prob_idx_3d
            self.logger.info(f"添加被试索引: {prob_idx}")

        # 上采样到目标形状（如果需要）
        if target_shape is not None and target_shape != current_shape:
            self.logger.info(f"上采样到目标形状: {current_shape} -> {target_shape}")
            data_3d = self.upsample_3d_data(data_3d, target_shape)

        return data_3d

    def upsample_3d_data(self, data_3d: Dict[str, np.ndarray],
                         target_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        """
        上采样3D数据到目标形状

        Args:
            data_3d: 3D数据字典
            target_shape: 目标形状 (x, y, z)

        Returns:
            上采样后的3D数据
        """
        upsampled = {}

        for key, value in data_3d.items():
            if value.ndim == 3:
                # 3D数据
                upsampled[key] = self.downsample_3d_volume(
                    value, target_shape=target_shape
                )
            elif value.ndim == 4:
                # 4D数据，保持特征维度
                target_4d = target_shape + (value.shape[-1],)
                upsampled[key] = self.downsample_3d_volume(
                    value, target_shape=target_4d
                )
            else:
                # 其他数据直接复制
                upsampled[key] = value.copy()

        return upsampled

    def map_1d_predictions_to_3d(self, predictions_1d: np.ndarray,
                                 region_mask: np.ndarray,
                                 target_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        将1D预测结果映射回3D空间

        Args:
            predictions_1d: (n_voxels,) 或 (n_voxels, n_classes) 1D预测结果
            region_mask: (x, y, z) 3D ROI掩码
            target_shape: 可选的目标形状，用于上采样

        Returns:
            3D预测结果
        """
        region_mask = region_mask.astype(bool)
        current_shape = region_mask.shape

        # 创建3D数组
        if predictions_1d.ndim == 1:
            # 类别预测
            predictions_3d = np.zeros(current_shape, dtype=predictions_1d.dtype)
            predictions_3d[region_mask] = predictions_1d
        else:
            # 概率预测
            n_classes = predictions_1d.shape[1]
            predictions_3d = np.zeros((*current_shape, n_classes), dtype=predictions_1d.dtype)
            predictions_3d[region_mask] = predictions_1d

        # 上采样（如果需要）
        if target_shape is not None and target_shape != current_shape:
            if predictions_3d.ndim == 3:
                predictions_3d = self.downsample_3d_volume(
                    predictions_3d, target_shape=target_shape
                )
            else:
                predictions_3d = self.downsample_3d_volume(
                    predictions_3d, target_shape=target_shape + (n_classes,)
                )

        self.logger.info(f"预测映射: {predictions_1d.shape} -> {predictions_3d.shape}")
        return predictions_3d

    def validate_conversion(self, data_1d_original: Dict[str, np.ndarray],
                           data_1d_recovered: Dict[str, np.ndarray]) -> bool:
        """
        验证1D数据转换的一致性

        Args:
            data_1d_original: 原始1D数据
            data_1d_recovered: 从3D恢复的1D数据

        Returns:
            是否通过验证
        """
        all_valid = True

        # 验证特征数据
        if 'multidim_data' in data_1d_original and 'multidim_data' in data_1d_recovered:
            features_match = np.allclose(
                data_1d_original['multidim_data'],
                data_1d_recovered['multidim_data'],
                rtol=1e-5, atol=1e-8
            )
            if not features_match:
                self.logger.warning("特征数据不匹配")
                all_valid = False
            else:
                self.logger.info("特征数据验证通过 ✓")

        # 验证标签数据
        if 'region_seg' in data_1d_original and 'region_seg' in data_1d_recovered:
            labels_match = np.array_equal(
                data_1d_original['region_seg'],
                data_1d_recovered['region_seg']
            )
            if not labels_match:
                self.logger.warning("标签数据不匹配")
                all_valid = False
            else:
                self.logger.info("标签数据验证通过 ✓")

        return all_valid

    def save_data(self, data: Dict[str, np.ndarray], filepath: Union[str, Path],
                  matlab_compatible: bool = True):
        """
        保存数据到HDF5文件

        Args:
            data: 数据字典
            filepath: 输出文件路径
            matlab_compatible: 是否调整为MATLAB兼容格式
        """
        filepath = Path(filepath)

        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                # 对于标量值，保存为属性
                if np.isscalar(value):
                    f.attrs[key] = value
                    continue

                # MATLAB兼容性调整
                if matlab_compatible:
                    if key == 'data' and value.ndim == 4:
                        # 4D数据：(x,y,z,features) -> (features,x,y,z)
                        value = np.moveaxis(value, -1, 0)
                    # 使用Fortran顺序
                    value = np.asfortranarray(value)

                # 选择压缩级别
                if key == 'data' or 'multidim_data' in key:
                    compression_opts = 1  # 低压缩
                else:
                    compression_opts = 4  # 中压缩

                f.create_dataset(key, data=value, compression='gzip',
                               compression_opts=compression_opts)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        self.logger.info(f"数据已保存: {filepath} ({file_size_mb:.1f} MB)")

    def load_data(self, filepath: Union[str, Path],
                  from_matlab: bool = True) -> Dict[str, np.ndarray]:
        """
        从HDF5文件加载数据

        Args:
            filepath: 输入文件路径
            from_matlab: 是否从MATLAB格式加载

        Returns:
            数据字典
        """
        filepath = Path(filepath)
        data = {}

        with h5py.File(filepath, 'r') as f:
            # 读取属性
            for key, value in f.attrs.items():
                data[key] = value

            # 读取数据集
            for key in f.keys():
                if not key.startswith('#'):
                    value = f[key][()]

                    # MATLAB格式转换 - 与原始代码完全一致
                    if from_matlab:
                        if key == 'multidim_data':
                            original_shape = f[key][()].shape

                            # 方法1：严格判断 - 与原始代码完全一致（仅当第一维==expected_features时转置）
                            if value.shape[0] == self.expected_features:
                                value = value.T
                                self.logger.info(f"✓ 转置 {key}: {original_shape} → {value.shape} (检测到标准{self.expected_features}特征)")
                            # 方法2：通用启发式（当行数<列数时，可能需要转置）
                            elif value.shape[0] < value.shape[1]:
                                self.logger.warning(f"⚠️ {key} 形状异常: {original_shape}")
                                self.logger.warning(f"  第一维({value.shape[0]})不是期望的{self.expected_features}，但小于第二维({value.shape[1]})")
                                self.logger.warning(f"  可能是: 1) 非标准特征数 2) 极小ROI(n_voxels<{value.shape[0]})")
                                # 询问用户或根据配置决定
                                self.logger.warning(f"  按通用规则转置: {original_shape} → {value.T.shape}")
                                value = value.T
                            else:
                                self.logger.info(f"✗ 不转置 {key}: 保持 {value.shape} (不符合转置条件)")
                                if value.shape[1] == self.expected_features:
                                    self.logger.info(f"  检测到数据可能已经是正确格式 (n_voxels={value.shape[0]}, features={self.expected_features})")

                        elif key == 'data' and value.ndim == 4:
                            # 4D数据：(features,x,y,z) -> (x,y,z,features)
                            if value.shape[0] < value.shape[1]:
                                value = np.moveaxis(value, 0, -1)
                                self.logger.debug(f"4D数据维度调整: {f[key][()].shape} → {value.shape}")
                        elif key == 'region_seg':
                            value = value.flatten()  # 展平

                    data[key] = value

        self.logger.info(f"数据已加载: {filepath}")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                self.logger.debug(f"  {key}: {value.shape} ({value.dtype})")

        return data


def example_usage():
    """示例用法"""
    import numpy as np

    # 创建映射器
    mapper = Data3D1DMapper(log_level='INFO')

    # 模拟3D数据
    print("\n=== 创建模拟3D数据 ===")
    region_mask = np.random.rand(64, 64, 64) > 0.7  # 随机ROI
    n_voxels = np.sum(region_mask)

    data_3d = {
        'data': np.random.randn(64, 64, 64, 10),  # 10个特征
        'region_mask': region_mask,
        'region_labels': np.random.randint(0, 5, (64, 64, 64)).astype(np.uint8),
        'big_seg': np.random.randint(0, 100, (64, 64, 64)).astype(np.uint8)
    }

    # 1. 3D转1D（带降采样）
    print("\n=== 3D转1D（降采样到32x32x32）===")
    data_1d = mapper.convert_3d_to_1d(data_3d, downsample_shape=(32, 32, 32))
    print(f"1D数据 multidim_data形状: {data_1d['multidim_data'].shape}")
    print(f"1D数据 region_seg形状: {data_1d['region_seg'].shape}")

    # 2. 1D转回3D
    print("\n=== 1D转回3D ===")
    data_3d_recovered = mapper.convert_1d_to_3d(data_1d, prob_idx=1)
    print(f"恢复的3D数据 data形状: {data_3d_recovered['data'].shape}")
    print(f"恢复的3D数据 region_labels形状: {data_3d_recovered['region_labels'].shape}")

    # 3. 验证转换
    print("\n=== 验证转换一致性 ===")
    # 从恢复的3D再转1D
    data_1d_recovered = mapper.convert_3d_to_1d(data_3d_recovered)
    is_valid = mapper.validate_conversion(data_1d, data_1d_recovered)
    print(f"转换验证: {'通过 ✅' if is_valid else '失败 ❌'}")

    # 4. 模拟预测映射
    print("\n=== 模拟1D预测映射到3D ===")
    # 假设有1D预测结果
    predictions_1d = np.random.randint(0, 5, data_1d['n_voxels'])
    predictions_3d = mapper.map_1d_predictions_to_3d(
        predictions_1d,
        data_1d['region'],
        target_shape=(64, 64, 64)  # 上采样回原始尺寸
    )
    print(f"预测结果3D形状: {predictions_3d.shape}")

    # 5. 保存和加载
    print("\n=== 保存和加载测试 ===")
    mapper.save_data(data_1d, 'test_1d_data.mat')
    loaded_data = mapper.load_data('test_1d_data.mat')
    print("数据保存和加载成功")

    # 清理测试文件
    Path('test_1d_data.mat').unlink(missing_ok=True)


if __name__ == "__main__":
    print("3D-1D数据映射工具")
    print("=" * 50)
    print("功能：")
    print("1. 3D数据转1D（支持降采样）")
    print("2. 1D数据转3D（支持上采样）")
    print("3. 1D预测映射到3D")
    print("4. 数据验证")
    print("5. MATLAB兼容的保存/加载")
    print("=" * 50)

    # 运行示例
    example_usage()