#!/usr/bin/env python3
"""
基于验证结果的最终3D数据转换脚本
根据实际数据形状正确处理每个key
"""

import numpy as np
import h5py
from pathlib import Path
import json
from typing import Dict, Tuple, Any, Optional
import time
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """验证结果数据类"""
    key: str
    passed: bool
    match_rate: float
    max_diff: float
    message: str


class MRIDataConverter3D:
    """MRI数据3D转换器 - 基于实际数据形状的最终版本"""
    
    def __init__(self):
        self.validation_results = []
    
    def load_mat_h5(self, path: Path) -> Dict[str, np.ndarray]:
        """
        正确加载MATLAB HDF5格式的MAT文件
        基于验证结果的实际形状处理
        
        实际形状（已验证）：
        - big_seg: (384, 336, 256) - 3D体，不需要转置
        - region: (384, 336, 256) - 3D体，不需要转置  
        - multidim_data: (351, n_voxels) - 需要转置为 (n_voxels, 351)
        - seg_one_hot: (102, n_voxels) - 保持原样，不转置
        - region_seg: (1, n_voxels) - 展平为1D
        """
        data = {}
        print(f"\n加载文件: {path.name}")
        
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if not k.startswith("#"):
                    v = f[k][()]
                    
                    # 根据验证结果处理每个key
                    if k == 'multidim_data':
                        # (351, n_voxels) → (n_voxels, 351)
                        # 只有这个需要转置！
                        if v.shape[0] == 351:
                            v = v.T
                            print(f"  {k}: {v.T.shape} → {v.shape} [转置]")
                        else:
                            print(f"  {k}: {v.shape} [已经是正确格式]")
                            
                    elif k == 'region_seg':
                        # (1, n_voxels) → (n_voxels,)
                        original_shape = v.shape
                        v = v.flatten()
                        print(f"  {k}: {original_shape} → {v.shape} [展平]")
                        
                    elif k in ['region', 'big_seg']:
                        # 3D体数据：保持原样 (384, 336, 256)
                        print(f"  {k}: {v.shape} [3D体，保持原样]")
                        
                    elif k == 'seg_one_hot':
                        # (102, n_voxels) - 保持原样
                        print(f"  {k}: {v.shape} [One-hot，保持原样]")
                        
                    else:
                        print(f"  {k}: {v.shape} [其他]")
                    
                    data[k] = v
        
        return data
    
    def save_mat_h5(self, path: Path, data_dict: Dict[str, np.ndarray]):
        """
        保存为MATLAB兼容的HDF5格式
        保持与原始文件相同的格式，便于MATLAB读取
        """
        print(f"\n保存文件: {path.name}")
        
        with h5py.File(path, 'w') as f:
            for key, value in data_dict.items():
                save_value = value
                
                # 根据key类型决定保存格式
                if key == 'data':
                    # 4D数据: (384, 336, 256, 351)
                    # 保存时转回 (351, 384, 336, 256) 以匹配MATLAB习惯
                    if value.ndim == 4:
                        # 将最后一维移到最前面
                        save_value = np.moveaxis(value, -1, 0)
                        print(f"  {key}: {value.shape} → {save_value.shape} [4D调整为MATLAB格式]")
                    else:
                        print(f"  {key}: {value.shape} [保持]")
                        
                elif key in ['region_labels', 'prob_idx', 'region_mask', 'region_seg_3d']:
                    # 3D数据：保持 (384, 336, 256)
                    print(f"  {key}: {value.shape} [3D体]")
                    
                elif key in ['region', 'big_seg']:
                    # 3D原始数据：保持原样
                    print(f"  {key}: {value.shape} [3D原始]")
                    
                else:
                    print(f"  {key}: {value.shape} [其他]")
                
                # 保存数据，使用Fortran连续性以优化MATLAB读取
                save_value = np.asfortranarray(save_value)
                
                # 根据数据大小选择压缩级别
                if key == 'data' and save_value.size > 1e7:
                    f.create_dataset(key, data=save_value, compression='gzip', compression_opts=1)
                else:
                    f.create_dataset(key, data=save_value, compression='gzip', compression_opts=4)
    
    def revert_reshape(self, array: np.ndarray, region: np.ndarray) -> np.ndarray:
        """
        Mert的revert_reshape函数
        将1D数组根据region掩码重塑为3D体积
        
        Parameters:
        - array: 1D或2D numpy数组 (num_voxels,) 或 (num_voxels, num_features)
        - region: 3D numpy数组 (384, 336, 256) 定义有效体素位置
        
        Returns:
        - big_img: 3D或4D numpy数组
        """
        if array.ndim == 1:
            array = array[:, np.newaxis]
        
        num_features = array.shape[1]
        # 创建输出数组 - region已经是 (384, 336, 256)
        big_img = np.zeros((*region.shape, num_features), dtype=array.dtype)
        
        # 使用Fortran顺序重塑和填充
        big_img = big_img.reshape(-1, array.shape[1], order='F')
        bool_mask = region.flatten(order='F').astype(bool)
        big_img[bool_mask] = array
        big_img = big_img.reshape((*region.shape, num_features), order='F')
        
        # 如果只有一个特征，去掉最后一维
        if num_features == 1:
            big_img = big_img.squeeze()
        
        return big_img
    
    def extract_1d_from_3d(self, volume: np.ndarray, region: np.ndarray) -> np.ndarray:
        """
        从3D体积提取1D数据（revert_reshape的逆操作）
        
        Parameters:
        - volume: 3D或4D数组 (384, 336, 256) 或 (384, 336, 256, n_features)
        - region: 3D布尔掩码 (384, 336, 256)
        
        Returns:
        - array_1d: 1D或2D数组 (n_voxels,) 或 (n_voxels, n_features)
        """
        if volume.ndim == 3:
            volume = volume[..., np.newaxis]
        
        # 使用Fortran顺序展平
        volume_flat = volume.reshape(-1, volume.shape[-1], order='F')
        mask_flat = region.flatten(order='F').astype(bool)
        
        # 提取有效体素
        array_1d = volume_flat[mask_flat]
        
        if array_1d.shape[1] == 1:
            array_1d = array_1d.squeeze()
        
        return array_1d
    
    def validate_round_trip(self, original_1d: np.ndarray, region: np.ndarray, 
                          key_name: str) -> ValidationResult:
        """
        验证1D→3D→1D往返转换的正确性
        """
        # 1D → 3D
        volume_3d = self.revert_reshape(original_1d, region)
        
        # 3D → 1D
        restored_1d = self.extract_1d_from_3d(volume_3d, region)
        
        # 确保形状一致
        if original_1d.shape != restored_1d.shape:
            return ValidationResult(
                key=key_name,
                passed=False,
                match_rate=0.0,
                max_diff=float('inf'),
                message=f"形状不匹配: 原始{original_1d.shape} vs 恢复{restored_1d.shape}"
            )
        
        # 计算匹配度
        if np.issubdtype(original_1d.dtype, np.integer):
            # 整数类型：精确匹配
            matches = np.sum(original_1d == restored_1d)
            total = original_1d.size
            match_rate = 100.0 * matches / total
            max_diff = 0 if match_rate == 100 else 1
            
            passed = match_rate == 100.0
            message = f"整数匹配率: {match_rate:.2f}% ({matches}/{total})"
        else:
            # 浮点类型：允许小误差
            max_diff = np.max(np.abs(original_1d - restored_1d))
            mean_diff = np.mean(np.abs(original_1d - restored_1d))
            
            # 计算相对误差
            non_zero_mask = original_1d != 0
            if np.any(non_zero_mask):
                rel_error = np.max(np.abs((original_1d[non_zero_mask] - restored_1d[non_zero_mask]) 
                                         / original_1d[non_zero_mask]))
            else:
                rel_error = 0
            
            passed = max_diff < 1e-6
            match_rate = 100.0 if passed else (100.0 * (1 - min(rel_error, 1)))
            message = f"最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e}"
        
        result = ValidationResult(
            key=key_name,
            passed=passed,
            match_rate=match_rate,
            max_diff=max_diff,
            message=message
        )
        
        # 打印结果
        status = "✅" if passed else "❌"
        print(f"  {status} {key_name}: {message}")
        
        return result
    
    def process_multidim_data(self, data: Dict, region: np.ndarray) -> Tuple[np.ndarray, ValidationResult]:
        """
        处理multidim_data: (n_voxels, 351) → (384, 336, 256, 351)
        注意：数据已经在加载时转置为 (n_voxels, 351)
        """
        print("\n处理 multidim_data...")
        multidim_data = data['multidim_data']  # 已经是 (n_voxels, 351)
        
        # 验证形状
        n_voxels = np.sum(region.astype(bool))
        print(f"  输入形状: {multidim_data.shape}")
        print(f"  期望体素数: {n_voxels:,}")
        
        if multidim_data.shape[0] != n_voxels:
            print(f"  ⚠️ 体素数不匹配: 实际{multidim_data.shape[0]:,}")
        
        # 批量转换为4D
        n_voxels, n_features = multidim_data.shape
        data_4d = np.zeros((*region.shape, n_features), dtype=multidim_data.dtype)
        
        # 使用Fortran顺序处理
        data_4d_flat = data_4d.reshape(-1, n_features, order='F')
        bool_mask = region.flatten(order='F').astype(bool)
        data_4d_flat[bool_mask] = multidim_data
        data_4d = data_4d_flat.reshape((*region.shape, n_features), order='F')
        
        print(f"  输出形状: {data_4d.shape}")
        
        # 验证（随机抽样几个特征维度）
        print("\n验证往返转换:")
        sample_dims = np.random.choice(n_features, min(3, n_features), replace=False)
        all_passed = True
        
        for dim in sample_dims:
            result = self.validate_round_trip(
                multidim_data[:, dim], 
                region, 
                f"特征维度_{dim}"
            )
            if not result.passed:
                all_passed = False
        
        # 创建总体验证结果
        validation = ValidationResult(
            key="multidim_data",
            passed=all_passed,
            match_rate=100.0 if all_passed else 99.0,
            max_diff=0 if all_passed else 1e-6,
            message=f"验证了{len(sample_dims)}个维度，{'全部通过' if all_passed else '部分失败'}"
        )
        
        return data_4d, validation
    
    def process_seg_one_hot(self, data: Dict, region: np.ndarray) -> Tuple[np.ndarray, ValidationResult]:
        """
        处理seg_one_hot: (102, n_voxels) → 3D整数标签 (384, 336, 256)
        注意：seg_one_hot已经是正确格式 (102, n_voxels)，不需要转置
        """
        print("\n处理 seg_one_hot...")
        seg_one_hot = data['seg_one_hot']  # (102, n_voxels)
        print(f"  输入形状: {seg_one_hot.shape}")
        
        # 验证形状
        if seg_one_hot.shape[0] != 102:
            print(f"  ⚠️ 警告：第一维应该是102，实际是{seg_one_hot.shape[0]}")
        
        # 转换为整数标签 - axis=0 因为是 (102, n_voxels)
        labels_1d = np.argmax(seg_one_hot, axis=0).astype(np.uint8)
        print(f"  1D标签形状: {labels_1d.shape}")
        print(f"  标签范围: {labels_1d.min()} - {labels_1d.max()}")
        print(f"  唯一标签数: {len(np.unique(labels_1d))}")
        
        # 转换为3D
        labels_3d = self.revert_reshape(labels_1d, region)
        print(f"  输出形状: {labels_3d.shape}")
        
        # 验证
        print("\n验证往返转换:")
        validation = self.validate_round_trip(labels_1d, region, "seg_one_hot标签")
        
        return labels_3d, validation
    
    def process_region_seg(self, data: Dict, region: np.ndarray) -> Tuple[np.ndarray, ValidationResult]:
        """
        处理region_seg: (n_voxels,) → 3D (384, 336, 256)
        注意：已经在加载时展平
        """
        print("\n处理 region_seg...")
        region_seg = data['region_seg']  # 已经是1D (n_voxels,)
        print(f"  输入形状: {region_seg.shape}")
        print(f"  值范围: {region_seg.min():.0f} - {region_seg.max():.0f}")
        print(f"  唯一值数: {len(np.unique(region_seg))}")
        
        # 转换为3D
        region_seg_3d = self.revert_reshape(region_seg, region)
        print(f"  输出形状: {region_seg_3d.shape}")
        
        # 验证
        print("\n验证往返转换:")
        validation = self.validate_round_trip(region_seg, region, "region_seg")
        
        return region_seg_3d, validation
    
    def create_prob_idx_volume(self, region: np.ndarray, prob_idx: int) -> np.ndarray:
        """
        创建prob_idx的3D体积
        """
        print(f"\n创建 prob_idx 体积...")
        print(f"  被试索引: {prob_idx}")
        
        # 创建填充了prob_idx的1D数组
        n_voxels = np.sum(region.astype(bool))
        prob_idx_1d = np.full(n_voxels, prob_idx, dtype=np.uint8)
        
        # 转换为3D
        prob_idx_3d = self.revert_reshape(prob_idx_1d, region)
        print(f"  输出形状: {prob_idx_3d.shape}")
        
        # 验证
        unique_vals = np.unique(prob_idx_3d[region.astype(bool)])
        if len(unique_vals) == 1 and unique_vals[0] == prob_idx:
            print(f"  ✅ 验证通过: 所有有效体素值为 {prob_idx}")
        else:
            print(f"  ❌ 验证失败: 期望 {prob_idx}, 实际 {unique_vals}")
        
        return prob_idx_3d
    
    def process_single_subject(self, mat_path: Path, prob_idx: int, 
                              output_dir: Path, validate_all: bool = True) -> Path:
        """
        处理单个被试的完整转换流程
        """
        print(f"\n{'='*60}")
        print(f"处理被试 {prob_idx}: {mat_path.name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 加载数据
        data = self.load_mat_h5(mat_path)
        
        # 获取region掩码 - 已经是 (384, 336, 256)
        region = data['region'].astype(bool)
        print(f"\nRegion信息:")
        print(f"  形状: {region.shape}")
        print(f"  有效体素数: {np.sum(region):,}")
        print(f"  占比: {100*np.sum(region)/region.size:.1f}%")
        
        # 准备输出数据字典
        output_data = {}
        self.validation_results = []
        
        # 1. 处理multidim_data
        if 'multidim_data' in data:
            data_4d, validation = self.process_multidim_data(data, region)
            output_data['data'] = data_4d
            self.validation_results.append(validation)
        
        # 2. 处理seg_one_hot → region标签
        if 'seg_one_hot' in data:
            labels_3d, validation = self.process_seg_one_hot(data, region)
            output_data['region_labels'] = labels_3d
            self.validation_results.append(validation)
        
        # 3. 处理region_seg
        if 'region_seg' in data:
            region_seg_3d, validation = self.process_region_seg(data, region)
            output_data['region_seg_3d'] = region_seg_3d
            self.validation_results.append(validation)
        
        # 4. 创建prob_idx体积
        prob_idx_3d = self.create_prob_idx_volume(region, prob_idx)
        output_data['prob_idx'] = prob_idx_3d
        
        # 5. 复制原始3D数据（保持原样）
        output_data['big_seg'] = data['big_seg']  # (384, 336, 256)
        output_data['region_mask'] = region.astype(np.uint8)  # (384, 336, 256)
        print(f"\n复制原始3D数据:")
        print(f"  big_seg: {data['big_seg'].shape}")
        print(f"  region_mask: {region.shape}")
        
        # 打印验证总结
        print(f"\n{'='*40}")
        print("验证总结:")
        all_passed = all(v.passed for v in self.validation_results)
        for result in self.validation_results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.key}: {result.message}")
        
        if all_passed:
            print(f"\n🎉 所有验证通过！")
        else:
            print(f"\n⚠️ 部分验证失败，请检查")
        
        # 保存数据
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{mat_path.stem}_3d.mat"
        output_path = output_dir / output_filename
        
        self.save_mat_h5(output_path, output_data)
        
        # 计算处理时间
        elapsed = time.time() - start_time
        print(f"\n处理完成:")
        print(f"  时间: {elapsed:.1f}秒")
        print(f"  输出: {output_path}")
        
        # 文件大小
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  大小: {file_size_mb:.1f} MB")
        
        return output_path
    
    def verify_saved_file(self, file_path: Path) -> bool:
        """
        验证保存的3D文件
        """
        print(f"\n{'='*60}")
        print(f"验证文件: {file_path.name}")
        print(f"{'='*60}")
        
        # 直接用h5py查看原始形状
        print("\nHDF5原始形状:")
        with h5py.File(file_path, 'r') as f:
            for key in sorted(f.keys()):
                dataset = f[key]
                print(f"  {key:20s}: {str(dataset.shape):20s}")
        
        # 用我们的加载器读取
        print("\n通过load_mat_h5加载:")
        data = self.load_mat_h5(file_path)
        
        print("\n加载后的形状:")
        for key in sorted(data.keys()):
            value = data[key]
            print(f"  {key:20s}: {str(value.shape):20s}")
        
        # 基本检查
        checks = {}
        
        # 检查必要的keys
        required_keys = ['data', 'region_labels', 'prob_idx', 'big_seg', 'region_mask']
        for key in required_keys:
            checks[f'has_{key}'] = key in data
        
        # 检查形状
        if 'data' in data:
            # 加载后应该是 (n_voxels, 351)，但保存时是 (351, 384, 336, 256)
            # 这里我们看到的是加载后的形状
            checks['data_dims'] = data['data'].ndim == 4
            
        if 'region_labels' in data:
            checks['labels_dims'] = data['region_labels'].ndim == 3
            checks['labels_shape'] = data['region_labels'].shape == (384, 336, 256)
        
        all_passed = all(checks.values())
        
        print(f"\n验证结果: {'✅ 通过' if all_passed else '❌ 失败'}")
        for check, passed in checks.items():
            print(f"  {'✅' if passed else '❌'} {check}")
        
        return all_passed


def main():
    """主函数：处理单个被试作为示例"""
    
    # 设置路径 - 请根据您的实际情况修改
    data_dir = Path("/home/jannik/Documents/mri_mat_onehot/1D")
    output_dir = Path("/home/jannik/Documents/mri_mat_onehot/3D_final")
    
    # 创建转换器
    converter = MRIDataConverter3D()
    
    # 处理第一个被试作为测试
    mat_files = sorted(list(data_dir.glob("*.mat")))
    if mat_files:
        first_file = mat_files[0]
        prob_idx = 1
        
        print(f"="*60)
        print("3D数据转换 - 最终版本")
        print(f"="*60)
        print(f"\n基于验证的数据处理策略:")
        print(f"  • region/big_seg: (384,336,256) 保持原样")
        print(f"  • multidim_data: (351,n_voxels) → 转置为 (n_voxels,351)")
        print(f"  • seg_one_hot: (102,n_voxels) 保持原样")
        print(f"  • region_seg: (1,n_voxels) → 展平")
        
        # 执行转换
        output_path = converter.process_single_subject(
            mat_path=first_file,
            prob_idx=prob_idx,
            output_dir=output_dir,
            validate_all=True
        )
        
        # 验证保存的文件
        converter.verify_saved_file(output_path)
        
        print(f"\n{'='*60}")
        print("完成！")
        print(f"{'='*60}")
    else:
        print(f"未找到MAT文件在: {data_dir}")


if __name__ == "__main__":
    main()