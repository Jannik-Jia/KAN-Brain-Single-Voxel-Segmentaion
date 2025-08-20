#!/usr/bin/env python3
"""
往返转换验证：1D→3D→1D 完整循环测试
验证数据转换的完整正确性，确保逐像素一致
"""

import numpy as np
import h5py
from pathlib import Path
import time
from typing import Dict, Tuple, Any

class RoundTripValidator:
    """1D→3D→1D往返转换验证器"""
    
    def __init__(self):
        self.results = {}
    
    def load_mat_h5_correct(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """正确的MAT文件加载方法"""
        print(f"加载原始MAT文件: {mat_path.name}")
        
        data = {}
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                if not k.startswith("#"):
                    v = f[k][()]
                    
                    # 关键：只转置需要转置的数据
                    if k == 'multidim_data' and v.shape[0] == 351:
                        # 特征矩阵：(351, n_voxels) → (n_voxels, 351)
                        print(f"  转置 {k}: {v.shape} → {v.T.shape}")
                        v = v.T
                    elif k == 'region_seg':
                        # 1D标签：(1, n_voxels) → (n_voxels,)
                        original_shape = v.shape
                        v = v.flatten()
                        print(f"  展平 {k}: {original_shape} → {v.shape}")
                    else:
                        print(f"  保持 {k}: {v.shape}")
                    
                    data[k] = v
        
        return data
    
    def convert_1d_to_3d(self, data: Dict[str, np.ndarray], prob_idx: int = 1) -> Dict[str, np.ndarray]:
        """将1D数据转换为3D格式"""
        print(f"\n执行 1D → 3D 转换...")
        
        region = data['region'].astype(bool)
        output_data = {}
        
        print(f"Region信息: {region.shape}, 有效体素: {np.sum(region):,}")
        
        # 1. 3D体数据（直接保留）
        output_data['big_seg'] = data['big_seg'].copy()
        output_data['region_mask'] = data['region'].astype(np.uint8)
        print(f"  保留3D数据: big_seg {data['big_seg'].shape}, region_mask {data['region'].shape}")
        
        # 2. 多模态特征数据：(n_voxels, 351) → (384, 336, 256, 351)
        features = data['multidim_data']  # 已经是 (n_voxels, 351)
        n_voxels, n_features = features.shape
        
        data_4d = np.zeros((*region.shape, n_features), dtype=features.dtype)
        mask = region.astype(bool)
        data_4d[mask] = features  # 直接布尔索引
        output_data['data'] = data_4d
        print(f"  特征重构: {features.shape} → {data_4d.shape}")
        
        # 3. One-Hot标签转换：(102, n_voxels) → (384, 336, 256)
        seg_one_hot = data['seg_one_hot']  # (102, n_voxels)
        labels_1d = np.argmax(seg_one_hot, axis=0).astype(np.uint8)
        
        labels_3d = np.zeros(region.shape, dtype=np.uint8)
        labels_3d[mask] = labels_1d
        output_data['region_labels'] = labels_3d
        print(f"  标签转换: {seg_one_hot.shape} → argmax → {labels_3d.shape}")
        
        # 4. 原始标签重构：(n_voxels,) → (384, 336, 256)
        region_seg = data['region_seg']  # (n_voxels,)
        region_seg_3d = np.zeros(region.shape, dtype=region_seg.dtype)
        region_seg_3d[mask] = region_seg
        output_data['region_seg_3d'] = region_seg_3d
        print(f"  region_seg重构: {region_seg.shape} → {region_seg_3d.shape}")
        
        # 5. 被试索引体积
        prob_idx_3d = np.zeros(region.shape, dtype=np.uint8)
        prob_idx_3d[mask] = prob_idx
        output_data['prob_idx'] = prob_idx_3d
        print(f"  创建prob_idx: 标量 {prob_idx} → {prob_idx_3d.shape}")
        
        return output_data
    
    def convert_3d_to_1d(self, data_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """将3D数据转换回1D格式"""
        print(f"\n执行 3D → 1D 转换...")
        
        # 获取region掩码
        region_mask = data_3d['region_mask'].astype(bool)
        n_voxels = np.sum(region_mask)
        print(f"有效体素数: {n_voxels:,}")
        
        output_data = {}
        
        # 1. 3D体数据（直接保留）
        output_data['big_seg'] = data_3d['big_seg'].copy()
        output_data['region'] = data_3d['region_mask'].copy()
        print(f"  保留3D数据: big_seg {data_3d['big_seg'].shape}, region {data_3d['region_mask'].shape}")
        
        # 2. 4D特征数据提取：(384, 336, 256, 351) → (n_voxels, 351)
        data_4d = data_3d['data']  # (384, 336, 256, 351)
        features_extracted = data_4d[region_mask]  # 直接布尔索引
        output_data['multidim_data'] = features_extracted
        print(f"  特征提取: {data_4d.shape} → {features_extracted.shape}")
        
        # 3. 3D标签提取：(384, 336, 256) → (n_voxels,)
        labels_3d = data_3d['region_labels']
        labels_extracted = labels_3d[region_mask]
        
        # 转换为one-hot格式：(n_voxels,) → (102, n_voxels)
        n_classes = 102
        one_hot = np.zeros((n_classes, n_voxels), dtype=np.uint8)
        one_hot[labels_extracted, np.arange(n_voxels)] = 1
        output_data['seg_one_hot'] = one_hot
        print(f"  标签提取: {labels_3d.shape} → {labels_extracted.shape} → one-hot {one_hot.shape}")
        
        # 4. region_seg提取：(384, 336, 256) → (n_voxels,)
        region_seg_3d = data_3d['region_seg_3d']
        region_seg_extracted = region_seg_3d[region_mask]
        output_data['region_seg'] = region_seg_extracted
        print(f"  region_seg提取: {region_seg_3d.shape} → {region_seg_extracted.shape}")
        
        return output_data
    
    def compare_data(self, original: Dict[str, np.ndarray], recovered: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """逐像素对比原始数据和恢复数据"""
        print(f"\n执行逐像素对比验证...")
        
        comparison_results = {}
        
        # 比较每个数据字段
        for key in ['big_seg', 'region', 'multidim_data', 'region_seg', 'seg_one_hot']:
            if key in original and key in recovered:
                orig = original[key]
                recov = recovered[key]
                
                result = self._compare_arrays(orig, recov, key)
                comparison_results[key] = result
                
                status = "✅" if result['perfect_match'] else "❌"
                print(f"  {status} {key:15s}: {result['match_rate']:6.2f}% ({result['matches']:,}/{result['total']:,})")
                
                if not result['perfect_match']:
                    print(f"      最大差异: {result['max_diff']:.2e}, 平均差异: {result['mean_diff']:.2e}")
            else:
                print(f"  ⚠️ {key:15s}: 数据缺失")
        
        return comparison_results
    
    def _compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray, name: str) -> Dict[str, Any]:
        """比较两个数组的一致性"""
        result = {
            'name': name,
            'shape_match': arr1.shape == arr2.shape,
            'dtype_match': arr1.dtype == arr2.dtype
        }
        
        if not result['shape_match']:
            result.update({
                'perfect_match': False,
                'match_rate': 0.0,
                'matches': 0,
                'total': min(arr1.size, arr2.size),
                'max_diff': float('inf'),
                'mean_diff': float('inf'),
                'error': f"形状不匹配: {arr1.shape} vs {arr2.shape}"
            })
            return result
        
        # 计算差异
        if np.issubdtype(arr1.dtype, np.integer):
            # 整数类型：精确匹配
            matches = np.sum(arr1 == arr2)
            total = arr1.size
            perfect_match = matches == total
            max_diff = 0 if perfect_match else 1
            mean_diff = 0 if perfect_match else np.mean(arr1 != arr2)
        else:
            # 浮点类型：允许小误差
            diff = np.abs(arr1 - arr2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            tolerance = 1e-6
            perfect_match = max_diff < tolerance
            matches = np.sum(diff < tolerance)
            total = arr1.size
        
        result.update({
            'perfect_match': perfect_match,
            'match_rate': 100.0 * matches / total,
            'matches': matches,
            'total': total,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        })
        
        return result
    
    def save_3d_data(self, data_3d: Dict[str, np.ndarray], output_path: Path):
        """保存3D数据为MAT文件"""
        print(f"\n保存3D数据到: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            for key, value in data_3d.items():
                # 保存时为了MATLAB兼容性，可能需要调整格式
                if key == 'data' and value.ndim == 4:
                    # 4D数据可能需要调整轴顺序
                    save_value = np.moveaxis(value, -1, 0)  # (384,336,256,351) → (351,384,336,256)
                    print(f"  {key}: {value.shape} → {save_value.shape} [调整为MATLAB格式]")
                else:
                    save_value = value
                    print(f"  {key}: {value.shape}")
                
                f.create_dataset(key, data=save_value, compression='gzip', compression_opts=4)
    
    def run_full_validation(self, mat_path: Path, output_dir: Path = None, prob_idx: int = 1) -> bool:
        """执行完整的往返验证"""
        print(f"{'='*80}")
        print(f"往返转换验证：{mat_path.name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # 1. 加载原始数据
        original_data = self.load_mat_h5_correct(mat_path)
        
        # 2. 1D → 3D 转换
        data_3d = self.convert_1d_to_3d(original_data, prob_idx)
        
        # 3. 保存3D数据（可选）
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_3d_path = output_dir / f"{mat_path.stem}_3d_validated.mat"
            self.save_3d_data(data_3d, output_3d_path)
        
        # 4. 3D → 1D 转换
        recovered_data = self.convert_3d_to_1d(data_3d)
        
        # 5. 逐像素对比
        comparison_results = self.compare_data(original_data, recovered_data)
        
        # 6. 总结结果
        elapsed_time = time.time() - start_time
        all_perfect = all(result['perfect_match'] for result in comparison_results.values())
        
        print(f"\n{'='*80}")
        print(f"验证结果总结")
        print(f"{'='*80}")
        print(f"处理时间: {elapsed_time:.1f}秒")
        print(f"总体结果: {'✅ 完美匹配' if all_perfect else '❌ 存在差异'}")
        
        # 详细结果
        print(f"\n详细结果:")
        for key, result in comparison_results.items():
            status = "✅" if result['perfect_match'] else "❌"
            print(f"  {status} {key:15s}: {result['match_rate']:6.2f}%")
            if not result['perfect_match'] and 'error' not in result:
                print(f"      差异统计: max={result['max_diff']:.2e}, mean={result['mean_diff']:.2e}")
        
        # 保存验证结果
        self.results[mat_path.name] = {
            'all_perfect': all_perfect,
            'processing_time': elapsed_time,
            'details': comparison_results
        }
        
        if all_perfect:
            print(f"\n🎉 往返转换验证完全成功！数据转换方法正确无误。")
        else:
            print(f"\n⚠️ 存在数据不一致，需要进一步调查。")
        
        return all_perfect

def main():
    """主函数"""
    # 设置路径
    test_file = Path("/home/jannik/Documents/mri_mat_onehot/1D/ODP_01_qhlazec.mat")
    output_dir = Path("/home/jannik/Documents/mri_mat_onehot/3D_validated")
    
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    # 创建验证器
    validator = RoundTripValidator()
    
    # 执行验证
    success = validator.run_full_validation(
        mat_path=test_file,
        output_dir=output_dir,
        prob_idx=1
    )
    
    if success:
        print(f"\n✅ 验证成功！可以放心使用这个数据转换方法。")
    else:
        print(f"\n❌ 验证失败，需要进一步优化转换方法。")

if __name__ == "__main__":
    main()