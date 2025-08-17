#!/usr/bin/env python3
"""
测试数据加载后的内存布局与Mert函数的兼容性
关键问题：Fortran vs C order对revert_reshape的影响
"""

import numpy as np
import h5py
from pathlib import Path
import sys

def load_mat_original_way(path):
    """原始方式：直接加载，可能保持MATLAB的Fortran顺序"""
    data = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("#"):
                v = f[k][()]
                data[k] = v
    return data

def load_mat_transpose_way(path):
    """转置方式：如create_3d_final.py中那样处理"""
    data = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("#"):
                v = f[k][()]
                if k == 'multidim_data' and v.shape[0] == 351:
                    v = v.T  # 转置
                elif k == 'region_seg':
                    v = v.flatten()
                # region和big_seg保持原样
                data[k] = v
    return data

def mert_original_revert_reshape(array, region):
    """Mert的原始revert_reshape函数"""
    if array.ndim == 1:
        array = array[:, np.newaxis]
    
    num_features = array.shape[1]
    big_img = np.zeros((*region.shape, num_features), dtype=array.dtype)
    big_img = big_img.reshape(-1, array.shape[1], order='F')
    big_img[region.flatten(order='F')] = array  # 注意：这里用的是region.flatten(order='F')
    big_img = big_img.reshape((*region.shape, num_features), order='F')
    
    big_img = big_img.squeeze()
    return big_img

def create_3d_final_revert_reshape(array, region):
    """create_3d_final.py中的revert_reshape函数"""
    if array.ndim == 1:
        array = array[:, np.newaxis]
    
    num_features = array.shape[1]
    big_img = np.zeros((*region.shape, num_features), dtype=array.dtype)
    big_img = big_img.reshape(-1, array.shape[1], order='F')
    bool_mask = region.flatten(order='F').astype(bool)
    big_img[bool_mask] = array  # 注意：这里用的是布尔掩码
    big_img = big_img.reshape((*region.shape, num_features), order='F')
    
    if num_features == 1:
        big_img = big_img.squeeze()
    
    return big_img

def test_memory_layout_effects(filepath):
    """测试内存布局对函数行为的影响"""
    print(f"\n{'='*80}")
    print(f"测试文件: {Path(filepath).name}")
    print(f"{'='*80}")
    
    # 1. 加载数据（两种方式）
    print("\n1. 加载数据...")
    data_original = load_mat_original_way(filepath)
    data_transposed = load_mat_transpose_way(filepath)
    
    print("\n原始加载方式:")
    for key in ['region', 'big_seg', 'multidim_data', 'region_seg']:
        if key in data_original:
            arr = data_original[key]
            print(f"  {key:15s}: {str(arr.shape):20s} flags: C={arr.flags.c_contiguous}, F={arr.flags.f_contiguous}")
    
    print("\n转置加载方式:")
    for key in ['region', 'big_seg', 'multidim_data', 'region_seg']:
        if key in data_transposed:
            arr = data_transposed[key]
            print(f"  {key:15s}: {str(arr.shape):20s} flags: C={arr.flags.c_contiguous}, F={arr.flags.f_contiguous}")
    
    # 2. 测试region的flatten行为
    print(f"\n2. 测试region.flatten()的差异...")
    
    # 原始region
    region_orig = data_original['region'].astype(bool)
    region_trans = data_transposed['region'].astype(bool)
    
    print(f"region形状: 原始={region_orig.shape}, 转置后={region_trans.shape}")
    
    # flatten行为对比
    flatten_orig_F = region_orig.flatten(order='F')
    flatten_orig_C = region_orig.flatten(order='C')
    flatten_trans_F = region_trans.flatten(order='F')
    flatten_trans_C = region_trans.flatten(order='C')
    
    print(f"flatten结果对比:")
    print(f"  原始数据 F-order: {np.sum(flatten_orig_F)} 个True")
    print(f"  原始数据 C-order: {np.sum(flatten_orig_C)} 个True")
    print(f"  转置数据 F-order: {np.sum(flatten_trans_F)} 个True")
    print(f"  转置数据 C-order: {np.sum(flatten_trans_C)} 个True")
    
    # 检查flatten顺序是否一致
    flatten_order_match_F = np.array_equal(flatten_orig_F, flatten_trans_F)
    flatten_order_match_C = np.array_equal(flatten_orig_C, flatten_trans_C)
    
    print(f"  Fortran order 一致性: {'✅' if flatten_order_match_F else '❌'}")
    print(f"  C order 一致性: {'✅' if flatten_order_match_C else '❌'}")
    
    # 3. 测试用region_seg验证重构
    print(f"\n3. 使用region_seg验证重构一致性...")
    
    # 获取region_seg数据
    if 'region_seg' in data_original:
        region_seg_orig = data_original['region_seg'].flatten()
        region_seg_trans = data_transposed['region_seg']
        
        print(f"region_seg形状: 原始={region_seg_orig.shape}, 转置后={region_seg_trans.shape}")
        
        # 使用Mert的函数重构
        print("\n使用Mert原始函数重构:")
        reconstructed_mert_orig = mert_original_revert_reshape(region_seg_orig, region_orig)
        reconstructed_mert_trans = mert_original_revert_reshape(region_seg_trans, region_trans)
        
        print(f"  原始数据重构形状: {reconstructed_mert_orig.shape}")
        print(f"  转置数据重构形状: {reconstructed_mert_trans.shape}")
        
        # 使用create_3d_final的函数重构
        print("\n使用create_3d_final函数重构:")
        reconstructed_final_orig = create_3d_final_revert_reshape(region_seg_orig, region_orig)
        reconstructed_final_trans = create_3d_final_revert_reshape(region_seg_trans, region_trans)
        
        print(f"  原始数据重构形状: {reconstructed_final_orig.shape}")
        print(f"  转置数据重构形状: {reconstructed_final_trans.shape}")
        
        # 4. 验证重构结果的一致性
        print(f"\n4. 验证重构结果一致性...")
        
        # 与big_seg对比（作为ground truth）
        big_seg_orig = data_original['big_seg']
        big_seg_trans = data_transposed['big_seg']
        
        def verify_reconstruction(reconstructed, big_seg, region, method_name):
            """验证重构结果与big_seg的匹配度"""
            mask = region.astype(bool)
            matches = np.sum(reconstructed[mask] == big_seg[mask])
            total = int(mask.sum())
            match_rate = 100.0 * matches / total
            print(f"  {method_name:30s}: {match_rate:6.2f}% ({matches}/{total})")
            return match_rate
        
        print("\n原始数据加载:")
        rate1 = verify_reconstruction(reconstructed_mert_orig, big_seg_orig, region_orig, "Mert函数")
        rate2 = verify_reconstruction(reconstructed_final_orig, big_seg_orig, region_orig, "create_3d_final函数")
        
        print("\n转置数据加载:")
        rate3 = verify_reconstruction(reconstructed_mert_trans, big_seg_trans, region_trans, "Mert函数")
        rate4 = verify_reconstruction(reconstructed_final_trans, big_seg_trans, region_trans, "create_3d_final函数")
        
        # 5. 结论
        print(f"\n5. 结论:")
        print(f"{'='*50}")
        
        all_rates = [rate1, rate2, rate3, rate4]
        if all(rate > 99.9 for rate in all_rates):
            print("✅ 所有方法都得到了正确的结果")
        else:
            print("❌ 存在不一致，需要进一步调查")
            
        if abs(rate1 - rate3) < 0.1 and abs(rate2 - rate4) < 0.1:
            print("✅ 数据加载方式（是否转置）对结果没有影响")
        else:
            print("⚠️ 数据加载方式可能影响结果")
            
        if abs(rate1 - rate2) < 0.1 and abs(rate3 - rate4) < 0.1:
            print("✅ Mert函数和create_3d_final函数结果一致")
        else:
            print("⚠️ 两个函数实现存在差异")
            
        return {
            'flatten_consistency': flatten_order_match_F,
            'reconstruction_rates': all_rates,
            'data_loading_consistent': abs(rate1 - rate3) < 0.1,
            'function_implementations_consistent': abs(rate1 - rate2) < 0.1
        }

def main():
    """主函数"""
    default_path = "/home/jannik/Documents/mri_mat_onehot/1D/ODP_01_qhlazec.mat"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = default_path
        print(f"使用默认测试文件: {test_file}")
    
    test_path = Path(test_file)
    
    if not test_path.exists():
        print(f"❌ 文件不存在: {test_path}")
        return
    
    # 执行测试
    results = test_memory_layout_effects(test_path)
    
    print(f"\n{'='*80}")
    print("最终建议:")
    print(f"{'='*80}")
    
    if results['flatten_consistency'] and results['data_loading_consistent']:
        print("✅ 可以安全使用create_3d_final.py的数据加载方式")
    else:
        print("⚠️ 建议保持与Mert函数完全一致的数据处理方式")
    
    if not results['function_implementations_consistent']:
        print("🔧 建议修改create_3d_final.py中的revert_reshape函数以完全匹配Mert的实现")

if __name__ == "__main__":
    main()