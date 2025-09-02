#!/usr/bin/env python3
"""
调试softmax数据的问题
"""

import numpy as np
import nibabel as nib
import json

def debug_softmax_data(softmax_path, info_path):
    """调试softmax数据"""
    print("🔍 Loading data for debugging...")
    
    # 加载数据
    softmax_nii = nib.load(softmax_path)
    softmax = softmax_nii.get_fdata()
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print(f"📊 Softmax shape: {softmax.shape}")
    print(f"📊 Data type: {softmax.dtype}")
    print(f"📊 Include background: {info['include_background']}")
    
    # 基本统计
    print(f"\n📈 Basic Statistics:")
    print(f"  Min: {softmax.min():.2e}")
    print(f"  Max: {softmax.max():.6f}")
    print(f"  Mean: {softmax.mean():.6f}")
    print(f"  Std: {softmax.std():.6f}")
    
    # 检查概率和
    prob_sums = np.sum(softmax, axis=-1)
    print(f"\n🎯 Probability Sums:")
    print(f"  Min sum: {prob_sums.min():.6f}")
    print(f"  Max sum: {prob_sums.max():.6f}")
    print(f"  Mean sum: {prob_sums.mean():.6f}")
    
    # 检查argmax分布
    predictions = np.argmax(softmax, axis=-1)
    unique_preds, counts = np.unique(predictions, return_counts=True)
    print(f"\n🎲 Prediction Distribution:")
    for pred, count in zip(unique_preds, counts):
        percentage = count / np.prod(predictions.shape) * 100
        print(f"  Class {pred}: {count:,} voxels ({percentage:.2f}%)")
    
    # 检查每个类的最大概率
    print(f"\n📊 Per-Class Maximum Probabilities:")
    for i in range(min(10, softmax.shape[-1])):  # 只显示前10个类
        max_prob = softmax[..., i].max()
        print(f"  Class {i}: max_prob = {max_prob:.6f}")
    
    # 检查是否有异常值
    nan_count = np.isnan(softmax).sum()
    inf_count = np.isinf(softmax).sum()
    print(f"\n⚠️ Anomaly Check:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    return softmax

if __name__ == "__main__":
    # 分析两种模式
    print("="*60)
    print("🌟 BACKGROUND INCLUDED MODE")
    print("="*60)
    bg_incl_softmax = "/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.nii.gz"
    bg_incl_info = "/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.json"
    
    softmax_incl = debug_softmax_data(bg_incl_softmax, bg_incl_info)
    
    print("\n" + "="*60)
    print("🎯 BACKGROUND EXCLUDED MODE")  
    print("="*60)
    bg_excl_softmax = "/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.nii.gz"
    bg_excl_info = "/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_excl_20250827_154806.json"
    
    softmax_excl = debug_softmax_data(bg_excl_softmax, bg_excl_info)
    
    # 比较分析
    print("\n" + "="*60)
    print("🔄 COMPARISON")
    print("="*60)
    
    # 检查是否是相同数据
    if softmax_incl.shape == softmax_excl.shape:
        diff = np.abs(softmax_incl - softmax_excl)
        print(f"📊 Data Difference:")
        print(f"  Max difference: {diff.max():.2e}")
        print(f"  Mean difference: {diff.mean():.2e}")
        
        if diff.max() < 1e-10:
            print("⚠️ WARNING: Softmax data appears identical!")
    
    print("\n🎉 Debug analysis completed!")