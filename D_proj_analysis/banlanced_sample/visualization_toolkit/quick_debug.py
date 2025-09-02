#!/usr/bin/env python3
"""
快速调试argmax问题
"""

import numpy as np
import nibabel as nib
import json

def quick_debug():
    print("🔍 Quick Debug - Background Included Mode")
    
    # 加载数据
    softmax_path = "/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.nii.gz"
    softmax_nii = nib.load(softmax_path)
    softmax = softmax_nii.get_fdata()
    
    print(f"Softmax shape: {softmax.shape}")
    
    # 检查概率和
    prob_sum = np.sum(softmax, axis=-1)
    print(f"Prob sum range: [{prob_sum.min():.6f}, {prob_sum.max():.6f}]")
    
    # 计算argmax
    predictions = np.argmax(softmax, axis=-1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: {predictions.min()}-{predictions.max()}")
    
    # 检查每个类别的预测数量
    unique, counts = np.unique(predictions, return_counts=True)
    print("\n🎲 Prediction distribution:")
    for cls, count in zip(unique, counts):
        pct = count / np.prod(predictions.shape) * 100
        print(f"  Class {cls}: {count:,} voxels ({pct:.2f}%)")
    
    # 检查各个类别的最大概率
    print(f"\n📊 Max probability per class:")
    for i in range(min(10, softmax.shape[-1])):
        max_prob = softmax[..., i].max()
        mean_prob = softmax[..., i].mean()
        print(f"  Class {i}: max={max_prob:.6f}, mean={mean_prob:.6f}")
    
    # 检查是否有非背景预测
    non_bg_predictions = predictions[predictions > 0]
    print(f"\n🔍 Non-background predictions: {len(non_bg_predictions):,}")
    
    if len(non_bg_predictions) > 0:
        unique_non_bg, counts_non_bg = np.unique(non_bg_predictions, return_counts=True)
        print("Non-background classes found:")
        for cls, count in zip(unique_non_bg, counts_non_bg):
            print(f"  Class {cls}: {count:,} voxels")
    else:
        print("❌ NO NON-BACKGROUND PREDICTIONS FOUND!")
        
        # 进一步调试：检查第二高的概率
        print("\n🔍 Checking second-highest probabilities...")
        second_max = np.partition(softmax, -2, axis=-1)[..., -2]  # 第二大的概率
        print(f"Second max prob range: [{second_max.min():.6f}, {second_max.max():.6f}]")
        print(f"Second max prob mean: {second_max.mean():.6f}")

if __name__ == "__main__":
    quick_debug()