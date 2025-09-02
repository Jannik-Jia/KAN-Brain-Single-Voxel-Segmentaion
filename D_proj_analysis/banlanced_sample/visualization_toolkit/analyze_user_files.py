#!/usr/bin/env python3
"""
直接分析用户提供的softmax文件
"""

import numpy as np
import nibabel as nib
import json

def analyze_user_softmax():
    """分析用户的softmax文件"""
    print("🔍 Analyzing user's softmax files...")
    
    # 文件路径（从用户消息中提取）
    softmax_path = "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.nii.gz"
    info_path = "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.json"
    
    try:
        # 加载数据
        print("📂 Loading softmax data...")
        softmax_nii = nib.load(softmax_path)
        softmax = softmax_nii.get_fdata()
        
        print("📂 Loading info data...")
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"✅ Successfully loaded data")
        print(f"  Softmax shape: {softmax.shape}")
        print(f"  Softmax dtype: {softmax.dtype}")
        print(f"  Info classes: {info['n_classes']}")
        print(f"  Include background: {info['include_background']}")
        
        # 基本统计
        print(f"\n📊 Basic Statistics:")
        print(f"  Min: {softmax.min():.2e}")
        print(f"  Max: {softmax.max():.6f}")
        print(f"  Mean: {softmax.mean():.6f}")
        print(f"  Std: {softmax.std():.6f}")
        
        # 检查NaN和Inf
        nan_count = np.isnan(softmax).sum()
        inf_count = np.isinf(softmax).sum()
        print(f"\n⚠️ Anomaly Check:")
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        
        # 检查概率和
        prob_sums = np.sum(softmax, axis=-1)
        print(f"\n🎯 Probability Sum Check:")
        print(f"  Min sum: {prob_sums.min():.6f}")
        print(f"  Max sum: {prob_sums.max():.6f}")
        print(f"  Mean sum: {prob_sums.mean():.6f}")
        print(f"  Std sum: {prob_sums.std():.6f}")
        
        # 检查是否是有效的概率分布
        valid_prob = np.all(softmax >= 0) and np.all(softmax <= 1)
        print(f"  Valid probability range [0,1]: {valid_prob}")
        
        # 计算argmax
        print(f"\n🎲 Prediction Analysis:")
        predictions = np.argmax(softmax, axis=-1)
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions range: [{predictions.min()}, {predictions.max()}]")
        
        # 预测分布
        unique_preds, counts = np.unique(predictions, return_counts=True)
        print(f"  Unique predictions: {len(unique_preds)}")
        print(f"\n  Top 10 predictions:")
        
        # 按数量排序
        sorted_indices = np.argsort(counts)[::-1]
        for i in range(min(10, len(unique_preds))):
            idx = sorted_indices[i]
            pred = unique_preds[idx]
            count = counts[idx]
            percentage = count / np.prod(predictions.shape) * 100
            print(f"    Class {pred}: {count:,} voxels ({percentage:.2f}%)")
        
        # 检查每个类别的概率统计
        print(f"\n📈 Per-Class Probability Analysis:")
        for i in range(min(10, softmax.shape[-1])):
            class_probs = softmax[..., i]
            max_prob = class_probs.max()
            mean_prob = class_probs.mean()
            above_threshold = (class_probs > 0.5).sum()
            print(f"  Class {i}: max={max_prob:.6f}, mean={mean_prob:.6f}, >0.5: {above_threshold:,}")
        
        # 关键：检查是否真的没有前景预测
        non_bg_predictions = predictions[predictions > 0]
        print(f"\n🔍 Non-background Analysis:")
        print(f"  Non-background predictions: {len(non_bg_predictions):,}")
        
        if len(non_bg_predictions) > 0:
            print("✅ Found non-background predictions!")
            unique_non_bg, counts_non_bg = np.unique(non_bg_predictions, return_counts=True)
            for pred, count in zip(unique_non_bg, counts_non_bg):
                percentage = count / len(non_bg_predictions) * 100
                print(f"    Class {pred}: {count:,} ({percentage:.2f}%)")
        else:
            print("❌ NO non-background predictions found!")
            
            # 进一步分析：检查第二高概率
            print("\n🔍 Second-highest probability analysis:")
            second_highest_idx = np.argpartition(softmax, -2, axis=-1)[..., -2]
            second_highest_prob = np.take_along_axis(softmax, second_highest_idx[..., None], axis=-1).squeeze()
            
            print(f"  Second highest prob range: [{second_highest_prob.min():.6f}, {second_highest_prob.max():.6f}]")
            print(f"  Second highest prob mean: {second_highest_prob.mean():.6f}")
            
            # 检查背景类的主导程度
            bg_prob = softmax[..., 0]
            bg_dominance = (bg_prob > 0.9).sum() / np.prod(bg_prob.shape) * 100
            print(f"  Background dominance (>0.9): {bg_dominance:.2f}%")
        
        return softmax, info
        
    except Exception as e:
        print(f"❌ Error analyzing files: {e}")
        return None, None

if __name__ == "__main__":
    analyze_user_softmax()