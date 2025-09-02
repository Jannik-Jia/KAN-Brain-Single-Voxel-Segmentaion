#!/usr/bin/env python3
"""
Quick analysis of the softmax issue - optimized for large files
"""

import numpy as np
import nibabel as nib
import json

def quick_analysis():
    print("🔍 Quick Analysis - Investigating F1 discrepancy")
    
    # File paths
    softmax_path = "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.nii.gz"
    info_path = "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.json"
    
    try:
        # Load info first
        print("📂 Loading info...")
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"✅ Info loaded:")
        print(f"  Shape: {info['softmax_shape']}")
        print(f"  Classes: {info['n_classes']}")
        print(f"  Background included: {info['include_background']}")
        print(f"  Min prob: {info['prediction_stats']['min_prob']:.2e}")
        print(f"  Max prob: {info['prediction_stats']['max_prob']:.6f}")
        print(f"  Mean prob: {info['prediction_stats']['mean_prob']:.6f}")
        
        # Load softmax with memory mapping
        print("📂 Loading softmax (memory mapped)...")
        softmax_nii = nib.load(softmax_path)
        softmax = softmax_nii.get_fdata()
        
        print(f"✅ Softmax loaded:")
        print(f"  Shape: {softmax.shape}")
        print(f"  Dtype: {softmax.dtype}")
        
        # Sample a small region to check predictions
        print("\n🎲 Sampling predictions...")
        # Take middle slice for analysis
        mid_slice = softmax.shape[2] // 2
        sample_slice = softmax[:, :, mid_slice, :]
        
        print(f"Sample slice shape: {sample_slice.shape}")
        
        # Check argmax on sample
        sample_predictions = np.argmax(sample_slice, axis=-1)
        unique_preds, counts = np.unique(sample_predictions, return_counts=True)
        
        print(f"\n📊 Predictions in middle slice:")
        for pred, count in zip(unique_preds, counts):
            percentage = count / np.prod(sample_predictions.shape) * 100
            print(f"  Class {pred}: {count:,} voxels ({percentage:.2f}%)")
        
        # Check if there are non-background predictions
        non_bg = sample_predictions[sample_predictions > 0]
        print(f"\n🔍 Non-background predictions in slice: {len(non_bg):,}")
        
        if len(non_bg) > 0:
            print("✅ Found non-background predictions!")
            unique_non_bg, counts_non_bg = np.unique(non_bg, return_counts=True)
            for pred, count in zip(unique_non_bg, counts_non_bg):
                print(f"    Class {pred}: {count:,} voxels")
        else:
            print("❌ NO non-background predictions in sample!")
            
            # Check probability values for non-background classes
            print("\n🔍 Checking class probabilities in sample...")
            for i in range(min(5, sample_slice.shape[-1])):
                class_probs = sample_slice[..., i]
                max_prob = class_probs.max()
                mean_prob = class_probs.mean()
                above_01 = (class_probs > 0.1).sum()
                print(f"  Class {i}: max={max_prob:.6f}, mean={mean_prob:.6f}, >0.1: {above_01:,}")
        
        # Check background dominance
        if sample_slice.shape[-1] > 0:
            bg_probs = sample_slice[..., 0]
            bg_high = (bg_probs > 0.9).sum()
            bg_very_high = (bg_probs > 0.99).sum()
            total_voxels = np.prod(bg_probs.shape)
            
            print(f"\n🏷️ Background dominance in sample:")
            print(f"  Background >0.9: {bg_high:,}/{total_voxels:,} ({bg_high/total_voxels*100:.2f}%)")
            print(f"  Background >0.99: {bg_very_high:,}/{total_voxels:,} ({bg_very_high/total_voxels*100:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_analysis()