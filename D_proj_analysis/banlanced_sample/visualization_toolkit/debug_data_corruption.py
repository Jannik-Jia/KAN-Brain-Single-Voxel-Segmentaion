#!/usr/bin/env python3
"""
Debug the data corruption issue
"""

import numpy as np
import nibabel as nib
import json

def debug_data_corruption():
    print("🔍 Debugging Data Corruption Issue")
    
    # File paths
    softmax_path = "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_3d_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.nii.gz"
    info_path = "/Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/results/test_softmax_info_FOR_016_20250204_reproducibility_bg_incl_20250829_151320.json"
    
    try:
        # Load info
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print("📊 Info file stats:")
        stats = info['prediction_stats']
        print(f"  Min: {stats['min_prob']:.2e}")
        print(f"  Max: {stats['max_prob']:.6f}")
        print(f"  Mean: {stats['mean_prob']:.6f}")
        
        # Load NIfTI header first
        print("\n📂 Checking NIfTI file...")
        nii = nib.load(softmax_path)
        print(f"  Header datatype: {nii.header.get_data_dtype()}")
        print(f"  Header shape: {nii.header.get_data_shape()}")
        print(f"  Slope: {nii.header.get_slope_inter()[0]}")
        print(f"  Intercept: {nii.header.get_slope_inter()[1]}")
        
        # Load raw data without scaling
        print("\n📊 Loading raw data...")
        raw_data = nii.get_fdata(dtype=np.float64)
        print(f"  Raw data shape: {raw_data.shape}")
        print(f"  Raw data dtype: {raw_data.dtype}")
        print(f"  Raw min: {raw_data.min():.2e}")
        print(f"  Raw max: {raw_data.max():.6f}")
        print(f"  Raw mean: {raw_data.mean():.6f}")
        
        # Check if data is all zeros
        non_zero_count = np.count_nonzero(raw_data)
        total_count = np.prod(raw_data.shape)
        print(f"  Non-zero values: {non_zero_count:,}/{total_count:,} ({non_zero_count/total_count*100:.6f}%)")
        
        # Sample some values from different locations
        print(f"\n🎲 Sample values from different locations:")
        h, w, d, c = raw_data.shape
        
        # Sample from corners and center
        locations = [
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, -1),
            (h//2, w//2, d//2, 0), (h//2, w//2, d//2, 1), (h//2, w//2, d//2, -1),
            (-1, -1, -1, 0), (-1, -1, -1, 1), (-1, -1, -1, -1)
        ]
        
        for loc in locations:
            val = raw_data[loc]
            print(f"  {loc}: {val:.6f}")
        
        # Check if the problem is in the middle slice specifically
        print(f"\n🔍 Checking different slices...")
        for slice_idx in [0, d//4, d//2, 3*d//4, d-1]:
            slice_data = raw_data[:, :, slice_idx, :]
            slice_min = slice_data.min()
            slice_max = slice_data.max()
            slice_nonzero = np.count_nonzero(slice_data)
            slice_total = np.prod(slice_data.shape)
            print(f"  Slice {slice_idx}: min={slice_min:.6f}, max={slice_max:.6f}, non-zero={slice_nonzero:,}/{slice_total:,}")
        
        # Final check: is this a precision/scaling issue?
        print(f"\n⚠️ Checking for precision issues...")
        unique_values = np.unique(raw_data.flatten())
        print(f"  Number of unique values: {len(unique_values)}")
        if len(unique_values) <= 10:
            print(f"  Unique values: {unique_values}")
        else:
            print(f"  First 10 unique values: {unique_values[:10]}")
            print(f"  Last 10 unique values: {unique_values[-10:]}")
            
        return raw_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_data_corruption()