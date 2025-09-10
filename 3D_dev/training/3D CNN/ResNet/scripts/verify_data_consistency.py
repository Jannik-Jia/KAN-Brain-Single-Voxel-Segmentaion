#!/usr/bin/env python3
"""
Verify data loading consistency between ResNet and 3D CNN baseline

This script compares the data loading behavior to ensure they are identical.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import glob

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'models'))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import ResNet dataset
from dataset import MRIBrain2DPatchDataset

# Import 3D CNN dataset
from train_baseline_3x3_7x7 import Brain3DPatchDataset


def test_data_consistency(data_dir: str, patch_size: int = 7):
    """Test data loading consistency"""
    print("=== Data Loading Consistency Test ===")
    
    # Find MAT files using 3D CNN logic
    data_path = Path(data_dir)
    mat_files = sorted(data_path.glob('subject*_3d_validated.mat'))
    
    if len(mat_files) == 0:
        mat_files = sorted(data_path.glob('*.mat'))
    
    if len(mat_files) == 0:
        print(f"No MAT files found in {data_dir}")
        return False
    
    print(f"Found {len(mat_files)} MAT files")
    print(f"Testing with patch size: {patch_size}")
    
    # Test with first file only
    test_files = mat_files[:1]
    
    print("\n=== Creating 3D CNN Dataset ===")
    dataset_3d = Brain3DPatchDataset(
        test_files,
        patch_size=patch_size,
        samples_per_subject=100,  # Small number for testing
        is_train=True,
        cache_data=True
    )
    
    print("\n=== Creating ResNet Dataset ===")
    dataset_resnet = MRIBrain2DPatchDataset(
        test_files,
        patch_size=patch_size,
        samples_per_subject=100,  # Same small number
        is_train=True,
        cache_data=True,
        balance_classes=False,
        augmentation=False
    )
    
    print(f"\n=== Dataset Size Comparison ===")
    print(f"3D CNN dataset size: {len(dataset_3d)}")
    print(f"ResNet dataset size: {len(dataset_resnet)}")
    
    if len(dataset_3d) != len(dataset_resnet):
        print("❌ Dataset sizes don't match!")
        return False
    
    print("✅ Dataset sizes match")
    
    print(f"\n=== Sample Comparison ===")
    # Compare first few samples
    for i in range(min(3, len(dataset_3d))):
        print(f"\nSample {i}:")
        
        # Get samples
        patch_3d, label_3d = dataset_3d[i]
        patch_resnet, label_resnet = dataset_resnet[i]
        
        print(f"  3D CNN - Patch shape: {patch_3d.shape}, Label: {label_3d.item()}")
        print(f"  ResNet - Patch shape: {patch_resnet.shape}, Label: {label_resnet.item()}")
        
        # Check shapes
        if patch_3d.shape != patch_resnet.shape:
            print("  ❌ Patch shapes don't match!")
            return False
        
        # Check labels
        if label_3d.item() != label_resnet.item():
            print("  ❌ Labels don't match!")
            return False
        
        # Check patch values (should be very similar, allowing for floating point precision)
        patch_diff = torch.abs(patch_3d - patch_resnet).max().item()
        print(f"  Max patch difference: {patch_diff:.6f}")
        
        if patch_diff > 1e-5:
            print("  ❌ Patch values differ significantly!")
            return False
        
        print("  ✅ Sample matches")
    
    print(f"\n=== Data Loading Logic Test ===")
    # Test the load_subject_data function directly
    test_file = test_files[0]
    
    print(f"Testing file: {test_file.name}")
    
    # Load with 3D CNN method
    data_3d = dataset_3d.load_subject_data(test_file)
    
    # Load with ResNet method  
    data_resnet = dataset_resnet.load_subject_data(test_file)
    
    # Compare shapes
    print(f"3D CNN loaded shapes:")
    print(f"  data: {data_3d['data'].shape}")
    print(f"  labels: {data_3d['labels'].shape}")
    print(f"  mask: {data_3d['mask'].shape}")
    
    print(f"ResNet loaded shapes:")
    print(f"  data: {data_resnet['data'].shape}")
    print(f"  labels: {data_resnet['labels'].shape}")
    print(f"  mask: {data_resnet['mask'].shape}")
    
    # Check if shapes match
    for key in ['data', 'labels', 'mask']:
        if data_3d[key].shape != data_resnet[key].shape:
            print(f"❌ {key} shapes don't match!")
            return False
        
        # Check if values match
        diff = np.abs(data_3d[key].astype(float) - data_resnet[key].astype(float)).max()
        print(f"  {key} max difference: {diff:.6f}")
        
        if diff > 1e-5:
            print(f"❌ {key} values differ significantly!")
            return False
    
    print("✅ Data loading methods produce identical results (including z-score normalization)")
    
    print(f"\n=== Patch Extraction Test ===")
    # Test patch extraction directly
    x, y, z = 200, 150, 100  # Fixed coordinates for testing
    
    patch_3d_direct = dataset_3d.extract_patch_2d(data_3d['data'], x, y, z)
    patch_resnet_direct = dataset_resnet.extract_patch_2d(data_resnet['data'], x, y, z)
    
    print(f"Direct patch extraction at ({x}, {y}, {z}):")
    print(f"  3D CNN patch shape: {patch_3d_direct.shape}")
    print(f"  ResNet patch shape: {patch_resnet_direct.shape}")
    
    if patch_3d_direct.shape != patch_resnet_direct.shape:
        print("❌ Direct patch shapes don't match!")
        return False
    
    patch_direct_diff = np.abs(patch_3d_direct - patch_resnet_direct).max()
    print(f"  Max patch difference: {patch_direct_diff:.6f}")
    
    if patch_direct_diff > 1e-5:
        print("❌ Direct patches differ significantly!")
        return False
    
    print("✅ Patch extraction methods produce identical results")
    
    print(f"\n🎉 All tests passed! Data loading (with z-score normalization) is consistent between 3D CNN and ResNet.")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify data loading consistency')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing MAT files')
    parser.add_argument('--patch_size', type=int, default=7,
                       help='Patch size to test')
    
    args = parser.parse_args()
    
    try:
        success = test_data_consistency(args.data_dir, args.patch_size)
        if success:
            print("\n✅ Verification completed successfully!")
            exit(0)
        else:
            print("\n❌ Verification failed!")
            exit(1)
    except Exception as e:
        print(f"\n💥 Error during verification: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()