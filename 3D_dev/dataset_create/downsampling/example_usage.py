"""
Example Usage Scripts for MRI Downsampling Pipeline
===================================================

This file contains practical examples of how to use the pipeline
for different scenarios.

Author: Generated for KAN-Brain project
Date: 2025-01-11
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import h5py

from mri_downsampling_pipeline import MRIDownsamplingPipeline


# ==============================================================================
# Example 1: Basic Usage - Single Subject
# ==============================================================================

def example_1_basic_usage():
    """
    Basic usage: Load MAT file, downsample, save results
    """
    print("=" * 80)
    print("Example 1: Basic Single Subject Downsampling")
    print("=" * 80)

    # Paths
    input_file = Path("/path/to/your/subject1_3d_validated.mat")
    output_dir = Path("./downsampling_output")

    if not input_file.exists():
        print(f"⚠️  Input file not found: {input_file}")
        print("Please update the path in the script.")
        return

    # Load data from MAT file
    print("\n1. Loading data...")
    mat_data = sio.loadmat(input_file)

    data = mat_data['data']  # (384, 336, 256, 351) in (Z, X, Y, C) order
    region_mask = mat_data['region_mask']  # (384, 336, 256)
    region_labels = mat_data['region_labels']  # (384, 336, 256)

    print(f"   Data shape: {data.shape}")
    print(f"   Mask shape: {region_mask.shape}")
    print(f"   Labels shape: {region_labels.shape}")

    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = MRIDownsamplingPipeline(
        output_dir=output_dir,
        log_level='INFO',  # Use 'DEBUG' for more details
        random_seed=42
    )

    # Run pipeline
    print("\n3. Running downsampling pipeline...")
    results = pipeline.run(
        data=data,
        region_mask=region_mask,
        region_labels=region_labels,
        align_to_128x104x18=False  # Set True if you need exact matrix size
    )

    # Access results
    data_lr = results['data_lr']
    proba_labels = results['proba_labels']
    mask_lr = results['region_mask_lr']
    metadata = results['metadata']
    qa_metrics = results['qa_metrics']

    print(f"\n4. Results:")
    print(f"   Downsampled data shape: {data_lr.shape}")
    print(f"   Probability labels shape: {proba_labels.shape}")
    print(f"   Downsampled mask shape: {mask_lr.shape}")

    # Save results
    print("\n5. Saving results...")
    output_file = output_dir / 'subject1_downsampled.npz'
    np.savez_compressed(
        output_file,
        data_lr=data_lr,
        proba_labels=proba_labels,
        region_mask_lr=mask_lr
    )
    print(f"   ✓ Saved to: {output_file}")

    print("\n✓ Example 1 completed successfully!")


# ==============================================================================
# Example 2: Batch Processing Multiple Subjects
# ==============================================================================

def example_2_batch_processing():
    """
    Batch process multiple subjects with progress tracking
    """
    print("=" * 80)
    print("Example 2: Batch Processing Multiple Subjects")
    print("=" * 80)

    # Paths
    data_dir = Path("/path/to/3d_validated_data")
    output_dir = Path("./batch_downsampling_output")

    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {data_dir}")
        print("Please update the path in the script.")
        return

    # Find all subject files
    subject_files = sorted(data_dir.glob("subject*_3d_validated.mat"))
    print(f"\nFound {len(subject_files)} subjects to process")

    # Initialize pipeline once (reuse for all subjects)
    pipeline = MRIDownsamplingPipeline(
        output_dir=output_dir,
        log_level='INFO',
        random_seed=42
    )

    # Process each subject
    for i, subject_file in enumerate(subject_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing subject {i}/{len(subject_files)}: {subject_file.name}")
        print(f"{'=' * 80}")

        try:
            # Load data
            mat_data = sio.loadmat(subject_file)
            data = mat_data['data']
            region_mask = mat_data['region_mask']
            region_labels = mat_data['region_labels']

            # Run pipeline
            results = pipeline.run(
                data=data,
                region_mask=region_mask,
                region_labels=region_labels,
                align_to_128x104x18=False
            )

            # Save results
            output_file = output_dir / f"{subject_file.stem}_downsampled.npz"
            np.savez_compressed(
                output_file,
                data_lr=results['data_lr'],
                proba_labels=results['proba_labels'],
                region_mask_lr=results['region_mask_lr']
            )

            print(f"✓ Subject {i} completed: {output_file}")

        except Exception as e:
            print(f"✗ Error processing {subject_file.name}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("✓ Batch processing completed!")
    print(f"{'=' * 80}")


# ==============================================================================
# Example 3: Load and Inspect Results
# ==============================================================================

def example_3_inspect_results():
    """
    Load and inspect downsampled results
    """
    print("=" * 80)
    print("Example 3: Inspect Downsampled Results")
    print("=" * 80)

    # Load downsampled data
    result_file = Path("./downsampling_output/subject1_downsampled.npz")

    if not result_file.exists():
        print(f"⚠️  Result file not found: {result_file}")
        print("Please run Example 1 first.")
        return

    print(f"\nLoading: {result_file}")
    data = np.load(result_file)

    data_lr = data['data_lr']
    proba_labels = data['proba_labels']
    mask_lr = data['region_mask_lr']

    print(f"\n1. Array shapes:")
    print(f"   data_lr: {data_lr.shape}")
    print(f"   proba_labels: {proba_labels.shape}")
    print(f"   mask_lr: {mask_lr.shape}")

    print(f"\n2. Data statistics (brain voxels only):")
    brain_voxels = mask_lr > 0

    # Sample channel statistics
    sample_channels = [0, 100, 200, 300]  # QTI, DWI, CEST, QSM
    for ch in sample_channels:
        ch_data = data_lr[brain_voxels, ch]
        print(f"   Channel {ch:3d}: mean={ch_data.mean():8.4f}, "
              f"std={ch_data.std():8.4f}, "
              f"range=[{ch_data.min():8.4f}, {ch_data.max():8.4f}]")

    print(f"\n3. Probability label statistics:")
    prob_sum = proba_labels.sum(axis=-1)
    print(f"   Sum per voxel: mean={prob_sum.mean():.6f}, "
          f"std={prob_sum.std():.6f}")
    print(f"   Min probability: {proba_labels.min():.6f}")
    print(f"   Max probability: {proba_labels.max():.6f}")

    # Load metadata
    metadata_file = Path("./downsampling_output/pipeline_metadata.json")
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"\n4. Processing metadata:")
        print(f"   Input spacing: {metadata['config']['input_spacing_mm']} mm")
        print(f"   Output spacing: {metadata['config']['target_spacing_mm']} mm")
        print(f"   Slab thickness: {metadata['slab_localization']['slab_thickness_mm']:.2f} mm")
        print(f"   Coverage fallback: {metadata['slab_localization']['coverage_fallback']}")

    print("\n✓ Inspection completed!")


# ==============================================================================
# Example 4: Custom Configuration
# ==============================================================================

def example_4_custom_config():
    """
    Use custom configuration for specific needs
    """
    print("=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80)

    # Load data (same as Example 1)
    input_file = Path("/path/to/your/subject1_3d_validated.mat")

    if not input_file.exists():
        print(f"⚠️  Input file not found: {input_file}")
        return

    mat_data = sio.loadmat(input_file)
    data = mat_data['data']
    region_mask = mat_data['region_mask']
    region_labels = mat_data['region_labels']

    # Custom output directory
    output_dir = Path("./custom_output")

    # Initialize with custom settings
    pipeline = MRIDownsamplingPipeline(
        output_dir=output_dir,
        log_level='DEBUG',  # More verbose logging
        random_seed=123  # Different random seed
    )

    # Run with custom options
    results = pipeline.run(
        data=data,
        region_mask=region_mask,
        region_labels=region_labels,
        align_to_128x104x18=True  # Force exact matrix size
    )

    print(f"\n✓ Custom configuration completed!")
    print(f"   Output shape: {results['data_lr'].shape}")
    print(f"   Check logs in: {output_dir / 'logs'}")


# ==============================================================================
# Example 5: Load from HDF5 MAT Files
# ==============================================================================

def example_5_hdf5_loading():
    """
    Load from HDF5 format MAT files (v7.3+)
    """
    print("=" * 80)
    print("Example 5: Loading from HDF5 MAT Files")
    print("=" * 80)

    input_file = Path("/path/to/your/subject1_3d_validated.mat")

    if not input_file.exists():
        print(f"⚠️  Input file not found: {input_file}")
        return

    print(f"\nLoading HDF5 MAT file: {input_file}")

    # Load using h5py (for v7.3 MAT files)
    with h5py.File(input_file, 'r') as f:
        print("Available keys:", list(f.keys()))

        # Load arrays (note: may need transpose for MATLAB compatibility)
        data = f['data'][()]  # Shape depends on MATLAB save format
        region_mask = f['region_mask'][()]
        region_labels = f['region_labels'][()]

        print(f"Loaded shapes:")
        print(f"  data: {data.shape}")
        print(f"  region_mask: {region_mask.shape}")
        print(f"  region_labels: {region_labels.shape}")

        # Check if transpose needed (MATLAB uses Fortran order)
        # If data shape is (351, 256, 336, 384), need to transpose
        if data.shape[0] == 351:
            print("\nTransposing for (Z, X, Y, C) order...")
            data = np.transpose(data, (3, 2, 1, 0))
            region_mask = np.transpose(region_mask, (2, 1, 0))
            region_labels = np.transpose(region_labels, (2, 1, 0))

            print(f"After transpose:")
            print(f"  data: {data.shape}")
            print(f"  region_mask: {region_mask.shape}")

    # Now use with pipeline
    output_dir = Path("./hdf5_output")
    pipeline = MRIDownsamplingPipeline(output_dir=output_dir, log_level='INFO')

    results = pipeline.run(data, region_mask, region_labels)

    print("\n✓ HDF5 loading and processing completed!")


# ==============================================================================
# Example 6: Extract Specific Channel Families
# ==============================================================================

def example_6_extract_channels():
    """
    Extract and work with specific channel families
    """
    print("=" * 80)
    print("Example 6: Extract Specific Channel Families")
    print("=" * 80)

    # Load results
    result_file = Path("./downsampling_output/subject1_downsampled.npz")

    if not result_file.exists():
        print(f"⚠️  Result file not found: {result_file}")
        return

    data = np.load(result_file)
    data_lr = data['data_lr']

    print(f"\nLoaded data shape: {data_lr.shape}")

    # Import channel config
    from mri_downsampling_pipeline import ChannelConfig
    config = ChannelConfig()

    # Extract specific families
    print("\nExtracting channel families:")

    # QTI parameters (0-14, 0-based)
    qti_data = data_lr[..., config.FAMILIES['QTI_params'].indices]
    print(f"  QTI parameters: {qti_data.shape}")

    # DWI data (15-224, 0-based)
    dwi_data = data_lr[..., config.FAMILIES['DWI'].indices]
    print(f"  DWI data: {dwi_data.shape}")

    # Z-spectrum low B1 (230-283, 0-based)
    z_low = data_lr[..., config.Z_SPECTRUM_LOW_B1]
    print(f"  Z-spectrum (low B1): {z_low.shape}")

    # Z-spectrum high B1 (286-339, 0-based)
    z_high = data_lr[..., config.Z_SPECTRUM_HIGH_B1]
    print(f"  Z-spectrum (high B1): {z_high.shape}")

    # MPRAGE (341, 0-based)
    mprage = data_lr[..., config.FAMILIES['MPRAGE'].indices[0]]
    print(f"  MPRAGE: {mprage.shape}")

    # QSM (350, 0-based)
    qsm = data_lr[..., config.FAMILIES['QSM'].indices[0]]
    print(f"  QSM: {qsm.shape}")

    # Save individual modalities if needed
    output_dir = Path("./extracted_modalities")
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / 'qti_params.npy', qti_data)
    np.save(output_dir / 'dwi_data.npy', dwi_data)
    np.save(output_dir / 'z_spectrum_low_b1.npy', z_low)
    np.save(output_dir / 'mprage.npy', mprage)
    np.save(output_dir / 'qsm.npy', qsm)

    print(f"\n✓ Channel extraction completed!")
    print(f"   Saved to: {output_dir}")


# ==============================================================================
# Example 7: Save in Original Axis Order with 1D Data Generation
# ==============================================================================

def example_7_save_original_axis_order():
    """
    Demonstrate save_axis_order='orig' for compatibility with 1D<->3D mappers

    This example shows how to:
    1. Save 3D data in original (Z,X,Y,C) axis order
    2. Generate 1D format data (multidim_data, seg_one_hot, region_seg)
    3. Ensure 1D row ordering matches 3D C-order indexing
    """
    print("=" * 80)
    print("Example 7: Save in Original Axis Order with 1D Data")
    print("=" * 80)

    # Load data
    input_file = Path("/path/to/your/subject1_3d_validated.mat")
    output_dir = Path("./orig_axis_output")

    if not input_file.exists():
        print(f"⚠️  Input file not found: {input_file}")
        print("Please update the path in the script.")
        return

    print("\n1. Loading data...")
    mat_data = sio.loadmat(input_file)

    data = mat_data['data']  # (384, 336, 256, 351) in (Z, X, Y, C) order
    region_mask = mat_data['region_mask']  # (384, 336, 256)
    region_labels = mat_data['region_labels']  # (384, 336, 256)

    print(f"   Input data shape: {data.shape} (Z,X,Y,C)")
    print(f"   Input mask shape: {region_mask.shape} (Z,X,Y)")

    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = MRIDownsamplingPipeline(
        output_dir=output_dir,
        log_level='INFO',
        random_seed=42
    )

    # Run pipeline with save_axis_order='orig'
    print("\n3. Running downsampling with save_axis_order='orig'...")
    results = pipeline.run(
        data=data,
        region_mask=region_mask,
        region_labels=region_labels,
        align_to_128x104x18=False,
        save_axis_order='orig'  # KEY: Save in original axis order
    )

    # Access 3D results (in original axis order)
    data_lr = results['data_lr']  # (Z', X', Y', 351) - original order
    proba_labels = results['proba_labels']  # (Z', X', Y', 102)
    mask_lr = results['region_mask_lr']  # (Z', X', Y')

    # Access 1D results (with proper row reordering)
    multidim_data = results['multidim_data']  # (n_voxels, 351)
    seg_one_hot = results['seg_one_hot']  # (102, n_voxels)
    region_seg = results['region_seg']  # (n_voxels,)
    n_voxels = results['n_voxels']

    # Access metadata
    metadata = results['metadata']
    mapping_info = metadata['mapping']

    print(f"\n4. Results in ORIGINAL axis order:")
    print(f"   3D data:")
    print(f"      data_lr: {data_lr.shape} (Z,X,Y,C)")
    print(f"      proba_labels: {proba_labels.shape} (Z,X,Y,K)")
    print(f"      region_mask_lr: {mask_lr.shape} (Z,X,Y)")
    print(f"   1D data:")
    print(f"      multidim_data: {multidim_data.shape} (n_voxels, features)")
    print(f"      seg_one_hot: {seg_one_hot.shape} (n_classes, n_voxels)")
    print(f"      region_seg: {region_seg.shape} (n_voxels,)")
    print(f"      n_voxels: {n_voxels}")

    print(f"\n5. Mapping metadata:")
    print(f"   Axes: {mapping_info['axes']}")
    print(f"   Permutation: {mapping_info['permute']}")
    print(f"   Inverse permutation: {mapping_info['inv_perm']}")
    print(f"   Save axis order: {mapping_info['save_axis_order']}")
    print(f"   1D reorder applied: {mapping_info['one_d_reorder_applied']}")

    # Validate 1D-3D correspondence
    print(f"\n6. Validating 1D-3D correspondence...")
    n_check = min(10, n_voxels)
    check_indices = np.random.choice(n_voxels, size=n_check, replace=False)

    max_diff = 0.0
    for i in check_indices:
        # Get 1D feature vector
        feat_1d = multidim_data[i, :]

        # Find corresponding 3D position (using C-order indexing)
        saved_idx = np.where(mask_lr.ravel(order="C"))[0][i]
        z, x, y = np.unravel_index(saved_idx, mask_lr.shape, order="C")

        # Get 3D feature vector at same position
        feat_3d = data_lr[z, x, y, :]

        diff = np.abs(feat_1d - feat_3d).max()
        max_diff = max(max_diff, diff)

    print(f"   Checked {n_check} random voxels:")
    print(f"   Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print(f"   ✓ 1D-3D correspondence validated!")
    else:
        print(f"   ⚠️  Large difference detected: {max_diff:.2e}")

    # Save results
    print(f"\n7. Saving results...")

    # Save 3D data (in original axis order)
    output_3d = output_dir / 'subject1_3d_orig_order.npz'
    np.savez_compressed(
        output_3d,
        data_lr=data_lr,
        proba_labels=proba_labels,
        region_mask_lr=mask_lr
    )
    print(f"   ✓ 3D data saved: {output_3d}")

    # Save 1D data (compatible with data_3d_1d_mapper.py)
    output_1d = output_dir / 'subject1_1d_format.npz'
    np.savez_compressed(
        output_1d,
        multidim_data=multidim_data,
        seg_one_hot=seg_one_hot,
        region_seg=region_seg,
        region=mask_lr,  # Save mask for reconstruction
        n_voxels=n_voxels
    )
    print(f"   ✓ 1D data saved: {output_1d}")

    # Save mapping metadata
    import json
    metadata_file = output_dir / 'mapping_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    print(f"   ✓ Mapping metadata saved: {metadata_file}")

    print("\n✓ Example 7 completed successfully!")
    print("\nUse case:")
    print("  - 3D data is in original (Z,X,Y,C) order for visualization")
    print("  - 1D data is compatible with existing data_3d_1d_mapper.py tools")
    print("  - Row ordering ensures: multidim_data[i] ↔ data_lr[mask_lr][i]")


# ==============================================================================
# Example 8: Compare Processing vs Original Axis Order
# ==============================================================================

def example_8_compare_axis_orders():
    """
    Compare the two save_axis_order modes side-by-side
    """
    print("=" * 80)
    print("Example 8: Compare Processing vs Original Axis Orders")
    print("=" * 80)

    # Load data
    input_file = Path("/path/to/your/subject1_3d_validated.mat")

    if not input_file.exists():
        print(f"⚠️  Input file not found: {input_file}")
        print("Please update the path in the script.")
        return

    print("\n1. Loading data...")
    mat_data = sio.loadmat(input_file)

    data = mat_data['data']
    region_mask = mat_data['region_mask']
    region_labels = mat_data['region_labels']

    print(f"   Input shape: {data.shape} (Z,X,Y,C)")

    # Initialize pipeline
    pipeline = MRIDownsamplingPipeline(
        output_dir=Path("./compare_output"),
        log_level='INFO',
        random_seed=42
    )

    # Mode 1: Processing axis order (default)
    print("\n2. Running with save_axis_order='proc' (default)...")
    results_proc = pipeline.run(
        data=data,
        region_mask=region_mask,
        region_labels=region_labels,
        save_axis_order='proc'
    )

    # Mode 2: Original axis order
    print("\n3. Running with save_axis_order='orig'...")
    results_orig = pipeline.run(
        data=data,
        region_mask=region_mask,
        region_labels=region_labels,
        save_axis_order='orig'
    )

    # Compare results
    print("\n4. Comparison:")
    print(f"\n   Mode 1: save_axis_order='proc' (processing order)")
    print(f"      data_lr shape: {results_proc['data_lr'].shape} (X,Y,Z,C)")
    print(f"      Has multidim_data: {'multidim_data' in results_proc}")
    print(f"      Mapping: {results_proc['metadata']['mapping']['axes']}")

    print(f"\n   Mode 2: save_axis_order='orig' (original order)")
    print(f"      data_lr shape: {results_orig['data_lr'].shape} (Z,X,Y,C)")
    print(f"      Has multidim_data: {'multidim_data' in results_orig}")
    print(f"      multidim_data shape: {results_orig['multidim_data'].shape}")
    print(f"      seg_one_hot shape: {results_orig['seg_one_hot'].shape}")
    print(f"      Mapping: {results_orig['metadata']['mapping']['axes']}")

    # Verify data consistency (values should be identical, just reordered)
    print(f"\n5. Data consistency check:")

    # Extract ROI voxels from both
    voxels_proc = results_proc['data_lr'][results_proc['region_mask_lr'] > 0]
    voxels_orig_3d = results_orig['data_lr'][results_orig['region_mask_lr'] > 0]
    voxels_orig_1d = results_orig['multidim_data']

    # Compare value distributions (should be identical sets, just different order)
    print(f"   Processing order - mean: {voxels_proc.mean():.6f}, std: {voxels_proc.std():.6f}")
    print(f"   Original 3D order - mean: {voxels_orig_3d.mean():.6f}, std: {voxels_orig_3d.std():.6f}")
    print(f"   Original 1D order - mean: {voxels_orig_1d.mean():.6f}, std: {voxels_orig_1d.std():.6f}")

    print("\n✓ Example 8 completed successfully!")
    print("\nConclusion:")
    print("  - 'proc': For internal processing, no 1D data generation")
    print("  - 'orig': For compatibility with external tools, includes 1D format")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    import sys

    examples = {
        '1': ('Basic Usage', example_1_basic_usage),
        '2': ('Batch Processing', example_2_batch_processing),
        '3': ('Inspect Results', example_3_inspect_results),
        '4': ('Custom Configuration', example_4_custom_config),
        '5': ('HDF5 Loading', example_5_hdf5_loading),
        '6': ('Extract Channels', example_6_extract_channels),
        '7': ('Original Axis Order + 1D Data', example_7_save_original_axis_order),
        '8': ('Compare Axis Orders', example_8_compare_axis_orders),
    }

    print("\n" + "=" * 80)
    print("MRI Downsampling Pipeline - Example Usage")
    print("=" * 80)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("\nUsage: python example_usage.py [example_number]")
    print("Example: python example_usage.py 1")
    print("=" * 80 + "\n")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            func()
        else:
            print(f"Invalid example number: {choice}")
    else:
        # Run Example 1 by default
        print("Running Example 1 (default)...\n")
        example_1_basic_usage()
