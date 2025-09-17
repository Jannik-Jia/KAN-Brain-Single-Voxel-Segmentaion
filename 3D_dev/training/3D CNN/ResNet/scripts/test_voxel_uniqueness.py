#!/usr/bin/env python3
"""
测试体素唯一性和重复问题

对比原始批量数据集和改进版本在避免重复训练相同patch方面的效果
"""

import sys
import time
from pathlib import Path
from typing import List, Set, Dict
from collections import Counter

# 添加模型目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'models'))

from batch_dataset import BatchMRIBrain2DPatchDataset
from batch_dataset_improved import ImprovedBatchMRIBrain2DPatchDataset


def find_mat_files(data_dir: Path) -> List[Path]:
    """查找MAT文件"""
    mat_files = sorted(data_dir.glob('subject*_3d_validated.mat'))
    if len(mat_files) == 0:
        mat_files = sorted(data_dir.glob('*.mat'))
    return mat_files


def test_original_dataset_duplicates(mat_files: List[Path], batch_files: int = 3) -> Dict:
    """测试原始批量数据集的重复情况"""
    print("=== 测试原始批量数据集 ===")

    dataset = BatchMRIBrain2DPatchDataset(
        mat_files=mat_files,
        patch_size=7,
        samples_per_subject=1000,
        is_train=True,
        batch_files=batch_files
    )

    # 模拟一个epoch的训练
    epoch_voxels = set()
    batch_voxels_count = []
    duplicate_count = 0

    # 重置到第一批
    dataset.reset_to_first_batch()

    batch_info = dataset.get_current_batch_info()
    total_batches = batch_info['total_batches']

    print(f"总批次数: {total_batches}")

    for batch_idx in range(total_batches):
        current_info = dataset.get_current_batch_info()
        print(f"\\n批次 {current_info['current_batch']}/{current_info['total_batches']}:")
        print(f"  样本数: {len(dataset)}")

        # 收集当前批次的体素
        batch_voxels = set()
        for i in range(len(dataset.current_sample_indices)):
            file_path, x, y, z = dataset.current_sample_indices[i]
            voxel_id = f"{file_path}:{x}:{y}:{z}"

            if voxel_id in epoch_voxels:
                duplicate_count += 1
            else:
                epoch_voxels.add(voxel_id)

            batch_voxels.add(voxel_id)

        batch_voxels_count.append(len(batch_voxels))
        print(f"  唯一体素数: {len(batch_voxels)}")
        print(f"  累计重复数: {duplicate_count}")

        # 切换到下一批
        if batch_idx < total_batches - 1:
            has_next = dataset.next_batch()
            if not has_next:
                print("  意外的批次结束")
                break

    results = {
        'dataset_type': 'original',
        'total_batches': total_batches,
        'total_unique_voxels': len(epoch_voxels),
        'total_duplicates': duplicate_count,
        'batch_voxel_counts': batch_voxels_count,
        'avg_batch_size': sum(batch_voxels_count) / len(batch_voxels_count) if batch_voxels_count else 0
    }

    print(f"\\n原始数据集结果:")
    print(f"  总唯一体素: {results['total_unique_voxels']}")
    print(f"  总重复数: {results['total_duplicates']}")
    print(f"  平均批次大小: {results['avg_batch_size']:.1f}")

    return results


def test_improved_dataset_duplicates(mat_files: List[Path], batch_files: int = 3) -> Dict:
    """测试改进批量数据集的重复情况"""
    print("\\n=== 测试改进批量数据集 ===")

    dataset = ImprovedBatchMRIBrain2DPatchDataset(
        mat_files=mat_files,
        patch_size=7,
        samples_per_subject=1000,
        is_train=True,
        batch_files=batch_files
    )

    # 模拟一个epoch的训练
    epoch_voxels = set()
    batch_voxels_count = []
    duplicate_count = 0

    # 重置到第一批
    dataset.reset_to_first_batch()

    batch_info = dataset.get_current_batch_info()
    total_batches = batch_info['total_batches']

    print(f"总批次数: {total_batches}")
    print(f"Epoch总样本数: {batch_info['epoch_total_samples']}")

    for batch_idx in range(total_batches):
        current_info = dataset.get_current_batch_info()
        print(f"\\n批次 {current_info['current_batch']}/{current_info['total_batches']}:")
        print(f"  样本数: {len(dataset)}")

        # 收集当前批次的体素
        batch_voxels = set()
        for i in range(len(dataset.current_batch_indices)):
            file_path, x, y, z = dataset.current_batch_indices[i]
            voxel_id = f"{file_path}:{x}:{y}:{z}"

            if voxel_id in epoch_voxels:
                duplicate_count += 1
                print(f"    警告: 发现重复体素 {voxel_id}")
            else:
                epoch_voxels.add(voxel_id)

            batch_voxels.add(voxel_id)

        batch_voxels_count.append(len(batch_voxels))
        print(f"  唯一体素数: {len(batch_voxels)}")
        print(f"  累计重复数: {duplicate_count}")

        # 切换到下一批
        if batch_idx < total_batches - 1:
            has_next = dataset.next_batch()
            if not has_next:
                print("  意外的批次结束")
                break

    # 获取覆盖统计
    coverage_stats = dataset.get_epoch_coverage_stats()

    results = {
        'dataset_type': 'improved',
        'total_batches': total_batches,
        'total_unique_voxels': len(epoch_voxels),
        'total_duplicates': duplicate_count,
        'batch_voxel_counts': batch_voxels_count,
        'avg_batch_size': sum(batch_voxels_count) / len(batch_voxels_count) if batch_voxels_count else 0,
        'coverage_stats': coverage_stats
    }

    print(f"\\n改进数据集结果:")
    print(f"  总唯一体素: {results['total_unique_voxels']}")
    print(f"  总重复数: {results['total_duplicates']}")
    print(f"  平均批次大小: {results['avg_batch_size']:.1f}")
    print(f"  覆盖统计: {coverage_stats}")

    return results


def test_cross_epoch_uniqueness(mat_files: List[Path], num_epochs: int = 3) -> Dict:
    """测试跨epoch的体素分布唯一性"""
    print(f"\\n=== 测试跨{num_epochs}个Epoch的体素分布 ===")

    dataset = ImprovedBatchMRIBrain2DPatchDataset(
        mat_files=mat_files,
        patch_size=7,
        samples_per_subject=1000,
        is_train=True,
        batch_files=3
    )

    epoch_voxel_sets = []

    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch + 1}:")

        # 重置到第一批
        dataset.reset_to_first_batch()

        # 收集这个epoch的所有体素
        epoch_voxels = set()
        batch_info = dataset.get_current_batch_info()
        total_batches = batch_info['total_batches']

        for batch_idx in range(total_batches):
            # 收集当前批次的体素
            for i in range(len(dataset.current_batch_indices)):
                file_path, x, y, z = dataset.current_batch_indices[i]
                voxel_id = f"{file_path}:{x}:{y}:{z}"
                epoch_voxels.add(voxel_id)

            # 切换到下一批
            if batch_idx < total_batches - 1:
                dataset.next_batch()

        epoch_voxel_sets.append(epoch_voxels)
        print(f"  Epoch {epoch + 1}唯一体素数: {len(epoch_voxels)}")

        # 准备下一个epoch
        if epoch < num_epochs - 1:
            dataset.start_new_epoch()

    # 分析跨epoch的重叠情况
    print(f"\\n跨Epoch分析:")

    # 计算epoch间的重叠
    for i in range(len(epoch_voxel_sets)):
        for j in range(i + 1, len(epoch_voxel_sets)):
            overlap = len(epoch_voxel_sets[i] & epoch_voxel_sets[j])
            total_unique = len(epoch_voxel_sets[i] | epoch_voxel_sets[j])
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0

            print(f"  Epoch {i+1} vs Epoch {j+1}: "
                  f"重叠{overlap}个体素 "
                  f"(重叠率: {overlap_ratio:.2%})")

    # 总体统计
    all_voxels = set()
    for epoch_voxels in epoch_voxel_sets:
        all_voxels.update(epoch_voxels)

    results = {
        'num_epochs': num_epochs,
        'total_unique_across_epochs': len(all_voxels),
        'epoch_sizes': [len(vs) for vs in epoch_voxel_sets],
        'average_epoch_size': sum(len(vs) for vs in epoch_voxel_sets) / len(epoch_voxel_sets)
    }

    print(f"\\n总体统计:")
    print(f"  {num_epochs}个epoch总共覆盖: {len(all_voxels)}个唯一体素")
    print(f"  平均每个epoch: {results['average_epoch_size']:.1f}个体素")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='测试体素唯一性和重复问题')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='包含MAT文件的目录')
    parser.add_argument('--batch_files', type=int, default=3,
                       help='每批次加载的文件数量')
    parser.add_argument('--test_epochs', type=int, default=3,
                       help='测试的epoch数量')

    args = parser.parse_args()

    # 查找MAT文件
    data_dir = Path(args.data_dir)
    mat_files = find_mat_files(data_dir)

    if len(mat_files) < 6:
        print(f"需要至少6个MAT文件进行测试，当前只有{len(mat_files)}个")
        return

    # 使用前6个文件进行测试
    test_files = mat_files[:6]
    print(f"使用{len(test_files)}个MAT文件进行测试")

    # 测试原始数据集的重复情况
    original_results = test_original_dataset_duplicates(test_files, args.batch_files)

    # 测试改进数据集的重复情况
    improved_results = test_improved_dataset_duplicates(test_files, args.batch_files)

    # 测试跨epoch的唯一性
    cross_epoch_results = test_cross_epoch_uniqueness(test_files, args.test_epochs)

    # 对比结果
    print("\\n" + "="*50)
    print("对比结果总结:")
    print("="*50)

    print(f"\\n单Epoch内重复情况:")
    print(f"  原始数据集:")
    print(f"    - 总唯一体素: {original_results['total_unique_voxels']}")
    print(f"    - 重复次数: {original_results['total_duplicates']}")
    print(f"    - 重复率: {original_results['total_duplicates'] / (original_results['total_unique_voxels'] + original_results['total_duplicates']) * 100:.1f}%")

    print(f"\\n  改进数据集:")
    print(f"    - 总唯一体素: {improved_results['total_unique_voxels']}")
    print(f"    - 重复次数: {improved_results['total_duplicates']}")
    print(f"    - 重复率: {improved_results['total_duplicates'] / max(1, improved_results['total_unique_voxels'] + improved_results['total_duplicates']) * 100:.1f}%")

    # 改进效果
    if original_results['total_duplicates'] > 0:
        improvement = (original_results['total_duplicates'] - improved_results['total_duplicates']) / original_results['total_duplicates'] * 100
        print(f"\\n改进效果:")
        print(f"  - 重复减少: {improvement:.1f}%")
        if improved_results['total_duplicates'] == 0:
            print(f"  - ✅ 完全消除了epoch内重复！")

    print(f"\\n跨Epoch唯一性:")
    print(f"  - 测试了{cross_epoch_results['num_epochs']}个epoch")
    print(f"  - 总覆盖体素: {cross_epoch_results['total_unique_across_epochs']}")
    print(f"  - 平均每epoch: {cross_epoch_results['average_epoch_size']:.1f}")

    print("\\n测试完成！")


if __name__ == "__main__":
    main()