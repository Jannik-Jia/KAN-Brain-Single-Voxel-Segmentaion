#!/usr/bin/env python3
"""
测试批量数据加载功能

验证批量数据集是否正常工作，检查内存使用情况
"""

import sys
import time
import psutil
import os
from pathlib import Path
from typing import List

# 添加模型目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'models'))

from batch_dataset import BatchMRIBrain2DPatchDataset, create_batch_data_loaders
import torch
from torch.utils.data import DataLoader


def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def find_mat_files(data_dir: Path) -> List[Path]:
    """查找MAT文件"""
    mat_files = sorted(data_dir.glob('subject*_3d_validated.mat'))
    if len(mat_files) == 0:
        mat_files = sorted(data_dir.glob('*.mat'))
    return mat_files


def test_batch_dataset(data_dir: str, batch_files: int = 3):
    """测试批量数据集功能"""
    print("=== 批量数据加载测试 ===")

    # 查找MAT文件
    data_dir = Path(data_dir)
    mat_files = find_mat_files(data_dir)

    if len(mat_files) < 6:
        print(f"警告: 只找到{len(mat_files)}个MAT文件，建议至少6个文件进行测试")
        return

    print(f"找到{len(mat_files)}个MAT文件")

    # 使用前6个文件进行测试
    test_files = mat_files[:6]

    print(f"\\n=== 测试批量数据集 (每批{batch_files}个文件) ===")

    # 记录初始内存使用
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.1f} MB")

    # 创建批量数据集
    print("\\n创建批量数据集...")
    start_time = time.time()

    dataset = BatchMRIBrain2DPatchDataset(
        mat_files=test_files,
        patch_size=7,
        samples_per_subject=1000,  # 减少样本数以加快测试
        is_train=True,
        batch_files=batch_files
    )

    creation_time = time.time() - start_time
    creation_memory = get_memory_usage()

    print(f"数据集创建完成，耗时: {creation_time:.1f}秒")
    print(f"创建后内存使用: {creation_memory:.1f} MB (增加: {creation_memory - initial_memory:.1f} MB)")

    # 获取批次信息
    batch_info = dataset.get_current_batch_info()
    print(f"\\n批次信息: {batch_info}")

    # 测试数据加载
    print(f"\\n当前批次数据集大小: {len(dataset)}")

    # 测试数据访问
    print("\\n测试数据访问...")
    patch, label = dataset[0]
    print(f"图像块形状: {patch.shape}")  # 应该是 (351, 7, 7)
    print(f"标签: {label.item()}")       # 应该是 0-101

    # 测试DataLoader
    print("\\n测试DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_patches, batch_labels = next(iter(loader))
    print(f"批次图像块形状: {batch_patches.shape}")  # (4, 351, 7, 7)
    print(f"批次标签形状: {batch_labels.shape}")     # (4,)

    # 测试批次切换
    print("\\n=== 测试批次切换 ===")
    total_batches = batch_info['total_batches']

    for batch_idx in range(total_batches):
        current_info = dataset.get_current_batch_info()
        current_memory = get_memory_usage()

        print(f"\\n批次 {current_info['current_batch']}/{current_info['total_batches']}:")
        print(f"  当前文件数: {current_info['current_files']}")
        print(f"  当前样本数: {current_info['current_samples']}")
        print(f"  内存使用: {current_memory:.1f} MB")

        # 测试加载一些数据
        if len(dataset) > 0:
            test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            for i, (patches, labels) in enumerate(test_loader):
                if i >= 2:  # 只测试前2个批次
                    break
            print(f"  成功加载数据")

        # 切换到下一批（如果不是最后一批）
        if batch_idx < total_batches - 1:
            print(f"  切换到下一批...")
            has_next = dataset.next_batch()
            if not has_next:
                print(f"  警告: 意外的批次结束")

    final_memory = get_memory_usage()
    print(f"\\n=== 测试完成 ===")
    print(f"最终内存使用: {final_memory:.1f} MB")
    print(f"总内存增长: {final_memory - initial_memory:.1f} MB")


def test_vs_original_dataset(data_dir: str):
    """对比原始数据集和批量数据集的内存使用"""
    print("\\n=== 内存使用对比测试 ===")

    data_dir = Path(data_dir)
    mat_files = find_mat_files(data_dir)

    if len(mat_files) < 6:
        print("需要至少6个MAT文件进行对比测试")
        return

    test_files = mat_files[:6]

    # 测试原始数据集（一次性加载所有文件）
    print("\\n测试原始数据集（一次性加载）...")
    initial_memory = get_memory_usage()

    try:
        from dataset import MRIBrain2DPatchDataset

        start_time = time.time()
        original_dataset = MRIBrain2DPatchDataset(
            mat_files=test_files,
            patch_size=7,
            samples_per_subject=1000,
            is_train=True,
            cache_data=True  # 一次性加载所有数据
        )
        original_time = time.time() - start_time
        original_memory = get_memory_usage()

        print(f"原始数据集创建时间: {original_time:.1f}秒")
        print(f"原始数据集内存使用: {original_memory - initial_memory:.1f} MB")
        print(f"原始数据集样本数: {len(original_dataset)}")

        # 释放原始数据集
        del original_dataset

    except Exception as e:
        print(f"原始数据集测试失败: {e}")
        original_memory = initial_memory

    # 等待内存释放
    time.sleep(2)

    # 测试批量数据集
    print("\\n测试批量数据集（分批加载）...")
    reset_memory = get_memory_usage()

    start_time = time.time()
    batch_dataset = BatchMRIBrain2DPatchDataset(
        mat_files=test_files,
        patch_size=7,
        samples_per_subject=1000,
        is_train=True,
        batch_files=3  # 每次只加载3个文件
    )
    batch_time = time.time() - start_time
    batch_memory = get_memory_usage()

    print(f"批量数据集创建时间: {batch_time:.1f}秒")
    print(f"批量数据集内存使用: {batch_memory - reset_memory:.1f} MB")
    print(f"批量数据集当前样本数: {len(batch_dataset)}")

    # 内存节省计算
    if 'original_memory' in locals():
        memory_saved = (original_memory - initial_memory) - (batch_memory - reset_memory)
        print(f"\\n内存节省: {memory_saved:.1f} MB")
        print(f"内存节省比例: {memory_saved / (original_memory - initial_memory) * 100:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='测试批量数据加载功能')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='包含MAT文件的目录')
    parser.add_argument('--batch_files', type=int, default=3,
                       help='每批次加载的文件数量')
    parser.add_argument('--compare', action='store_true',
                       help='与原始数据集进行内存使用对比')

    args = parser.parse_args()

    print(f"Python进程PID: {os.getpid()}")
    print(f"可用内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")

    # 基本批量加载测试
    test_batch_dataset(args.data_dir, args.batch_files)

    # 内存对比测试（可选）
    if args.compare:
        test_vs_original_dataset(args.data_dir)

    print("\\n测试完成！")


if __name__ == "__main__":
    main()