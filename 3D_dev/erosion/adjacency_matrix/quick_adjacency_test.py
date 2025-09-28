#!/usr/bin/env python3
"""
邻接矩阵计算快速验证脚本

在不实际计算的情况下验证数据加载和系统功能。
"""

import sys
import numpy as np
import h5py
from pathlib import Path
from typing import List


def test_data_loading(data_dir: Path) -> bool:
    """测试数据加载功能和标签验证"""
    print("🔍 测试数据加载和标签验证...")

    # 查找MAT文件
    mat_files = sorted(data_dir.glob('subject*_3d_validated.mat'))
    if not mat_files:
        mat_files = sorted(data_dir.glob('*.mat'))

    if not mat_files:
        print(f"❌ 在{data_dir}中未找到MAT文件")
        return False

    print(f"✅ 找到{len(mat_files)}个MAT文件")

    # 测试第一个文件
    test_file = mat_files[0]
    print(f"📂 测试文件: {test_file}")

    try:
        from compute_adjacency_matrices import AdjacencyMatrixComputer

        # 创建计算器实例
        computer = AdjacencyMatrixComputer(connectivity=6)

        # 测试加载和验证
        region_labels, region_mask, data_loading_validation = computer.load_subject_data(test_file)
        label_validation = computer.validate_labels(region_labels, region_mask)

        print("📋 数据加载验证结果:")
        print(f"  - 文件大小: {data_loading_validation['file_size_mb']:.1f} MB")
        print(f"  - 原始形状: {data_loading_validation['original_shapes']}")
        print(f"  - 最终形状: {data_loading_validation['final_shapes']}")

        if data_loading_validation['warnings']:
            print(f"  - ⚠️ 警告: {data_loading_validation['warnings']}")

        print("\n🏷️ 标签验证结果:")
        range_analysis = label_validation['label_range_analysis']
        print(f"  - 检测到的标签格式: {range_analysis['detected_format']}")
        print(f"  - 标签范围: {range_analysis['min_label']}-{range_analysis['max_label']}")
        print(f"  - 唯一标签数: {range_analysis['actual_unique_count']}")

        if range_analysis['missing_labels_in_range']:
            print(f"  - ⚠️ 范围内缺失标签: {range_analysis['missing_labels_in_range']}")

        mask_stats = label_validation['mask_statistics']
        print(f"  - 有效体素: {mask_stats['valid_voxels']:,} ({mask_stats['valid_percentage']:.1f}%)")

        quality_checks = label_validation['data_quality_checks']
        if quality_checks['isolated_single_voxel_labels'] > 0:
            print(f"  - ⚠️ 孤立单体素区域: {quality_checks['isolated_single_voxel_labels']} 个")

        print(f"  - 中位区域大小: {quality_checks['median_region_size']:.0f} 体素")

        # 显示前20个标签
        unique_labels = label_validation['label_statistics']['valid_labels']['unique_values']
        print(f"  - 前20个标签: {unique_labels[:20]}")

    except Exception as e:
        print(f"❌ 数据加载和验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    print("✅ 数据加载和标签验证测试通过")
    return True


def test_adjacency_algorithm() -> bool:
    """测试邻接矩阵计算算法（使用模拟数据）"""
    print("\n🧮 测试邻接矩阵算法...")

    # 创建简单的3D测试数据
    test_shape = (10, 10, 10)
    region_labels = np.zeros(test_shape, dtype=np.int16)
    region_mask = np.zeros(test_shape, dtype=bool)

    # 创建几个简单的区域
    region_labels[2:5, 2:5, 2:5] = 0  # 区域0
    region_labels[6:9, 2:5, 2:5] = 1  # 区域1 (应该与区域0邻接)
    region_labels[2:5, 6:9, 2:5] = 2  # 区域2 (应该与区域0邻接)
    region_labels[6:9, 6:9, 6:9] = 3  # 区域3 (不与其他区域邻接)

    # 设置掩膜
    region_mask[region_labels >= 0] = True

    print(f"📊 测试数据: {test_shape}, 4个区域")

    try:
        from scipy import ndimage

        # 测试邻接检测
        adjacency_matrix = np.zeros((4, 4), dtype=np.uint8)
        structure = ndimage.generate_binary_structure(3, 1)  # 6连通

        for i in range(4):
            mask_i = (region_labels == i)
            if not np.any(mask_i):
                continue

            # 膨胀
            dilated_mask_i = ndimage.binary_dilation(mask_i, structure=structure)

            for j in range(i+1, 4):
                mask_j = (region_labels == j)
                if not np.any(mask_j):
                    continue

                # 检查重叠
                overlap = dilated_mask_i & mask_j
                if np.any(overlap):
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

        print(f"🔗 计算得到的邻接矩阵:")
        print(adjacency_matrix)

        # 验证预期结果
        expected_adjacencies = [(0, 1), (0, 2)]  # 区域0应该与区域1和2邻接
        actual_adjacencies = list(zip(*np.where(np.triu(adjacency_matrix) == 1)))

        print(f"✅ 预期邻接: {expected_adjacencies}")
        print(f"✅ 实际邻接: {actual_adjacencies}")

        if set(expected_adjacencies) == set(actual_adjacencies):
            print("✅ 邻接算法测试通过")
            return True
        else:
            print("❌ 邻接算法结果不符合预期")
            return False

    except Exception as e:
        print(f"❌ 邻接算法测试失败: {str(e)}")
        return False


def test_storage_format() -> bool:
    """测试存储格式"""
    print("\n💾 测试存储格式...")

    try:
        # 创建测试数据
        test_adjacency = np.random.randint(0, 2, (102, 102))
        test_adjacency = (test_adjacency + test_adjacency.T) // 2  # 确保对称
        np.fill_diagonal(test_adjacency, 0)  # 对角线为0

        test_contacts = np.random.randint(0, 100, (102, 102)) * test_adjacency

        # 测试HDF5存储
        test_file = Path('./test_adjacency.h5')

        with h5py.File(test_file, 'w') as f:
            # 创建组
            adj_group = f.create_group('adjacency')
            contact_group = f.create_group('contact_counts')
            meta_group = f.create_group('metadata')

            # 存储数据
            adj_group.create_dataset('matrix', data=test_adjacency, compression='gzip')
            contact_group.create_dataset('counts', data=test_contacts, compression='gzip')

            # 元数据
            meta_group.attrs['total_regions'] = 102
            meta_group.attrs['total_adjacencies'] = np.sum(test_adjacency) // 2

        # 测试读取
        with h5py.File(test_file, 'r') as f:
            loaded_adjacency = f['adjacency/matrix'][()]
            loaded_contacts = f['contact_counts/counts'][()]

            if np.array_equal(test_adjacency, loaded_adjacency):
                print("✅ 邻接矩阵存储/读取正确")
            else:
                print("❌ 邻接矩阵存储/读取错误")
                return False

            if np.array_equal(test_contacts, loaded_contacts):
                print("✅ 接触计数存储/读取正确")
            else:
                print("❌ 接触计数存储/读取错误")
                return False

        # 清理测试文件
        test_file.unlink()
        print("✅ 存储格式测试通过")
        return True

    except Exception as e:
        print(f"❌ 存储格式测试失败: {str(e)}")
        return False


def test_analysis_utils() -> bool:
    """测试分析工具"""
    print("\n🔬 测试分析工具...")

    try:
        # 创建模拟数据
        adjacency_matrix = np.random.randint(0, 2, (102, 102))
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) // 2
        np.fill_diagonal(adjacency_matrix, 0)

        confusion_matrix = np.random.rand(102, 102) * 0.2
        np.fill_diagonal(confusion_matrix, np.random.rand(102) * 0.8 + 0.2)

        # 测试Hadamard乘积
        hadamard = confusion_matrix * adjacency_matrix
        print(f"✅ Hadamard乘积计算: 形状={hadamard.shape}, 非零元素={np.count_nonzero(hadamard)}")

        # 测试相关性计算
        from scipy import stats
        upper_indices = np.triu_indices_from(adjacency_matrix, k=1)
        adj_values = adjacency_matrix[upper_indices]
        conf_values = confusion_matrix[upper_indices]

        correlation, p_value = stats.spearmanr(adj_values, conf_values)
        print(f"✅ Spearman相关性: r={correlation:.3f}, p={p_value:.3f}")

        print("✅ 分析工具测试通过")
        return True

    except Exception as e:
        print(f"❌ 分析工具测试失败: {str(e)}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='邻接矩阵系统快速验证')
    parser.add_argument('--data_dir', type=str,
                      default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated',
                      help='3D验证数据目录')

    args = parser.parse_args()

    print("🚀 邻接矩阵计算系统快速验证")
    print("=" * 50)

    data_dir = Path(args.data_dir)

    # 运行所有测试
    tests = [
        ("数据加载", lambda: test_data_loading(data_dir)),
        ("邻接算法", test_adjacency_algorithm),
        ("存储格式", test_storage_format),
        ("分析工具", test_analysis_utils)
    ]

    all_passed = True
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}测试: 通过")
            else:
                print(f"❌ {test_name}测试: 失败")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name}测试: 异常 - {str(e)}")
            all_passed = False

        print()

    print("=" * 50)
    if all_passed:
        print("🎉 所有测试通过！系统就绪。")
        print("\n📋 下一步:")
        print("1. 运行: python compute_adjacency_matrices.py --test_only")
        print("2. 如果成功，运行: bash run_adjacency_batch.sh")
        return 0
    else:
        print("❌ 部分测试失败，请检查环境和依赖。")
        return 1


if __name__ == '__main__':
    sys.exit(main())