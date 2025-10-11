#!/usr/bin/env python3
"""
验证批处理生成的1D数据
验证概率标签和3D-1D互转的正确性
"""

import numpy as np
import sys
from pathlib import Path

# 添加1d-3d-convert模块路径
sys.path.insert(0, str(Path(__file__).parent / '1d-3d-convert'))

from data_3d_1d_mapper import Data3D1DMapper


def verify_downsampled_data(npz_file: Path):
    """
    验证downsampled NPZ文件的完整性

    Args:
        npz_file: 批处理生成的NPZ文件路径
    """
    print("=" * 80)
    print(f"验证文件: {npz_file.name}")
    print("=" * 80)

    # 加载数据
    print("\n1. 加载数据...")
    data = np.load(npz_file)

    print("   文件包含的key:")
    for key in data.files:
        if isinstance(data[key], np.ndarray):
            print(f"     - {key}: {data[key].shape} ({data[key].dtype})")
        else:
            print(f"     - {key}: {data[key]} ({type(data[key])})")

    # 检查必需的key
    required_3d_keys = ['data_lr', 'proba_labels', 'region_mask_lr']
    required_1d_keys = ['multidim_data', 'seg_one_hot', 'region_seg', 'n_voxels']

    has_3d = all(k in data for k in required_3d_keys)
    has_1d = all(k in data for k in required_1d_keys)

    print(f"\n2. 数据完整性检查:")
    print(f"   3D数据: {'✓' if has_3d else '✗'}")
    print(f"   1D数据: {'✓' if has_1d else '✗'}")

    if not has_3d:
        print("   错误: 缺少必需的3D数据")
        return False

    if not has_1d:
        print("   警告: 未包含1D数据（可能使用了save_axis_order='proc'）")
        return True

    # 验证shape一致性
    print("\n3. 验证Shape一致性...")

    # 3D数据
    data_lr = data['data_lr']
    proba_labels = data['proba_labels']
    region_mask_lr = data['region_mask_lr']

    # 1D数据
    multidim_data = data['multidim_data']
    seg_one_hot = data['seg_one_hot']
    region_seg = data['region_seg']
    n_voxels = int(data['n_voxels'])

    print(f"   3D数据:")
    print(f"     data_lr: {data_lr.shape} (应该是 Z,X,Y,351)")
    print(f"     proba_labels: {proba_labels.shape} (应该是 Z,X,Y,102)")
    print(f"     region_mask_lr: {region_mask_lr.shape} (应该是 Z,X,Y)")

    print(f"   1D数据:")
    print(f"     multidim_data: {multidim_data.shape} (应该是 n_voxels,351)")
    print(f"     seg_one_hot: {seg_one_hot.shape} (应该是 102,n_voxels)")
    print(f"     region_seg: {region_seg.shape} (应该是 n_voxels,)")
    print(f"     n_voxels: {n_voxels}")

    # 验证体素数一致性
    n_voxels_from_mask = np.sum(region_mask_lr > 0)
    print(f"\n4. 验证体素数一致性...")
    print(f"   region_mask_lr中的体素数: {n_voxels_from_mask}")
    print(f"   记录的n_voxels: {n_voxels}")
    print(f"   multidim_data行数: {multidim_data.shape[0]}")
    print(f"   seg_one_hot列数: {seg_one_hot.shape[1]}")

    if n_voxels_from_mask != n_voxels:
        print(f"   ✗ 体素数不一致！")
        return False

    if multidim_data.shape[0] != n_voxels:
        print(f"   ✗ multidim_data行数不匹配！")
        return False

    if seg_one_hot.shape[1] != n_voxels:
        print(f"   ✗ seg_one_hot列数不匹配！")
        return False

    print("   ✓ 体素数一致性验证通过")

    # 验证概率和
    print(f"\n5. 验证概率标签...")
    prob_sum_3d = proba_labels[region_mask_lr > 0].sum(axis=-1).mean()
    prob_sum_1d = seg_one_hot.sum(axis=0).mean()

    print(f"   3D proba_labels概率和: {prob_sum_3d:.6f}")
    print(f"   1D seg_one_hot概率和: {prob_sum_1d:.6f}")

    if not (0.99 <= prob_sum_3d <= 1.01):
        print(f"   ✗ 3D概率和异常")
        return False

    if not (0.99 <= prob_sum_1d <= 1.01):
        print(f"   ✗ 1D概率和异常")
        return False

    print("   ✓ 概率和验证通过")

    # 验证3D-1D对应关系
    print(f"\n6. 验证3D-1D数据对应关系...")

    # 随机选择10个体素进行验证
    check_indices = np.random.choice(n_voxels, size=min(10, n_voxels), replace=False)

    # 获取3D mask的线性索引（C-order）
    mask_indices = np.where(region_mask_lr.ravel(order='C'))[0]

    max_feat_diff = 0.0
    max_label_diff = 0.0

    for i in check_indices:
        # 1D数据
        feat_1d = multidim_data[i, :]
        label_1d = seg_one_hot[:, i]

        # 找到对应的3D位置
        mask_idx = mask_indices[i]
        z, x, y = np.unravel_index(mask_idx, region_mask_lr.shape, order='C')

        # 3D数据
        feat_3d = data_lr[z, x, y, :]
        label_3d = proba_labels[z, x, y, :]

        # 计算差异
        feat_diff = np.abs(feat_1d - feat_3d).max()
        label_diff = np.abs(label_1d - label_3d).max()

        max_feat_diff = max(max_feat_diff, feat_diff)
        max_label_diff = max(max_label_diff, label_diff)

    print(f"   特征最大差异: {max_feat_diff:.2e}")
    print(f"   标签最大差异: {max_label_diff:.2e}")

    if max_feat_diff > 1e-5 or max_label_diff > 1e-5:
        print(f"   ✗ 3D-1D对应关系验证失败")
        return False

    print("   ✓ 3D-1D对应关系验证通过")

    # 测试互转
    print(f"\n7. 测试3D-1D互转...")
    mapper = Data3D1DMapper(log_level='WARNING')

    # 构建data_3d字典
    data_3d = {
        'data': data_lr,
        'region_mask': region_mask_lr,
        'proba_labels': proba_labels
    }

    # 3D转1D
    data_1d_converted = mapper.convert_3d_to_1d(data_3d)

    # 比较转换结果
    feat_match = np.allclose(data_1d_converted['multidim_data'], multidim_data, rtol=1e-5, atol=1e-8)
    label_match = np.allclose(data_1d_converted['seg_one_hot'], seg_one_hot, rtol=1e-5, atol=1e-8)

    print(f"   特征一致性: {'✓' if feat_match else '✗'}")
    print(f"   标签一致性: {'✓' if label_match else '✗'}")

    if not feat_match or not label_match:
        print(f"   ✗ 互转测试失败")
        return False

    print("   ✓ 互转测试通过")

    print("\n" + "=" * 80)
    print("✅ 所有验证通过！数据格式正确。")
    print("=" * 80)

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证批处理生成的1D数据')
    parser.add_argument('npz_file', type=str, help='NPZ文件路径')

    args = parser.parse_args()

    npz_file = Path(args.npz_file)

    if not npz_file.exists():
        print(f"错误: 文件不存在: {npz_file}")
        sys.exit(1)

    success = verify_downsampled_data(npz_file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
