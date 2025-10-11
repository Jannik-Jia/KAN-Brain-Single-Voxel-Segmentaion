#!/usr/bin/env python3
"""
测试概率标签转换功能
"""

import numpy as np
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from data_3d_1d_mapper import Data3D1DMapper


def test_proba_labels_conversion():
    """测试概率标签的3D-1D互转"""
    print("=" * 80)
    print("测试概率标签转换功能")
    print("=" * 80)

    # 创建mapper
    mapper = Data3D1DMapper(log_level='INFO')

    # 创建模拟数据
    print("\n1. 创建模拟3D概率标签...")
    shape_3d = (10, 12, 8)  # 小尺寸便于测试
    n_classes = 102

    # 创建随机ROI mask
    region_mask = np.random.rand(*shape_3d) > 0.6
    n_voxels = np.sum(region_mask)
    print(f"   3D形状: {shape_3d}, ROI体素数: {n_voxels}")

    # 创建概率标签（模拟downsampling pipeline的输出）
    proba_labels_3d = np.random.rand(*shape_3d, n_classes).astype(np.float32)

    # 归一化使每个体素的概率和为1
    prob_sum = proba_labels_3d.sum(axis=-1, keepdims=True)
    proba_labels_3d = proba_labels_3d / (prob_sum + 1e-10)

    # 验证概率和
    actual_sum = proba_labels_3d[region_mask].sum(axis=-1).mean()
    print(f"   概率和: {actual_sum:.6f} (应该接近1.0)")

    # 2. 测试3D到1D转换
    print("\n2. 测试3D到1D转换...")
    seg_proba_1d = mapper.convert_proba_labels_3d_to_1d(proba_labels_3d, region_mask)
    print(f"   ✓ 转换成功: {proba_labels_3d.shape} -> {seg_proba_1d.shape}")
    print(f"   预期形状: (102, {n_voxels}), 实际形状: {seg_proba_1d.shape}")

    assert seg_proba_1d.shape == (102, n_voxels), "1D形状不正确"

    # 验证概率和
    sum_1d = seg_proba_1d.sum(axis=0).mean()
    print(f"   1D概率和: {sum_1d:.6f}")

    # 3. 测试1D到3D转换
    print("\n3. 测试1D到3D转换...")
    proba_labels_3d_recovered = mapper.convert_proba_labels_1d_to_3d(seg_proba_1d, region_mask)
    print(f"   ✓ 转换成功: {seg_proba_1d.shape} -> {proba_labels_3d_recovered.shape}")
    print(f"   预期形状: {proba_labels_3d.shape}, 实际形状: {proba_labels_3d_recovered.shape}")

    assert proba_labels_3d_recovered.shape == proba_labels_3d.shape, "恢复的3D形状不正确"

    # 4. 验证往返一致性
    print("\n4. 验证往返转换一致性...")

    # 只比较ROI内的数据
    original_roi = proba_labels_3d[region_mask]
    recovered_roi = proba_labels_3d_recovered[region_mask]

    max_diff = np.abs(original_roi - recovered_roi).max()
    mean_diff = np.abs(original_roi - recovered_roi).mean()

    print(f"   最大差异: {max_diff:.2e}")
    print(f"   平均差异: {mean_diff:.2e}")

    if max_diff < 1e-6:
        print("   ✓ 往返转换验证通过！")
    else:
        print(f"   ✗ 往返转换有误差: {max_diff:.2e}")
        return False

    # 5. 测试集成到convert_3d_to_1d
    print("\n5. 测试集成到convert_3d_to_1d方法...")
    data_3d = {
        'data': np.random.randn(*shape_3d, 351).astype(np.float32),
        'region_mask': region_mask,
        'proba_labels': proba_labels_3d
    }

    data_1d = mapper.convert_3d_to_1d(data_3d)

    print(f"   ✓ 转换成功")
    print(f"   multidim_data: {data_1d['multidim_data'].shape}")
    print(f"   seg_one_hot: {data_1d['seg_one_hot'].shape}")
    assert 'seg_one_hot' in data_1d, "缺少seg_one_hot"
    assert data_1d['seg_one_hot'].shape == (102, n_voxels), "seg_one_hot形状不正确"

    # 6. 测试集成到convert_1d_to_3d
    print("\n6. 测试集成到convert_1d_to_3d方法...")
    data_3d_recovered = mapper.convert_1d_to_3d(data_1d)

    print(f"   ✓ 转换成功")
    print(f"   data: {data_3d_recovered['data'].shape}")
    print(f"   proba_labels: {data_3d_recovered['proba_labels'].shape}")
    print(f"   region_labels: {data_3d_recovered['region_labels'].shape}")

    assert 'proba_labels' in data_3d_recovered, "缺少proba_labels"
    assert 'region_labels' in data_3d_recovered, "缺少region_labels"

    # 验证一致性
    original_roi = data_3d['proba_labels'][region_mask]
    recovered_roi = data_3d_recovered['proba_labels'][region_mask]
    max_diff = np.abs(original_roi - recovered_roi).max()
    print(f"   完整流程最大差异: {max_diff:.2e}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)
    return True


def test_comparison_with_onehot():
    """对比概率标签和one-hot标签的区别"""
    print("\n" + "=" * 80)
    print("对比概率标签和one-hot标签")
    print("=" * 80)

    mapper = Data3D1DMapper(log_level='INFO')

    shape_3d = (10, 12, 8)
    region_mask = np.random.rand(*shape_3d) > 0.6
    n_voxels = np.sum(region_mask)

    # 1. 创建one-hot标签
    print("\n1. One-hot标签 (严格二值)...")
    region_labels = np.random.randint(0, 102, shape_3d).astype(np.uint8)
    data_3d_onehot = {
        'data': np.random.randn(*shape_3d, 351).astype(np.float32),
        'region_mask': region_mask,
        'region_labels': region_labels
    }

    data_1d_onehot = mapper.convert_3d_to_1d(data_3d_onehot)
    seg_onehot = data_1d_onehot['seg_one_hot']
    print(f"   seg_one_hot范围: [{seg_onehot.min():.1f}, {seg_onehot.max():.1f}]")
    print(f"   每列非零元素数: {(seg_onehot > 0).sum(axis=0).mean():.1f} (应该是1)")

    # 2. 创建概率标签
    print("\n2. 概率标签 (软标签)...")
    proba_labels_3d = np.random.rand(*shape_3d, 102).astype(np.float32)
    prob_sum = proba_labels_3d.sum(axis=-1, keepdims=True)
    proba_labels_3d = proba_labels_3d / prob_sum

    data_3d_proba = {
        'data': np.random.randn(*shape_3d, 351).astype(np.float32),
        'region_mask': region_mask,
        'proba_labels': proba_labels_3d
    }

    data_1d_proba = mapper.convert_3d_to_1d(data_3d_proba)
    seg_proba = data_1d_proba['seg_one_hot']
    print(f"   seg_one_hot范围: [{seg_proba.min():.4f}, {seg_proba.max():.4f}]")
    print(f"   每列非零元素数: {(seg_proba > 0.01).sum(axis=0).mean():.1f} (可以>1)")

    # 3. 测试1D到3D恢复
    print("\n3. 测试恢复...")
    data_3d_onehot_recovered = mapper.convert_1d_to_3d(data_1d_onehot)
    data_3d_proba_recovered = mapper.convert_1d_to_3d(data_1d_proba)

    print(f"   One-hot恢复: {'region_labels' in data_3d_onehot_recovered}, "
          f"{'proba_labels' not in data_3d_onehot_recovered}")
    print(f"   Proba恢复: {'region_labels' in data_3d_proba_recovered}, "
          f"{'proba_labels' in data_3d_proba_recovered}")

    print("\n✓ 对比测试完成")


if __name__ == "__main__":
    print("\n开始测试概率标签转换功能...\n")

    success = True

    try:
        test_proba_labels_conversion()
    except Exception as e:
        print(f"\n✗ 主测试失败: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        test_comparison_with_onehot()
    except Exception as e:
        print(f"\n✗ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！概率标签转换功能正常工作。")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ 测试失败，请检查代码")
        print("=" * 80)
        sys.exit(1)
