#!/usr/bin/env python3
"""
验证MAT文件加载后的实际数据形状
用于确认数据加载的前提条件
"""

import numpy as np
import h5py
from pathlib import Path
import sys


def inspect_mat_file_raw(filepath):
    """
    直接查看MAT文件中的原始数据形状（不做任何转换）
    """
    print(f"\n{'='*60}")
    print(f"文件: {Path(filepath).name}")
    print(f"{'='*60}")
    
    print("\n1. HDF5原始数据形状（MATLAB存储的实际形状）:")
    print("-" * 40)
    
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if not key.startswith("#"):
                dataset = f[key]
                print(f"  {key:20s}: {str(dataset.shape):20s} dtype={dataset.dtype}")
    
    print("\n2. 使用f[key][()]加载后的形状（Python numpy数组）:")
    print("-" * 40)
    
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if not key.startswith("#"):
                v = f[key][()]
                data[key] = v
                print(f"  {key:20s}: {str(v.shape):20s} dtype={v.dtype}")
    
    print("\n3. README中的预期形状:")
    print("-" * 40)
    expected_shapes = {
        'big_seg': '(384, 336, 256) - FreeSurfer完整分割',
        'region': '(384, 336, 256) - 脑组织二值掩膜',
        'multidim_data': '(351, n_voxels) - 多模态特征',
        'region_seg': '(1, n_voxels) - 体素级FreeSurfer标签',
        'seg_one_hot': '(102, n_voxels) - One-Hot编码'
    }
    
    for key, expected in expected_shapes.items():
        print(f"  {key:20s}: {expected}")
    
    print("\n4. 详细分析:")
    print("-" * 40)
    
    # 分析region
    if 'region' in data:
        region = data['region']
        print(f"\nregion分析:")
        print(f"  形状: {region.shape}")
        print(f"  唯一值: {np.unique(region)}")
        print(f"  非零值数量: {np.sum(region > 0):,}")
        
        # 计算有效体素数
        n_voxels_from_region = np.sum(region.astype(bool))
        print(f"  有效体素数(region==1): {n_voxels_from_region:,}")
    
    # 分析multidim_data
    if 'multidim_data' in data:
        mdata = data['multidim_data']
        print(f"\nmultidim_data分析:")
        print(f"  形状: {mdata.shape}")
        print(f"  第一维: {mdata.shape[0]} {'(应该是351或n_voxels)' if mdata.shape[0] in [351] else ''}")
        print(f"  第二维: {mdata.shape[1]} {'(应该是n_voxels或351)' if mdata.shape[1] in [351] else ''}")
        
        # 判断哪个维度是351
        if mdata.shape[0] == 351:
            print(f"  → 看起来是 (351, n_voxels) 格式")
            n_voxels_from_mdata = mdata.shape[1]
        elif mdata.shape[1] == 351:
            print(f"  → 看起来是 (n_voxels, 351) 格式")
            n_voxels_from_mdata = mdata.shape[0]
        else:
            print(f"  → 无法确定格式")
            n_voxels_from_mdata = 0
        
        if 'region' in data and n_voxels_from_mdata > 0:
            if n_voxels_from_mdata == n_voxels_from_region:
                print(f"  ✅ 体素数匹配: {n_voxels_from_mdata:,}")
            else:
                print(f"  ❌ 体素数不匹配: multidim_data={n_voxels_from_mdata:,}, region={n_voxels_from_region:,}")
    
    # 分析seg_one_hot
    if 'seg_one_hot' in data:
        seg = data['seg_one_hot']
        print(f"\nseg_one_hot分析:")
        print(f"  形状: {seg.shape}")
        print(f"  第一维: {seg.shape[0]} {'(应该是102或n_voxels)' if seg.shape[0] in [102] else ''}")
        print(f"  第二维: {seg.shape[1]} {'(应该是n_voxels或102)' if seg.shape[1] in [102] else ''}")
        
        # 判断哪个维度是102
        if seg.shape[0] == 102:
            print(f"  → 看起来是 (102, n_voxels) 格式")
            n_voxels_from_seg = seg.shape[1]
        elif seg.shape[1] == 102:
            print(f"  → 看起来是 (n_voxels, 102) 格式")
            n_voxels_from_seg = seg.shape[0]
        else:
            print(f"  → 无法确定格式")
            n_voxels_from_seg = 0
        
        if 'region' in data and n_voxels_from_seg > 0:
            if n_voxels_from_seg == n_voxels_from_region:
                print(f"  ✅ 体素数匹配: {n_voxels_from_seg:,}")
            else:
                print(f"  ❌ 体素数不匹配: seg_one_hot={n_voxels_from_seg:,}, region={n_voxels_from_region:,}")
    
    # 分析region_seg
    if 'region_seg' in data:
        rseg = data['region_seg']
        print(f"\nregion_seg分析:")
        print(f"  形状: {rseg.shape}")
        print(f"  总元素: {rseg.size:,}")
        
        if 'region' in data:
            if rseg.size == n_voxels_from_region:
                print(f"  ✅ 元素数匹配region有效体素数")
            else:
                print(f"  ❌ 元素数不匹配: region_seg={rseg.size:,}, region={n_voxels_from_region:,}")
    
    # 分析big_seg
    if 'big_seg' in data:
        bseg = data['big_seg']
        print(f"\nbig_seg分析:")
        print(f"  形状: {bseg.shape}")
        print(f"  唯一值数量: {len(np.unique(bseg))}")
        print(f"  值范围: {bseg.min()} - {bseg.max()}")
    
    print("\n5. 转置测试（模拟.T的效果）:")
    print("-" * 40)
    
    for key in ['region', 'big_seg', 'multidim_data', 'seg_one_hot']:
        if key in data:
            original = data[key]
            transposed = original.T
            print(f"  {key:20s}: {str(original.shape):20s} → .T → {str(transposed.shape):20s}")
    
    return data


def compare_with_readme_specs():
    """
    对比README中的规格说明
    """
    print("\n" + "="*60)
    print("README规格总结:")
    print("="*60)
    
    specs = """
    根据README文档：
    
    原始MAT文件（HDF5格式）中的数据：
    1. big_seg: (384, 336, 256) - FreeSurfer完整分割 [需要转置]
    2. region: (384, 336, 256) - 脑组织二值掩膜 [需要转置]
    3. multidim_data: (351, n_voxels) - 多模态特征 [需要转置]
    4. region_seg: (1, n_voxels) - 体素级标签 [展平即可]
    5. seg_one_hot: (102, n_voxels) - One-Hot编码 [需要转置]
    
    ⚠️ 重要提示（来自README第7-11行）：
    "HDF5文件中数据使用MATLAB的Fortran顺序（列优先）存储，Python加载时需要转置"
    - 原始存储：MATLAB格式，使用Fortran顺序
    - Python处理：需要转置以适配行优先习惯
    - 正确加载方式：多维数组必须转置（.T）
    
    但是根据您的指出：
    - 对3D体数据使用.T会反转轴顺序，导致空间方向错误
    - 应该根据数据类型精细处理，而不是一刀切转置
    """
    print(specs)


def main():
    """主函数"""
    
    # 默认测试路径
    default_path = "/home/jannik/Documents/mri_mat_onehot/1D/OHC_13_ckgulxe.mat"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = default_path
        print(f"使用默认测试文件（可通过命令行参数指定其他文件）")
    
    test_path = Path(test_file)
    
    if not test_path.exists():
        print(f"❌ 文件不存在: {test_path}")
        print("\n请指定正确的MAT文件路径:")
        print("  python verify_data_shapes.py /path/to/your/file.mat")
        return
    
    # 检查文件
    data = inspect_mat_file_raw(test_path)
    
    # 对比README
    compare_with_readme_specs()
    
    # 结论
    print("\n" + "="*60)
    print("结论和建议:")
    print("="*60)
    
    print("""
    基于实际数据形状，建议的处理策略：
    
    1. 3D体数据 (region, big_seg):
       - 如果是 (256, 336, 384) → 需要处理为 (384, 336, 256)
       - 如果已经是 (384, 336, 256) → 保持不变
       - 不要用.T（会变成 (384, 336, 256)），考虑用 np.transpose(v, (2, 1, 0))
    
    2. 2D特征矩阵 (multidim_data):
       - 如果是 (351, n_voxels) → 转置为 (n_voxels, 351)
       - 如果已经是 (n_voxels, 351) → 保持不变
    
    3. One-hot编码 (seg_one_hot):
       - 如果是 (102, n_voxels) → 保持（这是正确格式）
       - 如果是 (n_voxels, 102) → 转置为 (102, n_voxels)
    
    4. 1D标签 (region_seg):
       - 展平为 (n_voxels,) 即可
    """)


if __name__ == "__main__":
    main()