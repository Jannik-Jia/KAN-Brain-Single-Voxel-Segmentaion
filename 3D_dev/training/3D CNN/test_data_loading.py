#!/usr/bin/env python3
"""
测试数据加载脚本
验证3D MAT文件的加载和patch提取
"""

import h5py
import numpy as np
from pathlib import Path
import argparse

def test_load_mat_file(mat_file: Path):
    """测试加载单个MAT文件"""
    print(f"\n测试文件: {mat_file.name}")
    print("="*50)
    
    with h5py.File(mat_file, 'r') as f:
        # 列出所有keys
        print("文件包含的keys:")
        for key in f.keys():
            if not key.startswith('#'):
                shape = f[key].shape
                dtype = f[key].dtype
                print(f"  {key}: shape={shape}, dtype={dtype}")
        
        # 加载并检查关键数据
        if 'data' in f:
            data = f['data'][()]
            print(f"\n原始 'data' 形状: {data.shape}")
            
            # 根据实际的数据格式进行正确的转置
            if data.shape[0] == 351:
                # 如果第一个维度是351，说明是 (351, 384, 336, 256) 格式
                data = np.transpose(data, (1, 2, 3, 0))  # -> (384, 336, 256, 351)
            elif data.shape[-1] == 351:
                # 如果最后一个维度是351，已经是正确格式
                pass
            else:
                print(f"警告：无法识别数据格式，shape: {data.shape}")
                
            print(f"转置后 'data' 形状: {data.shape}")
            print(f"  数据范围: [{data.min():.2f}, {data.max():.2f}]")
            print(f"  数据类型: {data.dtype}")
            print(f"  特征维度: {data.shape[-1]}")
        
        if 'region_labels' in f:
            labels = f['region_labels'][()]
            
            # labels的处理
            if labels.shape != (384, 336, 256):
                labels = labels.T
                
            print(f"\n'region_labels' 形状: {labels.shape}")
            unique_labels = np.unique(labels)
            print(f"  唯一标签数: {len(unique_labels)}")
            print(f"  标签范围: [{unique_labels.min()}, {unique_labels.max()}]")
            
            # 统计每个标签的体素数
            label_counts = {}
            for label in unique_labels[:10]:  # 只显示前10个
                count = np.sum(labels == label)
                label_counts[label] = count
            print(f"  前10个标签分布: {label_counts}")
        
        if 'region_mask' in f:
            mask = f['region_mask'][()]
            
            # mask的处理
            if mask.shape != (384, 336, 256):
                mask = mask.T
                
            print(f"\n'region_mask' 形状: {mask.shape}")
            valid_voxels = np.sum(mask > 0)
            total_voxels = mask.size
            print(f"  有效体素数: {valid_voxels:,} / {total_voxels:,} ({100*valid_voxels/total_voxels:.2f}%)")

def test_patch_extraction(mat_file: Path, patch_size: int = 3):
    """测试2D patch提取"""
    print(f"\n\n测试 {patch_size}×{patch_size} 2D Patch 提取")
    print("="*50)
    
    with h5py.File(mat_file, 'r') as f:
        # 加载数据
        data = f['data'][()]
        labels = f['region_labels'][()]
        mask = f['region_mask'][()]
        
        print(f"原始数据形状: data={data.shape}, labels={labels.shape}, mask={mask.shape}")
        
        # 根据实际的数据格式进行正确的转置
        if data.shape[0] == 351:
            # 如果第一个维度是351，说明是 (351, 384, 336, 256) 格式
            data = np.transpose(data, (1, 2, 3, 0))  # -> (384, 336, 256, 351)
        elif data.shape[-1] == 351:
            # 如果最后一个维度是351，已经是正确格式
            pass
        else:
            print(f"警告：无法识别数据格式，shape: {data.shape}")
        
        # labels和mask的处理
        if labels.shape != (384, 336, 256):
            labels = labels.T
        if mask.shape != (384, 336, 256):
            mask = mask.T
            
        print(f"转置后数据形状: data={data.shape}, labels={labels.shape}, mask={mask.shape}")
        
        # 找到一个有效体素
        valid_positions = np.where((mask > 0) & (labels > 0))
        if len(valid_positions[0]) > 0:
            # 随机选择一个位置
            idx = np.random.randint(0, len(valid_positions[0]))
            x, y, z = valid_positions[0][idx], valid_positions[1][idx], valid_positions[2][idx]
            
            print(f"选择的体素位置: ({x}, {y}, {z})")
            print(f"该位置的标签: {labels[x, y, z]}")
            print(f"在z={z}切片上提取{patch_size}×{patch_size}的2D patch")
            
            # 提取2D patch
            p = patch_size // 2
            
            # 在x-y平面上计算边界
            x_min = max(0, x - p)
            x_max = min(data.shape[0], x + p + 1)
            y_min = max(0, y - p)
            y_max = min(data.shape[1], y + p + 1)
            
            # 提取2D patch（z固定）
            patch_2d = data[x_min:x_max, y_min:y_max, z, :]
            print(f"提取的2D patch形状: {patch_2d.shape}")
            
            # 如果需要padding
            if patch_2d.shape[:2] != (patch_size, patch_size):
                print(f"需要padding到 ({patch_size}, {patch_size}, 351)")
                padded = np.zeros((patch_size, patch_size, 351), dtype=data.dtype)
                
                x_off = p - (x - x_min)
                y_off = p - (y - y_min)
                
                padded[x_off:x_off+patch_2d.shape[0], 
                       y_off:y_off+patch_2d.shape[1], :] = patch_2d
                
                print(f"Padding后形状: {padded.shape}")
                patch_2d = padded
            
            # 转换为模型输入格式 (C, H, W)
            patch_tensor = patch_2d.transpose(2, 0, 1)
            print(f"模型输入形状 (C, H, W): {patch_tensor.shape}")
            print(f"  C=351个特征通道")
            print(f"  H×W={patch_size}×{patch_size}空间维度")
            
            return True
        else:
            print("未找到有效体素")
            return False

def main():
    parser = argparse.ArgumentParser(description='测试数据加载')
    parser.add_argument('--mat_file', type=str, required=True,
                       help='要测试的MAT文件路径')
    parser.add_argument('--patch_size', type=int, default=3,
                       help='Patch大小')
    
    args = parser.parse_args()
    
    mat_file = Path(args.mat_file)
    
    if not mat_file.exists():
        print(f"错误：文件 {mat_file} 不存在")
        return
    
    # 测试加载
    test_load_mat_file(mat_file)
    
    # 测试patch提取
    test_patch_extraction(mat_file, args.patch_size)
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()