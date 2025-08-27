#!/usr/bin/env python
# coding: utf-8

"""
检查Balanced数据一致性工具
验证每个受试者的balanced_data_4d10000.nii.gz和balanced_labels_3d10000.nii.gz
是否真的属于同一个受试者，避免复制错误
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import hashlib
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def compute_file_hash(file_path, chunk_size=8192):
    """计算文件的MD5哈希值"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def check_spatial_alignment(img_4d, label_3d):
    """检查4D影像和3D标签的空间对齐"""
    # 检查空间维度
    spatial_match = img_4d.shape[:3] == label_3d.shape
    
    # 检查仿射矩阵
    affine_match = np.allclose(img_4d.affine, label_3d.affine, rtol=1e-5)
    
    # 检查体素尺寸
    voxel_4d = img_4d.header.get_zooms()[:3]
    voxel_3d = label_3d.header.get_zooms()
    voxel_match = np.allclose(voxel_4d, voxel_3d, rtol=1e-5)
    
    return {
        'spatial_dims_match': spatial_match,
        'affine_match': affine_match,
        'voxel_size_match': voxel_match,
        'all_match': spatial_match and affine_match and voxel_match
    }

def analyze_data_correlation(data_4d, label_3d, n_samples=1000):
    """
    分析4D数据和3D标签的相关性
    通过采样检查标签边界是否与影像特征对应
    """
    # 创建标签边界掩码
    from scipy import ndimage
    edges = ndimage.sobel(label_3d.astype(float))
    edge_mask = np.abs(edges) > 0
    
    # 在边界位置采样
    edge_coords = np.where(edge_mask)
    n_edge_points = len(edge_coords[0])
    
    if n_edge_points == 0:
        return {'correlation_score': 0, 'message': 'No edges found'}
    
    # 随机采样边界点
    sample_size = min(n_samples, n_edge_points)
    sample_indices = np.random.choice(n_edge_points, sample_size, replace=False)
    
    # 计算边界处的影像梯度
    gradient_scores = []
    for idx in sample_indices:
        x, y, z = edge_coords[0][idx], edge_coords[1][idx], edge_coords[2][idx]
        
        # 检查邻域内的影像变化（使用第一个模态）
        window = 1
        x_min, x_max = max(0, x-window), min(data_4d.shape[0], x+window+1)
        y_min, y_max = max(0, y-window), min(data_4d.shape[1], y+window+1)
        z_min, z_max = max(0, z-window), min(data_4d.shape[2], z+window+1)
        
        # 计算局部标准差作为梯度指标
        local_region = data_4d[x_min:x_max, y_min:y_max, z_min:z_max, 0]
        local_std = np.std(local_region)
        gradient_scores.append(local_std)
    
    # 计算平均梯度分数
    mean_gradient = np.mean(gradient_scores)
    
    # 同时在非边界区域采样作为对照
    non_edge_mask = ~edge_mask
    non_edge_coords = np.where(non_edge_mask)
    n_non_edge = len(non_edge_coords[0])
    
    if n_non_edge > 0:
        sample_size = min(n_samples, n_non_edge)
        sample_indices = np.random.choice(n_non_edge, sample_size, replace=False)
        
        non_edge_gradients = []
        for idx in sample_indices:
            x, y, z = non_edge_coords[0][idx], non_edge_coords[1][idx], non_edge_coords[2][idx]
            
            window = 1
            x_min, x_max = max(0, x-window), min(data_4d.shape[0], x+window+1)
            y_min, y_max = max(0, y-window), min(data_4d.shape[1], y+window+1)
            z_min, z_max = max(0, z-window), min(data_4d.shape[2], z+window+1)
            
            local_region = data_4d[x_min:x_max, y_min:y_max, z_min:z_max, 0]
            local_std = np.std(local_region)
            non_edge_gradients.append(local_std)
        
        mean_non_edge_gradient = np.mean(non_edge_gradients)
        
        # 计算相关性分数（边界梯度应该更高）
        if mean_non_edge_gradient > 0:
            correlation_score = (mean_gradient - mean_non_edge_gradient) / mean_non_edge_gradient
        else:
            correlation_score = 0
    else:
        correlation_score = mean_gradient
        mean_non_edge_gradient = 0
    
    return {
        'correlation_score': float(correlation_score),
        'edge_gradient': float(mean_gradient),
        'non_edge_gradient': float(mean_non_edge_gradient),
        'higher_edge_gradient': mean_gradient > mean_non_edge_gradient
    }

def check_label_consistency(label_3d):
    """检查标签的内部一致性"""
    unique_labels = np.unique(label_3d)
    
    # 检查是否有孤立的小区域
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(label_3d > 0)
    
    # 计算每个连通组件的大小
    component_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        component_sizes.append(size)
    
    # 检查是否有异常小的组件（可能是噪声）
    if component_sizes:
        min_size = min(component_sizes)
        max_size = max(component_sizes)
        suspicious_components = sum(1 for s in component_sizes if s < 100)  # 小于100体素的组件
    else:
        min_size = 0
        max_size = 0
        suspicious_components = 0
    
    return {
        'num_unique_labels': len(unique_labels),
        'num_connected_components': num_features,
        'min_component_size': min_size,
        'max_component_size': max_size,
        'suspicious_small_components': suspicious_components,
        'likely_consistent': suspicious_components < 5  # 少于5个可疑小组件
    }

def check_subject_data_consistency(subject_path):
    """
    全面检查单个受试者的数据一致性
    """
    subject_path = Path(subject_path)
    balanced_dir = subject_path / "balanced_output"
    
    result = {
        'subject_id': subject_path.name,
        'status': 'unknown',
        'checks': {}
    }
    
    # 检查文件是否存在
    data_4d_path = balanced_dir / "balanced_data_4d10000.nii.gz"
    label_3d_path = balanced_dir / "balanced_labels_3d10000.nii.gz"
    
    if not balanced_dir.exists():
        result['status'] = 'no_balanced_dir'
        return result
    
    if not data_4d_path.exists() or not label_3d_path.exists():
        result['status'] = 'missing_files'
        result['checks']['4d_exists'] = data_4d_path.exists()
        result['checks']['3d_exists'] = label_3d_path.exists()
        return result
    
    try:
        # 计算文件哈希
        result['checks']['4d_hash'] = compute_file_hash(data_4d_path)
        result['checks']['3d_hash'] = compute_file_hash(label_3d_path)
        
        # 加载数据
        print(f"  加载数据: {subject_path.name}")
        img_4d = nib.load(data_4d_path)
        label_3d = nib.load(label_3d_path)
        
        # 基本信息
        result['checks']['4d_shape'] = img_4d.shape
        result['checks']['3d_shape'] = label_3d.shape
        
        # 空间对齐检查
        alignment = check_spatial_alignment(img_4d, label_3d)
        result['checks']['spatial_alignment'] = alignment
        
        # 标签一致性检查
        print(f"  检查标签一致性...")
        label_data = label_3d.get_fdata()
        label_consistency = check_label_consistency(label_data)
        result['checks']['label_consistency'] = label_consistency
        
        # 数据相关性分析（这会加载完整数据，比较慢）
        print(f"  分析数据相关性...")
        data_4d = img_4d.get_fdata()
        correlation = analyze_data_correlation(data_4d, label_data)
        result['checks']['data_correlation'] = correlation
        
        # 综合判断
        is_consistent = (
            alignment['all_match'] and
            label_consistency['likely_consistent'] and
            correlation['higher_edge_gradient']
        )
        
        result['status'] = 'consistent' if is_consistent else 'suspicious'
        result['checks']['overall_consistent'] = is_consistent
        
        # 如果发现问题，记录详细信息
        if not is_consistent:
            problems = []
            if not alignment['all_match']:
                problems.append("空间不对齐")
            if not label_consistency['likely_consistent']:
                problems.append(f"标签可疑(小组件:{label_consistency['suspicious_small_components']})")
            if not correlation['higher_edge_gradient']:
                problems.append("边界梯度异常")
            result['problems'] = problems
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result

def check_cross_subject_consistency(all_results):
    """
    跨受试者一致性检查，找出可能的复制错误
    """
    # 检查是否有相同的文件哈希
    hash_dict = {}
    duplicates = []
    
    for result in all_results:
        if 'checks' in result and '4d_hash' in result['checks']:
            hash_4d = result['checks']['4d_hash']
            hash_3d = result['checks']['3d_hash']
            
            # 检查4D数据哈希
            if hash_4d in hash_dict:
                duplicates.append({
                    'type': '4D_data',
                    'subjects': [hash_dict[hash_4d], result['subject_id']],
                    'hash': hash_4d
                })
            else:
                hash_dict[hash_4d] = result['subject_id']
            
            # 检查3D标签哈希
            if hash_3d in hash_dict:
                duplicates.append({
                    'type': '3D_label',
                    'subjects': [hash_dict[hash_3d], result['subject_id']],
                    'hash': hash_3d
                })
            else:
                hash_dict[hash_3d] = result['subject_id']
    
    return duplicates

def main():
    """主函数"""
    # 设置路径
    ROOT_DIR = Path("/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS")
    
    # 创建输出目录
    output_dir = Path("./balanced")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Balanced数据一致性检查工具")
    print("=" * 80)
    print(f"数据目录: {ROOT_DIR}")
    print(f"输出目录: {output_dir}")
    
    # 查找所有受试者
    subject_dirs = sorted([d for d in ROOT_DIR.iterdir() 
                          if d.is_dir() and d.name.startswith("FOR_")])
    
    print(f"\n找到 {len(subject_dirs)} 个受试者")
    
    # 检查每个受试者
    all_results = []
    suspicious_subjects = []
    
    print("\n开始检查数据一致性...")
    print("-" * 40)
    
    for subject_dir in tqdm(subject_dirs, desc="检查进度"):
        result = check_subject_data_consistency(subject_dir)
        all_results.append(result)
        
        if result['status'] == 'suspicious':
            suspicious_subjects.append(result)
    
    # 跨受试者检查
    print("\n检查是否有重复文件...")
    duplicates = check_cross_subject_consistency(all_results)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("检查结果汇总")
    print("=" * 80)
    
    # 统计各种状态
    status_counts = {}
    for result in all_results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\n状态统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} 个受试者")
    
    # 显示可疑的受试者
    if suspicious_subjects:
        print(f"\n⚠️ 发现 {len(suspicious_subjects)} 个可疑的受试者:")
        for subj in suspicious_subjects:
            print(f"\n  {subj['subject_id']}:")
            if 'problems' in subj:
                for problem in subj['problems']:
                    print(f"    - {problem}")
    
    # 显示重复文件
    if duplicates:
        print(f"\n❌ 发现 {len(duplicates)} 个重复文件:")
        for dup in duplicates:
            print(f"\n  {dup['type']} 重复:")
            print(f"    受试者: {', '.join(dup['subjects'])}")
            print(f"    哈希: {dup['hash'][:16]}...")
    else:
        print("\n✅ 未发现重复文件")
    
    # 保存详细结果
    report = {
        'check_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_subjects': len(subject_dirs),
        'status_summary': status_counts,
        'suspicious_subjects': suspicious_subjects,
        'duplicate_files': duplicates,
        'all_results': all_results
    }
    
    # 保存JSON报告
    report_file = output_dir / f"consistency_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细报告已保存到: {report_file}")
    
    # 生成CSV摘要
    df_data = []
    for result in all_results:
        row = {
            'subject_id': result['subject_id'],
            'status': result['status']
        }
        
        if 'checks' in result:
            checks = result['checks']
            if 'spatial_alignment' in checks:
                row['spatial_match'] = checks['spatial_alignment']['all_match']
            if 'label_consistency' in checks:
                row['label_consistent'] = checks['label_consistency']['likely_consistent']
                row['num_labels'] = checks['label_consistency']['num_unique_labels']
                row['small_components'] = checks['label_consistency']['suspicious_small_components']
            if 'data_correlation' in checks:
                row['edge_gradient_ok'] = checks['data_correlation']['higher_edge_gradient']
                row['correlation_score'] = checks['data_correlation']['correlation_score']
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    csv_file = output_dir / f"consistency_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"💾 CSV摘要已保存到: {csv_file}")
    
    # 给出建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    
    if duplicates:
        print("\n⚠️ 发现重复文件，请检查是否有复制错误!")
        print("   建议重新复制正确的文件到对应的balanced_output目录")
    
    if suspicious_subjects:
        print(f"\n⚠️ 发现 {len(suspicious_subjects)} 个数据不一致的受试者")
        print("   建议检查这些受试者的数据是否正确配对")
    
    if not duplicates and not suspicious_subjects:
        print("\n✅ 所有数据看起来一致性良好!")
        print("   可以继续进行数据处理")
    
    print("\n检查完成!")
    
    return report

if __name__ == "__main__":
    report = main()