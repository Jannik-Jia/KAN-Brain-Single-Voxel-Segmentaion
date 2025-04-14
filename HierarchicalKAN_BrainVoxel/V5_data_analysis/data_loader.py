#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载和预处理模块
用于高效加载脑部MRI数据集并进行初步处理
"""

import os
import sys
import numpy as np
import h5py
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from tqdm import tqdm

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_loader')

# 检测是否可以使用GPU
try:
    import cupy as cp
    import cudf
    import cuml
    HAS_GPU = True
    logger.info("GPU加速可用")
except ImportError:
    HAS_GPU = False
    logger.warning("未检测到GPU加速库，将使用CPU进行计算")

# 特征组定义
FEATURE_GROUPS = {
    'diffusion': list(range(0, 15)),           # 扩散特征 (0-14)
    'qti': list(range(15, 225)),              # QTI特征 (15-224)
    'cest': list(range(225, 341)),            # CEST特征 (225-340)
    'all_features': list(range(0, 341))       # 全部特征 (0-340)
}

def load_brain_voxel_test_data(check_normalization=True):
    """
    加载脑部MRI测试数据集，处理one-hot标签格式
    
    参数:
        check_normalization: 是否检查数据是否已标准化
        
    返回:
        dataset: 包含数据和标签的字典
    """
    # 固定的数据路径
    base_path = '/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data'
    test_label_dir = os.path.join(base_path, 'test_set_by_label')
    
    logger.info(f"从固定路径加载测试数据: {test_label_dir}")
    
    # 读取标签索引文件
    index_file = os.path.join(test_label_dir, "test_label_index.txt")
    if not os.path.exists(index_file):
        index_file = os.path.join(test_label_dir, "label_index.txt")
        if not os.path.exists(index_file):
            logger.error(f"标签索引文件不存在: {index_file}")
            return None
    
    # 加载标签信息
    label_info = {}
    with open(index_file, 'r') as f:
        # 跳过表头（如果有）
        first_line = f.readline().strip()
        if not first_line[0].isdigit():  # 如果第一行不是以数字开头，认为是表头
            pass  # 已经跳过了第一行
        else:
            # 如果第一行是数据，重新处理
            parts = first_line.split(',')
            if len(parts) >= 3:
                label_id = int(parts[0])
                voxel_count = int(parts[1])
                filename = parts[2] if parts[2] else None
                label_info[label_id] = {'count': voxel_count, 'filename': filename}
        
        # 处理剩余行
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                label_id = int(parts[0])
                voxel_count = int(parts[1])
                filename = parts[2] if parts[2] else None
                label_info[label_id] = {'count': voxel_count, 'filename': filename}
    
    # 获取有效标签（有体素数据的标签）
    valid_labels = [label_id for label_id, info in label_info.items() if info['count'] > 0]
    logger.info(f"找到 {len(valid_labels)} 个有效标签")
    
    # 初始化数据和标签列表
    all_data = []
    all_onehot_labels = []  # 存储one-hot编码标签
    
    # 加载每个标签的数据
    for label_id in tqdm(valid_labels, desc="加载标签数据"):
        info = label_info[label_id]
        if info['count'] == 0 or not info['filename']:
            continue
            
        # 构建文件路径
        data_file = os.path.join(test_label_dir, info['filename'])
        
        try:
            # 加载数据文件
            label_data = np.load(data_file)
            
            # 假设标签文件包含特征数据和one-hot编码标签
            # 我们需要确定哪部分是特征，哪部分是标签
            
            # 如果数据是二维的，可能已经只包含特征
            if len(label_data.shape) == 2:
                features = label_data
                # 创建该标签的one-hot编码
                n_samples = features.shape[0]
                onehot = np.zeros((n_samples, 102))
                onehot[:, label_id] = 1
            # 如果数据包含标签，可能需要分离
            # 注意：这里假设最后102列是one-hot标签，实际情况可能需要调整
            elif label_data.shape[1] > 341 + 102:  # 特征341维 + 标签102维
                features = label_data[:, :-102]  # 假设前面是特征
                onehot = label_data[:, -102:]    # 假设后面是标签
            else:
                # 如果数据格式不明确，可能需要特殊处理
                # 这里假设数据只包含特征
                features = label_data
                # 创建该标签的one-hot编码
                n_samples = features.shape[0]
                onehot = np.zeros((n_samples, 102))
                onehot[:, label_id] = 1
            
            all_data.append(features)
            all_onehot_labels.append(onehot)
                
            logger.info(f"加载标签 {label_id} 的 {features.shape[0]} 个样本，特征维度: {features.shape[1]}")
            
        except Exception as e:
            logger.error(f"加载标签 {label_id} 的数据时出错: {str(e)}")
    
    # 转换为numpy数组
    all_data = np.vstack(all_data) if all_data else np.array([])
    all_onehot_labels = np.vstack(all_onehot_labels) if all_onehot_labels else np.array([])
    
    # 检查数据是否已标准化
    if check_normalization and all_data.size > 0:
        # 计算每个特征的均值和标准差
        feature_means = np.mean(all_data, axis=0)
        feature_stds = np.std(all_data, axis=0)
        
        # 检查均值是否接近0，标准差是否接近1
        mean_near_zero = np.allclose(feature_means, 0, atol=0.1)
        std_near_one = np.allclose(feature_stds, 1, atol=0.5)
        
        if mean_near_zero and std_near_one:
            logger.info("数据检查: 数据已经过标准化处理（均值接近0，标准差接近1）")
        else:
            logger.info("数据检查: 数据可能未标准化")
            logger.info(f"  特征均值范围: [{np.min(feature_means):.4f}, {np.max(feature_means):.4f}]")
            logger.info(f"  特征标准差范围: [{np.min(feature_stds):.4f}, {np.max(feature_stds):.4f}]")
    
    # 从one-hot转换为整数标签（便于某些分析）
    int_labels = np.argmax(all_onehot_labels, axis=1) if all_onehot_labels.size > 0 else np.array([])
    
    # 统计各标签的样本数量
    if int_labels.size > 0:
        unique_labels, counts = np.unique(int_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"标签 {label}: {count} 个样本")
    
    # 构建结果字典
    result = {
        'test_samples': all_data,
        'test_labels': int_labels,  # 整数标签，便于某些分析
        'test_labels_onehot': all_onehot_labels,  # 原始one-hot标签
        'feature_groups': FEATURE_GROUPS,  # 使用预定义的特征组
        'label_info': label_info,
        'valid_labels': valid_labels
    }
    
    logger.info(f"测试数据加载完成: {all_data.shape}, 标签: {all_onehot_labels.shape}")
    if all_data.size > 0:
        logger.info(f"数据范围: [{np.min(all_data):.4f}, {np.max(all_data):.4f}]")
    
    return result


def load_voxel_data_by_label(test_label_dir, feature_dim=341, check_normalization=True):
    """
    加载按label分类存储的脑体素数据，支持one-hot标签格式
    
    参数:
        test_label_dir: 测试数据目录，包含按标签分类的数据
        feature_dim: 特征维度，默认为341
        check_normalization: 是否检查数据是否已标准化
        
    返回:
        dataset: 包含数据和标签的字典
    """
    logger.info(f"加载按标签分类的测试数据，从目录: {test_label_dir}")
    
    # 读取标签索引文件
    index_file = os.path.join(test_label_dir, "label_index.txt")
    if not os.path.exists(index_file):
        index_file = os.path.join(test_label_dir, "test_label_index.txt")
        if not os.path.exists(index_file):
            logger.error(f"标签索引文件不存在: {index_file}")
            return None
    
    # 加载标签信息
    label_info = {}
    with open(index_file, 'r') as f:
        # 跳过表头
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                label_id = int(parts[0])
                voxel_count = int(parts[1])
                filename = parts[2] if parts[2] else None
                label_info[label_id] = {'count': voxel_count, 'filename': filename}
    
    # 获取有效标签（有体素数据的标签）
    valid_labels = [label_id for label_id, info in label_info.items() if info['count'] > 0]
    logger.info(f"找到 {len(valid_labels)} 个有效标签")
    
    # 初始化数据和标签列表
    all_data = []
    all_labels = []
    
    # 加载每个标签的数据
    for label_id in tqdm(valid_labels, desc="加载标签数据"):
        info = label_info[label_id]
        if info['count'] == 0 or not info['filename']:
            continue
            
        # 构建文件路径
        data_file = os.path.join(test_label_dir, info['filename'])
        
        try:
            # 加载数据文件
            label_data = np.load(data_file)
            
            # 创建该标签的one-hot编码（初始化为零矩阵）
            # 假设总共有102个标签类别
            n_samples = label_data.shape[0]
            
            # 对于每个样本，添加数据和one-hot标签
            for i in range(n_samples):
                all_data.append(label_data[i])
                
                # 创建one-hot标签
                one_hot = np.zeros(102)
                one_hot[label_id] = 1
                all_labels.append(one_hot)
                
            logger.info(f"加载标签 {label_id} 的 {n_samples} 个样本")
            
        except Exception as e:
            logger.error(f"加载标签 {label_id} 的数据时出错: {str(e)}")
    
    # 转换为numpy数组
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # 检查数据是否已标准化
    if check_normalization:
        # 计算每个特征的均值和标准差
        feature_means = np.mean(all_data, axis=0)
        feature_stds = np.std(all_data, axis=0)
        
        # 检查均值是否接近0，标准差是否接近1
        mean_near_zero = np.allclose(feature_means, 0, atol=0.1)
        std_near_one = np.allclose(feature_stds, 1, atol=0.5)
        
        if mean_near_zero and std_near_one:
            logger.info("数据检查: 数据已经过标准化处理（均值接近0，标准差接近1）")
        else:
            logger.info("数据检查: 数据可能未标准化")
            logger.info(f"  特征均值范围: [{np.min(feature_means):.4f}, {np.max(feature_means):.4f}]")
            logger.info(f"  特征标准差范围: [{np.min(feature_stds):.4f}, {np.max(feature_stds):.4f}]")
    
    # 统计各标签的样本数量
    if len(all_labels) > 0:
        label_counts = np.sum(all_labels, axis=0)
        for i, count in enumerate(label_counts):
            if count > 0:
                logger.info(f"标签 {i}: {int(count)} 个样本")
    
    # 从one-hot转换回单一整数标签，便于后续分析
    # 找出每个样本中为1的位置作为类别标签
    labels_single = np.argmax(all_labels, axis=1)
    
    # 构建结果字典
    result = {
        'test_samples': all_data,
        'test_labels': labels_single,  # 使用整数标签
        'test_labels_onehot': all_labels,  # 保留one-hot标签
        'feature_groups': FEATURE_GROUPS,  # 使用预定义的特征组
        'label_info': label_info,
        'valid_labels': valid_labels
    }
    
    logger.info(f"测试数据加载完成: {all_data.shape}, 标签: {all_labels.shape}")
    logger.info(f"数据范围: [{np.min(all_data):.4f}, {np.max(all_data):.4f}]")
    
    return result




def load_mat_file(file_path):
    """
    加载.mat文件
    
    参数:
        file_path: .mat文件路径
    返回:
        数据字典
    """
    try:
        return sio.loadmat(file_path)
    except:
        # 尝试使用h5py加载较新版本的.mat文件
        try:
            return h5py.File(file_path, 'r')
        except Exception as e:
            logger.error(f"无法加载MAT文件 {file_path}: {str(e)}")
            return None

def load_offset_list(file_path):
    """
    加载offset列表文件
    
    参数:
        file_path: 文件路径
    返回:
        offset列表
    """
    try:
        with open(file_path, 'r') as f:
            offset_list = [float(line.strip()) for line in f if line.strip()]
        return offset_list
    except Exception as e:
        logger.error(f"无法加载offset列表文件 {file_path}: {str(e)}")
        return None

def preprocess_data(data, labels, feature_groups=None, normalize_method='robust', 
                   use_gpu=False, verbose=True):
    """
    数据预处理
    
    参数:
        data: 输入数据
        labels: 标签
        feature_groups: 特征组字典
        normalize_method: 标准化方法 ('standard', 'robust', 'none')
        use_gpu: 是否使用GPU加速
        verbose: 是否输出详细信息
    
    返回:
        processed_data: 处理后的特征字典
        preprocessing_info: 预处理信息
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS
    
    if verbose:
        logger.info(f"使用{normalize_method}方法进行数据预处理...")
    
    processed_data = {}
    preprocessing_info = {'scalers': {}}
    
    # 使用GPU加速（如果可用）
    if use_gpu and HAS_GPU:
        try:
            data_gpu = cp.asarray(data)
            for group_name, indices in tqdm(feature_groups.items(), desc="预处理特征组"):
                # 提取特征组
                group_data = data_gpu[:, indices].get()  # 转回CPU进行预处理
                
                # 标准化
                if normalize_method == 'standard':
                    scaler = StandardScaler()
                elif normalize_method == 'robust':
                    scaler = RobustScaler()
                else:  # 'none'
                    processed_data[group_name] = group_data
                    preprocessing_info['scalers'][group_name] = None
                    continue
                
                # 应用标准化
                normalized_data = scaler.fit_transform(group_data)
                processed_data[group_name] = normalized_data
                preprocessing_info['scalers'][group_name] = scaler
                
                if verbose:
                    logger.info(f"处理特征组 {group_name}: {normalized_data.shape}")
                    logger.info(f"  标准化后范围: [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")
            
            # 释放GPU内存
            del data_gpu
            import gc
            gc.collect()
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
        
        except Exception as e:
            logger.error(f"GPU处理失败: {str(e)}")
            logger.info("回退到CPU处理...")
            use_gpu = False
    
    # 使用CPU处理
    if not use_gpu or not HAS_GPU:
        for group_name, indices in tqdm(feature_groups.items(), desc="预处理特征组"):
            # 提取特征组
            group_data = data[:, indices]
            
            # 标准化
            if normalize_method == 'standard':
                scaler = StandardScaler()
            elif normalize_method == 'robust':
                scaler = RobustScaler()
            else:  # 'none'
                processed_data[group_name] = group_data
                preprocessing_info['scalers'][group_name] = None
                continue
            
            # 应用标准化
            normalized_data = scaler.fit_transform(group_data)
            processed_data[group_name] = normalized_data
            preprocessing_info['scalers'][group_name] = scaler
            
            if verbose:
                logger.info(f"处理特征组 {group_name}: {normalized_data.shape}")
                logger.info(f"  标准化后范围: [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")
    
    return processed_data, preprocessing_info

def define_big_classes():
    """
    定义脑部分类的大类映射
    
    返回:
        fine_to_big: 细分类到大类的映射
        big_to_fine: 大类到细分类的映射
        big_class_names: 大类名称列表
    """
    # 大类名称
    big_class_names = [
        "脑室系统 (Ventricular System)",  # 0
        "白质 (White Matter)",             # 1
        "灰质-皮层 (Cortical Gray Matter)", # 2
        "深部灰质核团 (Deep Gray Nuclei)",  # 3
        "边缘系统 (Limbic System)",        # 4
        "脑干 (Brain Stem)",              # 5
        "其他结构 (Other Structures)"      # 6
    ]
    
    # 初始化映射字典
    fine_to_big = {}
    big_to_fine = {i: [] for i in range(len(big_class_names))}
    
    # 脑室系统
    ventricular_classes = [1, 2, 9, 10, 14, 19, 34]
    for cls in ventricular_classes:
        fine_to_big[cls] = 0
        big_to_fine[0].append(cls)
    
    # 白质
    white_matter_classes = [
        3, 22, 96, 97, 98, 99, 100,  # 白质区域
        22, 23, 24, 25, 26,  # 胼胝体
        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 101, 102  # 各脑区白质
    ]
    for cls in white_matter_classes:
        if cls not in fine_to_big:  # 避免重复添加
            fine_to_big[cls] = 1
            big_to_fine[1].append(cls)
    
    # 灰质-皮层
    cortical_gray_matter_classes = [
        4, 23,  # 小脑皮层等
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62  # 大脑皮层区域
    ]
    for cls in cortical_gray_matter_classes:
        if cls not in fine_to_big:  # 避免重复添加
            fine_to_big[cls] = 2
            big_to_fine[2].append(cls)
    
    # 深部灰质核团
    deep_gray_nuclei = [5, 6, 7, 8, 15, 16, 17, 27, 24, 25, 26, 27, 30, 31, 32]
    for cls in deep_gray_nuclei:
        if cls not in fine_to_big:  # 避免重复添加
            fine_to_big[cls] = 3
            big_to_fine[3].append(cls)
    
    # 边缘系统
    limbic_system = [12, 13, 28, 29]
    for cls in limbic_system:
        if cls not in fine_to_big:  # 避免重复添加
            fine_to_big[cls] = 4
            big_to_fine[4].append(cls)
    
    # 脑干
    brain_stem = [11]
    for cls in brain_stem:
        if cls not in fine_to_big:  # 避免重复添加
            fine_to_big[cls] = 5
            big_to_fine[5].append(cls)
    
    # 对于未分类的标签，归为"其他结构"
    for cls in range(103):  # 假设有0-102的标签
        if cls not in fine_to_big:
            fine_to_big[cls] = 6
            big_to_fine[6].append(cls)
    
    logger.info("大类映射定义完成")
    for i, name in enumerate(big_class_names):
        logger.info(f"大类 {i} - {name}: {len(big_to_fine[i])}个细分类")
    
    return fine_to_big, big_to_fine, big_class_names

def map_to_big_classes(fine_labels, fine_to_big):
    """
    将细分类标签映射为大类标签
    
    参数:
        fine_labels: 细分类标签
        fine_to_big: 细分类到大类的映射
        
    返回:
        big_labels: 大类标签
    """
    # 使用向量化操作加速
    mapper = np.vectorize(lambda x: fine_to_big.get(x, 6))  # 默认映射到"其他结构"
    big_labels = mapper(fine_labels)
    
    # 打印大类分布
    unique_classes, counts = np.unique(big_labels, return_counts=True)
    logger.info("\n大类标签分布:")
    for cls, count in zip(unique_classes, counts):
        logger.info(f"大类 {cls}: {count} 个样本 ({count/len(big_labels)*100:.2f}%)")
    
    return big_labels

def sample_balanced_data(data, labels, n_per_class=None, max_samples=10000):
    """
    从数据集中采样平衡的子集
    
    参数:
        data: 输入数据
        labels: 标签
        n_per_class: 每个类别的样本数，如果为None则自动计算
        max_samples: 最大总样本数
        
    返回:
        sampled_data: 采样后的数据
        sampled_labels: 采样后的标签
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    
    if n_per_class is None:
        # 计算每类应保留的样本数，确保总数不超过max_samples
        n_per_class = min(np.min(counts), max_samples // n_classes)
    
    logger.info(f"从每个类别中采样{n_per_class}个样本，共{n_classes}个类别")
    
    sampled_indices = []
    for label in unique_labels:
        # 获取当前类别的所有样本索引
        indices = np.where(labels == label)[0]
        # 随机选择n_per_class个样本
        if len(indices) > n_per_class:
            selected = np.random.choice(indices, n_per_class, replace=False)
        else:
            selected = indices
        sampled_indices.extend(selected)
    
    # 打乱顺序
    np.random.shuffle(sampled_indices)
    sampled_data = data[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    logger.info(f"采样后数据集大小: {sampled_data.shape}")
    
    return sampled_data, sampled_labels

def visualize_class_distribution(labels, class_names=None, save_path=None):
    """
    可视化类别分布
    
    参数:
        labels: 类别标签
        class_names: 类别名称
        save_path: 保存路径
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 排序以便更好地可视化
    sort_idx = np.argsort(counts)[::-1]
    sorted_labels = unique_labels[sort_idx]
    sorted_counts = counts[sort_idx]
    
    # 如果类别太多，只显示前20个
    if len(sorted_labels) > 20:
        sorted_labels = sorted_labels[:20]
        sorted_counts = sorted_counts[:20]
        title_suffix = " (Top 20 Classes)"
    else:
        title_suffix = ""
    
    # 准备标签
    if class_names is not None:
        x_labels = [class_names[label] if label < len(class_names) else f"Class {label}" 
                   for label in sorted_labels]
    else:
        x_labels = [f"Class {label}" for label in sorted_labels]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_labels, sorted_counts)
    
    # 添加数值标签
    for bar, count in zip(bars, sorted_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count} ({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', rotation=45 if len(sorted_labels) > 10 else 0)
    
    plt.title(f"Class Distribution{title_suffix}")
    plt.ylabel("Number of Samples")
    plt.xlabel("Class")
    plt.xticks(rotation=90 if len(sorted_labels) > 10 else 0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"类别分布图保存至: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    # 测试数据加载和预处理
    print("数据加载模块测试")
    
    # 示例用法
    # dataset = load_multiclass_data("/path/to/data", subset="val", sample_ratio=0.1)
    # data = dataset["val_samples"]
    # labels = dataset["val_labels"]
    
    # 测试大类映射
    fine_to_big, big_to_fine, big_class_names = define_big_classes()
    print(f"大类数量: {len(big_class_names)}")
    
    # 假数据测试可视化
    np.random.seed(42)
    test_labels = np.random.randint(0, 7, 1000)
    visualize_class_distribution(test_labels, big_class_names, "test_class_dist.png")
    
    print("测试完成")