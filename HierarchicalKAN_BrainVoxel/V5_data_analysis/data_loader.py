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

def load_multiclass_data(data_dir, subset='val', sample_ratio=1.0, verbose=True):
    """
    加载MRI数据集
    
    参数:
        data_dir: 数据目录
        subset: 子集名称 ('train', 'val', 'test', 'all')
        sample_ratio: 采样比例，用于减少计算量
        verbose: 是否输出详细信息
    
    返回:
        dataset: 包含数据和标签的字典
    """
    if verbose:
        logger.info(f"加载{subset}子集数据，采样比例: {sample_ratio:.2f}")
    
    # 确定数据路径
    if subset == 'all':
        subsets = ['train', 'val', 'test']
    else:
        subsets = [subset]
    
    result = {}
    
    for set_name in subsets:
        try:
            # 尝试直接使用numpy加载预处理好的数据
            data_path = os.path.join(data_dir, f"{set_name}_samples.npy")
            labels_path = os.path.join(data_dir, f"{set_name}_labels.npy")
            
            if os.path.exists(data_path) and os.path.exists(labels_path):
                data = np.load(data_path)
                labels = np.load(labels_path)
                
                if verbose:
                    logger.info(f"从NPY文件加载{set_name}数据成功: {data.shape}")
            else:
                # 尝试从原始数据文件夹加载数据
                data_dir_subset = os.path.join(data_dir, set_name)
                if not os.path.exists(data_dir_subset):
                    logger.warning(f"未找到{set_name}数据目录: {data_dir_subset}")
                    continue
                
                # 实现原始数据加载逻辑，这里简化处理
                # 假设与之前提供的代码中的load_multiclass_data_from_dirs类似
                logger.info(f"从原始文件加载{set_name}数据...")
                
                # 此处应该根据实际情况实现数据加载
                # 为简化示例，这里假设数据已预处理好
                raise NotImplementedError("原始数据加载功能尚未实现，请提供预处理好的NPY文件")
            
            # 如果需要采样以减少计算量
            if sample_ratio < 1.0:
                n_samples = int(len(data) * sample_ratio)
                indices = np.random.choice(len(data), n_samples, replace=False)
                data = data[indices]
                labels = labels[indices]
                if verbose:
                    logger.info(f"采样后数据大小: {data.shape}")
            
            # 存储加载的数据
            result[f"{set_name}_samples"] = data
            result[f"{set_name}_labels"] = labels
            
            if verbose:
                logger.info(f"{set_name}数据加载完成: {data.shape}, 标签: {labels.shape}")
                logger.info(f"数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
                unique_labels = np.unique(labels)
                logger.info(f"标签类别数: {len(unique_labels)}")
        
        except Exception as e:
            logger.error(f"加载{set_name}数据时出错: {str(e)}")
    
    # 添加特征组信息
    result['feature_groups'] = FEATURE_GROUPS
    
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