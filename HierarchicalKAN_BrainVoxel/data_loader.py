#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块 - 负责数据的加载、预处理和增强
"""

import os
import numpy as np
import glob
from sklearn.decomposition import PCA
from utils import analyze_pca_variance

def load_multiclass_data_from_dirs(data_dirs, apply_pca=True, n_components=24, norm=True, logger=None):
    """
    加载多类别脑体素数据，使用与原始BrainVoxelSampler类似的逻辑
    
    参数:
        data_dirs: 包含训练、测试和验证数据目录的字典
        apply_pca: 是否应用PCA
        n_components: PCA组件数
        norm: 是否归一化
        logger: 日志记录器
    
    返回:
        包含处理后数据的字典
    """
    from config import NUM_CLASS
    
    msg = f"从目录加载多类别数据..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"训练集目录: {data_dirs['train_dir']}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"测试集目录: {data_dirs['test_dir']}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"验证集目录: {data_dirs['val_dir']}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 加载所有标签的数据
    train_data_all = []
    train_labels_all = []
    test_data_all = []
    test_labels_all = []
    val_data_all = []
    val_labels_all = []
    
    # 获取目录中的所有体素文件
    train_files = glob.glob(os.path.join(data_dirs['train_dir'], "label_*_count_*_voxels.npy"))
    test_files = glob.glob(os.path.join(data_dirs['test_dir'], "label_*_count_*_voxels.npy"))
    val_files = glob.glob(os.path.join(data_dirs['val_dir'], "label_*_count_*_voxels.npy"))
    
    msg = f"找到 {len(train_files)} 个训练文件, {len(test_files)} 个测试文件, {len(val_files)} 个验证文件"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 处理训练集
    for file_path in train_files:
        # 从文件名提取标签ID
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'label':
            try:
                label_id = int(parts[1])
                # 确保标签在有效范围内 (0 到 NUM_CLASS-1)
                if 0 <= label_id < NUM_CLASS:
                    msg = f"加载特征数据: {filename}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    voxels = np.load(file_path)
                    labels = np.full(len(voxels), label_id)
                    train_data_all.append(voxels)
                    train_labels_all.append(labels)
                else:
                    msg = f"警告: 跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
            except ValueError:
                msg = f"警告: 无法从 {filename} 提取标签ID"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
    
    # 处理测试集
    for file_path in test_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'label':
            try:
                label_id = int(parts[1])
                if 0 <= label_id < NUM_CLASS:
                    msg = f"加载特征数据: {filename}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    voxels = np.load(file_path)
                    labels = np.full(len(voxels), label_id)
                    test_data_all.append(voxels)
                    test_labels_all.append(labels)
                else:
                    msg = f"警告: 跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
            except ValueError:
                msg = f"警告: 无法从 {filename} 提取标签ID"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
    
    # 处理验证集
    for file_path in val_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'label':
            try:
                label_id = int(parts[1])
                if 0 <= label_id < NUM_CLASS:
                    msg = f"加载特征数据: {filename}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                        
                    voxels = np.load(file_path)
                    labels = np.full(len(voxels), label_id)
                    val_data_all.append(voxels)
                    val_labels_all.append(labels)
                else:
                    msg = f"警告: 跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
            except ValueError:
                msg = f"警告: 无法从 {filename} 提取标签ID"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
    
    # 合并所有数据
    if train_data_all:
        train_data = np.vstack(train_data_all)
        train_labels = np.concatenate(train_labels_all)
    else:
        msg = "没有找到有效的训练数据"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    if test_data_all:
        test_data = np.vstack(test_data_all)
        test_labels = np.concatenate(test_labels_all)
    else:
        msg = "没有找到有效的测试数据"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    if val_data_all:
        val_data = np.vstack(val_data_all)
        val_labels = np.concatenate(val_labels_all)
    else:
        msg = "没有找到有效的验证数据"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    
    # 打印数据范围
    msg = f"训练集数据范围: {np.min(train_data)} 至 {np.max(train_data)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"测试集数据范围: {np.min(test_data)} 至 {np.max(test_data)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"验证集数据范围: {np.min(val_data)} 至 {np.max(val_data)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 检查标签范围
    msg = f"训练集标签范围: {np.min(train_labels)} 至 {np.max(train_labels)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"测试集标签范围: {np.min(test_labels)} 至 {np.max(test_labels)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"验证集标签范围: {np.min(val_labels)} 至 {np.max(val_labels)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # 应用PCA
    if apply_pca:
        # 合并所有数据用于PCA拟合
        all_data = np.vstack([train_data, test_data, val_data])
        
        if n_components > 0:
            pca_model = PCA(n_components=n_components)
            pca_model.fit(all_data)
        else:
            # 自动选择PCA组件数（从utils模块导入）
            from config import SAVE_PATH
            n_components, _, _ = analyze_pca_variance(all_data, plot=True, save_path=SAVE_PATH, logger=logger)
            pca_model = PCA(n_components=n_components)
            pca_model.fit(all_data)
            
        # 应用PCA变换
        train_data = pca_model.transform(train_data)
        test_data = pca_model.transform(test_data)
        val_data = pca_model.transform(val_data)
        
        # 如果需要，进行归一化
        if norm:
            # 基于所有样本计算归一化参数
            all_transformed = np.vstack([train_data, test_data, val_data])
            mins = np.min(all_transformed, axis=0)
            maxs = np.max(all_transformed, axis=0)
            ranges = maxs - mins + 1e-10  # 避免除零
            
            # 应用归一化
            train_data = (train_data - mins) / ranges
            test_data = (test_data - mins) / ranges
            val_data = (val_data - mins) / ranges
        
        feature_dim = train_data.shape[1]
        msg = f"PCA后的特征维度: {feature_dim}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    else:
        feature_dim = train_data.shape[1]
        pca_model = None
        msg = f"未应用PCA，特征维度: {feature_dim}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 统计各类别样本数
    class_counts = np.zeros((NUM_CLASS, 3), dtype=int)  # [训练集, 测试集, 验证集]
    for i in range(NUM_CLASS):
        class_counts[i, 0] = np.sum(train_labels == i)
        class_counts[i, 1] = np.sum(test_labels == i)
        class_counts[i, 2] = np.sum(val_labels == i)
    
    # 打印类别分布
    msg = "\n类别分布:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"{'类别ID':^10}{'训练集':^10}{'测试集':^10}{'验证集':^10}{'总计':^10}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = "-" * 50
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for i in range(NUM_CLASS):
        if np.sum(class_counts[i]) > 0:  # 只打印有样本的类别
            total = np.sum(class_counts[i])
            msg = f"{i:^10}{class_counts[i, 0]:^10}{class_counts[i, 1]:^10}{class_counts[i, 2]:^10}{total:^10}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    
    total_train = len(train_labels)
    total_test = len(test_labels)
    total_val = len(val_labels)
    total_all = total_train + total_test + total_val
    
    msg = "-" * 50
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    msg = f"{'总计':^10}{total_train:^10}{total_test:^10}{total_val:^10}{total_all:^10}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return {
        'train_samples': train_data,
        'train_labels': train_labels,
        'test_samples': test_data,
        'test_labels': test_labels,
        'val_samples': val_data,
        'val_labels': val_labels,
        'feature_dim': feature_dim,
        'pca_model': pca_model,
        'class_counts': class_counts
    }


def define_big_classes(logger=None):
    """
    定义基于神经解剖学的大类标签映射关系
    将102个细分类别映射到7个大类

    返回:
        fine_to_big: 细分类别到大类的映射字典
        big_to_fine: 大类到细分类别的映射字典
        big_class_names: 大类名称列表
    """
    from config import NUM_CLASS
    
    # 定义大类名称
    big_class_names = [
        "脑室系统 (Ventricular System)",  # 0
        "白质 (White Matter)",            # 1
        "灰质-皮层 (Cortical Gray Matter)", # 2
        "深部灰质核团 (Deep Gray Nuclei)",  # 3
        "边缘系统 (Limbic System)",        # 4
        "脑干 (Brain Stem)",              # 5
        "其他结构 (Other Structures)"      # 6
    ]
    
    # 初始化映射字典
    fine_to_big = {}
    big_to_fine = {i: [] for i in range(len(big_class_names))}
    
    # 1. 脑室系统 (Ventricular System)
    ventricular_system_classes = [1, 2, 9, 10, 14, 19, 34]
    for cls in ventricular_system_classes:
        fine_to_big[cls] = 0
        big_to_fine[0].append(cls)
    
    # 2. 白质 (White Matter)
    white_matter_classes = [
        3, 22, 96, 97, 98, 99, 100, 
        # 胼胝体
        22, 23, 24, 25, 26,
        # 各脑区白质标签(wm-前缀)
        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 101, 102
    ]
    for cls in white_matter_classes:
        fine_to_big[cls] = 1
        if cls not in big_to_fine[1]:
            big_to_fine[1].append(cls)
    
    # 3. 灰质-皮层 (Cortical Gray Matter)
    cortical_gray_matter_classes = [
        4, 23,  # 小脑皮层
        # 大脑皮层标签(ctx-前缀)
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
    ]
    for cls in cortical_gray_matter_classes:
        fine_to_big[cls] = 2
        if cls not in big_to_fine[2]:
            big_to_fine[2].append(cls)
    
    # 4. 深部灰质核团 (Deep Gray Nuclei)
    deep_gray_nuclei_classes = [
        5, 6, 7, 8, 15, 16, 17, 27,  # 丘脑、尾状核、壳核等
        24, 25, 26, 27, 30, 31, 32  # 右侧对应结构
    ]
    for cls in deep_gray_nuclei_classes:
        fine_to_big[cls] = 3
        if cls not in big_to_fine[3]:
            big_to_fine[3].append(cls)
    
    # 5. 边缘系统 (Limbic System)
    limbic_system_classes = [12, 13, 28, 29]  # 海马、杏仁核
    for cls in limbic_system_classes:
        fine_to_big[cls] = 4
        if cls not in big_to_fine[4]:
            big_to_fine[4].append(cls)
    
    # 6. 脑干 (Brain Stem)
    brain_stem_classes = [11]
    for cls in brain_stem_classes:
        fine_to_big[cls] = 5
        big_to_fine[5].append(cls)
    
    # 7. 其他结构 (Other Structures)
    # 包括未包含在上述类别中的所有其他结构
    for cls in range(NUM_CLASS):
        if cls not in fine_to_big:
            fine_to_big[cls] = 6
            big_to_fine[6].append(cls)
    
    # 打印大类统计信息
    msg = "大类划分统计:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for i, name in enumerate(big_class_names):
        msg = f"大类 {i} - {name}: {len(big_to_fine[i])} 个细分类别"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        msg = f"    包含的细分类别: {big_to_fine[i]}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return fine_to_big, big_to_fine, big_class_names


def map_to_big_classes(fine_labels, fine_to_big, logger=None):
    """
    将细分类别标签映射为大类标签
    
    参数:
        fine_labels (array): 细分类别标签
        fine_to_big (dict): 细分类别到大类的映射字典
        logger: 日志记录器
    
    返回:
        big_labels: 大类标签
    """
    # 使用numpy的vectorize功能提高效率
    mapper = np.vectorize(lambda x: fine_to_big.get(x, 6))  # 默认映射到"其他结构"类别
    big_labels = mapper(fine_labels)
    
    # 打印大类分布
    unique_classes, counts = np.unique(big_labels, return_counts=True)
    
    msg = "\n大类标签分布:"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    for cls, count in zip(unique_classes, counts):
        msg = f"大类 {cls}: {count} 个样本"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return big_labels