#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块，负责从文件系统加载脑体素数据
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import sys
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import NUM_CLASS, DATA_DIRS
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_multiclass_data_from_dirs(data_dirs=None, subset='val', verbose=True):
    """
    加载多类别脑体素数据
    
    参数:
        data_dirs: 包含训练、测试和验证数据目录的字典，默认使用config中的DATA_DIRS
        subset: 指定要加载哪个子集，可选值为'train', 'test', 'val', 'all'
        verbose: 是否打印详细信息
        
    返回:
        包含处理后数据的字典
    """
    if data_dirs is None:
        data_dirs = DATA_DIRS
    
    if verbose:
        logger.info(f"从目录加载{subset}数据集...")
        logger.info(f"训练集目录: {data_dirs['train_dir']}")
        logger.info(f"测试集目录: {data_dirs['test_dir']}")
        logger.info(f"验证集目录: {data_dirs['val_dir']}")
    
    # 根据subset参数确定要加载的子集
    dirs_to_load = {}
    if subset == 'all' or subset == 'train':
        dirs_to_load['train'] = data_dirs['train_dir']
    if subset == 'all' or subset == 'test':
        dirs_to_load['test'] = data_dirs['test_dir']
    if subset == 'all' or subset == 'val':
        dirs_to_load['val'] = data_dirs['val_dir']
    
    # 初始化结果字典
    result = {}
    
    # 加载每个指定的子集
    for set_name, dir_path in dirs_to_load.items():
        if verbose:
            logger.info(f"加载{set_name}数据...")
        
        # 获取目录中的所有体素文件
        files = glob.glob(os.path.join(dir_path, "label_*_count_*_voxels.npy"))
        
        if verbose:
            logger.info(f"找到 {len(files)} 个{set_name}文件")
        
        data_all = []
        labels_all = []
        
        # 处理每个文件
        for file_path in tqdm(files, desc=f"加载{set_name}数据", disable=not verbose):
            # 从文件名提取标签ID
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) >= 4 and parts[0] == 'label':
                try:
                    label_id = int(parts[1])
                    # 确保标签在有效范围内 (0 到 NUM_CLASS-1)
                    if 0 <= label_id < NUM_CLASS:
                        if verbose:
                            logger.debug(f"加载特征数据: {filename}")
                        voxels = np.load(file_path)
                        labels = np.full(len(voxels), label_id)
                        data_all.append(voxels)
                        labels_all.append(labels)
                    else:
                        logger.warning(f"跳过标签 {label_id}，超出范围 [0, {NUM_CLASS-1}]")
                except ValueError:
                    logger.warning(f"无法从 {filename} 提取标签ID")
        
        # 合并所有数据
        if data_all:
            data = np.vstack(data_all)
            labels = np.concatenate(labels_all)
            
            # 将结果添加到返回字典中
            result[f"{set_name}_samples"] = data
            result[f"{set_name}_labels"] = labels
            
            if verbose:
                logger.info(f"{set_name}数据: {data.shape}, 标签: {labels.shape}")
                logger.info(f"{set_name}数据范围: {np.min(data)} 至 {np.max(data)}")
                logger.info(f"{set_name}标签范围: {np.min(labels)} 至 {np.max(labels)}")
        else:
            logger.error(f"没有找到有效的{set_name}数据")
    
    # 如果加载了数据，统计各类别样本数
    if result:
        class_counts = {}
        for set_name in dirs_to_load.keys():
            if f"{set_name}_labels" in result:
                labels = result[f"{set_name}_labels"]
                unique, counts = np.unique(labels, return_counts=True)
                class_counts[set_name] = dict(zip(unique, counts))
        
        result['class_counts'] = class_counts
        
        if verbose:
            logger.info("\n类别分布:")
            # 获取所有出现的类别
            all_classes = set()
            for counts in class_counts.values():
                all_classes.update(counts.keys())
            all_classes = sorted(all_classes)
            
            # 打印表头
            header = ["类别ID"] + list(class_counts.keys()) + ["总计"]
            logger.info("\t".join(header))
            logger.info("-" * 50)
            
            # 打印每个类别的数据
            for class_id in all_classes:
                row = [str(class_id)]
                total = 0
                for set_name in class_counts.keys():
                    count = class_counts[set_name].get(class_id, 0)
                    row.append(str(count))
                    total += count
                row.append(str(total))
                logger.info("\t".join(row))
            
            # 打印总计
            total_row = ["总计"]
            grand_total = 0
            for set_name in class_counts.keys():
                set_total = sum(class_counts[set_name].values())
                total_row.append(str(set_total))
                grand_total += set_total
            total_row.append(str(grand_total))
            logger.info("-" * 50)
            logger.info("\t".join(total_row))
    
    return result

def define_big_classes():
    """
    定义基于神经解剖学的大类标签映射关系
    将102个细分类别映射到7个大类
    
    返回:
        fine_to_big: 细分类别到大类的映射字典
        big_to_fine: 大类到细分类别的映射字典
        big_class_names: 大类名称列表
    """
    logger.info("定义基于神经解剖学的大类标签映射关系")
    
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
    logger.info("大类划分统计:")
    for i, name in enumerate(big_class_names):
        logger.info(f"大类 {i} - {name}: {len(big_to_fine[i])} 个细分类别")
        logger.info(f"    包含的细分类别: {big_to_fine[i]}")
    
    return fine_to_big, big_to_fine, big_class_names

def map_to_big_classes(fine_labels, fine_to_big):
    """
    将细分类别标签映射为大类标签
    
    参数:
        fine_labels (array): 细分类别标签
        fine_to_big (dict): 细分类别到大类的映射字典
    
    返回:
        big_labels: 大类标签
    """
    # 使用numpy的vectorize功能提高效率
    mapper = np.vectorize(lambda x: fine_to_big.get(x, 6))  # 默认映射到"其他结构"类别
    big_labels = mapper(fine_labels)
    
    # 打印大类分布
    unique_classes, counts = np.unique(big_labels, return_counts=True)
    logger.info("\n大类标签分布:")
    for cls, count in zip(unique_classes, counts):
        logger.info(f"大类 {cls}: {count} 个样本")
    
    return big_labels

if __name__ == "__main__":
    # 测试数据加载
    dataset = load_multiclass_data_from_dirs(subset='val')
    print("数据加载完成")
    
    # 测试大类定义和映射
    fine_to_big, big_to_fine, big_class_names = define_big_classes()
    val_big_labels = map_to_big_classes(dataset['val_labels'], fine_to_big)
    print(f"细分类标签数量: {len(dataset['val_labels'])}")
    print(f"大类标签数量: {len(val_big_labels)}")