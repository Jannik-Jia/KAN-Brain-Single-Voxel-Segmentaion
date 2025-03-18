#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型工具模块，提供模型保存、加载和辅助功能
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODELS_DIR, RESULTS_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def save_model(model, model_name, extra_info=None):
    """
    保存模型和相关信息
    
    参数:
        model: 要保存的模型对象
        model_name: 模型名称
        extra_info: 额外信息字典
        
    返回:
        model_path: 保存的模型文件路径
    """
    # 确保模型目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 创建带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    model_path = os.path.join(MODELS_DIR, filename)
    
    # 准备要保存的内容
    save_dict = {
        'model': model,
        'timestamp': timestamp,
        'name': model_name
    }
    
    # 添加额外信息
    if extra_info:
        save_dict.update(extra_info)
    
    # 保存模型
    with open(model_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    logger.info(f"模型已保存至: {model_path}")
    
    return model_path

def load_model(model_path):
    """
    加载模型和相关信息
    
    参数:
        model_path: 模型文件路径
        
    返回:
        model: 加载的模型对象
        info: 相关信息字典
    """
    # 加载模型
    with open(model_path, 'rb') as f:
        saved_dict = pickle.load(f)
    
    # 提取模型和信息
    model = saved_dict.pop('model')
    info = saved_dict
    
    logger.info(f"已加载模型: {model_path}")
    logger.info(f"模型信息: {info}")
    
    return model, info

def save_results(results, result_name, format='pkl'):
    """
    保存分析结果
    
    参数:
        results: 要保存的结果数据
        result_name: 结果名称
        format: 保存格式，'pkl' 或 'json'
        
    返回:
        result_path: 保存的结果文件路径
    """
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 创建带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format.lower() == 'pkl':
        filename = f"{result_name}_{timestamp}.pkl"
        result_path = os.path.join(RESULTS_DIR, filename)
        
        # 保存为pickle格式
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format.lower() == 'json':
        import json
        filename = f"{result_name}_{timestamp}.json"
        result_path = os.path.join(RESULTS_DIR, filename)
        
        # 保存为JSON格式
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    else:
        raise ValueError(f"不支持的保存格式: {format}")
    
    logger.info(f"结果已保存至: {result_path}")
    
    return result_path

def load_results(result_path):
    """
    加载分析结果
    
    参数:
        result_path: 结果文件路径
        
    返回:
        results: 加载的结果数据
    """
    # 根据文件扩展名确定加载方式
    if result_path.endswith('.pkl'):
        # 加载pickle格式
        with open(result_path, 'rb') as f:
            results = pickle.load(f)
    
    elif result_path.endswith('.json'):
        # 加载JSON格式
        import json
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    else:
        raise ValueError(f"不支持的文件格式: {result_path}")
    
    logger.info(f"已加载结果: {result_path}")
    
    return results

def generate_bigclass_mapping(clustering_result, original_labels, big_class_names=None, mapping_type='majority'):
    """
    生成聚类结果到大类的映射关系
    
    参数:
        clustering_result: 聚类结果，包含标签数组
        original_labels: 原始标签
        big_class_names: 大类名称列表
        mapping_type: 映射类型，'majority'或'optimal'
        
    返回:
        mapping: 聚类到大类的映射字典
        new_labels: 基于聚类的新标签
    """
    # 获取聚类标签
    cluster_labels = clustering_result['labels']
    unique_clusters = np.unique(cluster_labels)
    unique_originals = np.unique(original_labels)
    
    # 计算聚类与原始标签的混淆矩阵
    matrix = np.zeros((len(unique_originals), len(unique_clusters)))
    for i, orig in enumerate(unique_originals):
        for j, cluster in enumerate(unique_clusters):
            matrix[i, j] = np.sum((original_labels == orig) & (cluster_labels == cluster))
    
    # 确定映射关系
    if mapping_type == 'majority':
        # 每个聚类映射到最多出现的原始标签
        mapping = {}
        for j, cluster in enumerate(unique_clusters):
            if np.sum(matrix[:, j]) > 0:  # 避免空聚类
                best_match = np.argmax(matrix[:, j])
                mapping[int(cluster)] = int(unique_originals[best_match])
    
    elif mapping_type == 'optimal':
        # 使用匈牙利算法找到全局最优映射
        from scipy.optimize import linear_sum_assignment
        cost_matrix = -matrix.copy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            # 原始标签 -> 聚类标签
            mapping[int(unique_clusters[j])] = int(unique_originals[i])
    
    else:
        raise ValueError(f"不支持的映射类型: {mapping_type}")
    
    # 生成新标签
    new_labels = np.array([mapping.get(int(label), -1) for label in cluster_labels])
    
    # 打印映射信息
    logger.info(f"聚类到大类的映射关系 (映射类型: {mapping_type}):")
    
    for cluster, original in mapping.items():
        name = big_class_names[original] if big_class_names and original < len(big_class_names) else f"Class {original}"
        count = np.sum(cluster_labels == cluster)
        percentage = count / len(cluster_labels) * 100
        logger.info(f"  聚类 {cluster} -> {name} (包含 {count} 样本, {percentage:.1f}%)")
    
    # 计算映射一致性
    matched_count = np.sum(new_labels == original_labels)
    consistency = matched_count / len(original_labels)
    logger.info(f"映射一致性: {consistency:.4f} ({matched_count}/{len(original_labels)})")
    
    return mapping, new_labels

def get_fine_to_big_mapping(big_to_fine_dict):
    """
    将大类到细分类的映射转换为细分类到大类的映射
    
    参数:
        big_to_fine_dict: 大类到细分类的映射字典
        
    返回:
        fine_to_big: 细分类到大类的映射字典
    """
    fine_to_big = {}
    
    for big_class, fine_classes in big_to_fine_dict.items():
        for fine_class in fine_classes:
            fine_to_big[fine_class] = big_class
    
    return fine_to_big

def visualize_mapping_changes(original_mapping, new_mapping, big_class_names=None, save_path=None):
    """
    可视化映射关系的变化
    
    参数:
        original_mapping: 原始细分类到大类的映射字典
        new_mapping: 新的细分类到大类的映射字典
        big_class_names: 大类名称列表
        save_path: 图表保存路径
    """
    # 获取所有细分类
    all_fine_classes = set(original_mapping.keys()) | set(new_mapping.keys())
    
    # 找出发生变化的细分类
    changed_classes = []
    for fine_class in all_fine_classes:
        if fine_class in original_mapping and fine_class in new_mapping:
            if original_mapping[fine_class] != new_mapping[fine_class]:
                changed_classes.append(fine_class)
    
    if not changed_classes:
        logger.info("没有细分类的归属发生变化")
        return
    
    # 准备绘图数据
    changes = []
    for fine_class in changed_classes:
        old_big = original_mapping.get(fine_class, -1)
        new_big = new_mapping.get(fine_class, -1)
        old_name = big_class_names[old_big] if big_class_names and old_big < len(big_class_names) else f"Class {old_big}"
        new_name = big_class_names[new_big] if big_class_names and new_big < len(big_class_names) else f"Class {new_big}"
        changes.append((fine_class, old_big, new_big, old_name, new_name))
    
    # 绘制变化图
    plt.figure(figsize=(12, len(changes) * 0.5 + 2))
    
    # 坐标
    y_pos = np.arange(len(changes))
    
    # 绘制从旧到新的连线
    for i, (fine_class, old_big, new_big, old_name, new_name) in enumerate(changes):
        plt.plot([0, 1], [i, i], 'o-', linewidth=1.5, 
                markersize=8, alpha=0.7)
        plt.text(-0.1, i, old_name, ha='right', va='center')
        plt.text(1.1, i, new_name, ha='left', va='center')
        plt.text(0.5, i, f"Class {fine_class}", ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.ylim(-1, len(changes))
    plt.xlim(-0.5, 1.5)
    plt.axis('off')
    plt.title('Changes in Fine-to-Big Class Mapping')
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"映射变化图表已保存至: {save_path}")
    
    plt.close()
    
    # 打印变化信息
    logger.info(f"共有 {len(changed_classes)} 个细分类的归属发生变化:")
    for fine_class, old_big, new_big, old_name, new_name in changes:
        logger.info(f"  细分类 {fine_class}: {old_name} -> {new_name}")

if __name__ == "__main__":
    # 测试模型工具
    # 模拟数据
    class DummyModel:
        def __init__(self, name):
            self.name = name
        
        def predict(self, X):
            return np.zeros(len(X))
    
    model = DummyModel("测试模型")
    
    # 保存模型
    model_path = save_model(model, "dummy_model", {"param1": 10, "param2": "test"})
    
    # 加载模型
    loaded_model, info = load_model(model_path)
    print(f"加载的模型名称: {loaded_model.name}")
    print(f"加载的模型信息: {info}")
    
    # 测试结果保存和加载
    results = {"accuracy": 0.85, "f1": 0.82}
    result_path = save_results(results, "test_results", format='json')
    loaded_results = load_results(result_path)
    print(f"加载的结果: {loaded_results}")
    
    print("模型工具测试完成")