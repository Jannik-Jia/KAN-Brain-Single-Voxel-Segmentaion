#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本数据分析模块
用于分析MRI数据的基本统计特性
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import logging
from scipy import stats
from tqdm import tqdm

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basic_analysis')

# 检测是否可以使用GPU
try:
    import cupy as cp
    import cudf
    HAS_GPU = True
    logger.info("GPU加速可用")
except ImportError:
    HAS_GPU = False
    logger.warning("未检测到GPU加速库，将使用CPU进行计算")

def analyze_basic_stats(data, feature_groups=None, use_gpu=False, save_dir=None):
    """
    分析数据的基本统计特性
    
    参数:
        data: 输入数据
        feature_groups: 特征组字典
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
    
    返回:
        stats_dict: 统计结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    stats_dict = {}
    
    for group_name, indices in tqdm(feature_groups.items(), desc="分析基本统计量"):
        logger.info(f"分析特征组: {group_name} ({len(indices)}个特征)")
        
        # 提取特征组数据
        group_data = data[:, indices]
        
        # 计算基本统计量
        # 使用GPU加速（如果可用）
        if use_gpu and HAS_GPU:
            try:
                # 转移到GPU
                group_data_gpu = cp.asarray(group_data)
                
                # 计算统计量
                mins = cp.min(group_data_gpu, axis=0).get()
                maxs = cp.max(group_data_gpu, axis=0).get()
                means = cp.mean(group_data_gpu, axis=0).get()
                stds = cp.std(group_data_gpu, axis=0).get()
                medians = cp.median(group_data_gpu, axis=0).get()
                
                # 计算四分位数
                q1 = cp.percentile(group_data_gpu, 25, axis=0).get()
                q3 = cp.percentile(group_data_gpu, 75, axis=0).get()
                
                # 计算零值比例
                zero_counts = cp.sum(cp.abs(group_data_gpu) < 1e-10, axis=0).get()
                zero_ratios = zero_counts / group_data.shape[0]
                
                # 释放GPU内存
                del group_data_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                logger.error(f"GPU处理失败: {str(e)}")
                logger.info("回退到CPU处理...")
                
                # 计算统计量（CPU）
                mins = np.min(group_data, axis=0)
                maxs = np.max(group_data, axis=0)
                means = np.mean(group_data, axis=0)
                stds = np.std(group_data, axis=0)
                medians = np.median(group_data, axis=0)
                
                # 计算四分位数
                q1 = np.percentile(group_data, 25, axis=0)
                q3 = np.percentile(group_data, 75, axis=0)
                
                # 计算零值比例
                zero_counts = np.sum(np.abs(group_data) < 1e-10, axis=0)
                zero_ratios = zero_counts / group_data.shape[0]
        else:
            # 计算统计量（CPU）
            mins = np.min(group_data, axis=0)
            maxs = np.max(group_data, axis=0)
            means = np.mean(group_data, axis=0)
            stds = np.std(group_data, axis=0)
            medians = np.median(group_data, axis=0)
            
            # 计算四分位数
            q1 = np.percentile(group_data, 25, axis=0)
            q3 = np.percentile(group_data, 75, axis=0)
            
            # 计算零值比例
            zero_counts = np.sum(np.abs(group_data) < 1e-10, axis=0)
            zero_ratios = zero_counts / group_data.shape[0]
        
        # 检查是否存在异常值
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 计算每个特征的异常值数量
        outlier_counts = np.zeros(len(indices))
        for i in range(len(indices)):
            outlier_counts[i] = np.sum((group_data[:, i] < lower_bound[i]) | (group_data[:, i] > upper_bound[i]))
        
        outlier_ratios = outlier_counts / group_data.shape[0]
        
        # 计算数据稀疏性
        # 稀疏性定义为零值比例
        sparsity = np.mean(zero_ratios)
        
        # 检查分布偏斜性
        skewness = stats.skew(group_data, axis=0)
        
        # 计算峰度
        kurtosis = stats.kurtosis(group_data, axis=0)
        
        # 保存结果
        stats_dict[group_name] = {
            'min': mins,
            'max': maxs,
            'mean': means,
            'std': stds,
            'median': medians,
            'q1': q1,
            'q3': q3,
            'zero_ratio': zero_ratios,
            'outlier_ratio': outlier_ratios,
            'sparsity': sparsity,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # 打印统计摘要
        logger.info(f"特征组 {group_name} 统计摘要:")
        logger.info(f"  数据范围: [{np.min(mins):.4f}, {np.max(maxs):.4f}]")
        logger.info(f"  均值范围: [{np.min(means):.4f}, {np.max(means):.4f}]")
        logger.info(f"  标准差范围: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
        logger.info(f"  零值比例: {sparsity:.4f}")
        logger.info(f"  异常值比例: {np.mean(outlier_ratios):.4f}")
        logger.info(f"  偏度范围: [{np.min(skewness):.4f}, {np.max(skewness):.4f}]")
        logger.info(f"  峰度范围: [{np.min(kurtosis):.4f}, {np.max(kurtosis):.4f}]")
        
        # 可视化分析结果
        if save_dir:
            # 绘制分布直方图
            plt.figure(figsize=(16, 12))
            
            # 选择一些有代表性的特征进行可视化
            n_features = min(16, len(indices))
            sample_indices = np.linspace(0, len(indices)-1, n_features, dtype=int)
            
            for i, idx in enumerate(sample_indices):
                plt.subplot(4, 4, i+1)
                sns.histplot(group_data[:, idx], kde=True)
                plt.title(f"Feature {indices[idx]}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_distributions.png"), dpi=300)
            plt.close()
            
            # 绘制箱线图
            plt.figure(figsize=(16, 8))
            sns.boxplot(data=group_data[:, sample_indices])
            plt.title(f"{group_name} Feature Boxplot")
            plt.xlabel("Feature Index")
            plt.ylabel("Value")
            plt.xticks(range(len(sample_indices)), [str(indices[i]) for i in sample_indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_boxplot.png"), dpi=300)
            plt.close()
            
            # 绘制零值比例和异常值比例
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(zero_ratios, bins=20)
            plt.title(f"{group_name} Zero Value Ratio Distribution")
            plt.xlabel("Zero Value Ratio")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 2, 2)
            plt.hist(outlier_ratios, bins=20)
            plt.title(f"{group_name} Outlier Ratio Distribution")
            plt.xlabel("Outlier Ratio")
            plt.ylabel("Frequency")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_ratios.png"), dpi=300)
            plt.close()
            
            # 保存统计结果到CSV
            stats_df = pd.DataFrame({
                'feature_idx': indices,
                'min': mins,
                'max': maxs,
                'mean': means,
                'std': stds,
                'median': medians,
                'q1': q1,
                'q3': q3,
                'zero_ratio': zero_ratios,
                'outlier_ratio': outlier_ratios,
                'skewness': skewness,
                'kurtosis': kurtosis
            })
            stats_df.to_csv(os.path.join(save_dir, f"{group_name}_stats.csv"), index=False)
            logger.info(f"统计结果已保存至 {os.path.join(save_dir, f'{group_name}_stats.csv')}")
    
    return stats_dict

# def analyze_feature_correlation(data, feature_groups=None, use_gpu=False, save_dir=None, 
#                               max_features_per_group=100):
#     """
#     分析特征之间的相关性
    
#     参数:
#         data: 输入数据
#         feature_groups: 特征组字典
#         use_gpu: 是否使用GPU加速
#         save_dir: 结果保存目录
#         max_features_per_group: 每个特征组最多分析的特征数量
    
#     返回:
#         corr_dict: 相关性分析结果字典
#     """
#     if feature_groups is None:
#         feature_groups = {
#             'all_features': list(range(data.shape[1]))
#         }
    
#     if save_dir and not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     corr_dict = {}
    
#     for group_name, indices in tqdm(feature_groups.items(), desc="分析特征相关性"):
#         logger.info(f"分析特征组 {group_name} 内部相关性")
        
#         # 如果特征太多，随机采样一部分
#         if len(indices) > max_features_per_group:
#             logger.info(f"特征数量 ({len(indices)}) 超过限制 ({max_features_per_group})，随机采样")
#             sampled_indices = np.random.choice(indices, max_features_per_group, replace=False)
#             sampled_indices.sort()  # 保持索引有序
#         else:
#             sampled_indices = indices
        
#         # 提取特征组数据
#         group_data = data[:, sampled_indices]
        
#         # 计算相关性矩阵
#         if use_gpu and HAS_GPU:
#             try:
#                 # 使用cuDF计算相关性
#                 import cudf
#                 df_gpu = cudf.DataFrame(group_data)
#                 corr_matrix = df_gpu.corr().values.get()
                
#                 # 释放GPU内存
#                 del df_gpu
#                 cp.get_default_memory_pool().free_all_blocks()
                
#             except Exception as e:
#                 logger.error(f"GPU处理失败: {str(e)}")
#                 logger.info("回退到CPU处理...")
#                 corr_matrix = np.corrcoef(group_data.T)
#         else:
#             corr_matrix = np.corrcoef(group_data.T)
        
#         # 分析相关性统计
#         # 取相关性矩阵的上三角部分
#         upper_triangle = np.triu(corr_matrix, k=1)
#         mask = upper_triangle != 0
#         if np.any(mask):
#             abs_corrs = np.abs(upper_triangle[mask])
#             high_corr_ratio = np.mean(abs_corrs > 0.7)
#             mean_abs_corr = np.mean(abs_corrs)
#         else:
#             high_corr_ratio = 0
#             mean_abs_corr = 0
        
#         logger.info(f"  平均绝对相关系数: {mean_abs_corr:.4f}")
#         logger.info(f"  高度相关 (|r| > 0.7) 特征占比: {high_corr_ratio:.4f}")
        
#         # 保存相关性分析结果
#         corr_dict[group_name] = {
#             'corr_matrix': corr_matrix,
#             'feature_indices': sampled_indices,
#             'mean_abs_corr': mean_abs_corr,
#             'high_corr_ratio': high_corr_ratio
#         }
        
#         # 可视化相关性矩阵
#         if save_dir:
#             plt.figure(figsize=(12, 10))
#             mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#             sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", 
#                        vmin=-1, vmax=1, center=0, annot=False, 
#                        square=True, linewidths=.5)
#             plt.title(f"{group_name} Feature Correlation Matrix")
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, f"{group_name}_correlation.png"), dpi=300)
#             plt.close()
            
#             # 保存相关性统计
#             with open(os.path.join(save_dir, f"{group_name}_correlation_stats.txt"), 'w') as f:
#                 f.write(f"特征组: {group_name}\n")
#                 f.write(f"特征数量: {len(sampled_indices)}\n")
#                 f.write(f"平均绝对相关系数: {mean_abs_corr:.4f}\n")
#                 f.write(f"高度相关 (|r| > 0.7) 特征占比: {high_corr_ratio:.4f}\n")
                
#                 # 找出高度相关的特征对
#                 highly_correlated = []
#                 for i in range(corr_matrix.shape[0]):
#                     for j in range(i+1, corr_matrix.shape[1]):
#                         if abs(corr_matrix[i, j]) > 0.7:
#                             highly_correlated.append((
#                                 sampled_indices[i], 
#                                 sampled_indices[j], 
#                                 corr_matrix[i, j]
#                             ))
                
#                 # 排序并输出
#                 highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
#                 f.write("\n高度相关的特征对 (top 20):\n")
#                 for i, (idx1, idx2, corr) in enumerate(highly_correlated[:20]):
#                     f.write(f"{i+1}. 特征 {idx1} - 特征 {idx2}: {corr:.4f}\n")
    
#     # 分析特征组间相关性
#     if len(feature_groups) > 1:
#         logger.info("分析特征组间相关性")
        
#         # 计算每个特征组的代表性特征
#         group_representatives = {}
#         for group_name, indices in feature_groups.items():
#             # 使用主成分分析提取代表性特征
#             if len(indices) > 10:
#                 # 随机选择10个特征
#                 group_representatives[group_name] = np.random.choice(indices, 10, replace=False)
#             else:
#                 group_representatives[group_name] = indices
        
#         # 提取所有代表性特征
#         all_repr_indices = []
#         group_indices = []  # 记录每个特征属于哪个组
#         for group_id, (group_name, indices) in enumerate(group_representatives.items()):
#             all_repr_indices.extend(indices)
#             group_indices.extend([group_id] * len(indices))
        
#         all_repr_data = data[:, all_repr_indices]
        
#         # 计算相关性矩阵
#         if use_gpu and HAS_GPU:
#             try:
#                 import cudf
#                 df_gpu = cudf.DataFrame(all_repr_data)
#                 cross_corr_matrix = df_gpu.corr().values.get()
                
#                 # 释放GPU内存
#                 del df_gpu
#                 cp.get_default_memory_pool().free_all_blocks()
#             except Exception as e:
#                 logger.error(f"GPU处理失败: {str(e)}")
#                 logger.info("回退到CPU处理...")
#                 cross_corr_matrix = np.corrcoef(all_repr_data.T)
#         else:
#             cross_corr_matrix = np.corrcoef(all_repr_data.T)
        
#         # 计算特征组间平均相关性
#         group_names = list(group_representatives.keys())
#         n_groups = len(group_names)
#         between_group_corr = np.zeros((n_groups, n_groups))
        
#         for i in range(n_groups):
#             for j in range(n_groups):
#                 if i == j:
#                     continue
                
#                 # 获取组i和组j的索引
#                 i_indices = [k for k, g in enumerate(group_indices) if g == i]
#                 j_indices = [k for k, g in enumerate(group_indices) if g == j]
                
#                 # 提取组间相关系数
#                 cross_corrs = np.abs(cross_corr_matrix[np.ix_(i_indices, j_indices)])
#                 between_group_corr[i, j] = np.mean(cross_corrs)
        
#         logger.info("特征组间平均绝对相关系数:")
#         for i in range(n_groups):
#             for j in range(i+1, n_groups):
#                 logger.info(f"  {group_names[i]} - {group_names[j]}: {between_group_corr[i, j]:.4f}")
        
#         # 可视化特征组间相关性
#         if save_dir:
#             plt.figure(figsize=(10, 8))
#             sns.heatmap(between_group_corr, annot=True, cmap="YlGnBu", 
#                        xticklabels=group_names, yticklabels=group_names)
#             plt.title("Between-Group Feature Correlation")
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, "between_group_correlation.png"), dpi=300)
#             plt.close()
            
#             # 保存组间相关性结果
#             np.save(os.path.join(save_dir, "between_group_correlation.npy"), between_group_corr)
            
#             # 创建详细的交叉相关性可视化
#             plt.figure(figsize=(14, 12))
            
#             # 创建组标签
#             group_labels = []
#             for group_id, group_name in enumerate(group_names):
#                 count = len([g for g in group_indices if g == group_id])
#                 group_labels.extend([group_name] * count)
            
#             # 重新排序相关性矩阵，按特征组划分
#             ordered_indices = np.argsort(group_indices)
#             ordered_matrix = cross_corr_matrix[np.ix_(ordered_indices, ordered_indices)]
            
#             # 绘制热图
#             sns.heatmap(ordered_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1)
            
#             # 添加特征组分隔线
#             group_sizes = [len(indices) for indices in group_representatives.values()]
#             cumulative_sizes = np.cumsum(group_sizes)
            
#             for size in cumulative_sizes[:-1]:
#                 plt.axhline(y=size, color='black', linestyle='-', linewidth=1)
#                 plt.axvline(x=size, color='black', linestyle='-', linewidth=1)
            
#             # 添加组标签
#             plt.title("Cross-Feature Correlation Matrix")
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, "cross_feature_correlation.png"), dpi=300)
#             plt.close()
        
#         # 添加组间相关性结果到返回字典
#         corr_dict['between_groups'] = {
#             'corr_matrix': between_group_corr,
#             'group_names': group_names
#         }
    
#     return corr_dict

def analyze_feature_correlation(data, feature_groups=None, use_gpu=False, save_dir=None, 
                              max_features_per_group=100):
    """
    分析特征之间的相关性
    
    参数:
        data: 输入数据
        feature_groups: 特征组字典
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
        max_features_per_group: 每个特征组最多分析的特征数量
    
    返回:
        corr_dict: 相关性分析结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    corr_dict = {}
    
    # 如果使用GPU，一次性将整个数据转移到GPU内存
    if use_gpu and HAS_GPU:
        try:
            data_gpu = cp.asarray(data)
            using_gpu = True
            logger.info("数据已成功转移到GPU")
        except Exception as e:
            logger.error(f"无法将数据转移到GPU: {str(e)}")
            logger.info("将使用CPU进行计算")
            using_gpu = False
    else:
        using_gpu = False
    
    # 计算每个特征组的内部相关性
    for group_name, indices in tqdm(feature_groups.items(), desc="分析特征相关性"):
        logger.info(f"分析特征组 {group_name} 内部相关性")
        
        # 如果特征太多，随机采样一部分
        if len(indices) > max_features_per_group:
            logger.info(f"特征数量 ({len(indices)}) 超过限制 ({max_features_per_group})，随机采样")
            sampled_indices = np.random.choice(indices, max_features_per_group, replace=False)
            sampled_indices.sort()  # 保持索引有序
        else:
            sampled_indices = indices
        
        # 使用GPU加速计算相关性矩阵
        if using_gpu:
            try:
                # 直接在GPU上索引数据，避免数据传输
                group_data_gpu = data_gpu[:, sampled_indices]
                
                # 使用cuPy直接计算相关性矩阵
                # 注：对于大型矩阵，cuPy的corrcoef可能更高效
                corr_matrix = cp.corrcoef(group_data_gpu.T)
                
                # 仅在需要时将结果转回CPU
                corr_matrix_cpu = corr_matrix.get()
                
                # 如果内存紧张，可以释放不再需要的GPU变量
                del corr_matrix
                
            except Exception as e:
                logger.error(f"GPU计算相关性失败: {str(e)}")
                logger.info("回退到CPU计算...")
                
                # 提取特征组数据(CPU)
                group_data = data[:, sampled_indices]
                corr_matrix_cpu = np.corrcoef(group_data.T)
        else:
            # CPU方式计算相关性
            group_data = data[:, sampled_indices]
            corr_matrix_cpu = np.corrcoef(group_data.T)
        
        # 分析相关性统计(在CPU上)
        # 取相关性矩阵的上三角部分，排除对角线
        upper_triangle = np.triu(corr_matrix_cpu, k=1)
        mask = upper_triangle != 0
        
        if np.any(mask):
            abs_corrs = np.abs(upper_triangle[mask])
            # 添加空数组检查
            if abs_corrs.size > 0:
                high_corr_ratio = np.mean(abs_corrs > 0.7)
                mean_abs_corr = np.mean(abs_corrs)
            else:
                high_corr_ratio = 0
                mean_abs_corr = 0
        else:
            high_corr_ratio = 0
            mean_abs_corr = 0
        
        logger.info(f"  平均绝对相关系数: {mean_abs_corr:.4f}")
        logger.info(f"  高度相关 (|r| > 0.7) 特征占比: {high_corr_ratio:.4f}")
        
        # 保存相关性分析结果
        corr_dict[group_name] = {
            'corr_matrix': corr_matrix_cpu,
            'feature_indices': sampled_indices,
            'mean_abs_corr': mean_abs_corr,
            'high_corr_ratio': high_corr_ratio
        }
        
        # 可视化相关性矩阵
        if save_dir:
            plt.figure(figsize=(12, 10))
            mask_plt = np.triu(np.ones_like(corr_matrix_cpu, dtype=bool))
            sns.heatmap(corr_matrix_cpu, mask=mask_plt, cmap="coolwarm", 
                       vmin=-1, vmax=1, center=0, annot=False, 
                       square=True, linewidths=.5)
            plt.title(f"{group_name} Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_correlation.png"), dpi=300)
            plt.close()
            
            # 保存相关性统计
            with open(os.path.join(save_dir, f"{group_name}_correlation_stats.txt"), 'w') as f:
                f.write(f"特征组: {group_name}\n")
                f.write(f"特征数量: {len(sampled_indices)}\n")
                f.write(f"平均绝对相关系数: {mean_abs_corr:.4f}\n")
                f.write(f"高度相关 (|r| > 0.7) 特征占比: {high_corr_ratio:.4f}\n")
                
                # 找出高度相关的特征对
                highly_correlated = []
                for i in range(corr_matrix_cpu.shape[0]):
                    for j in range(i+1, corr_matrix_cpu.shape[1]):
                        if abs(corr_matrix_cpu[i, j]) > 0.7:
                            highly_correlated.append((
                                sampled_indices[i], 
                                sampled_indices[j], 
                                corr_matrix_cpu[i, j]
                            ))
                
                # 排序并输出
                highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
                f.write("\n高度相关的特征对 (top 20):\n")
                for i, (idx1, idx2, corr) in enumerate(highly_correlated[:20]):
                    f.write(f"{i+1}. 特征 {idx1} - 特征 {idx2}: {corr:.4f}\n")
    
    # 分析特征组间相关性
    if len(feature_groups) > 1:
        logger.info("分析特征组间相关性")
        
        # 计算每个特征组的代表性特征
        group_representatives = {}
        for group_name, indices in feature_groups.items():
            # 如果特征组太大，随机选择代表性特征
            if len(indices) > 10:
                group_representatives[group_name] = np.random.choice(indices, 10, replace=False)
            else:
                group_representatives[group_name] = indices
        
        # 提取所有代表性特征
        all_repr_indices = []
        group_indices = []  # 记录每个特征属于哪个组
        for group_id, (group_name, indices) in enumerate(group_representatives.items()):
            all_repr_indices.extend(indices)
            group_indices.extend([group_id] * len(indices))
        
        # 使用GPU计算特征组间相关性
        if using_gpu:
            try:
                # 直接在GPU上索引数据
                all_repr_data_gpu = data_gpu[:, all_repr_indices]
                cross_corr_matrix = cp.corrcoef(all_repr_data_gpu.T).get()
                
            except Exception as e:
                logger.error(f"GPU计算特征组间相关性失败: {str(e)}")
                logger.info("回退到CPU计算...")
                
                all_repr_data = data[:, all_repr_indices]
                cross_corr_matrix = np.corrcoef(all_repr_data.T)
        else:
            all_repr_data = data[:, all_repr_indices]
            cross_corr_matrix = np.corrcoef(all_repr_data.T)
        
        # 计算特征组间平均相关性
        group_names = list(group_representatives.keys())
        n_groups = len(group_names)
        between_group_corr = np.zeros((n_groups, n_groups))
        
        for i in range(n_groups):
            for j in range(n_groups):
                if i == j:
                    continue
                
                # 获取组i和组j的索引
                i_indices = [k for k, g in enumerate(group_indices) if g == i]
                j_indices = [k for k, g in enumerate(group_indices) if g == j]
                
                # 提取组间相关系数
                cross_corrs = np.abs(cross_corr_matrix[np.ix_(i_indices, j_indices)])
                between_group_corr[i, j] = np.mean(cross_corrs) if cross_corrs.size > 0 else 0
        
        logger.info("特征组间平均绝对相关系数:")
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                logger.info(f"  {group_names[i]} - {group_names[j]}: {between_group_corr[i, j]:.4f}")
        
        # 可视化特征组间相关性
        if save_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(between_group_corr, annot=True, cmap="YlGnBu", 
                       xticklabels=group_names, yticklabels=group_names)
            plt.title("Between-Group Feature Correlation")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "between_group_correlation.png"), dpi=300)
            plt.close()
            
            # 保存组间相关性结果
            np.save(os.path.join(save_dir, "between_group_correlation.npy"), between_group_corr)
            
            # 创建详细的交叉相关性可视化
            plt.figure(figsize=(14, 12))
            
            # 创建组标签
            group_labels = []
            for group_id, group_name in enumerate(group_names):
                count = len([g for g in group_indices if g == group_id])
                group_labels.extend([group_name] * count)
            
            # 重新排序相关性矩阵，按特征组划分
            ordered_indices = np.argsort(group_indices)
            ordered_matrix = cross_corr_matrix[np.ix_(ordered_indices, ordered_indices)]
            
            # 绘制热图
            sns.heatmap(ordered_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1)
            
            # 添加特征组分隔线
            group_sizes = [len(indices) for indices in group_representatives.values()]
            cumulative_sizes = np.cumsum(group_sizes)
            
            for size in cumulative_sizes[:-1]:
                plt.axhline(y=size, color='black', linestyle='-', linewidth=1)
                plt.axvline(x=size, color='black', linestyle='-', linewidth=1)
            
            # 添加组标签
            plt.title("Cross-Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "cross_feature_correlation.png"), dpi=300)
            plt.close()
        
        # 添加组间相关性结果到返回字典
        corr_dict['between_groups'] = {
            'corr_matrix': between_group_corr,
            'group_names': group_names
        }
    
    # 最后释放GPU内存
    if using_gpu:
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    return corr_dict

    
def analyze_class_separability(data, labels, feature_groups=None, use_gpu=False, save_dir=None):
    """
    分析各特征组对类别区分的贡献度
    
    参数:
        data: 输入数据
        labels: 类别标签
        feature_groups: 特征组字典
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
    
    返回:
        separability_dict: 可分性分析结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    separability_dict = {}
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    logger.info(f"分析类别可分性，共{n_classes}个类别")
    
    # 使用ANOVA F值评估特征的区分能力
    from sklearn.feature_selection import f_classif
    
    for group_name, indices in tqdm(feature_groups.items(), desc="分析类别可分性"):
        logger.info(f"分析特征组 {group_name} 的类别区分能力")
        
        # 提取特征组数据
        group_data = data[:, indices]
        
        # 计算F值和p值
        f_vals, p_vals = f_classif(group_data, labels)
        
        # 分析特征区分能力
        significant_features = np.sum(p_vals < 0.05)
        highly_significant = np.sum(p_vals < 0.001)
        
        logger.info(f"  统计显著特征 (p < 0.05) 数量: {significant_features} "
                   f"({significant_features/len(indices)*100:.1f}%)")
        logger.info(f"  高度显著特征 (p < 0.001) 数量: {highly_significant} "
                   f"({highly_significant/len(indices)*100:.1f}%)")
        
        # 计算平均F值作为整体区分能力指标
        mean_f = np.mean(f_vals)
        top_f = np.mean(np.sort(f_vals)[-min(10, len(f_vals)):])  # 前10个特征平均F值
        
        logger.info(f"  平均F值: {mean_f:.4f}")
        logger.info(f"  前10特征平均F值: {top_f:.4f}")
        
        # 保存结果
        separability_dict[group_name] = {
            'f_values': f_vals,
            'p_values': p_vals,
            'mean_f': mean_f,
            'top_f': top_f,
            'significant_ratio': significant_features / len(indices),
            'highly_significant_ratio': highly_significant / len(indices)
        }
        
        # 可视化F值分布
        if save_dir:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(f_vals, bins=30)
            plt.title(f"{group_name} F-values Distribution")
            plt.xlabel("F-value")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 2, 2)
            # 处理p值为0或接近0的情况
            log_p_vals = -np.log10(p_vals)
            # 替换无穷大值为最大有限值的1.1倍
            finite_mask = np.isfinite(log_p_vals)
            if not np.all(finite_mask):
                # 如果有无穷大值，找出最大有限值
                if np.any(finite_mask):
                    max_finite = np.max(log_p_vals[finite_mask])
                    # 替换无穷大值为最大有限值的1.1倍
                    log_p_vals[~finite_mask] = max_finite * 1.1
                else:
                    # 如果全部是无穷大，设置为一个合理的大值
                    log_p_vals = np.ones_like(log_p_vals) * 20  # 相当于p值为10^-20
            
            plt.hist(log_p_vals, bins=30)
            plt.title(f"{group_name} -log10(p-value) Distribution")
            plt.xlabel("-log10(p-value)")
            plt.ylabel("Frequency")
            plt.axvline(x=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
            plt.axvline(x=-np.log10(0.001), color='g', linestyle='--', label='p=0.001')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_feature_significance.png"), dpi=300)
            plt.close()
            
            # 绘制top-20最显著特征
            sorted_indices = np.argsort(f_vals)[::-1]
            top_k = min(20, len(indices))
            
            plt.figure(figsize=(14, 8))
            
            top_feature_indices = [indices[i] for i in sorted_indices[:top_k]]
            top_f_vals = f_vals[sorted_indices[:top_k]]
            
            plt.barh(range(top_k), top_f_vals)
            plt.yticks(range(top_k), [f"Feature {idx}" for idx in top_feature_indices])
            plt.xlabel("F-value")
            plt.title(f"Top {top_k} Most Discriminative Features in {group_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_top_features.png"), dpi=300)
            plt.close()
            
            # 保存特征重要性排序结果
            importance_df = pd.DataFrame({
                'feature_idx': indices,
                'f_value': f_vals,
                'p_value': p_vals,
                '-log10(p)': log_p_vals  # 使用处理后的log_p_vals
            })
            importance_df = importance_df.sort_values('f_value', ascending=False)
            importance_df.to_csv(os.path.join(save_dir, f"{group_name}_feature_importance.csv"), index=False)
    
    # 比较特征组之间的区分能力
    if len(feature_groups) > 1:
        mean_f_vals = [separability_dict[group]['mean_f'] for group in feature_groups]
        top_f_vals = [separability_dict[group]['top_f'] for group in feature_groups]
        sig_ratios = [separability_dict[group]['significant_ratio'] for group in feature_groups]
        
        # 可视化比较
        if save_dir:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 3, 1)
            plt.bar(feature_groups.keys(), mean_f_vals)
            plt.title("Mean F-value by Feature Group")
            plt.ylabel("Mean F-value")
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 2)
            plt.bar(feature_groups.keys(), top_f_vals)
            plt.title("Top-10 Mean F-value by Feature Group")
            plt.ylabel("Top-10 Mean F-value")
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 3)
            plt.bar(feature_groups.keys(), sig_ratios)
            plt.title("Significant Feature Ratio by Group")
            plt.ylabel("Ratio of Significant Features")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "feature_group_comparison.png"), dpi=300)
            plt.close()
            
            # 保存特征组比较结果
            group_comparison = pd.DataFrame({
                'feature_group': list(feature_groups.keys()),
                'mean_f_value': mean_f_vals,
                'top_f_value': top_f_vals,
                'significant_ratio': sig_ratios
            })
            group_comparison.to_csv(os.path.join(save_dir, "feature_group_comparison.csv"), index=False)
    
    return separability_dict

if __name__ == "__main__":
    # 测试基本分析功能
    print("基本分析模块测试")
    
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.randn(1000, 100)
    test_labels = np.random.randint(0, 5, 1000)
    
    test_groups = {
        'group1': list(range(0, 30)),
        'group2': list(range(30, 70)),
        'group3': list(range(70, 100))
    }
    
    # 创建测试输出目录
    test_save_dir = "test_results"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # 测试基本统计分析
    stats = analyze_basic_stats(test_data, test_groups, False, test_save_dir)
    
    # 测试相关性分析
    corr = analyze_feature_correlation(test_data, test_groups, False, test_save_dir)
    
    # 测试类别可分性分析
    sep = analyze_class_separability(test_data, test_labels, test_groups, False, test_save_dir)
    
    print("测试完成")