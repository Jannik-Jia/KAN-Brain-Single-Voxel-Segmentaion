#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块
包含Multi-Subject-Out数据分割和预处理功能
"""

import numpy as np
import h5py
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_and_prepare_data_multi_subject_out(data_path='/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38_no_label43.mat',
                                          train_subjects_range=(1, 31),    # 受试者1-30用于训练
                                          val_subjects_range=(31, 38),     # 受试者31-37用于验证  
                                          test_subject=38,                 # 受试者38用于测试
                                          random_state=42):
    """
    Multi-Subject-Out数据分割策略
    
    Args:
        data_path: 数据文件路径
        train_subjects_range: 训练集受试者范围 (start, end) - 左闭右开
        val_subjects_range: 验证集受试者范围 (start, end) - 左闭右开  
        test_subject: 测试集受试者ID
        random_state: 随机种子
        
    Returns:
        dict: 包含所有数据和受试者信息的字典
    """
    
    logger.info("📂 加载数据 - Multi-Subject-Out策略...")
    
    try:
        # 加载原始数据
        f = h5py.File(data_path, 'r')
        arrays = {}
        for k, v in f.items():
            arrays[k] = np.array(v)
        f.close()
        
        train_data = arrays['data'].transpose()
        train_region = arrays['region'].transpose()
        prob_idx = arrays['prob_idx'].transpose().flatten()  # 确保是1D数组
        
        logger.info(f"原始数据形状: {train_data.shape}")
        logger.info(f"原始标签形状: {train_region.shape}")
        logger.info(f"prob_idx形状: {prob_idx.shape}")
        logger.info(f"受试者ID范围: {np.min(prob_idx)} - {np.max(prob_idx)}")
        
        del arrays, f
        
        # 按受试者分割数据
        logger.info(f"\n🎯 按Multi-Subject-Out策略分割数据...")
        logger.info(f"  训练集: 受试者 {train_subjects_range[0]}-{train_subjects_range[1]-1}")
        logger.info(f"  验证集: 受试者 {val_subjects_range[0]}-{val_subjects_range[1]-1}")  
        logger.info(f"  测试集: 受试者 {test_subject}")
        
        # 创建受试者掩码
        train_subjects_ids = list(range(train_subjects_range[0], train_subjects_range[1]))
        val_subjects_ids = list(range(val_subjects_range[0], val_subjects_range[1]))
        
        train_mask = np.isin(prob_idx, train_subjects_ids)
        val_mask = np.isin(prob_idx, val_subjects_ids)
        test_mask = prob_idx == test_subject
        
        # 验证分割完整性
        total_samples = len(prob_idx)
        assigned_samples = np.sum(train_mask) + np.sum(val_mask) + np.sum(test_mask)
        
        logger.info(f"\n📊 分割结果验证:")
        logger.info(f"  总样本数: {total_samples:,}")
        logger.info(f"  已分配样本数: {assigned_samples:,}")
        logger.info(f"  未分配样本数: {total_samples - assigned_samples:,}")
        
        if assigned_samples != total_samples:
            missing_subjects = set(np.unique(prob_idx)) - set(train_subjects_ids + val_subjects_ids + [test_subject])
            logger.warning(f"  ⚠️ 警告: 存在未分配的受试者: {sorted(missing_subjects)}")
        
        # 提取各数据集
        X_train = train_data[train_mask]
        y_train = train_region[train_mask] 
        subjects_train = prob_idx[train_mask]
        
        X_val = train_data[val_mask]
        y_val = train_region[val_mask]
        subjects_val = prob_idx[val_mask]
        
        X_test = train_data[test_mask]
        y_test = train_region[test_mask]
        subjects_test = prob_idx[test_mask]
        
        logger.info(f"\n📈 最终数据集统计:")
        logger.info(f"  训练集: {X_train.shape[0]:,} 样本, {len(np.unique(subjects_train))} 个受试者")
        logger.info(f"  验证集: {X_val.shape[0]:,} 样本, {len(np.unique(subjects_val))} 个受试者") 
        logger.info(f"  测试集: {X_test.shape[0]:,} 样本, {len(np.unique(subjects_test))} 个受试者")
        
        # 受试者样本分布统计
        logger.info(f"\n👥 受试者样本分布:")
        
        logger.info("  训练集受试者:")
        for subject_id in sorted(np.unique(subjects_train)):
            count = np.sum(subjects_train == subject_id)
            logger.info(f"    受试者{subject_id}: {count:,} 样本")
        
        logger.info("  验证集受试者:")
        for subject_id in sorted(np.unique(subjects_val)):
            count = np.sum(subjects_val == subject_id)
            logger.info(f"    受试者{subject_id}: {count:,} 样本")
        
        test_count = np.sum(subjects_test == test_subject)
        logger.info(f"  测试集受试者{test_subject}: {test_count:,} 样本")
        
        # 清理内存
        del train_data, train_region, prob_idx
        
        # 应用标准化
        logger.info(f"\n📊 应用标准化...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("✅ Multi-Subject-Out数据准备完成")
        
        # 返回完整的数据字典
        return {
            # 原始数据
            'X_train': X_train, 'y_train': y_train, 'subjects_train': subjects_train,
            'X_val': X_val, 'y_val': y_val, 'subjects_val': subjects_val,
            'X_test': X_test, 'y_test': y_test, 'subjects_test': subjects_test,
            
            # 标准化数据
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled, 
            'X_test_scaled': X_test_scaled,
            
            # 元信息
            'scaler': scaler,
            'train_subjects_ids': train_subjects_ids,
            'val_subjects_ids': val_subjects_ids,
            'test_subject_id': test_subject,
            'feature_dim': X_train.shape[1],
            'n_classes': y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(np.argmax(y_train, axis=1))),
            
            # 分割策略信息
            'split_strategy': 'Multi-Subject-Out',
            'split_config': {
                'train_subjects_range': train_subjects_range,
                'val_subjects_range': val_subjects_range, 
                'test_subject': test_subject
            }
        }
        
    except Exception as e:
        logger.error(f"数据加载过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        return None