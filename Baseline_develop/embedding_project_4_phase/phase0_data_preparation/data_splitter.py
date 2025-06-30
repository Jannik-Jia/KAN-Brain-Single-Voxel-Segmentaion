#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分割器
实现Multi-Subject-Out数据分割策略
"""

import numpy as np
import h5py
import logging
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class DataSplitter:
    """数据分割器"""
    
    def __init__(self, data_path: str,
                 train_subjects_range: Tuple[int, int] = (1, 31),
                 val_subjects_range: Tuple[int, int] = (31, 38),
                 test_subject: int = 38,
                 random_state: int = 42):
        """
        初始化数据分割器
        
        Args:
            data_path: 数据文件路径
            train_subjects_range: 训练集受试者范围
            val_subjects_range: 验证集受试者范围
            test_subject: 测试集受试者ID
            random_state: 随机种子
        """
        self.data_path = data_path
        self.train_subjects_range = train_subjects_range
        self.val_subjects_range = val_subjects_range
        self.test_subject = test_subject
        self.random_state = random_state
        
        # 设置随机种子
        np.random.seed(random_state)
    
    def load_raw_data(self) -> Dict[str, np.ndarray]:
        """加载原始数据"""
        logger.info(f"加载数据文件: {self.data_path}")
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                arrays = {}
                for k, v in f.items():
                    arrays[k] = np.array(v)
            
            # 转置数据以获得正确的形状
            train_data = arrays['data'].transpose()
            train_region = arrays['region'].transpose()
            prob_idx = arrays['prob_idx'].transpose().flatten()
            
            logger.info(f"原始数据形状: {train_data.shape}")
            logger.info(f"原始标签形状: {train_region.shape}")
            logger.info(f"受试者索引形状: {prob_idx.shape}")
            logger.info(f"受试者ID范围: {np.min(prob_idx)} - {np.max(prob_idx)}")
            
            return {
                'data': train_data,
                'region': train_region,
                'prob_idx': prob_idx
            }
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def split_by_subjects(self, raw_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """按受试者分割数据"""
        data = raw_data['data']
        region = raw_data['region']
        prob_idx = raw_data['prob_idx']
        
        # 创建受试者ID列表
        train_subjects_ids = list(range(self.train_subjects_range[0], self.train_subjects_range[1]))
        val_subjects_ids = list(range(self.val_subjects_range[0], self.val_subjects_range[1]))
        
        logger.info(f"\n执行Multi-Subject-Out分割...")
        logger.info(f"训练集受试者: {train_subjects_ids}")
        logger.info(f"验证集受试者: {val_subjects_ids}")
        logger.info(f"测试集受试者: {self.test_subject}")
        
        # 创建掩码
        train_mask = np.isin(prob_idx, train_subjects_ids)
        val_mask = np.isin(prob_idx, val_subjects_ids)
        test_mask = prob_idx == self.test_subject
        
        # 验证分割完整性
        total_samples = len(prob_idx)
        assigned_samples = np.sum(train_mask) + np.sum(val_mask) + np.sum(test_mask)
        
        logger.info(f"\n分割结果验证:")
        logger.info(f"总样本数: {total_samples:,}")
        logger.info(f"已分配样本数: {assigned_samples:,}")
        logger.info(f"未分配样本数: {total_samples - assigned_samples:,}")
        
        if assigned_samples != total_samples:
            missing_subjects = set(np.unique(prob_idx)) - set(train_subjects_ids + val_subjects_ids + [self.test_subject])
            logger.warning(f"警告: 存在未分配的受试者: {sorted(missing_subjects)}")
        
        # 提取各数据集
        split_data = {
            'X_train': data[train_mask],
            'y_train': region[train_mask],
            'subjects_train': prob_idx[train_mask],
            'X_val': data[val_mask],
            'y_val': region[val_mask],
            'subjects_val': prob_idx[val_mask],
            'X_test': data[test_mask],
            'y_test': region[test_mask],
            'subjects_test': prob_idx[test_mask],
            'train_subjects_ids': train_subjects_ids,
            'val_subjects_ids': val_subjects_ids,
            'test_subject_id': self.test_subject
        }
        
        # 打印统计信息
        self._print_split_statistics(split_data)
        
        return split_data
    
    def apply_standardization(self, split_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用标准化"""
        logger.info("\n应用特征标准化...")
        
        scaler = StandardScaler()
        
        # 在训练集上拟合标准化器
        scaler.fit(split_data['X_train'])
        
        # 应用标准化
        split_data['X_train_scaled'] = scaler.transform(split_data['X_train'])
        split_data['X_val_scaled'] = scaler.transform(split_data['X_val'])
        split_data['X_test_scaled'] = scaler.transform(split_data['X_test'])
        split_data['scaler'] = scaler
        
        logger.info("标准化完成")
        logger.info(f"特征均值范围: [{np.min(scaler.mean_):.3f}, {np.max(scaler.mean_):.3f}]")
        logger.info(f"特征标准差范围: [{np.min(scaler.scale_):.3f}, {np.max(scaler.scale_):.3f}]")
        
        return split_data
    
    def split_data(self) -> Optional[Dict[str, np.ndarray]]:
        """执行完整的数据分割流程"""
        try:
            # 加载原始数据
            raw_data = self.load_raw_data()
            
            # 按受试者分割
            split_data = self.split_by_subjects(raw_data)
            
            # 应用标准化
            split_data = self.apply_standardization(split_data)
            
            # 添加元信息
            split_data['feature_dim'] = split_data['X_train'].shape[1]
            split_data['n_classes'] = (split_data['y_train'].shape[1] 
                                      if len(split_data['y_train'].shape) > 1 
                                      else len(np.unique(np.argmax(split_data['y_train'], axis=1))))
            split_data['split_strategy'] = 'Multi-Subject-Out'
            split_data['split_config'] = {
                'train_subjects_range': self.train_subjects_range,
                'val_subjects_range': self.val_subjects_range,
                'test_subject': self.test_subject
            }
            
            logger.info("\n数据分割成功完成!")
            
            return split_data
            
        except Exception as e:
            logger.error(f"数据分割过程中出现错误: {e}")
            return None
    
    def _print_split_statistics(self, split_data: Dict[str, np.ndarray]):
        """打印分割统计信息"""
        logger.info(f"\n最终数据集统计:")
        logger.info(f"训练集: {split_data['X_train'].shape[0]:,} 样本, "
                   f"{len(np.unique(split_data['subjects_train']))} 个受试者")
        logger.info(f"验证集: {split_data['X_val'].shape[0]:,} 样本, "
                   f"{len(np.unique(split_data['subjects_val']))} 个受试者")
        logger.info(f"测试集: {split_data['X_test'].shape[0]:,} 样本, "
                   f"{len(np.unique(split_data['subjects_test']))} 个受试者")
        
        # 受试者样本分布
        logger.info(f"\n受试者样本分布:")
        
        logger.info("训练集:")
        for subject_id in sorted(np.unique(split_data['subjects_train'])):
            count = np.sum(split_data['subjects_train'] == subject_id)
            logger.info(f"  受试者{int(subject_id)}: {count:,} 样本")
        
        logger.info("\n验证集:")
        for subject_id in sorted(np.unique(split_data['subjects_val'])):
            count = np.sum(split_data['subjects_val'] == subject_id)
            logger.info(f"  受试者{int(subject_id)}: {count:,} 样本")
        
        logger.info(f"\n测试集:")
        test_count = len(split_data['subjects_test'])
        logger.info(f"  受试者{self.test_subject}: {test_count:,} 样本")