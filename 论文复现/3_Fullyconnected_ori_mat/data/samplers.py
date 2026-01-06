#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据采样器实现
"""

import os
import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict


class BrainVoxelSampler:
    """脑体素数据采样器，提供多种采样策略"""
    
    def __init__(self, data_dir):
        """
        初始化采样器
        
        参数:
            data_dir: 数据集目录
        """
        self.data_dir = data_dir
        self.label_info = self._load_label_index()
        self.valid_labels = [label for label, info in self.label_info.items() if info['count'] > 0]
    
    def _load_label_index(self):
        """加载标签索引文件"""
        index_file = os.path.join(self.data_dir, "label_index.txt")
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
        
        return label_info
    
    def get_file_path(self, label_id):
        """获取指定标签的文件路径"""
        if label_id not in self.label_info:
            return None
        
        filename = self.label_info[label_id]['filename']
        if not filename:
            return None
            
        return os.path.join(self.data_dir, filename)


class PatientAwareBatchSampler(Sampler):
    """
    确保每个批次包含来自多个患者的样本，避免批次内样本全部来自同一患者
    """
    def __init__(self, patient_indices, batch_size, drop_last=False, min_patients_per_batch=3):
        """
        参数:
            patient_indices: 每个样本对应的患者ID数组
            batch_size: 批次大小
            drop_last: 是否丢弃最后一个不完整的批次
            min_patients_per_batch: 每个批次最少包含的患者数
        """
        self.patient_indices = np.array(patient_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.min_patients_per_batch = min_patients_per_batch
        
        # 构建患者到样本索引的映射
        self.patient_to_indices = defaultdict(list)
        for idx, patient_id in enumerate(patient_indices):
            self.patient_to_indices[patient_id].append(idx)
        
        self.patients = list(self.patient_to_indices.keys())
        self.n_patients = len(self.patients)
        
        # 计算每个患者的样本数
        self.patient_sample_counts = {
            pid: len(indices) for pid, indices in self.patient_to_indices.items()
        }
        
        print(f"PatientAwareBatchSampler初始化:")
        print(f"  总样本数: {len(patient_indices)}")
        print(f"  患者数: {self.n_patients}")
        print(f"  批次大小: {batch_size}")
        print(f"  每批次最少患者数: {min_patients_per_batch}")
        
    def __iter__(self):
        # 为每个患者的样本创建随机顺序
        patient_iterators = {}
        for pid, indices in self.patient_to_indices.items():
            shuffled_indices = np.random.permutation(indices).tolist()
            patient_iterators[pid] = iter(shuffled_indices)
        
        # 生成批次
        batch = []
        patients_in_batch = set()
        
        while True:
            # 随机选择一个还有剩余样本的患者
            available_patients = [
                pid for pid in self.patients 
                if pid in patient_iterators
            ]
            
            if not available_patients:
                break
            
            # 优先选择当前批次中还没有的患者
            patients_not_in_batch = [
                pid for pid in available_patients 
                if pid not in patients_in_batch
            ]
            
            if patients_not_in_batch and len(patients_in_batch) < self.min_patients_per_batch:
                # 需要更多患者来满足最小患者数要求
                selected_patient = np.random.choice(patients_not_in_batch)
            else:
                # 随机选择任意可用患者
                selected_patient = np.random.choice(available_patients)
            
            # 从选定患者获取一个样本
            try:
                sample_idx = next(patient_iterators[selected_patient])
                batch.append(sample_idx)
                patients_in_batch.add(selected_patient)
            except StopIteration:
                # 该患者的样本用完了
                del patient_iterators[selected_patient]
                continue
            
            # 检查是否形成完整批次
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                patients_in_batch = set()
        
        # 处理最后一个批次
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.patient_indices) // self.batch_size
        else:
            return (len(self.patient_indices) + self.batch_size - 1) // self.batch_size


class BalancedPatientSampler(Sampler):
    """
    平衡的患者采样器，确保每个epoch中各患者的样本被均匀采样
    """
    def __init__(self, patient_indices, samples_per_patient_per_epoch=None):
        """
        参数:
            patient_indices: 每个样本对应的患者ID数组
            samples_per_patient_per_epoch: 每个患者每个epoch采样的样本数
                                           如果为None，则使用最少患者的样本数
        """
        self.patient_indices = np.array(patient_indices)
        
        # 构建患者到样本索引的映射
        self.patient_to_indices = defaultdict(list)
        for idx, patient_id in enumerate(patient_indices):
            self.patient_to_indices[patient_id].append(idx)
        
        # 计算每个患者的样本数
        self.patient_sample_counts = {
            pid: len(indices) for pid, indices in self.patient_to_indices.items()
        }
        
        # 确定每个患者每个epoch的采样数
        if samples_per_patient_per_epoch is None:
            self.samples_per_patient = min(self.patient_sample_counts.values())
        else:
            self.samples_per_patient = min(
                samples_per_patient_per_epoch,
                min(self.patient_sample_counts.values())
            )
        
        self.total_samples = len(self.patient_to_indices) * self.samples_per_patient
        
        print(f"BalancedPatientSampler初始化:")
        print(f"  患者数: {len(self.patient_to_indices)}")
        print(f"  每患者采样数: {self.samples_per_patient}")
        print(f"  总采样数: {self.total_samples}")
        
    def __iter__(self):
        indices = []
        
        # 为每个患者采样指定数量的样本
        for pid, patient_indices in self.patient_to_indices.items():
            # 随机采样（可重复）
            sampled = np.random.choice(
                patient_indices, 
                size=self.samples_per_patient,
                replace=len(patient_indices) < self.samples_per_patient
            )
            indices.extend(sampled)
        
        # 打乱所有索引
        np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.total_samples


def create_patient_aware_dataloader(dataset, patient_indices, batch_size=128, 
                                  shuffle=True, sampler_type='aware', **kwargs):
    """
    创建考虑患者分布的DataLoader
    
    参数:
        dataset: PyTorch数据集
        patient_indices: 每个样本对应的患者ID
        batch_size: 批次大小
        shuffle: 是否打乱（仅在sampler_type='default'时有效）
        sampler_type: 'aware'(PatientAwareBatchSampler), 
                     'balanced'(BalancedPatientSampler),
                     'default'(标准DataLoader)
        **kwargs: 传递给DataLoader的其他参数
    """
    from torch.utils.data import DataLoader
    
    if sampler_type == 'aware':
        batch_sampler = PatientAwareBatchSampler(
            patient_indices, 
            batch_size, 
            drop_last=kwargs.get('drop_last', False),
            min_patients_per_batch=kwargs.get('min_patients_per_batch', 3)
        )
        # 使用batch_sampler时，不能指定batch_size, shuffle, sampler, drop_last
        return DataLoader(dataset, batch_sampler=batch_sampler, **{
            k: v for k, v in kwargs.items() 
            if k not in ['batch_size', 'shuffle', 'sampler', 'drop_last', 'min_patients_per_batch']
        })
    
    elif sampler_type == 'balanced':
        sampler = BalancedPatientSampler(
            patient_indices,
            samples_per_patient_per_epoch=kwargs.get('samples_per_patient', None)
        )
        # 使用sampler时，不能指定shuffle
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=sampler,
            **{k: v for k, v in kwargs.items() 
               if k not in ['shuffle', 'samples_per_patient']}
        )
    
    else:  # default
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )