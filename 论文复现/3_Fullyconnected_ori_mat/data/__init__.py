#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载接口（支持原版和MAT版本，以及Patientwise标准化）
"""

from .dataset import BrainVoxelDataset, load_multiclass_data
from .mat_loader import load_and_process_data as load_and_process_data_original
from .mat_loader_patientwise import (
    load_and_process_data_with_patientwise,
    load_external_mat_data_patientwise,
    patientwise_standardize,
    BrainVoxelMatDataset
)
from .samplers import (
    BrainVoxelSampler, 
    PatientAwareBatchSampler, 
    BalancedPatientSampler,
    create_patient_aware_dataloader
)
from config import is_mat_format


def load_data(config, mode='train', model_path=None):
    """
    统一的数据加载接口，根据配置自动选择合适的加载方式
    
    参数:
        config: 配置字典
        mode: 'train' 或 'eval'
        model_path: 模型路径（用于eval模式）
    
    返回:
        dataset_dict: 数据集字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 检查是否使用MAT格式
    if is_mat_format(config):
        # MAT格式：检查是否使用patientwise标准化
        if config.get('standardization_method', 'global') == 'patientwise':
            print("使用MAT格式数据 + Patientwise标准化")
            return load_and_process_data_with_patientwise(config, mode, model_path)
        else:
            print("使用MAT格式数据 + 全局标准化")
            return load_and_process_data_original(config, mode, model_path)
    else:
        # 原版格式：暂不支持patientwise标准化
        print("使用原版数据格式")
        if config.get('standardization_method', 'global') == 'patientwise':
            print("⚠️ 警告: 原版数据格式暂不支持patientwise标准化，将使用全局标准化")
        
        # 使用原版数据加载
        dataset_dict = load_multiclass_data(
            config['data_dirs'],
            apply_pca_flag=config.get('apply_pca', False),
            n_components=config.get('n_pca', 0),
            norm=config.get('norm', True)
        )
        
        # 创建数据集
        from torch.utils.data import DataLoader
        train_dataset = BrainVoxelDataset(
            dataset_dict['train_samples'], 
            dataset_dict['train_labels'],
            config=config
        )
        test_dataset = BrainVoxelDataset(
            dataset_dict['test_samples'], 
            dataset_dict['test_labels'],
            config=config
        )
        val_dataset = BrainVoxelDataset(
            dataset_dict['val_samples'], 
            dataset_dict['val_labels'],
            config=config
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', 128), 
            shuffle=(mode == 'train')
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.get('batch_size', 128), 
            shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.get('batch_size', 128), 
            shuffle=False
        )
        
        return dataset_dict, train_loader, val_loader, test_loader


def load_external_data(mat_file_path, config):
    """
    加载外部MAT数据文件的统一接口
    
    参数:
        mat_file_path: MAT文件路径
        config: 配置字典
    
    返回:
        处理后的数据字典
    """
    if config.get('standardization_method', 'global') == 'patientwise':
        return load_external_mat_data_patientwise(mat_file_path, config)
    else:
        # 使用原有的加载方式
        from .mat_loader import load_external_mat_data
        return load_external_mat_data(
            mat_file_path, 
            scaler=config.get('scaler'),
            scaler_path=None
        )


# 导出常用函数和类
__all__ = [
    'BrainVoxelDataset',
    'BrainVoxelMatDataset',
    'BrainVoxelSampler',
    'PatientAwareBatchSampler',
    'BalancedPatientSampler',
    'create_patient_aware_dataloader',
    'load_data',
    'load_external_data',
    'load_multiclass_data',
    'patientwise_standardize'
]