#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件，定义了模型训练和评估的默认参数（支持两种数据格式）
"""

import os
import json
import torch

# 基础配置
CONFIG = {
    # 随机种子和设备设置
    'random_seed': 666,
    'device': 0,  # -1表示CPU，>=0表示使用对应索引的GPU
    
    # 数据路径和处理设置 - 支持两种格式
    # 方式1：原版分散文件格式
    'data_dirs': {
        'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
        'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
        'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
    },
    
    # 方式2：MAT文件格式（如果设置了mat_file_path，将优先使用这种方式）

    'mat_file_path': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/TRAIN38.mat", 
    'test_size': 0.01,      # 从训练集中分割出测试集的比例（仅在随机划分模式下使用）
    'demo_mat_path': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/DATA/DEMO38.mat",  # 用于评估的DEMO38.mat文件路径
    
    # 🔧 新增：固定测试集配置（MAT格式专用）
    'test_prob_idx': [13, 23, 38],     # 固定测试集的prob_idx列表，例如: [13, 23, 38]
                               # 设为None则使用原有的随机划分模式
    'train_val_ratio': 0.75,   # 在排除测试集后，训练集在训练+验证中的比例
                               # 0.75表示训练:验证 = 3:1 (75%:25%)
    
    # MAT文件的患者ID配置 - 自定义数据集分割（已废弃，建议使用test_prob_idx）
    'dataset_split': {
        'train_patients': [28, 5, 25, 30, 34, 32, 33, 11, 12, 20, 29, 17, 37, 7, 26, 1, 36, 14, 19, 3, 35, 31, 22, 8],
        'val_patients': [4, 24, 9, 15, 16, 18, 2],
        'test_patients': [38, 6, 21, 13, 10, 23, 27]
    },
    

    # 背景像素处理配置
    'filter_background': True,          # 是否过滤背景像素（默认True保持向后兼容）
    'background_label_original': 0,     # 原始数据中的背景标签值
    'background_label_target': -1,      # 训练时的背景标签值（-1表示ignore，仅在filter_background=False时使用）
    'include_background_in_classes': False,  # 是否将背景作为一个分类类别（仅在filter_background=False时使用）

    'apply_pca': False,  # 是否应用PCA降维
    'n_pca': 0,          # PCA保留的主成分数量，0表示不进行PCA
    'norm': True,        # 是否进行数据标准化
    
    # 模型基本参数
    'model_name': 'BrainVoxel_102Class_MLP',
    'dataset_name': 'BrainVoxel',
    'feature_dim': 341,  # 原始特征维度
    'num_class': 102,    # 类别数量
    
    # 标签处理配置
    'label_format': 'auto',  # 'original' (1-102, 背景-1), 'mat' (0-101, 背景0), 'auto' (自动检测)
    
    # 模型架构参数
    'model_type': 'base_mlp',  # 'base_mlp', 'deep_mlp', 'residual_mlp'
    'hidden_units': [4096, 4096, 4096, 4096],  # MLP隐藏层大小
    'dropout_rate': 0.5,        # Dropout率
    'activation': 'relu',       # 激活函数: 'relu', 'gelu', 'swish'
    
    # 训练参数
    'epochs': 30,        # 训练轮数
    'val_epochs': 3,     # 验证频率
    'batch_size': 128,   # 批处理大小
    'lr': 1e-5,          # 学习率
    'weight_decay': 1e-5,# 权重衰减
    
    # 学习率调度参数
    'use_lr_scheduler': True,           # 是否使用学习率调度
    'lr_scheduler_type': 'cosine',      # 'multistep', 'cosine', 'plateau'
    'lr_milestones': [10, 20],          # 多步调度的里程碑轮次
    'lr_gamma': 0.5,                    # 学习率降低的倍数因子
    
    # 优化器设置
    'optimizer': 'adamw',  # 'adam', 'adamw'
    
    # 贝叶斯优化参数
    'run_bayesian_opt': True,  # 是否运行贝叶斯优化
    'n_trials': 30,            # 贝叶斯优化的试验次数
    'pruning_patience': 5,     # 提前终止的耐心值
    
    # 结果保存
    'save_dir': './results',    # 结果保存目录
    'log_dir': './logs',        # 日志保存目录
    'save_checkpoints': True,   # 是否保存检查点
    'use_old_zipfile_serialization': True,  # 使用旧的zipfile序列化方式（兼容性更好）
}

def load_config(config_file=None):
    """
    从JSON文件加载配置，并与默认配置合并
    
    参数:
        config_file: 配置文件路径，None表示使用默认配置
    
    返回:
        config: 合并后的配置字典
    """
    config = CONFIG.copy()
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # 确保结果和日志目录存在
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 数据格式自适应逻辑
    _auto_detect_data_format(config)
    
    return config

def _auto_detect_data_format(config):
    """
    自动检测数据格式并设置相应参数
    """
    # 如果设置了mat_file_path且文件存在，优先使用MAT格式
    if config.get('mat_file_path') and os.path.exists(config['mat_file_path']):
        print(f"检测到MAT文件: {config['mat_file_path']}")
        print("将使用MAT数据加载格式")
        
        # 🔧 新增：检查测试集划分模式
        if config.get('test_prob_idx') is not None:
            print(f"🎯 使用固定prob_idx测试集划分模式")
            print(f"  测试集prob_idx: {config['test_prob_idx']}")
            print(f"  训练/验证比例: {config.get('train_val_ratio', 0.75):.2f}/{1-config.get('train_val_ratio', 0.75):.2f}")
        else:
            print(f"🎲 使用随机测试集划分模式（原有方式）")
        
        # 确保MAT格式的相关配置
        if config.get('label_format') == 'auto':
            config['label_format'] = 'mat'
        
        # 简化方案：不使用ignore_index
        config['ignore_index'] = None  # 修改：不忽略任何标签
        
    # 否则检查是否有传统的数据目录
    elif config.get('data_dirs') and all(
        config['data_dirs'].get(key) and os.path.exists(config['data_dirs'][key]) 
        for key in ['train_dir', 'test_dir', 'val_dir']
    ):
        print("检测到传统数据目录格式")
        print("将使用原版数据加载格式")
        
        # 原版格式不支持固定prob_idx，给出提示
        if config.get('test_prob_idx') is not None:
            print("⚠️ 警告: 原版数据格式不支持test_prob_idx配置，该配置将被忽略")
        
        # 确保原版格式的相关配置
        if config.get('label_format') == 'auto':
            config['label_format'] = 'original'
        
        # 简化方案：不使用ignore_index
        config['ignore_index'] = None  # 修改：不忽略任何标签
        
    else:
        print("警告: 未检测到有效的数据源配置")
        print("请设置 'mat_file_path' 或完整的 'data_dirs'")
    
    # 修改：更新标签格式说明
    print("标签格式说明:")
    print('- 原始数据: 0=背景, 1-102=有效类别')
    print('- 处理后数据: 1-102=有效类别 (背景已过滤)')  
    print('- 训练时: 0-101=有效类别 (映射后)')
    print("- 模型输出: 102个类别 (0-101)")

def save_config(config, filepath):
    """
    保存配置到JSON文件
    
    参数:
        config: 配置字典
        filepath: 保存路径
    """
    # 创建一个副本，移除不可序列化的对象
    config_to_save = config.copy()
    
    # 移除可能存在的不可序列化对象
    if 'scaler' in config_to_save:
        del config_to_save['scaler']
    if 'pca_model' in config_to_save:
        del config_to_save['pca_model']
    
    with open(filepath, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    
    return filepath

def get_data_format(config):
    """
    获取当前配置使用的数据格式
    
    返回:
        'mat' 或 'original'
    """
    if config.get('mat_file_path') and os.path.exists(config['mat_file_path']):
        return 'mat'
    elif config.get('data_dirs') and all(config['data_dirs'].values()):
        return 'original'
    else:
        return 'unknown'

def is_mat_format(config):
    """
    检查是否使用MAT格式
    """
    return get_data_format(config) == 'mat'

def is_original_format(config):
    """
    检查是否使用原版格式
    """
    return get_data_format(config) == 'original'

def get_data_split_mode(config):
    """
    🔧 新增：获取数据划分模式
    
    返回:
        'fixed_test': 使用固定prob_idx测试集
        'random': 使用随机划分
        'legacy': 使用原有的dataset_split配置（已废弃）
    """
    if is_mat_format(config):
        if config.get('test_prob_idx') is not None:
            return 'fixed_test'
        else:
            return 'random'
    else:
        return 'legacy'

def validate_config(config):
    """
    🔧 新增：验证配置的有效性
    
    参数:
        config: 配置字典
    
    返回:
        bool: 配置是否有效
    """
    errors = []
    warnings = []
    
    # 检查基本配置
    if not config.get('mat_file_path') and not all(config.get('data_dirs', {}).values()):
        errors.append("必须设置 'mat_file_path' 或完整的 'data_dirs'")
    
    # 检查固定测试集配置
    if config.get('test_prob_idx') is not None:
        if not is_mat_format(config):
            warnings.append("test_prob_idx只在MAT格式下有效，当前配置将被忽略")
        elif not isinstance(config['test_prob_idx'], list):
            errors.append("test_prob_idx必须是一个列表")
        elif len(config['test_prob_idx']) == 0:
            warnings.append("test_prob_idx是空列表，将使用随机划分模式")
    
    # 检查训练验证比例
    train_val_ratio = config.get('train_val_ratio', 0.75)
    if not 0 < train_val_ratio < 1:
        errors.append(f"train_val_ratio必须在0和1之间，当前值: {train_val_ratio}")
    
    # 新增：验证背景处理配置
    if not validate_background_config(config):
        return False
    
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("⚠️ 配置警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ 配置验证通过")
    return True


def validate_background_config(config):
    """
    验证背景处理配置的有效性
    
    参数:
        config: 配置字典
    
    返回:
        bool: 配置是否有效
    """
    errors = []
    warnings = []
    
    filter_bg = config.get('filter_background', True)
    bg_target = config.get('background_label_target', -1)
    include_bg = config.get('include_background_in_classes', False)
    num_classes = config.get('num_class', 102)
    
    if filter_bg:
        # 过滤背景模式
        if num_classes != 102:
            warnings.append(f"过滤背景模式下，num_class应为102，当前为{num_classes}")
        if include_bg:
            warnings.append("过滤背景模式下，include_background_in_classes将被忽略")
    else:
        # 保留背景模式
        if include_bg:
            # 背景作为分类类别
            if num_classes != 103:
                errors.append(f"包含背景分类模式下，num_class应为103，当前为{num_classes}")
            if bg_target != 0:
                warnings.append(f"包含背景分类模式下，background_label_target应为0，当前为{bg_target}")
        else:
            # 背景被忽略
            if num_classes != 102:
                warnings.append(f"忽略背景模式下，num_class应为102，当前为{num_classes}")
            if bg_target != -1:
                warnings.append(f"忽略背景模式下，background_label_target应为-1，当前为{bg_target}")
    
    # 输出结果
    if errors:
        print("❌ 背景配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("⚠️ 背景配置警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return True

def get_effective_num_classes(config):
    """
    获取实际的类别数量
    
    返回:
        int: 实际类别数量
    """
    if config.get('filter_background', True):
        return 102  # 过滤背景：1-102 → 0-101
    else:
        if config.get('include_background_in_classes', False):
            return 103  # 包含背景：0-102 → 0-102
        else:
            return 102  # 忽略背景：0→-1, 1-102 → 0-101

def print_background_config_info(config):
    """
    打印背景处理配置信息
    """
    print("\n🎯 背景处理配置:")
    filter_bg = config.get('filter_background', True)
    
    if filter_bg:
        print("  模式: 过滤背景像素（当前默认）")
        print("  处理: 在数据加载阶段移除background=0的样本")
        print("  标签映射: 1-102 → 0-101")
        print("  模型输出: 102个类别")
        print("  损失函数: CrossEntropyLoss（无ignore_index）")
    else:
        include_bg = config.get('include_background_in_classes', False)
        bg_target = config.get('background_label_target', -1)
        
        print("  模式: 保留背景像素")
        if include_bg:
            print("  处理: 背景作为第0类进行分类")
            print("  标签映射: 0-102 → 0-102")
            print("  模型输出: 103个类别")
            print("  损失函数: CrossEntropyLoss（无ignore_index）")
        else:
            print("  处理: 背景像素在训练时被忽略")
            print("  标签映射: 0→-1, 1-102→0-101")
            print("  模型输出: 102个类别")
            print("  损失函数: CrossEntropyLoss（ignore_index=-1）")
    
    print(f"  实际类别数: {get_effective_num_classes(config)}")
    print()


# 🔧 新增：预设配置模板
PRESET_CONFIGS = {
    'default_fixed_test': {
        'test_prob_idx': [13, 23, 38],
        'train_val_ratio': 0.75,
        'description': '默认固定测试集配置：prob_idx [13,23,38] 作为测试集，75%训练25%验证'
    },
    'small_test_set': {
        'test_prob_idx': [38],
        'train_val_ratio': 0.8,
        'description': '小测试集配置：仅prob_idx 38作为测试集，80%训练20%验证'
    },
    'large_test_set': {
        'test_prob_idx': [6, 10, 13, 21, 23, 27, 38],
        'train_val_ratio': 0.75,
        'description': '大测试集配置：多个prob_idx作为测试集，75%训练25%验证'
    },
    'random_split': {
        'test_prob_idx': None,
        'train_val_ratio': 0.75,
        'description': '随机划分模式：使用原有的随机划分逻辑'
    }
}

def apply_preset_config(config, preset_name):
    """
    🔧 新增：应用预设配置
    
    参数:
        config: 当前配置字典
        preset_name: 预设配置名称
    
    返回:
        bool: 是否成功应用
    """
    if preset_name not in PRESET_CONFIGS:
        print(f"❌ 未知的预设配置: {preset_name}")
        print(f"可用的预设配置: {list(PRESET_CONFIGS.keys())}")
        return False
    
    preset = PRESET_CONFIGS[preset_name]
    config.update({
        'test_prob_idx': preset['test_prob_idx'],
        'train_val_ratio': preset['train_val_ratio']
    })
    
    print(f"✅ 已应用预设配置: {preset_name}")
    print(f"   {preset['description']}")
    return True

def print_data_split_info(config):
    """
    🔧 新增：打印数据划分信息
    """
    print("\n📊 数据划分配置信息:")
    print(f"  数据格式: {'MAT格式' if is_mat_format(config) else '原版格式'}")
    
    split_mode = get_data_split_mode(config)
    if split_mode == 'fixed_test':
        print(f"  划分模式: 固定prob_idx测试集")
        print(f"  测试集prob_idx: {config['test_prob_idx']}")
        print(f"  训练/验证比例: {config.get('train_val_ratio', 0.75):.2f}/{1-config.get('train_val_ratio', 0.75):.2f}")
    elif split_mode == 'random':
        print(f"  划分模式: 随机划分")
        print(f"  测试集比例: {config.get('test_size', 0.01):.2f}")
    else:
        print(f"  划分模式: 原版格式（使用数据目录）")
    print()