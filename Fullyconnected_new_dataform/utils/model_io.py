#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型加载和保存工具函数
"""

import torch
import os
import json
import numpy as np


def save_model_with_architecture(model, optimizer, config, training_info, save_path, 
                               normalization_params=None, lr_scheduler=None, 
                               use_old_zipfile_serialization=True):
    
    """
    保存模型并包含完整的架构和超参数信息
    
    参数:
        model: 模型对象
        optimizer: 优化器对象
        config: 配置字典
        training_info: 训练信息字典，包含损失曲线、准确率等
        save_path: 保存路径
        lr_scheduler: 学习率调度器（可选）
        use_old_zipfile_serialization: 是否使用旧的PyTorch序列化格式
        normalization_params: 包含均值和标准差的字典，用于特征标准化
    
    返回:
        save_dict: 保存的字典
        save_path: 保存的文件路径
    """
    print(f"保存模型及架构信息到: {save_path}")
    
    # 收集模型架构信息
    model_arch_info = {
        # 基本模型信息
        'model_type': config.get('model_type', 'unknown'),
        'model_name': config.get('model_name', 'unknown'),
        'feature_dim': config.get('feature_dim', model.layers[0].in_features if hasattr(model, 'layers') else 0),
        'num_class': config.get('num_class', 0),
        
        # 架构详细信息
        'hidden_units': config.get('hidden_units', []),
        'activation': config.get('activation', 'relu'),
        'dropout_rate': config.get('dropout_rate', 0.5),
        'use_skip_connections': config.get('use_skip_connections', False),
        'use_bottleneck': config.get('use_bottleneck', False),
        'bottleneck_factor': config.get('bottleneck_factor', 0.5),
        
        # 训练超参数
        'optimizer': config.get('optimizer', 'adam'),
        'lr': config.get('lr', 0.001),
        'weight_decay': config.get('weight_decay', 0),
        'batch_size': config.get('batch_size', 128),
        'epochs': config.get('epochs', 0),
        
        # 学习率调度器信息
        'use_lr_scheduler': config.get('use_lr_scheduler', False),
        'lr_scheduler_type': config.get('lr_scheduler_type', 'none'),
        'lr_milestones': config.get('lr_milestones', []),
        'lr_gamma': config.get('lr_gamma', 0.1),
        'cosine_t_max': config.get('cosine_t_max', 0),
        'cosine_eta_min': config.get('cosine_eta_min', 0),
        'plateau_patience': config.get('plateau_patience', 5),
        'plateau_factor': config.get('plateau_factor', 0.5),
        'plateau_threshold': config.get('plateau_threshold', 1e-4),
        
        # 数据处理信息
        'apply_pca': config.get('apply_pca', False),
        'n_pca': config.get('n_pca', 0),
        'norm': config.get('norm', True),
        'scaler_path': config.get('scaler_path', None),
        
        # 其他配置信息
        'random_seed': config.get('random_seed', 666),
        'experiment_name': config.get('experiment_name', 'unknown'),
        'normalization_params': normalization_params,
    }
    
    # 收集模型额外特征（如果有自定义类或属性）
    if hasattr(model, 'get_model_info'):
        model_arch_info.update(model.get_model_info())
    
    # 创建主保存字典
    save_dict = {
        # 模型核心信息
        'state_dict': model.state_dict(),
        'model_arch_info': model_arch_info,
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        
        # 训练过程信息
        'training_info': training_info,
        
        # 学习率调度器状态
        'lr_scheduler_state': lr_scheduler.state_dict() if lr_scheduler else None,
        'lr_scheduler_type': config.get('lr_scheduler_type', 'none') if lr_scheduler else 'none',
        
        # 标准化信息
        'normalization_params': normalization_params,
        'scaler_path': config.get('scaler_path', None),
        
        # 元信息
        'created_with': f'PyTorch {torch.__version__}',
        'numpy_version': f'{np.__version__}',
        'save_format_version': 1.1,
        'save_timestamp': import_time().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # 尝试保存模型，兼容性设置
    try:
        # 先尝试使用标准方法
        torch.save(save_dict, save_path, _use_new_zipfile_serialization=not use_old_zipfile_serialization)
        print(f"已保存模型及完整架构信息到: {save_path}")
        
        # 保存一个独立的JSON文件，便于查看和解析
        json_path = os.path.splitext(save_path)[0] + '_architecture.json'
        with open(json_path, 'w') as f:
            # 过滤掉不可JSON序列化的项目
            json_safe_dict = {k: v for k, v in model_arch_info.items() if k != 'state_dict' and k != 'optimizer_state' and k != 'lr_scheduler_state'}
            json.dump(json_safe_dict, f, indent=4)
        print(f"已保存模型架构信息到JSON文件: {json_path}")
        
    except TypeError as e:
        # 如果不支持 _use_new_zipfile_serialization 参数
        if "_use_new_zipfile_serialization" in str(e):
            try:
                # 尝试直接保存
                torch.save(save_dict, save_path)
                print(f"已使用默认序列化方式保存模型到: {save_path}")
                
                # 保存JSON架构文件
                json_path = os.path.splitext(save_path)[0] + '_architecture.json'
                with open(json_path, 'w') as f:
                    json_safe_dict = {k: v for k, v in model_arch_info.items() if k != 'state_dict' and k != 'optimizer_state' and k != 'lr_scheduler_state'}
                    json.dump(json_safe_dict, f, indent=4)
                print(f"已保存模型架构信息到JSON文件: {json_path}")
                
            except Exception as save_e:
                print(f"保存模型失败: {save_e}")
                raise save_e
        else:
            raise e
    
    return save_dict, save_path

def import_time():
    """导入时间模块并返回，避免全局导入"""
    import time
    return time

def safe_load_model(model_path, device='cpu'):
    """
    安全加载模型，处理 PyTorch 版本兼容性问题

    参数:
        model_path: 模型文件路径
        device: 设备（'cpu' 或 'cuda:0' 等）

    返回:
    参数:
        model_path: 模型文件路径
        device: 设备（'cpu' 或 'cuda:0' 等）
        
    返回:
        checkpoint: 加载的模型检查点
    """
    import torch
    import numpy as np
    print(f"安全加载模型: {model_path}")
    
    # 确保numpy相关类型被添加到安全全局变量
    try:
        # 添加多个可能需要的numpy类型
        safe_globals = [
            np.dtype,
            np.core.multiarray.scalar,
            np.ndarray,
            np.generic,
            np.float64,
            np.float32,
            np.int64,
            np.int32
        ]
        torch.serialization.add_safe_globals(safe_globals)
        print("已添加numpy类型到安全全局变量列表")
    except Exception as e:
        print(f"添加安全全局变量时出错 (可忽略): {e}")
    
    # 1. 使用上下文管理器和安全全局变量尝试加载
    try:
        with torch.serialization.safe_globals([np.dtype, np.core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=device)
            print("成功使用safe_globals上下文管理器加载模型")
            return checkpoint
    except Exception as e:
        print(f"使用上下文管理器加载失败: {e}")
    
    # 2. 显式使用weights_only=False尝试加载
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("成功使用weights_only=False加载模型")
        return checkpoint
    except Exception as e:
        print(f"使用weights_only=False加载失败: {e}")
    
    # 3. 使用pickle加载尝试
    try:
        import pickle
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("成功使用pickle直接加载模型")
        return checkpoint
    except Exception as e:
        print(f"使用pickle加载失败: {e}")
    
    # 4. 最后尝试使用默认设置
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("成功使用默认设置加载模型")
        return checkpoint
    except Exception as e:
        print(f"所有加载方法都失败: {e}")
        
        # 提供更详细的错误信息
        print("\n尝试恢复模型失败，可能原因包括:")
        print("1. PyTorch版本不兼容 (当前版本与保存模型时的版本不同)")
        print("2. 模型文件被损坏或不完整")
        print("3. 序列化格式变更导致的兼容性问题")
        print("\n建议解决方案:")
        print("- 检查PyTorch版本，可能需要回退到保存模型时使用的版本")
        print("- 在主程序开始添加: torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar])")
        print("- 如果可能，考虑重新训练模型并使用更简单的保存方法")
        
        raise RuntimeError(f"无法加载模型文件 {model_path}")

def load_model_with_architecture(model_path, device='cpu', strict=True):
    """
    从保存的文件完整重建模型架构并加载权重
    
    参数:
        model_path: 模型文件路径
        device: 设备
        strict: 是否严格加载权重（如果模型结构有差异会报错）
        
    返回:
        model: 重建并加载了权重的模型
        checkpoint: 完整的检查点数据
    """
    # 安全加载检查点
    checkpoint = safe_load_model(model_path, device)
    
    # 检查是否包含模型架构信息
    if 'model_arch_info' not in checkpoint:
        raise ValueError(f"模型文件 {model_path} 不包含架构信息，无法完整重建模型。请提供模型类型和参数。")
    
    # 获取架构信息
    arch_info = checkpoint['model_arch_info']
    model_type = arch_info.get('model_type', 'base_mlp')
    
    # 从models模块导入get_model函数
    from models import get_model
    
    # 构建模型参数
    model_args = {
        'input_dim': arch_info.get('feature_dim', 0),
        'hidden_dims': arch_info.get('hidden_units', [4096, 4096, 4096, 4096]),
        'num_classes': arch_info.get('num_class', 0),
        'dropout_rate': arch_info.get('dropout_rate', 0.5),
        'activation': arch_info.get('activation', 'relu'),
    }
    
    # 根据模型类型添加特定参数
    if model_type == 'deep_mlp':
        model_args['use_skip_connections'] = arch_info.get('use_skip_connections', False)
    elif model_type == 'residual_mlp':
        model_args['use_bottleneck'] = arch_info.get('use_bottleneck', False)
        model_args['bottleneck_factor'] = arch_info.get('bottleneck_factor', 0.5)
    
    # 创建模型
    print(f"重建模型架构: {model_type}")
    print(f"输入维度: {model_args['input_dim']}, 输出类别: {model_args['num_classes']}")
    print(f"隐藏层: {model_args['hidden_dims']}, 激活函数: {model_args['activation']}")
    
    model = get_model(model_type, **model_args)
    
    # 加载权重
    model.load_state_dict(checkpoint['state_dict'], strict=strict)
    model = model.to(device)
    model.eval()
    
    print(f"模型架构重建并加载权重成功")
    
    # 打印训练信息摘要
    if 'training_info' in checkpoint:
        train_info = checkpoint['training_info']
        if 'last_epoch' in train_info:
            print(f"训练轮数: {train_info['last_epoch']}")
        if 'val_f1_macro_list' in train_info and len(train_info['val_f1_macro_list']) > 0:
            print(f"最佳验证F1分数: {max(train_info['val_f1_macro_list']):.4f}")
        if 'train_time' in train_info:
            print(f"训练时间: {train_info['train_time']:.2f} 秒")
    
    return model, checkpoint











    """
    加载模型并重建其架构
    
    参数:
        model_path: 模型文件路径
        model_class: 模型类
        model_args: 模型初始化参数（字典）
        device: 设备
        
    返回:
        model: 加载了权重的模型
        checkpoint: 完整的检查点数据
    """
    # 安全加载检查点
    checkpoint = safe_load_model(model_path, device)
    
    # 创建模型
    model = model_class(**model_args)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型架构重建并加载成功")
    return model, checkpoint