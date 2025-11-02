#!/usr/bin/env python
# coding: utf-8

"""
模型模块
Models module

支持的模型:
- RegModel: 深度全连接神经网络（Alex identical structure）
- ResNetMLP: 带残差连接的全连接神经网络
- SimpleMLP: 简单多层感知机
- BrainVoxelKAN: 基于KAN的体素分类模型
"""

from .reg_model import RegModel, get_model_config as get_reg_model_config
from .resnet_model import ResNetMLP, get_model_config as get_resnet_model_config
from .simple_mlp import SimpleMLP, get_model_config as get_simple_mlp_config
from .kan_model import BrainVoxelKAN, get_model_config as get_kan_model_config
from .deep_mlp import DeepMLP, get_model_config as get_deep_mlp_config


# 模型注册表
MODEL_REGISTRY = {
    'reg_model': {
        'class': RegModel,
        'config_fn': get_reg_model_config
    },
    'resnet_mlp': {
        'class': ResNetMLP,
        'config_fn': get_resnet_model_config
    },
    'simple_mlp': {
        'class': SimpleMLP,
        'config_fn': get_simple_mlp_config
    },
    'kan': {
        'class': BrainVoxelKAN,
        'config_fn': get_kan_model_config
    },
    'deep_mlp': {
        'class': DeepMLP,
        'config_fn': get_deep_mlp_config
    }
}


def get_model(model_name, input_dim, num_classes, **kwargs):
    """
    根据模型名称获取模型实例

    Parameters:
    -----------
    model_name : str
        模型名称 ('reg_model', 'resnet_mlp', 'simple_mlp')
    input_dim : int
        输入特征维度
    num_classes : int
        输出类别数
    **kwargs : dict
        模型特定参数

    Returns:
    --------
    model : nn.Module
        模型实例
    model_config : dict
        模型配置信息
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"未知的模型名称: {model_name}. 可用模型: {available_models}")

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info['class']
    config_fn = model_info['config_fn']

    # 获取默认配置
    default_config = config_fn()

    # 合并用户提供的参数
    model_params = {
        'input_dim': input_dim,
        'num_classes': num_classes
    }

    # 对于reg_model，使用特定参数名
    if model_name == 'reg_model':
        model_params['hidden_dim'] = kwargs.get('hidden_dim', default_config['hidden_dim'])
        model_params['num_hidden_layers'] = kwargs.get('num_hidden_layers',
                                                       default_config['num_hidden_layers'])
        model_params['dropout_rate'] = kwargs.get('dropout_rate', default_config['dropout_rate'])

    # 对于resnet_mlp，使用特定参数名
    elif model_name == 'resnet_mlp':
        model_params['hidden_dim'] = kwargs.get('hidden_dim', default_config['hidden_dim'])
        model_params['num_residual_blocks'] = kwargs.get('num_residual_blocks',
                                                         default_config['num_residual_blocks'])
        model_params['dropout_rate'] = kwargs.get('dropout_rate', default_config['dropout_rate'])

    # 对于simple_mlp，使用特定参数名
    elif model_name == 'simple_mlp':
        model_params['hidden_dims'] = kwargs.get('hidden_dims', default_config['hidden_dims'])
        model_params['dropout_rate'] = kwargs.get('dropout_rate', default_config['dropout_rate'])

    # 对于kan，使用特定参数名
    elif model_name == 'kan':
        model_params['hidden_dims'] = kwargs.get('hidden_dims', default_config['hidden_dims'])
        model_params['grid_size'] = kwargs.get('grid_size', default_config['grid_size'])

    # 对于deep_mlp，使用特定参数名
    elif model_name == 'deep_mlp':
        model_params['hidden_dims'] = kwargs.get('hidden_dims', default_config['hidden_dims'])
        model_params['use_feature_interaction'] = kwargs.get('use_feature_interaction',
                                                             default_config['use_feature_interaction'])
        model_params['use_trilinear'] = kwargs.get('use_trilinear', default_config['use_trilinear'])
        model_params['use_residual'] = kwargs.get('use_residual', default_config['use_residual'])
        model_params['use_self_attention'] = kwargs.get('use_self_attention',
                                                        default_config['use_self_attention'])
        model_params['num_attn_heads'] = kwargs.get('num_attn_heads', default_config['num_attn_heads'])
        model_params['attn_layers'] = kwargs.get('attn_layers', default_config['attn_layers'])
        model_params['use_mixed_activation'] = kwargs.get('use_mixed_activation',
                                                          default_config['use_mixed_activation'])
        model_params['use_shake_shake'] = kwargs.get('use_shake_shake', default_config['use_shake_shake'])
        model_params['stochastic_depth_rate'] = kwargs.get('stochastic_depth_rate',
                                                           default_config['stochastic_depth_rate'])
        model_params['dropout_rate'] = kwargs.get('dropout_rate', default_config['dropout_rate'])

    # 创建模型
    model = model_class(**model_params)

    # 更新配置信息
    config = default_config.copy()
    config.update(model_params)

    return model, config


def list_available_models():
    """
    列出所有可用的模型

    Returns:
    --------
    list : 可用模型名称列表
    """
    return list(MODEL_REGISTRY.keys())


def get_model_description(model_name):
    """
    获取模型描述

    Parameters:
    -----------
    model_name : str
        模型名称

    Returns:
    --------
    str : 模型描述
    """
    if model_name not in MODEL_REGISTRY:
        return None

    config_fn = MODEL_REGISTRY[model_name]['config_fn']
    config = config_fn()
    return config.get('description', 'No description available')


__all__ = [
    'RegModel',
    'ResNetMLP',
    'SimpleMLP',
    'BrainVoxelKAN',
    'DeepMLP',
    'get_model',
    'list_available_models',
    'get_model_description',
    'MODEL_REGISTRY'
]
