#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型加载和保存工具函数
"""

import torch
import os
import numpy as np

def safe_load_model(model_path, device='cpu'):
    """
    安全加载模型，处理 PyTorch 版本兼容性问题
    
    参数:
        model_path: 模型文件路径
        device: 设备（'cpu' 或 'cuda:0' 等）
        
    返回:
        checkpoint: 加载的模型检查点
    """
    print(f"安全加载模型: {model_path}")
    
    # 首先尝试使用 weights_only=False
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("成功使用 weights_only=False 加载模型")
        return checkpoint
    except Exception as e:
        print(f"使用 weights_only=False 加载失败: {e}")
    
    # 尝试添加安全全局变量
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(model_path, map_location=device)
        print("成功使用安全全局变量加载模型")
        return checkpoint
    except Exception as e:
        print(f"使用安全全局变量加载失败: {e}")
    
    # 最后尝试使用默认设置
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("成功使用默认设置加载模型")
        return checkpoint
    except Exception as e:
        print(f"所有加载方法都失败: {e}")
        raise RuntimeError(f"无法加载模型文件 {model_path}，请检查文件格式或 PyTorch 版本兼容性")

def load_model_with_architecture(model_path, model_class, model_args, device='cpu'):
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