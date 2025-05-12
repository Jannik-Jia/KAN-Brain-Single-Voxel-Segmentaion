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