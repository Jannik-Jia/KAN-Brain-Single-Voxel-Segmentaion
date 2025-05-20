#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估函数
"""

import os
import sys
import torch
import numpy as np
from utils.metrics import evaluate_model

# 添加导入
from data.mat_loader import load_external_mat_data, BrainVoxelMatDataset

# 为保持向后兼容性的包装函数
def evaluate_model_detailed(model, data_loader, device, result_path, dataset_name="test", class_names=None):
    """
    使用统一评估函数的包装，保持向后兼容性
    """
    return evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        result_path=result_path,
        dataset_name=dataset_name,
        class_names=class_names,
        detailed=True,
        plot=True
    )

def load_mat_evaluation_data(config, model_path):
    """从mat文件加载评估数据"""
    # 首先尝试加载与模型一起保存的scaler
    try:
        from utils.model_io import load_model_with_architecture
        _, _, scaler = load_model_with_architecture(
            model_path=model_path,
            device='cpu',  # 只需要scaler，不需要模型
            load_scaler=True
        )
    except Exception as e:
        print(f"加载scaler时出错: {e}")
        scaler = None
        
    # 然后从配置的路径加载mat数据
    if config.get('mat_file_path'):
        mat_path = config['mat_file_path']
    elif config.get('demo_mat_path'):
        mat_path = config['demo_mat_path']
    else:
        raise ValueError("配置中缺少'mat_file_path'或'demo_mat_path'，请指定要评估的mat文件路径")
    
    print(f"从路径加载mat评估数据: {mat_path}")
    data_dict = load_external_mat_data(
        mat_file_path=mat_path,
        scaler=scaler
    )
    
    # 创建数据集和加载器
    from torch.utils.data import DataLoader
    
    if data_dict['labels'] is not None:
        # 如果有标签，创建评估数据集
        dataset = BrainVoxelMatDataset(data_dict['data'], data_dict['labels'])
        loader = DataLoader(dataset, batch_size=config.get('batch_size', 128), shuffle=False)
        
        return dataset, loader
    else:
        # 如果没有标签，只能做前向传播，不能评估
        print("警告: 加载的mat文件没有标签数据，无法评估模型性能")
        return None, None

def predict_with_mat_data(model, mat_file_path, scaler=None, device='cpu', batch_size=128, output_path=None):
    """
    使用模型对mat文件数据进行预测
    
    参数:
        model: 训练好的模型
        mat_file_path: mat文件路径
        scaler: 标准化器
        device: 设备
        batch_size: 批处理大小
        output_path: 输出路径，None表示不保存
        
    返回:
        predictions: 预测结果
    """
    # 加载数据
    data_dict = load_external_mat_data(
        mat_file_path=mat_file_path,
        scaler=scaler
    )
    
    # 获取数据
    data = data_dict['data']
    
    # 转换为torch张量并预测
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 批量处理，避免内存不足
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = torch.FloatTensor(data[start_idx:end_idx]).to(device)
            
            outputs = model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())
    
    # 合并所有批次的预测结果
    all_predictions = np.vstack(predictions)
    
    # 保存预测结果
    if output_path:
        import scipy.io
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scipy.io.savemat(output_path, {'predicted_ann': all_predictions})
        print(f"预测结果已保存至: {output_path}")
    
    return all_predictions