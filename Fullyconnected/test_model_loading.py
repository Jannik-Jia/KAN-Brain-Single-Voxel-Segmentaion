#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型加载测试脚本 - 使用现有代码结构
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 确保能找到项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入已有的配置、模型和数据加载函数
from config import load_config
from models import get_model
from data import BrainVoxelDataset, load_multiclass_data
from utils.metrics import evaluate_model
from utils.model_io import safe_load_model  # 这里使用你现有的函数

# 修复numpy数据类型加载问题
def setup_torch_loading():
    """设置PyTorch加载兼容性"""
    try:
        # 添加所需的numpy类型到安全全局变量
        safe_globals = [
            np.dtype,
            np.core.multiarray.scalar,
            np.ndarray,
            np.generic,
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            np.bool_
        ]
        for sg in safe_globals:
            try:
                torch.serialization.add_safe_globals([sg])
                print(f"成功添加: {sg}")
            except Exception as e:
                print(f"添加 {sg} 失败: {e}")
        
        print("PyTorch模型加载兼容性设置完成")
        return True
    except Exception as e:
        print(f"设置PyTorch加载兼容性时出错: {e}")
        return False

def main():
    """主函数"""
    # 设置PyTorch加载兼容性
    setup_torch_loading()
    
    # 1. 加载配置文件
    experiment_dir = "./results/BrainVoxel_102Class_MLP_20250511_192244"
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(experiment_dir, "optimized_config.json")
    
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 2. 设置设备
    device = torch.device(f"cuda:{config['device']}" if config['device'] >= 0 and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 3. 加载数据集
    print("加载数据集...")
    dataset_dict = load_multiclass_data(
        config['data_dirs'],
        apply_pca_flag=config['apply_pca'],
        n_components=config['n_pca'],
        norm=config['norm']
    )
    
    # 创建数据集
    train_dataset = BrainVoxelDataset(dataset_dict['train_samples'], dataset_dict['train_labels'])
    test_dataset = BrainVoxelDataset(dataset_dict['test_samples'], dataset_dict['test_labels'])
    val_dataset = BrainVoxelDataset(dataset_dict['val_samples'], dataset_dict['val_labels'])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 4. 创建模型
    print("创建模型...")
    if config['model_type'] == 'deep_mlp':
        # 深层MLP特殊处理
        hidden_dims = [config.get('width_factor', 2048)] * config.get('depth', 6)
    else:
        hidden_dims = config['hidden_units']
    
    model = get_model(
        model_type=config['model_type'],
        input_dim=dataset_dict['feature_dim'],
        hidden_dims=hidden_dims,
        num_classes=config['num_class'],
        dropout_rate=config['dropout_rate'],
        activation=config['activation']
    )
    model = model.to(device)
    
    # 5. 查找并加载模型
    print("查找最佳模型...")
    model_files = [f for f in os.listdir(experiment_dir) if f.endswith(".pth")]
    if not model_files:
        print(f"错误: 在 {experiment_dir} 中没有找到模型文件")
        return
    
    # 按照修改时间排序，最新的在前
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(experiment_dir, x)), reverse=True)
    model_path = os.path.join(experiment_dir, model_files[0])
    print(f"选择模型文件: {model_path}")
    
    # 6. 加载模型
    print("加载模型...")
    try:
        # 使用你现有的safe_load_model函数
        checkpoint = safe_load_model(model_path, device)
        model.load_state_dict(checkpoint['state_dict'])
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将使用未训练的模型继续评估")
    
    # 7. 评估模型
    print("\n开始模型评估...")
    model.eval()
    
    # 在训练集上评估
    print("\n在训练集上评估...")
    train_results = evaluate_model(
        model=model,
        data_loader=train_loader,
        device=device,
        result_path=experiment_dir,
        dataset_name="train_test",
        detailed=True,
        plot=True,
        disable_progress=False,
        show_class_metrics=True
    )

    # 在验证集上评估
    print("\n在验证集上评估...")
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        result_path=experiment_dir,
        dataset_name="val_test",
        detailed=True,
        plot=True,
        disable_progress=False,
        show_class_metrics=True
    )

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=experiment_dir,
        dataset_name="test_test",
        detailed=True,
        plot=True,
        disable_progress=False,
        show_class_metrics=True
    )
    
    # 8. 打印结果摘要
    print("\n评估结果摘要:")
    print(f"训练集 - 准确率: {train_results['accuracy']:.4f}, F1宏平均: {train_results['f1_macro']:.4f}")
    print(f"验证集 - 准确率: {val_results['accuracy']:.4f}, F1宏平均: {val_results['f1_macro']:.4f}")
    print(f"测试集 - 准确率: {test_results['accuracy']:.4f}, F1宏平均: {test_results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()