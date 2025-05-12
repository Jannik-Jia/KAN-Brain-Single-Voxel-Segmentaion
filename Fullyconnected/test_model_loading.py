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
    model_path = None
    
    # 查找最新模型文件
    model_files = [f for f in os.listdir(experiment_dir) if f.endswith(".pth")]
    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(experiment_dir, x)), reverse=True)
        model_path = os.path.join(experiment_dir, model_files[0])
    
    if not model_path:
        print("错误: 未找到模型文件")
        return
    
    print(f"使用模型文件: {model_path}")
    
    # 2. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 3. 使用新的加载函数
    try:
        from utils.model_io import load_model_with_architecture
        model, checkpoint = load_model_with_architecture(model_path, device)
        print("成功完整重建模型架构并加载权重")
        
        # 如果需要，可以打印模型架构信息
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            print("\n模型架构信息:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        
        # 创建数据加载器用于评估
        config_path = os.path.join(experiment_dir, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(experiment_dir, "optimized_config.json")
        
        config = load_config(config_path)
        
        # 4. 加载数据集
        print("\n加载数据集...")
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
        
        # 5. 评估模型
        print("\n开始模型评估...")
        model.eval()
        
        from utils.metrics import evaluate_model
        
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
        
        # 打印结果摘要
        print("\n评估结果摘要:")
        print(f"训练集 - 准确率: {train_results['accuracy']:.4f}, F1宏平均: {train_results['f1_macro']:.4f}")
        print(f"验证集 - 准确率: {val_results['accuracy']:.4f}, F1宏平均: {val_results['f1_macro']:.4f}")
        print(f"测试集 - 准确率: {test_results['accuracy']:.4f}, F1宏平均: {test_results['f1_macro']:.4f}")
        
    except Exception as e:
        print(f"模型加载或评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()