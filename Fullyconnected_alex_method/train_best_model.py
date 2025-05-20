#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用贝叶斯优化找到的最佳超参数训练完整模型，并确保正确的标准化
"""

import os
import sys
import torch
import numpy as np
import time
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append('.')

# 导入必要的模块
from config import load_config
from models import get_model
from data import BrainVoxelDataset, load_multiclass_data
from utils.metrics import calculate_class_weights, evaluate_model
from train import train_brain_voxel_mlp_multiclass

def train_best_model():
    """使用贝叶斯优化找到的最佳参数训练模型"""
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(666)
    np.random.seed(666)
    
    # 创建结果目录
    result_dir = "./best_model_results"
    os.makedirs(result_dir, exist_ok=True)
    log_dir = "./best_model_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置实验名称
    experiment_name = f"BrainVoxel_BestParams_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 设置数据目录
    data_dirs = {
        'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
        'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
        'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val"
    }
    
    # 加载原始数据（不进行标准化）
    print("加载原始数据集（不标准化）...")
    dataset_dict = load_multiclass_data(
        data_dirs,
        apply_pca_flag=False,
        n_components=0,
        norm=False  # 关闭内部标准化
    )
    
    # 提取原始样本数据
    train_samples = dataset_dict['train_samples']
    test_samples = dataset_dict['test_samples']
    val_samples = dataset_dict['val_samples']
    train_labels = dataset_dict['train_labels']
    test_labels = dataset_dict['test_labels']
    val_labels = dataset_dict['val_labels']
    
    # 创建并拟合 StandardScaler（仅使用训练数据）
    print("使用训练数据拟合 StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(train_samples)
    
    # 保存 scaler 以供将来使用
    scaler_path = os.path.join(result_dir, f"{experiment_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"已保存 StandardScaler 到: {scaler_path}")
    
    # 应用相同的 scaler 转换所有数据集
    print("应用相同的 scaler 转换所有数据集...")
    train_samples_scaled = scaler.transform(train_samples)
    test_samples_scaled = scaler.transform(test_samples)
    val_samples_scaled = scaler.transform(val_samples)
    
    # 创建数据集（使用标准化后的数据）
    train_dataset = BrainVoxelDataset(train_samples_scaled, train_labels)
    test_dataset = BrainVoxelDataset(test_samples_scaled, test_labels)
    val_dataset = BrainVoxelDataset(val_samples_scaled, val_labels)
    
    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 记录标准化统计信息
    normalization_params = {
        'mean': scaler.mean_.tolist(),
        'std': np.sqrt(scaler.var_).tolist()
    }
    
    # 将标准化参数保存为文本文件，方便查看
    with open(os.path.join(result_dir, f"{experiment_name}_normalization_params.txt"), 'w') as f:
        f.write("Normalization Parameters (StandardScaler)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of features: {len(normalization_params['mean'])}\n\n")
        f.write("Mean range: [{:.6f}, {:.6f}]\n".format(
            min(normalization_params['mean']), 
            max(normalization_params['mean'])
        ))
        f.write("Std range: [{:.6f}, {:.6f}]\n\n".format(
            min(normalization_params['std']), 
            max(normalization_params['std'])
        ))
        
        # 打印前10个特征的详细信息
        f.write("First 10 features details:\n")
        f.write("{:<5} {:<15} {:<15}\n".format("Idx", "Mean", "Std"))
        for i in range(min(10, len(normalization_params['mean']))):
            f.write("{:<5} {:<15.6f} {:<15.6f}\n".format(
                i, 
                normalization_params['mean'][i], 
                normalization_params['std'][i]
            ))
    
    # 创建模型 - 使用贝叶斯优化找到的最佳参数
    input_dim = dataset_dict['feature_dim']
    num_classes = 102  # 修改为您的实际类别数
    
    # 最佳参数
    hidden_dims = [2048] * 6  # 6层，每层2048单元
    dropout_rate = 0.2567125567148536
    activation = 'gelu'
    learning_rate = 0.0001697347212847238
    weight_decay = 0.00001328825998652095
    use_skip_connections = True
    
    # 创建模型
    model = get_model(
        model_type='deep_mlp',
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        activation=activation,
        use_skip_connections=use_skip_connections
    )
    model = model.to(device)
    
    # 计算类别权重
    class_weights = calculate_class_weights(train_labels, num_classes).to(device)
    
    # 创建损失函数
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 创建学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,  # 每2轮降低一次学习率
        gamma=0.22157302560547384  # 学习率乘以此因子
    )
    
    # 创建配置字典
    config = {
        'model_type': 'deep_mlp',
        'hidden_units': hidden_dims,
        'dropout_rate': dropout_rate,
        'activation': activation,
        'optimizer': 'adam',
        'lr': learning_rate,
        'weight_decay': weight_decay,
        'use_lr_scheduler': True,
        'lr_scheduler_type': 'step',
        'lr_step_size': 2,
        'lr_gamma': 0.22157302560547384,
        'use_skip_connections': use_skip_connections,
        'batch_size': batch_size,
        'epochs': 30,
        'num_class': num_classes,
        'feature_dim': input_dim,
        'experiment_name': experiment_name,
        'normalization_method': 'StandardScaler',
        'normalization_params': normalization_params
    }
    
    # 训练模型
    print(f"\n开始训练最佳模型...")
    print(f"模型类型: deep_mlp, 层数: 6, 每层单元: 2048")
    print(f"Dropout率: {dropout_rate}, 激活函数: {activation}")
    print(f"学习率: {learning_rate}, 权重衰减: {weight_decay}")
    print(f"总训练轮数: 30")
    
    training_results = train_brain_voxel_mlp_multiclass(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=30,  # 训练30轮
        val_epoch=1,    # 每轮验证一次
        save_path=result_dir,
        lr_scheduler=lr_scheduler,
        use_old_zipfile_serialization=True,
        experiment_name=experiment_name,
        config=config,  # 传递配置
        normalization_params=normalization_params  # 传递标准化参数
    )
    
    # 评估最佳模型
    print("\n训练完成，评估最终模型...")
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=result_dir,
        dataset_name="test_final",
        detailed=True,
        plot=True,
        disable_progress=False,
        show_class_metrics=True
    )
    
    # 打印最终结果
    print("\n训练和评估完成!")
    print(f"最终测试集准确率: {test_results['accuracy']:.4f}")
    print(f"最终测试集F1宏平均: {test_results['f1_macro']:.4f}")
    print(f"最终测试集Kappa系数: {test_results['kappa']:.4f}")
    print(f"所有结果已保存至: {result_dir}")
    print(f"标准化参数已保存至: {scaler_path}")
    print(f"使用以下命令将相同的标准化应用到demo38数据：")
    print(f"python predict.py --model <模型路径> --data_path <demo38路径> --normalize same_as_training")
    
    return training_results, test_results, scaler_path

if __name__ == "__main__":
    try:
        # 添加PyTorch安全全局变量
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception as e:
        print(f"添加安全全局变量失败: {e}")
    
    train_best_model()
