#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用贝叶斯优化找到的最佳超参数训练完整模型，并适配方法1的数据结构
"""

import os
import sys
import torch
import numpy as np
import time
import joblib
import random

# 添加项目根目录到路径
sys.path.append('.')

# 导入必要的模块
from config import load_config
from models import get_model
from utils.metrics import calculate_class_weights, evaluate_model
from train import train_brain_voxel_mlp_multiclass
from data_loader import load_and_prepare_data, load_scaler

def train_best_model():
    """使用贝叶斯优化找到的最佳参数训练模型，适配方法1的数据结构"""
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(666)
    np.random.seed(666)
    random.seed(666)
    
    # 创建结果目录
    result_dir = "./best_model_results"
    os.makedirs(result_dir, exist_ok=True)
    log_dir = "./best_model_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置实验名称
    experiment_name = f"BrainVoxel_BestParams_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 设置数据目录 - 使用方法1的数据路径
    data_base_dir = "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/processed_data"
    data_format = 'mat'  # 或 'npy'，取决于您的偏好
    batch_size = 128
    
    # 加载数据和创建数据加载器
    print("\n准备数据...")
    train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(
        base_dir=data_base_dir,
        format=data_format,
        batch_size=batch_size,
        shuffle=True,
        seed=666
    )
    
    # 保存scaler以供将来使用
    scaler_path = os.path.join(result_dir, f"{experiment_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"已保存StandardScaler到: {scaler_path}")
    
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
    input_dim = feature_dim
    
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
    
    # 计算类别权重 - 需要提取训练集的标签
    # 从train_loader中提取第一批次的标签和统计总体分布
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)
    
    class_weights = calculate_class_weights(all_labels, num_classes).to(device)
    
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
        'normalization_params': normalization_params,
        'data_method': 'method1',  # 标记使用方法1的数据结构
        'data_format': data_format
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