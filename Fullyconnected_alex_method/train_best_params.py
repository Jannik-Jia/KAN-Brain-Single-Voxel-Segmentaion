#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用最优超参数和MAT数据格式进行训练
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
import joblib

# 确保能够导入主项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试添加numpy类型到PyTorch安全全局变量
try:
    # 添加可能需要的numpy类型到安全全局变量列表
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
    print("已添加numpy类型到PyTorch安全全局变量列表")
except Exception as e:
    print(f"添加安全全局变量时出错 (可忽略): {e}")

# 导入自定义模块
from data.mat_loader import process_train38_data, create_dataloaders_from_mat
from models import get_model
from train import train_brain_voxel_mlp_multiclass
from utils.metrics import evaluate_model
from utils.model_io import save_model_with_architecture

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用最优超参数训练脑体素MLP')
    
    # 必要参数
    parser.add_argument('--mat_file_path', type=str, required=True, help='TRAIN38.mat文件路径')
    parser.add_argument('--experiment_name', type=str, default=f"BrainVoxel_BestParams_{time.strftime('%Y%m%d_%H%M%S')}", help='实验名称')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存目录')
    parser.add_argument('--device', type=int, default=0, help='使用的设备（-1表示CPU）')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--test_size', type=float, default=0.01, help='测试集比例')
    parser.add_argument('--seed', type=int, default=666, help='随机种子')
    
    return parser.parse_args()

def setup_environment(seed):
    """设置环境，包括随机种子和设备"""
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子设置为: {seed}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    setup_environment(args.seed)
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 记录超参数
    with open(os.path.join(experiment_dir, "best_hyperparameters.txt"), "w") as f:
        f.write("最优超参数(Trial 82):\n")
        f.write("learning_rate: 9.247073789581185e-05\n")
        f.write("weight_decay: 1.041622803083192e-06\n")
        f.write("optimizer: adamw\n")
        f.write("dropout_rate: 0.1840571430245583\n")
        f.write("activation: gelu\n")
        f.write("lr_scheduler: step\n")
        f.write("model_type: base_mlp\n")
        f.write("layer_sizes_idx: 0 (corresponds to [4096, 4096, 4096, 4096])\n")
        f.write("step_size: 4\n")
        f.write("step_gamma: 0.16254869619779228\n")
    
    # 加载数据
    print("加载数据...")
    scaler_save_path = os.path.join(experiment_dir, "scaler.joblib")
    dataset_dict = process_train38_data(
        mat_file_path=args.mat_file_path,
        test_size=args.test_size,
        random_state=args.seed,
        scaler_save_path=scaler_save_path
    )
    
    # 创建数据加载器
    dataloaders = create_dataloaders_from_mat(
        dataset_dict,
        batch_size=args.batch_size,
        shuffle_train=True
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # 创建模型 - 使用最优参数
    input_dim = dataset_dict['feature_dim']
    num_classes = dataset_dict['num_classes']
    hidden_dims = [4096, 4096, 4096, 4096]  # 这对应于layer_sizes_idx=0
    
    print(f"创建模型: base_mlp, 隐藏层: {hidden_dims}")
    model = get_model(
        model_type='base_mlp',
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=0.1840571430245583,  # 最优dropout_rate
        activation='gelu'                 # 最优activation
    )
    model = model.to(device)
    
    # 创建优化器 - 使用最优参数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=9.247073789581185e-05,        # 最优learning_rate
        weight_decay=1.041622803083192e-06  # 最优weight_decay
    )
    
    # 创建学习率调度器 - 使用最优参数
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=4,                      # 最优step_size
        gamma=0.16254869619779228         # 最优step_gamma
    )
    
    # 创建损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # 训练配置
    config = {
        'model_type': 'base_mlp',
        'feature_dim': input_dim,
        'num_class': num_classes,
        'hidden_units': hidden_dims,
        'dropout_rate': 0.1840571430245583,
        'activation': 'gelu',
        'optimizer': 'adamw',
        'lr': 9.247073789581185e-05,
        'weight_decay': 1.041622803083192e-06,
        'lr_scheduler_type': 'step',
        'step_size': 4,
        'step_gamma': 0.16254869619779228,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'random_seed': args.seed,
        'experiment_name': args.experiment_name,
        'scaler': dataset_dict['scaler']
    }
    
    # 保存配置
    import json
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        # 过滤掉不可JSON序列化的项
        json_safe_config = {k: v for k, v in config.items() if k != 'scaler'}
        json.dump(json_safe_config, f, indent=4)
    
    # 训练模型
    print(f"\n开始训练模型...")
    print(f"训练轮数: {args.epochs}, 批大小: {args.batch_size}")
    training_results = train_brain_voxel_mlp_multiclass(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        val_epoch=1,  # 每个epoch都验证
        save_path=experiment_dir,
        lr_scheduler=lr_scheduler,
        use_old_zipfile_serialization=True,
        experiment_name=args.experiment_name,
        config=config,
        normalization_params=None
    )
    
    # 在测试集上评估
    print("\n在测试集上评估最终模型...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=experiment_dir,
        dataset_name="test",
        detailed=True,
        plot=True,
        disable_progress=False
    )
    
    # 保存最终结果
    with open(os.path.join(experiment_dir, "final_results.txt"), "w") as f:
        f.write(f"测试集性能指标:\n")
        f.write(f"准确率: {test_results['accuracy']:.6f}\n")
        f.write(f"平衡准确率: {test_results['balanced_accuracy']:.6f}\n")
        f.write(f"宏平均F1: {test_results['f1_macro']:.6f}\n")
        f.write(f"加权F1: {test_results['f1_weighted']:.6f}\n")
        f.write(f"Kappa系数: {test_results['kappa']:.6f}\n")
    
    print(f"\n训练和评估完成!")
    print(f"最终测试集F1分数: {test_results['f1_macro']:.6f}")
    print(f"所有结果已保存至: {experiment_dir}")

if __name__ == "__main__":
    main()