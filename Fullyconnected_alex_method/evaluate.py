#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本，从已训练的模型继续评估流程
"""

import os
import sys
import json
import argparse
import torch
import torch.serialization
import numpy as np
from data.mat_loader import load_external_mat_data, BrainVoxelMatDataset

# 设置PyTorch序列化安全变量 - 添加这部分
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
    print("将尝试在需要时再添加安全全局变量")

    
from torch.utils.data import DataLoader

# 导入自定义模块
from config import load_config
from models import get_model
from data import BrainVoxelDataset, load_multiclass_data
from utils.metrics import evaluate_model
from utils.visualization import visualize_dataset_distribution


try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    print("已添加 numpy.core.multiarray.scalar 到安全全局变量列表")
except Exception as e:
    print(f"添加安全全局变量时出错: {e}")
    print("将尝试使用 weights_only=False 加载模型")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BrainVoxel MLP Evaluation')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--experiment_dir', type=str, required=True, help='实验结果目录')
    parser.add_argument('--model_path', type=str, required=True, help='已训练模型的路径')
    parser.add_argument('--device', type=int, default=0, help='使用的设备（-1表示CPU）')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default=None, help='训练数据目录')
    parser.add_argument('--test_dir', type=str, default=None, help='测试数据目录')
    parser.add_argument('--val_dir', type=str, default=None, help='验证数据目录')
    parser.add_argument('--mat_file_path', type=str, default=None, 
                        help='评估用的mat文件路径')
    parser.add_argument('--use_mat_format', action='store_true',
                        help='是否使用mat文件格式')
    
    # 其他参数
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    
    return parser.parse_args()

def setup_environment(config):
    """设置环境，包括随机种子和设备"""
    import random
    import torch
    
    # 设置随机种子
    random.seed(config.get('random_seed', 666))
    torch.manual_seed(config.get('random_seed', 666))
    torch.cuda.manual_seed(config.get('random_seed', 666))
    torch.cuda.manual_seed_all(config.get('random_seed', 666))
    np.random.seed(config.get('random_seed', 666))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置设备
    device = torch.device(f"cuda:{config.get('device', 0)}" if config.get('device', 0) >= 0 and torch.cuda.is_available() else "cpu")
    
    print(f"随机种子设置为: {config.get('random_seed', 666)}")
    print(f"使用设备: {device}")
    
    return device


def convert_onehot_to_indices(labels_onehot):
    """
    将one-hot编码的标签转换为索引
    
    参数:
        labels_onehot: One-hot编码的标签，形状为(n_samples, num_classes)
        
    返回:
        labels_indices: 类别索引，形状为(n_samples,)
    """
    return np.argmax(labels_onehot, axis=1)



def load_datasets(config):
    """加载数据集"""
    print("开始加载数据集...")
    
    # 确保数据目录存在且不为None
    if 'data_dirs' not in config or not all(key in config['data_dirs'] and config['data_dirs'][key] is not None 
                                         for key in ['train_dir', 'test_dir', 'val_dir']):
        print("配置中数据目录不完整，尝试使用命令行参数中的路径...")
        
        # 使用命令行参数中的路径重建data_dirs
        config['data_dirs'] = {
            'train_dir': config.get('train_dir'),
            'test_dir': config.get('test_dir'),
            'val_dir': config.get('val_dir')
        }
    
    # 最终检查确保所有路径都存在
    for key, path in config['data_dirs'].items():
        if path is None:
            raise ValueError(f"错误: {key} 路径为None，请检查配置文件或命令行参数")
        if not os.path.exists(path):
            raise ValueError(f"错误: {key} 路径不存在: {path}")
        print(f"使用 {key}: {path}")
    
    # 继续原有的数据加载流程...
    dataset_dict = load_multiclass_data(
        config['data_dirs'],
        apply_pca_flag=config.get('apply_pca', False),
        n_components=config.get('n_pca', 0),
        norm=config.get('norm', True)
    )
    
    # 创建数据集
    train_dataset = BrainVoxelDataset(dataset_dict['train_samples'], dataset_dict['train_labels'])
    test_dataset = BrainVoxelDataset(dataset_dict['test_samples'], dataset_dict['test_labels'])
    val_dataset = BrainVoxelDataset(dataset_dict['val_samples'], dataset_dict['val_labels'])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
    
    print(f"数据加载完成! 共载入 {len(train_dataset)} 个训练样本，{len(val_dataset)} 个验证样本，{len(test_dataset)} 个测试样本")
    print(f"特征维度: {dataset_dict['feature_dim']}")
    
    return dataset_dict, train_loader, val_loader, test_loader




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
    if not config.get('mat_file_path'):
        raise ValueError("配置中缺少'mat_file_path'，请指定要评估的mat文件路径")
    
    data_dict = load_external_mat_data(
        mat_file_path=config['mat_file_path'],
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
    
    

def load_model_from_checkpoint(model_path, device):
    """从检查点加载模型，使用新的加载函数"""
    print(f"从检查点加载模型: {model_path}")
    
    # 使用新的加载函数
    from utils.model_io import load_model_with_architecture
    
    try:
        model, checkpoint = load_model_with_architecture(model_path, device)
        print("成功完整重建模型架构并加载权重")
        
        # 打印模型架构信息
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            print("\n模型架构信息:")
            for key, value in model_info.items():
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    value = f"{value[:5]}...共{len(value)}项"
                print(f"  {key}: {value}")
        
        return model, checkpoint
    except Exception as e:
        print(f"使用新方法加载模型失败: {e}, 尝试旧方法")
        
        # 如果新方法失败，回退到旧方法
        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 从配置文件获取模型参数
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(model_path), "optimized_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件，无法重建模型架构")
        
        # 加载配置
        from config import load_config
        config = load_config(config_path)
        
        # 创建模型
        from models import get_model
        model = get_model(
            model_type=config.get('model_type', 'base_mlp'),
            input_dim=config.get('feature_dim', 0),
            hidden_dims=config.get('hidden_units', [4096, 4096, 4096, 4096]),
            num_classes=config.get('num_class', 102),
            dropout_rate=config.get('dropout_rate', 0.5),
            activation=config.get('activation', 'relu')
        )
        
        # 加载模型状态
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"使用旧方法成功加载模型")
        
        return model, checkpoint
    
def evaluate_loaded_model(model, train_loader, val_loader, test_loader, device, result_path):
    """评估已加载的模型"""
    print("\n开始评估模型...")
    
    # 在训练集上评估
    print("\n在训练集上评估...")
    train_results = evaluate_model(
        model=model,
        data_loader=train_loader,
        device=device,
        result_path=result_path,
        dataset_name="train",
        detailed=True,
        plot=True,
        disable_progress=True,
        show_class_metrics=True  # 显示每个标签的指标
    )

    # 在验证集上评估
    print("\n在验证集上评估...")
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        result_path=result_path,
        dataset_name="val",
        detailed=True,
        plot=True,
        disable_progress=True,
        show_class_metrics=True  # 显示每个标签的指标
    )

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        result_path=result_path,
        dataset_name="test",
        detailed=True,
        plot=True,
        disable_progress=True,
        show_class_metrics=True  # 显示每个标签的指标
    )
    
    # 保存评估结果摘要
    summary_path = os.path.join(result_path, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("评估结果摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write("训练集结果:\n")
        f.write(f"  准确率: {train_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {train_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {train_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {train_results['kappa']:.4f}\n\n")
        
        f.write("验证集结果:\n")
        f.write(f"  准确率: {val_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {val_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {val_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {val_results['kappa']:.4f}\n\n")
        
        f.write("测试集结果:\n")
        f.write(f"  准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"  宏平均F1: {test_results['f1_macro']:.4f}\n")
        f.write(f"  平衡准确率: {test_results['balanced_accuracy']:.4f}\n")
        f.write(f"  Kappa系数: {test_results['kappa']:.4f}\n\n")
        
    print(f"评估结果摘要已保存至: {summary_path}")
    
    return train_results, val_results, test_results

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 校验路径参数
    if not os.path.exists(args.experiment_dir):
        print(f"错误: 实验目录 {args.experiment_dir} 不存在")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        sys.exit(1)
    
    # 加载配置
    config_path = args.config
    if config_path is None:
        # 尝试从实验目录加载配置
        possible_configs = [
            os.path.join(args.experiment_dir, "optimized_config.json"),
            os.path.join(args.experiment_dir, "config.json")
        ]
        for path in possible_configs:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        print("错误: 无法找到有效的配置文件")
        sys.exit(1)
    
    # 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 用命令行参数覆盖配置
    if args.train_dir is not None:
        config['train_dir'] = args.train_dir
    if args.test_dir is not None:
        config['test_dir'] = args.test_dir
    if args.val_dir is not None:
        config['val_dir'] = args.val_dir

        
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.device is not None:
        config['device'] = args.device
    
    # 设置环境
    device = setup_environment(config)
    
    # 加载数据集
    dataset_dict, train_loader, val_loader, test_loader = load_datasets(config)
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    try:
        # 从配置中获取模型参数
        model_type = config.get('model_type', 'base_mlp')
        hidden_dims = config.get('hidden_units', [4096, 4096, 4096, 4096])
        dropout_rate = config.get('dropout_rate', 0.5)
        activation = config.get('activation', 'relu')
        
        # 加载模型
        model, checkpoint = load_model_from_checkpoint(
            model_path=args.model_path,
            model_type=model_type,
            input_dim=dataset_dict['feature_dim'],
            hidden_dims=hidden_dims,
            num_classes=config.get('num_class', 102),
            dropout_rate=dropout_rate,
            activation=activation,
            device=device
        )
        
        # 评估模型
        train_results, val_results, test_results = evaluate_loaded_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            result_path=args.experiment_dir
        )
        
        # 输出最终结果
        print("\n评估完成!")
        print(f"测试集宏平均F1: {test_results['f1_macro']:.4f}")
        print(f"测试集准确率: {test_results['accuracy']:.4f}")
        print(f"测试集Kappa: {test_results['kappa']:.4f}")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()