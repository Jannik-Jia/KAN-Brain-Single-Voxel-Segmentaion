#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段二：基线模型训练脚本
使用特征工程后的数据训练基线模型
"""

import os
import argparse
import json
import sys
import time
import numpy as np
import torch
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from models.baseline import BaselineMLP, FeatureGroupMLP, DeepMLP
from training.baseline_trainer import BaselineTrainer
from evaluation.group_evaluator import GroupEvaluator
from utils.logging_utils import Logger

def load_h5_data(file_path):
    """加载HDF5格式的数据"""
    import h5py
    data_dict = {}
    
    print(f"开始加载数据文件: {file_path}")
    with h5py.File(file_path, 'r') as f:
        # 打印H5文件的所有组
        print("H5文件组结构:")
        def visit_for_debug(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  数据集: {name}, 形状: {obj.shape}, 类型: {obj.dtype}")
            else:
                print(f"  组: {name}")
        f.visititems(visit_for_debug)
        
        # 原来的加载逻辑
        def visit_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                parts = name.split('/')
                current_dict = data_dict
                for i, part in enumerate(parts[:-1]):
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                current_dict[parts[-1]] = obj[()]
                print(f"已加载数据集: {name}")
        
        f.visititems(visit_group)
    
    # 打印data_dict的结构
    print("加载后的数据字典结构:")
    def print_dict_structure(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                print_dict_structure(v, prefix + "  ")
            else:
                print(f"{prefix}{k}: 形状={v.shape if hasattr(v, 'shape') else '标量'}")
    
    print_dict_structure(data_dict)
    return data_dict

def main():
    """基线模型训练主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脑MRI数据基线模型训练')
    parser.add_argument('--config', type=str, default='configs/base_config.json', help='配置文件路径')
    parser.add_argument('--selected_features', type=str, default=None, help='特征选择结果文件路径')
    parser.add_argument('--transformed_features', type=str, default=None, help='特征变换结果文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，若不指定则使用配置文件中的设置')
    parser.add_argument('--model_type', type=str, default='deep_mlp', 
                       choices=['mlp', 'group_mlp', 'deep_mlp'], help='模型类型')
    parser.add_argument('--feature_type', type=str, default='selected', 
                       choices=['original', 'selected', 'pca', 'combined'], help='使用的特征类型')
    parser.add_argument('--feature_selection', type=str, default=None,
                       choices=['rf', 'mi', 'permutation', 'combined'], help='使用的特征选择方法')
    parser.add_argument('--feature_subset', type=str, default=None,
                       help='特征子集名称，如 "importance_90pct", "top_50" 等')
    args = parser.parse_args()
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger_manager = Logger("Stage2_BaselineTraining", log_dir="logs/stage2")
    logger = logger_manager.get_logger()
    
    logger.info("="*80)
    logger.info("阶段二：开始基线模型训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"特征类型: {args.feature_type}")
    if args.feature_selection:
        logger.info(f"特征选择方法: {args.feature_selection}")
    if args.feature_subset:
        logger.info(f"特征子集: {args.feature_subset}")
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("成功加载配置文件")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置输出目录
    base_output_dir = args.output_dir or config.get('paths', {}).get('results_dir', 'results')
    output_dir = os.path.join(base_output_dir, 'baseline', f"{args.model_type}_{args.feature_type}_{timestamp}")
    if args.feature_selection:
        output_dir = output_dir.replace(args.feature_type, f"{args.feature_type}_{args.feature_selection}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 记录实验开始
    experiment_name = f"阶段二：{args.model_type}基线模型训练({args.feature_type}特征)"
    if args.feature_selection:
        experiment_name += f" 使用{args.feature_selection}特征选择"
    logger_manager.log_experiment_start(experiment_name, f"时间戳: {timestamp}")
    
    start_time = time.time()
    
    # 步骤1：加载特征数据
    logger.info("步骤1: 加载特征数据...")
    features_path = None
    
    if args.feature_type == 'selected' and args.selected_features:
        features_path = args.selected_features
        logger.info(f"使用特征选择结果: {features_path}")
        data_dict = load_h5_data(features_path)
    elif args.feature_type == 'pca' and args.transformed_features:
        features_path = args.transformed_features
        logger.info(f"使用特征变换结果: {features_path}")
        data_dict = load_h5_data(features_path)
    elif args.feature_type == 'combined':
        # 如果使用组合特征，同时加载选择和变换的特征
        if args.selected_features and args.transformed_features:
            selected_data = load_h5_data(args.selected_features)
            transformed_data = load_h5_data(args.transformed_features)
            data_dict = selected_data
            logger.info(f"使用组合特征 (特征选择+PCA变换)")
        else:
            logger.error("组合特征模式需要同时提供特征选择和特征变换结果路径")
            return
    else:
        # 默认使用预处理后的原始特征
        features_path = config.get('paths', {}).get('processed_data_dir', 'data/processed')
        features_path = os.path.join(features_path, 'preprocessed_data.h5')
        if not os.path.exists(features_path):
            logger.error(f"找不到预处理数据文件: {features_path}")
            return
        logger.info(f"使用预处理后的原始特征: {features_path}")
        data_dict = load_h5_data(features_path) 

    
    # 加载特征数据
    try:
        # 更灵活的数据提取逻辑
        train_features = None
        train_labels = None
        val_features = None
        val_labels = None
        test_features = None
        test_labels = None
        
        # 1. 首先尝试从顶级结构中提取
        if 'train' in data_dict and 'features' in data_dict['train']:
            train_features = data_dict['train']['features']
            train_labels = data_dict['train']['labels'] if 'labels' in data_dict['train'] else None
            val_features = data_dict['val']['features'] if 'val' in data_dict and 'features' in data_dict['val'] else None
            val_labels = data_dict['val']['labels'] if 'val' in data_dict and 'labels' in data_dict['val'] else None
            test_features = data_dict['test']['features'] if 'test' in data_dict and 'features' in data_dict['test'] else None
            test_labels = data_dict['test']['labels'] if 'test' in data_dict and 'labels' in data_dict['test'] else None
            logger.info("从顶级结构中提取数据")
        
        # 2. 如果上面失败，尝试从 'all' 组提取
        elif 'all' in data_dict:
            # 检查嵌套结构
            if 'train' in data_dict['all']:
                if isinstance(data_dict['all']['train'], dict) and 'features' in data_dict['all']['train']:
                    train_features = data_dict['all']['train']['features']
                    train_labels = data_dict['all']['train']['labels'] if 'labels' in data_dict['all']['train'] else None
                    val_features = data_dict['all']['val']['features'] if 'val' in data_dict['all'] and 'features' in data_dict['all']['val'] else None
                    val_labels = data_dict['all']['val']['labels'] if 'val' in data_dict['all'] and 'labels' in data_dict['all']['val'] else None
                    test_features = data_dict['all']['test']['features'] if 'test' in data_dict['all'] and 'features' in data_dict['all']['test'] else None
                    test_labels = data_dict['all']['test']['labels'] if 'test' in data_dict['all'] and 'labels' in data_dict['all']['test'] else None
                    logger.info("从嵌套的 'all' 组结构中提取数据")
                else:  # 可能是数组而不是字典
                    train_features = data_dict['all']['train']
                    train_labels = data_dict['all']['train_labels'] if 'train_labels' in data_dict['all'] else None
                    val_features = data_dict['all']['val'] if 'val' in data_dict['all'] else None
                    val_labels = data_dict['all']['val_labels'] if 'val_labels' in data_dict['all'] else None
                    test_features = data_dict['all']['test'] if 'test' in data_dict['all'] else None
                    test_labels = data_dict['all']['test_labels'] if 'test_labels' in data_dict['all'] else None
                    logger.info("从 'all' 组的单一结构中提取数据")
        
        # 3. 如果还是失败，尝试其他可能的结构
        elif args.feature_type == 'selected' and args.feature_selection:
            # 尝试找到包含选定特征选择方法的键
            for key in data_dict.keys():
                if args.feature_selection in key.lower():
                    if 'train' in data_dict[key]:
                        if isinstance(data_dict[key]['train'], dict) and 'features' in data_dict[key]['train']:
                            train_features = data_dict[key]['train']['features']
                            train_labels = data_dict[key]['train']['labels'] if 'labels' in data_dict[key]['train'] else None
                        else:
                            train_features = data_dict[key]['train']
                            train_labels = data_dict[key]['train_labels'] if 'train_labels' in data_dict[key] else None
                        
                        # 提取验证集和测试集
                        if 'val' in data_dict[key]:
                            val_features = data_dict[key]['val']['features'] if isinstance(data_dict[key]['val'], dict) and 'features' in data_dict[key]['val'] else data_dict[key]['val']
                            val_labels = data_dict[key]['val']['labels'] if isinstance(data_dict[key]['val'], dict) and 'labels' in data_dict[key]['val'] else data_dict[key]['val_labels'] if 'val_labels' in data_dict[key] else None
                        
                        if 'test' in data_dict[key]:
                            test_features = data_dict[key]['test']['features'] if isinstance(data_dict[key]['test'], dict) and 'features' in data_dict[key]['test'] else data_dict[key]['test']
                            test_labels = data_dict[key]['test']['labels'] if isinstance(data_dict[key]['test'], dict) and 'labels' in data_dict[key]['test'] else data_dict[key]['test_labels'] if 'test_labels' in data_dict[key] else None
                        
                        logger.info(f"从特征选择结果 '{key}' 中提取数据")
                        break
        
        # 4. 最后一次尝试 - 直接搜索任何包含训练特征和标签的路径
        if train_features is None:
            logger.warning("无法通过正常路径找到数据，尝试搜索任何可能的路径...")
            
            # 尝试搜索特征和标签
            for key1 in data_dict.keys():
                if train_features is not None:
                    break
                    
                if isinstance(data_dict[key1], dict):
                    for key2 in data_dict[key1].keys():
                        if key2 == 'train' or key2 == 'features' or 'train' in key2:
                            if isinstance(data_dict[key1][key2], np.ndarray):
                                # 可能是特征
                                if len(data_dict[key1][key2].shape) == 2:
                                    train_features = data_dict[key1][key2]
                                    logger.info(f"找到可能的训练特征: {key1}/{key2} 形状={train_features.shape}")
                                    
                                    # 尝试找到相关的标签
                                    label_keys = [k for k in data_dict[key1].keys() if 'label' in k.lower()]
                                    if label_keys:
                                        train_labels = data_dict[key1][label_keys[0]]
                                        logger.info(f"找到可能的训练标签: {key1}/{label_keys[0]} 形状={train_labels.shape}")
                                    break
        
        # 检查是否成功提取特征和标签
        if train_features is None:
            logger.error("无法从数据中提取特征")
            return
            
        if train_labels is None:
            # 尝试在整个数据字典中查找任何标签数据
            def find_labels(d, path=""):
                for k, v in d.items():
                    if 'label' in k.lower() and isinstance(v, np.ndarray):
                        logger.info(f"找到可能的标签数据: {path}/{k} 形状={v.shape}")
                        return v
                    elif isinstance(v, dict):
                        result = find_labels(v, f"{path}/{k}")
                        if result is not None:
                            return result
                return None
                
            train_labels = find_labels(data_dict)
            
            if train_labels is None:
                logger.error("无法从数据中提取标签")
                return
        
        # 提取特征组数据 (用于group_mlp模型)
        group_features = {}
        for key in data_dict.keys():
            if key not in ['all', 'original'] and not key.startswith('_'):
                if 'train' in data_dict[key]:
                    group_features[key] = {
                        'train': data_dict[key]['train']['features'] if isinstance(data_dict[key]['train'], dict) and 'features' in data_dict[key]['train'] else data_dict[key]['train'],
                        'val': data_dict[key]['val']['features'] if 'val' in data_dict[key] and isinstance(data_dict[key]['val'], dict) and 'features' in data_dict[key]['val'] else data_dict[key]['val'] if 'val' in data_dict[key] else None,
                        'test': data_dict[key]['test']['features'] if 'test' in data_dict[key] and isinstance(data_dict[key]['test'], dict) and 'features' in data_dict[key]['test'] else data_dict[key]['test'] if 'test' in data_dict[key] else None
                    }
                    logger.info(f"提取到特征组 '{key}'")



        
        # 检查是否成功提取特征和标签
        if train_features is None:
            logger.error("无法从数据中提取特征")
            return
            
        if train_labels is None:
            logger.error("无法从数据中提取标签")
            return
            
        if val_features is None or val_labels is None:
            logger.warning("无法从数据中提取验证集，尝试使用训练集的一部分作为验证集")
            # 分割训练集的20%作为验证集
            from sklearn.model_selection import train_test_split
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )
            
        if test_features is None or test_labels is None:
            logger.warning("无法从数据中提取测试集，尝试使用训练集的一部分作为测试集")
            # 分割训练集的10%作为测试集
            from sklearn.model_selection import train_test_split
            train_features, test_features, train_labels, test_labels = train_test_split(
                train_features, train_labels, test_size=0.1, random_state=42, stratify=train_labels
            )
            
        # 在加载特征后添加
        print(f"Feature stats - Min: {np.min(train_features)}, Max: {np.max(train_features)}")
        print(f"Feature sample (first 5 elements): {train_features[0, :5]}")
        print(f"Any NaN values: {np.isnan(train_features).any()}")
        print(f"Any Inf values: {np.isinf(train_features).any()}")

        logger.info(f"成功提取特征 - 训练集: {train_features.shape}, 验证集: {val_features.shape}, 测试集: {test_features.shape}")
        if group_features:
            logger.info(f"提取了 {len(group_features)} 个特征组用于group_mlp模型")
            
    except Exception as e:
        logger.error(f"处理特征数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 步骤2：创建模型
    logger.info("步骤2: 创建模型...")
    try:
        model_config = config.get('baseline_model', {})
        
        if args.model_type == 'mlp':
            # 基础MLP模型
            hidden_dims = model_config.get('hidden_dims', [2048, 1536, 1024, 512, 256, 128])
            dropout_rate = model_config.get('dropout_rate', 0.4)
            
            input_dim = train_features.shape[1]
            num_classes = len(np.unique(train_labels))
            
            model = BaselineMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
            
            logger.info(f"创建BaselineMLP模型 - 输入维度: {input_dim}, 隐藏层: {hidden_dims}, 输出类别: {num_classes}")
            
        elif args.model_type == 'group_mlp':
            # 特征组MLP模型
            # 提取每个特征组的维度
            group_dims = {}
            for group_name, group_data in group_features.items():
                if group_data['train'] is not None and len(group_data['train'].shape) > 0:
                    group_dims[group_name] = group_data['train'].shape[1]
            
            # 使用配置文件中的隐藏层维度或设置默认值
            group_hidden_dims = {}
            for group_name in group_dims.keys():
                if group_name in model_config.get('group_hidden_dims', {}):
                    group_hidden_dims[group_name] = model_config['group_hidden_dims'][group_name]
                else:
                    # 默认隐藏层维度
                    group_hidden_dims[group_name] = [256, 128]
            
            num_classes = len(np.unique(train_labels))
            fusion_method = model_config.get('fusion_method', 'attention')
            dropout_rate = model_config.get('dropout_rate', 0.3)
            
            model = FeatureGroupMLP(
                group_dims=group_dims,
                hidden_dims=group_hidden_dims,
                num_classes=num_classes,
                fusion_method=fusion_method,
                dropout_rate=dropout_rate
            )
            
            logger.info(f"创建FeatureGroupMLP模型 - 特征组: {list(group_dims.keys())}, 融合方法: {fusion_method}")
            
        elif args.model_type == 'deep_mlp':
            # 增强版DeepMLP模型
            hidden_dims = model_config.get('hidden_dims', [2048, 1536, 1024, 512, 256, 128])
            dropout_rate = model_config.get('dropout_rate', 0.3)
            use_residual = model_config.get('use_residual', True)
            use_self_attention = model_config.get('use_self_attention', True)
            use_feature_interaction = model_config.get('use_feature_interaction', True)
            num_attn_heads = model_config.get('num_attn_heads', 12)
            attn_layers = model_config.get('attn_layers', [1, 3])
            
            input_dim = train_features.shape[1]
            
            
            # 检查实际标签范围
            unique_labels = np.unique(train_labels)
            min_label = np.min(unique_labels)
            max_label = np.max(unique_labels)
            num_classes = len(unique_labels)

            # 如果标签是从1开始的，手动设置为从0开始
            label_offset = 0
            if min_label == 1 and max_label == 102:
                logger.info("检测到标签从1开始，将进行0-101重新映射")
                train_labels = train_labels - 1
                val_labels = val_labels - 1
                test_labels = test_labels - 1
                label_offset = 1
                # 重新计算唯一值和范围
                unique_labels = np.unique(train_labels)
                min_label = np.min(unique_labels)
                max_label = np.max(unique_labels)
                num_classes = len(unique_labels)

            logger.info(f"调整后标签范围: {min_label} - {max_label}")
            logger.info(f"实际类别数量: {num_classes}")
            logger.info(f"使用的类别数量: {max_label + 1}")  # 因为标签是从0开始的

            # 确保模型输出维度正确
            model_num_classes = int(max_label + 1) # 使用最大标签值+1作为类别数

            logger.info(f"After setting - max_label: {max_label}, type: {type(max_label)}")
            logger.info(f"After setting - model_num_classes: {model_num_classes}, type: {type(model_num_classes)}")

            logger.info(f"model_num_classes type: {type(model_num_classes)}, value: {model_num_classes}")

            model = DeepMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=model_num_classes,  # 使用正确的类别数
                use_residual=False,
                use_self_attention=use_self_attention,
                use_feature_interaction=use_feature_interaction,
                dropout_rates=[dropout_rate] * len(hidden_dims),
                num_attn_heads=num_attn_heads,
                attn_layers=attn_layers
            )

            
            logger.info(f"创建DeepMLP模型 - 输入维度: {input_dim}, 隐藏层: {hidden_dims}, 使用残差: {use_residual}, 使用自注意力: {use_self_attention}")
        
        # 记录模型参数数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数量: {total_params:,}")
        
    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
        # 在stage2_baseline_training.py的main函数中，创建模型后添加
    print(f"Model structure:\n{model}")
    print(f"Input shape: {train_features.shape}")
    print(f"Expected first layer input: {model.layers[0].in_features if hasattr(model.layers[0], 'in_features') else 'Unknown'}")
    # 步骤3：训练模型
    logger.info("步骤3: 训练模型...")
    try:
        # 更新配置，指定保存目录
        model_config['save_dir'] = output_dir
        
        # 创建临时配置文件
        temp_config_path = os.path.join(output_dir, 'model_config.json')
        with open(temp_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # 创建训练器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        if args.model_type in ['mlp', 'deep_mlp']:
            # 使用标准训练器
            trainer = BaselineTrainer(model, temp_config_path, device)

            if trainer.criterion is None:
                # 根据配置创建损失函数
                criterion_type = model_config.get('criterion', 'cross_entropy')
                label_smoothing = model_config.get('label_smoothing', 0.1)
                
                if criterion_type.lower() == 'cross_entropy':
                    trainer.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                elif criterion_type.lower() == 'focal':
                    # 简化版Focal Loss实现
                    from torch.nn import functional as F
                    class FocalLoss(nn.Module):
                        def __init__(self, gamma=2.0, reduction='mean'):
                            super(FocalLoss, self).__init__()
                            self.gamma = gamma
                            self.reduction = reduction
                        
                        def forward(self, input, target):
                            ce_loss = F.cross_entropy(input, target, reduction='none')
                            pt = torch.exp(-ce_loss)
                            focal_loss = (1 - pt) ** self.gamma * ce_loss
                            
                            if self.reduction == 'mean':
                                return focal_loss.mean()
                            elif self.reduction == 'sum':
                                return focal_loss.sum()
                            else:
                                return focal_loss
                    
                    trainer.criterion = FocalLoss()

            
            # 准备数据
            import torch.utils.data as data
            num_classes = len(np.unique(train_labels))
            model_num_classes = model.classifier.out_features
            print(f"数据中的唯一类别数: {num_classes}")
            print(f"模型输出的类别数: {model_num_classes}")
            print(f"标签值范围: {np.min(train_labels)} - {np.max(train_labels)}")

            # 预处理标签
            # 无需再次截断标签，因为我们已经正确映射了
            logger.info(f"最终标签值范围: {np.min(train_labels)} - {np.max(train_labels)}")
            logger.info(f"模型输出类别数: {model_num_classes}")

            # 只检查一下是否有问题，而不进行截断
            if np.max(train_labels) >= model_num_classes:
                logger.warning(f"警告：仍有 {np.sum(train_labels >= model_num_classes)} 个标签值超出模型类别数")
                # 在这种情况下可能需要进行截断，但这应该是极少数情况
                train_labels = np.clip(train_labels, 0, model_num_classes - 1)
                val_labels = np.clip(val_labels, 0, model_num_classes - 1)
                test_labels = np.clip(test_labels, 0, model_num_classes - 1)
                logger.info(f"截断后标签值范围: {np.min(train_labels)} - {np.max(train_labels)}")


            # 创建张量和数据集
            train_tensor_x = torch.tensor(train_features, dtype=torch.float32)
            train_tensor_y = torch.tensor(train_labels, dtype=torch.long)
            val_tensor_x = torch.tensor(val_features, dtype=torch.float32)
            val_tensor_y = torch.tensor(val_labels, dtype=torch.long)
            test_tensor_x = torch.tensor(test_features, dtype=torch.float32)
            test_tensor_y = torch.tensor(test_labels, dtype=torch.long)

            train_dataset = data.TensorDataset(train_tensor_x, train_tensor_y)
            val_dataset = data.TensorDataset(val_tensor_x, val_tensor_y)
            test_dataset = data.TensorDataset(test_tensor_x, test_tensor_y)

            # 创建数据加载器
            batch_size = model_config.get('batch_size', 128)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            data_loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }

            # 使用普通Trainer
            trainer = BaselineTrainer(model, temp_config_path, device)
            # 确保损失函数已初始化
            if trainer.criterion is None:
                trainer.criterion = nn.CrossEntropyLoss()
            
            # 训练模型
            history = trainer.train(data_loaders)
            
            # 评估模型
            metrics = trainer.evaluate(data_loaders['test'])
            
            logger.info(f"训练完成 - 测试集准确率: {metrics['accuracy']:.4f}, 测试集F1分数: {metrics['f1_weighted']:.4f}")
            
        elif args.model_type == 'group_mlp':
            # 使用特征组评估器进行训练
            # 首先创建一个简化版的特征组数据集字典
            group_data_dict = {}
            
            # 添加训练集
            train_group = {}
            for group_name, group_data in group_features.items():
                if group_data['train'] is not None:
                    train_group[group_name] = torch.tensor(group_data['train'], dtype=torch.float32)
            group_data_dict['train'] = {
                'features': train_group,
                'labels': torch.tensor(train_labels, dtype=torch.long)
            }
            
            # 添加验证集
            val_group = {}
            for group_name, group_data in group_features.items():
                if group_data['val'] is not None:
                    val_group[group_name] = torch.tensor(group_data['val'], dtype=torch.float32)
            group_data_dict['val'] = {
                'features': val_group,
                'labels': torch.tensor(val_labels, dtype=torch.long)
            }
            
            # 添加测试集
            test_group = {}
            for group_name, group_data in group_features.items():
                if group_data['test'] is not None:
                    test_group[group_name] = torch.tensor(group_data['test'], dtype=torch.float32)
            group_data_dict['test'] = {
                'features': test_group,
                'labels': torch.tensor(test_labels, dtype=torch.long)
            }
            
            # 使用GroupEvaluator训练和评估模型
            evaluator = GroupEvaluator(
                model_template=model,
                config_path=temp_config_path,
                output_dir=output_dir,
                logger=logger,
                device=device
            )
            
            # 准备数据加载器

            # 详细检查标签范围
            logger.info("标签详细检查：")
            logger.info(f"标签最小值：{np.min(train_labels)}")
            logger.info(f"标签最大值：{np.max(train_labels)}")
            logger.info(f"标签唯一值的数量：{len(np.unique(train_labels))}")

            # 检查是否存在大于等于101的标签
            if np.max(train_labels) >= 101:
                high_labels = np.sum(train_labels >= 101)
                logger.info(f"存在 {high_labels} 个值>=101的标签")
                # 查看这些高值标签的具体分布
                for label in range(101, int(np.max(train_labels))+1):
                    count = np.sum(train_labels == label)
                    if count > 0:
                        logger.info(f"  标签值 {label}: {count} 个样本")

            # 添加标签分布统计
            label_counts = {}
            for i in range(int(np.max(train_labels)) + 1):
                count = np.sum(train_labels == i)
                if count > 0:
                    label_counts[i] = int(count)

            logger.info("标签分布概要:")
            for label, count in sorted(label_counts.items()):
                    logger.info(f"  类别 {label}: {count} 样本")

            # 检查是否存在稀有类别（样本数较少的类别）
            rare_labels = {label: count for label, count in label_counts.items() if count < 100}
            if rare_labels:
                logger.info("警告: 存在稀有类别（样本数<100）:")
                for label, count in sorted(rare_labels.items()):
                    logger.info(f"  类别 {label}: 仅有 {count} 样本")

            data_loaders, feature_dims = evaluator._prepare_data_loaders(group_data_dict)
            
            # 训练和评估
            metrics = evaluator.train_and_evaluate(data_loaders, feature_dims, list(group_features.keys()))
            
            logger.info(f"训练完成 - 测试集准确率: {metrics['accuracy']:.4f}, 测试集F1分数: {metrics['f1_weighted']:.4f}")
        
    except Exception as e:
        logger.error(f"训练模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger_manager.log_experiment_end(experiment_name, {"状态": "失败", "阶段": "模型训练", "错误": str(e)})
        return
    
    # 步骤4：评估特征有效性（对比不同特征类型）
    if args.feature_type != 'original':
        logger.info("步骤4: 评估特征工程有效性...")
        try:
            # 创建评估结果总结
            evaluation_summary = {
                "model_type": args.model_type,
                "feature_type": args.feature_type,
                "feature_selection": args.feature_selection,
                "feature_subset": args.feature_subset,
                "accuracy": float(metrics['accuracy']),
                "f1_weighted": float(metrics['f1_weighted']),
                "f1_macro": float(metrics.get('f1_macro', 0)),
                "params_count": total_params,
                "feature_dims": train_features.shape[1]
            }
            
            # 保存评估结果
            summary_path = os.path.join(output_dir, 'evaluation_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(evaluation_summary, f, indent=4)
                
            logger.info(f"特征工程评估摘要已保存至 {summary_path}")
            
            # 创建可读性报告
            report_path = os.path.join(output_dir, 'feature_effectiveness_report.md')
            with open(report_path, 'w') as f:
                f.write(f"# {args.model_type.upper()} 模型特征有效性评估\n\n")
                f.write(f"## 特征类型: {args.feature_type}\n\n")
                if args.feature_selection:
                    f.write(f"## 特征选择方法: {args.feature_selection}\n\n")
                if args.feature_subset:
                    f.write(f"## 特征子集: {args.feature_subset}\n\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 模型性能\n\n")
                f.write(f"- 准确率: {metrics['accuracy']:.4f}\n")
                f.write(f"- 加权F1分数: {metrics['f1_weighted']:.4f}\n")
                if 'f1_macro' in metrics:
                    f.write(f"- 宏平均F1分数: {metrics['f1_macro']:.4f}\n")
                f.write(f"- 特征维度: {train_features.shape[1]}\n")
                f.write(f"- 模型参数数量: {total_params:,}\n\n")
                
                f.write("## 特征有效性分析\n\n")
                
                if args.feature_type == 'selected':
                    # 特征选择有效性分析
                    original_dims = config.get('feature_groups', {}).get('diffusion', {}).get('input_dim', 0) + \
                                    config.get('feature_groups', {}).get('qti', {}).get('input_dim', 0) + \
                                    config.get('feature_groups', {}).get('cest', {}).get('input_dim', 0)
                    if original_dims == 0:
                        original_dims = 341  # 默认值
                        
                    reduction_ratio = (original_dims - train_features.shape[1]) / original_dims * 100
                    
                    f.write(f"### 特征选择效果\n\n")
                    f.write(f"- 原始特征维度: {original_dims}\n")
                    f.write(f"- 选择后特征维度: {train_features.shape[1]}\n")
                    f.write(f"- 维度减少比例: {reduction_ratio:.1f}%\n\n")
                    
                    if args.feature_selection:
                        f.write(f"- 特征选择方法: {args.feature_selection}\n")
                    if args.feature_subset:
                        f.write(f"- 使用的特征子集: {args.feature_subset}\n\n")
                    
                    if reduction_ratio > 70:
                        f.write("特征选择效果**显著**，保留了最重要的特征同时大幅减少了维度。\n\n")
                    elif reduction_ratio > 30:
                        f.write("特征选择效果**中等**，适度减少了特征维度。\n\n")
                    else:
                        f.write("特征选择效果**有限**，仅少量减少了特征维度。\n\n")
                    
                elif args.feature_type == 'pca':
                    # PCA降维有效性分析
                    original_dims = config.get('feature_groups', {}).get('diffusion', {}).get('input_dim', 0) + \
                                    config.get('feature_groups', {}).get('qti', {}).get('input_dim', 0) + \
                                    config.get('feature_groups', {}).get('cest', {}).get('input_dim', 0)
                    if original_dims == 0:
                        original_dims = 341  # 默认值
                        
                    reduction_ratio = (original_dims - train_features.shape[1]) / original_dims * 100
                    
                    f.write(f"### PCA降维效果\n\n")
                    f.write(f"- 原始特征维度: {original_dims}\n")
                    f.write(f"- PCA降维后维度: {train_features.shape[1]}\n")
                    f.write(f"- 维度减少比例: {reduction_ratio:.1f}%\n\n")
                    
                    if reduction_ratio > 70:
                        f.write("PCA降维效果**显著**，极大地减少了特征维度同时保留了主要信息。\n\n")
                    elif reduction_ratio > 30:
                        f.write("PCA降维效果**中等**，在保留信息的同时减少了特征维度。\n\n")
                    else:
                        f.write("PCA降维效果**有限**，仅少量减少了特征维度。\n\n")
                    
                elif args.feature_type == 'combined':
                    # 组合特征有效性分析
                    f.write(f"### 组合特征效果\n\n")
                    
                    if args.feature_selection:
                        f.write(f"- 使用的特征选择方法: {args.feature_selection}\n")
                    if args.feature_subset:
                        f.write(f"- 使用的特征子集: {args.feature_subset}\n")
                        
                    f.write(f"- 特征选择和PCA降维的组合提供了一种平衡的特征表示。\n")
                    f.write(f"- 最终特征维度: {train_features.shape[1]}\n\n")
                    
                f.write("## 建议\n\n")
                
                # 根据评估结果给出建议
                f.write("基于当前评估结果，对模型训练提出以下建议：\n\n")
                
                if metrics['accuracy'] > 0.8:
                    f.write("1. **模型性能良好**：当前模型已达到较高准确率，可以考虑部署使用。\n")
                elif metrics['accuracy'] > 0.6:
                    f.write("1. **模型性能中等**：可尝试调整模型超参数或尝试更复杂的模型架构提升性能。\n")
                else:
                    f.write("1. **模型性能有限**：建议重新审视特征工程策略，或考虑更高级的模型架构。\n")
                
                if args.model_type == 'group_mlp':
                    f.write("2. **特征组融合**：当前使用特征组分别建模再融合的策略，可以进一步优化每个特征组的处理方式和融合机制。\n")
                else:
                    f.write("2. **特征表示**：可以尝试不同的特征表示方法，比如结合特征选择和降维的混合策略。\n")
                
                if args.feature_selection:
                    f.write(f"3. **特征选择方法**：当前使用的是{args.feature_selection}方法进行特征选择，可以尝试其他方法如{'互信息' if args.feature_selection != 'mi' else '随机森林'}进行比较。\n")
                
                f.write(f"4. **进一步实验**：建议与其他模型架构和特征工程策略进行对比实验，寻找最佳组合。\n")
                
            logger.info(f"特征有效性报告已保存至 {report_path}")
            
        except Exception as e:
            logger.error(f"评估特征有效性失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 记录实验结束
    results = {
        "状态": "成功",
        "训练时间(秒)": training_time,
        "准确率": float(metrics['accuracy']),
        "F1分数": float(metrics['f1_weighted']),
        "模型类型": args.model_type,
        "特征类型": args.feature_type,
        "特征选择方法": args.feature_selection,
        "特征子集": args.feature_subset,
        "特征维度": train_features.shape[1],
        "参数数量": total_params
    }
    logger_manager.log_experiment_end(experiment_name, results)
    
    logger.info(f"基线模型训练完成！总耗时: {training_time:.2f} 秒")
    logger.info(f"模型保存在: {output_dir}")
    
    return {
        'model_dir': output_dir,
        'accuracy': float(metrics['accuracy']),
        'f1_score': float(metrics['f1_weighted']),
        'timestamp': timestamp
    }

if __name__ == "__main__":
    main()