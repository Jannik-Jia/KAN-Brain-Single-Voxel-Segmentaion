#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用训练好的模型进行体素分类预测并映射回3D空间
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import h5py
import time
from torch.utils.data import DataLoader, Dataset
from scipy.io import savemat

# 确保能找到项目模块
sys.path.append('.')

# 导入必要的模块
from models import get_model
from utils.model_io import safe_load_model

class SingleVoxelDataset(Dataset):
    """单体素数据集"""
    def __init__(self, voxels):
        """
        初始化
        
        参数:
            voxels: 形状为(n_samples, 341)的numpy数组
        """
        self.voxels = voxels
    
    def __len__(self):
        return len(self.voxels)
    
    def __getitem__(self, idx):
        x = self.voxels[idx]
        x = torch.FloatTensor(x)
        return x

def load_optimized_model(model_path=None):
    """
    加载优化后的模型
    
    参数:
        model_path: 模型文件路径，如果为None则自动查找
        
    返回:
        model: 加载好的模型
        config: 模型配置
    """
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 如果未提供模型路径，尝试自动查找
    if model_path is None:
        print("尝试自动查找最新的模型文件...")
        
        possible_dirs = [
            "./best_model_results", 
            "./results", 
            "./extended_bayes_results"
        ]
        
        found = False
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                model_files = [f for f in os.listdir(dir_path) if f.endswith(".pth")]
                if model_files:
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
                    model_path = os.path.join(dir_path, model_files[0])
                    found = True
                    break
        
        if not found:
            raise FileNotFoundError("无法自动找到模型文件，请手动指定模型路径")
    
    print(f"使用模型: {model_path}")
    
    # 贝叶斯优化的最佳参数 - 硬编码以确保正确
    model_config = {
        'model_type': 'deep_mlp',
        'hidden_dims': [2048] * 6,
        'input_dim': 341,
        'num_classes': 102,
        'dropout_rate': 0.2567125567148536,
        'activation': 'gelu',
        'use_skip_connections': True
    }
    
    print("使用贝叶斯优化的最佳参数创建模型:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # 创建模型
    model = get_model(
        model_type=model_config['model_type'],
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout_rate'],
        activation=model_config['activation'],
        use_skip_connections=model_config['use_skip_connections']
    )
    
    # 加载模型权重
    try:
        checkpoint = safe_load_model(model_path, device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()  # 设置为评估模式
        print("成功加载模型权重")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise
    
    return model, model_config, device

def predict_voxels(model, voxels, batch_size=128, device=None):
    """
    预测体素分类
    
    参数:
        model: 训练好的模型
        voxels: 形状为(n_samples, 341)的numpy数组
        batch_size: 批处理大小
        device: 计算设备
        
    返回:
        predictions: 预测的类别，从0开始
        probabilities: 每个类别的概率
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集和数据加载器
    dataset = SingleVoxelDataset(voxels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 收集预测结果
    all_probs = []
    
    # 预测
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # 合并结果
    all_probs = np.vstack(all_probs)
    predictions = np.argmax(all_probs, axis=1)
    
    # 转换为从1开始的标签（与原始数据匹配）
    predictions = predictions + 1
    
    return predictions, all_probs

def normalize_voxels(voxels, method='standard'):
    """
    标准化体素数据
    
    参数:
        voxels: 形状为(n_samples, 341)的numpy数组
        method: 标准化方法，'standard'或'minmax'
        
    返回:
        normalized_voxels: 标准化后的体素数据
    """
    if method == 'standard':
        # 标准化（Z-score）
        mean = np.mean(voxels, axis=0)
        std = np.std(voxels, axis=0)
        std[std == 0] = 1e-10  # 避免除零
        normalized_voxels = (voxels - mean) / std
    elif method == 'minmax':
        # 最小-最大归一化
        min_vals = np.min(voxels, axis=0)
        max_vals = np.max(voxels, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-10  # 避免除零
        normalized_voxels = (voxels - min_vals) / range_vals
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    return normalized_voxels

def load_matlab_data(data_path, features_key=None, region_key=None):
    """
    加载MATLAB .mat文件数据，支持v7.3格式
    
    参数:
        data_path: .mat文件路径
        features_key: MAT文件中特征数据的键名 (可选)
        region_key: MAT文件中区域掩码的键名 (可选)
    
    返回:
        features: 特征矩阵
        original_shape: 原始3D形状
        metadata: 其他元数据
    """
    print(f"加载数据: {data_path}")
    try:
        # 检查文件格式并选择加载方法
        import h5py
        import scipy.io as sio
        
        try:
            # 尝试使用scipy.io.loadmat加载
            mat_data = sio.loadmat(data_path)
            print("使用scipy.io.loadmat成功加载数据")
            is_hdf5 = False
        except NotImplementedError:
            # 如果是v7.3格式，使用h5py加载
            print("检测到MATLAB v7.3 (HDF5)格式，使用h5py加载")
            mat_data = h5py.File(data_path, 'r')
            is_hdf5 = True
        
        print(f"MAT文件中的键: {list(mat_data.keys())}")
        
        # 初始化变量
        features = None
        original_shape = None
        metadata = {}
        
        # 处理HDF5格式
        if is_hdf5:
            # 使用指定的特征键
            if features_key and features_key in mat_data:
                try:
                    features = np.array(mat_data[features_key])
                    if features.ndim > 1:
                        features = features.T
                    print(f"使用指定键'{features_key}'加载特征数据，形状: {features.shape}")
                except Exception as e:
                    print(f"使用指定键加载数据失败: {e}")
            
            # 尝试获取区域掩码
            if region_key and region_key in mat_data:
                try:
                    region_mask = np.array(mat_data[region_key])
                    if region_mask.ndim > 1:
                        region_mask = region_mask.T
                    metadata['region_mask'] = region_mask
                    print(f"使用指定键'{region_key}'加载区域掩码，形状: {region_mask.shape}")
                    
                    # 从掩码获取原始形状
                    original_shape = region_mask.shape
                    print(f"从区域掩码获取原始形状: {original_shape}")
                except Exception as e:
                    print(f"加载区域掩码失败: {e}")
        # 处理传统格式
        else:
            # 使用指定的特征键
            if features_key and features_key in mat_data:
                features = mat_data[features_key]
                print(f"使用指定键'{features_key}'加载特征数据，形状: {features.shape}")
            
            # 获取区域掩码
            if region_key and region_key in mat_data:
                metadata['region_mask'] = mat_data[region_key]
                print(f"使用指定键'{region_key}'加载区域掩码，形状: {metadata['region_mask'].shape}")
                
                # 从掩码获取原始形状
                original_shape = metadata['region_mask'].shape
                print(f"从区域掩码获取原始形状: {original_shape}")
        
        # 如果找不到特征数据，报错
        if features is None:
            raise ValueError(f"无法在MAT文件中找到特征数据。可用键: {list(mat_data.keys())}")
            
        # 关闭HDF5文件
        if is_hdf5:
            mat_data.close()
            
        return features, original_shape, metadata
        
    except Exception as e:
        print(f"加载MAT文件出错: {e}")
        raise

def map_predictions_to_3d(predictions, original_shape, region_mask=None, probabilities=None):
    """
    将预测结果映射回3D空间
    
    参数:
        predictions: 预测的类别
        original_shape: 原始3D形状
        region_mask: 掩码，指定哪些位置应该被填充
        probabilities: 预测的概率 (可选)
        
    返回:
        volume: 3D体积，包含预测结果
        prob_volume: 4D体积，包含概率 (如果probabilities不为None)
    """
    print(f"将预测映射回3D空间，目标形状: {original_shape}")
    
    # 创建3D体积并初始化为-1（未知）
    volume = np.zeros(original_shape, dtype=np.int32) - 1
    
    # 如果有概率矩阵，创建4D体积
    prob_volume = None
    if probabilities is not None:
        n_classes = probabilities.shape[1]
        prob_volume = np.zeros(original_shape + (n_classes,), dtype=np.float32)
    
    # 如果有区域掩码，使用它来定位预测位置
    if region_mask is not None:
        # 确保region_mask是布尔型
        if not np.issubdtype(region_mask.dtype, np.bool_):
            region_mask = region_mask > 0
        
        # 将掩码展平
        mask_flat = region_mask.flatten()
        
        # 确保预测数量与区域掩码中的True数量匹配
        if np.sum(mask_flat) != len(predictions):
            print(f"警告: 预测数量({len(predictions)})与区域掩码中的True数量({np.sum(mask_flat)})不匹配")
            # 使用最小的数量避免错误
            n_samples = min(len(predictions), np.sum(mask_flat))
            mask_indices = np.where(mask_flat)[0][:n_samples]
        else:
            # 如果数量匹配，获取所有True位置的索引
            mask_indices = np.where(mask_flat)[0]
        
        # 填充预测
        volume_flat = volume.flatten()
        volume_flat[mask_indices] = predictions[:len(mask_indices)]
        volume = volume_flat.reshape(original_shape)
        
        # 如果有概率数据，也填充它们
        if prob_volume is not None:
            for c in range(probabilities.shape[1]):
                # 创建临时3D体积
                temp_vol = np.zeros(original_shape, dtype=np.float32)
                temp_vol_flat = temp_vol.flatten()
                temp_vol_flat[mask_indices] = probabilities[:len(mask_indices), c]
                # 将1D数组重新整形为3D并保存到相应通道
                prob_volume[..., c] = temp_vol_flat.reshape(original_shape)
    else:
        # 如果没有掩码，尝试直接重塑预测数组
        try:
            n_voxels = np.prod(original_shape)
            if len(predictions) == n_voxels:
                volume = predictions.reshape(original_shape)
                print("直接重塑预测数组到指定形状")
                
                # 如果有概率数据，也重塑它们
                if prob_volume is not None:
                    for c in range(probabilities.shape[1]):
                        prob_volume[..., c] = probabilities[:, c].reshape(original_shape)
            else:
                print(f"警告: 预测数量({len(predictions)})与体积体素数量({n_voxels})不匹配，无法重塑")
        except Exception as e:
            print(f"重塑预测数组时出错: {e}")
    
    # 统计非背景体素数量
    n_valid = np.sum(volume != -1)
    print(f"体积中的有效体素数量: {n_valid} / {volume.size} ({n_valid/volume.size*100:.2f}%)")
    
    # 统计不同类别的数量
    unique_classes = np.unique(volume)
    unique_classes = unique_classes[unique_classes != -1]  # 排除背景
    print(f"体积中的唯一类别: {len(unique_classes)}")
    
    # 显示前5个类别的统计
    class_counts = [(cls, np.sum(volume == cls)) for cls in unique_classes]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    if class_counts:
        print(f"前5个类别统计: {class_counts[:5]}")
    
    return volume, prob_volume

def save_results(predictions, probabilities, volume, prob_volume=None, metadata=None, output_dir=None):
    """
    保存预测结果，仅使用MAT格式
    
    参数:
        predictions: 预测的类别
        probabilities: 预测的概率
        volume: 3D体积
        prob_volume: 概率体积 (可选)
        metadata: 原始数据的元数据
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = './prediction_results_' + time.strftime("%Y%m%d_%H%M%S")
    
    print(f"保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建要保存的数据字典
    output_data = {
        'predictions': predictions,
        'probabilities': probabilities,
        'volume_3d': volume
    }
    
    if prob_volume is not None:
        output_data['probability_volume'] = prob_volume
    
    # 添加元数据
    if metadata:
        for key, value in metadata.items():
            if key not in output_data:
                # 只添加简单的元数据，避免过大的数据
                try:
                    if isinstance(value, np.ndarray) and value.size > 1000000:
                        print(f"跳过保存大型元数据 '{key}'，大小: {value.shape}")
                    else:
                        output_data[key] = value
                except:
                    print(f"无法保存元数据 '{key}'")
    
    # 保存MAT文件
    mat_file_path = os.path.join(output_dir, 'prediction_results.mat')
    try:
        print(f"正在保存MAT文件: {mat_file_path}")
        savemat(mat_file_path, output_data)
        print(f"成功保存MAT文件")
    except Exception as e:
        print(f"保存完整MAT文件时出错: {e}")
        print("尝试保存不包含元数据的简化版本...")
        
        # 尝试保存简化版本
        simple_output = {
            'predictions': predictions,
            'volume_3d': volume
        }
        simple_mat_path = os.path.join(output_dir, 'prediction_results_simple.mat')
        try:
            savemat(simple_mat_path, simple_output)
            print(f"成功保存简化版MAT文件: {simple_mat_path}")
        except Exception as e2:
            print(f"保存简化MAT文件也失败: {e2}")
            
            # 最后尝试分割保存
            print("尝试分别保存各个组件...")
            for key, value in simple_output.items():
                try:
                    component_path = os.path.join(output_dir, f'{key}.mat')
                    savemat(component_path, {key: value})
                    print(f"成功保存组件: {component_path}")
                except:
                    print(f"无法保存组件: {key}")
    
    # 保存一些基本统计信息（作为文本文件）
    stats_file = os.path.join(output_dir, 'prediction_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("预测统计信息\n")
        f.write("=" * 50 + "\n\n")
        
        # 预测类别统计
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        f.write("预测类别分布:\n")
        for cls, count in zip(unique_preds, pred_counts):
            if cls != -1:  # 排除背景/未知类别
                f.write(f"  类别 {cls}: {count} 样本 ({count/len(predictions)*100:.2f}%)\n")
        
        # 体积统计
        f.write("\n体积信息:\n")
        f.write(f"  形状: {volume.shape}\n")
        f.write(f"  占用比例: {np.sum(volume != -1) / volume.size * 100:.2f}%\n")
        
        # 最大概率值统计
        max_probs = np.max(probabilities, axis=1)
        f.write("\n置信度统计:\n")
        f.write(f"  平均置信度: {np.mean(max_probs):.4f}\n")
        f.write(f"  中位数置信度: {np.median(max_probs):.4f}\n")
        f.write(f"  最小置信度: {np.min(max_probs):.4f}\n")
        f.write(f"  最大置信度: {np.max(max_probs):.4f}\n")
    
    print(f"已保存统计信息到: {stats_file}")
    
    return output_dir

def visualize_3d_volume(volume, colormap='jet', save_path=None, show=True, max_points=10000):
    """
    可视化3D体积
    
    参数:
        volume: 3D体积，包含预测结果
        colormap: 颜色映射
        save_path: 保存路径
        show: 是否显示图像
        max_points: 每个类别显示的最大点数
    """
    print("生成3D可视化...")
    
    # 获取唯一类别（排除背景-1）
    unique_classes = np.unique(volume)
    unique_classes = unique_classes[unique_classes != -1]
    
    if len(unique_classes) == 0:
        print("警告：体积中没有有效类别，无法生成可视化")
        return
    
    # 创建颜色映射
    cmap = plt.get_cmap(colormap)
    class_colors = {}
    
    # 为每个类别分配颜色
    for i, cls in enumerate(unique_classes):
        class_colors[cls] = cmap(i / max(1, len(unique_classes) - 1))
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为了更好的渲染性能，可以进行下采样或只显示边界点
    # 这里采用下采样的方法
    downsample = 1
    if max(volume.shape) > 100:
        downsample = max(1, max(volume.shape) // 100)
        print(f"由于体积较大，应用下采样因子: {downsample}")
    
    # 收集每个类别的坐标
    for cls in unique_classes:
        # 获取此类别的所有点
        x, y, z = np.where(volume == cls)
        
        # 下采样
        x = x[::downsample]
        y = y[::downsample]
        z = z[::downsample]
        
        # 如果点数仍然很多，进一步下采样
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]
        
        # 绘制此类别的点
        ax.scatter(x, y, z, c=[class_colors[cls]], marker='.', alpha=0.8, label=f'Class {cls}')
    
    # 设置图表属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Predictions')
    
    # 添加图例（但限制数量，避免过多）
    if len(unique_classes) <= 20:
        ax.legend()
    else:
        # 只显示前10个类别和后10个类别的图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:10] + handles[-10:], labels[:10] + labels[-10:])
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path)
        print(f"已保存3D可视化到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='体素分类预测并映射回3D空间')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径，如果不指定则自动查找')
    parser.add_argument('--data_path', type=str, required=True, help='输入MATLAB文件路径')
    parser.add_argument('--features_key', type=str, default='multidim_data', help='MATLAB文件中特征数据的键名')
    parser.add_argument('--region_key', type=str, default='region', help='MATLAB文件中区域掩码的键名')
    parser.add_argument('--output_dir', type=str, default='./prediction_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--normalize', type=str, default='standard', choices=['standard', 'minmax', 'none'], 
                      help='标准化方法: standard=Z标准化, minmax=最小-最大归一化, none=不标准化')
    parser.add_argument('--threshold', type=float, default=0.0, help='预测概率阈值，低于此值的预测将被忽略')
    parser.add_argument('--save_3d', action='store_true', help='是否保存3D可视化结果')
    parser.add_argument('--colormap', type=str, default='jet', help='3D可视化使用的颜色映射')
    parser.add_argument('--no_display', action='store_true', help='不显示可视化，只保存')
    parser.add_argument('--max_points', type=int, default=10000, help='3D可视化中显示的最大点数')
    
    args = parser.parse_args()
    
    # 加载模型
    model, config, device = load_optimized_model(args.model)
    
    # 加载输入数据
    print(f"加载输入数据: {args.data_path}")
    features, original_shape, metadata = load_matlab_data(
        args.data_path, 
        features_key=args.features_key, 
        region_key=args.region_key
    )
    
    # 检查输入数据
    if features.shape[1] != 341:
        raise ValueError(f"输入数据应有341个特征，但发现了{features.shape[1]}个特征")
    
    # 标准化数据
    if args.normalize != 'none':
        print(f"使用{args.normalize}方法标准化数据")
        features = normalize_voxels(features, method=args.normalize)
    
    # 预测
    print("开始预测...")
    predictions, probabilities = predict_voxels(
        model, 
        features, 
        batch_size=args.batch_size,
        device=device
    )
    
    # 应用阈值
    if args.threshold > 0:
        # 获取每个预测的最大概率
        max_probs = np.max(probabilities, axis=1)
        # 创建掩码，标记低于阈值的样本
        mask = max_probs < args.threshold
        # 将低于阈值的预测设为-1（表示未知）
        predictions[mask] = -1
        print(f"应用阈值 {args.threshold}，{np.sum(mask)} 个样本 ({np.sum(mask)/len(mask)*100:.2f}%) 被标记为未知")
    
    # 打印预测类别分布
    unique_classes, counts = np.unique(predictions, return_counts=True)
    print(f"\n预测出的不同类别数量: {len(unique_classes[unique_classes != -1])}")
    
    # 按类别汇总（显示前10个最常见的类别）
    print("\n按类别汇总 (前10个):")
    class_counts = [(cls, cnt) for cls, cnt in zip(unique_classes, counts) if cls != -1]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    for cls, count in class_counts[:10]:
        percentage = count / len(predictions) * 100
        print(f"类别 {cls}: {count} 个样本 ({percentage:.2f}%)")
    
    # 将预测映射回3D空间
    region_mask = metadata.get('region_mask', None)
    volume, prob_volume = map_predictions_to_3d(
        predictions, 
        original_shape, 
        region_mask=region_mask,
        probabilities=probabilities
    )
    
    # 保存结果
    output_dir = save_results(
        predictions, 
        probabilities, 
        volume, 
        prob_volume=prob_volume, 
        metadata=None,  # 不保存元数据以避免文件过大
        output_dir=args.output_dir
    )
    
    # 可视化 (如果需要)
    if args.save_3d:
        vis_file = os.path.join(output_dir, '3d_visualization.png')
        visualize_3d_volume(
            volume, 
            colormap=args.colormap, 
            save_path=vis_file, 
            show=not args.no_display,
            max_points=args.max_points
        )
        print(f"3D可视化已保存到: {vis_file}")
    
    print("\n预测和3D映射完成!")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()