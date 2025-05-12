#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测脚本：使用训练好的模型对新数据进行预测并进行3D映射
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
import pickle
import time

# 设置PyTorch序列化安全变量
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

# 导入自定义模块
from utils.model_io import load_model_with_architecture, safe_load_model
from data import apply_pca  # 假设你的数据模块中有这个函数

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑体素预测和3D映射工具')
    
    # 基本参数
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='要预测的MAT文件路径')
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./prediction_results', help='结果保存目录')
    parser.add_argument('--device', type=int, default=0, help='使用的设备（-1表示CPU）')
    
    # 数据处理参数
    parser.add_argument('--apply_pca', action='store_true', help='是否应用PCA降维')
    parser.add_argument('--pca_model', type=str, default=None, help='PCA模型路径，如果有的话')
    parser.add_argument('--normalize', action='store_true', help='是否标准化特征')
    parser.add_argument('--region_key', type=str, default=None, help='MAT文件中region掩码的键名')
    parser.add_argument('--features_key', type=str, default=None, help='MAT文件中特征数据的键名')
    
    # 预测参数
    parser.add_argument('--batch_size', type=int, default=1024, help='预测时的批处理大小')
    parser.add_argument('--threshold', type=float, default=0.0, help='预测概率阈值，低于此值的预测将被忽略')
    parser.add_argument('--save_probability_maps', action='store_true', help='是否保存概率体积（可能较大）')
    
    # 可视化参数
    parser.add_argument('--save_3d', action='store_true', help='是否保存3D可视化结果')
    parser.add_argument('--colormap', type=str, default='jet', help='3D可视化使用的颜色映射')
    parser.add_argument('--no_display', action='store_true', help='不显示可视化，只保存')
    parser.add_argument('--max_points', type=int, default=10000, help='3D可视化中显示的最大点数')
    
    return parser.parse_args()

def setup_environment(device_idx):
    """设置环境，包括CUDA设备"""
    # 确定设备
    if device_idx >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_idx}")
        print(f"使用CUDA设备: {torch.cuda.get_device_name(device_idx)}")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    
    return device

def load_config(config_path):
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"已加载配置文件: {config_path}")
        return config
    return {}

def load_matlab_data(data_path, features_key=None, region_key=None):
    """
    加载MATLAB .mat文件数据，包括region掩码
    
    参数:
        data_path: .mat文件路径
        features_key: 特征数据的键名，如果指定则优先使用
        region_key: region掩码的键名，如果指定则优先使用
    
    返回:
        features: 特征矩阵
        original_shape: 原始3D形状
        metadata: 其他元数据
    """
    print(f"加载数据: {data_path}")
    try:
        mat_data = loadmat(data_path)
        
        # 打印mat文件中的键以帮助调试
        print(f"MAT文件中的键: {list(mat_data.keys())}")
        
        # 预期的数据结构可能需要根据你的具体.mat文件调整
        features = None
        original_shape = None
        region_mask = None
        metadata = {}
        
        # 获取特征数据
        if features_key and features_key in mat_data:
            features = mat_data[features_key]
            print(f"使用指定键 '{features_key}' 获取特征数据，形状: {features.shape}")
        else:
            # 尝试常见的键名
            possible_feature_keys = ['data', 'features', 'X', 'voxel_data']
            for key in possible_feature_keys:
                if key in mat_data and mat_data[key] is not None:
                    features = mat_data[key]
                    print(f"找到特征数据，使用键: {key}，形状: {features.shape}")
                    break
        
        # 获取region掩码
        if region_key and region_key in mat_data:
            region_mask = mat_data[region_key]
            print(f"使用指定键 '{region_key}' 获取region掩码，形状: {region_mask.shape}")
            
            # 如果我们有region掩码，可以从中推断原始形状
            original_shape = region_mask.shape
            print(f"从region掩码推断原始形状: {original_shape}")
        else:
            # 尝试常见的键名
            possible_region_keys = ['region', 'mask', 'template', 'brain_mask']
            for key in possible_region_keys:
                if key in mat_data and mat_data[key] is not None:
                    region_mask = mat_data[key]
                    print(f"找到region掩码，使用键: {key}，形状: {region_mask.shape}")
                    
                    # 如果我们有region掩码，可以从中推断原始形状
                    original_shape = region_mask.shape
                    print(f"从region掩码推断原始形状: {original_shape}")
                    break
        
        # 尝试获取原始形状信息(如果region掩码中没有)
        if original_shape is None:
            possible_shape_keys = ['original_shape', 'shape', 'dimensions', 'voxel_shape']
            for key in possible_shape_keys:
                if key in mat_data and mat_data[key] is not None:
                    original_shape = mat_data[key]
                    if isinstance(original_shape, np.ndarray):
                        original_shape = tuple(original_shape.flatten())
                    print(f"找到原始形状信息: {original_shape}")
                    break
        
        # 添加其他可能有用的元数据
        for key in mat_data.keys():
            if key not in ['__header__', '__version__', '__globals__'] and key not in metadata:
                metadata[key] = mat_data[key]
        
        # 验证是否找到了必要数据
        if features is None:
            raise ValueError(f"无法在MAT文件中找到特征数据。可用键: {list(mat_data.keys())}")
            
        metadata['region_mask'] = region_mask
        
        return features, original_shape, metadata
        
    except Exception as e:
        print(f"加载MAT文件出错: {e}")
        raise

def preprocess_features(features, config=None, model_info=None, apply_pca_flag=False, pca_model_path=None, normalize=True):
    """
    预处理特征数据，确保使用与训练时相同的标准化参数
    
    参数:
        features: 特征矩阵
        config: 模型配置信息，包含标准化参数
        model_info: 从checkpoint加载的模型信息
        apply_pca_flag: 是否应用PCA降维
        pca_model_path: PCA模型路径
        normalize: 是否标准化特征
    
    返回:
        processed_features: 处理后的特征
        pca_model: 使用的PCA模型 (若有)
    """
    print(f"预处理特征，原始形状: {features.shape}")
    
    # 确保特征是2D矩阵
    original_dims = features.shape
    if len(features.shape) > 2:
        features = features.reshape(-1, features.shape[-1])
        print(f"将特征重塑为2D矩阵: {features.shape}")
    
    pca_model = None
    
    # 应用PCA (如果需要)
    if apply_pca_flag:
        if pca_model_path and os.path.exists(pca_model_path):
            # 加载已有PCA模型
            print(f"加载PCA模型: {pca_model_path}")
            with open(pca_model_path, 'rb') as f:
                pca_model = pickle.load(f)
            
            # 应用变换
            features = pca_model.transform(features)
            print(f"应用PCA后的特征形状: {features.shape}")
        elif model_info and 'pca_model' in model_info:
            print("使用模型中保存的PCA参数")
            pca_model = model_info['pca_model']
            features = pca_model.transform(features)
            print(f"应用PCA后的特征形状: {features.shape}")
        else:
            print("未找到PCA模型，跳过PCA")
    
    # 使用与训练时相同的标准化参数
    if normalize:
        # 首先尝试从模型信息中获取标准化参数
        mean = None
        std = None
        
        # 从模型信息中提取标准化参数
        if model_info and 'normalization_params' in model_info:
            print("使用模型中保存的标准化参数")
            norm_params = model_info['normalization_params']
            mean = norm_params.get('mean')
            std = norm_params.get('std')
            
            # 如果参数是列表，转换为numpy数组
            if mean is not None and isinstance(mean, list):
                mean = np.array(mean)
            if std is not None and isinstance(std, list):
                std = np.array(std)
        
        # 从配置中提取标准化参数
        elif config and 'normalization_params' in config:
            print("使用配置文件中的标准化参数")
            norm_params = config['normalization_params']
            mean = norm_params.get('mean')
            std = norm_params.get('std')
            
            # 如果参数是列表，转换为numpy数组
            if mean is not None and isinstance(mean, list):
                mean = np.array(mean)
            if std is not None and isinstance(std, list):
                std = np.array(std)
        
        # 如果找不到保存的参数，则尝试从训练数据集加载
        if mean is None or std is None:
            try:
                # 这里仅为示例，你需要根据实际情况调整
                from data import load_multiclass_data
                
                print("未找到保存的标准化参数，尝试从训练集计算...")
                if config and 'data_dirs' in config:
                    dataset_dict = load_multiclass_data(
                        config['data_dirs'],
                        apply_pca_flag=False,
                        norm=False  # 先不标准化，我们要获取原始值
                    )
                    
                    # 合并所有训练数据来计算标准化参数
                    all_train_samples = dataset_dict['train_samples']
                    
                    mean = np.mean(all_train_samples, axis=0)
                    std = np.std(all_train_samples, axis=0)
                    # 避免除零
                    std[std == 0] = 1e-10
                    
                    print("成功从训练集计算标准化参数")
                else:
                    raise ValueError("无法找到标准化参数，且无法从训练集计算")
            except Exception as e:
                print(f"尝试计算标准化参数时出错: {e}")
                print("警告：将使用当前数据进行标准化，这可能导致不一致的结果")
                # 如果无法获取训练时的标准化参数，退回到使用当前数据
                mean = np.mean(features, axis=0)
                std = np.std(features, axis=0)
                std[std == 0] = 1e-10
        
        # 应用标准化
        print("应用标准化处理")
        features = (features - mean) / std
        
        # 保存使用的标准化参数，便于调试
        normalization_used = {
            'mean': mean,
            'std': std,
            'source': 'model' if model_info and 'normalization_params' in model_info else 
                     'config' if config and 'normalization_params' in config else 
                     'train_data' if 'dataset_dict' in locals() else 'current_data'
        }
        
        print(f"使用标准化参数源: {normalization_used['source']}")
    
    return features, pca_model

def predict_with_model(model, features, device, batch_size=1024, threshold=0.0):
    """
    使用模型进行预测
    
    参数:
        model: 加载的模型
        features: 处理后的特征
        device: 计算设备
        batch_size: 批处理大小
        threshold: 预测概率阈值
    
    返回:
        predictions: 预测的类别
        probabilities: 预测的概率
    """
    print("使用模型进行预测...")
    
    # 确保模型处于评估模式
    model.eval()
    
    # 获取样本数量
    n_samples = features.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 初始化结果存储
    all_probs = []
    all_preds = []
    
    # 批处理预测
    with torch.no_grad():
        for i in range(n_batches):
            # 获取当前批次
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_features = features[start_idx:end_idx]
            
            # 转换为Tensor
            batch_tensor = torch.FloatTensor(batch_features).to(device)
            
            # 前向传播
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            # 转移到CPU并转换为NumPy
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"已处理 {end_idx}/{n_samples} 个样本 ({end_idx/n_samples*100:.1f}%)")
    
    # 合并所有批次结果
    probabilities = np.vstack(all_probs)
    predictions = np.concatenate(all_preds)
    
    # 应用阈值
    if threshold > 0:
        # 获取每个预测的最大概率
        max_probs = np.max(probabilities, axis=1)
        # 创建掩码，标记低于阈值的样本
        mask = max_probs < threshold
        # 将低于阈值的预测设为-1（表示未知）
        predictions[mask] = -1
        
        print(f"应用阈值 {threshold}，{np.sum(mask)} 个样本 ({np.sum(mask)/len(mask)*100:.2f}%) 被标记为未知")
    
    # 将预测类别从模型输出 (0 to n-1) 映射回原始标签 (1 to n)
    # 假设背景类为-1，其他类从0开始
    valid_mask = predictions != -1
    predictions[valid_mask] += 1  # 将0-(n-1)映射到1-n
    
    return predictions, probabilities

def map_predictions_to_3d(predictions, original_shape, region_mask=None, metadata=None):
    """
    将预测结果映射回3D空间，使用与MATLAB revert_reshape相同的逻辑
    
    参数:
        predictions: 预测的类别或概率矩阵 [n_voxels, n_classes] 或 [n_voxels]
        original_shape: 原始3D体积的形状
        region_mask: 掩码，指定哪些位置应该被填充 (等同于MATLAB中的region参数)
        metadata: 其他元数据
        
    返回:
        volume: 3D或4D体积，包含映射后的预测结果
    """
    print(f"将预测映射回3D空间，目标形状: {original_shape}")
    
    # 检查预测是否为多类别概率
    multi_class = len(predictions.shape) > 1
    
    if multi_class:
        # 创建4D体积 [x, y, z, n_classes]
        n_classes = predictions.shape[1]
        volume = np.zeros(original_shape + (n_classes,), dtype=np.float32)
        
        # 如果有region_mask，使用它来确定哪些位置应该被填充
        if region_mask is not None:
            # 确保region_mask是布尔型
            if not np.issubdtype(region_mask.dtype, np.bool_):
                region_mask = region_mask > 0
                
            # 为每个类别单独处理
            for c in range(n_classes):
                # 创建临时3D体积
                temp_vol = np.zeros(original_shape, dtype=np.float32)
                # 将掩码展平为1D
                mask_flat = region_mask.flatten()
                # 只有掩码为True的位置才填充预测值
                temp_vol_flat = temp_vol.flatten()
                temp_vol_flat[mask_flat] = predictions[:, c]
                # 将1D数组重新整形为3D
                temp_vol = temp_vol_flat.reshape(original_shape)
                # 保存到4D体积的相应通道
                volume[..., c] = temp_vol
        else:
            # 如果没有region_mask，假设所有体素都是有效的
            # 将1D预测重塑为4D体积
            for c in range(n_classes):
                volume[..., c] = predictions[:, c].reshape(original_shape)
    else:
        # 创建3D体积
        volume = np.zeros(original_shape, dtype=np.int32) - 1  # 初始化为-1（未知）
        
        # 如果有region_mask，使用它来确定哪些位置应该被填充
        if region_mask is not None:
            # 确保region_mask是布尔型
            if not np.issubdtype(region_mask.dtype, np.bool_):
                region_mask = region_mask > 0
                
            # 创建临时3D体积
            temp_vol = np.full(original_shape, -1, dtype=np.int32)  # 初始化为-1
            # 将掩码展平为1D
            mask_flat = region_mask.flatten()
            # 只有掩码为True的位置才填充预测值
            temp_vol_flat = temp_vol.flatten()
            temp_vol_flat[mask_flat] = predictions
            # 将1D数组重新整形为3D
            volume = temp_vol_flat.reshape(original_shape)
        else:
            # 如果没有region_mask，假设特征和体积的排列顺序相同
            volume = predictions.reshape(original_shape)
    
    # 计算统计信息
    if not multi_class:
        unique_classes = np.unique(predictions)
        class_counts = {cls: np.sum(predictions == cls) for cls in unique_classes if cls != -1}
        
        print(f"体积中的唯一类别: {len(class_counts)}")
        print(f"前5个类别统计: {sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return volume

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

def save_results(predictions, probabilities, volume, probability_volume=None, metadata=None, output_dir=None):
    """
    保存预测结果
    
    参数:
        predictions: 预测的类别
        probabilities: 预测的概率
        volume: 3D体积
        probability_volume: 概率体积（可选）
        metadata: 原始数据的元数据
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = './prediction_results_' + time.strftime("%Y%m%d_%H%M%S")
    
    print(f"保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为NumPy文件
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'probabilities.npy'), probabilities)
    np.save(os.path.join(output_dir, 'volume_3d.npy'), volume)
    
    if probability_volume is not None:
        np.save(os.path.join(output_dir, 'probability_volume.npy'), probability_volume)
    
    # 保存为MAT文件
    output_data = {
        'predictions': predictions,
        'probabilities': probabilities,
        'volume_3d': volume
    }
    
    if probability_volume is not None:
        output_data['probability_volume'] = probability_volume
    
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
    try:
        savemat(os.path.join(output_dir, 'prediction_results.mat'), output_data)
    except Exception as e:
        print(f"保存MAT文件时出错: {e}")
        print("尝试保存不包含元数据的简化版本...")
        
        # 尝试保存简化版本
        simple_output = {
            'predictions': predictions,
            'volume_3d': volume
        }
        savemat(os.path.join(output_dir, 'prediction_results_simple.mat'), simple_output)
    
    # 保存一些基本统计信息
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

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config_path)
    
    # 设置环境
    device = setup_environment(args.device)
    
    try:
        # 加载模型
        print(f"\n加载模型: {args.model_path}")
        model, checkpoint = load_model_with_architecture(args.model_path, device)
        print("模型加载成功")
        
        # 获取模型信息
        model_info = checkpoint.get('model_arch_info', {})
        training_info = checkpoint.get('training_info', {})
        
        # 加载数据
        features, original_shape, metadata = load_matlab_data(
            args.data_path, 
            features_key=args.features_key,
            region_key=args.region_key
        )
        
        # 获取region_mask
        region_mask = metadata.get('region_mask', None)
        
        # 预处理特征
        processed_features, pca_model = preprocess_features(
            features,
            config=config,
            model_info=model_info,
            apply_pca_flag=args.apply_pca,
            pca_model_path=args.pca_model,
            normalize=args.normalize
        )
        
        # 检查特征维度是否与模型兼容
        expected_dim = model_info.get('feature_dim', None)
        if expected_dim and processed_features.shape[1] != expected_dim:
            print(f"警告: 预处理后的特征维度 ({processed_features.shape[1]}) 与模型期望的维度 ({expected_dim}) 不匹配!")
            print("这可能导致错误或不准确的预测结果。")
            
            # 如果可能，尝试调整维度
            if args.apply_pca:
                print("建议: 请确保使用与训练模型相同的PCA配置。")
            else:
                print("建议: 如果训练使用了PCA，请添加 --apply_pca 参数并提供相同的PCA模型。")
        
        # 进行预测
        predictions, probabilities = predict_with_model(
            model, 
            processed_features, 
            device,
            batch_size=args.batch_size,
            threshold=args.threshold
        )
        
        # 映射回3D
        volume = map_predictions_to_3d(
            predictions, 
            original_shape, 
            region_mask=region_mask,
            metadata=metadata
        )
        
        # 如果需要，映射概率体积
        probability_volume = None
        if args.save_probability_maps:
            print("映射概率体积...")
            probability_volume = map_predictions_to_3d(
                probabilities,
                original_shape,
                region_mask=region_mask,
                metadata=metadata
            )
        
        # 保存结果
        output_dir = save_results(
            predictions, 
            probabilities, 
            volume, 
            probability_volume=probability_volume, 
            metadata=metadata, 
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
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())