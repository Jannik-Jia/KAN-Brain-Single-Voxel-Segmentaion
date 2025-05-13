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
import joblib  # 新增: 用于加载保存的scaler
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
    加载优化后的模型，如果是完整模型直接加载，否则尝试从同名JSON文件读取配置，
    如果两者都失败则报错。
    
    参数:
        model_path: 模型文件路径，如果为None则自动查找
        
    返回:
        model: 加载好的模型
        config: 模型配置
        device: 计算设备
        normalization_params: 标准化参数(如果模型中包含)
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
    
    # 策略1: 尝试加载完整模型
    try:
        print("尝试加载完整模型...")
        checkpoint = safe_load_model(model_path, device)
        
        # 检查是否包含模型架构信息
        if 'model_arch_info' in checkpoint:
            print("成功! 这是一个完整模型，包含架构信息")
            model_config = checkpoint['model_arch_info']
            
            # 尝试从检查点获取标准化参数
            normalization_params = None
            if 'normalization_params' in checkpoint:
                normalization_params = checkpoint['normalization_params']
                print("从模型检查点获取到标准化参数")
            elif 'training_info' in checkpoint and 'normalization_params' in checkpoint['training_info']:
                normalization_params = checkpoint['training_info']['normalization_params']
                print("从训练信息中获取到标准化参数")
            
            # 获取模型架构参数
            model_type = model_config.get('model_type', 'deep_mlp')
            input_dim = model_config.get('input_dim', 341)
            # 支持两种可能的隐藏层参数名
            hidden_dims = model_config.get('hidden_dims', model_config.get('hidden_units', [2048] * 6))
            num_classes = model_config.get('num_classes', model_config.get('num_class', 102))
            dropout_rate = model_config.get('dropout_rate', 0.2567125567148536)
            activation = model_config.get('activation', 'gelu')
            use_skip_connections = model_config.get('use_skip_connections', True)
            
            # 打印关键模型配置
            print(f"模型类型: {model_type}")
            print(f"输入维度: {input_dim}")
            print(f"隐藏层: {hidden_dims}")
            print(f"输出类别数: {num_classes}")
            print(f"Dropout率: {dropout_rate}")
            print(f"激活函数: {activation}")
            
            # 创建模型
            model = get_model(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                activation=activation,
                use_skip_connections=use_skip_connections
            )
            
            # 加载模型权重
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            model.eval()
            print("成功加载模型权重")
            
            return model, model_config, device, normalization_params
        else:
            print("模型检查点不包含架构信息，将尝试从同名JSON文件加载")
    except Exception as e:
        print(f"从完整模型加载失败: {e}")
        print("将尝试从JSON文件加载配置")
    
    # 策略2: 尝试从同名JSON配置文件加载
    json_config_path = None
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # 检查几种可能的JSON配置文件名称
    possible_json_names = [
        os.path.splitext(model_name)[0] + '_architecture.json',  # 标准命名格式
        os.path.splitext(model_name)[0] + '.json',               # 简化命名
        'architecture.json',                                      # 通用架构文件
        'config.json',                                            # 通用配置
        'optimized_config.json'                                   # 优化后的配置
    ]
    
    for json_name in possible_json_names:
        json_path = os.path.join(model_dir, json_name)
        if os.path.exists(json_path):
            json_config_path = json_path
            print(f"找到模型配置文件: {json_config_path}")
            break
    
    if json_config_path:
        try:
            with open(json_config_path, 'r') as f:
                model_config = json.load(f)
            print("成功从JSON文件加载模型配置")
            
            # 再次尝试加载检查点以获取权重
            checkpoint = safe_load_model(model_path, device)
            
            # 尝试获取标准化参数
            normalization_params = None
            if 'normalization_params' in model_config:
                normalization_params = model_config['normalization_params']
                print("从配置文件获取到标准化参数")
            
            # 获取模型架构参数
            model_type = model_config.get('model_type', 'deep_mlp')
            input_dim = model_config.get('input_dim', 341)
            hidden_dims = model_config.get('hidden_dims', model_config.get('hidden_units', [2048] * 6))
            num_classes = model_config.get('num_classes', model_config.get('num_class', 102))
            dropout_rate = model_config.get('dropout_rate', 0.2567125567148536)
            activation = model_config.get('activation', 'gelu')
            use_skip_connections = model_config.get('use_skip_connections', True)
            
            # 打印关键模型配置
            print(f"模型类型: {model_type}")
            print(f"输入维度: {input_dim}")
            print(f"隐藏层: {hidden_dims}")
            print(f"输出类别数: {num_classes}")
            print(f"Dropout率: {dropout_rate}")
            print(f"激活函数: {activation}")
            
            # 创建模型
            model = get_model(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                activation=activation,
                use_skip_connections=use_skip_connections
            )
            
            # 加载模型权重
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            model.eval()
            print("成功从JSON配置创建模型并加载权重")
            
            return model, model_config, device, normalization_params
            
        except Exception as e:
            print(f"从JSON配置加载失败: {e}")
    else:
        print("无法找到相关的JSON配置文件")
    
    # 两种策略都失败，报错
    raise ValueError(
        "无法加载模型: 既不能作为完整模型加载，也找不到有效的JSON配置文件。"
        "请确保模型文件包含架构信息或者同目录下有对应的JSON配置文件。"
    )


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

def save_essential_results(predictions, probabilities, volume, output_dir):
    """
    只保存必要的三个文件：predictions.mat, probabilities.mat, volume_3d.mat
    
    参数:
        predictions: 预测的类别 (1D数组)
        probabilities: 预测的概率 (2D数组)
        volume: 3D体积
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = './prediction_results_' + time.strftime("%Y%m%d_%H%M%S")
    
    print(f"保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测标签
    print(f"保存预测标签，形状: {predictions.shape}")
    savemat(os.path.join(output_dir, 'predictions.mat'), {'predictions': predictions})
    
    # 保存预测概率
    print(f"保存预测概率，形状: {probabilities.shape}")
    savemat(os.path.join(output_dir, 'probabilities.mat'), {'probabilities': probabilities})
    
    # 保存3D体积
    print(f"保存3D体积，形状: {volume.shape}")
    savemat(os.path.join(output_dir, 'volume_3d.mat'), {'volume_3d': volume})
    
    print(f"文件保存完成。")
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


def revert_reshape_python(array, region):
    """
    Python实现的revert_reshape函数，与MATLAB版本完全一致
    
    参数:
        array: 数组，可以是1D(n_samples,)或2D(n_samples, n_features)
        region: 3D掩码，指示哪些位置有效
    
    返回:
        big_img: 重建的3D或4D图像
    """
    # 确保region是布尔值
    if not np.issubdtype(region.dtype, np.bool_):
        region = region > 0
    
    # 确保array是2D的
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)
    
    # 创建输出数组 - 与MATLAB一致，使用single精度
    big_img = np.zeros(region.shape + (array.shape[1],), dtype=np.float32)
    
    # 对每一列进行处理
    for ii in range(array.shape[1]):
        # 提取当前切片
        img = big_img[..., ii]
        
        # 展平图像和掩码
        img_index = img.flatten()
        template_index = region.flatten()
        
        # 在掩码为True的位置填入值
        img_index[template_index] = array[:, ii]
        
        # 重塑并保存回原数组
        img = img_index.reshape(region.shape)
        big_img[..., ii] = img
    
    # 如果原始数组是1D的，返回3D结果
    if array.shape[1] == 1:
        return big_img[..., 0]
    else:
        return big_img
    


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


# 可以移除 compute_normalization_params_from_datasets 函数，因为我们将使用保存的参数
def normalize_data_with_params(features, mean=None, std=None, scaler=None):
    """
    使用预先计算的均值和标准差或者保存的scaler对数据进行标准化
    
    参数:
        features: 需要标准化的特征数据
        mean: 均值向量
        std: 标准差向量
        scaler: 保存的StandardScaler对象
        
    返回:
        normalized_features: 标准化后的特征数据
    """
    if scaler is not None:
        # 优先使用scaler对象
        try:
            return scaler.transform(features)
        except Exception as e:
            print(f"使用scaler标准化失败: {e}")
            # 如果scaler失败，尝试使用均值和标准差
            if mean is not None and std is not None:
                print("回退到使用均值和标准差标准化")
            else:
                raise
    
    if mean is not None and std is not None:
        # 使用均值和标准差
        # 确保mean和std是numpy数组
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean)
        if not isinstance(std, np.ndarray):
            std = np.array(std)
            
        # 确保维度匹配
        if mean.shape[0] != features.shape[1] or std.shape[0] != features.shape[1]:
            raise ValueError(f"特征维度不匹配: 特征维度={features.shape[1]}, 均值维度={mean.shape[0]}, 标准差维度={std.shape[0]}")
            
        # 避免除零
        std_safe = std.copy()
        std_safe[std_safe < 1e-10] = 1e-10
        
        # 执行标准化
        return (features - mean) / std_safe
    
    raise ValueError("必须提供scaler对象或者均值和标准差")

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='体素分类预测并映射回3D空间')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径，如果不指定则自动查找')
    parser.add_argument('--data_path', type=str, required=True, help='输入MATLAB文件路径')
    parser.add_argument('--features_key', type=str, default='multidim_data', help='MATLAB文件中特征数据的键名')
    parser.add_argument('--region_key', type=str, default='region', help='MATLAB文件中区域掩码的键名')
    parser.add_argument('--output_dir', type=str, default='./prediction_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--scaler_path', type=str, default=None, help='StandardScaler保存路径，如果不指定则尝试从模型中读取')
    parser.add_argument('--threshold', type=float, default=0.0, help='预测概率阈值，低于此值的预测将被忽略')
    parser.add_argument('--save_3d', action='store_true', help='是否保存3D可视化结果')
    parser.add_argument('--colormap', type=str, default='jet', help='3D可视化使用的颜色映射')
    parser.add_argument('--no_display', action='store_true', help='不显示可视化，只保存')
    parser.add_argument('--max_points', type=int, default=10000, help='3D可视化中显示的最大点数')
    args = parser.parse_args()
    
    # 加载模型（现在会返回标准化参数）
    model, config, device, normalization_params = load_optimized_model(args.model)
    
    # 如果未指定scaler_path，尝试自动查找
    if not args.scaler_path:
        model_dir = os.path.dirname(args.model) if args.model else "."
        model_name = os.path.splitext(os.path.basename(args.model))[0] if args.model else ""
        
        # 可能的scaler文件名模式
        possible_scaler_names = [
            f"{model_name}_scaler.pkl",
            f"{model_name.split('_epoch_')[0]}_scaler.pkl" if "_epoch_" in model_name else "",  # 处理有epoch信息的模型名
            "model_scaler.pkl",
            "scaler.pkl"
        ]
        
        for scaler_name in possible_scaler_names:
            if not scaler_name:  # 跳过空字符串
                continue
            scaler_path = os.path.join(model_dir, scaler_name)
            if os.path.exists(scaler_path):
                args.scaler_path = scaler_path
                print(f"自动找到scaler: {args.scaler_path}")
                break
    
    # 加载scaler（如果提供路径）
    scaler = None
    if args.scaler_path and os.path.exists(args.scaler_path):
        try:
            scaler = joblib.load(args.scaler_path)
            print(f"从 {args.scaler_path} 加载 StandardScaler")
        except Exception as e:
            print(f"加载 StandardScaler 时出错: {e}")
    
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
    print("\n开始标准化数据...")
    try:
        if scaler is not None:
            # 使用加载的scaler
            print(f"使用从 {args.scaler_path} 加载的StandardScaler")
            
            # 打印一些scaler的统计信息，便于调试
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'var_'):
                print(f"Scaler均值范围: [{np.min(scaler.mean_):.6f}, {np.max(scaler.mean_):.6f}]")
                print(f"Scaler标准差范围: [{np.min(np.sqrt(scaler.var_)):.6f}, {np.max(np.sqrt(scaler.var_)):.6f}]")
            
            # 记录标准化前的统计信息
            features_mean_before = np.mean(features, axis=0)
            features_std_before = np.std(features, axis=0)
            print(f"标准化前数据均值范围: [{np.min(features_mean_before):.6f}, {np.max(features_mean_before):.6f}]")
            print(f"标准化前数据标准差范围: [{np.min(features_std_before):.6f}, {np.max(features_std_before):.6f}]")
            
            # 应用标准化
            features = scaler.transform(features)
            
            # 记录标准化后的统计信息
            features_mean_after = np.mean(features, axis=0)
            features_std_after = np.std(features, axis=0)
            print(f"标准化后数据均值范围: [{np.min(features_mean_after):.6f}, {np.max(features_mean_after):.6f}]")
            print(f"标准化后数据标准差范围: [{np.min(features_std_after):.6f}, {np.max(features_std_after):.6f}]")
            
        elif normalization_params is not None:
            # 使用从模型中提取的标准化参数
            mean = np.array(normalization_params['mean'])
            std = np.array(normalization_params['std'])
            
            print("使用训练模型中保存的标准化参数")
            print(f"标准化参数均值范围: [{np.min(mean):.6f}, {np.max(mean):.6f}]")
            print(f"标准化参数标准差范围: [{np.min(std):.6f}, {np.max(std):.6f}]")
            
            # 记录标准化前的统计信息
            features_mean_before = np.mean(features, axis=0)
            features_std_before = np.std(features, axis=0)
            print(f"标准化前数据均值范围: [{np.min(features_mean_before):.6f}, {np.max(features_mean_before):.6f}]")
            print(f"标准化前数据标准差范围: [{np.min(features_std_before):.6f}, {np.max(features_std_before):.6f}]")
            
            # 应用标准化
            features = normalize_data_with_params(features, mean, std)
            
            # 记录标准化后的统计信息
            features_mean_after = np.mean(features, axis=0)
            features_std_after = np.std(features, axis=0)
            print(f"标准化后数据均值范围: [{np.min(features_mean_after):.6f}, {np.max(features_mean_after):.6f}]")
            print(f"标准化后数据标准差范围: [{np.min(features_std_after):.6f}, {np.max(features_std_after):.6f}]")
        else:
            print("警告: 无法获取标准化参数，将使用原始数据进行预测")
            print("这可能导致预测结果不准确，因为模型在标准化数据上训练")
            
            # 提供用户选择
            print("\n请选择:")
            print("1. 使用原始数据继续预测 (不推荐)")
            print("2. 退出程序")
            
            # 在非交互环境中默认继续
            try:
                choice = input("请选择 (1/2, 默认1): ").strip()
                if choice == "2":
                    print("用户选择退出程序")
                    return
            except:
                print("非交互环境，默认继续使用原始数据")
            
            print("继续使用原始数据进行预测")
            
    except Exception as e:
        print(f"标准化数据时出错: {e}")
        print("将使用原始数据进行预测，但这可能导致结果不准确")
    
    # 预测
    print("\n开始预测...")
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
    
    # 获取区域掩码
    region_mask = metadata.get('region_mask', None)
    if region_mask is None:
        print("警告: 未找到区域掩码，将尝试使用原始形状直接重塑")
        # 创建一个全1的掩码
        region_mask = np.ones(original_shape, dtype=bool)
        # 确保掩码中的True数量与预测数量一致
        n_elements = np.prod(original_shape)
        if n_elements > len(predictions):
            # 如果掩码太大，只使用部分
            flat_mask = region_mask.flatten()
            flat_mask[len(predictions):] = False
            region_mask = flat_mask.reshape(original_shape)
        elif n_elements < len(predictions):
            print(f"错误: 原始形状({original_shape})元素数量小于预测数量({len(predictions)})")
            # 尝试猜测更合适的形状
            side = int(np.ceil(len(predictions)**(1/3)))
            new_shape = (side, side, side)
            print(f"尝试使用新形状: {new_shape}")
            region_mask = np.zeros(new_shape, dtype=bool)
            flat_mask = region_mask.flatten()
            flat_mask[:len(predictions)] = True
            region_mask = flat_mask.reshape(new_shape)
            original_shape = new_shape
    
    # 将预测映射回3D空间
    print(f"将预测映射回3D空间，目标形状: {original_shape}")
    volume = revert_reshape_python(predictions, region_mask)
    
    # 如果需要概率体积，也使用revert_reshape_python
    prob_volume = None
    if args.save_3d or 'prob_volume' in locals():
        prob_volume = revert_reshape_python(probabilities, region_mask)
    
    # 创建并确保输出目录存在
    if args.output_dir is None:
        args.output_dir = './prediction_results_' + time.strftime("%Y%m%d_%H%M%S")
    
    # 保存结果
    output_dir = save_essential_results(
        predictions, 
        probabilities, 
        volume, 
        output_dir=args.output_dir
    )
    
    # 保存标准化信息以便追踪
    norm_info_path = os.path.join(output_dir, 'normalization_info.txt')
    with open(norm_info_path, 'w') as f:
        f.write("标准化信息\n")
        f.write("=" * 50 + "\n\n")
        
        if scaler is not None:
            f.write(f"使用了从 {args.scaler_path} 加载的StandardScaler\n\n")
            
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'var_'):
                f.write(f"Scaler均值范围: [{np.min(scaler.mean_):.6f}, {np.max(scaler.mean_):.6f}]\n")
                f.write(f"Scaler标准差范围: [{np.min(np.sqrt(scaler.var_)):.6f}, {np.max(np.sqrt(scaler.var_)):.6f}]\n\n")
        elif normalization_params is not None:
            f.write("使用了从模型中提取的标准化参数\n\n")
            
            mean = np.array(normalization_params['mean'])
            std = np.array(normalization_params['std'])
            f.write(f"标准化参数均值范围: [{np.min(mean):.6f}, {np.max(mean):.6f}]\n")
            f.write(f"标准化参数标准差范围: [{np.min(std):.6f}, {np.max(std):.6f}]\n\n")
        else:
            f.write("警告: 未使用任何标准化\n\n")
        
        # 记录数据统计信息
        f.write("预测数据统计:\n")
        f.write(f"特征维度: {features.shape[1]}\n")
        f.write(f"样本数量: {features.shape[0]}\n")
        f.write(f"预测的类别数量: {len(unique_classes[unique_classes != -1])}\n\n")
        
        # 记录预测分布
        f.write("预测类别分布 (前10个):\n")
        for i, (cls, count) in enumerate(class_counts[:10]):
            percentage = count / len(predictions) * 100
            f.write(f"类别 {cls}: {count} 个样本 ({percentage:.2f}%)\n")
    
    print(f"标准化信息已保存到: {norm_info_path}")
    
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
    
    # 提供使用建议
    print("\n使用建议:")
    if scaler is not None:
        print(f"✓ 成功使用了与训练相同的StandardScaler")
    elif normalization_params is not None:
        print(f"✓ 成功使用了从模型中提取的标准化参数")
    else:
        print("⚠ 警告: 未使用任何标准化，预测结果可能不准确")
    
    if args.threshold > 0:
        print(f"✓ 应用了概率阈值 {args.threshold}，低置信度预测被标记为未知")
    else:
        print("ℹ 提示: 可以使用 --threshold 参数设置概率阈值，过滤低置信度预测")
    
    return output_dir


if __name__ == "__main__":
    main()
