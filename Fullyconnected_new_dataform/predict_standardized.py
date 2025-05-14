#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门针对demo38数据的预测脚本，先标准化再预测
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from torch.utils.data import DataLoader, Dataset
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler

# 导入必要的模块
from models import get_model
from utils.model_io import safe_load_model

class SingleVoxelDataset(Dataset):
    def __init__(self, voxels):
        self.voxels = voxels
    
    def __len__(self):
        return len(self.voxels)
    
    def __getitem__(self, idx):
        x = self.voxels[idx]
        x = torch.FloatTensor(x)
        return x

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    checkpoint = safe_load_model(model_path, device)
    if 'model_arch_info' in checkpoint:
        model_config = checkpoint['model_arch_info']
        
        model_type = model_config.get('model_type', 'deep_mlp')
        input_dim = model_config.get('input_dim', 341)
        hidden_dims = model_config.get('hidden_dims', model_config.get('hidden_units', [2048] * 6))
        num_classes = model_config.get('num_classes', model_config.get('num_class', 102))
        dropout_rate = model_config.get('dropout_rate', 0.5)
        activation = model_config.get('activation', 'relu')
        use_skip_connections = model_config.get('use_skip_connections', True)
        
        model = get_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation,
            use_skip_connections=use_skip_connections
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, device
    else:
        raise ValueError("模型文件不包含架构信息")

def load_data(data_path, features_key="multidim_data", region_key="region"):
    print(f"加载数据: {data_path}")
    
    try:
        f = h5py.File(data_path, 'r')
        print(f"数据文件键: {list(f.keys())}")
        
        features = np.array(f[features_key]).T
        print(f"特征数据形状: {features.shape}")
        
        region_mask = np.array(f[region_key]).T
        print(f"区域掩码形状: {region_mask.shape}")
        
        f.close()
        return features, region_mask
    except Exception as e:
        print(f"加载数据出错: {e}")
        raise

def standardize_features(features):
    """先标准化特征再预测"""
    print("\n对特征进行标准化...")
    print(f"标准化前均值范围: [{np.min(np.mean(features, axis=0)):.6f}, {np.max(np.mean(features, axis=0)):.6f}]")
    print(f"标准化前标准差范围: [{np.min(np.std(features, axis=0)):.6f}, {np.max(np.std(features, axis=0)):.6f}]")
    
    # 应用标准化
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    
    print(f"标准化后均值范围: [{np.min(np.mean(features_std, axis=0)):.6f}, {np.max(np.mean(features_std, axis=0)):.6f}]")
    print(f"标准化后标准差范围: [{np.min(np.std(features_std, axis=0)):.6f}, {np.max(np.std(features_std, axis=0)):.6f}]")
    
    return features_std

def predict(model, features, batch_size=128, device=None):
    """预测类别和概率"""
    dataset = SingleVoxelDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    
    print("\n开始预测...")
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    predictions = np.argmax(all_probs, axis=1)
    
    # 转换为从1开始的标签
    predictions = predictions + 1
    
    print(f"预测完成，形状: {predictions.shape}")
    return predictions, all_probs

def map_to_3d(predictions, probabilities, region_mask):
    """将预测映射回3D空间"""
    print("\n将预测映射回3D空间...")
    
    # 确保region_mask是布尔型
    if not np.issubdtype(region_mask.dtype, np.bool_):
        region_mask = region_mask > 0
    
    # 创建3D体积
    volume = np.zeros(region_mask.shape, dtype=np.int32) - 1
    
    # 创建概率体积
    n_classes = probabilities.shape[1]
    prob_volume = np.zeros(region_mask.shape + (n_classes,), dtype=np.float32)
    
    # 展平掩码并获取True索引
    flat_mask = region_mask.flatten()
    mask_indices = np.where(flat_mask)[0]
    
    # 验证预测数量与掩码中True数量是否匹配
    if len(mask_indices) != len(predictions):
        print(f"警告: 预测数量({len(predictions)})与掩码中的True数量({len(mask_indices)})不匹配")
        min_len = min(len(predictions), len(mask_indices))
        mask_indices = mask_indices[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
    
    # 填充预测到3D体积
    volume_flat = volume.flatten()
    volume_flat[mask_indices] = predictions
    volume = volume_flat.reshape(region_mask.shape)
    
    # 填充概率到4D体积
    for c in range(n_classes):
        prob_flat = np.zeros(np.prod(region_mask.shape), dtype=np.float32)
        prob_flat[mask_indices] = probabilities[:, c]
        prob_volume[..., c] = prob_flat.reshape(region_mask.shape)
    
    # 统计信息
    n_valid = np.sum(volume != -1)
    print(f"有效体素数量: {n_valid} / {volume.size} ({n_valid/volume.size*100:.2f}%)")
    
    return volume, prob_volume

def save_results(predictions, probabilities, volume, prob_volume, output_dir):
    """保存预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n保存结果到: {output_dir}")
    
    # 保存预测标签
    savemat(os.path.join(output_dir, 'predictions.mat'), {'predictions': predictions})
    
    # 保存预测概率
    savemat(os.path.join(output_dir, 'probabilities.mat'), {'probabilities': probabilities})
    
    # 保存3D体积
    savemat(os.path.join(output_dir, 'volume_3d.mat'), {'volume_3d': volume})
    
    # 保存概率体积（可选）
    savemat(os.path.join(output_dir, 'probability_volume.mat'), {'probability_volume': prob_volume})
    
    # 保存预测统计信息
    with open(os.path.join(output_dir, 'prediction_stats.txt'), 'w') as f:
        f.write("预测统计信息\n")
        f.write("=" * 50 + "\n\n")
        
        unique_classes, counts = np.unique(predictions, return_counts=True)
        f.write("类别分布:\n")
        
        # 按频率排序
        sorted_indices = np.argsort(counts)[::-1]
        
        for i in range(min(20, len(sorted_indices))):
            idx = sorted_indices[i]
            cls = unique_classes[idx]
            count = counts[idx]
            percentage = count / len(predictions) * 100
            f.write(f"类别 {cls}: {count} 个样本 ({percentage:.2f}%)\n")
    
    print("结果已保存")
    return output_dir

def visualize_3d(volume, output_dir, max_points=10000):
    """可视化3D预测结果"""
    print("\n生成3D可视化...")
    
    # 获取唯一类别（排除背景-1）
    unique_classes = np.unique(volume)
    unique_classes = unique_classes[unique_classes != -1]
    
    if len(unique_classes) == 0:
        print("警告：体积中没有有效类别，无法生成可视化")
        return
    
    # 创建颜色映射
    cmap = plt.get_cmap('jet')
    class_colors = {}
    
    # 为每个类别分配颜色
    for i, cls in enumerate(unique_classes):
        class_colors[cls] = cmap(i / max(1, len(unique_classes) - 1))
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 下采样
    downsample = max(1, max(volume.shape) // 100)
    print(f"应用下采样因子: {downsample}")
    
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
    ax.set_title('3D Visualization of Predictions (Standard Normalized)')
    
    # 添加图例
    if len(unique_classes) <= 20:
        ax.legend()
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:10] + handles[-10:], labels[:10] + labels[-10:])
    
    # 保存图表
    vis_path = os.path.join(output_dir, '3d_visualization.png')
    plt.savefig(vis_path)
    plt.close()
    
    print(f"3D可视化已保存到: {vis_path}")

def main():
    # 命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='使用先标准化再预测的方法进行预测')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data', type=str, required=True, help='demo38数据文件路径')
    parser.add_argument('--output', type=str, default='./prediction_results_standardized', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    args = parser.parse_args()
    
    # 全局开始时间
    start_time = time.time()
    
    # 1. 加载模型
    model, device = load_model(args.model)
    
    # 2. 加载数据
    features, region_mask = load_data(args.data)
    
    # 3. 标准化特征
    features_std = standardize_features(features)
    
    # 4. 预测
    predictions, probabilities = predict(model, features_std, batch_size=args.batch_size, device=device)
    
    # 5. 映射到3D
    volume, prob_volume = map_to_3d(predictions, probabilities, region_mask)
    
    # 6. 保存结果
    output_dir = save_results(predictions, probabilities, volume, prob_volume, args.output)
    
    # 7. 可视化
    visualize_3d(volume, output_dir)
    
    # 总结
    end_time = time.time()
    print(f"\n预测完成！总用时: {end_time - start_time:.2f} 秒")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    # 设置numpy精度显示
    np.set_printoptions(precision=6, suppress=True)
    main()