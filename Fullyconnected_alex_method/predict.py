#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用训练好的模型进行预测
"""

import os
import sys
import argparse
import torch
import numpy as np
import scipy.io

from data.mat_loader import load_external_mat_data
from utils.model_io import load_model_with_architecture

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BrainVoxel MLP Prediction')
    
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input_mat_path', type=str, required=True, help='输入.mat文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出预测结果的路径')
    parser.add_argument('--device', type=int, default=0, help='使用的设备（-1表示CPU）')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和scaler
    try:
        model, checkpoint, scaler = load_model_with_architecture(
            model_path=args.model_path,
            device=device,
            load_scaler=True
        )
        print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)
    
    # 如果scaler不存在，尝试从模型目录中找
    if scaler is None:
        scaler_path = os.path.join(os.path.dirname(args.model_path), "scaler.joblib")
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            print(f"从目录加载scaler: {scaler_path}")
    
    # 加载输入数据
    try:
        data_dict = load_external_mat_data(
            mat_file_path=args.input_mat_path,
            scaler=scaler
        )
        input_data = data_dict['data']
        print(f"成功加载输入数据: {args.input_mat_path}")
        print(f"数据形状: {input_data.shape}")
    except Exception as e:
        print(f"加载输入数据失败: {e}")
        sys.exit(1)
    
    # 转换为torch张量并预测
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 批量处理，避免内存不足
        batch_size = 1000
        num_batches = (len(input_data) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(input_data))
            batch_data = torch.FloatTensor(input_data[start_idx:end_idx]).to(device)
            
            outputs = model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())
    
    # 合并所有批次的预测结果
    all_predictions = np.vstack(predictions)
    
    # 保存预测结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    scipy.io.savemat(args.output_path, {'predicted_ann': all_predictions})
    print(f"预测结果已保存至: {args.output_path}")
    
    # 基本统计信息
    predicted_classes = np.argmax(all_predictions, axis=1)
    class_counts = np.bincount(predicted_classes)
    top_classes = np.argsort(class_counts)[::-1][:10]
    
    print("\n预测结果统计:")
    print(f"总样本数: {len(predicted_classes)}")
    print("前10个最常预测的类别:")
    for cls in top_classes:
        print(f"  类别 {cls}: {class_counts[cls]} 个样本 ({class_counts[cls]/len(predicted_classes)*100:.2f}%)")

if __name__ == "__main__":
    main()