#!/usr/bin/env python3
"""
手动指定模型路径进行预测并保存softmax
"""

import sys
import os
import numpy as np
import torch
import nibabel as nib
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入训练代码中的函数，并设置logger
try:
    import train_with_3d_prediction_save
    train_with_3d_prediction_save.logger = logger
    
    from train_with_3d_prediction_save import (
        load_and_process_subject_with_mask,
        predictions_to_3d_volume,
        create_label_mapping,
        STANDARD_LABELS
    )
except ImportError as e:
    print(f"❌ 无法导入训练代码: {e}")
    sys.exit(1)

def predict_and_save_softmax(model_path, output_dir=None, subject_path=None, output_name=None):
    """
    使用指定模型进行预测并保存softmax
    
    Parameters:
    -----------
    model_path : str
        模型文件路径 (.pth)
    output_dir : str, optional
        输出目录，默认为results文件夹
    subject_path : str, optional
        受试者数据路径，默认使用FOR_016_20250204_reproducibility
    output_name : str, optional
        输出文件名前缀，默认为current_prediction
    """
    
    print(f"🔍 手动预测softmax并保存")
    print(f"📦 模型路径: {model_path}")
    
    # 验证模型文件存在
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    # 设置输出目录
    if output_dir is None:
        base_dir = Path("/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample")
        output_dir = base_dir / "results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 设置受试者数据路径
    if subject_path is None:
        subject_path = "/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS/FOR_016_20250204_reproducibility"
    
    print(f"📂 数据路径: {subject_path}")
    
    try:
        # 加载模型
        print(f"\n📦 加载模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {device}")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✅ 模型加载成功")
        
        # 解析checkpoint
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            if 'config' in checkpoint and 'include_background' in checkpoint['config']:
                include_background = checkpoint['config']['include_background']
            else:
                include_background = 'bg_incl' in str(model_path)
            
            print(f"📊 模型信息:")
            if 'best_epoch' in checkpoint:
                print(f"  Best Epoch: {checkpoint['best_epoch']}")
            print(f"  Include background: {include_background}")
        else:
            print(f"❌ 无效的checkpoint格式")
            return False
        
        # 加载数据
        print(f"\n📂 加载测试数据...")
        test_data_info = load_and_process_subject_with_mask(
            subject_dir=subject_path,
            include_background=include_background
        )
        test_data_info['subject_id'] = Path(subject_path).name
        
        print(f"✅ 数据加载成功")
        print(f"  Features shape: {test_data_info['features'].shape}")
        print(f"  Labels shape: {test_data_info['labels'].shape}")
        
        # 重建模型
        print(f"\n🏗️ 重建模型架构...")
        
        # 从模型权重获取维度
        first_layer_weight = model_state['fc1.weight']
        last_layer_weight = model_state['fc5.weight']
        n_features = first_layer_weight.shape[1]
        n_classes = last_layer_weight.shape[0]
        
        print(f"  模型输入维度: {n_features}")
        print(f"  模型输出类别: {n_classes}")
        
        # 调整数据维度
        actual_features = test_data_info['features'].shape[1]
        if n_features != actual_features:
            print(f"⚠️ 调整特征维度: {actual_features} -> {n_features}")
            if actual_features > n_features:
                test_data_info['features'] = test_data_info['features'][:, :n_features]
            else:
                padding = np.zeros((test_data_info['features'].shape[0], n_features - actual_features))
                test_data_info['features'] = np.concatenate([test_data_info['features'], padding], axis=1)
        
        # 重建模型
        from torch import nn
        import torch.nn.functional as F
        
        class RegModel(nn.Module):
            def __init__(self, input_dim=42, num_classes=52, hidden_dim=4096, 
                         num_hidden_layers=4, dropout_rate=0.5):
                super(RegModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, hidden_dim)
                self.fc5 = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.dropout(F.relu(self.fc2(x)))
                x = self.dropout(F.relu(self.fc3(x)))
                x = self.dropout(F.relu(self.fc4(x)))
                x = self.fc5(x)
                return x
        
        model = RegModel(input_dim=n_features, num_classes=n_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        print(f"✅ 模型架构重建成功")
        
        # 进行预测
        print(f"\n🔮 开始预测...")
        
        test_features = test_data_info['features']
        batch_size = 8192
        all_predictions = []
        n_samples = len(test_features)
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                if start_idx % (batch_size * 10) == 0:  # 每10个batch显示进度
                    print(f"  进度: {start_idx:,}/{n_samples:,} ({start_idx/n_samples*100:.1f}%)")
                
                end_idx = min(start_idx + batch_size, n_samples)
                batch_features = torch.FloatTensor(test_features[start_idx:end_idx]).to(device)
                batch_logits = model(batch_features)
                batch_pred = torch.softmax(batch_logits, dim=1)
                all_predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        
        print(f"✅ 预测完成")
        print(f"  预测形状: {predictions.shape}")
        print(f"  预测范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        # 检查预测分布
        pred_classes = np.argmax(predictions, axis=1)
        non_bg_preds = pred_classes[pred_classes > 0]
        print(f"  非背景预测: {len(non_bg_preds):,}/{len(pred_classes):,} ({len(non_bg_preds)/len(pred_classes)*100:.2f}%)")
        
        # 转换为3D volume
        print(f"\n🔄 转换为3D volume...")
        
        volume_3d = predictions_to_3d_volume(
            predictions, test_data_info, 
            include_background=include_background
        )
        
        print(f"✅ 3D volume转换完成")
        print(f"  Volume形状: {volume_3d.shape}")
        print(f"  Volume范围: [{volume_3d.min():.6f}, {volume_3d.max():.6f}]")
        
        # 保存文件
        print(f"\n💾 保存softmax文件...")
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bg_str = 'incl' if include_background else 'excl'
        subject_id = test_data_info['subject_id']
        
        if output_name is None:
            output_name = "current_prediction"
        
        # 准备NIfTI保存
        affine = test_data_info['affine']
        header = test_data_info['header'].copy()
        header.set_data_shape(volume_3d.shape)
        header.set_data_dtype(np.float32)
        
        softmax_img = nib.Nifti1Image(volume_3d, affine, header)
        
        # 保存softmax文件
        softmax_path = output_dir / f"{output_name}_softmax_3d_{subject_id}_bg_{bg_str}_{timestamp}.nii.gz"
        nib.save(softmax_img, softmax_path)
        
        print(f"✅ Softmax文件已保存: {softmax_path}")
        print(f"  文件大小: {softmax_path.stat().st_size / (1024**2):.2f} MB")
        
        # 保存info文件
        info_dict = {
            'subject_id': subject_id,
            'model_path': str(model_path),
            'original_shape_3d': test_data_info['original_shape_3d'],
            'softmax_shape': volume_3d.shape,
            'include_background': include_background,
            'n_valid_voxels': len(test_features),
            'n_classes': predictions.shape[1],
            'timestamp': timestamp,
            'prediction_stats': {
                'min_prob': float(predictions.min()),
                'max_prob': float(predictions.max()),
                'mean_prob': float(predictions.mean()),
                'non_bg_predictions': int(len(non_bg_preds)),
                'non_bg_percentage': float(len(non_bg_preds)/len(pred_classes)*100)
            }
        }
        
        info_path = output_dir / f"{output_name}_softmax_info_{subject_id}_bg_{bg_str}_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        
        print(f"✅ Info文件已保存: {info_path}")
        
        print(f"\n🎉 预测完成！")
        print(f"📁 输出文件:")
        print(f"  - {softmax_path.name}")
        print(f"  - {info_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='手动指定模型路径进行softmax预测')
    
    parser.add_argument('model_path', 
                       help='模型文件路径 (.pth)')
    parser.add_argument('-o', '--output', 
                       help='输出目录 (默认: results文件夹)')
    parser.add_argument('-s', '--subject', 
                       help='受试者数据路径 (默认: FOR_016_20250204_reproducibility)')
    parser.add_argument('-n', '--name', 
                       help='输出文件名前缀 (默认: current_prediction)')
    
    args = parser.parse_args()
    
    # 运行预测
    success = predict_and_save_softmax(
        model_path=args.model_path,
        output_dir=args.output,
        subject_path=args.subject,
        output_name=args.name
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()