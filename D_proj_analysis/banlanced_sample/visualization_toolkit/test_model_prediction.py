#!/usr/bin/env python3
"""
加载训练好的模型进行预测，检测softmax保存前后的数据正确性
"""

import sys
import os
import numpy as np
import torch
import nibabel as nib
import json
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
    # 设置训练代码中需要的logger
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

def test_model_prediction_and_save():
    """测试模型预测和保存过程"""
    
    print("🔍 测试模型预测和softmax保存过程")
    
    # 服务器路径设置
    base_dir = Path("/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample")
    results_dir = base_dir / "results"
    
    # 寻找训练好的模型
    model_patterns = [
        "*bg_incl*.pth",
        "*bg_incl*.pt", 
        "*.pth",
        "*.pt"
    ]
    
    model_path = None
    for pattern in model_patterns:
        models = list(results_dir.glob(pattern))
        if models:
            # 选择最新的模型
            model_path = max(models, key=lambda x: x.stat().st_mtime)
            break
    
    if not model_path:
        print(f"❌ 未找到训练好的模型文件在 {results_dir}")
        print("📁 检查结果目录内容:")
        for f in results_dir.iterdir():
            print(f"  {f.name}")
        return False
    
    print(f"📦 找到模型: {model_path}")
    
    # 加载模型
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {device}")
        
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✅ 模型加载成功")
        
        # 检查checkpoint内容
        print("📊 Checkpoint keys:", list(checkpoint.keys()))
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                print("📊 Checkpoint信息:")
                if 'best_epoch' in checkpoint:
                    print(f"  Best Epoch: {checkpoint['best_epoch']}")
                if 'config' in checkpoint and 'include_background' in checkpoint['config']:
                    include_background = checkpoint['config']['include_background']
                    print(f"  Include background: {include_background}")
                else:
                    # 从文件名推断
                    include_background = 'bg_incl' in str(model_path)
                    print(f"  Include background (推断): {include_background}")
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
                include_background = 'bg_incl' in str(model_path)
            else:
                model_state = checkpoint
                include_background = 'bg_incl' in str(model_path)
        else:
            model_state = checkpoint
            include_background = 'bg_incl' in str(model_path)
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 准备测试数据
    try:
        print("\n📂 加载测试数据...")
        
        # 服务器数据路径
        subject_path = "/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS/FOR_016_20250204_reproducibility"
        
        # 检查数据文件是否存在
        balanced_dir = Path(subject_path) / "balanced_output"
        data_4d_path = balanced_dir / "balanced_data_4d10000.nii.gz"
        label_3d_path = balanced_dir / "balanced_labels_3d10000.nii.gz"
        
        if not data_4d_path.exists():
            print(f"❌ 数据文件不存在: {data_4d_path}")
            return False
            
        if not label_3d_path.exists():
            print(f"❌ 标签文件不存在: {label_3d_path}")
            return False
        
        print(f"使用服务器数据: {balanced_dir}")
        
        # 加载并处理数据（使用subject_dir参数）
        test_data_info = load_and_process_subject_with_mask(
            subject_dir=subject_path,
            include_background=include_background
        )
        
        # 添加subject_id字段
        test_data_info['subject_id'] = "FOR_016_20250204_reproducibility"
        
        print(f"✅ 数据加载成功")
        print(f"  Features shape: {test_data_info['features'].shape}")
        print(f"  Labels shape: {test_data_info['labels'].shape}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 重建模型架构（这里需要根据你的实际模型结构调整）
    try:
        print("\n🏗️ 重建模型架构...")
        
        # 从checkpoint的config中获取正确的维度信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            n_features = config.get('input_dim', test_data_info['features'].shape[1])
            n_classes = config.get('num_classes', len(np.unique(test_data_info['labels'])))
            print(f"  从config获取维度:")
            print(f"    输入特征维度: {n_features}")
            print(f"    输出类别数: {n_classes}")
        else:
            # 从模型状态推断维度
            first_layer_weight = model_state['fc1.weight']
            n_features = first_layer_weight.shape[1]
            n_classes = model_state['fc5.weight'].shape[0]
            print(f"  从模型状态推断维度:")
            print(f"    输入特征维度: {n_features}")
            print(f"    输出类别数: {n_classes}")
        
        # 检查数据维度是否匹配
        actual_features = test_data_info['features'].shape[1]
        if n_features != actual_features:
            print(f"⚠️ 特征维度不匹配: 模型期望{n_features}，数据实际{actual_features}")
            # 调整数据以匹配模型
            if actual_features > n_features:
                print(f"  截取前{n_features}个特征")
                test_data_info['features'] = test_data_info['features'][:, :n_features]
            else:
                print(f"  用零填充到{n_features}个特征")
                padding = np.zeros((test_data_info['features'].shape[0], n_features - actual_features))
                test_data_info['features'] = np.concatenate([test_data_info['features'], padding], axis=1)
        
        print(f"  最终使用维度: 特征{test_data_info['features'].shape[1]}, 类别{n_classes}")
        
        # 使用训练代码中的RegModel架构
        from torch import nn
        import torch.nn.functional as F
        
        class RegModel(nn.Module):
            """Deep fully connected network for MRI voxel classification (Alex identical structure)"""
            
            def __init__(self, input_dim=42, num_classes=52, hidden_dim=4096, 
                         num_hidden_layers=4, dropout_rate=0.5):
                super(RegModel, self).__init__()
                
                # 完全对应Alex的结构：无BatchNorm，使用单独的层定义
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, hidden_dim)
                self.fc5 = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                # 严格按照Alex的前向传播结构
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.dropout(F.relu(self.fc2(x)))
                x = self.dropout(F.relu(self.fc3(x)))
                x = self.dropout(F.relu(self.fc4(x)))
                x = self.fc5(x)  # 输出层无激活函数
                return x
        
        model = RegModel(input_dim=n_features, num_classes=n_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        print("✅ 模型架构重建成功")
        
    except Exception as e:
        print(f"❌ 模型架构重建失败: {e}")
        print("💡 请根据你的实际模型架构修改这部分代码")
        return False
    
    # 进行预测
    try:
        print("\n🔮 开始预测...")
        
        test_features = test_data_info['features']
        batch_size = 8192
        
        all_predictions = []
        n_samples = len(test_features)
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_features = torch.FloatTensor(test_features[start_idx:end_idx]).to(device)
                batch_logits = model(batch_features)
                # 将logits转换为softmax概率
                batch_pred = torch.softmax(batch_logits, dim=1)
                all_predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        
        print(f"✅ 预测完成")
        print(f"  预测形状: {predictions.shape}")
        print(f"  预测范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"  预测均值: {predictions.mean():.6f}")
        
        # 检查预测的有效性
        pred_sums = np.sum(predictions, axis=1)
        print(f"  概率和范围: [{pred_sums.min():.6f}, {pred_sums.max():.6f}]")
        
        # 检查argmax分布
        pred_classes = np.argmax(predictions, axis=1)
        unique_classes, counts = np.unique(pred_classes, return_counts=True)
        print(f"\n📊 预测类别分布:")
        for cls, count in zip(unique_classes, counts):
            percentage = count / len(pred_classes) * 100
            print(f"  类别 {cls}: {count:,} ({percentage:.2f}%)")
        
        # 检查非背景预测
        non_bg_preds = pred_classes[pred_classes > 0]
        print(f"  非背景预测: {len(non_bg_preds):,}/{len(pred_classes):,} ({len(non_bg_preds)/len(pred_classes)*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 转换为3D volume（保存前检测）
    try:
        print(f"\n🔄 转换为3D volume...")
        
        volume_3d = predictions_to_3d_volume(
            predictions, test_data_info, 
            include_background=include_background
        )
        
        print(f"✅ 3D volume转换完成")
        print(f"  Volume形状: {volume_3d.shape}")
        print(f"  Volume范围: [{volume_3d.min():.6f}, {volume_3d.max():.6f}]")
        print(f"  Volume均值: {volume_3d.mean():.6f}")
        
        # 检查非零值数量
        non_zero_count = np.count_nonzero(volume_3d)
        total_count = np.prod(volume_3d.shape)
        print(f"  非零值: {non_zero_count:,}/{total_count:,} ({non_zero_count/total_count*100:.6f}%)")
        
        # 检查概率和
        prob_sums = np.sum(volume_3d, axis=-1)
        print(f"  概率和范围: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
        
        # 检查中间切片的预测
        mid_slice = volume_3d.shape[2] // 2
        slice_data = volume_3d[:, :, mid_slice, :]
        slice_predictions = np.argmax(slice_data, axis=-1)
        unique_slice_preds, slice_counts = np.unique(slice_predictions, return_counts=True)
        
        print(f"\n🎲 中间切片 #{mid_slice} 预测分布:")
        for pred, count in zip(unique_slice_preds, slice_counts):
            percentage = count / np.prod(slice_predictions.shape) * 100
            print(f"  类别 {pred}: {count:,} ({percentage:.2f}%)")
        
        # 检查非背景预测
        slice_non_bg = slice_predictions[slice_predictions > 0]
        print(f"  中间切片非背景预测: {len(slice_non_bg):,}")
        
    except Exception as e:
        print(f"❌ 3D volume转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存softmax文件
    try:
        print(f"\n💾 保存softmax文件...")
        
        # 创建输出目录（保存在results文件夹中）
        output_dir = results_dir
        output_dir.mkdir(exist_ok=True)
        
        # 准备保存
        bg_str = 'incl' if include_background else 'excl'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        subject_id = test_data_info['subject_id']
        
        # 使用原始的仿射矩阵和头信息
        affine = test_data_info['affine']
        header = test_data_info['header'].copy()
        
        # 更新头信息
        header.set_data_shape(volume_3d.shape)
        header.set_data_dtype(np.float32)
        
        # 创建NIfTI图像
        softmax_img = nib.Nifti1Image(volume_3d, affine, header)
        
        # 保存文件（添加verification前缀标识这是验证测试）
        softmax_path = output_dir / f"verification_test_softmax_3d_{subject_id}_bg_{bg_str}_{timestamp}.nii.gz"
        nib.save(softmax_img, softmax_path)
        
        print(f"✅ 文件已保存: {softmax_path}")
        print(f"  文件大小: {softmax_path.stat().st_size / (1024**2):.2f} MB")
        
        # 保存info信息
        info_dict = {
            'subject_id': subject_id,
            'original_shape_3d': test_data_info['original_shape_3d'],
            'softmax_shape': volume_3d.shape,
            'include_background': include_background,
            'n_valid_voxels': len(test_features),
            'n_classes': predictions.shape[1],
            'timestamp': timestamp,
            'model_path': str(model_path),
            'prediction_stats': {
                'min_prob': float(predictions.min()),
                'max_prob': float(predictions.max()),
                'mean_prob': float(predictions.mean())
            }
        }
        
        info_path = output_dir / f"verification_test_softmax_info_{subject_id}_bg_{bg_str}_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        
        print(f"✅ Info文件已保存: {info_path}")
        
    except Exception as e:
        print(f"❌ 文件保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 重新加载保存的文件进行验证
    try:
        print(f"\n🔍 验证保存的文件...")
        
        # 重新加载
        loaded_nii = nib.load(softmax_path)
        loaded_data = loaded_nii.get_fdata()
        
        print(f"✅ 文件重新加载成功")
        print(f"  加载形状: {loaded_data.shape}")
        print(f"  加载范围: [{loaded_data.min():.6f}, {loaded_data.max():.6f}]")
        print(f"  加载均值: {loaded_data.mean():.6f}")
        
        # 检查非零值
        loaded_non_zero = np.count_nonzero(loaded_data)
        loaded_total = np.prod(loaded_data.shape)
        print(f"  非零值: {loaded_non_zero:,}/{loaded_total:,} ({loaded_non_zero/loaded_total*100:.6f}%)")
        
        # 检查中间切片
        loaded_slice = loaded_data[:, :, mid_slice, :]
        loaded_slice_preds = np.argmax(loaded_slice, axis=-1)
        loaded_unique, loaded_counts = np.unique(loaded_slice_preds, return_counts=True)
        
        print(f"\n🎲 加载后中间切片预测分布:")
        for pred, count in zip(loaded_unique, loaded_counts):
            percentage = count / np.prod(loaded_slice_preds.shape) * 100
            print(f"  类别 {pred}: {count:,} ({percentage:.2f}%)")
        
        loaded_slice_non_bg = loaded_slice_preds[loaded_slice_preds > 0]
        print(f"  加载后中间切片非背景预测: {len(loaded_slice_non_bg):,}")
        
        # 比较保存前后的数据
        print(f"\n📊 保存前后数据比较:")
        data_diff = np.abs(volume_3d - loaded_data).max()
        print(f"  最大差异: {data_diff:.2e}")
        
        if data_diff < 1e-6:
            print("✅ 数据保存和加载完全一致！")
        else:
            print("⚠️ 数据保存和加载存在差异")
            
        return True
        
    except Exception as e:
        print(f"❌ 文件验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_prediction_and_save()
    if success:
        print("\n🎉 测试完成！")
    else:
        print("\n❌ 测试失败")