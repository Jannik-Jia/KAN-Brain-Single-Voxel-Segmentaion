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
    parser.add_argument('--save_every', type=int, default=1, help='每隔多少个epoch保存一次模型')
    parser.add_argument('--eval_every', type=int, default=1, help='每隔多少个epoch评估一次模型')
    
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        valid_mask = target != -1  # 忽略背景(-1)的准确率计算
        total += valid_mask.sum().item()
        correct += (predicted[valid_mask] == target[valid_mask]).sum().item()
        
        # 每100个batch打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # 只评估非背景像素
            valid_mask = target != -1
            all_preds.extend(predicted[valid_mask].cpu().numpy())
            all_targets.extend(target[valid_mask].cpu().numpy())
    
    # 计算平均损失
    val_loss = running_loss / len(val_loader)
    
    # 计算F1分数
    from sklearn.metrics import f1_score, accuracy_score
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    val_acc = accuracy_score(all_targets, all_preds)
    
    return val_loss, val_acc, val_f1, all_preds, all_targets

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
    
    # 创建CSV日志
    csv_log_file = os.path.join(experiment_dir, f"{args.experiment_name}_metrics.csv")
    with open(csv_log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_F1,LR\n")
    
    print(f"\n开始训练模型...")
    print(f"训练轮数: {args.epochs}, 批大小: {args.batch_size}")
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    # 保存训练过程中的指标
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    learning_rates = []
    
    best_val_f1 = 0.0
    best_model_path = None
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}, Learning Rate: {current_lr:.6f}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证模型
        if (epoch + 1) % args.eval_every == 0:
            val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            
            # 记录日志
            with open(csv_log_file, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_f1:.6f},{current_lr:.8f}\n")
            
            # 输出结果
            print(f"Epoch {epoch+1}/{args.epochs} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")
            
            # 保存当前模型
            if (epoch + 1) % args.save_every == 0:
                model_filename = f"{args.experiment_name}_epoch_{epoch+1}_acc_{val_acc:.4f}_f1_{val_f1:.4f}.pth"
                save_path = os.path.join(experiment_dir, model_filename)
                
                # 收集训练信息
                training_info = {
                    'epoch': epoch+1, 
                    'train_losses': train_losses, 
                    'train_accs': train_accs,
                    'val_losses': val_losses,
                    'val_accs': val_accs,
                    'val_f1s': val_f1s,
                    'learning_rates': learning_rates,
                    'train_time': time.time() - train_start_time
                }
                
                # 保存模型
                _, save_path_full = save_model_with_architecture(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    training_info=training_info,
                    save_path=save_path,
                    lr_scheduler=lr_scheduler,
                    use_old_zipfile_serialization=True,
                    scaler=dataset_dict['scaler']
                )
                print(f"  模型已保存至: {save_path_full}")
                
                # 更新最佳模型
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_path = save_path_full
                    print(f"  新的最佳模型! F1: {val_f1:.4f}")
                    
                    # 保存一个特殊的"best_model"副本
                    best_save_path = os.path.join(experiment_dir, f"{args.experiment_name}_best_model.pth")
                    torch.save(torch.load(save_path_full), best_save_path)
                    print(f"  最佳模型副本已保存至: {best_save_path}")
        
        # 更新学习率
        lr_scheduler.step()
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 完成, 耗时: {epoch_time:.2f}秒")
    
    # 训练结束时间
    train_total_time = time.time() - train_start_time
    print(f"\n训练完成! 总耗时: {train_total_time:.2f}秒")
    
    # 加载最佳模型
    if best_model_path is not None:
        print(f"加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
    
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
        f.write(f"训练信息:\n")
        f.write(f"总轮数: {args.epochs}\n")
        f.write(f"总训练时间: {train_total_time:.2f}秒\n")
        f.write(f"最佳验证F1: {best_val_f1:.6f}\n\n")
        
        f.write(f"测试集性能指标:\n")
        f.write(f"准确率: {test_results['accuracy']:.6f}\n")
        f.write(f"平衡准确率: {test_results['balanced_accuracy']:.6f}\n")
        f.write(f"宏平均F1: {test_results['f1_macro']:.6f}\n")
        f.write(f"加权F1: {test_results['f1_weighted']:.6f}\n")
        f.write(f"Kappa系数: {test_results['kappa']:.6f}\n")
    
    print(f"\n训练和评估完成!")
    print(f"最终测试集F1分数: {test_results['f1_macro']:.6f}")
    print(f"所有结果已保存至: {experiment_dir}")
    print(f"最佳模型保存在: {best_model_path}")

if __name__ == "__main__":
    main()