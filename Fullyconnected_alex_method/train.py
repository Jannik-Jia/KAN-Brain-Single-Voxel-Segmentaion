#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练函数
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, balanced_accuracy_score
import sys

def train_brain_voxel_mlp_multiclass(model, train_loader, val_loader, criterion, optimizer, device, 
                          num_epochs=100, val_epoch=1, save_path="./Results",
                          lr_scheduler=None, use_old_zipfile_serialization=True, experiment_name=None,
                          config=None, normalization_params=None):  # 新增normalization_params参数


    """
    训练脑体素MLP多分类模型，并输出训练集和验证集的性能指标
    
    参数:
        model: MLP模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        val_epoch: 验证频率
        save_path: 模型保存路径
        lr_scheduler: 学习率调度器
        use_old_zipfile_serialization: 是否使用旧的序列化方式
        experiment_name: 实验名称
        config: 配置字典，包含模型架构和超参数信息
    
    返回:
        训练结果统计信息
    """
    # 导入模型保存函数
    from utils.model_io import save_model_with_architecture
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 实验名称用于区分不同实验的保存文件
    if experiment_name is None:
        experiment_name = time.strftime("%Y%m%d_%H%M%S")
    
    # 初始化统计变量
    loss_list = []
    acc_list = []
    f1_macro_list = []  # 训练集F1宏平均记录
    
    val_acc_list = []
    val_epoch_list = []
    val_f1_macro_list = []
    val_kappa_list = []
    val_balanced_acc_list = []
    lr_list = []  # 学习率列表
    e = 0  # 初始化epoch计数器
    
    # 保存起始时间
    train_st = time.time()
    
    # 计算批次数量和样本数量
    batch_num = len(train_loader)
    train_num = len(train_loader.dataset)
    val_num = len(val_loader.dataset)
    
    # 创建日志文件
    log_file = os.path.join(save_path, f"{experiment_name}_training_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model architecture: {model.__class__.__name__}\n")
        f.write(f"Total epochs: {num_epochs}, Validation frequency: {val_epoch}\n")
        f.write(f"Train samples: {train_num}, Validation samples: {val_num}\n")
        f.write(f"Batch size: {train_loader.batch_size}\n\n")
        f.write("Epoch,Train_Loss,Train_Acc,Train_F1,Val_Acc,Val_F1,Val_Kappa,Val_BalAcc,LR\n")
    
    # 创建CSV日志
    csv_log_file = os.path.join(save_path, f"{experiment_name}_metrics.csv")
    with open(csv_log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Acc,Train_F1,Val_Acc,Val_F1,Val_Kappa,Val_BalAcc,LR\n")
    
    try:
        # 训练循环
        for e in range(num_epochs):
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)
            
            # 设置模型为训练模式
            model.train()
            avg_loss = 0.0
            train_acc = 0
            valid_count = 0
            
            # 收集训练集的预测结果
            train_all_preds = []
            train_all_targets = []
            
            # 批次循环 - 移除tqdm
            for batch_idx, (data, target) in enumerate(train_loader):
                # 将数据移动到指定设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 累计损失和准确率
                avg_loss += loss.item()
                _, pred = torch.max(out, dim=1)
                valid_mask = target != -1  # 忽略背景(-1)的准确率计算
                train_acc += (pred[valid_mask] == target[valid_mask]).sum().item()
                valid_count += valid_mask.sum().item()
                
                # 收集预测和目标用于计算F1等指标
                train_all_preds.extend(pred[valid_mask].cpu().numpy())
                train_all_targets.extend(target[valid_mask].cpu().numpy())
            
            # 计算本轮平均损失和准确率
            loss_list.append(avg_loss / batch_num)
            acc_list.append(train_acc / valid_count if valid_count > 0 else 0)
            
            # 计算训练集的F1分数
            train_all_preds = np.array(train_all_preds)
            train_all_targets = np.array(train_all_targets)
            train_f1_macro = f1_score(train_all_targets, train_all_preds, average='macro')
            f1_macro_list.append(train_f1_macro)
            
            epoch_msg = f"Epoch {e+1}/{num_epochs} Loss:{loss_list[-1]:.4f} Train Acc:{acc_list[-1]:.4f} Train F1:{train_f1_macro:.4f} LR:{current_lr:.6f}"
            print(epoch_msg)
            sys.stdout.flush()  # 确保立即输出到nohup.out
            
            # 验证阶段
            if (e+1) % val_epoch == 0 or (e+1) == num_epochs:
                val_acc = 0
                valid_count = 0
                model.eval()
                
                # 收集验证数据的预测结果
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    # 移除tqdm
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data, target = data.to(device), target.to(device)
                        out = model(data)
                        _, pred = torch.max(out, dim=1)
                        
                        # 收集有效预测（非背景）
                        valid_mask = target != -1
                        all_preds.extend(pred[valid_mask].cpu().numpy())
                        all_targets.extend(target[valid_mask].cpu().numpy())
                        val_acc += (pred[valid_mask] == target[valid_mask]).sum().item()
                        valid_count += valid_mask.sum().item()
                
                # 计算全面的评估指标
                all_preds = np.array(all_preds)
                all_targets = np.array(all_targets)
                val_accuracy = val_acc / valid_count if valid_count > 0 else 0
                val_f1_macro = f1_score(all_targets, all_preds, average='macro')
                val_kappa = cohen_kappa_score(all_targets, all_preds)
                val_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
                
                # 保存验证结果
                val_acc_list.append(val_accuracy)
                val_epoch_list.append(e)
                val_f1_macro_list.append(val_f1_macro)
                val_kappa_list.append(val_kappa)
                val_balanced_acc_list.append(val_balanced_acc)
                
                # 显示对比训练集和验证集的评估指标
                train_val_diff = train_f1_macro - val_f1_macro  # 训练集和验证集F1的差异（用于评估过拟合）
                
                val_msg = f"Epoch {e+1}/{num_epochs}\n  Train: Acc:{acc_list[-1]:.4f}  F1:{train_f1_macro:.4f}\n  Val:   Acc:{val_accuracy:.4f}  F1:{val_f1_macro:.4f}  Kappa:{val_kappa:.4f}  Balanced Acc:{val_balanced_acc:.4f}\n  Diff:  F1:{train_val_diff:.4f} (Training-Validation)"
                print(val_msg)
                sys.stdout.flush()  # 确保立即输出到nohup.out

                # 保存当前模型
                save_name = os.path.join(save_path, f"{experiment_name}_epoch_{e+1}_acc_{val_accuracy:.4f}_f1_{val_f1_macro:.4f}.pth")
                
                # 收集训练信息
                training_info = {
                    'epoch': e+1, 
                    'loss_list': loss_list, 
                    'acc_list': acc_list,
                    'f1_macro_list': f1_macro_list,
                    'val_acc_list': val_acc_list, 
                    'val_epoch_list': val_epoch_list,
                    'val_f1_macro_list': val_f1_macro_list,
                    'val_kappa_list': val_kappa_list,
                    'val_balanced_acc_list': val_balanced_acc_list,
                    'lr_list': lr_list,
                    'last_epoch': e+1,
                    'train_time': time.time() - train_st
                }
                
                # 使用新的保存函数保存模型及其完整架构

                try:
                    # 如果没有提供配置对象，创建一个基本配置
                    if config is None:
                        config = {
                            'model_type': model.__class__.__name__,
                            'feature_dim': model.layers[0].in_features if hasattr(model, 'layers') else 0,
                            'num_class': model.layers[-1].out_features if hasattr(model, 'layers') else 0,
                            'experiment_name': experiment_name
                        }

                    
                    # 保存模型
                    _, save_path_full = save_model_with_architecture(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        training_info=training_info,
                        normalization_params=normalization_params,
                        save_path=save_name,
                        lr_scheduler=lr_scheduler,
                        use_old_zipfile_serialization=use_old_zipfile_serialization,
                        scaler=config.get('scaler', None)  # 从配置中获取scaler
                    )
                    print(f"已保存模型及完整架构信息到: {save_path_full}")

                    
                except Exception as save_e:
                    print(f"保存模型时出错: {save_e}")
                    # 尝试使用旧的保存方法作为备选
                    try:
                        save_dict = {
                            'state_dict': model.state_dict(), 
                            'epoch': e+1, 
                            'optimizer': optimizer.state_dict(),
                            'training_info': training_info,
                            'created_with': f'PyTorch {torch.__version__}',
                            'save_format_version': 1.0,
                            'numpy_version': f'{np.__version__}'
                        }
                        torch.save(save_dict, save_name, _use_new_zipfile_serialization=not use_old_zipfile_serialization)
                        print(f"已使用备选方法保存模型到: {save_name}")
                    except Exception as backup_e:
                        print(f"备选保存方法也失败: {backup_e}")

                # 记录日志
                log_line = f"{e+1},{loss_list[-1]:.6f},{acc_list[-1]:.6f},{train_f1_macro:.6f},{val_accuracy:.6f},{val_f1_macro:.6f},{val_kappa:.6f},{val_balanced_acc:.6f},{current_lr:.8f}\n"
                with open(log_file, 'a') as f:
                    f.write(log_line)
                with open(csv_log_file, 'a') as f:
                    f.write(log_line)
                
                # 更新ReduceLROnPlateau类型的学习率调度器
                if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_f1_macro)  # 使用验证集F1宏平均指导学习率调度
            
            # 更新其他类型的学习率调度器
            if lr_scheduler is not None and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()
                
    except Exception as exc:
        print(exc)
        import traceback
        traceback.print_exc()
        
        # 记录错误
        with open(log_file, 'a') as f:
            f.write(f"\nTraining stopped due to error at epoch {e+1}: {str(exc)}\n")
            f.write(traceback.format_exc())
        
    finally:
        print(f'Training stopped at epoch {e+1}')
        
        # 记录训练结束信息
        with open(log_file, 'a') as f:
            f.write(f"\nTraining completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Last epoch: {e+1}")
    
    # 计算总训练时间
    train_time = time.time() - train_st
    print(f"Training time: {train_time:.2f} seconds")
    
    # 返回训练结果
    return {
        'loss_list': loss_list,
        'acc_list': acc_list,
        'f1_macro_list': f1_macro_list,
        'val_acc_list': val_acc_list,
        'val_epoch_list': val_epoch_list,
        'val_f1_macro_list': val_f1_macro_list,
        'val_kappa_list': val_kappa_list,
        'val_balanced_acc_list': val_balanced_acc_list,
        'train_time': train_time,
        'lr_list': lr_list,
        'last_epoch': e+1
    }