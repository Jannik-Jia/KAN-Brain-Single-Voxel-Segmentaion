#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
贝叶斯优化相关函数
"""

import os
import json
import torch
import optuna
import numpy as np
from sklearn.metrics import f1_score
from models import get_model
import time
import datetime
import shutil
def create_mlp_model(trial, input_dim, num_classes, param_space=None):
    """
    使用Optuna trial创建MLP模型
    
    参数:
        trial: Optuna trial对象
        input_dim: 输入特征维度
        num_classes: 类别数量
        param_space: 参数空间字典
    
    返回:
        model: MLP模型
    """
    if param_space is None:
        # 默认参数空间
        param_space = {
            'dropout_rate': (0.3, 0.7),
            'activation': ['relu', 'gelu', 'swish'],
            'layer_sizes': [
                [4096, 4096, 4096, 4096],  # 标准4x4096网络
                [3072, 3072, 3072, 3072],  # 更小的网络
                [2048, 4096, 4096, 2048]   # 钟形网络
            ]
        }
    
    # 从参数空间采样
    dropout_rate = trial.suggest_float('dropout_rate', *param_space['dropout_rate'])
    activation = trial.suggest_categorical('activation', param_space['activation'])
    layer_sizes_idx = trial.suggest_int('layer_sizes_idx', 0, len(param_space['layer_sizes'])-1)
    hidden_dims = param_space['layer_sizes'][layer_sizes_idx]
    
    # 创建模型
    model_type = trial.suggest_categorical('model_type', ['base_mlp', 'deep_mlp', 'residual_mlp'])
    model = get_model(
        model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        activation=activation
    )
    
    return model



def objective(trial, data_loaders, input_dim, num_classes, device, param_space=None, config=None):
    """
    Optuna优化目标函数，具有增强的早停和检查点保存功能
    
    参数:
        trial: Optuna trial对象
        data_loaders: 包含训练和验证数据加载器的字典
        input_dim: 输入特征维度
        num_classes: 类别数量
        device: 计算设备
        param_space: 参数空间字典
        config: 配置字典，包含早停参数等
    
    返回:
        best_val_f1: 最佳验证F1分数
    """
    # 配置信息
    epochs = config.get('epochs', 30) if config else 30
    # 使用贝叶斯优化专用的最大轮数
    train_epochs = config.get('bo_max_epochs', 100) if config else 100
    # 早停参数
    patience = config.get('early_stop_patience', 10) if config else 10
    min_delta = config.get('early_stop_min_delta', 0.001) if config else 0.001
    save_checkpoints = config.get('save_trial_checkpoints', True) if config else True
    save_path = config.get('save_dir', './results') if config else './results'
    
    # ===== 共享参数空间（所有架构通用）=====
    # 这些参数对所有架构保持一致
    shared_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),  # 统一的dropout范围
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
        'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['cosine', 'step', 'plateau', 'none']),
    }
    
    # 首先选择模型类型
    model_type = trial.suggest_categorical('model_type', ['base_mlp', 'deep_mlp', 'residual_mlp'])
    
    # ===== 架构特定参数空间 =====
    if model_type == 'base_mlp':
        # 基础MLP特有参数
        layer_sizes_options = [
            [4096, 4096, 4096, 4096],  # 标准4x4096网络
            [3072, 3072, 3072, 3072],  # 更小的网络
            [2048, 2048, 2048, 2048],  # 更小的网络
        ]
        layer_sizes_idx = trial.suggest_int('layer_sizes_idx', 0, len(layer_sizes_options)-1)
        hidden_dims = layer_sizes_options[layer_sizes_idx]
        model_params = {}  # 基础MLP没有额外参数
        
    elif model_type == 'deep_mlp':
        # 深层MLP特有参数
        depth = trial.suggest_int('depth', 5, 8)  # 更深的网络
        width_factor = trial.suggest_categorical('width_factor', [1024, 2048, 3072])
        hidden_dims = [width_factor] * depth
        model_params = {
            'use_skip_connections': trial.suggest_categorical('use_skip_connections', [True, False])
        }
        
    elif model_type == 'residual_mlp':
        # 残差MLP特有参数
        layer_sizes_options = [
            [4096, 4096, 4096, 4096],  # 标准尺寸
            [2048, 2048, 2048, 2048],  # 较小尺寸
            [1024, 2048, 2048, 1024],  # 钟形结构
            [4096, 2048, 2048, 4096]   # 沙漏形结构
        ]
        layer_sizes_idx = trial.suggest_int('layer_sizes_idx', 0, len(layer_sizes_options)-1)
        hidden_dims = layer_sizes_options[layer_sizes_idx]
        model_params = {
            'use_bottleneck': trial.suggest_categorical('use_bottleneck', [True, False]),
        }
        # 只有当use_bottleneck为True时才添加bottleneck_factor参数
        if model_params['use_bottleneck']:
            model_params['bottleneck_factor'] = trial.suggest_float('bottleneck_factor', 0.25, 0.5)
        else:
            model_params['bottleneck_factor'] = 0.5  # 默认值
    
    # 创建模型
    model_kwargs = {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'num_classes': num_classes,
        'dropout_rate': shared_params['dropout_rate'],
        'activation': shared_params['activation'],
        **model_params  # 添加模型特定参数
    }
    
    model = get_model(model_type, **model_kwargs)
    model = model.to(device)
    
    # 创建优化器
    if shared_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=shared_params['learning_rate'], 
            weight_decay=shared_params['weight_decay']
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=shared_params['learning_rate'], 
            weight_decay=shared_params['weight_decay']
        )
    
    # 学习率调度器
    lr_scheduler = None
    if shared_params['lr_scheduler'] == 'cosine':
        # 修复t_max参数：确保low <= high
        t_max = min(train_epochs, max(2, train_epochs // 2))  # 确保t_max在合理范围内
        eta_min = trial.suggest_float('cosine_eta_min', 1e-7, 1e-5, log=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif shared_params['lr_scheduler'] == 'step':
        # 确保step_size <= train_epochs
        step_size = trial.suggest_int('step_size', 1, max(1, train_epochs // 2))
        gamma = trial.suggest_float('step_gamma', 0.1, 0.5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif shared_params['lr_scheduler'] == 'plateau':
        # 确保patience不超过训练轮数
        plateau_patience = trial.suggest_int('plateau_patience', 1, max(1, train_epochs // 3))
        factor = trial.suggest_float('plateau_factor', 0.1, 0.5)
        threshold = trial.suggest_float('plateau_threshold', 1e-4, 1e-2, log=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=plateau_patience, 
            threshold=threshold, verbose=True
        )
    
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # 确保标准化参数被保存在trial的用户属性中
    if config and 'normalization_params' in config and config['normalization_params'] is not None:
        trial.set_user_attr('normalization_params', config['normalization_params'])
    
    if config and 'scaler_path' in config and config['scaler_path'] is not None:
        trial.set_user_attr('scaler_path', config['scaler_path'])

    # 训练模型 - 增强版本，支持Early Stopping
    val_f1_values = []
    train_losses = []
    lr_history = []
    best_val_f1 = 0.0
    no_improve_epochs = 0
    best_model_state = None
    restart_from_best = config.get('restart_from_best', True) if config else True
    
    # 训练循环
    for epoch in range(train_epochs):
        # 训练阶段
        model.train()
        batch_losses = []
        for batch_idx, (data, target) in enumerate(data_loaders['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        # 计算平均训练损失
        avg_train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        train_losses.append(avg_train_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 验证阶段
        model.eval()
        all_preds = []
        all_targets = []
        val_losses = []
        
        with torch.no_grad():
            for data, target in data_loaders['val']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # 计算验证损失
                val_loss = criterion(output, target).item()
                val_losses.append(val_loss)
                
                _, preds = torch.max(output, 1)
                
                # 只评估非背景像素
                valid_mask = target != -1
                all_preds.extend(preds[valid_mask].cpu().numpy())
                all_targets.extend(target[valid_mask].cpu().numpy())
        
        # 计算验证指标
        val_f1_macro = f1_score(all_targets, all_preds, average='macro')
        val_f1_values.append(val_f1_macro)
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        
        # 更新学习率调度器
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_f1_macro)
            else:
                lr_scheduler.step()
        
        # 输出当前轮次信息
        print(f"Trial {trial.number} | Epoch {epoch+1}/{train_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1_macro:.4f} | LR: {current_lr:.6f}")
        
        # 报告进度给Optuna
        trial.report(val_f1_macro, epoch)
        
        # Early Stopping 检查
        if val_f1_macro > best_val_f1 + min_delta:
            # 性能有改进
            best_val_f1 = val_f1_macro
            no_improve_epochs = 0
            
            # 保存最佳模型状态
            if save_checkpoints:
                best_model_state = {
                    'trial_number': trial.number,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_f1_macro': val_f1_macro,
                    'params': trial.params,
                    'train_losses': train_losses,
                    'val_f1_values': val_f1_values,
                    'lr_history': lr_history,
                    'best_epoch': epoch,
                    'model_config': {
                        'model_type': model_type,
                        'input_dim': input_dim,
                        'hidden_dims': hidden_dims,
                        'num_classes': num_classes,
                        'dropout_rate': shared_params['dropout_rate'],
                        'activation': shared_params['activation'],
                        **model_params
                    },
                    # ===== 添加标准化信息 =====
                    'normalization_params': config.get('normalization_params') if config else None,
                    'scaler_path': config.get('scaler_path') if config else None
                }
                
                # 保存检查点
                checkpoint_dir = os.path.join(save_path, "trial_checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"trial_{trial.number}_best.pth")
                torch.save(best_model_state, checkpoint_path)
                print(f"Trial {trial.number}: 保存epoch {epoch}的最佳模型，F1={val_f1_macro:.4f}")
        else:
            # 性能没有改进
            no_improve_epochs += 1
            
            # 如果启用回到最佳权重的选项，并且当前是回到最佳权重的合适时机
            if restart_from_best and no_improve_epochs >= patience // 2 and best_model_state is not None:
                print(f"Trial {trial.number}: 回到最佳模型权重（Epoch {best_model_state['epoch']}）")
                model.load_state_dict(best_model_state['state_dict'])
                # 可选：调整学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                # 重置学习率调度器
                if lr_scheduler is not None and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler = type(lr_scheduler)(optimizer, **{k: v for k, v in lr_scheduler.__dict__.items() 
                                                               if not k.startswith('_') and k != 'optimizer'})
            
            # 检查是否应该提前停止
            if no_improve_epochs >= patience:
                print(f"Trial {trial.number}: Early stopping at epoch {epoch+1}/{train_epochs}, best F1={best_val_f1:.4f}")
                break
        
        # Optuna提前停止
        if trial.should_prune():
            print(f"Trial {trial.number}: Pruned by Optuna at epoch {epoch+1}")
            raise optuna.TrialPruned()
    
    # 计算多个评估指标以全面了解模型性能
    if len(all_targets) > 0:
        from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, balanced_accuracy_score
        
        # 多种指标计算
        metrics = {
            'f1_macro': val_f1_macro,
            'precision_macro': precision_score(all_targets, all_preds, average='macro'),
            'recall_macro': recall_score(all_targets, all_preds, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
            'kappa': cohen_kappa_score(all_targets, all_preds)
        }
        
        # 将所有指标都报告给trial，便于后续分析
        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(metric_name, float(metric_value))
        
        # 记录额外的训练信息
        trial.set_user_attr('train_epochs_completed', epoch + 1)
        trial.set_user_attr('early_stopped', no_improve_epochs >= patience)
        trial.set_user_attr('best_epoch', best_model_state['epoch'] if best_model_state else epoch)
    
    # 保存最终的指标曲线图，便于后续分析
    if save_checkpoints and len(val_f1_values) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
            import matplotlib.pyplot as plt
            
            # 创建指标曲线图目录
            curves_dir = os.path.join(save_path, "trial_curves")
            os.makedirs(curves_dir, exist_ok=True)
            
            # 绘制F1和训练损失曲线
            plt.figure(figsize=(12, 5))
            
            # 训练损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.title(f'Trial {trial.number} Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # F1曲线
            plt.subplot(1, 2, 2)
            plt.plot(val_f1_values)
            plt.axhline(y=best_val_f1, color='r', linestyle='--', label=f'Best: {best_val_f1:.4f}')
            plt.title(f'Trial {trial.number} Validation F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, f"trial_{trial.number}_curves.png"))
            plt.close()
        except Exception as e:
            print(f"绘制指标曲线时出错: {e}")
    
    # 确保最佳模型被保存，即使在最后一轮达到最佳性能
    if save_checkpoints and best_model_state is None and len(val_f1_values) > 0:
        best_val_f1 = max(val_f1_values)
        best_epoch = val_f1_values.index(best_val_f1)
        
        best_model_state = {
            'trial_number': trial.number,
            'epoch': best_epoch,
            'state_dict': model.state_dict(),  # 注意：这将保存最后一轮的权重，而不是最佳轮次的权重
            'optimizer': optimizer.state_dict(),
            'val_f1_macro': best_val_f1,
            'params': trial.params,
            'train_losses': train_losses,
            'val_f1_values': val_f1_values,
            'lr_history': lr_history,
            'best_epoch': best_epoch,
            'model_config': {
                'model_type': model_type,
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'num_classes': num_classes,
                'dropout_rate': shared_params['dropout_rate'],
                'activation': shared_params['activation'],
                **model_params
            },
            # ===== 添加标准化信息 =====
            'normalization_params': config.get('normalization_params') if config else None,
            'scaler_path': config.get('scaler_path') if config else None
        }
        
        # 保存检查点
        checkpoint_dir = os.path.join(save_path, "trial_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"trial_{trial.number}_best.pth")
        torch.save(best_model_state, checkpoint_path)
        print(f"Trial {trial.number}: 保存最终模型，F1={best_val_f1:.4f}（最佳在epoch {best_epoch}）")
    
    # 返回最佳F1分数
    return best_val_f1


def run_bayesian_optimization(data_loaders, input_dim, num_classes, device, param_space=None, 
                             n_trials=30, study_name="mlp_optimization", save_path="./results", config=None):
    """
    运行贝叶斯优化，查找最优超参数并保存最佳模型
    
    参数:
        data_loaders: 包含训练和验证数据加载器的字典
        input_dim: 输入特征维度
        num_classes: 类别数量
        device: 计算设备
        param_space: 参数空间字典
        n_trials: 优化试验次数
        study_name: 研究名称
        save_path: 结果保存路径
        config: 配置字典
    
    返回:
        study: Optuna study对象
        best_params: 最佳参数
        best_model_dir: 最佳模型目录
    """
    import os
    import json
    import torch
    import optuna
    import numpy as np
    import time
    import datetime
    import shutil
    from models import get_model
    
    # 选择pruner类型
    pruner_type = config.get('pruner_type', 'median') if config else 'median'
    
    if pruner_type == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=5,  # 最小轮数
            max_resource=config.get('bo_max_epochs', 100) if config else 100,  # 最大轮数
            reduction_factor=3  # 资源减少因子
        )
    else:
        # 默认使用MedianPruner
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    
    # 优化TPESampler配置
    sampler = optuna.samplers.TPESampler(
        seed=np.random.randint(1, 10000),
        n_startup_trials=max(10, n_trials // 5),  # 更多的随机试验
        consider_prior=True,                      # 考虑先验
        prior_weight=1.0,                         # 先验权重
        consider_magic_clip=True,                 # 使用魔术剪裁
        consider_endpoints=True,                  # 考虑端点
        n_ei_candidates=100,                      # 更多候选点提高探索性
        multivariate=True                         # 使用多变量采样
    )
    
    
    # 创建研究
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name
    )
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(save_path, "trial_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 优化开始时间
    start_time = time.time()
    
    # 运行优化
    try:
        study.optimize(
            lambda trial: objective(trial, data_loaders, input_dim, num_classes, device, param_space, config),
            n_trials=n_trials,
            callbacks=[lambda study, trial: print_architecture_distribution(study)]
        )
    except KeyboardInterrupt:
        print("用户中断了优化过程。将保存已完成的试验结果。")
    
    # 优化结束时间和总时间
    end_time = time.time()
    total_duration = end_time - start_time
    duration_str = str(datetime.timedelta(seconds=int(total_duration)))
    
    # 获取所有完成的试验
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    # 打印优化结果
    print("\n" + "="*70)
    print(f"贝叶斯优化完成.")
    print(f"总试验次数: {len(study.trials)}")
    print(f"完成的试验: {len(completed_trials)}")
    print(f"提前终止的试验: {len(pruned_trials)}")
    print(f"总耗时: {duration_str}")
    print(f"最佳F1分数: {study.best_value:.4f}")
    
    print("\n最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存优化结果
    result_path = os.path.join(save_path, f"{study_name}_results.json")
    with open(result_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len(pruned_trials),
            'duration_seconds': total_duration,
            'duration_formatted': duration_str,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'all_trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value if t.value is not None else None,
                    'state': t.state.name
                }
                for t in study.trials
            ]
        }, f, indent=2)
    
    # 创建并准备最佳模型目录
    best_model_dir = os.path.join(save_path, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 找到最佳trial的检查点
    best_trial_checkpoint = os.path.join(checkpoint_dir, f"trial_{study.best_trial.number}_best.pth")
    best_model_path = None
    
    if os.path.exists(best_trial_checkpoint):
        # 复制到最佳模型目录
        best_model_path = os.path.join(best_model_dir, "best_model.pth")
        shutil.copy2(best_trial_checkpoint, best_model_path)
        print(f"\n已复制最佳模型检查点: {best_model_path}")
        
        # 保存最佳超参数到单独文件
        best_params_path = os.path.join(best_model_dir, "best_params.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
            
        # 加载最佳trial模型状态
        best_state = torch.load(best_trial_checkpoint, map_location=device)
        
        # 从最佳参数创建模型架构信息
        best_params = study.best_params
        model_type = best_params.get('model_type', 'base_mlp')
        
        # 处理隐藏层参数
        hidden_dims = None
        if 'layer_sizes_idx' in best_params:
            layer_sizes_idx = best_params.get('layer_sizes_idx', 0)
            # 使用param_space中的层大小选项
            if param_space and 'layer_sizes' in param_space:
                hidden_dims = param_space['layer_sizes'][layer_sizes_idx]
        elif 'depth' in best_params and 'width_factor' in best_params:
            # 深层MLP的情况
            depth = best_params.get('depth', 6)
            width = best_params.get('width_factor', 2048)
            hidden_dims = [width] * depth
            
        # 如果无法确定隐藏层大小，使用默认值
        if not hidden_dims:
            hidden_dims = [4096, 4096, 4096, 4096]
            
        # 创建包含模型完整信息的架构字典
        arch_info = {
            'model_type': model_type,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'dropout_rate': best_params.get('dropout_rate', 0.5),
            'activation': best_params.get('activation', 'relu'),
        }
        
        # 添加模型特定参数
        if model_type == 'deep_mlp':
            arch_info['use_skip_connections'] = best_params.get('use_skip_connections', False)
        elif model_type == 'residual_mlp':
            arch_info['use_bottleneck'] = best_params.get('use_bottleneck', False)
            if arch_info['use_bottleneck']:
                arch_info['bottleneck_factor'] = best_params.get('bottleneck_factor', 0.5)
        
        # 保存架构信息
        arch_info_path = os.path.join(best_model_dir, "architecture.json")
        with open(arch_info_path, 'w') as f:
            json.dump(arch_info, f, indent=2)
        
        # 保存完整配置信息
        complete_config = config.copy() if config else {}
        complete_config.update({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'optimization_completed': True,
            'n_trials': n_trials,
            'study_name': study_name,
            'completed_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_duration': duration_str
        })
            
        # 如果存在标准化器路径，复制到最佳模型目录
        if study.best_trial.user_attrs.get('scaler_path') and os.path.exists(study.best_trial.user_attrs.get('scaler_path')):
            best_scaler_path = os.path.join(best_model_dir, "scaler.pkl")
            shutil.copy2(study.best_trial.user_attrs.get('scaler_path'), best_scaler_path)
            print(f"标准化器已复制到最佳模型目录: {best_scaler_path}")
            
            # 更新完整配置
            complete_config['scaler_path'] = best_scaler_path
            
        config_path = os.path.join(best_model_dir, "best_config.json")
        with open(config_path, 'w') as f:
            json.dump(complete_config, f, indent=2)
        
        # 创建详细训练元数据
        metadata = {
            'date_completed': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'early_stopped_trials': len(pruned_trials),
            'total_duration': duration_str,
            'device': str(device),
            'optimization_study': study_name,
            'pytorch_version': torch.__version__,
            'optuna_version': optuna.__version__ if hasattr(optuna, '__version__') else 'unknown',
            'training_details': {
                'best_epoch': best_state.get('epoch', 0) if 'epoch' in best_state else 0,
                'learning_rate': best_params.get('learning_rate', 0),
                'optimizer': best_params.get('optimizer', 'adam'),
                'learning_rate_scheduler': best_params.get('lr_scheduler', 'none'),
            }
        }
        
        # 添加验证性能历史（如果可用）
        if 'val_f1_history' in best_state:
            metadata['training_details']['val_f1_history'] = best_state['val_f1_history']
            
        if 'train_losses' in best_state:
            metadata['training_details']['train_losses'] = best_state['train_losses']
            
        if 'lr_history' in best_state:
            metadata['training_details']['lr_history'] = best_state['lr_history']
        
        # 保存元数据
        metadata_path = os.path.join(best_model_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # 创建模型加载脚本
        load_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
加载最佳模型的简便脚本
\"\"\"

import os
import json
import torch
import sys

# 确保可以导入models模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_best_model(device='cpu'):
    \"\"\"
    加载最佳模型
    
    参数:
        device: 计算设备
        
    返回:
        model: 加载了权重的模型
    \"\"\"
    try:
        # 导入get_model函数
        from models import get_model
        
        # 加载架构信息
        with open('architecture.json', 'r') as f:
            arch_info = json.load(f)
        
        # 创建模型
        model = get_model(
            model_type=arch_info['model_type'],
            input_dim=arch_info['input_dim'],
            hidden_dims=arch_info['hidden_dims'],
            num_classes=arch_info['num_classes'],
            dropout_rate=arch_info['dropout_rate'],
            activation=arch_info['activation']
        )
        
        # 如果存在其他特定架构参数，设置它们
        if arch_info['model_type'] == 'deep_mlp' and 'use_skip_connections' in arch_info:
            model.use_skip_connections = arch_info['use_skip_connections']
        elif arch_info['model_type'] == 'residual_mlp':
            if 'use_bottleneck' in arch_info:
                model.use_bottleneck = arch_info['use_bottleneck']
            if 'bottleneck_factor' in arch_info:
                model.bottleneck_factor = arch_info['bottleneck_factor']
        
        # 加载模型权重
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        # 将模型设置为评估模式
        model.to(device)
        model.eval()
        
        print(f"成功加载最佳模型!")
        print(f"模型类型: {arch_info['model_type']}")
        print(f"输入维度: {arch_info['input_dim']}")
        print(f"类别数量: {arch_info['num_classes']}")
        
        return model
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

if __name__ == "__main__":
    model = load_best_model()
    if model:
        print("\\n模型加载成功，可以用于推理!")
"""
        
        # 保存加载脚本
        with open(os.path.join(best_model_dir, "load_model.py"), 'w') as f:
            f.write(load_script)
            
        # 如果配置中启用，创建推理脚本
        if config and config.get('create_inference_script', True):
            inference_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
使用最佳模型进行推理的示例脚本
\"\"\"

import torch
import numpy as np
import pickle
from load_model import load_best_model

def predict(model, features, device='cpu', apply_normalization=True, scaler_path='scaler.pkl'):
    \"\"\"
    使用模型进行预测
    
    参数:
        model: 模型
        features: 输入特征 (numpy数组或torch张量)
        device: 计算设备
        apply_normalization: 是否应用标准化
        scaler_path: 标准化器路径
        
    返回:
        predictions: 预测的类别
    \"\"\"
    # 应用标准化（如果需要）
    if apply_normalization and scaler_path:
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"已加载标准化器: {scaler_path}")
            
            # 确保输入是numpy数组
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
                
            # 应用标准化
            features_scaled = scaler.transform(features_np)
            
            # 转换回原始类型
            if isinstance(features, torch.Tensor):
                features = torch.FloatTensor(features_scaled)
            else:
                features = features_scaled
                
        except Exception as e:
            print(f"标准化处理失败: {e}")
            print("使用原始特征继续...")
    
    # 确保输入格式正确
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    
    # 移动到指定设备
    features = features.to(device)
    
    # 添加批次维度（如果需要）
    if len(features.shape) == 1:
        features = features.unsqueeze(0)
    
    # 进行预测
    with torch.no_grad():
        model.eval()
        outputs = model(features)
        _, predictions = torch.max(outputs, 1)
    
    return predictions.cpu().numpy()

if __name__ == "__main__":
    # 加载模型
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_best_model(device)
    
    if model:
        # 创建一些随机测试数据
        import json
        with open('architecture.json', 'r') as f:
            arch_info = json.load(f)
        
        input_dim = arch_info['feature_dim']
        num_samples = 5
        
        # 生成随机测试数据
        test_data = np.random.rand(num_samples, input_dim).astype(np.float32)
        
        # 进行预测
        predictions = predict(model, test_data, device, apply_normalization=True, scaler_path='scaler.pkl')
        
        print(f"\\n为{num_samples}个随机样本生成预测结果:")
        for i, pred in enumerate(predictions):
            print(f"样本 {i+1}: 预测类别 = {pred}")
        
        print("\\n这是一个示例脚本。在真实应用中，您应该加载实际数据并应用与训练时相同的预处理步骤。")
"""
            # 保存推理脚本
            with open(os.path.join(best_model_dir, "inference.py"), 'w') as f:
                f.write(inference_script)
        
        # 输出成功信息
        print("\n" + "="*70)
        print(f"已成功保存最佳模型及所有相关文件到: {best_model_dir}")
        print(f"架构信息: {arch_info_path}")
        print(f"配置信息: {config_path}")
        print(f"元数据: {metadata_path}")
        print("使用说明:")
        print(f"  1. 切换到 {best_model_dir} 目录")
        print(f"  2. 运行 python load_model.py 加载模型")
        print(f"  3. 运行 python inference.py 进行推理示例")
        
    else:
        print(f"\n警告: 未找到最佳trial检查点: {best_trial_checkpoint}")
        print("这可能是因为未启用检查点保存，或者所有试验都被pruned。")
        best_model_dir = None
    
    # 生成可视化分析
    visualization_path = os.path.join(save_path, f"{study_name}_analysis")
    try:
        visualize_optimization_results(study, visualization_path)
        print(f"\n优化分析可视化已保存到: {visualization_path}")
    except Exception as e:
        print(f"\n生成优化分析可视化时出错: {e}")
    
    # 尝试创建和导出最佳模型（用于部署）
    if best_model_path and config and config.get('prepare_deployment', False):
        try:
            # 创建最佳模型
            best_model = get_model(
                model_type=study.best_params.get('model_type', 'base_mlp'),
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=study.best_params.get('dropout_rate', 0.5),
                activation=study.best_params.get('activation', 'relu')
            )
            
            # 加载权重
            checkpoint = torch.load(best_model_path, map_location='cpu')
            best_model.load_state_dict(checkpoint['state_dict'])
            best_model.eval()
            
            # 导出为部署格式
            export_best_model_for_deployment(best_model_dir, best_model, {
                'feature_dim': input_dim,
                'num_classes': num_classes
            })
        except Exception as e:
            print(f"\n准备部署模型时出错: {e}")
    
    # 返回study对象、最佳参数和最佳模型目录
    return study, study.best_params, best_model_dir


def export_best_model_for_deployment(best_model_dir, model, config):
    """
    将最佳模型导出为便于部署的格式
    
    参数:
        best_model_dir: 最佳模型目录
        model: 模型对象
        config: 配置字典
    """
    import os
    import torch
    import time
    
    # 确保模型处于CPU上和评估模式
    model = model.cpu()
    model.eval()
    
    # 创建导出目录
    deployment_dir = os.path.join(best_model_dir, "deployment")
    os.makedirs(deployment_dir, exist_ok=True)
    
    # 保存PyTorch原生格式
    torch_path = os.path.join(deployment_dir, "model.pth")
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, torch_path)
    
    print(f"\n已保存PyTorch原生格式: {torch_path}")
    
    # 尝试导出为TorchScript格式
    try:
        # 创建示例输入
        example_input = torch.randn(1, config.get('feature_dim', 341))
        
        # 转换为TorchScript
        script_model = torch.jit.trace(model, example_input)
        script_path = os.path.join(deployment_dir, "model.pt")
        script_model.save(script_path)
        
        print(f"已导出TorchScript格式: {script_path}")
    except Exception as e:
        print(f"导出TorchScript模型时出错: {e}")
    
    # 尝试导出为ONNX格式
    try:
        # 检查是否安装了onnx
        import importlib.util
        onnx_spec = importlib.util.find_spec("onnx")
        if onnx_spec is None:
            print("未安装ONNX库，跳过ONNX导出")
        else:
            import onnx
            
            # 创建示例输入
            dummy_input = torch.randn(1, config.get('feature_dim', 341))
            
            # 导出ONNX
            onnx_path = os.path.join(deployment_dir, "model.onnx")
            torch.onnx.export(
                model,               # 模型
                dummy_input,         # 示例输入
                onnx_path,           # 输出路径
                export_params=True,  # 存储模型权重
                opset_version=11,    # ONNX操作集版本
                do_constant_folding=True,  # 是否执行常量折叠优化
                input_names=['input'],     # 输入名称
                output_names=['output'],   # 输出名称
                dynamic_axes={             # 动态轴（批次大小是动态的）
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            print(f"已导出ONNX格式: {onnx_path}")
    except Exception as e:
        print(f"导出ONNX模型时出错: {e}")
    
    # 创建简单的部署README
    readme = f"""# 模型部署指南

该目录包含经过优化的模型，可用于部署到生产环境。

## 模型格式

- `model.pth`: 原生PyTorch格式
- `model.pt`: TorchScript格式（可在C++/Java等环境使用）
- `model.onnx`: ONNX格式（跨平台格式，支持多种框架和硬件）

## 推理示例

### PyTorch原生格式
```python
import torch
from models import get_model  # 需要引入原始项目的模型定义

# 加载模型架构信息
with open('../architecture.json', 'r') as f:
    import json
    arch_info = json.load(f)

# 创建模型
model = get_model(
    model_type=arch_info['model_type'],
    input_dim=arch_info['input_dim'],
    hidden_dims=arch_info['hidden_dims'],
    num_classes=arch_info['num_classes'],
    dropout_rate=arch_info['dropout_rate'],
    activation=arch_info['activation']
)

# 加载权重
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 推理
with torch.no_grad():
    inputs = torch.randn(1, {config.get('feature_dim', 341)})  # 替换为实际输入
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
```

### TorchScript格式
```python
import torch

# 加载模型
model = torch.jit.load('model.pt')
model.eval()

# 推理
with torch.no_grad():
    inputs = torch.randn(1, {config.get('feature_dim', 341)})  # 替换为实际输入
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
```

### ONNX格式
```python
import onnxruntime
import numpy as np

# 加载ONNX模型
session = onnxruntime.InferenceSession('model.onnx')

# 准备输入
input_name = session.get_inputs()[0].name
inputs = np.random.randn(1, {config.get('feature_dim', 341)}).astype(np.float32)  # 替换为实际输入

# 推理
outputs = session.run(None, {{input_name: inputs}})
predictions = np.argmax(outputs[0], axis=1)
```

## 模型信息

- 输入维度: {config.get('feature_dim', 341)}
- 输出类别数: {config.get('num_classes', 102)}
- 模型类型: {model.__class__.__name__}

## 注意事项

- 确保输入数据经过与训练相同的预处理
- 部署前建议进行完整的集成测试
"""
    
    # 保存README
    with open(os.path.join(deployment_dir, "README.md"), 'w') as f:
        f.write(readme)
    
    print(f"已创建部署指南: {os.path.join(deployment_dir, 'README.md')}")
    print(f"模型已准备好用于部署: {deployment_dir}")


def print_architecture_distribution(study):
    """输出不同架构的试验分布情况"""
    # 统计不同架构的试验次数
    arch_counts = {}
    arch_values = {}
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        if 'model_type' in trial.params:
            model_type = trial.params['model_type']
            if model_type not in arch_counts:
                arch_counts[model_type] = 0
                arch_values[model_type] = []
            
            arch_counts[model_type] += 1
            if trial.value is not None:
                arch_values[model_type].append(trial.value)
    
    # 打印分布情况
    print("\n当前架构分布情况:")
    for arch, count in arch_counts.items():
        avg_value = np.mean(arch_values[arch]) if arch_values[arch] else float('nan')
        max_value = np.max(arch_values[arch]) if arch_values[arch] else float('nan')
        print(f"  {arch}: {count} 次试验, 平均 F1: {avg_value:.4f}, 最佳 F1: {max_value:.4f}")


def visualize_optimization_results(study, save_path):
    """
    生成并保存贝叶斯优化的详细可视化分析
    
    参数:
        study: Optuna study对象
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from optuna.visualization import plot_param_importances, plot_optimization_history, plot_slice
    
    os.makedirs(save_path, exist_ok=True)
    
    # 获取所有完成的试验
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # 创建数据框
    trial_data = []
    for t in completed_trials:
        data = {
            'number': t.number,
            'value': t.value,
            **t.params
        }
        trial_data.append(data)
    
    df = pd.DataFrame(trial_data)
    
    # 1. 每种架构的性能箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model_type', y='value', data=df)
    plt.title('Performance by Model Architecture')
    plt.xlabel('Model Architecture')
    plt.ylabel('Validation F1 Score')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_path, 'architecture_performance_boxplot.png'))
    plt.close()
    
    # 2. 每种架构的试验计数
    arch_counts = df['model_type'].value_counts()
    plt.figure(figsize=(8, 6))
    arch_counts.plot(kind='bar')
    plt.title('Number of Trials per Architecture')
    plt.xlabel('Model Architecture')
    plt.ylabel('Number of Trials')
    plt.grid(True, axis='y')
    for i, v in enumerate(arch_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'architecture_trial_counts.png'))
    plt.close()
    
    # 3. 性能随时间变化
    plt.figure(figsize=(12, 6))
    for arch in df['model_type'].unique():
        arch_df = df[df['model_type'] == arch]
        plt.plot(arch_df['number'], arch_df['value'], 'o-', label=arch)
    plt.title('Performance Evolution by Architecture')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'performance_evolution.png'))
    plt.close()
    
    # 4. 参数重要性
    try:
        importance_fig = plot_param_importances(study)
        importance_fig.write_image(os.path.join(save_path, 'parameter_importance.png'))
    except Exception as e:
        print(f"无法生成参数重要性图: {e}")
    
    # 5. 优化历史
    try:
        history_fig = plot_optimization_history(study)
        history_fig.write_image(os.path.join(save_path, 'optimization_history.png'))
    except Exception as e:
        print(f"无法生成优化历史图: {e}")
    
    # 6. 为每种架构生成参数相关性热图
    for arch in df['model_type'].unique():
        arch_df = df[df['model_type'] == arch]
        if len(arch_df) < 5:  # 跳过试验次数不足的架构
            continue
            
        # 选择数值型参数
        numeric_cols = arch_df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['number']]
        
        if len(numeric_cols) >= 2:  # 至少需要2个数值参数才能计算相关性
            plt.figure(figsize=(10, 8))
            sns.heatmap(arch_df[numeric_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Parameter Correlation for {arch}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{arch}_parameter_correlation.png'))
            plt.close()
    
    # 7. 生成详细报告
    with open(os.path.join(save_path, 'optimization_report.txt'), 'w') as f:
        f.write(f"贝叶斯优化结果报告\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"优化名称: {study.study_name}\n")
        f.write(f"总试验次数: {len(completed_trials)}\n")
        f.write(f"最佳F1分数: {study.best_value:.6f}\n\n")
        
        f.write(f"最佳参数:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        
        f.write(f"\n架构分布统计:\n")
        for arch, count in arch_counts.items():
            arch_best = df[df['model_type'] == arch]['value'].max()
            arch_mean = df[df['model_type'] == arch]['value'].mean()
            f.write(f"  {arch}: {count} 次试验, 最佳 F1: {arch_best:.6f}, 平均 F1: {arch_mean:.6f}\n")
        
        # 验证每种架构是否被充分探索
        f.write(f"\n架构探索充分性分析:\n")
        for arch in df['model_type'].unique():
            arch_df = df[df['model_type'] == arch]
            if len(arch_df) < 5:
                f.write(f"  {arch}: 警告! 仅有 {len(arch_df)} 次试验, 可能未被充分探索\n")
            else:
                # 计算最近5次试验的性能变化
                recent_trials = arch_df.sort_values('number', ascending=False).head(5)
                improvement = recent_trials['value'].max() - recent_trials['value'].min()
                if improvement > 0.02:  # 如果仍有明显改进
                    f.write(f"  {arch}: 最近5次试验仍有 {improvement:.4f} 的性能改进, 可能需要更多试验\n")
                else:
                    f.write(f"  {arch}: 最近5次试验性能稳定, 探索可能已充分\n")