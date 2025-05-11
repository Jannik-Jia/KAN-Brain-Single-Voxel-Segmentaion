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
    Optuna优化目标函数，保持共享参数空间一致性
    """
    # 配置信息
    epochs = config.get('epochs', 30) if config else 30
    # 确保epochs至少为5，防止t_max参数出错
    train_epochs = max(5, min(epochs, 10))  # 贝叶斯优化时使用较少的epoch，但至少5轮
    
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
        patience = trial.suggest_int('plateau_patience', 1, max(1, train_epochs // 3))
        factor = trial.suggest_float('plateau_factor', 0.1, 0.5)
        threshold = trial.suggest_float('plateau_threshold', 1e-4, 1e-2, log=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, 
            threshold=threshold, verbose=True
        )
    
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # 训练模型 - 简化版本，只训练几个epoch用于评估
    val_f1_values = []
    
    # 训练循环
    for epoch in range(train_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loaders['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loaders['val']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                
                # 只评估非背景像素
                valid_mask = target != -1
                all_preds.extend(preds[valid_mask].cpu().numpy())
                all_targets.extend(target[valid_mask].cpu().numpy())
        
        # 计算F1分数
        val_f1_macro = f1_score(all_targets, all_preds, average='macro')
        val_f1_values.append(val_f1_macro)
        
        # 更新学习率调度器
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_f1_macro)
            else:
                lr_scheduler.step()
        
        # 报告进度
        trial.report(val_f1_macro, epoch)
        
        # 提前停止
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # 返回最佳F1分数
    return max(val_f1_values)


def run_bayesian_optimization(data_loaders, input_dim, num_classes, device, param_space=None, 
                             n_trials=30, study_name="mlp_optimization", save_path="./results", config=None):
    """
    运行贝叶斯优化
    
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
    """
    # 创建研究
    study = optuna.create_study(
        direction="maximize",
        # 修改采样器设置，增加早期随机探索
        sampler=optuna.samplers.TPESampler(
            seed=np.random.randint(1, 10000),
            n_startup_trials=10,  # 增加初始随机试验数量
            n_ei_candidates=24,   # 增加候选点数量，提高探索性
            prior_weight=1.0,     # 给先验分布更多权重
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=study_name
    )
    
    # 运行优化
    study.optimize(
        lambda trial: objective(trial, data_loaders, input_dim, num_classes, device, param_space, config),
        n_trials=n_trials,
        callbacks=[lambda study, trial: print_architecture_distribution(study)]
    )
    
    # 打印优化结果
    print("贝叶斯优化完成.")
    print(f"最佳F1分数: {study.best_value:.4f}")
    print("最佳参数:")
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
    
    # 生成可视化分析
    visualization_path = os.path.join(save_path, f"{study_name}_analysis")
    visualize_optimization_results(study, visualization_path)
    
    return study, study.best_params



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