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
from utils.label_processing import create_criterion_with_background_config

def objective(trial, data_loaders, input_dim, num_classes, device, param_space=None, config=None):
    """
    扩展版的Optuna优化目标函数 - 更大的搜索空间
    """
    epochs = config.get('epochs', 30) if config else 30
    train_epochs = max(5, min(epochs, 20))  # 优化时使用15-20轮
    
    # ===== 大幅扩展的共享参数空间 =====
    shared_params = {
        # 学习率：扩大范围，更细致的搜索
        'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True),
        
        # 权重衰减：使用log scale，但避免0值
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True),
        
        # 优化器：添加更多选项
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']),
        
        # Dropout：更大范围
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.8),
        
        # 激活函数：只使用现有模型支持的激活函数
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
        
        # 学习率调度器
        'lr_scheduler': trial.suggest_categorical('lr_scheduler', 
            ['cosine', 'step', 'plateau', 'none']),
    }
    
    # SGD特有参数
    if shared_params['optimizer'] == 'sgd':
        shared_params['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
        shared_params['nesterov'] = trial.suggest_categorical('nesterov', [True, False])
    
    # RMSprop特有参数
    if shared_params['optimizer'] == 'rmsprop':
        shared_params['rmsprop_alpha'] = trial.suggest_float('rmsprop_alpha', 0.9, 0.999)
    
    # 模型类型
    model_type = trial.suggest_categorical('model_type', ['base_mlp', 'deep_mlp', 'residual_mlp'])
    
    # ===== 架构特定参数空间（大幅扩展）=====
    if model_type == 'base_mlp':
        # 更多层数和宽度选项
        layer_sizes_options = [
            [8192, 8192, 8192, 8192],           # 超大网络
            [6144, 6144, 6144, 6144],           # 大网络
            [4096, 4096, 4096, 4096],           # 标准大网络
            [3072, 3072, 3072, 3072],           # 中等网络
            [2048, 2048, 2048, 2048],           # 较小网络
            [1024, 1024, 1024, 1024],           # 小网络
            [4096, 2048, 1024, 512],            # 递减网络
            [512, 1024, 2048, 4096],            # 递增网络
            [2048, 4096, 4096, 2048],           # 钟形网络
            [4096, 2048, 2048, 4096],           # 沙漏网络
            [1024, 2048, 4096, 2048, 1024],    # 5层钟形
            [4096, 4096, 4096, 4096, 4096, 4096], # 6层网络
        ]
        layer_sizes_idx = trial.suggest_int('layer_sizes_idx', 0, len(layer_sizes_options)-1)
        hidden_dims = layer_sizes_options[layer_sizes_idx]
        model_params = {}
        
    elif model_type == 'deep_mlp':
        # 深度网络：更多层数选择
        depth = trial.suggest_int('depth', 4, 12)  # 4-12层
        
        # 宽度策略
        width_strategy = trial.suggest_categorical('width_strategy', 
            ['constant', 'decreasing', 'increasing', 'hourglass', 'bell'])
        
        base_width = trial.suggest_categorical('base_width', 
            [512, 768, 1024, 1536, 2048, 3072, 4096, 6144])
        
        # 根据策略生成层宽度
        if width_strategy == 'constant':
            hidden_dims = [base_width] * depth
        elif width_strategy == 'decreasing':
            hidden_dims = [max(256, int(base_width * (0.8 ** i))) for i in range(depth)]
        elif width_strategy == 'increasing':
            hidden_dims = [min(8192, int(base_width * (1.2 ** i))) for i in range(depth)]
        elif width_strategy == 'hourglass':
            mid = depth // 2
            hidden_dims = ([max(256, int(base_width * (0.7 ** i))) for i in range(mid)] + 
                          [max(256, int(base_width * (0.7 ** (depth-i-1)))) for i in range(mid, depth)])
        else:  # bell
            mid = depth // 2
            hidden_dims = ([min(8192, int(base_width * (1.3 ** i))) for i in range(mid)] + 
                          [min(8192, int(base_width * (1.3 ** (depth-i-1)))) for i in range(mid, depth)])
        
        model_params = {
            'use_skip_connections': trial.suggest_categorical('use_skip_connections', [True, False]),
        }
        
    elif model_type == 'residual_mlp':
        # 残差网络：更多配置选项
        num_blocks = trial.suggest_int('num_blocks', 2, 8)
        block_width = trial.suggest_categorical('block_width', 
            [512, 768, 1024, 1536, 2048, 3072, 4096, 6144])
        
        # 每个块使用相同宽度
        hidden_dims = [block_width] * (num_blocks * 2)  # 每个残差块通常有2层
        
        model_params = {
            'use_bottleneck': trial.suggest_categorical('use_bottleneck', [True, False]),
        }
        if model_params['use_bottleneck']:
            model_params['bottleneck_factor'] = trial.suggest_float('bottleneck_factor', 0.1, 0.5)
    
    # 创建模型
    model_kwargs = {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'num_classes': num_classes,
        'dropout_rate': shared_params['dropout_rate'],
        'activation': shared_params['activation'],
        **model_params
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
    elif shared_params['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=shared_params['learning_rate'], 
            weight_decay=shared_params['weight_decay']
        )
    elif shared_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=shared_params['learning_rate'], 
            weight_decay=shared_params['weight_decay'],
            momentum=shared_params.get('momentum', 0.9),
            nesterov=shared_params.get('nesterov', False)
        )
    elif shared_params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=shared_params['learning_rate'], 
            weight_decay=shared_params['weight_decay'],
            alpha=shared_params.get('rmsprop_alpha', 0.99)
        )
    
    # 学习率调度器
    lr_scheduler = None
    if shared_params['lr_scheduler'] == 'cosine':
        t_max = train_epochs
        eta_min = trial.suggest_float('cosine_eta_min', 1e-8, 1e-6, log=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif shared_params['lr_scheduler'] == 'step':
        step_size = trial.suggest_int('step_size', 1, max(1, train_epochs // 2))
        gamma = trial.suggest_float('step_gamma', 0.1, 0.5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif shared_params['lr_scheduler'] == 'plateau':
        patience = trial.suggest_int('plateau_patience', 1, max(1, train_epochs // 3))
        factor = trial.suggest_float('plateau_factor', 0.1, 0.5)
        threshold = trial.suggest_float('plateau_threshold', 1e-4, 1e-2, log=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, 
            threshold=threshold, verbose=False
        )
    
    # 创建损失函数
    criterion = create_criterion_with_background_config(config)
    
    # 训练模型
    val_f1_values = []
    
    for epoch in range(train_epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(data_loaders['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪（如果需要）
            if trial.suggest_categorical('use_gradient_clip', [True, False]):
                grad_clip_value = trial.suggest_float('grad_clip_value', 0.5, 5.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        # 验证
        model.eval()
        all_preds = []
        all_targets = []
        
        from utils.label_processing import get_ignore_index
        ignore_index = get_ignore_index(config) if config else None
        
        with torch.no_grad():
            for data, target in data_loaders['val']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                
                if ignore_index is not None:
                    valid_mask = (target != ignore_index)
                    if valid_mask.sum() > 0:
                        all_preds.extend(preds[valid_mask].cpu().numpy())
                        all_targets.extend(target[valid_mask].cpu().numpy())
                else:
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        # 计算F1分数
        if len(all_preds) > 0 and len(all_targets) > 0:
            val_f1_macro = f1_score(all_targets, all_preds, average='macro')
        else:
            val_f1_macro = 0.0
            
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
    
    return max(val_f1_values) if val_f1_values else 0.0


# 新增函数：针对Patientwise的专门优化
def run_patientwise_optimization(data_loaders, input_dim, num_classes, device, 
                                 n_trials=30, study_name="patientwise_optimization", 
                                 save_path="./results", config=None):
    """
    专门针对Patientwise标准化的贝叶斯优化
    
    这个函数假设数据已经使用patientwise标准化处理
    主要优化模型架构和训练超参数
    """
    import optuna
    
    # 确保使用patientwise标准化
    if config:
        config['standardization_method'] = 'patientwise'
    
    # 定义参数空间 - 可能需要针对patientwise调整
    param_space = {
        'learning_rate': (1e-6, 1e-4),  # 可能需要更小的学习率
        'batch_size': [64, 128, 256],   # 批次大小可能影响患者分布
        'weight_decay': (1e-6, 1e-3),
        'dropout_rate': (0.3, 0.7),      # 可能需要更高的dropout
        'activation': ['relu', 'gelu', 'swish'],
        'optimizer': ['adam', 'adamw'],
        'lr_scheduler': ['cosine', 'plateau', 'none'],  # 去掉step调度器
        'model_type': ['base_mlp', 'deep_mlp', 'residual_mlp'],
    }
    
    # 创建研究
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=np.random.randint(1, 10000),
            n_startup_trials=10,
            n_ei_candidates=24,
            prior_weight=1.0,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=5,
            n_min_trials=5  # 确保至少有5个试验完成
        ),
        study_name=study_name
    )
    
    # 定义专门的目标函数
    def patientwise_objective(trial):
        # 特别注意：验证集只有一个患者(20)，可能需要特殊处理
        # 可以考虑使用训练集的一部分患者作为额外验证
        
        return objective(trial, data_loaders, input_dim, num_classes, device, param_space, config)
    
    # 运行优化
    study.optimize(
        patientwise_objective,
        n_trials=n_trials,
        callbacks=[
            lambda study, trial: print_optimization_progress(study, trial),
            lambda study, trial: check_patient_performance(study, trial, config)
        ]
    )
    
    # ... 保存结果等后续处理 ...
    
    return study, study.best_params

def check_patient_performance(study, trial, config):
    """
    检查每个患者的性能，确保模型不会过度偏向某些患者
    """
    if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
        # 这里可以添加代码来分析不同患者的性能差异
        # 例如，记录训练集中不同患者的平均损失
        pass

def print_optimization_progress(study, trial):
    """
    打印优化进度，包括当前最佳结果
    """
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"\n试验 {trial.number} 完成:")
        print(f"  F1分数: {trial.value:.4f}")
        print(f"  当前最佳F1: {study.best_value:.4f}")
        
        # 打印关键参数
        important_params = ['model_type', 'learning_rate', 'dropout_rate', 'standardization_method']
        for param in important_params:
            if param in trial.params:
                print(f"  {param}: {trial.params[param]}")


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
    # 检查是否使用patientwise标准化
    if config and config.get('standardization_method') == 'patientwise':
        print("检测到Patientwise标准化，使用专门的优化策略")
        return run_patientwise_optimization(
            data_loaders, input_dim, num_classes, device, 
            n_trials, study_name + "_patientwise", save_path, config
        )
    

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