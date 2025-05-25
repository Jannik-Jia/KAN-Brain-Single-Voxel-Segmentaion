#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脑体素分类快速耦合验证脚本
完全基于原始brain_voxel.ipynb的训练流程
验证不同架构在不同学习率下的表现差异
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from datetime import datetime

class OriginalStyleMLP(nn.Module):
    """
    完全复制原始notebook的MLP架构
    包含L2正则化 (kernel_regularizer=tf.keras.regularizers.l2(0.00001))
    """
    def __init__(self, input_dim=341, hidden_dims=[4096, 4096, 4096, 4096], 
                 num_classes=102, dropout_rate=0.5, l2_lambda=1e-5):
        super(OriginalStyleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 中间隐藏层
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层 (对应原始的activation='softmax')
        layers.append(nn.Linear(self.hidden_dims[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"    Model: {self.hidden_dims} -> {num_classes}")
        print(f"    Parameters: {total_params:,}")
        
    def forward(self, x):
        return self.network(x)
    
    def get_l2_loss(self):
        """计算L2正则化损失，对应原始的kernel_regularizer"""
        l2_loss = 0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, 2) ** 2
        return self.l2_lambda * l2_loss

def load_original_data(mat_file_path, test_size=0.01, random_state=42):
    """
    完全复制原始notebook的数据加载和预处理流程
    """
    print(f"Loading data from: {mat_file_path}")
    
    # 原始代码: f = h5py.File('TRAIN38.mat','r')
    f = h5py.File(mat_file_path, 'r')
    arrays = {}
    for k, v in f.items():
        arrays[k] = np.array(v)
    f.close()
    
    # 原始代码: 转置操作
    train_data = arrays['data'].transpose()
    train_region = arrays['region'].transpose()
    prob_idx = arrays['prob_idx'].transpose()
    
    print(f"Original data shape: {train_data.shape}")
    print(f"Original region shape: {train_region.shape}")
    
    # 原始代码: 数据分割
    # curr_set = np.where(prob_idx != 38)[0]
    curr_set = np.where(prob_idx != 38)[0]
    set_data = train_data[curr_set, :]
    set_region = train_region[curr_set, :]
    print(f"Training set shape: {set_data.shape}")
    
    # curr_val = np.where(prob_idx == 38)[0]
    curr_val = np.where(prob_idx == 38)[0]
    val_data = train_data[curr_val, :]
    val_label = train_region[curr_val, :]
    print(f"Validation set shape: {val_data.shape}")
    
    # 原始代码: 进一步分割训练集
    # X_train1, X_train2, y_train1, y_train2 = train_test_split(set_data,set_region,test_size = 0.01, random_state = 42)
    X_train1, X_train2, y_train1, y_train2 = train_test_split(
        set_data, set_region, test_size=test_size, random_state=random_state
    )
    
    # 原始代码: StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(X_train1)
    # X_train1 = scaler.transform(X_train1)
    # val_data = scaler.transform(val_data)
    scaler = StandardScaler()
    scaler.fit(X_train1)
    X_train1_scaled = scaler.transform(X_train1)
    val_data_scaled = scaler.transform(val_data)
    
    print(f"Final training data shape: {X_train1_scaled.shape}")
    print(f"Final validation data shape: {val_data_scaled.shape}")
    
    return {
        'X_train': X_train1_scaled,
        'y_train': y_train1,  # one-hot format
        'X_val': val_data_scaled,
        'y_val': val_label,   # one-hot format
        'scaler': scaler,
        'input_dim': X_train1_scaled.shape[1]  # 341
    }

def train_original_style(model, X_train, y_train, X_val, y_val, 
                        learning_rate=1e-5, batch_size=128, epochs=25, device='cuda'):
    """
    完全复制原始notebook的训练流程
    """
    model = model.to(device)
    
    # 原始代码: optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 原始代码: loss='categorical_crossentropy'
    criterion = nn.CrossEntropyLoss()
    
    # 转换数据到PyTorch格式
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # 将one-hot转为类别索引
    y_train_idx = torch.argmax(y_train_tensor, dim=1)
    y_val_idx = torch.argmax(y_val_tensor, dim=1)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_kappa': []
    }
    
    print(f"    Training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 添加L2正则化 (对应原始的kernel_regularizer)
            l2_loss = model.get_l2_loss()
            total_loss = loss + l2_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_idx)
            val_loss += model.get_l2_loss()
            
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == y_val_idx).sum().item() / y_val_idx.size(0)
            
            # 计算F1和Kappa指标
            y_val_np = y_val_idx.cpu().numpy()
            val_pred_np = val_predicted.cpu().numpy()
            val_f1 = f1_score(y_val_np, val_pred_np, average='macro')
            val_kappa = cohen_kappa_score(y_val_np, val_pred_np)
        
        # 记录历史
        train_acc = train_correct / train_total
        history['train_loss'].append(epoch_train_loss / len(dataloader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_kappa'].append(val_kappa)
        
        # 每5轮或第1轮打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs}: "
                  f"Train Loss: {epoch_train_loss/len(dataloader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val F1: {val_f1:.4f}")
    
    training_time = time.time() - start_time
    final_f1 = history['val_f1'][-1]
    best_f1 = max(history['val_f1'])
    
    print(f"    ✓ Completed in {training_time:.1f}s, Final F1: {final_f1:.4f}, Best F1: {best_f1:.4f}")
    
    return {
        'history': history,
        'training_time': training_time,
        'final_f1': final_f1,
        'best_f1': best_f1,
        'final_acc': history['val_acc'][-1],
        'best_acc': max(history['val_acc'])
    }

def run_coupling_experiment(mat_file_path, output_dir="./coupling_test_results", device='cuda'):
    """
    运行快速耦合验证实验
    """
    print("=" * 80)
    print("脑体素分类 - 架构超参数耦合验证实验")
    print("基于原始brain_voxel.ipynb训练流程")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n📊 数据加载与预处理...")
    data = load_original_data(mat_file_path)
    
    # 实验配置
    architectures = {
        'original': [4096, 4096, 4096, 4096],           # 原始配置 (baseline)
        'wide_shallow': [8192, 8192],                   # 宽而浅
        'narrow_deep': [2048, 2048, 2048, 2048, 2048, 2048]  # 窄而深
    }
    
    # 学习率围绕原始值 (1e-5) 进行测试
    learning_rates = [5e-6, 1e-5, 2e-5, 5e-5]
    
    # 固定参数 (与原始notebook一致)
    fixed_params = {
        'batch_size': 128,    # 原始: batch_size = 128
        'epochs': 12,         # 快速测试 (原始: no_epochs = 25)
        'dropout_rate': 0.5,  # 原始: Dropout(0.5)
        'input_dim': data['input_dim'],  # 341
        'num_classes': 102,   # 原始: no_classes = 102
        'l2_lambda': 1e-5     # 原始: kernel_regularizer=tf.keras.regularizers.l2(0.00001)
    }
    
    total_experiments = len(architectures) * len(learning_rates)
    print(f"\n🧪 实验配置:")
    print(f"   架构数量: {len(architectures)} ({list(architectures.keys())})")
    print(f"   学习率数量: {len(learning_rates)} ({learning_rates})")
    print(f"   总实验数: {total_experiments}")
    print(f"   预计时间: {total_experiments * 8}分钟")
    
    # 开始实验
    results = {}
    experiment_log = []
    current_exp = 0
    
    print(f"\n🚀 开始耦合测试实验...")
    overall_start = time.time()
    
    for arch_name, hidden_dims in architectures.items():
        print(f"\n📐 测试架构: {arch_name.upper()}")
        print(f"   Hidden layers: {hidden_dims}")
        
        results[arch_name] = {}
        
        for lr in learning_rates:
            current_exp += 1
            exp_name = f"{arch_name}_lr{lr:.0e}"
            
            print(f"\n  [{current_exp:2d}/{total_experiments}] {exp_name}")
            print(f"     Learning rate: {lr:.0e}")
            
            try:
                # 创建模型
                model = OriginalStyleMLP(
                    input_dim=fixed_params['input_dim'],
                    hidden_dims=hidden_dims,
                    num_classes=fixed_params['num_classes'],
                    dropout_rate=fixed_params['dropout_rate'],
                    l2_lambda=fixed_params['l2_lambda']
                )
                
                # 训练模型
                train_result = train_original_style(
                    model=model,
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_val=data['X_val'],
                    y_val=data['y_val'],
                    learning_rate=lr,
                    batch_size=fixed_params['batch_size'],
                    epochs=fixed_params['epochs'],
                    device=device
                )
                
                # 保存结果
                results[arch_name][lr] = {
                    'best_f1': train_result['best_f1'],
                    'final_f1': train_result['final_f1'],
                    'best_acc': train_result['best_acc'],
                    'final_acc': train_result['final_acc'],
                    'training_time': train_result['training_time'],
                    'converged': True
                }
                
                # 记录实验日志
                experiment_log.append({
                    'experiment': exp_name,
                    'architecture': arch_name,
                    'hidden_dims': hidden_dims,
                    'learning_rate': lr,
                    'best_f1': train_result['best_f1'],
                    'final_f1': train_result['final_f1'],
                    'training_time': train_result['training_time'],
                    'success': True
                })
                
            except Exception as e:
                print(f"     ✗ 实验失败: {str(e)}")
                results[arch_name][lr] = {'error': str(e), 'converged': False}
                experiment_log.append({
                    'experiment': exp_name,
                    'architecture': arch_name,
                    'learning_rate': lr,
                    'error': str(e),
                    'success': False
                })
    
    total_time = time.time() - overall_start
    print(f"\n🎉 所有实验完成! 总用时: {total_time/60:.1f}分钟")
    
    # 保存和分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyze_and_save_results(results, experiment_log, output_dir, timestamp)
    
    return results

def analyze_and_save_results(results, experiment_log, output_dir, timestamp):
    """
    分析耦合实验结果并保存
    """
    print(f"\n📈 结果分析...")
    
    # 分析每种架构的表现
    analysis = {
        'coupling_strength': 'unknown',
        'best_configs': {},
        'recommendations': []
    }
    
    print(f"\n{'='*60}")
    print("各架构最佳配置分析:")
    print(f"{'='*60}")
    
    for arch_name in results:
        if not any(results[arch_name][lr].get('converged', False) for lr in results[arch_name]):
            print(f"{arch_name}: 所有配置都失败了")
            continue
            
        # 找到最佳配置
        best_f1 = 0
        best_lr = None
        all_f1s = []
        
        for lr in results[arch_name]:
            if results[arch_name][lr].get('converged', False):
                f1 = results[arch_name][lr]['best_f1']
                all_f1s.append(f1)
                if f1 > best_f1:
                    best_f1 = f1
                    best_lr = lr
        
        # 计算性能方差和范围
        variance = np.var(all_f1s) if len(all_f1s) > 1 else 0
        range_val = max(all_f1s) - min(all_f1s) if len(all_f1s) > 1 else 0
        
        analysis['best_configs'][arch_name] = {
            'best_lr': best_lr,
            'best_f1': best_f1,
            'variance': variance,
            'range': range_val,
            'all_f1s': all_f1s
        }
        
        print(f"{arch_name:12s}: 最佳LR={best_lr:.0e}, 最佳F1={best_f1:.4f}, "
              f"方差={variance:.6f}, 范围={range_val:.4f}")
        
        # 显示详细结果
        lr_results = []
        for lr in sorted(results[arch_name].keys()):
            if results[arch_name][lr].get('converged', False):
                f1 = results[arch_name][lr]['best_f1']
                lr_results.append(f"LR={lr:.0e}→F1={f1:.4f}")
        print(f"             详细: {' | '.join(lr_results)}")
    
    # 判断耦合强度
    variances = [info['variance'] for info in analysis['best_configs'].values()]
    ranges = [info['range'] for info in analysis['best_configs'].values()]
    
    avg_variance = np.mean(variances) if variances else 0
    avg_range = np.mean(ranges) if ranges else 0
    
    if avg_variance > 0.002 or avg_range > 0.05:
        coupling_strength = "强耦合"
        recommendation = "建议使用联合优化 (贝叶斯优化或网格搜索)"
    elif avg_variance > 0.0005 or avg_range > 0.02:
        coupling_strength = "中等耦合"
        recommendation = "建议使用分层优化或小规模联合优化"
    else:
        coupling_strength = "弱耦合"
        recommendation = "可以分步优化: 先确定最佳架构,再调整超参数"
    
    analysis['coupling_strength'] = coupling_strength
    analysis['avg_variance'] = avg_variance
    analysis['avg_range'] = avg_range
    analysis['recommendations'] = [recommendation]
    
    # 找出总体最佳配置
    overall_best_f1 = 0
    overall_best_config = None
    
    for arch_name in analysis['best_configs']:
        if analysis['best_configs'][arch_name]['best_f1'] > overall_best_f1:
            overall_best_f1 = analysis['best_configs'][arch_name]['best_f1']
            overall_best_config = (arch_name, analysis['best_configs'][arch_name]['best_lr'])
    
    print(f"\n{'='*60}")
    print("耦合分析结果:")
    print(f"{'='*60}")
    print(f"耦合强度: {coupling_strength}")
    print(f"平均方差: {avg_variance:.6f}")
    print(f"平均范围: {avg_range:.4f}")
    print(f"建议策略: {recommendation}")
    
    if overall_best_config:
        arch, lr = overall_best_config
        print(f"最佳配置: {arch} + LR={lr:.0e} (F1={overall_best_f1:.4f})")
    
    # 保存结果文件
    results_file = os.path.join(output_dir, f"coupling_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        # 转换为可序列化格式
        serializable_results = {}
        for arch in results:
            serializable_results[arch] = {}
            for lr in results[arch]:
                if 'best_f1' in results[arch][lr]:
                    serializable_results[arch][lr] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in results[arch][lr].items()
                    }
                else:
                    serializable_results[arch][lr] = results[arch][lr]
        
        json.dump({
            'results': serializable_results,
            'analysis': analysis,
            'experiment_log': experiment_log,
            'timestamp': timestamp
        }, f, indent=2, default=str)
    
    # 生成报告文件
    report_file = os.path.join(output_dir, f"coupling_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write("脑体素分类 - 架构超参数耦合验证报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验配置: 3种架构 × 4个学习率 = 12个实验\n\n")
        
        f.write("各架构最佳配置:\n")
        f.write("-" * 30 + "\n")
        for arch_name in analysis['best_configs']:
            config = analysis['best_configs'][arch_name]
            f.write(f"{arch_name}:\n")
            f.write(f"  最佳学习率: {config['best_lr']:.0e}\n")
            f.write(f"  最佳F1分数: {config['best_f1']:.4f}\n")
            f.write(f"  性能方差: {config['variance']:.6f}\n")
            f.write(f"  性能范围: {config['range']:.4f}\n\n")
        
        f.write("耦合分析:\n")
        f.write("-" * 30 + "\n")
        f.write(f"耦合强度: {coupling_strength}\n")
        f.write(f"平均方差: {avg_variance:.6f}\n")
        f.write(f"平均范围: {avg_range:.4f}\n")
        f.write(f"优化建议: {recommendation}\n\n")
        
        if overall_best_config:
            arch, lr = overall_best_config
            f.write(f"推荐配置: {arch} 架构 + {lr:.0e} 学习率\n")
            f.write(f"预期F1分数: {overall_best_f1:.4f}\n\n")
        
        f.write("实验详情:\n")
        f.write("-" * 30 + "\n")
        for log_entry in experiment_log:
            if log_entry.get('success', False):
                f.write(f"{log_entry['experiment']}: F1={log_entry['best_f1']:.4f}, "
                       f"时间={log_entry['training_time']:.1f}s\n")
            else:
                f.write(f"{log_entry['experiment']}: 失败 - {log_entry.get('error', 'Unknown')}\n")
    
    print(f"\n💾 结果已保存:")
    print(f"   详细结果: {results_file}")
    print(f"   分析报告: {report_file}")
    
    return analysis

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脑体素分类架构-超参数耦合验证')
    parser.add_argument('--mat_file_path', type=str, required=True, help='TRAIN38.mat文件路径')
    parser.add_argument('--output_dir', type=str, default='./coupling_test_results', help='结果输出目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查文件是否存在
    if not os.path.exists(args.mat_file_path):
        print(f"错误: 数据文件不存在: {args.mat_file_path}")
        sys.exit(1)
    
    # 运行实验
    try:
        results = run_coupling_experiment(
            mat_file_path=args.mat_file_path,
            output_dir=args.output_dir,
            device=device
        )
        print(f"\n✅ 耦合测试实验成功完成!")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()