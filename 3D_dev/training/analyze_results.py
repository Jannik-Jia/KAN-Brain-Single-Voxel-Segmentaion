#!/usr/bin/env python3
"""
分析Leave-one-out训练结果
计算平均性能和标准差
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_results(results_dir: Path, patch_size: int):
    """加载所有测试结果"""
    results = []
    
    for test_subject in range(1, 39):
        history_file = results_dir / f'history_patch{patch_size}_test{test_subject}.json'
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                
                # 获取最佳结果
                best_f1 = max(history['test_f1'])
                best_epoch = history['test_f1'].index(best_f1) + 1
                final_train_f1 = history['train_f1'][best_epoch - 1]
                
                results.append({
                    'test_subject': test_subject,
                    'patch_size': patch_size,
                    'best_test_f1': best_f1,
                    'best_epoch': best_epoch,
                    'final_train_f1': final_train_f1,
                    'final_test_loss': history['test_loss'][best_epoch - 1],
                    'total_epochs': len(history['test_f1'])
                })
    
    return results

def analyze_and_plot(results_3x3, results_7x7, output_dir):
    """分析结果并生成图表"""
    
    # 转换为DataFrame
    df_3x3 = pd.DataFrame(results_3x3)
    df_7x7 = pd.DataFrame(results_7x7)
    
    # 计算统计数据
    stats = {
        '3x3': {
            'mean_f1': df_3x3['best_test_f1'].mean(),
            'std_f1': df_3x3['best_test_f1'].std(),
            'min_f1': df_3x3['best_test_f1'].min(),
            'max_f1': df_3x3['best_test_f1'].max(),
            'median_f1': df_3x3['best_test_f1'].median(),
            'mean_epoch': df_3x3['best_epoch'].mean(),
            'n_subjects': len(df_3x3)
        },
        '7x7': {
            'mean_f1': df_7x7['best_test_f1'].mean(),
            'std_f1': df_7x7['best_test_f1'].std(),
            'min_f1': df_7x7['best_test_f1'].min(),
            'max_f1': df_7x7['best_test_f1'].max(),
            'median_f1': df_7x7['best_test_f1'].median(),
            'mean_epoch': df_7x7['best_epoch'].mean(),
            'n_subjects': len(df_7x7)
        }
    }
    
    # 打印统计结果
    print("\n" + "="*60)
    print("Leave-One-Out 交叉验证结果汇总")
    print("="*60)
    
    for patch_size, stat in stats.items():
        print(f"\n{patch_size} Patch 模型:")
        print(f"  测试被试数: {stat['n_subjects']}")
        print(f"  平均 F1 Score: {stat['mean_f1']:.4f} ± {stat['std_f1']:.4f}")
        print(f"  中位数 F1 Score: {stat['median_f1']:.4f}")
        print(f"  最小 F1 Score: {stat['min_f1']:.4f}")
        print(f"  最大 F1 Score: {stat['max_f1']:.4f}")
        print(f"  平均最佳轮数: {stat['mean_epoch']:.1f}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. F1 Score 分布对比
    ax = axes[0, 0]
    data_to_plot = [df_3x3['best_test_f1'].values, df_7x7['best_test_f1'].values]
    bp = ax.boxplot(data_to_plot, labels=['3×3', '7×7'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax.set_ylabel('Test F1 Score')
    ax.set_title('F1 Score Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    # 2. 每个被试的F1 Score
    ax = axes[0, 1]
    subjects = df_3x3['test_subject'].values
    ax.plot(subjects, df_3x3['best_test_f1'].values, 'o-', label='3×3', alpha=0.7)
    ax.plot(subjects, df_7x7['best_test_f1'].values, 's-', label='7×7', alpha=0.7)
    ax.set_xlabel('Test Subject')
    ax.set_ylabel('Test F1 Score')
    ax.set_title('F1 Score by Subject')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 训练vs测试 F1对比
    ax = axes[1, 0]
    ax.scatter(df_3x3['final_train_f1'], df_3x3['best_test_f1'], 
              label='3×3', alpha=0.6, s=50)
    ax.scatter(df_7x7['final_train_f1'], df_7x7['best_test_f1'], 
              label='7×7', alpha=0.6, s=50, marker='s')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # 对角线
    ax.set_xlabel('Train F1 Score')
    ax.set_ylabel('Test F1 Score')
    ax.set_title('Train vs Test Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 收敛速度（最佳epoch分布）
    ax = axes[1, 1]
    bins = np.arange(0, max(df_3x3['best_epoch'].max(), df_7x7['best_epoch'].max()) + 5, 5)
    ax.hist(df_3x3['best_epoch'], bins=bins, alpha=0.5, label='3×3', color='blue')
    ax.hist(df_7x7['best_epoch'], bins=bins, alpha=0.5, label='7×7', color='green')
    ax.set_xlabel('Best Epoch')
    ax.set_ylabel('Count')
    ax.set_title('Convergence Speed Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存详细结果到CSV
    combined_df = pd.concat([
        df_3x3.assign(model='3x3'),
        df_7x7.assign(model='7x7')
    ])
    combined_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # 保存统计摘要
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(output_dir / 'summary_statistics.csv')
    
    print(f"\n结果已保存到: {output_dir}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='分析Leave-one-out结果')
    parser.add_argument('--results_dir', type=str, default='./results_leave_one_out',
                       help='结果目录')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # 加载3×3和7×7的结果
    print("加载3×3模型结果...")
    results_3x3 = load_results(results_dir / '3x3', patch_size=3)
    
    print("加载7×7模型结果...")
    results_7x7 = load_results(results_dir / '7x7', patch_size=7)
    
    if len(results_3x3) == 0 or len(results_7x7) == 0:
        print("警告：未找到足够的结果文件")
        return
    
    # 分析和可视化
    analyze_and_plot(results_3x3, results_7x7, results_dir)

if __name__ == '__main__':
    main()