# -*- coding: utf-8 -*-

"""
特征分析模块
用于分析MRI数据的特征重要性和特征选择
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import logging
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_analysis')

# 检测是否可以使用GPU
try:
    import cupy as cp
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAS_GPU = True
    logger.info("GPU加速可用")
except ImportError:
    HAS_GPU = False
    logger.warning("未检测到GPU加速库，将使用CPU进行计算")






def evaluate_feature_selection_methods(data, labels, feature_groups=None, methods=None, 
                                     k_values=None, use_gpu=False, save_dir=None):
    """
    评估不同特征选择方法的性能
    
    参数:
        data: 输入数据
        labels: 类别标签
        feature_groups: 特征组字典
        methods: 特征选择方法列表，可选值: 'rf', 'mi', 'chi2'
        k_values: 特征数量列表
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
    
    返回:
        selection_dict: 特征选择评估结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if methods is None:
        methods = ['rf', 'mi']  # 默认使用随机森林和互信息
    
    if k_values is None:
        # 根据特征数量自动设置k值
        max_features = max([len(indices) for indices in feature_groups.values()])
        k_values = [10, 20, 50, 100]
        if max_features > 100:
            k_values.extend([200, 500])
        if max_features > 500:
            k_values.append(1000)
        # 确保k值不超过最大特征数
        k_values = [k for k in k_values if k <= max_features]
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    selection_dict = {}
    
    # 定义评估方法
    def evaluate_features(selected_data):
        # 使用随机森林和交叉验证评估特征选择结果
        if use_gpu and HAS_GPU:
            try:
                # 使用GPU版本
                rf = cuRF(n_estimators=100, random_state=42)
                scores = cross_val_score(rf, selected_data, labels, cv=5, scoring='balanced_accuracy')
                
                # 释放GPU内存
                del rf
                cp.get_default_memory_pool().free_all_blocks()
                
                return scores.mean(), scores.std()
            except Exception as e:
                logger.error(f"GPU评估失败: {str(e)}")
                logger.info("回退到CPU评估...")
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                scores = cross_val_score(rf, selected_data, labels, cv=5, scoring='balanced_accuracy')
                return scores.mean(), scores.std()
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            scores = cross_val_score(rf, selected_data, labels, cv=5, scoring='balanced_accuracy')
            return scores.mean(), scores.std()
    
    # 特征选择方法实现
    def select_features(method, data, labels, k):
        if method == 'rf':
            # 使用随机森林重要性选择特征
            if use_gpu and HAS_GPU:
                try:
                    rf = cuRF(n_estimators=100, random_state=42)
                    rf.fit(data, labels)
                    importances = rf.feature_importances_
                    
                    del rf
                    cp.get_default_memory_pool().free_all_blocks()
                    
                except Exception as e:
                    logger.error(f"GPU随机森林失败: {str(e)}")
                    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    rf.fit(data, labels)
                    importances = rf.feature_importances_
            else:
                rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                rf.fit(data, labels)
                importances = rf.feature_importances_
            
            # 选择top-k特征
            indices = np.argsort(importances)[::-1][:k]
            return data[:, indices], indices
        
        elif method == 'mi':
            # 使用互信息选择特征
            mi_scores = mutual_info_classif(data, labels, random_state=42)
            indices = np.argsort(mi_scores)[::-1][:k]
            return data[:, indices], indices
        
        elif method == 'chi2':
            # 使用卡方检验选择特征
            # 首先确保数据是非负的
            data_nonneg = data.copy()
            data_nonneg = data_nonneg - data_nonneg.min(axis=0) + 1e-8
            
            # 应用卡方检验
            chi2_scores, _ = chi2(data_nonneg, labels)
            indices = np.argsort(chi2_scores)[::-1][:k]
            return data[:, indices], indices
        
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
    
    # 为每个特征组评估特征选择方法
    for group_name, indices in tqdm(feature_groups.items(), desc="评估特征选择方法"):
        logger.info(f"评估特征组 {group_name} 的特征选择方法")
        
        # 提取特征组数据
        group_data = data[:, indices]
        
        # 计算基线性能（使用所有特征）
        baseline_acc, baseline_std = evaluate_features(group_data)
        logger.info(f"  基线性能 (所有 {len(indices)} 特征): {baseline_acc:.4f} ± {baseline_std:.4f}")
        
        # 初始化结果
        method_results = {
            'baseline': {
                'accuracy': baseline_acc,
                'std': baseline_std,
                'n_features': len(indices)
            }
        }
        
        # 评估每种特征选择方法
        for method in methods:
            logger.info(f"  评估方法: {method}")
            k_results = {}
            
            # 评估不同的特征数量
            for k in k_values:
                if k >= len(indices):
                    continue  # 跳过超过特征总数的k值
                
                logger.info(f"    特征数量 k={k}")
                selected_data, selected_indices = select_features(method, group_data, labels, k)
                acc, std = evaluate_features(selected_data)
                
                logger.info(f"      准确率: {acc:.4f} ± {std:.4f}")
                
                k_results[k] = {
                    'accuracy': acc,
                    'std': std,
                    'indices': selected_indices,
                    'relative_improvement': (acc - baseline_acc) / baseline_acc
                }
            
            method_results[method] = k_results
        
        selection_dict[group_name] = method_results
        
        # 可视化特征选择结果
        if save_dir and method_results:
            plt.figure(figsize=(10, 6))
            
            # 绘制不同方法和特征数量的性能
            for method in methods:
                if method not in method_results:
                    continue
                
                k_list = sorted(method_results[method].keys())
                accuracy = [method_results[method][k]['accuracy'] for k in k_list]
                
                plt.plot(k_list, accuracy, 'o-', label=method)
            
            # 绘制基线
            plt.axhline(y=method_results['baseline']['accuracy'], color='r', linestyle='--', 
                       label=f'Baseline ({method_results["baseline"]["n_features"]} features)')
            
            plt.xlabel('Number of Selected Features')
            plt.ylabel('Balanced Accuracy')
            plt.title(f'Feature Selection Performance for {group_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_feature_selection_performance.png"), dpi=300)
            plt.close()
            
            # 保存特征选择结果
            selection_summary = {
                'group': group_name,
                'baseline_accuracy': method_results['baseline']['accuracy'],
                'baseline_std': method_results['baseline']['std'],
                'total_features': method_results['baseline']['n_features']
            }
            
            for method in methods:
                if method not in method_results:
                    continue
                
                # 找出最佳k值
                best_k = max(method_results[method].keys(), key=lambda k: method_results[method][k]['accuracy'])
                best_acc = method_results[method][best_k]['accuracy']
                
                selection_summary[f'{method}_best_k'] = best_k
                selection_summary[f'{method}_best_accuracy'] = best_acc
                selection_summary[f'{method}_improvement'] = (best_acc - method_results['baseline']['accuracy']) / method_results['baseline']['accuracy']
            
            # 保存摘要结果
            pd.DataFrame([selection_summary]).to_csv(
                os.path.join(save_dir, f"{group_name}_feature_selection_summary.csv"), index=False)
    
    # 比较不同特征组和方法的结果
    if len(feature_groups) > 1 and save_dir:
        # 整理比较数据
        comparison_data = []
        
        for group_name in feature_groups:
            group_result = selection_dict[group_name]
            
            for method in methods:
                if method not in group_result:
                    continue
                
                # 找出最佳k值
                if not group_result[method]:  # 检查是否为空字典
                    continue
                    
                best_k = max(group_result[method].keys(), key=lambda k: group_result[method][k]['accuracy'])
                best_acc = group_result[method][best_k]['accuracy']
                baseline_acc = group_result['baseline']['accuracy']
                
                comparison_data.append({
                    'group': group_name,
                    'method': method,
                    'best_k': best_k,
                    'best_accuracy': best_acc,
                    'baseline_accuracy': baseline_acc,
                    'improvement': (best_acc - baseline_acc) / baseline_acc
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # 绘制比较图
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            
            # 使用seaborn绘制方法和特征组的比较
            df_pivot = comparison_df.pivot(index='group', columns='method', values='best_accuracy')
            sns.heatmap(df_pivot, annot=True, cmap="YlGnBu")
            plt.title('Best Accuracy by Group and Method')
            
            plt.subplot(1, 2, 2)
            df_pivot = comparison_df.pivot(index='group', columns='method', values='improvement')
            sns.heatmap(df_pivot, annot=True, cmap="RdYlGn", center=0)
            plt.title('Relative Improvement over Baseline')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "feature_selection_comparison.png"), dpi=300)
            plt.close()
            
            # 保存比较结果
            comparison_df.to_csv(os.path.join(save_dir, "feature_selection_comparison.csv"), index=False)
    
    return selection_dict

def analyze_feature_combinations(data, labels, feature_groups=None, top_k=50, 
                               use_gpu=False, save_dir=None):
    """
    分析不同特征组合的性能
    
    参数:
        data: 输入数据
        labels: 类别标签
        feature_groups: 特征组字典
        top_k: 每个特征组选取的top-k重要特征
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
    
    返回:
        combinations_dict: 特征组合分析结果字典
    """
    if feature_groups is None or len(feature_groups) < 2:
        logger.warning("至少需要两个特征组才能分析组合")
        return {}
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logger.info(f"分析特征组合性能，每组选取top-{top_k}特征")
    
    # 对每个特征组选择top-k重要特征
    selected_features = {}
    for group_name, indices in tqdm(feature_groups.items(), desc="选择每组top特征"):
        logger.info(f"为特征组 {group_name} 选择top-{top_k}特征")
        
        # 提取特征组数据
        group_data = data[:, indices]
        
        # 限制k不超过特征总数
        actual_k = min(top_k, len(indices))
        
        # 使用随机森林选择特征
        if use_gpu and HAS_GPU:
            try:
                rf = cuRF(n_estimators=100, random_state=42)
                rf.fit(group_data, labels)
                importances = rf.feature_importances_
                
                del rf
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                logger.error(f"GPU随机森林失败: {str(e)}")
                rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                rf.fit(group_data, labels)
                importances = rf.feature_importances_
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(group_data, labels)
            importances = rf.feature_importances_
        
        # 选择top-k特征
        top_indices = np.argsort(importances)[::-1][:actual_k]
        selected_indices = [indices[i] for i in top_indices]
        selected_features[group_name] = selected_indices
        
        logger.info(f"  选择了 {len(selected_indices)} 个特征")
    
    # 定义评估方法
    def evaluate_combination(combined_data):
        if use_gpu and HAS_GPU:
            try:
                rf = cuRF(n_estimators=100, random_state=42)
                scores = cross_val_score(rf, combined_data, labels, cv=5, scoring='balanced_accuracy')
                
                del rf
                cp.get_default_memory_pool().free_all_blocks()
                
                return scores.mean(), scores.std()
            except Exception as e:
                logger.error(f"GPU评估失败: {str(e)}")
                rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                scores = cross_val_score(rf, combined_data, labels, cv=5, scoring='balanced_accuracy')
                return scores.mean(), scores.std()
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            scores = cross_val_score(rf, combined_data, labels, cv=5, scoring='balanced_accuracy')
            return scores.mean(), scores.std()
    
    # 生成和评估所有可能的组合
    import itertools
    group_names = list(feature_groups.keys())
    combinations_dict = {}
    
    # 单个特征组的性能
    for group in group_names:
        logger.info(f"评估单个特征组: {group}")
        selected_data = data[:, selected_features[group]]
        acc, std = evaluate_combination(selected_data)
        logger.info(f"  准确率: {acc:.4f} ± {std:.4f}")
        
        combinations_dict[group] = {
            'accuracy': acc,
            'std': std,
            'n_features': len(selected_features[group])
        }
    
    # 两两组合
    for i, j in itertools.combinations(range(len(group_names)), 2):
        group_i = group_names[i]
        group_j = group_names[j]
        combo_name = f"{group_i}+{group_j}"
        
        logger.info(f"评估两两组合: {combo_name}")
        
        # 组合特征
        combined_indices = selected_features[group_i] + selected_features[group_j]
        selected_data = data[:, combined_indices]
        
        acc, std = evaluate_combination(selected_data)
        logger.info(f"  准确率: {acc:.4f} ± {std:.4f}")
        
        combinations_dict[combo_name] = {
            'accuracy': acc,
            'std': std,
            'n_features': len(combined_indices),
            'groups': [group_i, group_j]
        }
    
    # 如果有多于2个特征组，评估3个组合
    if len(group_names) > 2:
        for i, j, k in itertools.combinations(range(len(group_names)), 3):
            group_i = group_names[i]
            group_j = group_names[j]
            group_k = group_names[k]
            combo_name = f"{group_i}+{group_j}+{group_k}"
            
            logger.info(f"评估三个组合: {combo_name}")
            
            # 组合特征
            combined_indices = selected_features[group_i] + selected_features[group_j] + selected_features[group_k]
            selected_data = data[:, combined_indices]
            
            acc, std = evaluate_combination(selected_data)
            logger.info(f"  准确率: {acc:.4f} ± {std:.4f}")
            
            combinations_dict[combo_name] = {
                'accuracy': acc,
                'std': std,
                'n_features': len(combined_indices),
                'groups': [group_i, group_j, group_k]
            }
    
    # 所有特征组的组合
    if len(group_names) > 3:
        combo_name = "+".join(group_names)
        logger.info(f"评估所有特征组组合: {combo_name}")
        
        all_indices = []
        for group in group_names:
            all_indices.extend(selected_features[group])
        
        selected_data = data[:, all_indices]
        acc, std = evaluate_combination(selected_data)
        logger.info(f"  准确率: {acc:.4f} ± {std:.4f}")
        
        combinations_dict[combo_name] = {
            'accuracy': acc,
            'std': std,
            'n_features': len(all_indices),
            'groups': group_names
        }
    
    # 找出最佳组合
    best_combo = max(combinations_dict.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"最佳特征组合: {best_combo[0]}, 准确率: {best_combo[1]['accuracy']:.4f}")
    
    # 可视化组合性能
    if save_dir:
        # 按准确率排序
        sorted_combos = sorted(combinations_dict.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        combo_names = [combo[0] for combo in sorted_combos]
        accuracies = [combo[1]['accuracy'] for combo in sorted_combos]
        stds = [combo[1]['std'] for combo in sorted_combos]
        n_features = [combo[1]['n_features'] for combo in sorted_combos]
        
        # 绘制准确率比较
        plt.figure(figsize=(12, 6))
        plt.bar(combo_names, accuracies, yerr=stds)
        plt.xlabel('Feature Combination')
        plt.ylabel('Balanced Accuracy')
        plt.title('Feature Combination Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_combination_performance.png"), dpi=300)
        plt.close()
        
        # 绘制特征数量和准确率的关系
        plt.figure(figsize=(10, 6))
        for i, (name, result) in enumerate(sorted_combos):
            plt.scatter(result['n_features'], result['accuracy'], label=name)
            plt.text(result['n_features'], result['accuracy'], name, fontsize=9)
        
        plt.xlabel('Number of Features')
        plt.ylabel('Balanced Accuracy')
        plt.title('Accuracy vs. Number of Features')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "accuracy_vs_features.png"), dpi=300)
        plt.close()
        
        # 保存组合性能结果
        combo_results = []
        for name, result in combinations_dict.items():
            combo_results.append({
                'combination': name,
                'accuracy': result['accuracy'],
                'std': result['std'],
                'n_features': result['n_features'],
                'groups': result.get('groups', [name])
            })
        
        combo_df = pd.DataFrame(combo_results)
        combo_df = combo_df.sort_values('accuracy', ascending=False)
        combo_df.to_csv(os.path.join(save_dir, "feature_combination_results.csv"), index=False)
    
    return combinations_dict

if __name__ == "__main__":
    # 测试特征分析功能
    print("特征分析模块测试")
    
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.randn(1000, 100)
    test_labels = np.random.randint(0, 5, 1000)
    
    test_groups = {
        'group1': list(range(0, 30)),
        'group2': list(range(30, 70)),
        'group3': list(range(70, 100))
    }
    
    # 创建测试输出目录
    test_save_dir = "test_results/feature_analysis"
    os.makedirs(test_save_dir, exist_ok=True)
    
    # 测试特征重要性分析
    importance = analyze_feature_importance(test_data, test_labels, test_groups, 
                                          use_gpu=False, save_dir=test_save_dir)
    
    # 测试特征选择方法评估
    selection = evaluate_feature_selection_methods(test_data, test_labels, test_groups,
                                                methods=['rf', 'mi'], k_values=[5, 10, 20],
                                                use_gpu=False, save_dir=test_save_dir)
    
    # 测试特征组合分析
    combinations = analyze_feature_combinations(test_data, test_labels, test_groups, top_k=10,
                                              use_gpu=False, save_dir=test_save_dir)
    
    print("测试完成")#!/usr/bin/env python




def analyze_feature_importance(data, labels, feature_groups=None, n_estimators=100, 
                            use_gpu=False, save_dir=None):
    """
    使用随机森林分析特征重要性
    
    参数:
        data: 输入数据
        labels: 类别标签
        feature_groups: 特征组字典
        n_estimators: 随机森林中树的数量
        use_gpu: 是否使用GPU加速
        save_dir: 结果保存目录
    
    返回:
        importance_dict: 特征重要性分析结果字典
    """
    if feature_groups is None:
        feature_groups = {
            'all_features': list(range(data.shape[1]))
        }
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    importance_dict = {}
    
    for group_name, indices in tqdm(feature_groups.items(), desc="分析特征重要性"):
        logger.info(f"分析特征组 {group_name} 的特征重要性")
        
        # 提取特征组数据
        group_data = data[:, indices]
        # 训练随机森林模型
        start_time = time.time()
        if use_gpu and HAS_GPU:
            try:
                # 使用GPU版本的随机森林
                forest = cuRF(n_estimators=n_estimators, random_state=42)
                forest.fit(group_data, labels)
                importances = forest.feature_importances_
                
                # cuML的feature_importances_属性返回numpy数组，无需转换
                
                # 释放GPU内存
                del forest
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                logger.error(f"GPU处理失败: {str(e)}")
                logger.info("回退到CPU处理...")
                
                forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
                forest.fit(group_data, labels)
                importances = forest.feature_importances_
        else:
            forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
            forest.fit(group_data, labels)
            importances = forest.feature_importances_
        
        elapsed = time.time() - start_time
        logger.info(f"  随机森林训练完成，耗时 {elapsed:.2f} 秒")
        
        # 排序并分析重要性
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_feature_indices = [indices[i] for i in range(len(indices))]
        cumulative_importance = np.cumsum(sorted_importances)
        
        # 计算统计信息
        n_features_90 = np.argmax(cumulative_importance >= 0.9) + 1
        n_features_95 = np.argmax(cumulative_importance >= 0.95) + 1
        n_features_99 = np.argmax(cumulative_importance >= 0.99) + 1
        
        logger.info(f"  覆盖90%重要性需要的特征数: {n_features_90} ({n_features_90/len(indices)*100:.1f}%)")
        logger.info(f"  覆盖95%重要性需要的特征数: {n_features_95} ({n_features_95/len(indices)*100:.1f}%)")
        logger.info(f"  覆盖99%重要性需要的特征数: {n_features_99} ({n_features_99/len(indices)*100:.1f}%)")
        
        # 保存结果
        importance_dict[group_name] = {
            'importances': importances,
            'sorted_indices': indices,
            'sorted_importances': sorted_importances,
            'cumulative_importance': cumulative_importance,
            'n_features_90': n_features_90,
            'n_features_95': n_features_95,
            'n_features_99': n_features_99
        }
        
        # 可视化特征重要性
        if save_dir:
            # 绘制特征重要性条形图
            plt.figure(figsize=(14, 8))
            
            # 只绘制前30个特征
            n_top = min(30, len(indices))
            plt.barh(range(n_top), sorted_importances[:n_top])
            plt.yticks(range(n_top), [f"Feature {indices[i]}" for i in range(n_top)])
            plt.xlabel("Feature Importance")
            plt.title(f"Top {n_top} Feature Importance for {group_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_feature_importance.png"), dpi=300)
            plt.close()
            
            # 绘制累积重要性曲线
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(indices)+1), cumulative_importance)
            plt.axhline(y=0.9, color='r', linestyle='--', label='90% Importance')
            plt.axhline(y=0.95, color='g', linestyle='--', label='95% Importance')
            plt.axhline(y=0.99, color='b', linestyle='--', label='99% Importance')
            plt.axvline(x=n_features_90, color='r', linestyle=':')
            plt.axvline(x=n_features_95, color='g', linestyle=':')
            plt.axvline(x=n_features_99, color='b', linestyle=':')
            plt.xlabel("Number of Features")
            plt.ylabel("Cumulative Importance")
            plt.title(f"Cumulative Feature Importance for {group_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{group_name}_cumulative_importance.png"), dpi=300)
            plt.close()
            
            # 保存特征重要性排名到CSV
            importance_df = pd.DataFrame({
                'feature_idx': [indices[i] for i in range(len(indices))],
                'importance': sorted_importances,
                'cumulative_importance': cumulative_importance
            })
            importance_df.to_csv(os.path.join(save_dir, f"{group_name}_feature_importance.csv"), index=False)
    
    # 比较不同特征组的重要性分布
    if len(feature_groups) > 1 and save_dir:
        plt.figure(figsize=(12, 6))
        
        # 绘制每个特征组所需的特征比例
        groups = list(feature_groups.keys())
        n90_ratios = [importance_dict[g]['n_features_90'] / len(feature_groups[g]) for g in groups]
        n95_ratios = [importance_dict[g]['n_features_95'] / len(feature_groups[g]) for g in groups]
        n99_ratios = [importance_dict[g]['n_features_99'] / len(feature_groups[g]) for g in groups]
        
        x = np.arange(len(groups))
        width = 0.25
        
        plt.bar(x - width, n90_ratios, width, label='90% Importance')
        plt.bar(x, n95_ratios, width, label='95% Importance')
        plt.bar(x + width, n99_ratios, width, label='99% Importance')
        
        plt.xlabel('Feature Group')
        plt.ylabel('Proportion of Features Needed')
        plt.title('Feature Importance Distribution Across Groups')
        plt.xticks(x, groups)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_group_importance_comparison.png"), dpi=300)
        plt.close()
        
        # 保存比较结果
        comparison_df = pd.DataFrame({
            'group': groups,
            'total_features': [len(feature_groups[g]) for g in groups],
            'n_features_90': [importance_dict[g]['n_features_90'] for g in groups],
            'n_features_95': [importance_dict[g]['n_features_95'] for g in groups],
            'n_features_99': [importance_dict[g]['n_features_99'] for g in groups],
            'ratio_90': n90_ratios,
            'ratio_95': n95_ratios,
            'ratio_99': n99_ratios
        })
        comparison_df.to_csv(os.path.join(save_dir, "feature_group_importance_comparison.csv"), index=False)
    
    return importance_dict
