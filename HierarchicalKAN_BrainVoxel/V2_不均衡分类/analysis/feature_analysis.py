#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征分析模块，负责分析特征重要性和特征组合的效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FIGURES_DIR, EVALUATION
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def analyze_feature_importance(data, labels, feature_names=None, n_estimators=100, verbose=True, plot=True):
    """
    使用随机森林分析特征重要性
    
    参数:
        data: 输入数据
        labels: 类别标签
        feature_names: 特征名称列表
        n_estimators: 随机森林中树的数量
        verbose: 是否打印详细信息
        plot: 是否绘制重要性分布图
        
    返回:
        importance: 特征重要性数组
        indices: 按重要性排序的特征索引
    """
    if verbose:
        logger.info("使用随机森林分析特征重要性...")
    
    # 训练随机森林模型
    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    forest.fit(data, labels)
    
    # 获取特征重要性
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # 如果没有提供特征名称，则使用索引作为名称
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(data.shape[1])]
    
    if verbose:
        logger.info("\n特征重要性排名:")
        for i, idx in enumerate(indices[:20]):  # 只打印前20个
            logger.info(f"#{i+1}: 特征 '{feature_names[idx]}' (索引 {idx}) - 重要性: {importance[idx]:.6f}")
    
    if plot:
        # 绘制特征重要性
        plt.figure(figsize=(12, 6))
        
        # 绘制前30个特征的重要性
        top_n = min(30, len(indices))
        plt.barh(range(top_n), importance[indices[:top_n]], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        
        # 保存图表
        save_path = os.path.join(FIGURES_DIR, 'feature_importance.png')
        plt.savefig(save_path, dpi=300)
        if verbose:
            logger.info(f"特征重要性图表已保存至: {save_path}")
        
        plt.close()
    
    return importance, indices

def analyze_feature_group_combinations(feature_groups, labels, group_names=None, verbose=True, plot=True):
    """
    分析不同特征组合的分类性能（并行版本）
    
    参数：
        feature_groups: 特征组字典
        labels: 标签
        group_names: 特征组名称列表
        verbose: 是否打印详细信息
        plot: 是否绘制图表
        
    返回：
        results: 不同组合的性能结果列表
    """
    from joblib import Parallel, delayed
    from itertools import combinations
    
    if group_names is None:
        group_names = list(feature_groups.keys())
    
    if verbose:
        logger.info("分析不同特征组合的分类性能...")
    
    # 定义评估单个特征组合的函数
    def evaluate_combination(combination):
        # 获取组合名称
        if isinstance(combination, str):
            combination = [combination]  # 单个特征组
        
        combination_str = ' + '.join(combination)
        
        if verbose:
            logger.info(f"测试特征组合: {combination_str}")
        
        # 合并特征
        features_list = [feature_groups[group] for group in combination]
        X = np.hstack(features_list)
        
        # 使用随机森林评估性能
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        cv_folds = EVALUATION['classification']['cv_folds']
        
        # 交叉验证评估
        acc_scores = cross_val_score(rf, X, labels, cv=cv_folds, scoring='accuracy')
        balanced_acc_scores = cross_val_score(rf, X, labels, cv=cv_folds, scoring='balanced_accuracy')
        f1_scores = cross_val_score(rf, X, labels, cv=cv_folds, scoring='f1_weighted')
        
        # 计算Kappa系数(需要自定义评分函数)
        def cohen_kappa_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return cohen_kappa_score(y, y_pred)
        
        kappa_scores = cross_val_score(rf, X, labels, cv=cv_folds, scoring=cohen_kappa_scorer)
        
        if verbose:
            logger.info(f"  准确率: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
            logger.info(f"  平衡准确率: {balanced_acc_scores.mean():.4f} ± {balanced_acc_scores.std():.4f}")
            logger.info(f"  加权F1分数: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
            logger.info(f"  Cohen's Kappa: {kappa_scores.mean():.4f} ± {kappa_scores.std():.4f}")
        
        # 返回结果
        return {
            'combination': list(combination),
            'accuracy': acc_scores.mean(),
            'accuracy_std': acc_scores.std(),
            'balanced_accuracy': balanced_acc_scores.mean(),
            'balanced_accuracy_std': balanced_acc_scores.std(),
            'f1_weighted': f1_scores.mean(),
            'f1_weighted_std': f1_scores.std(),
            'kappa': kappa_scores.mean(),
            'kappa_std': kappa_scores.std()
        }
    
    # 生成所有可能的组合
    all_combinations = []
    
    # 单个特征组
    all_combinations.extend(group_names)
    
    # 两个特征组的组合
    if len(group_names) >= 2:
        all_combinations.extend(combinations(group_names, 2))
    
    # 三个特征组的组合
    if len(group_names) >= 3:
        all_combinations.extend(combinations(group_names, 3))
    
    # 所有特征组
    if len(group_names) >= 4:
        all_combinations.append(tuple(group_names))
    
    # 并行评估所有组合
    results = Parallel(n_jobs=-1)(delayed(evaluate_combination)(combo) for combo in all_combinations)
    
    # 对结果排序（使用平衡准确率作为主要指标）
    results.sort(key=lambda x: x['balanced_accuracy'], reverse=True)
    
    if verbose:
        logger.info("\n所有特征组合的性能排名:")
        for i, result in enumerate(results):
            combination_str = ' + '.join(result['combination'])
            logger.info(f"{i+1}. {combination_str}:")
            logger.info(f"   平衡准确率: {result['balanced_accuracy']:.4f} ± {result['balanced_accuracy_std']:.4f}")
            logger.info(f"   加权F1分数: {result['f1_weighted']:.4f} ± {result['f1_weighted_std']:.4f}")
            logger.info(f"   Cohen's Kappa: {result['kappa']:.4f} ± {result['kappa_std']:.4f}")
    
    if plot:
        # 可视化结果
        plt.figure(figsize=(14, 10))
        
        # 准备数据
        combinations = [' + '.join(r['combination']) for r in results]
        balanced_accs = [r['balanced_accuracy'] for r in results]
        balanced_acc_stds = [r['balanced_accuracy_std'] for r in results]
        f1_scores = [r['f1_weighted'] for r in results]
        f1_stds = [r['f1_weighted_std'] for r in results]
        kappas = [r['kappa'] for r in results]
        kappa_stds = [r['kappa_std'] for r in results]
        
        # 绘制平衡准确率
        plt.subplot(3, 1, 1)
        plt.bar(combinations, balanced_accs, yerr=balanced_acc_stds, alpha=0.8)
        plt.axhline(y=1/len(np.unique(labels)), color='r', linestyle='--', 
                label=f'Random Guessing ({1/len(np.unique(labels)):.4f})')
        plt.xlabel('Feature Combinations')
        plt.ylabel('Balanced Accuracy')
        plt.title('Classification Performance: Balanced Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.legend()
        
        # 绘制加权F1分数
        plt.subplot(3, 1, 2)
        plt.bar(combinations, f1_scores, yerr=f1_stds, alpha=0.8)
        plt.xlabel('Feature Combinations')
        plt.ylabel('Weighted F1 Score')
        plt.title('Classification Performance: Weighted F1 Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        # 绘制Kappa系数
        plt.subplot(3, 1, 3)
        plt.bar(combinations, kappas, yerr=kappa_stds, alpha=0.8)
        plt.xlabel('Feature Combinations')
        plt.ylabel("Cohen's Kappa")
        plt.title("Classification Performance: Cohen's Kappa")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(FIGURES_DIR, 'feature_combination_performance.png')
        plt.savefig(save_path, dpi=300)
        if verbose:
            logger.info(f"特征组合性能图表已保存至: {save_path}")
        
        plt.close()
    
    return results
def analyze_class_separability(feature_groups, labels, verbose=True, use_sampling=False, sample_ratio=0.1):
    """
    分析特征空间中的类别可分性（并行处理版本）
    
    参数：
        feature_groups: 处理后的特征组
        labels: 类别标签
        verbose: 是否打印详细信息
        use_sampling: 是否使用采样来加速分析
        sample_ratio: 采样比例
        
    返回：
        separability_scores: 各特征组的可分性评分
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedShuffleSplit
    import time
    from joblib import Parallel, delayed
    
    if verbose:
        logger.info("分析特征空间中的类别可分性...")
    
    separability_scores = {}
    
    # 如果使用采样，创建一个采样子集
    if use_sampling:
        if verbose:
            logger.info(f"使用{sample_ratio*100:.1f}%的数据进行分析")
        
        from sklearn.model_selection import train_test_split
        
        # 使用分层采样保持类别分布
        indices = np.arange(len(labels))
        _, sampled_indices, _, sampled_labels = train_test_split(
            indices, labels, test_size=sample_ratio, stratify=labels, random_state=42
        )
        
        if verbose:
            logger.info(f"原始数据: {len(labels)} 样本")
            logger.info(f"采样后: {len(sampled_labels)} 样本")
            unique, counts = np.unique(sampled_labels, return_counts=True)
            logger.info(f"采样后类别分布: {len(unique)} 个唯一类别")
        
        # 对每个特征组进行采样
        sampled_groups = {}
        for group_name, group_data in feature_groups.items():
            sampled_groups[group_name] = group_data[sampled_indices]
        
        # 使用采样后的数据
        analysis_groups = sampled_groups
        analysis_labels = sampled_labels
    else:
        # 使用全部数据
        analysis_groups = feature_groups
        analysis_labels = labels
    
    # 使用交叉验证分割器
    cv_splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    
    # 定义分类器
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', C=1, probability=True, class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    }
    
    # 定义单个分类器评估任务
    def evaluate_classifier(group_name, group_data, clf_name, clf):
        start_time = time.time()
        if verbose:
            print(f"  计算 {group_name} - {clf_name} 分类器性能...", flush=True)
        
        # 使用自定义的交叉验证
        acc_scores = []
        balanced_acc_scores = []
        f1_scores = []
        kappa_scores = []
        
        for train_idx, test_idx in cv_splitter.split(group_data, analysis_labels):
            X_train, X_test = group_data[train_idx], group_data[test_idx]
            y_train, y_test = analysis_labels[train_idx], analysis_labels[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            acc_scores.append(accuracy_score(y_test, y_pred))
            balanced_acc_scores.append(balanced_accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            kappa_scores.append(cohen_kappa_score(y_test, y_pred))
        
        # 计算平均分数
        mean_acc = np.mean(acc_scores)
        std_acc = np.std(acc_scores)
        mean_balanced_acc = np.mean(balanced_acc_scores)
        std_balanced_acc = np.std(balanced_acc_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_kappa = np.mean(kappa_scores)
        std_kappa = np.std(kappa_scores)
        
        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"  {group_name} - {clf_name} 完成! ({elapsed:.1f}秒)")
            logger.info(f"    准确率: {mean_acc:.4f} ± {std_acc:.4f}")
            logger.info(f"    平衡准确率: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")
            logger.info(f"    加权F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
            logger.info(f"    Cohen's Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
        
        return group_name, clf_name, {
            'accuracy': {'mean': mean_acc, 'std': std_acc},
            'balanced_accuracy': {'mean': mean_balanced_acc, 'std': std_balanced_acc},
            'f1_weighted': {'mean': mean_f1, 'std': std_f1},
            'kappa': {'mean': mean_kappa, 'std': std_kappa}
        }
    
    # 准备并行任务
    tasks = []
    for group_name, group_data in analysis_groups.items():
        for clf_name, clf in classifiers.items():
            tasks.append((group_name, group_data, clf_name, clf))
    
    # 并行执行所有评估任务
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_classifier)(group_name, group_data, clf_name, clf) 
        for group_name, group_data, clf_name, clf in tasks
    )
    
    # 处理结果
    # 整理结果为嵌套字典
    for group_name, clf_name, metrics in results:
        if group_name not in separability_scores:
            separability_scores[group_name] = {
                'classifiers': {}
            }
        separability_scores[group_name]['classifiers'][clf_name] = metrics
    
    # 计算随机猜测基线
    n_classes = len(np.unique(analysis_labels))
    random_guess_acc = 1.0 / n_classes
    
    # 计算每个特征组的总体可分性得分
    for group_name in separability_scores:
        # 计算平均得分作为总体可分性评分
        mean_accuracy = np.mean([separability_scores[group_name]['classifiers'][clf_name]['accuracy']['mean'] 
                              for clf_name in classifiers])
        mean_balanced_accuracy = np.mean([separability_scores[group_name]['classifiers'][clf_name]['balanced_accuracy']['mean'] 
                                       for clf_name in classifiers])
        mean_f1 = np.mean([separability_scores[group_name]['classifiers'][clf_name]['f1_weighted']['mean'] 
                        for clf_name in classifiers])
        mean_kappa = np.mean([separability_scores[group_name]['classifiers'][clf_name]['kappa']['mean'] 
                           for clf_name in classifiers])
        
        separability_scores[group_name]['overall'] = mean_balanced_accuracy  # 使用平衡准确率作为总体评分
        separability_scores[group_name]['accuracy'] = mean_accuracy
        separability_scores[group_name]['balanced_accuracy'] = mean_balanced_accuracy
        separability_scores[group_name]['f1_weighted'] = mean_f1
        separability_scores[group_name]['kappa'] = mean_kappa
        separability_scores[group_name]['random_guess'] = random_guess_acc
        
        if verbose:
            logger.info(f"\n{group_name} 特征组总体可分性评分: {mean_balanced_accuracy:.4f}")
            logger.info(f"  相对于随机猜测 ({random_guess_acc:.4f}) 的提升: {(mean_balanced_accuracy/random_guess_acc - 1)*100:.1f}%")
    
    # 可视化结果
    plt.figure(figsize=(14, 10))
    
    # 准备数据
    groups = list(separability_scores.keys())
    clf_names = list(classifiers.keys())
    metrics = ['balanced_accuracy', 'f1_weighted', 'kappa']
    metric_names = {'balanced_accuracy': 'Balanced Accuracy', 
                   'f1_weighted': 'Weighted F1 Score', 
                   'kappa': "Cohen's Kappa"}
    
    # 设置子图布局
    for m_idx, metric in enumerate(metrics):
        plt.subplot(3, 1, m_idx+1)
        
        x = np.arange(len(groups))
        width = 0.25
        offsets = np.linspace(-width, width, len(clf_names))
        
        # 绘制各分类器的性能
        for i, clf_name in enumerate(clf_names):
            means = [separability_scores[g]['classifiers'][clf_name][metric]['mean'] for g in groups]
            stds = [separability_scores[g]['classifiers'][clf_name][metric]['std'] for g in groups]
            
            plt.bar(x + offsets[i], means, width, label=clf_name, yerr=stds)
        
        # 如果是平衡准确率，绘制随机猜测基线
        if metric == 'balanced_accuracy':
            plt.axhline(y=random_guess_acc, color='r', linestyle='--', 
                      label=f'Random Guessing ({random_guess_acc:.4f})')
        
        plt.xlabel('Feature Groups')
        plt.ylabel(metric_names[metric])
        plt.title(f'Class Separability: {metric_names[metric]}')
        plt.xticks(x, groups)
        plt.legend()
        plt.grid(axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, 'class_separability.png')
    plt.savefig(save_path, dpi=300)
    if verbose:
        logger.info(f"类别可分性图表已保存至: {save_path}")
    
    plt.close()
    
    return separability_scores


if __name__ == "__main__":
    # 测试特征分析功能
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_multiclass_data_from_dirs, define_big_classes, map_to_big_classes
    from data.preprocessing import preprocess_feature_groups
    
    # 加载验证集数据
    dataset = load_multiclass_data_from_dirs(subset='val')
    val_data, val_labels = dataset['val_samples'], dataset['val_labels']
    
    # 获取大类标签
    fine_to_big, big_to_fine, big_class_names = define_big_classes()
    big_labels = map_to_big_classes(val_labels, fine_to_big)
    
    # 预处理特征
    processed_groups, _ = preprocess_feature_groups(val_data, big_labels)
    
    # 测试特征组合分析
    combination_results = analyze_feature_group_combinations(processed_groups, big_labels)
    
    # 测试类别可分性分析
    separability_scores = analyze_class_separability(processed_groups, big_labels)
    
    print("特征分析测试完成")