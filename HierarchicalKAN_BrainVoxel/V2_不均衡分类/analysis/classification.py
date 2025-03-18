#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分类性能评估模块，用于评估大类分类效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import EVALUATION, FIGURES_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def compare_classification_models(X, y, classifiers=None, cv=5, verbose=True, plot=True, save_name=None):
    """
    比较不同分类模型的性能（并行版本）
    
    参数:
        X: 特征数据
        y: 目标标签
        classifiers: 分类器字典，默认为None（使用预定义的分类器）
        cv: 交叉验证折数
        verbose: 是否打印详细信息
        plot: 是否绘制性能比较图
        save_name: 保存文件名(不含扩展名)
        
    返回:
        results: 各分类器的性能评估结果字典
    """
    from joblib import Parallel, delayed
    
    if classifiers is None:
        classifiers = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', C=1, probability=True, class_weight='balanced'),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        }
    
    if verbose:
        logger.info(f"比较{len(classifiers)}种分类模型的性能...")
    
    # 使用层化K折交叉验证
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 定义单个分类器评估任务
    def evaluate_classifier(clf_name, clf):
        if verbose:
            logger.info(f"\n评估 {clf_name} 分类器...")
        
        # 评估指标
        metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1_weighted': [],
            'f1_macro': [],
            'kappa': []
        }
        
        # 每个类别的单独指标
        class_metrics = {}
        for class_idx in np.unique(y):
            class_metrics[int(class_idx)] = {'precision': [], 'recall': [], 'f1': []}
        
        # 对每个折叠进行评估
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 训练模型
            clf.fit(X_train, y_train)
            
            # 预测
            y_pred = clf.predict(X_test)
            
            # 计算整体指标
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
            metrics['f1_weighted'].append(f1_score(y_test, y_pred, average='weighted'))
            metrics['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
            metrics['kappa'].append(cohen_kappa_score(y_test, y_pred))
            
            # 计算每个类别的指标
            report = classification_report(y_test, y_pred, output_dict=True)
            for class_idx in class_metrics:
                if str(class_idx) in report:
                    class_metrics[class_idx]['precision'].append(report[str(class_idx)]['precision'])
                    class_metrics[class_idx]['recall'].append(report[str(class_idx)]['recall'])
                    class_metrics[class_idx]['f1'].append(report[str(class_idx)]['f1-score'])
        
        # 计算平均指标
        mean_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
        std_metrics = {metric: np.std(scores) for metric, scores in metrics.items()}
        
        # 计算每个类别的平均指标
        mean_class_metrics = {}
        std_class_metrics = {}
        for class_idx, metrics_dict in class_metrics.items():
            mean_class_metrics[class_idx] = {metric: np.mean(scores) for metric, scores in metrics_dict.items()}
            std_class_metrics[class_idx] = {metric: np.std(scores) for metric, scores in metrics_dict.items()}
        
        if verbose:
            logger.info(f"  {clf_name} 评估完成")
            logger.info(f"  准确率: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
            logger.info(f"  平衡准确率: {mean_metrics['balanced_accuracy']:.4f} ± {std_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  加权F1分数: {mean_metrics['f1_weighted']:.4f} ± {std_metrics['f1_weighted']:.4f}")
            logger.info(f"  宏平均F1分数: {mean_metrics['f1_macro']:.4f} ± {std_metrics['f1_macro']:.4f}")
            logger.info(f"  Cohen's Kappa: {mean_metrics['kappa']:.4f} ± {std_metrics['kappa']:.4f}")
            
            logger.info("  各类别性能:")
            for class_idx in sorted(mean_class_metrics.keys()):
                logger.info(f"    类别 {class_idx}: F1 = {mean_class_metrics[class_idx]['f1']:.4f}, "
                           f"精确率 = {mean_class_metrics[class_idx]['precision']:.4f}, "
                           f"召回率 = {mean_class_metrics[class_idx]['recall']:.4f}")
        
        # 返回结果
        return clf_name, {
            'mean': mean_metrics,
            'std': std_metrics,
            'class_mean': mean_class_metrics,
            'class_std': std_class_metrics
        }
    
    # 并行执行所有分类器评估
    clf_results = Parallel(n_jobs=-1)(
        delayed(evaluate_classifier)(clf_name, clf) for clf_name, clf in classifiers.items()
    )
    
    # 处理结果
    results = dict(clf_results)
    
    # 绘制性能比较图
    if plot:
        plt.figure(figsize=(15, 10))
        
        metrics_to_plot = ['balanced_accuracy', 'f1_weighted', 'f1_macro', 'kappa']
        metric_names = {
            'balanced_accuracy': 'Balanced Accuracy',
            'f1_weighted': 'Weighted F1 Score',
            'f1_macro': 'Macro F1 Score',
            'kappa': "Cohen's Kappa"
        }
        
        # 准备数据
        clf_names = list(results.keys())
        
        # 绘制整体性能比较
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            
            means = [results[clf]['mean'][metric] for clf in clf_names]
            stds = [results[clf]['std'][metric] for clf in clf_names]
            
            plt.bar(clf_names, means, yerr=stds, alpha=0.7)
            plt.title(f'Performance Comparison: {metric_names[metric]}')
            plt.xlabel('Classifier')
            plt.ylabel(metric_names[metric])
            plt.grid(axis='y')
            
            # 如果是平衡准确率，添加随机猜测的基线
            if metric == 'balanced_accuracy':
                n_classes = len(np.unique(y))
                plt.axhline(y=1/n_classes, color='r', linestyle='--', 
                          label=f'Random Guessing ({1/n_classes:.4f})')
                plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if save_name:
            save_path = os.path.join(FIGURES_DIR, f'{save_name}_classifier_comparison.png')
            plt.savefig(save_path, dpi=300)
            if verbose:
                logger.info(f"分类器性能比较图表已保存至: {save_path}")
        
        plt.close()
        
        # 绘制每个类别的F1分数
        plt.figure(figsize=(15, 8))
        
        # 获取所有类别
        all_classes = set()
        for clf_results in results.values():
            all_classes.update(clf_results['class_mean'].keys())
        all_classes = sorted(all_classes)
        
        # 设置x轴位置
        x = np.arange(len(all_classes))
        width = 0.8 / len(clf_names)
        offsets = np.linspace(-width * (len(clf_names)-1)/2, width * (len(clf_names)-1)/2, len(clf_names))
        
        # 绘制每个分类器的每个类别F1分数
        for i, clf_name in enumerate(clf_names):
            class_f1_means = []
            class_f1_stds = []
            
            for class_idx in all_classes:
                if class_idx in results[clf_name]['class_mean']:
                    class_f1_means.append(results[clf_name]['class_mean'][class_idx]['f1'])
                    class_f1_stds.append(results[clf_name]['class_std'][class_idx]['f1'])
                else:
                    class_f1_means.append(0)
                    class_f1_stds.append(0)
            
            plt.bar(x + offsets[i], class_f1_means, width, label=clf_name, yerr=class_f1_stds, alpha=0.7)
        
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Score by Class for Different Classifiers')
        plt.xticks(x, all_classes)
        plt.legend()
        plt.grid(axis='y')
        
        # 保存图表
        if save_name:
            save_path = os.path.join(FIGURES_DIR, f'{save_name}_class_f1_comparison.png')
            plt.savefig(save_path, dpi=300)
            if verbose:
                logger.info(f"类别F1分数比较图表已保存至: {save_path}")
        
        plt.close()
    
    # 找出最佳分类器
    best_clf = max(results.items(), key=lambda x: x[1]['mean']['balanced_accuracy'])[0]
    if verbose:
        logger.info(f"\n最佳分类器: {best_clf}")
        logger.info(f"  平衡准确率: {results[best_clf]['mean']['balanced_accuracy']:.4f}")
        logger.info(f"  宏平均F1分数: {results[best_clf]['mean']['f1_macro']:.4f}")
    
    return results

def evaluate_combined_features(feature_groups, labels, best_groups=None, clf=None, 
                             verbose=True, plot=True, save_name=None):
    """
    评估最佳特征组合的分类性能
    
    参数:
        feature_groups: 特征组字典
        labels: 目标标签
        best_groups: 最佳特征组列表，如果为None则评估所有可能的组合
        clf: 分类器，如果为None则使用随机森林
        verbose: 是否打印详细信息
        plot: 是否绘制性能图表
        save_name: 保存文件名(不含扩展名)
        
    返回:
        best_combination: 最佳特征组合
        best_performance: 最佳组合的性能指标
    """
    if verbose:
        logger.info("评估特征组合的分类性能...")
    
    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        if verbose:
            logger.info(f"使用随机森林分类器")
    
    # 如果未指定最佳组，则测试所有组合
    if best_groups is None:
        group_names = list(feature_groups.keys())
        # 测试所有组合（包括单个组）
        combinations = []
        for r in range(1, len(group_names) + 1):
            from itertools import combinations as iter_combinations
            combinations.extend(list(iter_combinations(group_names, r)))
    else:
        combinations = [tuple(best_groups)]
    
    results = []
    
    for combo in tqdm(combinations, desc="评估特征组合"):
        if verbose:
            logger.info(f"\n评估特征组合: {' + '.join(combo)}")
        
        # 合并特征
        features_list = [feature_groups[group] for group in combo]
        X_combined = np.hstack(features_list)
        
        # 评估分类性能
        cv = EVALUATION['classification']['cv_folds']
        acc_scores = cross_val_score(clf, X_combined, labels, cv=cv, scoring='accuracy')
        balanced_acc_scores = cross_val_score(clf, X_combined, labels, cv=cv, scoring='balanced_accuracy')
        f1_weighted_scores = cross_val_score(clf, X_combined, labels, cv=cv, scoring='f1_weighted')
        f1_macro_scores = cross_val_score(clf, X_combined, labels, cv=cv, scoring='f1_macro')
        
        # Cohen's Kappa需要自定义评分函数
        def cohen_kappa_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return cohen_kappa_score(y, y_pred)
        
        kappa_scores = cross_val_score(clf, X_combined, labels, cv=cv, scoring=cohen_kappa_scorer)
        
        # 计算每个类别的性能指标
        class_metrics = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for train_index, test_index in skf.split(X_combined, labels):
            X_train, X_test = X_combined[train_index], X_combined[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            for class_label, metrics in report.items():
                if class_label.isdigit() or (isinstance(class_label, (int, float)) and not isinstance(class_label, bool)):
                    class_idx = int(class_label) if class_label.isdigit() else class_label
                    if class_idx not in class_metrics:
                        class_metrics[class_idx] = {'precision': [], 'recall': [], 'f1': []}
                    
                    class_metrics[class_idx]['precision'].append(metrics['precision'])
                    class_metrics[class_idx]['recall'].append(metrics['recall'])
                    class_metrics[class_idx]['f1'].append(metrics['f1-score'])
        
        # 计算平均类别指标
        mean_class_metrics = {}
        for class_idx, metrics_dict in class_metrics.items():
            mean_class_metrics[class_idx] = {
                metric: np.mean(scores) for metric, scores in metrics_dict.items()
            }
        
        # 存储结果
        result = {
            'combination': combo,
            'accuracy': acc_scores.mean(),
            'accuracy_std': acc_scores.std(),
            'balanced_accuracy': balanced_acc_scores.mean(),
            'balanced_accuracy_std': balanced_acc_scores.std(),
            'f1_weighted': f1_weighted_scores.mean(),
            'f1_weighted_std': f1_weighted_scores.std(),
            'f1_macro': f1_macro_scores.mean(),
            'f1_macro_std': f1_macro_scores.std(),
            'kappa': kappa_scores.mean(),
            'kappa_std': kappa_scores.std(),
            'class_metrics': mean_class_metrics
        }
        
        results.append(result)
        
        if verbose:
            logger.info(f"  准确率: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
            logger.info(f"  平衡准确率: {balanced_acc_scores.mean():.4f} ± {balanced_acc_scores.std():.4f}")
            logger.info(f"  加权F1分数: {f1_weighted_scores.mean():.4f} ± {f1_weighted_scores.std():.4f}")
            logger.info(f"  宏平均F1分数: {f1_macro_scores.mean():.4f} ± {f1_macro_scores.std():.4f}")
            logger.info(f"  Cohen's Kappa: {kappa_scores.mean():.4f} ± {kappa_scores.std():.4f}")
    
    # 按平衡准确率排序
    results.sort(key=lambda x: x['balanced_accuracy'], reverse=True)
    
    # 输出排序后的结果
    if verbose and len(results) > 1:
        logger.info("\n特征组合性能排名:")
        for i, result in enumerate(results[:5]):  # 只显示前5个
            combo_str = ' + '.join(result['combination'])
            logger.info(f"{i+1}. {combo_str}:")
            logger.info(f"   平衡准确率: {result['balanced_accuracy']:.4f} ± {result['balanced_accuracy_std']:.4f}")
            logger.info(f"   宏平均F1分数: {result['f1_macro']:.4f} ± {result['f1_macro_std']:.4f}")
    
    # 绘制组合性能比较图
    if plot and len(results) > 1:
        plt.figure(figsize=(14, 10))
        
        # 准备数据（只绘制前10个组合，否则图表太拥挤）
        top_n = min(10, len(results))
        top_results = results[:top_n]
        
        combinations = [' + '.join(r['combination']) for r in top_results]
        balanced_accs = [r['balanced_accuracy'] for r in top_results]
        balanced_acc_stds = [r['balanced_accuracy_std'] for r in top_results]
        f1_macros = [r['f1_macro'] for r in top_results]
        f1_macro_stds = [r['f1_macro_std'] for r in top_results]
        kappas = [r['kappa'] for r in top_results]
        kappa_stds = [r['kappa_std'] for r in top_results]
        
        # 绘制平衡准确率
        plt.subplot(3, 1, 1)
        plt.bar(combinations, balanced_accs, yerr=balanced_acc_stds, alpha=0.8)
        n_classes = len(np.unique(labels))
        plt.axhline(y=1/n_classes, color='r', linestyle='--', 
                   label=f'Random Guessing ({1/n_classes:.4f})')
        plt.xlabel('Feature Combinations')
        plt.ylabel('Balanced Accuracy')
        plt.title('Classification Performance: Balanced Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.legend()
        
        # 绘制宏平均F1分数
        plt.subplot(3, 1, 2)
        plt.bar(combinations, f1_macros, yerr=f1_macro_stds, alpha=0.8)
        plt.xlabel('Feature Combinations')
        plt.ylabel('Macro F1 Score')
        plt.title('Classification Performance: Macro F1 Score')
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
        if save_name:
            save_path = os.path.join(FIGURES_DIR, f'{save_name}_feature_combination_performance.png')
            plt.savefig(save_path, dpi=300)
            if verbose:
                logger.info(f"特征组合性能图表已保存至: {save_path}")
        
        plt.close()
    
    # 返回最佳组合
    best_combination = results[0]['combination']
    best_performance = {
        'balanced_accuracy': results[0]['balanced_accuracy'],
        'f1_macro': results[0]['f1_macro'],
        'kappa': results[0]['kappa'],
        'class_metrics': results[0]['class_metrics']
    }
    
    if verbose:
        logger.info(f"\n最佳特征组合: {' + '.join(best_combination)}")
        logger.info(f"  平衡准确率: {best_performance['balanced_accuracy']:.4f}")
        logger.info(f"  宏平均F1分数: {best_performance['f1_macro']:.4f}")
        logger.info(f"  Cohen's Kappa: {best_performance['kappa']:.4f}")
    
    return best_combination, best_performance

def visualize_confusion_matrix(X, y, clf=None, class_names=None, test_size=0.3, 
                             verbose=True, save_name=None):
    """
    可视化分类器的混淆矩阵
    
    参数:
        X: 特征数据
        y: 目标标签
        clf: 分类器，如果为None则使用随机森林
        class_names: 类别名称列表
        test_size: 测试集比例
        verbose: 是否打印详细信息
        save_name: 保存文件名(不含扩展名)
        
    返回:
        cm: 混淆矩阵
        clf: 训练好的分类器
    """
    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        if verbose:
            logger.info(f"使用随机森林分类器")
    
    if verbose:
        logger.info("可视化分类器的混淆矩阵...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 训练分类器
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)
    
    if verbose:
        logger.info(f"测试集性能:")
        logger.info(f"  准确率: {accuracy:.4f}")
        logger.info(f"  平衡准确率: {balanced_accuracy:.4f}")
        logger.info(f"  加权F1分数: {f1_weighted:.4f}")
        logger.info(f"  宏平均F1分数: {f1_macro:.4f}")
        logger.info(f"  Cohen's Kappa: {kappa:.4f}")
        
        # 打印分类报告
        logger.info("\n分类报告:")
        report = classification_report(y_test, y_pred)
        for line in report.split('\n'):
            logger.info(f"  {line}")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(12, 10))
    
    # 使用类别名称（如果提供）
    if class_names is not None:
        labels = class_names
    else:
        labels = [f'Class {i}' for i in range(len(np.unique(y)))]
    
    # 计算归一化的混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制原始混淆矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 绘制归一化的混淆矩阵
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    
    # 保存图表
    if save_name:
        save_path = os.path.join(FIGURES_DIR, f'{save_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300)
        if verbose:
            logger.info(f"混淆矩阵图表已保存至: {save_path}")
    
    plt.close()
    
    return cm, clf

if __name__ == "__main__":
    # 测试分类评估功能
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
    
    # 测试分类器比较
    classifier_results = compare_classification_models(
        processed_groups['all_features'], big_labels, save_name='all_features'
    )
    
    # 测试特征组合评估
    best_combination, best_performance = evaluate_combined_features(
        processed_groups, big_labels, save_name='feature_combinations'
    )
    
    # 测试混淆矩阵可视化
    # 合并最佳特征组合
    features_list = [processed_groups[group] for group in best_combination]
    X_combined = np.hstack(features_list)
    
    cm, clf = visualize_confusion_matrix(
        X_combined, big_labels, class_names=big_class_names, save_name='best_combination'
    )
    
    print("分类评估测试完成")