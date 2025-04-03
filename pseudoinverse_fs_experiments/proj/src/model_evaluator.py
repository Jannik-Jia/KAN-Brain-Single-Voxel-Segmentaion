# src/model_evaluator.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from src.bilingual_logger import BilingualLogger
class ModelEvaluator:
    """模型评估类，用于评估模型性能并生成可视化结果"""
    
    def __init__(self, logger=None):
        """
        初始化评估器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger if logger else BilingualLogger()
        self.results = {}  # 存储评估结果
    
    def evaluate(self, model, X, y, dataset_name="test"):
        """
        评估模型性能
        
        参数:
            model: 已拟合的模型
            X: 特征矩阵
            y: 真实标签
            dataset_name: 数据集名称
            
        返回:
            包含各种评估指标的字典
        """
        self.logger.info(f"开始评估模型在{dataset_name}集上的性能", 
                       f"Starting model evaluation on {dataset_name} set")
        
        # 获取预测
        y_pred = model.predict(X)
        
        try:
            # 计算概率预测（可选）
            y_proba = model.predict_proba(X)
        except:
            y_proba = None
        
        # 计算评估指标
        accuracy = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
        
        # 处理可能的警告（某些类别可能没有样本）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1_macro = f1_score(y, y_pred, average='macro')
            f1_weighted = f1_score(y, y_pred, average='weighted')
            class_f1 = f1_score(y, y_pred, average=None)
            
        # 只有当所有类别都有样本时才计算Kappa
        try:
            kappa = cohen_kappa_score(y, y_pred)
        except:
            kappa = float('nan')
        
        # 分类报告和混淆矩阵
        try:
            report = classification_report(y, y_pred, output_dict=True)
        except:
            report = {}
        
        try:
            conf_matrix = confusion_matrix(y, y_pred)
        except:
            conf_matrix = np.array([])
        
        # 统计每个类别的样本数
        class_counts = {}
        unique_classes = np.unique(y)
        for cls in unique_classes:
            class_counts[int(cls)] = np.sum(y == cls)
        
        # 将结果存储到字典中
        result = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'kappa': kappa,
            'class_f1': class_f1,
            'report': report,
            'confusion_matrix': conf_matrix,
            'class_counts': class_counts,
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        # 记录评估结果
        self.logger.info(
            f"{dataset_name}集评估结果:\n"
            f"  准确率: {accuracy:.4f}\n"
            f"  平衡准确率: {balanced_acc:.4f}\n"
            f"  宏平均F1: {f1_macro:.4f}\n"
            f"  加权F1: {f1_weighted:.4f}\n"
            f"  Kappa系数: {kappa:.4f}",
            
            f"{dataset_name} set evaluation results:\n"
            f"  Accuracy: {accuracy:.4f}\n"
            f"  Balanced Accuracy: {balanced_acc:.4f}\n"
            f"  Macro F1: {f1_macro:.4f}\n"
            f"  Weighted F1: {f1_weighted:.4f}\n"
            f"  Kappa: {kappa:.4f}"
        )
        
        # 保存结果
        self.results[dataset_name] = result
        
        return result

    def visualize_performance(self, save_dir=None, prefix=""):
        """
        可视化评估结果
        
        参数:
            save_dir: 图表保存目录
            prefix: 文件名前缀
        """
        if not self.results:
            self.logger.warning("没有可视化的评估结果", "No evaluation results to visualize")
            return
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 1. 绘制性能指标比较图
        self._plot_performance_comparison(save_dir, prefix)
        
        # 2. 绘制每个数据集的混淆矩阵
        for dataset_name, result in self.results.items():
            self._plot_confusion_matrix(result, dataset_name, save_dir, prefix)
        
        # 3. 绘制类别性能分析图（对测试集）
        if 'test' in self.results:
            self._plot_class_performance(self.results['test'], save_dir, prefix)
    
    def _plot_performance_comparison(self, save_dir, prefix):
        """绘制不同数据集间的性能比较"""
        # 提取数据
        datasets = list(self.results.keys())
        metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'kappa']
        metric_names = ['Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', 'Kappa']
        
        values = np.zeros((len(metrics), len(datasets)))
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                values[j, i] = self.results[dataset][metric]
        
        # 绘制柱状图
        plt.figure(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.7 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            offset = (i - len(datasets)/2 + 0.5) * width
            plt.bar(x + offset, values[:, i], width, label=dataset.capitalize())
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metric_names)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                offset = (i - len(datasets)/2 + 0.5) * width
                plt.text(j + offset, values[j, i] + 0.01, f"{values[j, i]:.3f}", 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}performance_comparison.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_confusion_matrix(self, result, dataset_name, save_dir, prefix):
        """为指定数据集绘制混淆矩阵热图"""
        conf_matrix = result['confusion_matrix']
        
        if conf_matrix.size == 0:
            return
        
        # 对大型混淆矩阵使用对数缩放
        if conf_matrix.shape[0] > 10:
            # 添加一个小值防止log(0)
            conf_matrix_vis = np.log1p(conf_matrix)
            title = f"Confusion Matrix (log scale) - {dataset_name.capitalize()} Set"
            vmax = np.log1p(conf_matrix.max())
        else:
            conf_matrix_vis = conf_matrix
            title = f"Confusion Matrix - {dataset_name.capitalize()} Set"
            vmax = conf_matrix.max()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_vis, annot=False, fmt='d', cmap='Blues', 
                   vmin=0, vmax=vmax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}confusion_matrix_{dataset_name}.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_class_performance(self, result, save_dir, prefix):
        """绘制类别级别的性能分析图表"""
        class_f1 = result['class_f1']
        class_counts = result['class_counts']
        
        # 准备数据
        classes = list(class_counts.keys())
        classes.sort()
        counts = [class_counts[cls] for cls in classes]
        f1_scores = [class_f1[i] if i < len(class_f1) else 0 for i in range(len(classes))]
        
        # 根据样本数量筛选类别（只显示有样本的类别）
        valid_idx = [i for i, count in enumerate(counts) if count > 0]
        classes = [classes[i] for i in valid_idx]
        counts = [counts[i] for i in valid_idx]
        f1_scores = [f1_scores[i] for i in valid_idx]
        
        if not classes:
            return
        
        # 1. 绘制F1分数柱状图（按分数排序）
        sorted_idx = np.argsort(f1_scores)
        sorted_classes = [classes[i] for i in sorted_idx]
        sorted_f1 = [f1_scores[i] for i in sorted_idx]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_classes)), sorted_f1)
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Score by Class (Sorted)')
        
        # 如果类别太多，只显示一部分刻度
        if len(classes) > 20:
            tick_step = len(classes) // 10
            plt.xticks(range(0, len(sorted_classes), tick_step), 
                      [sorted_classes[i] for i in range(0, len(sorted_classes), tick_step)])
        else:
            plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=90)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}class_f1_sorted.png"), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 2. 绘制样本数量与F1分数的散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(counts, f1_scores, alpha=0.6)
        plt.xlabel('Sample Count')
        plt.ylabel('F1 Score')
        plt.title('Relationship Between Sample Count and F1 Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 尝试拟合一条趋势线
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(counts, f1_scores)
            x = np.array([min(counts), max(counts)])
            y = slope * x + intercept
            plt.plot(x, y, 'r--', label=f'Trend line (r={r_value:.2f})')
            plt.legend()
        except:
            pass
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}sample_count_vs_f1.png"), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 3. 为样本最多和最少的几个类别绘制详细分析
        top_n = 5
        if len(classes) > top_n * 2:
            # 样本最多的类别
            top_idx = np.argsort(counts)[-top_n:]
            top_classes = [classes[i] for i in top_idx]
            top_counts = [counts[i] for i in top_idx]
            top_f1 = [f1_scores[i] for i in top_idx]
            
            # 样本最少的类别
            bottom_idx = np.argsort(counts)[:top_n]
            bottom_classes = [classes[i] for i in bottom_idx]
            bottom_counts = [counts[i] for i in bottom_idx]
            bottom_f1 = [f1_scores[i] for i in bottom_idx]
            
            plt.figure(figsize=(12, 6))
            
            # 左侧：样本最多的类别
            plt.subplot(1, 2, 1)
            plt.bar(range(len(top_classes)), top_f1)
            plt.title('Classes with Most Samples')
            plt.xlabel('Class')
            plt.ylabel('F1 Score')
            plt.xticks(range(len(top_classes)), top_classes, rotation=45)
            for i, (cls, count) in enumerate(zip(top_classes, top_counts)):
                plt.text(i, top_f1[i] + 0.02, f"{count}", ha='center')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 右侧：样本最少的类别
            plt.subplot(1, 2, 2)
            plt.bar(range(len(bottom_classes)), bottom_f1)
            plt.title('Classes with Least Samples')
            plt.xlabel('Class')
            plt.ylabel('F1 Score')
            plt.xticks(range(len(bottom_classes)), bottom_classes, rotation=45)
            for i, (cls, count) in enumerate(zip(bottom_classes, bottom_counts)):
                plt.text(i, bottom_f1[i] + 0.02, f"{count}", ha='center')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{prefix}top_bottom_classes.png"), dpi=300)
                plt.close()
            else:
                plt.show()