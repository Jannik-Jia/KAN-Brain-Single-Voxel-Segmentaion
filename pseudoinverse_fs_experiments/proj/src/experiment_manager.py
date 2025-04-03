# src/experiment_manager.py

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .bilingual_logger import BilingualLogger
from .feature_selector import FeatureSelector
from .pseudoinverse_model import PseudoInverseModel
from .model_evaluator import ModelEvaluator
from .visualization_utils import (
    visualize_feature_selection, 
    visualize_feature_stability, 
    visualize_feature_importance, 
    visualize_weight_distribution
)

class ExperimentManager:
    """实验管理器，处理批量参数测试和结果管理"""
    
    def __init__(self, data_loader, base_dir="experiments", logger=None):
        """
        初始化实验管理器
        
        参数:
            data_loader: 数据加载器实例
            base_dir: 实验结果基础目录
            logger: 日志记录器
        """
        self.data_loader = data_loader
        self.base_dir = base_dir
        self.logger = logger if logger else BilingualLogger()
        
        # 确保基础目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        # 创建实验记录文件
        self.experiment_log_path = os.path.join(base_dir, "experiment_log.csv")
        if not os.path.exists(self.experiment_log_path):
            with open(self.experiment_log_path, 'w') as f:
                f.write("experiment_id,timestamp,apply_pca,n_components,normalization,"
                       "class_balance,target_samples,regularization,alpha,"
                       "train_accuracy,test_accuracy,val_accuracy,train_f1,test_f1,val_f1,"
                       "train_time,evaluation_time,status\n")
        
        self.logger.info(f"实验管理器初始化，基础目录: {base_dir}", 
                       f"Experiment manager initialized with base directory: {base_dir}")
    
    def run_experiment(self, params, force_rerun=False):
        """
        运行单个实验
        
        参数:
            params: 实验参数字典
            force_rerun: 是否强制重新运行已完成的实验
            
        返回:
            experiment_id: 实验ID
            results: 实验结果
        """
        # 生成实验ID
        experiment_id = self._generate_experiment_id(params)
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        
        # 检查是否已完成实验
        if os.path.exists(experiment_dir) and not force_rerun:
            status_path = os.path.join(experiment_dir, "status.json")
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    if status.get('status') == 'completed':
                        self.logger.info(f"实验 {experiment_id} 已完成，跳过", 
                                       f"Experiment {experiment_id} already completed, skipping")
                        return experiment_id, self._load_results(experiment_dir)
        
        # 创建实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 保存参数
        with open(os.path.join(experiment_dir, "params.json"), 'w') as f:
            json.dump(params, f, indent=4)
        
        # 更新状态为运行中
        with open(os.path.join(experiment_dir, "status.json"), 'w') as f:
            json.dump({"status": "running", "start_time": str(datetime.now())}, f, indent=4)
        
        # 记录实验开始
        self.logger.info(f"开始实验 {experiment_id}", f"Starting experiment {experiment_id}")
        self.logger.info(f"参数: {params}", f"Parameters: {params}")
        
        try:
            # 处理数据
            start_time = time.time()
            processed_data = self.data_loader.preprocess_data(
                apply_pca=params.get('apply_pca', False),
                n_components=params.get('n_components', 50),
                normalization=params.get('normalization', None),
                class_balance=params.get('class_balance', False),
                target_samples=params.get('target_samples', 1000)
            )
            
            # 训练模型
            model = PseudoInverseModel(num_classes=102, logger=self.logger)
            model.fit(
                processed_data['train_X'], 
                processed_data['train_y'],
                regularization=params.get('regularization', None),
                alpha=params.get('alpha', 0.0)
            )
            train_time = time.time() - start_time
            
            # 保存模型
            model.save(os.path.join(experiment_dir, "model.pkl"))
            
            # 评估模型
            eval_start_time = time.time()
            evaluator = ModelEvaluator(logger=self.logger)
            
            # 评估训练集
            train_result = evaluator.evaluate(
                model, processed_data['train_X'], processed_data['train_y'], "train")
            
            # 评估测试集
            test_result = evaluator.evaluate(
                model, processed_data['test_X'], processed_data['test_y'], "test")
            
            # 评估验证集
            val_result = evaluator.evaluate(
                model, processed_data['val_X'], processed_data['val_y'], "val")
            
            evaluation_time = time.time() - eval_start_time
            
            # 生成可视化
            evaluator.visualize_performance(save_dir=experiment_dir)
            
            # 特征重要性可视化
            visualize_feature_importance(
                model, top_n=30, 
                save_path=os.path.join(experiment_dir, "feature_importance.png"))
            
            # 权重分布可视化
            visualize_weight_distribution(
                model, save_path=os.path.join(experiment_dir, "weight_distribution.png"))
            
            # 保存评估结果
            results = {
                'train': train_result,
                'test': test_result,
                'val': val_result,
                'train_time': train_time,
                'evaluation_time': evaluation_time
            }

            # 保存评估结果
            self._save_results(experiment_dir, results)

            # 更新状态为已完成
            with open(os.path.join(experiment_dir, "status.json"), 'w') as f:
                json.dump({
                    "status": "completed", 
                    "start_time": str(datetime.now()),
                    "end_time": str(datetime.now()),
                    "train_time": train_time,
                    "evaluation_time": evaluation_time
                }, f, indent=4)
            
            # 更新实验日志
            self._update_experiment_log(
                experiment_id, params, train_result, test_result, val_result, 
                train_time, evaluation_time, "completed")
            
            self.logger.info(f"实验 {experiment_id} 完成", f"Experiment {experiment_id} completed")
            
            return experiment_id, results
            
        except Exception as e:
            # 记录错误
            self.logger.error(f"实验 {experiment_id} 失败: {str(e)}", 
                            f"Experiment {experiment_id} failed: {str(e)}")
            
            # 更新状态为失败
            with open(os.path.join(experiment_dir, "status.json"), 'w') as f:
                json.dump({
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, f, indent=4)
            
            # 更新实验日志
            self._update_experiment_log(
                experiment_id, params, None, None, None, 0, 0, "failed")
            
            raise e
    
    def run_batch_experiments(self, param_grid, max_experiments=None):
        """
        运行批量实验
        
        参数:
            param_grid: 参数网格，包含每个参数的可能值列表
            max_experiments: 最大实验数量，None表示不限制
            
        返回:
            completed_experiments: 已完成实验的ID列表
        """
        # 生成所有参数组合
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # 计算总实验数
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        if max_experiments and max_experiments < total_combinations:
            self.logger.info(f"限制实验数量为 {max_experiments}/{total_combinations}", 
                           f"Limiting to {max_experiments}/{total_combinations} experiments")
            total_combinations = max_experiments
        
        self.logger.info(f"开始批量实验，共 {total_combinations} 个组合", 
                       f"Starting batch experiments with {total_combinations} combinations")
        
        # 生成所有参数组合
        param_combinations = []
        
        def generate_combinations(keys, values, current=0, current_params={}):
            if current == len(keys):
                param_combinations.append(current_params.copy())
                return
            
            for value in values[current]:
                current_params[keys[current]] = value
                generate_combinations(keys, values, current + 1, current_params)
                
                # 如果达到最大实验数，则停止
                if max_experiments and len(param_combinations) >= max_experiments:
                    break
        
        generate_combinations(param_keys, param_values)
        
        # 打乱参数组合，使实验更加随机化
        np.random.shuffle(param_combinations)
        
        # 运行所有实验
        completed_experiments = []
        total_start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            # 估计剩余时间
            if i > 0:
                elapsed_time = time.time() - total_start_time
                avg_time_per_exp = elapsed_time / i
                remaining_time = avg_time_per_exp * (len(param_combinations) - i)
                
                self.logger.info(
                    f"进度: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%), "
                    f"预计剩余时间: {remaining_time/60:.1f} 分钟",
                    
                    f"Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%), "
                    f"Estimated time remaining: {remaining_time/60:.1f} minutes"
                )
            
            try:
                experiment_id, _ = self.run_experiment(params)
                completed_experiments.append(experiment_id)
            except Exception as e:
                self.logger.error(f"实验失败: {str(e)}", f"Experiment failed: {str(e)}")
                # 继续下一个实验
                continue
        
        total_elapsed_time = time.time() - total_start_time
        self.logger.info(
            f"批量实验完成，共 {len(completed_experiments)}/{len(param_combinations)} 个实验成功，"
            f"总耗时: {total_elapsed_time/60:.1f} 分钟",
            
            f"Batch experiments completed, {len(completed_experiments)}/{len(param_combinations)} "
            f"experiments succeeded, total time: {total_elapsed_time/60:.1f} minutes"
        )
        
        return completed_experiments
    
    def _generate_experiment_id(self, params):
        """生成唯一的实验ID"""
        # 使用参数的哈希值作为ID的一部分
        param_str = '_'.join([f"{k}_{v}" for k, v in sorted(params.items())])
        param_hash = abs(hash(param_str)) % 10000
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        
        # 生成ID
        experiment_id = f"exp_{timestamp}_{param_hash}"
        
        return experiment_id
    
    def _update_experiment_log(self, experiment_id, params, train_result, test_result, 
                              val_result, train_time, evaluation_time, status):
        """更新实验日志"""
        # 提取评估指标
        train_acc = train_result['accuracy'] if train_result else float('nan')
        test_acc = test_result['accuracy'] if test_result else float('nan')
        val_acc = val_result['accuracy'] if val_result else float('nan')
        
        train_f1 = train_result['f1_macro'] if train_result else float('nan')
        test_f1 = test_result['f1_macro'] if test_result else float('nan')
        val_f1 = val_result['f1_macro'] if val_result else float('nan')
        
        # 准备日志条目
        log_entry = (f"{experiment_id},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{params.get('apply_pca', False)},{params.get('n_components', 0)},"
                    f"{params.get('normalization', 'none')},{params.get('class_balance', False)},"
                    f"{params.get('target_samples', 0)},{params.get('regularization', 'none')},"
                    f"{params.get('alpha', 0.0)},"
                    f"{train_acc},{test_acc},{val_acc},{train_f1},{test_f1},{val_f1},"
                    f"{train_time},{evaluation_time},{status}\n")
        
        # 追加到日志文件
        with open(self.experiment_log_path, 'a') as f:
            f.write(log_entry)
    
    def _save_results(self, experiment_dir, results):
        """保存实验结果到JSON文件"""
        # 定义递归转换NumPy类型的函数
        def convert_numpy_types(obj):
            """递归转换字典中的NumPy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist() if obj.size > 0 else []
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif str(type(obj)).startswith("<class 'numpy"):  # 捕获其他NumPy类型
                return obj.item() if hasattr(obj, 'item') else str(obj)
            else:
                return obj
        
        # 保存评估结果
        with open(os.path.join(experiment_dir, "results.json"), 'w') as f:
            # 转换NumPy类型为Python原生类型
            results_json = convert_numpy_types(results)
            json.dump(results_json, f, indent=4)
    
    def _load_results(self, experiment_dir):
        """加载保存的实验结果"""
        results_path = os.path.join(experiment_dir, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def generate_summary_report(self, top_n=10):
        """
        生成汇总报告
        
        参数:
            top_n: 展示的顶部实验数量
        """
        # 读取实验日志
        if not os.path.exists(self.experiment_log_path):
            self.logger.warning("实验日志不存在", "Experiment log does not exist")
            return
        
        try:
            log_df = pd.read_csv(self.experiment_log_path)
        except:
            self.logger.warning("无法读取实验日志", "Cannot read experiment log")
            return
        
        # 过滤成功的实验
        log_df = log_df[log_df['status'] == 'completed']
        
        if len(log_df) == 0:
            self.logger.warning("没有已完成的实验", "No completed experiments")
            return
        
        # 按测试集准确率排序
        log_df_sorted = log_df.sort_values('test_accuracy', ascending=False)
        
        # 创建报告目录
        report_dir = os.path.join(self.base_dir, "summary_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存排序后的实验日志
        log_df_sorted.to_csv(os.path.join(report_dir, "experiments_sorted.csv"), index=False)
        
        # 生成顶部实验表格
        top_df = log_df_sorted.head(top_n)
        
        # 生成HTML报告
        html_report = f"""
        <html>
        <head>
            <title>Pseudo-Inverse Experiments Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                h2 {{ color: #666699; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .chart {{ margin: 10px; max-width: 600px; }}
            </style>
        </head>
        <body>
            <h1>Pseudo-Inverse Experiments Summary Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total experiments: {len(log_df)}</p>
            
            <h2>Top {top_n} Experiments by Test Accuracy</h2>
            <table>
                <tr>
                    <th>Experiment ID</th>
                    <th>Test Accuracy</th>
                    <th>Test F1</th>
                    <th>Val Accuracy</th>
                    <th>PCA</th>
                    <th>Components</th>
                    <th>Normalization</th>
                    <th>Class Balance</th>
                    <th>Regularization</th>
                </tr>
        """
        

        for _, row in top_df.iterrows():
            html_report += f"""
                <tr>
                    <td>{row['experiment_id']}</td>
                    <td>{row['test_accuracy']:.4f}</td>
                    <td>{row['test_f1']:.4f}</td>
                    <td>{row['val_accuracy']:.4f}</td>
                    <td>{'Yes' if row['apply_pca'] else 'No'}</td>
                    <td>{row['n_components']}</td>
                    <td>{row['normalization']}</td>
                    <td>{'Yes' if row['class_balance'] else 'No'}</td>
                    <td>{row['regularization']}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Parameter Impact Analysis</h2>
            <div class="chart-container">
        """
        
        # 生成参数影响可视化
        # 1. PCA对准确率的影响
        if 'apply_pca' in log_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='apply_pca', y='test_accuracy', data=log_df)
            plt.title('Impact of PCA on Test Accuracy')
            plt.xlabel('PCA Applied')
            plt.ylabel('Test Accuracy')
            plt.tight_layout()
            pca_impact_path = os.path.join(report_dir, "pca_impact.png")
            plt.savefig(pca_impact_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(pca_impact_path, self.base_dir)}" alt="PCA Impact" width="100%">
                </div>
            """
        
        # 2. 组件数量对准确率的影响
        if 'n_components' in log_df.columns and 'apply_pca' in log_df.columns:
            pca_df = log_df[log_df['apply_pca'] == True]
            if len(pca_df) > 1:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x='n_components', y='test_accuracy', data=pca_df)
                plt.title('Impact of PCA Components on Test Accuracy')
                plt.xlabel('Number of Components')
                plt.ylabel('Test Accuracy')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                components_impact_path = os.path.join(report_dir, "components_impact.png")
                plt.savefig(components_impact_path, dpi=300)
                plt.close()
                
                html_report += f"""
                    <div class="chart">
                        <img src="{os.path.relpath(components_impact_path, self.base_dir)}" alt="Components Impact" width="100%">
                    </div>
                """
        
        # 3. 标准化方法对准确率的影响
        if 'normalization' in log_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='normalization', y='test_accuracy', data=log_df)
            plt.title('Impact of Normalization on Test Accuracy')
            plt.xlabel('Normalization Method')
            plt.ylabel('Test Accuracy')
            plt.tight_layout()
            norm_impact_path = os.path.join(report_dir, "normalization_impact.png")
            plt.savefig(norm_impact_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(norm_impact_path, self.base_dir)}" alt="Normalization Impact" width="100%">
                </div>
            """
        
        # 4. 类别平衡对准确率的影响
        if 'class_balance' in log_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='class_balance', y='test_accuracy', data=log_df)
            plt.title('Impact of Class Balancing on Test Accuracy')
            plt.xlabel('Class Balancing Applied')
            plt.ylabel('Test Accuracy')
            plt.tight_layout()
            balance_impact_path = os.path.join(report_dir, "class_balance_impact.png")
            plt.savefig(balance_impact_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(balance_impact_path, self.base_dir)}" alt="Class Balance Impact" width="100%">
                </div>
            """
        
        # 5. 正则化方法对准确率的影响
        if 'regularization' in log_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='regularization', y='test_accuracy', data=log_df)
            plt.title('Impact of Regularization on Test Accuracy')
            plt.xlabel('Regularization Method')
            plt.ylabel('Test Accuracy')
            plt.tight_layout()
            reg_impact_path = os.path.join(report_dir, "regularization_impact.png")
            plt.savefig(reg_impact_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(reg_impact_path, self.base_dir)}" alt="Regularization Impact" width="100%">
                </div>
            """
        
        html_report += """
            </div>
            
            <h2>Best Experiment Details</h2>
        """
        
        # 获取最佳实验的详细信息
        if len(top_df) > 0:
            best_exp_id = top_df.iloc[0]['experiment_id']
            best_exp_dir = os.path.join(self.base_dir, best_exp_id)
            
            # 复制最佳实验的图表到报告目录
            for img_file in ['feature_importance.png', 'weight_distribution.png', 
                            'performance_comparison.png', 'confusion_matrix_test.png']:
                img_path = os.path.join(best_exp_dir, img_file)
                if os.path.exists(img_path):
                    dest_path = os.path.join(report_dir, f"best_{img_file}")
                    import shutil
                    shutil.copy(img_path, dest_path)
                    
                    html_report += f"""
                        <div class="chart">
                            <h3>{img_file.replace('_', ' ').replace('.png', '').title()}</h3>
                            <img src="{os.path.relpath(dest_path, self.base_dir)}" alt="{img_file}" width="100%">
                        </div>
                    """
        
        html_report += """
            <h2>Conclusion and Recommendations</h2>
            <p>Based on the experimental results, here are the key findings:</p>
            <ul>
        """
        
        # 添加结论（基于实验结果）
        # PCA
        if 'apply_pca' in log_df.columns:
            pca_impact = log_df.groupby('apply_pca')['test_accuracy'].mean()
            if len(pca_impact) > 1:
                pca_better = pca_impact[True] > pca_impact[False]
                html_report += f"""
                    <li>PCA {'improves' if pca_better else 'reduces'} model performance: 
                        Average accuracy with PCA: {pca_impact[True]:.4f}, 
                        without PCA: {pca_impact[False]:.4f}</li>
                """
        
        # 标准化
        if 'normalization' in log_df.columns:
            norm_impact = log_df.groupby('normalization')['test_accuracy'].mean()
            if len(norm_impact) > 1:
                best_norm = norm_impact.idxmax()
                html_report += f"""
                    <li>Best normalization method: {best_norm} (Avg. accuracy: {norm_impact[best_norm]:.4f})</li>
                """
        
        # 类别平衡
        if 'class_balance' in log_df.columns:
            balance_impact = log_df.groupby('class_balance')['test_accuracy'].mean()
            if len(balance_impact) > 1:
                balance_better = balance_impact[True] > balance_impact[False]
                html_report += f"""
                    <li>Class balancing {'improves' if balance_better else 'reduces'} model performance: 
                        Average accuracy with balancing: {balance_impact[True]:.4f}, 
                        without balancing: {balance_impact[False]:.4f}</li>
                """
        
        # 正则化
        if 'regularization' in log_df.columns:
            reg_impact = log_df.groupby('regularization')['test_accuracy'].mean()
            if len(reg_impact) > 1:
                best_reg = reg_impact.idxmax()
                html_report += f"""
                    <li>Best regularization method: {best_reg} (Avg. accuracy: {reg_impact[best_reg]:.4f})</li>
                """
        
        # 最佳参数组合
        if len(top_df) > 0:
            best_row = top_df.iloc[0]
            html_report += f"""
                <li>Best parameter combination:
                    <ul>
                        <li>PCA: {'Yes' if best_row['apply_pca'] else 'No'}</li>
                        {'<li>Components: ' + str(best_row['n_components']) + '</li>' if best_row['apply_pca'] else ''}
                        <li>Normalization: {best_row['normalization']}</li>
                        <li>Class balancing: {'Yes' if best_row['class_balance'] else 'No'}</li>
                        <li>Regularization: {best_row['regularization']}</li>
                        <li>Alpha: {best_row['alpha']}</li>
                    </ul>
                </li>
            """
        
        html_report += """
            </ul>
        </body>
        </html>
        """
        
        # 保存HTML报告
        with open(os.path.join(report_dir, "summary_report.html"), 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"汇总报告已生成: {os.path.join(report_dir, 'summary_report.html')}", 
                       f"Summary report generated: {os.path.join(report_dir, 'summary_report.html')}")


class ExperimentManagerWithFeatureSelection(ExperimentManager):
    """扩展实验管理器，支持特征选择"""
    
    def __init__(self, data_loader, base_dir="experiments", logger=None, 
                feature_selection_mode='global'):
        """
        初始化实验管理器
        
        参数:
            data_loader: 数据加载器实例
            base_dir: 实验结果基础目录
            logger: 日志记录器
            feature_selection_mode: 特征选择模式，'global'或'per_experiment'
        """
        super().__init__(data_loader, base_dir, logger)
        self.feature_selection_mode = feature_selection_mode
        
        # 全局特征选择器
        self.global_selector = None
        # 全局特征选择前后的原始数据和处理后数据
        self.original_data = None
        self.global_selected_data = None
        
        # 更新实验日志文件头
        self.experiment_log_path = os.path.join(base_dir, "experiment_log_with_fs.csv")
        if not os.path.exists(self.experiment_log_path):
            with open(self.experiment_log_path, 'w') as f:
                f.write("experiment_id,timestamp,apply_pca,n_components,normalization,"
                       "class_balance,target_samples,regularization,alpha,"
                       "feature_selection,selection_mode,selection_threshold,max_features,l1_ratio,"
                       "train_accuracy,test_accuracy,val_accuracy,train_f1,test_f1,val_f1,"
                       "original_features,selected_features,selection_ratio,"
                       "train_time,evaluation_time,status\n")
        
        self.logger.info(f"扩展实验管理器初始化，特征选择模式: {feature_selection_mode}", 
                       f"Extended experiment manager initialized with feature selection mode: {feature_selection_mode}")
    
    def apply_global_feature_selection(self, selection_params):
        """
        应用全局特征选择
        
        参数:
            selection_params: 特征选择参数字典
            
        返回:
            feature_selector: 训练好的特征选择器
        """
        self.logger.info("开始全局特征选择", "Starting global feature selection")
        
        # 加载原始数据
        if self.original_data is None:
            self.original_data = self.data_loader.load_all_data()
        
        # 应用预处理，但不包括特征选择
        preprocessed_data = self.data_loader.preprocess_data(
            apply_pca=selection_params.get('apply_pca', False),
            n_components=selection_params.get('n_components', 50),
            normalization=selection_params.get('normalization', 'standard'),
            class_balance=selection_params.get('class_balance', False),
            target_samples=selection_params.get('target_samples', 1000)
        )
        
        # 创建特征选择器
        self.global_selector = FeatureSelector(
            method=selection_params.get('feature_selection', 'lasso'),
            selection_mode=selection_params.get('selection_mode', 'threshold'),
            selection_threshold=selection_params.get('selection_threshold', 0.01),
            max_features=selection_params.get('max_features', 100),
            l1_ratio=selection_params.get('l1_ratio', 1.0),
            cv_folds=selection_params.get('cv_folds', 5),
            random_state=42,
            scaling_before_selection=selection_params.get('scaling_before_selection', True),
            selection_metric=selection_params.get('selection_metric', 'coefficient'),
            logger=self.logger
        )
        
        # 拟合特征选择器
        self.global_selector.fit(
            preprocessed_data['train_X'], 
            preprocessed_data['train_y']
        )
        
        # 转换所有数据集
        train_X_selected = self.global_selector.transform(preprocessed_data['train_X'])
        test_X_selected = self.global_selector.transform(preprocessed_data['test_X'])
        val_X_selected = self.global_selector.transform(preprocessed_data['val_X'])
        
        # 保存全局特征选择后的数据
        self.global_selected_data = {
            'train_X': train_X_selected,
            'train_y': preprocessed_data['train_y'],
            'test_X': test_X_selected,
            'test_y': preprocessed_data['test_y'],
            'val_X': val_X_selected,
            'val_y': preprocessed_data['val_y'],
            'original_dim': preprocessed_data['train_X'].shape[1],
            'selected_dim': train_X_selected.shape[1]
        }
        
        
        # 评估特征选择的稳定性
        stability_metrics = self.global_selector.evaluate_stability(
            preprocessed_data['train_X'], 
            preprocessed_data['train_y']
        )
        
        # 将稳定性指标保存到全局选择器中，以便后续可视化使用
        self.global_selector.selection_frequency = stability_metrics['selection_frequency']
        self.global_selector.jaccard_matrix = stability_metrics['jaccard_matrix']

        # 保存全局特征选择结果
        global_fs_dir = os.path.join(self.base_dir, "global_feature_selection")
        os.makedirs(global_fs_dir, exist_ok=True)
        
        # 保存特征选择器
        self.global_selector.save(os.path.join(global_fs_dir, "global_selector.pkl"))
        
        # 可视化特征选择结果
        visualize_feature_selection(
            self.global_selector, 
            save_path=os.path.join(global_fs_dir, "feature_selection_visualization.png")
        )
        
        # 可视化特征选择稳定性
        visualize_feature_stability(
            stability_metrics, 
            save_path=os.path.join(global_fs_dir, "feature_stability_visualization.png")
        )
        
        self.logger.info(f"全局特征选择完成，从 {preprocessed_data['train_X'].shape[1]} 个特征中选择了 "
                      f"{train_X_selected.shape[1]} 个特征",
                      f"Global feature selection completed, selected {train_X_selected.shape[1]} "
                      f"features from {preprocessed_data['train_X'].shape[1]}")
        
        return self.global_selector
    
    def run_experiment(self, params, force_rerun=False):
        """
        运行单个实验，支持特征选择
        
        参数:
            params: 实验参数字典
            force_rerun: 是否强制重新运行已完成的实验
            
        返回:
            experiment_id: 实验ID
            results: 实验结果
        """
        # 生成实验ID
        experiment_id = self._generate_experiment_id(params)
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        
        # 检查是否已完成实验
        if os.path.exists(experiment_dir) and not force_rerun:
            status_path = os.path.join(experiment_dir, "status.json")
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    if status.get('status') == 'completed':
                        self.logger.info(f"实验 {experiment_id} 已完成，跳过", 
                                       f"Experiment {experiment_id} already completed, skipping")
                        return experiment_id, self._load_results(experiment_dir)
        
        # 创建实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 保存参数
        with open(os.path.join(experiment_dir, "params.json"), 'w') as f:
            json.dump(params, f, indent=4)
        
        # 更新状态为运行中
        with open(os.path.join(experiment_dir, "status.json"), 'w') as f:
            json.dump({"status": "running", "start_time": str(datetime.now())}, f, indent=4)
        
        # 记录实验开始
        self.logger.info(f"开始实验 {experiment_id}", f"Starting experiment {experiment_id}")
        self.logger.info(f"参数: {params}", f"Parameters: {params}")
        
        try:
            # 判断是否使用特征选择
            use_feature_selection = params.get('feature_selection') is not None
            
            # 处理数据
            start_time = time.time()
            
            # 根据特征选择模式处理数据
            if use_feature_selection and self.feature_selection_mode == 'global':
                # 全局特征选择模式
                if self.global_selector is None:
                    # 如果全局选择器尚未创建，则创建
                    self.apply_global_feature_selection(params)
                
                # 使用全局选择后的数据
                processed_data = {
                    'train_X': self.global_selected_data['train_X'],
                    'train_y': self.global_selected_data['train_y'],
                    'test_X': self.global_selected_data['test_X'],
                    'test_y': self.global_selected_data['test_y'],
                    'val_X': self.global_selected_data['val_X'],
                    'val_y': self.global_selected_data['val_y']
                }
                
                # 记录原始和选择后的特征维度
                original_dim = self.global_selected_data['original_dim']
                selected_dim = self.global_selected_data['selected_dim']
                
            elif use_feature_selection and self.feature_selection_mode == 'per_experiment':
                # 每次实验单独进行特征选择
                processed_data = self.data_loader.preprocess_data_with_feature_selection(
                    apply_pca=params.get('apply_pca', False),
                    n_components=params.get('n_components', 50),
                    normalization=params.get('normalization', None),
                    class_balance=params.get('class_balance', False),
                    target_samples=params.get('target_samples', 1000),
                    feature_selection=params.get('feature_selection'),
                    selection_mode=params.get('selection_mode', 'threshold'),
                    selection_threshold=params.get('selection_threshold', 0.01),
                    max_features=params.get('max_features', 100),
                    l1_ratio=params.get('l1_ratio', 1.0),
                    cv_folds=params.get('cv_folds', 5),
                    scaling_before_selection=params.get('scaling_before_selection', True),
                    selection_metric=params.get('selection_metric', 'coefficient')
                )
                
                # 获取特征选择器并保存
                selector = processed_data.pop('feature_selector')
                selector.save(os.path.join(experiment_dir, "feature_selector.pkl"))
                
                # 可视化特征选择结果
                visualize_feature_selection(
                    selector, 
                    save_path=os.path.join(experiment_dir, "feature_selection_visualization.png")
                )
                
                # 记录原始和选择后的特征维度
                original_dim = selector.feature_importance.shape[0]
                selected_dim = len(selector.selected_indices)
                
            else:
                # 不使用特征选择，正常处理数据
                processed_data = self.data_loader.preprocess_data(
                    apply_pca=params.get('apply_pca', False),
                    n_components=params.get('n_components', 50),
                    normalization=params.get('normalization', None),
                    class_balance=params.get('class_balance', False),
                    target_samples=params.get('target_samples', 1000)
                )
                
                # 原始和选择后的特征维度相同
                original_dim = processed_data['train_X'].shape[1]
                selected_dim = original_dim
            
            # 训练伪逆模型
            model = PseudoInverseModel(num_classes=102, logger=self.logger)
            model.fit(
                processed_data['train_X'], 
                processed_data['train_y'],
                regularization=params.get('regularization', None),
                alpha=params.get('alpha', 0.0)
            )
            
            train_time = time.time() - start_time
            
            # 保存模型
            model.save(os.path.join(experiment_dir, "model.pkl"))
            
            # 评估模型
            eval_start_time = time.time()
            evaluator = ModelEvaluator(logger=self.logger)
            
            # 评估训练集
            train_result = evaluator.evaluate(
                model, processed_data['train_X'], processed_data['train_y'], "train")
            
            # 评估测试集
            test_result = evaluator.evaluate(
                model, processed_data['test_X'], processed_data['test_y'], "test")
            
            # 评估验证集
            val_result = evaluator.evaluate(
                model, processed_data['val_X'], processed_data['val_y'], "val")
            
            evaluation_time = time.time() - eval_start_time
            
            # 生成可视化
            evaluator.visualize_performance(save_dir=experiment_dir)
            
            # 特征重要性可视化
            visualize_feature_importance(
                model, top_n=30, 
                save_path=os.path.join(experiment_dir, "feature_importance.png"))
            
            # 权重分布可视化
            visualize_weight_distribution(
                model, save_path=os.path.join(experiment_dir, "weight_distribution.png"))
            # 保存评估结果
            results = {
                'train': train_result,
                'test': test_result,
                'val': val_result,
                'train_time': train_time,
                'evaluation_time': evaluation_time,
                'original_dim': original_dim,
                'selected_dim': selected_dim
            }

            # 保存评估结果
            self._save_results(experiment_dir, results)

            # 更新状态为已完成
            with open(os.path.join(experiment_dir, "status.json"), 'w') as f:
                json.dump({
                    "status": "completed", 
                    "start_time": str(datetime.now()),
                    "end_time": str(datetime.now()),
                    "train_time": train_time,
                    "evaluation_time": evaluation_time,
                    "original_dim": original_dim,
                    "selected_dim": selected_dim
                }, f, indent=4)
            
            # 更新实验日志
            self._update_experiment_log_with_fs(
                experiment_id, params, train_result, test_result, val_result, 
                original_dim, selected_dim, train_time, evaluation_time, "completed")
            
            self.logger.info(f"实验 {experiment_id} 完成", f"Experiment {experiment_id} completed")
            
            return experiment_id, results
            
        except Exception as e:
            # 记录错误
            self.logger.error(f"实验 {experiment_id} 失败: {str(e)}", 
                            f"Experiment {experiment_id} failed: {str(e)}")
            
            # 更新状态为失败
            with open(os.path.join(experiment_dir, "status.json"), 'w') as f:
                json.dump({
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, f, indent=4)
            
            # 更新实验日志
            self._update_experiment_log_with_fs(
                experiment_id, params, None, None, None, 0, 0, 0, 0, "failed")
            
            raise e
    
    def _update_experiment_log_with_fs(self, experiment_id, params, train_result, test_result, 
                                    val_result, original_dim, selected_dim, train_time, 
                                    evaluation_time, status):
        """更新包含特征选择信息的实验日志"""
        # 提取评估指标
        train_acc = train_result['accuracy'] if train_result else float('nan')
        test_acc = test_result['accuracy'] if test_result else float('nan')
        val_acc = val_result['accuracy'] if val_result else float('nan')
        
        train_f1 = train_result['f1_macro'] if train_result else float('nan')
        test_f1 = test_result['f1_macro'] if test_result else float('nan')
        val_f1 = val_result['f1_macro'] if val_result else float('nan')
        
        # 计算特征选择比例
        selection_ratio = selected_dim / original_dim if original_dim > 0 else 1.0
        
        # 处理None值，统一转换为字符串'none'用于日志
        feature_selection = 'none' if params.get('feature_selection') is None else params.get('feature_selection')
        selection_mode = 'none' if params.get('selection_mode') is None else params.get('selection_mode')
        regularization = 'none' if params.get('regularization') is None else params.get('regularization')
        normalization = 'none' if params.get('normalization') is None else params.get('normalization')
        
        # 准备日志条目
        log_entry = (f"{experiment_id},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"\
                    f"{params.get('apply_pca', False)},{params.get('n_components', 0)},"\
                    f"{normalization},{params.get('class_balance', False)},"\
                    f"{params.get('target_samples', 0)},{regularization},"\
                    f"{params.get('alpha', 0.0)},"\
                    f"{feature_selection},"\
                    f"{selection_mode},"\
                    f"{params.get('selection_threshold', 0.0)},"\
                    f"{params.get('max_features', 0)},"\
                    f"{params.get('l1_ratio', 0.0)},"\
                    f"{train_acc},{test_acc},{val_acc},{train_f1},{test_f1},{val_f1},"\
                    f"{original_dim},{selected_dim},{selection_ratio},"\
                    f"{train_time},{evaluation_time},{status}\n")
        
        # 追加到日志文件
        with open(self.experiment_log_path, 'a') as f:
            f.write(log_entry)
    
    def generate_summary_report_with_fs(self, top_n=10):
        """
        生成包含特征选择信息的汇总报告
        
        参数:
            top_n: 展示的顶部实验数量
        """
        # 读取实验日志
        if not os.path.exists(self.experiment_log_path):
            self.logger.warning("实验日志不存在", "Experiment log does not exist")
            return
        
        try:
            log_df = pd.read_csv(self.experiment_log_path)
        except:
            self.logger.warning("无法读取实验日志", "Cannot read experiment log")
            return
        
        # 过滤成功的实验
        log_df = log_df[log_df['status'] == 'completed']
        
        if len(log_df) == 0:
            self.logger.warning("没有已完成的实验", "No completed experiments")
            return
        
        # 按测试集准确率排序
        log_df_sorted = log_df.sort_values('test_accuracy', ascending=False)
        
        # 创建报告目录
        report_dir = os.path.join(self.base_dir, "summary_report_with_fs")
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存排序后的实验日志
        log_df_sorted.to_csv(os.path.join(report_dir, "experiments_sorted.csv"), index=False)
        
        # 生成顶部实验表格
        top_df = log_df_sorted.head(top_n)
        
        # 生成HTML报告
        html_report = f"""
        <html>
        <head>
            <title>Pseudo-Inverse Experiments with Feature Selection Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                h2 {{ color: #666699; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .chart {{ margin: 10px; max-width: 600px; }}
            </style>
        </head>
        <body>
            <h1>Pseudo-Inverse Experiments with Feature Selection Summary Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total experiments: {len(log_df)}</p>
            
            <h2>Top {top_n} Experiments by Test Accuracy</h2>
            <table>
                <tr>
                    <th>Experiment ID</th>
                    <th>Test Accuracy</th>
                    <th>Test F1</th>
                    <th>Val Accuracy</th>
                    <th>PCA</th>
                    <th>Feature Selection</th>
                    <th>Original Features</th>
                    <th>Selected Features</th>
                    <th>Selection Ratio</th>
                </tr>
        """
        
        for _, row in top_df.iterrows():
            html_report += f"""
                <tr>
                    <td>{row['experiment_id']}</td>
                    <td>{row['test_accuracy']:.4f}</td>
                    <td>{row['test_f1']:.4f}</td>
                    <td>{row['val_accuracy']:.4f}</td>
                    <td>{'Yes' if row['apply_pca'] else 'No'}</td>
                    <td>{row['feature_selection']}</td>
                    <td>{int(row['original_features'])}</td>
                    <td>{int(row['selected_features'])}</td>
                    <td>{row['selection_ratio']:.2f}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Feature Selection Impact Analysis</h2>
            <div class="chart-container">
        """
        
        # 生成特征选择影响可视化
        # 1. 特征选择方法对准确率的影响
        if 'feature_selection' in log_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='feature_selection', y='test_accuracy', data=log_df)
            plt.title('Impact of Feature Selection Method on Test Accuracy')
            plt.xlabel('Feature Selection Method')
            plt.ylabel('Test Accuracy')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            fs_impact_path = os.path.join(report_dir, "feature_selection_impact.png")
            plt.savefig(fs_impact_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(fs_impact_path, self.base_dir)}" alt="Feature Selection Impact" width="100%">
                </div>
            """
        
        # 2. 特征选择比例与准确率的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(log_df['selection_ratio'], log_df['test_accuracy'], alpha=0.7)
        plt.title('Relationship Between Feature Selection Ratio and Test Accuracy')
        plt.xlabel('Feature Selection Ratio')
        plt.ylabel('Test Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        ratio_impact_path = os.path.join(report_dir, "selection_ratio_impact.png")
        plt.savefig(ratio_impact_path, dpi=300)
        plt.close()
        
        html_report += f"""
            <div class="chart">
                <img src="{os.path.relpath(ratio_impact_path, self.base_dir)}" alt="Selection Ratio Impact" width="100%">
            </div>
        """
        
        # 3. 使用不同L1比例的比较
        if 'l1_ratio' in log_df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='l1_ratio', y='test_accuracy', data=log_df)
            plt.title('Impact of L1 Ratio on Test Accuracy')
            plt.xlabel('L1 Ratio')
            plt.ylabel('Test Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            l1_impact_path = os.path.join(report_dir, "l1_ratio_impact.png")
            plt.savefig(l1_impact_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(l1_impact_path, self.base_dir)}" alt="L1 Ratio Impact" width="100%">
                </div>
            """
        
        # 4. 特征选择前后对比（使用和不使用特征选择）
        with_fs = log_df[log_df['feature_selection'] != 'none']
        without_fs = log_df[log_df['feature_selection'] == 'none']
        
        if len(with_fs) > 0 and len(without_fs) > 0:
            plt.figure(figsize=(12, 6))
            
            # 准确率对比
            plt.subplot(1, 2, 1)
            data = {
                'With FS': with_fs['test_accuracy'].mean(),
                'Without FS': without_fs['test_accuracy'].mean()
            }
            plt.bar(data.keys(), data.values())
            plt.ylabel('Average Test Accuracy')
            plt.title('Accuracy With vs Without Feature Selection')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 特征数量对比
            plt.subplot(1, 2, 2)
            data = {
                'Original': log_df['original_features'].mean(),
                'After Selection': with_fs['selected_features'].mean()
            }
            plt.bar(data.keys(), data.values())
            plt.ylabel('Average Feature Count')
            plt.title('Feature Count Before vs After Selection')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            fs_comparison_path = os.path.join(report_dir, "feature_selection_comparison.png")
            plt.savefig(fs_comparison_path, dpi=300)
            plt.close()
            
            html_report += f"""
                <div class="chart">
                    <img src="{os.path.relpath(fs_comparison_path, self.base_dir)}" alt="Feature Selection Comparison" width="100%">
                </div>
            """
        
        html_report += """
            </div>
            
            <h2>Best Experiment Details</h2>
        """
        
        # 获取最佳实验的详细信息
        if len(top_df) > 0:
            best_exp_id = top_df.iloc[0]['experiment_id']
            best_exp_dir = os.path.join(self.base_dir, best_exp_id)
            
            # 复制最佳实验的图表到报告目录
            important_images = [
                'feature_selection_visualization.png',
                'feature_importance.png',
                'weight_distribution.png', 
                'performance_comparison.png', 
                'confusion_matrix_test.png'
            ]
            
            for img_file in important_images:
                img_path = os.path.join(best_exp_dir, img_file)
                if os.path.exists(img_path):
                    dest_path = os.path.join(report_dir, f"best_{img_file}")
                    import shutil
                    shutil.copy(img_path, dest_path)
                    
                    img_title = img_file.replace('_', ' ').replace('.png', '').title()
                    html_report += f"""
                        <div class="chart">
                            <h3>{img_title}</h3>
                            <img src="{os.path.relpath(dest_path, self.base_dir)}" alt="{img_file}" width="100%">
                        </div>
                    """
        
        html_report += """
            <h2>Conclusion and Recommendations</h2>
            <p>Based on the experimental results, here are the key findings related to feature selection:</p>
            <ul>
        """
        
        # 添加特征选择相关结论
        # 特征选择方法比较
        if 'feature_selection' in log_df.columns:
            fs_impact = log_df.groupby('feature_selection')['test_accuracy'].mean()
            best_fs = fs_impact.idxmax()
            html_report += f"""
                <li>Best feature selection method: {best_fs} (Avg. accuracy: {fs_impact[best_fs]:.4f})</li>
            """
        
        # 特征选择比例分析
        avg_ratio = with_fs['selection_ratio'].mean() if len(with_fs) > 0 else 0
        html_report += f"""
            <li>Average feature selection ratio: {avg_ratio:.2f} ({avg_ratio*100:.1f}% of original features)</li>
        """
        
        # 特征选择前后对比
        if len(with_fs) > 0 and len(without_fs) > 0:
            acc_diff = with_fs['test_accuracy'].mean() - without_fs['test_accuracy'].mean()
            html_report += f"""
                <li>Feature selection {'improves' if acc_diff > 0 else 'reduces'} model performance by {abs(acc_diff):.4f} on average</li>
            """
        
        # 最佳特征选择参数
        if len(top_df) > 0 and top_df.iloc[0]['feature_selection'] != 'none':
            best_row = top_df.iloc[0]
            html_report += f"""
                <li>Best feature selection parameters:
                    <ul>
                        <li>Method: {best_row['feature_selection']}</li>
                        <li>Selection mode: {best_row['selection_mode']}</li>
                        <li>Threshold/Max features: {best_row['selection_threshold'] if best_row['selection_mode'] == 'threshold' else best_row['max_features']}</li>
                        <li>L1 ratio: {best_row['l1_ratio']}</li>
                        <li>Feature reduction: from {int(best_row['original_features'])} to {int(best_row['selected_features'])} features ({best_row['selection_ratio']:.2f} ratio)</li>
                    </ul>
                </li>
            """
        
        html_report += """
            </ul>
        </body>
        </html>
        """
        
        # 保存HTML报告
        with open(os.path.join(report_dir, "summary_report_with_fs.html"), 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"特征选择汇总报告已生成: {os.path.join(report_dir, 'summary_report_with_fs.html')}", 
                       f"Feature selection summary report generated: {os.path.join(report_dir, 'summary_report_with_fs.html')}")