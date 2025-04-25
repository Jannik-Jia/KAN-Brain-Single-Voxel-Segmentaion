import os
import json
import itertools
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.logging_utils import Logger


class HyperparameterOptimizer:
    """超参数优化工具"""
    
    def __init__(self, base_config_path, output_dir, logger=None):
        """
        初始化超参数优化器
        
        参数:
            base_config_path: 基础配置文件路径
            output_dir: 输出目录
            logger: 日志记录器
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志记录器
        if logger:
            self.logger = logger
        else:
            from utils.logging_utils import Logger
            log_manager = Logger("HyperparameterOptimizer", log_dir="logs/utils")
            self.logger = log_manager.get_logger()
        
        # 加载基础配置
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)
            
        self.logger.info(f"Loaded base config from {base_config_path}")
        self.results = []
    

    # 在HyperparameterOptimizer类中添加专用方法
    def get_next_bayesian_config(self, iteration):
        """获取贝叶斯优化的下一个配置"""
        if not hasattr(self, 'optimizer'):
            raise RuntimeError("贝叶斯优化器未初始化")
            
        suggested_params_list = self.optimizer.ask(n_points=1)
        suggested_params = suggested_params_list[0]
        
        # 将参数列表转换为字典
        params = {name: value for name, value in zip(self.dimension_names, suggested_params)}
        
        # 创建配置
        return self._create_config(params, iteration)

    def grid_search(self, param_grid, max_combinations=None):
        """
        执行网格搜索
        
        参数:
            param_grid: 参数网格字典，例如 {'learning_rate': [1e-3, 1e-4], 'dropout': [0.3, 0.5]}
            max_combinations: 最大组合数，若不指定则测试所有组合
            
        返回:
            configs: 生成的配置列表
        """
        self.logger.info(f"Performing grid search with parameters: {param_grid}")
        
        # 创建参数网格
        grid = list(ParameterGrid(param_grid))
        
        # 如果指定了最大组合数，随机选择子集
        if max_combinations and len(grid) > max_combinations:
            self.logger.info(f"Sampling {max_combinations} configurations from {len(grid)} possibilities")
            indices = np.random.choice(len(grid), size=max_combinations, replace=False)
            grid = [grid[i] for i in indices]
        
        # 生成配置
        configs = []
        for i, params in enumerate(grid):
            config = self._create_config(params, i)
            configs.append(config)
        
        self.logger.info(f"Generated {len(configs)} configurations for grid search")
        return configs
    
    def random_search(self, param_distributions, n_iter=10):
        """
        执行随机搜索
        
        参数:
            param_distributions: 参数分布字典
            n_iter: 迭代次数
            
        返回:
            configs: 生成的配置列表
        """
        self.logger.info(f"Performing random search with {n_iter} iterations")
        
        # 创建参数采样
        param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))
        
        # 生成配置
        configs = []
        for i, params in enumerate(param_list):
            config = self._create_config(params, i)
            configs.append(config)
        
        self.logger.info(f"Generated {len(configs)} configurations for random search")
        return configs
    
    def bayesian_optimization(self, param_space, n_iter=10, early_stopping=5, improvement_threshold=0.001):
        """
        执行贝叶斯优化
        
        参数:
            param_space: 参数空间，格式为 {'param_name': (low, high, 'prior')} 或 {'param_name': [choice1, choice2, ...]}
            n_iter: 迭代次数
            early_stopping: 连续多少次无改进后停止
            improvement_threshold: 改进阈值，小于此值视为无改进
            
        返回:
            configs: 生成的配置列表
        """
        try:
            from skopt import Optimizer
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            self.logger.error("scikit-optimize未安装，请使用'pip install scikit-optimize'安装")
            return []
        
        self.logger.info(f"执行贝叶斯优化，迭代次数: {n_iter}，早停阈值: {early_stopping}")
        
        # 转换参数空间为skopt空间
        dimensions = []
        dimension_names = []
        
        for name, space in param_space.items():
            dimension_names.append(name)
            
            if isinstance(space, tuple) and len(space) >= 2:
                # 连续参数
                low, high = space[:2]
                prior = 'uniform'
                if len(space) > 2:
                    prior = space[2]
                    
                if isinstance(low, int) and isinstance(high, int):
                    dimensions.append(Integer(low, high))
                elif prior == 'log-uniform':
                    dimensions.append(Real(low, high, prior='log-uniform'))
                else:
                    dimensions.append(Real(low, high, prior=prior))
                    
            elif isinstance(space, list):
                # 分类参数
                dimensions.append(Categorical(space))
            else:
                self.logger.warning(f"无法识别的参数空间格式 '{name}': {space}，使用默认范围")
                dimensions.append(Real(0.0, 1.0))
        
        self.logger.info(f"参数空间: {dimension_names}")
        
        # 创建贝叶斯优化器
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",  # 高斯过程
            acq_func="EI",        # 期望改进
            acq_optimizer="auto",
            random_state=42
        )
        
        # 创建初始配置（随机采样1个）
        configs = []
        
        # 首先尝试使用默认配置
        default_params = {}
        for name, space in param_space.items():
            if isinstance(space, tuple) and len(space) >= 2:
                # 对于连续参数，使用范围中点
                low, high = space[:2]
                default_params[name] = (low + high) / 2
            elif isinstance(space, list):
                # 对于分类参数，使用第一个值
                default_params[name] = space[0]
        
        default_config = self._create_config(default_params, 0)
        configs.append(default_config)
        
        # 添加早停相关变量
        best_score = float('inf')  # 因为我们在最小化负的F1
        no_improvement_count = 0
        
        # 然后生成n_iter-1个优化建议
        for i in range(1, n_iter):
            # 让贝叶斯优化器建议下一组参数
            suggested_params_list = optimizer.ask(n_points=1)  # 要求1个点
            suggested_params = suggested_params_list[0]
            
            # 将参数列表转换为字典
            params = {name: value for name, value in zip(dimension_names, suggested_params)}
            
            # 创建配置
            config = self._create_config(params, i)
            configs.append(config)
        
        self.logger.info(f"生成了{len(configs)}个配置用于贝叶斯优化")
        
        # 注册更新优化器的方法（当获得评估结果后调用）
        def update_optimizer(config_id, metrics):
            """更新贝叶斯优化器"""
            # 找到配置
            nonlocal optimizer, dimension_names, best_score, no_improvement_count
            config = next((c for c in configs if c['id'] == config_id), None)
            if not config:
                self.logger.warning(f"无法找到配置 {config_id} 以更新优化器")
                return False
            
            # 获取参数值
            params = [config['params'].get(name) for name in dimension_names]
            
            # 获取指标值（最大化F1分数）
            current_score = -metrics.get('f1_macro', 0)  # 负号因为optimizer最小化目标函数
            
            # 告诉优化器结果
            optimizer.tell(params, current_score)
            self.logger.info(f"更新贝叶斯优化器: config_id={config_id}, f1_macro={-current_score}")
            
            # 检查是否有改进
            if current_score < best_score - improvement_threshold:  # 比最佳分数更好
                improvement = best_score - current_score
                best_score = current_score
                no_improvement_count = 0
                self.logger.info(f"发现新的最佳分数: {-best_score:.6f}, 改进了: {improvement:.6f}")
                return False  # 不需要早停
            else:
                no_improvement_count += 1
                self.logger.info(f"未发现显著改进，当前无改进次数: {no_improvement_count}/{early_stopping}")
                
                # 检查是否需要早停
                if no_improvement_count >= early_stopping:
                    self.logger.info(f"触发早停条件: 连续{early_stopping}次迭代无显著改进")
                    return True  # 建议早停
                return False
        
        # 保存更新方法以供外部使用
        self.update_bayesian_optimizer = update_optimizer
        
        # 添加检查是否应该早停的方法
        self.should_stop_early = lambda: no_improvement_count >= early_stopping
        
        return configs


    

    def _create_config(self, params, index):
        """创建具体配置"""
        # 复制基础配置
        config = deepcopy(self.base_config)
        
        # 更新配置
        for key, value in params.items():
            # 处理嵌套键，例如 'training.learning_rate'
            if '.' in key:
                parts = key.split('.')
                target = config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                config[key] = value
        
        # 确保loss配置部分存在
        if 'loss' not in config:
            config['loss'] = {
                'aux_weight': 0.3,
                'attn_reg_weight': 0.01,
                'class_balance_method': 'effective_samples',
                'focal_gamma': 2.0,
                'cb_beta': 0.9999
            }
        
        # 设置配置路径和ID
        config_id = f"config_{index:03d}"
        config_path = os.path.join(self.output_dir, f"{config_id}.json")
        
        # 保存配置
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        self.logger.info(f"Created configuration {config_id}: {params}")
        
        return {
            'id': config_id,
            'path': config_path,
            'params': params,
            'config': config
        }


    
    def record_result(self, config_id, metrics):
        """
        记录训练结果，并在贝叶斯优化时更新优化器
        
        参数:
            config_id: 配置ID
            metrics: 性能指标
        """
        result = {
            'config_id': config_id,
            'metrics': metrics
        }
        
        self.results.append(result)
        
        # 更新结果文件
        results_path = os.path.join(self.output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        self.logger.info(f"Recorded result for {config_id}: {metrics}")
        
        # 如果正在进行贝叶斯优化，更新优化器
        if hasattr(self, 'update_bayesian_optimizer') and callable(self.update_bayesian_optimizer):
            try:
                self.update_bayesian_optimizer(config_id, metrics)
                self.logger.info(f"已更新贝叶斯优化器，配置 {config_id}")
            except Exception as e:
                self.logger.error(f"更新贝叶斯优化器时出错: {e}")

    
    def get_best_config(self, metric='f1_macro', higher_is_better=True):
        """
        获取最佳配置
        
        参数:
            metric: 用于比较的指标
            higher_is_better: 是否值越高越好
            
        返回:
            best_config: 最佳配置
        """
        if not self.results:
            self.logger.warning("No results recorded yet")
            return None
        
        # 根据指标排序
        sorted_results = sorted(
            self.results, 
            key=lambda x: x['metrics'].get(metric, 0), 
            reverse=higher_is_better
        )
        
        best_result = sorted_results[0]
        best_config_id = best_result['config_id']
        best_config_path = os.path.join(self.output_dir, f"{best_config_id}.json")
        
        # 加载最佳配置
        with open(best_config_path, 'r') as f:
            best_config = json.load(f)
        
        self.logger.info(f"Best config is {best_config_id} with {metric}={best_result['metrics'].get(metric, 0)}")
        
        return best_config
    
    def visualize_results(self):
        """可视化优化结果"""
        if not self.results:
            self.logger.warning("No results to visualize")
            return
        
        # 提取主要指标
        config_ids = [r['config_id'] for r in self.results]
        accuracy = [r['metrics'].get('accuracy', 0) for r in self.results]
        f1_macro = [r['metrics'].get('f1_macro', 0) for r in self.results]
        
        # 绘制性能对比图
        plt.figure(figsize=(10, 6))
        x = range(len(config_ids))
        plt.plot(x, accuracy, 'o-', label='Accuracy')
        plt.plot(x, f1_macro, 's-', label='F1-macro')
        plt.xticks(x, config_ids, rotation=45)
        plt.xlabel('Configuration')
        plt.ylabel('Performance')
        plt.title('Hyperparameter Optimization Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.output_dir, "optimization_results.png")
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Optimization results visualization saved to {save_path}")
        
        # 如果参数空间是二维的，可以绘制热图
        if len(self.results) > 0 and len(self.results[0]['params']) == 2:
            param_names = list(self.results[0]['params'].keys())
            param1_values = []
            param2_values = []
            f1_scores = []
            
            for result in self.results:
                param1_values.append(result['params'][param_names[0]])
                param2_values.append(result['params'][param_names[1]])
                f1_scores.append(result['metrics'].get('f1_macro', 0))
            
            # 绘制热图
            plt.figure(figsize=(10, 8))
            plt.scatter(param1_values, param2_values, c=f1_scores, cmap='viridis', 
                      s=100, alpha=0.8, edgecolors='k')
            plt.colorbar(label='F1-macro')
            plt.xlabel(param_names[0])
            plt.ylabel(param_names[1])
            plt.title('Hyperparameter Space F1 Scores')
            plt.grid(True, alpha=0.3)
            
            # 保存热图
            heatmap_path = os.path.join(self.output_dir, "param_space_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            
            self.logger.info(f"Parameter space heatmap saved to {heatmap_path}")


    def generate_initial_config(self, param_space):
        """生成初始配置"""
        # 使用默认参数
        default_params = {}
        for name, space in param_space.items():
            if isinstance(space, tuple) and len(space) >= 2:
                # 对于连续参数，使用范围中点
                low, high = space[:2]
                default_params[name] = (low + high) / 2
            elif isinstance(space, list):
                # 对于分类参数，使用第一个值
                default_params[name] = space[0]
        
        return self._create_config(default_params, 0)

    def generate_next_config_bayesian(self, index):
        """基于贝叶斯优化器生成下一个配置"""
        if not hasattr(self, 'optimizer') or not hasattr(self, 'dimension_names'):
            self.logger.error("贝叶斯优化器未初始化")
            raise RuntimeError("贝叶斯优化器未初始化")
        
        # 让优化器建议下一组参数
        suggested_params_list = self.optimizer.ask(n_points=1)
        suggested_params = suggested_params_list[0]
        
        # 将参数列表转换为字典
        params = {name: value for name, value in zip(self.dimension_names, suggested_params)}
        
        # 创建配置
        return self._create_config(params, index)