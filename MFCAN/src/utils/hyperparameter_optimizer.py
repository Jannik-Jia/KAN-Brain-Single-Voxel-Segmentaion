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
    
    def bayesian_optimization(self, param_space, n_iter=10):
        """
        执行贝叶斯优化
        
        参数:
            param_space: 参数空间，格式为 {'param_name': (low, high, 'prior')} 或 {'param_name': [choice1, choice2, ...]}
            n_iter: 迭代次数
            
        返回:
            configs: 生成的配置列表
        """
        try:
            from skopt import Optimizer
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            self.logger.error("scikit-optimize未安装，请使用'pip install scikit-optimize'安装")
            return []
        
        self.logger.info(f"执行贝叶斯优化，迭代次数: {n_iter}")
        
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
        
        # 创建贝叶斯优化器 - 注意设置不同的随机种子
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",  # 高斯过程
            acq_func="EI",        # 期望改进
            acq_optimizer="auto",
            random_state=np.random.randint(0, 10000)  # 使用随机种子
        )
        
        # 创建初始配置
        configs = []
        
        # 只生成初始随机点，后续配置会在评估后动态生成
        initial_points = min(3, n_iter)  # 最多初始生成3个点，或根据n_iter调整
        
        # 记录操作日志
        self.logger.info(f"生成{initial_points}个初始随机配置")
        
        # 生成初始随机配置
        for i in range(initial_points):
            # 对每个点使用一个不同的随机种子以确保多样性
            random_seed = np.random.randint(0, 10000)
            np.random.seed(random_seed)
            
            # 让贝叶斯优化器建议一个点
            suggested_params_list = optimizer.ask(n_points=1)
            suggested_params = suggested_params_list[0]
            
            # 将参数列表转换为字典
            params = {name: value for name, value in zip(dimension_names, suggested_params)}
            
            # 创建配置
            config = self._create_config(params, i)
            configs.append(config)
            
            # 记录生成的配置
            self.logger.info(f"初始配置 {i+1}/{initial_points}: {params}")
        
        # 保存优化器和必要的上下文供外部使用
        self.optimizer = optimizer
        self.dimension_names = dimension_names
        self.next_config_index = initial_points
        self.total_configs_needed = n_iter
        
        self.logger.info(f"初始阶段生成了{len(configs)}个配置，剩余{n_iter - len(configs)}个将在评估后动态生成")
        
        return configs


    def generate_next_config(self):
        """生成下一个配置，基于之前的评估结果"""
        if not hasattr(self, 'optimizer') or not hasattr(self, 'dimension_names'):
            self.logger.error("贝叶斯优化器未初始化，无法生成下一个配置")
            return None
        
        if not hasattr(self, 'next_config_index'):
            self.next_config_index = 0
        
        # 使用不同的随机种子
        np.random.seed(np.random.randint(0, 10000))
        
        # 获取下一个建议参数
        suggested_params = self.optimizer.ask()[0]
        params = {name: value for name, value in zip(self.dimension_names, suggested_params)}
        
        # 创建配置
        config = self._create_config(params, self.next_config_index)
        self.next_config_index += 1
        
        return config

    def update_optimizer(self, config_id, metrics):
        """更新贝叶斯优化器"""
        if not hasattr(self, 'optimizer') or not hasattr(self, 'dimension_names'):
            self.logger.error("贝叶斯优化器未初始化，无法更新")
            return
        
        # 找到配置
        config = next((r for r in self.results if r['config_id'] == config_id), None)
        if not config:
            self.logger.warning(f"无法找到配置结果 {config_id} 以更新优化器")
            return
        
        # 从结果中获取参数
        params_dict = config.get('params', {})
        if not params_dict:
            # 尝试从配置文件中加载参数
            config_path = os.path.join(self.output_dir, f"{config_id}.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    try:
                        full_config = json.load(f)
                        params = []
                        for name in self.dimension_names:
                            parts = name.split('.')
                            value = full_config
                            for part in parts:
                                if part in value:
                                    value = value[part]
                                else:
                                    value = None
                                    break
                            params.append(value)
                    except Exception as e:
                        self.logger.error(f"无法从配置文件加载参数: {e}")
                        return
            else:
                self.logger.error(f"找不到配置文件 {config_path}")
                return
        else:
            # 使用结果中记录的参数
            params = []
            for name in self.dimension_names:
                params.append(params_dict.get(name))
        
        # 获取指标值（最大化F1分数）
        f1_score = -metrics.get('f1_macro', 0)  # 负号因为optimizer最小化目标函数
        
        # 告诉优化器结果
        try:
            self.optimizer.tell(params, f1_score)
            self.logger.info(f"更新贝叶斯优化器: config_id={config_id}, f1_macro={-f1_score}")
        except Exception as e:
            self.logger.error(f"更新优化器失败: {e}")



    def _create_config(self, params, index):
        """
        创建具体配置
        
        参数:
            params: 参数字典
            index: 配置索引
            
        返回:
            config: 配置字典
        """
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
        记录训练结果
        
        参数:
            config_id: 配置ID
            metrics: 性能指标
        """
        # 加载配置文件，获取参数信息
        config_path = os.path.join(self.output_dir, f"{config_id}.json")
        params = {}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # 如果存在dimension_names，提取相关参数
                if hasattr(self, 'dimension_names'):
                    for name in self.dimension_names:
                        parts = name.split('.')
                        value = config
                        for part in parts:
                            if part in value:
                                value = value[part]
                            else:
                                value = None
                                break
                        params[name] = value
            except Exception as e:
                self.logger.warning(f"无法从配置文件加载参数: {e}")
        
        # 组合结果
        result = {
            'config_id': config_id,
            'metrics': metrics,
            'params': params  # 保存参数信息
        }
        
        self.results.append(result)
        
        # 更新结果文件
        results_path = os.path.join(self.output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        self.logger.info(f"Recorded result for {config_id}: {metrics}")

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