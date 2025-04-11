# src/utils.py

import numpy as np

def create_feature_selection_param_grid():
    """创建专注于PCA的参数网格"""
    
    # 基本参数网格 - 专注于PCA
    base_param_grid = {
        'apply_pca': [True],  # 只使用PCA
        'n_components': [50, 80, 100, 150, 200, 250, 300],  # 更多PCA组件选项以找到最佳点
        'normalization': ['standard', 'minmax', None],
        'class_balance': [False, True],
        'regularization': [None, 'l2', 'truncated'],
        'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    }
    
    # 特征选择参数网格 - 不使用特征选择
    feature_selection_param_grid = {
        'feature_selection': [None],  # 不使用特征选择
        'selection_mode': ['threshold'],
        'selection_threshold': [0.01],
        'max_features': [100],
        'l1_ratio': [1.0],
        'scaling_before_selection': [True],
        'selection_metric': ['coefficient']
    }
    
    # 合并参数网格
    full_param_grid = {**base_param_grid, **feature_selection_param_grid}
    
    return full_param_grid

def create_feature_selection_without_pca_param_grid():
    """创建不使用PCA，专注于直接特征选择的参数网格"""
    
    # 基本参数网格 - 不使用PCA
    base_param_grid = {
        'apply_pca': [False],  # 不使用PCA
        'normalization': ['standard', 'minmax', None],  # 包括不使用标准化
        'class_balance': [False, True],
        'regularization': [None, 'l2', 'truncated'],
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]  # 增加更多正则化强度值
    }
    
    # 特征选择参数网格 - 使用多种算法，参数和迭代次数
    feature_selection_param_grid = {
        'feature_selection': ['lasso', 'elastic_net_torch'],  # 添加PyTorch实现
        'selection_mode': ['threshold', 'fixed'],
        'selection_threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],  # 增加更多阈值
        'max_features': [50, 100, 150, 200, 250, 300, 341],  # 增加更多特征数选项
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # 增加更多L1比例
        'scaling_before_selection': [True],
        'selection_metric': ['coefficient'],
        'max_iter': [500, 1000, 2000, 5000],  # 添加不同的最大迭代次数
        'tol': [1e-4, 1e-5, 1e-6]  # 添加不同的收敛阈值
    }
    
    # 合并参数网格
    full_param_grid = {**base_param_grid, **feature_selection_param_grid}
    
    return full_param_grid

def sample_parameter_combinations_for_feature_selection(param_grid, max_samples=100, random_state=42):
    """
    从特征选择参数网格中采样有效的参数组合，专注于直接特征选择
    
    参数:
        param_grid: 参数网格
        max_samples: 最大采样数量
        random_state: 随机种子
        
    返回:
        param_combinations: 参数组合列表
    """
    np.random.seed(random_state)
    
    # 提取所有参数可能的值
    param_values = list(param_grid.values())
    param_keys = list(param_grid.keys())
    
    # 计算所有可能组合的数量
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"特征选择可能的参数组合数: {total_combinations}")
    
    # 采样参数组合
    param_combinations = []
    
    # 首先添加几个基准配置
    # 1. LASSO + 标准化 + 阈值选择
    lasso_threshold_config = {
        'apply_pca': False,
        'normalization': 'standard',
        'class_balance': False,
        'regularization': None,
        'alpha': 0.01,
        'feature_selection': 'lasso',
        'selection_mode': 'threshold',
        'selection_threshold': 0.01,
        'max_features': 100,
        'l1_ratio': 1.0,
        'scaling_before_selection': True,
        'selection_metric': 'coefficient'
    }
    param_combinations.append(lasso_threshold_config)
    
    # 2. LASSO + 标准化 + 固定特征数量
    lasso_fixed_config = lasso_threshold_config.copy()
    lasso_fixed_config['selection_mode'] = 'fixed'
    lasso_fixed_config['max_features'] = 100
    param_combinations.append(lasso_fixed_config)
    
    # 3. 弹性网络 + 标准化 + 阈值选择
    elastic_threshold_config = lasso_threshold_config.copy()
    elastic_threshold_config['feature_selection'] = 'elastic_net_torch'
    elastic_threshold_config['l1_ratio'] = 0.5
    param_combinations.append(elastic_threshold_config)
    
    # 4. 弹性网络 + 标准化 + 固定特征数量
    elastic_fixed_config = elastic_threshold_config.copy()
    elastic_fixed_config['selection_mode'] = 'fixed'
    elastic_fixed_config['max_features'] = 100
    param_combinations.append(elastic_fixed_config)

    
    
    # 5. LASSO + 不标准化
    lasso_no_norm_config = lasso_threshold_config.copy()
    lasso_no_norm_config['normalization'] = None
    param_combinations.append(lasso_no_norm_config)
    
    # 优先测试不同的特征选择方法和参数组合
    selection_methods = param_grid['feature_selection']
    selection_modes = param_grid['selection_mode']
    thresholds = param_grid['selection_threshold']
    max_features_options = param_grid['max_features']
    l1_ratios = param_grid['l1_ratio']
    normalization_options = param_grid['normalization']
    
    # 确保每种特征选择方法和模式至少有一些样本
    for method in selection_methods:
        for mode in selection_modes:
            if mode == 'threshold':
                for threshold in thresholds:
                    # 为每种阈值创建至少一个配置
                    if len(param_combinations) < max_samples:
                        config = lasso_threshold_config.copy()
                        config['feature_selection'] = method
                        config['selection_threshold'] = threshold
                        
                        # 随机选择其他参数
                        config['normalization'] = np.random.choice(normalization_options)
                        if method == 'elastic_net_torch':
                            config['l1_ratio'] = np.random.choice(l1_ratios)
                        
                        if config not in param_combinations:
                            param_combinations.append(config)
            else:  # 'fixed' mode
                for max_feat in max_features_options:
                    # 为每种最大特征数创建至少一个配置
                    if len(param_combinations) < max_samples:
                        config = lasso_fixed_config.copy()
                        config['feature_selection'] = method
                        config['max_features'] = max_feat
                        
                        # 随机选择其他参数
                        config['normalization'] = np.random.choice(normalization_options)
                        if method == 'elastic_net_torch':
                            config['l1_ratio'] = np.random.choice(l1_ratios)
                        
                        if config not in param_combinations:
                            param_combinations.append(config)
    
    # 随机填充剩余的配置
    attempts = 0
    max_attempts = total_combinations * 2
    
    while len(param_combinations) < max_samples and attempts < max_attempts:
        attempts += 1
        config = {}
        
        # 采样每个参数
        for key, values in param_grid.items():
            config[key] = np.random.choice(values)
        
        # 如果配置没有出现在之前的组合中，添加它
        if config not in param_combinations:
            param_combinations.append(config)
    
    print(f"总共生成了 {len(param_combinations)} 个特征选择参数组合")
    return param_combinations