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



def sample_parameter_combinations(param_grid, max_samples=100, random_state=42):
    """
    从参数网格中采样有效的参数组合
    
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
    
    print(f"总共可能的参数组合数: {total_combinations}")
    
    # 如果总组合数小于最大采样数，返回所有组合
    if total_combinations <= max_samples:
        # 生成所有组合
        all_combinations = []
        
        def generate_combinations(keys, values, current=0, current_params={}):
            if current == len(keys):
                all_combinations.append(current_params.copy())
                return
            
            for value in values[current]:
                current_params[keys[current]] = value
                generate_combinations(keys, values, current + 1, current_params)
        
        generate_combinations(param_keys, param_values)
        return all_combinations
    
    # 采样参数组合
    param_combinations = []
    
    # 首先添加几个基准配置
    # 1. PCA + 标准化，不使用特征选择，无正则化 (基础配置)
    baseline_config = {
        'apply_pca': True,
        'n_components': 100,  # 较小的组件数
        'normalization': 'standard',
        'class_balance': False,
        'regularization': None,
        'alpha': 0.01,
        'feature_selection': None,
        'selection_mode': 'threshold',
        'selection_threshold': 0.01,
        'max_features': 100,
        'l1_ratio': 1.0,
        'scaling_before_selection': True,
        'selection_metric': 'coefficient'
    }
    param_combinations.append(baseline_config)
    
    # 2. PCA + 标准化 + L2正则化
    pca_l2_config = baseline_config.copy()
    pca_l2_config['regularization'] = 'l2'
    pca_l2_config['alpha'] = 0.1
    param_combinations.append(pca_l2_config)
    
    # 3. PCA + 标准化 + 截断SVD正则化
    pca_truncated_config = baseline_config.copy()
    pca_truncated_config['regularization'] = 'truncated'
    pca_truncated_config['alpha'] = 0.1
    param_combinations.append(pca_truncated_config)
    
    # 4. PCA + 标准化 + 类别平衡
    pca_balanced_config = baseline_config.copy()
    pca_balanced_config['class_balance'] = True
    param_combinations.append(pca_balanced_config)
    
    # 5. PCA (更多组件) + 标准化
    pca_more_comp_config = baseline_config.copy()
    pca_more_comp_config['n_components'] = 200
    param_combinations.append(pca_more_comp_config)
    
    # 随机采样剩余配置
    remaining_samples = max_samples - len(param_combinations)
    
    # 优先测试不同的PCA组件数量和正则化参数组合
    n_components_options = param_grid['n_components']
    regularization_options = param_grid['regularization']
    alpha_options = param_grid['alpha']
    normalization_options = param_grid['normalization']
    class_balance_options = param_grid['class_balance']
    
    # 确保每个PCA组件数量至少有一些样本
    for n_comp in n_components_options:
        for reg in regularization_options:
            # 为每个组件数和正则化方法组合创建至少一个配置
            if len(param_combinations) < max_samples:
                config = baseline_config.copy()
                config['n_components'] = n_comp
                config['regularization'] = reg
                if reg is not None:  # 只有在使用正则化时才设置alpha
                    config['alpha'] = np.random.choice(alpha_options)
                
                # 随机选择其他参数
                config['normalization'] = np.random.choice(normalization_options)
                config['class_balance'] = np.random.choice(class_balance_options)
                
                if config not in param_combinations:
                    param_combinations.append(config)
    
    # 避免选择无效组合，随机填充剩余的配置
    attempts = 0
    max_attempts = total_combinations * 2  # 设置一个上限以避免无限循环
    
    while len(param_combinations) < max_samples and attempts < max_attempts:
        attempts += 1
        config = {}
        
        # 采样每个参数
        for key, values in param_grid.items():
            config[key] = np.random.choice(values)
        
        # 修正无效组合
        # 如果配置没有出现在之前的组合中，添加它
        if config not in param_combinations:
            param_combinations.append(config)
    
    print(f"总共生成了 {len(param_combinations)} 个参数组合")
    return param_combinations

 

def create_feature_selection_without_pca_param_grid():
    """创建不使用PCA，专注于直接特征选择的参数网格"""
    
    # 基本参数网格 - 不使用PCA
    base_param_grid = {
        'apply_pca': [False],  # 不使用PCA
        'normalization': ['standard', 'minmax', None],  # 包括不使用标准化
        'class_balance': [False, True],
        'regularization': [None, 'l2', 'truncated'],
        'alpha': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 5.0]  # 正则化强度值
    }
    
    # 特征选择参数网格 - 使用多种算法，参数和迭代次数
    feature_selection_param_grid = {
        'feature_selection': ['lasso', 'elastic_net_torch'],  # 特征选择算法
        'selection_mode': ['fixed'],  # 改为只使用fixed模式
        'selection_threshold': [0.0001],  # 保留一个小的阈值，虽然在fixed模式下不会使用
        'max_features': [50, 100, 150, 200, 250, 300],  # 固定特征数量
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # L1比例
        'scaling_before_selection': [True],
        'selection_metric': ['coefficient'],
        'max_iter': [1000, 2000],  # 最大迭代次数
        'tol': [1e-4, 1e-5]  # 收敛阈值
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
    # 1. LASSO + 标准化 + 固定特征数量 (50)
    lasso_fixed_50_config = {
        'apply_pca': False,
        'normalization': 'standard',
        'class_balance': False,
        'regularization': None,
        'alpha': 0.01,
        'feature_selection': 'lasso',
        'selection_mode': 'fixed',
        'selection_threshold': 0.0001,
        'max_features': 50,
        'l1_ratio': 1.0,
        'scaling_before_selection': True,
        'selection_metric': 'coefficient',
        'max_iter': 1000,
        'tol': 1e-4
    }
    param_combinations.append(lasso_fixed_50_config)
    
    # 2. LASSO + 标准化 + 固定特征数量 (100)
    lasso_fixed_100_config = lasso_fixed_50_config.copy()
    lasso_fixed_100_config['max_features'] = 100
    param_combinations.append(lasso_fixed_100_config)
    
    # 3. LASSO + 标准化 + 固定特征数量 (200)
    lasso_fixed_200_config = lasso_fixed_50_config.copy()
    lasso_fixed_200_config['max_features'] = 200
    param_combinations.append(lasso_fixed_200_config)
    
    # 4. 弹性网络 + 标准化 + 固定特征数量 (100)
    elastic_fixed_config = lasso_fixed_100_config.copy()
    elastic_fixed_config['feature_selection'] = 'elastic_net_torch'
    elastic_fixed_config['l1_ratio'] = 0.5
    param_combinations.append(elastic_fixed_config)
    
    # 5. 弹性网络 + 标准化 + 固定特征数量 (200) + 较小的正则化强度
    elastic_fixed_200_config = elastic_fixed_config.copy()
    elastic_fixed_200_config['max_features'] = 200
    elastic_fixed_200_config['alpha'] = 0.001
    param_combinations.append(elastic_fixed_200_config)
    
    # 6. LASSO + MinMax归一化 + 固定特征数量 (150)
    lasso_minmax_config = lasso_fixed_100_config.copy()
    lasso_minmax_config['normalization'] = 'minmax'
    lasso_minmax_config['max_features'] = 150
    param_combinations.append(lasso_minmax_config)
    
    # 优先测试不同的特征选择方法和参数组合
    # 确保测试各种特征数量和l1_ratio组合
    selection_methods = param_grid['feature_selection']
    max_features_options = param_grid['max_features']
    l1_ratios = param_grid['l1_ratio']
    alpha_options = param_grid['alpha']
    max_iter_options = param_grid['max_iter']
    tol_options = param_grid['tol']
    normalization_options = param_grid['normalization']
    
    # 确保每种特征选择方法与不同特征数量的组合被测试
    for method in selection_methods:
        for max_feat in max_features_options:
            if len(param_combinations) < max_samples:
                config = lasso_fixed_50_config.copy()
                config['feature_selection'] = method
                config['max_features'] = max_feat
                
                # 随机选择其他参数
                config['normalization'] = np.random.choice(normalization_options)
                config['alpha'] = np.random.choice(alpha_options)
                config['max_iter'] = np.random.choice(max_iter_options)
                config['tol'] = np.random.choice(tol_options)
                
                if method == 'elastic_net_torch':
                    config['l1_ratio'] = np.random.choice(l1_ratios)
                
                if config not in param_combinations:
                    param_combinations.append(config)
    
    # 确保测试不同的L1比例（针对elastic_net_torch）
    if 'elastic_net_torch' in selection_methods:
        for l1_ratio in l1_ratios:
            if len(param_combinations) < max_samples:
                config = elastic_fixed_config.copy()
                config['l1_ratio'] = l1_ratio
                config['max_features'] = np.random.choice(max_features_options)
                
                # 随机选择其他参数
                config['normalization'] = np.random.choice(normalization_options)
                config['alpha'] = np.random.choice(alpha_options)
                config['max_iter'] = np.random.choice(max_iter_options)
                config['tol'] = np.random.choice(tol_options)
                
                if config not in param_combinations:
                    param_combinations.append(config)
    
    # 测试不同的正则化强度
    for alpha in alpha_options:
        for method in selection_methods:
            if len(param_combinations) < max_samples:
                config = lasso_fixed_100_config.copy() if method == 'lasso' else elastic_fixed_config.copy()
                config['feature_selection'] = method
                config['alpha'] = alpha
                
                # 随机选择其他参数
                config['normalization'] = np.random.choice(normalization_options)
                config['max_features'] = np.random.choice(max_features_options)
                config['max_iter'] = np.random.choice(max_iter_options)
                config['tol'] = np.random.choice(tol_options)
                
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