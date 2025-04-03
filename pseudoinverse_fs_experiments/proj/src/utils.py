# src/utils.py

import numpy as np

def create_feature_selection_param_grid():
    """创建包含特征选择参数的网格"""
    
    # 基本参数网格
    base_param_grid = {
        'apply_pca': [True, False],
        'n_components': [50, 100],
        'normalization': ['standard', 'minmax', None],
        'class_balance': [False, True],
        'regularization': [None, 'l2'],
        'alpha': [0.001, 0.01, 0.1]
    }
    
    # 特征选择参数网格
    feature_selection_param_grid = {
        'feature_selection': [None, 'lasso', 'elastic_net'],
        'selection_mode': ['threshold', 'fixed'],
        'selection_threshold': [0.001, 0.01, 0.05],
        'max_features': [50, 100, 200],
        'l1_ratio': [0.5, 0.8, 1.0],
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
    # 1. 不使用特征选择的基准配置
    baseline_config = {
        'apply_pca': False,
        'n_components': 50,
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
    
    # 2. 使用LASSO特征选择的基准配置
    lasso_config = baseline_config.copy()
    lasso_config['feature_selection'] = 'lasso'
    param_combinations.append(lasso_config)
    
    # 3. 使用弹性网络特征选择的基准配置
    elastic_net_config = baseline_config.copy()
    elastic_net_config['feature_selection'] = 'elastic_net'
    elastic_net_config['l1_ratio'] = 0.5
    param_combinations.append(elastic_net_config)
    
    # 随机采样其余配置
    remaining_samples = max_samples - len(param_combinations)
    
    # 避免选择无效组合
    while len(param_combinations) < max_samples:
        config = {}
        
        # 采样每个参数
        for key, values in param_grid.items():
            config[key] = np.random.choice(values)
        
        # 修正无效组合
        # 1. 如果不使用特征选择，调整相关参数
        if config['feature_selection'] is None:
            config['selection_mode'] = 'threshold'  # 不影响实际行为
            config['selection_threshold'] = 0.01
            config['max_features'] = 100
            config['l1_ratio'] = 1.0
        
        # 2. 如果使用的是LASSO，设置l1_ratio为1.0
        if config['feature_selection'] == 'lasso':
            config['l1_ratio'] = 1.0
        
        # 3. 如果不使用PCA，调整相关参数
        if not config['apply_pca']:
            config['n_components'] = 50  # 设为默认值，不影响
        
        # 4. 如果selection_mode是threshold，max_features不相关
        if config['selection_mode'] == 'threshold':
            config['max_features'] = 100  # 设为默认值
        
        # 5. 如果selection_mode是fixed，selection_threshold不相关
        if config['selection_mode'] == 'fixed':
            config['selection_threshold'] = 0.01  # 设为默认值
        
        # 检查是否已经存在相同配置
        if config not in param_combinations:
            param_combinations.append(config)
    
    return param_combinations