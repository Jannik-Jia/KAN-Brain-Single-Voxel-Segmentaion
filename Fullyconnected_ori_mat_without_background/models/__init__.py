from .base_mlp import BrainVoxelMLP
from .deep_mlp import DeepMLP
from .residual_mlp import ResidualBrainVoxelMLP

def get_model(model_type, **kwargs):
    """
    根据模型类型名称返回对应的模型类
    
    参数:
        model_type: 模型类型名称
        **kwargs: 传递给模型构造函数的参数
    
    返回:
        model: 构造的模型对象
    """
    models = {
        'base_mlp': BrainVoxelMLP,
        'deep_mlp': DeepMLP,
        'residual_mlp': ResidualBrainVoxelMLP
    }
    
    if model_type not in models:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(models.keys())}")
    
    # 过滤掉不被当前模型支持的参数
    if model_type == 'base_mlp':
        valid_params = ['input_dim', 'hidden_dims', 'num_classes', 'dropout_rate', 'activation']
    elif model_type == 'deep_mlp':
        valid_params = ['input_dim', 'hidden_dims', 'num_classes', 'dropout_rate', 'activation', 'use_skip_connections']
    elif model_type == 'residual_mlp':
        valid_params = ['input_dim', 'hidden_dims', 'num_classes', 'dropout_rate', 'activation', 'use_bottleneck', 'bottleneck_factor']
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return models[model_type](**filtered_kwargs)