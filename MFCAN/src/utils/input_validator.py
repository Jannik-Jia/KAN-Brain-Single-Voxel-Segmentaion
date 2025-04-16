import numpy as np
import logging
import json
import os
import torch
from typing import Dict, List, Any, Union, Tuple, Optional
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.logging_utils import Logger

class InputValidator:
    """输入验证工具，用于验证模型和数据输入的有效性"""
    
    def __init__(self, logger=None):
        """
        初始化输入验证器
        
        参数:
            logger: 日志记录器
        """
        # 设置日志记录器
        if logger:
            self.logger = logger
        else:
            log_manager = Logger("InputValidator", log_dir="logs/utils")
            self.logger = log_manager.get_logger()
    
    def validate_config(self, config: Dict, required_fields: List[str], 
                        field_types: Dict[str, type] = None) -> Tuple[bool, List[str]]:
        """
        验证配置文件是否包含必要字段并且类型正确
        
        参数:
            config: 配置字典
            required_fields: 必要字段列表
            field_types: 字段类型字典
            
        返回:
            valid: 配置是否有效
            errors: 错误信息列表
        """
        self.logger.info(f"验证配置文件，必须字段: {required_fields}")
        errors = []
        
        # 检查必要字段
        for field in required_fields:
            if field not in config:
                error_msg = f"配置缺少必要字段: {field}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        # 如果提供了field_types，检查类型
        if field_types:
            for field, expected_type in field_types.items():
                if field in config:
                    if not isinstance(config[field], expected_type):
                        actual_type = type(config[field]).__name__
                        expected_type_name = expected_type.__name__
                        
                        error_msg = f"字段 {field} 类型错误，期望 {expected_type_name}，实际为 {actual_type}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
        
        valid = len(errors) == 0
        if valid:
            self.logger.info("配置验证通过")
        else:
            self.logger.warning(f"配置验证失败，发现 {len(errors)} 个错误")
            
        return valid, errors
    
    def validate_model_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        验证模型配置是否有效
        
        参数:
            config: 模型配置字典
            
        返回:
            valid: 配置是否有效
            errors: 错误信息列表
        """
        self.logger.info("验证模型配置")
        errors = []
        
        # 检查模型类型
        model_type = config.get('model_type')
        if not model_type:
            errors.append("配置中缺少 'model_type' 字段")
        elif model_type not in ['mlp', 'group_mlp', 'deep_mlp', 'mfcan', 'ensemble']:
            errors.append(f"不支持的模型类型: {model_type}")
        
        # 根据模型类型验证特定配置
        if model_type == 'mlp':
            required_fields = ['input_dim', 'hidden_dims', 'num_classes']
            for field in required_fields:
                if field not in config:
                    errors.append(f"MLP模型配置缺少字段: {field}")
                    
            # 验证hidden_dims是否为列表
            if 'hidden_dims' in config and not isinstance(config['hidden_dims'], list):
                errors.append("'hidden_dims' 必须是列表类型")
                
        elif model_type == 'group_mlp':
            required_fields = ['group_dims', 'num_classes']
            for field in required_fields:
                if field not in config:
                    errors.append(f"GroupMLP模型配置缺少字段: {field}")
                    
            # 验证group_dims是否为字典
            if 'group_dims' in config and not isinstance(config['group_dims'], dict):
                errors.append("'group_dims' 必须是字典类型")
                
        elif model_type == 'deep_mlp':
            required_fields = ['input_dim', 'hidden_dims', 'num_classes']
            for field in required_fields:
                if field not in config:
                    errors.append(f"DeepMLP模型配置缺少字段: {field}")
            
            # 验证hidden_dims是否为列表
            if 'hidden_dims' in config and not isinstance(config['hidden_dims'], list):
                errors.append("'hidden_dims' 必须是列表类型")
                
            # 验证dropout_rates是否为列表或None
            if 'dropout_rates' in config and config['dropout_rates'] is not None and not isinstance(config['dropout_rates'], list):
                errors.append("'dropout_rates' 必须是列表类型或None")
                
            # 验证attn_layers是否为列表或None
            if 'attn_layers' in config and config['attn_layers'] is not None and not isinstance(config['attn_layers'], list):
                errors.append("'attn_layers' 必须是列表类型或None")
                
        # 添加MFCAN等其他模型的验证逻辑...
                
        # 验证训练参数
        if 'num_epochs' in config and (not isinstance(config['num_epochs'], int) or config['num_epochs'] <= 0):
            errors.append("'num_epochs' 必须是正整数")
            
        if 'batch_size' in config and (not isinstance(config['batch_size'], int) or config['batch_size'] <= 0):
            errors.append("'batch_size' 必须是正整数")
            
        if 'learning_rate' in config and (not isinstance(config['learning_rate'], (float, int)) or config['learning_rate'] <= 0):
            errors.append("'learning_rate' 必须是正数")
        
        valid = len(errors) == 0
        if valid:
            self.logger.info("模型配置验证通过")
        else:
            self.logger.warning(f"模型配置验证失败，发现 {len(errors)} 个错误")
            
        return valid, errors
    
    def validate_data_dimensions(self, data: Dict[str, np.ndarray], 
                                expected_dims: Dict[str, Tuple]) -> Tuple[bool, List[str]]:
        """
        验证数据维度是否符合期望
        
        参数:
            data: 数据字典
            expected_dims: 期望的维度字典
            
        返回:
            valid: 数据维度是否有效
            errors: 错误信息列表
        """
        self.logger.info("验证数据维度")
        errors = []
        
        for key, expected in expected_dims.items():
            if key not in data:
                errors.append(f"缺少数据: {key}")
                continue
                
            actual = data[key].shape
            
            # 检查维度是否匹配
            valid_dims = True
            if len(actual) != len(expected):
                valid_dims = False
            else:
                for a, e in zip(actual, expected):
                    # 如果期望维度为None，表示任意值都可以
                    if e is not None and a != e:
                        valid_dims = False
                        break
            
            if not valid_dims:
                errors.append(f"数据 {key} 维度不匹配, 期望 {expected}, 实际为 {actual}")
        
        valid = len(errors) == 0
        if valid:
            self.logger.info("数据维度验证通过")
        else:
            self.logger.warning(f"数据维度验证失败，发现 {len(errors)} 个错误")
            
        return valid, errors
    
    def validate_feature_range(self, features: np.ndarray, 
                             min_val: float = -float('inf'), 
                             max_val: float = float('inf')) -> Tuple[bool, Dict[str, Any]]:
        """
        验证特征值是否在合理范围内
        
        参数:
            features: 特征矩阵
            min_val: 最小合理值
            max_val: 最大合理值
            
        返回:
            valid: 特征值是否有效
            stats: 统计信息字典
        """
        self.logger.info(f"验证特征值范围 [{min_val}, {max_val}]")
        
        # 计算基本统计量
        feature_min = np.min(features)
        feature_max = np.max(features)
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        
        # 检查是否有超出范围的值
        below_min = (features < min_val).sum() if min_val > -float('inf') else 0
        above_max = (features > max_val).sum() if max_val < float('inf') else 0
        
        stats = {
            'min': float(feature_min),
            'max': float(feature_max),
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'below_min_count': int(below_min),
            'above_max_count': int(above_max),
            'total_invalid': int(nan_count + inf_count + below_min + above_max)
        }
        
        valid = stats['total_invalid'] == 0
        
        if valid:
            self.logger.info("特征值范围验证通过")
        else:
            self.logger.warning(f"特征值范围验证失败，发现 {stats['total_invalid']} 个无效值")
            
            if nan_count > 0:
                self.logger.warning(f"包含 {nan_count} 个NaN值")
            if inf_count > 0:
                self.logger.warning(f"包含 {inf_count} 个Inf值")
            if below_min > 0:
                self.logger.warning(f"包含 {below_min} 个小于最小值 {min_val} 的值")
            if above_max > 0:
                self.logger.warning(f"包含 {above_max} 个大于最大值 {max_val} 的值")
        
        return valid, stats
    
    def validate_labels(self, labels: np.ndarray, num_classes: int) -> Tuple[bool, Dict[str, Any]]:
        """
        验证分类标签是否有效
        
        参数:
            labels: 标签数组
            num_classes: 类别数量
            
        返回:
            valid: 标签是否有效
            stats: 统计信息字典
        """
        self.logger.info(f"验证分类标签，类别数: {num_classes}")
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        min_label = np.min(unique_labels)
        max_label = np.max(unique_labels)
        
        # 检查标签是否在有效范围内
        if min_label < 0:
            invalid_labels = (labels < 0).sum()
            self.logger.warning(f"存在 {invalid_labels} 个负标签值")
        else:
            invalid_labels = 0
            
        if max_label >= num_classes:
            out_of_range = (labels >= num_classes).sum()
            self.logger.warning(f"存在 {out_of_range} 个超出类别范围的标签值")
            invalid_labels += out_of_range
        
        # 计算每个类别的样本数量
        label_counts = {}
        for i in range(num_classes):
            count = np.sum(labels == i)
            label_counts[str(i)] = int(count)
        
        # 计算类别不平衡比例
        non_zero_counts = [count for count in label_counts.values() if count > 0]
        if non_zero_counts:
            imbalance_ratio = max(non_zero_counts) / max(min(non_zero_counts), 1)
        else:
            imbalance_ratio = 0
            
        # 检查是否有缺失的类别
        missing_classes = [i for i in range(num_classes) if label_counts.get(str(i), 0) == 0]
        
        stats = {
            'unique_labels': len(unique_labels),
            'min_label': int(min_label),
            'max_label': int(max_label),
            'invalid_count': int(invalid_labels),
            'label_counts': label_counts,
            'imbalance_ratio': float(imbalance_ratio),
            'missing_classes': missing_classes
        }
        
        valid = invalid_labels == 0 and not missing_classes
        
        if valid:
            self.logger.info("标签验证通过")
        else:
            self.logger.warning(f"标签验证失败")
            
            if invalid_labels > 0:
                self.logger.warning(f"包含 {invalid_labels} 个无效标签")
            if missing_classes:
                self.logger.warning(f"缺少类别: {missing_classes}")
            if imbalance_ratio > 10:
                self.logger.warning(f"类别严重不平衡，比例为 {imbalance_ratio:.2f}:1")
        
        return valid, stats
    
    def validate_tensors(self, tensors: Dict[str, torch.Tensor], 
                        expected_dims: Dict[str, Tuple] = None) -> Tuple[bool, List[str]]:
        """
        验证PyTorch张量是否有效
        
        参数:
            tensors: 张量字典
            expected_dims: 期望的维度字典
            
        返回:
            valid: 张量是否有效
            errors: 错误信息列表
        """
        self.logger.info("验证PyTorch张量")
        errors = []
        
        for name, tensor in tensors.items():
            # 检查是否为张量
            if not isinstance(tensor, torch.Tensor):
                errors.append(f"{name} 不是有效的PyTorch张量")
                continue
                
            # 检查是否包含NaN或Inf
            if torch.isnan(tensor).any():
                errors.append(f"{name} 包含NaN值")
            if torch.isinf(tensor).any():
                errors.append(f"{name} 包含Inf值")
                
            # 检查维度
            if expected_dims and name in expected_dims:
                expected = expected_dims[name]
                actual = tensor.shape
                
                if len(actual) != len(expected):
                    errors.append(f"{name} 维度数量不匹配，期望 {len(expected)}，实际为 {len(actual)}")
                else:
                    for i, (a, e) in enumerate(zip(actual, expected)):
                        if e is not None and a != e:
                            errors.append(f"{name} 在维度 {i} 不匹配，期望 {e}，实际为 {a}")
        
        valid = len(errors) == 0
        if valid:
            self.logger.info("PyTorch张量验证通过")
        else:
            self.logger.warning(f"PyTorch张量验证失败，发现 {len(errors)} 个错误")
            
        return valid, errors
    
    def validate_model_inputs(self, model, inputs: Dict[str, Any], expected_inputs: List[str]) -> Tuple[bool, List[str]]:
        """
        验证模型输入是否符合期望
        
        参数:
            model: PyTorch模型
            inputs: 输入字典
            expected_inputs: 期望的输入列表
            
        返回:
            valid: 输入是否有效
            errors: 错误信息列表
        """
        self.logger.info(f"验证模型输入，期望输入: {expected_inputs}")
        errors = []
        
        # 检查是否提供了所有期望的输入
        for input_name in expected_inputs:
            if input_name not in inputs:
                errors.append(f"缺少必要的输入: {input_name}")
        
        # 如果缺少必要输入，直接返回
        if errors:
            self.logger.warning(f"模型输入验证失败，缺少必要输入")
            return False, errors
            
        try:
            # 尝试执行前向传播
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                
            self.logger.info(f"模型前向传播成功，输出形状: {outputs.shape}")
        except Exception as e:
            error_msg = f"模型前向传播失败: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        valid = len(errors) == 0
        if valid:
            self.logger.info("模型输入验证通过")
        else:
            self.logger.warning(f"模型输入验证失败")
            
        return valid, errors