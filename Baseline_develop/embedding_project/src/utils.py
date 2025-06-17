#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
包含日志设置、文件保存等功能
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')


def setup_logging(log_file, log_level='INFO'):
    """
    设置日志系统
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
        
    Returns:
        logger: 配置好的日志器
    """
    # 确保日志目录存在
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 获取主日志器
    logger = logging.getLogger()
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)
    
    return logger


def save_config(config_data, config_file):
    """
    保存配置信息到JSON文件
    
    Args:
        config_data: 配置数据字典
        config_file: 配置文件路径
    """
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)


def save_analysis_results(results, output_file):
    """
    保存分析结果到文件
    
    Args:
        results: 分析结果
        output_file: 输出文件路径
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理numpy数组，转换为可序列化的格式
    serializable_results = _make_serializable(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)


def _make_serializable(obj):
    """
    将对象转换为可JSON序列化的格式
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size < 1000:  # 只保存小数组
            return obj.tolist()
        else:
            return f"numpy.ndarray(shape={obj.shape}, dtype={obj.dtype})"
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return f"object({type(obj).__name__})"
    else:
        return obj


def format_time(seconds):
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def create_progress_tracker():
    """
    创建进度跟踪器
    
    Returns:
        进度跟踪函数
    """
    start_time = datetime.now()
    
    def track_progress(current_step, total_steps, step_name=""):
        """
        跟踪进度
        
        Args:
            current_step: 当前步骤
            total_steps: 总步骤数
            step_name: 步骤名称
        """
        progress = current_step / total_steps * 100
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if current_step > 0:
            estimated_total = elapsed * total_steps / current_step
            remaining = estimated_total - elapsed
            remaining_str = format_time(remaining)
        else:
            remaining_str = "未知"
        
        logger = logging.getLogger(__name__)
        logger.info(f"进度: {progress:.1f}% ({current_step}/{total_steps}) - {step_name} - 剩余时间: {remaining_str}")
    
    return track_progress


def ensure_directory(path):
    """
    确保目录存在
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path):
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小字符串
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return "文件不存在"
    
    size_bytes = file_path.stat().st_size
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"


def cleanup_memory():
    """
    清理内存
    """
    import gc
    gc.collect()