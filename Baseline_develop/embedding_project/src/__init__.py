# src/__init__.py
"""
脑区感知Subject Embedding分析包
"""

__version__ = "1.0.0"
__author__ = "Brain Analysis Team"
__description__ = "Brain-Aware Subject Embedding Feasibility Analysis"

from .analyzer import BrainAwareSubjectEmbeddingAnalyzer
from .data_loader import load_and_prepare_data_multi_subject_out
from .utils import setup_logging, save_config, ensure_directory

__all__ = [
    'BrainAwareSubjectEmbeddingAnalyzer',
    'load_and_prepare_data_multi_subject_out',
    'setup_logging',
    'save_config',
    'ensure_directory'
]


# config/__init__.py
"""
配置模块
"""

from .settings import Config

__all__ = ['Config']