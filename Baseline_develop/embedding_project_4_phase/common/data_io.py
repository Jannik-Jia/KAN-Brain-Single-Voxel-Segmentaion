#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据输入输出模块
负责标准化的数据读写操作
"""

import json
import pickle
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class DataIO:
    """数据输入输出管理器"""
    
    @staticmethod
    def save_phase_output(phase_number: int, output_dir: Path, results: Dict[str, Any], 
                         scores: Dict[str, float], metadata: Optional[Dict] = None):
        """
        保存Phase输出的标准化数据包
        
        Args:
            phase_number: Phase编号
            output_dir: 输出目录
            results: Phase特定的结果
            scores: 决策相关得分
            metadata: 元信息
        """
        output_data = {
            "phase_info": {
                "phase_number": phase_number,
                "execution_time": datetime.now().isoformat(),
                "version": "1.0"
            },
            "results": results,
            "scores": scores,
            "metadata": metadata or {}
        }
        
        output_file = output_dir / f'phase{phase_number}_results.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Phase {phase_number} 结果已保存到: {output_file}")
    
    @staticmethod
    def load_phase_output(phase_number: int, output_dir: Path) -> Dict[str, Any]:
        """
        加载Phase输出数据
        
        Args:
            phase_number: Phase编号
            output_dir: 输出目录
            
        Returns:
            Phase输出数据
        """
        output_file = output_dir / f'phase{phase_number}_results.json'
        
        if not output_file.exists():
            raise FileNotFoundError(f"Phase {phase_number} 输出文件不存在: {output_file}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"已加载 Phase {phase_number} 结果从: {output_file}")
        return data
    
    @staticmethod
    def save_numpy_data(data_dict: Dict[str, np.ndarray], output_file: Union[str, Path]):
        """
        保存numpy数组数据
        
        Args:
            data_dict: 包含numpy数组的字典
            output_file: 输出文件路径
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_file, **data_dict)
        logger.info(f"Numpy数据已保存到: {output_file}")
    
    @staticmethod
    def load_numpy_data(input_file: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        加载numpy数组数据
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            包含numpy数组的字典
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Numpy数据文件不存在: {input_file}")
        
        data = np.load(input_file)
        data_dict = {key: data[key] for key in data.files}
        
        logger.info(f"已加载Numpy数据从: {input_file}")
        return data_dict
    
    @staticmethod
    def save_pickle(obj: Any, output_file: Union[str, Path]):
        """
        保存pickle对象
        
        Args:
            obj: 要保存的对象
            output_file: 输出文件路径
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(obj, f)
        
        logger.info(f"Pickle对象已保存到: {output_file}")
    
    @staticmethod
    def load_pickle(input_file: Union[str, Path]) -> Any:
        """
        加载pickle对象
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            加载的对象
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Pickle文件不存在: {input_file}")
        
        with open(input_file, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info(f"已加载Pickle对象从: {input_file}")
        return obj
    
    @staticmethod
    def save_json(data: Dict, output_file: Union[str, Path]):
        """
        保存JSON数据
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"JSON数据已保存到: {output_file}")
    
    @staticmethod
    def load_json(input_file: Union[str, Path]) -> Dict:
        """
        加载JSON数据
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            加载的数据
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"JSON文件不存在: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"已加载JSON数据从: {input_file}")
        return data


class NumpyEncoder(json.JSONEncoder):
    """用于处理numpy类型的JSON编码器"""

    def encode(self, obj):
        obj = self.convert_keys(obj)
        return super().encode(obj)

    def convert_keys(self, obj):
        if isinstance(obj, dict):
            return {self.convert_key(k): self.convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_keys(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_keys(item) for item in obj)
        else:
            return obj

    def convert_key(self, key):
        if isinstance(key, (np.integer, np.int64, np.int32)):
            return int(key)
        elif isinstance(key, (np.floating, np.float64, np.float32)):
            return float(key)
        elif isinstance(key, np.bool_):
            return bool(key)
        else:
            return key

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if obj.size < 100:  # 小数组直接转换
                return obj.tolist()
            else:  # 大数组只保存描述
                return f"<numpy.ndarray: shape={obj.shape}, dtype={obj.dtype}>"
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
