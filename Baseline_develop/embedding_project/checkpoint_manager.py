#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存档点管理系统
支持分析过程中的状态保存和恢复功能
"""

import os
import json
import pickle
import gzip
import time
import logging
import traceback
import psutil
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """存档点管理器"""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self.recovery_log = self.checkpoint_dir / 'recovery_log.txt'
        
        # 存档点配置
        self.config = {
            'max_checkpoints': 15,
            'auto_cleanup_days': 7,
            'compression_level': 6,
            'keep_phase_checkpoints': True,
            'include_deep_networks': True,
            'checkpoint_format_version': '1.0'
        }
        
        # 初始化元数据
        self._initialize_metadata()
        
        logger.info(f"🔄 存档点管理器初始化完成: {self.checkpoint_dir}")
    
    def _initialize_metadata(self):
        """初始化存档点元数据"""
        if not self.metadata_file.exists():
            metadata = {
                'checkpoints': {},
                'creation_time': datetime.now().isoformat(),
                'format_version': self.config['checkpoint_format_version'],
                'total_checkpoints_created': 0
            }
            self._save_metadata(metadata)
    
    def _load_metadata(self) -> Dict:
        """加载存档点元数据"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载元数据失败: {e}")
            return {'checkpoints': {}, 'total_checkpoints_created': 0}
    
    def _save_metadata(self, metadata: Dict):
        """保存存档点元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def _get_memory_usage(self) -> str:
        """获取当前内存使用量"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        except:
            return "Unknown"
    
    def _compress_data(self, data: Any) -> bytes:
        """压缩数据"""
        try:
            pickled_data = pickle.dumps(data)
            compressed_data = gzip.compress(pickled_data, compresslevel=self.config['compression_level'])
            return compressed_data
        except Exception as e:
            logger.error(f"数据压缩失败: {e}")
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """解压数据"""
        try:
            # 尝试解压缩
            decompressed_data = gzip.decompress(compressed_data)
            return pickle.loads(decompressed_data)
        except:
            # 如果解压失败，尝试直接反序列化（可能没有压缩）
            try:
                return pickle.loads(compressed_data)
            except Exception as e:
                logger.error(f"数据解压失败: {e}")
                raise
    
    def create_checkpoint(self, 
                         analyzer_instance,
                         checkpoint_name: str,
                         phase_completed: int = -1,
                         description: str = "",
                         is_auto: bool = True,
                         include_deep_networks: bool = None) -> str:
        """
        创建存档点
        
        Args:
            analyzer_instance: 分析器实例
            checkpoint_name: 存档点名称
            phase_completed: 已完成的阶段 (-1表示未完成任何阶段)
            description: 描述信息
            is_auto: 是否为自动创建
            include_deep_networks: 是否包含深度网络状态
            
        Returns:
            str: 存档点文件路径
        """
        start_time = time.time()
        
        if include_deep_networks is None:
            include_deep_networks = self.config['include_deep_networks']
        
        # 生成存档点文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_auto:
            checkpoint_filename = f"auto_{checkpoint_name}_{timestamp}.ckpt"
        else:
            checkpoint_filename = f"manual_{checkpoint_name}_{timestamp}.ckpt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        logger.info(f"🔄 创建存档点: {checkpoint_name}")
        logger.info(f"📁 存档路径: {checkpoint_path}")
        
        try:
            # 准备存档数据
            checkpoint_data = self._prepare_checkpoint_data(
                analyzer_instance, 
                checkpoint_name, 
                phase_completed, 
                description,
                include_deep_networks
            )
            
            # 保存存档点
            with open(checkpoint_path, 'wb') as f:
                compressed_data = self._compress_data(checkpoint_data)
                f.write(compressed_data)
            
            # 更新元数据
            self._update_checkpoint_metadata(
                checkpoint_filename,
                checkpoint_data['metadata']
            )
            
            creation_time = time.time() - start_time
            file_size = checkpoint_path.stat().st_size / 1024 / 1024  # MB
            
            logger.info(f"✅ 存档点创建成功")
            logger.info(f"   📊 耗时: {creation_time:.2f}秒")
            logger.info(f"   📦 文件大小: {file_size:.1f}MB")
            logger.info(f"   💾 内存使用: {checkpoint_data['metadata']['memory_usage']}")
            
            # 自动清理旧存档点
            self._auto_cleanup_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ 存档点创建失败: {e}")
            logger.exception("详细错误信息:")
            raise
    
    def _prepare_checkpoint_data(self, 
                                analyzer_instance,
                                checkpoint_name: str,
                                phase_completed: int,
                                description: str,
                                include_deep_networks: bool) -> Dict:
        """准备存档点数据"""
        
        # 获取分析器状态
        analyzer_state = {
            'analysis_results': getattr(analyzer_instance, 'analysis_results', {}),
            'decision_scores': getattr(analyzer_instance, 'decision_scores', {}),
            'data': getattr(analyzer_instance, 'data', {}),
        }
        
        # 深度网络状态处理
        deep_network_state = {}
        if include_deep_networks and hasattr(analyzer_instance, 'deep_network_cache'):
            deep_network_state = self._extract_deep_network_state(analyzer_instance)
        
        # 计算进度
        total_phases = 4
        progress_percentage = (phase_completed / total_phases * 100) if phase_completed >= 0 else 0
        
        # 准备元数据
        metadata = {
            'checkpoint_name': checkpoint_name,
            'creation_time': datetime.now().isoformat(),
            'phase_completed': phase_completed,
            'total_phases': total_phases,
            'progress_percentage': progress_percentage,
            'description': description,
            'memory_usage': self._get_memory_usage(),
            'include_deep_networks': include_deep_networks,
            'data_shapes': self._get_data_shapes(analyzer_instance),
        }
        
        # 获取配置信息
        config_info = getattr(analyzer_instance, 'config_info', {})
        
        # 性能统计
        performance_stats = {
            'phase_durations': getattr(analyzer_instance, 'phase_durations', []),
            'memory_peaks': [],
            'gpu_usage': torch.cuda.is_available(),
        }
        
        # 版本信息
        version_info = {
            'analyzer_version': 'Brain-Aware v1.0',
            'checkpoint_format_version': self.config['checkpoint_format_version'],
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'torch_version': torch.__version__ if hasattr(analyzer_instance, 'device') else 'N/A',
            'cuda_available': torch.cuda.is_available() if hasattr(analyzer_instance, 'device') else False,
        }
        
        return {
            'analyzer_state': analyzer_state,
            'deep_network_state': deep_network_state,
            'metadata': metadata,
            'config': config_info,
            'performance_stats': performance_stats,
            'version_info': version_info,
        }
    
    def _extract_deep_network_state(self, analyzer_instance) -> Dict:
        """提取深度网络状态"""
        deep_network_state = {}
        
        try:
            # 保存深度网络缓存
            if hasattr(analyzer_instance, 'deep_network_cache'):
                deep_network_state['cache'] = analyzer_instance.deep_network_cache
            
            # 保存alex超参数
            if hasattr(analyzer_instance, 'alex_hyperparams'):
                deep_network_state['alex_hyperparams'] = analyzer_instance.alex_hyperparams
            
            # 如果存在训练好的网络，保存其状态
            if hasattr(analyzer_instance, '_last_trained_network'):
                network = getattr(analyzer_instance, '_last_trained_network')
                if hasattr(network, 'state_dict'):
                    deep_network_state['network_state_dict'] = network.state_dict()
                
            logger.info(f"   🧠 深度网络状态已保存")
            
        except Exception as e:
            logger.warning(f"深度网络状态保存失败: {e}")
            deep_network_state['error'] = str(e)
        
        return deep_network_state
    
    def _get_data_shapes(self, analyzer_instance) -> Dict:
        """获取数据形状信息用于兼容性检查"""
        shapes = {}
        
        if hasattr(analyzer_instance, 'data') and analyzer_instance.data:
            data = analyzer_instance.data
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    shapes[key] = value.shape
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        shapes[key] = f"list_of_arrays_{len(value)}"
                    else:
                        shapes[key] = f"list_{len(value)}"
        
        return shapes
    
    def _update_checkpoint_metadata(self, filename: str, checkpoint_metadata: Dict):
        """更新存档点元数据"""
        metadata = self._load_metadata()
        
        metadata['checkpoints'][filename] = checkpoint_metadata
        metadata['total_checkpoints_created'] = metadata.get('total_checkpoints_created', 0) + 1
        metadata['last_checkpoint'] = filename
        metadata['last_update'] = datetime.now().isoformat()
        
        self._save_metadata(metadata)
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有可用的存档点"""
        metadata = self._load_metadata()
        checkpoints = []
        
        for filename, checkpoint_info in metadata.get('checkpoints', {}).items():
            checkpoint_path = self.checkpoint_dir / filename
            
            if checkpoint_path.exists():
                file_size = checkpoint_path.stat().st_size / 1024 / 1024  # MB
                
                checkpoint_entry = {
                    'filename': filename,
                    'name': checkpoint_info.get('checkpoint_name', 'Unknown'),
                    'creation_time': checkpoint_info.get('creation_time', 'Unknown'),
                    'phase_completed': checkpoint_info.get('phase_completed', -1),
                    'progress_percentage': checkpoint_info.get('progress_percentage', 0),
                    'description': checkpoint_info.get('description', ''),
                    'file_size_mb': file_size,
                    'memory_usage': checkpoint_info.get('memory_usage', 'Unknown'),
                    'include_deep_networks': checkpoint_info.get('include_deep_networks', False),
                }
                checkpoints.append(checkpoint_entry)
        
        # 按创建时间排序
        checkpoints.sort(key=lambda x: x['creation_time'], reverse=True)
        return checkpoints
    
    def load_checkpoint(self, checkpoint_identifier: str, analyzer_instance) -> bool:
        """
        加载存档点
        
        Args:
            checkpoint_identifier: 存档点标识符（文件名或阶段名）
            analyzer_instance: 分析器实例
            
        Returns:
            bool: 加载是否成功
        """
        logger.info(f"🔄 开始加载存档点: {checkpoint_identifier}")
        
        # 查找存档点文件
        checkpoint_path = self._find_checkpoint_file(checkpoint_identifier)
        if not checkpoint_path:
            logger.error(f"❌ 未找到存档点: {checkpoint_identifier}")
            return False
        
        try:
            # 加载存档点数据
            with open(checkpoint_path, 'rb') as f:
                compressed_data = f.read()
                checkpoint_data = self._decompress_data(compressed_data)
            
            # 兼容性检查
            if not self._check_compatibility(checkpoint_data, analyzer_instance):
                logger.error("❌ 存档点兼容性检查失败")
                return False
            
            # 恢复分析器状态
            self._restore_analyzer_state(checkpoint_data, analyzer_instance)
            
            # 记录恢复日志
            self._log_recovery(checkpoint_path, checkpoint_data)
            
            logger.info(f"✅ 存档点加载成功: {checkpoint_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 存档点加载失败: {e}")
            logger.exception("详细错误信息:")
            return False
    
    def _find_checkpoint_file(self, identifier: str) -> Optional[Path]:
        """查找存档点文件"""
        # 如果是完整文件名
        direct_path = self.checkpoint_dir / identifier
        if direct_path.exists():
            return direct_path
        
        # 如果不包含扩展名，添加.ckpt
        if not identifier.endswith('.ckpt'):
            direct_path = self.checkpoint_dir / f"{identifier}.ckpt"
            if direct_path.exists():
                return direct_path
        
        # 按阶段名查找最新的存档点
        phase_keywords = {
            'phase0': 'data_prepared',
            'phase1': 'subject_analysis',
            'phase2': 'separability',
            'phase3': 'embedding_design',
            'phase4': 'final_decision',
            'data': 'data_prepared',
            'subject': 'subject_analysis',
            'separability': 'separability',
            'embedding': 'embedding_design',
            'decision': 'final_decision',
        }
        
        keyword = phase_keywords.get(identifier.lower())
        if keyword:
            # 查找包含关键词的最新存档点
            matching_files = []
            for checkpoint_file in self.checkpoint_dir.glob("*.ckpt"):
                if keyword in checkpoint_file.name:
                    matching_files.append(checkpoint_file)
            
            if matching_files:
                # 返回最新的文件
                return max(matching_files, key=lambda f: f.stat().st_mtime)
        
        return None
    
    def _check_compatibility(self, checkpoint_data: Dict, analyzer_instance) -> bool:
        """检查存档点兼容性"""
        try:
            version_info = checkpoint_data.get('version_info', {})
            checkpoint_version = version_info.get('checkpoint_format_version', '1.0')
            
            if checkpoint_version != self.config['checkpoint_format_version']:
                logger.warning(f"存档点格式版本不匹配: {checkpoint_version} vs {self.config['checkpoint_format_version']}")
                # 版本不匹配时可以尝试兼容性转换
            
            # 检查数据形状兼容性
            if hasattr(analyzer_instance, 'data') and analyzer_instance.data:
                checkpoint_shapes = checkpoint_data.get('metadata', {}).get('data_shapes', {})
                current_shapes = self._get_data_shapes(analyzer_instance)
                
                for key, shape in checkpoint_shapes.items():
                    if key in current_shapes and current_shapes[key] != shape:
                        logger.warning(f"数据形状不匹配 {key}: {shape} vs {current_shapes[key]}")
            
            return True
            
        except Exception as e:
            logger.error(f"兼容性检查失败: {e}")
            return False
    
    def _restore_analyzer_state(self, checkpoint_data: Dict, analyzer_instance):
        """恢复分析器状态"""
        analyzer_state = checkpoint_data.get('analyzer_state', {})
        
        # 恢复分析结果
        if 'analysis_results' in analyzer_state:
            analyzer_instance.analysis_results = analyzer_state['analysis_results']
            logger.info("   📊 分析结果已恢复")
        
        # 恢复决策得分
        if 'decision_scores' in analyzer_state:
            analyzer_instance.decision_scores = analyzer_state['decision_scores']
            logger.info("   📈 决策得分已恢复")
        
        # 恢复数据状态
        if 'data' in analyzer_state and analyzer_state['data']:
            # 只恢复关键数据，避免大数组重复
            for key, value in analyzer_state['data'].items():
                if key not in ['X_train', 'X_val', 'X_test']:  # 跳过大数组
                    setattr(analyzer_instance, key, value)
            analyzer_instance.data.update(analyzer_state['data'])
            logger.info("   💾 数据状态已恢复")
        
        # 恢复深度网络状态
        deep_network_state = checkpoint_data.get('deep_network_state', {})
        if deep_network_state and hasattr(analyzer_instance, 'deep_network_cache'):
            if 'cache' in deep_network_state:
                analyzer_instance.deep_network_cache = deep_network_state['cache']
                logger.info("   🧠 深度网络缓存已恢复")
            
            if 'alex_hyperparams' in deep_network_state:
                analyzer_instance.alex_hyperparams = deep_network_state['alex_hyperparams']
                logger.info("   ⚙️ Alex超参数已恢复")
        
        # 恢复配置信息
        config_info = checkpoint_data.get('config', {})
        if config_info:
            analyzer_instance.config_info = config_info
    
    def _log_recovery(self, checkpoint_path: Path, checkpoint_data: Dict):
        """记录恢复日志"""
        try:
            metadata = checkpoint_data.get('metadata', {})
            recovery_info = {
                'recovery_time': datetime.now().isoformat(),
                'checkpoint_file': checkpoint_path.name,
                'checkpoint_name': metadata.get('checkpoint_name', 'Unknown'),
                'creation_time': metadata.get('creation_time', 'Unknown'),
                'phase_completed': metadata.get('phase_completed', -1),
                'progress_percentage': metadata.get('progress_percentage', 0),
            }
            
            with open(self.recovery_log, 'a', encoding='utf-8') as f:
                f.write(f"{json.dumps(recovery_info, ensure_ascii=False)}\n")
                
        except Exception as e:
            logger.warning(f"记录恢复日志失败: {e}")
    
    def _auto_cleanup_checkpoints(self):
        """自动清理旧存档点"""
        try:
            checkpoints = self.list_checkpoints()
            
            if len(checkpoints) <= self.config['max_checkpoints']:
                return
            
            # 按创建时间排序，保留最新的
            checkpoints.sort(key=lambda x: x['creation_time'], reverse=True)
            
            # 确定要删除的存档点
            to_delete = checkpoints[self.config['max_checkpoints']:]
            
            # 保留Phase级别的存档点
            if self.config['keep_phase_checkpoints']:
                phase_names = ['data_prepared', 'subject_analysis', 'separability', 'embedding_design', 'final_decision']
                to_delete = [ckpt for ckpt in to_delete 
                           if not any(phase in ckpt['name'] for phase in phase_names)]
            
            # 删除旧存档点
            deleted_count = 0
            for checkpoint in to_delete:
                checkpoint_path = self.checkpoint_dir / checkpoint['filename']
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"🧹 自动清理完成，删除了 {deleted_count} 个旧存档点")
                
        except Exception as e:
            logger.warning(f"自动清理失败: {e}")
    
    def create_emergency_checkpoint(self, 
                                  analyzer_instance,
                                  error_info: str = "",
                                  stack_trace: str = "") -> str:
        """创建紧急存档点（异常时调用）"""
        try:
            emergency_name = f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = f"异常紧急保存: {error_info[:100]}"
            
            checkpoint_path = self.create_checkpoint(
                analyzer_instance=analyzer_instance,
                checkpoint_name=emergency_name,
                phase_completed=-1,
                description=description,
                is_auto=True,
                include_deep_networks=False  # 紧急情况下跳过深度网络状态
            )
            
            # 额外保存错误信息
            error_log_path = self.checkpoint_dir / f"{emergency_name}_error.log"
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Emergency Checkpoint Created: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_info}\n")
                f.write(f"Stack Trace:\n{stack_trace}\n")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"紧急存档点创建失败: {e}")
            return ""
    
    def get_recovery_options(self) -> List[Dict]:
        """获取恢复选项列表"""
        checkpoints = self.list_checkpoints()
        
        recovery_options = []
        
        # 添加重新开始选项
        recovery_options.append({
            'index': 0,
            'name': '重新开始完整分析',
            'description': '从头开始执行完整分析',
            'time_saved': '0分钟',
            'checkpoint_file': None
        })
        
        # 添加存档点恢复选项
        for i, checkpoint in enumerate(checkpoints, 1):
            phase_completed = checkpoint['phase_completed']
            
            # 估算节省时间
            if phase_completed >= 0:
                estimated_time_saved = (phase_completed + 1) * 15  # 假设每Phase平均15分钟
                time_saved_str = f"{estimated_time_saved}分钟"
            else:
                time_saved_str = "未知"
            
            # 生成友好的描述
            phase_descriptions = {
                0: "数据准备完成",
                1: "受试者差异分析完成", 
                2: "可分离性评估完成",
                3: "Embedding设计完成",
                4: "最终决策完成"
            }
            
            if phase_completed in phase_descriptions:
                description = f"Phase {phase_completed}: {phase_descriptions[phase_completed]}"
            else:
                description = checkpoint['description'] or checkpoint['name']
            
            recovery_options.append({
                'index': i,
                'name': description,
                'description': f"{checkpoint['creation_time']} - {checkpoint['file_size_mb']:.1f}MB",
                'time_saved': time_saved_str,
                'checkpoint_file': checkpoint['filename']
            })
        
        return recovery_options[:11]  # 最多显示10个存档点选项 + 重新开始
    
    def interactive_recovery_selection(self) -> Optional[str]:
        """交互式恢复选择"""
        recovery_options = self.get_recovery_options()
        
        if len(recovery_options) <= 1:
            logger.info("📂 未找到可用的存档点，将重新开始分析")
            return None
        
        print("\n" + "="*80)
        print("🔄 检测到存档点，请选择继续方式：")
        print("="*80)
        
        for option in recovery_options:
            print(f"[{option['index']}] {option['name']}")
            print(f"    {option['description']} - 节省 {option['time_saved']}")
            print()
        
        while True:
            try:
                choice = input(f"请输入选择 [0-{len(recovery_options)-1}]: ").strip()
                
                if choice == '':
                    continue
                
                choice_index = int(choice)
                
                if 0 <= choice_index < len(recovery_options):
                    selected_option = recovery_options[choice_index]
                    
                    if choice_index == 0:
                        print("✅ 选择重新开始完整分析")
                        return None
                    else:
                        checkpoint_file = selected_option['checkpoint_file']
                        print(f"✅ 选择恢复存档点: {selected_option['name']}")
                        return checkpoint_file
                else:
                    print(f"❌ 请输入 0 到 {len(recovery_options)-1} 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n\n🛑 用户取消，将重新开始分析")
                return None


def create_checkpoint_decorator(checkpoint_manager, checkpoint_name: str, phase: int):
    """创建存档点装饰器"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                # 执行原函数
                result = func(self, *args, **kwargs)
                
                # 创建存档点
                checkpoint_manager.create_checkpoint(
                    analyzer_instance=self,
                    checkpoint_name=checkpoint_name,
                    phase_completed=phase,
                    description=f"Phase {phase} 完成: {func.__name__}",
                    is_auto=True
                )
                
                return result
                
            except Exception as e:
                # 异常时创建紧急存档点
                checkpoint_manager.create_emergency_checkpoint(
                    analyzer_instance=self,
                    error_info=str(e),
                    stack_trace=traceback.format_exc()
                )
                raise
        
        return wrapper
    return decorator