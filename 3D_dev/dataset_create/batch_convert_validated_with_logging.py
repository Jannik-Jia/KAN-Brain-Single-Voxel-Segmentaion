#!/usr/bin/env python3
"""
基于验证成功的方法进行批量转换 - 增强版
包含完整的验证、错误处理和详细日志记录
"""

import numpy as np
import h5py
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
import pandas as pd
import logging
import sys
import traceback
import psutil
import os
from datetime import datetime
import signal
import pickle

class LoggedBatchConverter:
    """带详细日志记录的批量转换器"""
    
    def __init__(self, data_dir: Path, output_dir: Path, log_level: str = 'DEBUG'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志系统
        self.setup_logging(log_level)
        
        # 结果跟踪
        self.conversion_results = []
        self.validation_results = []
        self.failed_subjects = []
        
        # 进程监控
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 断点续传
        self.checkpoint_file = self.output_dir / "conversion_checkpoint.pkl"
        self.completed_subjects = self.load_checkpoint()
        
        # 信号处理（优雅退出）
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info(f"转换器初始化完成")
        self.logger.info(f"数据目录: {self.data_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"初始内存使用: {self.initial_memory:.1f} MB")
        
    def setup_logging(self, log_level: str):
        """设置日志系统"""
        # 创建日志目录
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 生成时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"conversion_{timestamp}.log"
        
        # 配置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        # 设置根日志器
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"日志系统已启动，日志文件: {log_file}")
        
    def signal_handler(self, signum, frame):
        """处理中断信号"""
        self.logger.warning(f"收到信号 {signum}，正在保存检查点并退出...")
        self.save_checkpoint()
        self.generate_emergency_report()
        sys.exit(0)
        
    def load_checkpoint(self) -> set:
        """加载检查点"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.logger.info(f"加载检查点: 已完成 {len(checkpoint)} 个被试")
                    return checkpoint
            except Exception as e:
                self.logger.error(f"加载检查点失败: {e}")
        return set()
        
    def save_checkpoint(self):
        """保存检查点"""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.completed_subjects, f)
            self.logger.debug(f"检查点已保存: {len(self.completed_subjects)} 个被试完成")
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
    
    def log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.initial_memory
        self.logger.debug(f"内存使用 {context}: {current_memory:.1f} MB (变化: {memory_delta:+.1f} MB)")
        
        # 如果内存使用过高，发出警告
        if current_memory > 8000:  # 8GB
            self.logger.warning(f"内存使用过高: {current_memory:.1f} MB")
            
    def load_mat_h5_correct(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """正确的MAT文件加载方法（带日志）"""
        self.logger.debug(f"开始加载文件: {mat_path}")
        start_time = time.time()
        
        try:
            data = {}
            with h5py.File(mat_path, "r") as f:
                self.logger.debug(f"文件包含的键: {list(f.keys())}")
                
                for k in f.keys():
                    if not k.startswith("#"):
                        v = f[k][()]
                        original_shape = v.shape
                        
                        # 关键：只转置需要转置的数据
                        if k == 'multidim_data' and v.shape[0] == 351:
                            v = v.T  # 特征矩阵转置
                            self.logger.debug(f"转置 {k}: {original_shape} → {v.shape}")
                        elif k == 'region_seg':
                            v = v.flatten()  # 展平
                            self.logger.debug(f"展平 {k}: {original_shape} → {v.shape}")
                        else:
                            self.logger.debug(f"保持原样 {k}: {v.shape}")
                        
                        data[k] = v
                        
            load_time = time.time() - start_time
            self.logger.info(f"文件加载成功: {mat_path.name} (耗时: {load_time:.2f}秒)")
            self.log_memory_usage(f"加载 {mat_path.name} 后")
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载文件失败 {mat_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def convert_1d_to_3d(self, data: Dict[str, np.ndarray], prob_idx: int) -> Dict[str, np.ndarray]:
        """将1D数据转换为3D格式（带日志）"""
        self.logger.debug(f"开始转换被试 {prob_idx} 的数据到3D格式")
        start_time = time.time()
        
        try:
            region = data['region'].astype(bool)
            n_voxels = np.sum(region)
            self.logger.debug(f"ROI体素数: {n_voxels}")
            
            output_data = {}
            
            # 1. 3D体数据（直接保留）
            output_data['big_seg'] = data['big_seg'].copy()
            output_data['region_mask'] = data['region'].astype(np.uint8)
            self.logger.debug(f"3D体数据形状: big_seg={data['big_seg'].shape}, region={data['region'].shape}")
            
            # 2. 多模态特征数据：(n_voxels, 351) → (384, 336, 256, 351)
            features = data['multidim_data']
            n_voxels_check, n_features = features.shape
            
            if n_voxels != n_voxels_check:
                self.logger.warning(f"体素数不匹配: region={n_voxels}, features={n_voxels_check}")
            
            self.logger.debug(f"开始转换特征数据: {features.shape} → 4D")
            data_4d = np.zeros((*region.shape, n_features), dtype=features.dtype)
            data_4d[region] = features  # 直接布尔索引
            output_data['data'] = data_4d
            self.logger.debug(f"特征数据转换完成: {data_4d.shape}")
            
            # 3. One-Hot标签转换：(102, n_voxels) → (384, 336, 256)
            seg_one_hot = data['seg_one_hot']
            self.logger.debug(f"One-hot标签形状: {seg_one_hot.shape}")
            labels_1d = np.argmax(seg_one_hot, axis=0).astype(np.uint8)
            
            labels_3d = np.zeros(region.shape, dtype=np.uint8)
            labels_3d[region] = labels_1d
            output_data['region_labels'] = labels_3d
            
            # 4. 原始标签重构：(n_voxels,) → (384, 336, 256)
            region_seg = data['region_seg']
            region_seg_3d = np.zeros(region.shape, dtype=region_seg.dtype)
            region_seg_3d[region] = region_seg
            output_data['region_seg_3d'] = region_seg_3d
            
            # 5. 被试索引体积
            prob_idx_3d = np.zeros(region.shape, dtype=np.uint8)
            prob_idx_3d[region] = prob_idx
            output_data['prob_idx'] = prob_idx_3d
            
            convert_time = time.time() - start_time
            self.logger.info(f"被试 {prob_idx} 转换完成 (耗时: {convert_time:.2f}秒)")
            self.log_memory_usage(f"转换被试 {prob_idx} 后")
            
            return output_data
            
        except Exception as e:
            self.logger.error(f"转换被试 {prob_idx} 失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def save_3d_data(self, data_3d: Dict[str, np.ndarray], output_path: Path):
        """保存3D数据为MAT文件（带日志）"""
        self.logger.debug(f"开始保存3D数据到: {output_path}")
        start_time = time.time()
        
        try:
            with h5py.File(output_path, 'w') as f:
                for key, value in data_3d.items():
                    original_shape = value.shape
                    
                    # 为MATLAB兼容性调整格式
                    if key == 'data' and value.ndim == 4:
                        # 4D数据：(384,336,256,351) → (351,384,336,256)
                        save_value = np.moveaxis(value, -1, 0)
                        self.logger.debug(f"调整 {key} 维度: {original_shape} → {save_value.shape}")
                    else:
                        save_value = value
                    
                    # 使用Fortran连续性优化MATLAB读取
                    save_value = np.asfortranarray(save_value)
                    
                    # 根据数据大小选择压缩级别
                    if key == 'data':
                        f.create_dataset(key, data=save_value, compression='gzip', compression_opts=1)
                        self.logger.debug(f"保存 {key}: 使用低压缩级别")
                    else:
                        f.create_dataset(key, data=save_value, compression='gzip', compression_opts=4)
                        self.logger.debug(f"保存 {key}: 使用中压缩级别")
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            save_time = time.time() - start_time
            self.logger.info(f"文件保存成功: {output_path.name} ({file_size_mb:.1f} MB, 耗时: {save_time:.2f}秒)")
            
        except Exception as e:
            self.logger.error(f"保存文件失败 {output_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def quick_validation(self, original_data: Dict[str, np.ndarray], 
                        converted_3d: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """快速验证转换结果（带日志）"""
        self.logger.debug("开始验证转换结果")
        
        try:
            region = original_data['region'].astype(bool)
            results = {}
            
            # 验证特征数据
            original_features = original_data['multidim_data']
            recovered_features = converted_3d['data'][region]
            features_match = np.allclose(original_features, recovered_features)
            results['multidim_data'] = features_match
            if not features_match:
                self.logger.warning("特征数据验证失败")
            
            # 验证region_seg
            original_region_seg = original_data['region_seg']
            recovered_region_seg = converted_3d['region_seg_3d'][region]
            seg_match = np.array_equal(original_region_seg, recovered_region_seg)
            results['region_seg'] = seg_match
            if not seg_match:
                self.logger.warning("region_seg验证失败")
            
            # 验证one-hot标签
            original_one_hot = original_data['seg_one_hot']
            original_labels = np.argmax(original_one_hot, axis=0)
            recovered_labels = converted_3d['region_labels'][region]
            labels_match = np.array_equal(original_labels, recovered_labels)
            results['seg_one_hot'] = labels_match
            if not labels_match:
                self.logger.warning("one-hot标签验证失败")
            
            # 验证3D数据
            results['big_seg'] = np.array_equal(original_data['big_seg'], converted_3d['big_seg'])
            results['region'] = np.array_equal(original_data['region'], converted_3d['region_mask'])
            
            all_valid = all(results.values())
            if all_valid:
                self.logger.info("所有验证通过 ✅")
            else:
                self.logger.warning(f"验证结果: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"验证失败: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': False}
    
    def process_single_subject(self, mat_path: Path, prob_idx: int) -> Dict[str, Any]:
        """处理单个被试（带完整日志和错误处理）"""
        start_time = time.time()
        subject_id = mat_path.stem
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"开始处理被试 {prob_idx}: {subject_id}")
        self.logger.info(f"文件: {mat_path.name}")
        
        result = {
            'prob_idx': prob_idx,
            'subject_id': subject_id,
            'filename': mat_path.name,
            'start_time': start_time
        }
        
        try:
            # 1. 加载原始数据
            self.logger.debug("步骤1: 加载原始数据")
            original_data = self.load_mat_h5_correct(mat_path)
            
            # 2. 转换为3D
            self.logger.debug("步骤2: 转换为3D格式")
            converted_3d = self.convert_1d_to_3d(original_data, prob_idx)
            
            # 3. 快速验证
            self.logger.debug("步骤3: 验证转换结果")
            validation_results = self.quick_validation(original_data, converted_3d)
            all_valid = all(validation_results.values())
            
            # 4. 保存文件
            self.logger.debug("步骤4: 保存3D数据")
            output_filename = f"{subject_id}_3d_validated.mat"
            output_path = self.output_dir / output_filename
            self.save_3d_data(converted_3d, output_path)
            
            # 5. 记录结果
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            processing_time = time.time() - start_time
            
            result.update({
                'status': 'success',
                'output_path': str(output_path),
                'file_size_mb': file_size_mb,
                'processing_time': processing_time,
                'validation_passed': all_valid,
                'validation_details': validation_results,
                'n_voxels': np.sum(original_data['region'].astype(bool)),
                'features_shape': original_data['multidim_data'].shape
            })
            
            # 标记为已完成
            self.completed_subjects.add(prob_idx)
            
            if all_valid:
                self.logger.info(f"✅ 被试 {prob_idx} 处理成功 ({processing_time:.1f}秒, {file_size_mb:.1f}MB)")
            else:
                self.logger.warning(f"⚠️ 被试 {prob_idx} 处理成功但验证未完全通过")
                for key, valid in validation_results.items():
                    if not valid:
                        self.logger.warning(f"    - {key}: 验证失败")
        
        except Exception as e:
            self.logger.error(f"❌ 被试 {prob_idx} 处理失败")
            self.logger.error(f"错误类型: {type(e).__name__}")
            self.logger.error(f"错误信息: {str(e)}")
            self.logger.error(f"详细堆栈:\n{traceback.format_exc()}")
            
            result.update({
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'processing_time': time.time() - start_time
            })
            self.failed_subjects.append(prob_idx)
        
        finally:
            # 定期保存检查点
            if len(self.completed_subjects) % 5 == 0:
                self.save_checkpoint()
            
            # 清理内存
            import gc
            gc.collect()
            
        return result
    
    def create_dataset_index(self) -> Dict[int, Dict[str, str]]:
        """创建数据集索引（带日志）"""
        self.logger.info("创建数据集索引...")
        
        mat_files = sorted(list(self.data_dir.glob("*.mat")))
        
        if len(mat_files) == 0:
            self.logger.error(f"未找到MAT文件在: {self.data_dir}")
            raise ValueError(f"未找到MAT文件在: {self.data_dir}")
        
        self.logger.info(f"找到 {len(mat_files)} 个MAT文件")
        
        index_mapping = {}
        for idx, filepath in enumerate(mat_files, start=1):
            index_mapping[idx] = {
                "prob_idx": idx,
                "filename": filepath.name,
                "filepath": str(filepath),
                "subject_id": filepath.stem,
                "file_size_mb": filepath.stat().st_size / (1024 * 1024)
            }
            self.logger.debug(f"索引 {idx}: {filepath.name} ({index_mapping[idx]['file_size_mb']:.1f} MB)")
        
        # 保存索引
        json_path = self.output_dir / "dataset_index_logged.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"索引已保存到: {json_path}")
        return index_mapping
    
    def process_all_subjects(self, start_idx: int = 1, end_idx: int = 38, 
                           skip_existing: bool = False) -> Tuple[List, List]:
        """批量处理所有被试（带完整日志）"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"批量处理开始")
        self.logger.info(f"处理范围: 被试 {start_idx} 到 {end_idx}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"跳过已存在: {skip_existing}")
        self.logger.info(f"已完成被试数: {len(self.completed_subjects)}")
        self.logger.info(f"{'='*80}")
        
        # 创建索引
        index_mapping = self.create_dataset_index()
        
        total_start_time = time.time()
        successful_subjects = []
        
        # 处理每个被试
        for prob_idx in tqdm(range(start_idx, end_idx + 1), desc="处理进度"):
            # 检查是否已经处理过（断点续传）
            if prob_idx in self.completed_subjects:
                self.logger.info(f"跳过已完成的被试: {prob_idx}")
                successful_subjects.append(prob_idx)
                continue
            
            if prob_idx not in index_mapping:
                self.logger.warning(f"跳过不存在的prob_idx: {prob_idx}")
                continue
            
            subject_info = index_mapping[prob_idx]
            mat_path = Path(subject_info['filepath'])
            
            # 检查是否跳过已存在的文件
            if skip_existing:
                output_filename = f"{subject_info['subject_id']}_3d_validated.mat"
                output_path = self.output_dir / output_filename
                if output_path.exists():
                    self.logger.info(f"跳过已存在文件: {subject_info['subject_id']}")
                    self.completed_subjects.add(prob_idx)
                    successful_subjects.append(prob_idx)
                    continue
            
            # 处理被试
            result = self.process_single_subject(mat_path, prob_idx)
            self.conversion_results.append(result)
            
            if result['status'] == 'success':
                successful_subjects.append(prob_idx)
            
            # 更新进度条描述
            tqdm.write(f"内存: {self.process.memory_info().rss / 1024 / 1024:.1f} MB | "
                      f"成功: {len(successful_subjects)} | "
                      f"失败: {len(self.failed_subjects)}")
        
        total_time = time.time() - total_start_time
        
        # 保存最终检查点
        self.save_checkpoint()
        
        # 生成报告
        self._generate_batch_report(total_time)
        
        return successful_subjects, self.failed_subjects
    
    def generate_emergency_report(self):
        """生成紧急报告（程序中断时）"""
        self.logger.info("生成紧急报告...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        emergency_report = {
            'timestamp': timestamp,
            'completed_subjects': list(self.completed_subjects),
            'failed_subjects': self.failed_subjects,
            'last_results': self.conversion_results[-10:] if self.conversion_results else [],
            'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024
        }
        
        emergency_file = self.output_dir / f"emergency_report_{timestamp}.json"
        with open(emergency_file, 'w') as f:
            json.dump(emergency_report, f, indent=2, default=str)
        
        self.logger.info(f"紧急报告已保存: {emergency_file}")
    
    def _generate_batch_report(self, total_time: float):
        """生成批量处理报告（带详细日志）"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("生成批量处理报告")
        self.logger.info(f"{'='*80}")
        
        successful = [r for r in self.conversion_results if r['status'] == 'success']
        failed = [r for r in self.conversion_results if r['status'] == 'failed']
        fully_validated = [r for r in successful if r.get('validation_passed', False)]
        
        self.logger.info(f"处理统计:")
        self.logger.info(f"  总处理数: {len(self.conversion_results)}")
        self.logger.info(f"  成功转换: {len(successful)}")
        self.logger.info(f"  验证通过: {len(fully_validated)}")
        self.logger.info(f"  处理失败: {len(failed)}")
        self.logger.info(f"  总时间: {total_time/60:.1f}分钟")
        self.logger.info(f"  最终内存: {self.process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        if successful:
            avg_time = np.mean([r['processing_time'] for r in successful])
            avg_size = np.mean([r['file_size_mb'] for r in successful])
            total_size = sum([r['file_size_mb'] for r in successful])
            self.logger.info(f"  平均处理时间: {avg_time:.1f}秒")
            self.logger.info(f"  平均文件大小: {avg_size:.1f}MB")
            self.logger.info(f"  总输出大小: {total_size:.1f}MB")
        
        # 失败的被试
        if failed:
            self.logger.error(f"\n处理失败的被试:")
            for r in failed:
                self.logger.error(f"  - 被试 {r['prob_idx']}: {r['subject_id']}")
                self.logger.error(f"    错误类型: {r.get('error_type', 'Unknown')}")
                self.logger.error(f"    错误信息: {r.get('error', 'Unknown')}")
        
        # 验证问题
        validation_issues = [r for r in successful if not r.get('validation_passed', False)]
        if validation_issues:
            self.logger.warning(f"\n验证问题的被试:")
            for r in validation_issues:
                self.logger.warning(f"  - 被试 {r['prob_idx']}: {r['subject_id']}")
                for key, valid in r.get('validation_details', {}).items():
                    if not valid:
                        self.logger.warning(f"    ❌ {key}")
        
        # 保存详细报告
        self._save_detailed_reports()
    
    def _save_detailed_reports(self):
        """保存详细报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON报告
        json_report_path = self.output_dir / f"batch_report_{timestamp}.json"
        report_data = {
            'timestamp': timestamp,
            'completed_subjects': list(self.completed_subjects),
            'failed_subjects': self.failed_subjects,
            'conversion_results': self.conversion_results,
            'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024
        }
        
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        self.logger.info(f"JSON报告已保存: {json_report_path}")
        
        # CSV摘要
        if self.conversion_results:
            summary_data = []
            for r in self.conversion_results:
                summary_data.append({
                    'prob_idx': r['prob_idx'],
                    'subject_id': r['subject_id'],
                    'status': r['status'],
                    'validation_passed': r.get('validation_passed', False),
                    'file_size_mb': r.get('file_size_mb', 0),
                    'processing_time': r['processing_time'],
                    'n_voxels': r.get('n_voxels', 0),
                    'error': r.get('error', ''),
                    'error_type': r.get('error_type', '')
                })
            
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / f"batch_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV摘要已保存: {csv_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量转换MRI数据到3D格式（带日志版）')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D',
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated',
                       help='输出目录')
    parser.add_argument('--start', type=int, default=1,
                       help='开始的prob_idx')
    parser.add_argument('--end', type=int, default=38,
                       help='结束的prob_idx')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的文件')
    parser.add_argument('--test-only', action='store_true',
                       help='只处理第一个被试用于测试')
    parser.add_argument('--log-level', type=str, default='DEBUG',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 创建批量转换器
    converter = LoggedBatchConverter(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        log_level=args.log_level
    )
    
    try:
        if args.test_only:
            print("测试模式：只处理第一个被试")
            successful, failed = converter.process_all_subjects(start_idx=1, end_idx=1)
        else:
            successful, failed = converter.process_all_subjects(
                start_idx=args.start,
                end_idx=args.end,
                skip_existing=args.skip_existing
            )
        
        # 最终结果
        if len(failed) == 0:
            converter.logger.info(f"\n🎉 所有被试处理成功！")
        else:
            converter.logger.warning(f"\n⚠️ 有 {len(failed)} 个被试处理失败，请检查日志")
            
    except KeyboardInterrupt:
        converter.logger.warning("用户中断程序")
        converter.save_checkpoint()
        converter.generate_emergency_report()
    except Exception as e:
        converter.logger.error(f"程序异常退出: {e}")
        converter.logger.error(traceback.format_exc())
        converter.save_checkpoint()
        converter.generate_emergency_report()
        raise

if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    
    if len(sys.argv) == 1:
        print("批量转换工具（增强版）")
        print("-" * 50)
        print("特性：")
        print("1. 详细日志记录")
        print("2. 断点续传支持")
        print("3. 内存监控")
        print("4. 错误追踪")
        print("5. 优雅退出")
        print("-" * 50)
        
        converter = LoggedBatchConverter(
            data_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D"),
            output_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated"),
            log_level='DEBUG'
        )
        
        response = input("\n选择模式:\n1. 测试（只处理第一个被试）\n2. 处理所有38个被试\n3. 处理前5个被试\n4. 断点续传（继续之前的任务）\n请输入 (1/2/3/4): ")
        
        try:
            if response == '1':
                successful, failed = converter.process_all_subjects(start_idx=1, end_idx=1)
            elif response == '2':
                successful, failed = converter.process_all_subjects(start_idx=1, end_idx=38)
            elif response == '3':
                successful, failed = converter.process_all_subjects(start_idx=1, end_idx=5)
            elif response == '4':
                print(f"已完成被试: {sorted(converter.completed_subjects)}")
                successful, failed = converter.process_all_subjects(start_idx=1, end_idx=38)
            else:
                print("无效选择")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            converter.save_checkpoint()
            converter.generate_emergency_report()
    else:
        main()