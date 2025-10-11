#!/usr/bin/env python3
"""
MRI Multi-modal Downsampling Pipeline - Batch Processing
=========================================================

自动化批处理脚本，用于将3D_validated数据集下采样到CEST分辨率

特性：
- 自动遍历所有3D验证数据
- 详细日志记录和进度跟踪
- 断点续传支持
- 内存监控和错误处理
- 生成QA报告和处理摘要

Author: KAN-Brain Project
Date: 2025-01-11
Version: 1.2.0 (with BUGFIX v1.2 - anisotropic Gaussian and probability labels fixed)
"""

import numpy as np
import scipy.io as sio
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
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 导入downsampling pipeline
from mri_downsampling_pipeline import MRIDownsamplingPipeline


class BatchDownsamplingProcessor:
    """批量downsampling处理器，带完整日志和错误处理"""

    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 log_level: str = 'INFO',
                 target_spacing: Tuple[float, float, float] = (1.8, 1.8, 3.0)):
        """
        初始化批量处理器

        Args:
            input_dir: 输入目录（3D_validated数据）
            output_dir: 输出目录（downsampled数据）
            log_level: 日志级别
            target_spacing: 目标分辨率 (X, Y, Z) in mm
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_spacing = target_spacing

        # 设置日志系统
        self.setup_logging(log_level)

        # 结果跟踪
        self.processing_results = []
        self.failed_subjects = []

        # 进程监控
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # 断点续传
        self.checkpoint_file = self.output_dir / "downsampling_checkpoint.pkl"
        self.completed_subjects = self.load_checkpoint()

        # 信号处理（优雅退出）
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.logger.info(f"批量Downsampling处理器初始化完成")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"目标分辨率: {self.target_spacing} mm")
        self.logger.info(f"初始内存使用: {self.initial_memory:.1f} MB")

    def setup_logging(self, log_level: str):
        """设置日志系统"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"batch_downsampling_{timestamp}.log"

        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

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

        if current_memory > 16000:  # 16GB
            self.logger.warning(f"内存使用过高: {current_memory:.1f} MB")

    def load_3d_validated_data(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """
        加载3D验证数据

        Args:
            mat_path: MAT文件路径

        Returns:
            包含data, region_mask, region_labels的字典
        """
        self.logger.debug(f"加载3D验证数据: {mat_path.name}")
        start_time = time.time()

        try:
            # 尝试使用scipy.io.loadmat加载
            try:
                mat_data = sio.loadmat(mat_path)
                self.logger.debug(f"使用scipy.io.loadmat加载成功")
            except (NotImplementedError, ValueError) as e:
                # 如果是HDF5格式或版本不兼容，使用h5py
                self.logger.info(f"scipy.io.loadmat失败 ({str(e)}), 尝试使用h5py加载HDF5格式")
                mat_data = {}
                with h5py.File(mat_path, 'r') as f:
                    # 读取关键数据
                    for key in ['data', 'region_mask', 'region_labels']:
                        if key in f:
                            mat_data[key] = f[key][()]
                            self.logger.debug(f"读取 {key}: {mat_data[key].shape}")

                            # MATLAB保存的HDF5可能需要转置
                            # MATLAB是列优先（Fortran order），Python是行优先（C order）
                            if mat_data[key].ndim >= 2:
                                # 对于2D以上的数组，可能需要转置轴
                                # data应该是(C, Z, Y, X) -> (Z, X, Y, C)或(X, Y, Z, C)
                                # 我们需要检查并调整
                                shape = mat_data[key].shape
                                self.logger.debug(f"  原始shape: {shape}")

                                # 如果是4D数据且第一维是351，需要转置
                                if key == 'data' and len(shape) == 4 and shape[0] == 351:
                                    # MATLAB: (351, Z, Y, X) -> Python: (Z, X, Y, 351)
                                    mat_data[key] = np.transpose(mat_data[key], (3, 2, 1, 0))
                                    self.logger.debug(f"  转置后shape: {mat_data[key].shape}")
                                # 如果是3D数据，可能也需要转置
                                elif key in ['region_mask', 'region_labels'] and len(shape) == 3:
                                    # 检查是否需要转置（MATLAB可能是 (Z, Y, X)）
                                    if shape != (384, 336, 256):
                                        # 尝试转置
                                        mat_data[key] = np.transpose(mat_data[key], (2, 1, 0))
                                        self.logger.debug(f"  转置后shape: {mat_data[key].shape}")

                self.logger.info(f"使用h5py加载HDF5格式成功")

            # 验证必需的keys
            required_keys = ['data', 'region_mask', 'region_labels']
            for key in required_keys:
                if key not in mat_data:
                    raise ValueError(f"缺少必需的key: {key}")

            # 验证和修正数据shape
            data = mat_data['data']
            region_mask = mat_data['region_mask']
            region_labels = mat_data['region_labels']

            # 验证data shape
            if data.ndim != 4:
                raise ValueError(f"data应该是4D数组，实际是{data.ndim}D: {data.shape}")

            # 如果data的最后一维不是351，可能需要调整
            if data.shape[-1] != 351:
                self.logger.warning(f"data shape异常: {data.shape}, 尝试调整...")
                # 如果第一维是351，转置
                if data.shape[0] == 351:
                    data = np.transpose(data, (1, 2, 3, 0))
                    self.logger.info(f"data转置后shape: {data.shape}")
                else:
                    raise ValueError(f"无法识别的data shape: {data.shape}, 期望最后一维是351")

            # 验证3D shape（期望是384, 336, 256）
            expected_spatial = (384, 336, 256)
            data_spatial = data.shape[:3]

            # 如果空间维度不匹配，尝试转置
            if data_spatial != expected_spatial:
                self.logger.warning(f"空间维度不匹配: data={data_spatial}, 期望={expected_spatial}")

                # 尝试所有可能的转置组合，找到匹配的
                from itertools import permutations
                found = False
                for perm in permutations([0, 1, 2]):
                    test_shape = tuple(data_spatial[i] for i in perm)
                    if test_shape == expected_spatial:
                        self.logger.info(f"找到匹配的转置: {perm}")
                        # 应用转置（包括通道维度）
                        full_perm = tuple(list(perm) + [3])
                        data = np.transpose(data, full_perm)
                        # 对mask和labels应用同样的转置（只前3维）
                        if region_mask.shape != expected_spatial:
                            region_mask = np.transpose(region_mask, perm)
                        if region_labels.shape != expected_spatial:
                            region_labels = np.transpose(region_labels, perm)
                        found = True
                        break

                if not found:
                    raise ValueError(f"无法将data shape {data_spatial}转换为期望的{expected_spatial}")

            # 最终验证
            if data.shape[:3] != expected_spatial or data.shape[3] != 351:
                raise ValueError(f"最终data shape验证失败: {data.shape}, 期望: {expected_spatial + (351,)}")

            if region_mask.shape != expected_spatial:
                raise ValueError(f"region_mask shape验证失败: {region_mask.shape}, 期望: {expected_spatial}")

            if region_labels.shape != expected_spatial:
                raise ValueError(f"region_labels shape验证失败: {region_labels.shape}, 期望: {expected_spatial}")

            load_time = time.time() - start_time
            self.logger.info(f"数据加载成功: {mat_path.name} (耗时: {load_time:.2f}秒)")
            self.logger.info(f"  data shape: {data.shape}")
            self.logger.info(f"  region_mask shape: {region_mask.shape}")
            self.logger.info(f"  region_labels shape: {region_labels.shape}")
            self.log_memory_usage(f"加载 {mat_path.name} 后")

            return {
                'data': data,
                'region_mask': region_mask,
                'region_labels': region_labels
            }

        except Exception as e:
            self.logger.error(f"加载文件失败 {mat_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def save_downsampled_data(self,
                             results: Dict[str, Any],
                             output_path: Path,
                             subject_id: str):
        """
        保存downsampled数据

        Args:
            results: Pipeline输出结果
            output_path: 输出文件路径
            subject_id: 被试ID
        """
        self.logger.debug(f"保存downsampled数据到: {output_path}")
        start_time = time.time()

        try:
            # 保存为.npz格式（压缩）
            np.savez_compressed(
                output_path,
                data_lr=results['data_lr'],
                proba_labels=results['proba_labels'],
                region_mask_lr=results['region_mask_lr']
            )

            # 保存metadata为JSON
            metadata_path = output_path.parent / f"{subject_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(results['metadata'], f, indent=2, default=str)

            # 保存QA metrics为JSON
            qa_path = output_path.parent / f"{subject_id}_qa_metrics.json"
            with open(qa_path, 'w') as f:
                json.dump(results['qa_metrics'], f, indent=2, default=str)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            save_time = time.time() - start_time
            self.logger.info(f"文件保存成功: {output_path.name} ({file_size_mb:.1f} MB, 耗时: {save_time:.2f}秒)")

            return file_size_mb

        except Exception as e:
            self.logger.error(f"保存文件失败 {output_path}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def process_single_subject(self, mat_path: Path, subject_id: str) -> Dict[str, Any]:
        """
        处理单个被试

        Args:
            mat_path: 输入MAT文件路径
            subject_id: 被试ID

        Returns:
            处理结果字典
        """
        start_time = time.time()

        self.logger.info(f"{'='*80}")
        self.logger.info(f"开始处理被试: {subject_id}")
        self.logger.info(f"输入文件: {mat_path.name}")

        result = {
            'subject_id': subject_id,
            'filename': mat_path.name,
            'start_time': datetime.now().isoformat()
        }

        try:
            # 步骤1: 加载3D验证数据
            self.logger.info("步骤1: 加载3D验证数据")
            input_data = self.load_3d_validated_data(mat_path)

            # 验证输入shape
            data_shape = input_data['data'].shape
            self.logger.info(f"输入数据shape: {data_shape}")
            if data_shape[:3] != (384, 336, 256) or data_shape[3] != 351:
                self.logger.warning(f"输入数据shape不符合预期: {data_shape}")

            # 步骤2: 初始化downsampling pipeline
            self.logger.info("步骤2: 初始化Downsampling Pipeline")
            pipeline_output_dir = self.output_dir / subject_id
            pipeline_output_dir.mkdir(exist_ok=True)

            pipeline = MRIDownsamplingPipeline(
                output_dir=pipeline_output_dir,
                log_level='INFO',
                random_seed=42
            )

            # 步骤3: 运行downsampling
            self.logger.info("步骤3: 运行Downsampling Pipeline")
            pipeline_results = pipeline.run(
                data=input_data['data'],
                region_mask=input_data['region_mask'],
                region_labels=input_data['region_labels'],
                align_to_128x104x18=False
            )

            # 步骤4: 保存结果
            self.logger.info("步骤4: 保存Downsampled数据")
            output_filename = f"{subject_id}_downsampled.npz"
            output_path = self.output_dir / output_filename
            file_size_mb = self.save_downsampled_data(
                pipeline_results,
                output_path,
                subject_id
            )

            # 记录结果
            processing_time = time.time() - start_time

            result.update({
                'status': 'success',
                'output_path': str(output_path),
                'file_size_mb': file_size_mb,
                'processing_time': processing_time,
                'input_shape': data_shape,
                'output_shape': pipeline_results['data_lr'].shape,
                'proba_labels_shape': pipeline_results['proba_labels'].shape,
                'slab_thickness_mm': pipeline_results['metadata']['slab_localization']['slab_thickness_mm'],
                'coverage_fallback': pipeline_results['metadata']['slab_localization']['coverage_fallback']
            })

            # 标记为已完成
            self.completed_subjects.add(subject_id)

            self.logger.info(f"✅ 被试 {subject_id} 处理成功")
            self.logger.info(f"   处理时间: {processing_time:.1f}秒")
            self.logger.info(f"   输出大小: {file_size_mb:.1f}MB")
            self.logger.info(f"   输出shape: {pipeline_results['data_lr'].shape}")

        except Exception as e:
            self.logger.error(f"❌ 被试 {subject_id} 处理失败")
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
            self.failed_subjects.append(subject_id)

        finally:
            # 定期保存检查点
            if len(self.completed_subjects) % 3 == 0:
                self.save_checkpoint()

            # 清理内存
            import gc
            gc.collect()

        return result

    def create_dataset_index(self) -> Dict[str, Dict[str, str]]:
        """创建数据集索引"""
        self.logger.info("创建数据集索引...")

        # 查找所有3d_validated.mat文件
        mat_files = sorted(list(self.input_dir.glob("*_3d_validated.mat")))

        if len(mat_files) == 0:
            self.logger.error(f"未找到3D验证文件在: {self.input_dir}")
            raise ValueError(f"未找到3D验证文件在: {self.input_dir}")

        self.logger.info(f"找到 {len(mat_files)} 个3D验证文件")

        index_mapping = {}
        for filepath in mat_files:
            # 提取subject_id（去掉_3d_validated后缀）
            subject_id = filepath.stem.replace('_3d_validated', '')

            index_mapping[subject_id] = {
                "subject_id": subject_id,
                "filename": filepath.name,
                "filepath": str(filepath),
                "file_size_mb": filepath.stat().st_size / (1024 * 1024)
            }
            self.logger.debug(f"索引: {subject_id} - {filepath.name} ({index_mapping[subject_id]['file_size_mb']:.1f} MB)")

        # 保存索引
        json_path = self.output_dir / "downsampling_dataset_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, indent=2, ensure_ascii=False)

        self.logger.info(f"索引已保存到: {json_path}")
        return index_mapping

    def process_all_subjects(self, skip_existing: bool = True) -> Tuple[List, List]:
        """
        批量处理所有被试

        Args:
            skip_existing: 是否跳过已存在的文件

        Returns:
            (成功列表, 失败列表)
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"批量Downsampling处理开始")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"跳过已存在: {skip_existing}")
        self.logger.info(f"已完成被试数: {len(self.completed_subjects)}")
        self.logger.info(f"{'='*80}")

        # 创建索引
        index_mapping = self.create_dataset_index()

        total_start_time = time.time()
        successful_subjects = []

        # 处理每个被试
        for subject_id, subject_info in tqdm(index_mapping.items(), desc="Downsampling进度"):
            # 检查是否已经处理过（断点续传）
            if subject_id in self.completed_subjects:
                self.logger.info(f"跳过已完成的被试: {subject_id}")
                successful_subjects.append(subject_id)
                continue

            # 检查是否跳过已存在的文件
            if skip_existing:
                output_filename = f"{subject_id}_downsampled.npz"
                output_path = self.output_dir / output_filename
                if output_path.exists():
                    self.logger.info(f"跳过已存在文件: {subject_id}")
                    self.completed_subjects.add(subject_id)
                    successful_subjects.append(subject_id)
                    continue

            # 处理被试
            mat_path = Path(subject_info['filepath'])
            result = self.process_single_subject(mat_path, subject_id)
            self.processing_results.append(result)

            if result['status'] == 'success':
                successful_subjects.append(subject_id)

            # 更新进度信息
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
            'last_results': self.processing_results[-10:] if self.processing_results else [],
            'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024
        }

        emergency_file = self.output_dir / f"emergency_report_{timestamp}.json"
        with open(emergency_file, 'w') as f:
            json.dump(emergency_report, f, indent=2, default=str)

        self.logger.info(f"紧急报告已保存: {emergency_file}")

    def _generate_batch_report(self, total_time: float):
        """生成批量处理报告"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("生成批量Downsampling报告")
        self.logger.info(f"{'='*80}")

        successful = [r for r in self.processing_results if r['status'] == 'success']
        failed = [r for r in self.processing_results if r['status'] == 'failed']

        self.logger.info(f"处理统计:")
        self.logger.info(f"  总处理数: {len(self.processing_results)}")
        self.logger.info(f"  成功处理: {len(successful)}")
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
                self.logger.error(f"  - {r['subject_id']}")
                self.logger.error(f"    错误类型: {r.get('error_type', 'Unknown')}")
                self.logger.error(f"    错误信息: {r.get('error', 'Unknown')}")

        # 保存详细报告
        self._save_detailed_reports()

    def _save_detailed_reports(self):
        """保存详细报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON报告
        json_report_path = self.output_dir / f"downsampling_report_{timestamp}.json"
        report_data = {
            'timestamp': timestamp,
            'pipeline_version': '1.2.0',
            'target_spacing_mm': self.target_spacing,
            'completed_subjects': list(self.completed_subjects),
            'failed_subjects': self.failed_subjects,
            'processing_results': self.processing_results,
            'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024
        }

        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        self.logger.info(f"JSON报告已保存: {json_report_path}")

        # CSV摘要
        if self.processing_results:
            summary_data = []
            for r in self.processing_results:
                summary_data.append({
                    'subject_id': r['subject_id'],
                    'status': r['status'],
                    'file_size_mb': r.get('file_size_mb', 0),
                    'processing_time': r['processing_time'],
                    'input_shape': str(r.get('input_shape', '')),
                    'output_shape': str(r.get('output_shape', '')),
                    'slab_thickness_mm': r.get('slab_thickness_mm', 0),
                    'coverage_fallback': r.get('coverage_fallback', False),
                    'error': r.get('error', ''),
                    'error_type': r.get('error_type', '')
                })

            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / f"downsampling_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV摘要已保存: {csv_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量Downsampling MRI数据到CEST分辨率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理所有被试
  python batch_downsampling_pipeline.py

  # 指定输入输出目录
  python batch_downsampling_pipeline.py --input-dir /path/to/3D_validated --output-dir /path/to/output

  # 测试模式（处理前3个）
  python batch_downsampling_pipeline.py --test-only

  # 强制重新处理（不跳过已存在文件）
  python batch_downsampling_pipeline.py --no-skip-existing
        """
    )

    parser.add_argument('--input-dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated',
                       help='输入目录（3D验证数据）')
    parser.add_argument('--output-dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling/3d',
                       help='输出目录（downsampled数据）')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='不跳过已存在的文件（强制重新处理）')
    parser.add_argument('--test-only', action='store_true',
                       help='测试模式（只处理前3个被试）')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--target-spacing', type=str, default='1.8,1.8,3.0',
                       help='目标分辨率 (X,Y,Z) in mm，逗号分隔')

    args = parser.parse_args()

    # 解析target_spacing
    target_spacing = tuple(map(float, args.target_spacing.split(',')))

    # 创建批量处理器
    processor = BatchDownsamplingProcessor(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        log_level=args.log_level,
        target_spacing=target_spacing
    )

    try:
        if args.test_only:
            print("测试模式：处理前3个被试")
            # 手动处理前3个
            index_mapping = processor.create_dataset_index()
            test_subjects = list(index_mapping.keys())[:3]
            for subject_id in test_subjects:
                mat_path = Path(index_mapping[subject_id]['filepath'])
                result = processor.process_single_subject(mat_path, subject_id)
                processor.processing_results.append(result)
            processor._generate_batch_report(0)
        else:
            successful, failed = processor.process_all_subjects(
                skip_existing=not args.no_skip_existing
            )

        # 最终结果
        if len(processor.failed_subjects) == 0:
            processor.logger.info(f"\n🎉 所有被试Downsampling成功！")
        else:
            processor.logger.warning(f"\n⚠️ 有 {len(processor.failed_subjects)} 个被试处理失败，请检查日志")

    except KeyboardInterrupt:
        processor.logger.warning("用户中断程序")
        processor.save_checkpoint()
        processor.generate_emergency_report()
    except Exception as e:
        processor.logger.error(f"程序异常退出: {e}")
        processor.logger.error(traceback.format_exc())
        processor.save_checkpoint()
        processor.generate_emergency_report()
        raise


if __name__ == "__main__":
    # 如果直接运行，提供交互式界面
    import sys

    if len(sys.argv) == 1:
        print("="*80)
        print("MRI Multi-modal Downsampling Pipeline - 批量处理工具")
        print("Version 1.2.0 (BUGFIX v1.2 applied)")
        print("="*80)
        print("\n特性：")
        print("1. 自动遍历所有3D验证数据")
        print("2. 详细日志记录和进度跟踪")
        print("3. 断点续传支持")
        print("4. 内存监控和错误处理")
        print("5. 生成QA报告和处理摘要")
        print("="*80)

        processor = BatchDownsamplingProcessor(
            input_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated"),
            output_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/downsampling/3d"),
            log_level='INFO'
        )

        print(f"\n输入目录: {processor.input_dir}")
        print(f"输出目录: {processor.output_dir}")
        print(f"已完成被试: {len(processor.completed_subjects)}")

        response = input("\n选择模式:\n1. 测试（只处理前3个被试）\n2. 处理所有被试\n3. 断点续传（继续之前的任务）\n请输入 (1/2/3): ")

        try:
            if response == '1':
                print("\n测试模式启动...")
                index_mapping = processor.create_dataset_index()
                test_subjects = list(index_mapping.keys())[:3]
                for subject_id in test_subjects:
                    mat_path = Path(index_mapping[subject_id]['filepath'])
                    result = processor.process_single_subject(mat_path, subject_id)
                    processor.processing_results.append(result)
                processor._generate_batch_report(0)
            elif response == '2':
                print("\n全量处理模式启动...")
                successful, failed = processor.process_all_subjects(skip_existing=True)
            elif response == '3':
                print(f"\n断点续传模式启动...")
                print(f"已完成被试: {sorted(processor.completed_subjects)}")
                successful, failed = processor.process_all_subjects(skip_existing=True)
            else:
                print("无效选择")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            processor.save_checkpoint()
            processor.generate_emergency_report()
    else:
        main()
