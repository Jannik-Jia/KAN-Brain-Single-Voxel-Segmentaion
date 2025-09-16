#!/usr/bin/env python3
"""
完整的训练性能诊断脚本

检查训练慢的根本原因：
1. h5py I/O性能
2. 数据预处理开销
3. 内存使用情况
4. GPU利用率
5. 不同数据加载模式对比
"""

import os
import sys
import time
import psutil
import threading
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import json

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 添加模型路径
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from dataset import MRIBrain2DPatchDataset, create_data_loaders
from resnet import mri_resnet50


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.cpu_percent = []
        self.memory_percent = []
        self.memory_gb = []
        self.gpu_memory = []
        self.gpu_utilization = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None

        # 检查GPU
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_nvidia_ml = True
            except:
                self.has_nvidia_ml = False
                print("Warning: pynvml not available, GPU monitoring disabled")
        else:
            self.has_nvidia_ml = False
            print("No GPU detected")

    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            # CPU和内存
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            self.cpu_percent.append(cpu_percent)
            self.memory_percent.append(memory.percent)
            self.memory_gb.append(memory.used / 1024**3)

            # GPU监控
            if self.has_nvidia_ml:
                try:
                    import pynvml
                    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

                    self.gpu_memory.append(gpu_info.used / 1024**3)
                    self.gpu_utilization.append(gpu_util.gpu)
                except:
                    self.gpu_memory.append(0)
                    self.gpu_utilization.append(0)
            else:
                self.gpu_memory.append(0)
                self.gpu_utilization.append(0)

            self.timestamps.append(time.time())
            time.sleep(1)  # 每秒采样一次

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.timestamps:
            return {}

        return {
            'cpu_avg': np.mean(self.cpu_percent),
            'cpu_max': np.max(self.cpu_percent),
            'memory_avg_gb': np.mean(self.memory_gb),
            'memory_max_gb': np.max(self.memory_gb),
            'memory_avg_percent': np.mean(self.memory_percent),
            'gpu_memory_avg_gb': np.mean(self.gpu_memory),
            'gpu_memory_max_gb': np.max(self.gpu_memory),
            'gpu_util_avg': np.mean(self.gpu_utilization),
            'gpu_util_max': np.max(self.gpu_utilization),
            'duration_seconds': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        }


def test_h5py_io_performance(mat_files: List[Path], num_tests: int = 100) -> Dict:
    """测试h5py I/O性能"""
    print(f"\n=== H5PY I/O性能测试 ===")
    print(f"测试文件数量: {len(mat_files)}")
    print(f"每个文件测试次数: {num_tests}")

    results = []

    for mat_file in mat_files:
        print(f"\n测试文件: {mat_file.name}")

        # 测试完整文件加载
        start_time = time.time()
        with h5py.File(mat_file, 'r') as f:
            data_shape = f['data'].shape
            labels_shape = f['region_labels'].shape
            mask_shape = f['region_mask'].shape
        load_info_time = time.time() - start_time

        # 测试完整数据加载
        start_time = time.time()
        with h5py.File(mat_file, 'r') as f:
            data = f['data'][()]
            labels = f['region_labels'][()]
            mask = f['region_mask'][()]
        full_load_time = time.time() - start_time

        data_size_gb = data.nbytes / 1024**3
        labels_size_mb = labels.nbytes / 1024**2
        mask_size_mb = mask.nbytes / 1024**2

        print(f"  数据形状: data={data_shape}, labels={labels_shape}, mask={mask_shape}")
        print(f"  数据大小: data={data_size_gb:.2f}GB, labels={labels_size_mb:.1f}MB, mask={mask_size_mb:.1f}MB")
        print(f"  文件信息加载时间: {load_info_time:.3f}s")
        print(f"  完整数据加载时间: {full_load_time:.3f}s")
        print(f"  加载速度: {data_size_gb/full_load_time:.2f} GB/s")

        # 测试随机patch提取性能
        patch_times = []
        patch_size = 7
        p = patch_size // 2

        print(f"  测试{num_tests}次随机patch提取...")

        # 生成随机坐标
        valid_positions = np.where((mask > 0) & (labels > 0))
        valid_coords = list(zip(valid_positions[0], valid_positions[1], valid_positions[2]))
        test_coords = np.random.choice(len(valid_coords), min(num_tests, len(valid_coords)), replace=False)

        for i in tqdm(test_coords, desc="  Patch提取"):
            x, y, z = valid_coords[i]

            # 计算patch边界
            x_min = max(0, x - p)
            x_max = min(data_shape[1] if data_shape[0] == 351 else data_shape[0], x + p + 1)
            y_min = max(0, y - p)
            y_max = min(data_shape[2] if data_shape[0] == 351 else data_shape[1], y + p + 1)

            start_time = time.time()
            with h5py.File(mat_file, 'r') as f:
                # 模拟实际的patch提取（包括数据格式处理）
                data_ref = f['data']
                if data_ref.shape[0] == 351:
                    patch = data_ref[:, x_min:x_max, y_min:y_max, z][()]
                    patch = patch.transpose(1, 2, 0)
                else:
                    patch = data_ref[x_min:x_max, y_min:y_max, z, :][()]

                # 模拟z-score标准化
                patch = patch.astype(np.float32)
                for ch in range(patch.shape[2]):
                    channel_data = patch[:, :, ch]
                    mean_val = np.mean(channel_data)
                    std_val = np.std(channel_data)
                    if std_val > 1e-8:
                        patch[:, :, ch] = (channel_data - mean_val) / std_val
                    else:
                        patch[:, :, ch] = 0

            patch_time = time.time() - start_time
            patch_times.append(patch_time)

        avg_patch_time = np.mean(patch_times)
        print(f"  平均patch提取时间: {avg_patch_time*1000:.2f}ms")
        print(f"  patch提取速度: {1/avg_patch_time:.1f} patches/s")

        results.append({
            'file': mat_file.name,
            'data_shape': data_shape,
            'data_size_gb': data_size_gb,
            'full_load_time': full_load_time,
            'load_speed_gb_s': data_size_gb / full_load_time,
            'avg_patch_time_ms': avg_patch_time * 1000,
            'patch_speed_per_s': 1 / avg_patch_time,
            'num_valid_coords': len(valid_coords)
        })

    return {
        'individual_files': results,
        'summary': {
            'avg_patch_time_ms': np.mean([r['avg_patch_time_ms'] for r in results]),
            'avg_patch_speed_per_s': np.mean([r['patch_speed_per_s'] for r in results]),
            'total_valid_patches': sum([r['num_valid_coords'] for r in results])
        }
    }


def test_dataset_loading_modes(mat_files: List[Path], batch_size: int = 32) -> Dict:
    """测试不同数据加载模式的性能"""
    print(f"\n=== 数据加载模式性能对比 ===")

    results = {}

    # 测试配置
    configs = [
        {
            'name': 'Memory-Efficient Mode',
            'memory_efficient': True,
            'samples_per_subject': None,
            'num_workers': 0
        },
        {
            'name': 'Original Mode (1K samples)',
            'memory_efficient': False,
            'samples_per_subject': 1000,
            'num_workers': 4
        },
        {
            'name': 'Original Mode (10K samples)',
            'memory_efficient': False,
            'samples_per_subject': 10000,
            'num_workers': 4
        }
    ]

    train_files = mat_files[:2]  # 只用前2个文件测试
    test_files = mat_files[:1]

    for config in configs:
        print(f"\n--- {config['name']} ---")

        monitor = PerformanceMonitor()

        try:
            # 创建数据集
            monitor.start_monitoring()
            start_time = time.time()

            train_loader, test_loader = create_data_loaders(
                train_files=train_files,
                test_files=test_files,
                batch_size=batch_size,
                num_workers=config['num_workers'],
                samples_per_subject=config['samples_per_subject'],
                memory_efficient=config['memory_efficient']
            )

            dataset_creation_time = time.time() - start_time

            print(f"  数据集创建时间: {dataset_creation_time:.2f}s")
            print(f"  训练样本数: {len(train_loader.dataset):,}")
            print(f"  批次数: {len(train_loader)}")

            # 测试数据加载速度
            num_batches_to_test = min(10, len(train_loader))
            batch_times = []

            print(f"  测试前{num_batches_to_test}个批次的加载速度...")

            for i, (batch_data, batch_labels) in enumerate(train_loader):
                if i >= num_batches_to_test:
                    break

                batch_start = time.time()
                # 模拟一些处理
                _ = batch_data.shape
                _ = batch_labels.shape
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                print(f"    Batch {i+1}: {batch_time*1000:.1f}ms, Shape: {batch_data.shape}")

            monitor.stop_monitoring()

            avg_batch_time = np.mean(batch_times)
            samples_per_second = batch_size / avg_batch_time

            stats = monitor.get_stats()

            results[config['name']] = {
                'dataset_creation_time': dataset_creation_time,
                'train_samples': len(train_loader.dataset),
                'num_batches': len(train_loader),
                'avg_batch_time_ms': avg_batch_time * 1000,
                'samples_per_second': samples_per_second,
                'estimated_epoch_time_minutes': (len(train_loader) * avg_batch_time) / 60,
                'memory_usage': stats
            }

            print(f"  平均批次加载时间: {avg_batch_time*1000:.1f}ms")
            print(f"  样本处理速度: {samples_per_second:.1f} samples/s")
            print(f"  预估单epoch时间: {results[config['name']]['estimated_epoch_time_minutes']:.1f}分钟")
            print(f"  内存使用: {stats.get('memory_avg_gb', 0):.1f}GB (平均), {stats.get('memory_max_gb', 0):.1f}GB (峰值)")

        except Exception as e:
            monitor.stop_monitoring()
            print(f"  错误: {str(e)}")
            results[config['name']] = {'error': str(e)}

    return results


def test_model_forward_speed(batch_size: int = 32) -> Dict:
    """测试模型前向传播速度"""
    print(f"\n=== 模型前向传播速度测试 ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    results = {}

    # 测试不同模型大小
    configs = [
        {'name': 'Small (base_width=32)', 'base_width': 32},
        {'name': 'Medium (base_width=64)', 'base_width': 64},
        {'name': 'Large (base_width=104)', 'base_width': 104},
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")

        try:
            # 创建模型
            model = mri_resnet50(
                input_channels=351,
                num_classes=102,
                base_width=config['base_width']
            ).to(device)

            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")

            # 创建随机输入
            dummy_input = torch.randn(batch_size, 351, 7, 7).to(device)

            # 预热
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)

            # 测试前向传播速度
            forward_times = []
            num_tests = 50

            print(f"  测试{num_tests}次前向传播...")

            with torch.no_grad():
                for _ in tqdm(range(num_tests), desc="  Forward pass"):
                    start_time = time.time()
                    output = model(dummy_input)
                    torch.cuda.synchronize() if device == 'cuda' else None
                    forward_time = time.time() - start_time
                    forward_times.append(forward_time)

            avg_forward_time = np.mean(forward_times)
            samples_per_second = batch_size / avg_forward_time

            results[config['name']] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'avg_forward_time_ms': avg_forward_time * 1000,
                'samples_per_second': samples_per_second,
                'batch_size': batch_size
            }

            print(f"  平均前向传播时间: {avg_forward_time*1000:.2f}ms")
            print(f"  样本处理速度: {samples_per_second:.1f} samples/s")

            # 清理GPU内存
            del model
            del dummy_input
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  错误: {str(e)}")
            results[config['name']] = {'error': str(e)}

    return results


def analyze_data_bottlenecks(mat_files: List[Path]) -> Dict:
    """分析数据瓶颈"""
    print(f"\n=== 数据瓶颈分析 ===")

    # 分析文件大小和分布
    file_sizes = []
    total_patches = 0

    for mat_file in mat_files:
        file_size = mat_file.stat().st_size / 1024**3  # GB
        file_sizes.append(file_size)

        # 快速估算patch数量
        with h5py.File(mat_file, 'r') as f:
            mask = f['region_mask'][()]
            labels = f['region_labels'][()]
            if mask.shape != (384, 336, 256):
                mask = mask.T
            if labels.shape != (384, 336, 256):
                labels = labels.T

            valid_voxels = np.sum((mask > 0) & (labels > 0))
            total_patches += valid_voxels

        print(f"  {mat_file.name}: {file_size:.2f}GB, {valid_voxels:,} patches")

    avg_file_size = np.mean(file_sizes)
    total_file_size = sum(file_sizes)

    # 估算理论训练时间
    theoretical_analysis = {}

    # 假设不同的patch处理速度
    patch_speeds = [10, 50, 100, 500, 1000]  # patches per second

    for speed in patch_speeds:
        time_per_epoch = total_patches / speed
        theoretical_analysis[f'{speed}_patches_per_s'] = {
            'time_per_epoch_hours': time_per_epoch / 3600,
            'time_100_epochs_hours': time_per_epoch * 100 / 3600
        }

    return {
        'file_stats': {
            'num_files': len(mat_files),
            'avg_file_size_gb': avg_file_size,
            'total_file_size_gb': total_file_size,
            'individual_sizes_gb': file_sizes
        },
        'patch_stats': {
            'total_patches': total_patches,
            'avg_patches_per_file': total_patches / len(mat_files)
        },
        'theoretical_training_time': theoretical_analysis
    }


def generate_report(results: Dict, output_dir: Path):
    """生成诊断报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    with open(output_dir / 'performance_diagnosis_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # 生成总结报告
    report = []
    report.append("# 训练性能诊断报告\n")

    # H5PY性能分析
    if 'h5py_performance' in results:
        h5py_results = results['h5py_performance']
        report.append("## H5PY I/O性能分析\n")

        summary = h5py_results.get('summary', {})
        avg_patch_time = summary.get('avg_patch_time_ms', 0)
        avg_patch_speed = summary.get('avg_patch_speed_per_s', 0)

        report.append(f"- 平均patch提取时间: {avg_patch_time:.2f}ms")
        report.append(f"- 平均patch提取速度: {avg_patch_speed:.1f} patches/s")

        if avg_patch_time > 100:
            report.append("⚠️ **警告**: patch提取时间过长 (>100ms)，h5py I/O是主要瓶颈")
        elif avg_patch_time > 50:
            report.append("⚠️ **注意**: patch提取时间较长 (>50ms)，建议优化I/O")
        else:
            report.append("✅ patch提取速度正常")

        report.append("")

    # 数据加载模式对比
    if 'dataset_performance' in results:
        report.append("## 数据加载模式性能对比\n")

        dataset_results = results['dataset_performance']
        for mode, data in dataset_results.items():
            if 'error' not in data:
                report.append(f"### {mode}")
                report.append(f"- 数据集创建时间: {data.get('dataset_creation_time', 0):.2f}s")
                report.append(f"- 平均批次加载时间: {data.get('avg_batch_time_ms', 0):.1f}ms")
                report.append(f"- 样本处理速度: {data.get('samples_per_second', 0):.1f} samples/s")
                report.append(f"- 预估单epoch时间: {data.get('estimated_epoch_time_minutes', 0):.1f}分钟")

                memory_stats = data.get('memory_usage', {})
                report.append(f"- 内存使用: {memory_stats.get('memory_avg_gb', 0):.1f}GB (平均)")
                report.append("")

        # 找出最佳模式
        best_mode = None
        best_speed = 0
        for mode, data in dataset_results.items():
            if 'error' not in data and data.get('samples_per_second', 0) > best_speed:
                best_speed = data['samples_per_second']
                best_mode = mode

        if best_mode:
            report.append(f"📊 **推荐模式**: {best_mode} (速度: {best_speed:.1f} samples/s)")
            report.append("")

    # 模型前向传播分析
    if 'model_performance' in results:
        report.append("## 模型前向传播性能\n")

        model_results = results['model_performance']
        for model_name, data in model_results.items():
            if 'error' not in data:
                report.append(f"### {model_name}")
                report.append(f"- 参数数量: {data.get('total_params', 0):,}")
                report.append(f"- 前向传播时间: {data.get('avg_forward_time_ms', 0):.2f}ms")
                report.append(f"- 样本处理速度: {data.get('samples_per_second', 0):.1f} samples/s")
                report.append("")

    # 瓶颈分析总结
    if 'bottleneck_analysis' in results:
        report.append("## 瓶颈分析总结\n")

        bottleneck = results['bottleneck_analysis']
        patch_stats = bottleneck.get('patch_stats', {})
        total_patches = patch_stats.get('total_patches', 0)

        report.append(f"- 总patch数量: {total_patches:,}")

        theoretical = bottleneck.get('theoretical_training_time', {})
        for speed_key, time_data in theoretical.items():
            speed = speed_key.replace('_patches_per_s', '')
            epoch_hours = time_data.get('time_per_epoch_hours', 0)
            report.append(f"- 以{speed} patches/s速度: 单epoch需要{epoch_hours:.1f}小时")

        report.append("")

    # 优化建议
    report.append("## 优化建议\n")

    # 基于结果给出具体建议
    if 'h5py_performance' in results:
        avg_patch_time = results['h5py_performance']['summary'].get('avg_patch_time_ms', 0)
        if avg_patch_time > 100:
            report.append("### 紧急优化 (h5py瓶颈)")
            report.append("1. **关闭memory_efficient模式**: 使用`--memory_efficient false`")
            report.append("2. **预处理数据**: 将h5py文件转换为PyTorch格式")
            report.append("3. **减少样本数**: 使用`--samples_per_subject 1000`进行快速测试")
            report.append("4. **使用SSD存储**: 如果数据在机械硬盘上，移至SSD")
            report.append("")

    if 'dataset_performance' in results:
        dataset_results = results['dataset_performance']
        memory_efficient_error = 'Memory-Efficient Mode' in dataset_results and 'error' in dataset_results['Memory-Efficient Mode']

        if memory_efficient_error or ('Memory-Efficient Mode' in dataset_results and
            dataset_results['Memory-Efficient Mode'].get('samples_per_second', 0) < 10):
            report.append("### 数据加载优化")
            report.append("1. **避免memory-efficient模式**: 该模式I/O开销太大")
            report.append("2. **增加num_workers**: 使用`--num_workers 8`")
            report.append("3. **增加batch_size**: 在显存允许情况下增大到512或1024")
            report.append("4. **启用数据预加载**: 确保`cache_data=True`")
            report.append("")

    report.append("### 通用优化建议")
    report.append("1. **减少模型复杂度**: 先用`--base_width 64`测试")
    report.append("2. **使用混合精度**: 确保开启AMP")
    report.append("3. **监控GPU利用率**: 确保GPU充分利用")
    report.append("4. **批量数据预处理**: 考虑预先处理所有数据")

    # 保存报告
    with open(output_dir / 'performance_diagnosis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n✅ 诊断报告已保存到: {output_dir}")
    print(f"   - 详细结果: performance_diagnosis_detailed.json")
    print(f"   - 总结报告: performance_diagnosis_report.md")


def main():
    parser = argparse.ArgumentParser(description='训练性能诊断工具')
    parser.add_argument('--data_dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/',
                       help='MAT文件目录')
    parser.add_argument('--output_dir', type=str, default='./diagnosis_results',
                       help='诊断结果输出目录')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='测试用批次大小')
    parser.add_argument('--num_h5py_tests', type=int, default=20,
                       help='h5py测试次数')
    parser.add_argument('--skip_h5py', action='store_true',
                       help='跳过h5py测试(如果文件太大)')
    parser.add_argument('--skip_dataset', action='store_true',
                       help='跳过数据集测试')
    parser.add_argument('--skip_model', action='store_true',
                       help='跳过模型测试')

    args = parser.parse_args()

    # 查找MAT文件
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    mat_files = sorted(data_dir.glob('*.mat'))
    if not mat_files:
        print(f"错误: 在{data_dir}中未找到MAT文件")
        return

    print(f"找到{len(mat_files)}个MAT文件")
    for i, f in enumerate(mat_files[:5]):  # 只显示前5个
        print(f"  {i+1}. {f.name}")
    if len(mat_files) > 5:
        print(f"  ... 还有{len(mat_files)-5}个文件")

    # 开始诊断
    print(f"\n🚀 开始性能诊断...")
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    results = {}

    # H5PY性能测试
    if not args.skip_h5py:
        try:
            results['h5py_performance'] = test_h5py_io_performance(
                mat_files[:3],  # 只测试前3个文件
                num_tests=args.num_h5py_tests
            )
        except Exception as e:
            print(f"H5PY测试失败: {e}")
            results['h5py_performance'] = {'error': str(e)}

    # 数据集加载性能测试
    if not args.skip_dataset:
        try:
            results['dataset_performance'] = test_dataset_loading_modes(
                mat_files[:3],  # 只测试前3个文件
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"数据集测试失败: {e}")
            results['dataset_performance'] = {'error': str(e)}

    # 模型前向传播测试
    if not args.skip_model:
        try:
            results['model_performance'] = test_model_forward_speed(
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"模型测试失败: {e}")
            results['model_performance'] = {'error': str(e)}

    # 数据瓶颈分析
    try:
        results['bottleneck_analysis'] = analyze_data_bottlenecks(mat_files)
    except Exception as e:
        print(f"瓶颈分析失败: {e}")
        results['bottleneck_analysis'] = {'error': str(e)}

    # 生成报告
    output_dir = Path(args.output_dir)
    generate_report(results, output_dir)

    print(f"\n🎯 诊断完成! 请查看报告文件了解详细分析结果。")


if __name__ == "__main__":
    main()