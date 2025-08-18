#!/usr/bin/env python3
"""
基于验证成功的方法进行批量转换
包含完整的验证和错误处理
"""

import numpy as np
import h5py
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm
import pandas as pd

class ValidatedBatchConverter:
    """经过验证的批量转换器"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果跟踪
        self.conversion_results = []
        self.validation_results = []
        self.failed_subjects = []
        
    def load_mat_h5_correct(self, mat_path: Path) -> Dict[str, np.ndarray]:
        """正确的MAT文件加载方法"""
        data = {}
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                if not k.startswith("#"):
                    v = f[k][()]
                    
                    # 关键：只转置需要转置的数据
                    if k == 'multidim_data' and v.shape[0] == 351:
                        v = v.T  # 特征矩阵转置
                    elif k == 'region_seg':
                        v = v.flatten()  # 展平
                    # 其他数据保持原样
                    data[k] = v
        return data
    
    def convert_1d_to_3d(self, data: Dict[str, np.ndarray], prob_idx: int) -> Dict[str, np.ndarray]:
        """将1D数据转换为3D格式（验证过的方法）"""
        region = data['region'].astype(bool)
        output_data = {}
        
        # 1. 3D体数据（直接保留）
        output_data['big_seg'] = data['big_seg'].copy()
        output_data['region_mask'] = data['region'].astype(np.uint8)
        
        # 2. 多模态特征数据：(n_voxels, 351) → (384, 336, 256, 351)
        features = data['multidim_data']
        n_voxels, n_features = features.shape
        
        data_4d = np.zeros((*region.shape, n_features), dtype=features.dtype)
        data_4d[region] = features  # 直接布尔索引
        output_data['data'] = data_4d
        
        # 3. One-Hot标签转换：(102, n_voxels) → (384, 336, 256)
        seg_one_hot = data['seg_one_hot']
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
        
        return output_data
    
    def save_3d_data(self, data_3d: Dict[str, np.ndarray], output_path: Path):
        """保存3D数据为MAT文件"""
        with h5py.File(output_path, 'w') as f:
            for key, value in data_3d.items():
                # 为MATLAB兼容性调整格式
                if key == 'data' and value.ndim == 4:
                    # 4D数据：(384,336,256,351) → (351,384,336,256)
                    save_value = np.moveaxis(value, -1, 0)
                else:
                    save_value = value
                
                # 使用Fortran连续性优化MATLAB读取
                save_value = np.asfortranarray(save_value)
                
                # 根据数据大小选择压缩级别
                if key == 'data':
                    f.create_dataset(key, data=save_value, compression='gzip', compression_opts=1)
                else:
                    f.create_dataset(key, data=save_value, compression='gzip', compression_opts=4)
    
    def quick_validation(self, original_data: Dict[str, np.ndarray], 
                        converted_3d: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """快速验证转换结果"""
        region = original_data['region'].astype(bool)
        results = {}
        
        # 验证特征数据
        original_features = original_data['multidim_data']
        recovered_features = converted_3d['data'][region]
        results['multidim_data'] = np.allclose(original_features, recovered_features)
        
        # 验证region_seg
        original_region_seg = original_data['region_seg']
        recovered_region_seg = converted_3d['region_seg_3d'][region]
        results['region_seg'] = np.array_equal(original_region_seg, recovered_region_seg)
        
        # 验证one-hot标签
        original_one_hot = original_data['seg_one_hot']
        original_labels = np.argmax(original_one_hot, axis=0)
        recovered_labels = converted_3d['region_labels'][region]
        results['seg_one_hot'] = np.array_equal(original_labels, recovered_labels)
        
        # 验证3D数据
        results['big_seg'] = np.array_equal(original_data['big_seg'], converted_3d['big_seg'])
        results['region'] = np.array_equal(original_data['region'], converted_3d['region_mask'])
        
        return results
    
    def process_single_subject(self, mat_path: Path, prob_idx: int) -> Dict[str, Any]:
        """处理单个被试"""
        start_time = time.time()
        subject_id = mat_path.stem
        
        result = {
            'prob_idx': prob_idx,
            'subject_id': subject_id,
            'filename': mat_path.name,
            'start_time': start_time
        }
        
        try:
            # 1. 加载原始数据
            original_data = self.load_mat_h5_correct(mat_path)
            
            # 2. 转换为3D
            converted_3d = self.convert_1d_to_3d(original_data, prob_idx)
            
            # 3. 快速验证
            validation_results = self.quick_validation(original_data, converted_3d)
            all_valid = all(validation_results.values())
            
            # 4. 保存文件
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
            
            if not all_valid:
                print(f"⚠️ {subject_id}: 验证未完全通过")
                for key, valid in validation_results.items():
                    if not valid:
                        print(f"    - {key}: 验证失败")
        
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            self.failed_subjects.append(prob_idx)
            print(f"❌ {subject_id}: 处理失败 - {str(e)}")
        
        return result
    
    def create_dataset_index(self) -> Dict[int, Dict[str, str]]:
        """创建数据集索引"""
        mat_files = sorted(list(self.data_dir.glob("*.mat")))
        
        if len(mat_files) == 0:
            raise ValueError(f"未找到MAT文件在: {self.data_dir}")
        
        print(f"找到 {len(mat_files)} 个MAT文件")
        
        index_mapping = {}
        for idx, filepath in enumerate(mat_files, start=1):
            index_mapping[idx] = {
                "prob_idx": idx,
                "filename": filepath.name,
                "filepath": str(filepath),
                "subject_id": filepath.stem
            }
        
        # 保存索引
        json_path = self.data_dir / "dataset_index_validated.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"索引已保存到: {json_path}")
        return index_mapping
    
    def process_all_subjects(self, start_idx: int = 1, end_idx: int = 38, 
                           skip_existing: bool = False) -> Tuple[List, List]:
        """批量处理所有被试"""
        print(f"\n{'='*80}")
        print(f"批量处理被试 {start_idx} 到 {end_idx}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}")
        
        # 创建索引
        index_mapping = self.create_dataset_index()
        
        total_start_time = time.time()
        successful_subjects = []
        
        # 处理每个被试
        for prob_idx in tqdm(range(start_idx, end_idx + 1), desc="处理进度"):
            if prob_idx not in index_mapping:
                print(f"⚠️ 跳过不存在的prob_idx: {prob_idx}")
                continue
            
            subject_info = index_mapping[prob_idx]
            mat_path = Path(subject_info['filepath'])
            
            # 检查是否跳过已存在的文件
            if skip_existing:
                output_filename = f"{subject_info['subject_id']}_3d_validated.mat"
                output_path = self.output_dir / output_filename
                if output_path.exists():
                    tqdm.write(f"跳过已存在: {subject_info['subject_id']}")
                    continue
            
            # 处理被试
            result = self.process_single_subject(mat_path, prob_idx)
            self.conversion_results.append(result)
            
            if result['status'] == 'success':
                successful_subjects.append(prob_idx)
                if result['validation_passed']:
                    status_icon = "✅"
                else:
                    status_icon = "⚠️"
                
                tqdm.write(f"{status_icon} {result['subject_id']}: "
                          f"{result['file_size_mb']:.1f}MB, "
                          f"{result['processing_time']:.1f}s")
        
        total_time = time.time() - total_start_time
        self._generate_batch_report(total_time)
        
        return successful_subjects, self.failed_subjects
    
    def _generate_batch_report(self, total_time: float):
        """生成批量处理报告"""
        print(f"\n{'='*80}")
        print("批量处理报告")
        print(f"{'='*80}")
        
        successful = [r for r in self.conversion_results if r['status'] == 'success']
        failed = [r for r in self.conversion_results if r['status'] == 'failed']
        fully_validated = [r for r in successful if r['validation_passed']]
        
        print(f"处理统计:")
        print(f"  总处理数: {len(self.conversion_results)}")
        print(f"  成功转换: {len(successful)}")
        print(f"  验证通过: {len(fully_validated)}")
        print(f"  处理失败: {len(failed)}")
        print(f"  总时间: {total_time/60:.1f}分钟")
        
        if successful:
            avg_time = np.mean([r['processing_time'] for r in successful])
            avg_size = np.mean([r['file_size_mb'] for r in successful])
            total_size = sum([r['file_size_mb'] for r in successful])
            print(f"  平均处理时间: {avg_time:.1f}秒")
            print(f"  平均文件大小: {avg_size:.1f}MB")
            print(f"  总输出大小: {total_size:.1f}MB")
        
        # 失败的被试
        if failed:
            print(f"\n处理失败的被试:")
            for r in failed:
                print(f"  - {r['prob_idx']}: {r['subject_id']} - {r['error']}")
        
        # 验证问题
        validation_issues = [r for r in successful if not r['validation_passed']]
        if validation_issues:
            print(f"\n验证问题的被试:")
            for r in validation_issues:
                print(f"  - {r['prob_idx']}: {r['subject_id']}")
                for key, valid in r['validation_details'].items():
                    if not valid:
                        print(f"    ❌ {key}")
        
        # 保存详细报告
        self._save_detailed_reports()
    
    def _save_detailed_reports(self):
        """保存详细报告"""
        # JSON报告
        json_report_path = self.output_dir / "batch_conversion_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.conversion_results, f, indent=2, default=str)
        print(f"\nJSON报告已保存: {json_report_path}")
        
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
                    'error': r.get('error', '')
                })
            
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / "batch_conversion_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"CSV摘要已保存: {csv_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量转换MRI数据到3D格式（验证版）')
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
    
    args = parser.parse_args()
    
    # 创建批量转换器
    converter = ValidatedBatchConverter(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir)
    )
    
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
        print(f"\n🎉 所有被试处理成功！")
    else:
        print(f"\n⚠️ 有 {len(failed)} 个被试处理失败，请检查报告")

if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    
    if len(sys.argv) == 1:
        print("使用默认设置运行批量转换...")
        converter = ValidatedBatchConverter(
            data_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/1D"),
            output_dir=Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated")
        )
        
        response = input("\n选择模式:\n1. 测试（只处理第一个被试）\n2. 处理所有38个被试\n3. 处理前5个被试\n请输入 (1/2/3): ")
        
        if response == '1':
            successful, failed = converter.process_all_subjects(start_idx=1, end_idx=1)
        elif response == '2':
            successful, failed = converter.process_all_subjects(start_idx=1, end_idx=38)
        elif response == '3':
            successful, failed = converter.process_all_subjects(start_idx=1, end_idx=5)
        else:
            print("无效选择")
    else:
        main()