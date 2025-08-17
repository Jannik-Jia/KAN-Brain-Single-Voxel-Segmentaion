#!/usr/bin/env python3
"""
批量处理所有38个被试的3D转换
包含完整的验证和错误处理
"""

import numpy as np
import json
from pathlib import Path
import time
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import pandas as pd
from create_3d_with_validation import MRIDataConverter3D


class BatchConverter:
    """批量转换管理器"""
    
    def __init__(self, data_dir: Path, output_dir: Path, json_path: Path = None):
        """
        初始化批量转换器
        
        Parameters:
        - data_dir: 包含原始MAT文件的目录
        - output_dir: 输出目录
        - json_path: dataset_index.json路径（可选）
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.converter = MRIDataConverter3D()
        
        # 创建或加载索引
        if json_path and json_path.exists():
            self.load_index(json_path)
        else:
            self.create_index()
        
        # 初始化结果跟踪
        self.results = []
        self.failed_subjects = []
        
    def create_index(self) -> Dict:
        """创建数据集索引"""
        mat_files = sorted(list(self.data_dir.glob("*.mat")))
        
        if len(mat_files) == 0:
            raise ValueError(f"未找到MAT文件在: {self.data_dir}")
        
        print(f"找到 {len(mat_files)} 个MAT文件")
        
        # 创建索引映射
        self.index_mapping = {}
        for idx, filepath in enumerate(mat_files, start=1):
            self.index_mapping[idx] = {
                "prob_idx": idx,
                "filename": filepath.name,
                "filepath": str(filepath),
                "subject_id": filepath.stem
            }
        
        # 保存JSON
        json_path = self.data_dir / "dataset_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"索引已保存到: {json_path}")
        return self.index_mapping
    
    def load_index(self, json_path: Path):
        """加载现有索引"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 处理不同的JSON格式
        if isinstance(data, dict):
            self.index_mapping = {int(k): v for k, v in data.items()}
        elif isinstance(data, list):
            self.index_mapping = {}
            for item in data:
                self.index_mapping[item['prob_idx']] = item
        
        print(f"加载了 {len(self.index_mapping)} 个被试的索引")
    
    def process_subject(self, prob_idx: int) -> Dict:
        """
        处理单个被试
        
        Returns:
        - result: 包含处理结果的字典
        """
        subject_info = self.index_mapping[prob_idx]
        mat_path = Path(subject_info['filepath'])
        
        result = {
            'prob_idx': prob_idx,
            'subject_id': subject_info['subject_id'],
            'filename': subject_info['filename'],
            'start_time': time.time()
        }
        
        try:
            # 执行转换
            output_path = self.converter.process_single_subject(
                mat_path=mat_path,
                prob_idx=prob_idx,
                output_dir=self.output_dir,
                validate_all=True
            )
            
            # 验证输出文件
            verification_passed = self.converter.verify_saved_file(output_path)
            
            # 收集验证结果
            validation_summary = {
                'all_passed': all(v.passed for v in self.converter.validation_results),
                'details': [
                    {
                        'key': v.key,
                        'passed': v.passed,
                        'match_rate': v.match_rate,
                        'message': v.message
                    }
                    for v in self.converter.validation_results
                ]
            }
            
            result.update({
                'status': 'success',
                'output_path': str(output_path),
                'output_size_mb': output_path.stat().st_size / (1024 * 1024),
                'verification_passed': verification_passed,
                'validation_summary': validation_summary,
                'processing_time': time.time() - result['start_time']
            })
            
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - result['start_time']
            })
            self.failed_subjects.append(prob_idx)
        
        return result
    
    def process_all(self, start_from: int = 1, end_at: int = 38, 
                   skip_existing: bool = False):
        """
        处理所有被试
        
        Parameters:
        - start_from: 开始的prob_idx
        - end_at: 结束的prob_idx
        - skip_existing: 是否跳过已存在的输出文件
        """
        print(f"\n{'='*60}")
        print(f"批量处理被试 {start_from} 到 {end_at}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
        
        total_start = time.time()
        
        # 创建进度条
        subject_range = range(start_from, end_at + 1)
        pbar = tqdm(subject_range, desc="处理进度")
        
        for prob_idx in pbar:
            if prob_idx not in self.index_mapping:
                print(f"\n⚠️ 跳过不存在的prob_idx: {prob_idx}")
                continue
            
            # 检查是否跳过
            if skip_existing:
                subject_info = self.index_mapping[prob_idx]
                output_filename = f"{Path(subject_info['filename']).stem}_3d_validated.mat"
                output_path = self.output_dir / output_filename
                if output_path.exists():
                    pbar.set_description(f"跳过已存在: {subject_info['subject_id']}")
                    continue
            
            # 更新进度条描述
            subject_info = self.index_mapping[prob_idx]
            pbar.set_description(f"处理: {subject_info['subject_id']}")
            
            # 处理被试
            result = self.process_subject(prob_idx)
            self.results.append(result)
            
            # 打印简要状态
            if result['status'] == 'success':
                validation_status = "✅" if result['validation_summary']['all_passed'] else "⚠️"
                print(f"\n{validation_status} 完成 {result['subject_id']}: "
                      f"{result['output_size_mb']:.1f}MB, "
                      f"{result['processing_time']:.1f}秒")
            else:
                print(f"\n❌ 失败 {result['subject_id']}: {result['error']}")
        
        total_time = time.time() - total_start
        
        # 生成报告
        self.generate_report(total_time)
    
    def generate_report(self, total_time: float):
        """生成处理报告"""
        print(f"\n{'='*60}")
        print("批量处理报告")
        print(f"{'='*60}\n")
        
        # 统计
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        fully_validated = [r for r in successful 
                          if r['validation_summary']['all_passed']]
        
        print(f"处理统计:")
        print(f"  总处理数: {len(self.results)}")
        print(f"  成功: {len(successful)}")
        print(f"  失败: {len(failed)}")
        print(f"  完全验证通过: {len(fully_validated)}")
        print(f"  总时间: {total_time/60:.1f}分钟")
        
        if successful:
            avg_time = np.mean([r['processing_time'] for r in successful])
            avg_size = np.mean([r['output_size_mb'] for r in successful])
            print(f"  平均处理时间: {avg_time:.1f}秒")
            print(f"  平均文件大小: {avg_size:.1f}MB")
        
        # 失败的被试
        if failed:
            print(f"\n失败的被试:")
            for r in failed:
                print(f"  - {r['prob_idx']}: {r['subject_id']} - {r['error']}")
        
        # 验证问题
        validation_issues = []
        for r in successful:
            if not r['validation_summary']['all_passed']:
                validation_issues.append(r)
        
        if validation_issues:
            print(f"\n验证问题的被试:")
            for r in validation_issues:
                print(f"  - {r['prob_idx']}: {r['subject_id']}")
                for detail in r['validation_summary']['details']:
                    if not detail['passed']:
                        print(f"    ❌ {detail['key']}: {detail['message']}")
        
        # 保存详细报告
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """保存详细的CSV和JSON报告"""
        # JSON报告
        json_report_path = self.output_dir / "conversion_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nJSON报告已保存: {json_report_path}")
        
        # CSV摘要
        if self.results:
            summary_data = []
            for r in self.results:
                summary_data.append({
                    'prob_idx': r['prob_idx'],
                    'subject_id': r['subject_id'],
                    'status': r['status'],
                    'validation_passed': r.get('validation_summary', {}).get('all_passed', False) if r['status'] == 'success' else False,
                    'file_size_mb': r.get('output_size_mb', 0),
                    'time_seconds': r['processing_time'],
                    'error': r.get('error', '')
                })
            
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / "conversion_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"CSV摘要已保存: {csv_path}")
    
    def resume_from_checkpoint(self):
        """从检查点恢复处理"""
        # 检查已完成的文件
        existing_files = list(self.output_dir.glob("*_3d_validated.mat"))
        completed_subjects = set()
        
        for f in existing_files:
            # 从文件名提取subject_id
            subject_id = f.stem.replace('_3d_validated', '')
            # 找到对应的prob_idx
            for prob_idx, info in self.index_mapping.items():
                if info['subject_id'] == subject_id:
                    completed_subjects.add(prob_idx)
                    break
        
        print(f"找到 {len(completed_subjects)} 个已完成的被试")
        
        # 找到下一个要处理的
        for prob_idx in range(1, 39):
            if prob_idx not in completed_subjects:
                print(f"从被试 {prob_idx} 恢复处理")
                return prob_idx
        
        print("所有被试已完成！")
        return None


def main():
    parser = argparse.ArgumentParser(description='批量转换MRI数据到3D格式')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/jannik/Documents/mri_mat_onehot/1D',
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='/home/jannik/Documents/mri_mat_onehot/3D_validated',
                       help='输出目录')
    parser.add_argument('--start', type=int, default=1,
                       help='开始的prob_idx')
    parser.add_argument('--end', type=int, default=38,
                       help='结束的prob_idx')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的文件')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复')
    parser.add_argument('--test-only', action='store_true',
                       help='只处理第一个被试用于测试')
    
    args = parser.parse_args()
    
    # 创建批量转换器
    converter = BatchConverter(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir)
    )
    
    if args.test_only:
        # 测试模式：只处理第一个被试
        print("测试模式：只处理第一个被试")
        converter.process_all(start_from=1, end_at=1)
    elif args.resume:
        # 恢复模式
        next_idx = converter.resume_from_checkpoint()
        if next_idx:
            converter.process_all(
                start_from=next_idx,
                end_at=args.end,
                skip_existing=True
            )
    else:
        # 正常处理
        converter.process_all(
            start_from=args.start,
            end_at=args.end,
            skip_existing=args.skip_existing
        )


if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    
    if len(sys.argv) == 1:
        # 没有命令行参数，使用默认设置
        print("使用默认设置运行...")
        converter = BatchConverter(
            data_dir=Path("/home/jannik/Documents/mri_mat_onehot/1D"),
            output_dir=Path("/home/jannik/Documents/mri_mat_onehot/3D_validated")
        )
        
        # 询问用户
        response = input("\n选择模式:\n1. 测试（只处理第一个被试）\n2. 处理所有38个被试\n3. 从检查点恢复\n请输入 (1/2/3): ")
        
        if response == '1':
            converter.process_all(start_from=1, end_at=1)
        elif response == '2':
            converter.process_all(start_from=1, end_at=38)
        elif response == '3':
            next_idx = converter.resume_from_checkpoint()
            if next_idx:
                converter.process_all(start_from=next_idx, end_at=38, skip_existing=True)
        else:
            print("无效选择")
    else:
        main()