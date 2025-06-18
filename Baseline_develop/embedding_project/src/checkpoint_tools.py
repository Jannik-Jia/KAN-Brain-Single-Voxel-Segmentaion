#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存档点管理工具
提供存档点的查看、分析、清理和维护功能
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# 添加当前目录到Python路径以导入checkpoint_manager
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from checkpoint_manager import CheckpointManager


class CheckpointAnalyzer:
    """存档点分析器"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    def analyze_size_usage(self):
        """分析存档点空间使用"""
        print("📊 存档点空间使用分析")
        print("=" * 60)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            print("📂 未找到任何存档点")
            return
        
        total_size = sum(ckpt['file_size_mb'] for ckpt in checkpoints)
        
        print(f"存档点总数: {len(checkpoints)}")
        print(f"总空间使用: {total_size:.1f} MB")
        print(f"平均文件大小: {total_size/len(checkpoints):.1f} MB")
        print()
        
        # 按阶段分组统计
        phase_stats = {}
        for ckpt in checkpoints:
            phase = ckpt.get('phase_completed', -1)
            if phase not in phase_stats:
                phase_stats[phase] = {'count': 0, 'size': 0}
            phase_stats[phase]['count'] += 1
            phase_stats[phase]['size'] += ckpt['file_size_mb']
        
        print("按阶段分组:")
        for phase in sorted(phase_stats.keys()):
            stats = phase_stats[phase]
            phase_name = f"Phase {phase}" if phase >= 0 else "未完成阶段"
            print(f"  {phase_name}: {stats['count']} 个文件, {stats['size']:.1f} MB")
        
        print()
        
        # 显示最大的存档点
        largest_checkpoints = sorted(checkpoints, key=lambda x: x['file_size_mb'], reverse=True)[:5]
        print("最大的存档点:")
        for i, ckpt in enumerate(largest_checkpoints, 1):
            print(f"  {i}. {ckpt['name']} - {ckpt['file_size_mb']:.1f} MB")
    
    def analyze_timeline(self):
        """分析存档点时间线"""
        print("⏰ 存档点时间线分析")
        print("=" * 60)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            print("📂 未找到任何存档点")
            return
        
        # 按创建时间排序
        checkpoints.sort(key=lambda x: x['creation_time'])
        
        print("创建时间线:")
        for ckpt in checkpoints:
            creation_time = ckpt['creation_time']
            if isinstance(creation_time, str):
                try:
                    dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = creation_time
            else:
                time_str = str(creation_time)
            
            phase_str = f"Phase {ckpt['phase_completed']}" if ckpt['phase_completed'] >= 0 else "未完成"
            print(f"  {time_str} - {ckpt['name']} ({phase_str})")
        
        # 分析创建频率
        if len(checkpoints) > 1:
            first_time = datetime.fromisoformat(checkpoints[0]['creation_time'].replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(checkpoints[-1]['creation_time'].replace('Z', '+00:00'))
            total_duration = last_time - first_time
            
            print()
            print(f"首个存档点: {first_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"最新存档点: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"时间跨度: {total_duration}")
            print(f"平均创建间隔: {total_duration / (len(checkpoints) - 1)}")
    
    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str):
        """对比两个存档点"""
        print(f"🔍 对比存档点: {checkpoint1} vs {checkpoint2}")
        print("=" * 60)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        # 查找存档点
        ckpt1 = None
        ckpt2 = None
        
        for ckpt in checkpoints:
            if checkpoint1 in ckpt['filename'] or checkpoint1 in ckpt['name']:
                ckpt1 = ckpt
            if checkpoint2 in ckpt['filename'] or checkpoint2 in ckpt['name']:
                ckpt2 = ckpt
        
        if not ckpt1:
            print(f"❌ 未找到存档点: {checkpoint1}")
            return
        
        if not ckpt2:
            print(f"❌ 未找到存档点: {checkpoint2}")
            return
        
        print("基本信息对比:")
        print(f"  存档点1: {ckpt1['name']}")
        print(f"    创建时间: {ckpt1['creation_time']}")
        print(f"    阶段进度: Phase {ckpt1['phase_completed']}")
        print(f"    文件大小: {ckpt1['file_size_mb']:.1f} MB")
        print()
        print(f"  存档点2: {ckpt2['name']}")
        print(f"    创建时间: {ckpt2['creation_time']}")
        print(f"    阶段进度: Phase {ckpt2['phase_completed']}")
        print(f"    文件大小: {ckpt2['file_size_mb']:.1f} MB")
        print()
        
        # 进度对比
        progress_diff = ckpt2['progress_percentage'] - ckpt1['progress_percentage']
        size_diff = ckpt2['file_size_mb'] - ckpt1['file_size_mb']
        
        print("差异分析:")
        print(f"  进度差异: {progress_diff:+.1f}%")
        print(f"  大小差异: {size_diff:+.1f} MB")
        
        # 时间差异
        try:
            time1 = datetime.fromisoformat(ckpt1['creation_time'].replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(ckpt2['creation_time'].replace('Z', '+00:00'))
            time_diff = time2 - time1
            print(f"  时间差异: {time_diff}")
        except:
            print("  时间差异: 无法计算")
    
    def cleanup_old_checkpoints(self, days: int = 7, keep_count: int = 5, dry_run: bool = True):
        """清理旧存档点"""
        action = "模拟清理" if dry_run else "清理"
        print(f"🧹 {action}旧存档点 (保留{days}天内和最新{keep_count}个)")
        print("=" * 60)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            print("📂 未找到任何存档点")
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 按创建时间排序，最新的在前
        checkpoints.sort(key=lambda x: x['creation_time'], reverse=True)
        
        # 要保留的存档点
        keep_checkpoints = set()
        
        # 保留最新的N个
        for i, ckpt in enumerate(checkpoints[:keep_count]):
            keep_checkpoints.add(ckpt['filename'])
            if i == 0:
                print(f"✅ 保留最新: {ckpt['name']}")
            else:
                print(f"✅ 保留前{i+1}: {ckpt['name']}")
        
        # 保留指定天数内的
        recent_count = 0
        for ckpt in checkpoints:
            try:
                ckpt_time = datetime.fromisoformat(ckpt['creation_time'].replace('Z', '+00:00'))
                if ckpt_time > cutoff_date:
                    keep_checkpoints.add(ckpt['filename'])
                    if ckpt['filename'] not in [c['filename'] for c in checkpoints[:keep_count]]:
                        recent_count += 1
            except:
                # 如果时间解析失败，保守地保留
                keep_checkpoints.add(ckpt['filename'])
        
        if recent_count > 0:
            print(f"✅ 保留{days}天内: {recent_count} 个")
        
        # 确定要删除的存档点
        to_delete = [ckpt for ckpt in checkpoints if ckpt['filename'] not in keep_checkpoints]
        
        if not to_delete:
            print("📂 没有需要清理的存档点")
            return
        
        total_size_to_delete = sum(ckpt['file_size_mb'] for ckpt in to_delete)
        
        print()
        print(f"将要删除的存档点 ({len(to_delete)} 个，{total_size_to_delete:.1f} MB):")
        for ckpt in to_delete:
            print(f"  ❌ {ckpt['name']} - {ckpt['creation_time']} ({ckpt['file_size_mb']:.1f} MB)")
        
        if not dry_run:
            deleted_count = 0
            for ckpt in to_delete:
                checkpoint_path = self.checkpoint_dir / ckpt['filename']
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        deleted_count += 1
                        print(f"  ✅ 已删除: {ckpt['filename']}")
                except Exception as e:
                    print(f"  ❌ 删除失败: {ckpt['filename']} - {e}")
            
            print(f"\n🧹 清理完成，删除了 {deleted_count} 个存档点，释放 {total_size_to_delete:.1f} MB 空间")
        else:
            print(f"\n💡 这是模拟运行，使用 --execute 参数执行实际删除")
    
    def export_summary(self, output_file: str):
        """导出存档点摘要"""
        print(f"📄 导出存档点摘要到: {output_file}")
        print("=" * 60)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        summary = {
            'export_time': datetime.now().isoformat(),
            'total_checkpoints': len(checkpoints),
            'checkpoint_directory': str(self.checkpoint_dir),
            'checkpoints': []
        }
        
        total_size = 0
        for ckpt in checkpoints:
            total_size += ckpt['file_size_mb']
            
            checkpoint_summary = {
                'filename': ckpt['filename'],
                'name': ckpt['name'],
                'creation_time': ckpt['creation_time'],
                'phase_completed': ckpt['phase_completed'],
                'progress_percentage': ckpt['progress_percentage'],
                'file_size_mb': ckpt['file_size_mb'],
                'description': ckpt['description'],
                'memory_usage': ckpt['memory_usage'],
                'include_deep_networks': ckpt['include_deep_networks']
            }
            summary['checkpoints'].append(checkpoint_summary)
        
        summary['total_size_mb'] = total_size
        summary['average_size_mb'] = total_size / len(checkpoints) if checkpoints else 0
        
        # 写入文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 摘要已导出: {output_path}")
        print(f"📊 包含 {len(checkpoints)} 个存档点的详细信息")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='存档点管理工具')
    
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='./brain_aware_analysis_output/checkpoints',
                       help='存档点目录路径')
    
    # 分析命令
    parser.add_argument('--analyze-size', action='store_true',
                       help='分析存档点空间使用')
    
    parser.add_argument('--analyze-timeline', action='store_true',
                       help='分析存档点时间线')
    
    parser.add_argument('--compare', nargs=2, metavar=('CHECKPOINT1', 'CHECKPOINT2'),
                       help='对比两个存档点')
    
    # 清理命令
    parser.add_argument('--cleanup', action='store_true',
                       help='清理旧存档点')
    
    parser.add_argument('--cleanup-days', type=int, default=7,
                       help='保留多少天内的存档点 (默认: 7)')
    
    parser.add_argument('--keep-count', type=int, default=5,
                       help='最少保留多少个最新存档点 (默认: 5)')
    
    parser.add_argument('--execute', action='store_true',
                       help='执行实际删除操作 (默认为模拟运行)')
    
    # 导出命令
    parser.add_argument('--export-summary', type=str, metavar='OUTPUT_FILE',
                       help='导出存档点摘要到指定文件')
    
    # 列表命令
    parser.add_argument('--list', action='store_true',
                       help='列出所有存档点')
    
    args = parser.parse_args()
    
    # 检查存档点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ 存档点目录不存在: {checkpoint_dir}")
        print("💡 请先运行一次分析以创建存档点")
        return 1
    
    # 创建分析器
    analyzer = CheckpointAnalyzer(checkpoint_dir)
    
    # 执行相应的命令
    try:
        if args.list:
            checkpoints = analyzer.checkpoint_manager.list_checkpoints()
            if not checkpoints:
                print("📂 未找到任何存档点")
            else:
                print("📂 存档点列表:")
                print("=" * 80)
                for i, ckpt in enumerate(checkpoints):
                    print(f"[{i}] {ckpt['name']}")
                    print(f"    文件: {ckpt['filename']}")
                    print(f"    创建时间: {ckpt['creation_time']}")
                    print(f"    阶段: Phase {ckpt['phase_completed']} ({ckpt['progress_percentage']:.1f}%)")
                    print(f"    大小: {ckpt['file_size_mb']:.1f} MB")
                    print(f"    描述: {ckpt['description']}")
                    print()
        
        elif args.analyze_size:
            analyzer.analyze_size_usage()
        
        elif args.analyze_timeline:
            analyzer.analyze_timeline()
        
        elif args.compare:
            analyzer.compare_checkpoints(args.compare[0], args.compare[1])
        
        elif args.cleanup:
            analyzer.cleanup_old_checkpoints(
                days=args.cleanup_days,
                keep_count=args.keep_count,
                dry_run=not args.execute
            )
        
        elif args.export_summary:
            analyzer.export_summary(args.export_summary)
        
        else:
            # 默认：显示概览
            print("🔄 存档点管理工具")
            print("=" * 60)
            
            checkpoints = analyzer.checkpoint_manager.list_checkpoints()
            total_size = sum(ckpt['file_size_mb'] for ckpt in checkpoints) if checkpoints else 0
            
            print(f"存档点目录: {checkpoint_dir}")
            print(f"存档点总数: {len(checkpoints)}")
            print(f"总空间使用: {total_size:.1f} MB")
            
            if checkpoints:
                latest = checkpoints[0]
                print(f"最新存档点: {latest['name']} ({latest['creation_time']})")
            
            print()
            print("可用命令:")
            print("  --list                  列出所有存档点")
            print("  --analyze-size          分析空间使用")
            print("  --analyze-timeline      分析时间线")
            print("  --cleanup               清理旧存档点")
            print("  --export-summary FILE   导出摘要")
            print("  --compare CKPT1 CKPT2   对比存档点")
            print()
            print("使用 --help 查看完整选项")
        
        return 0
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)