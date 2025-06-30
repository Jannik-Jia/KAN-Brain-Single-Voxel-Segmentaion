#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主编排器
用于按顺序运行所有Phase或单独运行某个Phase
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.config import Config


class PhaseOrchestrator:
    """Phase编排器"""
    
    def __init__(self):
        self.phase_scripts = {
            0: 'phase0_data_preparation/main.py',
            1: 'phase1_subject_analysis/main.py',
            2: 'phase2_separability/main.py',
            3: 'phase3_embedding_design/main.py',
            4: 'phase4_decision_maker/main.py'
        }
        
        # 确保输出目录存在
        Config.ensure_directories()
    
    def run_phase(self, phase_num: int, args: list = None) -> int:
        """
        运行单个Phase
        
        Args:
            phase_num: Phase编号
            args: 额外的命令行参数
            
        Returns:
            返回码
        """
        if phase_num not in self.phase_scripts:
            print(f"错误: 不存在Phase {phase_num}")
            return 1
        
        script_path = Path(self.phase_scripts[phase_num])
        if not script_path.exists():
            print(f"错误: 找不到脚本 {script_path}")
            return 1
        
        # 构建命令
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        print(f"\n{'='*80}")
        print(f"运行 Phase {phase_num}: {script_path.name}")
        print(f"{'='*80}\n")
        
        # 运行Phase
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n错误: Phase {phase_num} 执行失败")
            return result.returncode
        
        print(f"\n✅ Phase {phase_num} 执行成功")
        return 0
    
    def run_all_phases(self, start_from: int = 0, args_dict: dict = None) -> int:
        """
        按顺序运行所有Phase
        
        Args:
            start_from: 从哪个Phase开始
            args_dict: 每个Phase的参数字典
            
        Returns:
            返回码
        """
        print(f"开始运行分析流程 (从Phase {start_from}开始)")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for phase_num in range(start_from, 5):
            args = args_dict.get(phase_num, []) if args_dict else []
            ret_code = self.run_phase(phase_num, args)
            
            if ret_code != 0:
                print(f"\n分析流程在Phase {phase_num}失败")
                return ret_code
        
        print(f"\n{'='*80}")
        print(f"🎉 所有Phase执行成功!")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        return 0
    
    def check_phase_output(self, phase_num: int) -> bool:
        """检查Phase输出是否存在"""
        output_dirs = {
            0: Config.PHASE0_OUTPUT_DIR,
            1: Config.PHASE1_OUTPUT_DIR,
            2: Config.PHASE2_OUTPUT_DIR,
            3: Config.PHASE3_OUTPUT_DIR
        }
        
        if phase_num not in output_dirs:
            return False
        
        output_dir = output_dirs[phase_num]
        result_file = output_dir / f'phase{phase_num}_results.json'
        
        return result_file.exists()
    
    def list_phase_status(self):
        """列出所有Phase的状态"""
        print("\nPhase执行状态:")
        print("-" * 50)
        
        for phase_num in range(5):
            if phase_num < 4:  # Phase 0-3有输出
                status = "✅ 已完成" if self.check_phase_output(phase_num) else "❌ 未完成"
            else:  # Phase 4是最终决策
                status = "🎯 最终决策"
            
            print(f"Phase {phase_num}: {status}")
        
        print("-" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脑区感知Subject Embedding分析系统编排器')
    
    # 运行模式
    parser.add_argument('--run', type=str, choices=['all', 'phase'], default='all',
                       help='运行模式: all-运行所有phase, phase-运行单个phase')
    
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3, 4],
                       help='要运行的Phase编号')
    
    parser.add_argument('--start-from', type=int, default=0,
                       help='从哪个Phase开始运行（仅在run=all时有效）')
    
    parser.add_argument('--list-status', action='store_true',
                       help='列出所有Phase的执行状态')
    
    # Phase 0参数
    parser.add_argument('--data-path', type=str,
                       help='数据文件路径（Phase 0）')
    
    parser.add_argument('--train-start', type=int,
                       help='训练集起始受试者ID（Phase 0）')
    
    parser.add_argument('--train-end', type=int,
                       help='训练集结束受试者ID（Phase 0）')
    
    # 通用参数
    parser.add_argument('--skip-visualization', action='store_true',
                       help='跳过可视化生成')
    
    args = parser.parse_args()
    
    orchestrator = PhaseOrchestrator()
    
    # 列出状态
    if args.list_status:
        orchestrator.list_phase_status()
        return 0
    
    # 准备Phase参数
    phase_args = []
    
    # Phase 0特定参数
    if args.data_path:
        phase_args.extend(['--data_path', args.data_path])
    if args.train_start:
        phase_args.extend(['--train_start', str(args.train_start)])
    if args.train_end:
        phase_args.extend(['--train_end', str(args.train_end)])
    
    # 通用参数
    if args.skip_visualization:
        phase_args.append('--skip_visualization')
    
    # 运行模式
    if args.run == 'phase':
        if args.phase is None:
            print("错误: 运行单个phase时必须指定--phase参数")
            return 1
        
        return orchestrator.run_phase(args.phase, phase_args)
    
    else:  # run all
        # 构建每个Phase的参数
        args_dict = {}
        if phase_args:
            # Phase 0使用特定参数
            args_dict[0] = phase_args
            # 其他Phase使用通用参数
            for i in range(1, 5):
                if args.skip_visualization:
                    args_dict[i] = ['--skip_visualization']
        
        return orchestrator.run_all_phases(args.start_from, args_dict)


if __name__ == "__main__":
    sys.exit(main())