#!/usr/bin/env python3
"""
检查排除逻辑是否正确
在实际训练前验证哪些数据集会被排除
"""

import argparse
from pathlib import Path

def load_exclude_list(exclude_file: Path) -> set:
    """从文件中读取要排除的被试名列表"""
    exclude_set = set()
    if exclude_file and exclude_file.exists():
        with open(exclude_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    exclude_set.add(line)
    return exclude_set

def should_exclude_subject(subject_name: str, exclude_set: set) -> bool:
    """判断被试是否应该被排除"""
    for exclude_name in exclude_set:
        if exclude_name in subject_name:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description='检查排除逻辑')
    parser.add_argument('--data_dir_1d', type=str, required=True,
                       help='1D数据目录')
    parser.add_argument('--exclude_file', type=str, required=True,
                       help='排除列表文件')
    parser.add_argument('--exclude_single', type=str, default=None,
                       help='单次排除的被试名')

    args = parser.parse_args()

    # 读取排除列表
    exclude_file = Path(args.exclude_file)
    exclude_set = load_exclude_list(exclude_file)

    if args.exclude_single:
        exclude_set.add(args.exclude_single)

    print(f"\n排除列表: {exclude_set}")
    print(f"排除关键字数量: {len(exclude_set)}")
    print()

    # 扫描数据目录
    data_dir = Path(args.data_dir_1d)
    mat_files = sorted(data_dir.glob('*.mat'))

    print(f"找到 {len(mat_files)} 个MAT文件")
    print()

    # 分类文件
    excluded_files = []
    included_files = []

    for mat_file in mat_files:
        subject_name = mat_file.stem
        if should_exclude_subject(subject_name, exclude_set):
            excluded_files.append((mat_file.name, subject_name))
        else:
            included_files.append((mat_file.name, subject_name))

    # 输出结果
    print("=" * 80)
    print(f"✅ 将被包含的数据集 ({len(included_files)}个):")
    print("=" * 80)
    for filename, subject_name in included_files:
        print(f"  ✓ {filename}")
    print()

    print("=" * 80)
    print(f"❌ 将被排除的数据集 ({len(excluded_files)}个):")
    print("=" * 80)
    if excluded_files:
        for filename, subject_name in excluded_files:
            # 找出匹配的排除关键字
            matched_keys = [key for key in exclude_set if key in subject_name]
            print(f"  ✗ {filename} (匹配: {matched_keys})")
    else:
        print("  (无)")
    print()

    # 统计
    print("=" * 80)
    print("统计:")
    print("=" * 80)
    print(f"  原始数据集: {len(mat_files)} 个")
    print(f"  排除数据集: {len(excluded_files)} 个")
    print(f"  剩余数据集: {len(included_files)} 个")
    print(f"  排除比例: {len(excluded_files)/len(mat_files)*100:.1f}%")
    print()

    # 警告
    if len(included_files) < 10:
        print("⚠️  警告: 剩余数据集少于10个，可能不足以进行有效训练")

    if len(excluded_files) == 0:
        print("⚠️  警告: 没有数据集被排除，请检查排除列表是否正确")

if __name__ == '__main__':
    main()
