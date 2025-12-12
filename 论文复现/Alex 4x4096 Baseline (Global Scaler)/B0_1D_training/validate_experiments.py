#!/usr/bin/env python3
"""
实验记录验证工具
用于检查实验 JSON 文件是否符合标准 schema
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# 导入 schema 验证函数
from experiment_schema import validate_experiment_json, get_schema_summary


def validate_single_experiment(json_path: Path, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    验证单个实验 JSON 文件

    Args:
        json_path: JSON 文件路径
        verbose: 是否打印详细信息

    Returns:
        (is_valid, errors): 是否有效和错误列表
    """
    if not json_path.exists():
        error_msg = f"文件不存在: {json_path}"
        if verbose:
            print(f"❌ {error_msg}")
        return False, [error_msg]

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            experiment = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"JSON 解析错误: {e}"
        if verbose:
            print(f"❌ {error_msg}")
        return False, [error_msg]
    except Exception as e:
        error_msg = f"读取文件错误: {e}"
        if verbose:
            print(f"❌ {error_msg}")
        return False, [error_msg]

    if verbose:
        print(f"\n正在验证: {json_path.name}")

    return validate_experiment_json(experiment, verbose=verbose)


def validate_batch_experiments(
    json_dir: Path,
    pattern: str = "*experiment*.json",
    verbose: bool = True
) -> Dict[str, Tuple[bool, List[str]]]:
    """
    批量验证目录下的所有实验 JSON 文件

    Args:
        json_dir: 包含 JSON 文件的目录
        pattern: 文件名匹配模式
        verbose: 是否打印详细信息

    Returns:
        结果字典: {文件名: (is_valid, errors)}
    """
    if not json_dir.exists() or not json_dir.is_dir():
        print(f"❌ 目录不存在或不是目录: {json_dir}")
        return {}

    json_files = list(json_dir.glob(pattern))

    if not json_files:
        print(f"❌ 在 {json_dir} 中没有找到匹配 '{pattern}' 的文件")
        return {}

    print(f"\n找到 {len(json_files)} 个实验 JSON 文件")
    print("=" * 80)

    results = {}
    valid_count = 0
    invalid_count = 0

    for json_file in sorted(json_files):
        is_valid, errors = validate_single_experiment(json_file, verbose=verbose)
        results[json_file.name] = (is_valid, errors)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

        if verbose and len(json_files) > 1:
            print("\n" + "=" * 80 + "\n")

    # 打印总结
    print("\n" + "=" * 80)
    print("批量验证总结")
    print("=" * 80)
    print(f"总计: {len(json_files)} 个文件")
    print(f"✅ 有效: {valid_count} 个")
    print(f"❌ 无效: {invalid_count} 个")

    if invalid_count > 0:
        print("\n无效的文件:")
        for filename, (is_valid, errors) in results.items():
            if not is_valid:
                print(f"  - {filename}: {len(errors)} 个问题")

    print("=" * 80)

    return results


def generate_validation_report(
    results: Dict[str, Tuple[bool, List[str]]],
    output_path: Path
) -> None:
    """
    生成验证报告并保存为文本文件

    Args:
        results: 验证结果字典
        output_path: 输出文件路径
    """
    lines = [
        "=" * 80,
        "实验记录验证报告",
        "=" * 80,
        "",
        f"总计: {len(results)} 个实验文件",
        "",
    ]

    valid_files = [f for f, (valid, _) in results.items() if valid]
    invalid_files = [f for f, (valid, _) in results.items() if not valid]

    lines.extend([
        f"✅ 有效文件: {len(valid_files)}",
        f"❌ 无效文件: {len(invalid_files)}",
        "",
        "=" * 80,
    ])

    if valid_files:
        lines.extend([
            "",
            "有效的实验文件:",
            "",
        ])
        for filename in sorted(valid_files):
            lines.append(f"  ✅ {filename}")

    if invalid_files:
        lines.extend([
            "",
            "=" * 80,
            "无效的实验文件及问题详情:",
            "=" * 80,
            "",
        ])
        for filename in sorted(invalid_files):
            _, errors = results[filename]
            lines.extend([
                f"❌ {filename}",
                f"   问题数量: {len(errors)}",
                "",
            ])
            for i, error in enumerate(errors, 1):
                lines.append(f"   {i}. {error}")
            lines.append("")

    lines.extend([
        "=" * 80,
        "报告生成完成",
        "=" * 80,
    ])

    report_text = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n📄 验证报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='验证实验记录 JSON 文件是否符合标准 schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 验证单个文件
  python validate_experiments.py --file experiment.json

  # 批量验证目录下的所有实验文件
  python validate_experiments.py --dir ./results

  # 批量验证并生成报告
  python validate_experiments.py --dir ./results --report validation_report.txt

  # 显示 schema 摘要
  python validate_experiments.py --show-schema
        """
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='验证单个实验 JSON 文件'
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='批量验证目录下的所有实验 JSON 文件'
    )

    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*experiment*.json',
        help='文件名匹配模式 (默认: *experiment*.json)'
    )

    parser.add_argument(
        '--report', '-r',
        type=str,
        help='生成验证报告并保存到指定文件'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，只显示摘要'
    )

    parser.add_argument(
        '--show-schema', '-s',
        action='store_true',
        help='显示 schema 摘要信息'
    )

    args = parser.parse_args()

    # 显示 schema 摘要
    if args.show_schema:
        print(get_schema_summary())
        return 0

    # 验证单个文件
    if args.file:
        json_path = Path(args.file)
        is_valid, errors = validate_single_experiment(json_path, verbose=not args.quiet)

        if is_valid:
            print(f"\n✅ 验证通过: {json_path.name}")
            return 0
        else:
            print(f"\n❌ 验证失败: {json_path.name} ({len(errors)} 个问题)")
            return 1

    # 批量验证目录
    elif args.dir:
        json_dir = Path(args.dir)
        results = validate_batch_experiments(
            json_dir,
            pattern=args.pattern,
            verbose=not args.quiet
        )

        # 生成报告
        if args.report:
            report_path = Path(args.report)
            generate_validation_report(results, report_path)

        # 返回状态码
        invalid_count = sum(1 for _, (valid, _) in results.items() if not valid)
        return 0 if invalid_count == 0 else 1

    else:
        parser.print_help()
        print("\n❌ 错误: 必须指定 --file 或 --dir 参数")
        return 1


if __name__ == "__main__":
    sys.exit(main())
