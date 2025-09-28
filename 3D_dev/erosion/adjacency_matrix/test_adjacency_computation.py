#!/usr/bin/env python3
"""
邻接矩阵计算测试脚本

测试邻接矩阵计算和分析功能，验证系统正确性。
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from adjacency_analysis_utils import AdjacencyMatrixReader, ConfusionAdjacencyAnalyzer


def test_single_adjacency_matrix(adjacency_file: Path):
    """测试单个邻接矩阵文件"""
    print(f"测试文件: {adjacency_file}")

    # 创建读取器
    reader = AdjacencyMatrixReader(adjacency_file)

    # 测试元数据读取
    metadata = reader.metadata
    print(f"元数据: {metadata}")

    # 测试完整矩阵读取
    adjacency_matrix = reader.load_full_adjacency_matrix()
    print(f"邻接矩阵形状: {adjacency_matrix.shape}")
    print(f"邻接对数量: {np.sum(adjacency_matrix) // 2}")

    # 测试子矩阵读取
    submatrix = reader.load_adjacency_submatrix([0, 1, 2], [0, 1, 2])
    print(f"子矩阵(前3×3):\n{submatrix}")

    # 测试稀疏格式读取
    row_indices, col_indices, adj_values, contact_values = reader.load_sparse_adjacency()
    print(f"稀疏格式: {len(row_indices)}个邻接对")
    print(f"最大接触体素数: {np.max(contact_values)}")

    # 测试邻接对获取
    adjacency_pairs = reader.get_adjacency_pairs()
    print(f"邻接对示例: {adjacency_pairs[:5]}")

    # 测试特定区域的邻居
    neighbors = reader.get_region_neighbors(0)
    print(f"区域0的邻居: {neighbors}")

    return adjacency_matrix


def test_confusion_adjacency_analysis(adjacency_matrix: np.ndarray):
    """测试混淆-邻接分析功能"""
    print("\n测试混淆-邻接分析...")

    # 创建模拟的混淆矩阵
    np.random.seed(42)
    confusion_matrix = np.random.rand(102, 102) * 0.1
    # 添加对角线为主要预测
    np.fill_diagonal(confusion_matrix, np.random.rand(102) * 0.8 + 0.2)

    # 创建分析器
    analyzer = ConfusionAdjacencyAnalyzer()

    # 测试Hadamard乘积
    hadamard_product = analyzer.compute_hadamard_product(confusion_matrix, adjacency_matrix)
    print(f"Hadamard乘积形状: {hadamard_product.shape}")
    print(f"Hadamard乘积统计: 最大值={np.max(hadamard_product):.4f}, 平均值={np.mean(hadamard_product):.4f}")

    # 测试rank相关性
    spearman_corr, p_value = analyzer.compute_rank_correlation(confusion_matrix, adjacency_matrix)
    print(f"Spearman相关性: r={spearman_corr:.4f}, p={p_value:.4f}")

    # 测试高混淆邻接对
    high_pairs = analyzer.find_high_confusion_adjacency_pairs(
        confusion_matrix, adjacency_matrix, threshold=0.05
    )
    print(f"高混淆邻接对数量: {len(high_pairs)}")
    if high_pairs:
        print(f"前3个高混淆对: {high_pairs[:3]}")

    # 测试完整分析
    analysis_results = analyzer.analyze_adjacency_confusion_relationship(
        confusion_matrix, adjacency_matrix
    )
    print(f"完整分析结果keys: {list(analysis_results.keys())}")

    return analysis_results


def visualize_adjacency_matrix(adjacency_matrix: np.ndarray, output_dir: Path):
    """可视化邻接矩阵"""
    plt.figure(figsize=(12, 10))

    # 创建热图
    sns.heatmap(adjacency_matrix[:50, :50],  # 只显示前50个区域
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False)

    plt.title('3D脑区邻接矩阵 (前50个区域)')
    plt.xlabel('脑区ID')
    plt.ylabel('脑区ID')

    # 保存图像
    output_file = output_dir / 'adjacency_matrix_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"邻接矩阵可视化已保存: {output_file}")

    # 统计图
    plt.figure(figsize=(10, 6))

    # 每个区域的邻接数量
    adjacency_counts = np.sum(adjacency_matrix, axis=1)
    plt.subplot(1, 2, 1)
    plt.hist(adjacency_counts, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('邻接区域数量')
    plt.ylabel('区域数量')
    plt.title('每个脑区的邻接分布')

    # 邻接矩阵密度
    plt.subplot(1, 2, 2)
    density = np.sum(adjacency_matrix) / (102 * 101)  # 排除对角线
    plt.bar(['邻接密度'], [density], color='lightcoral')
    plt.ylabel('密度')
    plt.title(f'邻接矩阵密度: {density:.3f}')

    plt.tight_layout()
    output_file = output_dir / 'adjacency_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"邻接统计图已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='测试邻接矩阵计算功能')
    parser.add_argument('--adjacency_file', type=str,
                      help='单个邻接矩阵文件路径（用于测试）')
    parser.add_argument('--adjacency_dir', type=str, default='./results',
                      help='邻接矩阵目录（用于批量测试）')
    parser.add_argument('--visualize', action='store_true',
                      help='生成可视化图像')

    args = parser.parse_args()

    output_dir = Path(args.adjacency_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("邻接矩阵计算测试")
    print("=" * 50)

    # 测试单个文件
    if args.adjacency_file:
        adjacency_file = Path(args.adjacency_file)
        if not adjacency_file.exists():
            print(f"错误: 文件不存在 {adjacency_file}")
            return 1

        # 测试邻接矩阵读取
        adjacency_matrix = test_single_adjacency_matrix(adjacency_file)

        # 测试分析功能
        test_confusion_adjacency_analysis(adjacency_matrix)

        # 可视化
        if args.visualize:
            visualize_adjacency_matrix(adjacency_matrix, output_dir)

    else:
        # 查找已计算的邻接矩阵文件
        adjacency_files = list(output_dir.glob("*_adjacency_*.h5"))

        if not adjacency_files:
            print(f"在{output_dir}中未找到邻接矩阵文件")
            print("请先运行compute_adjacency_matrices.py")
            return 1

        print(f"找到{len(adjacency_files)}个邻接矩阵文件")

        # 测试第一个文件
        test_file = adjacency_files[0]
        print(f"\n测试第一个文件: {test_file}")
        adjacency_matrix = test_single_adjacency_matrix(test_file)

        # 测试分析功能
        test_confusion_adjacency_analysis(adjacency_matrix)

        # 可视化
        if args.visualize:
            visualize_adjacency_matrix(adjacency_matrix, output_dir)

    print("\n✅ 所有测试完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())