#!/usr/bin/env python3
"""
3D-1D数据转换工作流示例
展示完整的降采样、转换、预测、映射流程
"""

import numpy as np
from pathlib import Path
from data_3d_1d_mapper import Data3D1DMapper


def complete_workflow_example():
    """
    完整工作流程示例：
    1. 加载原始3D数据 (384, 336, 256)
    2. 降采样到目标尺寸 (例如 192, 168, 128)
    3. 转换为1D格式
    4. 模拟1D预测
    5. 映射回降采样后的3D空间
    """

    # 创建映射器
    mapper = Data3D1DMapper(log_level='INFO')

    print("\n" + "="*60)
    print("完整的3D-1D转换工作流程")
    print("="*60)

    # ============================================
    # 步骤1: 加载原始3D数据
    # ============================================
    print("\n步骤1: 加载原始3D数据")

    # 方法A: 从已转换的3D文件加载
    # data_3d = mapper.load_data('path_to_3d_validated.mat', from_matlab=True)

    # 方法B: 模拟数据（用于演示）
    original_shape = (384, 336, 256)
    n_features = 351

    # 创建模拟的ROI掩码（实际使用时从文件加载）
    region_mask = np.random.rand(*original_shape) > 0.8  # 约20%的体素在ROI内

    # 创建模拟的3D数据
    data_3d = {
        'data': np.random.randn(*original_shape, n_features).astype(np.float32),
        'region_mask': region_mask.astype(np.uint8),
        'region_labels': np.random.randint(0, 102, original_shape).astype(np.uint8),
        'big_seg': np.random.randint(0, 102, original_shape).astype(np.uint8)
    }

    print(f"原始数据形状:")
    print(f"  - data: {data_3d['data'].shape}")
    print(f"  - region_mask: {data_3d['region_mask'].shape}")
    print(f"  - ROI内体素数: {np.sum(region_mask)}")

    # ============================================
    # 步骤2: 3D到1D转换（带降采样）
    # ============================================
    print("\n步骤2: 3D到1D转换（带降采样）")

    # 定义降采样目标尺寸（减半）
    downsampled_shape = (192, 168, 128)

    # 执行转换
    data_1d = mapper.convert_3d_to_1d(
        data_3d,
        downsample_shape=downsampled_shape
    )

    print(f"1D数据形状:")
    print(f"  - multidim_data: {data_1d['multidim_data'].shape}")
    print(f"  - region_seg: {data_1d['region_seg'].shape}")
    print(f"  - seg_one_hot: {data_1d['seg_one_hot'].shape}")
    print(f"  - 降采样后region: {data_1d['region'].shape}")
    print(f"  - 降采样后ROI体素数: {data_1d['n_voxels']}")

    # ============================================
    # 步骤3: 保存1D数据（可选）
    # ============================================
    print("\n步骤3: 保存1D数据")
    output_1d_path = Path("example_1d_data.mat")
    mapper.save_data(data_1d, output_1d_path, matlab_compatible=True)
    print(f"1D数据已保存到: {output_1d_path}")

    # ============================================
    # 步骤4: 模拟1D模型预测
    # ============================================
    print("\n步骤4: 模拟1D模型预测")

    # 假设这是模型的预测输出
    n_voxels = data_1d['n_voxels']
    predictions_1d = np.random.randint(0, 102, n_voxels).astype(np.uint8)
    print(f"1D预测结果形状: {predictions_1d.shape}")

    # 也可以是概率预测
    predictions_proba_1d = np.random.rand(n_voxels, 102).astype(np.float32)
    predictions_proba_1d = predictions_proba_1d / predictions_proba_1d.sum(axis=1, keepdims=True)
    print(f"1D概率预测形状: {predictions_proba_1d.shape}")

    # ============================================
    # 步骤5: 将1D预测映射回3D空间
    # ============================================
    print("\n步骤5: 将1D预测映射回3D空间")

    # 映射到降采样后的3D空间
    predictions_3d_downsampled = mapper.map_1d_predictions_to_3d(
        predictions_1d,
        region_mask=data_1d['region'],
        target_shape=None  # 保持降采样尺寸
    )
    print(f"降采样3D预测形状: {predictions_3d_downsampled.shape}")

    # 如果需要，可以上采样回原始尺寸
    predictions_3d_original = mapper.map_1d_predictions_to_3d(
        predictions_1d,
        region_mask=data_1d['region'],
        target_shape=original_shape  # 上采样到原始尺寸
    )
    print(f"原始尺寸3D预测形状: {predictions_3d_original.shape}")

    # ============================================
    # 步骤6: 验证往返转换的一致性
    # ============================================
    print("\n步骤6: 验证往返转换的一致性")

    # 将1D数据转回3D
    data_3d_recovered = mapper.convert_1d_to_3d(data_1d, prob_idx=1)

    # 再次转换为1D
    data_1d_recovered = mapper.convert_3d_to_1d(data_3d_recovered)

    # 验证
    is_valid = mapper.validate_conversion(data_1d, data_1d_recovered)
    print(f"数据往返转换验证: {'✅ 通过' if is_valid else '❌ 失败'}")

    # ============================================
    # 步骤7: 清理临时文件
    # ============================================
    print("\n步骤7: 清理临时文件")
    if output_1d_path.exists():
        output_1d_path.unlink()
        print(f"已删除临时文件: {output_1d_path}")

    print("\n" + "="*60)
    print("工作流程完成！")
    print("="*60)


def batch_processing_example():
    """
    批量处理多个被试的示例
    """
    print("\n" + "="*60)
    print("批量处理示例")
    print("="*60)

    mapper = Data3D1DMapper(log_level='INFO')

    # 假设有多个3D文件
    data_dir = Path("/path/to/3d/data")
    output_dir = Path("/path/to/1d/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 降采样配置
    target_shape = (192, 168, 128)  # 降采样到一半

    # 获取所有3D文件（示例）
    # mat_files = list(data_dir.glob("*_3d_validated.mat"))

    # 这里用模拟数据演示
    for prob_idx in range(1, 4):  # 处理3个被试
        print(f"\n处理被试 {prob_idx}...")

        # 模拟加载3D数据
        region_mask = np.random.rand(384, 336, 256) > 0.8
        data_3d = {
            'data': np.random.randn(384, 336, 256, 351).astype(np.float32),
            'region_mask': region_mask.astype(np.uint8),
            'region_labels': np.random.randint(0, 102, (384, 336, 256)).astype(np.uint8),
        }

        # 转换为1D（带降采样）
        data_1d = mapper.convert_3d_to_1d(data_3d, downsample_shape=target_shape)

        # 保存1D数据
        output_path = output_dir / f"subject_{prob_idx:03d}_1d.mat"
        mapper.save_data(data_1d, output_path)
        print(f"  已保存: {output_path}")

        # 清理（实际使用时可选）
        output_path.unlink(missing_ok=True)

    print("\n批量处理完成！")


def load_and_convert_real_data():
    """
    加载真实数据并转换的示例
    """
    print("\n" + "="*60)
    print("真实数据转换示例")
    print("="*60)

    mapper = Data3D1DMapper(log_level='INFO')

    # 路径配置
    input_3d_path = Path("/path/to/your/3d_validated.mat")
    output_1d_path = Path("/path/to/output/1d_data.mat")

    # 检查文件是否存在
    if not input_3d_path.exists():
        print(f"文件不存在: {input_3d_path}")
        print("请更新文件路径后运行")
        return

    # 1. 加载3D数据
    print("\n1. 加载3D数据...")
    data_3d = mapper.load_data(input_3d_path, from_matlab=True)

    # 2. 转换为1D（带降采样）
    print("\n2. 转换为1D（降采样到192x168x128）...")
    data_1d = mapper.convert_3d_to_1d(
        data_3d,
        downsample_shape=(192, 168, 128)
    )

    # 3. 保存1D数据
    print("\n3. 保存1D数据...")
    mapper.save_data(data_1d, output_1d_path, matlab_compatible=True)

    print(f"\n转换完成！")
    print(f"输出文件: {output_1d_path}")

    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  原始形状: {data_3d.get('region_mask', data_3d.get('region', np.array([]))).shape}")
    print(f"  降采样后: {data_1d['region'].shape}")
    print(f"  ROI体素数: {data_1d['n_voxels']}")
    print(f"  特征数: {data_1d['multidim_data'].shape[1] if 'multidim_data' in data_1d else 'N/A'}")


if __name__ == "__main__":
    print("3D-1D数据转换工作流示例")
    print("="*60)
    print("选择示例:")
    print("1. 完整工作流程（模拟数据）")
    print("2. 批量处理示例")
    print("3. 真实数据转换（需要更新路径）")
    print("="*60)

    choice = input("请选择 (1/2/3) [默认=1]: ").strip() or "1"

    if choice == "1":
        complete_workflow_example()
    elif choice == "2":
        batch_processing_example()
    elif choice == "3":
        load_and_convert_real_data()
    else:
        print("无效选择，运行默认示例")
        complete_workflow_example()