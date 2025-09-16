#!/usr/bin/env python3
"""
快速转换脚本 - MRI数据集Z-Score标准化

这个脚本会：
1. 使用预配置的数据集路径
2. 创建z-score标准化版本的数据集
3. 保存到同级目录下的 _zscore_normalized 文件夹
"""

from pathlib import Path
import sys

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from zscore_dataset_converter import ZScoreDatasetConverter

def convert_datasets():
    """转换数据集 - 使用预配置路径"""

    print("🧠 MRI数据集Z-Score标准化转换工具")
    print("=" * 60)

    # 专注于3D数据集的z-score标准化（用于7×7×1 patch训练）
    data_paths = {
        "3D": "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/"
    }

    print("📂 Z-Score标准化目标:")
    print(f"   3D数据: {data_paths['3D']}")
    print("   (1D数据集保持原始格式，不需要z-score标准化)")
    print()

    # 让用户选择要转换的数据集
    print("请选择要转换的数据集:")
    print("1. 3D数据 (用于patch训练) - 推荐")
    print("2. 使用自定义路径")

    choice = input("请输入选择 (1/2): ").strip()

    conversions = []

    if choice == "1":
        conversions.append(("3D", data_paths["3D"]))
    elif choice == "2":
        # 自定义路径
        custom_path = input("请输入数据集路径: ").strip()
        if Path(custom_path).exists():
            conversions.append(("Custom", custom_path))
        else:
            print(f"❌ 路径不存在: {custom_path}")
            return
    else:
        print("❌ 无效选择")
        return

    # 确认转换
    print(f"\n将要转换 {len(conversions)} 个数据集:")
    for data_type, path in conversions:
        print(f"  {data_type}: {path}")

    # 检查路径是否存在
    for data_type, path in conversions:
        if not Path(path).exists():
            print(f"⚠️ 警告: {data_type}数据路径不存在: {path}")
            print("这可能是因为路径配置为服务器路径，而您在本地运行")

            # 询问用户是否想要指定本地路径
            use_local = input(f"是否指定{data_type}数据的本地路径? (y/n): ").strip().lower()
            if use_local == 'y':
                local_path = input(f"请输入{data_type}数据的本地路径: ").strip()
                conversions = [(dt, local_path if dt == data_type else p) for dt, p in conversions]

    proceed = input("\n开始转换? (y/n): ").strip().lower()
    if proceed != 'y':
        print("已取消转换")
        return

    # 开始转换
    for data_type, input_path in conversions:
        print(f"\n{'=' * 80}")
        print(f"🚀 开始转换 {data_type} 数据集")
        print(f"{'=' * 80}")

        input_dir = Path(input_path)

        # 创建输出目录名
        output_dir = input_dir.parent / f"{input_dir.name}_zscore_normalized"

        print(f"📥 输入目录: {input_dir}")
        print(f"📤 输出目录: {output_dir}")

        # 检查输入目录是否存在
        if not input_dir.exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            continue

        # 检查是否有MAT文件
        mat_files = list(input_dir.glob("*.mat"))
        if len(mat_files) == 0:
            print(f"❌ 在 {input_dir} 中未找到.mat文件")
            continue

        print(f"📋 找到 {len(mat_files)} 个MAT文件")

        # 询问是否测试模式
        test_mode = input("是否只转换第一个文件进行测试? (y/n): ").strip().lower() == 'y'

        try:
            # 创建转换器 - 使用优化的7×7×1 patch chunk size
            converter = ZScoreDatasetConverter(
                input_dir=input_dir,
                output_dir=output_dir,
                log_level='INFO',
                chunk_size=(32, 32, 1, 351)  # 优化7×7×1 patch提取，减小解压开销
            )

            if test_mode:
                print("🔍 测试模式：只处理第一个文件")
                result = converter.process_single_subject(mat_files[0])

                if result['status'] == 'success':
                    print(f"✅ 测试成功!")
                    print(f"   输出文件: {result['output_path']}")
                    print(f"   文件大小: {result['file_size_mb']:.1f} MB")
                    print(f"   验证状态: {'✅ 通过' if result['validation_passed'] else '⚠️ 需要检查'}")

                    # 询问是否继续处理所有文件
                    continue_all = input("\n测试成功！是否继续转换所有文件? (y/n): ").strip().lower() == 'y'
                    if continue_all:
                        successful, failed = converter.process_all_subjects()
                        print(f"🎯 转换完成！成功: {len(successful)}, 失败: {len(failed)}")
                    else:
                        print("只完成了测试转换")
                else:
                    print(f"❌ 测试失败: {result.get('error', '未知错误')}")

            else:
                # 直接处理所有文件
                successful, failed = converter.process_all_subjects()

                if len(failed) == 0:
                    print(f"🎉 所有文件转换成功！")
                    print(f"   转换了 {len(successful)} 个文件")
                    print(f"   输出目录: {output_dir}")
                else:
                    print(f"⚠️ 转换完成，但有 {len(failed)} 个文件失败")
                    print(f"   成功: {len(successful)}")
                    print(f"   失败: {failed}")

        except Exception as e:
            print(f"❌ 转换{data_type}数据集时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎊 所有数据集转换完成！")

if __name__ == "__main__":
    print("🚀 快速数据集转换工具")
    print("=" * 40)
    print("这个工具会自动:")
    print("✅ 使用预配置的数据集路径")
    print("✅ 对每个病人的351维特征进行z-score标准化")
    print("✅ 创建7×7×1 patch友好的HDF5格式")
    print("✅ 保持所有体素的空间位置不变")
    print("✅ 保存到 _zscore_normalized 目录")
    print()

    try:
        convert_datasets()
    except KeyboardInterrupt:
        print("\n用户中断了转换过程")
    except Exception as e:
        print(f"\n转换失败: {e}")
        import traceback
        traceback.print_exc()