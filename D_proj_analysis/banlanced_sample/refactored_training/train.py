#!/usr/bin/env python
# coding: utf-8

"""
通用训练脚本 - 支持多种模型
Universal training script - Support multiple models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import sys
import argparse
import time

# 添加父目录到Python路径，以便导入visualization_toolkit
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 导入模型
from models import get_model, list_available_models, get_model_description

# 导入工具函数
from utils import (
    load_balanced_dataset_with_spatial_info,
    create_label_mapping,
    STANDARD_LABELS,
    train_epoch,
    evaluate,
    predict_and_save_3d_softmax,
    plot_training_history,
    compute_class_weights
)

# 检查是否能导入可视化工具包
try:
    from visualization_toolkit.per_class_analyzer import (
        calculate_per_class_metrics_detailed,
        create_comprehensive_visualizations,
        save_detailed_results
    )
    VISUALIZATION_AVAILABLE = True
    print("✓ visualization_toolkit loaded successfully")
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"⚠️  Warning: visualization_toolkit not found ({e}). Per-class analysis will be skipped.")


def setup_logging(log_file='training.log'):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(config, logger):
    """
    主训练函数

    Parameters:
    -----------
    config : dict
        训练配置
    logger : logging.Logger
        日志记录器
    """
    logger.info("=" * 80)
    logger.info(f"训练开始 - 模型: {config['model_name']}")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    logger.info("\n📂 加载balanced数据...")
    all_data, scalers_info = load_balanced_dataset_with_spatial_info(
        config['root_dir'],
        include_background=config['include_background'],
        exclude_features=config.get('exclude_features')
    )

    # train/test分割
    test_data_info = all_data[0]  # 第一个做测试
    train_data_list = all_data[1:]  # 其他做训练

    logger.info(f"\n数据分割:")
    logger.info(f"  测试集: {test_data_info['subject_id']} ({test_data_info['n_voxels']:,} 体素)")
    logger.info(f"  训练集: {len(train_data_list)} 个受试者")

    # 合并训练数据
    train_features = np.vstack([data['features'] for data in train_data_list])
    train_labels = np.hstack([data['labels'] for data in train_data_list])

    test_features = test_data_info['features']
    test_labels = test_data_info['labels']

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # 创建模型
    actual_input_dim = train_features.shape[1]
    logger.info(f"\n🧠 创建模型: {config['model_name']}")

    model, model_config = get_model(
        config['model_name'],
        input_dim=actual_input_dim,
        num_classes=config['num_classes'],
        **config.get('model_params', {})
    )
    model = model.to(device)

    logger.info(f"模型描述: {get_model_description(config['model_name'])}")
    logger.info(f"输入维度: {actual_input_dim}")
    logger.info(f"输出类别数: {config['num_classes']}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])

    # 损失函数（检查模型是否需要类权重）
    if model_config.get('use_class_weights', False):
        logger.info("\n⚖️ 计算类权重以处理类别不平衡...")
        class_weights = compute_class_weights(train_labels, config['num_classes'])
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info("✅ 使用加权损失函数 (Weighted CrossEntropyLoss)")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("使用标准损失函数（无类权重）")

    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': []
    }

    best_test_f1 = 0
    best_epoch = 0
    best_model_state = None

    logger.info(f"\n🚀 开始训练 ({config['num_epochs']} epochs)...")
    start_time = time.time()

    # 训练循环
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # 评估
        test_metrics = evaluate(model, test_features, test_labels, criterion, device,
                               batch_size=config.get('test_batch_size', 8192))

        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['macro_f1'])

        # 保存最佳模型
        if test_metrics['macro_f1'] > best_test_f1:
            best_test_f1 = test_metrics['macro_f1']
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        # 打印进度
        logger.info(f"\nEpoch [{epoch+1}/{config['num_epochs']}] "
                   f"Time: {time.time()-epoch_start:.2f}s")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['macro_f1']:.4f}")
        logger.info(f"  Test  - Loss: {test_metrics['loss']:.4f}, "
                   f"Acc: {test_metrics['accuracy']:.4f}, "
                   f"F1: {test_metrics['macro_f1']:.4f}")

        torch.cuda.empty_cache()

    training_time = time.time() - start_time
    logger.info(f"\n✅ 训练完成! 总耗时: {training_time:.2f}秒")
    logger.info(f"最佳测试F1: {best_test_f1:.4f} (Epoch {best_epoch+1})")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存结果
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)

    # 预测并保存3D softmax
    volume_3d, softmax_path = predict_and_save_3d_softmax(
        model, test_data_info, device, output_dir,
        include_background=config['include_background'],
        batch_size=config.get('test_batch_size', 8192),
        class_labels=STANDARD_LABELS
    )

    # 保存模型和其他结果
    bg_str = 'incl' if config['include_background'] else 'excl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_str = config['model_name']

    model_path = output_dir / f"{model_name_str}_bg_{bg_str}_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'config': config,
        'history': history,
        'best_epoch': best_epoch,
        'scalers_info': scalers_info,
        'test_subject_id': test_data_info['subject_id'],
        'softmax_path': str(softmax_path)
    }, model_path)

    # 绘制训练历史
    plot_path = output_dir / f"training_history_{model_name_str}_bg_{bg_str}_{timestamp}.png"
    plot_training_history(history, plot_path)

    logger.info(f"💾 模型已保存: {model_path}")

    # Per-Class性能分析
    if VISUALIZATION_AVAILABLE:
        logger.info("="*80)
        logger.info("🎯 开始Per-Class性能量化分析...")
        logger.info("="*80)

        try:
            model.eval()

            # 准备测试数据
            X_test = test_data_info['features']
            y_test = test_data_info['labels']

            logger.info(f"📊 对 {X_test.shape[0]:,} 个测试样本进行预测...")

            # 批量预测
            batch_size = min(8192, len(X_test))
            all_predictions = []

            with torch.no_grad():
                for i in tqdm(range(0, len(X_test), batch_size), desc="预测进度"):
                    batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                    batch_pred = model(batch_X)
                    batch_pred_labels = torch.argmax(batch_pred, dim=1).cpu().numpy()
                    all_predictions.extend(batch_pred_labels)

                    del batch_X, batch_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            predictions_array = np.array(all_predictions)

            logger.info(f"✅ 预测完成!")

            # 计算per-class指标
            per_class_analysis_dir = output_dir / f"per_class_analysis_{model_name_str}_bg_{bg_str}_{timestamp}"
            per_class_analysis_dir.mkdir(exist_ok=True)

            volume_info = {
                'subject_id': test_data_info.get('subject_id', 'unknown'),
                'bg_mode': 'included' if config['include_background'] else 'excluded',
                'timestamp': timestamp,
                'total_voxels': X_test.shape[0],
                'n_classes': len(np.unique(y_test)),
                'model_name': config['model_name']
            }

            forward_mapping, reverse_mapping = create_label_mapping()

            metrics_dict = calculate_per_class_metrics_detailed(
                y_test, predictions_array,
                volume_info=volume_info,
                reverse_mapping=reverse_mapping
            )

            # 生成可视化
            create_comprehensive_visualizations(metrics_dict, per_class_analysis_dir)

            # 保存详细结果
            save_detailed_results(metrics_dict, per_class_analysis_dir)

            logger.info(f"\n💾 Per-class分析结果已保存到: {per_class_analysis_dir}")

        except Exception as e:
            logger.error(f"❌ Per-class分析失败: {str(e)}")
            logger.error("训练结果已保存，但per-class分析未完成")
    else:
        logger.info("⚠️ 跳过per-class分析 (visualization_toolkit未找到)")

    return {
        'model': model,
        'history': history,
        'config': config,
        'softmax_path': softmax_path
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通用MRI体素分割训练脚本')

    # 模型选择
    parser.add_argument('--model', type=str, default='reg_model',
                       choices=list_available_models(),
                       help='选择模型 (default: reg_model)')

    # 数据参数
    parser.add_argument('--root_dir', type=str,
                       default='/home/jovyan/gpu_space/workspace_jiayi/new_datasets/NEW_DATASET_ANALYSIS',
                       help='数据集根目录')
    parser.add_argument('--include_background', action='store_true',
                       help='包含背景体素（默认排除）')
    parser.add_argument('--exclude_features', type=int, nargs='*', default=[14],
                       help='要排除的特征索引（默认[14]）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8192, help='批大小')
    parser.add_argument('--lr', type=float, default=0.00001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='权重衰减')

    # 模型特定参数 (可选)
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='隐藏层维度（针对RegModel和ResNetMLP）')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='Dropout率')
    parser.add_argument('--grid_size', type=int, default=None,
                       help='KAN网格分段数（针对KAN模型，默认8）')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    bg_str = 'incl' if args.include_background else 'excl'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{args.model}_bg_{bg_str}_{timestamp}.log"
    logger = setup_logging(log_file)

    # 打印可用模型列表
    logger.info("可用模型:")
    for model_name in list_available_models():
        desc = get_model_description(model_name)
        logger.info(f"  - {model_name}: {desc}")

    # 配置
    config = {
        'model_name': args.model,
        'root_dir': args.root_dir,
        'include_background': args.include_background,
        'exclude_features': args.exclude_features if args.exclude_features else None,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'test_batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_classes': 52,
        'model_params': {}
    }

    # 添加模型特定参数
    if args.hidden_dim is not None:
        config['model_params']['hidden_dim'] = args.hidden_dim
    if args.dropout_rate is not None:
        config['model_params']['dropout_rate'] = args.dropout_rate
    if args.grid_size is not None:
        config['model_params']['grid_size'] = args.grid_size

    # 训练模型
    result = train_model(config, logger)

    logger.info("\n" + "=" * 80)
    logger.info("训练完成!")
    logger.info("=" * 80)

    logger.info(f"\n💡 3D softmax已保存到: {result['softmax_path']}")
    logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
