# src/main.py

import os
import argparse
import time
import numpy as np
from datetime import datetime

from src.bilingual_logger import BilingualLogger

from src.utils import create_feature_selection_param_grid, sample_parameter_combinations
from src.gpu_utils import init_gpu  # 导入GPU初始化函数


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='伪逆线性模型特征选择实验')
    
    # 数据路径
    parser.add_argument('--train_dir', type=str, required=True, help='训练数据目录路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试数据目录路径')
    parser.add_argument('--val_dir', type=str, required=True, help='验证数据目录路径')
    
    # 实验设置
    parser.add_argument('--exp_dir', type=str, default='pseudoinverse_with_fs_experiments', 
                        help='实验结果保存目录')
    parser.add_argument('--fs_mode', type=str, default='global', choices=['global', 'per_experiment'],
                        help='特征选择模式: global(全局选择) 或 per_experiment(每次实验单独选择)')
    parser.add_argument('--max_experiments', type=int, default=100, 
                        help='最大实验数量')
    parser.add_argument('--run_baseline', action='store_true', 
                        help='只运行基准实验')
    
    # GPU 设置
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速计算')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8, 
                        help='GPU内存使用比例上限 (0.0-1.0)')

    # 日志设置
    parser.add_argument('--max_samples_per_label', type=int, default=None, 
                        help='每个标签最多使用的样本数量')
    parser.add_argument('--apply_pca', type=lambda x: (str(x).lower() == 'true'), 
                        default=False, help='是否应用PCA降维')
    parser.add_argument('--pca_components', type=int, default=50, 
                        help='PCA组件数量')
    
    
    return parser.parse_args()

def run_baseline_experiments(experiment_manager, logger):
    """运行基准测试实验"""
    logger.info("运行基准测试实验（包含特征选择）", "Running baseline test experiment with feature selection")

    # 基准参数 - 不使用特征选择
    baseline_params_no_fs = {
        'apply_pca': False,
        'n_components': 50,
        'normalization': 'standard',
        'class_balance': True,
        'target_samples': 5000,  # 每类最多500个样本
        'regularization': None,
        'alpha': 0.01,
        'feature_selection': None,
        'selection_mode': 'threshold',
        'selection_threshold': 0.01,
        'max_features': 100,
        'l1_ratio': 1.0,
        'scaling_before_selection': True,
        'selection_metric': 'coefficient'
    }

    # 基准参数 - 使用LASSO特征选择
    baseline_params_lasso = {
        'apply_pca': False,
        'n_components': 50,
        'normalization': 'standard',
        'class_balance': False,
        'regularization': None,
        'alpha': 0.01,
        'feature_selection': 'lasso',
        'selection_mode': 'threshold',
        'selection_threshold': 0.01,
        'max_features': 100,
        'l1_ratio': 1.0,
        'scaling_before_selection': True,
        'selection_metric': 'coefficient'
    }

    try:
        # 运行不使用特征选择的基准实验
        logger.info("运行不使用特征选择的基准实验", "Running baseline experiment without feature selection")
        experiment_id_no_fs, results_no_fs = experiment_manager.run_experiment(baseline_params_no_fs)
        
        logger.info(f"基准实验（无特征选择）{experiment_id_no_fs} 完成", 
                   f"Baseline experiment (no feature selection) {experiment_id_no_fs} completed")
        
        # 打印主要评估指标
        logger.info(
            f"基准实验（无特征选择）结果:\n"
            f"  训练集准确率: {results_no_fs['train']['accuracy']:.4f}\n"
            f"  测试集准确率: {results_no_fs['test']['accuracy']:.4f}\n"
            f"  验证集准确率: {results_no_fs['val']['accuracy']:.4f}",
            
            f"Baseline experiment (no feature selection) results:\n"
            f"  Training accuracy: {results_no_fs['train']['accuracy']:.4f}\n"
            f"  Test accuracy: {results_no_fs['test']['accuracy']:.4f}\n"
            f"  Validation accuracy: {results_no_fs['val']['accuracy']:.4f}"
        )
        
        # 运行使用LASSO特征选择的基准实验
        logger.info("运行使用LASSO特征选择的基准实验", "Running baseline experiment with LASSO feature selection")
        experiment_id_lasso, results_lasso = experiment_manager.run_experiment(baseline_params_lasso)
        
        logger.info(f"基准实验（LASSO特征选择）{experiment_id_lasso} 完成", 
                   f"Baseline experiment (LASSO feature selection) {experiment_id_lasso} completed")
        
        # 打印主要评估指标
        logger.info(
            f"基准实验（LASSO特征选择）结果:\n"
            f"  训练集准确率: {results_lasso['train']['accuracy']:.4f}\n"
            f"  测试集准确率: {results_lasso['test']['accuracy']:.4f}\n"
            f"  验证集准确率: {results_lasso['val']['accuracy']:.4f}\n"
            f"  原始特征维度: {results_lasso['original_dim']}\n"
            f"  选择后特征维度: {results_lasso['selected_dim']}\n"
            f"  特征选择比例: {results_lasso['selected_dim']/results_lasso['original_dim']:.2f}",
            
            f"Baseline experiment (LASSO feature selection) results:\n"
            f"  Training accuracy: {results_lasso['train']['accuracy']:.4f}\n"
            f"  Test accuracy: {results_lasso['test']['accuracy']:.4f}\n"
            f"  Validation accuracy: {results_lasso['val']['accuracy']:.4f}\n"
            f"  Original feature dimension: {results_lasso['original_dim']}\n"
            f"  Selected feature dimension: {results_lasso['selected_dim']}\n"
            f"  Feature selection ratio: {results_lasso['selected_dim']/results_lasso['original_dim']:.2f}"
        )
        
        # 对比两个基准实验结果
        logger.info(
            f"基准实验对比:\n"
            f"  无特征选择 vs LASSO特征选择:\n"
            f"  测试集准确率: {results_no_fs['test']['accuracy']:.4f} vs {results_lasso['test']['accuracy']:.4f}\n"
            f"  准确率差异: {results_lasso['test']['accuracy'] - results_no_fs['test']['accuracy']:.4f}",
            
            f"Baseline experiments comparison:\n"
            f"  No feature selection vs LASSO feature selection:\n"
            f"  Test accuracy: {results_no_fs['test']['accuracy']:.4f} vs {results_lasso['test']['accuracy']:.4f}\n"
            f"  Accuracy difference: {results_lasso['test']['accuracy'] - results_no_fs['test']['accuracy']:.4f}"
        )

        return True
        
    except Exception as e:
        logger.error(f"基准测试实验失败: {str(e)}", f"Baseline test experiments failed: {str(e)}")
        print(f"基准测试实验失败: {str(e)}")
        # 显示堆栈跟踪以便调试
        import traceback
        traceback.print_exc()
        
        return False

def run_full_experiments(experiment_manager, max_experiments, logger):
    """运行完整实验集"""
    # 创建参数网格
    param_grid = create_feature_selection_param_grid()
    
    logger.info(f"参数网格创建完成，包含 {len(param_grid)} 个参数", 
               f"Parameter grid created with {len(param_grid)} parameters")
    
    # 从参数网格中采样合理的组合
    param_combinations = sample_parameter_combinations(param_grid, max_samples=max_experiments)
    
    logger.info(f"采样了 {len(param_combinations)} 个参数组合", 
               f"Sampled {len(param_combinations)} parameter combinations")
               
    # 批量运行实验
    logger.info(f"开始批量实验，最大实验数: {len(param_combinations)}", 
               f"Starting batch experiments, max experiments: {len(param_combinations)}")

    try:
        # 打乱组合顺序，避免类似参数连续运行
        np.random.shuffle(param_combinations)
        
        completed_experiments = experiment_manager.run_batch_experiments(param_combinations)
        
        # 生成结果汇总报告
        logger.info("生成结果汇总报告", "Generating summary report")
        experiment_manager.generate_summary_report_with_fs(top_n=20)
        logger.info("汇总报告生成完成", "Summary report generation completed")
        
        return True
        
    except Exception as e:
        logger.error(f"批量实验过程中出错: {str(e)}", f"Error during batch experiments: {str(e)}")
        print(f"批量实验过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return False

def main():
    """主函数"""
    # 解析命令行参数
    args = setup_args()
    
    # 创建实验目录
    os.makedirs(args.exp_dir, exist_ok=True)
    
    # 创建日志记录器
    log_dir = os.path.join(args.exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = BilingualLogger(
        log_dir=log_dir, 
        log_name=f"experiment_with_fs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger.info("开始伪逆线性模型实验（包含特征选择）", 
               "Starting pseudo-inverse linear model experiments with feature selection")
    
    # 初始化GPU支持
    if args.use_gpu:
        logger.info(f"尝试初始化GPU加速，内存使用限制: {args.gpu_memory_fraction}", 
                   f"Trying to initialize GPU acceleration, memory limit: {args.gpu_memory_fraction}")
        
        gpu_available = init_gpu(use_gpu=args.use_gpu, memory_fraction=args.gpu_memory_fraction)
    
        from src.gpu_utils import USE_GPU
        logger.info(f"GPU全局状态检查: USE_GPU = {USE_GPU}", 
                f"Global GPU status check: USE_GPU = {USE_GPU}")

        # 确保所有导入都在GPU初始化之后
        from src.brain_voxel_dataloader import BrainVoxelDataLoader
        from src.experiment_manager import ExperimentManagerWithFeatureSelection


        
        if gpu_available:
            logger.info("GPU初始化成功，将使用GPU加速计算", "GPU initialization successful, using GPU acceleration")
        else:
            logger.warning("GPU初始化失败，将使用CPU进行计算", "GPU initialization failed, using CPU for computation")
    else:
        logger.info("不使用GPU加速，所有计算将在CPU上进行", "Not using GPU acceleration, all computation will be done on CPU")
        init_gpu(use_gpu=False)

    # 打印实验参数
    logger.info(f"实验参数：\n"
               f"  训练数据目录：{args.train_dir}\n"
               f"  测试数据目录：{args.test_dir}\n"
               f"  验证数据目录：{args.val_dir}\n"
               f"  实验结果目录：{args.exp_dir}\n"
               f"  特征选择模式：{args.fs_mode}\n"
               f"  最大实验数量：{args.max_experiments}\n"
               f"  使用GPU加速：{args.use_gpu}", 
               
               f"Experiment parameters:\n"
               f"  Train directory: {args.train_dir}\n"
               f"  Test directory: {args.test_dir}\n"
               f"  Validation directory: {args.val_dir}\n"
               f"  Experiment directory: {args.exp_dir}\n"
               f"  Feature selection mode: {args.fs_mode}\n"
               f"  Maximum number of experiments: {args.max_experiments}\n"
               f"  Use GPU acceleration: {args.use_gpu}")
               

    # 加载数据，设置每个标签的最大样本数
    logger.info(f"加载数据集，每个标签最多 {args.max_samples_per_label} 个样本", 
               f"Loading datasets, max {args.max_samples_per_label} samples per label")
    
    # 修改加载方法，传入限制参数
    data_loader.load_all_data(max_samples_per_label=args.max_samples_per_label)
    

    # 创建带特征选择的实验管理器
    experiment_manager = ExperimentManagerWithFeatureSelection(
        data_loader, 
        base_dir=args.exp_dir, 
        logger=logger,
        feature_selection_mode=args.fs_mode
    )

    logger.info(f"实验管理器初始化完成，特征选择模式: {args.fs_mode}", 
               f"Experiment manager initialized with feature selection mode: {args.fs_mode}")
    
    # 运行实验
    start_time = time.time()
    
    if args.run_baseline:
        # 只运行基准实验
        success = run_baseline_experiments(experiment_manager, logger)
    else:
        # 先运行基准实验
        baseline_success = run_baseline_experiments(experiment_manager, logger)
        
        # 然后运行完整实验集
        if baseline_success:
            success = run_full_experiments(experiment_manager, args.max_experiments, logger)
        else:
            logger.error("基准实验失败，跳过完整实验集", 
                         "Baseline experiments failed, skipping full experiment set")
            success = False
    
    # 计算总运行时间
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if success:
        logger.info(f"所有实验和分析已完成, 总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒", 
                   f"All experiments and analysis completed, total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    else:
        logger.error(f"实验过程中出现错误, 总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒", 
                    f"Errors occurred during experiments, total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    print(f"\n{'所有实验和分析已完成' if success else '实验过程中出现错误'}。请查看结果目录获取详细报告。")
    print(f"结果目录: {os.path.abspath(args.exp_dir)}")

if __name__ == "__main__":
    main()