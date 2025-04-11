# src/main.py

import os
import argparse
import time
import numpy as np
from datetime import datetime

from src.bilingual_logger import BilingualLogger
from src.utils import create_feature_selection_param_grid, create_feature_selection_without_pca_param_grid
from src.utils import sample_parameter_combinations, sample_parameter_combinations_for_feature_selection
from src.gpu_utils import init_gpu, has_cuda_ml  # 导入GPU初始化函数


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='伪逆线性模型特征选择实验')
    
    # 数据路径
    parser.add_argument('--train_dir', type=str, required=True, help='训练数据目录路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试数据目录路径')
    parser.add_argument('--val_dir', type=str, required=True, help='验证数据目录路径')
    
    # 实验设置
    parser.add_argument('--exp_dir', type=str, default='pseudoinverse_experiments', 
                        help='实验结果保存目录')
    parser.add_argument('--fs_mode', type=str, default='global', choices=['global', 'per_experiment'],
                        help='特征选择模式: global(全局选择) 或 per_experiment(每次实验单独选择)')
    parser.add_argument('--max_experiments', type=int, default=100, 
                        help='最大实验数量')
    parser.add_argument('--run_baseline', action='store_true', 
                        help='只运行基准实验')
    parser.add_argument('--focus_on_fs', action='store_true',
                        help='专注于直接特征选择实验（不使用PCA）')
    
    # 新增参数：特征选择时是否完全跳过PCA
    parser.add_argument('--skip_pca', action='store_true', 
                        help='特征选择时完全跳过PCA，直接在原始特征上进行选择')
    
    # GPU 设置
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速计算')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8, 
                        help='GPU内存使用比例上限 (0.0-1.0)')

    # 数据处理参数
    parser.add_argument('--max_samples_per_label', type=int, default=None, 
                        help='每个标签最多使用的样本数量')
    parser.add_argument('--apply_pca', type=lambda x: (str(x).lower() == 'true'), 
                        default=True, help='是否应用PCA降维')
    parser.add_argument('--pca_components', type=int, default=50, 
                        help='PCA组件数量')
    parser.add_argument('--auto_pca_variance', type=float, default=0.95,
                        help='自动选择PCA组件数量以解释指定比例的方差 (0.0-1.0)')
    parser.add_argument('--scaling_before_pca', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='是否在PCA降维前进行标准化')
    
    return parser.parse_args()




def run_baseline_experiments(experiment_manager, logger, args):
    """运行基准测试实验"""
    
    # 添加这个检查：如果是特征选择模式并且需要跳过PCA，则使用无PCA的基准测试
    if args.focus_on_fs or args.skip_pca:
        logger.info("运行基准测试实验（专注于特征选择，不使用PCA）", 
                   "Running baseline test experiment focusing on feature selection without PCA")
        
        # 特征选择基准参数 - 不使用PCA
        baseline_params_fs = {
            'apply_pca': False,  # 不使用PCA
            'normalization': 'standard',
            'class_balance': False,
            'target_samples': args.max_samples_per_label,
            'regularization': None,
            'alpha': 0.01,
            'feature_selection': 'lasso',  # 使用LASSO特征选择
            'selection_mode': 'fixed',
            'selection_threshold': 0.01,
            'max_features': 100,
            'l1_ratio': 1.0,
            'scaling_before_selection': True,
            'selection_metric': 'coefficient'
        }

        # 带L2正则化的特征选择基准
        baseline_params_fs_l2 = baseline_params_fs.copy()
        baseline_params_fs_l2['regularization'] = 'l2'
        baseline_params_fs_l2['alpha'] = 0.1
        
        try:
            # 运行基本特征选择实验
            logger.info("运行基本特征选择实验（无PCA）", "Running basic feature selection experiment (without PCA)")
            experiment_id_fs, results_fs = experiment_manager.run_experiment(baseline_params_fs)
            
            logger.info(f"基准实验（特征选择）{experiment_id_fs} 完成", 
                       f"Baseline experiment (feature selection) {experiment_id_fs} completed")
            
            # 打印主要评估指标
            logger.info(
                f"基准实验（特征选择）结果:\n"
                f"  训练集准确率: {results_fs['train']['accuracy']:.4f}\n"
                f"  测试集准确率: {results_fs['test']['accuracy']:.4f}\n"
                f"  验证集准确率: {results_fs['val']['accuracy']:.4f}",
                
                f"Baseline experiment (feature selection) results:\n"
                f"  Training accuracy: {results_fs['train']['accuracy']:.4f}\n"
                f"  Test accuracy: {results_fs['test']['accuracy']:.4f}\n"
                f"  Validation accuracy: {results_fs['val']['accuracy']:.4f}"
            )
            
            # 运行带L2正则化的特征选择实验
            logger.info("运行带L2正则化的特征选择实验", "Running feature selection experiment with L2 regularization")
            experiment_id_fs_l2, results_fs_l2 = experiment_manager.run_experiment(baseline_params_fs_l2)
            
            logger.info(f"基准实验（特征选择+L2正则化）{experiment_id_fs_l2} 完成", 
                       f"Baseline experiment (feature selection+L2 regularization) {experiment_id_fs_l2} completed")
            
            # 对比实验结果
            logger.info(
                f"基准实验对比:\n"
                f"  特征选择 vs 特征选择+L2正则化:\n"
                f"  测试集准确率: {results_fs['test']['accuracy']:.4f} vs {results_fs_l2['test']['accuracy']:.4f}\n"
                f"  准确率差异: {results_fs_l2['test']['accuracy'] - results_fs['test']['accuracy']:.4f}",
                
                f"Baseline experiments comparison:\n"
                f"  Feature selection vs Feature selection+L2 regularization:\n"
                f"  Test accuracy: {results_fs['test']['accuracy']:.4f} vs {results_fs_l2['test']['accuracy']:.4f}\n"
                f"  Accuracy difference: {results_fs_l2['test']['accuracy'] - results_fs['test']['accuracy']:.4f}"
            )

            return True
            
        except Exception as e:
            logger.error(f"基准测试实验失败: {str(e)}", f"Baseline test experiments failed: {str(e)}")
            print(f"基准测试实验失败: {str(e)}")
            # 显示堆栈跟踪以便调试
            import traceback
            traceback.print_exc()
            
            return False
    
    # 原有的基于PCA的基准测试代码（当不是特征选择模式时执行）
    else:
        logger.info("运行基准测试实验（专注于PCA方法）", "Running baseline test experiment focusing on PCA methods")
                
        # 基准参数 - 使用PCA自动组件数量
        baseline_params_pca = {
            'apply_pca': args.apply_pca,
            'n_components': args.pca_components,
            'auto_pca_variance': args.auto_pca_variance,
            'scaling_before_pca': args.scaling_before_pca,
            'normalization': 'standard',
            'class_balance': False,
            'target_samples': args.max_samples_per_label,
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

        # 带L2正则化的PCA基准
        baseline_params_pca_l2 = baseline_params_pca.copy()
        baseline_params_pca_l2['regularization'] = 'l2'
        baseline_params_pca_l2['alpha'] = 0.1
        
        # 带类别平衡的PCA基准
        baseline_params_pca_balanced = baseline_params_pca.copy()
        baseline_params_pca_balanced['class_balance'] = True

        try:
            # 运行基本PCA实验
            logger.info("运行基本PCA实验", "Running basic PCA experiment")
            experiment_id_pca, results_pca = experiment_manager.run_experiment(baseline_params_pca)
            
            logger.info(f"基准实验（PCA）{experiment_id_pca} 完成", 
                    f"Baseline experiment (PCA) {experiment_id_pca} completed")
            
            # 打印主要评估指标
            logger.info(
                f"基准实验（PCA）结果:\n"
                f"  训练集准确率: {results_pca['train']['accuracy']:.4f}\n"
                f"  测试集准确率: {results_pca['test']['accuracy']:.4f}\n"
                f"  验证集准确率: {results_pca['val']['accuracy']:.4f}",
                
                f"Baseline experiment (PCA) results:\n"
                f"  Training accuracy: {results_pca['train']['accuracy']:.4f}\n"
                f"  Test accuracy: {results_pca['test']['accuracy']:.4f}\n"
                f"  Validation accuracy: {results_pca['val']['accuracy']:.4f}"
            )
            
            # 运行带L2正则化的PCA实验
            logger.info("运行带L2正则化的PCA实验", "Running PCA experiment with L2 regularization")
            experiment_id_pca_l2, results_pca_l2 = experiment_manager.run_experiment(baseline_params_pca_l2)
            
            logger.info(f"基准实验（PCA+L2正则化）{experiment_id_pca_l2} 完成", 
                    f"Baseline experiment (PCA+L2 regularization) {experiment_id_pca_l2} completed")
            
            # 打印主要评估指标
            logger.info(
                f"基准实验（PCA+L2正则化）结果:\n"
                f"  训练集准确率: {results_pca_l2['train']['accuracy']:.4f}\n"
                f"  测试集准确率: {results_pca_l2['test']['accuracy']:.4f}\n"
                f"  验证集准确率: {results_pca_l2['val']['accuracy']:.4f}",
                
                f"Baseline experiment (PCA+L2 regularization) results:\n"
                f"  Training accuracy: {results_pca_l2['train']['accuracy']:.4f}\n"
                f"  Test accuracy: {results_pca_l2['test']['accuracy']:.4f}\n"
                f"  Validation accuracy: {results_pca_l2['val']['accuracy']:.4f}"
            )
            
            # 运行带类别平衡的PCA实验
            logger.info("运行带类别平衡的PCA实验", "Running PCA experiment with class balancing")
            experiment_id_pca_balanced, results_pca_balanced = experiment_manager.run_experiment(baseline_params_pca_balanced)
            
            logger.info(f"基准实验（PCA+类平衡）{experiment_id_pca_balanced} 完成", 
                    f"Baseline experiment (PCA+class balancing) {experiment_id_pca_balanced} completed")
            
            # 打印主要评估指标
            logger.info(
                f"基准实验（PCA+类平衡）结果:\n"
                f"  训练集准确率: {results_pca_balanced['train']['accuracy']:.4f}\n"
                f"  测试集准确率: {results_pca_balanced['test']['accuracy']:.4f}\n"
                f"  验证集准确率: {results_pca_balanced['val']['accuracy']:.4f}",
                
                f"Baseline experiment (PCA+class balancing) results:\n"
                f"  Training accuracy: {results_pca_balanced['train']['accuracy']:.4f}\n"
                f"  Test accuracy: {results_pca_balanced['test']['accuracy']:.4f}\n"
                f"  Validation accuracy: {results_pca_balanced['val']['accuracy']:.4f}"
            )
            
            # 对比实验结果
            logger.info(
                f"基准实验对比:\n"
                f"  PCA vs PCA+L2正则化:\n"
                f"  测试集准确率: {results_pca['test']['accuracy']:.4f} vs {results_pca_l2['test']['accuracy']:.4f}\n"
                f"  准确率差异: {results_pca_l2['test']['accuracy'] - results_pca['test']['accuracy']:.4f}\n"
                f"  PCA vs PCA+类平衡:\n"
                f"  测试集准确率: {results_pca['test']['accuracy']:.4f} vs {results_pca_balanced['test']['accuracy']:.4f}\n"
                f"  准确率差异: {results_pca_balanced['test']['accuracy'] - results_pca['test']['accuracy']:.4f}",
                
                f"Baseline experiments comparison:\n"
                f"  PCA vs PCA+L2 regularization:\n"
                f"  Test accuracy: {results_pca['test']['accuracy']:.4f} vs {results_pca_l2['test']['accuracy']:.4f}\n"
                f"  Accuracy difference: {results_pca_l2['test']['accuracy'] - results_pca['test']['accuracy']:.4f}\n"
                f"  PCA vs PCA+class balancing:\n"
                f"  Test accuracy: {results_pca['test']['accuracy']:.4f} vs {results_pca_balanced['test']['accuracy']:.4f}\n"
                f"  Accuracy difference: {results_pca_balanced['test']['accuracy'] - results_pca['test']['accuracy']:.4f}"
            )

            return True
            
        except Exception as e:
            logger.error(f"基准测试实验失败: {str(e)}", f"Baseline test experiments failed: {str(e)}")
            print(f"基准测试实验失败: {str(e)}")
            # 显示堆栈跟踪以便调试
            import traceback
            traceback.print_exc()
            
            return False


def run_full_experiments(experiment_manager, max_experiments, logger, use_feature_selection=False, skip_pca=False):
    """
    运行完整实验集，支持PCA或直接特征选择
    
    参数:
        experiment_manager: 实验管理器
        max_experiments: 最大实验数量 
        logger: 日志记录器
        use_feature_selection: 是否使用直接特征选择（不使用PCA）
        skip_pca: 特征选择时是否完全跳过PCA
    """
    # 根据类型选择参数网格
    if use_feature_selection:
        logger.info("创建直接特征选择参数网格（不使用PCA）", 
                   "Creating direct feature selection parameter grid (without PCA)")
        param_grid = create_feature_selection_without_pca_param_grid()
        
        # 添加对skip_pca的记录
        logger.info(f"特征选择时{'完全跳过PCA' if skip_pca else '可能使用PCA'}", 
                   f"{'Completely skipping PCA' if skip_pca else 'May use PCA'} during feature selection")
        
        # 从参数网格中采样组合
        param_combinations = sample_parameter_combinations_for_feature_selection(
            param_grid, max_samples=max_experiments)
    else:
        logger.info("创建PCA参数网格", "Creating PCA parameter grid")
        param_grid = create_feature_selection_param_grid()
        
        # 从参数网格中采样组合
        param_combinations = sample_parameter_combinations(
            param_grid, max_samples=max_experiments)
    
    logger.info(f"采样了 {len(param_combinations)} 个参数组合", 
               f"Sampled {len(param_combinations)} parameter combinations")
    
    # 预计算常用的数据变换
    if hasattr(experiment_manager.data_loader, 'precompute_transformations'):
        logger.info("预计算常用数据变换，减少重复计算", 
                   "Precomputing common data transformations to reduce redundant computation")
        
        # 选择几个常用配置进行预计算
        common_configs = []
        
        # 根据是否跳过PCA添加不同的预计算配置
        if use_feature_selection and skip_pca:
            # 跳过PCA的特征选择配置
            common_configs.extend([
                {'apply_pca': False, 'skip_pca': True, 'normalization': 'standard'},
                {'apply_pca': False, 'skip_pca': True, 'normalization': 'minmax'},
                {'apply_pca': False, 'skip_pca': True, 'normalization': None}
            ])
            logger.info("预计算直接特征选择（跳过PCA）的常用配置", 
                      "Precomputing common configurations for direct feature selection (skipping PCA)")
        else:
            # 基础PCA配置
            common_configs.extend([
                {'apply_pca': True, 'n_components': 100, 'normalization': 'standard'},
                {'apply_pca': True, 'n_components': 100, 'normalization': 'minmax'},
                {'apply_pca': True, 'n_components': 100, 'normalization': None},
                # 非PCA配置
                {'apply_pca': False, 'normalization': 'standard'},
                {'apply_pca': False, 'normalization': 'minmax'},
                {'apply_pca': False, 'normalization': None},
            ])
        
        # 预计算这些配置
        experiment_manager.data_loader.precompute_transformations(common_configs)
               
    # 批量运行实验
    logger.info(f"开始批量实验，最大实验数: {len(param_combinations)}", 
               f"Starting batch experiments, max experiments: {len(param_combinations)}")

    try:
        # 运行批量实验，直接传递参数组合列表
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
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = BilingualLogger(
        log_dir=log_dir, 
        log_name=f"experiment_{timestamp}.log"
    )

    # 特征选择时是否跳过PCA
    skip_pca = args.skip_pca or args.focus_on_fs  # 如果focus_on_fs为True，自动设置skip_pca为True

    if args.focus_on_fs:
        logger.info("开始伪逆线性模型实验（专注于直接特征选择）", 
                   "Starting pseudo-inverse linear model experiments focusing on direct feature selection")
        logger.info(f"特征选择时{'完全跳过PCA' if skip_pca else '可能使用PCA'}", 
                   f"{'Completely skipping PCA' if skip_pca else 'May use PCA'} during feature selection")
    else:
        logger.info("开始伪逆线性模型实验（专注于PCA方法）", 
                   "Starting pseudo-inverse linear model experiments focusing on PCA")
    
    # 初始化GPU支持
    if args.use_gpu:
        logger.info(f"尝试初始化GPU加速，内存使用限制: {args.gpu_memory_fraction}", 
                f"Trying to initialize GPU acceleration, memory limit: {args.gpu_memory_fraction}")
        
        gpu_available = init_gpu(use_gpu=args.use_gpu, memory_fraction=args.gpu_memory_fraction)

        from src.gpu_utils import USE_GPU
        logger.info(f"GPU全局状态检查: USE_GPU = {USE_GPU}", 
                f"Global GPU status check: USE_GPU = {USE_GPU}")
        
        if gpu_available:
            logger.info("GPU初始化成功，将使用GPU加速计算", "GPU initialization successful, using GPU acceleration")
        else:
            logger.warning("GPU初始化失败，将使用CPU进行计算", "GPU initialization failed, using CPU for computation")
    else:
        logger.info("不使用GPU加速，所有计算将在CPU上进行", "Not using GPU acceleration, all computation will be done on CPU")
        init_gpu(use_gpu=False)

    # 确保在任何条件下都导入必要的模块
    from src.brain_voxel_dataloader import BrainVoxelDataLoader
    from src.experiment_manager import ExperimentManagerWithFeatureSelection

    # 打印实验参数
    logger.info(f"实验参数：\n"
            f"  训练数据目录：{args.train_dir}\n"
            f"  测试数据目录：{args.test_dir}\n"
            f"  验证数据目录：{args.val_dir}\n"
            f"  实验结果目录：{args.exp_dir}\n"
            f"  特征选择模式：{args.fs_mode}\n"
            f"  最大实验数量：{args.max_experiments}\n"
            f"  使用GPU加速：{args.use_gpu}\n"
            f"  直接特征选择：{args.focus_on_fs}\n"
            f"  特征选择时跳过PCA：{skip_pca}\n"
            f"  自动PCA方差阈值：{args.auto_pca_variance if args.auto_pca_variance is not None else 'None'}", 
            
            f"Experiment parameters:\n"
            f"  Train directory: {args.train_dir}\n"
            f"  Test directory: {args.test_dir}\n"
            f"  Validation directory: {args.val_dir}\n"
            f"  Experiment directory: {args.exp_dir}\n"
            f"  Feature selection mode: {args.fs_mode}\n"
            f"  Maximum number of experiments: {args.max_experiments}\n"
            f"  Use GPU acceleration: {args.use_gpu}\n"
            f"  Focus on feature selection: {args.focus_on_fs}\n"
            f"  Skip PCA during feature selection: {skip_pca}\n"
            f"  Auto PCA variance threshold: {args.auto_pca_variance if args.auto_pca_variance is not None else 'None'}")
            
    # 创建数据加载器
    data_loader = BrainVoxelDataLoader(args.train_dir, args.test_dir, args.val_dir, logger=logger)

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
        success = run_baseline_experiments(experiment_manager, logger, args)
    else:
        # 先运行基准实验
        baseline_success = run_baseline_experiments(experiment_manager, logger, args)
        
        # 然后运行完整实验集
        if baseline_success:
            success = run_full_experiments(
                experiment_manager, 
                args.max_experiments, 
                logger,
                use_feature_selection=args.focus_on_fs,  # 是否专注于特征选择
                skip_pca=skip_pca  # 特征选择时是否跳过PCA
            )
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

