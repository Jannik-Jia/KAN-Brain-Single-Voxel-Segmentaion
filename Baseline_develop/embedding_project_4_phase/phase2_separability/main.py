#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: 可分离性评估模块主程序
负责评估跨受试者泛化性能和Subject Embedding必要性
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import Config
from common.data_io import DataIO
from common.visualization import Visualizer
from phase2_separability.baseline_tester import BaselineTester
from phase2_separability.loso_evaluator import LOSOEvaluator
from phase2_separability.embedding_assessor import EmbeddingNeedAssessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Phase 2: 可分离性评估')
    
    parser.add_argument('--phase0_dir', type=str,
                       default=str(Config.PHASE0_OUTPUT_DIR),
                       help='Phase 0输出目录')
    
    parser.add_argument('--phase1_dir', type=str,
                       default=str(Config.PHASE1_OUTPUT_DIR),
                       help='Phase 1输出目录')
    
    parser.add_argument('--output_dir', type=str,
                       default=str(Config.PHASE2_OUTPUT_DIR),
                       help='输出目录')
    
    parser.add_argument('--skip_visualization', action='store_true',
                       help='跳过可视化生成')
    
    parser.add_argument('--save_models', action='store_true',
                       help='保存训练好的模型')
    
    parser.add_argument('--test_models', type=str, nargs='+',
                       default=['rf', 'lr', 'deep'],
                       choices=['rf', 'lr', 'deep', 'deep_lightweight'],
                       help='要测试的模型类型')
    
    parser.add_argument('--max_loso_subjects', type=int,
                       default=8,
                       help='LOSO测试的最大受试者数（节省时间）')
    
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    
    return parser.parse_args()


def load_data(phase0_dir: Path, phase1_dir: Path) -> dict:
    """加载Phase 0和Phase 1的数据"""
    logger.info("加载数据...")
    
    # 加载Phase 0数据
    train_data = DataIO.load_numpy_data(phase0_dir / Config.TRAIN_DATA_FILE)
    val_data = DataIO.load_numpy_data(phase0_dir / Config.VAL_DATA_FILE)
    data_stats = DataIO.load_json(phase0_dir / Config.DATA_STATS_FILE)
    label_mapping = DataIO.load_json(phase0_dir / Config.LABEL_MAPPING_FILE)
    
    # 加载Phase 1结果
    phase1_results = DataIO.load_phase_output(1, phase1_dir)
    region_specificity = DataIO.load_json(phase1_dir / 'region_specificity.json')
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'data_stats': data_stats,
        'label_mapping': label_mapping,
        'phase1_results': phase1_results,
        'region_specificity': region_specificity
    }


def main():
    """主函数"""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("Phase 2: 可分离性评估开始")
    logger.info("="*80)
    
    # 设置路径
    phase0_dir = Path(args.phase0_dir)
    phase1_dir = Path(args.phase1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    viz_dir = output_dir / 'visualizations'
    if not args.skip_visualization:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_models:
        models_dir = output_dir / 'trained_models'
        models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载数据
        data = load_data(phase0_dir, phase1_dir)
        
        # 准备训练数据
        X_train = data['train_data']['X_scaled']
        y_train = data['train_data']['y']
        subjects_train = data['train_data']['subjects']
        
        X_val = data['val_data']['X_scaled']
        y_val = data['val_data']['y']
        subjects_val = data['val_data']['subjects']
        
        logger.info(f"训练数据形状: {X_train.shape}")
        logger.info(f"验证数据形状: {X_val.shape}")
        logger.info(f"设备: {args.device}")
        
        # 步骤1: Baseline性能测试
        logger.info("\n步骤1: 执行Baseline性能测试...")
        baseline_tester = BaselineTester(
            model_types=args.test_models,
            device=args.device,
            save_models=args.save_models,
            models_dir=models_dir if args.save_models else None
        )
        
        baseline_results = baseline_tester.test_baseline_performance(
            X_train, y_train, subjects_train,
            X_val, y_val, subjects_val,
            label_mapping=data['label_mapping']
        )
        
        # 保存baseline结果
        DataIO.save_json(baseline_results, output_dir / 'baseline_results.json')
        
        # 步骤2: LOSO评估
        logger.info("\n步骤2: 执行Leave-One-Subject-Out评估...")
        loso_evaluator = LOSOEvaluator(
            model_types=args.test_models,
            device=args.device,
            max_subjects=args.max_loso_subjects
        )
        
        # 合并训练和验证数据用于LOSO
        X_all = np.vstack([X_train, X_val])
        y_all = np.vstack([y_train, y_val])
        subjects_all = np.concatenate([subjects_train, subjects_val])
        
        loso_results = loso_evaluator.evaluate_loso_performance(
            X_all, y_all, subjects_all,
            label_mapping=data['label_mapping']
        )
        
        # 保存LOSO结果
        DataIO.save_json(loso_results, output_dir / 'loso_results.json')
        
        # 步骤3: 深度网络权威分析
        logger.info("\n步骤3: 执行深度网络权威分析...")
        deep_network_analysis = analyze_deep_network_authority(
            baseline_results, loso_results
        )
        
        DataIO.save_json(deep_network_analysis, output_dir / 'deep_network_analysis.json')
        
        # 步骤4: 分脑区Embedding需求评估
        logger.info("\n步骤4: 执行分脑区Embedding需求评估...")
        embedding_assessor = EmbeddingNeedAssessor(device=args.device)
        
        # 处理标签
        if data['label_mapping']['is_one_hot']:
            y_regions = np.argmax(y_all, axis=1)
        else:
            y_regions = y_all.flatten()
        
        region_embedding_needs = embedding_assessor.assess_region_embedding_needs(
            X_all, y_regions, subjects_all,
            region_specificity=data['region_specificity'],
            baseline_results=baseline_results,
            loso_results=loso_results
        )
        
        DataIO.save_json(region_embedding_needs, output_dir / 'region_embedding_needs.json')
        
        # 步骤5: 计算Phase 2决策得分
        logger.info("\n步骤5: 计算决策得分...")
        phase2_scores = compute_phase2_scores(
            baseline_results, loso_results, 
            deep_network_analysis, region_embedding_needs
        )
        
        DataIO.save_json(phase2_scores, output_dir / 'phase2_scores.json')
        
        # 步骤6: 生成可视化
        if not args.skip_visualization:
            logger.info("\n步骤6: 生成可视化图表...")
            generate_visualizations(
                baseline_results, loso_results, 
                region_embedding_needs, viz_dir
            )
        
        # 保存Phase 2完整结果
        phase2_results = {
            'baseline_summary': {
                'best_model': baseline_results['best_model'],
                'best_accuracy': baseline_results['best_accuracy'],
                'global_vs_region_improvement': baseline_results.get('global_vs_region_improvement', 0)
            },
            'loso_summary': {
                'mean_generalization_gap': loso_results['mean_generalization_gap'],
                'deep_network_advantage': loso_results.get('deep_network_advantage', 0)
            },
            'deep_network_authority': deep_network_analysis['authority_level'],
            'embedding_recommendation': {
                'global_recommendation': deep_network_analysis['recommendation'],
                'critical_regions_count': len(region_embedding_needs['critical_regions']),
                'high_priority_regions_count': len(region_embedding_needs['high_priority_regions'])
            }
        }
        
        DataIO.save_phase_output(
            phase_number=2,
            output_dir=output_dir,
            results=phase2_results,
            scores=phase2_scores,
            metadata={
                'models_tested': args.test_models,
                'device': args.device,
                'execution_time': datetime.now().isoformat()
            }
        )
        
        logger.info("\n" + "="*80)
        logger.info("Phase 2: 可分离性评估完成!")
        logger.info("="*80)
        logger.info(f"结果已保存到: {output_dir}")
        
        # 打印关键发现
        print_key_findings(baseline_results, loso_results, 
                          deep_network_analysis, region_embedding_needs, phase2_scores)
        
        return 0
        
    except Exception as e:
        logger.error(f"Phase 2 执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


def analyze_deep_network_authority(baseline_results, loso_results):
    """分析深度网络的权威性"""
    
    # 提取深度网络性能
    deep_baseline = None
    deep_loso = None
    
    for model_name, results in baseline_results['model_results'].items():
        if 'deep' in model_name:
            deep_baseline = results
            break
    
    for model_name, results in loso_results['model_results'].items():
        if 'deep' in model_name:
            deep_loso = results
            break
    
    if not deep_baseline or not deep_loso:
        return {
            'authority_level': 'UNAVAILABLE',
            'recommendation': '深度网络分析不可用',
            'confidence': 0.0
        }
    
    # 计算权威性指标
    # 1. 性能优越性
    performance_superiority = 0.0
    traditional_scores = []
    
    for model_name, results in baseline_results['model_results'].items():
        if 'deep' not in model_name:
            traditional_scores.append(results['global_accuracy'])
    
    if traditional_scores:
        avg_traditional = np.mean(traditional_scores)
        performance_superiority = (deep_baseline['global_accuracy'] - avg_traditional) / avg_traditional
    
    # 2. 泛化能力
    generalization_gap = deep_baseline['global_accuracy'] - deep_loso['mean_accuracy']
    relative_gap = generalization_gap / deep_baseline['global_accuracy']
    
    # 3. 训练稳定性
    training_stability = 1.0 - deep_baseline.get('accuracy_std', 0.1)
    
    # 综合权威性评分
    authority_score = (
        max(0, performance_superiority) * 0.4 +
        max(0, 1 - relative_gap) * 0.4 +
        training_stability * 0.2
    )
    
    # 确定权威性等级和建议
    if authority_score > 0.8:
        authority_level = "AUTHORITATIVE"
        recommendation = "深度网络显示决定性优势，强烈推荐Subject Embedding"
        confidence = "高"
    elif authority_score > 0.6:
        authority_level = "HIGHLY_CREDIBLE"
        recommendation = "深度网络表现优异，建议使用Subject Embedding"
        confidence = "中高"
    elif authority_score > 0.4:
        authority_level = "MODERATELY_CREDIBLE"
        recommendation = "深度网络有一定优势，可考虑Subject Embedding"
        confidence = "中"
    else:
        authority_level = "LIMITED_CREDIBILITY"
        recommendation = "深度网络优势有限，Subject Embedding价值需进一步评估"
        confidence = "低"
    
    return {
        'authority_score': float(authority_score),
        'authority_level': authority_level,
        'recommendation': recommendation,
        'confidence': confidence,
        'performance_superiority': float(performance_superiority),
        'generalization_ability': float(1 - relative_gap),
        'training_stability': float(training_stability),
        'baseline_accuracy': float(deep_baseline['global_accuracy']),
        'loso_accuracy': float(deep_loso['mean_accuracy']),
        'generalization_gap': float(generalization_gap)
    }


def compute_phase2_scores(baseline_results, loso_results, 
                         deep_network_analysis, region_embedding_needs):
    """计算Phase 2的决策得分"""
    
    # 全局可分离性得分
    best_baseline = baseline_results['best_accuracy']
    random_baseline = 1.0 / baseline_results['n_classes']
    global_separability = (best_baseline - random_baseline) / (1 - random_baseline)
    
    # 泛化能力得分
    generalization_gap = loso_results['mean_generalization_gap']
    generalization_score = max(0, 1 - generalization_gap)
    
    # 深度网络权威性得分
    deep_authority = deep_network_analysis['authority_score']
    
    # 分脑区需求多样性
    total_regions = len(region_embedding_needs['all_regions'])
    critical_regions = len(region_embedding_needs['critical_regions'])
    high_regions = len(region_embedding_needs['high_priority_regions'])
    
    region_diversity = (critical_regions + high_regions * 0.5) / total_regions if total_regions > 0 else 0
    
    # Subject Embedding必要性综合评分
    embedding_necessity = (
        (1 - global_separability) * 0.2 +  # 可分离性越低，越需要embedding
        (1 - generalization_score) * 0.3 +  # 泛化越差，越需要embedding
        deep_authority * 0.3 +              # 深度网络权威性
        region_diversity * 0.2              # 脑区需求多样性
    )
    
    # 实施优先级
    if embedding_necessity > 0.7:
        implementation_priority = "HIGH"
    elif embedding_necessity > 0.5:
        implementation_priority = "MEDIUM"
    else:
        implementation_priority = "LOW"
    
    return {
        'global_separability': float(global_separability),
        'generalization_ability': float(generalization_score),
        'deep_network_authority': float(deep_authority),
        'region_diversity_score': float(region_diversity),
        'embedding_necessity': float(embedding_necessity),
        'implementation_priority': implementation_priority,
        'critical_regions_ratio': float(critical_regions / total_regions) if total_regions > 0 else 0,
        'high_priority_regions_ratio': float(high_regions / total_regions) if total_regions > 0 else 0
    }


def generate_visualizations(baseline_results, loso_results, 
                           region_embedding_needs, viz_dir):
    """生成可视化图表"""
    visualizer = Visualizer(save_dir=viz_dir)
    
    # 1. 性能对比图
    models = []
    baseline_accs = []
    loso_accs = []
    
    for model_name in baseline_results['model_results']:
        if model_name in loso_results['model_results']:
            models.append(model_name)
            baseline_accs.append(baseline_results['model_results'][model_name]['global_accuracy'])
            loso_accs.append(loso_results['model_results'][model_name]['mean_accuracy'])
    
    if models:
        # 创建分组柱状图
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline (随机分割)', alpha=0.8)
        bars2 = ax.bar(x + width/2, loso_accs, width, label='LOSO (跨受试者)', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('模型')
        ax.set_ylabel('准确率')
        ax.set_title('Baseline vs LOSO 性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_comparison.png', dpi=300)
        plt.close()
    
    # 2. 脑区Embedding需求热图
    if region_embedding_needs['region_scores']:
        # 提取前30个脑区的需求得分
        region_scores = sorted(region_embedding_needs['region_scores'].items(),
                             key=lambda x: x[1], reverse=True)[:30]
        
        regions = [f"R{r[0]}" for r in region_scores]
        scores = [r[1] for r in region_scores]
        
        # 根据需求等级着色
        colors = []
        for score in scores:
            if score > 0.8:
                colors.append('darkred')
            elif score > 0.6:
                colors.append('red')
            elif score > 0.4:
                colors.append('orange')
            else:
                colors.append('yellow')
        
        visualizer.plot_bar_chart(
            scores, regions,
            title="脑区Subject Embedding需求得分 (Top 30)",
            ylabel="需求得分",
            save_name="embedding_necessity_map.png",
            colors=colors
        )


def print_key_findings(baseline_results, loso_results, 
                      deep_network_analysis, region_embedding_needs, phase2_scores):
    """打印关键发现"""
    logger.info("\n关键发现:")
    logger.info(f"- 最佳Baseline准确率: {baseline_results['best_accuracy']:.3f} ({baseline_results['best_model']})")
    logger.info(f"- 平均泛化差距: {loso_results['mean_generalization_gap']:.3f}")
    logger.info(f"- 深度网络权威性: {deep_network_analysis['authority_level']}")
    logger.info(f"- 需要紧急处理的脑区: {len(region_embedding_needs['critical_regions'])}个")
    logger.info(f"- Subject Embedding必要性得分: {phase2_scores['embedding_necessity']:.3f}")
    logger.info(f"- 实施优先级: {phase2_scores['implementation_priority']}")


if __name__ == "__main__":
    sys.exit(main())