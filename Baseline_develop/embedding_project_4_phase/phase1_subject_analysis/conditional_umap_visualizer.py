#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件UMAP可视化器
生成全面的患者差异可视化，包括102个脑区的独立UMAP
使用GPU加速进行大规模可视化
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# UMAP相关导入
try:
    from cuml import UMAP as cuUMAP  # GPU版本
    GPU_UMAP_AVAILABLE = True
except ImportError:
    GPU_UMAP_AVAILABLE = False
    from umap import UMAP  # CPU版本
    warnings.warn("cuML未安装，将使用CPU版本的UMAP")

logger = logging.getLogger(__name__)


class ConditionalUMAPVisualizer:
    """条件UMAP可视化器 - 支持GPU加速"""
    
    def __init__(self, output_dir: Path, use_gpu: bool = True, 
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 random_state: int = 42):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            use_gpu: 是否使用GPU
            n_neighbors: UMAP参数
            min_dist: UMAP参数
            random_state: 随机种子
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu and GPU_UMAP_AVAILABLE
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        
        # 创建子目录
        self.global_dir = self.output_dir / 'umap_global'
        self.regions_dir = self.output_dir / 'umap_regions'
        self.conditional_dir = self.output_dir / 'umap_conditional'
        self.summary_dir = self.output_dir / 'summary'
        
        for dir_path in [self.global_dir, self.regions_dir, 
                        self.conditional_dir, self.summary_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 为102个脑区创建子目录
        self.region_subdirs = {}
        for i in range(102):
            region_dir = self.regions_dir / f'region_{i:03d}'
            region_dir.mkdir(exist_ok=True)
            self.region_subdirs[i] = region_dir
        
        logger.info(f"条件UMAP可视化器初始化完成 (GPU: {self.use_gpu})")
    
    def generate_all_visualizations(self, X: np.ndarray, subjects: np.ndarray,
                                  regions: np.ndarray, 
                                  ml_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        生成所有可视化
        
        Args:
            X: 特征矩阵
            subjects: 患者标签
            regions: 脑区标签
            ml_results: 多标签分析结果
            
        Returns:
            可视化结果和评分
        """
        logger.info("开始生成全面的UMAP可视化套件...")
        
        results = {}
        
        # 1. 全局UMAP可视化
        logger.info("\n1. 生成全局UMAP可视化...")
        global_results = self._create_global_umap(X, subjects, regions)
        results['global_umap'] = global_results
        
        # 2. 102个脑区的独立UMAP
        logger.info("\n2. 生成102个脑区的独立UMAP...")
        region_results = self._create_all_region_umaps(X, subjects, regions)
        results['region_umaps'] = region_results
        
        # 3. 条件UMAP（移除脑区效应）
        logger.info("\n3. 生成条件UMAP可视化...")
        conditional_results = self._create_conditional_umap(X, subjects, regions)
        results['conditional_umap'] = conditional_results
        
        # 4. 生成汇总可视化
        logger.info("\n4. 生成汇总可视化...")
        summary_results = self._create_summary_visualizations(
            region_results, ml_results
        )
        results['summary'] = summary_results
        
        # 5. 3D交互式可视化（如果数据量合适）
        if len(X) < 100000:
            logger.info("\n5. 生成3D交互式可视化...")
            interactive_results = self._create_3d_interactive_visualization(
                X, subjects, regions
            )
            results['interactive'] = interactive_results
        
        logger.info("\n✅ 所有可视化生成完成!")
        return results
    
    def _create_global_umap(self, X: np.ndarray, subjects: np.ndarray,
                          regions: np.ndarray) -> Dict[str, Any]:
        """创建全局UMAP可视化"""
        
        # 如果数据太大，进行分层采样
        if len(X) > 50000:
            logger.info("  数据量大，进行分层采样...")
            sample_idx = self._stratified_sample(subjects, regions, n_samples=50000)
            X_sample = X[sample_idx]
            subjects_sample = subjects[sample_idx]
            regions_sample = regions[sample_idx]
        else:
            X_sample = X
            subjects_sample = subjects
            regions_sample = regions
        
        # 计算UMAP嵌入
        logger.info(f"  计算UMAP嵌入 (样本数: {len(X_sample)})...")
        embedding = self._compute_umap_embedding(X_sample, n_components=2)
        
        # 生成两个子图：按患者着色和按脑区着色
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 子图1：按患者着色
        unique_subjects = np.unique(subjects_sample)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
        
        for i, subj in enumerate(unique_subjects):
            mask = subjects_sample == subj
            ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[i]], s=1, alpha=0.6, label=f'Patient {int(subj)}')
        
        ax1.set_title('Global UMAP - Colored by Patient', fontsize=16)
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
        
        # 子图2：按脑区着色
        unique_regions = np.unique(regions_sample)
        
        # 使用更多的颜色映射
        if len(unique_regions) <= 20:
            colors_regions = plt.cm.tab20(np.linspace(0, 1, len(unique_regions)))
        else:
            colors_regions = plt.cm.gist_ncar(np.linspace(0, 0.9, len(unique_regions)))
        
        for i, reg in enumerate(unique_regions):
            mask = regions_sample == reg
            ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors_regions[i]], s=1, alpha=0.6, label=f'Region {int(reg)}')
        
        ax2.set_title('Global UMAP - Colored by Brain Region', fontsize=16)
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        
        # 保存图片
        plt.tight_layout()
        global_path = self.global_dir / 'global_umap_comparison.png'
        plt.savefig(global_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分别保存高分辨率版本
        self._save_single_umap(embedding, subjects_sample, 'Patient',
                             self.global_dir / 'global_by_patient.png')
        self._save_single_umap(embedding, regions_sample, 'Region',
                             self.global_dir / 'global_by_region.png')
        
        # 计算聚类质量指标
        patient_clustering_score = self._compute_clustering_score(
            embedding, subjects_sample
        )
        
        logger.info(f"  全局UMAP完成，患者聚类得分: {patient_clustering_score:.3f}")
        
        return {
            'embedding': embedding,
            'patient_clustering_score': patient_clustering_score,
            'n_samples': len(X_sample),
            'output_paths': {
                'comparison': str(global_path),
                'by_patient': str(self.global_dir / 'global_by_patient.png'),
                'by_region': str(self.global_dir / 'global_by_region.png')
            }
        }
    
    def _create_all_region_umaps(self, X: np.ndarray, subjects: np.ndarray,
                                regions: np.ndarray) -> Dict[int, Dict]:
        """为所有102个脑区创建独立的UMAP"""
        
        unique_regions = np.unique(regions)
        region_results = {}
        
        # 使用并行处理加速
        logger.info(f"  使用并行处理生成{len(unique_regions)}个脑区的UMAP...")
        
        # 准备参数
        region_data_list = []
        for reg in unique_regions:
            mask = regions == reg
            if np.sum(mask) >= 50:  # 至少50个样本
                region_data_list.append({
                    'region_id': int(reg),
                    'X': X[mask],
                    'subjects': subjects[mask],
                    'output_dir': self.region_subdirs.get(int(reg), self.regions_dir)
                })
        
        # 批量处理避免内存溢出
        batch_size = 10
        for i in range(0, len(region_data_list), batch_size):
            batch = region_data_list[i:i+batch_size]
            logger.info(f"  处理脑区批次 {i//batch_size + 1}/{(len(region_data_list)-1)//batch_size + 1}")
            
            # 串行处理每个批次（避免GPU内存问题）
            for data in batch:
                try:
                    result = self._process_single_region_umap(data)
                    if result:
                        region_results[result['region_id']] = result
                except Exception as e:
                    logger.error(f"处理脑区 {data['region_id']} 时出错: {e}")
        
        # 生成汇总图
        self._create_region_overview(region_results)
        
        # 识别高/低患者差异脑区
        high_variance_regions = []
        low_variance_regions = []
        
        scores = [(rid, r['patient_clustering_score']) 
                 for rid, r in region_results.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(scores) > 10:
            high_variance_regions = [s[0] for s in scores[:10]]
            low_variance_regions = [s[0] for s in scores[-10:]]
        
        logger.info(f"  完成{len(region_results)}个脑区的UMAP生成")
        logger.info(f"  最高患者差异脑区: {high_variance_regions[:5]}")
        
        return {
            'individual_results': region_results,
            'high_variance_regions': high_variance_regions,
            'low_variance_regions': low_variance_regions,
            'mean_clustering_score': np.mean([r['patient_clustering_score'] 
                                             for r in region_results.values()])
        }
    
    def _process_single_region_umap(self, data: Dict) -> Optional[Dict]:
        """处理单个脑区的UMAP"""
        region_id = data['region_id']
        X_region = data['X']
        subjects_region = data['subjects']
        output_dir = data['output_dir']
        
        unique_subjects = np.unique(subjects_region)
        
        # 跳过患者太少的脑区
        if len(unique_subjects) < 3:
            return None
        
        # 采样如果数据太多
        if len(X_region) > 10000:
            sample_idx = self._stratified_sample(subjects_region, 
                                               np.zeros(len(subjects_region)),
                                               n_samples=10000)
            X_region = X_region[sample_idx]
            subjects_region = subjects_region[sample_idx]
        
        # 计算UMAP
        try:
            embedding = self._compute_umap_embedding(X_region, n_components=2)
        except Exception as e:
            logger.warning(f"脑区 {region_id} UMAP计算失败: {e}")
            return None
        
        # 绘制图像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 为每个患者分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
        
        for i, subj in enumerate(unique_subjects):
            mask = subjects_region == subj
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[colors[i]], s=20, alpha=0.7,
                      label=f'Patient {int(subj)}')
        
        ax.set_title(f'Brain Region {region_id} UMAP\n'
                    f'({len(X_region)} voxels, {len(unique_subjects)} patients)',
                    fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        if len(unique_subjects) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        output_path = output_dir / f'region_{region_id:03d}_umap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 计算患者聚类得分
        clustering_score = self._compute_clustering_score(embedding, subjects_region)
        
        return {
            'region_id': region_id,
            'n_voxels': len(X_region),
            'n_patients': len(unique_subjects),
            'patient_clustering_score': clustering_score,
            'output_path': str(output_path)
        }
    
    def _create_conditional_umap(self, X: np.ndarray, subjects: np.ndarray,
                               regions: np.ndarray) -> Dict[str, Any]:
        """创建条件UMAP（移除脑区效应）"""
        
        logger.info("  移除脑区效应...")
        X_residual = self._remove_region_effects(X, regions)
        
        # 采样
        if len(X) > 50000:
            sample_idx = self._stratified_sample(subjects, regions, n_samples=50000)
            X_sample = X[sample_idx]
            X_residual_sample = X_residual[sample_idx]
            subjects_sample = subjects[sample_idx]
            regions_sample = regions[sample_idx]
        else:
            X_sample = X
            X_residual_sample = X_residual
            subjects_sample = subjects
            regions_sample = regions
        
        # 计算原始和条件UMAP
        logger.info("  计算原始空间UMAP...")
        embedding_original = self._compute_umap_embedding(X_sample, n_components=2)
        
        logger.info("  计算条件空间UMAP...")
        embedding_conditional = self._compute_umap_embedding(X_residual_sample, n_components=2)
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        unique_subjects = np.unique(subjects_sample)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
        
        # 原始空间
        for i, subj in enumerate(unique_subjects):
            mask = subjects_sample == subj
            ax1.scatter(embedding_original[mask, 0], embedding_original[mask, 1],
                       c=[colors[i]], s=1, alpha=0.6, label=f'P{int(subj)}')
        
        ax1.set_title('Original Space UMAP', fontsize=16)
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # 条件空间
        for i, subj in enumerate(unique_subjects):
            mask = subjects_sample == subj
            ax2.scatter(embedding_conditional[mask, 0], embedding_conditional[mask, 1],
                       c=[colors[i]], s=1, alpha=0.6, label=f'P{int(subj)}')
        
        ax2.set_title('Conditional Space UMAP\n(Region Effects Removed)', fontsize=16)
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        comparison_path = self.conditional_dir / 'before_after_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算聚类改善
        score_original = self._compute_clustering_score(embedding_original, subjects_sample)
        score_conditional = self._compute_clustering_score(embedding_conditional, subjects_sample)
        improvement = score_conditional - score_original
        
        logger.info(f"  条件UMAP完成，聚类改善: {improvement:.3f}")
        
        return {
            'embedding_original': embedding_original,
            'embedding_conditional': embedding_conditional,
            'clustering_score_original': score_original,
            'clustering_score_conditional': score_conditional,
            'clustering_improvement': improvement,
            'output_path': str(comparison_path)
        }
    
    def _create_summary_visualizations(self, region_results: Dict,
                                     ml_results: Optional[Dict]) -> Dict[str, Any]:
        """创建汇总可视化"""
        
        # 1. 脑区患者效应热图
        if ml_results and 'variance_decomposition' in ml_results:
            region_effects = ml_results['variance_decomposition'].get(
                'region_specific_patient_effects', {}
            )
            
            if region_effects:
                self._create_region_effects_heatmap(region_effects)
        
        # 2. 方差成分饼图
        if ml_results and 'variance_decomposition' in ml_results:
            self._create_variance_pie_chart(ml_results['variance_decomposition'])
        
        # 3. 脑区聚类得分分布
        if region_results and 'individual_results' in region_results:
            self._create_clustering_score_distribution(region_results['individual_results'])
        
        return {
            'heatmap_path': str(self.summary_dir / 'region_effects_heatmap.png'),
            'pie_chart_path': str(self.summary_dir / 'variance_components.png'),
            'distribution_path': str(self.summary_dir / 'clustering_scores.png')
        }
    
    def _create_region_overview(self, region_results: Dict[int, Dict]):
        """创建102个脑区的总览图"""
        n_regions = len(region_results)
        
        # 创建网格布局
        n_cols = 10
        n_rows = (n_regions + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(30, 3 * n_rows))
        
        # 按脑区ID排序
        sorted_regions = sorted(region_results.items())
        
        for idx, (region_id, result) in enumerate(sorted_regions):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # 显示缩略信息
            score = result['patient_clustering_score']
            n_patients = result['n_patients']
            
            # 根据得分着色
            color = plt.cm.RdYlBu_r(score)
            ax.text(0.5, 0.6, f'R{region_id}', ha='center', va='center',
                   fontsize=14, weight='bold')
            ax.text(0.5, 0.4, f'Score: {score:.2f}', ha='center', va='center',
                   fontsize=10)
            ax.text(0.5, 0.2, f'{n_patients} patients', ha='center', va='center',
                   fontsize=8)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor(color)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加边框表示高差异脑区
            if score > 0.7:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        
        plt.suptitle('Brain Region UMAP Overview - Patient Clustering Scores\n'
                    '(Red border = High patient variance)', fontsize=20)
        plt.tight_layout()
        
        overview_path = self.regions_dir / 'overview_all_102_regions.png'
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  生成脑区总览图: {overview_path}")
    
    def _create_region_effects_heatmap(self, region_effects: Dict[int, float]):
        """创建脑区患者效应热图"""
        # 创建10x11的矩阵（共110个位置，填充102个脑区）
        matrix = np.full((11, 10), np.nan)
        
        for region_id, effect in region_effects.items():
            if region_id < 102:
                row = region_id // 10
                col = region_id % 10
                matrix[row, col] = effect
        
        plt.figure(figsize=(12, 10))
        
        # 创建热图
        mask = np.isnan(matrix)
        sns.heatmap(matrix, cmap='YlOrRd', mask=mask,
                   vmin=0, vmax=max(region_effects.values()),
                   cbar_kws={'label': 'Patient Effect Ratio'},
                   square=True, linewidths=0.5,
                   annot=True, fmt='.2f', annot_kws={'size': 8})
        
        # 添加脑区标签
        for region_id in range(102):
            row = region_id // 10
            col = region_id % 10
            plt.text(col + 0.5, row + 0.1, f'R{region_id}',
                    ha='center', va='top', fontsize=6, color='blue')
        
        plt.title('Patient Effect Strength by Brain Region', fontsize=16)
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        heatmap_path = self.summary_dir / 'region_effects_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_variance_pie_chart(self, variance_decomp: Dict):
        """创建方差成分饼图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 饼图1：包含交互效应
        labels1 = ['Patient', 'Region', 'Interaction', 'Error']
        sizes1 = [
            variance_decomp['patient_variance_ratio'],
            variance_decomp['region_variance_ratio'],
            variance_decomp['interaction_variance_ratio'],
            variance_decomp['error_variance_ratio']
        ]
        colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
        
        wedges, texts, autotexts = ax1.pie(sizes1, labels=labels1, colors=colors1,
                                           autopct='%1.1f%%', startangle=90)
        ax1.set_title('Variance Components (Full Model)', fontsize=14)
        
        # 饼图2：只显示主效应
        labels2 = ['Patient Effect', 'Region Effect', 'Other']
        patient_ratio = variance_decomp['patient_variance_ratio']
        region_ratio = variance_decomp['region_variance_ratio']
        other_ratio = 1 - patient_ratio - region_ratio
        sizes2 = [patient_ratio, region_ratio, other_ratio]
        colors2 = ['#FF6B6B', '#4ECDC4', '#BDC3C7']
        
        wedges2, texts2, autotexts2 = ax2.pie(sizes2, labels=labels2, colors=colors2,
                                              autopct='%1.1f%%', startangle=90)
        ax2.set_title('Main Effects Only', fontsize=14)
        
        # 添加文字说明
        fig.text(0.5, 0.02, f"Patient Effect: {patient_ratio:.1%} - " +
                ("Strong evidence for Subject Embedding" if patient_ratio > 0.1 
                 else "Weak patient effect"),
                ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        pie_path = self.summary_dir / 'variance_components.png'
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_clustering_score_distribution(self, region_results: Dict[int, Dict]):
        """创建聚类得分分布图"""
        scores = [r['patient_clustering_score'] for r in region_results.values()]
        region_ids = list(region_results.keys())
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 直方图
        ax1.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(scores):.3f}')
        ax1.axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(scores):.3f}')
        ax1.set_xlabel('Patient Clustering Score')
        ax1.set_ylabel('Number of Brain Regions')
        ax1.set_title('Distribution of Patient Clustering Scores Across Brain Regions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 排序的条形图（前20和后20）
        sorted_idx = np.argsort(scores)[::-1]
        top_20_idx = sorted_idx[:20]
        bottom_20_idx = sorted_idx[-20:]
        
        combined_idx = np.concatenate([top_20_idx, bottom_20_idx])
        combined_scores = [scores[i] for i in combined_idx]
        combined_labels = [f'R{region_ids[i]}' for i in combined_idx]
        
        colors = ['red' if i < 20 else 'blue' for i in range(40)]
        
        ax2.bar(range(40), combined_scores, color=colors, alpha=0.7)
        ax2.set_xticks(range(40))
        ax2.set_xticklabels(combined_labels, rotation=90, ha='right')
        ax2.set_ylabel('Patient Clustering Score')
        ax2.set_title('Top 20 (Red) and Bottom 20 (Blue) Brain Regions by Patient Clustering')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 添加阈值线
        ax2.axhline(0.5, color='black', linestyle=':', alpha=0.5,
                   label='Moderate clustering')
        ax2.axhline(0.7, color='red', linestyle=':', alpha=0.5,
                   label='Strong clustering')
        ax2.legend()
        
        plt.tight_layout()
        dist_path = self.summary_dir / 'clustering_scores_distribution.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_3d_interactive_visualization(self, X: np.ndarray, subjects: np.ndarray,
                                           regions: np.ndarray) -> Dict[str, Any]:
        """创建3D交互式可视化（HTML）"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 采样
            if len(X) > 10000:
                sample_idx = self._stratified_sample(subjects, regions, n_samples=10000)
                X_sample = X[sample_idx]
                subjects_sample = subjects[sample_idx]
                regions_sample = regions[sample_idx]
            else:
                X_sample = X
                subjects_sample = subjects
                regions_sample = regions
            
            # 计算3D UMAP
            logger.info("  计算3D UMAP嵌入...")
            embedding_3d = self._compute_umap_embedding(X_sample, n_components=3)
            
            # 创建交互式3D散点图
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=('Colored by Patient', 'Colored by Brain Region')
            )
            
            # 按患者着色
            unique_subjects = np.unique(subjects_sample)
            for i, subj in enumerate(unique_subjects):
                mask = subjects_sample == subj
                fig.add_trace(
                    go.Scatter3d(
                        x=embedding_3d[mask, 0],
                        y=embedding_3d[mask, 1],
                        z=embedding_3d[mask, 2],
                        mode='markers',
                        marker=dict(size=2, opacity=0.6),
                        name=f'Patient {int(subj)}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # 按脑区着色
            unique_regions = np.unique(regions_sample)
            for i, reg in enumerate(unique_regions[:20]):  # 只显示前20个脑区
                mask = regions_sample == reg
                fig.add_trace(
                    go.Scatter3d(
                        x=embedding_3d[mask, 0],
                        y=embedding_3d[mask, 1],
                        z=embedding_3d[mask, 2],
                        mode='markers',
                        marker=dict(size=2, opacity=0.6),
                        name=f'Region {int(reg)}',
                        showlegend=(i < 20)
                    ),
                    row=1, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title='3D Interactive UMAP Visualization',
                scene=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3'
                ),
                scene2=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3'
                ),
                height=800,
                showlegend=True
            )
            
            # 保存HTML
            html_path = self.global_dir / 'global_3d_interactive.html'
            fig.write_html(str(html_path))
            
            logger.info(f"  3D交互式可视化已保存: {html_path}")
            
            return {
                'embedding_3d': embedding_3d,
                'output_path': str(html_path)
            }
            
        except ImportError:
            logger.warning("Plotly未安装，跳过3D交互式可视化")
            return {}
    
    def _compute_umap_embedding(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """计算UMAP嵌入"""
        if self.use_gpu:
            # 使用GPU版本
            umap_model = cuUMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
                verbose=False
            )
        else:
            # 使用CPU版本
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        embedding = umap_model.fit_transform(X)
        
        # 如果使用GPU，转换回numpy
        if self.use_gpu and hasattr(embedding, 'get'):
            embedding = embedding.get()
        
        return embedding
    
    def _compute_clustering_score(self, embedding: np.ndarray, labels: np.ndarray) -> float:
        """
        计算聚类质量得分
        使用Calinski-Harabasz指数的归一化版本
        """
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        try:
            # Calinski-Harabasz得分（越高越好）
            ch_score = calinski_harabasz_score(embedding, labels)
            
            # Davies-Bouldin得分（越低越好）
            db_score = davies_bouldin_score(embedding, labels)
            
            # 归一化并组合
            # 使用sigmoid函数将CH分数映射到0-1
            normalized_ch = 1 / (1 + np.exp(-ch_score / 100))
            
            # DB分数反转并归一化
            normalized_db = 1 / (1 + db_score)
            
            # 组合得分
            combined_score = 0.7 * normalized_ch + 0.3 * normalized_db
            
            return float(combined_score)
            
        except Exception:
            return 0.0
    
    def _remove_region_effects(self, X: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """移除脑区主效应"""
        X_residual = X.copy()
        grand_mean = np.mean(X, axis=0)
        
        for region in np.unique(regions):
            mask = regions == region
            if np.sum(mask) > 0:
                region_mean = np.mean(X[mask], axis=0)
                region_effect = region_mean - grand_mean
                X_residual[mask] -= region_effect
        
        return X_residual
    
    def _stratified_sample(self, subjects: np.ndarray, regions: np.ndarray,
                          n_samples: int) -> np.ndarray:
        """分层采样确保各类别都有代表"""
        # 创建组合标签
        combined_labels = subjects * 1000 + regions  # 假设脑区ID < 1000
        unique_combinations = np.unique(combined_labels)
        
        # 计算每个组合应该采样的数量
        n_per_combination = max(1, n_samples // len(unique_combinations))
        
        sampled_indices = []
        remaining_quota = n_samples
        
        for combo in unique_combinations:
            if remaining_quota <= 0:
                break
                
            combo_indices = np.where(combined_labels == combo)[0]
            n_combo = len(combo_indices)
            
            # 采样数量：最少1个，最多不超过该组合的总数
            n_to_sample = min(n_per_combination, n_combo, remaining_quota)
            
            if n_to_sample > 0:
                sampled = np.random.choice(combo_indices, n_to_sample, replace=False)
                sampled_indices.extend(sampled)
                remaining_quota -= n_to_sample
        
        return np.array(sampled_indices)
    
    def _save_single_umap(self, embedding: np.ndarray, labels: np.ndarray,
                         label_type: str, output_path: Path):
        """保存单个UMAP图"""
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        
        # 选择合适的颜色映射
        if len(unique_labels) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        else:
            colors = plt.cm.gist_ncar(np.linspace(0, 0.9, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[i]], s=2, alpha=0.6,
                       label=f'{label_type} {int(label)}')
        
        plt.title(f'UMAP Projection - Colored by {label_type}', fontsize=16)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        
        if len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()