#!/usr/bin/env python3
"""
3D Dataset Label Voxel Count Statistics
Analyzes label distribution in the 3D validated dataset
"""

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class LabelVoxelAnalyzer:
    """Analyzer for label voxel count statistics"""

    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "statistics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.subject_stats = []  # Per-subject statistics
        self.aggregate_stats = {}  # Aggregated statistics across all subjects

        self.logger.info(f"Analyzer initialized")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_3d_file(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load 3D validated MAT file"""
        self.logger.debug(f"Loading file: {file_path}")

        try:
            data = {}
            with h5py.File(file_path, 'r') as f:
                # Load region_labels (contains 0-101 labels)
                if 'region_labels' in f:
                    data['region_labels'] = f['region_labels'][()]
                else:
                    self.logger.error(f"'region_labels' not found in {file_path}")
                    return None

                # Load region mask
                if 'region_mask' in f:
                    data['region_mask'] = f['region_mask'][()]
                elif 'region' in f:
                    data['region_mask'] = f['region'][()]
                else:
                    self.logger.warning(f"No region mask found in {file_path}")
                    data['region_mask'] = None

                self.logger.debug(f"Loaded shapes: region_labels={data['region_labels'].shape}")
                if data['region_mask'] is not None:
                    self.logger.debug(f"region_mask={data['region_mask'].shape}")

            return data

        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return None

    def analyze_single_subject(self, file_path: Path) -> Dict:
        """Analyze label distribution for a single subject"""
        subject_id = file_path.stem.replace('_3d_validated', '')
        self.logger.info(f"Analyzing subject: {subject_id}")

        # Load data
        data = self.load_3d_file(file_path)
        if data is None:
            return None

        region_labels = data['region_labels']
        region_mask = data['region_mask']

        # Apply region mask if available
        if region_mask is not None:
            mask = region_mask.astype(bool)
            valid_labels = region_labels[mask]
            total_brain_voxels = np.sum(mask)
        else:
            valid_labels = region_labels[region_labels > 0]  # Exclude background
            total_brain_voxels = len(valid_labels)

        # Count each label (0-101)
        label_counts = {}
        for label_id in range(102):  # 0 to 101
            count = np.sum(valid_labels == label_id)
            label_counts[label_id] = count

        # Calculate percentages
        label_percentages = {}
        for label_id, count in label_counts.items():
            percentage = (count / total_brain_voxels * 100) if total_brain_voxels > 0 else 0
            label_percentages[label_id] = percentage

        # Summary statistics
        non_zero_labels = [label_id for label_id, count in label_counts.items() if count > 0]
        unique_labels_count = len(non_zero_labels)

        result = {
            'subject_id': subject_id,
            'file_path': str(file_path),
            'total_brain_voxels': total_brain_voxels,
            'unique_labels_count': unique_labels_count,
            'label_counts': label_counts,
            'label_percentages': label_percentages,
            'non_zero_labels': non_zero_labels
        }

        self.logger.info(f"Subject {subject_id}: {total_brain_voxels} brain voxels, {unique_labels_count} unique labels")

        return result

    def analyze_all_subjects(self) -> Tuple[List[Dict], Dict]:
        """Analyze all subjects in the dataset"""
        self.logger.info("Starting analysis of all subjects...")

        # Find all 3D validated files
        mat_files = list(self.data_dir.glob("*_3d_validated.mat"))

        if len(mat_files) == 0:
            self.logger.error(f"No *_3d_validated.mat files found in {self.data_dir}")
            return [], {}

        self.logger.info(f"Found {len(mat_files)} 3D validated files")

        # Analyze each subject
        subject_results = []
        aggregate_counts = {label_id: 0 for label_id in range(102)}
        total_subjects = 0

        for file_path in tqdm(sorted(mat_files), desc="Analyzing subjects"):
            result = self.analyze_single_subject(file_path)
            if result is not None:
                subject_results.append(result)
                total_subjects += 1

                # Aggregate counts
                for label_id, count in result['label_counts'].items():
                    aggregate_counts[label_id] += count

        # Calculate aggregate statistics
        total_voxels_all = sum(aggregate_counts.values())
        aggregate_percentages = {}
        for label_id, count in aggregate_counts.items():
            percentage = (count / total_voxels_all * 100) if total_voxels_all > 0 else 0
            aggregate_percentages[label_id] = percentage

        aggregate_stats = {
            'total_subjects': total_subjects,
            'total_voxels_all_subjects': total_voxels_all,
            'aggregate_label_counts': aggregate_counts,
            'aggregate_label_percentages': aggregate_percentages,
            'labels_present_in_dataset': [label_id for label_id, count in aggregate_counts.items() if count > 0]
        }

        self.logger.info(f"Analysis complete: {total_subjects} subjects analyzed")
        self.logger.info(f"Total voxels across all subjects: {total_voxels_all:,}")
        self.logger.info(f"Labels present in dataset: {len(aggregate_stats['labels_present_in_dataset'])}")

        return subject_results, aggregate_stats

    def save_csv_reports(self, subject_results: List[Dict], aggregate_stats: Dict):
        """Save detailed CSV reports"""
        self.logger.info("Saving CSV reports...")

        # 1. Per-subject summary
        summary_data = []
        for result in subject_results:
            summary_data.append({
                'subject_id': result['subject_id'],
                'total_brain_voxels': result['total_brain_voxels'],
                'unique_labels_count': result['unique_labels_count'],
                'non_zero_labels': ','.join(map(str, result['non_zero_labels']))
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "subject_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Subject summary saved: {summary_path}")

        # 2. Detailed per-subject label counts
        detailed_data = []
        for result in subject_results:
            for label_id in range(102):
                detailed_data.append({
                    'subject_id': result['subject_id'],
                    'label_id': label_id,
                    'voxel_count': result['label_counts'][label_id],
                    'percentage': result['label_percentages'][label_id]
                })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = self.output_dir / "detailed_label_counts.csv"
        detailed_df.to_csv(detailed_path, index=False)
        self.logger.info(f"Detailed label counts saved: {detailed_path}")

        # 3. Aggregate statistics
        agg_data = []
        for label_id in range(102):
            agg_data.append({
                'label_id': label_id,
                'total_voxel_count': aggregate_stats['aggregate_label_counts'][label_id],
                'percentage_of_all_voxels': aggregate_stats['aggregate_label_percentages'][label_id],
                'present_in_subjects': sum(1 for result in subject_results
                                         if result['label_counts'][label_id] > 0)
            })

        agg_df = pd.DataFrame(agg_data)
        agg_path = self.output_dir / "aggregate_label_statistics.csv"
        agg_df.to_csv(agg_path, index=False)
        self.logger.info(f"Aggregate statistics saved: {agg_path}")

        return summary_df, detailed_df, agg_df

    def create_visualizations(self, subject_results: List[Dict], aggregate_stats: Dict):
        """Create visualization plots"""
        self.logger.info("Creating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Aggregate label distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Label Voxel Count Statistics - 3D Dataset', fontsize=16, fontweight='bold')

        # Top-left: Overall distribution (non-zero labels only)
        non_zero_labels = [label_id for label_id, count in aggregate_stats['aggregate_label_counts'].items() if count > 0]
        non_zero_counts = [aggregate_stats['aggregate_label_counts'][label_id] for label_id in non_zero_labels]

        axes[0, 0].bar(range(len(non_zero_labels)), non_zero_counts, alpha=0.7)
        axes[0, 0].set_title('Aggregate Voxel Counts by Label (Non-zero Only)')
        axes[0, 0].set_xlabel('Label Index')
        axes[0, 0].set_ylabel('Total Voxel Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        if len(non_zero_labels) > 20:
            step = len(non_zero_labels) // 20
            axes[0, 0].set_xticks(range(0, len(non_zero_labels), step))
            axes[0, 0].set_xticklabels([str(non_zero_labels[i]) for i in range(0, len(non_zero_labels), step)])
        else:
            axes[0, 0].set_xticks(range(len(non_zero_labels)))
            axes[0, 0].set_xticklabels([str(label) for label in non_zero_labels])

        # Top-right: Top 20 most frequent labels
        sorted_labels = sorted(non_zero_labels,
                             key=lambda x: aggregate_stats['aggregate_label_counts'][x],
                             reverse=True)[:20]
        top_counts = [aggregate_stats['aggregate_label_counts'][label_id] for label_id in sorted_labels]

        bars = axes[0, 1].bar(range(len(sorted_labels)), top_counts, alpha=0.7, color='coral')
        axes[0, 1].set_title('Top 20 Most Frequent Labels')
        axes[0, 1].set_xlabel('Label ID')
        axes[0, 1].set_ylabel('Total Voxel Count')
        axes[0, 1].set_xticks(range(len(sorted_labels)))
        axes[0, 1].set_xticklabels([str(label) for label in sorted_labels], rotation=45)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, top_counts)):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{count:,}', ha='center', va='bottom', fontsize=8)

        # Bottom-left: Distribution of unique labels per subject
        unique_counts = [result['unique_labels_count'] for result in subject_results]
        axes[1, 0].hist(unique_counts, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Distribution of Unique Labels per Subject')
        axes[1, 0].set_xlabel('Number of Unique Labels')
        axes[1, 0].set_ylabel('Number of Subjects')
        axes[1, 0].axvline(np.mean(unique_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(unique_counts):.1f}')
        axes[1, 0].legend()

        # Bottom-right: Brain voxel count distribution
        brain_voxels = [result['total_brain_voxels'] for result in subject_results]
        axes[1, 1].hist(brain_voxels, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 1].set_title('Distribution of Brain Voxel Counts per Subject')
        axes[1, 1].set_xlabel('Total Brain Voxels')
        axes[1, 1].set_ylabel('Number of Subjects')
        axes[1, 1].axvline(np.mean(brain_voxels), color='red', linestyle='--',
                          label=f'Mean: {np.mean(brain_voxels):,.0f}')
        axes[1, 1].legend()

        plt.tight_layout()
        overview_path = self.output_dir / "label_statistics_overview.png"
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Overview plot saved: {overview_path}")
        plt.close()

        # 2. Label presence heatmap (subjects vs labels)
        self.logger.info("Creating label presence heatmap...")

        # Create presence matrix (subjects x top labels)
        top_30_labels = sorted(non_zero_labels,
                              key=lambda x: aggregate_stats['aggregate_label_counts'][x],
                              reverse=True)[:30]

        presence_matrix = np.zeros((len(subject_results), len(top_30_labels)))
        subject_ids = []

        for i, result in enumerate(subject_results):
            subject_ids.append(result['subject_id'])
            for j, label_id in enumerate(top_30_labels):
                presence_matrix[i, j] = 1 if result['label_counts'][label_id] > 0 else 0

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(presence_matrix,
                    xticklabels=[f'Label {label}' for label in top_30_labels],
                    yticklabels=subject_ids,
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Label Present'},
                    ax=ax)
        ax.set_title('Label Presence Across Subjects (Top 30 Labels)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Label ID')
        ax.set_ylabel('Subject ID')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        heatmap_path = self.output_dir / "label_presence_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Heatmap saved: {heatmap_path}")
        plt.close()

        # 3. Summary statistics text plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # Prepare summary text
        summary_text = f"""
3D Dataset Label Statistics Summary

Dataset Overview:
• Total Subjects: {aggregate_stats['total_subjects']}
• Total Brain Voxels (all subjects): {aggregate_stats['total_voxels_all_subjects']:,}
• Labels Present in Dataset: {len(aggregate_stats['labels_present_in_dataset'])} out of 102

Per-Subject Statistics:
• Average Brain Voxels per Subject: {np.mean([r['total_brain_voxels'] for r in subject_results]):,.0f}
• Average Unique Labels per Subject: {np.mean([r['unique_labels_count'] for r in subject_results]):.1f}
• Min Unique Labels per Subject: {min([r['unique_labels_count'] for r in subject_results])}
• Max Unique Labels per Subject: {max([r['unique_labels_count'] for r in subject_results])}

Top 10 Most Frequent Labels:
"""

        # Add top 10 labels
        top_10_labels = sorted(non_zero_labels,
                              key=lambda x: aggregate_stats['aggregate_label_counts'][x],
                              reverse=True)[:10]

        for i, label_id in enumerate(top_10_labels, 1):
            count = aggregate_stats['aggregate_label_counts'][label_id]
            percentage = aggregate_stats['aggregate_label_percentages'][label_id]
            summary_text += f"{i:2d}. Label {label_id:2d}: {count:,} voxels ({percentage:.2f}%)\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        ax.set_title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        summary_path = self.output_dir / "dataset_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Summary plot saved: {summary_path}")
        plt.close()

    def save_json_report(self, subject_results: List[Dict], aggregate_stats: Dict):
        """Save complete results as JSON"""
        self.logger.info("Saving JSON report...")

        # Prepare data for JSON serialization
        json_data = {
            'analysis_info': {
                'data_directory': str(self.data_dir),
                'total_subjects_analyzed': len(subject_results),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'aggregate_statistics': aggregate_stats,
            'per_subject_results': subject_results
        }

        json_path = self.output_dir / "complete_label_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

        self.logger.info(f"Complete JSON report saved: {json_path}")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        self.logger.info("Starting complete label voxel analysis...")

        # Analyze all subjects
        subject_results, aggregate_stats = self.analyze_all_subjects()

        if not subject_results:
            self.logger.error("No subjects were successfully analyzed. Exiting.")
            return

        # Save reports
        self.save_csv_reports(subject_results, aggregate_stats)
        self.create_visualizations(subject_results, aggregate_stats)
        self.save_json_report(subject_results, aggregate_stats)

        self.logger.info(f"Analysis complete! Results saved to: {self.output_dir}")

        # Print quick summary
        print(f"\n{'='*60}")
        print(f"LABEL VOXEL ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Subjects analyzed: {len(subject_results)}")
        print(f"Total brain voxels: {aggregate_stats['total_voxels_all_subjects']:,}")
        print(f"Labels present: {len(aggregate_stats['labels_present_in_dataset'])}/102")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")

def main():
    # Hardcoded paths - no command line arguments needed
    data_dir = '/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated'
    output_dir = '/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_validated/statistics'

    print("Starting 3D Dataset Label Voxel Analysis...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Create analyzer and run
    analyzer = LabelVoxelAnalyzer(data_dir, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()