#!/usr/bin/env python3
"""
ResNet Leave-One-Out Results Analysis

Analyzes and summarizes results from Leave-One-Out cross-validation
of MRI ResNet brain region classification.

Usage:
    python analyze_resnet_results.py --results_dir ./results_leave_one_out_resnet
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


class ResNetResultsAnalyzer:
    """Analyzer for ResNet Leave-One-Out cross-validation results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.subject_results = {}
        self.summary_stats = {}
        
        # Load all subject results
        self._load_results()
    
    def _load_results(self):
        """Load results from all subjects"""
        print(f"Loading results from {self.results_dir}")
        
        # Find all subject result directories
        subject_dirs = list(self.results_dir.glob('resnet_test_subject_*'))
        subject_dirs.sort(key=lambda x: int(x.name.split('_')[-1]))
        
        print(f"Found {len(subject_dirs)} subject result directories")
        
        for subject_dir in subject_dirs:
            subject_num = int(subject_dir.name.split('_')[-1])
            results_file = subject_dir / 'training_results.json'
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    self.subject_results[subject_num] = {
                        'best_epoch': data.get('best_epoch', None),
                        'best_f1': data.get('best_f1', None),
                        'final_f1': data.get('final_f1', None),
                        'history': data.get('history', {}),
                        'final_metrics': data.get('final_metrics', {}),
                        'subject_dir': subject_dir
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to load results for subject {subject_num}: {e}")
            else:
                print(f"Warning: No results file found for subject {subject_num}")
        
        print(f"Successfully loaded results for {len(self.subject_results)} subjects")
    
    def compute_summary_statistics(self) -> Dict:
        """Compute summary statistics across all subjects"""
        if not self.subject_results:
            print("No results to analyze")
            return {}
        
        # Extract key metrics
        best_f1_scores = []
        final_f1_scores = []
        best_epochs = []
        
        # Per-class metrics
        per_class_f1 = defaultdict(list)
        per_class_precision = defaultdict(list)
        per_class_recall = defaultdict(list)
        
        for subject_num, results in self.subject_results.items():
            if results['best_f1'] is not None:
                best_f1_scores.append(results['best_f1'])
            
            if results['final_f1'] is not None:
                final_f1_scores.append(results['final_f1'])
            
            if results['best_epoch'] is not None:
                best_epochs.append(results['best_epoch'])
            
            # Per-class metrics
            final_metrics = results.get('final_metrics', {})
            if 'per_class_f1' in final_metrics:
                for class_id, f1 in enumerate(final_metrics['per_class_f1']):
                    per_class_f1[class_id].append(f1)
            
            if 'per_class_precision' in final_metrics:
                for class_id, prec in enumerate(final_metrics['per_class_precision']):
                    per_class_precision[class_id].append(prec)
            
            if 'per_class_recall' in final_metrics:
                for class_id, rec in enumerate(final_metrics['per_class_recall']):
                    per_class_recall[class_id].append(rec)
        
        # Compute statistics
        summary = {
            'num_subjects': len(self.subject_results),
            'num_successful': len(best_f1_scores),
            
            # Best F1 statistics
            'best_f1_mean': np.mean(best_f1_scores) if best_f1_scores else 0,
            'best_f1_std': np.std(best_f1_scores) if best_f1_scores else 0,
            'best_f1_min': np.min(best_f1_scores) if best_f1_scores else 0,
            'best_f1_max': np.max(best_f1_scores) if best_f1_scores else 0,
            'best_f1_median': np.median(best_f1_scores) if best_f1_scores else 0,
            
            # Final F1 statistics
            'final_f1_mean': np.mean(final_f1_scores) if final_f1_scores else 0,
            'final_f1_std': np.std(final_f1_scores) if final_f1_scores else 0,
            'final_f1_min': np.min(final_f1_scores) if final_f1_scores else 0,
            'final_f1_max': np.max(final_f1_scores) if final_f1_scores else 0,
            'final_f1_median': np.median(final_f1_scores) if final_f1_scores else 0,
            
            # Training characteristics
            'best_epoch_mean': np.mean(best_epochs) if best_epochs else 0,
            'best_epoch_std': np.std(best_epochs) if best_epochs else 0,
            
            # Raw data for further analysis
            'best_f1_scores': best_f1_scores,
            'final_f1_scores': final_f1_scores,
            'best_epochs': best_epochs,
            'per_class_f1': dict(per_class_f1),
            'per_class_precision': dict(per_class_precision),
            'per_class_recall': dict(per_class_recall)
        }
        
        self.summary_stats = summary
        return summary
    
    def create_detailed_dataframe(self) -> pd.DataFrame:
        """Create detailed DataFrame with all subject results"""
        rows = []
        
        for subject_num in sorted(self.subject_results.keys()):
            results = self.subject_results[subject_num]
            
            row = {
                'Subject': subject_num,
                'Best_Epoch': results.get('best_epoch'),
                'Best_F1': results.get('best_f1'),
                'Final_F1': results.get('final_f1'),
            }
            
            # Add final metrics if available
            final_metrics = results.get('final_metrics', {})
            if final_metrics:
                row.update({
                    'Final_Macro_Precision': final_metrics.get('macro_precision'),
                    'Final_Macro_Recall': final_metrics.get('macro_recall'),
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_results_summary(self, output_dir: Path):
        """Create comprehensive results visualization"""
        if not self.subject_results:
            print("No results to plot")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. F1 Score Distribution
        ax1 = plt.subplot(3, 3, 1)
        f1_scores = self.summary_stats['best_f1_scores']
        if f1_scores:
            plt.hist(f1_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(f1_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(f1_scores):.4f}')
            plt.xlabel('Best F1 Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Best F1 Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. F1 Score by Subject
        ax2 = plt.subplot(3, 3, 2)
        subjects = sorted(self.subject_results.keys())
        subject_f1s = [self.subject_results[s]['best_f1'] for s in subjects 
                       if self.subject_results[s]['best_f1'] is not None]
        valid_subjects = [s for s in subjects 
                         if self.subject_results[s]['best_f1'] is not None]
        
        if subject_f1s:
            plt.plot(valid_subjects, subject_f1s, 'o-', markersize=4, linewidth=1)
            plt.xlabel('Test Subject')
            plt.ylabel('Best F1 Score')
            plt.title('F1 Score by Test Subject')
            plt.grid(True, alpha=0.3)
            
            # Highlight best/worst performers
            best_idx = np.argmax(subject_f1s)
            worst_idx = np.argmin(subject_f1s)
            plt.annotate(f'Best: S{valid_subjects[best_idx]}\n{subject_f1s[best_idx]:.4f}',
                        xy=(valid_subjects[best_idx], subject_f1s[best_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))
            plt.annotate(f'Worst: S{valid_subjects[worst_idx]}\n{subject_f1s[worst_idx]:.4f}',
                        xy=(valid_subjects[worst_idx], subject_f1s[worst_idx]),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7))
        
        # 3. Best Epoch Distribution
        ax3 = plt.subplot(3, 3, 3)
        best_epochs = self.summary_stats['best_epochs']
        if best_epochs:
            plt.hist(best_epochs, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.axvline(np.mean(best_epochs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(best_epochs):.1f}')
            plt.xlabel('Best Epoch')
            plt.ylabel('Frequency')
            plt.title('Distribution of Best Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Box Plot Comparison
        ax4 = plt.subplot(3, 3, 4)
        data_to_plot = []
        labels = []
        
        if self.summary_stats['best_f1_scores']:
            data_to_plot.append(self.summary_stats['best_f1_scores'])
            labels.append('Best F1')
        
        if self.summary_stats['final_f1_scores']:
            data_to_plot.append(self.summary_stats['final_f1_scores'])
            labels.append('Final F1')
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('F1 Score')
            plt.title('F1 Score Distribution')
            plt.grid(True, alpha=0.3)
        
        # 5. Training Convergence
        ax5 = plt.subplot(3, 3, 5)
        # Plot learning curves for a few representative subjects
        sample_subjects = list(sorted(self.subject_results.keys()))[:5]  # First 5 subjects
        
        for subject in sample_subjects:
            results = self.subject_results[subject]
            history = results.get('history', {})
            
            if 'test_f1' in history and len(history['test_f1']) > 0:
                epochs = range(1, len(history['test_f1']) + 1)
                plt.plot(epochs, history['test_f1'], 
                        label=f'Subject {subject}', alpha=0.7, linewidth=1)
        
        plt.xlabel('Epoch')
        plt.ylabel('Test F1 Score')
        plt.title('Training Convergence (Sample Subjects)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 6. Per-Class Performance (if available)
        ax6 = plt.subplot(3, 3, 6)
        per_class_f1 = self.summary_stats.get('per_class_f1', {})
        if per_class_f1:
            # Compute mean F1 per class
            class_means = []
            class_ids = []
            
            for class_id in sorted(per_class_f1.keys()):
                if per_class_f1[class_id]:  # Non-empty list
                    class_means.append(np.mean(per_class_f1[class_id]))
                    class_ids.append(class_id)
            
            if class_means:
                # Plot top 20 and bottom 20 classes
                sorted_indices = np.argsort(class_means)
                worst_20 = sorted_indices[:20]
                best_20 = sorted_indices[-20:]
                
                plt.figure(figsize=(12, 6))
                
                # Plot worst 20 classes
                plt.subplot(1, 2, 1)
                worst_classes = [class_ids[i] for i in worst_20]
                worst_f1s = [class_means[i] for i in worst_20]
                plt.bar(range(len(worst_classes)), worst_f1s, color='lightcoral')
                plt.xlabel('Class ID')
                plt.ylabel('Mean F1 Score')
                plt.title('Worst 20 Classes')
                plt.xticks(range(len(worst_classes)), worst_classes, rotation=45)
                
                # Plot best 20 classes
                plt.subplot(1, 2, 2)
                best_classes = [class_ids[i] for i in best_20]
                best_f1s = [class_means[i] for i in best_20]
                plt.bar(range(len(best_classes)), best_f1s, color='lightgreen')
                plt.xlabel('Class ID')
                plt.ylabel('Mean F1 Score')
                plt.title('Best 20 Classes')
                plt.xticks(range(len(best_classes)), best_classes, rotation=45)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'per_class_performance.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Remove empty subplot
        plt.delaxes(ax6)
        
        # 7. Training vs Test Performance
        ax7 = plt.subplot(3, 3, 7)
        train_f1s = []
        test_f1s = []
        
        for subject_num, results in self.subject_results.items():
            history = results.get('history', {})
            if 'train_f1' in history and 'test_f1' in history:
                if len(history['train_f1']) > 0 and len(history['test_f1']) > 0:
                    # Use F1 at best epoch
                    best_epoch = results.get('best_epoch', len(history['test_f1']))
                    if best_epoch <= len(history['train_f1']) and best_epoch <= len(history['test_f1']):
                        train_f1s.append(history['train_f1'][best_epoch - 1])
                        test_f1s.append(history['test_f1'][best_epoch - 1])
        
        if train_f1s and test_f1s:
            plt.scatter(train_f1s, test_f1s, alpha=0.6)
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Perfect correlation line
            plt.xlabel('Training F1 Score')
            plt.ylabel('Test F1 Score')
            plt.title('Training vs Test F1 (at Best Epoch)')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(train_f1s, test_f1s)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 8. Success Rate and Statistics
        ax8 = plt.subplot(3, 3, 8)
        plt.axis('off')
        
        # Create text summary
        summary = self.summary_stats
        success_rate = summary['num_successful'] / summary['num_subjects'] * 100
        
        stats_text = f"""
        === ResNet Leave-One-Out Results ===
        
        Subjects Processed: {summary['num_subjects']}
        Successful Runs: {summary['num_successful']} ({success_rate:.1f}%)
        
        Best F1 Statistics:
        Mean ± Std: {summary['best_f1_mean']:.4f} ± {summary['best_f1_std']:.4f}
        Min - Max: {summary['best_f1_min']:.4f} - {summary['best_f1_max']:.4f}
        Median: {summary['best_f1_median']:.4f}
        
        Training Characteristics:
        Avg Best Epoch: {summary['best_epoch_mean']:.1f} ± {summary['best_epoch_std']:.1f}
        
        Model: ResNet-50 (~50M parameters)
        Loss: Class-Balanced Focal Loss
        Data: 7×7 MRI patches, 351 channels
        """
        
        plt.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
        
        # Save main summary plot
        plt.tight_layout()
        plt.savefig(output_dir / 'resnet_results_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, output_dir: Path):
        """Save detailed results to CSV and JSON"""
        
        # Save detailed DataFrame
        df = self.create_detailed_dataframe()
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        # Save summary statistics
        with open(output_dir / 'summary_statistics.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_stats = {}
            for key, value in self.summary_stats.items():
                if isinstance(value, np.ndarray):
                    serializable_stats[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = value.item()
                else:
                    serializable_stats[key] = value
            
            json.dump(serializable_stats, f, indent=2)
        
        # Save readable summary report
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write("MRI ResNet Leave-One-Out Cross-Validation Results\n")
            f.write("=" * 50 + "\n\n")
            
            summary = self.summary_stats
            success_rate = summary['num_successful'] / summary['num_subjects'] * 100
            
            f.write(f"Total Subjects: {summary['num_subjects']}\n")
            f.write(f"Successful Runs: {summary['num_successful']} ({success_rate:.1f}%)\n\n")
            
            f.write("Best F1 Score Statistics:\n")
            f.write(f"  Mean: {summary['best_f1_mean']:.4f}\n")
            f.write(f"  Std:  {summary['best_f1_std']:.4f}\n")
            f.write(f"  Min:  {summary['best_f1_min']:.4f}\n")
            f.write(f"  Max:  {summary['best_f1_max']:.4f}\n")
            f.write(f"  Median: {summary['best_f1_median']:.4f}\n\n")
            
            if summary['final_f1_scores']:
                f.write("Final F1 Score Statistics:\n")
                f.write(f"  Mean: {summary['final_f1_mean']:.4f}\n")
                f.write(f"  Std:  {summary['final_f1_std']:.4f}\n")
                f.write(f"  Min:  {summary['final_f1_min']:.4f}\n")
                f.write(f"  Max:  {summary['final_f1_max']:.4f}\n")
                f.write(f"  Median: {summary['final_f1_median']:.4f}\n\n")
            
            f.write("Training Characteristics:\n")
            f.write(f"  Average Best Epoch: {summary['best_epoch_mean']:.1f} ± {summary['best_epoch_std']:.1f}\n\n")
            
            # Individual subject results
            f.write("Individual Subject Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Subject':>7} {'Best Epoch':>10} {'Best F1':>10} {'Final F1':>10}\n")
            f.write("-" * 40 + "\n")
            
            for subject in sorted(self.subject_results.keys()):
                results = self.subject_results[subject]
                best_epoch = results.get('best_epoch', 'N/A')
                best_f1 = results.get('best_f1')
                final_f1 = results.get('final_f1')
                
                best_f1_str = f"{best_f1:.4f}" if best_f1 is not None else "N/A"
                final_f1_str = f"{final_f1:.4f}" if final_f1 is not None else "N/A"
                
                f.write(f"{subject:>7} {best_epoch:>10} {best_f1_str:>10} {final_f1_str:>10}\n")
    
    def analyze(self, output_dir: Path):
        """Run complete analysis pipeline"""
        print("\n=== ResNet Leave-One-Out Results Analysis ===")
        
        # Compute summary statistics
        print("Computing summary statistics...")
        summary = self.compute_summary_statistics()
        
        if summary['num_successful'] == 0:
            print("No successful runs found. Nothing to analyze.")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        print("Generating visualizations...")
        self.plot_results_summary(output_dir)
        
        # Save detailed results
        print("Saving detailed results...")
        self.save_detailed_results(output_dir)
        
        # Print summary to console
        print(f"\n=== Summary Statistics ===")
        print(f"Subjects processed: {summary['num_subjects']}")
        print(f"Successful runs: {summary['num_successful']} ({summary['num_successful']/summary['num_subjects']*100:.1f}%)")
        print(f"Best F1 - Mean: {summary['best_f1_mean']:.4f} ± {summary['best_f1_std']:.4f}")
        print(f"Best F1 - Range: {summary['best_f1_min']:.4f} - {summary['best_f1_max']:.4f}")
        print(f"Average best epoch: {summary['best_epoch_mean']:.1f}")
        
        print(f"\nResults saved to: {output_dir}")
        print("Generated files:")
        print("  - resnet_results_summary.png: Main visualization")
        print("  - per_class_performance.png: Per-class analysis")
        print("  - detailed_results.csv: Detailed data")
        print("  - summary_statistics.json: Summary stats")
        print("  - summary_report.txt: Human-readable report")


def main():
    parser = argparse.ArgumentParser(description='Analyze ResNet Leave-One-Out results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing Leave-One-Out results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: results_dir/analysis)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return
    
    # Default output directory
    if args.output_dir is None:
        output_dir = results_dir / 'analysis'
    else:
        output_dir = Path(args.output_dir)
    
    # Run analysis
    analyzer = ResNetResultsAnalyzer(results_dir)
    analyzer.analyze(output_dir)


if __name__ == "__main__":
    main()