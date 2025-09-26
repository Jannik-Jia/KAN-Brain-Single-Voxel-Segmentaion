#!/usr/bin/env python3
"""
Analysis script for cross-validation results
Generates plots and detailed reports from training outputs
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def load_results(results_path):
    """Load cross-validation results from JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)

def load_history(history_dir, fold):
    """Load training history for a specific fold"""
    history_path = os.path.join(history_dir, f'history_fold{fold}.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def plot_cross_validation_summary(results, output_dir):
    """Create summary plots for cross-validation results"""
    valid_results = [r for r in results if 'test_accuracy' in r]

    if not valid_results:
        print("No valid results to plot")
        return

    folds = [r['fold'] for r in valid_results]
    accuracies = [r['test_accuracy'] for r in valid_results]
    f1_scores = [r['test_f1'] for r in valid_results]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Accuracy per fold
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(folds, accuracies, color='steelblue', alpha=0.8)
    ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.4f}')
    ax1.set_xlabel('Fold (Subject ID)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy per Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: F1 Score per fold
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(folds, f1_scores, color='forestgreen', alpha=0.8)
    ax2.axhline(y=np.mean(f1_scores), color='r', linestyle='--',
                label=f'Mean: {np.mean(f1_scores):.4f}')
    ax2.set_xlabel('Fold (Subject ID)')
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_title('Macro F1 Score per Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Accuracy distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.boxplot(accuracies, vert=True)
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Accuracy Distribution')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: F1 distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.boxplot(f1_scores, vert=True)
    ax4.set_ylabel('Macro F1 Score')
    ax4.set_title('F1 Score Distribution')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Accuracy vs F1 scatter
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(accuracies, f1_scores, alpha=0.6, s=50)
    for i, fold in enumerate(folds):
        ax5.annotate(fold, (accuracies[i], f1_scores[i]), fontsize=8, alpha=0.7)
    ax5.set_xlabel('Test Accuracy')
    ax5.set_ylabel('Macro F1 Score')
    ax5.set_title('Accuracy vs F1 Score')
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Summary statistics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""Cross-Validation Summary Statistics

Accuracy:
  Mean: {np.mean(accuracies):.4f}
  Std:  {np.std(accuracies):.4f}
  Min:  {np.min(accuracies):.4f} (Fold {folds[np.argmin(accuracies)]})
  Max:  {np.max(accuracies):.4f} (Fold {folds[np.argmax(accuracies)]})

Macro F1 Score:
  Mean: {np.mean(f1_scores):.4f}
  Std:  {np.std(f1_scores):.4f}
  Min:  {np.min(f1_scores):.4f} (Fold {folds[np.argmin(f1_scores)]})
  Max:  {np.max(f1_scores):.4f} (Fold {folds[np.argmax(f1_scores)]})

Total Folds: {len(valid_results)}/{len(results)}
Failed Folds: {len(results) - len(valid_results)}"""

    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.suptitle('38-Fold Cross Validation Results', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, 'cross_validation_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {plot_path}")
    plt.show()

def plot_training_curves(history_dir, fold, output_dir):
    """Plot training curves for a specific fold"""
    history = load_history(history_dir, fold)

    if not history:
        print(f"No history found for fold {fold}")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].plot(epochs, history['test_acc'], 'g-', label='Test Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 curves
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[1, 0].plot(epochs, history['test_f1'], 'g-', label='Test F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].set_title('Macro F1 Score over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Final metrics text
    axes[1, 1].axis('off')
    if 'final_metrics' in history:
        fm = history['final_metrics']
        metrics_text = f"""Final Test Metrics (Fold {fold})

Gross Accuracy: {fm['gross_accuracy']:.4f}

Macro Average:
  Precision: {fm['macro_precision']:.4f}
  Recall:    {fm['macro_recall']:.4f}
  F1:        {fm['macro_f1']:.4f}

Micro F1:     {fm['micro_f1']:.4f}
Weighted F1:  {fm['weighted_f1']:.4f}"""

        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11,
                       family='monospace', verticalalignment='center')

    plt.suptitle(f'Training History - Fold {fold}', fontsize=16)
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, f'training_history_fold{fold}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    plt.show()

def generate_report(results, output_dir):
    """Generate detailed text report"""
    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("38-FOLD CROSS VALIDATION ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        valid_results = [r for r in results if 'test_accuracy' in r]
        failed_results = [r for r in results if 'error' in r]

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 30 + "\n")

        if valid_results:
            accuracies = [r['test_accuracy'] for r in valid_results]
            f1_scores = [r['test_f1'] for r in valid_results]
            val_accuracies = [r['val_accuracy'] for r in valid_results]
            val_f1_scores = [r['val_f1'] for r in valid_results]

            f.write(f"Total Folds Completed: {len(valid_results)}/{len(results)}\n")
            f.write(f"Failed Folds: {len(failed_results)}\n\n")

            f.write("Test Set Performance:\n")
            f.write(f"  Accuracy - Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
            f.write(f"           - Min:  {np.min(accuracies):.4f}\n")
            f.write(f"           - Max:  {np.max(accuracies):.4f}\n")
            f.write(f"  Macro F1 - Mean: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
            f.write(f"           - Min:  {np.min(f1_scores):.4f}\n")
            f.write(f"           - Max:  {np.max(f1_scores):.4f}\n\n")

            f.write("Validation Set Performance:\n")
            f.write(f"  Accuracy - Mean: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}\n")
            f.write(f"  Macro F1 - Mean: {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}\n\n")

        # Per-fold results
        f.write("PER-FOLD RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Fold':<6} {'Subject ID':<20} {'Test Acc':<10} {'Test F1':<10} {'Status':<10}\n")
        f.write("-" * 60 + "\n")

        for r in results:
            if 'test_accuracy' in r:
                f.write(f"{r['fold']:<6} {r['subject_id']:<20} "
                       f"{r['test_accuracy']:<10.4f} {r['test_f1']:<10.4f} SUCCESS\n")
            else:
                f.write(f"{r['fold']:<6} {'N/A':<20} {'N/A':<10} {'N/A':<10} FAILED\n")

        # Failed folds details
        if failed_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("FAILED FOLDS DETAILS\n")
            f.write("-" * 30 + "\n")
            for r in failed_results:
                f.write(f"Fold {r['fold']}: {r.get('error', 'Unknown error')}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Report generated at: {os.path.abspath(report_path)}\n")

    print(f"Analysis report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze cross-validation results')

    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing training outputs')
    parser.add_argument('--plot-fold', type=int, default=None,
                       help='Plot training curves for specific fold')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip summary plots')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots and reports')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    results_path = os.path.join(args.results_dir, 'cross_validation_results.json')

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Make sure you've completed training first.")
        return

    results = load_results(results_path)
    print(f"Loaded {len(results)} fold results")

    # Generate text report
    generate_report(results, args.output_dir)

    # Create summary plots
    if not args.no_summary:
        plot_cross_validation_summary(results, args.output_dir)

    # Plot specific fold if requested
    if args.plot_fold:
        history_dir = os.path.join(args.results_dir, 'history')
        if os.path.exists(history_dir):
            plot_training_curves(history_dir, args.plot_fold, args.output_dir)
        else:
            print(f"History directory not found: {history_dir}")

    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()