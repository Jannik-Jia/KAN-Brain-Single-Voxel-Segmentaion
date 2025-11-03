#!/usr/bin/env python
# coding: utf-8

"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_training_history(history, save_path):
    """
    Plot training history curves

    Parameters:
    -----------
    history : dict
        Training history dictionary
    save_path : Path
        Save path
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # F1 Score
    ax3.plot(epochs, history['train_f1'], 'b-', label='Train F1')
    ax3.plot(epochs, history['test_f1'], 'r-', label='Test F1')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro F1')
    ax3.set_title('Macro F1 Score')
    ax3.legend()
    ax3.grid(True)

    # Test metrics
    ax4.plot(epochs, history['test_loss'], 'g-', label='Test Loss')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, history['test_f1'], 'orange', label='Test F1')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Loss', color='g')
    ax4_twin.set_ylabel('Test F1', color='orange')
    ax4.set_title('Test Performance')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"💾 Training charts saved: {save_path}")
