"""Visualization utilities for model results."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Handles visualization of training results and model evaluation."""

    def __init__(self, save_dir: str = "results/figures", dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save figures
            dpi: DPI for saved figures
        """
        self.save_dir = save_dir
        self.dpi = dpi

        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_training_history(
        self,
        history: Dict,
        save_name: Optional[str] = "training_history.png"
    ):
        """
        Plot training and validation loss/accuracy.

        Args:
            history: Training history dictionary
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: Optional[str] = "confusion_matrix.png",
        normalize: bool = False
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: Names of classes
            save_name: Filename to save plot
            normalize: Whether to normalize the confusion matrix
        """
        if class_names is None:
            class_names = ['Negative', 'Positive']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_roc_curve(
        self,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        save_name: Optional[str] = "roc_curve.png"
    ):
        """
        Plot ROC curve.

        Args:
            true_labels: True labels
            probabilities: Predicted probabilities for positive class
            save_name: Filename to save plot
        """
        fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        plt.close()

    def plot_metrics_comparison(
        self,
        metrics: Dict,
        save_name: Optional[str] = "metrics_comparison.png"
    ):
        """
        Plot comparison of different metrics.

        Args:
            metrics: Dictionary of metric names and values
            save_name: Filename to save plot
        """
        # Filter out confusion matrix
        plot_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}

        if not plot_metrics:
            logger.warning("No metrics to plot")
            return

        metric_names = list(plot_metrics.keys())
        metric_values = list(plot_metrics.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color='steelblue', alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=10
            )

        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")

        plt.close()

    def plot_learning_rate_schedule(
        self,
        history: Dict,
        save_name: Optional[str] = "learning_rate_schedule.png"
    ):
        """
        Plot learning rate schedule.

        Args:
            history: Training history with learning rates
            save_name: Filename to save plot
        """
        if 'learning_rate' not in history:
            logger.warning("No learning rate information in history")
            return

        epochs = range(1, len(history['learning_rate']) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['learning_rate'], 'b-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Learning rate schedule saved to {save_path}")

        plt.close()

    def plot_class_distribution(
        self,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: Optional[str] = "class_distribution.png"
    ):
        """
        Plot class distribution.

        Args:
            labels: Array of labels
            class_names: Names of classes
            save_name: Filename to save plot
        """
        if class_names is None:
            class_names = ['Negative', 'Positive']

        counts = np.bincount(labels.astype(int))

        plt.figure(figsize=(8, 6))
        bars = plt.bar(class_names, counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=12
            )

        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Class distribution saved to {save_path}")

        plt.close()

    def create_evaluation_report(
        self,
        history: Dict,
        metrics: Dict,
        true_labels: np.ndarray,
        probabilities: np.ndarray
    ):
        """
        Create comprehensive evaluation report with all plots.

        Args:
            history: Training history
            metrics: Evaluation metrics
            true_labels: True labels
            probabilities: Predicted probabilities
        """
        logger.info("Creating comprehensive evaluation report...")

        # Training history
        self.plot_training_history(history, "training_history.png")

        # Confusion matrix
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(metrics['confusion_matrix'], save_name="confusion_matrix.png")
            self.plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_name="confusion_matrix_normalized.png",
                normalize=True
            )

        # ROC curve
        self.plot_roc_curve(true_labels, probabilities[:, 1], "roc_curve.png")

        # Metrics comparison
        self.plot_metrics_comparison(metrics, "metrics_comparison.png")

        # Learning rate schedule
        if 'learning_rate' in history:
            self.plot_learning_rate_schedule(history, "learning_rate_schedule.png")

        # Class distribution
        self.plot_class_distribution(true_labels, save_name="class_distribution.png")

        logger.info(f"Evaluation report saved to {self.save_dir}")


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Convenience function to plot training history.

    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    visualizer = Visualizer()
    save_name = os.path.basename(save_path) if save_path else "training_history.png"
    visualizer.plot_training_history(history, save_name)
