"""Training callbacks for model checkpointing and early stopping."""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1

    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.monitor_op(current_score, self.best_score + self.min_delta):
            # Improvement
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                logger.info(f"Validation metric improved to {current_score:.4f}")
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best: {self.best_score:.4f}, current: {current_score:.4f})"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
                return True

            return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training."""

    def __init__(
        self,
        save_dir: str = 'models/checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 5,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.

        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to save only the best model
            save_frequency: Save every N epochs (if save_best_only is False)
            verbose: Whether to print messages
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.verbose = verbose

        self.best_score = None

        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf

        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        current_score: float,
        metrics: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            current_score: Current metric value
            metrics: Additional metrics to save

        Returns:
            Path to saved checkpoint or None
        """
        should_save = False
        checkpoint_path = None

        if self.save_best_only:
            # Save only if best
            if self.monitor_op(current_score, self.best_score):
                self.best_score = current_score
                should_save = True
                checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')

                if self.verbose:
                    logger.info(
                        f"Saving best model at epoch {epoch} "
                        f"({self.monitor}={current_score:.4f})"
                    )
        else:
            # Save at regular intervals
            if epoch % self.save_frequency == 0:
                should_save = True
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')

        if should_save and checkpoint_path:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                self.monitor: current_score,
            }

            if metrics:
                checkpoint['metrics'] = metrics

            torch.save(checkpoint, checkpoint_path)

            if self.verbose and not self.save_best_only:
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            return checkpoint_path

        return None


class LearningRateSchedulerCallback:
    """Callback for learning rate scheduling."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
            verbose: Whether to print messages
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def __call__(self, metrics: Dict):
        """
        Step the learning rate scheduler.

        Args:
            metrics: Dictionary of metrics
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.monitor is None:
                raise ValueError("Monitor metric must be specified for ReduceLROnPlateau")

            if self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])

                if self.verbose:
                    current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                    logger.info(f"Learning rate: {current_lr:.6f}")
        else:
            self.scheduler.step()

            if self.verbose:
                current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")


class GradientClipping:
    """Gradient clipping callback."""

    def __init__(self, max_norm: float = 1.0, verbose: bool = False):
        """
        Initialize gradient clipping.

        Args:
            max_norm: Maximum gradient norm
            verbose: Whether to print gradient norms
        """
        self.max_norm = max_norm
        self.verbose = verbose

    def __call__(self, model: torch.nn.Module) -> float:
        """
        Clip gradients.

        Args:
            model: Model with gradients

        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm
        )

        if self.verbose:
            logger.debug(f"Gradient norm: {total_norm:.4f}")

        return total_norm.item()
