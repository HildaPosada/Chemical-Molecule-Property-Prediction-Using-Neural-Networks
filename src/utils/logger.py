"""Logging utilities."""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "molecule_prediction",
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file created: {log_file}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get existing logger or create a new one.

    Args:
        name: Logger name (None for root logger)

    Returns:
        Logger instance
    """
    if name is None:
        name = "molecule_prediction"

    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str = "runs", experiment_name: Optional[str] = None):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of experiment (will use timestamp if None)
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            if experiment_name is None:
                experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            log_path = os.path.join(log_dir, experiment_name)
            self.writer = SummaryWriter(log_path)
            self.enabled = True
            print(f"TensorBoard logging enabled: {log_path}")
            print(f"Run: tensorboard --logdir={log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag: str, figure, step: int):
        """Log matplotlib figure."""
        if self.enabled:
            self.writer.add_figure(tag, figure, step)

    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


class WandbLogger:
    """Weights & Biases logging wrapper."""

    def __init__(
        self,
        project: str = "molecule-prediction",
        entity: Optional[str] = None,
        config: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize Weights & Biases logger.

        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            config: Configuration dictionary to log
            name: Run name
        """
        try:
            import wandb

            wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=name,
            )
            self.wandb = wandb
            self.enabled = True
            print(f"W&B logging enabled: {project}")
        except ImportError:
            print("W&B not available. Install with: pip install wandb")
            self.wandb = None
            self.enabled = False

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters."""
        if self.enabled:
            self.wandb.watch(model, log_freq=log_freq)

    def finish(self):
        """Finish the run."""
        if self.enabled:
            self.wandb.finish()
