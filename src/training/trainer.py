"""Training loop for molecular property prediction models."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateSchedulerCallback, GradientClipping
from ..utils.logger import get_logger, TensorBoardLogger

logger = get_logger(__name__)


class Trainer:
    """Handles model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cpu')

        # Move model to device
        self.model.to(self.device)

        # Training config
        self.train_config = config['training']
        self.num_epochs = self.train_config['num_epochs']

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss function
        self.criterion = self._create_criterion()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        self.lr_scheduler_callback = None
        if self.scheduler:
            self.lr_scheduler_callback = LearningRateSchedulerCallback(
                self.scheduler,
                monitor='val_loss'
            )

        # Setup callbacks
        self.early_stopping = None
        if self.train_config.get('early_stopping', False):
            self.early_stopping = EarlyStopping(
                patience=self.train_config.get('patience', 10),
                min_delta=self.train_config.get('min_delta', 0.001),
                mode='min',
                verbose=True
            )

        self.checkpoint_callback = ModelCheckpoint(
            save_dir=config['checkpointing']['save_dir'],
            monitor=config['checkpointing']['monitor'],
            mode=config['checkpointing']['mode'],
            save_best_only=config['checkpointing']['save_best_only'],
            save_frequency=config['checkpointing']['save_frequency'],
            verbose=True
        )

        # Gradient clipping
        self.gradient_clipper = None
        if self.train_config.get('gradient_clipping', False):
            self.gradient_clipper = GradientClipping(
                max_norm=self.train_config.get('max_grad_norm', 1.0)
            )

        # TensorBoard logging
        self.tensorboard_logger = None
        if config['logging'].get('use_tensorboard', False):
            self.tensorboard_logger = TensorBoardLogger(
                log_dir=config['logging']['tensorboard_dir']
            )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        logger.info(f"Trainer initialized with device: {self.device}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        optimizer_name = self.train_config['optimizer'].lower()
        lr = float(self.train_config['learning_rate'])
        weight_decay = float(self.train_config.get('weight_decay', 0.0))

        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        logger.info(f"Created {optimizer_name.upper()} optimizer with lr={lr}")
        return optimizer

    def _create_criterion(self) -> nn.Module:
        """Create loss function from config."""
        # Get class weights if using weighted loss
        if self.train_config.get('use_class_weights', False):
            class_weights = self.train_loader.dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
            logger.info(f"Using class weights: {class_weights}")
        else:
            class_weights = None

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        if not self.train_config.get('use_scheduler', False):
            return None

        scheduler_type = self.train_config.get(
            'scheduler_type', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.train_config.get('scheduler_factor', 0.5),
                patience=self.train_config.get('scheduler_patience', 10),
                verbose=True
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs
            )
        else:
            scheduler = None

        if scheduler:
            logger.info(f"Created {scheduler_type} learning rate scheduler")

        return scheduler

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clipper:
                self.gradient_clipper(self.model)

            # Update weights
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Calculate loss
                loss = self.criterion(outputs, labels)

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self) -> Dict:
        """
        Train model for multiple epochs.

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Logging
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"Time: {epoch_time:.2f}s"
            )

            # TensorBoard logging
            if self.tensorboard_logger:
                self.tensorboard_logger.log_scalar(
                    'Loss/train', train_loss, epoch)
                self.tensorboard_logger.log_scalar('Loss/val', val_loss, epoch)
                self.tensorboard_logger.log_scalar(
                    'Accuracy/train', train_acc, epoch)
                self.tensorboard_logger.log_scalar(
                    'Accuracy/val', val_acc, epoch)
                self.tensorboard_logger.log_scalar(
                    'Learning_Rate', current_lr, epoch)

            # Learning rate scheduling
            if self.lr_scheduler_callback:
                self.lr_scheduler_callback({'val_loss': val_loss})

            # Model checkpointing
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            self.checkpoint_callback(
                epoch, self.model, self.optimizer, val_loss, metrics)

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        # Close TensorBoard logger
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

        return self.history

    def save_model(self, save_path: str):
        """
        Save final model.

        Args:
            save_path: Path to save model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, save_path)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load model from checkpoint.

        Args:
            load_path: Path to load model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"Model loaded from {load_path}")
