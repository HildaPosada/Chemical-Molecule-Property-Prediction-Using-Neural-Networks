"""PyTorch Dataset classes for molecular data."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular features."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.

        Args:
            features: Feature matrix (N x D)
            labels: Label array (N,)
            transform: Optional transform to apply
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels.astype(np.int64))
        self.transform = transform

        logger.info(f"Created dataset with {len(self)} samples")
        logger.info(f"Feature shape: {self.features.shape}")
        logger.info(f"Label distribution: {np.bincount(labels.astype(int))}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, label)
        """
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        labels_np = self.labels.numpy()
        class_counts = np.bincount(labels_np)
        total_samples = len(labels_np)

        # Calculate weights as inverse frequency
        class_weights = total_samples / (len(class_counts) * class_counts)
        weights_tensor = torch.FloatTensor(class_weights)

        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Class weights: {class_weights}")

        return weights_tensor

    def get_num_classes(self) -> int:
        """
        Get number of unique classes.

        Returns:
            Number of classes
        """
        return len(torch.unique(self.labels))

    def get_feature_dim(self) -> int:
        """
        Get feature dimensionality.

        Returns:
            Feature dimension
        """
        return self.features.shape[1]


def create_dataloaders(
    train_dataset: MoleculeDataset,
    val_dataset: MoleculeDataset,
    test_dataset: MoleculeDataset,
    config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for all splits.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    shuffle = config['training']['shuffle']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Created DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches of size {batch_size}")
    logger.info(f"  Val: {len(val_loader)} batches of size {batch_size}")
    logger.info(f"  Test: {len(test_loader)} batches of size {batch_size}")

    return train_loader, val_loader, test_loader


def get_sample_weights(dataset: MoleculeDataset) -> torch.Tensor:
    """
    Calculate sample weights for weighted sampling.

    Args:
        dataset: MoleculeDataset instance

    Returns:
        Tensor of sample weights
    """
    class_weights = dataset.get_class_weights()
    labels = dataset.labels.numpy()

    # Assign weight to each sample based on its class
    sample_weights = torch.FloatTensor([class_weights[label] for label in labels])

    return sample_weights
