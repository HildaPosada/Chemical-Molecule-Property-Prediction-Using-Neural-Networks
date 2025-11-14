"""Model evaluation utilities."""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Handles model evaluation and metric calculation."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            device: Device to run evaluation on
            threshold: Classification threshold for binary predictions
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.threshold = threshold

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)

                # Get model outputs
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)

                # Get predictions
                predictions = torch.argmax(probabilities, dim=1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.numpy())

        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            np.array(all_labels)
        )

    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: DataLoader for the dataset
            metrics: List of metrics to calculate

        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        logger.info(f"Evaluating model on {len(data_loader.dataset)} samples...")

        # Get predictions
        predictions, probabilities, true_labels = self.predict(data_loader)

        results = {}

        # Calculate metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(true_labels, predictions)

        if 'precision' in metrics:
            results['precision'] = precision_score(
                true_labels, predictions, average='binary', zero_division=0
            )

        if 'recall' in metrics:
            results['recall'] = recall_score(
                true_labels, predictions, average='binary', zero_division=0
            )

        if 'f1' in metrics:
            results['f1'] = f1_score(
                true_labels, predictions, average='binary', zero_division=0
            )

        if 'roc_auc' in metrics:
            # Use probability of positive class
            try:
                results['roc_auc'] = roc_auc_score(true_labels, probabilities[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                results['roc_auc'] = 0.0

        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(true_labels, predictions)

        # Log results
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric}: {value:.4f}")

        return results

    def get_classification_report(
        self,
        data_loader: DataLoader,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            data_loader: DataLoader for the dataset
            target_names: Names of target classes

        Returns:
            Classification report string
        """
        predictions, _, true_labels = self.predict(data_loader)

        if target_names is None:
            target_names = ['Negative', 'Positive']

        report = classification_report(
            true_labels,
            predictions,
            target_names=target_names
        )

        return report

    def get_confusion_matrix(
        self,
        data_loader: DataLoader
    ) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Confusion matrix
        """
        predictions, _, true_labels = self.predict(data_loader)
        cm = confusion_matrix(true_labels, predictions)
        return cm

    def calculate_loss(
        self,
        data_loader: DataLoader,
        criterion: torch.nn.Module
    ) -> float:
        """
        Calculate average loss on dataset.

        Args:
            data_loader: DataLoader for the dataset
            criterion: Loss function

        Returns:
            Average loss
        """
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def predict_single(
        self,
        features: torch.Tensor
    ) -> Tuple[int, np.ndarray]:
        """
        Predict for a single sample.

        Args:
            features: Feature tensor

        Returns:
            Tuple of (prediction, probabilities)
        """
        features = features.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

        return prediction.item(), probabilities.cpu().numpy()[0]


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to evaluate a model.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        config: Optional configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = Evaluator(model, device)

    # Get metrics from config or use defaults
    if config and 'evaluation' in config:
        metrics = config['evaluation'].get('metrics', None)
    else:
        metrics = None

    results = evaluator.evaluate(test_loader, metrics)

    # Print classification report
    logger.info("\nClassification Report:")
    report = evaluator.get_classification_report(test_loader)
    logger.info(f"\n{report}")

    return results


def get_misclassified_samples(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    max_samples: int = 10
) -> List[Dict]:
    """
    Get misclassified samples for analysis.

    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device to run on
        max_samples: Maximum number of samples to return

    Returns:
        List of dictionaries with misclassified sample info
    """
    device = device or torch.device('cpu')
    model.to(device)
    model.eval()

    misclassified = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(data_loader):
            features = features.to(device)
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            # Find misclassified samples in batch
            mask = predictions != labels.to(device)
            misclassified_indices = torch.where(mask)[0]

            for idx in misclassified_indices:
                if len(misclassified) >= max_samples:
                    return misclassified

                misclassified.append({
                    'batch_idx': batch_idx,
                    'sample_idx': idx.item(),
                    'true_label': labels[idx].item(),
                    'predicted_label': predictions[idx].item(),
                    'probabilities': probabilities[idx].cpu().numpy(),
                    'features': features[idx].cpu().numpy()
                })

    logger.info(f"Found {len(misclassified)} misclassified samples")
    return misclassified
