"""Neural network architecture for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MoleculeNet(nn.Module):
    """
    Feed-forward neural network for molecular property prediction.

    Architecture:
        Input -> Dense(hidden_sizes[0]) -> ReLU -> Dropout ->
        Dense(hidden_sizes[1]) -> ReLU -> Dropout ->
        ... -> Dense(num_classes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 2,
        dropout: float = 0.3,
        activation: str = 'relu',
        batch_norm: bool = True
    ):
        """
        Initialize MoleculeNet.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
            batch_norm: Whether to use batch normalization
        """
        super(MoleculeNet, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.use_batch_norm = batch_norm

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))

            # Dropout
            self.dropouts.append(nn.Dropout(dropout))

            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Initialize weights
        self._initialize_weights()

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Created MoleculeNet:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden layers: {hidden_sizes}")
        logger.info(f"  Output size: {num_classes}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            # Activation
            x = self.activation(x)

            # Dropout
            x = self.dropouts[i](x)

        # Output layer
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probabilities tensor
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        probs = self.predict_proba(x)
        predictions = torch.argmax(probs, dim=1)
        return predictions

    def get_embeddings(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get intermediate layer embeddings.

        Args:
            x: Input tensor
            layer_idx: Index of layer to extract embeddings from (-1 for last hidden layer)

        Returns:
            Embedding tensor
        """
        self.eval()
        with torch.no_grad():
            # Forward through layers
            for i, layer in enumerate(self.layers):
                x = layer(x)

                if self.use_batch_norm:
                    x = self.batch_norms[i](x)

                x = self.activation(x)

                # Return embeddings from specified layer
                if i == layer_idx or (layer_idx == -1 and i == len(self.layers) - 1):
                    return x

                x = self.dropouts[i](x)

        return x

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': non_trainable
        }


def create_model(config: Dict, input_size: int) -> MoleculeNet:
    """
    Create MoleculeNet model from configuration.

    Args:
        config: Configuration dictionary
        input_size: Number of input features

    Returns:
        MoleculeNet instance
    """
    model_config = config['model']

    model = MoleculeNet(
        input_size=input_size,
        hidden_sizes=model_config['hidden_sizes'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout'],
        activation=model_config.get('activation', 'relu'),
        batch_norm=model_config.get('batch_norm', True)
    )

    return model


class MoleculeNetWithAttention(nn.Module):
    """
    MoleculeNet with self-attention mechanism.

    This is an advanced variant that can be used for better performance.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 2,
        dropout: float = 0.3,
        num_attention_heads: int = 4
    ):
        """
        Initialize MoleculeNet with attention.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability
            num_attention_heads: Number of attention heads
        """
        super(MoleculeNetWithAttention, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_sizes[0])

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[0],
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward layers
        self.layers = nn.ModuleList()
        prev_size = hidden_sizes[0]

        for hidden_size in hidden_sizes[1:]:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.activation = nn.ReLU()

        logger.info(f"Created MoleculeNetWithAttention with {num_attention_heads} attention heads")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_projection(x)
        x = self.activation(x)

        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(1)  # (batch_size, hidden_size)

        # Residual connection
        x = x.squeeze(1) + attn_output
        x = self.dropout(x)

        # Feed-forward layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Output
        x = self.output_layer(x)

        return x
