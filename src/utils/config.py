"""Configuration management utilities."""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create necessary directories
    _create_directories(config)

    # Set device
    config['training']['device'] = get_device(config['training']['device'])

    return config


def _create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories based on configuration."""
    dirs_to_create = [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['checkpointing']['save_dir'],
        config['model_save']['save_dir'],
        config['logging']['log_dir'],
        config['visualization']['figures_dir'],
        config['visualization']['metrics_dir'],
    ]

    if config['logging'].get('use_tensorboard', False):
        dirs_to_create.append(config['logging']['tensorboard_dir'])

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the best available device for PyTorch.

    Args:
        device_preference: Device preference ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        PyTorch device
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_preference)
        print(f"Using specified device: {device}")

    return device


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    # Convert torch.device to string for serialization
    config_copy = config.copy()
    if 'training' in config_copy and isinstance(config_copy['training'].get('device'), torch.device):
        config_copy['training']['device'] = str(config_copy['training']['device'])

    with open(save_path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)

    print(f"Configuration saved to {save_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.

    Args:
        config: Original configuration
        updates: Dictionary with updates (can be nested)

    Returns:
        Updated configuration
    """
    def _update_nested(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                d[k] = _update_nested(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return _update_nested(config.copy(), updates)
