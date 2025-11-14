"""Script to train molecular property prediction model."""

import os
import sys
import argparse
import torch
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import MoleculeDataLoader, MoleculePreprocessor, MoleculeDataset, create_dataloaders
from src.models import create_model
from src.training import Trainer
from src.utils import load_config, setup_logger

logger = setup_logger()


def main():
    """Train molecular property prediction model."""
    parser = argparse.ArgumentParser(description='Train molecular property prediction model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Set random seeds for reproducibility
    seed = config['training']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize data loader
    logger.info("Loading data...")
    data_loader = MoleculeDataLoader(config)

    try:
        # Load data from SQLite
        train_df, val_df, test_df = data_loader.load_from_sqlite()
    except FileNotFoundError:
        logger.info("Database not found. Loading and preparing data...")
        df = data_loader.load_bbbp_dataset()
        train_df, val_df, test_df = data_loader.split_data(df)
        data_loader.save_to_sqlite(train_df, val_df, test_df)

    # Initialize preprocessor
    logger.info("Preprocessing molecules...")
    preprocessor = MoleculePreprocessor(config)

    # Process data
    X_train, y_train, _ = preprocessor.process_dataframe(train_df, fit_scaler=True)
    X_val, y_val, _ = preprocessor.process_dataframe(val_df, fit_scaler=False)
    X_test, y_test, _ = preprocessor.process_dataframe(test_df, fit_scaler=False)

    # Save scaler
    scaler_path = os.path.join(config['data']['processed_dir'], 'scaler.pkl')
    preprocessor.save_scaler(scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Create PyTorch datasets
    logger.info("Creating datasets...")
    train_dataset = MoleculeDataset(X_train, y_train)
    val_dataset = MoleculeDataset(X_val, y_val)
    test_dataset = MoleculeDataset(X_test, y_test)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )

    # Create model
    logger.info("Creating model...")
    input_size = preprocessor.get_feature_dim()
    model = create_model(config, input_size)

    # Print model summary
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: {param_counts['trainable']:,} trainable, "
                f"{param_counts['total']:,} total")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['training']['device']
    )

    # Train model
    logger.info("Starting training...")
    history = trainer.train()

    # Save final model
    if config['model_save']['save_final']:
        save_dir = config['model_save']['save_dir']
        save_path = os.path.join(save_dir, 'final_model.pth')
        trainer.save_model(save_path)

    # Save training history
    history_path = os.path.join(config['visualization']['metrics_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
