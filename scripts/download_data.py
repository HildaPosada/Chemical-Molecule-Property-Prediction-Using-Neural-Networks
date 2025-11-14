"""Script to download and prepare molecular dataset."""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import MoleculeDataLoader, MoleculePreprocessor, download_bbbp_dataset
from src.utils import load_config, setup_logger

logger = setup_logger()


def main():
    """Download and prepare molecular dataset."""
    parser = argparse.ArgumentParser(description='Download and prepare molecular dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='BBBP',
        help='Dataset name (currently only BBBP is supported)'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Download dataset
    logger.info(f"Downloading {args.dataset} dataset...")
    data_loader = MoleculeDataLoader(config)

    try:
        # Load dataset
        df = data_loader.load_bbbp_dataset()
        logger.info(f"Dataset loaded successfully: {len(df)} molecules")

        # Print dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total molecules: {len(df)}")
        logger.info(f"Positive samples: {df['p_np'].sum()}")
        logger.info(f"Negative samples: {(1 - df['p_np']).sum()}")
        logger.info(f"Positive ratio: {df['p_np'].mean():.2%}")

        # Split data
        logger.info("\nSplitting data...")
        train_df, val_df, test_df = data_loader.split_data(df)

        # Save to SQLite
        logger.info("\nSaving to SQLite database...")
        db_path = data_loader.save_to_sqlite(train_df, val_df, test_df)

        logger.info(f"\nData preparation complete!")
        logger.info(f"Database saved to: {db_path}")
        logger.info("\nNext steps:")
        logger.info("1. Run preprocessing: python scripts/preprocess_data.py")
        logger.info("2. Train model: python scripts/train.py")

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise


if __name__ == '__main__':
    main()
