"""Data loading utilities for molecular datasets."""

import os
import requests
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MoleculeDataLoader:
    """Handles loading and splitting of molecular datasets."""

    def __init__(self, config: Dict):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.raw_dir = config['data']['raw_dir']
        self.processed_dir = config['data']['processed_dir']
        self.random_seed = config['data']['random_seed']

        # Create directories
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def load_bbbp_dataset(self) -> pd.DataFrame:
        """
        Load BBBP (Blood-Brain Barrier Penetration) dataset.

        Returns:
            DataFrame with molecular data
        """
        csv_path = os.path.join(self.raw_dir, "BBBP.csv")

        if not os.path.exists(csv_path):
            logger.info("BBBP dataset not found. Downloading...")
            download_bbbp_dataset(self.raw_dir)

        logger.info(f"Loading BBBP dataset from {csv_path}")
        df = pd.read_csv(csv_path)

        logger.info(f"Loaded {len(df)} molecules")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Basic data validation
        required_columns = ['smiles', 'p_np']  # SMILES and target label
        if not all(col in df.columns for col in required_columns):
            # Try alternative column names
            if 'name' in df.columns and 'p_np' in df.columns:
                df = df.rename(columns={'name': 'smiles'})
            else:
                raise ValueError(f"Dataset must contain columns: {required_columns}")

        # Remove invalid SMILES if configured
        if self.config['data'].get('remove_invalid_smiles', True):
            initial_count = len(df)
            df = df.dropna(subset=['smiles'])
            df = df[df['smiles'].str.len() > 0]
            removed = initial_count - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} invalid SMILES entries")

        # Limit dataset size if specified
        max_molecules = self.config['data'].get('max_molecules')
        if max_molecules is not None and max_molecules < len(df):
            df = df.sample(n=max_molecules, random_state=self.random_seed)
            logger.info(f"Limited dataset to {max_molecules} molecules")

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            stratify: Whether to stratify split by target labels

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = self.config['data']['train_split']
        val_ratio = self.config['data']['val_split']
        test_ratio = self.config['data']['test_split']

        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")

        # Get target column
        target_col = 'p_np'
        stratify_col = df[target_col] if stratify else None

        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=self.random_seed,
            stratify=stratify_col
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        stratify_col_train_val = train_val_df[target_col] if stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=stratify_col_train_val
        )

        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train class distribution: {train_df[target_col].value_counts().to_dict()}")
        logger.info(f"Val class distribution: {val_df[target_col].value_counts().to_dict()}")
        logger.info(f"Test class distribution: {test_df[target_col].value_counts().to_dict()}")

        return train_df, val_df, test_df

    def save_to_sqlite(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        db_name: str = "molecules.db"
    ) -> str:
        """
        Save datasets to SQLite database.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            db_name: Database filename

        Returns:
            Path to database file
        """
        db_path = os.path.join(self.processed_dir, db_name)

        logger.info(f"Saving data to SQLite database: {db_path}")

        conn = sqlite3.connect(db_path)

        # Save each split to a separate table
        train_df.to_sql('train', conn, if_exists='replace', index=False)
        val_df.to_sql('validation', conn, if_exists='replace', index=False)
        test_df.to_sql('test', conn, if_exists='replace', index=False)

        # Create metadata table
        metadata = pd.DataFrame({
            'key': ['dataset', 'created_at', 'train_size', 'val_size', 'test_size'],
            'value': [
                'BBBP',
                pd.Timestamp.now().isoformat(),
                str(len(train_df)),
                str(len(val_df)),
                str(len(test_df))
            ]
        })
        metadata.to_sql('metadata', conn, if_exists='replace', index=False)

        conn.close()
        logger.info(f"Data saved successfully to {db_path}")

        return db_path

    def load_from_sqlite(
        self,
        db_name: str = "molecules.db"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load datasets from SQLite database.

        Args:
            db_name: Database filename

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        db_path = os.path.join(self.processed_dir, db_name)

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        logger.info(f"Loading data from SQLite database: {db_path}")

        conn = sqlite3.connect(db_path)

        train_df = pd.read_sql('SELECT * FROM train', conn)
        val_df = pd.read_sql('SELECT * FROM validation', conn)
        test_df = pd.read_sql('SELECT * FROM test', conn)

        conn.close()

        logger.info(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df


def download_bbbp_dataset(save_dir: str) -> str:
    """
    Download BBBP dataset from DeepChem.

    Args:
        save_dir: Directory to save the dataset

    Returns:
        Path to downloaded file
    """
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    save_path = os.path.join(save_dir, "BBBP.csv")

    logger.info(f"Downloading BBBP dataset from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Downloaded successfully to {save_path}")
        return save_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate basic statistics for the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_molecules': len(df),
        'num_positive': int(df['p_np'].sum()),
        'num_negative': int((1 - df['p_np']).sum()),
        'positive_ratio': float(df['p_np'].mean()),
        'smiles_avg_length': float(df['smiles'].str.len().mean()),
        'smiles_max_length': int(df['smiles'].str.len().max()),
        'smiles_min_length': int(df['smiles'].str.len().min()),
    }

    return stats
