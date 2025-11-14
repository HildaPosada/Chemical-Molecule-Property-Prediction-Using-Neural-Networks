"""Molecular feature extraction and preprocessing."""

import numpy as np
import pandas as pd
import pickle
from typing import List, Optional, Tuple, Dict
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MoleculePreprocessor:
    """Handles molecular feature extraction and preprocessing."""

    def __init__(self, config: Dict):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required but not installed")

        self.config = config
        self.feature_config = config['features']

        # Feature extraction settings
        self.use_morgan = self.feature_config.get('use_morgan_fingerprints', True)
        self.morgan_radius = self.feature_config.get('morgan_radius', 2)
        self.morgan_bits = self.feature_config.get('morgan_bits', 1024)

        self.use_descriptors = self.feature_config.get('use_descriptors', True)
        self.descriptor_list = self.feature_config.get('descriptor_list', [])

        # Scaling
        self.scale_features = self.feature_config.get('scale_features', True)
        scaler_type = self.feature_config.get('scaler_type', 'standard')

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.feature_names = []
        self.is_fitted = False

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule object.

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logger.warning(f"Failed to parse SMILES '{smiles}': {e}")
            return None

    def calculate_morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Calculate Morgan fingerprint for a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Binary fingerprint array
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.morgan_radius,
            nBits=self.morgan_bits
        )
        return np.array(fp)

    def calculate_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """
        Calculate molecular descriptors.

        Args:
            mol: RDKit molecule object

        Returns:
            Array of descriptor values
        """
        descriptor_values = []

        for desc_name in self.descriptor_list:
            try:
                if hasattr(Descriptors, desc_name):
                    desc_func = getattr(Descriptors, desc_name)
                    value = desc_func(mol)
                    descriptor_values.append(value)
                else:
                    logger.warning(f"Descriptor '{desc_name}' not found in RDKit")
                    descriptor_values.append(0.0)
            except Exception as e:
                logger.warning(f"Error calculating descriptor '{desc_name}': {e}")
                descriptor_values.append(0.0)

        return np.array(descriptor_values)

    def extract_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        Extract features from a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Feature vector or None if extraction fails
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        features = []

        # Morgan fingerprint
        if self.use_morgan:
            morgan_fp = self.calculate_morgan_fingerprint(mol)
            features.append(morgan_fp)

        # Molecular descriptors
        if self.use_descriptors and self.descriptor_list:
            descriptors = self.calculate_descriptors(mol)
            features.append(descriptors)

        if not features:
            return None

        return np.concatenate(features)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Process entire DataFrame and extract features.

        Args:
            df: DataFrame with 'smiles' and 'p_np' columns
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            Tuple of (features, labels, valid_indices)
        """
        logger.info(f"Processing {len(df)} molecules...")

        features_list = []
        labels_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            smiles = row['smiles']
            label = row['p_np']

            features = self.extract_features(smiles)

            if features is not None:
                features_list.append(features)
                labels_list.append(label)
                valid_indices.append(idx)
            else:
                logger.warning(f"Failed to extract features for SMILES: {smiles}")

        if not features_list:
            raise ValueError("No valid features extracted from dataset")

        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Extracted features for {len(X)} molecules")
        logger.info(f"Feature shape: {X.shape}")

        # Generate feature names (only once)
        if not self.feature_names:
            self._generate_feature_names(X.shape[1])

        # Scale features
        if self.scale_features:
            if fit_scaler:
                logger.info("Fitting scaler on training data...")
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    raise ValueError("Scaler not fitted. Process training data first.")
                X = self.scaler.transform(X)

        return X, y, valid_indices

    def _generate_feature_names(self, num_features: int):
        """Generate feature names based on configuration."""
        names = []

        if self.use_morgan:
            names.extend([f'morgan_{i}' for i in range(self.morgan_bits)])

        if self.use_descriptors and self.descriptor_list:
            names.extend(self.descriptor_list)

        # Validate
        if len(names) != num_features:
            logger.warning(f"Feature name count mismatch. Expected {num_features}, got {len(names)}")
            names = [f'feature_{i}' for i in range(num_features)]

        self.feature_names = names
        logger.info(f"Generated {len(self.feature_names)} feature names")

    def save_scaler(self, save_path: str):
        """
        Save fitted scaler to disk.

        Args:
            save_path: Path to save scaler
        """
        if not self.is_fitted:
            logger.warning("Scaler not fitted yet")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Scaler saved to {save_path}")

    def load_scaler(self, load_path: str):
        """
        Load fitted scaler from disk.

        Args:
            load_path: Path to load scaler from
        """
        with open(load_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.is_fitted = True
        logger.info(f"Scaler loaded from {load_path}")

    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of feature vectors.

        Returns:
            Feature dimension
        """
        dim = 0

        if self.use_morgan:
            dim += self.morgan_bits

        if self.use_descriptors:
            dim += len(self.descriptor_list)

        return dim

    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        mol = self.smiles_to_mol(smiles)
        return mol is not None

    def get_molecule_info(self, smiles: str) -> Dict:
        """
        Get detailed information about a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with molecule information
        """
        mol = self.smiles_to_mol(smiles)

        if mol is None:
            return {'valid': False, 'error': 'Invalid SMILES'}

        info = {
            'valid': True,
            'smiles': smiles,
            'canonical_smiles': Chem.MolToSmiles(mol),
            'molecular_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
        }

        # Add descriptors if available
        if self.use_descriptors:
            for desc_name in self.descriptor_list:
                if hasattr(Descriptors, desc_name):
                    desc_func = getattr(Descriptors, desc_name)
                    info[desc_name] = desc_func(mol)

        return info
