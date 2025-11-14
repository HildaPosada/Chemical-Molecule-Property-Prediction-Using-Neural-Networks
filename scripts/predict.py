"""Script for making predictions on new molecules."""

import os
import sys
import argparse
import torch
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import MoleculePreprocessor
from src.models import create_model
from src.utils import load_config, setup_logger

logger = setup_logger()


def predict_from_smiles(
    smiles: str,
    model: torch.nn.Module,
    preprocessor: MoleculePreprocessor,
    device: torch.device
) -> dict:
    """
    Predict property for a single SMILES string.

    Args:
        smiles: SMILES string
        model: Trained model
        preprocessor: Molecule preprocessor
        device: Computation device

    Returns:
        Dictionary with prediction results
    """
    # Validate SMILES
    if not preprocessor.validate_smiles(smiles):
        return {
            'smiles': smiles,
            'valid': False,
            'error': 'Invalid SMILES string'
        }

    # Extract features
    features = preprocessor.extract_features(smiles)
    if features is None:
        return {
            'smiles': smiles,
            'valid': False,
            'error': 'Feature extraction failed'
        }

    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    return {
        'smiles': smiles,
        'valid': True,
        'prediction': int(prediction.item()),
        'prediction_label': 'Penetrates BBB' if prediction.item() == 1 else 'Does not penetrate BBB',
        'confidence': float(probabilities[0, prediction.item()].item()),
        'probability_negative': float(probabilities[0, 0].item()),
        'probability_positive': float(probabilities[0, 1].item())
    }


def main():
    """Make predictions on new molecules."""
    parser = argparse.ArgumentParser(description='Predict molecular properties')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--smiles',
        type=str,
        help='Single SMILES string to predict'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='CSV file with SMILES column'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file for predictions'
    )

    args = parser.parse_args()

    if not args.smiles and not args.input_file:
        parser.error("Either --smiles or --input-file must be provided")

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Load preprocessor
    logger.info("Loading preprocessor...")
    preprocessor = MoleculePreprocessor(config)
    scaler_path = os.path.join(config['data']['processed_dir'], 'scaler.pkl')

    try:
        preprocessor.load_scaler(scaler_path)
    except FileNotFoundError:
        logger.warning(f"Scaler not found at {scaler_path}. Features will not be scaled.")

    # Load model
    logger.info("Loading model...")
    input_size = preprocessor.get_feature_dim()
    model = create_model(config, input_size)

    device = config['training']['device']
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {args.model_path}")

    # Make predictions
    results = []

    if args.smiles:
        # Single SMILES prediction
        logger.info(f"Predicting for SMILES: {args.smiles}")
        result = predict_from_smiles(args.smiles, model, preprocessor, device)

        logger.info("\nPrediction Result:")
        logger.info(f"SMILES: {result['smiles']}")
        if result['valid']:
            logger.info(f"Prediction: {result['prediction_label']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Probability (No): {result['probability_negative']:.2%}")
            logger.info(f"Probability (Yes): {result['probability_positive']:.2%}")
        else:
            logger.error(f"Error: {result['error']}")

        results.append(result)

    elif args.input_file:
        # Batch prediction from file
        logger.info(f"Loading molecules from {args.input_file}")
        df = pd.read_csv(args.input_file)

        if 'smiles' not in df.columns:
            raise ValueError("Input file must contain 'smiles' column")

        logger.info(f"Predicting for {len(df)} molecules...")

        for idx, row in df.iterrows():
            smiles = row['smiles']
            result = predict_from_smiles(smiles, model, preprocessor, device)
            results.append(result)

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} molecules")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        logger.info(f"\nPredictions saved to {args.output}")

        # Print summary
        valid_predictions = results_df[results_df['valid'] == True]
        if len(valid_predictions) > 0:
            positive_count = (valid_predictions['prediction'] == 1).sum()
            logger.info(f"\nSummary:")
            logger.info(f"Total molecules: {len(results_df)}")
            logger.info(f"Valid predictions: {len(valid_predictions)}")
            logger.info(f"Penetrates BBB: {positive_count}")
            logger.info(f"Does not penetrate BBB: {len(valid_predictions) - positive_count}")

    logger.info("\nPrediction complete!")


if __name__ == '__main__':
    main()
