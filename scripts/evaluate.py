"""Script to evaluate trained model."""

from src.utils import load_config, setup_logger
from src.evaluation import Evaluator, Visualizer
from src.models import create_model
from src.data import MoleculeDataLoader, MoleculePreprocessor, MoleculeDataset, create_dataloaders
import os
import sys
import argparse
import torch
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


logger = setup_logger()


def main():
    """Evaluate trained model."""
    parser = argparse.ArgumentParser(
        description='Evaluate molecular property prediction model')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/saved_models/best_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Load data
    logger.info("Loading data...")
    data_loader = MoleculeDataLoader(config)
    train_df, val_df, test_df = data_loader.load_from_sqlite()

    # Load preprocessor
    logger.info("Loading preprocessor...")
    preprocessor = MoleculePreprocessor(config)
    scaler_path = os.path.join(config['data']['processed_dir'], 'scaler.pkl')
    preprocessor.load_scaler(scaler_path)

    # Process data
    X_train, y_train, _ = preprocessor.process_dataframe(
        train_df, fit_scaler=False)
    X_val, y_val, _ = preprocessor.process_dataframe(val_df, fit_scaler=False)
    X_test, y_test, _ = preprocessor.process_dataframe(
        test_df, fit_scaler=False)

    # Create datasets
    train_dataset = MoleculeDataset(X_train, y_train)
    val_dataset = MoleculeDataset(X_val, y_val)
    test_dataset = MoleculeDataset(X_test, y_test)

    # Select dataset
    if args.split == 'train':
        dataset = train_dataset
        labels = y_train
    elif args.split == 'val':
        dataset = val_dataset
        labels = y_val
    else:
        dataset = test_dataset
        labels = y_test

    # Create data loader
    from torch.utils.data import DataLoader
    data_loader_obj = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Create model
    logger.info("Loading model...")
    input_size = preprocessor.get_feature_dim()
    model = create_model(config, input_size)

    # Load checkpoint
    device = config['training']['device']
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    logger.info(f"Model loaded from {args.model_path}")

    # Create evaluator
    logger.info(f"Evaluating on {args.split} set...")
    evaluator = Evaluator(model, device)

    # Evaluate
    metrics = evaluator.evaluate(
        data_loader_obj,
        metrics=['accuracy', 'precision', 'recall',
                 'f1', 'roc_auc', 'confusion_matrix']
    )

    # Print results
    logger.info("\n" + "="*50)
    logger.info(f"Evaluation Results on {args.split.upper()} set")
    logger.info("="*50)
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"{metric.upper()}: {value:.4f}")

    # Print classification report
    logger.info("\nDetailed Classification Report:")
    report = evaluator.get_classification_report(data_loader_obj)
    logger.info(f"\n{report}")

    # Save metrics
    metrics_path = os.path.join(
        config['visualization']['metrics_dir'],
        f'{args.split}_metrics.json'
    )

    # Convert confusion matrix to list for JSON serialization
    metrics_save = metrics.copy()
    if 'confusion_matrix' in metrics_save:
        metrics_save['confusion_matrix'] = metrics_save['confusion_matrix'].tolist()

    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    visualizer = Visualizer(save_dir=config['visualization']['figures_dir'])

    # Get predictions for visualization
    predictions, probabilities, true_labels = evaluator.predict(
        data_loader_obj)

    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_name=f'{args.split}_confusion_matrix.png'
    )

    # Plot ROC curve
    visualizer.plot_roc_curve(
        true_labels,
        probabilities[:, 1],
        save_name=f'{args.split}_roc_curve.png'
    )

    # Plot metrics
    visualizer.plot_metrics_comparison(
        metrics,
        save_name=f'{args.split}_metrics.png'
    )

    logger.info(
        f"Visualizations saved to {config['visualization']['figures_dir']}")
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
