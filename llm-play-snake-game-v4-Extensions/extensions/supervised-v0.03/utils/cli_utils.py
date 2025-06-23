"""
CLI utilities for supervised learning v0.03.

Design Pattern: Command Pattern
- Centralized argument parsing
- Validation and help text
- User-friendly interface
"""

import argparse
from typing import Dict, Any, List
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all supervised learning options."""
    parser = argparse.ArgumentParser(
        description="Supervised Learning v0.03 - Multi-Model Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train MLP neural network
  python scripts/train.py --model MLP --grid-size 15 --epochs 200

  # Train XGBoost with custom parameters
  python scripts/train.py --model XGBOOST --max-depth 8 --learning-rate 0.05

  # Evaluate trained model
  python scripts/evaluate.py --model MLP --grid-size 15

  # Compare multiple models
  python scripts/evaluate.py --models MLP,XGBOOST,LIGHTGBM --grid-size 10
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"],
        help="Model type to train/evaluate"
    )
    
    # Game parameters
    parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=10,
        choices=range(5, 51),
        metavar="[5-50]",
        help="Grid size for the game (default: 10)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--hidden-size", "-hs",
        type=int,
        default=256,
        help="Hidden layer size for neural networks (default: 256)"
    )
    
    # Tree model parameters
    parser.add_argument(
        "--max-depth", "-md",
        type=int,
        default=6,
        help="Maximum depth for tree models (default: 6)"
    )
    
    parser.add_argument(
        "--n-estimators", "-ne",
        type=int,
        default=100,
        help="Number of estimators for ensemble models (default: 100)"
    )
    
    # Data parameters
    parser.add_argument(
        "--dataset-path", "-d",
        type=Path,
        help="Path to training dataset"
    )
    
    parser.add_argument(
        "--validation-split", "-vs",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for models (default: auto-generated)"
    )
    
    parser.add_argument(
        "--experiment-name", "-en",
        type=str,
        help="Experiment name (default: auto-generated)"
    )
    
    # Training control
    parser.add_argument(
        "--max-games", "-mg",
        type=int,
        default=1000,
        help="Maximum number of games to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--save-frequency", "-sf",
        type=int,
        default=100,
        help="Save model every N epochs (default: 100)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--models", "-ms",
        type=str,
        help="Comma-separated list of models to compare"
    )
    
    parser.add_argument(
        "--test-size", "-ts",
        type=float,
        default=0.2,
        help="Test set size ratio (default: 0.2)"
    )
    
    # General options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    errors = []
    
    # Validate grid size
    if args.grid_size < 5 or args.grid_size > 50:
        errors.append("Grid size must be between 5 and 50")
    
    # Validate learning rate
    if args.learning_rate <= 0 or args.learning_rate > 1:
        errors.append("Learning rate must be between 0 and 1")
    
    # Validate validation split
    if args.validation_split <= 0 or args.validation_split >= 1:
        errors.append("Validation split must be between 0 and 1")
    
    # Validate test size
    if args.test_size <= 0 or args.test_size >= 1:
        errors.append("Test size must be between 0 and 1")
    
    # Validate epochs
    if args.epochs <= 0:
        errors.append("Epochs must be positive")
    
    # Validate batch size
    if args.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    # Validate max games
    if args.max_games <= 0:
        errors.append("Max games must be positive")
    
    # Validate max depth
    if args.max_depth <= 0:
        errors.append("Max depth must be positive")
    
    # Validate n estimators
    if args.n_estimators <= 0:
        errors.append("Number of estimators must be positive")
    
    # Check for conflicts
    if args.verbose and args.quiet:
        errors.append("Cannot use both --verbose and --quiet")
    
    # Report errors
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def parse_model_list(models_str: str) -> List[str]:
    """Parse comma-separated model list."""
    if not models_str:
        return []
    
    models = [model.strip().upper() for model in models_str.split(",")]
    valid_models = ["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"]
    
    invalid_models = [model for model in models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Invalid models: {invalid_models}. Valid options: {valid_models}")
    
    return models


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command line arguments to configuration dictionary."""
    config = {
        "model": {
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "validation_split": args.validation_split,
        },
        "training": {
            "grid_size": args.grid_size,
            "max_games": args.max_games,
            "save_frequency": args.save_frequency,
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "experiment_name": args.experiment_name,
        },
        "evaluation": {
            "test_size": args.test_size,
        },
        "log_level": "INFO" if args.verbose and not args.quiet else "WARNING",
        "device": "auto"
    }
    
    return config 