#!/usr/bin/env python3
"""
Supervised Learning v0.03 - Evaluation Script
============================================

Focused evaluation script for trained models.
Follows elegance guidelines for clean, maintainable code.

Design Pattern: Strategy Pattern
- Multiple evaluation strategies
- Model comparison capabilities
- Performance benchmarking
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.training_cli_utils import create_parser, validate_args, parse_model_list
from extensions.common.training_logging_utils import setup_logging
from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()


def main():
    """Main evaluation entry point."""
    # Create parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Determine evaluation mode
        if args.models:
            # Multi-model comparison
            models = parse_model_list(args.models)
            compare_models(models, args, logger)
        else:
            # Single model evaluation
            evaluate_single_model(args.model, args, logger)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def evaluate_single_model(model_type: str, args, logger):
    """Evaluate a single trained model."""
    logger.info(f"Evaluating {model_type} model...")
    
    # Load the model
    agent = load_model(model_type, args.grid_size)
    
    # Load test data
    X_test, y_test = load_test_data(args.grid_size)
    
    # Evaluate the model
    results = agent.evaluate(X_test, y_test)
    
    # Display results
    display_results(model_type, results, logger)


def compare_models(models: list, args, logger):
    """Compare multiple trained models."""
    logger.info(f"Comparing models: {', '.join(models)}")
    
    results = {}
    
    for model_type in models:
        try:
            # Load and evaluate each model
            agent = load_model(model_type, args.grid_size)
            X_test, y_test = load_test_data(args.grid_size)
            model_results = agent.evaluate(X_test, y_test)
            results[model_type] = model_results
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    # Display comparison
    display_comparison(results, logger)


def load_model(model_type: str, grid_size: int):
    """Load a trained model."""
    if model_type in ["MLP", "CNN", "LSTM"]:
        return load_neural_model(model_type, grid_size)
    elif model_type in ["XGBOOST", "LIGHTGBM", "RANDOMFOREST"]:
        return load_tree_model(model_type, grid_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_neural_model(model_type: str, grid_size: int):
    """Load neural network model."""
    if model_type == "MLP":
        from models.neural_networks.agent_mlp import MLPAgent
        agent = MLPAgent(grid_size=grid_size)
    elif model_type == "CNN":
        from models.neural_networks.agent_cnn import CNNAgent
        agent = CNNAgent(grid_size=grid_size)
    elif model_type == "LSTM":
        from models.neural_networks.agent_lstm import LSTMAgent
        agent = LSTMAgent(grid_size=grid_size)
    
    # Load trained weights
    model_name = f"{model_type.lower()}_model"
    agent.load_model(model_name)
    
    return agent


def load_tree_model(model_type: str, grid_size: int):
    """Load tree-based model."""
    if model_type == "XGBOOST":
        from models.tree_models.agent_xgboost import XGBoostAgent
        agent = XGBoostAgent(grid_size=grid_size)
    elif model_type == "LIGHTGBM":
        from models.tree_models.agent_lightgbm import LightGBMAgent
        agent = LightGBMAgent(grid_size=grid_size)
    elif model_type == "RANDOMFOREST":
        from models.tree_models.agent_randomforest import RandomForestAgent
        agent = RandomForestAgent(grid_size=grid_size)
    
    # Load trained model
    model_name = f"{model_type.lower()}_model"
    agent.load_model(model_name)
    
    return agent


def load_test_data(grid_size: int):
    """Load test data for evaluation."""
    # This would load actual test data from heuristics datasets
    # For now, generate dummy test data
    import numpy as np
    
    n_samples = 200
    n_features = grid_size ** 2 + 4
    
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.randint(0, 4, n_samples)
    
    return X_test, y_test


def display_results(model_type: str, results: dict, logger):
    """Display evaluation results for a single model."""
    logger.info("=" * 50)
    logger.info(f"Evaluation Results for {model_type}")
    logger.info("=" * 50)
    
    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    logger.info("=" * 50)


def display_comparison(results: dict, logger):
    """Display comparison results for multiple models."""
    logger.info("=" * 60)
    logger.info("Model Comparison Results")
    logger.info("=" * 60)
    
    # Find common metrics
    all_metrics = set()
    for model_results in results.values():
        if isinstance(model_results, dict) and "error" not in model_results:
            all_metrics.update(model_results.keys())
    
    if not all_metrics:
        logger.info("No common metrics found for comparison")
        return
    
    # Display header
    header = f"{'Model':<15}"
    for metric in sorted(all_metrics):
        header += f"{metric:>12}"
    logger.info(header)
    logger.info("-" * 60)
    
    # Display results
    for model_type, model_results in results.items():
        if "error" in model_results:
            logger.info(f"{model_type:<15} {'ERROR':>12}")
        else:
            row = f"{model_type:<15}"
            for metric in sorted(all_metrics):
                value = model_results.get(metric, "N/A")
                if isinstance(value, float):
                    row += f"{value:>12.4f}"
                else:
                    row += f"{str(value):>12}"
            logger.info(row)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 