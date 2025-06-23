#!/usr/bin/env python3
"""
Supervised Learning v0.03 - Training Script
==========================================

Modern CLI training script using elegant utilities and configuration management.
Follows elegance guidelines for file organization and user experience.

Design Pattern: Template Method
- CLI interface with comprehensive arguments
- Standardized model training pipeline
- Grid size flexibility
- Configuration management
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import elegant utilities
from extensions.common.training_cli_utils import create_parser, validate_args, args_to_config
from extensions.common.training_config_utils import load_config, save_config, validate_config
from extensions.common.training_logging_utils import log_experiment_start, log_experiment_complete
from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()


def main():
    """Main training entry point with elegant CLI and configuration."""
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Load configuration
    config = load_configuration(args)
    
    # Setup logging
    log_dir = Path("logs/extensions/supervised-v0.03")
    logger = log_experiment_start(config, log_dir)
    
    try:
        # Train based on model type
        train_model(args.model, config, logger)
        
        # Log completion
        final_results = {"status": "success", "model": args.model}
        log_experiment_complete(logger, final_results)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def load_configuration(args) -> dict:
    """Load configuration from file or create from arguments."""
    # Load from config file if provided
    if args.config and args.config.exists():
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create from command line arguments
        config = args_to_config(args)
        logger.info("Created configuration from command line arguments")
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    # Save configuration for reproducibility
    config_path = Path("logs/extensions/supervised-v0.03/config.json")
    save_config(config, config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    return config


def train_model(model_type: str, config: dict, logger):
    """Train model based on type."""
    logger.info(f"Training {model_type} model...")
    
    if model_type in ["MLP", "CNN", "LSTM"]:
        train_neural_network(model_type, config, logger)
    elif model_type in ["XGBOOST", "LIGHTGBM", "RANDOMFOREST"]:
        train_tree_model(model_type, config, logger)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_neural_network(model_type: str, config: dict, logger):
    """Train neural network models."""
    # Import neural network components
    if model_type == "MLP":
        from models.neural_networks.agent_mlp import MLPAgent
        agent = MLPAgent(
            grid_size=config["training"]["grid_size"],
            hidden_size=config["model"]["hidden_size"],
            learning_rate=config["model"]["learning_rate"]
        )
    elif model_type == "CNN":
        from models.neural_networks.agent_cnn import CNNAgent
        agent = CNNAgent(
            grid_size=config["training"]["grid_size"],
            learning_rate=config["model"]["learning_rate"]
        )
    elif model_type == "LSTM":
        from models.neural_networks.agent_lstm import LSTMAgent
        agent = LSTMAgent(
            grid_size=config["training"]["grid_size"],
            hidden_size=config["model"]["hidden_size"],
            learning_rate=config["model"]["learning_rate"]
        )
    
    # Load training data
    X_train, y_train = load_training_data(config)
    
    # Train the model
    logger.info(f"Training {model_type} for {config['model']['epochs']} epochs...")
    training_result = agent.train(
        X_train, y_train,
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        validation_split=config["model"]["validation_split"]
    )
    
    # Save the model
    model_name = f"{model_type.lower()}_model"
    agent.save_model(model_name)
    
    logger.info(f"Training completed! Final accuracy: {training_result.get('final_accuracy', 'N/A')}")


def train_tree_model(model_type: str, config: dict, logger):
    """Train tree-based models."""
    # Import tree model components
    if model_type == "XGBOOST":
        from models.tree_models.agent_xgboost import XGBoostAgent
        agent = XGBoostAgent(
            grid_size=config["training"]["grid_size"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            n_estimators=config["model"]["n_estimators"]
        )
    elif model_type == "LIGHTGBM":
        from models.tree_models.agent_lightgbm import LightGBMAgent
        agent = LightGBMAgent(
            grid_size=config["training"]["grid_size"],
            learning_rate=config["model"]["learning_rate"],
            n_estimators=config["model"]["n_estimators"]
        )
    elif model_type == "RANDOMFOREST":
        from models.tree_models.agent_randomforest import RandomForestAgent
        agent = RandomForestAgent(
            grid_size=config["training"]["grid_size"],
            max_depth=config["model"]["max_depth"],
            n_estimators=config["model"]["n_estimators"]
        )
    
    # Load training data
    X_train, y_train = load_training_data(config)
    
    # Train the model
    logger.info(f"Training {model_type}...")
    training_result = agent.train(
        X_train, y_train,
        validation_split=config["model"]["validation_split"]
    )
    
    # Save the model
    model_name = f"{model_type.lower()}_model"
    agent.save_model(model_name)
    
    logger.info(f"Training completed! Final accuracy: {training_result.get('final_accuracy', 'N/A')}")


def load_training_data(config: dict):
    """Load training data from dataset."""
    # This would load data from heuristics-generated datasets
    # For now, return dummy data
    import numpy as np
    
    # Generate dummy training data
    n_samples = 1000
    n_features = config["training"]["grid_size"] ** 2 + 4  # Board + position features
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 4, n_samples)
    
    return X_train, y_train


if __name__ == "__main__":
    main() 