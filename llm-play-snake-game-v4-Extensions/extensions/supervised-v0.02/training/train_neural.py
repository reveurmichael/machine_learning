"""
Neural Network Training Script for Supervised Learning v0.02
--------------------

Trains neural network models (MLP, CNN, LSTM, GRU) using datasets generated
by heuristics extensions. Uses the common CSV schema and dataset loader
for consistent data processing across different grid sizes.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

from extensions.common.dataset_loader import load_dataset_for_training
from extensions.supervised_v0_02.models.neural_networks.agent_mlp import MLPAgent
from extensions.supervised_v0_02.models.neural_networks.agent_cnn import CNNAgent
from extensions.supervised_v0_02.models.neural_networks.agent_lstm import LSTMAgent


def create_agent(model_type: str, grid_size: int, **kwargs) -> Any:
    """
    Create a neural network agent based on model type.
    
    Args:
        model_type: Type of model (MLP, CNN, LSTM, GRU)
        grid_size: Size of the game grid
        **kwargs: Additional model parameters
        
    Returns:
        Initialized agent
    """
    model_type = model_type.upper()
    
    if model_type == "MLP":
        return MLPAgent(grid_size=grid_size, **kwargs)
    elif model_type == "CNN":
        return CNNAgent(grid_size=grid_size, **kwargs)
    elif model_type == "LSTM":
        return LSTMAgent(grid_size=grid_size, **kwargs)
    elif model_type == "GRU":
        # Note: GRU agent was deleted, so we'll use LSTM as fallback
        print("Warning: GRU agent not available, using LSTM instead")
        return LSTMAgent(grid_size=grid_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(model_type: str, dataset_paths: List[str], output_dir: str,
               grid_size: int = None, epochs: int = 100, batch_size: int = 32,
               learning_rate: float = 0.001, test_size: float = 0.2,
               val_size: float = 0.2, **kwargs) -> Dict[str, Any]:
    """
    Train a neural network model.
    
    Args:
        model_type: Type of model to train
        dataset_paths: Paths to dataset files
        output_dir: Directory to save trained model
        grid_size: Grid size (auto-detected if None)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        test_size: Proportion for test set
        val_size: Proportion for validation set
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary with training results
    """
    print(f"Training {model_type} model...")
    print(f"Dataset paths: {dataset_paths}")
    print(f"Grid size: {grid_size or 'auto-detect'}")
    
    # Load and prepare dataset
    X_train, X_val, X_test, y_train, y_val, y_test, dataset_info = load_dataset_for_training(
        dataset_paths, grid_size=grid_size, test_size=test_size, val_size=val_size
    )
    
    # Update grid_size from dataset info
    grid_size = dataset_info["grid_size"]
    
    # Create agent
    agent = create_agent(model_type, grid_size, **kwargs)
    
    # Train the model
    print(f"Training {model_type} with {len(X_train)} samples...")
    training_results = agent.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Evaluate on validation set
    val_predictions = agent.predict(X_val)
    val_accuracy = (val_predictions == y_val).mean()
    
    # Evaluate on test set
    test_predictions = agent.predict(X_test)
    test_accuracy = (test_predictions == y_test).mean()
    
    # Save model using grid-size aware structure
    # Ensure output_dir follows grid-size structure
    if "grid-size-" not in str(output_dir):
        # If output_dir doesn't contain grid-size, create proper structure
        base_output = Path(output_dir).parent if Path(output_dir).name.startswith("grid-size-") else Path(output_dir)
        output_path = base_output / f"grid-size-{grid_size}" / "pytorch"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use common model utilities for standardized saving
    from extensions.common.model_utils import save_model_standardized
    
    model_filename = f"{model_type.lower()}_grid{grid_size}"
    saved_path = save_model_standardized(
        model=agent.model,  # For neural networks, use the model directly
        framework='PyTorch',
        grid_size=grid_size,
        model_name=model_filename,
        model_class=agent.__class__.__name__,
        input_size=agent.input_size,
        output_size=4,  # 4 directions
        training_params={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'test_size': test_size,
            'val_size': val_size,
            **kwargs
        },
        export_onnx=True  # Enable ONNX export for PyTorch models
    )
    
    # Print results
    print("Training completed!")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {saved_path}")
    
    return {
        "model_type": model_type,
        "grid_size": grid_size,
        "dataset_info": dataset_info,
        "training_results": training_results,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "model_path": saved_path,
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test_size": test_size,
            "val_size": val_size,
            **kwargs
        }
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train neural network models for Snake game")
    
    # Dataset arguments
    parser.add_argument("--dataset-paths", nargs="+", required=True,
                       help="Paths to dataset files or directories")
    parser.add_argument("--grid-size", type=int, default=None,
                       help="Grid size (auto-detected if not specified)")
    
    # Model arguments
    parser.add_argument("--model", choices=["MLP", "CNN", "LSTM", "GRU"], 
                       default="MLP", help="Model type to train")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden layer size for MLP")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion for test set")
    parser.add_argument("--val-size", type=float, default=0.2,
                       help="Proportion for validation set")
    
    # Output arguments
    parser.add_argument("--output-dir", default="logs/extensions/models",
                       help="Directory to save trained models (will create grid-size-N subdirectory)")
    
    args = parser.parse_args()
    
    # Train the model
    try:
        results = train_model(
            model_type=args.model,
            dataset_paths=args.dataset_paths,
            output_dir=args.output_dir,
            grid_size=args.grid_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_size=args.test_size,
            val_size=args.val_size,
            hidden_size=args.hidden_size
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 