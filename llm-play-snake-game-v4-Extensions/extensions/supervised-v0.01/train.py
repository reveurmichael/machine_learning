#!/usr/bin/env python3
"""
Supervised Learning v0.01 - Training Script
==========================================

Simple training script for supervised learning v0.01, focusing on neural networks only.
Demonstrates basic training pipeline for proof of concept.

Design Pattern: Template Method
- Simple training interface with no complex arguments
- Focused on neural networks only
- Uses common CSV schema for data processing
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

from extensions.common.dataset_loader import load_dataset_for_training
from agent_neural import MLPAgent, CNNAgent, LSTMAgent


def create_agent(model_type: str, grid_size: int, **kwargs) -> Any:
    """
    Create a neural network agent based on model type.
    
    Factory pattern for agent creation in v0.01.
    
    Args:
        model_type: Type of model (MLP, CNN, LSTM)
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(model_type: str, dataset_paths: List[str], output_dir: str,
               grid_size: int = None, epochs: int = 100, **kwargs) -> Dict[str, Any]:
    """
    Train a neural network model.
    
    Simple training function for v0.01, focused on proof of concept.
        
        Args:
        model_type: Type of model to train
        dataset_paths: Paths to dataset files
        output_dir: Directory to save trained model
        grid_size: Grid size (auto-detected if None)
        epochs: Number of training epochs
        **kwargs: Additional training parameters
            
        Returns:
        Dictionary with training results
    """
    print(f"Training {model_type} model...")
    print(f"Dataset paths: {dataset_paths}")
    print(f"Grid size: {grid_size or 'auto-detect'}")
    
    # Load and prepare dataset
    X_train, X_val, X_test, y_train, y_val, y_test, dataset_info = load_dataset_for_training(
        dataset_paths, grid_size=grid_size
    )
    
    # Update grid_size from dataset info
    grid_size = dataset_info["grid_size"]
    
    # Create agent
    agent = create_agent(model_type, grid_size, **kwargs)
    
    # Train the model
    print(f"Training {model_type} with {len(X_train)} samples...")
    training_results = agent.train(
        X_train, y_train,
        epochs=epochs
    )
    
    # Evaluate on validation set
    val_predictions = agent.predict(X_val)
    val_accuracy = (val_predictions == y_val).mean()
    
    # Evaluate on test set
    test_predictions = agent.predict(X_test)
    test_accuracy = (test_predictions == y_test).mean()
    
    # Save model
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
            'model_type': model_type
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
        "training_results": training_results,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "model_path": saved_path
    }


def main():
    """
    Main training function for supervised learning v0.01.
    
    Simple CLI with minimal arguments, focused on neural networks only.
    """
    parser = argparse.ArgumentParser(description="Train neural network models for Snake game v0.01")
    
    # Simple arguments for v0.01
    parser.add_argument("--model", choices=["MLP", "CNN", "LSTM"], 
                       default="MLP", help="Model type to train (default: MLP)")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset file or directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--output-dir", default="./trained_models",
                       help="Directory to save trained models (default: ./trained_models)")
    
    args = parser.parse_args()
    
    # Convert single path to list for compatibility
    dataset_paths = [args.dataset_path]
    
    try:
        # Train the model
        results = train_model(
            model_type=args.model,
            dataset_paths=dataset_paths,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 