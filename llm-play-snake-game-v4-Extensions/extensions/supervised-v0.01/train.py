#!/usr/bin/env python3
"""
Neural Network Training Script for Supervised Learning v0.01
============================================================

Trains PyTorch neural networks on datasets generated from heuristic algorithms.
Demonstrates the evolution from heuristics (rule-based) to supervised learning (data-driven).

This script shows how expert demonstrations from heuristic algorithms can be used
to train neural networks that potentially generalize beyond the original algorithms.

Usage:
    # Train MLP on tabular data
    python train.py --model MLP --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_bfs_data.csv

    # Train CNN on board data  
    python train.py --model CNN --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_mixed_data.csv

    # Train LSTM on sequential data
    python train.py --model LSTM --dataset-path ../../logs/extensions/datasets/grid-size-10/sequential_mixed_data.npz

Features:
- Multiple neural network architectures (MLP, CNN, LSTM, GRU)
- Automatic train/validation/test splits
- Comprehensive evaluation metrics
- Model checkpointing and saving
- Performance comparison with heuristic baselines
- Extensive reuse of Task-0 utilities and common extensions
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Import neural network architectures
from agent_neural import MLPNetwork, CNNNetwork, RNNNetwork, create_neural_agent

# Import common extensions
from extensions.common import get_dataset_path, EXTENSIONS_LOGS_DIR, DEFAULT_GRID_SIZE


class SnakeDataset(Dataset):
    """
    PyTorch dataset for snake game data.
    
    Handles different data formats (CSV, NPZ) and structures (tabular, sequential).
    """
    
    def __init__(self, data_path: str, data_type: str = "tabular"):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to the dataset file
            data_type: Type of data ("tabular", "sequential", "graph")
        """
        self.data_type = data_type
        self.features, self.targets, self.metadata = self._load_data(data_path)
        
        # Normalize features for better training
        if data_type == "tabular":
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
    
    def _load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load data from file."""
        data_path = Path(data_path)
        
        if data_path.suffix == ".csv":
            # Load CSV data (tabular)
            df = pd.read_csv(data_path)
            
            # Separate features and targets
            feature_cols = [col for col in df.columns if col not in ["target_move", "game_id", "step_in_game", "algorithm", "round_number"]]
            features = df[feature_cols].values.astype(np.float32)
            targets = df["target_move"].values.astype(np.int64)
            
            metadata = {
                "feature_names": feature_cols,
                "target_names": ["UP", "DOWN", "LEFT", "RIGHT"],
                "num_samples": len(df),
                "num_features": len(feature_cols)
            }
            
        elif data_path.suffix == ".npz":
            # Load NPZ data (sequential or graph)
            data = np.load(data_path, allow_pickle=True)
            
            if self.data_type == "sequential":
                features = data["sequences"].astype(np.float32)
                targets = data["targets"].astype(np.int64)
                metadata = {
                    "sequence_length": data.get("sequence_length", 10),
                    "feature_dim": data.get("feature_dim", features.shape[-1] if len(features) > 0 else 0),
                    "target_names": data.get("target_names", ["UP", "DOWN", "LEFT", "RIGHT"]).tolist(),
                    "num_samples": len(features)
                }
            else:
                raise NotImplementedError(f"Data type {self.data_type} not yet implemented for NPZ files")
                
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        return features, targets, metadata
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = torch.FloatTensor(self.features[idx])
        target = torch.LongTensor([self.targets[idx]])
        return feature, target.squeeze()
    
    def get_metadata(self) -> Dict:
        return self.metadata


class SupervisedTrainer:
    """
    Trainer class for supervised learning models.
    
    Handles training, validation, and evaluation of neural networks
    on snake game datasets generated from heuristic algorithms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to run on ("cpu" or "cuda")
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_targets.size(0)
            correct_predictions += (predicted == batch_targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_targets.size(0)
                correct_predictions += (predicted == batch_targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        best_val_accuracy = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Track history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - start_time
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return {
            "best_val_accuracy": best_val_accuracy,
            "training_time": training_time,
            "epochs_trained": epoch + 1,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation on test set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_targets.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        class_report = classification_report(all_targets, all_predictions, 
                                           target_names=["UP", "DOWN", "LEFT", "RIGHT"],
                                           output_dict=True)
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        return {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "predictions": all_predictions,
            "targets": all_targets
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural networks on heuristic-generated datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train MLP on tabular data
  python train.py --model MLP --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_bfs_data.csv

  # Train CNN with custom parameters
  python train.py --model CNN --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_mixed_data.csv \\
      --hidden-size 512 --epochs 200 --batch-size 64

  # Train LSTM on sequential data
  python train.py --model LSTM --dataset-path ../../logs/extensions/datasets/grid-size-10/sequential_mixed_data.npz \\
      --learning-rate 0.0001 --dropout 0.4
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        choices=["MLP", "CNN", "LSTM", "GRU"],
        required=True,
        help="Type of neural network model to train"
    )
    
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the dataset file (CSV or NPZ format)"
    )
    
    # Model architecture arguments
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden layer size (default: 256)"
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of layers for RNN models (default: 2)"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 1e-4)"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)"
    )
    
    # Other arguments
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained models (default: models)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for training (default: auto)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training plots"
    )
    
    return parser.parse_args()


def create_model(args: argparse.Namespace, dataset: SnakeDataset) -> nn.Module:
    """Create model based on arguments and dataset."""
    metadata = dataset.get_metadata()
    
    if args.model == "MLP":
        input_size = metadata["num_features"]
        model = MLPNetwork(
            input_size=input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout
        )
    elif args.model == "CNN":
        model = CNNNetwork(
            board_size=10,  # Assuming 10x10 grid
            dropout=args.dropout
        )
    elif args.model in ["LSTM", "GRU"]:
        if dataset.data_type != "sequential":
            raise ValueError(f"{args.model} requires sequential data")
        
        input_size = metadata["feature_dim"]
        model = RNNNetwork(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            rnn_type=args.model
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    return model


def main():
    """Main training function."""
    print("🚀 Supervised Learning v0.01 - Neural Network Training")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🔧 Using device: {device}")
    
    # Load dataset
    print(f"📁 Loading dataset: {args.dataset_path}")
    
    # Determine data type from file path
    if "sequential" in args.dataset_path:
        data_type = "sequential"
    elif "graph" in args.dataset_path:
        data_type = "graph"
    else:
        data_type = "tabular"
    
    dataset = SnakeDataset(args.dataset_path, data_type=data_type)
    metadata = dataset.get_metadata()
    
    print(f"📊 Dataset info:")
    print(f"   Samples: {metadata['num_samples']}")
    if data_type == "tabular":
        print(f"   Features: {metadata['num_features']}")
    elif data_type == "sequential":
        print(f"   Sequence length: {metadata['sequence_length']}")
        print(f"   Feature dimension: {metadata['feature_dim']}")
    
    # Create model
    print(f"🧠 Creating {args.model} model...")
    model = create_model(args, dataset)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"📚 Data splits:")
    print(f"   Train: {train_size} ({args.train_split:.1%})")
    print(f"   Validation: {val_size} ({args.val_split:.1%})")
    print(f"   Test: {test_size} ({1 - args.train_split - args.val_split:.1%})")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print(f"🏋️  Training {args.model} for up to {args.epochs} epochs...")
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        verbose=args.verbose
    )
    
    print(f"✅ Training completed!")
    print(f"   Best validation accuracy: {training_history['best_val_accuracy']:.2f}%")
    print(f"   Training time: {training_history['training_time']:.1f} seconds")
    print(f"   Epochs trained: {training_history['epochs_trained']}")
    
    # Evaluate on test set
    print(f"🧪 Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    
    print(f"📊 Test Results:")
    print(f"   Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"   Classification Report:")
    
    # Print classification report
    report = test_results['classification_report']
    for move in ["UP", "DOWN", "LEFT", "RIGHT"]:
        if move in report:
            precision = report[move]['precision']
            recall = report[move]['recall']
            f1 = report[move]['f1-score']
            print(f"     {move:>5}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Save model and results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name = f"{args.model}_{data_type}_model.pth"
    model_path = output_dir / model_name
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model,
        'data_type': data_type,
        'training_history': training_history,
        'test_results': test_results,
        'args': vars(args),
        'dataset_metadata': metadata
    }, model_path)
    
    print(f"💾 Model saved to: {model_path}")
    
    # Save training plots if requested
    if args.save_plots:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history['train_losses'], label='Train Loss')
        plt.plot(training_history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(training_history['train_accuracies'], label='Train Accuracy')
        plt.plot(training_history['val_accuracies'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plot_path = output_dir / f"{args.model}_{data_type}_training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training plots saved to: {plot_path}")
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=["UP", "DOWN", "LEFT", "RIGHT"],
                   yticklabels=["UP", "DOWN", "LEFT", "RIGHT"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        confusion_path = output_dir / f"{args.model}_{data_type}_confusion_matrix.png"
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {confusion_path}")
    
    print("🎉 Training and evaluation completed successfully!")


if __name__ == "__main__":
    main() 