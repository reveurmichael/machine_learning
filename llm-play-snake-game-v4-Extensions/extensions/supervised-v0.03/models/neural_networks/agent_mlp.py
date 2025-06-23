"""
MLP Agent for Supervised Learning v0.03
======================================

Multi-Layer Perceptron agent using PyTorch for tabular feature data.
Implements standardized model saving/loading with ONNX export.

Design Pattern: Template Method
- Standardized training interface
- Grid size flexibility
- ONNX export for cross-platform deployment
- Rich metadata for model tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
from typing import Dict, Any

from core.game_agents import SnakeAgent
from extensions.common.path_utils import setup_extension_paths
from extensions.common.csv_schema import TabularFeatureExtractor, get_schema_info
from extensions.common.model_utils import save_model_standardized, load_model_standardized
setup_extension_paths()


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron model for Snake game.
    
    Design Pattern: Template Method
    - Configurable architecture with grid size support
    - Standardized forward pass
    - Dropout for regularization
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_classes: int = 4):
        """
        Initialize MLP model.
        
        Args:
            input_size: Number of input features (depends on grid size)
            hidden_size: Size of hidden layers
            num_classes: Number of output classes (4 directions)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # MLP architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.layers(x)


class MLPAgent(SnakeAgent):
    """
    Multi-Layer Perceptron agent for Snake game.
    
    Design Pattern: Strategy Pattern
    - Implements SnakeAgent interface
    - Configurable model architecture
    - Standardized training and prediction
    """
    
    def __init__(self, grid_size: int = 10, hidden_size: int = 256, num_classes: int = 4):
        """
        Initialize MLP agent.
        
        Args:
            grid_size: Size of the game grid
            hidden_size: Size of hidden layers in MLP
            num_classes: Number of output classes (4 directions)
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Get input size from schema
        schema_info = get_schema_info(grid_size)
        self.input_size = schema_info["feature_columns"]
        
        # Initialize model and components
        self.model = MLPModel(self.input_size, hidden_size, num_classes)
        self.feature_extractor = TabularFeatureExtractor()
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Direction mapping
        self.direction_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    def get_move(self, game_logic) -> str:
        """
        Get the next move for the current game state.
        
        Args:
            game_logic: Game logic instance containing current state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get current game state
        state = game_logic.get_state_snapshot()
        
        # Extract features
        features = self.feature_extractor.extract_features(state)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(features_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Convert to direction
        return self.direction_map[predicted_class]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              batch_size: int = 32, learning_rate: float = 0.001,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the MLP model.
        
        Args:
            X: Training features of shape (n_samples, input_size)
            y: Training labels of shape (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training metrics
        """
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Split into train/validation
        n_val = int(len(X) * validation_split)
        X_train, X_val = X_tensor[:-n_val], X_tensor[-n_val:]
        y_train, y_val = y_tensor[:-n_val], y_tensor[-n_val:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"Training MLP for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Input size: {self.input_size}, Hidden size: {self.hidden_size}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Mark as trained
        self.is_trained = True
        
        # Return training metrics
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "final_train_accuracy": train_accuracies[-1],
            "final_val_accuracy": val_accuracies[-1],
            "epochs_trained": epochs
        }
    
    def save_model(self, model_name: str, export_onnx: bool = True) -> str:
        """
        Save the trained model with metadata.
        
        Args:
            model_name: Name for the saved model
            export_onnx: Whether to export ONNX format
            
        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Prepare metadata
        metadata = {
            "model_type": "MLP",
            "grid_size": self.grid_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "torch_version": torch.__version__,
            "device": str(self.device),
            "timestamp": datetime.utcnow().isoformat(),
            "model_architecture": "MLP",
            "framework": "pytorch"
        }
        
        # Save using standardized utility
        model_path = save_model_standardized(
            model_name=model_name,
            model=self.model,
            metadata=metadata,
            framework="pytorch",
            grid_size=self.grid_size,
            export_onnx=export_onnx
        )
        
        print(f"MLP model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        # Load using standardized utility
        loaded_data = load_model_standardized(model_path, framework="pytorch")
        
        # Extract model and metadata
        self.model = loaded_data["model"]
        metadata = loaded_data["metadata"]
        
        # Update agent state
        self.grid_size = metadata["grid_size"]
        self.input_size = metadata["input_size"]
        self.hidden_size = metadata["hidden_size"]
        self.num_classes = metadata["num_classes"]
        self.device = torch.device(metadata["device"])
        self.model.to(self.device)
        self.is_trained = True
        
        print(f"MLP model loaded from: {model_path}")
        print(f"Grid size: {self.grid_size}, Hidden size: {self.hidden_size}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": "MLP",
            "grid_size": self.grid_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "is_trained": self.is_trained,
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        } 