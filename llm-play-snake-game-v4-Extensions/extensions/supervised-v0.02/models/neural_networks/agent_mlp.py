"""
MLPAgent for Supervised Learning v0.02
======================================

Implements a Multi-Layer Perceptron agent using PyTorch for tabular data.
Inherits from BaseAgent and provides train/predict interface.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

from core.game_agents import BaseAgent
from extensions.common.path_utils import setup_extension_paths
from extensions.common.csv_schema import TabularFeatureExtractor, get_schema_info
from extensions.common.model_utils import save_model_standardized, load_model_standardized
setup_extension_paths()


class BaseMLAgent(BaseAgent, ABC):
    """
    Abstract base class for all machine learning agents.
    
    Provides common interface for training and prediction across different
    model types (neural, tree, graph).
    """
    
    def __init__(self, grid_size: int):
        super().__init__()
        self.name = self.__class__.__name__
        self.description = "Base machine learning agent"
        self.model_name = self.__class__.__name__
        self.is_trained = False
        self.grid_size = grid_size
        self.feature_extractor = TabularFeatureExtractor()
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on given data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on given data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted outputs
        """
        pass
    
    def get_move(self, game_state: Dict[str, Any]) -> str:
        """
        Get the next move based on current game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Move direction string or "NO_PATH_FOUND"
        """
        if not self.is_trained:
            return "NO_PATH_FOUND"
        
        try:
            # Extract features from game state using common utilities
            features = self.feature_extractor.extract_features(game_state, self.grid_size)
            
            # Convert features to numpy array in the correct order
            feature_vector = self._features_to_vector(features)
            
            # Make prediction
            prediction = self.predict(feature_vector.reshape(1, -1))
            
            # Convert prediction to move
            moves = ["UP", "DOWN", "LEFT", "RIGHT"]
            move_idx = int(prediction[0])
            
            if 0 <= move_idx < len(moves):
                return moves[move_idx]
            else:
                return "NO_PATH_FOUND"
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return "NO_PATH_FOUND"
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features dictionary to numpy vector in the correct order.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Numpy array with features in the correct order
        """
        # Get schema info to know the correct feature order
        schema_info = get_schema_info(self.grid_size)
        feature_names = schema_info["feature_names"]
        
        # Create feature vector in the correct order
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                # Default value if feature is missing
                feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def set_grid_size(self, grid_size: int) -> None:
        """
        Set the grid size for feature extraction.
        
        Args:
            grid_size: Size of the game grid
        """
        self.grid_size = grid_size


class MLPAgent(BaseMLAgent):
    """
    Multi-Layer Perceptron agent for tabular feature data.
    
    Uses PyTorch to implement a feedforward neural network for
    learning from engineered features extracted from game state.
    """
    
    def __init__(self, grid_size: int = 10, hidden_size: int = 256, num_classes: int = 4):
        """
        Initialize MLP agent.
        
        Args:
            grid_size: Size of the game grid
            hidden_size: Size of hidden layers
            num_classes: Number of output classes (4 for UP, DOWN, LEFT, RIGHT)
        """
        super().__init__(grid_size)
        self.name = "MLP Agent"
        self.description = "Multi-Layer Perceptron for tabular data"
        self.model_name = "MLP"
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Get input size from schema
        schema_info = get_schema_info(grid_size)
        self.input_size = schema_info["feature_columns"]
        
        # Create neural network
        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001, **kwargs) -> Dict[str, Any]:
        """
        Train the MLP model.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # Validate input dimensions
        expected_features = get_schema_info(self.grid_size)["feature_columns"]
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Training loop
        self.model.train()
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).float().mean().item()
            
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")
        
        self.is_trained = True
        
        return {
            "final_loss": train_losses[-1],
            "final_accuracy": train_accuracies[-1],
            "train_losses": train_losses,
            "train_accuracies": train_accuracies
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class indices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate input dimensions
        expected_features = get_schema_info(self.grid_size)["feature_columns"]
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def save_model(self, filepath: str, export_onnx: bool = True) -> None:
        """
        Save the trained model using standardized utility and export ONNX.
        Args:
            filepath: Path to save model
            export_onnx: If True, also export ONNX format
        """
        training_params = {
            'hidden_size': self.hidden_size,
            'is_trained': self.is_trained
        }
        model_name = Path(filepath).stem
        saved_path = save_model_standardized(
            model=self.model,
            framework='PyTorch',
            grid_size=self.grid_size,
            model_name=model_name,
            model_class=self.__class__.__name__,
            input_size=self.input_size,
            output_size=self.num_classes,
            training_params=training_params,
            export_onnx=export_onnx
        )
        print(f"Model saved using standardized format: {saved_path}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model using standardized utility.
        Args:
            filepath: Path to load model from
        """
        loaded_model = load_model_standardized(
            filepath, 'PyTorch', self.__class__, 
            grid_size=self.grid_size, hidden_size=self.hidden_size
        )
        self.model.load_state_dict(loaded_model['model_state'])
        self.optimizer.load_state_dict(loaded_model['optimizer_state'])
        self.is_trained = loaded_model.get('is_trained', False) 