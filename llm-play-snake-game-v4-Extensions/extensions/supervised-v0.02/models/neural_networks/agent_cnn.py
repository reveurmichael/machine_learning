"""
CNNAgent for Supervised Learning v0.02
======================================

Implements a Convolutional Neural Network agent using PyTorch for board data.
Processes the game board as a 2D image for spatial pattern recognition.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from .agent_mlp import BaseMLAgent
from extensions.common.path_utils import setup_extension_paths
from extensions.common.csv_schema import TabularFeatureExtractor, get_schema_info
setup_extension_paths()


class CNNAgent(BaseMLAgent):
    """
    Convolutional Neural Network agent for board state data.
    
    Processes the game board as a 2D image, learning spatial patterns
    that are important for navigation and planning.
    """
    
    def __init__(self, grid_size: int = 10, num_classes: int = 4):
        """
        Initialize CNN agent.
        
        Args:
            grid_size: Size of the game board (assumes square board)
            num_classes: Number of output classes (4 for UP, DOWN, LEFT, RIGHT)
        """
        super().__init__()
        self.name = "CNN Agent"
        self.description = "Convolutional Neural Network for board data"
        self.model_name = "CNN"
        
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        # Calculate input size for the linear layer after convolutions
        conv_output_size = self._calculate_conv_output_size(grid_size)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 channels: snake, apple, empty
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def _calculate_conv_output_size(self, board_size: int) -> int:
        """Calculate the output size after convolutions."""
        # After conv1 + pool: size // 2
        # After conv2 + pool: size // 4
        # After conv3: same size (no pooling)
        size_after_pools = board_size // 4
        return 128 * size_after_pools * size_after_pools
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN."""
        # Reshape input to board format if needed
        if x.dim() == 2:
            batch_size = x.size(0)
            # Assume input is flattened board + additional features
            board_features = x[:, :self.grid_size * self.grid_size * 3]
            x = board_features.view(batch_size, 3, self.grid_size, self.grid_size)
        
        # Convolutional layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def _extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from game state for CNN prediction.
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Feature vector as numpy array (board state + additional features)
        """
        # Get basic state information
        head_pos = game_state.get("head_position", [0, 0])
        apple_pos = game_state.get("apple_position", [0, 0])
        snake_positions = game_state.get("snake_positions", [])
        grid_size = game_state.get("grid_size", self.grid_size)
        
        # Create 3-channel board representation
        # Channel 0: Snake body (1 for snake, 0 for empty)
        # Channel 1: Apple position (1 for apple, 0 for empty)
        # Channel 2: Snake head (1 for head, 0 for empty)
        board_channels = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        
        # Fill snake body channel
        for pos in snake_positions:
            if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                board_channels[0, pos[1], pos[0]] = 1.0
        
        # Fill apple channel
        if 0 <= apple_pos[0] < grid_size and 0 <= apple_pos[1] < grid_size:
            board_channels[1, apple_pos[1], apple_pos[0]] = 1.0
        
        # Fill head channel
        if 0 <= head_pos[0] < grid_size and 0 <= head_pos[1] < grid_size:
            board_channels[2, head_pos[1], head_pos[0]] = 1.0
        
        # Flatten board channels
        board_features = board_channels.flatten()
        
        # Get additional features using common utilities
        additional_features = self.feature_extractor.extract_features(game_state, grid_size)
        additional_features_vector = self._features_to_vector(additional_features)
        
        # Combine board features with additional features
        features = np.concatenate([board_features, additional_features_vector])
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001, **kwargs) -> Dict[str, Any]:
        """
        Train the CNN model.
        
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
        expected_features = self.grid_size * self.grid_size * 3 + get_schema_info(self.grid_size)["feature_columns"]
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Training loop
        self.train()
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(X_tensor)
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
        expected_features = self.grid_size * self.grid_size * 3 + get_schema_info(self.grid_size)["feature_columns"]
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grid_size': self.grid_size,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.grid_size = checkpoint.get('grid_size', 10)
        self.is_trained = checkpoint['is_trained'] 