#!/usr/bin/env python3
"""
Supervised Learning v0.01 - Neural Network Agents
--------------------

Neural network agents for supervised learning v0.01, focusing on PyTorch implementations.
Extends BaseAgent from Task-0, demonstrating perfect base class reuse.

Design Pattern: Strategy Pattern + Template Method
- Different neural network architectures as strategies
- Template method for consistent training and prediction interface
- Simple structure focused on proof of concept
"""

import sys
import os
from typing import Dict, Any
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from core.game_agents import SnakeAgent
from extensions.common.path_utils import setup_extension_paths
from extensions.common.csv_schema import TabularFeatureExtractor, get_schema_info
from extensions.common.model_utils import save_model_standardized, load_model_standardized
setup_extension_paths()


class BaseNeuralAgent(SnakeAgent):
    """
    Base class for all neural network agents in supervised learning v0.01.
    
    Extends BaseAgent to demonstrate perfect base class reuse.
    Provides common interface for all neural network architectures.
    
    Design Pattern: Template Method
    - Defines common interface for all neural agents
    - Implements shared functionality (feature extraction, validation)
    - Allows subclasses to implement specific architectures
    """
    
    def __init__(self, grid_size: int = 10):
        """
        Initialize base neural agent.
        
        Args:
            grid_size: Size of the game grid
        """
        super().__init__()
        
        # Agent metadata
        self.name = self.__class__.__name__
        self.description = "Base neural network agent"
        self.algorithm_name = "Neural Network"
        
        # Grid and feature configuration
        self.grid_size = grid_size
        self.feature_extractor = TabularFeatureExtractor()
        
        # Model state
        self.is_trained = False
        self.model = None
        
        # Get feature information from schema
        schema_info = get_schema_info(grid_size)
        self.input_size = schema_info["feature_columns"]
        self.feature_names = schema_info["feature_names"]
        
        print(f"Initialized {self.name} with {self.input_size} input features")
    
    def get_move(self, game_state: Dict[str, Any]) -> str:
        """
        Get the next move using neural network prediction.
        
        Extends base decision making with neural network inference.
        Demonstrates how supervised learning agents make decisions.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Next move direction (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        if not self.is_trained:
            return "NO_PATH_FOUND"
        
        try:
            # Extract features from game state
            features = self.feature_extractor.extract_features(game_state, self.grid_size)
            
            # Convert features to model input format
            feature_vector = self._features_to_vector(features)
            
            # Make prediction
            prediction = self._predict(feature_vector)
            
            # Convert prediction to move
            moves = ["UP", "DOWN", "LEFT", "RIGHT"]
            move_idx = int(prediction)
            
            if 0 <= move_idx < len(moves):
                return moves[move_idx]
            else:
                return "NO_PATH_FOUND"
                
        except Exception as e:
            print(f"Neural network prediction error: {e}")
            return "NO_PATH_FOUND"
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features dictionary to numpy vector in the correct order.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Numpy array with features in the correct order
        """
        # Create feature vector in the correct order
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                # Default value if feature is missing
                feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _predict(self, features: np.ndarray) -> int:
        """
        Make prediction using the neural network model.
        
        Abstract method to be implemented by subclasses.
        
        Args:
            features: Input feature vector
            
        Returns:
            Predicted class index
        """
        raise NotImplementedError("Subclasses must implement _predict method")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Abstract method to be implemented by subclasses.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def save_model(self, filepath: str, export_onnx: bool = False) -> None:
        """
        Save the trained model with full metadata for cross-platform and time-proofing.
        Args:
            filepath: Path to save model
            export_onnx: If True, also export ONNX format
        """
        
        # Use standardized saving with proper directory structure
        training_params = {
            'hidden_size': getattr(self, 'hidden_size', None),
            'is_trained': self.is_trained
        }
        
        # Extract model name from filepath
        model_name = Path(filepath).stem
        
        # Use common utility for standardized saving
        saved_path = save_model_standardized(
            model=self.model,
            framework='PyTorch',
            grid_size=self.grid_size,
            model_name=model_name,
            model_class=self.__class__.__name__,
            input_size=self.input_size,
            output_size=4,  # 4 directions
            training_params=training_params,
            export_onnx=export_onnx
        )
        
        print(f"Model saved using standardized format: {saved_path}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and check grid_size/metadata.
        Args:
            filepath: Path to load model from
        """
        
        # Use standardized loading
        loaded_model = load_model_standardized(
            filepath, 'PyTorch', self.__class__, 
            grid_size=self.grid_size, hidden_size=self.hidden_size
        )
        
        # Extract state dicts for LSTM components
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if self.lstm and checkpoint.get('lstm_state_dict'):
            self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        if self.fc and checkpoint.get('fc_state_dict'):
            self.fc.load_state_dict(checkpoint['fc_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.grid_size = checkpoint.get('grid_size', 10)
        self.is_trained = checkpoint.get('is_trained', False)
        
        metadata = checkpoint.get('metadata', {})
        loaded_grid_size = metadata.get('grid_size', None)
        if loaded_grid_size is not None and loaded_grid_size != self.grid_size:
            print(f"Warning: Loaded model grid_size {loaded_grid_size} != current {self.grid_size}")
        
        print(f"Model loaded from: {filepath}\nMetadata: {metadata}")


class MLPAgent(BaseNeuralAgent):
    """
    Multi-Layer Perceptron agent for tabular feature data.
    
    Simple feedforward neural network for learning from engineered features.
    Demonstrates basic neural network implementation for supervised learning.
    
    Design Pattern: Strategy Pattern
    - Concrete implementation of neural network strategy
    - Uses PyTorch for implementation
    - Focuses on tabular data processing
    """
    
    def __init__(self, grid_size: int = 10, hidden_size: int = 256):
        """
        Initialize MLP agent.
        
        Args:
            grid_size: Size of the game grid
            hidden_size: Size of hidden layers
        """
        super().__init__(grid_size)
        
        self.name = "MLP Agent"
        self.description = "Multi-Layer Perceptron for tabular data"
        self.hidden_size = hidden_size
        
        # Initialize PyTorch model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PyTorch MLP model."""
        try:
            import torch
            import torch.nn as nn
            
            # Create simple MLP architecture
            self.model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size // 2, 4)  # 4 output classes
            )
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
            
            print(f"Initialized MLP model with {self.input_size} inputs")
            
        except ImportError:
            print("PyTorch not available, using placeholder model")
            self.model = None
    
    def _predict(self, features: np.ndarray) -> int:
        """
        Make prediction using the MLP model.
        
        Args:
            features: Input feature vector
            
        Returns:
            Predicted class index
        """
        if self.model is None:
            return 0  # Default prediction
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                # Convert to tensor
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # Make prediction
                outputs = self.model(features_tensor)
                _, predicted = torch.max(outputs, 1)
        
                return predicted.item()
                
        except Exception as e:
            print(f"MLP prediction error: {e}")
            return 0
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Train the MLP model.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            print("Model not available, skipping training")
            return {"error": "Model not available"}
        
        try:
            import torch
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # Training loop
            self.model.train()
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
            self.is_trained = True
            
            return {
                "final_loss": losses[-1],
                "epochs": epochs,
                "model_type": "MLP"
            }
            
        except Exception as e:
            print(f"MLP training error: {e}")
            return {"error": str(e)}


class CNNAgent(BaseNeuralAgent):
    """
    Convolutional Neural Network agent for board data.
    
    Processes the game board as a 2D image for spatial pattern recognition.
    Demonstrates CNN implementation for supervised learning.
    
    Design Pattern: Strategy Pattern
    - Concrete implementation of neural network strategy
    - Uses PyTorch for CNN implementation
    - Focuses on spatial data processing
    """
    
    def __init__(self, grid_size: int = 10):
        """
        Initialize CNN agent.
        
        Args:
            grid_size: Size of the game grid
        """
        super().__init__(grid_size)
        
        self.name = "CNN Agent"
        self.description = "Convolutional Neural Network for board data"
        
        # Initialize PyTorch CNN model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PyTorch CNN model."""
        try:
            import torch
            import torch.nn as nn
            
            # Simple CNN architecture for board processing
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels: snake, apple, empty
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(64 * (self.grid_size // 4) ** 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 4)  # 4 output classes
            )
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
            
            print(f"Initialized CNN model for {self.grid_size}x{self.grid_size} board")
            
        except ImportError:
            print("PyTorch not available, using placeholder model")
            self.model = None
    
    def _predict(self, features: np.ndarray) -> int:
        """
        Make prediction using the CNN model.
        
        Args:
            features: Input feature vector (will be converted to board format)
            
        Returns:
            Predicted class index
        """
        if self.model is None:
            return 0  # Default prediction
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                # Convert features to board format (simplified)
                board_features = self._features_to_board(features)
                board_tensor = torch.FloatTensor(board_features).unsqueeze(0)
                
                # Make prediction
                outputs = self.model(board_tensor)
                _, predicted = torch.max(outputs, 1)
                
                return predicted.item()
            
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return 0
    
    def _features_to_board(self, features: np.ndarray) -> np.ndarray:
        """
        Convert feature vector to board format for CNN.
        
        Args:
            features: Input feature vector
            
        Returns:
            Board representation as 3-channel image
        """
        # Simplified conversion - in practice, this would use actual board state
        board = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Use first few features as board representation
        if len(features) >= 3:
            board[0, 0, 0] = features[0]  # Snake channel
            board[1, 0, 0] = features[1]  # Apple channel
            board[2, 0, 0] = features[2]  # Empty channel
        
        return board
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            print("Model not available, skipping training")
            return {"error": "Model not available"}
        
        try:
            import torch
            
            # Convert features to board format
            X_board = np.array([self._features_to_board(x) for x in X])
            X_tensor = torch.FloatTensor(X_board)
            y_tensor = torch.LongTensor(y)
            
            # Training loop
            self.model.train()
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
            self.is_trained = True
            
            return {
                "final_loss": losses[-1],
                "epochs": epochs,
                "model_type": "CNN"
            }
            
        except Exception as e:
            print(f"CNN training error: {e}")
            return {"error": str(e)}


class LSTMAgent(BaseNeuralAgent):
    """
    Long Short-Term Memory agent for sequential data.
    
    Processes game state as a sequence for temporal pattern recognition.
    Demonstrates RNN implementation for supervised learning.
    
    Design Pattern: Strategy Pattern
    - Concrete implementation of neural network strategy
    - Uses PyTorch for LSTM implementation
    - Focuses on sequential data processing
    """
    
    def __init__(self, grid_size: int = 10, hidden_size: int = 128):
        """
        Initialize LSTM agent.
        
        Args:
            grid_size: Size of the game grid
            hidden_size: Size of LSTM hidden layers
        """
        super().__init__(grid_size)
        
        self.name = "LSTM Agent"
        self.description = "Long Short-Term Memory for sequential data"
        self.hidden_size = hidden_size
        
        # Initialize PyTorch LSTM model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn
            
            # Simple LSTM architecture for sequence processing
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, 4)  # 4 output classes
            
            self.optimizer = torch.optim.Adam(list(self.lstm.parameters()) + list(self.fc.parameters()), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
            
            print(f"Initialized LSTM model with {self.input_size} input features")
            
        except ImportError:
            print("PyTorch not available, using placeholder model")
            self.lstm = None
            self.fc = None
    
    def _predict(self, features: np.ndarray) -> int:
        """
        Make prediction using the LSTM model.
        
        Args:
            features: Input feature vector
        
        Returns:
            Predicted class index
        """
        if self.lstm is None or self.fc is None:
            return 0  # Default prediction
        
        try:
            import torch
            
            self.lstm.eval()
            self.fc.eval()
            with torch.no_grad():
                # Convert to tensor and add sequence dimension
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, features)
                
                # LSTM forward pass
                lstm_out, _ = self.lstm(features_tensor)
                
                # Final prediction
                outputs = self.fc(lstm_out[:, -1, :])
                _, predicted = torch.max(outputs, 1)
                
                return predicted.item()
                
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return 0
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        if self.lstm is None or self.fc is None:
            print("Model not available, skipping training")
            return {"error": "Model not available"}
        
        try:
            import torch
            
            # Convert to tensors with sequence dimension
            X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (batch, 1, features)
            y_tensor = torch.LongTensor(y)
            
            # Training loop
            self.lstm.train()
            self.fc.train()
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                lstm_out, _ = self.lstm(X_tensor)
                outputs = self.fc(lstm_out[:, -1, :])
                loss = self.criterion(outputs, y_tensor)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
            self.is_trained = True
            
            return {
                "final_loss": losses[-1],
                "epochs": epochs,
                "model_type": "LSTM"
            }
            
        except Exception as e:
            print(f"LSTM training error: {e}")
            return {"error": str(e)}
    
    def save_model(self, filepath: str, export_onnx: bool = False) -> None:
        """
        Save the trained model with full metadata for cross-platform and time-proofing.
        Args:
            filepath: Path to save model
            export_onnx: If True, also export ONNX format
        """
        
        # Use standardized saving with proper directory structure
        training_params = {
            'hidden_size': getattr(self, 'hidden_size', None),
            'is_trained': self.is_trained
        }
        
        # Extract model name from filepath
        model_name = Path(filepath).stem
        
        # For LSTM, we need to save both lstm and fc components
        import torch
        combined_state_dict = {
            'lstm_state_dict': self.lstm.state_dict() if self.lstm else None,
            'fc_state_dict': self.fc.state_dict() if self.fc else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }
        
        # Create a temporary model wrapper for saving
        class ModelWrapper(torch.nn.Module):
            def __init__(self, lstm, fc):
                super().__init__()
                self.lstm = lstm
                self.fc = fc
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
            
            def state_dict(self):
                return combined_state_dict
        
        temp_model = ModelWrapper(self.lstm, self.fc)
        
        # Use common utility for standardized saving
        saved_path = save_model_standardized(
            model=temp_model,
            framework='PyTorch',
            grid_size=self.grid_size,
            model_name=model_name,
            model_class=self.__class__.__name__,
            input_size=self.input_size,
            output_size=4,  # 4 directions
            training_params=training_params,
            export_onnx=export_onnx
        )
        
        print(f"Model saved using standardized format: {saved_path}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and check grid_size/metadata.
        Args:
            filepath: Path to load model from
        """
        
        # Use standardized loading
        loaded_model = load_model_standardized(
            filepath, 'PyTorch', self.__class__, 
            grid_size=self.grid_size, hidden_size=self.hidden_size
        )
        
        # Extract state dicts for LSTM components
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if self.lstm and checkpoint.get('lstm_state_dict'):
            self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        if self.fc and checkpoint.get('fc_state_dict'):
            self.fc.load_state_dict(checkpoint['fc_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.grid_size = checkpoint.get('grid_size', 10)
        self.is_trained = checkpoint.get('is_trained', False)
        
        metadata = checkpoint.get('metadata', {})
        loaded_grid_size = metadata.get('grid_size', None)
        if loaded_grid_size is not None and loaded_grid_size != self.grid_size:
            print(f"Warning: Loaded model grid_size {loaded_grid_size} != current {self.grid_size}")
        
        print(f"Model loaded from: {filepath}\nMetadata: {metadata}") 