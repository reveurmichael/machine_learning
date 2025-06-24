"""
LightGBM Agent for Supervised Learning v0.03
--------------------

LightGBM gradient boosting agent for tabular feature data.
Implements standardized model saving/loading with text format.

Design Pattern: Strategy Pattern
- Implements BaseAgent interface
- Configurable hyperparameters
- Standardized training and prediction
- Text model format for cross-platform compatibility
"""

import lightgbm as lgb
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any

from core.game_agents import BaseAgent
from extensions.common.path_utils import setup_extension_paths
from extensions.common.csv_schema import TabularFeatureExtractor, get_schema_info
from extensions.common.model_utils import get_model_directory
setup_extension_paths()


class LightGBMAgent(BaseAgent):
    """
    LightGBM gradient boosting agent for Snake game.
    
    Design Pattern: Strategy Pattern
    - Implements BaseAgent interface
    - Configurable hyperparameters
    - Standardized training and prediction
    """
    
    def __init__(self, grid_size: int = 10, num_leaves: int = 31, learning_rate: float = 0.1,
                 n_estimators: int = 100, **kwargs):
        """
        Initialize LightGBM agent.
        
        Args:
            grid_size: Size of the game grid
            num_leaves: Number of leaves in trees
            learning_rate: Learning rate for gradient boosting
            n_estimators: Number of boosting rounds
            **kwargs: Additional LightGBM parameters
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        # Get input size from schema
        schema_info = get_schema_info(grid_size)
        self.input_size = schema_info["feature_columns"]
        
        # Initialize LightGBM model
        self.model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='multiclass',
            num_class=4,
            random_state=42,
            verbose=-1,  # Suppress verbose output
            **kwargs
        )
        
        # Initialize components
        self.feature_extractor = TabularFeatureExtractor()
        
        # Training state
        self.is_trained = False
        
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
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features_array)[0]
        
        # Convert to direction
        return self.direction_map[prediction]
    
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
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            
        Returns:
            Probability predictions of shape (n_samples, num_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the LightGBM model.
        
        Args:
            X: Training features of shape (n_samples, input_size)
            y: Training labels of shape (n_samples,)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training metrics
        """
        # Split into train/validation
        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        print("Training LightGBM model...")
        print(f"Grid size: {self.grid_size}, Input size: {self.input_size}")
        print(f"Num leaves: {self.num_leaves}, Learning rate: {self.learning_rate}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        # Mark as trained
        self.is_trained = True
        
        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val)
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        print("Training completed!")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Return training metrics
        return {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "feature_importance": feature_importance.tolist(),
            "best_iteration": self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else self.n_estimators,
            "model_params": self.model.get_params()
        }
    
    def save_model(self, model_name: str) -> str:
        """
        Save the trained model with metadata in text format.
        
        Args:
            model_name: Name for the saved model
            
        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Get model directory
        model_dir = get_model_directory("lightgbm", self.grid_size)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model in text format
        model_file = model_dir / f"{model_name}.txt"
        self.model.booster_.save_model(str(model_file))
        
        # Prepare metadata
        metadata = {
            "model_type": "LightGBM",
            "grid_size": self.grid_size,
            "input_size": self.input_size,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "lightgbm_version": lgb.__version__,
            "timestamp": datetime.utcnow().isoformat(),
            "model_architecture": "Gradient Boosting",
            "framework": "lightgbm",
            "model_params": self.model.get_params()
        }
        
        # Save metadata
        metadata_file = model_dir / f"{model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"LightGBM model saved to: {model_file}")
        print(f"Metadata saved to: {metadata_file}")
        return str(model_file)
    
    def load_model(self, model_name: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the saved model (without extension)
        """
        # Get model directory
        model_dir = get_model_directory("lightgbm", self.grid_size)
        
        # Load LightGBM model
        model_file = model_dir / f"{model_name}.txt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load the model
        booster = lgb.Booster(model_file=str(model_file))
        
        # Create new classifier and set the booster
        self.model = lgb.LGBMClassifier()
        self.model.booster_ = booster
        
        # Load metadata
        metadata_file = model_dir / f"{model_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Update agent state
            self.grid_size = metadata["grid_size"]
            self.input_size = metadata["input_size"]
            self.num_leaves = metadata["num_leaves"]
            self.learning_rate = metadata["learning_rate"]
            self.n_estimators = metadata["n_estimators"]
        
        self.is_trained = True
        
        print(f"LightGBM model loaded from: {model_file}")
        print(f"Grid size: {self.grid_size}, Num leaves: {self.num_leaves}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "model_type": "LightGBM",
            "grid_size": self.grid_size,
            "input_size": self.input_size,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "is_trained": self.is_trained,
            "lightgbm_version": lgb.__version__
        }
        
        if self.is_trained:
            info["feature_importance"] = self.model.feature_importances_.tolist()
            info["best_iteration"] = getattr(self.model, 'best_iteration_', self.n_estimators)
        
        return info
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        # Get feature names from schema
        schema_info = get_schema_info(self.grid_size)
        feature_names = schema_info["feature_names"]
        
        # Get importance scores
        importance_scores = self.model.feature_importances_
        
        # Create mapping
        feature_importance = dict(zip(feature_names, importance_scores))
        
        return feature_importance 