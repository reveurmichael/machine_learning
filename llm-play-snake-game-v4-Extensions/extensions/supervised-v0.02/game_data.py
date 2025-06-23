"""
Supervised Learning v0.02 - Game Data
--------------------

Game data for supervised learning v0.02, supporting all ML model types.
Extends BaseGameData from Task-0, demonstrating perfect base class reuse.

Design Pattern: Template Method
- Extends BaseGameData for consistent data management
- Implements multi-model-specific data tracking
- Organized structure supporting neural, tree, and graph models
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from core.game_data import BaseGameData
from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()


class SupervisedGameData(BaseGameData):
    """
    Game data for supervised learning v0.02.
    
    Extends BaseGameData to demonstrate perfect base class reuse.
    Supports all ML model types (neural, tree, graph) for comprehensive data tracking.
    
    Design Pattern: Template Method
    - Inherits core data management from BaseGameData
    - Implements multi-model-specific data tracking
    - Demonstrates how extensions can reuse core infrastructure
    """
    
    def __init__(self, grid_size: int = 10):
        """
        Initialize supervised learning game data.
        
        Args:
            grid_size: Size of the game grid
        """
        # Call parent constructor to inherit all base functionality
        super().__init__(grid_size=grid_size)
        
        # Supervised learning specific data tracking
        self.model_predictions = []
        self.model_confidence = []
        self.model_features = []
        self.model_category = None
        self.model_framework = None
        
        print(f"Initialized SupervisedGameData for grid size {grid_size}")
    
    def record_model_prediction(self, prediction: str, confidence: float = None, 
                              features: Dict[str, Any] = None):
        """
        Record a model prediction for analysis.
        
        Extends base data tracking with model-specific information.
        
        Args:
            prediction: Model's predicted move
            confidence: Model's confidence in the prediction
            features: Features used for the prediction
        """
        self.model_predictions.append(prediction)
        self.model_confidence.append(confidence)
        self.model_features.append(features)
    
    def set_model_info(self, model_type: str, model_category: str, framework: str):
        """
        Set model information for tracking.
        
        Args:
            model_type: Type of model being used
            model_category: Category of model (Neural, Tree, Graph)
            framework: Framework used by the model
        """
        self.model_category = model_category
        self.model_framework = framework
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get model-specific statistics.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.model_predictions:
            return {}
        
        # Calculate prediction distribution
        move_counts = {}
        for pred in self.model_predictions:
            move_counts[pred] = move_counts.get(pred, 0) + 1
        
        # Calculate average confidence
        avg_confidence = None
        if self.model_confidence and any(c is not None for c in self.model_confidence):
            valid_confidences = [c for c in self.model_confidence if c is not None]
            if valid_confidences:
                avg_confidence = sum(valid_confidences) / len(valid_confidences)
        
        return {
            "total_predictions": len(self.model_predictions),
            "prediction_distribution": move_counts,
            "average_confidence": avg_confidence,
            "model_category": self.model_category,
            "model_framework": self.model_framework
        }
    
    def save_game_data(self, filepath: str) -> None:
        """
        Save game data including model-specific information.
        
        Extends base save functionality with model data.
        
        Args:
            filepath: Path to save the game data
        """
        # Get base game data
        base_data = self.get_game_data()
        
        # Add supervised learning specific data
        supervised_data = {
            **base_data,
            "model_statistics": self.get_model_statistics(),
            "model_predictions": self.model_predictions,
            "model_confidence": self.model_confidence,
            "extension_type": "supervised_learning",
            "version": "v0.02"
        }
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(supervised_data, f, indent=2, default=str)
        
        print(f"Supervised learning game data saved to: {filepath}")
    
    def get_game_data(self) -> Dict[str, Any]:
        """
        Get complete game data including model information.
        
        Returns:
            Dictionary with all game data
        """
        # Get base game data
        base_data = super().get_game_data()
        
        # Add supervised learning specific data
        supervised_data = {
            **base_data,
            "model_statistics": self.get_model_statistics(),
            "extension_type": "supervised_learning",
            "version": "v0.02"
        }
        
        return supervised_data
    
    def reset(self) -> None:
        """
        Reset game data for a new game.
        
        Extends base reset with model-specific cleanup.
        """
        # Reset base data
        super().reset()
        
        # Reset supervised learning specific data
        self.model_predictions = []
        self.model_confidence = []
        self.model_features = []
        # Keep model info as it doesn't change between games 