"""
CSV Schema Utilities for Snake Game Extensions
=============================================

Provides flexible CSV schema generation and feature extraction utilities
that work with different grid sizes. Used by both heuristics (for dataset generation)
and supervised learning (for training) extensions.

This module implements the exact CSV schema specified in the documentation:
- game_id, step_in_game (metadata, not used as features)
- head_x, head_y, apple_x, apple_y, snake_length (basic state)
- apple_dir_up, apple_dir_down, apple_dir_left, apple_dir_right (relative direction)
- danger_straight, danger_left, danger_right (immediate collision detection)
- free_space_up, free_space_down, free_space_left, free_space_right (maneuvering space)
- target_move (supervised learning label)

Design Pattern: Strategy Pattern
- Different feature extraction strategies for different model types
- Pluggable schema generation for different grid sizes
- Consistent interface across all extensions
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CSVSchema:
    """
    Represents a CSV schema for Snake game data.
    
    Contains column definitions and metadata for different grid sizes.
    Follows the exact specification from the documentation.
    """
    grid_size: int
    feature_columns: List[str]
    target_column: str
    metadata_columns: List[str]
    total_columns: int
    
    def get_feature_count(self) -> int:
        """Get the number of feature columns (excluding metadata and target)."""
        return len(self.feature_columns)
    
    def get_column_names(self) -> List[str]:
        """Get all column names in the correct order."""
        return self.metadata_columns + self.feature_columns + [self.target_column]


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction strategies."""
    
    @abstractmethod
    def extract_features(self, game_state: Dict[str, Any], grid_size: int) -> Dict[str, Any]:
        """
        Extract features from game state.
        
        Args:
            game_state: Game state dictionary
            grid_size: Size of the game grid
            
        Returns:
            Dictionary mapping feature names to values
        """
        pass


class TabularFeatureExtractor(FeatureExtractor):
    """
    Extracts tabular features for traditional ML models (XGBoost, LightGBM, MLP).
    
    Creates engineered features optimized for gradient boosting and neural networks.
    Follows the exact CSV schema specification from the documentation.
    """
    
    def extract_features(self, game_state: Dict[str, Any], grid_size: int) -> Dict[str, Any]:
        """
        Extract tabular features from game state.
        
        Args:
            game_state: Game state dictionary
            grid_size: Size of the game grid
            
        Returns:
            Dictionary with all feature values following the exact schema specification
        """
        # Extract basic state information
        head_pos = game_state.get("head_position", [0, 0])
        apple_pos = game_state.get("apple_position", [0, 0])
        snake_positions = game_state.get("snake_positions", [])
        current_direction = game_state.get("current_direction", "UP")
        snake_length = game_state.get("snake_length", 1)
        
        # Convert snake positions to set for efficient lookup
        snake_body = set(tuple(pos) for pos in snake_positions)
        
        # Basic position features (exact schema specification)
        features = {
            "head_x": head_pos[0],
            "head_y": head_pos[1],
            "apple_x": apple_pos[0],
            "apple_y": apple_pos[1],
            "snake_length": snake_length,
        }
        
        # Apple direction features (relative to head) - exact schema specification
        features.update({
            "apple_dir_up": 1 if apple_pos[1] < head_pos[1] else 0,
            "apple_dir_down": 1 if apple_pos[1] > head_pos[1] else 0,
            "apple_dir_left": 1 if apple_pos[0] < head_pos[0] else 0,
            "apple_dir_right": 1 if apple_pos[0] > head_pos[0] else 0,
        })
        
        # Danger detection features (relative to current direction) - exact schema specification
        danger_features = self._calculate_danger_features(
            head_pos, current_direction, snake_body, grid_size
        )
        features.update(danger_features)
        
        # Free space features - exact schema specification
        free_space_features = self._calculate_free_space_features(
            head_pos, snake_body, grid_size
        )
        features.update(free_space_features)
        
        return features
    
    def _calculate_danger_features(self, head_pos: List[int], current_direction: str, 
                                 snake_body: set, grid_size: int) -> Dict[str, int]:
        """
        Calculate danger features based on current direction.
        
        Returns danger flags for straight, left, and right relative to current heading.
        Follows the exact schema specification.
        """
        head_x, head_y = head_pos
        
        # Define direction vectors
        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
        
        # Get current direction vector
        dx, dy = directions.get(current_direction, (0, 0))
        
        # Calculate positions for straight, left, and right relative to current direction
        # For left turn: rotate 90 degrees counterclockwise
        # For right turn: rotate 90 degrees clockwise
        if current_direction == "UP":
            left_pos = (head_x - 1, head_y)
            right_pos = (head_x + 1, head_y)
        elif current_direction == "DOWN":
            left_pos = (head_x + 1, head_y)
            right_pos = (head_x - 1, head_y)
        elif current_direction == "LEFT":
            left_pos = (head_x, head_y + 1)
            right_pos = (head_x, head_y - 1)
        elif current_direction == "RIGHT":
            left_pos = (head_x, head_y - 1)
            right_pos = (head_x, head_y + 1)
        else:
            left_pos = right_pos = (head_x, head_y)
        
        straight_pos = (head_x + dx, head_y + dy)
        
        # Check if positions are dangerous (wall or snake body)
        def is_dangerous(pos):
            x, y = pos
            return (x < 0 or x >= grid_size or y < 0 or y >= grid_size or 
                   pos in snake_body)
        
        return {
            "danger_straight": 1 if is_dangerous(straight_pos) else 0,
            "danger_left": 1 if is_dangerous(left_pos) else 0,
            "danger_right": 1 if is_dangerous(right_pos) else 0,
        }
    
    def _calculate_free_space_features(self, head_pos: List[int], snake_body: set, 
                                     grid_size: int) -> Dict[str, int]:
        """
        Calculate free space in each direction.
        
        Counts how many free squares exist in each cardinal direction before hitting
        a wall or snake body. Follows the exact schema specification.
        """
        head_x, head_y = head_pos
        
        def count_free_space(start_x: int, start_y: int, dx: int, dy: int) -> int:
            """Count free spaces in a given direction."""
            count = 0
            x, y = start_x + dx, start_y + dy
            
            while 0 <= x < grid_size and 0 <= y < grid_size and (x, y) not in snake_body:
                count += 1
                x += dx
                y += dy
            
            return count
        
        return {
            "free_space_up": count_free_space(head_x, head_y, 0, -1),
            "free_space_down": count_free_space(head_x, head_y, 0, 1),
            "free_space_left": count_free_space(head_x, head_y, -1, 0),
            "free_space_right": count_free_space(head_x, head_y, 1, 0),
        }


def generate_csv_schema(grid_size: int) -> CSVSchema:
    """
    Generate CSV schema for a given grid size.
    
    This function implements the exact CSV schema specification from the documentation.
    
    Args:
        grid_size: Size of the game grid (e.g., 8, 10, 12, 16, 20)
        
    Returns:
        CSVSchema object with column definitions following the exact specification
    """
    # Metadata columns (not used as features) - exact specification
    metadata_columns = ["game_id", "step_in_game"]
    
    # Feature columns (used as model inputs) - exact specification
    feature_columns = [
        "head_x", "head_y", "apple_x", "apple_y", "snake_length",
        "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
        "danger_straight", "danger_left", "danger_right",
        "free_space_up", "free_space_down", "free_space_left", "free_space_right"
    ]
    
    # Target column (what we want to predict) - exact specification
    target_column = "target_move"
    
    total_columns = len(metadata_columns) + len(feature_columns) + 1
    
    return CSVSchema(
        grid_size=grid_size,
        feature_columns=feature_columns,
        target_column=target_column,
        metadata_columns=metadata_columns,
        total_columns=total_columns
    )


def validate_game_state(game_state: Dict[str, Any], grid_size: int) -> bool:
    """
    Validate that a game state is compatible with the given grid size.
    
    Args:
        game_state: Game state dictionary
        grid_size: Expected grid size
        
    Returns:
        True if valid, False otherwise
    """
    try:
        head_pos = game_state.get("head_position", [])
        apple_pos = game_state.get("apple_position", [])
        
        if len(head_pos) != 2 or len(apple_pos) != 2:
            return False
        
        head_x, head_y = head_pos
        apple_x, apple_y = apple_pos
        
        # Check bounds
        if not (0 <= head_x < grid_size and 0 <= head_y < grid_size):
            return False
        
        if not (0 <= apple_x < grid_size and 0 <= apple_y < grid_size):
            return False
        
        return True
        
    except Exception:
        return False


def create_csv_row(game_state: Dict[str, Any], target_move: str, game_id: int, 
                  step_in_game: int, grid_size: int) -> Dict[str, Any]:
    """
    Create a CSV row from game state and target move.
    
    This function creates rows following the exact CSV schema specification.
    
    Args:
        game_state: Game state dictionary
        target_move: Target move (UP, DOWN, LEFT, RIGHT)
        game_id: Game session ID
        step_in_game: Step number in the game
        grid_size: Size of the game grid
        
    Returns:
        Dictionary representing a CSV row following the exact schema specification
    """
    # Validate inputs
    if not validate_game_state(game_state, grid_size):
        raise ValueError(f"Invalid game state for grid size {grid_size}")
    
    if target_move not in ["UP", "DOWN", "LEFT", "RIGHT"]:
        raise ValueError(f"Invalid target move: {target_move}")
    
    # Extract features using the exact schema specification
    extractor = TabularFeatureExtractor()
    features = extractor.extract_features(game_state, grid_size)
    
    # Create row following the exact schema specification
    row = {
        "game_id": game_id,
        "step_in_game": step_in_game,
        **features,
        "target_move": target_move
    }
    
    return row


def get_schema_info(grid_size: int) -> Dict[str, Any]:
    """
    Get information about the CSV schema for a given grid size.
    
    Args:
        grid_size: Size of the game grid
        
    Returns:
        Dictionary with schema information
    """
    schema = generate_csv_schema(grid_size)
    
    return {
        "grid_size": grid_size,
        "total_columns": schema.total_columns,
        "feature_columns": len(schema.feature_columns),
        "metadata_columns": len(schema.metadata_columns),
        "column_names": schema.get_column_names(),
        "feature_names": schema.feature_columns,
        "target_column": schema.target_column
    }


# Example usage and testing
if __name__ == "__main__":
    # Test with different grid sizes
    for grid_size in [8, 10, 12, 16, 20]:
        schema = generate_csv_schema(grid_size)
        info = get_schema_info(grid_size)
        print(f"Grid size {grid_size}: {info['feature_columns']} features, {info['total_columns']} total columns")
        
        # Test feature extraction
        test_state = {
            "head_position": [5, 5],
            "apple_position": [7, 3],
            "snake_positions": [[5, 5], [5, 6], [5, 7]],
            "current_direction": "UP",
            "score": 10,
            "steps": 50,
            "snake_length": 3
        }
        
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(test_state, grid_size)
        print(f"  Features: {len(features)} extracted")
        
        # Test CSV row creation
        row = create_csv_row(test_state, "UP", 1, 1, grid_size)
        print(f"  CSV row: {len(row)} columns")
        print() 