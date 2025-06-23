"""
Supervised Learning v0.02 - Game Logic
--------------------

Game logic for supervised learning v0.02, supporting all ML model types.
Extends BaseGameLogic from Task-0, demonstrating perfect base class reuse.

Design Pattern: Template Method
- Extends BaseGameLogic for consistent game mechanics
- Implements multi-model-specific decision making
- Organized structure supporting neural, tree, and graph models
"""

import sys
import os
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from core.game_logic import BaseGameLogic
from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()


class SupervisedGameLogic(BaseGameLogic):
    """
    Game logic for supervised learning v0.02.
    
    Extends BaseGameLogic to demonstrate perfect base class reuse.
    Supports all ML model types (neural, tree, graph) for comprehensive evaluation.
    
    Design Pattern: Template Method
    - Inherits core game mechanics from BaseGameLogic
    - Implements multi-model-specific decision making
    - Demonstrates how extensions can reuse core infrastructure
    """
    
    def __init__(self, agent, grid_size: int = 10, max_steps: int = 1000):
        """
        Initialize supervised learning game logic.
        
        Args:
            agent: Multi-model agent for decision making
            grid_size: Size of the game grid
            max_steps: Maximum steps per game
        """
        # Call parent constructor to inherit all base functionality
        super().__init__(grid_size=grid_size, max_steps=max_steps)
        
        # Supervised learning specific attributes
        self.agent = agent
        self.model_type = type(agent).__name__
        self.model_category = self._get_model_category()
        
        print(f"Initialized SupervisedGameLogic with {self.model_type} ({self.model_category})")
    
    def _get_model_category(self) -> str:
        """
        Get the category of the current model.
        
        Returns:
            Model category (Neural, Tree, or Graph)
        """
        neural_models = ['MLPAgent', 'CNNAgent', 'LSTMAgent', 'GRUAgent']
        tree_models = ['XGBoostAgent', 'LightGBMAgent', 'RandomForestAgent']
        graph_models = ['GCNAgent', 'GraphSAGEAgent', 'GATAgent']
        
        if self.model_type in neural_models:
            return "Neural"
        elif self.model_type in tree_models:
            return "Tree"
        elif self.model_type in graph_models:
            return "Graph"
        else:
            return "Unknown"
    
    def get_next_move(self, game_state: Dict[str, Any]) -> str:
        """
        Get the next move using multi-model agent.
        
        Extends base decision making with multi-model prediction.
        Demonstrates how supervised learning agents make decisions.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Next move direction (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        try:
            # Use multi-model agent to get move
            move = self.agent.get_move(game_state)
            
            # Validate move
            if move in ["UP", "DOWN", "LEFT", "RIGHT"]:
                return move
            else:
                # Fallback to random move if model fails
                return self._get_random_move(game_state)
                
        except Exception as e:
            print(f"Multi-model prediction error: {e}")
            # Fallback to random move
            return self._get_random_move(game_state)
    
    def _get_random_move(self, game_state: Dict[str, Any]) -> str:
        """
        Get a random valid move as fallback.
        
        Used when model prediction fails or is invalid.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Random valid move direction
        """
        import random
        
        # Get current direction and available moves
        current_direction = game_state.get("current_direction", "UP")
        head_position = game_state.get("head_position", [0, 0])
        snake_positions = game_state.get("snake_positions", [])
        
        # Define possible moves
        possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Filter out invalid moves (opposite direction, walls, snake body)
        valid_moves = []
        for move in possible_moves:
            if self._is_valid_move(move, current_direction, head_position, snake_positions):
                valid_moves.append(move)
        
        # Return random valid move or current direction if no valid moves
        if valid_moves:
            return random.choice(valid_moves)
        else:
            return current_direction
    
    def _is_valid_move(self, move: str, current_direction: str, 
                      head_position: List[int], snake_positions: List[List[int]]) -> bool:
        """
        Check if a move is valid.
        
        Args:
            move: Move to check
            current_direction: Current snake direction
            head_position: Current head position
            snake_positions: All snake body positions
            
        Returns:
            True if move is valid, False otherwise
        """
        # Check for opposite direction
        opposite_moves = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT"
        }
        
        if move == opposite_moves.get(current_direction):
            return False
        
        # Calculate new head position
        new_head = list(head_position)
        if move == "UP":
            new_head[1] -= 1
        elif move == "DOWN":
            new_head[1] += 1
        elif move == "LEFT":
            new_head[0] -= 1
        elif move == "RIGHT":
            new_head[0] += 1
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return False
        
        # Check snake body collision
        if new_head in snake_positions:
            return False
        
        return True
    
    def update_game_state(self, game_state: Dict[str, Any], move: str) -> Dict[str, Any]:
        """
        Update game state after a move.
        
        Extends base state update with multi-model specific tracking.
        
        Args:
            game_state: Current game state
            move: Move that was made
            
        Returns:
            Updated game state
        """
        # Call parent method to handle basic state update
        updated_state = super().update_game_state(game_state, move)
        
        # Add multi-model specific tracking
        updated_state["model_move"] = move
        updated_state["model_type"] = self.model_type
        updated_state["model_category"] = self.model_category
        
        return updated_state
    
    def check_game_end(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Check if the game has ended.
        
        Extends base game end checking with multi-model specific conditions.
        
        Args:
            game_state: Current game state
            
        Returns:
            End reason if game ended, None otherwise
        """
        # Call parent method for basic end conditions
        end_reason = super().check_game_end(game_state)
        
        # Add multi-model specific end conditions if needed
        if end_reason:
            # Log multi-model performance
            steps = game_state.get("steps", 0)
            score = game_state.get("score", 0)
            print(f"Game ended: {end_reason} (Steps: {steps}, Score: {score})")
        
        return end_reason
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """
        Get game statistics including multi-model metrics.
        
        Extends base statistics with multi-model specific metrics.
        
        Returns:
            Dictionary with game statistics
        """
        # Get base statistics from parent
        base_stats = super().get_game_statistics()
        
        # Add multi-model specific statistics
        model_stats = {
            **base_stats,
            "model_type": self.model_type,
            "model_category": self.model_category,
            "agent_trained": getattr(self.agent, 'is_trained', False),
            "model_architecture": type(self.agent).__name__,
            "framework": self._get_model_framework()
        }
        
        return model_stats
    
    def _get_model_framework(self) -> str:
        """
        Get the framework used by the current model.
        
        Returns:
            Framework name
        """
        neural_models = ['MLPAgent', 'CNNAgent', 'LSTMAgent', 'GRUAgent']
        tree_models = ['XGBoostAgent', 'LightGBMAgent', 'RandomForestAgent']
        graph_models = ['GCNAgent', 'GraphSAGEAgent', 'GATAgent']
        
        if self.model_type in neural_models:
            return "PyTorch"
        elif self.model_type in tree_models:
            if self.model_type == 'XGBoostAgent':
                return "XGBoost"
            elif self.model_type == 'LightGBMAgent':
                return "LightGBM"
            else:
                return "Scikit-learn"
        elif self.model_type in graph_models:
            return "PyTorch Geometric"
        else:
            return "Unknown" 