"""
Supervised Learning v0.01 - Game Logic
=====================================

Game logic for supervised learning v0.01, focusing on neural networks only.
Extends BaseGameLogic from Task-0, demonstrating perfect base class reuse.

Design Pattern: Template Method
- Extends BaseGameLogic for consistent game mechanics
- Implements neural network-specific decision making
- Simple structure focused on proof of concept
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
    Game logic for supervised learning v0.01.
    
    Extends BaseGameLogic to demonstrate perfect base class reuse.
    Focuses on neural networks only for proof of concept.
    
    Design Pattern: Template Method
    - Inherits core game mechanics from BaseGameLogic
    - Implements neural network-specific decision making
    - Demonstrates how extensions can reuse core infrastructure
    """
    
    def __init__(self, agent, grid_size: int = 10, max_steps: int = 1000):
        """
        Initialize supervised learning game logic.
        
        Args:
            agent: Neural network agent for decision making
            grid_size: Size of the game grid
            max_steps: Maximum steps per game
        """
        # Call parent constructor to inherit all base functionality
        super().__init__(grid_size=grid_size, max_steps=max_steps)
        
        # Supervised learning specific attributes
        self.agent = agent
        self.neural_network_type = type(agent).__name__
        
        print(f"Initialized SupervisedGameLogic with {self.neural_network_type}")
    
    def get_next_move(self, game_state: Dict[str, Any]) -> str:
        """
        Get the next move using neural network agent.
        
        Extends base decision making with neural network prediction.
        Demonstrates how supervised learning agents make decisions.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Next move direction (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        try:
            # Use neural network agent to get move
            move = self.agent.get_move(game_state)
            
            # Validate move
            if move in ["UP", "DOWN", "LEFT", "RIGHT"]:
                return move
            else:
                # Fallback to random move if neural network fails
                return self._get_random_move(game_state)
                
        except Exception as e:
            print(f"Neural network prediction error: {e}")
            # Fallback to random move
            return self._get_random_move(game_state)
    
    def _get_random_move(self, game_state: Dict[str, Any]) -> str:
        """
        Get a random valid move as fallback.
        
        Used when neural network prediction fails or is invalid.
        
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
        
        Extends base state update with neural network specific tracking.
        
        Args:
            game_state: Current game state
            move: Move that was made
            
        Returns:
            Updated game state
        """
        # Call parent method to handle basic state update
        updated_state = super().update_game_state(game_state, move)
        
        # Add neural network specific tracking
        updated_state["neural_network_move"] = move
        updated_state["neural_network_type"] = self.neural_network_type
        
        return updated_state
    
    def check_game_end(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Check if the game has ended.
        
        Extends base game end checking with neural network specific conditions.
        
        Args:
            game_state: Current game state
            
        Returns:
            End reason if game ended, None otherwise
        """
        # Call parent method for basic end conditions
        end_reason = super().check_game_end(game_state)
        
        # Add neural network specific end conditions if needed
        if end_reason:
            # Log neural network performance
            steps = game_state.get("steps", 0)
            score = game_state.get("score", 0)
            print(f"Game ended: {end_reason} (Steps: {steps}, Score: {score})")
        
        return end_reason
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """
        Get game statistics including neural network metrics.
        
        Extends base statistics with neural network specific metrics.
        
        Returns:
            Dictionary with game statistics
        """
        # Get base statistics from parent
        base_stats = super().get_game_statistics()
        
        # Add neural network specific statistics
        neural_stats = {
            **base_stats,
            "neural_network_type": self.neural_network_type,
            "agent_trained": getattr(self.agent, 'is_trained', False),
            "model_architecture": type(self.agent).__name__
        }
        
        return neural_stats 