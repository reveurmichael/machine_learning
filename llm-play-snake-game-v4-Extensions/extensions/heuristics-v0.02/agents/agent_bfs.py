"""
BFS Agent - Breadth-First Search pathfinding for Snake Game
--------------------

This module implements a simple BFS (Breadth-First Search) agent that finds
the shortest path to the apple while avoiding walls and its own body.

The agent implements the SnakeAgent protocol, making it compatible with
the existing game engine infrastructure.

Design Patterns:
- Strategy Pattern: BFS algorithm encapsulated as a strategy
- Protocol Pattern: Implements SnakeAgent interface for compatibility
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, TYPE_CHECKING

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from config import DIRECTIONS
from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSAgent:
    """
    Breadth-First Search agent for Snake game.
    
    This agent uses BFS to find the shortest path from the snake's head
    to the apple, avoiding obstacles (walls and snake body).
    
    Algorithm:
    1. Start from current head position
    2. Explore all valid adjacent positions using BFS
    3. Return the first move in the shortest path to apple
    4. If no path exists, return "NO_PATH_FOUND"
    
    Design Patterns:
    - Strategy Pattern: BFS pathfinding strategy
    - Template Method: Generic pathfinding with BFS implementation
    """
    
    def __init__(self):
        """Initialize BFS agent."""
        self.algorithm_name = "BFS"
        
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """
        Get next move using BFS pathfinding.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        try:
            # Extract game state
            head_pos = tuple(game.head_position)
            apple_pos = tuple(game.apple_position)
            snake_positions = {tuple(pos) for pos in game.snake_positions}
            grid_size = game.grid_size
            
            # Find path using BFS
            path = self._bfs_pathfind(head_pos, apple_pos, snake_positions, grid_size)
            
            if not path or len(path) < 2:
                return "NO_PATH_FOUND"
                
            # Get first move in path
            next_pos = path[1]  # path[0] is current head position
            direction = position_to_direction(head_pos, next_pos)
            
            return direction
            
        except Exception as e:
            print(f"BFS Agent error: {e}")
            return "NO_PATH_FOUND"
    
    def _bfs_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     obstacles: set, grid_size: int) -> List[Tuple[int, int]]:
        """
        Find shortest path using Breadth-First Search.
        
        Args:
            start: Starting position (head)
            goal: Goal position (apple)
            obstacles: Set of obstacle positions (snake body)
            grid_size: Size of the game grid
            
        Returns:
            List of positions representing the path, or empty list if no path
        """
        if start == goal:
            return [start]
            
        # BFS initialization
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current_pos, path = queue.popleft()
            
            # Check all adjacent positions
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                dx, dy = DIRECTIONS[direction]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Skip if out of bounds
                if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                    continue
                    
                # Skip if obstacle or already visited
                if next_pos in obstacles or next_pos in visited:
                    continue
                    
                # Create new path
                new_path = path + [next_pos]
                
                # Check if we reached the goal
                if next_pos == goal:
                    return new_path
                    
                # Add to queue for further exploration
                queue.append((next_pos, new_path))
                visited.add(next_pos)
        
        # No path found
        return []
    

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})" 