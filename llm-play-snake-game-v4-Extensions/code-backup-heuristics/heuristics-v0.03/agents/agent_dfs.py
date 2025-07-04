"""
DFS Agent - Depth-First Search pathfinding for Snake Game
----------------

This module implements a DFS (Depth-First Search) agent for educational
comparison with BFS. DFS explores paths deeply before backtracking,
which can lead to longer paths but uses less memory.

Note: DFS is generally not optimal for Snake game due to tendency to
find longer paths, but included for algorithmic comparison.

Design Patterns:
- Strategy Pattern: DFS algorithm encapsulated as a strategy
- Template Method: Generic pathfinding with DFS implementation
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from config import DIRECTIONS
from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class DFSAgent:
    """
    Depth-First Search agent for Snake game.
    
    This agent uses DFS to find a path from the snake's head to the apple.
    Unlike BFS which finds the shortest path, DFS may find longer paths
    but uses less memory and can be faster in some scenarios.
    
    Algorithm:
    1. Start from current head position
    2. Explore paths deeply using DFS with recursion limit
    3. Return the first valid path found to apple
    4. If no path exists, return "NO_PATH_FOUND"
    
    Educational Note:
    DFS is included for comparison with BFS. In practice, BFS is usually
    better for Snake game as it finds shorter paths to the apple.
    """
    
    def __init__(self):
        """Initialize DFS agent."""
        self.algorithm_name = "DFS"
        self.name = "Depth-First Search"
        self.description = (
            "Depth-First Search pathfinding. Explores paths deeply before "
            "backtracking. May find longer paths than BFS but uses less memory. "
            "Included for educational comparison with BFS."
        )
        self.max_depth = 50  # Prevent infinite recursion
        
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """
        Get next move using DFS pathfinding.
        
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
            
            # Find path using DFS
            path = self._dfs_pathfind(head_pos, apple_pos, snake_positions, grid_size)
            
            if not path or len(path) < 2:
                # No path found - try any safe move
                for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    dx, dy = DIRECTIONS[direction]
                    next_pos = (head_pos[0] + dx, head_pos[1] + dy)
                    
                    # Check if position is valid and not in obstacles
                    if (0 <= next_pos[0] < grid_size and 
                        0 <= next_pos[1] < grid_size and 
                        next_pos not in snake_positions):
                        return direction
                        
                # Absolutely no safe move found
                return "NO_PATH_FOUND"
                
            # Get first move in path
            next_pos = path[1]  # path[0] is current head position
            direction = position_to_direction(head_pos, next_pos)
            
            # Validate the direction is not a reverse move
            if direction and direction != "NO_PATH_FOUND":
                return direction
            
            # If direction is invalid, return NO_PATH_FOUND
            return "NO_PATH_FOUND"

        except Exception as e:
            print(f"DFS Agent error: {e}")
            return "NO_PATH_FOUND"
    
    def _dfs_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     obstacles: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """
        Find path using Depth-First Search.
        
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
            
        # Use iterative DFS to avoid recursion limits
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            current_pos, path = stack.pop()
            
            # Depth limit to prevent excessive searching
            if len(path) > self.max_depth:
                continue
            
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
                    
                # Add to stack for further exploration (DFS uses stack, not queue)
                stack.append((next_pos, new_path))
                visited.add(next_pos)
        
        # No path found
        return []
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"DFSAgent(algorithm={self.algorithm_name})"
