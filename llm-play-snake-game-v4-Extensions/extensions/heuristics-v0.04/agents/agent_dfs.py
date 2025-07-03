"""
DFS Agent - Depth-First Search pathfinding for Snake Game
--------------------

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
from typing import List, Tuple, Set, TYPE_CHECKING

# Ensure project root is set and properly configured
import sys
import os
from pathlib import Path

def _ensure_project_root():
    """Ensure we're working from project root"""
    current = Path(__file__).resolve()
    # Navigate up to find project root (contains config/ directory)
    for _ in range(10):
        if (current / "config").is_dir():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            os.chdir(str(current))
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError("Could not locate project root containing 'config/' folder")

_ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error
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
        self.max_depth = 150  # Prevent infinite recursion; increased for larger grids
        
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """
        Get next move using DFS pathfinding (simplified interface).
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move
    
    def get_move_with_explanation(self, game: HeuristicGameLogic) -> Tuple[str, str]:
        """
        Get next move using DFS pathfinding with detailed explanation.
        
        v0.04 Enhancement: Returns both move and natural language explanation
        for LLM fine-tuning dataset generation.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Tuple of (direction_string, explanation_string)
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
                # No path found, try a fallback safe move
                safe_move, explanation = self._find_safe_move_with_explanation(head_pos, snake_positions, grid_size)
                return safe_move, explanation
                
            # Get first move in path
            next_pos = path[1]
            direction = position_to_direction(head_pos, next_pos)
            
            # Generate explanation for this move
            explanation = self._generate_move_explanation(head_pos, path, direction)
            
            return direction, explanation
            
        except Exception as e:
            error_explanation = f"DFS agent encountered an error: {str(e)}. Unable to compute safe path."
            print_error(f"DFS Agent error: {e}")
            return "NO_PATH_FOUND", error_explanation
    
    def _generate_move_explanation(self, head_pos: Tuple[int, int], path: List[Tuple[int, int]], direction: str) -> str:
        """Generates a detailed explanation for a successful DFS-based move."""
        path_length = len(path) - 1
        explanation = (
            f"DFS found a valid path of length {path_length} from {head_pos} to the apple. "
            f"As an educational algorithm, DFS explores deeply, which can result in non-optimal (longer) paths compared to BFS. "
            f"The first step on this path is to move {direction}."
        )
        return explanation

    def _find_safe_move_with_explanation(self, head_pos: Tuple[int, int], snake_positions: set, grid_size: int) -> Tuple[str, str]:
        """Finds any valid move and provides an explanation for this fallback strategy."""
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dx, dy = DIRECTIONS[direction]
            next_pos = (head_pos[0] + dx, head_pos[1] + dy)
            
            if (0 <= next_pos[0] < grid_size and 
                0 <= next_pos[1] < grid_size and 
                next_pos not in snake_positions):
                
                explanation = (
                    "DFS could not find a direct path to the apple. "
                    "This is common for DFS in constrained spaces as it may explore dead-end paths first. "
                    f"As a fallback, a safe move ({direction}) is chosen to avoid immediate collision and continue searching."
                )
                return direction, explanation
        
        explanation = (
            "DFS found no path to the apple and no safe fallback moves are available. "
            "All adjacent positions are blocked by the snake's body or a wall, leading to a game over."
        )
        return "NO_PATH_FOUND", explanation

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
