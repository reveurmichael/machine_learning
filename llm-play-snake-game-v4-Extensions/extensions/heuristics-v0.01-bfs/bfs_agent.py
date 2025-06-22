"""
BFS Agent - Breadth-First Search pathfinding for Snake Game
==========================================================

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
from typing import List, Tuple, Optional, TYPE_CHECKING, Any
import numpy as np

from config.game_constants import DIRECTIONS, VALID_MOVES
from core.game_agents import SnakeAgent

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
            direction = self._get_direction(head_pos, next_pos)
            
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
    
    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """
        Convert position difference to direction string.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Based on DIRECTIONS mapping: UP:(0,1), DOWN:(0,-1), LEFT:(-1,0), RIGHT:(1,0)
        if dx == 0 and dy == 1:
            return "UP"       # Y increases = UP
        elif dx == 0 and dy == -1:
            return "DOWN"     # Y decreases = DOWN
        elif dx == 1 and dy == 0:
            return "RIGHT"    # X increases = RIGHT
        elif dx == -1 and dy == 0:
            return "LEFT"     # X decreases = LEFT
        else:
            return "NO_PATH_FOUND"  # Invalid move
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})" 