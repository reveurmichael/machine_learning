"""
BFS Agent - Simple Breadth-First Search Snake Agent
----------------

Minimal BFS implementation for pathfinding to apples.
"""

from __future__ import annotations
from typing import List, Tuple, Set
from collections import deque

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction


class BFSAgent:
    """
    Simple BFS agent for Snake game.
    
    Uses breadth-first search to find shortest path to apple.
    """
    
    def __init__(self) -> None:
        """Initialize BFS agent."""
        pass
    
    def get_move(self, game_logic) -> str:
        """
        Get next move using BFS pathfinding.
        
        Args:
            game_logic: Game logic instance containing current state
            
        Returns:
            Direction string or "NO_PATH_FOUND"
        """
        state = game_logic.get_state_snapshot()
        
        head = tuple(state["head_position"])
        apple = tuple(state["apple_position"])
        snake_body = set(tuple(pos) for pos in state["snake_positions"])
        grid_size = state["grid_size"]
        
        # Find path to apple
        path = self._bfs_path(head, apple, snake_body, grid_size)
        
        if not path:
            return "NO_PATH_FOUND"
        
        # Convert first step to direction
        next_pos = path[1] if len(path) > 1 else path[0]
        return position_to_direction(head, next_pos)
    
    def _bfs_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  obstacles: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """
        Find shortest path using BFS.
        
        Args:
            start: Starting position
            goal: Target position
            obstacles: Set of obstacle positions
            grid_size: Grid size
            
        Returns:
            List of positions from start to goal, or empty list if no path
        """
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for dx, dy in DIRECTIONS.values():
                next_x = current[0] + dx
                next_y = current[1] + dy
                next_pos = (next_x, next_y)
                
                # Check bounds
                if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                    continue
                
                # Check obstacles
                if next_pos in obstacles:
                    continue
                
                # Check if visited
                if next_pos in visited:
                    continue
                
                # Found goal
                if next_pos == goal:
                    return path + [next_pos]
                
                # Add to queue
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
        
        return []  # No path found 