"""
Simple BFS Agent for Snake Game
===============================

Minimal BFS pathfinding agent.
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, TYPE_CHECKING

from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSAgent:
    """Simple BFS pathfinding agent."""
    
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """Get next move using BFS pathfinding."""
        try:
            head_pos = tuple(game.head_position)
            apple_pos = tuple(game.apple_position)
            snake_positions = {tuple(pos) for pos in game.snake_positions}
            
            path = self._bfs_pathfind(head_pos, apple_pos, snake_positions, game.grid_size)
            
            if len(path) < 2:
                return "NO_PATH_FOUND"
                
            next_pos = path[1]
            return position_to_direction(head_pos, next_pos)
            
        except Exception:
            return "NO_PATH_FOUND"
    
    def _bfs_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     obstacles: set, grid_size: int) -> List[Tuple[int, int]]:
        """Find shortest path using BFS."""
        if start == goal:
            return [start]
            
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current_pos, path = queue.popleft()
            
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                dx, dy = DIRECTIONS[direction]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                    continue
                    
                if next_pos in obstacles or next_pos in visited:
                    continue
                    
                new_path = path + [next_pos]
                
                if next_pos == goal:
                    return new_path
                    
                queue.append((next_pos, new_path))
                visited.add(next_pos)
        
        return [] 