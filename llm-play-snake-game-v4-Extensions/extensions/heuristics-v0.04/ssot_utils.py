"""
SSOT Utilities - Single Source of Truth for shared logic.

This module provides shared implementations for BFS pathfinding, valid moves calculation,
and other logic that must be identical between agents and dataset generator.

⚠️  IMPORTANT: This is the ONLY source of truth for these calculations.
    - All BFS pathfinding must use ssot_bfs_pathfind()
    - All valid moves calculation must use ssot_calculate_valid_moves()

    Do NOT reimplement these functions elsewhere in the codebase.
    This ensures perfect consistency between agents and dataset generation.
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional, Set
import sys
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Ensure project root is on sys.path for config imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.game_constants import DIRECTIONS, VALID_MOVES


def ssot_bfs_pathfind(start: List[int], goal: List[int], obstacles: Set[Tuple[int, int]], grid_size: int) -> Optional[List[List[int]]]:
    """
    SSOT BFS pathfinding from start to goal, avoiding obstacles.
    
    This is the single source of truth for BFS pathfinding used by both
    agents and dataset generator to ensure perfect consistency.
    
    Args:
        start: Starting position [x, y]
        goal: Goal position [x, y]
        obstacles: Set of obstacle positions as tuples (x, y)
        grid_size: Size of the game grid
        
    Returns:
        List of positions forming the path from start to goal, or None if no path exists
    """
    if start == goal:
        return [start]
    
    # Convert to tuples for set operations
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    
    if start_tuple in obstacles or goal_tuple in obstacles:
        return None
    
    # BFS queue: (position, path)
    queue = deque([(start_tuple, [start])])
    visited = {start_tuple}
    
    # Direction vectors for BFS
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
    
    while queue:
        current_pos, path = queue.popleft()
        
        for dx, dy in directions:
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_pos = (next_x, next_y)
            
            # Check bounds
            if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                continue
            
            # Check if visited or obstacle
            if next_pos in visited or next_pos in obstacles:
                continue
            
            # Check if goal reached
            if next_pos == goal_tuple:
                return path + [[next_x, next_y]]
            
            # Add to queue
            visited.add(next_pos)
            queue.append((next_pos, path + [[next_x, next_y]]))
    
    return None


def ssot_calculate_valid_moves(head_pos: List[int], snake_positions: List[List[int]], grid_size: int) -> List[str]:
    """
    SSOT valid moves calculation.
    
    This is the single source of truth for valid moves calculation used by both
    agents and dataset generator to ensure perfect consistency.
    
    Args:
        head_pos: Current head position [x, y]
        snake_positions: All snake positions (head at index -1)
        grid_size: Size of the game grid
        
    Returns:
        List of valid moves (UP, DOWN, LEFT, RIGHT)
    """
    # SSOT: Obstacles are all body segments except head
    obstacles = set(tuple(p) for p in snake_positions[:-1] if len(p) >= 2)
    
    valid_moves = []
    for direction, (dx, dy) in DIRECTIONS.items():
        next_x = head_pos[0] + dx
        next_y = head_pos[1] + dy
        # Check bounds
        if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
            next_pos = (next_x, next_y)
            # Check if position is not occupied by snake body (excluding head)
            if next_pos not in obstacles:
                valid_moves.append(direction)
    
    return valid_moves
