"""
Heuristic Utilities - Helper functions for heuristic algorithms v0.04
----------------

This module contains utility functions that are specific to heuristic algorithms
and used across multiple agent implementations.

These functions were moved from agent_bfs.py to follow SSOT principles and 
improve code organization.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

from typing import List, Tuple, Set, Dict, Any
from collections import deque
from config.game_constants import DIRECTIONS


def count_obstacles_in_path(path: List[Tuple[int, int]], snake_positions: set) -> int:
    """Count how many snake body segments are near the path."""
    obstacles_near_path = 0
    for pos in path:
        # Check adjacent positions for snake body
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            adjacent_pos = (pos[0] + dx, pos[1] + dy)
            if adjacent_pos in snake_positions:
                obstacles_near_path += 1
    return obstacles_near_path


def get_neighbors(pos: Tuple[int, int], grid_size: int) -> List[Tuple[int, int]]:
    """Get valid neighboring positions."""
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = (pos[0] + dx, pos[1] + dy)
        if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
            neighbors.append(neighbor)
    return neighbors


def count_remaining_free_cells(snake_positions: set, grid_size: int) -> int:
    """Count how many empty cells are not occupied by the snake body."""
    total_cells = grid_size * grid_size
    return total_cells - len(snake_positions)


def calculate_valid_moves(game_state: dict) -> list:
    """
    Calculate valid moves using the same logic as agents.
    
    SSOT: Extract positions using exact same logic as dataset_generator.py
    to guarantee true single source of truth.
    
    Args:
        game_state: Complete game state dict containing all positions
    Returns:
        List of valid moves (UP, DOWN, LEFT, RIGHT)
    """
    # Import here to avoid circular imports
    from extensions.common.utils.game_state_utils import extract_body_positions
    
    snake_positions = game_state.get('snake_positions', [])
    head_pos = game_state.get('head_position', [0, 0])
    grid_size = game_state.get('grid_size', 10)
    
    # SSOT: Use centralized body_positions extraction
    body_positions = extract_body_positions(game_state)

    # Use body positions as obstacles (excluding head)
    obstacles = set(tuple(p) for p in body_positions)
    
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


def count_free_space_in_direction(game_state: dict, direction: str) -> int:
    """
    Count free space in a given direction from the current head position.
    
    SSOT: Extract positions using exact same logic as dataset_generator.py
    to guarantee true single source of truth.
    
    Args:
        game_state: Complete game state dict containing all positions
        direction: Direction to check ('UP', 'DOWN', 'LEFT', 'RIGHT')
        
    Returns:
        Number of free cells in the direction
    """
    # Import here to avoid circular imports
    from extensions.common.utils.game_state_utils import extract_body_positions
    
    snake_positions = game_state.get('snake_positions', [])
    head_pos = game_state.get('head_position', [0, 0])
    grid_size = game_state.get('grid_size', 10)
    
    # SSOT: Use exact same body_positions logic as dataset_generator.py
    body_positions = extract_body_positions(game_state)
    
    count = 0
    current_pos = list(head_pos)
    
    while True:
        if direction == 'UP':
            current_pos[1] += 1
        elif direction == 'DOWN':
            current_pos[1] -= 1
        elif direction == 'LEFT':
            current_pos[0] -= 1
        elif direction == 'RIGHT':
            current_pos[0] += 1
        
        # Check bounds
        if (current_pos[0] < 0 or current_pos[0] >= grid_size or 
            current_pos[1] < 0 or current_pos[1] >= grid_size):
            break
        
        # Check snake collision (using body positions)
        if current_pos in body_positions:
            break
        
        count += 1
        
        # Prevent infinite loop
        if count > grid_size * grid_size:
            break
    
    return count


def calculate_manhattan_distance(game_state: dict) -> int:
    """
    SSOT: Calculate Manhattan distance between head and apple.
    
    Single source of truth for Manhattan distance calculation.
    All other files must use this method instead of calculating distance.
    """
    # Import here to avoid circular imports
    from extensions.common.utils.game_state_utils import extract_head_position
    
    head_pos = extract_head_position(game_state)
    apple_pos = game_state.get('apple_position', [0, 0])
    return abs(head_pos[0] - apple_pos[0]) + abs(head_pos[1] - apple_pos[1])


def calculate_valid_moves_ssot(game_state: dict) -> List[str]:
    """
    SSOT: Calculate valid moves from current head position.
    
    Single source of truth for valid moves calculation.
    All other files must use this method instead of calculating valid moves.
    """
    # Use the existing calculate_valid_moves method for SSOT compliance
    return calculate_valid_moves(game_state)


def bfs_pathfind(start: List[int], goal: List[int], obstacles: Set[Tuple[int, int]], grid_size: int) -> List[List[int]] | None:
    """
    BFS pathfinding from start to goal, avoiding obstacles.
    
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
