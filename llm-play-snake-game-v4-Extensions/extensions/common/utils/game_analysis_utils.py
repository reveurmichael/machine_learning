"""
Game Analysis Utilities

This module provides pure, stateless functions for analyzing game states
and calculating various game metrics. These utilities are designed to be:

- Pure functions with no side effects
- Type-safe with comprehensive type hints
- Reusable across all game components and extensions
- Independent of specific game logic implementations

The functions handle core game analysis like danger assessment, apple direction,
and other metrics that are useful for AI agents and dataset generation.

All functions in this module are generic and not specific to any single
task (e.g., Task-0).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "calculate_apple_direction",
    "calculate_danger_assessment",
    "get_next_position",
]

# Type aliases for clarity
Position = Union[List[int], Tuple[int, int], NDArray[np.int_]]
PositionList = List[Position]


def get_next_position(head_position: Position, direction: str) -> Position:
    """
    Calculate the next position when moving in a given direction.
    
    Args:
        head_position: Current [x, y] position
        direction: Direction to move ("UP", "DOWN", "LEFT", "RIGHT")
        
    Returns:
        Next [x, y] position after moving in the specified direction
        
    Example:
        >>> get_next_position([5, 5], "UP")
        [5, 6]
        >>> get_next_position([5, 5], "LEFT")
        [4, 5]
    """
    x, y = head_position[0], head_position[1]
    
    if direction == "UP":
        return [x, y + 1]
    elif direction == "DOWN":
        return [x, y - 1]
    elif direction == "LEFT":
        return [x - 1, y]
    elif direction == "RIGHT":
        return [x + 1, y]
    else:
        raise ValueError(f"Invalid direction: {direction}")


def calculate_apple_direction(head_position: Position, apple_position: Position) -> Dict[str, int]:
    """
    Calculate which direction the apple is relative to the snake's head.
    
    Args:
        head_position: Current [x, y] position of snake's head
        apple_position: [x, y] position of the apple
        
    Returns:
        Dictionary with boolean flags for each direction (up, down, left, right)
        
    Example:
        >>> head = [5, 5]
        >>> apple = [7, 3]
        >>> calculate_apple_direction(head, apple)
        {'up': 0, 'down': 1, 'left': 0, 'right': 1}
    """
    head_x, head_y = head_position[0], head_position[1]
    apple_x, apple_y = apple_position[0], apple_position[1]
    
    return {
        'up': 1 if apple_y > head_y else 0,
        'down': 1 if apple_y < head_y else 0,
        'left': 1 if apple_x < head_x else 0,
        'right': 1 if apple_x > head_x else 0
    }


def calculate_danger_assessment(
    head_position: Position,
    snake_positions: PositionList,
    grid_size: int,
    current_move: str
) -> Dict[str, int]:
    """
    Calculate danger assessment for the current move and adjacent directions.
    
    This function evaluates the danger of moving in the current direction
    and the adjacent left/right directions relative to the current move.
    
    Args:
        head_position: Current [x, y] position of snake's head
        snake_positions: List of all snake segment positions
        grid_size: Size of the game grid
        current_move: The move being evaluated ("UP", "DOWN", "LEFT", "RIGHT")
        
    Returns:
        Dictionary with danger flags for straight, left, and right directions
        
    Example:
        >>> head = [5, 5]
        >>> snake = [[5, 5], [5, 4], [5, 3]]
        >>> calculate_danger_assessment(head, snake, 10, "UP")
        {'straight': 0, 'left': 0, 'right': 0}
    """
    head_x, head_y = head_position[0], head_position[1]
    
    # Initialize danger flags
    danger_straight = 0
    danger_left = 0
    danger_right = 0
    
    # Check for wall collision based on current move direction
    if current_move == "UP":
        # Straight: check if moving up hits top wall
        danger_straight = 1 if head_y + 1 >= grid_size else 0
        # Left: check if moving left hits left wall
        danger_left = 1 if head_x - 1 < 0 else 0
        # Right: check if moving right hits right wall
        danger_right = 1 if head_x + 1 >= grid_size else 0
        
    elif current_move == "DOWN":
        # Straight: check if moving down hits bottom wall
        danger_straight = 1 if head_y - 1 < 0 else 0
        # Left: check if moving right hits right wall (relative to down direction)
        danger_left = 1 if head_x + 1 >= grid_size else 0
        # Right: check if moving left hits left wall (relative to down direction)
        danger_right = 1 if head_x - 1 < 0 else 0
        
    elif current_move == "LEFT":
        # Straight: check if moving left hits left wall
        danger_straight = 1 if head_x - 1 < 0 else 0
        # Left: check if moving down hits bottom wall (relative to left direction)
        danger_left = 1 if head_y - 1 < 0 else 0
        # Right: check if moving up hits top wall (relative to left direction)
        danger_right = 1 if head_y + 1 >= grid_size else 0
        
    elif current_move == "RIGHT":
        # Straight: check if moving right hits right wall
        danger_straight = 1 if head_x + 1 >= grid_size else 0
        # Left: check if moving up hits top wall (relative to right direction)
        danger_left = 1 if head_y + 1 >= grid_size else 0
        # Right: check if moving down hits bottom wall (relative to right direction)
        danger_right = 1 if head_y - 1 < 0 else 0
    
    # Check for body collision in the straight direction
    next_pos = get_next_position(head_position, current_move)
    if next_pos in snake_positions:
        danger_straight = 1
    
    return {
        'straight': danger_straight,
        'left': danger_left,
        'right': danger_right
    } 