"""
Collision Detection Utilities

This module provides high-performance, stateless functions for detecting various
types of collisions in the Snake game. The utilities are designed to be:

- Fast and efficient for real-time game loops
- Pure functions with predictable behavior
- Comprehensive in collision detection scenarios
- Easy to test and reason about
- Compatible with different coordinate systems

The collision detection system handles:
- Wall collisions (boundary checking)
- Self-collision (snake body intersections)
- Apple consumption detection
- Generic position overlap checking

All functions in this module are generic and not specific to any single
task (e.g., Task-0).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "check_collision", 
    "check_wall_collision",
    "check_body_collision", 
    "check_apple_collision",
    "positions_overlap"
]

# Type aliases for better readability
Position = NDArray[np.int_]
PositionList = List[Position]
CollisionResult = Tuple[bool, bool]


def positions_overlap(pos1: Position, pos2: Position) -> bool:
    """
    Checks if two positions are the same.
    
    This is a utility function that provides a clear, readable way to check
    position equality with proper NumPy array handling.
    
    Args:
        pos1: First position as [x, y] coordinates.
        pos2: Second position as [x, y] coordinates.
        
    Returns:
        True if positions are identical, False otherwise.
        
    Example:
        >>> pos_a = np.array([3, 4])
        >>> pos_b = np.array([3, 4])
        >>> positions_overlap(pos_a, pos_b)
        True
    """
    return np.array_equal(pos1, pos2)


def check_wall_collision(head_position: Position, grid_size: int) -> bool:
    """
    Checks if the snake's head has collided with any wall (boundary).
    
    Args:
        head_position: The [x, y] coordinates of the snake's head.
        grid_size: The size of the square grid (valid coordinates: 0 to grid_size-1).
        
    Returns:
        True if the head is outside the valid grid boundaries, False otherwise.
        
    Raises:
        ValueError: If grid_size is not positive or head_position is invalid.
        
    Example:
        >>> head = np.array([10, 5])
        >>> check_wall_collision(head, 10)  # x=10 is outside 0-9 range
        True
        >>> head = np.array([5, 5])
        >>> check_wall_collision(head, 10)  # x=5, y=5 is valid
        False
    """
    if grid_size <= 0:
        raise ValueError(f"Invalid grid_size: {grid_size}. Must be positive.")
    
    if len(head_position) != 2:
        raise ValueError(f"Invalid head_position: {head_position}. Must be [x, y].")
    
    x, y = head_position
    return x < 0 or x >= grid_size or y < 0 or y >= grid_size


def check_body_collision(
    head_position: Position, 
    snake_body: PositionList, 
    is_apple_eaten: bool
) -> bool:
    """
    Checks if the snake's head has collided with its own body.
    
    The collision logic accounts for snake growth mechanics:
    - When an apple is eaten, the snake grows and the tail doesn't move
    - When no apple is eaten, the tail moves forward, creating space
    
    Args:
        head_position: The [x, y] coordinates of the snake's head.
        snake_body: List of [x, y] coordinates for snake body segments (excluding head).
        is_apple_eaten: Whether the snake just ate an apple (affects collision zones).
        
    Returns:
        True if the head collides with a body segment, False otherwise.
        
    Example:
        >>> head = np.array([2, 2])
        >>> body = [np.array([2, 3]), np.array([2, 4]), np.array([1, 4])]
        >>> check_body_collision(head, body, False)  # Tail will move
        False
        >>> check_body_collision(head, body, True)   # Snake growing, all body solid
        False
    """
    if len(snake_body) == 0:
        return False  # No body to collide with
    
    # Determine which body segments to check based on growth state
    if is_apple_eaten:
        # Snake is growing - check collision with all body segments
        segments_to_check = snake_body
    else:
        # Snake is moving - tail will move out of the way, so exclude it
        segments_to_check = snake_body[:-1] if len(snake_body) > 0 else []
    
    # Check for collision with remaining segments
    return any(positions_overlap(head_position, segment) for segment in segments_to_check)


def check_apple_collision(head_position: Position, apple_position: Position) -> bool:
    """
    Checks if the snake's head is at the same position as the apple.
    
    Args:
        head_position: The [x, y] coordinates of the snake's head.
        apple_position: The [x, y] coordinates of the apple.
        
    Returns:
        True if the head and apple positions are identical, False otherwise.
        
    Example:
        >>> head = np.array([5, 7])
        >>> apple = np.array([5, 7])
        >>> check_apple_collision(head, apple)
        True
    """
    return positions_overlap(head_position, apple_position)


def check_collision(
    head_position: Position,
    snake_body: PositionList,
    grid_size: int,
    is_apple_eaten: bool,
) -> CollisionResult:
    """
    Comprehensive collision detection for all collision types.
    
    This is the main collision detection function that checks for both wall
    and body collisions in a single call. It's optimized to short-circuit
    on the first collision found for better performance.

    Args:
        head_position: The [x, y] coordinates of the snake's head.
        snake_body: List of [x, y] coordinates for each segment of the
                    snake's body, excluding the head.
        grid_size: The size of the square grid (e.g., 10 for a 10x10 board).
        is_apple_eaten: Flag indicating if the snake has just eaten an
                        apple, which affects body collision logic.

    Returns:
        A tuple containing two boolean values:
        - wall_collision: True if the head has collided with a wall.
        - body_collision: True if the head has collided with a body segment.
        
    Note:
        If a wall collision is detected, body collision checking is skipped
        for performance optimization since the game ends either way.
        
    Example:
        >>> head = np.array([0, 5])
        >>> body = [np.array([1, 5]), np.array([2, 5])]
        >>> wall, body_col = check_collision(head, body, 10, False)
        >>> print(f"Wall: {wall}, Body: {body_col}")
        Wall: False, Body: False
    """
    # Check wall collision first (faster check)
    wall_collision = check_wall_collision(head_position, grid_size)
    
    if wall_collision:
        # No need to check body collision if we've hit a wall
        return True, False
    
    # Check body collision only if no wall collision
    body_collision = check_body_collision(head_position, snake_body, is_apple_eaten)
    
    return wall_collision, body_collision 