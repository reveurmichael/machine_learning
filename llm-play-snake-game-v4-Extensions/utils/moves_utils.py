"""
Movement Utilities

This module provides pure, stateless helper functions related to movement,
such as normalizing direction strings and analyzing spatial relationships
between game entities. Keeping them here centralizes movement-related logic.

All functions in this module are generic and not specific to any single
task (e.g., Task-0).
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "normalize_direction",
    "normalize_directions",
    "is_reverse",
    "get_relative_apple_direction_text",
    "position_to_direction",
]

# ----------------
# Canonicalising direction tokens
# ----------------

def normalize_direction(direction: str) -> str:
    """
    Normalizes a single direction string to a canonical format.

    - Non-string inputs are returned as-is to prevent runtime errors.
    - String inputs are converted to uppercase and stripped of whitespace.

    Example:
        >>> normalize_direction("  left\\n")
        'LEFT'

    Args:
        direction: The direction string to normalize.

    Returns:
        The normalized direction string (e.g., 'UP', 'DOWN', 'LEFT', 'RIGHT').
    """
    if not isinstance(direction, str):
        return direction  # Defensive pass-through
    return direction.strip().upper()


def normalize_directions(directions: List[str]) -> List[str]:
    """
    Normalizes a list of direction strings.

    Args:
        directions: A list of direction strings.

    Returns:
        A new list containing the normalized direction strings.
    """
    return [normalize_direction(d) for d in directions]


# ----------------
# Direction relationship helpers
# ----------------


def is_reverse(dir_a: str, dir_b: str) -> bool:
    """
    Checks if two directions are exact opposites (e.g., 'UP' and 'DOWN').

    Args:
        dir_a: The first direction string.
        dir_b: The second direction string.

    Returns:
        True if the directions are opposites, False otherwise.
    """
    dir_a = normalize_direction(dir_a)
    dir_b = normalize_direction(dir_b)
    opposites = {("UP", "DOWN"), ("LEFT", "RIGHT")}
    return (dir_a, dir_b) in opposites or (dir_b, dir_a) in opposites


def position_to_direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
    """
    Convert position difference to direction string.
    
    This function is commonly used by pathfinding algorithms to convert
    a position difference into the corresponding direction string.
    
    Args:
        from_pos: Starting position (x, y)
        to_pos: Target position (x, y)
        
    Returns:
        Direction string ('UP', 'DOWN', 'LEFT', 'RIGHT') or 'NO_PATH_FOUND'
        if the positions are not adjacent or invalid.
        
    Example:
        >>> position_to_direction((1, 1), (1, 2))
        'UP'
        >>> position_to_direction((5, 5), (6, 5))
        'RIGHT'
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    # Based on coordinate system: UP:(0,1), DOWN:(0,-1), LEFT:(-1,0), RIGHT:(1,0)
    if dx == 0 and dy == 1:
        return "UP"       # Y increases = UP
    elif dx == 0 and dy == -1:
        return "DOWN"     # Y decreases = DOWN
    elif dx == 1 and dy == 0:
        return "RIGHT"    # X increases = RIGHT
    elif dx == -1 and dy == 0:
        return "LEFT"     # X decreases = LEFT
    else:
        return "NO_PATH_FOUND"  # Invalid move (not adjacent or diagonal)


# ---------------------
# Simple positional analytics (used in prompt engineering)
# ---------------------

def get_relative_apple_direction_text(
    head_pos: NDArray[np.int_], apple_pos: NDArray[np.int_]
) -> str:
    """
    Generates a human-readable text describing the apple's position
    relative to the snake's head.

    This is primarily used for prompt engineering.

    Example:
        >>> head = np.array([10, 10])
        >>> apple = np.array([13, 8])
        >>> get_relative_apple_direction_text(head, apple)
        'The apple is 3 units to the RIGHT and 2 units DOWN.'

    Args:
        head_pos: The [x, y] coordinates of the snake's head.
        apple_pos: The [x, y] coordinates of the apple.

    Returns:
        A descriptive string of the relative positions.
    """
    dx = apple_pos[0] - head_pos[0]
    dy = apple_pos[1] - head_pos[1]

    x_direction = "RIGHT" if dx >= 0 else "LEFT"
    y_direction = "DOWN" if dy >= 0 else "UP"  # Inverted Y-axis in many game contexts

    # Correcting for typical screen coordinates (Y increases downwards)
    # If your game board's origin (0,0) is top-left, this is standard.
    # If dy is positive, it means apple_y > head_y, which is "DOWN".
    # If dy is negative, it means apple_y < head_y, which is "UP".
    y_direction = "UP" if dy < 0 else "DOWN"

    return (
        f"The apple is {abs(dx)} units to the {x_direction} "
        f"and {abs(dy)} units {y_direction}."
    ) 