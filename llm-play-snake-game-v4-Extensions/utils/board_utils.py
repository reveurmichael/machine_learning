"""
Board Manipulation Utilities

This module provides pure, stateless functions for manipulating the game board
array. These utilities follow functional programming principles and are designed
to be:

- Pure functions with no side effects (except update_board_array)
- Type-safe with comprehensive type hints
- Efficient with NumPy array operations
- Reusable across all game components
- Independent of specific game logic implementations

The functions handle core board operations like apple placement and board state
updates, ensuring consistency across different game modes and extensions.

All functions in this module are generic and not specific to any single
task (e.g., Task-0).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "generate_random_apple",
    "update_board_array",
    "is_position_valid",
    "get_empty_positions",
    "create_text_board",
]

# Type aliases for clarity
Position = NDArray[np.int_]
PositionList = List[Position]
BoardArray = NDArray[np.int8]
BoardInfo = Dict[str, int]
Coordinate = Union[List[int], tuple, NDArray[np.int_]]


def is_position_valid(position: Position, grid_size: int) -> bool:
    """
    Checks if a position is within the valid bounds of the game grid.
    
    Args:
        position: The [x, y] coordinates to validate.
        grid_size: The size of the square grid.
        
    Returns:
        True if the position is valid (within bounds), False otherwise.
        
    Example:
        >>> pos = np.array([5, 3])
        >>> is_position_valid(pos, 10)
        True
        >>> pos = np.array([10, 3])
        >>> is_position_valid(pos, 10)
        False
    """
    if len(position) != 2:
        return False
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size


def get_empty_positions(snake_positions: PositionList, grid_size: int) -> List[Position]:
    """
    Returns a list of all empty positions on the board (not occupied by snake).
    
    This is useful for apple placement algorithms or AI pathfinding that needs
    to know all available spaces.
    
    Args:
        snake_positions: List of [x, y] coordinates occupied by the snake.
        grid_size: The size of the square grid.
        
    Returns:
        List of all unoccupied [x, y] positions as numpy arrays.
        
    Example:
        >>> snake = [np.array([1, 1]), np.array([1, 2])]
        >>> empty = get_empty_positions(snake, 3)
        >>> len(empty)  # 3x3 grid minus 2 snake positions = 7 empty
        7
    """
    if not snake_positions:
        # If no snake, all positions are empty
        return [
            np.array([x, y], dtype=np.int_) 
            for x in range(grid_size) 
            for y in range(grid_size)
        ]
    
    # Create a set of occupied positions for fast lookup
    occupied = {tuple(pos) for pos in snake_positions}
    
    # Generate all positions and filter out occupied ones
    empty_positions = []
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) not in occupied:
                empty_positions.append(np.array([x, y], dtype=np.int_))
    
    return empty_positions


def generate_random_apple(
    snake_positions: PositionList, 
    grid_size: int, 
    max_attempts: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> Position:
    """
    Generates a random [x, y] position for an apple that doesn't conflict with the snake.

    This function uses a rejection sampling approach to find a valid position.
    For better performance on nearly-full boards, consider using get_empty_positions()
    and selecting randomly from the result.

    Args:
        snake_positions: List of NumPy arrays representing snake segment coordinates.
        grid_size: The size of the square grid (e.g., 10 for a 10x10 board).
        max_attempts: Maximum attempts before raising an exception (prevents infinite loops).
        rng: Optional random number generator for deterministic testing.

    Returns:
        A NumPy array representing the [x, y] coordinates of the new apple.

    Raises:
        RuntimeError: If no free space can be found within max_attempts.
        ValueError: If grid_size is invalid or snake occupies all positions.
        
    Example:
        >>> snake = [np.array([1, 1]), np.array([1, 2])]
        >>> apple_pos = generate_random_apple(snake, 10)
        >>> print(f"Apple at: {apple_pos}")
        Apple at: [3 7]  # Random position not occupied by snake
    """
    if grid_size <= 0:
        raise ValueError(f"Invalid grid_size: {grid_size}. Must be positive.")
    
    total_positions = grid_size * grid_size
    if len(snake_positions) >= total_positions:
        raise ValueError("Snake occupies all available positions on the board.")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # For heavily occupied boards, use a more efficient approach
    if len(snake_positions) > total_positions * 0.8:
        empty_positions = get_empty_positions(snake_positions, grid_size)
        if not empty_positions:
            raise RuntimeError("No empty positions available for apple placement.")
        return rng.choice(empty_positions)
    
    # Use rejection sampling for lighter occupation
    for attempt in range(max_attempts):
        pos = rng.integers(0, grid_size, 2, dtype=np.int_)
        
        # Check if position conflicts with any snake segment
        if not any(np.array_equal(pos, seg) for seg in snake_positions):
            return pos
    
    # If we reach here, we couldn't find a position (shouldn't happen with valid input)
    raise RuntimeError(
        f"Failed to find a free space for apple after {max_attempts} attempts. "
        f"Board may be too crowded (snake length: {len(snake_positions)}, "
        f"total positions: {total_positions})."
    )


def update_board_array(
    board: BoardArray,
    snake_positions: PositionList,
    apple_position: Position,
    board_info: BoardInfo,
) -> None:
    """
    Updates the game board array in-place with current game state.

    This function efficiently updates the entire board state, filling it with
    integer codes that represent different game elements. The board uses a
    coordinate system where board[y, x] corresponds to position [x, y].

    Args:
        board: The 2D NumPy array representing the game board (modified in-place).
        snake_positions: List of [x, y] coordinates for each snake segment.
        apple_position: The [x, y] coordinates of the apple.
        board_info: Dictionary mapping entity names to their integer codes.
                   Expected keys: 'empty', 'snake', 'apple'

    Raises:
        KeyError: If required keys are missing from board_info.
        ValueError: If positions are outside board boundaries.
        
    Example:
        >>> board = np.zeros((10, 10), dtype=np.int8)
        >>> snake = [np.array([1, 1]), np.array([1, 2])]
        >>> apple = np.array([5, 5])
        >>> info = {'empty': 0, 'snake': 1, 'apple': 2}
        >>> update_board_array(board, snake, apple, info)
        >>> print(board[1, 1])  # Should be 1 (snake)
        1
    """
    # Validate board_info has required keys
    required_keys = {'empty', 'snake', 'apple'}
    missing_keys = required_keys - board_info.keys()
    if missing_keys:
        raise KeyError(f"Missing required board_info keys: {missing_keys}")
    
    # Validate board dimensions
    board_height, board_width = board.shape
    
    # Clear the board with empty value
    board.fill(board_info["empty"])

    # Place snake segments (board indexing is [y, x])
    for position in snake_positions:
        if len(position) != 2:
            raise ValueError(f"Invalid snake position format: {position}")
        
        x, y = position
        if not (0 <= x < board_width and 0 <= y < board_height):
            raise ValueError(
                f"Snake position {[x, y]} is outside board bounds "
                f"(width: {board_width}, height: {board_height})"
            )
        
        board[y, x] = board_info["snake"]

    # Place apple
    if len(apple_position) != 2:
        raise ValueError(f"Invalid apple position format: {apple_position}")
    
    ax, ay = apple_position
    if not (0 <= ax < board_width and 0 <= ay < board_height):
        raise ValueError(
            f"Apple position {[ax, ay]} is outside board bounds "
            f"(width: {board_width}, height: {board_height})"
        )
    
    board[ay, ax] = board_info["apple"] 


def create_text_board(
    grid_size: int,
    head_position: Coordinate,
    body_positions: List[Coordinate],
    apple_position: Optional[Coordinate] = None,
    empty_char: str = '.',
    head_char: str = 'H',
    body_char: str = 'S',
    apple_char: str = 'A'
) -> str:
    """
    Creates a text-based representation of the game board.
    
    This function generates a human-readable board string that can be used
    in prompts, logs, or debugging. The board is displayed with the origin
    at the bottom-left, following the game's coordinate system.
    
    Args:
        grid_size: The size of the square grid.
        head_position: The [x, y] coordinates of the snake's head.
        body_positions: List of [x, y] coordinates for snake body segments (excluding head).
        apple_position: Optional [x, y] coordinates of the apple.
        empty_char: Character to represent empty cells.
        head_char: Character to represent the snake's head.
        body_char: Character to represent the snake's body.
        apple_char: Character to represent the apple.
        
    Returns:
        A string representation of the board with rows separated by newlines.
        
    Example:
        >>> head = [1, 1]
        >>> body = [[1, 2], [1, 3]]
        >>> apple = [5, 5]
        >>> board = create_text_board(10, head, body, apple)
        >>> print(board)
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . A
        . . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        . H S S . . . . . .
    """
    # Initialize empty board
    board = [[empty_char for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Place apple if provided
    if apple_position and len(apple_position) >= 2:
        try:
            apple_x, apple_y = apple_position[0], apple_position[1]
            if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                board[apple_y][apple_x] = apple_char
        except (KeyError, TypeError, IndexError):
            # Handle dict format as fallback
            if isinstance(apple_position, dict) and 'x' in apple_position and 'y' in apple_position:
                apple_x, apple_y = apple_position['x'], apple_position['y']
                if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                    board[apple_y][apple_x] = apple_char
    
    # Place body segments (excluding head)
    for pos in body_positions:
        if len(pos) >= 2:
            pos_x, pos_y = pos[0], pos[1]
            if 0 <= pos_x < grid_size and 0 <= pos_y < grid_size:
                board[pos_y][pos_x] = body_char
    
    # Place head (should be last to ensure it's not overwritten by body)
    if len(head_position) >= 2:
        head_x, head_y = head_position[0], head_position[1]
        if 0 <= head_x < grid_size and 0 <= head_y < grid_size:
            board[head_y][head_x] = head_char
    
    # Convert to string with rows reversed (origin at bottom-left)
    return "\n".join(" ".join(row) for row in reversed(board)) 