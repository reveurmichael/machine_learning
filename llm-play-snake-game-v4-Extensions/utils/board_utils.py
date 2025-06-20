from __future__ import annotations

"""Board-related pure helpers shared across the project.

They live in ``utils`` so that importing them never drags heavy core modules
(like NumPy game state) into lightweight callers.  This also keeps the naming
convention inside ``core/`` (all *game_*.py) intact.
"""

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "generate_random_apple",
    "update_board_array",
]


# This function is NOT Task0 specific.
def generate_random_apple(
    snake_positions: NDArray[np.int_],
    grid_size: int,
) -> NDArray[np.int_]:
    """Return a random [x, y] apple position not colliding with *snake_positions*."""

    while True:
        pos = np.random.randint(0, grid_size, 2)
        if not any(np.array_equal(pos, seg) for seg in snake_positions):
            return pos


# This function is NOT Task0 specific.
def update_board_array(
    board: NDArray[np.int_],
    snake_positions: NDArray[np.int_],
    apple_position: NDArray[np.int_],
    board_info: Mapping[str, int],
) -> None:
    """Fill *board* in-place given the entity positions."""

    # Clear the board
    board.fill(board_info["empty"])

    # Draw snake (board is indexed [y, x])
    for x, y in snake_positions:
        board[y, x] = board_info["snake"]

    # Draw apple
    ax, ay = apple_position
    board[ay, ax] = board_info["apple"] 