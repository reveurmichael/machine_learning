from __future__ import annotations

"""Geometry helpers for detecting collisions on the Snake board.

They live in ``utils`` so both `core` and high-level session code can import
without cyclic dependencies.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["check_collision"]


def check_collision(
    position: NDArray[np.int_] | list[int],
    snake_positions: NDArray[np.int_],
    grid_size: int,
    is_eating_apple_flag: bool = False,
) -> Tuple[bool, bool]:
    """Return (wall_collision, body_collision) for *position*.

    The rules replicate the Task-0 logic exactly so that existing behaviour is
    unchanged while the helper is now imported from a dedicated module.
    """

    x, y = position

    # ----- Wall -----
    wall_collision = x < 0 or x >= grid_size or y < 0 or y >= grid_size

    # Empty board guard (should never happen but keeps static-analysis happy)
    if len(snake_positions) == 0:
        return wall_collision, False

    # ----- Body -----
    if is_eating_apple_flag:
        # Tail does NOT move → need to check all body segments except the head
        body_segments = snake_positions[:-1]
    else:
        # Tail moves → safe to ignore current tail & head
        body_segments = snake_positions[1:-1] if len(snake_positions) > 2 else []

    body_collision = any(np.array_equal(position, seg) for seg in body_segments)
    return wall_collision, body_collision 