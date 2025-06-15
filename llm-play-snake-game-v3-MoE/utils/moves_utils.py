"""
Movement-centric helper functions for the Snake game.

This module hosts stateless utilities that work with direction tokens or
positions.  Keeping them here avoids scattering small movement helpers
across unrelated files (text_utils, game_manager_utils, etc.).
"""

from typing import Iterable, List, Union

# ------------------------------------------------------------
# Canonicalising direction tokens
# ------------------------------------------------------------

DIRECTION_STR = Union[str, int, float]


def normalize_direction(move: DIRECTION_STR):
    """Return a canonical representation of a single direction token.

    * Non-string values are returned unchanged (defensive pass-through).
    * Strings are upper-cased and stripped so that "right\n" → "RIGHT".
    """
    if isinstance(move, str):
        return move.strip().upper()
    return move


def normalize_directions(moves: Iterable[DIRECTION_STR]) -> List[DIRECTION_STR]:
    """Vectorised wrapper around *normalize_direction*."""
    return [normalize_direction(m) for m in moves]


# ------------------------------------------------------------
# Direction relationship helpers
# ------------------------------------------------------------


def is_reverse(dir_a: str, dir_b: str) -> bool:
    """Return *True* if *dir_a* is the exact opposite of *dir_b* (e.g. UP vs DOWN).

    This helper is intentionally private (prefixed with an underscore) because it
    encodes a very specific rule used by the game logic.  External callers
    should rely on higher-level APIs unless they really need this low-level
    check.
    """

    dir_a = dir_a.upper()
    dir_b = dir_b.upper()

    return (
        (dir_a == "UP" and dir_b == "DOWN") or
        (dir_a == "DOWN" and dir_b == "UP") or
        (dir_a == "LEFT" and dir_b == "RIGHT") or
        (dir_a == "RIGHT" and dir_b == "LEFT")
    )


# ------------------------------------------------------------
# Simple positional analytics (used in prompt engineering)
# ------------------------------------------------------------

def calculate_move_differences(head_pos, apple_pos):
    """Return a human-readable diff between *head_pos* and *apple_pos*.

    Example output:  "#RIGHT - #LEFT = 3, and #UP - #DOWN = 1"
    """
    head_x, head_y = head_pos
    apple_x, apple_y = apple_pos

    # Horizontal diff
    if head_x <= apple_x:
        x_diff_text = f"#RIGHT - #LEFT = {apple_x - head_x} (= {apple_x} - {head_x})"
    else:
        x_diff_text = f"#LEFT - #RIGHT = {head_x - apple_x} (= {head_x} - {apple_x})"

    # Vertical diff
    if head_y <= apple_y:
        y_diff_text = f"#UP - #DOWN = {apple_y - head_y} (= {apple_y} - {head_y})"
    else:
        y_diff_text = f"#DOWN - #UP = {head_y - apple_y} (= {head_y} - {apple_y})"

    return f"{x_diff_text}, and {y_diff_text}" 