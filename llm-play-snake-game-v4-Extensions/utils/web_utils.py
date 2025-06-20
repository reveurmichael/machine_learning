"""Shared helpers for Flask/Streamlit web front-ends.

Keeps common mappings (colours, end-reason translations, …) in a single
place so main_web.py, replay_web.py, human_play_web.py don't duplicate
the same literals – silencing R0801 and easing future changes.

This whole module is NOT Task0 specific.
"""

from __future__ import annotations

from typing import Optional

from config.game_constants import END_REASON_MAP
from config.ui_constants import COLORS

__all__ = [
    "build_color_map",
    "translate_end_reason",
    "to_list",
    "build_state_dict",
]


# This function is NOT Task0 specific.
def build_color_map() -> dict[str, tuple[int, int, int]]:
    """Return colour map expected by the front-end JS/HTML."""

    return {
        "snake_head": COLORS["SNAKE_HEAD"],
        "snake_body": COLORS["SNAKE_BODY"],
        "apple": COLORS["APPLE"],
        "background": COLORS["BACKGROUND"],
        "grid": COLORS["GRID"],
    }

# This function is NOT Task0 specific.
def translate_end_reason(code: Optional[str]) -> Optional[str]:
    """Human-readable game-end reason."""

    if not code:
        return None
    return END_REASON_MAP.get(code, code)


# --------------------------
# Convenience helpers for state construction
# --------------------------

# This function is NOT Task0 specific.
def to_list(obj) -> list | object:  # noqa: D401 – tiny utility
    """Return ``obj.tolist()`` when available, otherwise the original ``obj``.

    Useful for serialising NumPy arrays into JSON-friendly Python lists.
    """

    return obj.tolist() if hasattr(obj, "tolist") else obj

# This function is NOT Task0 specific.
def build_state_dict(
    snake_positions,
    apple_position,
    score: int,
    steps: int,
    grid_size: int,
    *,
    extra: dict | None = None,
):
    """Generic JSON-serialisable game state dict used by front-ends."""

    state = {
        "snake_positions": to_list(snake_positions),
        "apple_position": to_list(apple_position),
        "score": score,
        "steps": steps,
        "grid_size": grid_size,
        "colors": build_color_map(),
    }

    if extra:
        state.update(extra)

    return state 
