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
    "build_error_response",
    "build_success_response",
    "build_color_payload",
]


# This function is NOT Task0 specific.
def build_color_map(as_list: bool = False) -> dict[str, tuple[int, int, int]] | dict[str, list[int]]:
    """Return colour map expected by the front-end JS/HTML.
    
    Args:
        as_list: If True, convert RGB tuples to lists for JavaScript consumption
        
    Returns:
        Dictionary mapping color names to RGB tuples or lists
        
    Educational Value: Shows how to handle different frontend requirements
    Extension Pattern: Extensions can use for consistent color handling
    """
    color_map = {
        "snake_head": COLORS["SNAKE_HEAD"],
        "snake_body": COLORS["SNAKE_BODY"],
        "apple": COLORS["APPLE"],
        "background": COLORS["BACKGROUND"],
        "grid": COLORS["GRID"],
    }
    
    if as_list:
        return {k: list(v) for k, v in color_map.items()}
    return color_map

# This function is NOT Task0 specific.
def translate_end_reason(code: Optional[str]) -> Optional[str]:
    """Human-readable game-end reason."""

    if not code:
        return None
    return END_REASON_MAP.get(code, code)


# ---------------------
# Convenience helpers for state construction
# ---------------------

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
        "colors": build_color_map(as_list=True),
    }

    if extra:
        state.update(extra)

    return state


# Additional web utilities consolidated from web/utils.py for better SSOT compliance




def build_error_response(message: str, details: Optional[str] = None) -> dict[str, str]:
    """Build standardized error response.
    
    Args:
        message: Error message
        details: Optional additional details
        
    Returns:
        Standardized error response dictionary
        
    Educational Value: Shows consistent error response formatting
    Extension Pattern: Extensions can use this for standardized error handling
    """
    response = {
        'status': 'error',
        'message': message
    }
    if details:
        response['details'] = details
    return response


def build_success_response(message: str, data: Optional[dict] = None) -> dict[str, str]:
    """Build standardized success response.
    
    Args:
        message: Success message
        data: Optional additional data
        
    Returns:
        Standardized success response dictionary
        
    Educational Value: Shows consistent success response formatting
    Extension Pattern: Extensions can use this for standardized API responses
    """
    response = {
        'status': 'success',
        'message': message
    }
    if data:
        response.update(data)
    return response


# DEPRECATED alias – will be removed in future versions; use build_color_map(as_list=True)
# -----------------------------------------------------------------------------

def build_color_payload() -> dict[str, list[int]]:  # noqa: D401 – simple wrapper
    """Return colour map converted to lists (legacy helper).

    This function is kept as a thin wrapper around ``build_color_map`` solely to
    avoid a sweeping rename across dozens of files.  New code **MUST** call
    ``build_color_map(as_list=True)`` directly instead.  The alias will be
    removed once all call-sites migrate.
    """

    return build_color_map(as_list=True)


# -----------------------------------------------------------------------------
# Update re-export list
# -----------------------------------------------------------------------------

if "build_color_payload" not in __all__:
    __all__.append("build_color_payload")
