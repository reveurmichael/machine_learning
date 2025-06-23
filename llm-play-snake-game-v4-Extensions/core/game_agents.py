from __future__ import annotations

"""Public contract for any autonomous or interactive policy that can drive the
Snake game.  All first-class agents (LLM, heuristic, RL, human, …) MUST
implement this one-method interface so that the *core* package never needs to
know *who* is providing moves.

Placed in `core/game_agents.py` to follow the `game_*` naming convention that
makes the first-citizen folder easy to grep and keeps a tidy namespace:

    from core.game_agents import SnakeAgent

This whole module is NOT Task0 specific.
"""

from typing import Protocol, Any, runtime_checkable

__all__ = ["SnakeAgent"]


@runtime_checkable
class SnakeAgent(Protocol):
    """Minimal surface required by the game loop."""

    def get_move(self, game: Any) -> str | None:  # noqa: D401, ANN401
        """Return the next direction or ``None`` (treated as EMPTY)."""

# Backward compatibility alias for extensions expecting BaseAgent name
BaseAgent = SnakeAgent 