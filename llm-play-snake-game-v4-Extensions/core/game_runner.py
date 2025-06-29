"""Quick-play helper for rolling out a :class:`core.game_agents.BaseAgent`.

The module lives inside *core* so that both first- and second-citizen tasks can
perform ad-hoc roll-outs without adding an extra dependency layer.  It remains
LLM-agnostic and depends only on other core modules plus generic utilities.

Example
-------
>>> from core.game_agents import BaseAgent
>>> from core.game_runner import play
>>> trajectory = play(my_agent, max_steps=500, render=True, seed=123)
"""

from __future__ import annotations

from typing import List, Optional

from core.game_agents import BaseAgent
from core.game_logic import GameLogic

# Optional reproducibility helper (non-core dependency is acceptable)
from utils.seed_utils import seed_everything

__all__ = ["play"]


def play(
    agent: BaseAgent,
    max_steps: int = 1_000,
    render: bool = False,
    *,
    seed: Optional[int] = None,
) -> List[dict]:
    """Execute a game and return the trajectory as a list of state dictionaries.

    Parameters
    ----------
    agent
        Policy implementing the :class:`BaseAgent` protocol.
    max_steps
        Safety cap to stop the simulation after *max_steps* even if the game
        would continue.
    render
        When *True* the PyGame GUI draws each frame.
    seed
        If provided, :func:`utils.seed_utils.seed_everything` is called to seed
        RNGs across libraries for repeatability.  Defaults to *None* (no
        seeding).
    """

    if seed is not None:
        seed_everything(seed)

    game = GameLogic(use_gui=render)
    trajectory: List[dict] = []

    for _ in range(max_steps):
        trajectory.append(game.get_state_snapshot())

        move = agent.get_move(game) or "EMPTY"
        active, _ = game.make_move(move)
        if not active:
            break
        if render:
            game.draw()

    return trajectory 