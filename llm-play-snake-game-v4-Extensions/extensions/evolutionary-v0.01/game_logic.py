"""GA GameLogic (Evolutionary-v0.01)
----------------------------------

Thin wrapper around :class:`core.game_logic.BaseGameLogic` that delegates the
planning step to a :class:`~extensions.evolutionary_v0_01.agent_ga.GAAgent`.

Only **one** method is overridden – :py:meth:`plan_next_moves` – to keep the
surface area minimal and highlight how little glue code is needed to integrate
an evolutionary planner.
"""

from __future__ import annotations

from typing import Optional, List

from core.game_logic import BaseGameLogic
from config.game_constants import GRID_SIZE  # Default fallback

from agent_ga import GAAgent


class EGA_GameLogic(BaseGameLogic):
    """GameLogic that uses a Genetic Algorithm agent for planning."""

    # BaseGameLogic expects a reference to a *data* class but the default is
    # already fine for v0.01 so we leave `GAME_DATA_CLS` untouched.

    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = False):
        super().__init__(grid_size=grid_size, use_gui=use_gui)
        self._agent: Optional[GAAgent] = None

    # ------------------------------------------------------------------
    # Public helper – called by GameManager
    # ------------------------------------------------------------------

    def set_agent(self, agent: GAAgent) -> None:
        self._agent = agent

    # ------------------------------------------------------------------
    # BaseGameLogic contract – override
    # ------------------------------------------------------------------

    def plan_next_moves(self) -> None:  # pylint: disable=missing-function-docstring
        if self._agent is None:
            raise RuntimeError("GAAgent has not been injected via set_agent().")

        head = tuple(self.snake_positions[-1])  # type: ignore[arg-type]
        apple = tuple(self.apple_position)      # type: ignore[arg-type]
        plan: List[str] = self._agent.plan(head, apple, self.grid_size)

        # Store plan so BaseGameLogic machinery can pop() moves during the round
        self.planned_moves = plan 