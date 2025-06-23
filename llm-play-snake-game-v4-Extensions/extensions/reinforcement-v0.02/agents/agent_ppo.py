"""PPO Agent – v0.02 (skeleton)

This is a *placeholder* implementation meant to illustrate the multi-algorithm
architecture in RL v0.02.  A production-ready version would integrate a policy
& value network, GAE, and clipping objective – see Stable-Baselines3 for a
reference.

The simplified agent demonstrates the required public interface so that
managers, loggers, and dashboards can treat *all* RL algorithms uniformly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PPOAgent:  # noqa: D101 – docstring at class level
    """Proximal Policy Optimisation (stub)."""

    def __init__(self, state_size: int, action_size: int, **kwargs: Any) -> None:  # noqa: D401
        self.state_size = state_size
        self.action_size = action_size
        self._step_count = 0
        # Placeholder policy: uniform random

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:  # noqa: D401
        return int(np.random.randint(self.action_size))

    def remember(self, *transition: Any) -> None:  # noqa: D401
        # PPO batches episodes; in this stub we do nothing.
        pass

    def learn(self) -> None:  # noqa: D401
        # No-op learning – replace with PPO optimisation.
        self._step_count += 1

    # Observer pattern placeholder -------------------------------------

    def add_observer(self, callback):  # noqa: D401 – minimal stub
        # Observers ignored in stub
        pass 