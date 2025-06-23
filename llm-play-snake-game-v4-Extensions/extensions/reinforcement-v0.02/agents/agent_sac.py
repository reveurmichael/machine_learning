"""SAC Agent – v0.02 (skeleton)

Soft Actor-Critic placeholder. Highlights how additional algorithms can be
plugged in without touching the manager.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SACAgent:
    """Stub SAC implementation."""

    def __init__(self, state_size: int, action_size: int, **kwargs: Any) -> None:
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self, state: np.ndarray) -> int:  # noqa: D401
        return int(np.random.randint(self.action_size))

    def remember(self, *transition: Any) -> None:  # noqa: D401
        pass

    def learn(self) -> None:  # noqa: D401
        pass

    def add_observer(self, callback):  # noqa: D401
        pass 