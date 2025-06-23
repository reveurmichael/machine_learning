"""A3C Agent – v0.02 (skeleton)

Asynchronous Advantage Actor-Critic placeholder.  Real implementation would
spawn parallel environments and aggregate gradients; out-of-scope for this
demonstration.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class A3CAgent:
    """Minimal stub obeying :pydata:`RLAgentProtocol`."""

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