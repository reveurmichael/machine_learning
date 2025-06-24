"""DQN Agent – v0.02

This file **copies the public interface** of the DQN implementation shipped in
v0.01 but trims the learning internals to keep the example concise.  The goal
is to showcase *how multiple agents coexist* in v0.02; performance-critical
code can be ported verbatim from v0.01 if desired.

The class obeys :pyclass:`extensions.reinforcement-v0.02.agents.RLAgentProtocol`.
"""

from __future__ import annotations

from collections import deque
from random import random, sample
from typing import Any, Deque, Tuple

import numpy as np


class DQNAgent:  # noqa: D101 – docstring at class level
    """Deep Q-Network implementation (minimal, educational version).

    Design patterns
    ---------------
    • **Strategy** – encapsulates a learning strategy interchangeable at
      runtime via the factory in :pymod:`extensions.reinforcement-v0.02`.
    • **Observer**  – external callbacks can subscribe via
      :pymeth:`add_observer` to receive `'episode_complete'` events.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        **_: Any,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=buffer_size
        )

        # Very tiny toy-network: Q-table approximated by linear weights for brevity.
        self._weights = np.random.randn(state_size, action_size) * 0.01

        self._observers: list = []

    # ---------------------------------------------------------------------
    # Public API (Strategy Protocol)
    # ---------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:  # noqa: D401 – imperative name ok
        """ϵ-greedy action selection."""
        if random() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = state @ self._weights  # type: ignore[operator]
        return int(np.argmax(q_values))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:  # noqa: D401 – imperative name ok
        """Store transition in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> None:  # noqa: D401
        """One DQN update step (very simplified)."""
        if len(self.memory) < self.batch_size:
            return
        batch = sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(next_state @ self._weights)  # type: ignore[operator]
            q_update = state * (target - (state @ self._weights)[action])  # type: ignore[index,operator]
            self._weights[:, action] += self.lr * q_update

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Observer helpers
    # ------------------------------------------------------------------

    def add_observer(self, callback):
        self._observers.append(callback)

    def _notify(self, event_type: str, data: dict):
        for cb in self._observers:
            try:
                cb(event_type, data)
            except Exception:  # pragma: no cover – observer errors non-fatal
                pass

    # Convenience API -----------------------------------------------------

    def end_episode(self, episode_idx: int, episode_reward: float):
        """Hook for managers to signal episode completion."""
        self._notify("episode_complete", {"id": episode_idx, "reward": episode_reward}) 