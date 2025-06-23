from __future__ import annotations

"""RL Game Data – v0.02

Lightweight container extending :class:`core.game_data.BaseGameData` with
reinforcement-learning specific statistics.  Kept deliberately *minimal* here –
experiments may subclass this further (e.g. to record Q-value histograms).
"""

from typing import List

import numpy as np

from core.game_data import BaseGameData


class RLGameData(BaseGameData):
    """Extends generic data with RL episode bookkeeping."""

    def __init__(self):  # noqa: D401 (one-liner ok)
        super().__init__()
        self.episode_reward: float = 0.0
        self.episode_steps: int = 0
        self.total_reward: float = 0.0
        self.episode_count: int = 0

        # History buffers (rolling)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_scores: List[int] = []

    # ------------------------------------------------------------------
    # Helper API called by GameLogic / Manager
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:  # noqa: D401
        self.episode_reward = 0.0
        self.episode_steps = 0

    def record_step(self, reward: float, score: int) -> None:  # noqa: D401
        self.episode_reward += reward
        self.episode_steps += 1
        self.score = score

    def end_episode(self) -> None:  # noqa: D401
        self.episode_count += 1
        self.total_reward += self.episode_reward
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_steps)
        self.episode_scores.append(self.score)

    # Exposed metric helpers -------------------------------------------

    @property
    def moving_avg_reward(self) -> float:  # noqa: D401
        return float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0 