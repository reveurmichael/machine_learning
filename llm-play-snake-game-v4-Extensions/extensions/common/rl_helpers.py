"""rl_helpers.py – Extra utilities shared by RL extensions

Contains *non-essential* helper objects (logging, target-network sync,
random-seed setup) so that algorithmic code remains uncluttered in each
`reinforcement-v0.*` package.
"""

from __future__ import annotations

import random
from typing import Deque, Optional
from collections import deque

import numpy as np
import torch

__all__ = [
    "set_seed",
    "hard_update",
    "soft_update",
    "RunningRewardTracker",
]


# ---------------------
# Reproducibility helpers
# ---------------------

def set_seed(seed: int) -> None:
    """Seed `random`, `numpy`, *and* torch (both CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------
# Target-network synchronisation
# ---------------------

def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """`target ← source` (exact copy)."""
    target.load_state_dict(source.state_dict())


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 0.005) -> None:
    """Polyak averaging: `θ_target ← τ θ_src + (1−τ) θ_target`."""
    for t_p, s_p in zip(target.parameters(), source.parameters()):
        t_p.data.mul_(1.0 - tau).add_(s_p.data, alpha=tau)


# ---------------------
# Simple reward logger
# ---------------------

class RunningRewardTracker:
    """Keep rolling stats for episode returns (mean over window)."""

    def __init__(self, window: int = 100):
        self.window = window
        self._buffer: Deque[float] = deque(maxlen=window)

    def add(self, reward: float) -> None:  # noqa: D401
        self._buffer.append(reward)

    @property
    def mean(self) -> Optional[float]:  # noqa: D401
        if not self._buffer:
            return None
        return float(np.mean(self._buffer))

    def __repr__(self) -> str:  # noqa: D401
        m = self.mean
        mean_str = f"{m:.3f}" if m is not None else "N/A"
        return f"RunningRewardTracker(window={self.window}, mean={mean_str})" 