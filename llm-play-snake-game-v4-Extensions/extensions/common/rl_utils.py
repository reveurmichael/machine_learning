"""rl_utils.py – Generic helpers for lightweight RL agents

The goal is to keep algorithm-specific logic (network architecture, loss,
policy) inside each *reinforcement-v0.* extension while factoring out
re-usable, *non-essential* plumbing into `extensions/common`.

Currently used by both *reinforcement-v0.01* and *reinforcement-v0.02*.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple, Any, Dict

import numpy as np

__all__ = [
    "Experience",
    "ReplayBuffer",
    "linear_epsilon_scheduler",
]


# ---------------------
# Experience replay
# ---------------------


@dataclass(slots=True)
class Experience:
    """Single RL transition (state, action, reward, next_state, done)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """Fixed-size circular buffer with random sampling.

    Only *stores* and *samples*; learning logic lives in each agent.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    # ---------------------
    def push(self, exp: Experience) -> None:  # noqa: D401
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns, d = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in batch])
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(ns), np.array(d, dtype=np.uint8))

    def __len__(self) -> int:  # noqa: D401
        return len(self.buffer)


# ---------------------
# Exploration helpers
# ---------------------


def linear_epsilon_scheduler(start: float, end: float, decay_steps: int):
    """Return a closure implementing ε(t) = max(end, start – t/decay_steps)."""

    def _eps(step: int) -> float:  # noqa: D401
        frac = max(0.0, 1 - step / decay_steps)
        return max(end, start * frac)

    return _eps 