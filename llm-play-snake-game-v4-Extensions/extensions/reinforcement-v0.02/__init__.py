from __future__ import annotations

"""Reinforcement-Learning v0.02
--------------------------------
Multi-algorithm RL extension (DQN, PPO, A3C, SAC …).
Keeps Task-0 core intact by living under ``extensions/``.

This ``__init__`` exposes:
1. ``RLConfig`` – immutable configuration object shared by CLI & pipeline.
2. ``create_rl_agent`` – factory that maps ``algorithm`` → concrete class.

All concrete agents sit in :pymod:`extensions.reinforcement-v0.02.agents`.

Design patterns applied
=======================
• *Singleton*   – internal FileManager from ``extensions.common`` ensures a single log path.
• *Factory*     – ``create_rl_agent`` builds requested agent.
• *Strategy*    – Agents share a common interface but implement their own learning strategies.

Do **not** import this package from Task-0 modules; first-citizen code must remain oblivious to second-citizens.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class RLConfig:
    """Centralised, *immutable* configuration for RL experiments.

    Parameters
    ----------
    algorithm:
        Name of the algorithm to train (``DQN``, ``PPO`` …).
    grid_size:
        Board size; must match the environment.
    episodes:
        Number of training episodes.
    max_steps:
        Max steps per episode.
    seed:
        RNG seed for reproducibility.
    extra:
        Algorithm-specific hyper-parameters (learning-rate, buffer-size …).
    """

    algorithm: str = "DQN"
    grid_size: int = 10
    episodes: int = 1_000
    max_steps: int = 1_000
    seed: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Derived fields ---------------------------------------------------------

    timestamp: str = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:  # noqa: D401 – one-liner style ok here
        object.__setattr__(self, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        log_root = Path("logs") / "extensions" / f"reinforcement-{self.algorithm.lower()}_{self.timestamp}"
        object.__setattr__(self, "log_dir", log_root)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

from importlib import import_module


def _lazy_import(path: str, class_name: str):
    """Lazy importer to avoid heavyweight deps unless needed."""
    mod = import_module(path)
    return getattr(mod, class_name)


_AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "DQN": ("extensions.reinforcement-v0.02.agents.agent_dqn", "DQNAgent"),
    "PPO": ("extensions.reinforcement-v0.02.agents.agent_ppo", "PPOAgent"),
    "A3C": ("extensions.reinforcement-v0.02.agents.agent_a3c", "A3CAgent"),
    "SAC": ("extensions.reinforcement-v0.02.agents.agent_sac", "SACAgent"),
}


def create_rl_agent(algorithm: str, **kwargs):
    """Factory that builds an RL agent based on ``algorithm``.

    Examples
    --------
    >>> agent = create_rl_agent("DQN", state_size=19, action_size=4)
    """
    algo_upper = algorithm.upper()
    if algo_upper not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown RL algorithm '{algorithm}'. Available: {list(_AGENT_REGISTRY)}")

    module_path, class_name = _AGENT_REGISTRY[algo_upper]
    cls = _lazy_import(module_path, class_name)
    return cls(**kwargs)


# Public re-exports ----------------------------------------------------------

__all__ = [
    "RLConfig",
    "create_rl_agent",
]

# ---------------------------------------------------------------------------
# hyphen ↔ underscore alias to avoid ``ModuleNotFoundError`` in tools like
# Click or Streamlit that mangle module names.
# ---------------------------------------------------------------------------
import sys as _sys

_alias = __name__.replace('-', '_')  # "extensions.reinforcement_v0.02"
if _alias not in _sys.modules:
    _sys.modules[_alias] = _sys.modules[__name__]

# Lazily alias sub-packages when they are first imported ---------------------

def __getattr__(name: str):  # noqa: D401
    full_real = f"{__name__}.{name}"
    full_alias = f"{_alias}.{name}"
    if full_real in _sys.modules:
        _sys.modules[full_alias] = _sys.modules[full_real]
    return _sys.modules[full_real] 