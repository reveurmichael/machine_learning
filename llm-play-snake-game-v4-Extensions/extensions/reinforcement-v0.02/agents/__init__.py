from __future__ import annotations

from typing import Any, List

from utils.factory_utils import SimpleFactory

from .q_learning_agent import QLearningAgent
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent

_factory = SimpleFactory("RLV02AgentFactory")
_factory.register("Q_LEARNING", QLearningAgent)
_factory.register("DQN", DQNAgent)
_factory.register("PPO", PPOAgent)

DEFAULT_ALGORITHM = "DQN"

def create(algorithm_name: str, **kwargs) -> Any:
    return _factory.create(algorithm_name, **kwargs)

def get_available_algorithms() -> List[str]:
    return _factory.list_available()

__all__ = [
    "create",
    "get_available_algorithms",
    "DEFAULT_ALGORITHM",
    "QLearningAgent",
    "DQNAgent",
    "PPOAgent",
]