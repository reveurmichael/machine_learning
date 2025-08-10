from __future__ import annotations

from typing import Any, List

from utils.factory_utils import SimpleFactory

from .greedy_agent import GreedyAgent

_factory = SimpleFactory("SupervisedAgentFactory")
_factory.register("GREEDY", GreedyAgent)

DEFAULT_ALGORITHM = "GREEDY"

def create(algorithm_name: str, **kwargs) -> Any:
    return _factory.create(algorithm_name, **kwargs)

def get_available_algorithms() -> List[str]:
    return _factory.list_available()

__all__ = [
    "create",
    "get_available_algorithms",
    "DEFAULT_ALGORITHM",
    "GreedyAgent",
]