"""
Heuristics Agents Package - Canonical Factory Pattern Implementation

Available Algorithms (Token Variants Only):
- BFS-512, BFS-1024, BFS-2048, BFS-4096, BFS-SAFE-GREEDY-4096

Base classes (BFS, BFS-SAFE-GREEDY) are blueprints and not registered.
"""
from __future__ import annotations

from typing import List, Any

from utils.factory_utils import SimpleFactory

from .agent_bfs import BFSAgent  # blueprint (not registered)
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent  # blueprint (not registered)
from .agent_bfs_tokens_512 import BFS512TokenAgent
from .agent_bfs_tokens_1024 import BFS1024TokenAgent
from .agent_bfs_tokens_2048 import BFS2048TokenAgent
from .agent_bfs_tokens_4096 import BFS4096TokenAgent
from .agent_bfs_safe_greedy_tokens_4096 import BFSSafeGreedy4096TokenAgent

_factory = SimpleFactory("HeuristicAgentFactory")

# Register only token variants used for dataset generation
_factory.register("BFS-512", BFS512TokenAgent)
_factory.register("BFS-1024", BFS1024TokenAgent)
_factory.register("BFS-2048", BFS2048TokenAgent)
_factory.register("BFS-4096", BFS4096TokenAgent)
_factory.register("BFS-SAFE-GREEDY-4096", BFSSafeGreedy4096TokenAgent)

DEFAULT_ALGORITHM: str = "BFS-512"

def create(algorithm_name: str, **kwargs) -> Any:
    return _factory.create(algorithm_name, **kwargs)

def get_available_algorithms() -> List[str]:
    return _factory.list_available()

__all__ = [
    "create",
    "get_available_algorithms",
    "DEFAULT_ALGORITHM",
    # Blueprints (not registered)
    "BFSAgent",
    "BFSSafeGreedyAgent",
    # Registered variants
    "BFS512TokenAgent",
    "BFS1024TokenAgent",
    "BFS2048TokenAgent",
    "BFS4096TokenAgent",
    "BFSSafeGreedy4096TokenAgent",
] 
