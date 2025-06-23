"""
Heuristics v0.01 - Simple BFS Snake Agent
--------------------

Minimal proof of concept extension.
"""

from __future__ import annotations

from extensions.common.path_utils import ensure_project_root_on_path
ensure_project_root_on_path()

from .agent_bfs import BFSAgent
from .game_manager import HeuristicGameManager
from .game_logic import HeuristicGameLogic

__all__ = [
    "BFSAgent",
    "HeuristicGameManager", 
    "HeuristicGameLogic",
] 