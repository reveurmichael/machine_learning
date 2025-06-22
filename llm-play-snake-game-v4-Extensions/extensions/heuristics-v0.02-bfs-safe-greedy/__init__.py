"""
Heuristics v0.01 Extension - Simple BFS/DFS Snake Agents
======================================================

This is the first simple extension implementing heuristic algorithms for 
the Snake game. It extends the base classes from the core package to provide
BFS and DFS pathfinding agents.

Design Philosophy:
- Extends BaseGameManager, BaseGameLogic, and BaseGameData
- No GUI dependencies (headless by default)
- Generates the same log format as Task-0 (game_N.json, summary.json)
- Uses SnakeAgent protocol for clean integration

This extension demonstrates how future tasks can leverage the base classes
while implementing their own specific algorithms.
"""

from bfs_agent import BFSAgent
from game_manager import HeuristicGameManager
from game_logic import HeuristicGameLogic
from game_data import HeuristicGameData

__all__ = [
    "BFSAgent",
    "HeuristicGameManager", 
    "HeuristicGameLogic",
    "HeuristicGameData",
] 