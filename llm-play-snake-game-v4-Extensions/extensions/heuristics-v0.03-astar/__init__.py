"""
A* Heuristic Snake Game Extension (v0.02)
========================================

This extension implements the A* (A-star) pathfinding algorithm for playing Snake.
A* is more efficient than BFS as it uses a heuristic function to guide the search
towards the goal, typically resulting in faster pathfinding and better performance.

Key Features:
- A* pathfinding with Manhattan distance heuristic
- Priority queue for optimal path exploration
- Same architecture as heuristics-v0.01 but with A* algorithm
- Standalone implementation with no external dependencies

Design Philosophy:
- Extends BaseGameManager, BaseGameData, BaseGameLogic, BaseGameController
- Uses Factory pattern for pluggable components
- Compatible with Task-0 logging format
- Demonstrates perfect inheritance from core base classes

Algorithm Details:
- f(n) = g(n) + h(n) where:
  - g(n) = actual cost from start to node n
  - h(n) = heuristic cost from node n to goal (Manhattan distance)
- Uses priority queue to explore most promising nodes first
- Guarantees optimal path when heuristic is admissible
"""

__version__ = "0.02"
__author__ = "Snake-GTP Extensions"
__description__ = "A* Heuristic Snake Game Algorithm"

# Make key classes available at package level
from .astar_agent import AStarAgent
from .game_data import HeuristicGameData
from .game_logic import HeuristicGameLogic
from .game_manager import HeuristicGameManager

__all__ = [
    "AStarAgent",
    "HeuristicGameData", 
    "HeuristicGameLogic",
    "HeuristicGameManager"
] 